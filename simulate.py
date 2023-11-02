import multiprocessing

import numpy as np
from HawkesShotNoise import Hawkes_Shot_Noise
from tools import DataBox
from tqdm.auto import tqdm


class Hawkes_Shot_Noise_Simulate(Hawkes_Shot_Noise):
    def __init__(self, dim_endo: int, dim_exo: int, verbose: bool = False):
        self.verbose = verbose
        super().__init__(dim_endo, dim_exo)

    def save_setting(self, name, save_path):
        self.save_name = name
        self.save_path = save_path
        self.db_save = DataBox(save_path)

    def _tell_me_more(self):
        Lambda = self.R @ (
            self.endo_baseline + np.block([self.exo_baselines, self.exo_baselines])
        )
        self.exp_descendants = (
            Lambda
            - self.endo_baseline
            - np.block([self.exo_baselines, self.exo_baselines])
        ) * self.end_time

    def _generate_poisson_process(self, rate: float, end_time: float = None):
        """generate a poisson process PP(rate)

        Args:
            rate (float): _description_
            end_time (float, optional): _description_. Defaults to None.

        Returns:
            np.array
        """
        if end_time is None:
            end_time = self.end_time
        n_jumps = np.random.poisson(rate * end_time)
        times = np.random.uniform(0, end_time, n_jumps)
        return np.sort(times)

    def generate_delayed_poisson(self, rate: float, delays: list):
        """generate a bivariate poisson process with delay

        Args:
            rate (float): _description_
            delays (list): _description_

        Returns:
            list of np.array
        """
        pp = self._generate_poisson_process(rate)
        delayed_pp = []
        for delay in delays:
            if delay == 0:
                delayed_pp.append(pp)
            else:
                delay_t = np.random.exponential(delay, len(pp))
                new_pp = np.sort(pp + delay_t)
                new_pp = new_pp[new_pp < self.end_time]
                delayed_pp.append(new_pp)
        return delayed_pp

    def _generate_cluster(self, t: float, which_parent: int):
        # print(self.generation)
        # self.generation += 1
        descendants = [[] for j in range(self.dim_endo)]
        ##
        if t > self.end_time:
            return descendants
        ##
        j = which_parent
        for which_child in range(self.dim_endo):
            i = which_child
            # generate children of a generation
            # each sub_branch represents a decay
            # branch is a superposition of these sub_branches
            children = []
            for k, decay in enumerate(self.decays):
                if self.adjacencys[k][i, j] <= 0:
                    continue
                else:
                    alpha = self.adjacencys[k][i, j]
                    # F(t) = \int_0^t alpha beta exp(-beta (s)) ds
                    #      = alpha [1-exp(-beta * t)]
                    poisson_1 = self._generate_poisson_process(
                        1, alpha * (1 - np.exp(-decay * (self.end_time - t)))
                    )
                    # F^-1(t) = -log(1-t/alpha) / beta
                    sub_branch = -np.log(1 - poisson_1 / alpha) / decay + t
                    children += list(sub_branch)
            ##
            descendants[i] += children
            # print(self.end_time-t, j, '-->', i, len(children))
            for tt in children:
                grand_children = self._generate_cluster(tt, i)
                # print(round(self.end_time-tt, 3), i, len(grand_children[0]),len(grand_children[1]))

                for which_type in range(self.dim_endo):
                    descendants[which_type] += grand_children[which_type]
        return descendants

    def simulate_immigrants(self):
        immigrants = [[] for i in range(self.dim_endo)]
        immigrant_labels = [[] for i in range(self.dim_endo)]

        for i in range(self.dim_exo):
            mux = self.exo_baselines[i]
            exo_immigrants = self.generate_delayed_poisson(
                mux, [self.delays[i], self.delays[i + self.dim_exo]]
            )
            for m in range(2):
                mu = self.endo_baseline[i + self.dim_exo * m]
                endo_immigrants = self._generate_poisson_process(mu)
                immigrants[i + self.dim_exo * m] += list(endo_immigrants)
                immigrant_labels[i + self.dim_exo * m] += [0] * len(endo_immigrants)

                immigrants[i + self.dim_exo * m] += list(exo_immigrants[m])
                immigrant_labels[i + self.dim_exo * m] += [-1] * len(exo_immigrants[m])

        for i in range(self.dim_endo):
            sort = np.argsort(immigrants[i])
            immigrant_labels[i] = list(np.array(immigrant_labels[i])[sort])
            immigrants[i] = list(np.array(immigrants[i])[sort])

        return immigrants, immigrant_labels

    def simulate_one(self, seed=37):
        """_summary_

        Args:
            seed (int, optional): _description_. Defaults to 37.

        Returns:
            timestamps, origines: list, list
            ---
            timestamps[i] = list of timestamps of events of type i
            origines[i] = list of origines of events of type i, 0 if endogenous from Hawkes, 1 if immigrants from Hawkes, -1 if exogenous
        """
        timestamps = [[] for i in range(self.dim_endo)]
        origines = [[] for i in range(self.dim_endo)]
        np.random.seed(seed)

        immigrants, immigrant_labels = self.simulate_immigrants()

        if self.verbose:
            n_events = 0
            next_print = 0
            self._tell_me_more()

        for j in range(self.dim_endo):
            immigrants_j = immigrants[j]
            timestamps[j] += immigrants_j
            origines[j] += immigrant_labels[j]
            # -------------------------------------------------------
            if immigrants_j[-1] >= self.end_time:
                print(j, "immigrant", immigrants_j[-1], self.end_time)
            # -------------------------------------------------------

            for t in immigrants_j:
                descendants_of_t = self._generate_cluster(t, j)
                for i in range(self.dim_endo):
                    # -------------------------------------------------------
                    if len(descendants_of_t[i]) > 0:
                        if descendants_of_t[i][-1] > self.end_time:
                            print(i, descendants_of_t[i][-1], self.end_time)
                    # -------------------------------------------------------
                    timestamps[i] += descendants_of_t[i]
                    origines[i] += [1] * len(descendants_of_t[i])

                    if self.verbose:
                        n_events += len(descendants_of_t[i])
                        if n_events > self.exp_descendants.sum() * next_print / 10:
                            if next_print < 10:
                                print(f"{next_print * 10}%% finished")
                            next_print += 1
        if self.verbose:
            print("100% finished")

        for j in range(self.dim_endo):
            sort = np.argsort(timestamps[j])
            timestamps[j] = np.array(timestamps[j])[sort]
            origines[j] = np.array(origines[j])[sort]
        return timestamps, origines

    def simulate(
        self,
        end_time: float,
        n_realization: int = 1,
        seed: int = 37,
        multi_thread: bool = False,
        save: bool = False,
        **kwarg,
    ):
        self.end_time = end_time

        if multi_thread:
            values = tuple(
                zip(
                    range(seed, seed + n_realization),
                )
            )
            with multiprocessing.Pool() as pool:
                res = pool.starmap(self.simulate_one, values)
            timestamps_list = []
            origines_list = []
            for ts, orig in res:
                timestamps_list.append(ts)
                origines_list.append(orig)
        else:
            timestamps_list = []
            origines_list = []
            for n in tqdm(range(n_realization), desc="Simulating..."):
                timestamps, origines = self.simulate_one(seed + n)
                timestamps_list.append(timestamps)
                origines_list.append(origines)

        if save:
            self.save_setting(**kwarg)
            self.db_save.save_pickles([timestamps_list, origines_list], self.save_name)

        return timestamps_list, origines_list
