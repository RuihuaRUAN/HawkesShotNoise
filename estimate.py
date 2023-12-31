from abc import ABC, abstractmethod

import numpy as np
import torch

# from numba import njit
from scipy.linalg import norm, qr, sqrtm
from tick.hawkes import HawkesCumulantMatching
from tick.hawkes.inference.base import LearnerHawkesNoParam
from tqdm.auto import tqdm

from cumulants import compute_cumulants
from HawkesShotNoise import Hawkes_Shot_Noise
from tools import print_info, timefunc


class Hawkes_Shot_Noise_Estimate(Hawkes_Shot_Noise):
    def __init__(self, dim_endo: int, dim_exo: int, device="cpu"):
        """
        Args:
            dim_endo (int): observable dimension, can be 2 or 4
            dim_exo (int): exogenous dimension, can be 1 or 2
            device (str, optional): _description_. Defaults to "cpu".
        """
        super().__init__(dim_endo, dim_exo, device)

    @staticmethod
    def _get_end_times(times):
        T = max([max([t[-1] for t in ts]) for ts in times]) - min(
            [min([t[0] for t in ts]) for ts in times]
        )
        return T

    def set_data(self, times: list, end_time: float = None):
        self.timestamps = times
        if end_time is None:
            self.end_time = self._get_end_times(times)
        else:
            self.end_time = end_time

    @property
    def n_realizations(self):
        return len(self.timestamps)

    def moving_mean_intensity(self, day_times: list, loc_bins: int):
        """every 1 second, compute the local mean intensity

        Args:
            day_times (list): _description_
            loc_bins (int): _description_

        Returns:
            dictionary: _description_
        """

        tfs = timefunc(day_times, 1, self.end_time)  # 1 --> every 1 second
        ms = np.zeros((self.end_time - 2 * loc_bins, self.dim_endo))

        for t in range(self.end_time - 2 * loc_bins):
            for i, tf in enumerate(tfs):
                ms[t, i] = (tf.value(t + 2 * loc_bins) - tf.value(t)) / (2 * loc_bins)
        center_ts = np.arange(self.end_time).astype(int)
        ms = np.pad(ms, ((loc_bins, loc_bins), (0, 0)), "edge")

        return dict(zip(center_ts, ms))

    def estimate_cumulants(self, H: float, loc_bins: float = 0):
        """_summary_

        Args:
            H (float): _description_
            loc_bins (float, optional): _description_. Defaults to None.
        """
        if loc_bins == 0:
            nphc = HawkesCumulantMatching(H)
            LearnerHawkesNoParam.fit(
                nphc,
                self.timestamps,
                end_times=np.ones(self.n_realizations) * self.end_time,
            )
            nphc.compute_cumulants()
            self.L_emp = nphc.mean_intensity
            self.C_emp = nphc.covariance
            self.K_emp = nphc.skewness

        else:
            d = self.dim_endo
            self.C_emp = np.zeros((d, d))
            self._J = np.zeros((self.n_realizations, d, d))

            for day_times in tqdm(self.timestamps):
                day_C = np.zeros((d, d))
                t_to_m = self.moving_mean_intensity(day_times, loc_bins)
                for i, ts_i in enumerate(day_times):
                    for t in ts_i:
                        if t < H or t > self.end_time - H:
                            continue
                        for j, ts_j in enumerate(day_times):
                            mean_j = t_to_m[int(t)][j]
                            dN_j = np.sum(ts_j < t + H) - np.sum(ts_j < t - H)
                            dN_j -= 2 * H * mean_j
                            day_C[i, j] += dN_j
                day_C /= self.end_time - 2 * H
                self.C_emp += day_C / self.n_realizations

    def approximate_optimal_cs_ratio(self):
        """
        Heuristic to set covariance skewness ratio close to its
        optimal value
        """
        norm_sq_C = norm(self.C_emp) ** 2
        norm_sq_K = norm(self.K_emp) ** 2
        return norm_sq_K / (norm_sq_K + norm_sq_C)

    def starting_point_R(self, random=False):
        """
        Heuristic to find a starting point candidate for R

        Parameters
        ----------
        random : `bool`
            Use a random orthogonal matrix instead of identity

        Returns
        -------
        startint_point : `np.ndarray`, shape=(n_nodes, n_nodes)
            A starting point candidate
        """
        sqrt_C = sqrtm(self.C_emp)
        sqrt_L = np.sqrt(self.L_emp)
        if random:
            random_matrix = np.random.rand(self.dim_endo, self.dim_endo)
            M, _ = qr(random_matrix)
        else:
            M = np.eye(self.dim_endo)
        initial = np.dot(np.dot(sqrt_C, M), np.diag(1.0 / sqrt_L))
        return initial

    def set_optimizer(self, learning_rate=1e-3):
        self.optimizer = torch.optim.Adam(self.variables, lr=learning_rate)

    def set_init_values(self, phi=None, exo_baseline=None):
        pass

    def set_variables(self):
        pass

    def objective(self):
        pass

    @property
    def R(self):
        pass

    @property
    def endo_baseline(self):
        pass

    @property
    def exo_baseline(self):
        pass

    @property
    def adjacency(self):
        pass

    def fit(
        self,
        max_iter: int = 1000,
        learning_rate: float = 1e-3,
        tol: float = 1e-8,
        print_every: int = 100,
    ):
        """Fit the model according to the given training data.

        Args:
            max_iter (int, optional): Defaults to 1000.
                Maximum number of iterations of the       solver.
            learning_rate (float, optional): Defaults to 1e-3.
                Initial step size used for learning.
            tol (float, optional): Defaults to 1e-8.
                The tolerance of the solver (iterations stop when the stopping criterion is below it). If not reached the solver does ``max_iter`` iterations.
            print_every (int, optional): Defaults to 100.
                Print history information when ``n_iter`` (iteration number) is a multiple of ``print_every``.
        """
        self.set_optimizer(learning_rate)
        min_cost = np.inf
        best_var = [self.var_phi.detach().numpy(), self.var_exo_mu.detach().numpy()]
        for _iter in range(max_iter):
            self.optimizer.zero_grad()
            loss = self.objective()
            loss.backward()
            self.optimizer.step()
            if _iter == 0:
                prev_obj = loss.item()
            else:
                rel_obj = abs(prev_obj - loss.item()) / abs(prev_obj)
                prev_obj = loss.item()
                converged = rel_obj < tol
                print_info(_iter - 1, print_every, loss.item(), rel_obj)
                if converged:
                    print_info(_iter - 1, 1, loss.item(), rel_obj)
                    break

            if loss.item() < min_cost:
                min_cost = loss.item()
                best_var = [
                    self.var_phi.detach().numpy(),
                    self.var_exo_mu.detach().numpy(),
                ]

        print_info(max_iter, 1, loss.item(), rel_obj)
