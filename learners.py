"""

"""

import numpy as np
import torch
from scipy.linalg import norm, qr, sqrtm

from cumulants import compute_cumulants
from estimate import Hawkes_Shot_Noise_Estimate
from tools import print_info, timefunc


class general_R(Hawkes_Shot_Noise_Estimate):
    """_summary_

    Args:
        Hawkes_Shot_Noise_Estimate (_type_): _description_
    """

    def __init__(self, dim_endo: int, dim_exo: int, device="cpu"):
        super().__init__(dim_endo, dim_exo, device)

    def set_init_values(self, R=None, exo_baseline=None):
        if R is None:
            self.init_R = self.starting_point_R()
        else:
            self.init_R = R
        if exo_baseline is None:
            self.init_exo_baseline = np.zeros(self.dim_exo)
        else:
            self.init_exo_baseline = exo_baseline

    def set_variables(self):
        ## variables
        self.var_R = torch.tensor(
            self.init_R, requires_grad=True, device=self.device, dtype=torch.float32
        )
        self.var_exo_mu = torch.tensor(
            self.init_exo_baseline,
            requires_grad=True,
            device=self.device,
            dtype=torch.float32,
        )
        self.variables = [self.var_R, self.var_exo_mu]

    def set_optimizer(self, learning_rate=1e-3):
        self.optimizer = torch.optim.Adam(self.variables, lr=learning_rate)

    def objective(self, var_R=None, var_exo_mu=None):
        if var_R is None:
            var_R = self.var_R
        if var_exo_mu is None:
            var_exo_mu = self.var_exo_mu

        cs_ratio = self.approximate_optimal_cs_ratio()
        variable_covariance, variable_skewness = compute_cumulants(
            var_R, torch.tensor(self.L_emp, dtype=torch.float32), var_exo_mu
        )
        covariance_divergence = torch.sum(
            torch.square(
                variable_covariance - torch.tensor(self.C_emp, dtype=torch.float32)
            )
        )
        skewness_divergence = torch.sum(
            torch.square(
                variable_skewness - torch.tensor(self.K_emp, dtype=torch.float32)
            )
        )
        loss = cs_ratio * covariance_divergence + (1 - cs_ratio) * skewness_divergence

        ## non-negative constraint
        relu = torch.nn.ReLU()
        endo_baseline = torch.matmul(
            torch.linalg.inv(var_R),
            torch.tensor(self.L_emp, dtype=torch.float32),
        ) - torch.cat((var_exo_mu, var_exo_mu), axis=0)
        mux_loss = torch.sum(torch.square(relu(-var_exo_mu)))
        R_loss = torch.sum(torch.square(relu(-var_R)))
        # mu_loss = torch.sum(torch.square(relu(-endo_baseline)))
        return loss + (mux_loss + R_loss) * 1e6

    @property
    def R(self):
        return self.var_R.detach().numpy()

    @property
    def endo_baseline(self):
        endo_baseline = torch.matmul(
            torch.linalg.inv(self.var_R.detach()),
            torch.tensor(self.L_emp, dtype=torch.float32),
        ) - torch.cat((self.var_exo_mu.detach(), self.var_exo_mu.detach()), axis=0)
        return endo_baseline.numpy()

    @property
    def exo_baseline(self):
        return self.var_exo_mu.detach().numpy()

    @property
    def adjacency(self):
        return (
            torch.eye(self.dim_endo) - torch.linalg.inv(self.var_R.detach())
        ).numpy()

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
        best_var = [self.var_R.detach().numpy(), self.var_exo_mu.detach().numpy()]
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
                    self.var_R.detach().numpy(),
                    self.var_exo_mu.detach().numpy(),
                ]

        print_info(max_iter, 1, loss.item(), rel_obj)


class general_phi(Hawkes_Shot_Noise_Estimate):
    """_summary_

    Args:
        Hawkes_Shot_Noise_Estimate (_type_): _description_
    """

    def __init__(self, dim_endo: int, dim_exo: int, device="cpu"):
        super().__init__(dim_endo, dim_exo, device)

    def set_init_values(self, phi=None, exo_baseline=None):
        if phi is None:
            self.init_phi = np.eye(self.dim_endo) - np.linalg.inv(
                self.starting_point_R()
            )
        else:
            self.init_phi = phi
        if exo_baseline is None:
            self.init_exo_baseline = np.zeros(self.dim_exo)
        else:
            self.init_exo_baseline = exo_baseline

    def set_variables(self):
        ## variables
        self.var_phi = torch.tensor(
            self.init_phi, requires_grad=True, device=self.device, dtype=torch.float32
        )
        self.var_exo_mu = torch.tensor(
            self.init_exo_baseline,
            requires_grad=True,
            device=self.device,
            dtype=torch.float32,
        )
        self.variables = [self.var_phi, self.var_exo_mu]

    def set_optimizer(self, learning_rate=1e-3):
        self.optimizer = torch.optim.Adam(self.variables, lr=learning_rate)

    def objective(self, var_phi=None, var_exo_mu=None):
        if var_phi is None:
            var_phi = self.var_phi
        if var_exo_mu is None:
            var_exo_mu = self.var_exo_mu

        cs_ratio = self.approximate_optimal_cs_ratio()
        R = torch.linalg.inv(torch.eye(self.dim_endo) - var_phi)
        variable_covariance, variable_skewness = compute_cumulants(
            R, torch.tensor(self.L_emp, dtype=torch.float32), var_exo_mu
        )
        covariance_divergence = torch.sum(
            torch.square(
                variable_covariance - torch.tensor(self.C_emp, dtype=torch.float32)
            )
        )
        skewness_divergence = torch.sum(
            torch.square(
                variable_skewness - torch.tensor(self.K_emp, dtype=torch.float32)
            )
        )
        loss = cs_ratio * covariance_divergence + (1 - cs_ratio) * skewness_divergence

        ## non-negative constraint
        relu = torch.nn.ReLU()
        endo_baseline = torch.matmul(
            torch.eye(self.dim_endo) - var_phi,
            torch.tensor(self.L_emp, dtype=torch.float32),
        ) - torch.cat((var_exo_mu, var_exo_mu), axis=0)
        mux_loss = torch.sum(torch.square(relu(-var_exo_mu)))
        R_loss = torch.sum(torch.square(relu(-R)))
        # mu_loss = torch.sum(torch.square(relu(-endo_baseline)))
        return loss + (mux_loss + R_loss) * 1e6

    @property
    def R(self):
        return torch.linalg.inv(
            torch.eye(self.dim_endo) - self.var_phi.detach()
        ).numpy()

    @property
    def endo_baseline(self):
        endo_baseline = torch.matmul(
            torch.eye(self.dim_endo) - self.var_phi.detach(),
            torch.tensor(self.L_emp, dtype=torch.float32),
        ) - torch.cat((self.var_exo_mu.detach(), self.var_exo_mu.detach()), axis=0)
        return endo_baseline.numpy()

    @property
    def exo_baseline(self):
        return self.var_exo_mu.detach().numpy()

    @property
    def adjacency(self):
        return self.var_phi.detach().numpy()


class sparse_phi(Hawkes_Shot_Noise_Estimate):
    """_summary_
    only for 4 x 4 case
    Args:
        Hawkes_Shot_Noise_Estimate (_type_): _description_
    """

    def __init__(self, dim_endo: int, dim_exo: int, device="cpu"):
        super().__init__(dim_endo, dim_exo, device)

    def set_init_values(self, flat_phi=None, exo_baseline=None):
        if flat_phi is None:
            initial_phi = np.eye(self.dim_endo) - np.linalg.inv(self.starting_point_R())
            self.init_phi = torch.Tensor(
                [
                    initial_phi[0, 1],
                    initial_phi[0, 2],
                    initial_phi[3, 1],
                    initial_phi[3, 2],
                ]
            )
        else:
            self.init_phi = flat_phi
        if exo_baseline is None:
            self.init_exo_baseline = np.zeros(self.dim_exo // 2)
        else:
            self.init_exo_baseline = exo_baseline

    def set_variables(self):
        ## variables
        self.var_phi = torch.tensor(
            self.init_phi, requires_grad=True, device=self.device, dtype=torch.float32
        )
        self.var_exo_mu = torch.tensor(
            self.init_exo_baseline,
            requires_grad=True,
            device=self.device,
            dtype=torch.float32,
        )
        self.variables = [self.var_phi, self.var_exo_mu]

    def set_optimizer(self, learning_rate=1e-3):
        self.optimizer = torch.optim.Adam(self.variables, lr=learning_rate)

    def objective(self, var_phi=None, var_exo_mu=None):
        if var_phi is None:
            var_phi = self.var_phi
        if var_exo_mu is None:
            var_exo_mu = self.var_exo_mu

        cs_ratio = self.approximate_optimal_cs_ratio()
        I = torch.eye(self.dim_exo)
        cI = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
        Phi = torch.cat(
            (
                torch.cat((var_phi[0] * cI, var_phi[1] * I), axis=1),
                torch.cat((var_phi[2] * I, var_phi[3] * cI), axis=1),
            ),
            axis=0,
        )
        R = torch.linalg.inv(torch.eye(self.dim_endo) - Phi)

        exo_mu = torch.cat((var_exo_mu, var_exo_mu), axis=0)
        variable_covariance, variable_skewness = compute_cumulants(
            R, torch.tensor(self.L_emp, dtype=torch.float32), exo_mu
        )
        covariance_divergence = torch.sum(
            torch.square(
                variable_covariance - torch.tensor(self.C_emp, dtype=torch.float32)
            )
        )
        skewness_divergence = torch.sum(
            torch.square(
                variable_skewness - torch.tensor(self.K_emp, dtype=torch.float32)
            )
        )
        loss = cs_ratio * covariance_divergence + (1 - cs_ratio) * skewness_divergence

        ## non-negative constraint
        relu = torch.nn.ReLU()
        endo_baseline = torch.matmul(
            torch.eye(self.dim_endo) - Phi,
            torch.tensor(self.L_emp, dtype=torch.float32),
        ) - torch.cat((exo_mu, exo_mu), axis=0)
        mux_loss = torch.sum(torch.square(relu(-var_exo_mu)))
        R_loss = torch.sum(torch.square(relu(-R)))
        # mu_loss = torch.sum(torch.square(relu(-endo_baseline)))
        return loss + (mux_loss + R_loss) * 1e6

    @property
    def R(self):
        return np.linalg.inv(np.eye(self.dim_endo) - self.adjacency)

    @property
    def endo_baseline(self):
        exo = torch.tensor(self.exo_baseline, dtype=torch.float32)
        endo_baseline = torch.matmul(
            torch.eye(self.dim_endo)
            - torch.tensor(self.adjacency, dtype=torch.float32),
            torch.tensor(self.L_emp, dtype=torch.float32),
        ) - torch.cat((exo, exo), axis=0)
        return endo_baseline.numpy()

    @property
    def exo_baseline(self):
        return torch.cat((self.var_exo_mu, self.var_exo_mu), axis=0).detach().numpy()

    @property
    def adjacency(self):
        I = torch.eye(self.dim_exo)
        cI = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
        var_phi = self.var_phi.detach()
        Phi = torch.cat(
            (
                torch.cat((var_phi[0] * cI, var_phi[1] * I), axis=1),
                torch.cat((var_phi[2] * I, var_phi[3] * cI), axis=1),
            ),
            axis=0,
        )
        return Phi.numpy()


class sparse_phi_compress(Hawkes_Shot_Noise_Estimate):
    """_summary_
    only for 4 x 4 case
    Args:
        Hawkes_Shot_Noise_Estimate (_type_): _description_
    """

    def __init__(self, dim_endo: int, dim_exo: int, device="cpu"):
        super().__init__(dim_endo, dim_exo, device)

    def approximate_optimal_cs_ratio(self):
        """Heuristic to set covariance skewness ratio close to its
        optimal value
        """
        norm_sq_C = norm(self.C_emp) ** 2
        norm_sq_K = norm(self.K_emp) ** 2
        return norm_sq_K / (norm_sq_K + norm_sq_C)

    def set_init_values(self, flat_phi=None, exo_baseline=None):
        if flat_phi is None:
            initial_phi = np.eye(self.dim_endo) - np.linalg.inv(self.starting_point_R())
            self.init_phi = torch.Tensor(
                [
                    initial_phi[0, 1],
                    initial_phi[0, 2],
                    initial_phi[3, 1],
                    initial_phi[3, 2],
                ]
            )
        else:
            self.init_phi = flat_phi
        if exo_baseline is None:
            self.init_exo_baseline = np.zeros(self.dim_exo // 2)
        else:
            self.init_exo_baseline = exo_baseline

    def set_variables(self):
        ## variables
        self.var_phi = torch.tensor(
            self.init_phi, requires_grad=True, device=self.device, dtype=torch.float32
        )
        self.var_exo_mu = torch.tensor(
            self.init_exo_baseline,
            requires_grad=True,
            device=self.device,
            dtype=torch.float32,
        )
        self.variables = [self.var_phi, self.var_exo_mu]

    def set_optimizer(self, learning_rate=1e-3):
        self.optimizer = torch.optim.Adam(self.variables, lr=learning_rate)

    def objective(self, var_phi=None, var_exo_mu=None):
        if var_phi is None:
            var_phi = self.var_phi
        if var_exo_mu is None:
            var_exo_mu = self.var_exo_mu

        cs_ratio = self.approximate_optimal_cs_ratio()
        I = torch.eye(self.dim_exo)
        cI = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
        Phi = torch.cat(
            (
                torch.cat((var_phi[0] * cI, var_phi[1] * I), axis=1),
                torch.cat((var_phi[2] * I, var_phi[3] * cI), axis=1),
            ),
            axis=0,
        )
        R = torch.linalg.inv(torch.eye(self.dim_endo) - Phi)

        exo_mu = torch.cat((var_exo_mu, var_exo_mu), axis=0)
        variable_covariance, variable_skewness = compute_cumulants(
            R, torch.tensor(self.L_emp, dtype=torch.float32), exo_mu
        )
        covariance_divergence = torch.sum(
            torch.square(
                variable_covariance - torch.tensor(self.C_emp, dtype=torch.float32)
            )
        )
        skewness_divergence = torch.sum(
            torch.square(
                variable_skewness - torch.tensor(self.K_emp, dtype=torch.float32)
            )
        )
        loss = cs_ratio * covariance_divergence + (1 - cs_ratio) * skewness_divergence

        ## non-negative constraint
        relu = torch.nn.ReLU()
        endo_baseline = torch.matmul(
            torch.eye(self.dim_endo) - Phi,
            torch.tensor(self.L_emp, dtype=torch.float32),
        ) - torch.cat((exo_mu, exo_mu), axis=0)
        mux_loss = torch.sum(torch.square(relu(-var_exo_mu)))
        R_loss = torch.sum(torch.square(relu(-R)))
        # mu_loss = torch.sum(torch.square(relu(-endo_baseline)))
        return loss + (mux_loss + R_loss) * 1e6

    @property
    def R(self):
        return np.linalg.inv(np.eye(self.dim_endo) - self.adjacency)

    @property
    def endo_baseline(self):
        exo = torch.tensor(self.exo_baseline, dtype=torch.float32)
        endo_baseline = torch.matmul(
            torch.eye(self.dim_endo)
            - torch.tensor(self.adjacency, dtype=torch.float32),
            torch.tensor(self.L_emp, dtype=torch.float32),
        ) - torch.cat((exo, exo), axis=0)
        return endo_baseline.numpy()

    @property
    def exo_baseline(self):
        return torch.cat((self.var_exo_mu, self.var_exo_mu), axis=0).detach().numpy()

    @property
    def adjacency(self):
        I = torch.eye(self.dim_exo)
        cI = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
        var_phi = self.var_phi.detach()
        Phi = torch.cat(
            (
                torch.cat((var_phi[0] * cI, var_phi[1] * I), axis=1),
                torch.cat((var_phi[2] * I, var_phi[3] * cI), axis=1),
            ),
            axis=0,
        )
        return Phi.numpy()
