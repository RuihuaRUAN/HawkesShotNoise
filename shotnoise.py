"""
"""
from abc import ABC, abstractmethod

import numpy as np
import torch
from cumulants import compute_cumulants
from tick.hawkes import HawkesCumulantMatching
from tick.hawkes.inference.base import LearnerHawkesNoParam


class Hawkes_Shot_Noise:
    def __init__(self, dim_endo: int, dim_exo: int, device="cpu"):
        self.dim_endo = dim_endo
        self.dim_exo = dim_exo
        self.device = device

    def set_parameters(self, **kwarg):
        """_summary_"""
        self.set_exo_baseline(kwarg["exo_baseline"])
        self.set_endo_baseline(kwarg["endo_baseline"])
        if not isinstance(kwarg["beta"], list):
            betas = [kwarg["beta"]]
        else:
            betas = kwarg["beta"]
        self.set_betas(betas)

        if len(np.array(kwarg["alpha"]).shape) == 2:
            alphas = [kwarg["alpha"]]
        else:
            alphas = kwarg["alpha"]
        self.set_adjacencys(alphas)

        if not isinstance(kwarg["delay"], list):
            delays = [0, kwarg["delay"]]
        else:
            delays = kwarg["delay"]
        self.set_delays(delays)

    def set_exo_baseline(self, exo_baseline: list):
        assert (
            len(exo_baseline) == self.dim_exo
        ), f"length of exo_baseline should be {self.dim_exo}"
        self.exo_baselines = exo_baseline

    def set_betas(self, decays: list):
        self.decays = decays

    def set_delays(self, delays: list):
        print(delays)
        assert (
            len(delays) == self.dim_endo
        ), f"length of delays should be {self.dim_endo}"
        self.delays = delays

    def set_endo_baseline(self, endo_baseline: list):
        assert (
            len(endo_baseline) == self.dim_endo
        ), f"length of endo_baseline should be {self.dim_endo}"
        self.endo_baseline = endo_baseline

    def set_adjacencys(self, alphas):
        assert len(alphas) == len(
            self.decays
        ), "alphas and decays should have the same length"
        assert self.dim_endo == len(alphas[0]), ""
        self.adjacencys = alphas

    @property
    def kernel_norm(self):
        return np.sum(self.adjacencys, axis=0)

    @property
    def R(self):
        return np.linalg.inv(np.eye(self.dim_endo) - self.kernel_norm)

    @property
    def spectral_radius(self):
        return np.absolute(
            np.linalg.eigvals(
                np.maximum(self.kernel_norm, np.zeros((self.dim_endo, self.dim_endo)))
            )
        ).max()

    def compute_cumulants(self):
        self.L = self.R @ (
            self.endo_baseline + np.block([self.exo_baselines, self.exo_baselines])
        )
        self.C, self.K = compute_cumulants(
            torch.Tensor(self.R), torch.Tensor(self.L), torch.Tensor(self.exo_baselines)
        )

    # _______________________________________________________
    def set_data(self, times: list, T: float = None):
        self.timestamps = times
        if T is None:
            self.T = max([ts[-1] for ts in times])
        else:
            self.T = T

    @property
    def n_realizations(self):
        return len(self.timestamps)

"""
"""
from abc import ABC, abstractmethod

import numpy as np
import torch
from cumulants import compute_cumulants
from tick.hawkes import HawkesCumulantMatching
from tick.hawkes.inference.base import LearnerHawkesNoParam


class Hawkes_Shot_Noise:
    def __init__(self, dim_endo: int, dim_exo: int, device="cpu"):
        self.dim_endo = dim_endo
        self.dim_exo = dim_exo
        self.device = device

    def set_parameters(self, **kwarg):
        """_summary_"""
        self.set_exo_baseline(kwarg["exo_baseline"])
        self.set_endo_baseline(kwarg["endo_baseline"])
        if not isinstance(kwarg["beta"], list):
            betas = [kwarg["beta"]]
        else:
            betas = kwarg["beta"]
        self.set_betas(betas)

        if len(np.array(kwarg["alpha"]).shape) == 2:
            alphas = [kwarg["alpha"]]
        else:
            alphas = kwarg["alpha"]
        self.set_adjacencys(alphas)

        if not isinstance(kwarg["delay"], list):
            delays = [0, kwarg["delay"]]
        else:
            delays = kwarg["delay"]
        self.set_delays(delays)

    def set_exo_baseline(self, exo_baseline: list):
        assert (
            len(exo_baseline) == self.dim_exo
        ), f"length of exo_baseline should be {self.dim_exo}"
        self.exo_baselines = exo_baseline

    def set_betas(self, decays: list):
        self.decays = decays

    def set_delays(self, delays: list):
        print(delays)
        assert (
            len(delays) == self.dim_endo
        ), f"length of delays should be {self.dim_endo}"
        self.delays = delays

    def set_endo_baseline(self, endo_baseline: list):
        assert (
            len(endo_baseline) == self.dim_endo
        ), f"length of endo_baseline should be {self.dim_endo}"
        self.endo_baseline = endo_baseline

    def set_adjacencys(self, alphas):
        assert len(alphas) == len(
            self.decays
        ), "alphas and decays should have the same length"
        assert self.dim_endo == len(alphas[0]), ""
        self.adjacencys = alphas

    @property
    def kernel_norm(self):
        return np.sum(self.adjacencys, axis=0)

    @property
    def R(self):
        return np.linalg.inv(np.eye(self.dim_endo) - self.kernel_norm)

    @property
    def spectral_radius(self):
        return np.absolute(
            np.linalg.eigvals(
                np.maximum(self.kernel_norm, np.zeros((self.dim_endo, self.dim_endo)))
            )
        ).max()

    def compute_cumulants(self):
        self.L = self.R @ (
            self.endo_baseline + np.block([self.exo_baselines, self.exo_baselines])
        )
        self.C, self.K = compute_cumulants(
            torch.Tensor(self.R), torch.Tensor(self.L), torch.Tensor(self.exo_baselines)
        )

    # _______________________________________________________
    def set_data(self, times: list, T: float = None):
        self.timestamps = times
        if T is None:
            self.T = max([ts[-1] for ts in times])
        else:
            self.T = T

    @property
    def n_realizations(self):
        return len(self.timestamps)
