#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor
from torch.nn import Module


class BenchmarkFunction(Module):
    r"""This is a wrapper that wraps the function callable and implements additional
    functionality for other benchmarking related needs, e.g., saving noise free
    observations.
    """

    def __init__(
        self,
        base_f: Callable,
        num_outputs: Union[int, List[int]],
        ref_point: Tensor,
        dim: int,
        tkwargs: Dict[str, Union[torch.dtype, torch.device]],
        negate: bool = True,
        observation_noise_std: Optional[List[float]] = None,
        observation_noise_bias: Optional[List[float]] = None,
    ):
        super().__init__()
        self.base_f = base_f
        self.num_outputs = num_outputs
        self.ref_point = ref_point
        self.dim = dim
        self.tkwargs = tkwargs
        self.negate = negate
        # Make observation noise into Tensors.
        if observation_noise_std is None:
            self.observation_noise_std = torch.zeros(num_outputs, **tkwargs)
        elif len(observation_noise_std) != num_outputs:
            raise ValueError("`observation_noise_std` must be of size `num_outputs`!")
        else:
            self.observation_noise_std = torch.tensor(observation_noise_std, **tkwargs)
        if observation_noise_bias is None:
            self.observation_noise_bias = torch.zeros(num_outputs, **tkwargs)
        elif len(observation_noise_bias) != num_outputs:
            raise ValueError("`observation_noise_bias` must be of size `num_outputs`!")
        else:
            self.observation_noise_bias = torch.tensor(
                observation_noise_bias, **tkwargs
            )
        # Containers for storing observations.
        self.X = torch.zeros(0, dim, **tkwargs)
        num_outputs = [num_outputs] if type(num_outputs) is int else num_outputs
        self.Y_noise_free = torch.zeros(0, *num_outputs, **tkwargs)

        # Containers for saving outputs.
        self.noise_free_hv = []
        self.noise_free_pareto_X = []
        self.noise_free_pareto_Y = []

    def _evaluate_base_f(self, X: Tensor) -> Tensor:
        Y = self.base_f(X).to(**self.tkwargs)
        if self.negate:
            Y = -Y
        if self.num_outputs == 1:
            Y = Y.view(-1, 1)
        return Y

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the function and return the possibly noisy observations. Also
        calls all side-effect type functionality, and saves noise-free observations.
        """
        self.X = torch.cat([self.X, X], dim=0)
        noise_free_eval = self._evaluate_base_f(X)
        self.Y_noise_free = torch.cat([self.Y_noise_free, noise_free_eval], dim=0)
        return (
            noise_free_eval
            + torch.randn_like(noise_free_eval) * self.observation_noise_std
            + self.observation_noise_bias
        )

    def record_current_pf_and_hv(
        self,
        obj: Callable,
        constraints: Optional[List] = None,
    ) -> None:
        r"""Records the current noise-free PF and HV."""
        noise_free_obj = obj(self.Y_noise_free)
        if constraints is not None:
            # Set the infeasible points to ref_point.
            constraint_value = torch.stack(
                [c(self.Y_noise_free) for c in constraints], dim=-1
            )
            feas = (constraint_value <= 0.0).all(dim=-1)
            noise_free_obj[~feas] = self.ref_point
        pareto_mask = is_non_dominated(noise_free_obj)
        self.noise_free_pareto_X.append(self.X[pareto_mask].tolist())
        self.noise_free_pareto_Y.append(noise_free_obj[pareto_mask].tolist())
        partitioning = DominatedPartitioning(
            ref_point=self.ref_point, Y=noise_free_obj[pareto_mask]
        )
        nf_hv = partitioning.compute_hypervolume().item()
        print(f"Current noise-free hypervolume: {nf_hv:.3f}")
        self.noise_free_hv.append(nf_hv)

    def record_all_hvs(
        self,
        obj: Callable,
        constraints: Optional[List] = None,
    ) -> None:
        r"""Loop over observations one by one and record the resulting HVs."""
        noise_free_obj = obj(self.Y_noise_free)
        if constraints is not None:
            constraint_value = torch.stack(
                [c(self.Y_noise_free) for c in constraints], dim=-1
            )
            feas = (constraint_value <= 0.0).all(dim=-1)
            noise_free_obj[~feas] = self.ref_point

        self.all_noise_free_hvs = torch.zeros(len(self.Y_noise_free), **self.tkwargs)
        partitioning = DominatedPartitioning(ref_point=self.ref_point)
        for i in range(len(self.Y_noise_free)):
            partitioning.update(Y=noise_free_obj[i : i + 1])
            self.all_noise_free_hvs[i] = partitioning.compute_hypervolume()

    def get_outputs(self) -> Dict:
        return {
            "noise_free_hv": self.noise_free_hv,
            "noise_free_pareto_X": self.noise_free_pareto_X,
            "noise_free_pareto_Y": self.noise_free_pareto_Y,
            "all_hvs": self.all_noise_free_hvs.tolist(),
        }
