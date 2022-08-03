#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from math import ceil, log
from typing import Any, Callable, Dict, List, Optional

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.fit import fit_gpytorch_model
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.optim.fit import fit_gpytorch_torch
from botorch.utils.sampling import draw_sobol_samples
from gpytorch import settings as gpytorch_settings
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor
from torch.distributions import Normal


def sample_tr_discrete_points(
    X_center: Tensor, length: float, n_discrete_points: int, qmc: bool = False
) -> Tensor:
    r"""Sample points around `X_center` for use in discrete Thompson sampling.

    Sample perturbed points around `X_center` such that the added perturbations
        are sampled from N(0, (length/4)^2) and truncated to be within
        [-length/2, -length/2].

    Args:
        X_center: a `1 x d`-dim tensor containing the center of trust region. `X_center`
            must be normalized to be within `[0, 1]^d`.
        length: edge length of the trust region's hypercube.
        n_discrete_points: number of points to sample for use in discrete TS.
        qmc: boolean indicating whether to use qmc

    Returns:
        Tensor: a `n_discrete_points x d`-dim tensor containing the sampled points.
    """
    d = X_center.shape[1]
    # sample points from N(X_center, (length/4)^2), truncated to be within
    # [X_center-length/2, X_center+length/2].
    # To do this, we sample perturbations from N(0, (length/4)^2) truncated to be
    # within [max(-X_center, -L/2), min(1-X_center, L/2) using the inverse transform
    # and then add these perturbations to X_center.
    sigma = length / 4.0
    if qmc:
        bounds = torch.stack(
            [torch.zeros_like(X_center[0]), torch.ones_like(X_center[0])], dim=0
        )
        u = draw_sobol_samples(bounds=bounds, n=n_discrete_points, q=1).squeeze(1)
    else:
        u = torch.rand(
            (n_discrete_points, d), dtype=X_center.dtype, device=X_center.device
        )
    # compute bounds to sample from
    a = (-X_center).clamp_min(-length / 2.0)
    b = (1 - X_center).clamp_max(length / 2.0)
    # compute z-score of bounds
    alpha = a / sigma
    beta = b / sigma
    normal = Normal(0, 1)
    cdf_alpha = normal.cdf(alpha)
    perturbation = normal.icdf(cdf_alpha + u * (normal.cdf(beta) - cdf_alpha)) * sigma
    X_discrete = X_center + perturbation

    # Clip points that are still outside
    return X_discrete.clamp(0.0, 1.0)


def sample_tr_discrete_points_subset_d(
    best_X: Tensor,
    normalized_tr_bounds: Tensor,
    n_discrete_points: int,
    length: float,
    qmc: bool = False,
    trunc_normal_perturb: bool = False,
    prob_perturb: float = None,
) -> Tensor:
    r"""Sample discrete for TS by perturbing ~20 dims of `best_X`.

    If `trunc_normal_perturb=True`, the perturbed samples are truncated normal
    around `best_X`. Otherwise, these are uniformly distributed in
    `normalize_tr_bounds`.
    """
    assert normalized_tr_bounds.ndim == 2
    d = normalized_tr_bounds.shape[-1]
    if prob_perturb is None:
        # Only perturb a subset of the features
        prob_perturb = min(20.0 / d, 1.0)

    if best_X.shape[0] == 1:
        X_cand = best_X.repeat(n_discrete_points, 1)
    else:
        rand_indices = torch.randint(
            best_X.shape[0], (n_discrete_points,), device=best_X.device
        )
        X_cand = best_X[rand_indices]

    if trunc_normal_perturb:
        pert = sample_tr_discrete_points(
            X_center=X_cand, length=length, n_discrete_points=n_discrete_points, qmc=qmc
        )
        # make sure perturbations are in bounds
        # if X_cand contains pareto points, the perturbed points might not be in the TR
        # TODO: refactor to this into a `project_on_box` helper function T65690436
        pert = torch.min(
            torch.max(pert, normalized_tr_bounds[0]), normalized_tr_bounds[1]
        )
    elif qmc:
        pert = draw_sobol_samples(
            bounds=normalized_tr_bounds, n=n_discrete_points, q=1
        ).squeeze(1)
    else:
        pert = torch.rand(
            n_discrete_points,
            d,
            dtype=normalized_tr_bounds.dtype,
            device=normalized_tr_bounds.device,
        )
        pert = (
            normalized_tr_bounds[1] - normalized_tr_bounds[0]
        ) * pert + normalized_tr_bounds[0]

    # find cases where we are not perturbing any dimensions
    mask = (
        torch.rand(
            n_discrete_points,
            d,
            dtype=normalized_tr_bounds.dtype,
            device=normalized_tr_bounds.device,
        )
        <= prob_perturb
    )
    ind = (~mask).all(dim=-1).nonzero()
    # perturb `n_perturb` of the dimensions
    n_perturb = ceil(d * prob_perturb)
    perturb_mask = torch.zeros(d, dtype=mask.dtype, device=mask.device)
    perturb_mask[:n_perturb].fill_(1)
    for idx in ind:
        mask[idx] = perturb_mask[torch.randperm(d, device=normalized_tr_bounds.device)]
    # Create candidate points
    X_cand[mask] = pert[mask]
    return X_cand


def get_tr_center(X: Tensor, f_obj: Tensor) -> Tensor:
    r"""Find the best point in the trust region.

    Args:
        X: a `n x d`-dim tensor of points
        f_obj: a `n`-dim tensor of scalarized objective values. In the noiseless,
            setting these can be (scalarized) observed values. In the noisy setting,
            these can be (scalarized) posterior means.
    Returns:
        Tensor: a `1 x d`-dim tensor containing the trust region center point.
    """
    if f_obj.ndim != 1:
        raise BotorchTensorDimensionError(
            f"f_obj must have 1 dimension, got {f_obj.ndim} dimensions."
        )
    return X[f_obj.argmax()].view(1, -1)


def get_indices_in_hypercube(
    X_center: Tensor, X: Tensor, length: float, eps: float = 1e-10
) -> Tensor:
    r"""Get indices of observed points inside of trust region.

    Args:
        X_center: a `1 x d`-dim tensor containing the trust region center point.
            `X_center` must be normalized to be within `[0, 1]^d`.
        X: `n x d`-dim tensor containing all data points collected by this trust region.
        length: the edge length of the trust region's hypercube.
        eps: absolute tolerance for evaluating equality (necessary on CUDA).

    Returns:
        A `n'`-dim tensor containing the points inside the hypercube.
    """
    return ((X - X_center).abs() - length / 2 <= eps).all(dim=1).nonzero().view(-1)


def get_fitted_model(
    X: Tensor,
    Y: Tensor,
    use_ard: bool,
    max_cholesky_size: int,
    state_dict: Optional[Dict[str, Tensor]] = None,
    input_transform: Optional[InputTransform] = None,
    outcome_transform: Optional[OutcomeTransform] = None,
    fit_gpytorch_options: Optional[Dict[str, Any]] = None,
) -> Model:
    print("Fitting a model")
    use_fast_mvms = True if X.shape[0] > max_cholesky_size else False
    with gpytorch_settings.fast_computations(
        log_prob=use_fast_mvms,
        covar_root_decomposition=use_fast_mvms,
        solves=use_fast_mvms,
    ):
        models = []
        for i in range(Y.shape[-1]):
            ard_num_dims = X.shape[-1] if use_ard else 1
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=ard_num_dims,
                    lengthscale_constraint=Interval(0.05, 4.0),
                ),
            )
            likelihood = GaussianLikelihood(
                noise_constraint=GreaterThan(1e-6),
                noise_prior=GammaPrior(0.9, 10.0),
            )
            model = SingleTaskGP(
                train_X=X,
                train_Y=Y[:, i : i + 1],
                covar_module=covar_module,
                likelihood=likelihood,
                outcome_transform=outcome_transform.subset_output([i])
                if outcome_transform
                else None,
                input_transform=input_transform,
            )
            models.append(model)

        # TODO: replaced with batched-MO model once MTMVN refactor
        # lands: https://github.com/cornellius-gp/gpytorch/pull/1083
        if Y.shape[-1] > 1:
            model = ModelListGP(*models)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
        else:
            model = models[0]
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

        if state_dict is not None:
            model.load_state_dict(state_dict)
        # 50 iterations appears to be a good compromise between fit and overhead.
        fit_gpytorch_model(mll, options=fit_gpytorch_options)

    if X.is_cuda:
        print(f"after fitting: {torch.cuda.memory_allocated(X.device) / (1000 ** 3)}")
    return model


def coalesce(x1: Optional[Tensor], x2: Optional[Tensor]) -> Optional[Tensor]:
    r"""Helper function the performs a coalesce operation.

    If x1 is not None, it is returned. Otherwise x2 is returned.

    Args:
        x1: a tensor
        x2 a tensor

    Returns:
        A tensor if either of x1 or x2 is not None, otherwise None.
    """
    if x1 is None:
        x1 = x2
    return x1


def decay_function(n: int, n0: int, n_max: int, alpha: float = 1.0) -> float:
    r"""Decay function governed by the used and remaining optimization budget.

    Decay function from:
        Regis R.G., Shoemaker C.A. Combining radial basis function
        surrogates and dynamic coordinate search in high-dimensional
        expensive black-box optimization. Engineering Optimization, 45
        (5) (2013), pp. 529-555

    Args:
        n: number of completed function evaluations
        n0: number of initial function evaluations
        n_max: maximum number of function evaluations (budget)
        alpha: hyperparameter controlling decay

    Returns:
        The probabilty of perturbing a dimension.
    """
    return 1 - alpha * log(n - n0 + 1) / log(n_max - n0 + 1)


def get_constraint_slack_and_feasibility(
    Y: Tensor, constraints: List[Callable[[Tensor], Tensor]]
) -> Tensor:
    r"""Compute feasibility.

    Args:
        Y: A `batch_shape x n x m`-dim tensor of outcomes
        constraints: A list of constraint callables mapping outcomes to the
            constraint slack.

    Returns:
        A `batch_shape x n`-dim boolean tensor indicating whether each example in Y
            is feasible.
    """
    constraint_slack = torch.stack([c(Y) for c in constraints], dim=-1)
    return constraint_slack, (constraint_slack <= 0).all(dim=-1)
