#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

import dataclasses
from abc import abstractmethod, abstractproperty, ABC
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from botorch.acquisition.objective import AcquisitionObjective
from botorch.models.transforms.input import (
    ChainedInputTransform,
    Normalize,
)
from botorch.models.transforms.outcome import Standardize
from botorch.sampling import MCSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.objective import get_objective_weights_transform
from botorch.utils.sampling import draw_sobol_normal_samples
from botorch.utils.transforms import normalize
from morbo.utils import (
    get_constraint_slack_and_feasibility,
    get_fitted_model,
    get_indices_in_hypercube,
)
from scipy.stats.mstats import winsorize
from torch import Tensor
from torch.nn import Module


@dataclasses.dataclass
class TurboHParams:
    r"""Hyperparameters for TuRBO.

    Args:
        length_init: Initial edge length for the trust region
        length_min: Minimum edge length
        length_max: Maximum edge length
        success_streak: Number of consecutive successes necessary to increase length
        failure_streak: Number of consecutive failures necessary to decrease length
        n_trust_regions: Total number of trust regions. This is used in failure
            accounting.
        batch_size: Batch size
        eps: The minimum percent improvement in objective that qualifying as a
            "success".
        use_ard: Whether to use ARD when fitting GPs for this trust region.
        trim_trace: A boolean indicating whether to use all data from the trust
            region's trace for model fitting.
        verbose: A boolean indicating whether to print verbose output
        max_tr_size: The maximum number of points in a trust region. This can be
            used to avoid memory issues.
        min_tr_size: The minimum number of points allowed in the trust region. If there
            are too few points in the TR, sobol samples will be used to refill the TR.
        qmc: Whether to use qmc when possible or not
        sample_subset_d: Whether to perturb subset of the dimensions for generating
            discrete X
        track_history: If true, uses the historically observed points and points
            from other trs when the trust regions moves
        fixed_scalarization: If set, a fixed scalarization weight would be used
        max_cholesky_size: Maximum number of training points for which we will use a
            Cholesky decomposition.
        raw_samples: number of discrete points for Thompson Sampling
        n_initial_points: Number of initial sobol points
        n_restart_points: Number of sobol points to evaluate when we restart a TR if
            `init_with_add_sobol=True`
        max_reference_point: The maximum reference point (i.e. this is the closest that
            the reference point can get to the pareto front)
        hypervolume: Whether to use a hypervolume objective for MOO
        winsor_pct: Percentage of worst outcome observations to winsorize
        trun_normal_perturb: Whether to generate discrete points for Thompson sampling
            by perturbing with samples from a zero-mean truncated Gaussian.
        decay_restart_length_alpha: Factor controlling how much to decay (over time)
            the initial TR length when restarting a trust region.
        switch_strategy_freq: The frequency (in terms of function evaluations) at which
            the strategy should be switched between using a hypervolume objective and
            using random scalarizations.
        tabu_tenure: Number of BO iterations for which a previous X_center is considered
            tabu. A previous X_center is only considered tabu if it was the TR center
            when the TR was terminated.
        fill_strategy: (DEPRECATED) Set to "sobol" to fill trust regions with Sobol
            points until there are at least min_tr_size in each trust region. Set to
            "closest" to include the closest min_tr_size instead. Using "closest" is
            strongly recommended as filling with Sobol points may be very
            sample-inefficient.
        use_noisy_trbo: A boolean denoting whether to expect noisy observations and
            use model predictions for trust region computations.
        use_simple_rff: If True, the GP predictions are replaced with predictions from a
            1-RFF draw during candidate generation.
        batch_limit: default maximum size of joint posterior for TS, lower is less memory intensive,
            while higher is more memory intensive. default: [0,10] (for full posterior sizes) but
            only drawing 10 samples at once.
        use_approximate_hv_computations: Whether to use approximate hypervolume computations. This
            may speed up candidate generation, especially when there are more than 2 objectives.
        approximate_hv_alpha: The value of alpha passed to NondominatedPartitioning. The larger
            the value of alpha is, the faster and less accurate hv computations will be used, see
            NondominatedPartitioning for more details. This parameter only has an effect if
            use_approximate_hv_computations is set to True.
        pred_batch_limit: The maximum batch size to use for `_get_predictions`.
        infer_reference_point: Set this to true if you want to explore the entire Pareto frontier.
            `max_reference_point` will be ignored and we will rely on `infer_reference_point` to infer
            the reference point before generating new candidates.
        fit_gpytorch_options: Options for fitting GPs
        restart_hv_scalarizations: Whether to sample restart points using random scalarizations
    """

    length_init: float = 0.8
    length_min: float = 0.01
    length_max: float = 1.6
    success_streak: int = 10_000
    failure_streak: Optional[int] = None
    n_trust_regions: int = 5
    batch_size: int = 100
    eps: float = 1e-3
    use_ard: bool = True
    trim_trace: bool = True
    verbose: bool = False
    max_tr_size: int = 2000
    min_tr_size: int = 250
    qmc: bool = True
    sample_subset_d: bool = True
    track_history: bool = True
    fixed_scalarization: bool = False
    max_cholesky_size: int = 50_000  # Not using Cholesky causes stability issues
    raw_samples: int = 4096
    n_initial_points: int = 1000
    n_restart_points: int = 0
    max_reference_point: Optional[List[float]] = None
    hypervolume: bool = True
    winsor_pct: float = 5.0  # this will winsorize the bottom 5%
    trunc_normal_perturb: bool = False
    decay_restart_length_alpha: float = 0.5
    switch_strategy_freq: Optional[int] = None
    tabu_tenure: int = 100
    fill_strategy: str = "closest"
    use_noisy_trbo: bool = False
    use_simple_rff: bool = False
    batch_limit: List = None
    use_approximate_hv_computations: bool = False
    approximate_hv_alpha: Optional[float] = None  # Note: Should be >= 0.0
    pred_batch_limit: int = 1024
    infer_reference_point: bool = False
    fit_gpytorch_options: Optional[Dict[str, Any]] = None  # {"maxiter": 50}
    restart_hv_scalarizations: bool = False

    @classmethod
    def from_dict(cls, tr_hparams: Dict) -> None:
        r"""Construct a TurboHParams object from a dict.

        This automatically filters unexpected keys in order to allow deleting
        keys that are no longer being used.

        Args:
            tr_hparams: Dict of hyperparameters
        """
        expected_keys = {f.name for f in dataclasses.fields(cls)}
        received_keys = set(tr_hparams.keys())
        unexpected_keys = received_keys - expected_keys
        if unexpected_keys:
            print(
                warning(f"Got unexpected tr_hparams keys, ignoring: {unexpected_keys}")
            )
        filtered_keys = expected_keys & received_keys
        return cls(**{k: tr_hparams[k] for k in filtered_keys})


class TrustRegion(ABC, Module):
    r"""A trust region object.

    This is a variation of the TuRBO algorithm presented in:

    D. Eriksson, M. Pearce, J.R. Gardner, R. Turner, M. Poloczek.
    Scalable Global Optimization via Local Bayesian Optimization.
    NeurIPS 2019. https://arxiv.org/pdf/1910.01739.pdf

    We adapt the original algorithm by trimming the observed data used in the
    trust region to only include data within the hypercube with edge length
    (2 * length) around the trust region center.

    Args:
        X_init: a `n x d`-dim tensor of points
        Y_init: a `n x m`-dim tensor of observations
        bounds: a `2 x d`-dim tensor of bounds
        tr_hparams: hyperparameters for turbo
        objective: An objective function that selects the objectives
            (a subset of all modeled outcomes).
        constraints: List of potential outcome constraints
        extra_buffers: Additional buffers that should be registered
    """

    def __init__(
        self,
        X_init: Tensor,
        Y_init: Tensor,
        bounds: Tensor,
        tr_hparams: TurboHParams,
        objective: AcquisitionObjective,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        extra_buffers: Optional[Dict[str, Union[None, Tensor]]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.X = X_init
        self.Y = Y_init
        # NOTE: if `use_noisy_trbo = True`, this stores the this stores the model
        # predictions. Otherwise, it is set to `Y` to reduce the use of
        # conditional statements.
        self.Y_estimate = None
        self.dim = self.X.shape[-1]
        self.bounds = bounds
        self.tr_hparams = tr_hparams
        self._objective = objective
        self.constraints = constraints
        self.register_buffer("X_center", None)
        self.register_buffer("X_center_normalized", None)
        self.register_buffer("Y_center", None)
        length = torch.tensor(
            tr_hparams.length_init, device=bounds.device, dtype=bounds.dtype
        )
        self.register_buffer("length", length)
        self.register_buffer(
            "n_successes", torch.tensor(0, device=bounds.device, dtype=torch.int64)
        )
        self.register_buffer(
            "n_failures", torch.tensor(0, device=bounds.device, dtype=torch.int64)
        )
        self._register_extra_buffers(extra_buffers)
        self._reset_counters()
        self.model = None
        self.model_training_data = [None, None]
        self.update(
            X_all=X_init,
            Y_all=Y_init,
            **kwargs,
        )

    def _register_extra_buffers(self, kwargs: Dict[str, Union[None, Tensor]]) -> None:
        if kwargs:
            for k, v in kwargs.items():
                self.register_buffer(k, v)

    def _check_for_updates_to_model_training_data(self) -> bool:
        r"""Return True if the model training data was updated."""
        if self.model is None or self.model_training_data[0] is None:
            return True
        if torch.equal(self.model_training_data[0], self.X) and torch.equal(
            self.model_training_data[1].to(self.Y), self.Y
        ):
            return False
        return True

    def update_model(self) -> bool:
        r"""Update the model using available data.

        Returns:
            A boolean representing whether the model was updated.
        """
        # Only update the model if the training data has changed
        if self._check_for_updates_to_model_training_data():
            self.model_training_data = [self.X, self.Y]
            # Scale X from problem space bounds to [0, 1]
            intf = Normalize(d=self.dim, bounds=self.bounds)
            # X really occupies a potentially small hypercube in [0, 1]
            # Scale X using the bounds of that hypercube so that the bounds of
            # that local hypercube are expanded to [0, 1].
            # Only do this if `X_center` exists. Fitting a global model otherwise.
            if self.X_center is not None:
                intf2 = Normalize(d=self.dim, bounds=self.get_bounds(model_space=True))
                intf = ChainedInputTransform(intf1=intf, intf2=intf2)
            # Standardize Y
            winsorized_Y = torch.from_numpy(
                winsorize(
                    self.Y.cpu().numpy(),
                    limits=(self.tr_hparams.winsor_pct / 100.0, None),
                    axis=0,
                )
            ).to(self.Y)
            octf = Standardize(m=self.Y.shape[-1])

            if self.model is not None:
                self.model.train()
                torch.cuda.empty_cache()

            self.model = get_fitted_model(
                X=self.X,
                Y=winsorized_Y,
                use_ard=self.tr_hparams.use_ard,
                max_cholesky_size=self.tr_hparams.max_cholesky_size,
                input_transform=intf,
                outcome_transform=octf,
                fit_gpytorch_options=self.tr_hparams.fit_gpytorch_options,
            )
            return True
        return False

    def _get_predictions(
        self,
        X: Tensor,
        sampler: Optional[MCSampler] = None,
    ) -> Tensor:
        r"""Get the model predictions corresponding to the given inputs.

        Args:
            X: An `n x d`-dim tensor of inputs to get the predictions for.
            sampler: If given, this sampler is used to get the predictions. If None, this will return the posterior mean.

        Returns:
            An `n x m`-dim tensor of predictions.
        """
        base_samples = None
        with torch.no_grad():
            predictions = []
            for x_ in X.split(self.tr_hparams.pred_batch_limit):
                # Unsqueeze the q-dim to avoid large q-batch posterior computations when using a GP.
                # This should not matter since we're using (MC) posterior mean here.
                x_ = x_.unsqueeze(-2)
                posterior = self.model.posterior(x_)
                if sampler is not None:
                    # If given, use the sampler.
                    samples = sampler(posterior)
                else:
                    # If no sampler return the posterior mean.
                    pm = posterior.mean
                    predictions.append(pm.squeeze(-2) if gp_flag else pm)
                    continue

                samples = samples.squeeze(-2)
                # Record the sample mean as the MC prediction.
                predictions.append(samples.mean(dim=0))
            return torch.cat(predictions, dim=0)

    def update(
        self,
        X_all: Tensor,
        Y_all: Tensor,
        X_new: Optional[Tensor] = None,
        Y_new: Optional[Tensor] = None,
        invalid_centers: Optional[Tensor] = None,
        update_streaks: bool = True,
        global_model: Optional[Any] = None,
        **kwargs,
    ) -> bool:
        r"""Update trust region.

        Adjust the edge length, and update the data in the trust region for model
            fitting.

        Args:
            X_all: `n x d`-dim tensor of all points
            Y_all: `n x m`-dim tensor of all observations
            X_new: `q x d`-dim tensor of new points
            Y_new: `q x m`-dim tensor new observations
            invalid_centers: a `k x d`-dim tensor of points that cannot be used as
                center. Currently only used in `HypervolumeTrustRegion`.
            update_streaks: a boolean indicating whether the success/failure streaks
                should be updated. This should be True unless we are filling in the
                trust region with sobol points to ensure it contains enough points.

        Returns:
            A boolean indicating whether to restart the TR.
        """
        # Append new data
        if X_new is not None:
            Y_new = Y_new.to(self.Y)

            self.X = torch.cat((self.X, X_new), dim=0)
            self.Y = torch.cat((self.Y, Y_new), dim=0)
            n_new = X_new.shape[0]
            if self.tr_hparams.use_noisy_trbo:
                # update the model with the new data
                self.update_model()
                # update the Y_estimates
                self.Y_estimate = self._get_predictions(self.X)
            else:
                self.Y_estimate = self.Y
            if update_streaks:
                # currently this assumes noiseless observations
                if self._has_improved_objective(n_new):
                    self.n_successes.add_(1)
                    self.n_failures.zero_()
                else:
                    self.n_successes.zero_()
                    # NOTE: TuRBO-1 counts batches, but we always add #evals to
                    # be consistent with the way we use the success streak.
                    self.n_failures.add_(n_new)

                if self.n_successes >= self.tr_hparams.success_streak:
                    # Expand trust region
                    self.length.fill_(
                        min(2.0 * self.length, self.tr_hparams.length_max)
                    )
                    self._reset_counters()

                elif self.n_failures >= self.tr_hparams.failure_streak:
                    # Shrink trust region
                    self.length.div_(2.0)
                    self._reset_counters()
                    # length is too small, restart the TR
                    if self.length < self.tr_hparams.length_min:
                        return True
        if self.tr_hparams.use_noisy_trbo and self.model is None:
            # We need the `Y_estimate` to update the center. If no prior model
            # is available to form the estimates, we fit a global model.
            self.X = X_all
            self.Y = Y_all
            self.update_model()
            self.Y_estimate = self._get_predictions(self.X)
        else:
            self.Y_estimate = self.Y
        self._update_center_and_best_points(
            invalid_centers=invalid_centers,
            **kwargs,
        )
        if self.tr_hparams.track_history:
            self._update_training_data(X_all=X_all, Y_all=Y_all)
        else:
            self._update_training_data(X_all=self.X, Y_all=self.Y)

        # Update the model with the new training data
        if global_model:
            self.model = global_model
        elif self.update_model() and self.tr_hparams.use_noisy_trbo:
            # If the model was updated, update the `Y_estimate`, `Y_center`,
            # and `best_Y` with updated model predictions.
            self.Y_estimate = self._get_predictions(self.X)
            center_idx = (self.X == self.X_center).all(dim=-1).nonzero()[0].item()
            self._set_center_and_best_points(center_idx)

        if self.tr_hparams.verbose and X_new is not None:
            print(f"Num points in TR: {self.X.shape[0]}")
            print(f"length: {self.length}")
        return False

    def _set_Y_center(self, center_idx: int) -> None:
        self.Y_center = self.Y_estimate[center_idx : center_idx + 1]

    def _set_center_and_best_points(self, center_idx: int) -> None:
        self.X_center = self.X[center_idx : center_idx + 1]
        self._set_Y_center(center_idx)
        self.best_X = self.X_center
        self.best_Y = self.Y_center

    @abstractproperty
    def objective(self) -> Callable[[Tensor], Tensor]:
        pass

    @abstractmethod
    def _update_center_and_best_points(
        self,
        invalid_centers: Optional[Tensor] = None,
        **kwargs,
    ) -> None:
        pass

    @abstractmethod
    def _has_improved_objective(
        self,
        n_new: int,
    ) -> bool:
        pass

    def _update_training_data(self, X_all: Tensor, Y_all: Tensor) -> None:

        X_all_normalized = normalize(X=X_all, bounds=self.bounds)

        if self.tr_hparams.trim_trace:
            tr_indices = get_indices_in_hypercube(
                X_center=self.X_center_normalized,
                X=X_all_normalized,
                # used 2x the TR candidate space for model fitting
                length=self.length * 2,
            )
        else:
            tr_indices = torch.arange(len(X_all), device=X_all.device)

        if self.tr_hparams.fill_strategy != "closest":
            raise ValueError("Only fill_strategy=='closest' is currently supported.")

        dists = torch.cdist(X_all_normalized, self.X_center_normalized).squeeze()
        if X_all_normalized.shape[0] > 1:
            if tr_indices.shape[0] < self.tr_hparams.min_tr_size:
                # Pick closest min(n_total, min_tr_size) points
                npts = min(len(X_all), self.tr_hparams.min_tr_size)
                _, tr_indices = torch.topk(dists, npts, largest=False)
            elif tr_indices.shape[0] > self.tr_hparams.max_tr_size:
                # Pick closest max_tr_size points
                _, tr_indices = torch.topk(
                    dists, self.tr_hparams.max_tr_size, largest=False
                )
        self.X = X_all[tr_indices]
        self.Y = Y_all[tr_indices]
        if not (self.X_center == self.X).all(dim=-1).any():
            self.X[-1] = self.X_center
            self.Y[-1] = self.Y_center

    def _reset_counters(self) -> None:
        self.n_successes.zero_()
        self.n_failures.zero_()

    def get_bounds(self, model_space: bool = False) -> None:
        normalized_X_center = normalize(self.X_center, bounds=self.bounds)
        # the model space is 2X the TR candidate space if we are trimming the trace,
        # otherwise the modeling space is [0, 1]^d
        half_length = self.length.clone()
        if not self.tr_hparams.trim_trace and model_space:
            d = self.X_center.shape[-1]
            return torch.cat([torch.zeros(1, d), torch.ones(1, d)], dim=0).to(
                self.X_center
            )
        if not model_space:  # The trust region is [x - L / 2, x + L / 2]
            half_length /= 2
        return torch.cat(
            [normalized_X_center - half_length, normalized_X_center + half_length],
            dim=0,
        ).clamp(0.0, 1.0)


class ScalarizedTrustRegion(TrustRegion):
    r"""A scalarized trust region object.

    NOTE: If `self.tr_hparams.use_noisy_trbo is True`, this uses estimates of Y
    (`Y_estimate`) to determine the objective improvement and the TR center.

    Args:
        X_init: a `n x d`-dim tensor of points
        Y_init: a `n x m`-dim tensor of observations
        bounds: a `2 x d`-dim tensor of bounds
        tr_hparams: hyperparameters for turbo
        objective: An objective function that selects the objectives
            (a subset of all modeled outcomes).
        constraints: List of potential outcome constraints
        weights: a `m`-dim tensor of weights for the scalarized objective
    """

    def __init__(
        self,
        X_init: Tensor,
        Y_init: Tensor,
        bounds: Tensor,
        tr_hparams: TurboHParams,
        objective: Callable[[Tensor], Tensor],
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        weights: Optional[Tensor] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            X_init=X_init,
            Y_init=Y_init,
            bounds=bounds,
            tr_hparams=tr_hparams,
            objective=objective,
            constraints=constraints,
            extra_buffers={"scalarization_weights": weights},
            **kwargs,
        )

    def _gen_objective(self) -> None:
        """This creates a callable that applies _objective to select
        the relevant objectives and then scalarizes the objectives."""
        if self.scalarization_weights is not None:

            def objective(Y):
                obj = get_objective_weights_transform(
                    weights=self.scalarization_weights
                )
                return obj(self._objective(Y))

        else:
            objective = self._objective
        self._scalarization_objective = objective

    @property
    def objective(self) -> Callable[[Tensor], Tensor]:
        if not hasattr(self, "_scalarization_objective"):
            self._gen_objective()  # this will set self._scalarization_objective
        return self._scalarization_objective

    def _get_max_previous_objective(self, n_new: int) -> Tensor:
        r"""Get the maximum objective value from the previous observations.

        Args:
            n_new: Number of new points. These are excluded to get the old observations.

        Returns:
            A tensor denoting the maximum previous objective value.
        """
        if self.tr_hparams.use_noisy_trbo:
            # find the best objective value from previously evaluated points
            Y = self.Y_estimate[:-n_new].clone()
            if self.constraints is None:
                return self.objective(Y).max()
            else:
                constraint_slack, feas = get_constraint_slack_and_feasibility(
                    Y=Y, constraints=self.constraints
                )
                if feas.any():
                    # Return the best feasible point if any.
                    return Y[feas].max()
                else:
                    # Return the point with minimum constraint violation.
                    return Y[
                        constraint_slack.clamp_min(0.0).sum(dim=-1).argmin().item()
                    ]
        else:
            return self.objective(self.best_Y).clone()

    def _has_improved_objective(self, n_new: int) -> bool:
        """Determine whether new batch of n_new points has improved the objective.

        For a scalarized trust region we define improvement through as:
            new_obj > old_obj + eps * |old_obj|
        """
        k = self.Y_estimate.shape[0]
        Y_new = self.Y_estimate[-n_new:]
        if self.constraints is not None:
            constraint_slack, feas = get_constraint_slack_and_feasibility(
                Y=Y_new, constraints=self.constraints
            )
        else:
            feas = None

        max_prev_obj = self._get_max_previous_objective(n_new, k)
        # NOTE: objective does not necessarily create a new tensor, so we
        # need to clone here.
        new_obj = self.objective(Y_new).clone()
        if feas is not None:
            if feas.any():
                # if there is at least one feasible point,
                # set infeasible points to have -inf as the objective value
                new_obj[~feas] = float("-inf")
            else:
                # No new feasible points
                # Check if previous best point was feasible
                best_prev_con_slack, _ = get_constraint_slack_and_feasibility(
                    Y=self.best_Y, constraints=self.constraints
                )
                best_prev_tot_violation = best_prev_con_slack.clamp_min(0.0).sum(-1)
                if best_prev_tot_violation <= 0:
                    # previous best point was feasible
                    return False
                # define success to be getting closer to a feasible solution
                total_violation = constraint_slack.clamp_min(0.0).sum(dim=-1)
                return (best_prev_tot_violation > total_violation.min()).item()
        if new_obj.max() > max_prev_obj + self.tr_hparams.eps * max_prev_obj.abs():
            return True
        return False

    def _update_center_and_best_points(
        self,
        invalid_centers: Optional[Tensor] = None,
    ) -> None:
        """Update center and best points."""
        Y_center_prev = None if self.Y_center is None else self.Y_center.clone()
        # NOTE: objective does not necessarily create a new tensor, so we
        # need to clone here.
        obj = self.objective(self.Y_estimate).clone()

        center_idx = None
        if self.constraints is not None:
            # set infeasible points to have -inf as the objective value
            constraint_slack, feas = get_constraint_slack_and_feasibility(
                Y=self.Y_estimate, constraints=self.constraints
            )
            if feas.any():
                obj[~feas] = float("-inf")
            else:
                # if there are no feasible points, set point with minimum total
                # violation to be the TR center
                center_idx = constraint_slack.clamp_min(0.0).sum(dim=-1).argmin().item()
        if center_idx is None:
            center_idx = obj.argmax().item()
        self._set_center_and_best_points(center_idx=center_idx)
        if self.tr_hparams.verbose and (
            Y_center_prev is None
            or not torch.equal(Y_center_prev.to(self.Y_center), self.Y_center)
        ):
            print(f"New metrics for center: {self.objective(self.Y_center)}")

        self.X_center_normalized = normalize(X=self.X_center, bounds=self.bounds)

    def _set_Y_center(self, center_idx: int) -> None:
        self.Y_center = self.Y_estimate[center_idx : center_idx + 1]


class HypervolumeTrustRegion(TrustRegion):
    r"""A hypervolume trust region object.

    Args:
        X_init: a `n x d`-dim tensor of points
        Y_init: a `n x m`-dim tensor of observations
        bounds: a `2 x d`-dim tensor of bounds
        tr_hparams: hyperparameters for turbo
        objective: An objective function that selects the objectives
            (a subset of all modeled outcomes).
        constraints: List of potential outcome constraints
        pareto_X_better_than_ref: a `k x d`-dim tensor of points on the Pareto frontier
        pareto_Y_better_than_ref: a `k x m`-dim tensor of obs. on the Pareto frontier
        ref_point: a `m`-dim tensor of the reference point
        current_hypervolume: Current hypervolume of the pareto frontier.
        invalid_centers: Points that can't be used as the center.
    """

    def __init__(
        self,
        X_init: Tensor,
        Y_init: Tensor,
        bounds: Tensor,
        tr_hparams: TurboHParams,
        objective: AcquisitionObjective,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        pareto_X_better_than_ref: Optional[Tensor] = None,
        pareto_Y_better_than_ref: Optional[Tensor] = None,
        ref_point: Optional[Tensor] = None,
        current_hypervolume: Optional[float] = None,
        hv_contributions: Optional[Tensor] = None,
        invalid_centers: Optional[Tensor] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            X_init=X_init,
            Y_init=Y_init,
            bounds=bounds,
            tr_hparams=tr_hparams,
            objective=objective,
            constraints=constraints,
            pareto_X_better_than_ref=pareto_X_better_than_ref,
            pareto_Y_better_than_ref=pareto_Y_better_than_ref,
            ref_point=ref_point,
            current_hypervolume=current_hypervolume,
            hv_contributions=hv_contributions,
            invalid_centers=invalid_centers,
            extra_buffers={"ref_point": ref_point},
            **kwargs,
        )

    @property
    def objective(self):
        return self._objective

    def _update_center_and_best_points(
        self,
        pareto_X_better_than_ref: Optional[Tensor] = None,
        pareto_Y_better_than_ref: Optional[Tensor] = None,
        ref_point: Optional[Tensor] = None,
        current_hypervolume: Optional[float] = None,
        hv_contributions: Optional[Tensor] = None,
        invalid_centers: Optional[Tensor] = None,
        X_center: Optional[Tensor] = None,
    ) -> None:
        Y_center_prev = None if self.Y_center is None else self.Y_center.clone()
        # note these are all pareto points (inside/outside TR)
        self.best_X = pareto_X_better_than_ref
        self.best_Y = pareto_Y_better_than_ref
        self.ref_point = ref_point
        # hypervolume is used for objective improvement
        self.hv = current_hypervolume
        if X_center is not None:
            center_idx = (self.X == X_center).long().argmax()
            self._set_center_and_best_points(center_idx=center_idx)
        else:
            pareto_X_normalized = normalize(self.best_X, bounds=self.bounds)
            # set X_center random pareto point, inside the TR when possible
            indices = torch.arange(
                pareto_X_normalized.shape[0], device=pareto_X_normalized.device
            )
            if self.X_center_normalized is not None:
                indices_in_tr = get_indices_in_hypercube(
                    self.X_center_normalized, pareto_X_normalized, length=self.length
                )
                if indices_in_tr.shape[0] > 0:
                    indices = indices_in_tr
            if invalid_centers is not None:
                not_taken_mask = ~(
                    (self.best_X.unsqueeze(1) == invalid_centers.unsqueeze(0))
                    .all(dim=-1)
                    .any(dim=-1)
                )
            else:
                not_taken_mask = torch.ones(
                    self.best_X.shape[0], dtype=bool, device=self.best_X.device
                )
            # select center that is available and in tr if possible
            if hv_contributions.shape[0] > 0:
                indices_mask = torch.zeros_like(not_taken_mask)
                indices_mask[indices] = True
                eligible_mask = indices_mask & not_taken_mask
                if eligible_mask.any():
                    eligible_indices = eligible_mask.nonzero().view(-1)
                    # take max contributing hypervolume among eligible points
                    base_idx = hv_contributions[eligible_indices].argmax()
                    idx = eligible_indices[base_idx].item()
                elif not_taken_mask.any():
                    # if all pareto points are taken in the tr, take the best
                    # available point
                    not_taken_indices = not_taken_mask.nonzero().view(-1)
                    base_idx = hv_contributions[not_taken_indices].argmax()
                    idx = not_taken_indices[base_idx].item()
                else:
                    # take point with max hv contribution
                    idx = hv_contributions.argmax().item()
                self.X_center = pareto_X_better_than_ref[idx : idx + 1]
                self.Y_center = pareto_Y_better_than_ref[idx : idx + 1]
            else:
                # no pareto points better than the reference point, so select a random point
                # TODO: Select the center based on constraint violations instead
                center_idx = torch.randint(0, self.X.shape[0], (1,)).item()
                self._set_center_and_best_points(center_idx=center_idx)

        if self.tr_hparams.verbose and (
            Y_center_prev is None
            or not torch.equal(Y_center_prev.to(self.Y_center), self.Y_center)
        ):
            print(f"New metrics for center: {self.objective(self.Y_center)}")

        self.X_center_normalized = normalize(X=self.X_center, bounds=self.bounds)

    def _has_improved_objective(self, n_new: int) -> bool:
        """Determine whether new batch of n_new points has improved the objective.

        For a hypervolume trust region we define improvement through as:
            new_hv > (1 + eps) * old_hv
        """
        Y_new = self.Y[-n_new:]
        if self.constraints is not None:
            constraint_slack, feas = get_constraint_slack_and_feasibility(
                Y=Y_new, constraints=self.constraints
            )
        else:
            feas = None
        # filter Y to only be the objective
        obj = self.objective(Y_new).clone()
        # NOTE: objective does not necessarily create a new tensor, so we
        # need to clone here.
        if feas is not None:
            # set infeasible points to be the reference point
            # so that they have zero hypervolume
            obj[~feas] = self.ref_point
        # filter to only include feasible points
        better_than_ref = (obj > self.ref_point).all(dim=1)
        obj = obj[better_than_ref]
        if obj.shape[0] > 0:
            aug_obj = torch.cat([self.objective(self.best_Y), obj], dim=0)
            pareto_mask = is_non_dominated(aug_obj)
            if pareto_mask[-(obj.shape[0]) :].any():
                # at least one new point is pareto
                # check hypervolume improvement
                pareto_aug_obj = aug_obj[pareto_mask]
                better_than_ref = (pareto_aug_obj > self.ref_point).all(dim=-1)
                partitioning = DominatedPartitioning(
                    ref_point=self.ref_point,
                    Y=pareto_aug_obj[better_than_ref],
                )
                new_hv = partitioning.compute_hypervolume()

                if new_hv > (1 + self.tr_hparams.eps) * self.hv:
                    print(f"SUCCESS: hv ratio: {new_hv/self.hv}")
                    return True
        return False
