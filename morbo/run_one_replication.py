#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Run one replication.
"""
from typing import Callable, Dict, List, Optional, Union
import time
import torch
import numpy as np
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.test_functions import Ackley
from botorch.test_functions.multi_objective import (
    BraninCurrin,
    C2DTLZ2,
    DH2,
    DH3,
    DH4,
    DTLZ1,
    DTLZ2,
    MW7,
    VehicleSafety,
    WeldedBeam,
)
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.sampling import draw_sobol_samples
from morbo.gen import (
    TS_select_batch_MORBO,
)
from morbo.state import TRBOState
from morbo.trust_region import TurboHParams
from torch import Tensor

from morbo.problems.rover import get_rover_fn
from morbo.benchmark_function import (
    BenchmarkFunction,
)

supported_labels = ["morbo"]

BASE_SEED = 12346


def run_one_replication(
    seed: int,
    label: str,
    max_evals: int,
    evalfn: str,
    batch_size: int,
    dim: int,
    n_initial_points: int,
    n_trust_regions: int = TurboHParams.n_trust_regions,
    max_tr_size: int = TurboHParams.max_tr_size,
    min_tr_size: int = TurboHParams.min_tr_size,
    max_reference_point: Optional[List[float]] = None,
    failure_streak: Optional[int] = None,  # This is better to set automatically
    success_streak: int = TurboHParams.success_streak,
    raw_samples: int = TurboHParams.raw_samples,
    n_restart_points: int = TurboHParams.n_restart_points,
    length_init: float = TurboHParams.length_init,
    length_min: float = TurboHParams.length_min,
    length_max: float = TurboHParams.length_max,
    trim_trace: bool = TurboHParams.trim_trace,
    hypervolume: bool = TurboHParams.hypervolume,
    max_cholesky_size: int = TurboHParams.max_cholesky_size,
    use_ard: bool = TurboHParams.use_ard,
    verbose: bool = TurboHParams.verbose,
    qmc: bool = TurboHParams.qmc,
    track_history: bool = TurboHParams.track_history,
    sample_subset_d: bool = TurboHParams.sample_subset_d,
    fixed_scalarization: bool = TurboHParams.fixed_scalarization,
    winsor_pct: float = TurboHParams.winsor_pct,
    trunc_normal_perturb: bool = TurboHParams.trunc_normal_perturb,
    switch_strategy_freq: Optional[int] = TurboHParams.switch_strategy_freq,
    tabu_tenure: int = TurboHParams.tabu_tenure,
    decay_restart_length_alpha: float = TurboHParams.decay_restart_length_alpha,
    use_noisy_trbo: bool = TurboHParams.use_noisy_trbo,
    observation_noise_std: Optional[List[float]] = None,
    observation_noise_bias: Optional[List[float]] = None,
    use_simple_rff: bool = TurboHParams.use_simple_rff,
    use_approximate_hv_computations: bool = TurboHParams.use_approximate_hv_computations,
    approximate_hv_alpha: Optional[float] = TurboHParams.approximate_hv_alpha,
    recompute_all_hvs: bool = True,
    restart_hv_scalarizations: bool = True,
    dtype: torch.device = torch.double,
    device: Optional[torch.device] = None,
    save_callback: Optional[Callable[[Tensor], None]] = None,
    save_during_opt: bool = True,
) -> None:
    r"""Run the BO loop for given number of iterations. Supports restarting of
    prematurely killed experiments.

    Args:
        seed: The random seed.
        label: The algorith ("morbo")
        max_evals: evaluation budget
        evalfn: The test problem name
        batch_size: The size of each batch in BO
        dim: The input dimension (this is a parameter for some problems)
        n_initial_points: The number of initial sobol points

    The remaining parameters and default values are defined in trust_region.py.
    """
    assert label in supported_labels, "Label not supported!"
    start_time = time.time()
    seed = BASE_SEED + seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tkwargs = {"dtype": dtype, "device": device}
    bounds = torch.empty((2, dim), dtype=dtype, device=device)
    constraints = None
    objective = None

    if evalfn == "ackley":
        f = Ackley(dim=dim, negate=False)
        bounds[0] = -5.0
        bounds[1] = 10.0
        num_outputs = 1
        num_objectives = 1
    elif max_reference_point is None:
        raise ValueError(f"max_reference_point is required for {evalfn}")
    else:
        num_objectives = len(max_reference_point)

    if evalfn == "C2DTLZ2":
        problem = C2DTLZ2(
            num_objectives=num_objectives,
            dim=dim,
            negate=False,
        )
        bounds = problem.bounds.to(**tkwargs)
        num_outputs = problem.num_objectives + problem.num_constraints
        # Note: all outcomes are multiplied by -1 in `BenchmarkFunction` by default.
        constraints = (
            torch.tensor([[0.0, 0.0, 1.0]], **tkwargs),
            torch.tensor([[0.0]], **tkwargs),
        )

        def f(X):
            return torch.cat([problem(X), problem.evaluate_slack(X)], dim=-1)

        objective = IdentityMCMultiOutputObjective(
            outcomes=[0, 1], num_outcomes=num_outputs
        )
    elif evalfn == "rover":
        num_objectives, num_outputs = 2, 2
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}.")
        f, bounds = get_rover_fn(
            dim, device=device, dtype=dtype, force_goal=False, force_start=True
        )
    elif evalfn == "WeldedBeam":
        problem = WeldedBeam(negate=False)
        bounds = problem.bounds.to(**tkwargs)

        def f(X):
            return torch.cat([problem(X), problem.evaluate_slack(X)], dim=-1)

        num_objectives = 2
        num_constraints = 4
        num_outputs = num_objectives + num_constraints

        Z_ = torch.zeros(num_constraints, num_objectives, **tkwargs)
        A = torch.cat((Z_, torch.eye(num_constraints, **tkwargs)), dim=1)
        constraints = (A, torch.zeros(num_constraints, 1, **tkwargs))

        objective = IdentityMCMultiOutputObjective(
            outcomes=[0, 1], num_outcomes=num_outputs
        )
    elif evalfn == "MW7":
        problem = MW7(negate=False, dim=dim)
        bounds = problem.bounds.to(**tkwargs)

        def f(X):
            return torch.cat([problem(X), problem.evaluate_slack(X)], dim=-1)

        num_objectives = 2
        num_constraints = 2
        num_outputs = num_objectives + num_constraints

        Z_ = torch.zeros(num_constraints, num_objectives, **tkwargs)
        A = torch.cat((Z_, torch.eye(num_constraints, **tkwargs)), dim=1)
        constraints = (A, torch.zeros(num_constraints, 1, **tkwargs))

        objective = IdentityMCMultiOutputObjective(
            outcomes=[0, 1], num_outcomes=num_outputs
        )
    elif evalfn != "ackley":
        # Handle the non-constrained botorch test functions here.
        constructor_map = {
            "DH2": DH2,
            "DH3": DH3,
            "DH4": DH4,
            "DTLZ1": DTLZ1,
            "DTLZ2": DTLZ2,
            "BraninCurrin": BraninCurrin,
            "VehicleSafety": VehicleSafety,
        }
        constructor_args = {"negate": False}
        if evalfn not in ("BraninCurrin", "VehicleSafety"):
            constructor_args["dim"] = dim
        if "DTLZ" in evalfn:
            constructor_args["num_objectives"] = num_objectives
        if evalfn not in constructor_map:
            raise ValueError("Unknown `evalfn` specified!")
        f = constructor_map[evalfn](**constructor_args)
        bounds = f.bounds.to(**tkwargs)
        num_outputs = f.num_objectives

    f = BenchmarkFunction(
        base_f=f,
        num_outputs=num_outputs,
        ref_point=torch.tensor(max_reference_point, **tkwargs),
        dim=dim,
        tkwargs=tkwargs,
        negate=True,
        observation_noise_std=observation_noise_std,
        observation_noise_bias=observation_noise_bias,
    )

    # Automatically set the failure streak if it isn't specified
    failure_streak = max(dim // 3, 10) if failure_streak is None else failure_streak

    tr_hparams = TurboHParams(
        length_init=length_init,
        length_min=length_min,
        length_max=length_max,
        batch_size=batch_size,
        success_streak=success_streak,
        failure_streak=failure_streak,
        max_tr_size=max_tr_size,
        min_tr_size=min_tr_size,
        trim_trace=trim_trace,
        n_trust_regions=n_trust_regions,
        verbose=verbose,
        qmc=qmc,
        use_ard=use_ard,
        sample_subset_d=sample_subset_d,
        track_history=track_history,
        fixed_scalarization=fixed_scalarization,
        n_initial_points=n_initial_points,
        n_restart_points=n_restart_points,
        raw_samples=raw_samples,
        max_reference_point=max_reference_point,
        hypervolume=hypervolume,
        winsor_pct=winsor_pct,
        trunc_normal_perturb=trunc_normal_perturb,
        decay_restart_length_alpha=decay_restart_length_alpha,
        switch_strategy_freq=switch_strategy_freq,
        tabu_tenure=tabu_tenure,
        use_noisy_trbo=use_noisy_trbo,
        use_simple_rff=use_simple_rff,
        use_approximate_hv_computations=use_approximate_hv_computations,
        approximate_hv_alpha=approximate_hv_alpha,
        restart_hv_scalarizations=restart_hv_scalarizations,
    )

    trbo_state = TRBOState(
        dim=dim,
        max_evals=max_evals,
        num_outputs=num_outputs,
        num_objectives=num_objectives,
        bounds=bounds,
        tr_hparams=tr_hparams,
        constraints=constraints,
        objective=objective,
    )

    # For saving outputs
    n_evals = []
    true_hv = []
    pareto_X = []
    pareto_Y = []
    n_points_in_tr = [[] for _ in range(n_trust_regions)]
    n_points_in_tr_collected_by_other = [[] for _ in range(n_trust_regions)]
    n_points_in_tr_collected_by_sobol = [[] for _ in range(n_trust_regions)]
    tr_sizes = [[] for _ in range(n_trust_regions)]
    tr_centers = [[] for _ in range(n_trust_regions)]
    tr_restarts = [[] for _ in range(n_trust_regions)]
    fit_times = []
    gen_times = []
    true_ref_point = torch.tensor(max_reference_point, dtype=dtype, device=device)

    # Create initial points
    n_points = min(n_initial_points, max_evals - trbo_state.n_evals)
    X_init = draw_sobol_samples(bounds=bounds, n=n_points, q=1).squeeze(1)
    Y_init = f(X_init)
    trbo_state.update(
        X=X_init,
        Y=Y_init,
        new_ind=torch.full(
            (X_init.shape[0],), 0, dtype=torch.long, device=X_init.device
        ),
    )
    trbo_state.log_restart_points(X=X_init, Y=Y_init)

    # Initializing the trust regions. This also initializes the models.
    for i in range(n_trust_regions):
        trbo_state.initialize_standard(
            tr_idx=i,
            restart=False,
            switch_strategy=False,
            X_init=X_init,
            Y_init=Y_init,
        )

    # Update TRs data across trust regions, if necessary
    trbo_state.update_data_across_trs()

    # Set the initial TR indices to -2
    trbo_state.TR_index_history.fill_(-2)

    # Getting next suggestions
    all_tr_indices = [-1] * n_points
    while trbo_state.n_evals < max_evals:
        start_gen = time.time()
        selection_output = TS_select_batch_MORBO(trbo_state=trbo_state)
        gen_times.append(time.time() - start_gen)
        if trbo_state.tr_hparams.verbose:
            print(f"Time spent on generating candidates: {gen_times[-1]:.1f} seconds")

        X_cand = selection_output.X_cand
        tr_indices = selection_output.tr_indices
        all_tr_indices.extend(tr_indices.tolist())
        trbo_state.tabu_set.log_iteration()
        Y_cand = f(X_cand)

        # Log TR info
        for i, tr in enumerate(trbo_state.trust_regions):
            inds = torch.cat(
                [torch.where((x == trbo_state.X_history).all(dim=-1))[0] for x in tr.X]
            )
            tr_inds = trbo_state.TR_index_history[inds]
            assert len(tr_inds) == len(tr.X)
            n_points_in_tr[i].append(len(tr_inds))
            n_points_in_tr_collected_by_sobol[i].append(sum(tr_inds == -2).cpu().item())
            n_points_in_tr_collected_by_other[i].append(
                sum((tr_inds != i) & (tr_inds != -2)).cpu().item()
            )
            tr_sizes[i].append(tr.length.item())
            tr_centers[i].append(tr.X_center.cpu().squeeze().tolist())

        # Append data to the global history and fit new models
        start_fit = time.time()
        trbo_state.update(X=X_cand, Y=Y_cand, new_ind=tr_indices)
        should_restart_trs = trbo_state.update_trust_regions_and_log(
            X_cand=X_cand,
            Y_cand=Y_cand,
            tr_indices=tr_indices,
            batch_size=batch_size,
            verbose=verbose,
        )
        fit_times.append(time.time() - start_fit)
        if trbo_state.tr_hparams.verbose:
            print(f"Time spent on model fitting: {fit_times[-1]:.1f} seconds")

        switch_strategy = trbo_state.check_switch_strategy()
        if switch_strategy:
            should_restart_trs = [True for _ in should_restart_trs]
        if any(should_restart_trs):
            for i in range(trbo_state.tr_hparams.n_trust_regions):
                if should_restart_trs[i]:
                    n_points = min(n_restart_points, max_evals - trbo_state.n_evals)
                    if n_points <= 0:
                        break  # out of budget
                    if trbo_state.tr_hparams.verbose:
                        print(f"{trbo_state.n_evals}) Restarting trust region {i}")
                    trbo_state.TR_index_history[trbo_state.TR_index_history == i] = -1
                    init_kwargs = {}
                    if trbo_state.tr_hparams.restart_hv_scalarizations:
                        # generate new point
                        X_center = trbo_state.gen_new_restart_design()
                        Y_center = f(X_center)
                        init_kwargs["X_init"] = X_center
                        init_kwargs["Y_init"] = Y_center
                        init_kwargs["X_center"] = X_center
                        trbo_state.update(
                            X=X_center,
                            Y=Y_center,
                            new_ind=torch.tensor(
                                [i], dtype=torch.long, device=X_center.device
                            ),
                        )
                        trbo_state.log_restart_points(X=X_center, Y=Y_center)

                    trbo_state.initialize_standard(
                        tr_idx=i,
                        restart=True,
                        switch_strategy=switch_strategy,
                        **init_kwargs,
                    )
                    if trbo_state.tr_hparams.restart_hv_scalarizations:
                        # we initialized the TR with one data point.
                        # this passes historical information to that new TR
                        trbo_state.update_data_across_trs()
                    tr_restarts[i].append(
                        trbo_state.n_evals.item()
                    )  # Where it restarted

        if trbo_state.tr_hparams.verbose:
            print(f"Total refill points: {trbo_state.total_refill_points}")

        # Save state at this evaluation and move to cpu
        n_evals.append(trbo_state.n_evals.item())
        if trbo_state.hv is not None:
            # The objective is None if there are no constraints
            obj = objective if objective else lambda x: x
            partitioning = DominatedPartitioning(
                ref_point=true_ref_point, Y=obj(trbo_state.pareto_Y)
            )
            hv = partitioning.compute_hypervolume().item()
            if trbo_state.tr_hparams.verbose:
                print(f"{trbo_state.n_evals}) Current hypervolume: {hv:.3f}")

            pareto_X.append(trbo_state.pareto_X.tolist())
            pareto_Y.append(trbo_state.pareto_Y.tolist())
            true_hv.append(hv)

            if observation_noise_std is not None:
                f.record_current_pf_and_hv(obj=obj, constraints=trbo_state.constraints)
        else:
            if trbo_state.tr_hparams.verbose:
                print(f"{trbo_state.n_evals}) Current hypervolume is zero!")
            pareto_X.append([])
            pareto_Y.append([])
            true_hv.append(0.0)
        trbo_state.update_data_across_trs()

        output = {
            "n_evals": n_evals,
            "X_history": trbo_state.X_history.cpu(),
            "metric_history": trbo_state.Y_history.cpu(),
            "true_pareto_X": pareto_X,
            "true_pareto_Y": pareto_Y,
            "true_hv": true_hv,
            "n_points_in_tr": n_points_in_tr,
            "n_points_in_tr_collected_by_other": n_points_in_tr_collected_by_other,
            "n_points_in_tr_collected_by_sobol": n_points_in_tr_collected_by_sobol,
            "tr_sizes": tr_sizes,
            "tr_centers": tr_centers,
            "tr_restarts": tr_restarts,
            "fit_times": fit_times,
            "gen_times": gen_times,
            "tr_indices": all_tr_indices,
        }
        # Save the output.
        if save_during_opt is not None:
            save_callback(output)

    end_time = time.time()
    if trbo_state.tr_hparams.verbose:
        print(f"Total time: {end_time - start_time:.1f} seconds")

    if trbo_state.hv is not None and recompute_all_hvs:
        # Go back and compute all hypervolumes so we don't have to do that later...
        f.record_all_hvs(obj=obj, constraints=trbo_state.constraints)

    output = {
        "n_evals": n_evals,
        "X_history": trbo_state.X_history.cpu(),
        "metric_history": trbo_state.Y_history.cpu(),
        "true_pareto_X": pareto_X,
        "true_pareto_Y": pareto_Y,
        "true_hv": true_hv,
        "n_points_in_tr": n_points_in_tr,
        "n_points_in_tr_collected_by_other": n_points_in_tr_collected_by_other,
        "n_points_in_tr_collected_by_sobol": n_points_in_tr_collected_by_sobol,
        "tr_sizes": tr_sizes,
        "tr_centers": tr_centers,
        "tr_restarts": tr_restarts,
        "fit_times": fit_times,
        "gen_times": gen_times,
        "tr_indices": all_tr_indices,
    }
    if trbo_state.hv is not None and recompute_all_hvs:
        additional_outputs = f.get_outputs()
        output = {**output, **additional_outputs}

    # Save the final output
    save_callback(output)
