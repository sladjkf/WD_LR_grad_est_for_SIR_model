#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a grid of experiments comparing gradient estimators for the tSIR model:
- Matrix Power finite-difference targets
- Finite Difference (FD)
- Weak Derivatives (WD) for beta and v
- Likelihood Ratio (LR) for beta and v

Saves results incrementally as each task completes.
"""

import csv
from itertools import product
import multiprocessing as mp
from pathlib import Path
import pickle
import time

import numpy as np
from scipy import stats
from tsir_wd import *

# ==========================================
# Experiment Configuration & Parameters
# ==========================================
N_vals = [50, 100, 150, 200]
v_vals = [0.01, 0.3, 0.5, 0.8, 0.99]
beta_vals = [1, 2, 4, 8, 16]
T_vals = [5, 10, 15, 20]
i0 = 1

fd_eps = np.linspace(0.01, 0.1, 12)
fd_matrix = 0.001

n_micro = 40
n_macro = 250
cores = 12

# Execution Flags
COMPUTE_MATRIX_POWER = False
COMPUTE_FD = True
COMPUTE_WD = True
COMPUTE_LR = True

OUTPUT_DIR = Path("output/data/rev_table1/")


# ==========================================
# Helper & Micro-Replication Functions
# ==========================================
def total_inf(traj):
    """Calculates total infections over a trajectory."""
    return np.sum(traj[:, 1])


def tSIR_LR_beta_path(N, v, i0, beta, T, pop_seed, dyn_seed):
    """Simulates a tSIR trajectory and computes per-step score values for beta."""
    pop_rng = np.random.default_rng(pop_seed)
    dyn_rng = np.random.default_rng(dyn_seed)

    N_minus_i0 = N - i0
    V = stats.binom.ppf(q=pop_rng.random(), n=N_minus_i0, p=v)
    S = N_minus_i0 - V

    traj = np.empty((T + 1, 3), dtype=int)
    traj[0] = [S, i0, 0]
    step_scores = np.empty(T)

    for i in range(T):
        next_I = _get_infections_crn(traj[i], N, beta, dyn_rng.random())
        next_state = step_one(traj[i], next_I)
        traj[i + 1] = next_state
        step_scores[i] = LR_beta_term(next_state, traj[i], beta, N)

    return traj[:, 1], step_scores


# --- Micro-Replications ---


def wd_beta_micro_rep(N, v, beta, T, i0, n_micro, macro_idx):
    """Evaluates Weak Derivative for beta across n_micro runs."""
    seed_offset = macro_idx * n_micro
    t0 = time.perf_counter()
    samples = []
    for i in range(n_micro):
        wd_list = tSIR_WD_beta_CRN(
            N, v, i0, beta, T, (seed_offset + i) * 42, (seed_offset + i) * 17
        )
        grad = sum(
            scale * (np.sum(plus[:, 1]) - np.sum(minus[:, 1]))
            for scale, plus, minus in wd_list
        )
        samples.append(grad)
    return np.mean(samples), time.perf_counter() - t0


def wd_v_micro_rep(N, v, beta, T, i0, n_micro, macro_idx):
    """Evaluates Weak Derivative for v across n_micro runs."""
    seed_offset = macro_idx * n_micro
    t0 = time.perf_counter()
    scale = N - i0
    samples = []
    for i in range(n_micro):
        _, plus, minus = tSIR_WD_CRN(
            N, v, i0, beta, T, (seed_offset + i) * 42, (seed_offset + i) * 17
        )
        grad = scale * (np.sum(plus[:, 1]) - np.sum(minus[:, 1]))
        samples.append(grad)
    return np.mean(samples), time.perf_counter() - t0


def lr_v_micro_rep(N, v, beta, T, i0, n_micro, macro_idx):
    """Evaluates Likelihood Ratio for v with baseline reduction."""
    seed_offset = macro_idx * n_micro
    t0 = time.perf_counter()
    total_infs, scores = np.empty(n_micro), np.empty(n_micro)
    for i in range(n_micro):
        traj, score = tSIR_LR_v(
            N, v, i0, beta, T, (seed_offset + i) * 42, (seed_offset + i) * 17
        )
        total_infs[i] = total_inf(traj)
        scores[i] = score
    y_bar = np.mean(total_infs)
    return np.mean((total_infs - y_bar) * scores), time.perf_counter() - t0


def lr_beta_micro_rep(N, v, beta, T, i0, n_micro, macro_idx):
    """Evaluates Likelihood Ratio for beta across n_micro runs."""
    seed_offset = macro_idx * n_micro
    t0 = time.perf_counter()
    trajectories = np.empty((T + 1, n_micro), dtype=int)
    score_samples = np.empty((T, n_micro))
    for i in range(n_micro):
        i_traj, step_scores = tSIR_LR_beta_path(
            N, v, i0, beta, T, (seed_offset + i) * 42, (seed_offset + i) * 17
        )
        trajectories[:, i] = i_traj
        score_samples[:, i] = step_scores
    mean_beta, _ = grad_wrt_beta(trajectories, score_samples, T, n_micro)
    return mean_beta, time.perf_counter() - t0


def fd_micro_rep(N, v, beta, T, i0, eps, n_micro, macro_idx):
    """Evaluates Finite Difference for v and beta across n_micro runs."""
    seed_offset = macro_idx * n_micro

    t0 = time.perf_counter()
    samples_beta = [
        one_sided_fd_crn(
            N,
            v,
            i0,
            beta,
            T,
            eps,
            (seed_offset + i) * 42,
            (seed_offset + i) * 17,
            total_inf,
            param="beta",
        )
        for i in range(n_micro)
    ]
    time_beta = time.perf_counter() - t0

    t0 = time.perf_counter()
    samples_v = [
        one_sided_fd_crn(
            N,
            v,
            i0,
            beta,
            T,
            eps,
            (seed_offset + i) * 42,
            (seed_offset + i) * 17,
            total_inf,
            param="v",
        )
        for i in range(n_micro)
    ]
    time_v = time.perf_counter() - t0

    return np.mean(samples_v), np.mean(samples_beta), time_v, time_beta


# ==========================================
# Generic Workers & Experiment Runners
# ==========================================
def generic_single_param_worker(args):
    """Generic worker executing macro replications on single-parameter estimators."""
    micro_rep_fn, N, v, beta, T, i0, n_macro, n_micro, J_true = args
    reps = np.array(
        [
            micro_rep_fn(N, v, beta, T, i0, n_micro, macro_idx=m)
            for m in range(n_macro)
        ]
    )

    mean_est = np.mean(reps[:, 0])
    sd_est = np.std(reps[:, 0], ddof=1) if n_macro > 1 else 0.0
    mse_est = np.mean((reps[:, 0] - J_true) ** 2)
    avg_time = np.mean(reps[:, 1])

    return N, v, beta, T, i0, mean_est, sd_est, mse_est, avg_time


def fd_worker(args):
    """Worker executing macro replications for Finite Difference."""
    N, v, beta, T, i0, eps, n_macro, n_micro, J_v, J_beta = args
    reps = np.array(
        [
            fd_micro_rep(N, v, beta, T, i0, eps, n_micro, macro_idx=m)
            for m in range(n_macro)
        ]
    )

    mean_v, mean_beta = np.mean(reps[:, 0]), np.mean(reps[:, 1])
    sd_v = np.std(reps[:, 0], ddof=1) if n_macro > 1 else 0.0
    sd_beta = np.std(reps[:, 1], ddof=1) if n_macro > 1 else 0.0
    mse_v = np.mean((reps[:, 0] - J_v) ** 2)
    mse_beta = np.mean((reps[:, 1] - J_beta) ** 2)
    avg_time_v, avg_time_beta = np.mean(reps[:, 2]), np.mean(reps[:, 3])

    return (
        N,
        v,
        beta,
        T,
        i0,
        eps,
        mean_v,
        mean_beta,
        sd_v,
        sd_beta,
        mse_v,
        mse_beta,
        avg_time_v,
        avg_time_beta,
    )


def load_targets(csv_path):
    """Parses ground-truth targets from matrix power CSV."""
    targets = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            targets.append(
                {
                    "N": int(row["N"]),
                    "v": float(row["v"]),
                    "beta": float(row["beta"]),
                    "T": int(row["T"]),
                    "i0": int(row["i0"]),
                    "J_v": float(row["J_v"].strip('"')),
                    "J_beta": float(row["J_beta"].strip('"')),
                }
            )
    return targets


def run_single_param_experiment(
    micro_fn, target_param, filename, targets, pool
):
    """Pipeline harness to execute and incrementally export single-parameter estimator studies."""
    output_csv = OUTPUT_DIR / filename
    tasks = [
        (
            micro_fn,
            t["N"],
            t["v"],
            t["beta"],
            t["T"],
            t["i0"],
            n_macro,
            n_micro,
            t[f"J_{target_param}"],
        )
        for t in targets
    ]

    with open(output_csv, "w") as f:
        f.write(
            f"N,v,beta,T,i0,mean_{target_param},sd_{target_param},mse_{target_param},time_{target_param}\n"
        )
        # Process results as they finish and flush to disk immediately
        for N, v, beta, T, i0, mean_e, sd_e, mse_e, time_e in pool.imap_unordered(
            generic_single_param_worker, tasks
        ):
            f.write(
                f"{N},{v},{beta},{T},{i0},{mean_e:.6f},{sd_e:.6f},{mse_e:.6e},{time_e:.6f}\n"
            )
            f.flush()


# ==========================================
# Main Orchestrator
# ==========================================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_csv = OUTPUT_DIR / "matrix_power_gradients.csv"

    # 1. Compute Matrix Power Targets
    if COMPUTE_MATRIX_POWER:

        def matrix_power_worker(args):
            N, v, beta, T, i0, eps = args
            t0 = time.perf_counter()
            J_v = fd_matrix_powers_I(N, v, i0, beta, T, eps=eps, param="v")
            time_v = time.perf_counter() - t0

            t0 = time.perf_counter()
            J_beta = fd_matrix_powers_I(N, v, i0, beta, T, eps=eps, param="beta")
            time_beta = time.perf_counter() - t0
            return args, J_v, J_beta, time_v, time_beta

        param_combos = list(
            product(N_vals, v_vals, beta_vals, T_vals, [i0], [fd_matrix])
        )

        raw_results = []
        with mp.Pool(cores) as pool:
            with open(matrix_csv, "w") as f:
                f.write("N,v,beta,T,i0,eps,J_v,J_beta,time_v,time_beta\n")
                for (N, v, beta, T, i0, eps), J_v, J_beta, time_v, time_beta in pool.imap_unordered(
                    matrix_power_worker, param_combos
                ):
                    raw_results.append(
                        ((N, v, beta, T, i0, eps), (J_v, J_beta, time_v, time_beta))
                    )
                    f.write(
                        f'{N},{v},{beta},{T},{i0},{eps},"{J_v}","{J_beta}",{time_v:.6f},{time_beta:.6f}\n'
                    )
                    f.flush()

        with open(OUTPUT_DIR / "matrix_powers.pkl", "wb") as f:
            pickle.dump(raw_results, f)

    # 2. Run Simulation Experiments
    targets = load_targets(matrix_csv)

    with mp.Pool(cores) as pool:
        # Finite Difference Evaluation
        if COMPUTE_FD:
            fd_tasks = [
                (
                    t["N"],
                    t["v"],
                    t["beta"],
                    t["T"],
                    t["i0"],
                    eps,
                    n_macro,
                    n_micro,
                    t["J_v"],
                    t["J_beta"],
                )
                for t in targets
                for eps in fd_eps
            ]

            with open(OUTPUT_DIR / "fd_estimators_results.csv", "w") as f:
                f.write(
                    "N,v,beta,T,i0,eps,mean_v,mean_beta,sd_v,sd_beta,mse_v,mse_beta,time_v,time_beta\n"
                )
                for r in pool.imap_unordered(fd_worker, fd_tasks):
                    f.write(
                        f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.6f},"
                        f"{r[6]:.6f},{r[7]:.6f},{r[8]:.6f},{r[9]:.6f},"
                        f"{r[10]:.6e},{r[11]:.6e},{r[12]:.6f},{r[13]:.6f}\n"
                    )
                    f.flush()

        # Weak Derivatives
        if COMPUTE_WD:
            run_single_param_experiment(
                wd_beta_micro_rep,
                "beta",
                "wd_beta_estimators_results.csv",
                targets,
                pool,
            )
            run_single_param_experiment(
                wd_v_micro_rep,
                "v",
                "wd_v_estimators_results.csv",
                targets,
                pool,
            )

        # Likelihood Ratio
        if COMPUTE_LR:
            run_single_param_experiment(
                lr_v_micro_rep,
                "v",
                "lr_v_estimators_results.csv",
                targets,
                pool,
            )
            run_single_param_experiment(
                lr_beta_micro_rep,
                "beta",
                "lr_beta_estimators_results.csv",
                targets,
                pool,
            )