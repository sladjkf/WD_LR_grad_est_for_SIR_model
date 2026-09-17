#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:45:33 2026

@author: nicholasw

Run a grid of experiments. We want to study the behavior of

- LR for $\beta$ and $v$, 
- WD for $\beta$ and $v$, 
- FD for $\beta$ and $v$, 
- FSE for $\beta$ and $v$.

The following parameters could be chosen:

- $N = [50, 100, 150, 200]$
- $v = [0.1, 0.5, 0.9]$
- $beta = [1, 5, 10]$
- $T = [5, 10, 15, 20]$
- $epsilon_{FD} = 0.001$ to $0.1$
- micro-replications: 50
- macro-replications: 10,000

# Experimental: metrics

- Mean squared error of each estimator, taking matrix-powers + finite-difference as true value of derivative
- Mean and variance of each estimator
- Computation time for all methods (including Kolmogorov method) - defined as the time to complete one batch of micro-replications, possibly parallelized

"""
import numpy as np
import multiprocess as mp
from tsir_wd import *
from itertools import product
import time
import multiprocess as mp
import pickle
import pandas as pd
from pathlib import Path
import csv

#%%

# Parameter values
N_vals = [50, 100, 150, 200]
v_vals = [0.01, 0.3, 0.5, 0.8, 0.99]
beta_vals = [1, 2, 4, 8, 16]
T_vals = [5, 10, 15, 20]
i0 = 1

fd_eps = np.linspace(0.001, 0.1, 23)
fd_matrix = 0.001

n_micro = 50
n_macro = 1000

cores = 12
#%%
output_dir = Path("output/data/rev_table1/")
output_dir.mkdir(parents=True, exist_ok=True)

#%%

compute_matrix_power = False
compute_fd = False

#%%
if compute_matrix_power:
    def matrix_power_gradients_worker(args):
        """Worker function to compute finite difference gradients and execution times."""
        N, v, beta, T, i0, eps = args
    
        # Measure gradient w.r.t. v
        t0 = time.perf_counter()
        J_v = fd_matrix_powers_I(N, v, i0, beta, T, eps=eps, param="v")
        time_v = time.perf_counter() - t0
    
        # Measure gradient w.r.t. beta
        t0 = time.perf_counter()
        J_beta = fd_matrix_powers_I(N, v, i0, beta, T, eps=eps, param="beta")
        time_beta = time.perf_counter() - t0
    
        return J_v, J_beta, time_v, time_beta
    
    param_combos = list(product(N_vals, v_vals, beta_vals, T_vals, [i0], [fd_matrix]))
    
    with mp.Pool(cores) as pool:
        results = pool.map(matrix_power_gradients_worker, param_combos)
    
    # Save raw results
    with open(output_dir / "matrix_powers.pkl", "wb") as f:
        pickle.dump(list(zip(param_combos, results)), f)
    
    # Save CSV output
    csv_path = output_dir / "matrix_power_gradients.csv"
    with open(csv_path, "w") as f:
        f.write("N,v,beta,T,i0,eps,J_v,J_beta,time_v,time_beta\n")
        for (N, v, beta, T, i0, eps), (J_v, J_beta, time_v, time_beta) in zip(
            param_combos, results
        ):
            # Enclose array/vector outputs in quotes to preserve CSV formatting
            f.write(
                f'{N},{v},{beta},{T},{i0},{eps},"{J_v}","{J_beta}",{time_v:.6f},{time_beta:.6f}\n'
            )
#%%

def total_inf(traj):
    """Calculates total infections over the trajectory."""
    return np.sum(traj[:, 1])


def FD_estimator_micro_rep(N, v, beta, T, i0, eps, n_micro, macro_idx):
    """Average across micro replications of the FD estimator for a given macro run."""
    seed_offset = macro_idx * n_micro

    # Estimate w.r.t beta
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
    mean_beta = np.mean(samples_beta)

    # Estimate w.r.t v
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
    mean_v = np.mean(samples_v)

    return mean_v, mean_beta, time_v, time_beta


def FD_estimator_worker(args):
    """Worker function executing macro replications for a single (row, eps) combo."""
    N, v, beta, T, i0, eps, n_macro, n_micro, J_v, J_beta = args

    reps = np.array(
        [
            FD_estimator_micro_rep(
                N, v, beta, T, i0, eps, n_micro, macro_idx=m
            )
            for m in range(n_macro)
        ]
    )

    mean_v, mean_beta = np.mean(reps[:, 0]), np.mean(reps[:, 1])
    sd_v = np.std(reps[:, 0], ddof=1) if n_macro > 1 else 0.0
    sd_beta = np.std(reps[:, 1], ddof=1) if n_macro > 1 else 0.0

    mse_v = np.mean((reps[:, 0] - J_v) ** 2)
    mse_beta = np.mean((reps[:, 1] - J_beta) ** 2)

    avg_time_v = np.mean(reps[:, 2])
    avg_time_beta = np.mean(reps[:, 3])

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


# Execution Block
if compute_fd:
    input_csv = output_dir / "matrix_power_gradients.csv"
    output_csv = output_dir / "fd_estimators_results.csv"

    # Build tasks across all CSV rows and all values in fd_eps
    tasks = []
    with open(input_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            N = int(row["N"])
            v = float(row["v"])
            beta = float(row["beta"])
            T = int(row["T"])
            i0 = int(row["i0"])
            J_v = float(row["J_v"].strip('"'))
            J_beta = float(row["J_beta"].strip('"'))

            for eps in fd_eps:
                tasks.append(
                    (N, v, beta, T, i0, eps, n_macro, n_micro, J_v, J_beta)
                )

    # Parallel processing over all task combinations
    with mp.Pool(cores) as pool:
        fd_results = pool.map(FD_estimator_worker, tasks)

    # Save aggregated results
    with open(output_csv, "w") as f:
        f.write(
            "N,v,beta,T,i0,eps,mean_v,mean_beta,sd_v,sd_beta,mse_v,mse_beta,time_v,time_beta\n"
        )
        for row in fd_results:
            (
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
                time_v,
                time_beta,
            ) = row
            f.write(
                f"{N},{v},{beta},{T},{i0},{eps:.6f},{mean_v:.6f},{mean_beta:.6f},"
                f"{sd_v:.6f},{sd_beta:.6f},{mse_v:.6e},{mse_beta:.6e},"
                f"{time_v:.6f},{time_beta:.6f}\n"
            )
#%%
import csv
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np


def wd_beta_single_sample(N, v, i0, beta, T, pop_seed, dyn_seed):
    """Evaluates a single Weak-Derivative path sample for beta."""
    wd_list = tSIR_WD_beta_CRN(N, v, i0, beta, T, pop_seed, dyn_seed)
    
    # Sum scale * (sum(plus_traj) - sum(minus_traj)) across time-step perturbations
    grad_estimate = sum(
        scale * (np.sum(plus_traj[:, 1]) - np.sum(minus_traj[:, 1]))
        for scale, plus_traj, minus_traj in wd_list
    )
    return grad_estimate


def WD_estimator_micro_rep(N, v, beta, T, i0, n_micro, macro_idx):
    """Average across n_micro replications of the Weak-Derivative estimator."""
    seed_offset = macro_idx * n_micro

    t0 = time.perf_counter()
    samples_beta = [
        wd_beta_single_sample(
            N,
            v,
            i0,
            beta,
            T,
            pop_seed=(seed_offset + i) * 42,
            dyn_seed=(seed_offset + i) * 17,
        )
        for i in range(n_micro)
    ]
    time_beta = time.perf_counter() - t0

    return np.mean(samples_beta), time_beta


def WD_estimator_worker(args):
    """Worker executing macro replications and computing metrics against true J_beta."""
    N, v, beta, T, i0, n_macro, n_micro, J_beta = args

    reps = np.array(
        [
            WD_estimator_micro_rep(N, v, beta, T, i0, n_micro, macro_idx=m)
            for m in range(n_macro)
        ]
    )

    # Compute aggregate performance metrics across macro runs
    mean_beta = np.mean(reps[:, 0])
    sd_beta = np.std(reps[:, 0], ddof=1) if n_macro > 1 else 0.0
    mse_beta = np.mean((reps[:, 0] - J_beta) ** 2)
    avg_time_beta = np.mean(reps[:, 1])

    return N, v, beta, T, i0, mean_beta, sd_beta, mse_beta, avg_time_beta


# Execution Block
input_csv = output_dir / "matrix_power_gradients.csv"
output_csv = output_dir / "wd_beta_estimators_results.csv"

# Parse parameter sets and exact gradients J_beta from previous step
tasks = []
with open(input_csv, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tasks.append(
            (
                int(row["N"]),
                float(row["v"]),
                float(row["beta"]),
                int(row["T"]),
                int(row["i0"]),
                n_macro,
                n_micro,
                float(row["J_beta"].strip('"')),
            )
        )

# Parallel processing across parameter configurations
with mp.Pool(cores) as pool:
    wd_results = pool.map(WD_estimator_worker, tasks)

# Save Weak-Derivative estimation statistics
with open(output_csv, "w") as f:
    f.write("N,v,beta,T,i0,mean_beta,sd_beta,mse_beta,time_beta\n")
    for row in wd_results:
        N, v, beta, T, i0, mean_beta, sd_beta, mse_beta, time_beta = row
        f.write(
            f"{N},{v},{beta},{T},{i0},{mean_beta:.6f},"
            f"{sd_beta:.6f},{mse_beta:.6e},{time_beta:.6f}\n"
        )
#%%

def wd_v_single_sample(N, v, i0, beta, T, pop_seed, dyn_seed):
    """Evaluates a single Weak-Derivative path sample for parameter v."""
    orig_traj, plus_traj, minus_traj = tSIR_WD_CRN(
        N, v, i0, beta, T, pop_seed, dyn_seed
    )

    # Weak derivative constant for Binomial(N - i0, 1 - v) with respect to v
    scale = N - i0

    # Gradient estimate: scale * (sum(plus) - sum(minus))
    grad_estimate = scale * (np.sum(plus_traj[:, 1]) - np.sum(minus_traj[:, 1]))
    return grad_estimate


def WD_v_estimator_micro_rep(N, v, beta, T, i0, n_micro, macro_idx):
    """Average across n_micro replications of the Weak-Derivative estimator for v."""
    seed_offset = macro_idx * n_micro

    t0 = time.perf_counter()
    samples_v = [
        wd_v_single_sample(
            N,
            v,
            i0,
            beta,
            T,
            pop_seed=(seed_offset + i) * 42,
            dyn_seed=(seed_offset + i) * 17,
        )
        for i in range(n_micro)
    ]
    time_v = time.perf_counter() - t0

    return np.mean(samples_v), time_v


def WD_v_estimator_worker(args):
    """Worker executing macro replications and computing metrics against true J_v."""
    N, v, beta, T, i0, n_macro, n_micro, J_v = args

    reps = np.array(
        [
            WD_v_estimator_micro_rep(N, v, beta, T, i0, n_micro, macro_idx=m)
            for m in range(n_macro)
        ]
    )

    # Compute aggregate performance metrics across macro runs
    mean_v = np.mean(reps[:, 0])
    sd_v = np.std(reps[:, 0], ddof=1) if n_macro > 1 else 0.0
    mse_v = np.mean((reps[:, 0] - J_v) ** 2)
    avg_time_v = np.mean(reps[:, 1])

    return N, v, beta, T, i0, mean_v, sd_v, mse_v, avg_time_v



input_csv = output_dir / "matrix_power_gradients.csv"
output_csv = output_dir / "wd_v_estimators_results.csv"

# Parse parameter sets and exact gradients J_v from previous matrix powers step
tasks = []
with open(input_csv, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tasks.append(
            (
                int(row["N"]),
                float(row["v"]),
                float(row["beta"]),
                int(row["T"]),
                int(row["i0"]),
                n_macro,
                n_micro,
                float(row["J_v"].strip('"')),
            )
        )

# Parallel processing across parameter configurations
with mp.Pool(cores) as pool:
    wd_results = pool.map(WD_v_estimator_worker, tasks)

# Save Weak-Derivative estimation statistics for v
with open(output_csv, "w") as f:
    f.write("N,v,beta,T,i0,mean_v,sd_v,mse_v,time_v\n")
    for row in wd_results:
        N, v, beta, T, i0, mean_v, sd_v, mse_v, time_v = row
        f.write(
            f"{N},{v},{beta},{T},{i0},{mean_v:.6f},"
            f"{sd_v:.6f},{mse_v:.6e},{time_v:.6f}\n"
        )
        
#%%
def total_inf(traj):
    """Calculates total infections over the trajectory."""
    return np.sum(traj[:, 1])


def LR_v_estimator_micro_rep(N, v, beta, T, i0, n_micro, macro_idx):
    """Average across n_micro replications of LR estimator for v using the sample-mean baseline trick."""
    seed_offset = macro_idx * n_micro

    t0 = time.perf_counter()

    total_infs = np.empty(n_micro)
    scores = np.empty(n_micro)

    # 1. Sample all trajectories and score function values
    for i in range(n_micro):
        pop_seed = (seed_offset + i) * 42
        dyn_seed = (seed_offset + i) * 17
        traj, score = tSIR_LR_v(N, v, i0, beta, T, pop_seed, dyn_seed)

        total_infs[i] = total_inf(traj)
        scores[i] = score

    # 2. Baseline trick: center total infections around sample mean Y_bar
    y_bar = np.mean(total_infs)
    centered_samples = (total_infs - y_bar) * scores

    time_v = time.perf_counter() - t0

    return np.mean(centered_samples), time_v


def LR_v_estimator_worker(args):
    """Worker executing macro replications and computing performance metrics against true J_v."""
    N, v, beta, T, i0, n_macro, n_micro, J_v = args

    reps = np.array(
        [
            LR_v_estimator_micro_rep(N, v, beta, T, i0, n_micro, macro_idx=m)
            for m in range(n_macro)
        ]
    )

    # Compute aggregate performance metrics across macro runs
    mean_v = np.mean(reps[:, 0])
    sd_v = np.std(reps[:, 0], ddof=1) if n_macro > 1 else 0.0
    mse_v = np.mean((reps[:, 0] - J_v) ** 2)
    avg_time_v = np.mean(reps[:, 1])

    return N, v, beta, T, i0, mean_v, sd_v, mse_v, avg_time_v


input_csv = output_dir / "matrix_power_gradients.csv"
output_csv = output_dir / "lr_v_estimators_results.csv"

# Parse parameter sets and exact gradients J_v from matrix powers step
tasks = []
with open(input_csv, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tasks.append(
            (
                int(row["N"]),
                float(row["v"]),
                float(row["beta"]),
                int(row["T"]),
                int(row["i0"]),
                n_macro,
                n_micro,
                float(row["J_v"].strip('"')),
            )
        )

# Parallel processing across parameter configurations
with mp.Pool(cores) as pool:
    lr_results = pool.map(LR_v_estimator_worker, tasks)

# Save Likelihood Ratio estimation statistics for v
with open(output_csv, "w") as f:
    f.write("N,v,beta,T,i0,mean_v,sd_v,mse_v,time_v\n")
    for row in lr_results:
        N, v, beta, T, i0, mean_v, sd_v, mse_v, time_v = row
        f.write(
            f"{N},{v},{beta},{T},{i0},{mean_v:.6f},"
            f"{sd_v:.6f},{mse_v:.6e},{time_v:.6f}\n"
        )

#%%
def LR_beta_estimator_micro_rep(N, v, beta, T, i0, n_micro, macro_idx):
    """
    Gather n_micro sample paths into matrices and delegate gradient calculation 
    to grad_wrt_beta.
    """
    seed_offset = macro_idx * n_micro

    t0 = time.perf_counter()

    # Pre-allocate matrices required by grad_wrt_beta
    trajectories = np.empty((T + 1, n_micro), dtype=int)
    score_samples = np.empty((T, n_micro))

    for i in range(n_micro):
        pop_seed = (seed_offset + i) * 42
        dyn_seed = (seed_offset + i) * 17

        i_traj, step_scores = tSIR_LR_beta_path(
            N, v, i0, beta, T, pop_seed, dyn_seed
        )

        trajectories[:, i] = i_traj
        score_samples[:, i] = step_scores

    # Compute gradient mean via grad_wrt_beta
    mean_beta, _ = grad_wrt_beta(trajectories, score_samples, T, n_micro)
    time_beta = time.perf_counter() - t0

    return mean_beta, time_beta


def LR_beta_estimator_worker(args):
    """Worker executing macro replications and measuring performance against J_beta."""
    N, v, beta, T, i0, n_macro, n_micro, J_beta = args

    reps = np.array(
        [
            LR_beta_estimator_micro_rep(
                N, v, beta, T, i0, n_micro, macro_idx=m
            )
            for m in range(n_macro)
        ]
    )

    mean_beta = np.mean(reps[:, 0])
    sd_beta = np.std(reps[:, 0], ddof=1) if n_macro > 1 else 0.0
    mse_beta = np.mean((reps[:, 0] - J_beta) ** 2)
    avg_time_beta = np.mean(reps[:, 1])

    return N, v, beta, T, i0, mean_beta, sd_beta, mse_beta, avg_time_beta


# Execution Block
if __name__ == "__main__":
    input_csv = output_dir / "matrix_power_gradients.csv"
    output_csv = output_dir / "lr_beta_estimators_results.csv"

    tasks = []
    with open(input_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(
                (
                    int(row["N"]),
                    float(row["v"]),
                    float(row["beta"]),
                    int(row["T"]),
                    int(row["i0"]),
                    n_macro,
                    n_micro,
                    float(row["J_beta"].strip('"')),
                )
            )

    with mp.Pool(cores) as pool:
        lr_results = pool.map(LR_beta_estimator_worker, tasks)

    with open(output_csv, "w") as f:
        f.write("N,v,beta,T,i0,mean_beta,sd_beta,mse_beta,time_beta\n")
        for row in lr_results:
            N, v, beta, T, i0, mean_beta, sd_beta, mse_beta, time_beta = row
            f.write(
                f"{N},{v},{beta},{T},{i0},{mean_beta:.6f},"
                f"{sd_beta:.6f},{mse_beta:.6e},{time_beta:.6f}\n"
            )