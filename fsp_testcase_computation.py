#!/usr/bin/env python3
"""
Incremental Jacobian Calculator for tSIR Finite State Projection (FSP)
=======================================================================
Evaluates finite-difference Jacobians of cumulative expected infections 
sum_{t=0}^T E[I_{k,t}] with respect to vaccination parameters (V) and contact rates (beta)
for two benchmark spatial systems. Results are saved incrementally to CSV.
"""

import csv
import json
import heapq
import numpy as np
from scipy.stats import binom, poisson, nbinom
from itertools import product
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor


# =====================================================================
# GLOBAL CONFIGURATION & COMPUTATIONAL PARAMETERS
# =====================================================================
NUM_WORKERS = 4              # Number of CPU worker processes to spawn
MAX_SUCC_PER_STATE = 50      # Top-M successor state cap per active state
TOL = 1e-12                  # Truncation probability mass cutoff threshold
EPS_V = 1e-3                 # Finite-difference step size for V
EPS_BETA = 1e-3              # Finite-difference step size for beta
Z_SCORE = 6.0                # Standard deviation search radius for local mode windowing
MASS_CUTOFF = 1e-10          # 1D transition PMF cumulative tail cutoff target
OUTPUT_FILENAME = "fsp_jacobians_results.csv"  # Target output CSV path


# =====================================================================
# 1. OPTIMIZED FSP SOLVER ENGINE
# =====================================================================

def get_location_transition_pmf_adaptive(S_k, I_k, lambda_k, z_score=Z_SCORE, mass_cutoff=MASS_CUTOFF):
    """Generates a pruned 1D transition PMF using local mode windowing & tail truncation."""
    if S_k == 0 or lambda_k <= 0:
        return {0: 1.0}

    if I_k > 0:
        n_param = I_k
        p_param = I_k / (I_k + lambda_k)
        dist = nbinom(n=n_param, p=p_param)
        mean_val = n_param * (1 - p_param) / p_param
        var_val = mean_val / p_param
    else:
        dist = poisson(mu=lambda_k)
        mean_val = lambda_k
        var_val = lambda_k

    std_val = np.sqrt(var_val)
    low_bound = max(0, int(np.floor(mean_val - z_score * std_val)))
    high_bound = min(int(S_k), int(np.ceil(mean_val + z_score * std_val)))

    pmf_dict = {}
    for i in range(low_bound, high_bound):
        prob = dist.pmf(i)
        if prob > 1e-14:
            pmf_dict[i] = prob

    if high_bound >= S_k:
        p_max = max(0.0, 1.0 - dist.cdf(S_k - 1))
        if p_max > 1e-14:
            pmf_dict[int(S_k)] = p_max

    if len(pmf_dict) > 1:
        sorted_items = sorted(pmf_dict.items(), key=lambda x: x[1], reverse=True)
        trimmed_dict = {}
        cum_p = 0.0
        for val, p in sorted_items:
            trimmed_dict[val] = p
            cum_p += p
            if cum_p >= (1.0 - mass_cutoff):
                break
        return trimmed_dict

    return pmf_dict


def _process_state_chunk(chunk_states, N, beta, W, tol, max_succ_per_state):
    """Parallel worker task evaluating successor transitions for assigned state chunks."""
    K = len(N)
    P_sub = defaultdict(float)

    for state, p_state in chunk_states:
        if p_state < tol:
            continue

        S_vec = state[:K]
        I_vec = state[K:]

        effective_I = np.dot(I_vec, W)
        lambda_vec = (np.array(S_vec) * beta / N) * effective_I

        loc_pmfs = []
        for k in range(K):
            pmf_k = get_location_transition_pmf_adaptive(S_vec[k], I_vec[k], lambda_vec[k])
            loc_pmfs.append(pmf_k.items())

        scored_transitions = []
        for transition in product(*loc_pmfs):
            trans_prob = np.prod([t[1] for t in transition])
            if trans_prob > 0:
                scored_transitions.append((trans_prob, transition))

        if max_succ_per_state and len(scored_transitions) > max_succ_per_state:
            top_transitions = heapq.nlargest(
                max_succ_per_state, scored_transitions, key=lambda x: x[0]
            )
        else:
            top_transitions = scored_transitions

        for trans_prob, transition in top_transitions:
            new_I_tuple = tuple(t[0] for t in transition)
            new_S_tuple = tuple(S_vec[k] - new_I_tuple[k] for k in range(K))
            next_state = new_S_tuple + new_I_tuple
            P_sub[next_state] += p_state * trans_prob

    return P_sub


def tsir_fsp_step_parallel(
    P_curr, N, beta, W, tol=TOL, num_workers=NUM_WORKERS, max_succ_per_state=MAX_SUCC_PER_STATE
):
    """Orchestrates FSP step evaluation using round-robin striding for load balancing."""
    items = list(P_curr.items())
    if len(items) == 0:
        return {}

    chunks = [items[i::num_workers] for i in range(num_workers) if i < len(items)]
    P_next = defaultdict(float)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_process_state_chunk, chunk, N, beta, W, tol, max_succ_per_state)
            for chunk in chunks
        ]
        for future in futures:
            P_sub = future.result()
            for st, p in P_sub.items():
                P_next[st] += p

    return {st: p for st, p in P_next.items() if p >= tol}


def tsir_exact_fsp_optimized(
    N, i0, V, beta, W, T, tol=TOL, num_workers=NUM_WORKERS, max_succ_per_state=MAX_SUCC_PER_STATE
):
    """Executes multi-step FSP distribution propagation."""
    K = len(N)
    p_initial = 1.0 - (V / (N - i0))
    initial_loc_pmfs = []

    for k in range(K):
        n_k = int(N[k] - i0[k])
        p_k = p_initial[k]
        loc_dict = {s: binom.pmf(s, n_k, p_k) for s in range(n_k + 1) if binom.pmf(s, n_k, p_k) > tol}
        initial_loc_pmfs.append(loc_dict.items())

    P_0 = {}
    for S_init in product(*initial_loc_pmfs):
        S_tuple = tuple(t[0] for t in S_init)
        prob = np.prod([t[1] for t in S_init])
        state = S_tuple + tuple(i0)
        P_0[state] = prob

    distributions = [P_0]
    P_curr = P_0

    for t in range(T):
        P_curr = tsir_fsp_step_parallel(
            P_curr, N, beta, W, tol=tol, num_workers=num_workers, max_succ_per_state=max_succ_per_state
        )
        distributions.append(P_curr)

    return distributions


# =====================================================================
# 2. JACOBIAN EVALUATOR
# =====================================================================

def compute_fsp_jacobians(
    N, i0, V, beta, W, T, 
    eps_V=EPS_V, 
    eps_beta=EPS_BETA, 
    tol=TOL, 
    num_workers=NUM_WORKERS, 
    max_succ_per_state=MAX_SUCC_PER_STATE
):
    """
    Computes Jacobians of sum_{t=0}^T E[I_{k,t}] with respect to V and beta
    along with error intervals derived from cumulative FSP truncation loss.
    """
    N = np.asarray(N, dtype=float)
    i0 = np.asarray(i0, dtype=float)
    V = np.asarray(V, dtype=float)
    beta = np.asarray(beta, dtype=float)
    K = len(N)

    def _get_cumulative_expectation_and_bounds(fsp_dists):
        cum_E_I = np.zeros(K)
        tot_eps_fsp = 0.0
        for P_t in fsp_dists:
            retained_p = sum(P_t.values())
            tot_eps_fsp += max(0.0, 1.0 - retained_p)
            for state, prob in P_t.items():
                I_vec = state[K:]
                cum_E_I += prob * np.array(I_vec)
        a = cum_E_I
        b = cum_E_I + N * tot_eps_fsp
        return cum_E_I, a, b

    # Nominal calculation
    dists_nom = tsir_exact_fsp_optimized(
        N, i0, V, beta, W, T, tol=tol, num_workers=num_workers, max_succ_per_state=max_succ_per_state
    )
    E_nom, a_nom, b_nom = _get_cumulative_expectation_and_bounds(dists_nom)

    J_V = np.zeros((K, K))
    J_V_interval = np.zeros((K, K, 2))
    J_beta = np.zeros((K, K))
    J_beta_interval = np.zeros((K, K, 2))

    # Jacobians w.r.t V
    V_max = N - i0
    for k in range(K):
        if V[k] + eps_V <= V_max[k]:
            h_k = eps_V
        elif V[k] - eps_V >= 0.0:
            h_k = -eps_V
        else:
            h_k = min(eps_V, V_max[k] - V[k])

        V_pert = V.copy()
        V_pert[k] += h_k
        dists_pert = tsir_exact_fsp_optimized(
            N, i0, V_pert, beta, W, T, tol=tol, num_workers=num_workers, max_succ_per_state=max_succ_per_state
        )
        E_pert, a_pert, b_pert = _get_cumulative_expectation_and_bounds(dists_pert)

        J_V[:, k] = (E_pert - E_nom) / h_k
        if h_k > 0:
            J_V_interval[:, k, 0] = (a_pert - b_nom) / h_k
            J_V_interval[:, k, 1] = (b_pert - a_nom) / h_k
        else:
            J_V_interval[:, k, 0] = (b_pert - a_nom) / h_k
            J_V_interval[:, k, 1] = (a_pert - b_nom) / h_k

    # Jacobians w.r.t beta
    for k in range(K):
        h_k = eps_beta if beta[k] + eps_beta >= 0.0 else -eps_beta
        beta_pert = beta.copy()
        beta_pert[k] += h_k
        dists_pert = tsir_exact_fsp_optimized(
            N, i0, V, beta_pert, W, T, tol=tol, num_workers=num_workers, max_succ_per_state=max_succ_per_state
        )
        E_pert, a_pert, b_pert = _get_cumulative_expectation_and_bounds(dists_pert)

        J_beta[:, k] = (E_pert - E_nom) / h_k
        if h_k > 0:
            J_beta_interval[:, k, 0] = (a_pert - b_nom) / h_k
            J_beta_interval[:, k, 1] = (b_pert - a_nom) / h_k
        else:
            J_beta_interval[:, k, 0] = (b_pert - a_nom) / h_k
            J_beta_interval[:, k, 1] = (a_pert - b_nom) / h_k

    return {
        'J_V': J_V,
        'J_V_interval': J_V_interval,
        'J_beta': J_beta,
        'J_beta_interval': J_beta_interval,
        'cum_E_I_nom': E_nom
    }


# =====================================================================
# 3. INCREMENTAL CSV BENCHMARK DRIVER
# =====================================================================

def run_benchmarks_and_save_csv(output_filename=OUTPUT_FILENAME):
    """
    Generates parameter grids for Case 1 and Case 2, computes Jacobians,
    and writes results row-by-row to a CSV file as calculations complete.
    """
    fieldnames = [
        "case_id", "K", "N", "i0", "T", "W", "V", "beta",
        "cum_E_I_nom",
        "J_V", "J_V_interval",
        "J_beta", "J_beta_interval"
    ]

    with open(output_filename, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.flush()

        # -------------------------------------------------------------
        # CASE 1: 2-Location System (16 Combinations)
        # -------------------------------------------------------------
        print("\n=== Running Benchmark Case 1 (K=2, 16 Cases) ===")
        N_case1 = [15, 15]
        i0_case1 = [1, 0]
        T_case1 = 5
        W_case1 = np.array([[1.0, 0.25], [0.25, 1.0]])

        # Restricted grids: [1, 8]^2 x [4, 8]^2
        V_grid_case1 = list(product([1, 8], repeat=2))
        beta_grid_case1 = list(product([4, 8], repeat=2))
        total_c1 = len(V_grid_case1) * len(beta_grid_case1)

        count = 0
        for V_tuple in V_grid_case1:
            for beta_tuple in beta_grid_case1:
                count += 1
                V = np.array(V_tuple, dtype=float)
                beta = np.array(beta_tuple, dtype=float)

                print(f"[Case 1 ({count}/{total_c1})] Evaluating V={V_tuple}, beta={beta_tuple}...")
                
                res = compute_fsp_jacobians(N_case1, i0_case1, V, beta, W_case1, T_case1)

                row_data = {
                    "case_id": 1,
                    "K": 2,
                    "N": json.dumps(N_case1),
                    "i0": json.dumps(i0_case1),
                    "T": T_case1,
                    "W": json.dumps(W_case1.tolist()),
                    "V": json.dumps(list(V_tuple)),
                    "beta": json.dumps(list(beta_tuple)),
                    "cum_E_I_nom": json.dumps(res['cum_E_I_nom'].tolist()),
                    "J_V": json.dumps(res['J_V'].tolist()),
                    "J_V_interval": json.dumps(res['J_V_interval'].tolist()),
                    "J_beta": json.dumps(res['J_beta'].tolist()),
                    "J_beta_interval": json.dumps(res['J_beta_interval'].tolist())
                }

                writer.writerow(row_data)
                csv_file.flush()

        # -------------------------------------------------------------
        # CASE 2: 3-Location System (9 Combinations)
        # -------------------------------------------------------------
        print("\n=== Running Benchmark Case 2 (K=3, 9 Cases) ===")
        N_case2 = [10, 10, 10]
        i0_case2 = [1, 0, 0]
        T_case2 = 5
        W_case2 = np.array([
            [1.0, 0.25, 0.25],
            [0.25, 1.0, 0.25],
            [0.25, 0.25, 1.0]
        ])

        V_grid_case2 = [[8, 2, 2], [2, 8, 2], [2, 2, 8]]
        beta_grid_case2 = [[8, 4, 4], [4, 8, 4], [4, 4, 8]]
        total_c2 = len(V_grid_case2) * len(beta_grid_case2)

        count = 0
        for V_list in V_grid_case2:
            for beta_list in beta_grid_case2:
                count += 1
                V = np.array(V_list, dtype=float)
                beta = np.array(beta_list, dtype=float)

                print(f"[Case 2 ({count}/{total_c2})] Evaluating V={V_list}, beta={beta_list}...")

                res = compute_fsp_jacobians(N_case2, i0_case2, V, beta, W_case2, T_case2)

                row_data = {
                    "case_id": 2,
                    "K": 3,
                    "N": json.dumps(N_case2),
                    "i0": json.dumps(i0_case2),
                    "T": T_case2,
                    "W": json.dumps(W_case2.tolist()),
                    "V": json.dumps(V_list),
                    "beta": json.dumps(beta_list),
                    "cum_E_I_nom": json.dumps(res['cum_E_I_nom'].tolist()),
                    "J_V": json.dumps(res['J_V'].tolist()),
                    "J_V_interval": json.dumps(res['J_V_interval'].tolist()),
                    "J_beta": json.dumps(res['J_beta'].tolist()),
                    "J_beta_interval": json.dumps(res['J_beta_interval'].tolist())
                }

                writer.writerow(row_data)
                csv_file.flush()

    print(f"\nAll benchmark cases completed! Results saved to '{output_filename}'.")


if __name__ == "__main__":
    run_benchmarks_and_save_csv(output_filename=OUTPUT_FILENAME)