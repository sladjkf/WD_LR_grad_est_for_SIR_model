#!/usr/bin/env python3
"""
Stochastic tSIR Finite State Projection (FSP) vs. Monte Carlo Simulator
========================================================================
This module implements both a vectorized Monte Carlo simulator and an optimized
Parallel MapReduce Finite State Projection (FSP) solver for multi-population 
time-discrete SIR (tSIR) epidemic models. 

The FSP solver incorporates local mode windowing, adaptive 1D tail truncation, 
top-M successor state capping, and lock-free parallel execution to provide 
deterministic expected values with explicit error upper bounds.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson, nbinom
from itertools import product
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import heapq
import random

# =====================================================================
# 1. NOMINAL SIMULATOR ENGINE (FOR MONTE CARLO)
# =====================================================================

def step_wd(U_dyn, S, I, N, beta, W):
    """
    Advances a single discrete time step for M parallel simulation paths 
    across K locations using uniform random variates (Inversion Method).

    Parameters:
    -----------
    U_dyn : np.ndarray of shape (M, K)
        Uniform random variates U ~ Uniform(0, 1) for path transitions.
    S, I  : np.ndarray of shape (M, K)
        Current susceptible and infected populations for M paths across K locations.
    N, beta : np.ndarray of shape (K,)
        Total population size and contact rates per location.
    W     : np.ndarray of shape (K, K)
        Inter-location connectivity / mixing matrix.

    Returns:
    --------
    next_S, next_I : Tuple[np.ndarray, np.ndarray]
        Updated susceptible and infected counts of shape (M, K) for step t+1.
    """
    # Step 1: Compute effective infectious population at each location via mixing matrix W
    # I @ W yields an (M, K) matrix representing cross-location infection pressures
    effective_I = I @ W

    # Step 2: Compute force of infection lambda_k = (S_k * beta_k / N_k) * (I_eff)_k
    lambda_val = (S * beta / N) * effective_I

    # Step 3: Draw Poisson candidate infections (used when current infections I_k = 0)
    y_pois = poisson.ppf(U_dyn, lambda_val)

    # Step 4: Draw Negative Binomial candidate infections (used when I_k > 0)
    # Safe division masks avoid divide-by-zero errors in zero-infection paths
    safe_denom = np.maximum(I + lambda_val, 1e-12)
    p = np.where(I > 0, I / safe_denom, 1.0)
    safe_n = np.maximum(I, 1e-12)
    y_nbinom = nbinom.ppf(U_dyn, safe_n, p)

    # Step 5: Select Poisson model for I = 0 and Negative Binomial for I > 0
    Y = np.where(I > 0, y_nbinom, y_pois)

    # Step 6: Enforce capacity constraint — newly infected cannot exceed current susceptibles S
    next_I = np.minimum(S, Y)
    next_S = S - next_I

    return next_S, next_I


def simulate_tsir_trajectories(S_0, I_0, N, beta, W, T, dyn_seed):
    """
    Core vectorized simulation engine propagating M parallel state trajectories
    over time horizon T using Common Random Numbers (CRN) or independent paths.

    Parameters:
    -----------
    S_0, I_0 : np.ndarray of shape (M, K)
        Initial susceptible and infected populations for M parallel paths.
    N, beta  : np.ndarray of shape (K,)
        Location population sizes and transmission coefficients.
    W        : np.ndarray of shape (K, K)
        Spatial connectivity matrix.
    T        : int
        Total simulation time horizon (number of discrete time steps).
    dyn_seed : int
        RNG seed controlling dynamic transitions over time.

    Returns:
    --------
    S_traj, I_traj : Tuple[np.ndarray, np.ndarray]
        State trajectory tensors of shape (T+1, M, K).
    """
    # Extract dimensions: M parallel paths, K locations
    M, K = S_0.shape
    dyn_rng = np.random.default_rng(dyn_seed)

    # Initialize trajectory storage tensors: (TimeSteps, Paths, Locations)
    S_traj = np.zeros((T + 1, M, K))
    I_traj = np.zeros((T + 1, M, K))

    # Set initial state at time t = 0
    S_traj[0] = S_0
    I_traj[0] = I_0

    S_curr = S_0.copy()
    I_curr = I_0.copy()

    # Time-stepping loop: propagate state from t -> t+1
    for t in range(T):
        # Draw independent uniform random matrix of shape (M, K) for Monte Carlo paths
        U_dyn = dyn_rng.random((M, K))

        # Advance state across all M paths simultaneously
        S_curr, I_curr = step_wd(U_dyn, S_curr, I_curr, N, beta, W)

        # Record state at timestep t+1
        S_traj[t + 1] = S_curr
        I_traj[t + 1] = I_curr

    return S_traj, I_traj


def run_monte_carlo(N, i0, V, beta, W, T, num_sims, base_seed=42):
    """
    Wrapper function executing M Monte Carlo simulations initialized from a
    Binomial distribution over susceptible counts.

    Parameters:
    -----------
    N, i0, V, beta, W, T : Model parameters.
    num_sims : int
        Number of independent stochastic realizations M to sample.
    base_seed : int
        Seed for initial population sampling and dynamic progression.

    Returns:
    --------
    I_traj : np.ndarray of shape (T+1, num_sims, K)
        Infected population trajectories across all Monte Carlo paths.
    """
    K = len(N)
    pop_rng = np.random.default_rng(base_seed)
    
    # Compute initial susceptible Binomial success probability p = 1 - V / (N - i0)
    p = 1.0 - (V / (N - i0))

    # Sample initial uniform variates for population setup across all M simulations
    U_pop = pop_rng.random((num_sims, K))
    
    # Sample initial susceptible counts S_0 ~ Binomial(N - i0, p)
    S_0 = binom.ppf(U_pop, n=N - i0, p=p)

    # Tile initial infected vector i0 across all num_sims rows
    I_0 = np.tile(i0, (num_sims, 1)).astype(float)

    # Run vectorized simulation trajectories
    S_traj, I_traj = simulate_tsir_trajectories(S_0, I_0, N, beta, W, T, dyn_seed=base_seed + 1)
    return I_traj


# =====================================================================
# 2. OPTIMIZED FSP SOLVER WITH ERROR BOUND TRACKING
# =====================================================================

def get_location_transition_pmf_adaptive(S_k, I_k, lambda_k, z_score=6.0, mass_cutoff=1e-10):
    """
    Generates a pruned 1D transition PMF for location k using local mode 
    windowing (z-score bounding around mean) and cumulative mass truncation.

    Parameters:
    -----------
    S_k, I_k : float
        Current susceptible and infected counts at location k.
    lambda_k : float
        Current infection intensity parameter at location k.
    z_score : float
        Number of standard deviations around distribution mean to search.
    mass_cutoff : float
        Cumulative probability target to retain before cutting off the tail.

    Returns:
    --------
    pmf_dict : Dict[int, float]
        Dictionary mapping potential newly infected counts Y_k to probability P(Y_k).
    """
    # Edge case: No susceptible individuals remaining or zero infection force
    if S_k == 0 or lambda_k <= 0:
        return {0: 1.0}

    # Case A: Existing infection present -> Negative Binomial model
    if I_k > 0:
        n_param = I_k
        p_param = I_k / (I_k + lambda_k)
        dist = nbinom(n=n_param, p=p_param)
        mean_val = n_param * (1 - p_param) / p_param
        var_val = mean_val / p_param
    # Case B: Zero active infection -> Poisson model
    else:
        dist = poisson(mu=lambda_k)
        mean_val = lambda_k
        var_val = lambda_k

    std_val = np.sqrt(var_val)

    # Heuristic 3: Local Mode Windowing — bound search interval to [mean - z*std, mean + z*std]
    low_bound = max(0, int(np.floor(mean_val - z_score * std_val)))
    high_bound = min(int(S_k), int(np.ceil(mean_val + z_score * std_val)))

    pmf_dict = {}
    accumulated_prob = 0.0

    # Evaluate distribution PMF only within the high-density local window
    for i in range(low_bound, high_bound):
        prob = dist.pmf(i)
        if prob > 1e-14:
            pmf_dict[i] = prob
            accumulated_prob += prob

    # Boundary term handling: Absorption at max capacity S_k (P(Y_k >= S_k) = 1 - CDF(S_k - 1))
    if high_bound >= S_k:
        p_max = max(0.0, 1.0 - dist.cdf(S_k - 1))
        if p_max > 1e-14:
            pmf_dict[int(S_k)] = p_max

    # Heuristic 1: 1D Tail Truncation — keep only top probable states covering (1 - mass_cutoff)
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
    """
    Parallel worker task: Advances a partitioned subset of active states independently
    without shared memory mutex locks (Lock-Free Map phase).

    Parameters:
    -----------
    chunk_states : List[Tuple[Tuple, float]]
        Sub-list of state items [(state_tuple, prob), ...] assigned to this worker process.
    N, beta, W, tol, max_succ_per_state : Model parameters and truncation settings.

    Returns:
    --------
    P_sub : Dict[Tuple, float]
        Local thread-safe dictionary containing aggregated next-step probability mass.
    """
    K = len(N)
    P_sub = defaultdict(float)

    # Loop over all active states assigned to this process chunk
    for state, p_state in chunk_states:
        if p_state < tol:
            continue

        # Unpack state tuple into Susceptible (0..K-1) and Infected (K..2K-1) vectors
        S_vec = state[:K]
        I_vec = state[K:]

        # Compute effective infections and intensity vector lambda
        effective_I = np.dot(I_vec, W)
        lambda_vec = (np.array(S_vec) * beta / N) * effective_I

        # Build 1D adaptive transition PMFs for each location
        loc_pmfs = []
        for k in range(K):
            pmf_k = get_location_transition_pmf_adaptive(S_vec[k], I_vec[k], lambda_vec[k])
            loc_pmfs.append(pmf_k.items())

        # Single-pass generation: compute probability once and filter out <= 0 entries
        scored_transitions = []
        for transition in product(*loc_pmfs):
            trans_prob = np.prod([t[1] for t in transition])
            if trans_prob > 0:
                scored_transitions.append((trans_prob, transition))

        # Heuristic 1 (b): Extract top-M transitions using an efficient min-heap O(N log M)
        if max_succ_per_state and len(scored_transitions) > max_succ_per_state:
            top_transitions = heapq.nlargest(
                max_succ_per_state, scored_transitions, key=lambda x: x[0]
            )
        else:
            top_transitions = scored_transitions

        # Propagate probability mass using the precalculated transition probabilities
        for trans_prob, transition in top_transitions:
            new_I_tuple = tuple(t[0] for t in transition)
            new_S_tuple = tuple(S_vec[k] - new_I_tuple[k] for k in range(K))
            next_state = new_S_tuple + new_I_tuple

            # Accumulate probability mass: P_{t+1}(x') += P_t(x) * P(x'|x)
            P_sub[next_state] += p_state * trans_prob

    return P_sub

def tsir_fsp_step_parallel(P_curr, N, beta, W, tol=1e-12, num_workers=4, max_succ_per_state=50):
    """
    MapReduce manager orchestrating multi-process parallel evaluation of an 
    FSP time step across active states using deterministic round-robin load balancing.

    Parameters:
    -----------
    P_curr : Dict[Tuple, float]
        Current joint probability distribution mapping state tuple -> probability.
    N, beta, W, tol, max_succ_per_state : Model parameters and truncation settings.
    num_workers : int
        Number of parallel CPU worker processes to spawn.

    Returns:
    --------
    P_next : Dict[Tuple, float]
        Updated probability distribution at timestep t+1.
    """
    items = list(P_curr.items())
    if len(items) == 0:
        return {}

    # Map Step: Distribute items via round-robin striding across worker processes.
    # Worker 0 gets indices [0, num_workers, 2*num_workers, ...]
    # Worker 1 gets indices [1, 1 + num_workers, 1 + 2*num_workers, ...]
    # This interleaving guarantees contiguous high-workload state clusters are evenly split.
    chunks = [items[i::num_workers] for i in range(num_workers) if i < len(items)]

    P_next = defaultdict(float)

    # Lock-free execution pool across process boundaries
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_process_state_chunk, chunk, N, beta, W, tol, max_succ_per_state)
            for chunk in chunks
        ]
        
        # Reduce Step: Merge worker sub-dictionaries into master distribution
        for future in futures:
            P_sub = future.result()
            for st, p in P_sub.items():
                P_next[st] += p

    # Filter out states whose cumulative probability mass falls below tolerance
    return {st: p for st, p in P_next.items() if p >= tol}


def tsir_exact_fsp_optimized(N, i0, V, beta, W, T, tol=1e-12, num_workers=4, max_succ_per_state=50):
    """
    Main driver function performing multi-step FSP propagation to obtain time-series
    of state distributions P_t for t = 0..T.

    Returns:
    --------
    distributions : List[Dict[Tuple, float]]
        List of length T+1 containing exact state probability dictionaries.
    """
    K = len(N)
    p_initial = 1.0 - (V / (N - i0))
    initial_loc_pmfs = []

    # Construct initial Binomial PMFs for susceptible counts at t = 0
    for k in range(K):
        n_k = N[k] - i0[k]
        p_k = p_initial[k]
        loc_dict = {s: binom.pmf(s, n_k, p_k) for s in range(n_k + 1) if binom.pmf(s, n_k, p_k) > tol}
        initial_loc_pmfs.append(loc_dict.items())

    # Build joint initial probability distribution P_0 over initial state Cartesian product
    P_0 = {}
    for S_init in product(*initial_loc_pmfs):
        S_tuple = tuple(t[0] for t in S_init)
        prob = np.prod([t[1] for t in S_init])
        state = S_tuple + tuple(i0)
        P_0[state] = prob

    distributions = [P_0]
    P_curr = P_0

    # Sequentially advance state distribution through time steps t = 1..T
    for t in range(T):
        P_curr = tsir_fsp_step_parallel(
            P_curr, N, beta, W, tol=tol, num_workers=num_workers, max_succ_per_state=max_succ_per_state
        )
        distributions.append(P_curr)

    return distributions


# =====================================================================
# 3. BENCHMARKING & PRINTING PREDICTED ERROR BOUNDS
# =====================================================================
if __name__ == "__main__":
    # Define model hyper-parameters for a 2-location system
    N = np.array([10, 10, 10])
    i0 = np.array([1, 0, 1])
    V = np.array([1.0, 1.0, 1.0])
    beta = np.array([3.0, 4.0, 5.0])
    W = np.array([[1.0, 0.25, 0.25], 
                  [0.25, 1.0, 0.25],
                  [0.25, 0.25, 1.0]
                  ])
    T = 5
    K = len(N)
    num_sims = 10_000

    print("--- 1. Running Optimized FSP Solver ---")
    fsp_dists = tsir_exact_fsp_optimized(
        N, i0, V, beta, W, T, tol=1e-12, num_workers=10, max_succ_per_state=1000
    )

    # Initialize expectation arrays and error tracking metrics
    fsp_expected_I = np.zeros((T + 1, K))
    retained_mass = np.zeros(T + 1)
    discarded_mass = np.zeros(T + 1)
    error_bounds = np.zeros((T + 1, K))

    print("\n" + "="*85)
    print(f"{'t':<3} | {'Retained Mass':<15} | {'Discarded Mass (eps)':<22} | {'Max Error E[I_1]':<18} | {'Max Error E[I_2]':<18}")
    print("="*85)

    # Calculate exact expectations and compute rigorous truncation error upper bounds
    for t, P_t in enumerate(fsp_dists):
        retained_p = sum(P_t.values())
        eps_t = max(0.0, 1.0 - retained_p)  # Discarded probability mass
        
        retained_mass[t] = retained_p
        discarded_mass[t] = eps_t

        # Accumulate expectation: E[I_k] = sum_{x} i_k * P_t(x)
        for state, prob in P_t.items():
            I_vec = state[K:]
            fsp_expected_I[t, :] += prob * np.array(I_vec)

        # Mathematical upper bound on absolute error: Error_k <= N_k * eps_t
        error_bounds[t, :] = eps_t * N

        print(f"{t:<3} | {retained_p:<15.10f} | {eps_t:<22.6e} | {error_bounds[t, 0]:<18.6e} | {error_bounds[t, 1]:<18.6e}")

    print("="*85)

    print("\n--- 2. Running Monte Carlo Simulation (M = 10,000) ---")
    mc_I_traj = run_monte_carlo(N, i0, V, beta, W, T, num_sims=num_sims)

    # Compute Monte Carlo empirical mean and 95% Confidence Interval bounds
    mc_mean = np.mean(mc_I_traj, axis=1)
    mc_std = np.std(mc_I_traj, axis=1, ddof=1)
    mc_se = mc_std / np.sqrt(num_sims)
    ci_95 = 1.96 * mc_se
#%%
    # ---------------------------------------------------------
    # 4. Plotting Exact FSP Path vs. Monte Carlo
    # ---------------------------------------------------------
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 6))
    time_steps = np.arange(T + 1)
    #loc_colors = ['#1f77b4', '#ff7f0e', 'red']
    loc_colors = sns.color_palette("tab10", K).as_hex()

    for k in range(K):
        # Plot exact FSP path
        ax.plot(
            time_steps, 
            fsp_expected_I[:, k], 
            color=loc_colors[k], 
            linestyle='-', 
            linewidth=2.5, 
            label=f'FSP $\\mathbb{{E}}[I_{{{k+1}, t}}]$'
        )
        
        # Overlay Monte Carlo mean with 95% CI error bars
        ax.errorbar(
            time_steps, 
            mc_mean[:, k], 
            yerr=ci_95[:, k], 
            color=loc_colors[k], 
            fmt='o', 
            capsize=4, 
            capthick=1.5, 
            markersize=5, 
            linestyle='None', 
            label=f'MC Mean $\\pm$ 95% CI (Loc {k+1})'
        )

    ax.set_xlabel('Time Step ($t$)', fontsize=12)
    ax.set_ylabel('Expected Infected Population', fontsize=12)
    ax.set_title('Exact FSP vs. Monte Carlo with FSP Truncation Error Bounds', fontsize=14, pad=12)
    ax.legend(frameon=True, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()