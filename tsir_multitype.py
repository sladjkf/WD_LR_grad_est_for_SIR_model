#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 12:54:00 2026

@author: nicholasw


Simulators and gradient estimators for a multipopulation tSIR model.
"""
import numpy as np
from scipy.stats import nbinom, binom, poisson
from itertools import product
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import heapq

# def tsir_multipop(N, i0, V, beta, W, T, obj_fn, pop_seed, dyn_seed):
#     """
#     Simulate a single nominal path of a multi-population tSIR model and evaluate obj_fn.

#     Parameters:
#     --------
#     N: np.ndarray[int] of length K
#         Population vector indexed by location.
#     i0: np.ndarray[int] of length K
#         Number of initially infected individuals indexed by location.
#     V: np.ndarray[float] of length K
#         Number of vaccinated individuals indexed by location.
#     beta: np.ndarray[float] of length K
#         Contact rate parameters indexed by location.
#     W: np.ndarray[float, float] of size K x K
#         Weight/mixing matrix, doubly indexed by location.
#     T: int
#         Number of periods to run the simulation for.
#     obj_fn: Callable[[np.ndarray], float] 
#         Sample path statistic operating on an array of size (T+1) x 2K,
#         where rows represent timesteps and columns are [(S_1, ... , S_K), (I_1, ..., I_K)].
#     pop_seed: int
#         Seed value for RNG used to generate initial state.
#     dyn_seed: int
#         Seed value for RNG used to generate dynamics.
        
#     Returns:
#     -------
#     float
#         - The evaluated objective function value obj_fn(traj).
#     """
#     K = len(N)
#     assert len(i0) == K
#     assert len(V) == K
#     assert len(beta) == K
#     assert W.shape == (K, K)

#     pop_rng = np.random.default_rng(pop_seed)
#     dyn_rng = np.random.default_rng(dyn_seed)

#     # 1. Sample initial state S_0 ~ Binom(N - i0, 1 - V / (N - i0))
#     p = 1.0 - (V / (N - i0))
#     U_pop = pop_rng.random(K)
    
#     S_0 = binom.ppf(U_pop, n=N - i0, p=p)
#     I_0 = np.array(i0, dtype=float)

#     # Trajectory storage: (T+1) x K for S and I
#     S_traj = np.zeros((T + 1, K))
#     I_traj = np.zeros((T + 1, K))
    
#     S_traj[0] = S_0
#     I_traj[0] = I_0

#     # Reshape to (1, K) so step_wd handles it as M=1 path
#     S_curr = S_0.reshape(1, K)
#     I_curr = I_0.reshape(1, K)

#     # 2. Dynamic trajectory simulation
#     for t in range(T):
#         U_dyn = dyn_rng.random((1, K))
#         S_curr, I_curr = step_wd(U_dyn, S_curr, I_curr, N, beta, W)
        
#         S_traj[t + 1] = S_curr[0]
#         I_traj[t + 1] = I_curr[0]

#     # Stack S and I horizontally to create shape (T+1) x 2K
#     traj = np.hstack([S_traj, I_traj])

#     return obj_fn(traj)

# def tsir_multipop_v(N, i0, V, beta, W, T, obj_fn, pop_seed, dyn_seed):
#     """
#     Simulate paths of a multi-population tSIR model where initial susceptibles are 
#     perturbed per-location.
    
#     Parameters:
#     --------
#     N: np.ndarray[int] of length K
#         Population vector indexed by location.
#     i0: np.ndarray[int] of length K
#         Number of initially infected individuals indexed by location.
#     V: np.ndarray[float] of length K
#         Number of vaccinated individuals indexed by location.
#     beta: np.ndarray[float] of length K
#         Contact rate parameters indexed by location.
#     W: np.ndarray[float, float] of size K x K
#         Weight/mixing matrix, doubly indexed by location.
#     T: int
#         Number of periods to run the simulation for.
#     obj_fn: Callable[[np.ndarray], float] 
#         Sample path statistic operating on arrays of size (T+1) x 2K,
#         where rows represent timestep and columns are [(S_1, ... , S_K), (I_1, ..., I_K)].
#     pop_seed: int
#         Seed value for RNG used to generate initial state.
#     dyn_seed: int
#         Seed value for RNG used to generate dynamics.
        
#     Returns:
#     -------
#     np.ndarray[float]
#         - The weak-derivative gradient estimate, of length K.
#     """
#     # Shape checks
#     K = len(N)
#     assert len(i0) == K
#     assert len(V) == K
#     assert len(beta) == K
#     assert W.shape == (K, K)

#     # Setup initial state storage
#     # Rows are the index of which v got perturbed
#     # Columns are the index of location
#     # (so each row represents one state) 
#     S_plus_0 = np.zeros((K, K))
#     S_minus_0 = np.zeros((K, K))
#     I_plus_0 = np.tile(i0, (K, 1))
#     I_minus_0 = np.tile(i0, (K, 1))
    
#     pop_rng = np.random.default_rng(pop_seed)
#     dyn_rng = np.random.default_rng(dyn_seed)
#     U_pop = pop_rng.random(K)

#     # Parameterization: p_k = 1 - V_k / (N_k - i_0,k)
#     p = 1.0 - (V / (N - i0))

#     # Sample initial condition pairs using Binomial weak derivative identity
#     for k in range(K):
#         to_perturb = np.zeros(K)
#         to_perturb[k] = 1
        
#         # Sample base binomial Binom(n_k - 1, p_k)
#         this_S_plus = binom.ppf(U_pop, n=N - i0 - to_perturb, p=p)
        
#         # S_minus is identical to S_plus everywhere except location k (+1)
#         this_S_minus = this_S_plus.copy()
#         this_S_minus[k] += 1
        
#         S_plus_0[k, :] = this_S_plus
#         S_minus_0[k, :] = this_S_minus
    
#     # Trajectory storage: shape is (T+1, K_perturbations, K_locations)
#     S_plus_traj = np.zeros((T + 1, K, K))
#     S_minus_traj = np.zeros((T + 1, K, K))
#     I_plus_traj = np.zeros((T + 1, K, K))
#     I_minus_traj = np.zeros((T + 1, K, K))
    
#     S_plus_traj[0, :, :] = S_plus_0
#     S_minus_traj[0, :, :] = S_minus_0
#     I_plus_traj[0, :, :] = I_plus_0
#     I_minus_traj[0, :, :] = I_minus_0
    
#     # Dynamic trajectory propagation
#     for t in range(T):
#         last_S_plus = S_plus_traj[t, :, :]
#         last_S_minus = S_minus_traj[t, :, :]
#         last_I_plus = I_plus_traj[t, :, :]
#         last_I_minus = I_minus_traj[t, :, :]
        
#         U_dyn = dyn_rng.random(K)
        
#         next_S_plus, next_I_plus = step_wd(U_dyn, last_S_plus, last_I_plus, N, beta, W)
#         next_S_minus, next_I_minus = step_wd(U_dyn, last_S_minus, last_I_minus, N, beta, W)
        
#         S_plus_traj[t + 1, :, :] = next_S_plus
#         S_minus_traj[t + 1, :, :] = next_S_minus
#         I_plus_traj[t + 1, :, :] = next_I_plus
#         I_minus_traj[t + 1, :, :] = next_I_minus
    
#     # Objective evaluation and gradient assembly
#     grad = np.zeros(K)
    
#     for k in range(K):
#         # Format trajectory matrices into shape (T+1) x 2K: [S_1..S_K, I_1..I_K]
#         traj_plus = np.hstack([S_plus_traj[:, k, :], I_plus_traj[:, k, :]])
#         traj_minus = np.hstack([S_minus_traj[:, k, :], I_minus_traj[:, k, :]])
        
#         val_plus = obj_fn(traj_plus)
#         val_minus = obj_fn(traj_minus)
        
#         # Gradient directly simplifies to val_plus - val_minus
#         grad[k] = val_plus - val_minus
        
#     return grad

####################################
# Core simulation engine functions #
####################################

def step_wd(U_dyn, S, I, N, beta, W):
    """
    Compute a single step of the multipopulation TSIR model.
    Automatically broadcasts across M simulations and K locations.

    Parameters:
    -----------
    U_dyn : np.ndarray of shape (M, K)
        Uniform random variables U ~ Unif(0, 1) for inverse CDF sampling.
        (Broadcast 1 x K array of uniform RV for CRN)
        (Sample a full array in case you want independent replicates)
    S, I  : np.ndarray of shape (M, K)
        Current susceptible and infectious counts for M paths and K locations.
    N     : np.ndarray of shape (K,) or (1, K)
        Total population sizes for each location.
    beta  : np.ndarray of shape (K,) or (1, K)
        Transmission parameters for each location.
    W     : np.ndarray of shape (K, K)
        Spatial transmission matrix where W[j, k] is the contact weight from j to k.

    Returns:
    --------
    next_S, next_I : np.ndarray of shape (M, K)
        Updated susceptible and infectious populations.
    """
    # 1. Compute spatial contact sum: (I @ W)[m, k] = sum_j (I[m, j] * W[j, k])
    effective_I = I @ W
    
    # 2. Compute rate lambda_{k, t+1} for each path and location
    lambda_val = (S * beta / N) * effective_I

    # 3. Sample Y_{t+1} using Inverse Transform Sampling (PPF)
    
    # Case A: Poisson sample for I == 0
    y_pois = poisson.ppf(U_dyn, lambda_val)

    # Case B: Negative Binomial sample for I > 0
    # Map (lambda, I) to SciPy's nbinom parameters (n, p)
    safe_denom = np.maximum(I + lambda_val, 1e-12)
    p = np.where(I > 0, I / safe_denom, 1.0)
    safe_n = np.maximum(I, 1e-12)
    
    y_nbinom = nbinom.ppf(U_dyn, safe_n, p)

    # Combine based on condition I_t > 0
    Y = np.where(I > 0, y_nbinom, y_pois)

    # 4. Update states according to transition rules
    next_I = np.minimum(S, Y)
    next_S = S - next_I

    return next_S, next_I

def simulate_tsir_trajectories(S_0, I_0, N, beta, W, T, dyn_seed, crn=True):
    """
    Simulates M paths simultaneously over T steps given initial state matrices.
    If crn = true, synchronizes all M paths using common random numbers
    by broadcasting K random numbers across all M paths for each step.
    Otherwise, each path is sampled independently.
    
    Parameters:
    -----------
    S_0, I_0 : np.ndarray of shape (M, K)
        Initial susceptible and infected counts for M parallel paths.
    N, beta  : np.ndarray of shape (K,)
    W        : np.ndarray of shape (K, K)
    T        : int
    dyn_seed : int
    crn      : bool
        Whether to use the same stream of random numbers for all trajectories.
    
    Returns:
    --------
    S_traj, I_traj : np.ndarray of shape (T+1, M, K)
    """
    M, K = S_0.shape
    dyn_rng = np.random.default_rng(dyn_seed)

    S_traj = np.zeros((T + 1, M, K))
    I_traj = np.zeros((T + 1, M, K))

    S_traj[0] = S_0
    I_traj[0] = I_0

    S_curr = S_0.copy()
    I_curr = I_0.copy()

    for t in range(T):
        if crn:
            # Broadcast 1 set of random numbers (1, K) across all M paths for CRN synchronization
            U_dyn = dyn_rng.random((1, K))
        else:
            U_dyn = dyn_rng.random((M, K))
        S_curr, I_curr = step_wd(U_dyn, S_curr, I_curr, N, beta, W)

        S_traj[t + 1] = S_curr
        I_traj[t + 1] = I_curr

    return S_traj, I_traj

def tsir_multipop(N, i0, V, beta, W, T, obj_fn, num_sims, pop_seed, dyn_seed):
    """
    Simulate num_sims independent paths of a multi-population tSIR model and evaluate obj_fn on each path.

    Parameters:
    --------
    N: np.ndarray[int] of length K
        Population vector indexed by location.
    i0: np.ndarray[int] of length K
        Number of initially infected individuals indexed by location.
    V: np.ndarray[float] of length K
        Number of vaccinated individuals indexed by location.
    beta: np.ndarray[float] of length K
        Contact rate parameters indexed by location.
    W: np.ndarray[float, float] of size K x K
        Weight/mixing matrix, doubly indexed by location.
    T: int
        Number of periods to run the simulation for.
    obj_fn: Callable[np.ndarray[float], np.ndarray[float]] 
        Sample path statistic operating on an array of size (T+1) x 2K,
        where rows represent timesteps and columns are [(S_1, ... , S_K), (I_1, ..., I_K)].
        Can output any float-valued array; use the identity to get the trajectories back.
    num_sims: int
        Number of simulations to run.
    pop_seed: int
        Seed value for RNG used to generate initial state.
    dyn_seed: int
        Seed value for RNG used to generate dynamics.
        
    Returns:
    -------
    np.ndarray[float
        - The evaluated objective function value obj_fn(traj).
    """
    K = len(N)
    pop_rng = np.random.default_rng(pop_seed)

    # Initial condition: Binom(N - i0, 1 - V / (N - i0))
    p = 1.0 - (V / (N - i0))
    U_pop = pop_rng.random((num_sims, K))

    S_0 = binom.ppf(U_pop, n=N - i0, p=p).reshape(num_sims, K)
    I_0 = np.tile(np.array(i0, dtype=float), num_sims).reshape(num_sims, K)

    # Run core simulation for num_sims paths
    S_traj, I_traj = simulate_tsir_trajectories(S_0, I_0, N, beta, W, T, dyn_seed, crn=False)
    
    # Evaluate obj_fn on each trajectory: (t, sim_n, loc)
    result = np.array([
        obj_fn(np.hstack([S_traj[:, n, :], I_traj[:, n, :]]))
        for n in range(num_sims)
        ])
    
    return result


def tsir_multipop_v(N, i0, V, beta, W, T, obj_fn, pop_seed, dyn_seed):
    """
    Simulate paths of a multi-population tSIR model where initial susceptibles are 
    perturbed per-location according to the weak-derivative.
    
    Parameters:
    --------
    N: np.ndarray[int] of length K
        Population vector indexed by location.
    i0: np.ndarray[int] of length K
        Number of initially infected individuals indexed by location.
    V: np.ndarray[float] of length K
        Number of vaccinated individuals indexed by location.
    beta: np.ndarray[float] of length K
        Contact rate parameters indexed by location.
    W: np.ndarray[float, float] of size K x K
        Weight/mixing matrix, doubly indexed by location.
    T: int
        Number of periods to run the simulation for.
    obj_fn: Callable[[np.ndarray], float] 
        Sample path statistic operating on arrays of size (T+1) x 2K,
        where rows represent timestep and columns are [(S_1, ... , S_K), (I_1, ..., I_K)].
    pop_seed: int
        Seed value for RNG used to generate initial state.
    dyn_seed: int
        Seed value for RNG used to generate dynamics.
        
    Returns:
    -------
    np.ndarray[float]
        - The weak-derivative gradient estimate, of length K.
    """
    K = len(N)
    pop_rng = np.random.default_rng(pop_seed)

    p = 1.0 - (V / (N - i0))
    U_pop = pop_rng.random(K)

    # Build 2K initial states: rows 0..K-1 are S_plus, rows K..2K-1 are S_minus
    S_plus_0 = np.zeros((K, K))
    S_minus_0 = np.zeros((K, K))

    for k in range(K):
        to_perturb = np.zeros(K)
        to_perturb[k] = 1.0
        
        this_S_plus = binom.ppf(U_pop, n=N - i0 - to_perturb, p=p)
        this_S_minus = this_S_plus.copy()
        this_S_minus[k] += 1.0

        S_plus_0[k, :] = this_S_plus
        S_minus_0[k, :] = this_S_minus

    # Stack all 2K initial conditions into a single (2K, K) matrix
    S_0 = np.vstack([S_plus_0, S_minus_0])
    I_0 = np.tile(i0, (2 * K, 1)).astype(float)

    # Single vectorized execution across ALL 2K trajectories simultaneously
    S_traj, I_traj = simulate_tsir_trajectories(S_0, I_0, N, beta, W, T, dyn_seed)

    # Evaluate objective pairs and compute gradient vector
    grad = np.zeros(K)
    for k in range(K):
        traj_plus = np.hstack([S_traj[:, k, :], I_traj[:, k, :]])
        traj_minus = np.hstack([S_traj[:, K + k, :], I_traj[:, K + k, :]])

        grad[k] = obj_fn(traj_plus) - obj_fn(traj_minus)

    return grad

def step_wd_beta(U_dyn, S, I, N, beta, W):
    """
    Compute a perturbed step of the multipopulation tSIR model
    for the weak derivative with respect to beta.
    
    Parameters:
    -----------
    U_dyn : np.ndarray of shape (1, K)
        Uniform random variables U ~ Unif(0, 1) for inverse CDF sampling.
    S, I  : np.ndarray of shape (1, K)
        Current susceptible and infectious counts for K locations.
    N, beta : np.ndarray of shape (K,) or (1, K)
        Total population sizes and transmission parameters.
    W     : np.ndarray of shape (K, K)
        Spatial transmission matrix where W[j, k] is contact weight from j to k.

    Returns:
    --------
    next_S_plus, next_I_plus, next_S_minus, next_I_minus : np.ndarray of shape (K, K)
        Rows indicate which location k has been perturbed; columns are locations.
    scale : np.ndarray of shape (K,)
        Scale factor for weak derivatives per location.
    """
    S = np.asarray(S, dtype=float).reshape(1, -1)
    I = np.asarray(I, dtype=float).reshape(1, -1)
    N = np.asarray(N, dtype=float).reshape(1, -1)
    beta = np.asarray(beta, dtype=float).reshape(1, -1)
    
    K = S.shape[1]
    
    # -----------------------------------------------------------------
    # ORIGINAL IDEA 1: Compute spatial contact sum
    # (I @ W)[0, k] = sum_j (I[0, j] * W[j, k])
    # -----------------------------------------------------------------
    effective_I = I @ W  # shape (1, K)
    
    # -----------------------------------------------------------------
    # ORIGINAL IDEA 2: Compute rate lambda_{k, t+1} for each path & location
    # -----------------------------------------------------------------
    lambda_val = (S * beta / N) * effective_I  # shape (1, K)
    
    # -----------------------------------------------------------------
    # ORIGINAL IDEA 3: Sample Y_{t+1} using Inverse Transform Sampling (PPF)
    # -----------------------------------------------------------------
    
    # --- Case A: Poisson sample for I == 0 ---
    y_pois = poisson.ppf(U_dyn, lambda_val)
    
    # --- Case B: Negative Binomial sample for I > 0 ---
    # ORIGINAL IDEA: Map (lambda, I) to SciPy's nbinom parameters (n, p)
    # UPDATED LOGIC:
    # SciPy uses p = I / (I + lambda).
    # 1) Perturbed pair uses n = I + 1 to yield mean = (I + 1)*lambda / I
    # 2) Unperturbed locations use nominal n = I to keep standard distribution
    safe_denom = np.maximum(I + lambda_val, 1e-12)
    p = np.where(I > 0, I / safe_denom, 1.0)
    
    # Perturbed draw (n = I + 1): Used for location k being perturbed
    safe_n_perturbed = np.maximum(I + 1.0, 1.0)
    y_nbinom_perturbed = nbinom.ppf(U_dyn, safe_n_perturbed, p)
    base_Y_perturbed = np.where(I > 0, y_nbinom_perturbed, y_pois)
    
    # Nominal draw (n = I): Used for unperturbed locations j != k
    safe_n_nom = np.maximum(I, 1e-12)
    y_nbinom_nom = nbinom.ppf(U_dyn, safe_n_nom, p)
    nominal_Y = np.where(I > 0, y_nbinom_nom, y_pois)
    
    # -----------------------------------------------------------------
    # UPDATED LOGIC: Build (K x K) states for next_S_plus/minus, next_I_plus/minus
    # ORIGINAL IDEA needed rows indicating which location k was perturbed.
    # -----------------------------------------------------------------
    next_S_plus = np.zeros((K, K))
    next_I_plus = np.zeros((K, K))
    next_S_minus = np.zeros((K, K))
    next_I_minus = np.zeros((K, K))
    
    nom_I_next = np.minimum(S[0], nominal_Y[0])
    
    for k in range(K):
        # Start all locations with nominal unperturbed step
        I_plus_k = nom_I_next.copy()
        I_minus_k = nom_I_next.copy()
        
        # Override ONLY location k: Y_k^+ = 1 + Y_base, Y_k^- = Y_base
        I_plus_k[k] = min(S[0, k], base_Y_perturbed[0, k] + 1.0)
        I_minus_k[k] = min(S[0, k], base_Y_perturbed[0, k])
        
        next_I_plus[k, :] = I_plus_k
        next_S_plus[k, :] = S[0] - I_plus_k
        
        next_I_minus[k, :] = I_minus_k
        next_S_minus[k, :] = S[0] - I_minus_k
        
    # -----------------------------------------------------------------
    # ORIGINAL FORMULA: scale = S_k * (sum_j W_{jk} I_j) / N_k
    # Implemented via vector operation S * (I @ W) / N
    # -----------------------------------------------------------------
    scale = ((S[0] * effective_I[0]) / N[0]).flatten()
    
    return next_S_plus, next_I_plus, next_S_minus, next_I_minus, scale


def tsir_multipop_beta(N, i0, V, beta, W, T, obj_fn, pop_seed, dyn_seed):
    """
    Simulate paths of a multi-population tSIR model where initial susceptibles are 
    perturbed per-location according to the weak-derivative for beta
    
    Parameters:
    --------
    N: np.ndarray[int] of length K
        Population vector indexed by location.
    i0: np.ndarray[int] of length K
        Number of initially infected individuals indexed by location.
    V: np.ndarray[float] of length K
        Number of vaccinated individuals indexed by location.
    beta: np.ndarray[float] of length K
        Contact rate parameters indexed by location.
    W: np.ndarray[float, float] of size K x K
        Weight/mixing matrix, doubly indexed by location.
    T: int
        Number of periods to run the simulation for.
    obj_fn: Callable[[np.ndarray], float] 
        Sample path statistic operating on arrays of size (T+1) x 2K,
        where rows represent timestep and columns are [(S_1, ... , S_K), (I_1, ..., I_K)].
    pop_seed: int
        Seed value for RNG used to generate initial state.
    dyn_seed: int
        Seed value for RNG used to generate dynamics.
        
    Returns:
    -------
    np.ndarray[float]
        - The weak-derivative gradient estimate, of length K.
    """
    K = len(N)
    
    # -----------------------------------------------------------------
    # ORIGINAL IDEA 1: Simulate a nominal path
    # -----------------------------------------------------------------
    nominal_traj = tsir_multipop(
        N, i0, V, beta, W, T, lambda x: x, 1, pop_seed, dyn_seed)[0]

    gradient = np.zeros(K)
    
    # -----------------------------------------------------------------
    # ORIGINAL IDEA 2: For t in range(0, T), run 2K perturbed simulations
    # ORIGINAL COMMENT: "# make sure to make the seed different for each t in T"
    # UPDATED LOGIC: Pre-generate independent seed sequence per timestep t
    # -----------------------------------------------------------------
    master_rng = np.random.default_rng(dyn_seed)
    t_seeds = master_rng.integers(0, 2**31 - 1, size=T)

    for t in range(T):
        # Extract nominal state at time t
        S_t_nom = nominal_traj[t, :K].reshape(1, K)
        I_t_nom = nominal_traj[t, K:].reshape(1, K)

        t_rng = np.random.default_rng(t_seeds[t])
        U_dyn = t_rng.random((1, K))

        # Perform single perturbed step at time t
        next_S_plus, next_I_plus, next_S_minus, next_I_minus, scale = step_wd_beta(
            U_dyn, S_t_nom, I_t_nom, N, beta, W)

        # Stack 2K paths: rows 0..K-1 are plus, rows K..2K-1 are minus
        next_S_stacked = np.vstack([next_S_plus, next_S_minus])
        next_I_stacked = np.vstack([next_I_plus, next_I_minus])

        rem_steps = T - t - 1
        rem_seed = int(t_rng.integers(0, 2**31 - 1))

        # -------------------------------------------------------------
        # ORIGINAL PLACEHOLDER: S_full, I_full
        # ORIGINAL COMMENT: "# paste the trajectories together [nominal, pert, remainder of path]"
        # UPDATED LOGIC: Fully constructed full-horizon matrices (T+1, 2K, K)
        # -------------------------------------------------------------
        if rem_steps > 0:
            # Simulate forward remainder up to time T
            S_rem, I_rem = simulate_tsir_trajectories(
                next_S_stacked, next_I_stacked, N, beta, W, rem_steps, rem_seed, crn=True)
            
            S_full = np.zeros((T + 1, 2 * K, K))
            I_full = np.zeros((T + 1, 2 * K, K))
            
            # Segment 1: Nominal history [0..t] replicated across all 2K paths
            S_full[:t + 1, :, :] = nominal_traj[:t + 1, :K][:, np.newaxis, :]
            I_full[:t + 1, :, :] = nominal_traj[:t + 1, K:][:, np.newaxis, :]
            
            # Segment 2 & 3: Perturbed step at t+1 and remainder rollouts [t+1..T]
            S_full[t + 1:, :, :] = S_rem
            I_full[t + 1:, :, :] = I_rem
        else:
            # Boundary condition when t = T-1 (no forward steps remaining)
            S_full = np.zeros((T + 1, 2 * K, K))
            I_full = np.zeros((T + 1, 2 * K, K))
            
            S_full[:T, :, :] = nominal_traj[:T, :K][:, np.newaxis, :]
            I_full[:T, :, :] = nominal_traj[:T, K:][:, np.newaxis, :]
            
            S_full[T, :, :] = next_S_stacked
            I_full[T, :, :] = next_I_stacked

        # -------------------------------------------------------------
        # ORIGINAL IDEA:
        # "# evaluate the obj_fn on the paths"
        # "# compute the statistic on each path (for each K)"
        # -------------------------------------------------------------
        for k in range(K):
            traj_plus = np.hstack([S_full[:, k, :], I_full[:, k, :]])
            traj_minus = np.hstack([S_full[:, K + k, :], I_full[:, K + k, :]])
            
            result_plus = obj_fn(traj_plus)
            result_minus = obj_fn(traj_minus)
            
            # Accumulate: scale_k * (f(traj_k^+) - f(traj_k^-))
            gradient[k] += scale[k] * (result_plus - result_minus)

    return gradient
    
def tsir_multipop_fd_v(N, i0, V, beta, W, T, obj_fn, pop_seed, dyn_seed, eps=1, equal_eps=True):
    """
    Finite-difference gradient estimator for V using K+1 simultaneous simulations.

    Parameters:
    --------
    N, i0, V, beta, W, T, obj_fn, pop_seed, dyn_seed:
        Same signature as tsir_multipop / tsir_multipop_v.
    eps: float or np.ndarray[float]
        Perturbation step size(s). Can be a scalar or array of length K.
    equal_eps: bool (default=True)
        If True, assumes all coordinates use the scalar eps. If False, expects eps to 
        be an array of length K (or broadcasted).

    Returns:
    -------
    np.ndarray[float]
        Finite-difference gradient estimate of length K.
    """
    K = len(N)

    # 1. Parse and format epsilon vector
    if equal_eps:
        eps_vec = np.full(K, float(eps))
    else:
        eps_vec = np.asarray(eps, dtype=float)
        assert len(eps_vec) == K, f"eps array length ({len(eps_vec)}) must match K ({K})"

    pop_rng = np.random.default_rng(pop_seed)
    U_pop = pop_rng.random(K)

    # 2. Build initial state matrix for M = K + 1 paths:
    # Row 0: Nominal state (V)
    # Rows 1..K: Perturbed states (V + eps_k * e_k)
    S_0 = np.zeros((K + 1, K))
    
    # Nominal initial p
    p_nominal = 1.0 - (V / (N - i0))
    S_0[0, :] = binom.ppf(U_pop, n=N - i0, p=p_nominal)

    # Perturbed initial states using Common Random Numbers (U_pop)
    for k in range(K):
        this_V = V.copy().astype(float)
        this_V[k] += eps_vec[k]
        
        p_perturbed = 1.0 - (this_V / (N - i0))
        S_0[k + 1, :] = binom.ppf(U_pop, n=N - i0, p=p_perturbed)

    I_0 = np.tile(i0, (K + 1, 1)).astype(float)

    # 3. Simulate all K+1 paths simultaneously in a single engine call
    S_traj, I_traj = simulate_tsir_trajectories(S_0, I_0, N, beta, W, T, dyn_seed)

    # 4. Evaluate objective function on nominal and perturbed trajectories
    traj_0 = np.hstack([S_traj[:, 0, :], I_traj[:, 0, :]])
    f_0 = obj_fn(traj_0)

    grad_fd = np.zeros(K)
    for k in range(K):
        traj_k = np.hstack([S_traj[:, k + 1, :], I_traj[:, k + 1, :]])
        f_k = obj_fn(traj_k)

        # Forward finite-difference ratio
        grad_fd[k] = (f_k - f_0) / eps_vec[k]

    return grad_fd

def tsir_multipop_fd_beta(N, i0, V, beta, W, T, obj_fn, pop_seed, dyn_seed, eps=0.1, equal_eps=True):
    """
    Compute finite difference gradient estimate for beta in multi-population tSIR.
    Simulates K+1 paths in parallel (1 nominal + K per-location perturbations).

    Parameters:
    -----------
    N, i0, V, beta : np.ndarray of shape (K,)
        Population sizes, initial infecteds, vaccinated counts, and transmission rates.
    W : np.ndarray of shape (K, K)
        Spatial mixing matrix.
    T : int
        Number of time steps to simulate.
    obj_fn : Callable[[np.ndarray], float]
        Statistic operating on (T+1, 2K) path array [S_1..S_K, I_1..I_K].
    pop_seed, dyn_seed : int
        RNG seeds for population initialization and time step dynamics.
    eps : float or np.ndarray
        Perturbation step size parameter.
    equal_eps : bool
        If True, uses absolute step size (delta_k = eps).
        If False, uses relative step size (delta_k = eps * beta_k).

    Returns:
    --------
    grad_fd : np.ndarray of shape (K,)
        The finite difference gradient estimate for each location k.
    """
    beta = np.asarray(beta, dtype=float).flatten()
    K = len(N)

    # 1. Determine perturbation step sizes delta_k for each location k
    if equal_eps:
        delta = np.full(K, eps, dtype=float) if np.isscalar(eps) else np.asarray(eps, dtype=float)
    else:
        delta = beta * eps if np.isscalar(eps) else beta * np.asarray(eps, dtype=float)

    # 2. Build (K + 1, K) beta matrix
    # Row 0: nominal beta vector
    # Rows 1..K: perturbed beta vector (location k-1 boosted by delta[k-1])
    beta_nominal = beta.reshape(1, K)
    beta_perturbed = np.tile(beta_nominal, (K, 1)) + np.diag(delta)
    beta_matrix = np.vstack([beta_nominal, beta_perturbed])  # Shape: (K + 1, K)

    # 3. Generate initial state (S_0, I_0) shared identically across all K+1 paths
    pop_rng = np.random.default_rng(pop_seed)
    p = 1.0 - (V / (N - i0))
    U_pop = pop_rng.random((1, K))

    S_0_single = binom.ppf(U_pop, n=N - i0, p=p).reshape(1, K)
    I_0_single = np.array(i0, dtype=float).reshape(1, K)

    # Replicate shared initial state for all M = K + 1 parallel simulations
    S_0 = np.tile(S_0_single, (K + 1, 1))  # Shape: (K + 1, K)
    I_0 = np.tile(I_0_single, (K + 1, 1))  # Shape: (K + 1, K)

    # 4. Simulate all K+1 trajectories simultaneously using Common Random Numbers
    S_traj, I_traj = simulate_tsir_trajectories(S_0, I_0, N, beta_matrix, W, T, dyn_seed, crn=True)

    # 5. Evaluate objective function across all K+1 simulated paths
    results = np.zeros(K + 1)
    for m in range(K + 1):
        traj_m = np.hstack([S_traj[:, m, :], I_traj[:, m, :]])
        results[m] = obj_fn(traj_m)

    f_nom = results[0]
    f_pert = results[1:]

    # 6. Compute forward finite difference vector
    grad_fd = (f_pert - f_nom) / delta

    return grad_fd

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

def analyze_fsp_results(fsp_dists, N, print_table=True, max_cols_to_print=4):
    """
    Analyzes FSP output distributions to compute exact expectations E[I_{k,t}],
    retained/discarded probability masses, and theoretical error upper bounds
    for any arbitrary number of locations K and time steps T.

    Parameters:
    -----------
    fsp_dists : List[Dict[Tuple, float]]
        List of probability distributions per time step [P_0, P_1, ..., P_T].
    N : np.ndarray or List[float]
        Population sizes per location of length K.
    print_table : bool, default=True
        If True, prints a formatted diagnostic summary table to stdout.
    max_cols_to_print : int, default=4
        Maximum number of per-location error columns to show before consolidating
        into a single max-error column across all locations.

    Returns:
    --------
    results : dict
        - 'expected_I'   : np.ndarray of shape (T+1, K)
        - 'retained_mass': np.ndarray of shape (T+1,)
        - 'discarded_mass': np.ndarray of shape (T+1,)
        - 'error_bounds' : np.ndarray of shape (T+1, K)
    """
    N = np.asarray(N)
    K = len(N)
    T_plus_1 = len(fsp_dists)

    fsp_expected_I = np.zeros((T_plus_1, K))
    retained_mass = np.zeros(T_plus_1)
    discarded_mass = np.zeros(T_plus_1)
    error_bounds = np.zeros((T_plus_1, K))

    # Process distributions across all time steps
    for t, P_t in enumerate(fsp_dists):
        retained_p = sum(P_t.values())
        eps_t = max(0.0, 1.0 - retained_p)  # Total discarded probability mass

        retained_mass[t] = retained_p
        discarded_mass[t] = eps_t
        error_bounds[t, :] = eps_t * N

        # Accumulate exact expectations E[I_k] = sum_{x} i_k * P_t(x)
        for state, prob in P_t.items():
            I_vec = state[K:]
            fsp_expected_I[t, :] += prob * np.array(I_vec)

    # Print formatted diagnostic table
    if print_table:
        # Dynamic table header formatting based on K
        if K <= max_cols_to_print:
            err_headers = [f"Max Error E[I_{k+1}]" for k in range(K)]
            col_spec = " | ".join([f"{h:<18}" for h in err_headers])
        else:
            col_spec = f"{'Max Error (All K)':<20}"

        header = f"{'t':<3} | {'Retained Mass':<15} | {'Discarded Mass (eps)':<22} | " + col_spec
        divider = "=" * len(header)

        print("\n" + divider)
        print(header)
        print(divider)

        for t in range(T_plus_1):
            base_str = f"{t:<3} | {retained_mass[t]:<15.10f} | {discarded_mass[t]:<22.6e} | "
            if K <= max_cols_to_print:
                err_str = " | ".join([f"{error_bounds[t, k]:<18.6e}" for k in range(K)])
            else:
                err_str = f"{np.max(error_bounds[t, :]):<20.6e}"
            print(base_str + err_str)

        print(divider)

    return {
        'expected_I': fsp_expected_I,
        'retained_mass': retained_mass,
        'discarded_mass': discarded_mass,
        'error_bounds': error_bounds,
    }

def compute_fsp_jacobians(
    N, i0, V, beta, W, T, 
    eps_V=1e-3, 
    eps_beta=1e-3, 
    tol=1e-12, 
    num_workers=4, 
    max_succ_per_state=50
):
    """
    Computes finite-difference Jacobians of cumulative expected infected populations 
    sum_{t=0}^T E[I_{k,t}] with respect to parameter vectors V and beta, along with 
    rigorous lower and upper error bound intervals derived from FSP truncation 
    probability mass loss aggregated over time.

    Parameters:
    -----------
    N, i0, V, beta, W, T : Model parameters.
    eps_V, eps_beta : float
        Finite difference perturbation step sizes for V and beta vectors.
    tol, num_workers, max_succ_per_state : FSP solver truncation/parallel execution settings.

    Returns:
    --------
    results : dict
        - 'J_V'            : (K, K) np.ndarray, Jacobian d (sum_t E[I_t]) / d V
        - 'J_V_interval'   : (K, K, 2) np.ndarray, [min_bound, max_bound] for each J_V entry
        - 'J_beta'         : (K, K) np.ndarray, Jacobian d (sum_t E[I_t]) / d beta
        - 'J_beta_interval': (K, K, 2) np.ndarray, [min_bound, max_bound] for each J_beta entry
        - 'cum_E_I_nom'    : (K,) np.ndarray, nominal cumulative expected infected counts
    """
    N = np.asarray(N, dtype=float)
    i0 = np.asarray(i0, dtype=float)
    V = np.asarray(V, dtype=float)
    beta = np.asarray(beta, dtype=float)
    
    K = len(N)

    # Helper function to compute cumulative expectation sum_{t=0}^T E[I_t] 
    # and total aggregated theoretical error bounds [a, b] across time horizon T
    def _get_cumulative_expectation_and_bounds(fsp_dists):
        cum_E_I = np.zeros(K)
        tot_eps_fsp = 0.0

        # Accumulate expectations and truncation loss over all time steps t = 0..T
        for P_t in fsp_dists:
            retained_p = sum(P_t.values())
            eps_t = max(0.0, 1.0 - retained_p)
            tot_eps_fsp += eps_t

            for state, prob in P_t.items():
                I_vec = state[K:]
                cum_E_I += prob * np.array(I_vec)

        # Theoretical bounds over cumulative time horizon:
        # a_k <= sum_{t=0}^T E[I_{k,t}] <= b_k
        a = cum_E_I
        b = cum_E_I + N * tot_eps_fsp
        return cum_E_I, a, b

    # ------------------------------------------------------------------
    # 1. NOMINAL RUN (Shared baseline for all parameters)
    # ------------------------------------------------------------------
    dists_nom = tsir_exact_fsp_optimized(
        N, i0, V, beta, W, T, tol=tol, num_workers=num_workers, max_succ_per_state=max_succ_per_state
    )
    E_nom, a_nom, b_nom = _get_cumulative_expectation_and_bounds(dists_nom)

    # Output Jacobian and interval matrices initialization
    J_V = np.zeros((K, K))
    J_V_interval = np.zeros((K, K, 2))
    
    J_beta = np.zeros((K, K))
    J_beta_interval = np.zeros((K, K, 2))

    # ------------------------------------------------------------------
    # 2. JACOBIAN WITH RESPECT TO V (d sum_t E[I_t] / d V_k)
    # ------------------------------------------------------------------
    V_max = N - i0  # Boundary constraint: 0 <= V_k <= N_k - i0_k

    for k in range(K):
        # Select forward or backward difference step based on domain bounds
        if V[k] + eps_V <= V_max[k]:
            h_k = eps_V          # Forward step
        elif V[k] - eps_V >= 0.0:
            h_k = -eps_V         # Backward step if upper bound exceeded
        else:
            h_k = min(eps_V, V_max[k] - V[k])

        V_pert = V.copy()
        V_pert[k] += h_k

        dists_pert = tsir_exact_fsp_optimized(
            N, i0, V_pert, beta, W, T, tol=tol, num_workers=num_workers, max_succ_per_state=max_succ_per_state
        )
        E_pert, a_pert, b_pert = _get_cumulative_expectation_and_bounds(dists_pert)

        # Point estimate of derivative
        J_V[:, k] = (E_pert - E_nom) / h_k

        # Error interval computation accounting for cumulative truncation loss
        if h_k > 0:
            bound_min = (a_pert - b_nom) / h_k
            bound_max = (b_pert - a_nom) / h_k
        else:
            bound_min = (b_pert - a_nom) / h_k
            bound_max = (a_pert - b_nom) / h_k

        J_V_interval[:, k, 0] = bound_min
        J_V_interval[:, k, 1] = bound_max

    # ------------------------------------------------------------------
    # 3. JACOBIAN WITH RESPECT TO BETA (d sum_t E[I_t] / d beta_k)
    # ------------------------------------------------------------------
    for k in range(K):
        h_k = eps_beta if beta[k] + eps_beta >= 0.0 else -eps_beta

        beta_pert = beta.copy()
        beta_pert[k] += h_k

        dists_pert = tsir_exact_fsp_optimized(
            N, i0, V, beta_pert, W, T, tol=tol, num_workers=num_workers, max_succ_per_state=max_succ_per_state
        )
        E_pert, a_pert, b_pert = _get_cumulative_expectation_and_bounds(dists_pert)

        # Point estimate of derivative
        J_beta[:, k] = (E_pert - E_nom) / h_k

        # Error interval computation
        if h_k > 0:
            bound_min = (a_pert - b_nom) / h_k
            bound_max = (b_pert - a_nom) / h_k
        else:
            bound_min = (b_pert - a_nom) / h_k
            bound_max = (a_pert - b_nom) / h_k

        J_beta_interval[:, k, 0] = bound_min
        J_beta_interval[:, k, 1] = bound_max

    return {
        'J_V': J_V,
        'J_V_interval': J_V_interval,
        'J_beta': J_beta,
        'J_beta_interval': J_beta_interval,
        'cum_E_I_nom': E_nom
    }