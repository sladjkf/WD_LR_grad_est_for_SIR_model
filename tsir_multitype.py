#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 12:54:00 2026

@author: nicholasw


Simulators and gradient estimators for a multipopulation tSIR model.
"""
import numpy as np
from scipy.stats import nbinom, binom, poisson

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

def step_wd(U_dyn, S, I, N, beta, W):
    """
    Compute a single step of the multipopulation TSIR model.
    Automatically broadcasts across M simulations and K locations.

    Parameters:
    -----------
    U_dyn : np.ndarray of shape (M, K)
        Uniform random variables U ~ Unif(0, 1) for inverse CDF sampling.
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

# =====================================================================
# 1. Core Propagation Engine (Reused by nominal & gradient callers)
# =====================================================================
def simulate_tsir_trajectories(S_0, I_0, N, beta, W, T, dyn_seed):
    """
    Simulates M paths simultaneously over T steps given initial state matrices.
    
    Parameters:
    -----------
    S_0, I_0 : np.ndarray of shape (M, K)
        Initial susceptible and infected counts for M parallel paths.
    N, beta  : np.ndarray of shape (K,)
    W        : np.ndarray of shape (K, K)
    T        : int
    dyn_seed : int
    
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
        # Broadcast 1 set of random numbers (1, K) across all M paths for CRN synchronization
        U_dyn = dyn_rng.random((1, K))
        S_curr, I_curr = step_wd(U_dyn, S_curr, I_curr, N, beta, W)

        S_traj[t + 1] = S_curr
        I_traj[t + 1] = I_curr

    return S_traj, I_traj


# =====================================================================
# 2. Nominal Simulator (M = 1 path)
# =====================================================================
def tsir_multipop(N, i0, V, beta, W, T, obj_fn, pop_seed, dyn_seed):
    K = len(N)
    pop_rng = np.random.default_rng(pop_seed)

    # Initial condition: Binom(N - i0, 1 - V / (N - i0))
    p = 1.0 - (V / (N - i0))
    U_pop = pop_rng.random(K)

    S_0 = binom.ppf(U_pop, n=N - i0, p=p).reshape(1, K)
    I_0 = np.array(i0, dtype=float).reshape(1, K)

    # Run core simulation for 1 path
    S_traj, I_traj = simulate_tsir_trajectories(S_0, I_0, N, beta, W, T, dyn_seed)

    # Reformat shape (T+1, 1, K) -> (T+1, 2K)
    traj = np.hstack([S_traj[:, 0, :], I_traj[:, 0, :]])
    return obj_fn(traj)


# =====================================================================
# 3. Weak Derivative Estimator (M = 2K paths in 1 single pass)
# =====================================================================
def tsir_multipop_v(N, i0, V, beta, W, T, obj_fn, pop_seed, dyn_seed):
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

    # Single vectorized execution across ALL 2K trajectories simultaneously!
    S_traj, I_traj = simulate_tsir_trajectories(S_0, I_0, N, beta, W, T, dyn_seed)

    # Evaluate objective pairs and compute gradient vector
    grad = np.zeros(K)
    for k in range(K):
        traj_plus = np.hstack([S_traj[:, k, :], I_traj[:, k, :]])
        traj_minus = np.hstack([S_traj[:, K + k, :], I_traj[:, K + k, :]])

        grad[k] = obj_fn(traj_plus) - obj_fn(traj_minus)

    return grad

def tsir_multipop_fd_v(N, i0, V, beta, W, T, obj_fn, pop_seed, dyn_seed, eps=1e-4, equal_eps=True):
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