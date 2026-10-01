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
from tsir_multitype import *

# =====================================================================
# 3. BENCHMARKING & PRINTING PREDICTED ERROR BOUNDS
# =====================================================================
if __name__ == "__main__":
    # Define model hyper-parameters for a 2-location system
    N = np.array([10, 10, 5])
    i0 = np.array([1, 0, 0])
    V = np.array([1.0, 1.0, 1.0])
    beta = np.array([1.0, 1.0, 1.0])
    W = np.array([[1.0, 0.25, 0.25], 
                  [0.25, 1.0, 0.25],
                  [0.25, 0.25, 1.0]
                  ])
    # N = np.array([10, 10])
    # i0 = np.array([1, 0])
    # V = np.array([1, 1])
    # beta = np.array([4, 8])
    # W = np.array([
    #     [1.0, 0.25],
    #     [0.25, 1.0]
    #     ])
    T = 5
    K = len(N)
    num_sims = 10_000

    print("--- 1. Running Optimized FSP Solver ---")
    fsp_dists = tsir_exact_fsp_optimized(
        N, i0, V, beta, W, T, tol=1e-12, num_workers=10, max_succ_per_state=200
    )
    
    results = analyze_fsp_results(fsp_dists, N)
    print("\n--- 2. Running Monte Carlo Simulation (M = 10,000) ---")
    # Define objective function to extract infected trajectory I_t (columns K to 2K)
    extract_I_fn = lambda traj: traj[:, K:]

    pop_seed = 42
    dyn_seed = 123

    # mc_I_sims shape: (num_sims, T + 1, K)
    mc_I_sims = tsir_multipop(
        N, i0, V, beta, W, T, 
        obj_fn=extract_I_fn, 
        num_sims=num_sims, 
        pop_seed=pop_seed, 
        dyn_seed=dyn_seed
    )
    print(mc_I_sims)

    # Compute Monte Carlo empirical mean and 95% CI across simulation draws (axis=0)
    mc_mean = np.mean(mc_I_sims, axis=0)        # Shape: (T + 1, K)
    mc_std = np.std(mc_I_sims, axis=0, ddof=1)   # Shape: (T + 1, K)
    mc_se = mc_std / np.sqrt(num_sims)           # Shape: (T + 1, K)
    ci_95 = 1.96 * mc_se                         # Shape: (T + 1, K)
    
    analyze_fsp_results(fsp_dists, N)
#%%
    # ---------------------------------------------------------
    # 4. Plotting Exact FSP Path vs. Monte Carlo
    # ---------------------------------------------------------
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    def plot_fsp_vs_mc(results, mc_mean, ci_95):
        """
        Plots exact FSP trajectories from the results dictionary against 
        Monte Carlo empirical means and 95% confidence intervals.
        """
        # Extract expected trajectory matrix of shape (T+1, K) from results dict
        fsp_expected_I = results['expected_I']
        T_plus_1, K = fsp_expected_I.shape
        time_steps = np.arange(T_plus_1)
    
        fig, ax = plt.subplots(figsize=(10, 6))
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
    
    # Usage:
    plot_fsp_vs_mc(results, mc_mean, ci_95)