# Description of data files

## big_run_combined.csv

This file contains results for the comparison of:
- Finite difference estimators for $J_v$ and $J_\beta$
- Variance-reduced estimator for $J_v$
- Variance-reduced estimator for $J_\beta$

Over the grid of parameters:
- $v$ ranging from 0.01 to 0.99
- $beta$ ranging from 2 to 4

Columns description:
- `total_size_mean`: The sample average number of infections.
- `total_size_sd`: Sample standard deviation of the number of infections.
- `v_mean`: The sample average variance-reduced result for $\widehat J_v$.
- `v_sd`: The sample standard deviation of $\widehat J_v$.
- `beta_mean`: The sample average variance-reduced result for $\widehat J_\beta$.
- `beta_sd`: The sample standard deviation of $\widehat J_\beta$.
- `fd_{v,beta}_mean`: The sample average finite-difference estimate for $J_v$ or $J_\beta$.
- `fd_{v,beta}_sd`: The sample average finite-difference estimate for $J_v$ or $J_\beta$.
- `beta`: The value of $\beta$ used in the estimate.
- `v`: The value of $v$ used in the estimate.
- `N_samples, N, T, i0`: All constants. Respectively, 
    - Number of replicates of the estimators
    - Population size
    - Time horizon
    - Number initially infected.