# Description of data files

## `big_run_combined.csv`

This file contains results for the comparison of:
- Finite difference estimators for $J_v$ and $J_\beta$
- Variance-reduced estimator for $J_v$
- Variance-reduced estimator for $J_\beta$

Over the grid of parameters:
- $v$ ranging from 0.01 to 0.99
- $\beta$ ranging from 2 to 4

Columns description:
- `total_size_mean`: The sample average number of infections.
- `total_size_sd`: Sample standard deviation of the number of infections.
- `v_mean`: The sample average variance-reduced result for $\widehat J_v$.
- `v_sd`: The sample standard deviation of $\widehat J_v$.
- `beta_mean`: The sample average variance-reduced result for $\widehat J_\beta$.
- `beta_sd`: The sample standard deviation of $\widehat J_\beta$.
- `fd_{v,beta}_mean`: The sample average finite-difference estimate for $J_v$ or $J_\beta$.
- `fd_{v,beta}_sd`: The sample average finite-difference estimate for $J_v$ or $J_\beta$.
- `beta`: The value of $\beta$ used in the estimate (contact rate parameter).
- `v`: The value of $v$ used in the estimate (proportion immune).
- `N_samples, N, T, i0`: All constants. Respectively, 
    - Number of replicates of the estimators
    - Population size
    - Time horizon
    - Number initially infected.

## `semi_analytic_result.csv`

This file contains the results for the derivatives calculated via the 
semi-analytic final size method.

It is computed over the same grid of parameters as `big_run_combined.csv`.

Column description:
- `beta`: Contact rate parameter.
- `v`: Proportion immune.
- `N`: Population size.
- `size`: Final size (total number infected).
- `dv`: Derivative of the final size with respect to $v$.
- `db`: Derivative of the final size with respect to $\beta$.

## `no_vrt_table.csv`

This file contains the results for the derivative estimators computed without
variance reduction. This means that the weak-derivative estimators for $J_v$ are 
calculated from samples where $Z^+$ and $Z^-$ are independently simulated. 
We use the version $J_\beta$ where the summation is not simplified and the mean
is not used as a control variate.

We do not compute over the whole grid of parameters, just for the selected
values in the table.

- `v`: Proportion immune.
- `beta`: Contact rate parameter.
- `N`: Population size.
- `i0`: Number initially infected.
- `N_samples`: Number of replicates for the estimators.
- `T`: Time horizon.
- `v_mean`: The sample average variance-reduced result for $\widehat J_v$.
- `v_sd`: The sample standard deviation of $\widehat J_v$.
- `v_se`: The standard error of $\widehat J_v$.
- `beta_mean`: The sample average variance-reduced result for $\widehat J_\beta$.
- `beta_sd`: The sample standard deviation of $\widehat J_\beta$.
- `beta_se`: The standard error of $\widehat J_\beta$.

## `parametric_uncertainty/parametric_uncertainty_result.csv`

This file contains the results for the sensitivity analysis under parametric
uncertainty, calculated on the same grid of parameters as `big_run_combined.csv`.
In this case, $\beta$ and $v$ are sampled from a distribution rather than
being fixed at a certain value. The sensitivity is averaged across the randomness
in the transmission dynamics as well as the randomness in the parameters.
Strictly speaking, this is no longer a true derivative but rather a "sensitivity."

Constants:
- `beta_conf`: Confidence parameter for the distribution of $\hat \beta$. In this case,
it follows a $\text{Gamma}(a, \beta/a)$ distribution, where the parameters are shape and rate,
and $a$ is the value of `beta_conf`.
- `v_conf`: Confidence parameter for the distribution of $\hat \beta$. In this case,
it follows a $\text{Beta}(Cv, C(1-v))$ distribution. $C$ is the value of `v_conf`.
- `N`: Population size.
- `i0`: Number initially infected.
- `T`: Time horizon.
- `N_samples`: Number of replications of the estimators.
Variables:
- `v`: Proportion immune.
- `beta`: Contact rate parameter.
Outputs:
- `beta_grad`: Sensitivity of total infected with respect to $\beta$.
- `beta_grad_se`: Standard error of the estimate of `beta_grad`.
- `v_grad`: Sensitivity of total infected with respect to $v$.
- `v_grad_se`: Standard error of the estimate of `v_grad`.
- `total_size_mean`: Average number of total infections across the replicates.
- `total_size_se`: Standard error of the average number of total infections across the replicates.

## `{v,beta}_star_vs_cost.csv`

These files contain the solutions to the optimization problems. The parameter in the title denotes the decision variable (the variable we are solving for), and the other parameter is fixed and known. 

Column descriptions:
- `c_{v, beta}` - Cost scaling parameter for objective function.
- `{v, beta}_star` - Solution found by stochastic approximation routine.
- `{v, beta}_star_sd` - Estimate of error in the solution, calculated via batch means.
- `{v, beta}` - Value of the fixed parameter.
- `N` - Population size.
- `T` - Time horizon.

## `parametric_uncertainty/{v,beta}_star_vs_cost_{v,beta}_conf_10.csv`

These files contain the solutions to the optimization problems under parameter uncertainty. The first `v` or `beta` denotes the decision variable. The next variable is the uncertain parameter, and `conf_10` denotes that the confidence parameter (controlling the variance of the distribution of the uncertainty) is set to 10.

Column descriptions:
- `c_{v, beta}` - Cost scaling parameter for objective function.
- `{v, beta}_star` - Solution found by stochastic approximation routine.
- `{v, beta}_star_sd` - Estimate of error in the solution, calculated via batch means.
- `{v, beta}` - Mean of the fixed parameter.
- `N` - Population size.
- `T` - Time horizon.

