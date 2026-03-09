"""
tsir_wd.py
Implements the discrete-time stochastic SIR model.
- Weak derivative gradient estimator for total inf. wrt. v (prop. vacc)
  Using common random numbers for variance reduction.
- Likelihood-ratio gradient estimator for total inf. wrt. beta (contact rate)
  Using control variates for variance reduction.

"""

import numpy as np
import scipy.stats as stats
import multiprocess as mp


def tSIR_WD_CRN(N, v, i0, beta, T, pop_seed, dyn_seed):
    """
    Simulate a path of the SIR model.

    Parameters
    ----------
    N : int
        Total population size.
    v : float
        Immunization rate. Between 0 and 1.
    i0 : int
        Number of initially infected individuals.
    beta : float
        Contact rate. Positive number.
    T : int
        Time horizon (number of iterations to run the sim for)
    pop_seed : int
        Seed value for the generation of the number initially immune individuals.
    dyn_seed : int
        Seed value for the generation of the SIR model's dynamics.

    Returns
    -------
    orig_traj : np.array[int] of shape (T+1, 3)
        The unperturbed sample path.
    plus_traj : np.array[int] of shape (T+1, 3)
        The sample path using the \mu^+ distribution for number of immune
        individuals.
    minus_traj : np.array[int] of shape (T+1, 3)
        The sample path using the \mu^+ distribution for number of immune
        individuals.

    """
    pop_rng = np.random.default_rng(pop_seed)
    dyn_rng = np.random.default_rng(dyn_seed)

    pop_U = pop_rng.random()
    N_minus_i0 = N - i0
    assert N_minus_i0 > 0

    V = stats.binom.ppf(q=pop_U, n=N_minus_i0, p=v)
    V_minus = stats.binom.ppf(q=pop_U, n=N_minus_i0 - 1, p=v)
    V_plus = 1 + V_minus 
    
    orig_S = N - V
    plus_S = N - V_plus
    minus_S = N - V_minus

    traj_shape = (T + 1, 3)
    orig_traj = np.empty(traj_shape, dtype=int)
    plus_traj = np.empty(traj_shape, dtype=int)
    minus_traj = np.empty(traj_shape, dtype=int)

    orig_traj[0] = [orig_S, i0, 0]
    plus_traj[0] = [plus_S, i0, 0]
    minus_traj[0] = [minus_S, i0, 0]

    for i in range(T):
        orig_next, plus_next, minus_next = step(
            orig_traj[i], plus_traj[i], minus_traj[i], 
            beta=beta, N=N, rng=dyn_rng
        )
        
        orig_traj[i + 1] = orig_next
        plus_traj[i + 1] = plus_next
        minus_traj[i + 1] = minus_next

    return orig_traj, plus_traj, minus_traj

def draw_samples(N, i0, v, beta, T, N_samples, scrambler_seed, max_scrambler = 1e8, calc_LR = True):
    """
    Sample several simulation paths of the tSIR model.

    Parameters
    ----------
    N : int
        Total population size.
    v : float
        Immunization rate. Between 0 and 1.
    i0 : int
        Number of initially infected individuals.
    beta : float
        Contact rate. Positive number.
    T : int
        Time horizon (number of iterations to run the sim for)
    N_samples : int
        Number of sample paths to draw.
    scrambler_seed : int
        Random seed to use for the 'scrambler' 
        (generates random seeds for each simulation path.)
    max_scrambler : int, optional
        Maximum seed value the scrambler can generate. The default is 1e8.
    calc_LR : bool, optional
        Whether to calculate the likelihood-ratios of the transition kernel
        with respect to the contact rate. The default is True.

    Returns
    -------
    orig_samples : np.array[int] of length N_samples
        The total infected along the unperturbed sample paths.
    plus_samples : np.array[int] of length N_samples
        The total infected along the sample paths using the \mu^+ 
        distribution for number initially immune.
    minus_samples : np.array[int] of length N_samples
        The total infected along the sample paths using the \mu^- 
        distribution for number initially immune.
    trajectories : np.array[int] of shape (T+1, N_samples)
        The number infected at each timestep, for each sample path.
        Rows are timestep indices and columns are sample indices.
    score_samples : np.array[int] of shape (T, N_samples)
        The score function of the transition kernel P(X_{j+1} | X_j)
        with respect to the contact rate beta, evaluated at the
        the sampled states.
    """
    orig_samples = np.empty(N_samples, dtype=int)
    plus_samples = np.empty(N_samples, dtype=int)
    minus_samples = np.empty(N_samples, dtype=int)

    scrambler = np.random.default_rng(seed=scrambler_seed)
    
    if calc_LR:
        trajectories = np.empty((T+1, N_samples), dtype=int)
        score_samples = np.empty((T, N_samples), dtype=float)
    else:
        trajectories = None
        score_samples = None
    
    for i in range(N_samples):
        seed1 = scrambler.integers(0, max_scrambler)
        seed2 = scrambler.integers(0, max_scrambler)

        orig, plus, minus = tSIR_WD_CRN(
            N=N, v=v, i0=i0, beta=beta, T=T, pop_seed=seed1, dyn_seed=seed2
        )
        
        orig_samples[i] = np.sum(orig, axis=0)[1]
        plus_samples[i] = np.sum(plus, axis=0)[1]
        minus_samples[i] = np.sum(minus, axis=0)[1]

        if calc_LR:
            trajectories[:,i] = orig[:,1]
            
            for t in range(T):
                step_score = LR_beta_term(orig[t+1], orig[t], beta, N)
                score_samples[t, i] = step_score
        
    return orig_samples, plus_samples, minus_samples, trajectories, score_samples

def grad_wrt_v(plus_samples, minus_samples, N, i0):
    """
    Calculate the weak-derivative estimator with respect to v
    given the output of draw_samples.

    Parameters
    ----------
    plus_samples : np.array[float]
        Samples of the total infections, calculated under the distribution
        using \mu^+ for the number initially immune.
    minus_samples: np.array[float]
        Samples of the total infections, calculated under the distribution
        using \mu^- for the number initially immune.
    N : int
        Initial population size.
    i0 : int
        Number initially infected.

    Returns
    -------
    tuple[float, float]
        The 0th index is the sample mean of the gradient estimate.
        The 1st index is the sample standard deviation of the gradient estimate.
    """
    mean_grad = (N - i0) * (plus_samples - minus_samples)
    return np.mean(mean_grad), np.std(mean_grad, ddof=1)

def cvar_grad_wrt_v(orig_samples, plus_samples, minus_samples, N, i0, alpha = 0.95):
    assert len(plus_samples) == len(minus_samples) == len(orig_samples)
    n = len(plus_samples)
    m = int(np.floor(np.sqrt(n)))
    n_blocks = int(np.floor(n/m))
    quantiles = [np.quantile(orig_samples[(i*m):((i+1)*m)], alpha) for i in range(n_blocks)]
    grads = []
    for i in range(n_blocks):
        this_quantile = quantiles[((i+1) % n_blocks)]
        this_est = np.mean(np.maximum(plus_samples - this_quantile,0) - np.maximum(minus_samples - this_quantile,0))
        this_est = this_est*(1-alpha)**(-1)*(N-i0)
        grads.append(this_est)
    print(grads)
    return np.mean(grads), np.std(grads, ddof=1)

def grad_wrt_beta(trajectories, score_samples, T, N_samples):
    """
    Calculate the likelihood-ratio estimator with respect to beta
    giiven the output of draw_samples.
    
    Parameters
    ----------
    trajectories : np.array[int] of shape (T+1, N_samples)
        The number infected at each timestep, for each sample path.
        Rows are timestep indices and columns are sample indices.
    score_samples : np.array[int] of shape (T, N_samples)
        The score function of the transition kernel P(X_{j+1} | X_j)
        with respect to the contact rate beta, evaluated at the
        the sampled states.
    T : int
        The length of the time horizon.
    N_samples : int
        How many simulations were run.

    Returns
    -------
    tuple[float, float]
        The 0th index is the sample mean of the gradient estimate.
        The 1st index is the sample standard deviation of the gradient estimate.
    """
    baselines = np.mean(trajectories, axis=1) 
    gradients = np.zeros(N_samples)
    
    for i in range(N_samples):
        grad_i = 0.0
        
        # 2. Apply Causality + Baseline
        # We iterate backwards to easily sum the "Future Infections"
        future_cumulative_inf = 0.0
        
        for t in range(T-1, -1, -1):
            # The "Reward" for this step is only what happens AFTER this step
            # We subtract the baseline of that future step to reduce variance
            reward_at_step_t = trajectories[t, i]
            baseline_at_step_t = baselines[t]
            
            # Add to cumulative "Cost-to-Go"
            # Centering: (Reward - Average_Reward)
            centered_reward = reward_at_step_t - baseline_at_step_t
            future_cumulative_inf += centered_reward
            
            # Gradient accumulator: Score_t * (Sum of Future Centered Rewards)
            grad_i += score_samples[t, i] * future_cumulative_inf
            
        gradients[i] = grad_i

    return np.mean(gradients), np.std(gradients, ddof=1)
        
def y_pmf(y, prev_state, beta, N):
    r = prev_state[1]
    if y < 0.0:
        return 0.0
    mu = beta * prev_state[0] * prev_state[1] / N
    p = r / (r + mu)
    return stats.nbinom.pmf(y, n = r, p = p)

def LR_score(y, last_i, lambd, beta):
    if lambd == 0: return 0.0
    term = (y / lambd) - ((y + last_i) / (last_i + lambd))
    return term * (lambd / beta)

def LR_beta_term(next_state, prev_state, beta, N):
    next_s, next_i, next_r = next_state
    last_s, last_i, last_r = prev_state
    
    y = next_i
    lambd = beta * last_s * last_i / N
    
    if lambd <= 1e-9 or last_i == 0:
        return 0.0

    # If all S were infected (next_s == 0), we use the tail probability gradient
    if next_s == 0 and last_s > 0:
        cum_prob = 0.0
        grad_sum = 0.0
        
        for k in range(last_s):
            prob = y_pmf(k, prev_state, beta, N)
            score = LR_score(k, last_i, lambd, beta)
            
            cum_prob += prob
            grad_sum += score * prob
            
        denom = 1.0 - cum_prob
        if denom < 1e-12: return 0.0 # Numerical safety
        
        # d/dbeta log(1-F) = -f' / (1-F)
        return -grad_sum / denom

    # 3. Standard Case
    elif y >= 0:
        return LR_score(y, last_i, lambd, beta)
        
    return 0.0

def _get_infections_crn(state_vector, N, beta, u1):
    r = state_vector[1]
    if r < 1:
        return 0
    mu = beta * state_vector[0] * state_vector[1] / N
    p = r / (r + mu)
    Y = stats.nbinom.ppf(q = u1, n = r, p = p)
    I = np.minimum(Y, state_vector[0])
    return I

def step(state, state_plus, state_minus, beta, N, rng):
    u1 = rng.random()
    
    if state[1] < 1:
        next_state = state
    else:
        I_orig = _get_infections_crn(state, N, beta, u1)
        next_state = np.array([
            state[0] - I_orig,
            I_orig, 
            state[2] + state[1]            
        ])

    if state_plus[1] < 1:
        next_state_plus = state_plus
    else:
        I_plus = _get_infections_crn(state_plus, N, beta, u1)
        next_state_plus = np.array([
            state_plus[0] - I_plus,
            I_plus,
            state_plus[2] + state_plus[1]
        ])

    if state_minus[1] < 1:
        next_state_minus = state_minus
    else:
        I_minus = _get_infections_crn(state_minus, N, beta, u1)
        next_state_minus = np.array([
            state_minus[0] - I_minus,
            I_minus,
            state_minus[2] + state_minus[1]
        ])
    
    return next_state, next_state_plus, next_state_minus

def draw_samples_random_params(N, i0, 
                               v, beta, v_conf, beta_conf, 
                               T, 
                               N_samples, scrambler_seed, 
                               max_scrambler = 1e8, calc_LR = True,
                               v_seed=12345, beta_seed=51932, cores=1, 
                               random_beta=True, random_v=True):
    """
    Sample several simulation paths of the tSIR model, under a Beta
    prior on v and a Gamma prior on beta.

    Parameters
    ----------
    N : int
        Total population size.
    v : float
        Immunization rate. Between 0 and 1.
        Represents the mean of the prior distribution on v.
    v_conf: float
        Confidence parameter for the prior distribution.
        Larger means less variance. Variance scales as O(1/v_var).
    i0 : int
        Number of initially infected individuals.
    beta : float
        Contact rate. Positive number.
        Represents the mean of the prior distribution on beta.
    beta_conf: float
        Confidence parameter for the prior distribution of beta.
        Larger means less variance. Variance scales as O(1/v_var).
    T : int
        Time horizon (number of iterations to run the sim for)
    N_samples : int
        Number of sample paths to draw.
    scrambler_seed : int
        Random seed to use for the 'scrambler' 
        (generates random seeds for each simulation path.)
    max_scrambler : int, optional
        Maximum seed value the scrambler can generate. The default is 1e8.
    calc_LR : bool, optional
        Whether to calculate the likelihood-ratios of the transition kernel
        with respect to the contact rate. The default is True.

    Returns
    -------
    orig_samples : np.array[int] of length N_samples
        The total infected along the unperturbed sample paths.
    plus_samples : np.array[int] of length N_samples
        The total infected along the sample paths using the \mu^+ 
        distribution for number initially immune.
    minus_samples : np.array[int] of length N_samples
        The total infected along the sample paths using the \mu^- 
        distribution for number initially immune.
    trajectories : np.array[int] of shape (T+1, N_samples)
        The number infected at each timestep, for each sample path.
        Rows are timestep indices and columns are sample indices.
    score_samples : np.array[int] of shape (T, N_samples)
        The score function of the transition kernel P(X_{j+1} | X_j)
        with respect to the contact rate beta, evaluated at the
        the sampled states.
    """
    orig_samples = np.empty(N_samples, dtype=int)
    plus_samples = np.empty(N_samples, dtype=int)
    minus_samples = np.empty(N_samples, dtype=int)

    scrambler = np.random.default_rng(seed=scrambler_seed)

    prior_alpha = v_conf*v
    prior_beta = v_conf*(1-v)
    prior_a = beta_conf
    prior_b = beta/prior_a

    sampled_v = np.ones(N_samples)*v
    sampled_beta = np.ones(N_samples)*beta
    if random_v:
        sampled_v = stats.beta.rvs(a=prior_alpha, b=prior_beta, size=N_samples, random_state=v_seed)
    if random_beta:
        print("sampled random beta")
        sampled_beta = stats.gamma.rvs(a=prior_a, scale=prior_b, size=N_samples, random_state=beta_seed)
    
    if calc_LR:
        trajectories = np.empty((T+1, N_samples), dtype=int)
        score_samples = np.empty((T, N_samples), dtype=float)
    else:
        trajectories = None
        score_samples = None

    def inner_loop(i, s1, s2, n_val, i0_val, t_val, calc_lr_val, v_val, beta_val, s_v, s_beta):
        '''
        i : loop index
        s1 : population seed
        s2 : dynamics seed
        n_val: population size
        i0 : initial infected
        t_val : time horizon
        calc_lr_val : whether or not to calculate lr gradient
        v_val : mean value for posterior of dist
        beta_val : mean value for posterior
        s_v : sampled v
        s_beta : sampled beta
        '''
        orig, plus, minus = tSIR_WD_CRN(
            N=n_val, v=s_v, i0=i0_val, beta=s_beta, T=t_val, pop_seed=s1, dyn_seed=s2
        )
        res_orig = np.sum(orig, axis=0)[1]
        res_plus = np.sum(plus, axis=0)[1]
        res_minus = np.sum(minus, axis=0)[1]

        traj = None
        scores = None
        
        if calc_lr_val:
            traj = orig[:, 1]
            # Calculate scores as a list or numpy array
            scores = [LR_beta_term(orig[t+1], orig[t], s_beta, n_val) for t in range(t_val)]
            
        return res_orig, res_plus, res_minus, traj, scores
        
    if cores > 1:
        # 1. Prepare seeds
        seeds1 = scrambler.integers(0, max_scrambler, size=N_samples)
        seeds2 = scrambler.integers(0, max_scrambler, size=N_samples)
        
        # 2. Package all arguments into a list of tuples for starmap
        # Note: We pass everything the function needs to avoid global scope issues
        tasks = [
            (i, seeds1[i], seeds2[i], N, i0, T, calc_LR, v, beta, sampled_v[i], sampled_beta[i]) 
            for i in range(N_samples)
        ]

        with mp.Pool(cores) as pool:
            results = pool.starmap(inner_loop, tasks)

        # 3. Unpack results into your pre-allocated arrays
        for i, res in enumerate(results):
            orig_samples[i], plus_samples[i], minus_samples[i], traj, scores = res
            
            if calc_LR:
                trajectories[:, i] = traj
                score_samples[:, i] = scores
    else:
        for i in range(N_samples):
            seed1 = scrambler.integers(0, max_scrambler)
            seed2 = scrambler.integers(0, max_scrambler)
            this_beta = sampled_beta[i]
            this_v = sampled_v[i]
            orig, plus, minus = tSIR_WD_CRN(
                N=N, v=this_v, i0=i0, beta=this_beta, T=T, pop_seed=seed1, dyn_seed=seed2
            )
            
            orig_samples[i] = np.sum(orig, axis=0)[1]
            plus_samples[i] = np.sum(plus, axis=0)[1]
            minus_samples[i] = np.sum(minus, axis=0)[1]
    
            if calc_LR:
                trajectories[:,i] = orig[:,1]
                
                for t in range(T):
                    step_score = LR_beta_term(orig[t+1], orig[t], beta, N)
                    score_samples[t, i] = step_score
        
    return orig_samples, plus_samples, minus_samples, trajectories, score_samples



def estimators(orig_samples, plus_samples, minus_samples, N, i0, alpha = 0.9, eps = 1e-6):
    expr = lambda t: t + (1/(1-alpha)) * np.mean(np.maximum(0, orig_samples - t))
    # regul_expr = lambda t: t + (1/(1-alpha)) * np.mean(np.maximum(0, orig_samples - t)) + eps * t * t
    # soln = minimize(regul_expr, x0 = 0)
    # t = soln.x[0]

    # use the t = sup{x: \hat F(x) \leq \alpha} (left endpoint)
    N_samples = len(orig_samples)
    k = int(alpha * N_samples)
    t = np.partition(orig_samples, k)[k]

    cvar = expr(t)
    cvar_grad = np.mean((1/(1-alpha)) * (N - i0) * (np.maximum(plus_samples - t, 0.0) - np.maximum(minus_samples - t, 0.0)))
    mean = np.mean(orig_samples)
    mean_grad = np.mean((N - i0) * (plus_samples - minus_samples))
    return mean, mean_grad, cvar, cvar_grad, t

def bootstrap_est(orig_samples, plus_samples, minus_samples, N, i0, alpha=0.9, eps=1e-6, N_resamples=1000):
    N_samples = len(orig_samples)
    
    assert len(orig_samples) == len(plus_samples) == len(minus_samples)

    mean_resamples = np.empty(N_resamples, dtype=float)
    mean_grad_resamples = np.empty(N_resamples, dtype=float)
    cvar_resamples = np.empty(N_resamples, dtype=float)
    cvar_grad_resamples = np.empty(N_resamples, dtype=float)
    t_resamples = np.empty(N_resamples, dtype=float)

    for i in range(N_resamples):
        selected_indices = np.random.choice(N_samples, size=N_samples, replace=True)
        
        resampled_orig = orig_samples[selected_indices]
        resampled_plus = plus_samples[selected_indices]
        resampled_minus = minus_samples[selected_indices]
        
        mean, mean_grad, cvar, cvar_grad, t = estimators(
            resampled_orig, resampled_plus, resampled_minus, N, i0, alpha, eps
        )
        
        mean_resamples[i] = mean
        mean_grad_resamples[i] = mean_grad
        cvar_resamples[i] = cvar
        cvar_grad_resamples[i] = cvar_grad
        t_resamples[i] = t
        
    return {
        'mean': np.mean(mean_resamples),
        'mean_sd': np.std(mean_resamples, ddof=1),
        'mean_grad': np.mean(mean_grad_resamples),
        'mean_grad_sd': np.std(mean_grad_resamples, ddof=1),
        'cvar': np.mean(cvar_resamples),
        'cvar_sd': np.std(cvar_resamples, ddof=1),
        'cvar_grad': np.mean(cvar_grad_resamples),
        'cvar_grad_sd': np.std(cvar_grad_resamples, ddof=1),
        't': np.mean(t_resamples),
        't_sd': np.std(t_resamples, ddof=1)
    }