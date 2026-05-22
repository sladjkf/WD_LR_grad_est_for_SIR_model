#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 22:03:10 2026

This script calculates the optimal solution of a stochastic approximation
problem where one variable is treated as a decision variable, and the other
treated as random with a known distribution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tsir_wd import draw_samples_random_params, grad_wrt_v, grad_wrt_beta
import multiprocess as mp

# %%

# Define an objective to solve finding the optimal v
# under uncertainty in beta

def obj_and_grad(v, c, alpha, N, i0, beta, beta_conf, T, N_samples, scrambler_seed):
    orig, plus, minus, traj, score_samples = draw_samples_random_params(
        N, i0, v, beta, 0, beta_conf, T, N_samples, scrambler_seed,
        random_beta = True, random_v = False
    )
    obj = np.mean(orig)
    v_mean, v_sd = grad_wrt_v(plus, minus, N, i0)
    return alpha*obj + c/(1-v) - c, alpha*v_mean + c*(1-v)**(-2)

def stoch_approx_loop(v0, c, alpha, N, i0, beta, beta_conf, T, samples_per_iter, iters, scrambler_seed, max_int=1e16, print_every=100, stepsize=0.5):
    v = [v0]
    scrambler = np.random.default_rng(scrambler_seed)
    for i in range(iters):
        this_seed = scrambler.integers(0, max_int)
        obj, grad = obj_and_grad(v[-1], c, alpha, N, i0, beta, beta_conf, T, samples_per_iter, this_seed)
        if i % print_every == 0:
            print(i,v[-1],obj,grad)
        v += [np.clip(v[-1] - (stepsize/(1+i)) * grad, 0.0001, 0.9999)]
    return v

# %%

# Solve an optimization problem to find the optimal vaccination

alpha = 1
N = 1000
i0 = 1
v = 0.1
beta = 5
beta_conf = 10
T = 10

def find_v(c, seed, iters):
    alpha = 1
    N = 1000
    i0 = 1
    v = 0.1
    beta = 5
    beta_conf = 10
    T = 10
    N_samples=10
    v = stoch_approx_loop(1 - 1/beta, c, alpha, N, i0, beta, beta_conf, T, N_samples, iters, seed, stepsize=1e-3)
    return v

# %%
c = np.linspace(1,10)
with mp.Pool(10) as p:
    result = p.map(find_v, c)

# %%
c = np.linspace(3,15,25)
from functools import partial
path_samples = []
# take independent samples to form a CI
ind_reps = 10
iters_per_rep = 500
with mp.Pool(11) as p:
    for i in range(ind_reps):
        seed = 123535 + i*159302
        this_result = p.map(partial(find_v, seed=seed, iters=iters_per_rep), c)
        # this is estimated solutions for each cost in the vector c, given this seed
        this_path = np.array([x[-1] for x in this_result])
        path_samples.append(this_path)

# %%
path_samples = np.array(path_samples)
soln_mean = np.mean(path_samples, axis=0)
soln_sd = np.std(path_samples, axis=0, ddof=1)/np.sqrt(ind_reps)
df = pd.DataFrame({
    'c_v' : c,
    'v_star': soln_mean,
    'v_star_sd': soln_sd,
    'beta': beta,
    'N': N,
    'T': T,
    'ind_reps': ind_reps,
    'iters_per_rep': iters_per_rep
})
df.to_csv('v_star_vs_cost_ci_independent_beta_conf_10.csv', index=None)


# %%
# Compute standard deviations of the solutions we found using the method
# of batch means

alpha = 1
N = 1000
i0 = 1
v = 0.1
beta = 5
beta_conf = 10
T = 10
result = np.array(result)
batch_means_results = []
for this_c, index in zip(c,range(result.shape[0])):
    batch_size = 1000
    batches = [np.mean(result[index,(k)*batch_size:(k+1)*batch_size]) for k in range(1,(result.shape[1]//batch_size))]
    batch_means_results.append([this_c, np.mean(batches), np.std(batches,ddof=1)])
batch_means_results = np.array(batch_means_results)

df = pd.DataFrame(batch_means_results)
df.columns = ['c_v', 'v_star', 'v_star_sd']
df['beta'] = beta
df['beta_conf'] = beta_conf
df['N'] = N
df['T'] = T
df.to_csv('v_star_vs_cost_beta_conf_10.csv',index=None)

# %%

# Define an objective to solve finding the optimal beta
# under uncertainty in v

def obj_and_grad(beta, q, t, alpha, v, v_conf, N, i0, T, N_samples, scrambler_seed):
    orig, plus, minus, traj, score_samples = draw_samples_random_params(
        N,
        i0,
        v,
        beta,
        v_conf,
        10000,
        T,
        N_samples,
        scrambler_seed,
        random_beta=False,
        random_v=True,
    )
    obj = np.mean(orig)
    # v_mean, v_sd = grad_wrt_v(plus, minus, N, i0)
    beta_mean, beta_sd = grad_wrt_beta(traj, score_samples, T, N_samples)
    return alpha * obj + q / (beta - t), alpha * beta_mean - q * (beta - t) ** (-2)


def stoch_approx_loop(
    beta0,
    q,
    t,
    alpha,
    v,
    v_conf,
    N,
    i0,
    T,
    samples_per_iter,
    iters,
    scrambler_seed,
    max_int=1e16,
    print_every=100,
    stepsize=0.5,
):
    beta = [beta0]
    scrambler = np.random.default_rng(scrambler_seed)
    for i in range(iters):
        this_seed = scrambler.integers(0, max_int)
        obj, grad = obj_and_grad(
            beta[-1], q, t, alpha, v, v_conf, N, i0, T, samples_per_iter, this_seed
        )
        if i % print_every == 0:
            print(i, beta[-1], obj, grad)
        beta += [np.clip(beta[-1] - (stepsize / (1 + i)) * grad, t + 0.001, 10)]
    return beta

#%%
# Given the uncertainty in v, find the optimal beta
alpha = 1
N = 1000
i0 = 1
v = 0.3
beta = 5
T = 10
N_samples = 10
v_conf = 10

def find_beta(q, t, iters, seed):
    alpha = 1
    N = 1000
    i0 = 1
    v = 0.3
    beta = 5
    T = 10
    N_samples = 10
    v_conf = 10
    beta = stoch_approx_loop(
        1 / (1 - v),
        q,
        t,
        alpha,
        v,
        v_conf,
        N,
        i0,
        T,
        N_samples,
        iters,
        seed,
        stepsize=1e-3,
    )
    return beta
# %%



q = np.linspace(1, 6, num=20)
t = [1] * len(q)
with mp.Pool(11) as p:
    result = p.starmap(find_beta, zip(q, t))

result = np.array(result)
batch_means_results = []
for this_q, index in zip(q, range(result.shape[0])):
    batch_size = 1000
    batches = [
        np.mean(result[index, (k) * batch_size : (k + 1) * batch_size])
        for k in range(1, (result.shape[1] // batch_size))
    ]
    batch_means_results.append([this_q, np.mean(batches), np.std(batches, ddof=1)])
batch_means_results = np.array(batch_means_results)


# %%

# %%
q = np.linspace(3,10,25)
t = [1] * len(q)
path_samples = []
ind_reps = 10
iters_per_rep = 500
# take independent samples to form a CI
with mp.Pool(11) as p:
    for i in range(ind_reps):
        seed = 304134 + i*159302
        this_result = p.starmap(partial(find_beta, seed=seed, iters=iters_per_rep), zip(q,t))
        # this is estimated solutions for each cost in the vector c, given this seed
        this_path = np.array([x[-1] for x in this_result])
        path_samples.append(this_path)

# %%
path_samples = np.array(path_samples)
soln_mean = np.mean(path_samples, axis=0)
soln_sd = np.std(path_samples, axis=0, ddof=1)/np.sqrt(ind_reps)
df = pd.DataFrame({
    'c_v' : c,
    'beta_star': soln_mean,
    'beta_star_sd': soln_sd,
    'v': v,
    'N': N,
    'T': T,
    'ind_reps': ind_reps,
    'iters_per_rep': iters_per_rep
})
df.to_csv('beta_star_vs_cost_ci_independent_v_conf_10.csv', index=None)



# %%

# Save the results to a csv file
alpha = 1
N = 1000
i0 = 1
v = 0.3
beta = 5
T = 10
N_samples = 10
v_conf = 10
df = pd.DataFrame(batch_means_results)
df.columns = ["c_v", "beta_star", "beta_star_sd"]
df["v"] = v
df["N"] = N
df["T"] = T
df["t"] = t
df.to_csv("beta_star_vs_cost_v_conf_10.csv", index=None)
