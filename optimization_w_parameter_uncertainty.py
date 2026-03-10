#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 22:03:10 2026

@author: nick
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tsir_wd import draw_samples_random_params, grad_wrt_v, grad_wrt_beta
import multiprocess as mp

# %%

# Define an objective to solve finding the optimal v
# under uncertainty in beta

# def obj_and_grad(v, c, alpha, N, i0, beta, beta_conf, T, N_samples, scrambler_seed):
#     orig, plus, minus, traj, score_samples = draw_samples_random_params(
#         N, i0, v, beta, 0, beta_conf, T, N_samples, scrambler_seed,
#         random_beta = True, random_v = False
#     )
#     obj = np.mean(orig)
#     v_mean, v_sd = grad_wrt_v(plus, minus, N, i0)
#     return alpha*obj + c/(1-v) - c, alpha*v_mean + c*(1-v)**(-2)
#
# def stoch_approx_loop(v0, c, alpha, N, i0, beta, beta_conf, T, samples_per_iter, iters, scrambler_seed, max_int=1e16, print_every=100, stepsize=0.5):
#     v = [v0]
#     scrambler = np.random.default_rng(scrambler_seed)
#     for i in range(iters):
#         this_seed = scrambler.integers(0, max_int)
#         obj, grad = obj_and_grad(v[-1], c, alpha, N, i0, beta, beta_conf, T, samples_per_iter, this_seed)
#         if i % print_every == 0:
#             print(i,v[-1],obj,grad)
#         v += [np.clip(v[-1] - (stepsize/(1+i)) * grad, 0.0001, 0.9999)]
#     return v
#
# # %%
#
# # Solve an optimization problem to find the optimal vaccination
#
# alpha = 1
# N = 1000
# i0 = 1
# v = 0.1
# beta = 5
# beta_conf = 10
# T = 10
#
# def find_v(c):
#     alpha = 1
#     N = 1000
#     i0 = 1
#     v = 0.1
#     beta = 5
#     beta_conf = 10
#     T = 10
#     v = stoch_approx_loop(1 - 1/beta, c, alpha, N, i0, beta, beta_conf, T, 10, 11000, 157382, stepsize=1e-3)
#     return v
# c = np.linspace(1,10)
# with mp.Pool(10) as p:
#     result = p.map(find_v, c)
#
# # %%
#
# alpha = 1
# N = 1000
# i0 = 1
# v = 0.1
# beta = 5
# beta_conf = 10
# T = 10
# result = np.array(result)
# batch_means_results = []
# for this_c, index in zip(c,range(result.shape[0])):
#     batch_size = 1000
#     batches = [np.mean(result[index,(k)*batch_size:(k+1)*batch_size]) for k in range(1,(result.shape[1]//batch_size))]
#     batch_means_results.append([this_c, np.mean(batches), np.std(batches,ddof=1)])
# batch_means_results = np.array(batch_means_results)
#
# df = pd.DataFrame(batch_means_results)
# df.columns = ['c_v', 'v_star', 'v_star_sd']
# df['beta'] = beta
# df['beta_conf'] = beta_conf
# df['N'] = N
# df['T'] = T
# df.to_csv('v_star_vs_cost_beta_conf_10.csv',index=None)

# %%


def compute_price(v, t):
    N = 1000
    i0 = 1
    beta = 1 / (1 - v)
    T = 10
    N_samples = 10000
    scrambler_seed = 12345677
    orig, plus, minus, traj, score_samples = draw_samples(
        N, i0, v, beta, T, N_samples, scrambler_seed
    )
    # v_mean, v_sd = grad_wrt_v(plus, minus, N, i0)
    beta_mean, beta_sd = grad_wrt_beta(traj, score_samples, T, N_samples)
    return (beta - t) ** 2 * beta_mean, (beta - t) ** 2 * beta_sd / np.sqrt(N_samples)


v = np.linspace(0.001, 0.999)
t = [np.floor(1 / (1 - v[0]))] * len(v)
inputs = list(zip(v, t))
with mp.Pool(11) as p:
    result = p.starmap(compute_price, inputs)


# %%


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
        beta += [np.clip(beta[-1] - (stepsize / (1 + i)) * grad, t + 0.001, N)]
    return beta


# %%

# alpha = 1
# N = 1000
# i0 = 1
# v = 0.3
# beta = 5
# T = 10
# N_samples = 10
# v_conf = 10
# q = 1
# t = 1
# beta_star = stoch_approx_loop(
#     1 / (1 - v), q, t, alpha, v, v_conf, N, i0, T, 10, 11000, 157382, stepsize=0.99e-03
# )

# %%


import multiprocess as mp


def find_v(q, t):
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
        10,
        11000,
        157382,
        stepsize=0.99e-3,
    )
    return beta


q = np.linspace(1, 6, num=20)
t = [1] * len(q)
with mp.Pool(11) as p:
    result = p.starmap(find_v, zip(q, t))

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
# %%
def compute_price(beta):
    N = 1000
    i0 = 1
    v = 1 - 1 / beta
    T = 10
    N_samples = 50000
    scrambler_seed = 12345677
    beta_conf = 10
    orig, plus, minus, traj, score_samples = (
        draw_samples_random_params(
            N,
            i0,
            v,
            beta,
            0,
            beta_conf,
            T,
            N_samples,
            scrambler_seed,
            random_beta=False,
            random_v=False
        )
    )
    samples = -(beta ** (-2)) * (N - i0) * (plus - minus)
    norm_samples = samples * beta / N
    return (
        np.mean(samples),
        np.std(samples, ddof=1) / np.sqrt(N_samples),
        np.mean(norm_samples),
        np.std(norm_samples, ddof=1) / np.sqrt(N_samples),
    )


# %%
import multiprocess as mp

betas_to_try = np.linspace(2, 10)
with mp.Pool(11) as p:
    result = p.map(compute_price, betas_to_try)
# %%

df = pd.DataFrame(result)
df.columns = ["cv_star_mean", "cv_star_sd", "cv_norm_mean", "cv_norm_sd"]
df["beta"] = betas_to_try
df["N"] = 1000
df["i0"] = 1
df["T"] = 10
df["N_samples"] = 50000
df["beta_conf"] = 10
# df.to_csv("output/cv_star_vs_beta_2_10_beta_conf_10.csv", index=None)
df.to_csv("output/cv_star_vs_beta_2_10_fixed.csv", index=None)


# %%
plt.plot(df["beta"], 100 * df["cv_norm_mean"], marker="o")
plt.fill_between(
    df["beta"],
    y1=100 * (df["cv_norm_mean"] - 1.96 * df["cv_norm_sd"]),
    y2=100 * (df["cv_norm_mean"] + 1.96 * df["cv_norm_sd"]),
    alpha=0.25,
)

# %%


def compute_price(v):
    N = 1000
    i0 = 1
    beta = 1 / (1 - v)
    v_conf = 10
    t = 1
    T = 10
    N_samples = 20000
    scrambler_seed = 12345677
    orig, plus, minus, traj, score_samples = (
        draw_samples_random_params(
            N,
            i0,
            v,
            beta,
            v_conf,
            1,
            T,
            N_samples,
            scrambler_seed,
            random_beta=False,
            random_v=False
        )
    )
    grad_samples = grad_wrt_beta(traj, score_samples, T, N_samples, avg=False)
    samples = ((1 - v) ** (-1) - t) ** 2 * grad_samples
    norm_samples = (
        ((1 - v) ** (-1) - t) * grad_samples / (N * T * (1 - v) ** (-1))
    )
    return (
        np.mean(samples),
        np.std(samples, ddof=1) / np.sqrt(N_samples),
        np.mean(norm_samples),
        np.std(norm_samples, ddof=1) / np.sqrt(N_samples),
    )


# %%

vs_to_try = np.linspace(0.01, 0.99)
with mp.Pool(12) as p:
    result = p.map(compute_price, vs_to_try)

# %%

df = pd.DataFrame(result)
df.columns = ["cbeta_star_mean", "cbeta_star_sd", "cbeta_norm_mean", "cbeta_norm_sd"]
df["v"] = vs_to_try
df["N"] = 1000
df["i0"] = 1
df["T"] = 10
df["N_samples"] = 20000
#df["v_conf"] = 10 
#df.to_csv("output/cbeta_star_vs_v_v_conf_10.csv", index=None)
df.to_csv("output/cbeta_star_vs_v_v_fixed.csv", index=None)
