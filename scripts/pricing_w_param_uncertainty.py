#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:11:40 2026

Find "dual costs" (the largest cost parameter allowable that still satisfies
R_0 < 1), under parametric uncertainty.

We use the Beta distribution for v and the Gamma distribution for beta.

@author: nicholasw
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tsir_wd import draw_samples, draw_samples_random_params, grad_wrt_v, grad_wrt_beta
import multiprocess as mp

# %%
# Compute the c_v^* given a fixed mean for the distribution of beta.

N = 1000
i0 = 1
T = 10
N_samples = 10000
scrambler_seed = 12345677
beta_conf = 10

def compute_price(beta):
    N = 1000
    i0 = 1
    v = 1 - 1 / beta
    T = 10
    N_samples = 10000
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
            random_beta=True,
            #random_beta=False,
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
betas_to_try = np.linspace(2, 10)
with mp.Pool(11) as p:
    result = p.map(compute_price, betas_to_try)
# %%

df = pd.DataFrame(result)
df.columns = ["cv_star_mean", "cv_star_sd", "cv_norm_mean", "cv_norm_sd"]
df["beta"] = betas_to_try
df["N"] = N
df["i0"] = i0
df["T"] = T
df["N_samples"] = N_samples
df["beta_conf"] = beta_conf
df.to_csv("output/cv_star_vs_beta_2_10_beta_conf_10.csv", index=None)
#df.to_csv("output/cv_star_vs_beta_2_10_fixed.csv", index=None)


# %%
plt.plot(df["beta"], 100 * df["cv_norm_mean"], marker="o")
plt.fill_between(
    df["beta"],
    y1=100 * (df["cv_norm_mean"] - 1.96 * df["cv_norm_sd"]),
    y2=100 * (df["cv_norm_mean"] + 1.96 * df["cv_norm_sd"]),
    alpha=0.25,
)

# %%
# Compute the c_beta^* for a given mean value of v.
N = 1000
i0 = 1
v_conf = 10
t = 1
T = 10
N_samples = 10000
scrambler_seed = 12345677
def compute_price(v):
    N = 1000
    i0 = 1
    beta = 1 / (1 - v)
    v_conf = 10
    t = 1
    T = 10
    N_samples = 10000
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
            random_v=True
            #random_v=False
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
df["N"] = N
df["i0"] = i0
df["T"] = T
df["N_samples"] = N_samples
df["v_conf"] = 10 
df.to_csv("output/cbeta_star_vs_v_v_conf_10.csv", index=None)
# df.to_csv("output/cbeta_star_vs_v_v_fixed.csv", index=None)
