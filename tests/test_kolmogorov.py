#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:21:31 2026

@author: nicholasw
"""

from scipy.stats import nbinom, binom
from tsir_wd import tSIR_WD_CRN, tSIR_LR_v, inf_via_matrix_powers_tSIR
import numpy as np
import matplotlib.pyplot as plt

#%%
N = 101
v = 0.4
i0 = 5
beta = 3
T = 10

#%%

N = 11
v = 0.2
i0 = 1
beta = 7
T = 10

#%%
N_sims = 50000
sim_paths = [tSIR_LR_v(N, v, i0, beta, T, seed, seed*42) for seed in range(N_sims)]
I_expect_est = sum([path[0][:,1] for path in sim_paths])/N_sims
I_expect = inf_via_matrix_powers_tSIR(N, v, i0, beta, T)
sds = np.std(np.array([path[0][:,1] for path in sim_paths]),axis=0)/np.sqrt(N_sims)
#%%
plt.scatter(np.arange(T+1), I_expect, marker='o', color="orange", label = "matrix powers")
plt.errorbar(np.arange(T+1), I_expect_est, yerr = 1.96*sds, capsize= 8, label = "monte carlo")
plt.legend()
plt.xlabel("time")
plt.ylabel(r"$E[I_t]$")
plt.title("Compare matrix powers vs. monte carlo:\n N={}, v={}, i0={}, beta={}, N_sims={}".format(N, v, i0, beta, N_sims))

# %%
# Agrees with matrix powers but not the monte carlo???
# condition on S


def expect_1_S(S):
    r = i0
    mu = beta * S * i0 / N
    p = r / (r + mu)
    expect = np.array(
        [np.arange(0, S)]) @ nbinom.pmf(np.arange(0, S), p=p, n=r)
    expect += nbinom.sf(S-1, p=p, n=r) * S
    return expect


result = binom.pmf(np.arange(0, N-i0+1), p=1-v, n=N -
                   i0) @ [expect_1_S(s) for s in np.arange(0, N - i0 + 1)]
print(result)
#%%
I_expect[i0]