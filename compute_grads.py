#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tsir_wd import *
import multiprocess as mp
import pandas as pd


# %%
def no_vrt(v,beta):
    N = 1000
    i0 = 1
    N_samples = 10000
    T = 10
    scrambler_seed = 12345677
    orig, plus, _, traj, score_samples = draw_samples(
        N, i0, v, beta, T, N_samples, scrambler_seed
    )
    scrambler_seed = 8684838
    orig, _, minus, traj, score_samples = draw_samples(
        N, i0, v, beta, T, N_samples, scrambler_seed
    )
    v_grads =(N-i0) * (plus - minus)
    v_mean = np.mean(v_grads)
    v_sd = np.std(v_grads, ddof=1)
    beta_grads = orig * np.sum(score_samples, axis=0)
    beta_mean = np.mean(beta_grads)
    beta_sd = np.std(beta_grads, ddof=1)
    return {
        'v': v,
        'beta': beta,
        'N': N,
        'i0': i0,
        'N_samples': N_samples,
        'T': T,
        'v_mean': v_mean,
        'v_sd': v_sd,
        'beta_se': v_sd/np.sqrt(N_samples),
        'beta_mean': beta_mean,
        'beta_sd': beta_sd,
        'beta_se': beta_sd/np.sqrt(N_samples)
    }
vs_to_try = [0.1, 0.6]
betas_to_try = [2, 4]
tasks = [(v,beta) for beta in betas_to_try for v in vs_to_try] + [(0.3, 3)]
with mp.Pool(5) as pool:
    result = pool.starmap(lambda v,beta: no_vrt(v,beta), tasks)
# %%

import pandas as pd
df = pd.DataFrame(result)
df.to_csv('output/no_vrt_table.csv', index=False)
# In[2]:


N = 1000
i0 = 1
v = 0.1
beta = 5
T = 10
N_samples = 5000
scrambler_seed = 12345677
orig, plus, minus, traj, score_samples = draw_samples(
    N, i0, v, beta, T, N_samples, scrambler_seed
)
v_mean, v_sd = grad_wrt_v(plus, minus, N, i0)
beta_mean, beta_sd = grad_wrt_beta(traj, score_samples, T, N_samples)


# In[107]:


def compute_gradients(N, i0, v, beta, T, N_samples, scrambler_seed, eps_v=5e-3, eps_beta=5e-2):
    orig, plus, minus, traj, score_samples = draw_samples(
        N, i0, v, beta, T, N_samples, scrambler_seed
    )
    v_mean, v_sd = grad_wrt_v(plus, minus, N, i0)
    beta_mean, beta_sd = grad_wrt_beta(traj, score_samples, T, N_samples)

    fd_beta, _, _, _, _ = draw_samples(
        N, i0, v, beta + eps_beta, T, N_samples, scrambler_seed, calc_LR=False
    )

    fd_v, _, _, _, _ = draw_samples(
        N, i0, v + eps_v, beta, T, N_samples, scrambler_seed, calc_LR=False
    )

    fd_v_mean = np.mean((fd_v - orig)/eps_v)
    fd_v_sd = np.std((fd_v - orig)/eps_v, ddof=1)

    fd_beta_mean = np.mean((fd_beta - orig)/eps_beta)
    fd_beta_sd = np.std((fd_beta - orig)/eps_beta, ddof=1)
    return {
        'total_size_mean': np.mean(orig),
        'total_size_sd': np.std(orig),
        'v_mean': v_mean, 
        'v_sd': v_sd, 
        'beta_mean': beta_mean, 
        'beta_sd': beta_sd, 
        'fd_v_mean': fd_v_mean, 
        'fd_v_sd': fd_v_sd, 
        'fd_beta_mean': fd_beta_mean, 
        'fd_beta_sd': fd_beta_sd}


# In[108]:


compute_gradients(N, i0, v, beta, T, N_samples, scrambler_seed)


# In[109]:


df = pd.DataFrame([Out[12]])
df.index = [(0.1, 0.2)]
df


# In[170]:


betas_to_try = np.linspace(2,4,21)
betas_to_try = [4]
#vs_to_try = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
#vs_to_try = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
vs_to_try = np.linspace(0,1)
N = 100000
i0 = 1
v = 0.1
beta = 5
T = 10
N_samples = 10000
scrambler_seed = 12345677
eps_v = 5e-3
eps_beta = 5e-3
tasks = [(beta, v) for beta in betas_to_try for v in vs_to_try]
with mp.Pool(11) as pool:
    result = pool.starmap(lambda beta, v: compute_gradients(N, i0, v, beta, T, N_samples, scrambler_seed, eps_v, eps_beta), tasks)



# In[171]:


df = pd.DataFrame(result)
df['beta'] = [x[0] for x in tasks]
df['v'] = [x[1] for x in tasks]
df['N_samples'] = N_samples
df['N'] = N
df['T'] = T
df['i0'] = i0
#df.to_csv('big_run_results_2.csv', index=None)


# In[112]:


df = pd.DataFrame(result)
df['beta'] = [x[0] for x in tasks]
df['v'] = [x[1] for x in tasks]
df['N_samples'] = N_samples
df['N'] = N
df['T'] = T
df['i0'] = i0
df.to_csv('big_run_results.csv', index=None)


# In[173]:


df2 = df


# In[174]:


df = pd.read_csv('big_run_results.csv')


# In[176]:


df = pd.concat([df,df2])


# In[192]:


df.to_csv('big_run_combined.csv')

