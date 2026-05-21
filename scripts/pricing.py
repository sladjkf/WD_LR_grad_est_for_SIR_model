#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:58:41 2026

Find "dual costs" (the largest cost parameter allowable that still satisfies
R_0 < 1), given a fixed and known value of the other parameter.

For example, if we want to find a critical cost for v, then we must have a 
fixed and known value of beta.

@author: nicholasw
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tsir_wd import *
import multiprocess as mp
import pandas as pd
import random

# In[250]:

def compute_price(beta):
    N = 1000
    i0 = 1
    v = 1 - 1/beta
    T = 10
    N_samples = 10000
    scrambler_seed = 12360982735
    orig, plus, minus, traj, score_samples = draw_samples(
        N, i0, v, beta, T, N_samples, scrambler_seed
    )
    v_mean, v_sd = grad_wrt_v(plus, minus, N, i0)
    beta_mean, beta_sd = grad_wrt_beta(traj, score_samples, T, N_samples)
    return -(1-v)**2 * v_mean, v_sd/np.sqrt(N_samples) * (1-v)**2


# In[251]:
betas_to_try = np.linspace(2,10)
with mp.Pool(11) as p:
    result = p.map(compute_price, betas_to_try)


# In[252]:
result = np.array(result)
df = pd.DataFrame({
    'beta':
})


# In[246]:


plt.plot(betas_to_try,result[:,0], marker='o', markersize=4)
#plt.plot(betas_to_try,result[:,0] + 1.96*result[:,1], color="red", linestyle="--")
#plt.plot(betas_to_try,result[:,0] - 1.96*result[:,1], color="red", linestyle="--")
plt.fill_between(betas_to_try, y1=result[:,0] + 1.96*result[:,1], y2=result[:,0] - 1.96*result[:,1], alpha = 0.4)
plt.xlabel(r'$\beta$ (eff. contact rate)')
plt.ylabel(r'$c^*_v$ (critical cost parameter for $v$)')
plt.title(r'$c^*_v$ vs. $\beta$' + "\n" + "N=1000, T=10")
plt.savefig('crit_cost_v.pdf')


# In[190]:


plt.plot(betas_to_try,result[:,0] * (betas_to_try - 1))
plt.plot(betas_to_try,result[:,0]*(betas_to_try - 1) + 1.96*result[:,1]*(betas_to_try - 1), color="red", linestyle="--")
plt.plot(betas_to_try,result[:,0]*(betas_to_try - 1) - 1.96*result[:,1]*(betas_to_try - 1), color="red", linestyle="--")
plt.title(r"cost to vacc. up to threshold vs. $\beta$")
plt.xlabel(r"$\beta$")
plt.ylabel(r"$c^* (\beta - 1)$")


# In[191]:


result[0,0]


# In[220]:


numerator = N * (1- (1/betas_to_try))
denom = (betas_to_try - 1)
factor = denom/numerator
# plt.plot(betas_to_try, result[:,0] * factor)
# plt.plot(betas_to_try, (result[:,0] + 1.96*result[:,1]) * factor)
# plt.plot(betas_to_try, (result[:,0] - 1.96*result[:,1]) * factor)
#plt.errorbar(x= betas_to_try, y=result[:,0] * factor, yerr= 1.96*result[:,1]*factor, capsize=2)
plt.fill_between(x= betas_to_try, y1 = 100*(result[:,0] + 1.96 *result[:,1]) * factor, y2 = 100*(result[:,0] - 1.96 *result[:,1]) * factor, alpha=0.5)
plt.plot(betas_to_try, 100*result[:,0]*factor, marker='o', markersize=4)
plt.title("")
plt.xlabel(r"$\beta$ (eff. contact rate)")
plt.ylabel("Cost (% infection cost)")
plt.title("Cost to threshold / expected num. of immune at threshold"+"\n"+"N=1000, T=10")
plt.savefig("normalized_cost_v.pdf")


# In[267]:


# 1. Calculate the main curve (Y)
numerator = N * (1 - (1/betas_to_try))
denom = result[:,0] * (betas_to_try - 1)
y_values = numerator / denom

# 2. Calculate the error (Propagation of Uncertainty)
# The relative error of Y is the same as the relative error of result[:,0]
relative_error = result[:,1] / result[:,0]
y_sd = y_values * relative_error

# 3. Plotting with a shaded error region (Cleanest look)
plt.plot(betas_to_try, y_values)
plt.plot(betas_to_try, 
                 y_values - 1.96*y_sd, 
                 linestyle="dashed", color="red")
plt.plot(betas_to_try, 
                 y_values + 1.96*y_sd, 
                 linestyle="dashed", color="red")

plt.xlabel(r"$\beta$")
plt.ylabel(r"$N / c^* \beta$")
plt.title(r"'Value' of vaccines vs. $\beta$")


# cost for $\beta$

# In[254]:


def compute_price(v, t):
    N = 1000
    i0 = 1
    beta = 1/(1-v)
    T = 10
    N_samples = 10000
    scrambler_seed = 12345677
    orig, plus, minus, traj, score_samples = draw_samples(
        N, i0, v, beta, T, N_samples, scrambler_seed
    )
    #v_mean, v_sd = grad_wrt_v(plus, minus, N, i0)
    beta_mean, beta_sd = grad_wrt_beta(traj, score_samples, T, N_samples)
    return (beta - t)**2 * beta_mean, (beta-t)**2 * beta_sd / np.sqrt(N_samples)


# In[255]:


v = np.linspace(0.001,0.999)
t = [np.floor(1/(1-v[0]))] * len(v)
inputs = list(zip(v,t))
with mp.Pool(11) as p:
    result = p.starmap(compute_price, inputs)


# In[200]:


result = np.array(result)
plt.figure()
plt.plot(v, result[:,0])
plt.plot(v, result[:,0] - 1.96 * result[:,1], color="red", linestyle="dashed")
plt.plot(v, result[:,0] + 1.96 * result[:,1], color="red", linestyle="dashed")
plt.xlabel("v")
plt.ylabel(r"$c_\beta$")
plt.title("Cost vs. v")
plt.figure()
plt.xlabel("v")
plt.ylabel(r"Cost to reach threshold")
plt.title("Cost to reach threshold vs. v")
factor = 1/(1/(1-v) - t[0])
plt.plot(v, (result[:,0] * factor))
plt.plot(v, result[:,0]*factor - 1.96 * factor * result[:,1], color="red", linestyle="dashed")
plt.plot(v, result[:,0]*factor + 1.96 * factor * result[:,1], color="red", linestyle="dashed")
plt.figure()
plt.xlabel("v")
plt.ylabel("Cost to reach threshold / avg. number of contacts")
plt.title(r"Normalized cost")
factor = 1/(1/(1-v) - t[0]) * (1-v) / 10
plt.plot(v, result[:,0] * factor)
#plt.yscale('log')
plt.plot(v, factor * (result[:,0] - 1.96 * result[:,1]), color="red", linestyle="dashed")
plt.plot(v, factor * (result[:,0] + 1.96 * result[:,1]), color="red", linestyle="dashed")


# In[229]:


plt.figure()
plt.xlabel("v (proportion immune)")
plt.ylabel(r"$c_\beta^*$ (critical cost parameter for $\beta$)")
plt.title(r"$c_\beta^*$ vs. v" + "\n" + "N=1000, T=10")
plt.plot(v, 100*result[:,0] * factor, marker='o', markersize=3)
#plt.yscale('log')
#plt.plot(v, 100*factor * (result[:,0] - 1.96 * result[:,1]), color="red", linestyle="dashed")
#plt.plot(v, 100*factor * (result[:,0] + 1.96 * result[:,1]), color="red", linestyle="dashed")
plt.fill_between(x=v, y1=100*factor * (result[:,0] - 1.96 * result[:,1]), y2=100*factor * (result[:,0] + 1.96 * result[:,1]), alpha=0.4)
plt.savefig("crit_cost_beta.pdf")


# In[257]:


result = np.array(result)
plt.figure()
plt.xlabel("v (proportion immune)")
plt.ylabel("Cost (% infection cost)")
plt.title(r"Cost to threshold / avg. contacts at threshold" + "\n" + "N=1000, T=10")
N = 1000
i0 = 1
beta = 1/(1-v)
T = 10
N_samples = 10000
scrambler_seed = 12345677
factor = 1/(1/(1-v) - t[0]) * (1-v) / T / N
plt.plot(v, 100*result[:,0] * factor, marker='o', markersize=3)
#plt.yscale('log')
#plt.plot(v, 100*factor * (result[:,0] - 1.96 * result[:,1]), color="red", linestyle="dashed")
#plt.plot(v, 100*factor * (result[:,0] + 1.96 * result[:,1]), color="red", linestyle="dashed")
plt.fill_between(x=v, y1=100*factor * (result[:,0] - 1.96 * result[:,1]), y2=100*factor * (result[:,0] + 1.96 * result[:,1]), alpha=0.4)
plt.savefig("normalized_cost_beta.pdf")

