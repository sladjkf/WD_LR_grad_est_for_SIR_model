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
import random


# In[3]:

# Define a problem to solve for the optimal v    

def obj_and_grad(v, c, alpha, N, i0, beta, T, N_samples, scrambler_seed):
    orig, plus, minus, traj, score_samples = draw_samples(
        N, i0, v, beta, T, N_samples, scrambler_seed
    )
    obj = np.mean(orig)
    v_mean, v_sd = grad_wrt_v(plus, minus, N, i0)
    return alpha*obj + c/(1-v) - c, alpha*v_mean + c*(1-v)**(-2)

def stoch_approx_loop(v0, c, alpha, N, i0, beta, T, samples_per_iter, iters, scrambler_seed, max_int=1e16, print_every=100, stepsize=0.5):
    v = [v0]
    scrambler = np.random.default_rng(scrambler_seed)
    for i in range(iters):
        this_seed = scrambler.integers(0, max_int)
        obj, grad = obj_and_grad(v[-1], c, alpha, N, i0, beta, T, samples_per_iter, this_seed)
        if i % print_every == 0:
            print(i,v[-1],obj,grad)
        v += [np.clip(v[-1] - (stepsize/(1+i)) * grad, 0.0001, 0.99)]
    return v


# In[4]:

alpha = 1
N = 1000
i0 = 1
v = 0.1
beta = 5
T = 10
N_samples = 10    

# %%
# Calculate the optimal v for a variety of parameters    
from functools import partial
def find_v(c, seed, iters):
    alpha = 1
    N = 1000
    i0 = 1
    v = 0.1
    beta = 5
    T = 10
    N_samples = 10
    v = stoch_approx_loop(1 - 1/beta, c, alpha, N, i0, beta, T, N_samples, iters, seed, stepsize=1e-3)
    return v

# %%

c = np.linspace(3,15,25)
path_samples = []
ind_reps = 10
# take independent samples to form a CI
with mp.Pool(11) as p:
    for i in range(ind_reps):
        seed = 123535 + i*159302
        this_result = p.map(partial(find_v, seed=seed, iters=500), c)
        # this is estimated solutions for each cost in the vector c, given this seed
        this_path = np.array([x[-1] for x in this_result])
        # iterate averaging
        #this_path = np.array([np.mean(x) for x in this_result])
        path_samples.append(this_path)

# %%

path_samples = np.array(path_samples)
soln_mean = np.mean(path_samples, axis=0)
soln_sd = np.std(path_samples, axis=0, ddof=1)/np.sqrt(ind_reps)
df = pd.DataFrame({
    'c_v' : c,
    'v_star': soln_mean,
    'v_star_se': soln_sd,
    'beta': beta,
    'N': N,
    'T': T,
    'ind_reps':ind_reps,
    'iters_per_rep': 500
})
df.to_csv('v_star_vs_cost_ci_independent.csv', index=None)

# In[51]:

# Estimate the error using the batch means    

batch_means_results = []
for this_c, index in zip(c,range(result.shape[0])):
    batch_size = 1000
    batches = [np.mean(result[index,(k)*batch_size:(k+1)*batch_size]) for k in range(1,(result.shape[1]//batch_size))]
    batch_means_results.append([this_c, np.mean(batches), np.std(batches,ddof=1)])
batch_means_results = np.array(batch_means_results)

df = pd.DataFrame(batch_means_results)
df.columns = ['c_v', 'v_star', 'v_star_sd']
df['beta'] = beta
df['N'] = N
df['T'] = T
df.to_csv('v_star_vs_cost.csv',index=None)


# In[137]:

# Save the result as a CSV    

df = pd.read_csv('v_star_vs_cost.csv')
batch_means_results = np.array(df.iloc[:,:3])


# In[138]:

# Plot the solution

# plt.errorbar(x=batch_means_results[:,0], 
#              y=batch_means_results[:,1],
#              yerr =  1.96*batch_means_results[:,2],
#              linestyle="dotted",
#              capsize=3
#             )
alpha = 1
N = 1000
i0 = 1
v = 0.1
beta = 5
T = 10
N_samples = 10
plt.fill_between(x=batch_means_results[:,0], 
                 y1=batch_means_results[:,1]+1.96*batch_means_results[:,2], 
                 y2=batch_means_results[:,1]-1.96*batch_means_results[:,2], alpha=0.25)
plt.plot(batch_means_results[:,0], batch_means_results[:,1], markersize=3, marker="o")
#plt.scatter(batch_means_results[:,0], batch_means_results[:,1], markersize=1)
plt.plot([min(c), max(c)], [1-1/5, 1-1/5], linestyle = "dashed", label = r"Threshold: $ 1 - 1/\beta$")
plt.legend()
plt.xlabel(r"$c_v$ (cost parameter for vaccination)")
plt.ylabel(r"$v^*$ (Optimal v found by stoch. approx.)")
plt.title(r"Problem 1. $v^*$ vs. $c_v$" + "\n" + r"$\beta$={}, N={}, T={}".format(beta, N, T))
plt.savefig("v_star_vs_cost.pdf", bbox_inches="tight")


# In[5]:


plt.plot(c, [np.mean(x) for x in result])
#plt.scatter(c, [np.mean(x) for x in result])
plt.plot([min(c), max(c)], [1-1/5, 1-1/5], linestyle = "dashed", label = r"$1 - 1/\beta$")
plt.xlabel("cost parameter")
plt.ylabel("optimal v")
plt.title("How the optimal solution changes wrt. the cost parameter")
plt.legend()


# In[108]:

# Define a problem to solve for the optimal beta given a fixed v

def obj_and_grad(beta, q, t, alpha, v, N, i0, T, N_samples, scrambler_seed):
    orig, plus, minus, traj, score_samples = draw_samples(
        N, i0, v, beta, T, N_samples, scrambler_seed
    )
    obj = np.mean(orig)
    # v_mean, v_sd = grad_wrt_v(plus, minus, N, i0)
    beta_mean, beta_sd = grad_wrt_beta(traj, score_samples, T, N_samples)
    return alpha*obj + q/(beta - t), alpha*beta_mean - q*(beta - t)**(-2)

def stoch_approx_loop(beta0, q, t, alpha, v, N, i0, T, samples_per_iter, iters, scrambler_seed, max_int=1e16, print_every=100, stepsize=0.5):
    beta = [beta0]
    scrambler = np.random.default_rng(scrambler_seed)
    for i in range(iters):
        this_seed = scrambler.integers(0, max_int)
        obj, grad = obj_and_grad(beta[-1], q, t, alpha, v, N, i0, T, samples_per_iter, this_seed)
        if i % print_every == 0:
            print(i,beta[-1],obj,grad)
        beta += [np.clip(beta[-1] - (stepsize/(1+i)) * grad, t+0.001, 10)]
    return beta

#%%

# Calculate the solutions using stochastic approximation
alpha = 1
N = 1000
i0 = 1
v = 0.3
beta = 5
T = 10
N_samples = 10

import multiprocess as mp
def find_beta(q,t, iters, seed):
    alpha = 1
    N = 1000
    i0 = 1
    v = 0.3
    beta = 5
    T = 10
    N_samples = 10
    beta = stoch_approx_loop(1/(1-v), q, t, alpha, v, N, i0, T, N_samples, iters, seed, stepsize=1e-3)
    return beta
#q = np.linspace(1,6, num = 20)
#t = [1] * len(q)
#with mp.Pool(11) as p:
#    result = p.starmap(find_beta, zip(q,t))
# %%

find_beta(10,1,1100,304134)


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
df.to_csv('beta_star_vs_cost_ci_independent.csv', index=None)


# In[118]:

# Calculate the error using batch means and save the result

result = np.array(result)
batch_means_results = []
for this_q, index in zip(q,range(result.shape[0])):
    batch_size = 1000
    batches = [np.mean(result[index,(k)*batch_size:(k+1)*batch_size]) for k in range(1,(result.shape[1]//batch_size))]
    batch_means_results.append([this_q, np.mean(batches), np.std(batches,ddof=1)])
batch_means_results = np.array(batch_means_results)

df = pd.DataFrame(batch_means_results)
df.columns = ['c_v', 'v_star', 'v_star_sd']
df['beta'] = beta
df['N'] = N
df['T'] = T
df['t'] = t
df.to_csv('beta_star_vs_cost.csv',index=None)


# In[131]:

# Plot the solution as a function of the cost parameter

# plt.errorbar(x=batch_means_results[:,0], 
#              y=batch_means_results[:,1],
#              yerr =  1.96*batch_means_results[:,2],
#              linestyle="dotted",
#              capsize=3
#             )
alpha = 1
N = 1000
i0 = 1
v = 0.3
T = 10
N_samples = 10
plt.fill_between(x=batch_means_results[:,0], 
                 y1=batch_means_results[:,1]+1.96*batch_means_results[:,2], 
                 y2=batch_means_results[:,1]-1.96*batch_means_results[:,2], alpha=0.25)
plt.plot(batch_means_results[:,0], batch_means_results[:,1], markersize=3, marker="o")
#plt.scatter(batch_means_results[:,0], batch_means_results[:,1], markersize=1)
plt.plot([min(q), max(q)], [1/(1-v), 1/(1-v)],linestyle="dashed", label= "Threshold: 1/(1-v)")
plt.legend()
plt.xlabel(r"$c_\beta$ (cost parameter for contact reduction)")
plt.ylabel(r"$\beta^*$ (Optimal $\beta$ found by stoch. approx.)")
plt.title(r"Problem 2: $\beta^*$ vs. $c_\beta$" + "\n" + r"v={}, N={}, T={}".format(v, N, T))
plt.savefig("beta_star_vs_cost.pdf", bbox_inches="tight")


# In[113]:

v=0.3
plt.plot(q, [np.mean(x) for x in result])
plt.plot([min(q), max(q)], [1/(1-v), 1/(1-v)],linestyle="dashed")
plt.xlabel("cost")
plt.ylabel(r"optimal $\beta$")
plt.title(r"Optimal $\beta$ vs. cost when v=" + str(v))

# In[3]:

# Define a problem to simultaneously solve for v and betaa

def obj_and_grad(x, c, q, t, alpha, N, i0, T, N_samples, scrambler_seed):
    v, beta = x
    orig, plus, minus, traj, score_samples = draw_samples(
        N, i0, v, beta, T, N_samples, scrambler_seed
    )
    obj = np.mean(orig)
    v_mean, v_sd = grad_wrt_v(plus, minus, N, i0)
    beta_mean, beta_sd = grad_wrt_beta(traj, score_samples, T, N_samples)
    return alpha*obj + (c/(1-v) - c) + (q/(beta - t)), np.array([
        alpha*v_mean + c*(1-v)**(-2),
        alpha*beta_mean - q*(beta - t)**(-2)
    ])

def stoch_approx_loop(x0, c, q, t, alpha, N, i0, T, samples_per_iter, iters, scrambler_seed, max_int=1e16, print_every=100, stepsize=0.5, beta_max=1000):
    x = [x0]
    scrambler = np.random.default_rng(scrambler_seed)
    for i in range(iters):
        this_seed = scrambler.integers(0, max_int)
        obj, grad = obj_and_grad(x[-1], c, q, t, alpha, N, i0, T, samples_per_iter, this_seed)
        if i % print_every == 0:
            print(i,x[-1],obj,grad)
        x += [np.clip(x[-1] - (stepsize/(1+i)) * grad, [0.0001, t+0.0001], [0.9999,beta_max])]
    return x

# In[7]:


N = 1000
i0 = 1
v = 0.1
beta = 5
T = 10
N_samples = 10
scrambler_seed = 12345677
c = 950/19
q = 100
t = 5
alpha = 1
#obj_and_grad([0.5, 5], c, q, t, alpha, N, i0, T, 10000, 9999)
beta_start = 6
traj = stoch_approx_loop([1 - 1/beta_start, beta_start], c, q, t, alpha, N, i0, T, 10, 20000, 12343, stepsize=np.array([1e-3, 1e-2]), print_every=1000)


# In[8]:


traj = np.array(traj)
plt.plot(traj[:,1])
plt.plot(traj[:,0])


# In[9]:


np.mean(traj[:,0])


# In[10]:


obj_and_grad(traj[-1], c, q, t, alpha, N, i0, T, 10000, 9999)


# In[11]:


traj[-1]


# In[226]:


1 - 1/7.06463223


# In[12]:


traj[-1][0]/(1 - 1/traj[-1][1])


# # finding the critical c

# In[248]:


import multiprocess as mp


# In[249]:


betas_to_try = np.linspace(5,15)


