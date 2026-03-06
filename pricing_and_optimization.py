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


# Only wrt $v$

# In[210]:


N = 1000
i0 = 5
beta = 4
v = 1 - 1/beta
T = 10
N_samples = 20000
scrambler_seed = 12345677
orig, plus, minus, traj, score_samples = draw_samples(
    N, i0, v, beta, T, N_samples, scrambler_seed
)
v_mean, v_sd = grad_wrt_v(plus, minus, N, i0)
beta_mean, beta_sd = grad_wrt_beta(traj, score_samples, T, N_samples)


# In[211]:


print(-(1-v)**2 * v_mean, v_sd/np.sqrt(N_samples) * (1-v)**2)


# In[212]:


1/32.334390625


# In[3]:


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


# In[16]:


# cost scaling for cost of v
c = 950/19
alpha = 1
obj_and_grad(0.5, c, alpha, N, i0, beta, T, N_samples, scrambler_seed)


# how to set the cost level c?
# 
# - choose

# In[61]:


c = (1/2)*950/19
c = 12
alpha = 1
N = 1000
i0 = 1
v = 0.1
beta = 5
T = 10
N_samples = 10
v = stoch_approx_loop(1 - 1/beta, c, alpha, N, i0, beta, T, 10, 11000, 157382, stepsize=1e-3)


# In[4]:


import multiprocess as mp
def find_v(c):
    alpha = 1
    N = 1000
    i0 = 1
    v = 0.1
    beta = 5
    T = 10
    N_samples = 10
    v = stoch_approx_loop(1 - 1/beta, c, alpha, N, i0, beta, T, 10, 11000, 157382, stepsize=1e-3)
    return v
c = np.linspace(1,10)
with mp.Pool(11) as p:
    result = p.map(find_v, c)


# In[7]:


result = np.array(result)


# In[13]:


result[0,1000:]


# In[20]:


(result.shape[1]-batch_size) // batch_size


# In[47]:


c = np.linspace(1,10)


# In[51]:


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


df = pd.read_csv('v_star_vs_cost.csv')
batch_means_results = np.array(df.iloc[:,:3])


# In[138]:


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


# In[62]:


np.mean(v)


# In[63]:


np.std(v)/np.sqrt(10000)


# In[64]:


plt.hist(v)


# In[60]:


plt.plot(v)


# In[53]:


np.mean(v[4000:])


# In[132]:


obj_and_grad(0.7175264700759599, c, alpha, N, i0, beta, T, 10000, 9999)


# In[145]:


obj_and_grad(0.5898066989162875, c, alpha, N, i0, beta, T, 10000, 9999)


# # just beta

# In[108]:


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
        beta += [np.clip(beta[-1] - (stepsize/(1+i)) * grad, t+0.001, N)]
    return beta


# In[322]:


obj_and_grad(5, 10, 1, 1, 0.3, 1000, 1, 10, 10, 100010)


# In[327]:


alpha = 1
N = 1000
i0 = 1
v = 0.3
beta = 5
T = 10
N_samples = 5
t = 1
q = 20
result = stoch_approx_loop(1/(1-v), q, t, alpha, v, N, i0, T, N_samples, 10000, scrambler_seed, max_int=1e16, print_every=1000, stepsize=1e-3)


# In[328]:


np.mean(result)


# In[329]:


obj_and_grad(1.647609383660349, 20, 1, 1, 0.3, 1000, 1, 10, 10000, 100010)


# In[319]:


np.linspace(10,20)


# In[112]:


import multiprocess as mp
def find_v(q,t):
    alpha = 1
    N = 1000
    i0 = 1
    v = 0.3
    beta = 5
    T = 10
    N_samples = 10
    beta = stoch_approx_loop(1/(1-v), q, t, alpha, v, N, i0, T, 10, 11000, 157382, stepsize=1e-2)
    return beta
q = np.linspace(1,6, num = 20)
t = [1] * len(q)
with mp.Pool(11) as p:
    result = p.starmap(find_v, zip(q,t))


# In[118]:


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


# # wrt v and beta

# In[3]:


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


# In[4]:


10/4


# In[ ]:





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


# In[250]:


def compute_price(beta):
    N = 1000
    i0 = 1
    v = 1 - 1/beta
    T = 10
    N_samples = 40000
    scrambler_seed = 12360982735
    orig, plus, minus, traj, score_samples = draw_samples(
        N, i0, v, beta, T, N_samples, scrambler_seed
    )
    v_mean, v_sd = grad_wrt_v(plus, minus, N, i0)
    beta_mean, beta_sd = grad_wrt_beta(traj, score_samples, T, N_samples)
    return -(1-v)**2 * v_mean, v_sd/np.sqrt(N_samples) * (1-v)**2


# In[251]:


with mp.Pool(11) as p:
    result = p.map(compute_price, betas_to_try)


# In[252]:


result = np.array(result)


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

