#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fsolve
from tsir_wd import *
import multiprocess as mp
import pandas as pd


# In[2]:


df = pd.read_csv('big_run_combined.csv')
df


# In[3]:


df['v'].unique()


# In[4]:


df['beta'].unique()


# In[4]:


df = df.sort_values(by = ['v', 'beta'])


# In[6]:


N = df['N'].unique()[0]
T = df['T'].unique()[0]


# In[7]:


plt.rcParams.update({'font.size': 12})


# In[9]:


df = pd.read_csv('big_run_combined.csv')
plt.figure(figsize=(8,5))

selected_beta = 4

df = df[df['beta'] == selected_beta]
plt.scatter(df['v'], df['total_size_mean'])
plt.errorbar(df['v'], df['total_size_mean'], yerr = 1.96*df['total_size_sd']/np.sqrt(df['N_samples']), linestyle="dotted",capsize=4)
plt.vlines(x = 1 - 1/selected_beta, ymin = 0, ymax = 1000, linestyle="dashed", label=r"Threshold ($1 - \frac{1}{\beta}$)", color="red")

plt.legend()
plt.xlabel("v (proportion immune)")
plt.ylabel("Z (total infections)")
plt.title("Total infections vs. proportion immune\n" + r"$\beta$={}, N={}, T={}".format(selected_beta, N, T))
#plt.yscale('log')
plt.ylim((-5,1000))
#plt.xscale('log')
plt.savefig('Z_vs_v.pdf', format="pdf")


# In[32]:


df = pd.read_csv('big_run_combined.csv')
plt.figure(figsize=(8,5))

selected_beta = 2

df = df[df['beta'] == selected_beta]
plt.scatter(df['v'], df['total_size_mean'], color="blue")
plt.errorbar(df['v'], df['total_size_mean'], yerr = 1.96*df['total_size_sd']/np.sqrt(df['N_samples']), linestyle="dotted",capsize=4, color="blue", 
             label=r"$\beta=$"+str(selected_beta))
plt.vlines(x = 1 - 1/selected_beta, ymin = 0, ymax = 1000, linestyle="dashed", color="blue",alpha=0.3)

plt.legend()
plt.xlabel("v (proportion immune)")
plt.ylabel("Z (total infections)")
plt.title("Total infections vs. proportion immune\n" + r"$\beta$={}, N={}, T={}".format(selected_beta, N, T))
#plt.yscale('log')
plt.ylim((-5,1000))

selected_beta = 4
df = pd.read_csv('big_run_combined.csv')
df = df[df['beta'] == selected_beta]
plt.scatter(df['v'], df['total_size_mean'],color="green")
plt.errorbar(df['v'], df['total_size_mean'], yerr = 1.96*df['total_size_sd']/np.sqrt(df['N_samples']), linestyle="dotted",capsize=4, color="green", label=r"$\beta=$"+str(selected_beta))
plt.vlines(x = 1 - 1/selected_beta, ymin = 0, ymax = 1000, linestyle="dashed", color="green", alpha=0.5)

plt.legend()
plt.xlabel("v (proportion immune)")
plt.ylabel("Z (total infections)")
plt.title("Total infections vs. proportion immune\n" + r"$\beta$={},{}; N={}, T={}".format(2,4, N, T))
plt.yscale('log')
plt.ylim((-5,1000))
#plt.xscale('log')
plt.savefig("compare_Z_vs_v.pdf")


# In[44]:


v = 0.1
betas_to_solve = np.linspace(2,4)
solns = []
for beta in betas_to_solve:
    soln = fsolve(
        func = lambda tau: 1 + 1/1000 - tau - np.exp(- beta * (1-v) * (1/1000) * tau),
        x0 = 1
    )
    solns.append(soln)
solns = 1000*np.array(solns)
plt.plot(betas_to_solve, solns)
#plt.vlines(x = 1 - 1/beta, ymin = 0, ymax = 1, linestyle="dashed", label=r"Threshold ($1 - \frac{1}{\beta}$)", color="red")


# In[48]:


v = 0.1
betas_to_solve = np.linspace(2,4)
solns = []
for beta in betas_to_solve:
    soln = fsolve(
        func = lambda s: np.exp(-beta * (1-v) * (1-s)) - s,
        x0 = 0
    )
    solns.append(soln)
solns = 1 - np.array(solns)
plt.plot(betas_to_solve, solns, label="v={}".format(v))

v = 0.45
betas_to_solve = np.linspace(2,4)
solns = []
for beta in betas_to_solve:
    soln = fsolve(
        func = lambda s: np.exp(-beta * (1-v) * (1-s)) - s,
        x0 = 0
    )
    solns.append(soln)
solns = 1 - np.array(solns)
plt.plot(betas_to_solve, solns, label="v={}".format(v))
plt.title(r"Probability of a large outbreak vs. $\beta$")
plt.legend()
plt.xlabel(r"$\beta$ (eff. contact rate)")
plt.ylabel("Probability of a large outbreak")
#plt.vlines(x = 1 - 1/beta, ymin = 0, ymax = 1, linestyle="dashed", label=r"Threshold ($1 - \frac{1}{\beta}$)", color="red")
plt.savefig("prob_outbreak_vs_beta.pdf")


# In[41]:


beta = 4
vs_to_solve = np.linspace(0, 1)
solns = []
for v in vs_to_solve:
    soln = fsolve(
        func = lambda s: np.exp(-beta * (1-v) * (1-s)) - s,
        x0 = 0
    )
    solns.append(soln)
solns = 1 - np.array(solns)
plt.plot(vs_to_solve, solns)
plt.vlines(x = 1 - 1/beta, ymin = 0, ymax = 1, linestyle="dashed", label=r"Threshold ($1 - \frac{1}{\beta}$)", color="red")
plt.legend()
plt.xlabel("v (Proportion immune)")
plt.ylabel("1-q (Probability of an outbreak)")
plt.title("Probability of an outbreak vs. v\n"+r"$\beta$={}".format(beta, N, T))
plt.savefig("branching.pdf")


# In[22]:


fig, axs = plt.subplots(3,1, figsize=(10,7))
betas = [2, 3, 4]
for ax, selected_beta in zip(axs, betas):
    df = pd.read_csv('big_run_combined.csv')
    df = df[df['beta'] == selected_beta]
    ax.scatter(df['v'], df['total_size_mean'])
    ax.errorbar(df['v'], df['total_size_mean'], yerr = 1.96*df['total_size_sd']/np.sqrt(df['N_samples']), linestyle="dotted",capsize=4)
    ax.vlines(x = 1 - 1/selected_beta, ymin = 0, ymax = 1000, linestyle="dashed", label=r"Threshold ($1 - \frac{1}{\beta}$)", color="red")
    ax.set_yscale('log')
    ax.set_ylim((1,1000))

plt.legend()
plt.xlabel("v (proportion immune)")
plt.ylabel("Z (total infections)")
axs[0].set_title("Total infections vs. proportion immune\n" + r"$\beta$={}, N={}, T={}".format(selected_beta, N, T))


# In[55]:


df = pd.read_csv('big_run_combined.csv')
selected_v = 0.01
df = df[df['v'] == selected_v]
plt.scatter(df['beta'], df['total_size_mean'])
plt.errorbar(df['beta'], df['total_size_mean'], yerr = 1.96*df['total_size_sd']/np.sqrt(df['N_samples']), linestyle="dotted",capsize=4)
df = pd.read_csv('big_run_combined.csv')

plt.xlabel(r"$\beta$ (eff. contact rate)")
plt.ylabel("Z (total infections)")
plt.title("Total infections vs. eff. contact rate\n" + r"v={}, N={}, T={}".format(selected_v, N, T))
#lt.yscale('log')
#lt.vlines(x = 1/(1-selected_v), ymin = 0, ymax = 200, linestyle="dashed")
plt.ylim((0,800))


# In[42]:


df = pd.read_csv('big_run_combined.csv')
selected_v = 0.1
df = df[df['v'] == selected_v]
plt.scatter(df['beta'], df['total_size_mean'], label = "v={}".format(selected_v))
plt.errorbar(df['beta'], df['total_size_mean'], yerr = 1.96*df['total_size_sd']/np.sqrt(df['N_samples']), linestyle="dotted",capsize=4)
df = pd.read_csv('big_run_combined.csv')

plt.xlabel(r"$\beta$ (eff. contact rate)")
plt.ylabel("Z (total infections)")
plt.title("Total infections vs. eff. contact rate\n" + r"v={}, N={}, T={}".format(selected_v, N, T))
#lt.yscale('log')
#lt.vlines(x = 1/(1-selected_v), ymin = 0, ymax = 200, linestyle="dashed")
plt.ylim((0,800))

#plt.yscale('log')
#lt.vlines(x = 1/(1-selected_v), ymin = 0, ymax = 200, linestyle="dashed")



df = pd.read_csv('big_run_combined.csv')
selected_v = 0.45
df = df[df['v'] == selected_v]
plt.scatter(df['beta'], df['total_size_mean'], label="v={}".format(selected_v))
plt.errorbar(df['beta'], df['total_size_mean'], yerr = 1.96*df['total_size_sd']/np.sqrt(df['N_samples']), linestyle="dotted",capsize=4)
df = pd.read_csv('big_run_combined.csv')

plt.xlabel(r"$\beta$ (eff. contact rate)")
plt.ylabel("Z (total infections)")
plt.title("Total infections vs. eff. contact rate\n" + r"N={}, T={}".format(N, T))
plt.legend()
plt.savefig("Z_vs_beta.pdf")


# In[76]:





# In[20]:


df = pd.read_csv('big_run_combined.csv')
selected_v = 0.55
df = df[df['v'] == selected_v]
plt.scatter(df['beta'], df['total_size_mean'])
plt.errorbar(df['beta'], df['total_size_mean'], yerr = 1.96*df['total_size_sd']/np.sqrt(df['N_samples']), linestyle="dotted",capsize=4)
df = pd.read_csv('big_run_combined.csv')

plt.xlabel(r"$\beta$ (eff. contact rate)")
plt.ylabel("Z (total infections)")
plt.title("Total infections vs. eff. contact rate\n" + r"v={}, N={}, T={}".format(selected_v, N, T))
#lt.yscale('log')
plt.vlines(x = 1/(1-selected_v), ymin = 0, ymax = 200, linestyle="dashed")


# In[37]:


df = pd.read_csv('big_run_combined.csv')
selected_beta = 4
df = df[df['beta'] == selected_beta]
plt.scatter(df['v'], df['v_mean'])
plt.errorbar(df['v'], df['v_mean'], yerr = 1.96*df['v_sd']/np.sqrt(df['N_samples']), linestyle="dotted",capsize=4, label = "Weak derivative")
plt.scatter(df['v'], df['fd_v_mean'])
plt.errorbar(df['v'], df['fd_v_mean'], yerr = 1.96*df['fd_v_sd']/np.sqrt(df['N_samples']),linestyle="dotted",capsize=4, label="Finite difference")

plt.title(r'$\partial E[Z] / \partial v$ vs. v' + "\n" + r"$\beta$={}, N={}, T={}, $\varepsilon_v$={}".format(selected_beta, N, T, 5e-3))
plt.xlabel("v (proportion immune)")
plt.ylabel(r'$\partial E[Z] / \partial v$')
plt.vlines(x = 1 - 1/selected_beta, 
           ymin = min(df['v_mean']- 1.96*df['v_sd']/np.sqrt(df['N_samples'])), 
           ymax = max(df['v_mean']+ 1.96*df['v_sd']/np.sqrt(df['N_samples'])), 
           linestyle="dashed", label=r"Threshold ($1 - \frac{1}{\beta}$)", color="red"
)
plt.ylim((-2000,0))
plt.legend()
plt.yscale('symlog')
plt.savefig('v_deriv_beta_4.pdf')


# In[39]:


df = pd.read_csv('big_run_combined.csv')
selected_beta = 2
df = df[df['beta'] == selected_beta]
plt.scatter(df['v'], df['v_mean'])
plt.errorbar(df['v'], df['v_mean'], yerr = 1.96*df['v_sd']/np.sqrt(df['N_samples']), linestyle="dotted",capsize=4, label = "Weak derivative")
plt.scatter(df['v'], df['fd_v_mean'])
plt.errorbar(df['v'], df['fd_v_mean'], yerr = 1.96*df['fd_v_sd']/np.sqrt(df['N_samples']),linestyle="dotted",capsize=4, label="Finite difference")

plt.title(r'$\partial E[Z] / \partial v$ vs. v' + "\n" + r"$\beta$={}, N={}, T={}, $\varepsilon$={}".format(selected_beta, N, T, 5e-3))
plt.xlabel("v (proportion immune)")
plt.ylabel(r'$\partial E[Z] / \partial v$')
plt.vlines(x = 1 - 1/selected_beta, 
           ymin = min(df['v_mean']- 1.96*df['v_sd']/np.sqrt(df['N_samples'])), 
           ymax = 0, 
           linestyle="dashed", label=r"Threshold ($1 - \frac{1}{\beta}$)", color="red"
)
plt.ylim((-2000,0))
plt.legend()
plt.yscale('symlog')
plt.savefig('v_deriv_beta_2.pdf')


# In[91]:


df = pd.read_csv('big_run_combined.csv')
selected_beta = 4

df = df[df['beta'] == selected_beta]
plt.scatter(df['v'], -df['v_mean'])
plt.errorbar(df['v'], -df['v_mean'], yerr = 1.96*df['v_sd']/np.sqrt(df['N_samples']), linestyle="dotted",capsize=4, label = "Weak derivative")
plt.scatter(df['v'], -df['fd_v_mean'])
plt.errorbar(df['v'], -df['fd_v_mean'], yerr = 1.96*df['fd_v_sd']/np.sqrt(df['N_samples']),linestyle="dotted",capsize=4, label="Finite difference")

plt.title(r'$|\partial E[Z] / \partial v|$ vs. v' + "\n" + r"$\beta$={}, N={}, T={}, $\varepsilon$={}".format(selected_beta, N, T, 5e-3))
plt.xlabel("v (proportion immune)")
plt.ylabel(r'$\partial E[Z] / \partial v$')
plt.vlines(x = 1 - 1/selected_beta, 
           ymax = -min(df['v_mean']- 1.96*df['v_sd']/np.sqrt(df['N_samples'])), 
           ymin = 0.0, 
           linestyle="dashed", label=r"Threshold ($1 - \frac{1}{\beta}$)", color="red"
)
plt.legend()
plt.yscale('log')


# In[69]:


df = pd.read_csv('big_run_combined.csv')
selected_v = 0.5
df = df[df['v'] == selected_v]
plt.tight_layout()
plt.scatter(df['beta'], df['v_mean'])
plt.errorbar(df['beta'], df['v_mean'], yerr = 1.96*df['v_sd']/np.sqrt(df['N_samples']), linestyle="dotted", capsize=4, label = "WD")
plt.scatter(df['beta'], df['fd_v_mean'])
plt.errorbar(df['beta'], df['fd_v_mean'], yerr = 1.96*df['fd_v_sd']/np.sqrt(df['N_samples']),linestyle="dotted", capsize=4, label = "Finite difference")

plt.title(r'$\partial E[Z] / \partial v$ vs. $\beta$' + "\n" + r"v={}, N={}, T={}, $\varepsilon$={}".format(selected_v, N, T, 5e-3))
plt.xlabel(r"$\beta$ (eff. contact rate)")
plt.ylabel(r'$\partial E[Z] / \partial v$')
plt.legend()
plt.savefig('v_deriv_vs_beta.pdf', bbox_inches="tight")


# In[90]:


1/(1-0.6)


# In[46]:


selected_v = 0.45
df = pd.read_csv('big_run_combined.csv')
df = df[df['v'] == selected_v]
plt.errorbar(df['beta'], df['beta_mean'], yerr = 1.96*df['beta_sd']/np.sqrt(df['N_samples']), linestyle="dotted",capsize=4, label = "LR")
plt.errorbar(df['beta'], df['fd_beta_mean'], yerr = 1.96*df['fd_beta_sd']/np.sqrt(df['N_samples']),linestyle="dotted",capsize=4, label="FD")
plt.title(r"$\partial E[Z] / \partial\beta$ vs. $\beta$" + "\n" + "v={}, N={}, T={}".format(selected_v, N,T))
plt.xlabel(r"$\beta$ (eff. contact rate)")
plt.ylabel(r"$\partial E[Z]/ \partial \beta$")
plt.legend()
plt.ylim((0,500))
plt.savefig('beta_deriv_vs_beta_v_45.pdf')


# In[125]:


selected_v = 0.4
df = pd.read_csv('big_run_combined.csv')
df = df[df['v'] == selected_v]
plt.errorbar(df['beta'], df['beta_mean'], yerr = 1.96*df['beta_sd']/np.sqrt(df['N_samples']), linestyle="dotted",capsize=4)
plt.errorbar(df['beta'], df['fd_beta_mean'], yerr = 1.96*df['fd_beta_sd']/np.sqrt(df['N_samples']),linestyle="dotted",capsize=4)
print(1/(1-selected_v))


# In[67]:


df = pd.read_csv('big_run_combined.csv')
selected_beta = 3
df = df[df['beta'] == selected_beta]

plt.errorbar(df['v'], df['beta_mean'], yerr = 1.96*df['beta_sd']/np.sqrt(df['N_samples']), linestyle="dotted", capsize=4)
plt.errorbar(df['v'], df['fd_beta_mean'], yerr = 1.96*df['fd_beta_sd']/np.sqrt(df['N_samples']),linestyle="dotted", capsize=4)
plt.vlines(x = 1 - 1/selected_beta, 
           ymin = min(df['beta_mean']- 1.96*df['beta_sd']/np.sqrt(df['N_samples'])), 
           ymax = 350, 
           linestyle="dashed", label=r"Threshold ($1 - \frac{1}{\beta}$)", color="red"
)
plt.legend()
plt.title('')
plt.title(r"$\partial E[Z] / \partial\beta$ vs. v" + "\n" + r"$\beta$"+"={}, N={}, T={}".format(selected_beta, N,T))
plt.xlabel(r"v (proportion immune)")
plt.ylabel(r"$\partial E[Z]/ \partial \beta$")
plt.savefig('deriv_beta_vs_v.pdf')


# In[73]:


1 - 1/2

