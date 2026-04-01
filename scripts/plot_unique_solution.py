# -*- coding: utf-8 -*-
"""
plot the gradients as a function of parameters to show how solutions to first order
conditions might not be unique
"""

# %%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%

df = pd.read_csv('output/big_run_combined.csv')

# %%

selection = df[df['beta'] == 4]
plt.plot(selection['v'], selection['v_mean'], marker='o', label = r"est. of $\frac{d \mathbb{E}[Z]}{dv}$")
plt.fill_between(selection['v'], 
                 y1 = selection['v_mean'] - 1.96*selection['v_sd']/np.sqrt(selection['N_samples']),
                 y2 = selection['v_mean'] + 1.96*selection['v_sd']/np.sqrt(selection['N_samples']),
                 alpha=0.25
                 )
plt.plot(selection['v'], -4*(1-selection['v'])**(-2), label=r"$-c_v(1-v)^{-2}$")
plt.yscale('symlog')
plt.legend()
plt.savefig('dv_solution_unique.pdf')

# %%

selection = df[df['beta'] == 4]
plt.plot(selection['v'], selection['v_mean'], marker='o', label = r"est. of $\frac{d \mathbb{E}[Z]}{dv}$")
plt.fill_between(selection['v'], 
                 y1 = selection['v_mean'] - 1.96*selection['v_sd']/np.sqrt(selection['N_samples']),
                 y2 = selection['v_mean'] + 1.96*selection['v_sd']/np.sqrt(selection['N_samples']),
                 alpha=0.25
                 )
plt.plot(selection['v'], [-1000]*len(selection['v']), label = r"$-c_v = -1000$")
plt.legend()
plt.savefig('dv_solution_nonunique.pdf')

# %%

selection = df[df['v'] == 0.4]
plt.plot(selection['beta'], selection['beta_mean'], 
         marker='o', 
         label = r"est. of $\frac{d \mathbb{E}[Z]}{d\beta}$")
plt.fill_between(selection['beta'], 
                 y1 = selection['beta_mean'] - 1.96*selection['beta_sd']/np.sqrt(selection['N_samples']),
                 y2 = selection['beta_mean'] + 1.96*selection['beta_sd']/np.sqrt(selection['N_samples']),
                 alpha=0.25
                 )
plt.plot(selection['beta'], 100*(selection['beta']-1)**(-2), label = r"$100 (\beta - 1)^{-2}$")
plt.legend()
plt.savefig('dbeta_solution_unique.pdf')

# %%

selection = df[df['v'] == 0.4]
plt.plot(selection['beta'], selection['beta_mean'], 
         marker='o', 
         label = r"est. of $\frac{d \mathbb{E}[Z]}{d\beta}$")
plt.fill_between(selection['beta'], 
                 y1 = selection['beta_mean'] - 1.96*selection['beta_sd']/np.sqrt(selection['N_samples']),
                 y2 = selection['beta_mean'] + 1.96*selection['beta_sd']/np.sqrt(selection['N_samples']),
                 alpha=0.25
                 )
plt.plot(selection['beta'], [160]*len(selection['beta']), label = r"$c_v = 160$")
plt.legend()
plt.savefig('dbeta_solution_nonunique.pdf')
