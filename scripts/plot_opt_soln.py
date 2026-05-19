#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:43:44 2026

@author: nicholasw
"""

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})

# %%
plt.figure(figsize=(10,7))
#df1 = pd.read_csv('output/v_star_vs_cost.csv')
#df2 = pd.read_csv('output/parametric_uncertainty/v_star_vs_cost_beta_conf_10.csv')

df1 = pd.read_csv('output/data/v_star_vs_cost_ci_independent.csv')
df2 = pd.read_csv('output/data/v_star_vs_cost_ci_independent_beta_conf_10.csv')

plt.plot(df1['c_v'], 
         df1['v_star'], marker='o', markersize=3,
         label = r"Fixed and known $\beta$")
plt.fill_between(df1['c_v'], 
                 y1 = df1['v_star'] + 1.96*df1['v_star_sd'], 
                 y2 = df1['v_star'] - 1.96*df1['v_star_sd'],
                 alpha=0.25
                 )
plt.plot(df2['c_v'], df2['v_star'], marker='o', markersize=3,
         label = r"Uncertain $\beta$"
         )
plt.fill_between(df2['c_v'], 
                 y1 = df2['v_star'] + 1.96*df2['v_star_sd'], 
                 y2 = df2['v_star'] - 1.96*df2['v_star_sd'],
                 alpha=0.25,
                 )
plt.plot(df1['c_v'], 1 - 1/df1['beta'], color="red", linestyle="dashed",
         label = r"$v = 1 - 1/\beta$"
         )
plt.legend()

plt.xlabel(r'$c_v$ (cost parameter for vaccination)')
plt.ylabel(r'$v^*$ (v found by stoch. approx)')
title1 = r"Problem 1. Solution $v^*$ as a function of cost parameter"
title2 = r"$\beta=5, N=1000, T=10$"
plt.title(title1 + "\n" + title2)
plt.savefig('output/compare_v_star_beta_conf_10.pdf')
plt.show()

# %%

plt.figure(figsize=(10,7))
df1 = pd.read_csv('output/data/beta_star_vs_cost_ci_independent.csv')
df2 = pd.read_csv('output/data/beta_star_vs_cost_ci_independent_v_conf_10.csv')

plt.plot(df1['c_v'], 
         df1['beta_star'], marker='o', markersize=3,
         label = r"Fixed and known $v$")
plt.fill_between(df1['c_v'], 
                 y1 = df1['beta_star'] + 1.96*df1['beta_star_sd'], 
                 y2 = df1['beta_star'] - 1.96*df1['beta_star_sd'],
                 alpha=0.25
                 )
plt.plot(df2['c_v'], df2['beta_star'], marker='o', markersize=3,
         label = r"Uncertain $v$"
         )
plt.fill_between(df2['c_v'], 
                 y1 = df2['beta_star'] + 1.96*df2['beta_star_sd'], 
                 y2 = df2['beta_star'] - 1.96*df2['beta_star_sd'],
                 alpha=0.25,
                 )
plt.plot(df2['c_v'], 1/(1-df2['v']), color="red", linestyle="dashed",
         label = r"$\beta = 1/(1 - v)$")
plt.legend()

plt.xlabel(r'$c_\beta$ (cost parameter for contact reduction)')
plt.ylabel(r'$\beta^*$ ($\beta$ found by stoch. approx)')
title1 = r"Problem 2. Solution $\beta^*$ as a function of cost parameter"
title2 = r"$v=0.3, N=1000, T=10$"
plt.title(title1 + "\n" + title2)
plt.savefig('output/compare_beta_star_beta_conf_10.pdf')
plt.show()

# %%


df1 = pd.read_csv('output/cv_star_vs_beta_2_10_fixed.csv')
df2 = pd.read_csv('output/cv_star_vs_beta_2_10_beta_conf_10.csv')

plt.plot(df1['beta'], df1['cv_star_mean'])
plt.fill_between(df1['beta'], 
                 y1=df1['cv_star_mean'] - 1.96*df1['cv_star_sd'],
                 y2=df1['cv_star_mean'] + 1.96*df1['cv_star_sd'],
                 alpha=0.25
                 )
plt.plot(df2['beta'],df2['cv_star_mean'])
plt.fill_between(df2['beta'], 
                 y1=df2['cv_star_mean'] - 1.96*df2['cv_star_sd'],
                 y2=df2['cv_star_mean'] + 1.96*df2['cv_star_sd'],
                 alpha=0.25
                 )

# %%

df1 = pd.read_csv('output/cv_star_vs_beta_2_10_fixed.csv')
df2 = pd.read_csv('output/cv_star_vs_beta_2_10_beta_conf_10.csv')

plt.figure(figsize=(10,7))
plt.plot(df1['beta'], df1['cv_star_mean']*df1['beta']/df1['N']*100,
         label=r'Known $\beta$',
         marker='o'
         )
plt.fill_between(df1['beta'], 
                 y1=(df1['cv_star_mean'] - 1.96*df1['cv_star_sd'])*df1['beta']/df1['N'] * 100,
                 y2=(df1['cv_star_mean'] + 1.96*df1['cv_star_sd'])*df1['beta']/df1['N'] * 100,
                 alpha=0.25
                 )
plt.plot(df2['beta'],df2['cv_star_mean']*df2['beta']/df1['N']*100,
         label=r'Uncertain $\beta$',
         marker='o'
         )
plt.fill_between(df2['beta'], 
                 y1=(df2['cv_star_mean'] - 1.96*df2['cv_star_sd'])*df2['beta']/df1['N']*100,
                 y2=(df2['cv_star_mean'] + 1.96*df2['cv_star_sd'])*df2['beta']/df1['N']*100,
                 alpha=0.25
                 )
plt.xlabel(r'$\beta$ (eff. contact rate)')
plt.ylabel(r'Normalized cost (% infection cost)')
plt.legend()
plt.title(r'Comparison of normalized cost: uncertainty in $\beta$ vs. known $\beta$')
plt.savefig('output/norm_cost_beta_conf_10.pdf')
# %%

df1 = pd.read_csv('output/cv_star_vs_beta_2_10_fixed.csv')
df2 = pd.read_csv('output/cv_star_vs_beta_2_10_beta_conf_10.csv')

plt.plot(df1['beta'], df1['cv_norm_mean']*df1['beta']/df1['N'])
plt.fill_between(df1['beta'], 
                 y1=(df1['cv_star_mean'] - 1.96*df1['cv_norm_sd'])*df1['beta']/df1['N'],
                 y2=(df1['cv_norm_mean'] + 1.96*df1['cv_norm_sd'])*df1['beta']/df1['N'],
                 alpha=0.25
                 )
plt.plot(df2['beta'],df2['cv_norm_mean']*df2['beta']/df1['N'])
plt.fill_between(df2['beta'], 
                 y1=(df2['cv_norm_mean'] - 1.96*df2['cv_norm_sd'])*df1['beta']/df1['N'],
                 y2=(df2['cv_norm_mean'] + 1.96*df2['cv_norm_sd'])*df1['beta']/df1['N'],
                 alpha=0.25
                 )

# %%

df1 = pd.read_csv("output/cbeta_star_vs_v_v_conf_10.csv")
df2 = pd.read_csv("output/cbeta_star_vs_v_v_fixed.csv")

plt.plot(df1['v'], df1['cbeta_star_mean'])
plt.fill_between(df1['v'], 
                 y1=df1['cbeta_star_mean'] - 1.96*df1['cbeta_star_sd'],
                 y2=df1['cbeta_star_mean'] + 1.96*df1['cbeta_star_sd'],
                 alpha=0.25
                 )
plt.plot(df2['v'],df2['cbeta_star_mean'])
plt.fill_between(df2['v'], 
                 y1=df2['cbeta_star_mean'] - 1.96*df2['cbeta_star_sd'],
                 y2=df2['cbeta_star_mean'] + 1.96*df2['cbeta_star_sd'],
                 alpha=0.25
                 )
# %%

df2 = pd.read_csv("output/cbeta_star_vs_v_v_conf_10.csv")
df1 = pd.read_csv("output/cbeta_star_vs_v_v_fixed.csv")

plt.figure(figsize=(10,7))
plt.plot(df1['v'], df1['cbeta_norm_mean']*100, marker='o', label=r"Known $v$")
plt.fill_between(df1['v'], 
                 y1=(df1['cbeta_norm_mean'] - 1.96*df1['cbeta_norm_sd'])*100,
                 y2=(df1['cbeta_norm_mean'] + 1.96*df1['cbeta_norm_sd'])*100,
                 alpha=0.25,
                 )
plt.plot(df2['v'],df2['cbeta_norm_mean']*100, marker='o', label=r"Uncertain $v$")
plt.fill_between(df2['v'], 
                 y1=(df2['cbeta_norm_mean'] - 1.96*df2['cbeta_norm_sd'])*100,
                 y2=(df2['cbeta_norm_mean'] + 1.96*df2['cbeta_norm_sd'])*100,
                 alpha=0.25
                 )
plt.xlabel(r'$v$ (prop. immune)')
plt.ylabel(r'Normalized cost (% infection cost)')
plt.title(r'Comparison of normalized cost: uncertainty in $v$ vs. known $v$')
plt.legend()
plt.savefig('output/norm_cost_beta_conf_10.pdf')