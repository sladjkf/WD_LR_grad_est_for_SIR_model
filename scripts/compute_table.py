#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:10:51 2026

Script that ingests data from the previously computed simulations and combines 
them into a single table, while also doing the LaTeX formatting.
"""
# %%
import pandas as pd
import numpy as np

# %%

df_grad_fd = pd.read_csv('output/data/big_run_results.csv')
df_analytic = pd.read_csv('output/data/semi_analytic_result.csv')

# %%

v = 0.1
beta = 2

v = 0.6
beta = 2

v = 0.1
beta = 4

v = 0.6
beta = 4

selection_grad_fd = df_grad_fd[df_grad_fd['v'] == v]
selection_grad_fd = selection_grad_fd[selection_grad_fd['beta'] == beta]
# 2. Select ONLY the columns you want to unpack
cols_to_unpack = [
    'total_size_mean', 'total_size_sd', 'v_mean', 'v_sd', 'beta_mean',
    'beta_sd', 'fd_v_mean', 'fd_v_sd', 'fd_beta_mean', 'fd_beta_sd', 'N_samples'
]

# 3. Now unpack from the subsetted row
(total_size_mean, total_size_sd, v_mean, v_sd, beta_mean, 
 beta_sd, fd_v_mean, fd_v_sd, fd_beta_mean, fd_beta_sd, 
 N_samples) = selection_grad_fd[cols_to_unpack].iloc[0]

print(selection_grad_fd.T)

selection_a = df_analytic[df_analytic['v'] == v]
selection_a = selection_a[selection_a['beta'] == beta]
(size, dv, db) = selection_a[['size','dv','db']].iloc[0]
print(selection_a.T)


se = np.array([
    total_size_sd,
    v_sd,
    beta_sd,
    fd_v_sd,
    fd_beta_sd
])/np.sqrt(N_samples)

row1 = pd.Series({
 # 'size_eqn' : size,
 # 'size_mean' : total_size_mean,
 # 'size_se' : se[0],
 'dv_eqn' : dv,
 'dv_wd_mean' : v_mean,
 'dv_fd_mean' : fd_v_mean,
 'dv_wd_se' : se[1],
 'dv_fd_se' : se[3],
 # 'dbeta_eqn' : db,
 # 'dbeta_lr_mean': beta_mean,
 # 'dbeta_fd_mean' : fd_beta_mean,
 # 'dbeta_lr_se' : se[2],
 # 'dbeta_fd_se' : se[4]
})
# %%


df_table = pd.concat({'v=0.6, beta=2': row1, 'v=0.7, beta=4': row2, 'v=0.1, beta=4': row3}, axis=1)

len(row3)
latex_code = df_table.T.to_latex(
    index=True,           # Set to False if you don't want the row labels
    caption="Summary Statistics of Gradient Descent",
    label="tab:selection_results",
    column_format='c'*6,  # 'l' for left-aligned, 'r' for right-aligned
    escape=True,         # Useful if your data contains LaTeX math symbols
    position='h!'         # LaTeX position hint (here, exactly)
)
print(latex_code)

# %%

import pandas as pd
import numpy as np

# 1. Define your parameter sets
parameter_sets = [
    {'v': 0.1, 'beta': 2},
    {'v': 0.6, 'beta': 2},
    {'v': 0.1, 'beta': 4},
    {'v': 0.6, 'beta': 4}
]

v_rows = {}
beta_rows = {}

for p in parameter_sets:
    v_val = p['v']
    b_val = p['beta']
    label = f"v={v_val}, beta={b_val}"
    
    # --- Extraction Logic ---
    # Filter Grad FD
    sel_fd = df_grad_fd[(df_grad_fd['v'] == v_val) & (df_grad_fd['beta'] == b_val)]
    # Filter Analytic
    sel_a = df_analytic[(df_analytic['v'] == v_val) & (df_analytic['beta'] == b_val)]
    
    if not sel_fd.empty and not sel_a.empty:
        # Get scalar values
        fd = sel_fd.iloc[0]
        an = sel_a.iloc[0]
        
        # Calculate Standard Errors
        # Mapping indices: 1=v_sd, 3=fd_v_sd, 2=beta_sd, 4=fd_beta_sd
        n_sqrt = np.sqrt(fd['N_samples'])
        
        # --- Build the V-Stats Row ---
        v_rows[label] = pd.Series({
            'dv\_eqn': an['dv'],
            'dv\_wd\_mean': fd['v_mean'],
            'dv\_fd\_mean': fd['fd_v_mean'],
            'dv\_wd\_se': fd['v_sd'] / n_sqrt,
            'dv\_fd\_se': fd['fd_v_sd'] / n_sqrt
        })
        
        # --- Build the Beta-Stats Row ---
        beta_rows[label] = pd.Series({
            'dbeta\_eqn': an['db'],
            'dbeta\_lr\_mean': fd['beta_mean'],
            'dbeta\_fd\_mean': fd['fd_beta_mean'],
            'dbeta\_lr\_se': fd['beta_sd'] / n_sqrt,
            'dbeta\_fd\_se': fd['fd_beta_sd'] / n_sqrt
        })

# 2. Create the DataFrames
df_v = pd.DataFrame(v_rows).T
df_beta = pd.DataFrame(beta_rows).T

# 3. Generate LaTeX with rounding
# You can print them separately or join them. 
# Here is the rounded output for the V-block:
print(df_v.to_latex(float_format="%.2f", escape=False, position='h!'))
print(df_beta.to_latex(float_format="%.2f", escape=False, position='h!'))

latex_final = (
    "\\begin{table}[h!]\n\\centering\n\\caption{Combined Results}\n" +
    "\\textbf{V Statistics}\\\\\n" +
    df_v.to_latex(float_format="%.2f", escape=False) +
    "\n\\vspace{1em}\n\\textbf{Beta Statistics}\\\\\n" +
    df_beta.to_latex(float_format="%.2f", escape=False) +
    "\\end{table}"
)