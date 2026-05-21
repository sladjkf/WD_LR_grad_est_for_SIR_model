#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 20:05:37 2026

This script computes the sensitivities that incorporate parameter uncertainty,
i.e., we average across both the uncertainty in the transmission dynamics
and input uncertainties in the parameters.

The parameters are sampled indpendently.

@author: nick
"""
# %%

from tsir_wd import *
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import multiprocess as mp
import pandas as pd

# %%
import csv
import os
import numpy as np
import pandas as pd

old_run = pd.read_csv('output/data/big_run_results.csv')

output_file = 'parametric_uncertainty_results.csv'

# 1. Define all your column headers in one place
fieldnames = [
    'beta_conf', 'v_conf', 'N', 'i0', 'T', 'v', 'beta', 
    'beta_grad', 'beta_grad_se', 'v_grad', 'v_grad_se', 
    'total_size_mean', 'total_size_se', 'N_samples'
]

# 2. Open the file once in append mode ('a')
# Using 'a' allows you to restart the script and keep previous data if it crashes
file_exists = os.path.isfile(output_file)

with open(output_file, 'a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write header only if it's a new file
    if not file_exists:
        writer.writeheader()

    for beta_conf in [10]:
        for v_conf in [10]:
            for idx, row in old_run.iterrows():
                # --- Your existing simulation logic ---
                N_samples = 10000
                scrambler_seed = 551505
                this_v = row['v']
                this_beta = row['beta']
                N, i0, T = int(row['N']), int(row['i0']), int(row['T'])
                print(this_v, v_conf, this_beta, beta_conf)
                orig_samples, plus_samples, minus_samples, trajectories, score_samples = \
                    draw_samples_random_params(N, i0, this_v, this_beta, v_conf, beta_conf, 
                                               T, N_samples, scrambler_seed, cores=15)
                
                beta_grad, beta_grad_sd = grad_wrt_beta(trajectories, score_samples, T, N_samples)
                v_grad, v_grad_sd = grad_wrt_v(plus_samples, minus_samples, N, i0)
                
                # --- Prepare and Write the Result ---
                result_row = {
                    'beta_conf': beta_conf,
                    'v_conf': v_conf,
                    'N': N,
                    'i0': i0,
                    'T': T,
                    'v': this_v,
                    'beta': this_beta,
                    'beta_grad': beta_grad,
                    'beta_grad_se': beta_grad_sd/np.sqrt(N_samples),
                    'v_grad': v_grad,
                    'v_grad_se': v_grad_sd/np.sqrt(N_samples),
                    'total_size_mean': np.mean(orig_samples),
                    'total_size_se': np.std(orig_samples, ddof=1)/np.sqrt(N_samples),
                    'N_samples': N_samples
                }
                
                writer.writerow(result_row)
                
                # IMPORTANT: Push the data from Python's memory to the hard drive
                csvfile.flush()
                os.fsync(csvfile.fileno())
                print("flushed to file")