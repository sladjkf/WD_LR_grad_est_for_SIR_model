#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 17:25:20 2026

@author: nick
"""

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})


# %%
df = pd.read_csv("parametric_uncertainty_results.csv")
selection = df[(df['beta_conf'] == 10) & (df['v_conf']== 10)]
beta = np.unique(selection['beta'])
v = np.unique(selection['v'])
beta_grad = np.array(selection['beta_grad']).reshape(len(v),len(beta))
v_grad = np.array(selection['v_grad']).reshape(len(v),len(beta))
total_size = np.array(selection['total_size_mean']).reshape(len(v),len(beta))
crit_curve_v = 1 - 1/beta
plt.contourf(beta, v, np.abs(beta_grad))
plt.plot(beta, crit_curve_v,color="red")
plt.colorbar()

# %%

df = pd.read_csv("output/big_run_combined.csv")
selection = df
beta = np.unique(selection['beta'])
v = np.unique(selection['v'])
beta_grad = np.array(selection['beta_mean']).reshape(len(v),len(beta))
v_grad = np.array(selection['v_mean']).reshape(len(v),len(beta))
#total_size = np.array(selection['total_size_mean']).reshape(len(v),len(beta))
plt.contourf(beta, v, np.abs(beta_grad))
plt.plot(beta, crit_curve_v,color="red")
plt.colorbar()

# %%

####
# Plot the comparison of gradients with input uncertainty
# vs gradients with fixed inputs
####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and process first dataset
df1 = pd.read_csv("parametric_uncertainty_results.csv")
selection1 = df1[(df1['beta_conf'] == 10) & (df1['v_conf'] == 10)]
beta1 = np.unique(selection1['beta'])
v1 = np.unique(selection1['v'])
beta_grad1 = np.abs(np.array(selection1['beta_grad']).reshape(len(v1), len(beta1)))

# 2. Load and process second dataset
df2 = pd.read_csv("output/big_run_combined.csv")
# Note: Ensure selection filter logic for df2 matches your requirements
selection2 = df2 
beta2 = np.unique(selection2['beta'])
v2 = np.unique(selection2['v'])
# Based on your snippet: using 'beta_mean' for the second plot's gradient
beta_grad2 = np.abs(np.array(selection2['beta_mean']).reshape(len(v2), len(beta2)))

# Define critical curves
crit_curve_v1 = 1 - 1/beta1
crit_curve_v2 = 1 - 1/beta2

# 3. Calculate shared color scale
global_min = min(beta_grad1.min(), beta_grad2.min())
global_max = max(beta_grad1.max(), beta_grad2.max())
levels = np.linspace(0, 600, 25) # Ensures identical contours

# 4. Create Side-by-Side Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# Plot Left: Parametric Uncertainty
cf1 = ax1.contourf(beta1, v1, beta_grad1, levels=levels, cmap="magma")
ax1.plot(beta1, crit_curve_v1, color="green", linewidth=3, linestyle="--")
title_line1 = r"$E_{v,\beta}\left[ \frac{\partial}{\partial \beta} E[Z(v,\beta)] \right]$ vs. v, $\beta$"
title_line2 = r"(Sensitivity wrt. $\beta$ under uncertainty in $\beta, v)$"
ax1.set_title(title_line1 + "\n" + title_line2)
ax1.set_xlabel(r"$\beta$ (Mean of input distr.)")
ax1.set_ylabel("v (Mean of input distr.)")

# Plot Right: Big Run Combined
cf2 = ax2.contourf(beta2, v2, beta_grad2, levels=levels, cmap="magma")
ax2.plot(beta2, crit_curve_v2, color="green", linewidth=3, label=r"$R_0 = 1$", linestyle="--")
title_line1 = r"$\frac{\partial}{\partial \beta} E[Z(v,\beta)]$ vs. v, $\beta$"
title_line2 = r"(Sensitivity wrt. $\beta$ for fixed and known $\beta, v)$"
ax2.set_title(title_line1 + "\n" + title_line2)
ax2.set_ylabel(r"v (fixed value)")
ax2.set_xlabel(r"$\beta$ (fixed value)")

# Add a single colorbar for both plots
cbar = fig.colorbar(cf2, ax=[ax1, ax2], orientation='horizontal', fraction=0.1, pad=-0.35)
cbar.set_label('Magnitude')

plt.tight_layout()
plt.savefig("output/compare_beta_beta_conf_10_v_conf_10.pdf")
plt.legend()
plt.show()


# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and process first dataset (Parametric Uncertainty)
df1 = pd.read_csv("parametric_uncertainty_results.csv")
selection1 = df1[(df1['beta_conf'] == 10) & (df1['v_conf'] == 10)]
beta1 = np.unique(selection1['beta'])
v1 = np.unique(selection1['v'])
# Use v_grad for the first plot
v_grad1 = np.abs(np.array(selection1['v_grad']).reshape(len(v1), len(beta1)))

# 2. Load and process second dataset (Big Run)
df2 = pd.read_csv("output/big_run_combined.csv")
selection2 = df2 
beta2 = np.unique(selection2['beta'])
v2 = np.unique(selection2['v'])
# Use v_mean for the second plot (following your variable mapping)
v_grad2 = np.abs(np.array(selection2['v_mean']).reshape(len(v2), len(beta2)))

# Critical curve: v = 1 - 1/beta
crit_curve_v1 = 1 - 1/beta1
crit_curve_v2 = 1 - 1/beta2

# 3. Calculate shared color scale for v_grad
v_min = min(v_grad1.min(), v_grad2.min())
v_max = max(v_grad1.max(), v_grad2.max())
levels = np.linspace(v_min, v_max, 20)

# 4. Create Side-by-Side Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Left Plot: v_grad from results
cf1 = ax1.contourf(beta1, v1, v_grad1, levels=levels)
ax1.plot(beta1, crit_curve_v1, color="red", linewidth=2, label="Critical Curve")
ax1.set_title("Parametric Uncertainty (|v_grad|)")
ax1.set_xlabel("beta")
ax1.set_ylabel("v")

# Right Plot: v_mean from big run
cf2 = ax2.contourf(beta2, v2, v_grad2, levels=levels)
ax2.plot(beta2, crit_curve_v2, color="red", linewidth=2)
ax2.set_title("Big Run Combined (|v_mean|)")
ax2.set_xlabel("beta")

# Add shared colorbar
cbar = fig.colorbar(cf2, ax=[ax1, ax2], orientation='vertical', fraction=0.54, pad=.3)
cbar.set_label('v Gradient Magnitude')

plt.tight_layout()
plt.savefig("v_grad_comparison_plot.png")
plt.show()

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and process first dataset
df1 = pd.read_csv("parametric_uncertainty_results.csv")
selection1 = df1[(df1['beta_conf'] == 1000) & (df1['v_conf'] == 1000)]
beta1 = np.unique(selection1['beta'])
v1 = np.unique(selection1['v'])
size1 = np.array(selection1['total_size_mean']).reshape(len(v1), len(beta1))

# 2. Load and process second dataset
df2 = pd.read_csv("output/big_run_combined.csv")
# Note: Ensure 'total_size_mean' exists in this CSV
selection2 = df2 
beta2 = np.unique(selection2['beta'])
v2 = np.unique(selection2['v'])
size2 = np.array(selection2['total_size_mean']).reshape(len(v2), len(beta2))

# 3. Calculate shared color scale for Total Size
s_min = min(size1.min(), size2.min())
s_max = max(size1.max(), size2.max())
levels = np.linspace(s_min, s_max, 25) # Slightly more levels for smoother size transitions

# 4. Create Side-by-Side Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot Left: Parametric Uncertainty
cf1 = ax1.contourf(beta1, v1, size1, levels=levels, cmap='viridis')
ax1.plot(beta1, 1 - 1/beta1, color="red", linewidth=2, label="Critical Curve")
ax1.set_title("Parametric Uncertainty (Total Size)")
ax1.set_xlabel("beta")
ax1.set_ylabel("v")

# Plot Right: Big Run Combined
cf2 = ax2.contourf(beta2, v2, size2, levels=levels, cmap='viridis')
ax2.plot(beta2, 1 - 1/beta2, color="red", linewidth=2)
ax2.set_title("Big Run Combined (Total Size)")
ax2.set_xlabel("beta")

# Add a single colorbar
cbar = fig.colorbar(cf2, ax=[ax1, ax2], orientation='vertical', fraction=0.03, pad=0.04)
cbar.set_label('Total Size Mean')

plt.tight_layout()
plt.show()

# %%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# 1. Load Data
df1 = pd.read_csv("parametric_uncertainty_results.csv")
df2 = pd.read_csv("output/big_run_combined.csv")

# 2. Process Dataset 1
selection1 = df1[(df1['beta_conf'] == 10) & (df1['v_conf'] == 10)]
beta1 = np.unique(selection1['beta'])
v1 = np.unique(selection1['v'])
v_grad1 = np.abs(np.array(selection1['v_grad']).reshape(len(v1), len(beta1)))

# 3. Process Dataset 2
selection2 = df2 
beta2 = np.unique(selection2['beta'])
v2 = np.unique(selection2['v'])
v_grad2 = np.abs(np.array(selection2['v_mean']).reshape(len(v2), len(beta2)))

# --- APPLY GAUSSIAN SMOOTHING ---
# Sigma controls the "blur". Start small (e.g., 1.0) to avoid over-smoothing.
v_grad1_smooth = gaussian_filter(v_grad1, sigma=1)
v_grad2_smooth = gaussian_filter(v_grad2, sigma=1)
#v_grad1_smooth = v_grad1
#v_grad2_smooth = v_grad2

# 4. Global Scale & Critical Curves
v_min = min(v_grad1_smooth.min(), v_grad2_smooth.min())
v_max = max(v_grad1_smooth.max(), v_grad2_smooth.max())
levels = np.linspace(0, 1500, 31)

crit_curve1 = 1 - 1/beta1
crit_curve2 = 1 - 1/beta2

# 5. Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# Left: Smoothed v_grad
cf1 = ax1.contourf(beta1, v1, v_grad1_smooth, levels=levels, cmap='magma')
ax1.plot(beta1, crit_curve1, color="green", linewidth=3, linestyle='--')
title_line1 = r"$E_{v,\beta}\left[ \frac{\partial}{\partial v} E[Z(v,\beta)] \right]$ vs. v, $\beta$"
title_line2 = r"(Sensitivity wrt. $v$ under uncertainty in $\beta, v)$"
ax1.set_title(title_line1 + "\n" + title_line2)
ax1.set_xlabel(r"$\beta$ (Mean of input distr.)")
ax1.set_ylabel("v (Mean of input distr.)")

# Right: Smoothed v_mean
cf2 = ax2.contourf(beta2, v2, v_grad2_smooth, levels=levels, cmap='magma')
ax2.plot(beta2, crit_curve2, color="green", linewidth=3, linestyle='--', label=r"$R_0 = 1$")
title_line1 = r"$\frac{\partial}{\partial v} E[Z(v,\beta)]$ vs. v, $\beta$"
title_line2 = r"(Sensitivity wrt. $v$ under uncertainty in $\beta, v)$"
ax2.set_title(title_line1 + "\n" + title_line2)
ax2.set_ylabel(r"v (fixed value)")
ax2.set_xlabel(r"$\beta$ (fixed value)")

# Colorbar
cbar = fig.colorbar(cf2, ax=[ax1, ax2], orientation='horizontal', fraction=0.1, pad=-0.35)
cbar.set_label('Magnitude')

plt.tight_layout()
plt.savefig("output/compare_v_beta_conf_10_v_conf_10_smoothing_sigma_1.pdf")
plt.legend()
plt.show()