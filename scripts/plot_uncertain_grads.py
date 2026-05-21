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
plt.rcParams.update({'font.size': 17})


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
df1 = pd.read_csv("output/data/parametric_uncertainty/parametric_uncertainty_results.csv")
selection1 = df1[(df1['beta_conf'] == 10) & (df1['v_conf'] == 10)]
beta1 = np.unique(selection1['beta'])
v1 = np.unique(selection1['v'])
beta_grad1 = np.abs(np.array(selection1['beta_grad']).reshape(len(v1), len(beta1)).T)

# 2. Load and process second dataset
df2 = pd.read_csv("output/data/big_run_results.csv")
# Note: Ensure selection filter logic for df2 matches your requirements
selection2 = df2 
beta2 = np.unique(selection2['beta'])
v2 = np.unique(selection2['v'])
# Based on your snippet: using 'beta_mean' for the second plot's gradient
beta_grad2 = np.abs(np.array(selection2['beta_mean']).reshape(len(v2), len(beta2)).T)

df3 = pd.read_csv("output/data/semi_analytic_result.csv")
beta3 = np.unique(df3['beta'])
v3 = np.unique(df3['v'])
beta_grad3 = np.abs(np.array(df3['db'])).reshape(len(v3), len(beta3))


# Define critical curves
crit_curve_v1 = 1 - 1/beta1
crit_curve_v2 = 1 - 1/beta2
crit_curve_v3 = 1 - 1/beta3

# 3. Calculate shared color scale
global_min = min([beta_grad1.min(), beta_grad2.min(), beta_grad3.min()])
global_max = max(beta_grad1.max(), beta_grad2.max(), beta_grad3.max())
levels = np.linspace(0, 600, 25) # Ensures identical contours

# 4. Create Side-by-Side Plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))

cmap = 'magma'
# Plot Left: Parametric Uncertainty
cf1 = ax3.contourf(beta1, v1, beta_grad1, levels=levels, cmap=cmap)
ax3.plot(beta1, crit_curve_v1, color="red", linewidth=3, linestyle="--", label = r"$R_{eff} =1$")
title_line1 = r"Sensitivity of final size wrt. $\beta$"
title_line2 = r"Gradient estimator, uncertain parameters"
ax3.set_title(title_line1 + "\n" + title_line2)
ax3.set_xlabel(r"$\beta$ (Mean of input distr.)")
ax3.set_ylabel("v (Mean of input distr.)")
ax3.legend()

# Plot Right: Big Run Combined
cf2 = ax2.contourf(beta2, v2, beta_grad2, levels=levels, cmap=cmap)
ax2.plot(beta2, crit_curve_v2, color="red", linewidth=3, label = r"$R_{eff} =1$", linestyle="--")
title_line1 = r"Sensitivity of final size wrt. $\beta$"
title_line2 = r"Gradient estimator"
ax2.set_title(title_line1 + "\n" + title_line2)
ax2.set_ylabel(r"v (fixed value)")
ax2.set_xlabel(r"$\beta$ (fixed value)")
ax2.legend()

cf3 = ax1.contourf(beta3, v3, beta_grad3, levels=levels, cmap=cmap)
ax1.plot(beta2, crit_curve_v2, color="red", linewidth=3, label = r"$R_{eff} =1$", linestyle="--")
title_line1 = r"Sensitivity of final size wrt. $\beta$"
title_line2 = r"Final size equation"
ax1.set_title(title_line1 + "\n" + title_line2)
ax1.set_ylabel(r"v (fixed value)")
ax1.set_xlabel(r"$\beta$ (fixed value)")
ax1.legend()

# Add a single colorbar for both plots
cbar = fig.colorbar(cf2, ax=[ax1, ax2, ax3], orientation='horizontal', fraction=0.1, pad=-0.35)
cbar.set_label('Magnitude')

plt.tight_layout()
plt.legend()
plt.savefig("output/compare_dbeta_contour.pdf")
plt.show()





# %%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# 1. Load Data
df1 = pd.read_csv("output/data/parametric_uncertainty/parametric_uncertainty_results.csv")
df2 = pd.read_csv("output/data/big_run_results.csv")
df3 = pd.read_csv("output/data/semi_analytic_result.csv")

# 2. Process Dataset 1
selection1 = df1[(df1['beta_conf'] == 10) & (df1['v_conf'] == 10)]
beta1 = np.unique(selection1['beta'])
v1 = np.unique(selection1['v'])
v_grad1 = np.abs(np.array(selection1['v_grad']).reshape(len(v1), len(beta1)).T)

# 3. Process Dataset 2
selection2 = df2 
beta2 = np.unique(selection2['beta'])
v2 = np.unique(selection2['v'])
v_grad2 = np.abs(np.array(selection2['v_mean']).reshape(len(v2), len(beta2)).T)

beta3 = np.unique(df3['beta'])
v3 = np.unique(df3['v'])
v_grad3 = np.abs(np.array(df3['dv'])).reshape(len(v3), len(beta3))

# --- APPLY GAUSSIAN SMOOTHING ---
# Sigma controls the "blur". Start small (e.g., 1.0) to avoid over-smoothing.
v_grad1_smooth = gaussian_filter(v_grad1, sigma=1)
v_grad2_smooth = gaussian_filter(v_grad2, sigma=1)
#v_grad1_smooth = v_grad1
#v_grad2_smooth = v_grad2

# 4. Global Scale & Critical Curves
v_min = min(v_grad1_smooth.min(), v_grad2_smooth.min(), v_grad3.min())
v_max = max(v_grad1_smooth.max(), v_grad2_smooth.max(), v_grad3.max())
print(round(v_max,-1))
levels = np.linspace(0, 1550 , 32)

crit_curve1 = 1 - 1/beta1
crit_curve2 = 1 - 1/beta2

# 5. Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))

cmap = 'viridis'
# Left: Smoothed v_grad
cf1 = ax3.contourf(beta1, v1, v_grad1_smooth, levels=levels, cmap=cmap)
ax3.plot(beta1, crit_curve1, color="red", linewidth=3, linestyle='--', label=r"$R_{eff} = 1$")
title_line1 = r"Sensitivity of final size wrt. $v$"
title_line2 = "Gradient estimator, uncertain parameters"
ax3.set_title(title_line1 + "\n" + title_line2)
ax3.set_xlabel(r"$\beta$ (Mean of input distr.)")
ax3.set_ylabel("v (Mean of input distr.)")
ax3.legend()

# Right: Smoothed v_mean
title_line1 = r"Sensitivity of final size wrt. $v$"
title_line2 = "Gradient estimator"
cf2 = ax2.contourf(beta2, v2, v_grad2_smooth, levels=levels, cmap=cmap)
ax2.plot(beta2, crit_curve2, color="red", linewidth=3, linestyle='--', label=r"$R_{eff} = 1$")
ax2.set_title(title_line1 + "\n" + title_line2)
ax2.set_ylabel(r"v (fixed value)")
ax2.set_xlabel(r"$\beta$ (fixed value)")
ax2.legend()

title_line1 = r"Sensitivity of final size wrt. $v$"
title_line2 = "Final size equation"
cf3 = ax1.contourf(beta3, v3, v_grad3, levels=levels, cmap=cmap)
ax1.plot(beta2, crit_curve2, color="red", linewidth=3, linestyle='--', label=r"$R_{eff} = 1$")
ax1.set_title(title_line1 + "\n" + title_line2)
ax1.set_ylabel(r"v (fixed value)")
ax1.set_xlabel(r"$\beta$ (fixed value)")
ax1.legend()

# Colorbar
cbar = fig.colorbar(cf2, ax=[ax1, ax2, ax3], orientation='horizontal', fraction=0.1, pad=-0.35)
cbar.set_label('Magnitude')

plt.tight_layout()
plt.savefig("output/compare_dv_contourplot.pdf")
plt.legend()
plt.show()