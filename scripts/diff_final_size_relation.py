#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:44:25 2026

This code implements the semi-analytic method for solving the final-size equations
and also calculates the derivatives with respect to the parameters of interest.

@author: nicholasw
"""

# %%

import numpy as np
import matplotlib.pyplot as plt
import scipy

# %%

beta = 4
v = 0.1
R = beta * (1-v)

# %%
def extinct_prob(R):
    eqn = lambda q: np.exp(R*(q-1)) - q
    q_star = scipy.optimize.fsolve(eqn, 0.0)
    return q_star[0]

def prop_inf(beta,v):
    eqn = lambda tau: (1 - np.exp(-beta * tau))*(1-v) - tau
    tau_star = scipy.optimize.fsolve(eqn, 1)
    return tau_star[0]

def q_derivR(R, q):
    term1 = np.exp(R*(q-1))
    num = -term1*(q-1)
    denom = term1*R - 1
    return num/denom

def tau_derivR(R, tau):
    term1 = np.exp(-R*tau)
    num = tau * term1
    denom = 1-R*term1
    return num/denom

def size_deriv(beta, v, N):
    R = beta*(1-v)
    q = extinct_prob(R)
    tau = prop_inf(R)
    size_est = (1-q)*tau
    dq_dR = q_derivR(R, q)
    dtau_dR = tau_derivR(R, tau)
    dsize_dR = -dq_dR * tau + (1-q) * dtau_dR
    #dsize_dR = dtau_dR
    return np.array([size_est, dsize_dR * -beta, dsize_dR * (1-v)])

# %%

def tau_derivatives(beta, v, tau):
    # Common denominator for both derivatives
    term1 = np.exp(-beta * tau)
    denom = 1 - beta * (1 - v) * term1
    
    # d_tau / d_beta
    dtau_dbeta = ((1 - v) * tau * term1) / denom
    
    # d_tau / d_v
    # Using the simplified version: -(1 - exp(-beta*tau)) / denom
    dtau_dv = -(1 - term1) / denom
    
    return dtau_dbeta, dtau_dv

def size_deriv_corrected(beta, v):
    # R_eff for the branching process (extinction prob)
    Reff = beta * (1 - v)
    
    q = extinct_prob(Reff)
    tau = prop_inf(beta, v)
    
    # Prob of outbreak * Size of outbreak
    size_est = (1 - q) * tau
    
    # Partial derivatives for q (extinction prob)
    dq_dReff = q_derivR(Reff, q)
    # Chain rule for q:
    dq_dbeta = dq_dReff * (1 - v)
    dq_dv = dq_dReff * (-beta)
    
    # Derivatives for tau
    dtau_dbeta, dtau_dv = tau_derivatives(beta, v, tau)
    
    # Total derivatives for size_est = (1-q)*tau
    # Product rule: d/dx [(1-q)*tau] = -dq/dx * tau + (1-q) * dtau/dx
    dsize_dbeta = -dq_dbeta * tau + (1 - q) * dtau_dbeta
    dsize_dv = -dq_dv * tau + (1 - q) * dtau_dv
    
    return np.array([size_est, dsize_dbeta, dsize_dv])

#%%
# Compute on the same grid of values as in big_run_combined.csv
import pandas as pd
df = pd.read_csv('output/big_run_combined.csv')
result = {'beta':[], 
          'v':[], 
          'N':[], 
          'size':[], 
          'dv':[], 
          'db':[]
          }
for index, row in df.iterrows():
    size, db, dv = size_deriv_corrected(row['beta'],row['v'])*row['N']
    result['beta'].append(row['beta'])
    result['v'].append(row['v'])
    result['size'].append(size)
    result['N'].append(int(row['N']))
    result['dv'].append(dv)
    result['db'].append(db)
result = pd.DataFrame(result)
result.to_csv('output/semi_analytic_result.csv', index=None)