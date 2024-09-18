#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 00:00:45 2024

@author: carolinepioger
"""


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
from matplotlib.patches import Patch
import matplotlib.cm as cm
import ast 

from data_processing import (survey, data_principal, data_EDRP, data_censored, data_altruistic,
                                no_tradeoff_lottery_differences_principal, self_lottery_differences_principal, charity_lottery_differences_principal, 
                                no_tradeoff_lottery_differences_censored, self_lottery_differences_censored, charity_lottery_differences_censored,
                                no_tradeoff_lottery_differences_EDRP, self_lottery_differences_EDRP, charity_lottery_differences_EDRP,
                                no_tradeoff_lottery_differences_altruistic, self_lottery_differences_altruistic, charity_lottery_differences_altruistic,
                                no_tradeoff_lottery_differences_EDRP_censored, self_lottery_differences_EDRP_censored, charity_lottery_differences_EDRP_censored,
                                no_tradeoff_lottery_differences_principal_censored, self_lottery_differences_principal_censored, charity_lottery_differences_principal_censored, 
                                valuation_ASPS, valuation_ACPS, valuation_ACPC, valuation_ASPC,
                                ASPS_principal, ACPS_principal, ACPC_principal, ASPC_principal, 
                                mean_valuations, 
                                samplesize_principal, samplesize_adaptive, samplesize_altruistic, samplesize_censored, 
                                samplesize_EDRP_censored, samplesize_principal_censored)


# Get string version of variable name using globals()
def get_variable_name_from_globals(var):
    globals_dict = globals()
    for name, value in globals_dict.items():
        if value is var:
            return name
    return None


# In H3, we are interested to see the correlation between YSPS-YCPS and ASPS-ACPS
# and between YCPC-YSPC and ACPC-ASPC. Since the different self_lottery_differences
# and charity_lottery_differences dataframes use the inverse relation, we create
# new columns for each to give the desired relation for H3.

def lottery_inverse(lottery):
    if get_variable_name_from_globals(lottery).split('_')[0] == 'self':
        var = '_ACPS_ASPS'
        var_inverse = '_ASPS_ACPS'
    elif get_variable_name_from_globals(lottery).split('_')[0] == 'charity':
        var = '_ASPC_ACPC'
        var_inverse = '_ACPC_ASPC'
    
    lottery[f'valuation{var_inverse}_H3'] = (-1)*lottery[f'valuation{var}'] 
    lottery[f'dwell_time{var_inverse}_H3'] = (-1)*lottery[f'dwell_time{var}'] 
    return lottery 

for suffix in ['principal', 'censored', 'EDRP', 'altruistic', 'EDRP_censored', 'principal_censored']:
    self_df_name = f'self_lottery_differences_{suffix}'
    charity_df_name = f'charity_lottery_differences_{suffix}'
    
    lottery_inverse(globals()[self_df_name])
    lottery_inverse(globals()[charity_df_name])

    
    
# %%
# =============================================================================
# ANALYSE CORRELATION DATA BETWEEN ATTENTION AND VALUATION 
# =============================================================================

################################################
# Differentiating between self and charity lottery differences
################################################

# Using linear regression between attention and valuation differences

# Adaptive subjects 
# For self lottery difference
reg_model_self_EDRP = sm.OLS(self_lottery_differences_EDRP['valuation_ASPS_ACPS_H3'], 
                                  sm.add_constant(self_lottery_differences_EDRP['dwell_time_ASPS_ACPS_H3'])).fit()
print(reg_model_self_EDRP.summary())

# For charity lottery difference
reg_model_charity_EDRP = sm.OLS(charity_lottery_differences_EDRP['valuation_ACPC_ASPC_H3'], 
                                     sm.add_constant(charity_lottery_differences_EDRP['dwell_time_ACPC_ASPC_H3'])).fit()
print(reg_model_charity_EDRP.summary())


# Censored subjects
# For self lottery difference
reg_model_self_censored = sm.OLS(self_lottery_differences_censored['valuation_ASPS_ACPS_H3'], 
                                 sm.add_constant(self_lottery_differences_censored['dwell_time_ASPS_ACPS_H3'])).fit()
print(reg_model_self_censored.summary())

# For charity lottery difference
reg_model_charity_censored = sm.OLS(charity_lottery_differences_censored['valuation_ACPC_ASPC_H3'], 
                                    sm.add_constant(charity_lottery_differences_censored['dwell_time_ACPC_ASPC_H3'])).fit()
print(reg_model_charity_censored.summary())


# Adaptive and Censored subjects
# For self lottery difference
reg_model_self_EDRP_censored = sm.OLS(self_lottery_differences_EDRP_censored['valuation_ASPS_ACPS_H3'], 
                                      sm.add_constant(self_lottery_differences_EDRP_censored['dwell_time_ASPS_ACPS_H3'])).fit()
print(reg_model_self_EDRP_censored.summary())

# For charity lottery difference
reg_model_charity_EDRP_censored = sm.OLS(charity_lottery_differences_EDRP_censored['valuation_ACPC_ASPC_H3'], 
                                         sm.add_constant(charity_lottery_differences_EDRP_censored['dwell_time_ACPC_ASPC_H3'])).fit()
print(reg_model_charity_EDRP_censored.summary())


# Principal Analysis and Censored subjects
# For self lottery difference
reg_model_self_principal_censored = sm.OLS(self_lottery_differences_principal_censored['valuation_ASPS_ACPS_H3'], 
                                      sm.add_constant(self_lottery_differences_principal_censored['dwell_time_ASPS_ACPS_H3'])).fit()
print(reg_model_self_principal_censored.summary())

# For charity lottery difference
reg_model_charity_principal_censored = sm.OLS(charity_lottery_differences_principal_censored['valuation_ACPC_ASPC_H3'], 
                                         sm.add_constant(charity_lottery_differences_principal_censored['dwell_time_ACPC_ASPC_H3'])).fit()
print(reg_model_charity_principal_censored.summary())




# %%
# =============================================================================
# VISUALISE CORRELATION DATA BETWEEN ATTENTION AND VALUATION
# =============================================================================

# Define function that assigns a different color for each individual
def color_by_ind(database):
    individuals = database['number'].unique()
    colors = cm.tab20(np.linspace(0, 1, len(individuals)))
    individual_color_map = dict(zip(individuals, colors))
    return database['number'].map(individual_color_map)


# Define function of plots of correlation between Attention and Valuation differences
def plot_corr_attention_valuation(database, pop, samplesize):
    if get_variable_name_from_globals(database).split('_')[0] == 'self':
        x = '_ASPS_ACPS_H3'
    elif get_variable_name_from_globals(database).split('_')[0] == 'charity':
        x = '_ACPC_ASPC_H3'
    
    # scatter plot with each participant having a different color 
    plt.scatter(database[f'dwell_time{x}'], database[f'valuation{x}'],  c=color_by_ind(database))
    
    # adding regression line in red
    coef = np.polyfit(database[f'dwell_time{x}'], database[f'valuation{x}'], 1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(database[f'dwell_time{x}'], poly1d_fn(database[f'dwell_time{x}']), 
                 color='red', linewidth=2, label='Regression Line')

    plt.xlabel('Attention difference in %')
    plt.ylabel('Valuation difference in %')
    plt.title('Attention vs Valuation differences ' 
              + str(get_variable_name_from_globals(database).split('_')[0]) 
              + ' / ' + str(pop) + f' n = {samplesize}')
    plt.legend()
    # plt.grid(True)
    plt.show()
  
# Plot Attention vs Valuation difference for Adaptive subjects
plot_corr_attention_valuation(self_lottery_differences_EDRP, 'Adaptive', samplesize_adaptive)
plot_corr_attention_valuation(charity_lottery_differences_EDRP, 'Adaptive', samplesize_adaptive)

# Plot Attention vs Valuation difference for Censored subjects
plot_corr_attention_valuation(self_lottery_differences_censored, 'Censored', samplesize_censored)
plot_corr_attention_valuation(charity_lottery_differences_censored, 'Censored', samplesize_censored)

# Plot Attention vs Valuation difference for both Adaptive and Censored subjects
plot_corr_attention_valuation(self_lottery_differences_EDRP_censored, 'Adaptive + Censored', samplesize_EDRP_censored)
plot_corr_attention_valuation(charity_lottery_differences_EDRP_censored, 'Adaptive + Censored', samplesize_EDRP_censored)

# Plot Attention vs Valuation difference for both Principal and Censored subjects
plot_corr_attention_valuation(self_lottery_differences_principal_censored, 'Principal + Censored', samplesize_principal_censored)
plot_corr_attention_valuation(charity_lottery_differences_principal_censored, 'Principal + Censored', samplesize_principal_censored)

# We see a general trend that there is a small correlation between attention and
# valuation, which is negative for the self lottery and positive for the charity 
# lottery. We need to verify this statistically



