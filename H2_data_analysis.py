#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 23:58:19 2024

@author: carolinepioger
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
from matplotlib.patches import Patch
import ast 

from data_processing import (survey, data_principal, data_EDRP, data_censored, data_altruistic,
                                no_tradeoff_lottery_differences_principal, self_lottery_differences_principal, charity_lottery_differences_principal, 
                                no_tradeoff_lottery_differences_censored, self_lottery_differences_censored, charity_lottery_differences_censored,
                                no_tradeoff_lottery_differences_EDRP, self_lottery_differences_EDRP, charity_lottery_differences_EDRP,
                                no_tradeoff_lottery_differences_altruistic, self_lottery_differences_altruistic, charity_lottery_differences_altruistic,
                                no_tradeoff_lottery_differences_EDRP_censored, self_lottery_differences_EDRP_censored, charity_lottery_differences_EDRP_censored,
                                valuation_ASPS, valuation_ACPS, valuation_ACPC, valuation_ASPC,
                                ASPS_principal, ACPS_principal, ACPC_principal, ASPC_principal, 
                                mean_valuations, 
                                samplesize_principal, samplesize_adaptive, samplesize_altruistic, samplesize_censored, 
                                samplesize_EDRP_censored, samplesize_principal_censored)


# %%
# =============================================================================
# ANALYSE ATTENTION DATA 
# =============================================================================

################################################
# Verifying H2 through fixed effect regression models inspired by Exley
################################################

# The regression model is taken from Exley (2015) whilst additionally taking
# into account the case order

def fixed_regression_model(data, dependent_var, independent_var, want_print):
    # Add fixed effects of individuals and probabilities 
    database = data
    dummy_prob = pd.get_dummies(database['prob_option_A'], drop_first=True, dtype=int) # Create dummy variable for probabilities (+drop first to avoid multicollinearity)
    dummy_ind = pd.get_dummies(database['number'], drop_first=True, dtype=int)  # Create dummy variable for individuals (+drop first to avoid multicollinearity)
    database = pd.concat([database, dummy_ind, dummy_prob], axis=1)
    
    # Add controls (information of survey)
    database = database.merge(survey, on='number', how='left')
    control_variables = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + ['NEP_' + str(i) for i in range(1, 16)] + 
                     ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]
    
    # Create the design matrix and dependent variable
    X = database[independent_var + list(dummy_prob.columns) + list(dummy_ind.columns)]
    X = pd.concat([X, database[control_variables]], axis=1)
    X = sm.add_constant(X, has_constant='add') # add a first column full of ones to account for intercept of regression
    y = database[dependent_var]

    # Fit the regression model using Ordinary Least Squares
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': database['number']}) # cluster at individual level
    summary = model.summary2().tables[1]
    if want_print == 'yes':
        print(summary)
    elif want_print == 'no':
        pass
    return summary 


# Principal Analysis
fixed_model_principal_attention = fixed_regression_model(data_principal, 'dwell_time_relative', ['charity', 'tradeoff', 'interaction', 'case_order'], 'yes')
fixed_model_principal_attention.to_csv('Principal analysis Fixed regression results Attention H2.csv')

# Adaptive subjects
fixed_model_EDRP_attention = fixed_regression_model(data_EDRP, 'dwell_time_relative', ['charity', 'tradeoff', 'interaction', 'case_order'], 'yes')
fixed_model_EDRP_attention.to_csv('Adaptive Fixed regression results H2.csv')

# Altruistic subjects
fixed_model_altruistic_attention = fixed_regression_model(data_altruistic, 'dwell_time_relative', ['charity', 'tradeoff', 'interaction', 'case_order'], 'yes')
fixed_model_altruistic_attention.to_csv('Altruistic Fixed regression results H2.csv')

# Censored subjects
fixed_model_censored_attention = fixed_regression_model(data_censored, 'dwell_time_relative', ['charity', 'tradeoff', 'interaction', 'case_order'], 'yes')
fixed_model_censored_attention.to_csv('Censored Fixed regression results H2.csv')

# Principal analysis and censored subjects
data_for_analysis_principal_and_censored = pd.concat([data_principal, data_censored], 
                                                     ignore_index=True) # Data specifically for Principal Analysis and Censored subjects 
fixed_model_principal_censored_attention = fixed_regression_model(data_for_analysis_principal_and_censored, 'dwell_time_relative', ['charity', 'tradeoff', 'interaction', 'case_order'], 'yes')
fixed_model_principal_censored_attention.to_csv('Principal analysis and Censored Fixed regression results H2.csv')


# Adaptive and censored subjects
data_for_analysis_EDRP_and_censored = pd.concat([data_EDRP, data_censored], 
                                                     ignore_index=True) # Data specifically for Adaptive and Censored subjects 
fixed_model_EDRP_censored_attention = fixed_regression_model(data_for_analysis_EDRP_and_censored, 'dwell_time_relative', ['charity', 'tradeoff', 'interaction', 'case_order'], 'yes')
fixed_model_EDRP_censored_attention.to_csv('Adaptive and Censored Fixed regression results H2.csv')




# %%
# =============================================================================
# ATTENTION DATA VISUALIZATION
# =============================================================================


################################################
# Attention 
################################################

lottery_types_attention = ['$A^{S}(P^{S})$', '$A^{C}(P^{S})$', '$A^{C}(P^{C})$', '$A^{S}(P^{C})$']
offset = 0.02

# # Plot 4 Valuations allthogether for all probabilities (Principal Analysis)
# plt.errorbar(attention_ASPS.mean().index - offset, attention_ASPS.mean(), attention_ASPS.std(), ecolor = 'black', fmt='none', alpha=0.5, label='std')
# plt.plot(attention_ASPS.mean().index - offset, attention_ASPS.mean(), label=lottery_types_attention[0], color='blue', marker='o', linestyle='-')
# plt.errorbar(attention_ACPS.mean().index - offset/2, attention_ACPS.mean(), attention_ACPS.std(), ecolor = 'black', fmt='none', alpha=0.5)
# plt.plot(attention_ACPS.mean().index - offset/2, attention_ACPS.mean(), label=lottery_types_attention[1], color='dodgerblue', marker='o', linestyle='-')
# plt.errorbar(attention_ACPC.mean().index + offset/2, attention_ACPC.mean(), attention_ACPC.std(), ecolor = 'black', fmt='none', alpha=0.5)
# plt.plot(attention_ACPC.mean().index + offset/2, attention_ACPC.mean(), label=lottery_types_attention[2], color='green', marker='o', linestyle='-')
# plt.errorbar(attention_ASPC.mean().index + offset, attention_ASPC.mean(), attention_ASPC.std(), ecolor = 'black', fmt='none', alpha=0.5)
# plt.plot(attention_ASPC.mean().index + offset, attention_ASPC.mean(), label=lottery_types_attention[3], color='limegreen', marker='o', linestyle='-')
# plt.xlabel('Probability P of Non-Zero Amount')
# plt.ylabel('Attention (in %)')
# plt.title('4 Attention for all probabilities (Principal Analysis)')
# plt.grid(True)
# plt.legend()
# plt.savefig('All Lottery attentions plot H2.png', dpi=1200)
# plt.show()


# # Plot 4 Valuations allthogether for all probabilities (Adaptive subjects)
# attention_ASPS_EDRP = ASPS_EDRP.groupby('prob_option_A')['dwell_time_relative']
# attention_ACPS_EDRP = ACPS_EDRP.groupby('prob_option_A')['dwell_time_relative']
# attention_ACPC_EDRP = ACPC_EDRP.groupby('prob_option_A')['dwell_time_relative']
# attention_ASPC_EDRP = ASPC_EDRP.groupby('prob_option_A')['dwell_time_relative']

# plt.errorbar(attention_ASPS_EDRP.mean().index - offset, attention_ASPS_EDRP.mean(), attention_ASPS_EDRP.std(), ecolor = 'black', fmt='none', alpha=0.5, label='std')
# plt.plot(attention_ASPS_EDRP.mean().index - offset, attention_ASPS_EDRP.mean(), label=lottery_types_attention[0], color='blue', marker='o', linestyle='-')
# plt.errorbar(attention_ACPS_EDRP.mean().index - offset/2, attention_ACPS_EDRP.mean(), attention_ACPS_EDRP.std(), ecolor = 'black', fmt='none', alpha=0.5)
# plt.plot(attention_ACPS_EDRP.mean().index - offset/2, attention_ACPS_EDRP.mean(), label=lottery_types_attention[1], color='dodgerblue', marker='o', linestyle='-')
# plt.errorbar(attention_ACPC_EDRP.mean().index + offset/2, attention_ACPC_EDRP.mean(), attention_ACPC_EDRP.std(), ecolor = 'black', fmt='none', alpha=0.5)
# plt.plot(attention_ACPC_EDRP.mean().index + offset/2, attention_ACPC_EDRP.mean(), label=lottery_types_attention[2], color='green', marker='o', linestyle='-')
# plt.errorbar(attention_ASPC_EDRP.mean().index + offset, attention_ASPC_EDRP.mean(), attention_ASPC_EDRP.std(), ecolor = 'black', fmt='none', alpha=0.5)
# plt.plot(attention_ASPC_EDRP.mean().index + offset, attention_ASPC_EDRP.mean(), label=lottery_types_attention[3], color='limegreen', marker='o', linestyle='-')
# plt.xlabel('Probability P of Non-Zero Amount')
# plt.ylabel('Attention (in %)')
# plt.title('4 Attention for all probabilities (Adaptive subjects)')
# plt.grid(True)
# plt.legend()
# plt.savefig('All Lottery attentions plot Adaptive H2.png', dpi=1200)
# plt.show()

# Plot 4 Valuations allthogether for all probabilities (Censored subjects)
# attention_ASPS_censored = ASPS_censored.groupby('prob_option_A')['dwell_time_relative']
# attention_ACPS_censored = ACPS_censored.groupby('prob_option_A')['dwell_time_relative']
# attention_ACPC_censored = ACPC_censored.groupby('prob_option_A')['dwell_time_relative']
# attention_ASPC_censored = ASPC_censored.groupby('prob_option_A')['dwell_time_relative']

# plt.errorbar(attention_ASPS_censored.mean().index - offset, attention_ASPS_censored.mean(), attention_ASPS_censored.std(), ecolor = 'black', fmt='none', alpha=0.5, label='std')
# plt.plot(attention_ASPS_censored.mean().index - offset, attention_ASPS_censored.mean(), label=lottery_types_attention[0], color='blue', marker='o', linestyle='-')
# plt.errorbar(attention_ACPS_censored.mean().index - offset/2, attention_ACPS_censored.mean(), attention_ACPS_censored.std(), ecolor = 'black', fmt='none', alpha=0.5)
# plt.plot(attention_ACPS_censored.mean().index - offset/2, attention_ACPS_censored.mean(), label=lottery_types_attention[1], color='dodgerblue', marker='o', linestyle='-')
# plt.errorbar(attention_ACPC_censored.mean().index + offset/2, attention_ACPC_censored.mean(), attention_ACPC_censored.std(), ecolor = 'black', fmt='none', alpha=0.5)
# plt.plot(attention_ACPC_censored.mean().index + offset/2, attention_ACPC_censored.mean(), label=lottery_types_attention[2], color='green', marker='o', linestyle='-')
# plt.errorbar(attention_ASPC_censored.mean().index + offset, attention_ASPC_censored.mean(), attention_ASPC_censored.std(), ecolor = 'black', fmt='none', alpha=0.5)
# plt.plot(attention_ASPC_censored.mean().index + offset, attention_ASPC_censored.mean(), label=lottery_types_attention[3], color='limegreen', marker='o', linestyle='-')
# plt.xlabel('Probability P of Non-Zero Amount')
# plt.ylabel('Attention (in %)')
# plt.title('4 Attention for all probabilities (Censored subjects)')
# plt.grid(True)
# plt.legend()
# plt.savefig('All Lottery attentions plot Censored H2.png', dpi=1200)
# plt.show()


################################################
# Attention difference
################################################

lottery_types_difference_attention = ['$A^{C}(P^{C})-A^{S}(P^{S})$', 
                                      '$A^{C}(P^{S})-A^{S}(P^{S})$', 
                                      '$A^{S}(P^{C})-A^{C}(P^{C})$']
x = np.arange(len(lottery_types_difference_attention))

# Now we are interested in attention difference, namely ACPS-ASPS and ASPC-ACPC
# To verify for H2, we check for negative differences 


# 3 attention differences and standard errors at ind level for Principal Analysis, Adaptive, Altruistic and Censored subjects
# for Principal Analysis
principal_means_att = [no_tradeoff_lottery_differences_principal['dwell_time_ACPC_ASPS'].mean(), 
                   self_lottery_differences_principal['dwell_time_ACPS_ASPS'].mean(),
                   charity_lottery_differences_principal['dwell_time_ASPC_ACPC'].mean()]
principal_std_model_att = fixed_model_principal_attention['Std.Err.'][['charity', 'tradeoff', 'interaction']].to_numpy() # take std of 3 coef from model
principal_errors_att = [principal_std_model_att[0], principal_std_model_att[1], 
                    (principal_std_model_att[1] + principal_std_model_att[2])/2]   # the last std is the sum of beta2 and beta3 


# for Adaptive subjects
EDRP_means_att = [no_tradeoff_lottery_differences_EDRP['dwell_time_ACPC_ASPS'].mean(), 
              self_lottery_differences_EDRP['dwell_time_ACPS_ASPS'].mean(),
              charity_lottery_differences_EDRP['dwell_time_ASPC_ACPC'].mean()]
EDRP_std_model_att = fixed_model_EDRP_attention['Std.Err.'][['charity', 'tradeoff', 'interaction']].to_numpy() # take std of 3 coef from model
EDRP_errors_att = [EDRP_std_model_att[0], EDRP_std_model_att[1], 
                    (EDRP_std_model_att[1] + EDRP_std_model_att[2])/2]   # the last std is the sum of beta2 and beta3 

# for Altruistic subjects
altruistic_means_att = [no_tradeoff_lottery_differences_altruistic['dwell_time_ACPC_ASPS'].mean(), 
              self_lottery_differences_altruistic['dwell_time_ACPS_ASPS'].mean(),
              charity_lottery_differences_altruistic['dwell_time_ASPC_ACPC'].mean()]
altruistic_std_model_att = fixed_model_altruistic_attention['Std.Err.'][['charity', 'tradeoff', 'interaction']].to_numpy() # take std of 3 coef from model
altruistic_errors_att = [altruistic_std_model_att[0], altruistic_std_model_att[1], 
                    (altruistic_std_model_att[1] + altruistic_std_model_att[2])/2]   # the last std is the sum of beta2 and beta3 

# for Censored subjects
censored_means_att = [no_tradeoff_lottery_differences_censored['dwell_time_ACPC_ASPS'].mean(), 
                  self_lottery_differences_censored['dwell_time_ACPS_ASPS'].mean(),
                  charity_lottery_differences_censored['dwell_time_ASPC_ACPC'].mean()]
censored_std_model_att = fixed_model_censored_attention['Std.Err.'][['charity', 'tradeoff', 'interaction']].to_numpy() # take std of 3 coef from model
censored_errors_att = [censored_std_model_att[0], censored_std_model_att[1], 
                    (censored_std_model_att[1] + censored_std_model_att[2])/2]   # the last std is the sum of beta2 and beta3 


# for Adaptive and Censored subjects
EDRP_censored_means_att = [no_tradeoff_lottery_differences_EDRP_censored['dwell_time_ACPC_ASPS'].mean(), 
                  self_lottery_differences_EDRP_censored['dwell_time_ACPS_ASPS'].mean(),
                  charity_lottery_differences_EDRP_censored['dwell_time_ASPC_ACPC'].mean()]
EDRP_censored_std_model_att = fixed_model_EDRP_censored_attention['Std.Err.'][['charity', 'tradeoff', 'interaction']].to_numpy() # take std of 3 coef from model
EDRP_censored_errors_att = [EDRP_censored_std_model_att[0], EDRP_censored_std_model_att[1], 
                    (EDRP_censored_std_model_att[1] + EDRP_censored_std_model_att[2])/2]   # the last std is the sum of beta2 and beta3 


# Plot 3 Attention differences for all probabilities (Principal Analysis)
plt.axhline(y=0, color='grey', linestyle='--')
diff_proba_no_tradeoff_attention = no_tradeoff_lottery_differences_principal.groupby('prob_option_A')['dwell_time_ACPC_ASPS']
diff_proba_self_attention = self_lottery_differences_principal.groupby('prob_option_A')['dwell_time_ACPS_ASPS']
diff_proba_charity_attention = charity_lottery_differences_principal.groupby('prob_option_A')['dwell_time_ASPC_ACPC']
plt.errorbar(diff_proba_no_tradeoff_attention.mean().index - offset/2, diff_proba_no_tradeoff_attention.mean(), diff_proba_no_tradeoff_attention.std(), ecolor = 'black', fmt='none', alpha=0.4, label='std')
plt.plot(diff_proba_no_tradeoff_attention.mean().index - offset/2, diff_proba_no_tradeoff_attention.mean(), label=lottery_types_difference_attention[0], color='bisque', marker='o', linestyle='-')
plt.errorbar(diff_proba_self_attention.mean().index, diff_proba_self_attention.mean(), diff_proba_self_attention.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_self_attention.mean().index, diff_proba_self_attention.mean(), label=lottery_types_difference_attention[1], color='dodgerblue', marker='o', linestyle='-')
plt.errorbar(diff_proba_charity_attention.mean().index + offset/2, diff_proba_charity_attention.mean(), diff_proba_charity_attention.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_charity_attention.mean().index + offset/2, diff_proba_charity_attention.mean(), label=lottery_types_difference_attention[2], color='limegreen', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Amount')
plt.ylabel('Attention difference in %')
plt.title('Attention differences for Principal Analysis')
plt.legend()
plt.savefig('All Lottery difference plot Principal H2.png', dpi=1200)
plt.show()

# Plot 3 Attention differences for all probabilities (Adaptive subjects)
plt.axhline(y=0, color='grey', linestyle='--')
diff_proba_no_tradeoff_attention_EDRP = no_tradeoff_lottery_differences_EDRP.groupby('prob_option_A')['dwell_time_ACPC_ASPS']
diff_proba_self_attention_EDRP = self_lottery_differences_EDRP.groupby('prob_option_A')['dwell_time_ACPS_ASPS']
diff_proba_charity_attention_EDRP = charity_lottery_differences_EDRP.groupby('prob_option_A')['dwell_time_ASPC_ACPC']
plt.errorbar(diff_proba_no_tradeoff_attention_EDRP.mean().index - offset/2, diff_proba_no_tradeoff_attention_EDRP.mean(), diff_proba_no_tradeoff_attention_EDRP.std(), ecolor = 'black', fmt='none', alpha=0.4, label='std')
plt.plot(diff_proba_no_tradeoff_attention_EDRP.mean().index - offset/2, diff_proba_no_tradeoff_attention_EDRP.mean(), label=lottery_types_difference_attention[0], color='bisque', marker='o', linestyle='-')
plt.errorbar(diff_proba_self_attention_EDRP.mean().index, diff_proba_self_attention_EDRP.mean(), diff_proba_self_attention_EDRP.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_self_attention_EDRP.mean().index, diff_proba_self_attention_EDRP.mean(), label=lottery_types_difference_attention[1], color='dodgerblue', marker='o', linestyle='-')
plt.errorbar(diff_proba_charity_attention_EDRP.mean().index + offset/2, diff_proba_charity_attention_EDRP.mean(), diff_proba_charity_attention_EDRP.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_charity_attention_EDRP.mean().index + offset/2, diff_proba_charity_attention_EDRP.mean(), label=lottery_types_difference_attention[2], color='limegreen', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Amount')
plt.ylabel('Attention difference in %')
plt.title('Attention differences for Adaptive subjects')
plt.legend()
plt.savefig('All Lottery difference plot Adaptive H2.png', dpi=1200)
plt.show()

# Plot 3 Attention differences for all probabilities (Altruistic subjects)
plt.axhline(y=0, color='grey', linestyle='--')
diff_proba_no_tradeoff_attention_altruistic = no_tradeoff_lottery_differences_altruistic.groupby('prob_option_A')['dwell_time_ACPC_ASPS']
diff_proba_self_attention_altruistic = self_lottery_differences_altruistic.groupby('prob_option_A')['dwell_time_ACPS_ASPS']
diff_proba_charity_attention_altruistic = charity_lottery_differences_altruistic.groupby('prob_option_A')['dwell_time_ASPC_ACPC']
plt.errorbar(diff_proba_no_tradeoff_attention_altruistic.mean().index - offset/2, diff_proba_no_tradeoff_attention_altruistic.mean(), diff_proba_no_tradeoff_attention_altruistic.std(), ecolor = 'black', fmt='none', alpha=0.4, label='std')
plt.plot(diff_proba_no_tradeoff_attention_altruistic.mean().index - offset/2, diff_proba_no_tradeoff_attention_altruistic.mean(), label=lottery_types_difference_attention[0], color='bisque', marker='o', linestyle='-')
plt.errorbar(diff_proba_self_attention_altruistic.mean().index, diff_proba_self_attention_altruistic.mean(), diff_proba_self_attention_altruistic.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_self_attention_altruistic.mean().index, diff_proba_self_attention_altruistic.mean(), label=lottery_types_difference_attention[1], color='dodgerblue', marker='o', linestyle='-')
plt.errorbar(diff_proba_charity_attention_altruistic.mean().index + offset/2, diff_proba_charity_attention_altruistic.mean(), diff_proba_charity_attention_altruistic.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_charity_attention_altruistic.mean().index + offset/2, diff_proba_charity_attention_altruistic.mean(), label=lottery_types_difference_attention[2], color='limegreen', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Amount')
plt.ylabel('Attention difference in %')
plt.title('Attention differences for Altruistic subjects')
plt.legend()
plt.savefig('All Lottery difference plot Altruistic H2.png', dpi=1200)
plt.show()

# Plot 3 Attention differences for all probabilities (Censored subjects)
plt.axhline(y=0, color='grey', linestyle='--')
diff_proba_no_tradeoff_attention_censored = no_tradeoff_lottery_differences_censored.groupby('prob_option_A')['dwell_time_ACPC_ASPS']
diff_proba_self_attention_censored = self_lottery_differences_censored.groupby('prob_option_A')['dwell_time_ACPS_ASPS']
diff_proba_charity_attention_censored = charity_lottery_differences_censored.groupby('prob_option_A')['dwell_time_ASPC_ACPC']
plt.errorbar(diff_proba_no_tradeoff_attention_censored.mean().index - offset/2, diff_proba_no_tradeoff_attention_censored.mean(), diff_proba_no_tradeoff_attention_censored.std(), ecolor = 'black', fmt='none', alpha=0.4, label='std')
plt.plot(diff_proba_no_tradeoff_attention_censored.mean().index - offset/2, diff_proba_no_tradeoff_attention_censored.mean(), label=lottery_types_difference_attention[0], color='bisque', marker='o', linestyle='-')
plt.errorbar(diff_proba_self_attention_censored.mean().index, diff_proba_self_attention_censored.mean(), diff_proba_self_attention_censored.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_self_attention_censored.mean().index, diff_proba_self_attention_censored.mean(), label=lottery_types_difference_attention[1], color='dodgerblue', marker='o', linestyle='-')
plt.errorbar(diff_proba_charity_attention_censored.mean().index + offset/2, diff_proba_charity_attention_censored.mean(), diff_proba_charity_attention_censored.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_charity_attention_censored.mean().index + offset/2, diff_proba_charity_attention_censored.mean(), label=lottery_types_difference_attention[2], color='limegreen', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Amount')
plt.ylabel('Attention difference in %')
plt.title('Attention differences for Censored subjects')
plt.legend()
plt.savefig('All Lottery difference plot Censored H2.png', dpi=1200)
plt.show()

# Plot 3 Attention differences for all probabilities (Adaptive and Censored subjects)
plt.axhline(y=0, color='grey', linestyle='--')
diff_proba_no_tradeoff_attention_EDRP_censored = no_tradeoff_lottery_differences_EDRP_censored.groupby('prob_option_A')['dwell_time_ACPC_ASPS']
diff_proba_self_attention_EDRP_censored = self_lottery_differences_EDRP_censored.groupby('prob_option_A')['dwell_time_ACPS_ASPS']
diff_proba_charity_attention_EDRP_censored = charity_lottery_differences_EDRP_censored.groupby('prob_option_A')['dwell_time_ASPC_ACPC']
plt.errorbar(diff_proba_no_tradeoff_attention_EDRP_censored.mean().index - offset/2, diff_proba_no_tradeoff_attention_EDRP_censored.mean(), diff_proba_no_tradeoff_attention_EDRP_censored.std(), ecolor = 'black', fmt='none', alpha=0.4, label='std')
plt.plot(diff_proba_no_tradeoff_attention_EDRP_censored.mean().index - offset/2, diff_proba_no_tradeoff_attention_EDRP_censored.mean(), label=lottery_types_difference_attention[0], color='bisque', marker='o', linestyle='-')
plt.errorbar(diff_proba_self_attention_EDRP_censored.mean().index, diff_proba_self_attention_EDRP_censored.mean(), diff_proba_self_attention_EDRP_censored.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_self_attention_EDRP_censored.mean().index, diff_proba_self_attention_EDRP_censored.mean(), label=lottery_types_difference_attention[1], color='dodgerblue', marker='o', linestyle='-')
plt.errorbar(diff_proba_charity_attention_EDRP_censored.mean().index + offset/2, diff_proba_charity_attention_EDRP_censored.mean(), diff_proba_charity_attention_EDRP_censored.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_charity_attention_EDRP_censored.mean().index + offset/2, diff_proba_charity_attention_EDRP_censored.mean(), label=lottery_types_difference_attention[2], color='limegreen', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Amount')
plt.ylabel('Attention difference in %')
plt.title('Attention differences for Adaptive and Censored subjects')
plt.legend()
plt.savefig('All Lottery difference plot Adaptive and Censored H2.png', dpi=1200)
plt.show()


# Plot 3 Attention differences with probabilities combined (Principal Analysis)
plt.bar(lottery_types_difference_attention, principal_means_att, color = ['bisque', 'lightskyblue', 'lightgreen']) 
plt.errorbar(lottery_types_difference_attention, principal_means_att, principal_errors_att, ecolor = 'black', fmt='none', alpha=0.7, label = 'std ind level')
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery differences')
plt.ylabel('Attention difference in %')
plt.text(0.15, 0.9, f'n = {samplesize_principal}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
plt.legend()
plt.title('Attention differences with probabilities combined Principal Analysis')
plt.savefig('All Lottery difference bar H2.png', dpi=1200)
plt.show()

# Plot 3 Attention differences with probabilities combined (Adaptive subjects)
plt.bar(lottery_types_difference_attention, EDRP_means_att, color = ['bisque', 'lightskyblue', 'lightgreen']) 
plt.errorbar(lottery_types_difference_attention, EDRP_means_att, EDRP_errors_att, ecolor = 'black', fmt='none', alpha=0.7, label = 'std ind level')
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery differences')
plt.ylabel('Attention difference in %')
plt.text(0.15, 0.9, f'n = {samplesize_adaptive}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
plt.legend()
plt.title('Attention differences for Adaptive subjects')
plt.savefig('Lottery differences Adaptive H2.png', dpi=1200)
plt.show()

# Plot 3 Attention differences with probabilities combined (Altruistic subjects)
plt.bar(lottery_types_difference_attention, altruistic_means_att, color = ['bisque', 'lightskyblue', 'lightgreen']) 
plt.errorbar(lottery_types_difference_attention, altruistic_means_att, altruistic_errors_att, ecolor = 'black', fmt='none', alpha=0.7, label = 'std ind level')
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery differences')
plt.ylabel('Attention difference in %')
plt.text(0.15, 0.9, f'n = {samplesize_altruistic}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
plt.legend()
plt.title('Attention differences for Altruistic subjects')
plt.savefig('Lottery differences Altruistic H2.png', dpi=1200)
plt.show()

# Plot 3 Attention differences with probabilities combined (Censored subjects)
plt.bar(lottery_types_difference_attention, censored_means_att, color = ['bisque', 'lightskyblue', 'lightgreen']) 
plt.errorbar(lottery_types_difference_attention, censored_means_att, censored_errors_att, ecolor = 'black', fmt='none', alpha=0.7, label = 'std ind level')
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery differences')
plt.ylabel('Attention difference in %')
plt.text(0.85, 0.15, f'n = {samplesize_censored}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
plt.legend()
plt.title('Attention differences for Censored subjects')
plt.savefig('Lottery differences Censored H2.png', dpi=1200)
plt.show()

# Plot 3 Attention differences with probabilities combined (Adaptive and Censored subjects)
plt.bar(lottery_types_difference_attention, EDRP_censored_means_att, color = ['bisque', 'lightskyblue', 'lightgreen']) 
plt.errorbar(lottery_types_difference_attention, EDRP_censored_means_att, EDRP_censored_errors_att, ecolor = 'black', fmt='none', alpha=0.7, label = 'std ind level')
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery differences')
plt.ylabel('Attention difference in %')
plt.text(0.85, 0.15, f'n = {samplesize_EDRP_censored}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
plt.legend()
plt.title('Attention differences for Adaptive and Censored subjects')
plt.savefig('Lottery differences Adaptive and Censored H2.png', dpi=1200)
plt.show()

# Plot Valuation differences between Adaptive and Censored subjects
width = 0.35
plt.bar(x - width/2, EDRP_means_att, width, yerr=EDRP_errors_att, capsize=5, color=['bisque', 'lightskyblue', 'lightgreen'], label='Principal analysis')
plt.bar(x + width/2, censored_means_att, width, yerr=censored_errors_att, capsize=5, color=['bisque', 'lightskyblue', 'lightgreen'], hatch="//", label='Censored')
plt.xlabel('Lottery type')
plt.ylabel('Difference in attention in %')
plt.title('Difference in attention for Adaptive and Censored subjects H1')
plt.xticks(x, lottery_types_difference_attention)
plt.axhline(y=0, color='grey', linestyle='--')
proxy_artists = [Patch(facecolor='white', edgecolor='black', label=f'Adaptive n = {samplesize_adaptive}'),
                 Patch(facecolor='white', edgecolor='black', hatch="//", label=f'Censored n = {samplesize_censored}')]
plt.legend(handles=proxy_artists)
plt.savefig('Merged Attention Adaptive and Censored H2.png', dpi=1200)
plt.show()


################################################
# Attention according to case order and probability
################################################

# Check the effect of order in which case are presented on attention of lotteries

cases = ['first', 'second', 'third', 'fourth']

first_case = data_principal[data_principal['case_order']==1] # attention from the first case presented
second_case = data_principal[data_principal['case_order']==2] # attention from the second case presented
third_case = data_principal[data_principal['case_order']==3] # attention from the third case presented
fourth_case = data_principal[data_principal['case_order']==4] # attention from the fourth case presented

case_order_attention  = [first_case['dwell_time_relative'].mean(), second_case['dwell_time_relative'].mean(), 
               third_case['dwell_time_relative'].mean(), fourth_case['dwell_time_relative'].mean()]
case_order_attention_std = [first_case['dwell_time_relative'].std(), second_case['dwell_time_relative'].std(), 
                  third_case['dwell_time_relative'].std(), fourth_case['dwell_time_relative'].std()]

plt.bar(cases, case_order_attention, color = ['dimgray', 'darkgray', 'silver', 'lightgrey']) 
plt.errorbar(cases, case_order_attention, case_order_attention_std, ecolor = 'black', fmt='none', alpha=0.7, label = 'std')
plt.xlabel('Case order')
plt.ylabel('Attention (in %)')
plt.title('Effect of case order on attention (all cases combined)')
plt.savefig('Attention case order H2.png', dpi=1200)
plt.show()

# Effect of case order on attention using mixed effects model
model_case_order_att = smf.mixedlm("dwell_time_relative ~ case_order", data_principal, groups=data_principal["number"]).fit()
print(model_case_order_att.summary())


# We find that individuals generally attend less lotteries in the following case
# which suggest we should control for case order in analysis 

# Check the effect of probability on attention of lotteries 

# We group attention according to probabilities (regardless of case and order)
attention_per_proba = data_principal.groupby('prob_option_A')['dwell_time_relative']
probabilities = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

plt.plot(probabilities, attention_per_proba.mean(), color='black', marker='o', linestyle='-')
plt.errorbar(probabilities, attention_per_proba.mean(), attention_per_proba.std(), ecolor = 'black', fmt='none', alpha=0.7, label = 'std')
plt.xlabel('Probability')
plt.ylabel('Attention (in %)')
plt.title('Effect of probability on attention (all cases combined)')
plt.savefig('Attention probability H2.png', dpi=1200)
plt.show()

# Effect of probability on attention using mixed effects model
model_proba_att = smf.mixedlm("dwell_time_relative ~ prob_option_A", data_principal, groups=data_principal["number"]).fit()
print(model_proba_att.summary())

# Importantly, we find that less ambiguous representations of probabilities (high
# and low probabilities for which the proportion of green balls is more easily 
# processed) that attention is smaller beause it takes less time to process. There 
# is surprisingly a sharp drop at 0.5 although it should be the highest one intuitively

# Importantly, we need to compare the valuation of lotteries with the same probability



# %%
# =============================================================================
# DIFFERENCES OF MAGNITUDES
# =============================================================================

# We study the magnitudes of the self, charity and no tradeoff lottery differences
# Thus we compare the absolute values of ACPS-ASPS, ASPC-ACPC and ACPC-ASPS

################################################
# Principal analysis
################################################

t_statistic_att_diff, p_value_att_diff = ttest_ind(self_lottery_differences_principal['dwell_time_ACPS_ASPS'].abs(), charity_lottery_differences_principal['dwell_time_ASPC_ACPC'].abs())
print()
print('PRINCIPAL ANALYSIS')
print('Difference of magnitude between self and charity attention difference for principal analysis (t-test, p value):')
print(t_statistic_att_diff, p_value_att_diff)

################################################
# Censored subjects
################################################

t_statistic_att_diff_censored, p_value_att_diff_censored = ttest_ind(self_lottery_differences_censored['dwell_time_ACPS_ASPS'].abs(), charity_lottery_differences_censored['dwell_time_ASPC_ACPC'].abs())
print()
print('CENSORED SUBJECTS')
print('Difference of magnitude between self and charity attention difference for censored subjects (t-test, p value):')
print(t_statistic_att_diff_censored, p_value_att_diff_censored)
print()


################################################
# BETWEEN Adaptive and Censored subjects 
################################################

print('BETWEEN Adaptive and Censored subjects ')

t_statistic_no_tradeoff_att_EDRP_censored, p_value_no_tradeoff_att_EDRP_censored = ttest_ind(no_tradeoff_lottery_differences_EDRP['dwell_time_ACPC_ASPS'], 
                                                                                     no_tradeoff_lottery_differences_censored['dwell_time_ACPC_ASPS'])
print('Difference of magnitude of No Tradeoff Attention difference between Adaptive and censored (t-test, p value)')
print(t_statistic_no_tradeoff_att_EDRP_censored, p_value_no_tradeoff_att_EDRP_censored)
print()

t_statistic_self_att_EDRP_censored, p_value_self_att_EDRP_censored = ttest_ind(self_lottery_differences_EDRP['dwell_time_ACPS_ASPS'], 
                                                                               self_lottery_differences_censored['dwell_time_ACPS_ASPS'])
print('Difference of magnitudeof Self Attention difference between Adaptive and censored (t-test, p value)')
print(t_statistic_self_att_EDRP_censored, p_value_self_att_EDRP_censored)
print()

t_statistic_charity_att_EDRP_censored, p_value_charity_att_EDRP_censored = ttest_ind(charity_lottery_differences_EDRP['dwell_time_ASPC_ACPC'], 
                                                                                     charity_lottery_differences_censored['dwell_time_ASPC_ACPC'])
print('Difference of magnitude of Charity Attention difference between Adaptive and censored (t-test, p value)')
print(t_statistic_charity_att_EDRP_censored, p_value_charity_att_EDRP_censored)
print()



