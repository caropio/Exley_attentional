#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 23:56:31 2024

@author: carolinepioger
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf
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
# ANALYSE VALUATION DATA 
# =============================================================================

################################################
# Verifying H1 through fixed effect regression models from Exley
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
fixed_model_principal = fixed_regression_model(data_principal, 'valuation', ['charity', 'tradeoff', 'interaction', 'case_order'], 'yes')
fixed_model_principal.to_csv('Principal analysis Fixed regression results H1.csv')

# Adaptive subjects
fixed_model_EDRP = fixed_regression_model(data_EDRP, 'valuation', ['charity', 'tradeoff', 'interaction', 'case_order'], 'yes')
fixed_model_EDRP.to_csv('Adaptive Fixed regression results H1.csv')

# Censored subjects
fixed_model_censored = fixed_regression_model(data_censored, 'valuation', ['charity', 'tradeoff', 'interaction', 'case_order'], 'yes')
fixed_model_censored.to_csv('Censored Fixed regression results H1.csv')

# Principal Analysis and Censored subjects (replication of Exley)
data_for_analysis_principal_and_censored = pd.concat([data_principal, data_censored], 
                                                     ignore_index=True) # Data specifically for Principal Analysis and Censored subjects 
fixed_model_principal_and_censored = fixed_regression_model(data_for_analysis_principal_and_censored, 'valuation', ['charity', 'tradeoff', 'interaction', 'case_order'], 'yes')
fixed_model_principal_and_censored.to_csv('Principal analysis and Censored Fixed regression results H1.csv')

# ################################################
# # Heterogeneous effects of probabilities
# ################################################

# # Although not part of H1, we observe heterogeneous effects of probabilities in 
# # the self and charity valuation difference (YCPS-YSPS and YSPC-YCPC respectively)
# # More specifically, in Principal Analysis, we observe that the valuation difference 
# # switches signs for high proba for the self valuation difference and for small 
# # prob for the charity valuation difference (and converges to 0 for Censored subjects)

# # PRINCIPAL ANALYSIS
# # For the no tradeoff difference YCPC-YSPS
# model_no_tradeoff_principal = fixed_regression_model(no_tradeoff_lottery_differences_principal, 'valuation_ACPC_ASPS', [], 'yes')
# model_no_tradeoff_principal.to_csv('No tradeoff principal analysis Fixed regression results.csv')

# # For the self lottery difference YCPS-YSPS
# model_self_principal = fixed_regression_model(self_lottery_differences_principal, 'valuation_ACPS_ASPS', [], 'yes')
# model_self_principal.to_csv('Self principal analysis Fixed regression results.csv')

# # For the charity lottery difference YSPC-YCPC
# model_charity_principal = fixed_regression_model(charity_lottery_differences_principal, 'valuation_ASPC_ACPC', [], 'yes')
# model_charity_principal.to_csv('Charity principal analysis Fixed regression results.csv')


# # CENSORED SUBJECTS
# # For the no tradeoff difference YCPC-YSPS
# model_no_tradeoff_censored = fixed_regression_model(no_tradeoff_lottery_differences_censored, 'valuation_ACPC_ASPS', [], 'yes')
# model_no_tradeoff_censored.to_csv('No tradeoff censored subjects Fixed regression results.csv')

# # For the self lottery difference YCPS-YSPS
# model_self_censored = fixed_regression_model(self_lottery_differences_censored, 'valuation_ACPS_ASPS', [], 'yes')
# model_self_censored.to_csv('Self censored subjects Fixed regression results.csv')

# # For the charity lottery difference YSPC-YCPC
# model_charity_censored = fixed_regression_model(charity_lottery_differences_censored, 'valuation_ASPC_ACPC', [], 'yes')
# model_charity_censored.to_csv('Charity censored subjects Fixed regression results.csv')


# %%
# =============================================================================
# VALUATION DATA VISUALIZATION
# =============================================================================

################################################
# Valuation
################################################

lottery_types = ['$Y^{S}(P^{S})$', '$Y^{C}(P^{S})$', '$Y^{C}(P^{C})$', '$Y^{S}(P^{C})$']

# Scales of X- and Y- axis for valuation plots
x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)

# Plot Valuations in the No Tradeoff Context (Principal Analysis)
plt.figure(figsize=(5, 5))
plt.plot(x_fit, y_fit, color='grey', label='Expected value')
plt.plot(valuation_ASPS.mean().index, valuation_ASPS.mean(), label=lottery_types[0], color='blue', marker='o', linestyle='-')
plt.plot(valuation_ACPC.mean().index, valuation_ACPC.mean(), label=lottery_types[2], color='red', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Amount')
plt.ylabel('Valuation (in %)')
plt.title('Valuations for No Tradeoff Context ')
plt.grid(True)
plt.legend()
plt.savefig('No Tradeoff valuations H1.png', dpi=1200)
plt.show()

# Plot Valuations in the No Tradeoff Context (Replication Exley)
# Add the points (0, 0) and (1, 100) to replicate Exley's figures
plt.figure(figsize=(5, 5))
plt.plot(x_fit, y_fit, color='grey', label='Expected value')
valuation_ASPS_mean = valuation_ASPS.mean() if isinstance(valuation_ASPS, pd.core.groupby.SeriesGroupBy) else valuation_ASPS
valuation_ACPC_mean = valuation_ACPC.mean() if isinstance(valuation_ACPC, pd.core.groupby.SeriesGroupBy) else valuation_ACPC
valuation_ASPS_Exley = pd.concat([valuation_ASPS_mean, pd.Series({0: 0, 1: 100})]).sort_index()
valuation_ACPC_Exley = pd.concat([valuation_ACPC_mean, pd.Series({0: 0, 1: 100})]).sort_index()
plt.plot(valuation_ASPS_Exley.index, valuation_ASPS_Exley, label=lottery_types[0], color='blue', marker='o', linestyle='-')
plt.plot(valuation_ACPC_Exley.index, valuation_ACPC_Exley, label=lottery_types[2], color='red', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Valuation as % of Riskless Lottery')
plt.title('Valuations of No Tradeoff Context ')
plt.grid(True)
plt.legend()
plt.savefig('No Tradeoff valuations EXLEY H1.png', dpi=1200)
plt.show()

# Plot Valuations in the Tradeoff Context (Principal Analysis)
plt.figure(figsize=(5, 5))
plt.plot(x_fit, y_fit, color='grey', label='Expected value')
plt.plot(valuation_ACPS.mean().index, valuation_ACPS.mean(), label=lottery_types[1], color='blue', marker='o', linestyle='-')
plt.plot(valuation_ASPC.mean().index, valuation_ASPC.mean(), label=lottery_types[3], color='red', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Amount')
plt.ylabel('Valuation (in %)')
plt.title('Valuations for Tradeoff Context')
plt.grid(True)
plt.legend()
plt.savefig('Tradeoff valuations H1.png', dpi=1200)
plt.show()

# Plot Valuations in the Tradeoff Context (Replication Exley)
# Add the points (0, 0) and (1, 100) to replicate Exley's figures
plt.figure(figsize=(5, 5))
plt.plot(x_fit, y_fit, color='grey', label='Expected value')
valuation_ACPS_mean = valuation_ACPS.mean() if isinstance(valuation_ACPS, pd.core.groupby.SeriesGroupBy) else valuation_ACPS
valuation_ASPC_mean = valuation_ASPC.mean() if isinstance(valuation_ASPC, pd.core.groupby.SeriesGroupBy) else valuation_ASPC
valuation_ACPS_Exley = pd.concat([valuation_ACPS_mean, pd.Series({0: 0, 1: 100})]).sort_index()
valuation_ASPC_Exley = pd.concat([valuation_ASPC_mean, pd.Series({0: 0, 1: 100})]).sort_index()
plt.plot(valuation_ACPS_Exley.index, valuation_ACPS_Exley, label=lottery_types[1], color='blue', marker='o', linestyle='-')
plt.plot(valuation_ASPC_Exley.index, valuation_ASPC_Exley, label=lottery_types[3], color='red', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Valuation as % of Riskless Lottery')
plt.title('Valuations for Tradeoff Context')
plt.grid(True)
plt.legend()
plt.savefig('Tradeoff valuations EXLEY H1.png', dpi=1200)
plt.show()

# Plot Valuations of the Self Lottery (Principal Analysis)
plt.figure(figsize=(5, 5))
plt.plot(x_fit, y_fit, color='grey', label='Expected value')
plt.plot(valuation_ASPS.mean().index, valuation_ASPS.mean(), label=lottery_types[0], color='green', marker='o', linestyle='-')
plt.plot(valuation_ACPS.mean().index, valuation_ACPS.mean(), label=lottery_types[1], color='orange', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Amount')
plt.ylabel('Valuation (in %)')
plt.title('Valuations for Self Lottery')
plt.grid(True)
plt.legend()
plt.savefig('Self Lottery valuations H1.png', dpi=1200)
plt.show()

# Plot Valuation of the Charity Lottery (Principal Analysis)
plt.figure(figsize=(5, 5))
plt.plot(x_fit, y_fit, color='grey', label='Expected value')
plt.plot(valuation_ASPC.mean().index, valuation_ASPC.mean(), label=lottery_types[3], color='green', marker='o', linestyle='-')
plt.plot(valuation_ACPC.mean().index, valuation_ACPC.mean(), label=lottery_types[2], color='orange', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Amount')
plt.ylabel('Valuation (in %)')
plt.title('Valuations for Charity Lottery')
plt.grid(True)
plt.legend()
plt.savefig('Charity Lottery valuations H1.png', dpi=1200)
plt.show()

# Plot 4 Valuations allthogether for all probabilities (Principal Analysis)
offset = 0.015
plt.plot(x_fit, y_fit, color='grey', label='Expected value')
plt.errorbar(valuation_ASPS.mean().index - offset, valuation_ASPS.mean(), valuation_ASPS.std(), ecolor = 'black', fmt='none', alpha=0.5, label='std')
plt.plot(valuation_ASPS.mean().index - offset, valuation_ASPS.mean(), label=lottery_types[0], color='blue', marker='o', linestyle='-')
plt.errorbar(valuation_ACPS.mean().index - offset/2, valuation_ACPS.mean(), valuation_ACPS.std(), ecolor = 'black', fmt='none', alpha=0.5)
plt.plot(valuation_ACPS.mean().index - offset/2, valuation_ACPS.mean(), label=lottery_types[1], color='dodgerblue', marker='o', linestyle='-')
plt.errorbar(valuation_ACPC.mean().index + offset/2, valuation_ACPC.mean(), valuation_ACPC.std(), ecolor = 'black', fmt='none', alpha=0.5)
plt.plot(valuation_ACPC.mean().index + offset/2, valuation_ACPC.mean(), label=lottery_types[2], color='green', marker='o', linestyle='-')
plt.errorbar(valuation_ASPC.mean().index + offset, valuation_ASPC.mean(), valuation_ASPC.std(), ecolor = 'black', fmt='none', alpha=0.5)
plt.plot(valuation_ASPC.mean().index + offset, valuation_ASPC.mean(), label=lottery_types[3], color='limegreen', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Amount')
plt.ylabel('Valuation (in %)')
plt.title('4 Lottery Valuation for all probabilities')
plt.grid(True)
plt.legend()
plt.savefig('All Lottery valuations plot H1.png', dpi=1200)
plt.show()

# Plot 4 Valuations with probabilities combined (Principal Analysis)
error_valuation = [np.std(ASPS_principal['valuation']), np.std(ACPS_principal['valuation']), 
                  np.std(ACPC_principal['valuation']), np.std(ASPC_principal['valuation'])]
plt.bar(lottery_types, mean_valuations, color = ['blue', 'dodgerblue', 'green', 'limegreen']) 
plt.errorbar(lottery_types, mean_valuations, error_valuation, ecolor = 'black', fmt='none', alpha=0.7, label='std')
plt.xlabel('Case')
plt.ylabel('Valuation (in %)')
plt.title('4 Lottery Valuation with probabilities combined')
plt.legend()
plt.savefig('All Lottery valuations bar H1.png', dpi=1200)
plt.show()


################################################
# Valuation differences
################################################

# Now we are interested in valuation DIFFERENCES, namely YCPC-YSPS, YCPS-YSPS and YSPC-YCPC
# To verify for H1, we check for null, positive and negative differences respectively

lottery_types_difference = ['$Y^{C}(P^{C})-Y^{S}(P^{S})$', 
                            '$Y^{C}(P^{S})-Y^{S}(P^{S})$', 
                            '$Y^{S}(P^{C})-Y^{C}(P^{C})$']
x = np.arange(len(lottery_types_difference))
offset_2 = 0.02

# 3 valuation differences and standard errors at ind level for Principal Analysis, Adaptive and Censored subjects
# for Principal Analysis
principal_means = [no_tradeoff_lottery_differences_principal['valuation_ACPC_ASPS'].mean(),
                   self_lottery_differences_principal['valuation_ACPS_ASPS'].mean(),
                   charity_lottery_differences_principal['valuation_ASPC_ACPC'].mean()]
principal_std_model = fixed_model_principal['Std.Err.'][['charity', 'tradeoff', 'interaction']].to_numpy() # take std of 3 coef from model
principal_errors = [principal_std_model[0], principal_std_model[1], 
                    (principal_std_model[1]+principal_std_model[2])/2]   # the last std is the sum of beta2 and beta3 

# for Adaptive subjects
EDRP_means = [no_tradeoff_lottery_differences_EDRP['valuation_ACPC_ASPS'].mean(), 
              self_lottery_differences_EDRP['valuation_ACPS_ASPS'].mean(),
              charity_lottery_differences_EDRP['valuation_ASPC_ACPC'].mean()]
EDRP_std_model = fixed_model_EDRP['Std.Err.'][['charity', 'tradeoff', 'interaction']].to_numpy() # take std of 3 coef from model
EDRP_errors = [EDRP_std_model[0], EDRP_std_model[1], 
                    (EDRP_std_model[1]+EDRP_std_model[2])/2]   # the last std is the sum of beta2 and beta3 

# for Censored subjects
censored_means = [no_tradeoff_lottery_differences_censored['valuation_ACPC_ASPS'].mean(), 
                  self_lottery_differences_censored['valuation_ACPS_ASPS'].mean(),
                  charity_lottery_differences_censored['valuation_ASPC_ACPC'].mean()]
censored_std_model = fixed_model_censored['Std.Err.'][['charity', 'tradeoff', 'interaction']].to_numpy() # take std of 3 coef from model
censored_errors = [censored_std_model[0], censored_std_model[1], 
                    (censored_std_model[1]+censored_std_model[2])/2]   # the last std is the sum of beta2 and beta3 


# Plot 3 Valuation differences for all probabilities (Principal Analysis)
plt.axhline(y=0, color='grey', linestyle='--')
diff_proba_no_tradeoff = no_tradeoff_lottery_differences_principal.groupby('prob_option_A')['valuation_ACPC_ASPS']
diff_proba_self = self_lottery_differences_principal.groupby('prob_option_A')['valuation_ACPS_ASPS']
diff_proba_charity = charity_lottery_differences_principal.groupby('prob_option_A')['valuation_ASPC_ACPC']
plt.errorbar(diff_proba_no_tradeoff.mean().index - offset_2/2, diff_proba_no_tradeoff.mean(), diff_proba_no_tradeoff.std(), ecolor = 'black', fmt='none', alpha=0.4, label='std')
plt.plot(diff_proba_no_tradeoff.mean().index - offset_2/2, diff_proba_no_tradeoff.mean(), label=lottery_types_difference[0], color='bisque', marker='o', linestyle='-')
plt.errorbar(diff_proba_self.mean().index, diff_proba_self.mean(), diff_proba_self.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_self.mean().index, diff_proba_self.mean(), label=lottery_types_difference[1], color='dodgerblue', marker='o', linestyle='-')
plt.errorbar(diff_proba_charity.mean().index + offset_2/2, diff_proba_charity.mean(), diff_proba_charity.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_charity.mean().index + offset_2/2, diff_proba_charity.mean(), label=lottery_types_difference[2], color='limegreen', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Amount')
plt.ylabel('Valuation difference in %')
plt.title('Valuation differences for Principal Analysis')
plt.legend()
plt.savefig('All Lottery difference plot Principal H1.png', dpi=1200)
plt.show()

# Plot 3 Valuation differences for all probabilities (Censored subjects)
plt.axhline(y=0, color='grey', linestyle='--')
diff_proba_self_censored = self_lottery_differences_censored.groupby('prob_option_A')['valuation_ACPS_ASPS']
diff_proba_charity_censored = charity_lottery_differences_censored.groupby('prob_option_A')['valuation_ASPC_ACPC']
diff_proba_no_tradeoff_censored = no_tradeoff_lottery_differences_censored.groupby('prob_option_A')['valuation_ACPC_ASPS']
plt.errorbar(diff_proba_no_tradeoff_censored.mean().index - offset_2/2, diff_proba_no_tradeoff_censored.mean(), diff_proba_no_tradeoff_censored.std(), ecolor = 'black', fmt='none', alpha=0.4, label='std')
plt.plot(diff_proba_no_tradeoff_censored.mean().index - offset_2/2, diff_proba_no_tradeoff_censored.mean(), label=lottery_types_difference[0], color='bisque', marker='o', linestyle='-')
plt.errorbar(diff_proba_self_censored.mean().index, diff_proba_self_censored.mean(), diff_proba_self_censored.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_self_censored.mean().index, diff_proba_self_censored.mean(), label=lottery_types_difference[1], color='dodgerblue', marker='o', linestyle='-')
plt.errorbar(diff_proba_charity_censored.mean().index + offset_2/2, diff_proba_charity_censored.mean(), diff_proba_charity_censored.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_charity_censored.mean().index + offset_2/2, diff_proba_charity_censored.mean(), label=lottery_types_difference[2], color='limegreen', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Amount')
plt.ylabel('Valuation difference in %')
plt.title('Valuation differences for Censored subjects')
plt.legend()
plt.savefig('All Lottery difference plot Censored H1.png', dpi=1200)
plt.show()

# Plot 3 Valuation differences with probabilities combined (Principal Analysis)
plt.bar(lottery_types_difference, principal_means, color = ['bisque', 'lightskyblue', 'lightgreen']) 
plt.errorbar(lottery_types_difference, principal_means, principal_errors, ecolor = 'black', fmt='none', alpha=0.7, label = 'std ind level')
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery differences')
plt.ylabel('Valuation difference in %')
plt.text(0.15, 0.9, f'n = {samplesize_principal}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
plt.legend()
plt.title('Valuation differences with probabilities combined Principal Analysis')
plt.savefig('All Lottery difference bar H1.png', dpi=1200)
plt.show()

# Plot 3 Valuation differences with probabilities combined (Adaptive Subjects)
plt.bar(lottery_types_difference, EDRP_means, color = ['bisque', 'lightskyblue', 'lightgreen']) 
plt.errorbar(lottery_types_difference, EDRP_means, EDRP_errors, ecolor = 'black', fmt='none', alpha=0.7, label = 'std ind level')
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery differences')
plt.ylabel('Valuation difference in %')
plt.text(0.15, 0.9, f'n = {samplesize_adaptive}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
plt.legend()
plt.title('Valuation differences for Adaptive subjects')
plt.savefig('Lottery differences Adaptive H1.png', dpi=1200)
plt.show()

# Plot 3 Valuation differences with probabilities combined (Censored Subjects)
plt.bar(lottery_types_difference, censored_means, color = ['bisque', 'lightskyblue', 'lightgreen']) 
plt.errorbar(lottery_types_difference, censored_means, censored_errors, ecolor = 'black', fmt='none', alpha=0.7, label = 'std ind level')
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery differences')
plt.ylabel('Valuation difference in %')
plt.text(0.15, 0.9, f'n = {samplesize_censored}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
plt.legend()
plt.title('Valuation differences for Censored subjects')
plt.savefig('Lottery differences Censored H1.png', dpi=1200)
plt.show()

# Plot Valuation differences between Principal Analysis and Censored
width = 0.35
plt.bar(x - width/2, principal_means, width, yerr=principal_errors, capsize=5, color=['bisque', 'lightskyblue', 'lightgreen'], label='Principal analysis')
plt.bar(x + width/2, censored_means, width, yerr=censored_errors, capsize=5, color=['bisque', 'lightskyblue', 'lightgreen'], hatch="//", label='Censored')
plt.xlabel('Lottery type')
plt.ylabel('Difference in valuation in %')
plt.title('Difference in valuation for Principal analysis and Censored subjects H1')
plt.xticks(x, lottery_types_difference)
plt.axhline(y=0, color='grey', linestyle='--')
proxy_artists = [Patch(facecolor='white', edgecolor='black', label=f'Principal analysis n = {samplesize_principal}'),
                 Patch(facecolor='white', edgecolor='black', hatch="//", label=f'Censored n = {samplesize_censored}')]
plt.legend(handles=proxy_artists)
plt.savefig('Merged Valuation Principal Analysis and Censored H1.png', dpi=1200)
plt.show()

# Plot Valuation differences between Adaptive and Censored
plt.bar(x - width/2, EDRP_means, width, yerr=EDRP_errors, capsize=5, color=['bisque', 'lightskyblue', 'lightgreen'], label='Adaptive')
plt.bar(x + width/2, censored_means, width, yerr=censored_errors, capsize=5, color=['bisque', 'lightskyblue', 'lightgreen'], hatch="//", label='Censored')
plt.xlabel('Lottery type')
plt.ylabel('Difference in valuation in %')
plt.title('Difference in valuation for Adaptive and Censored subjects H1')
plt.xticks(x, lottery_types_difference)
plt.axhline(y=0, color='grey', linestyle='--')
proxy_artists = [Patch(facecolor='white', edgecolor='black', label=f'Adaptive n = {samplesize_adaptive}'),
                 Patch(facecolor='white', edgecolor='black', hatch="//", label=f'Censored n = {samplesize_censored}')]
plt.legend(handles=proxy_artists)
plt.savefig('Merged Valuation Adaptive and Censored H1.png', dpi=1200)
plt.show()


# Histogram of the self and charity valuation differences of Principal Analysis
plt.hist([self_lottery_differences_principal['valuation_ACPS_ASPS'], charity_lottery_differences_principal['valuation_ASPC_ACPC']], 
        bins = 20, color = ['lightskyblue', 'lightgreen'], label = lottery_types_difference[1:3]) 
plt.xlabel('Difference in lottery valuation (trad - no trad)')
plt.ylabel('Frequency')
plt.title('Self and charity valuation differences across probabilities')
plt.legend()
plt.savefig('Histo Valuation difference Principal H1.png', dpi=1200)
plt.show()

# Histogram of the self and charity valuation differences of Adaptive subjects
plt.hist([self_lottery_differences_EDRP['valuation_ACPS_ASPS'], charity_lottery_differences_EDRP['valuation_ASPC_ACPC']], 
        bins = 20, color = ['lightskyblue', 'lightgreen'], label = lottery_types_difference[1:3]) 
plt.xlabel('Difference in lottery valuation (trad - no trad)')
plt.ylabel('Frequency')
plt.title('Self and charity valuation differences across probabilities')
plt.legend()
plt.savefig('Histo Valuation difference Adaptive H1.png', dpi=1200)
plt.show()
 

################################################
# Valuation according to case order and probability
################################################

# Check the effect of order in which case are presented on valuation of lotteries

cases = ['first', 'second', 'third', 'fourth']

first_case = data_principal[data_principal['case_order']==1] # valuation from the first case presented
second_case = data_principal[data_principal['case_order']==2] # valuation from the second case presented
third_case = data_principal[data_principal['case_order']==3] # valuation from the third case presented
fourth_case = data_principal[data_principal['case_order']==4] # valuation from the fourth case presented

case_order  = [first_case['valuation'].mean(), second_case['valuation'].mean(), 
               third_case['valuation'].mean(), fourth_case['valuation'].mean()]
case_order_std = [first_case['valuation'].std(), second_case['valuation'].std(), 
                  third_case['valuation'].std(), fourth_case['valuation'].std()]

plt.bar(cases, case_order, color = ['dimgray', 'darkgray', 'silver', 'lightgrey']) 
plt.errorbar(cases, case_order, case_order_std, ecolor = 'black', fmt='none', alpha=0.7, label = 'std')
plt.xlabel('Case order')
plt.ylabel('Valuation (in %)')
plt.title('Effect of case order on valuation (all cases combined)')
plt.savefig('Valuation case order H1.png', dpi=1200)
plt.show()

# Effect of case order on attention using mixed effects model
model_case_order = smf.mixedlm("valuation ~ case_order", data_principal, groups=data_principal["number"]).fit()
print(model_case_order.summary())

# We find that individuals generally value less lotteries in the following case
# which suggest we should control for case order in analysis 

# Check the effect of probability on valuation of lotteries 

# We group valuations according to probabilities (regardless of case and order)
valuation_per_proba = data_principal.groupby('prob_option_A')['valuation'] 
probabilities = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

plt.plot(x_fit, y_fit, color='grey', label='Expected value')
plt.plot(probabilities, valuation_per_proba.mean(), color='black', marker='o', linestyle='-')
plt.errorbar(probabilities, valuation_per_proba.mean(), valuation_per_proba.std(), ecolor = 'black', fmt='none', alpha=0.7, label = 'std')
plt.xlabel('Probability')
plt.ylabel('Valuation (in %)')
plt.title('Effect of probability on valuation (all cases combined)')
plt.savefig('Valuation probability H1.png', dpi=1200)
plt.show()

# Effect of probability on valuation using mixed effects model
model_proba = smf.mixedlm("valuation ~ prob_option_A", data_principal, groups=data_principal["number"]).fit()
print(model_proba.summary())

# We indeed find the standard empirical finding in risky decision-making that 
# the valuation is superior to expected value for small probabilities and inferior
# for high probabilities meaning that individuals are generally risk seeking for 
# small probabilities and more risk averse for high probabilities

# Importantly, we need to compare the valuation of lotteries with the same probability


# %%
# =============================================================================
# DIFFERENCES OF MAGNITUDES
# =============================================================================

# We study the magnitudes of the self, charity and no tradeoff lottery differences
# Thus we compare the absolute values of YCPS-YSPS, YSPC-YCPC and YCPC-YSPS

################################################
# Principal analysis
################################################

t_statistic_diff, p_value_diff = ttest_ind(self_lottery_differences_principal['valuation_ACPS_ASPS'].abs(), charity_lottery_differences_principal['valuation_ASPC_ACPC'].abs())
print()
print('PRINCIPAL ANALYSIS')
print('Difference of magnitude between self and charity valuation difference for principal analysis (t-test, p value):')
print(t_statistic_diff, p_value_diff)

################################################
# Censored subjects
################################################

t_statistic_diff_censored, p_value_diff_censored = ttest_ind(self_lottery_differences_censored['valuation_ACPS_ASPS'].abs(), charity_lottery_differences_censored['valuation_ASPC_ACPC'].abs())
print()
print('CENSORED SUBJECTS')
print('Difference of magnitude between self and charity valuation difference for censored subjects (t-test, p value):')
print(t_statistic_diff_censored, p_value_diff_censored)
print()

################################################
# BETWEEN Principal analysis and Censored subjects 
################################################

print('BETWEEN Principal analysis and Censored subjects ')

t_statistic_no_tradeoff, p_value_no_tradeoff = ttest_ind(no_tradeoff_lottery_differences_principal['valuation_ACPC_ASPS'], no_tradeoff_lottery_differences_censored['valuation_ACPC_ASPS'])
print('Difference of magnitude of No tradeoff difference between Principal analysis and censored (t-test, p value)')
print(t_statistic_no_tradeoff, p_value_no_tradeoff)
print()

t_statistic_self, p_value_self = ttest_ind(self_lottery_differences_principal['valuation_ACPS_ASPS'], self_lottery_differences_censored['valuation_ACPS_ASPS'])
print('Difference of magnitude of Self difference between Principal analysis and censored (t-test, p value)')
print(t_statistic_self, p_value_self)
print()

t_statistic_charity, p_value_charity = ttest_ind(charity_lottery_differences_principal['valuation_ASPC_ACPC'], charity_lottery_differences_censored['valuation_ASPC_ACPC'])
print('Difference of magnitude of Charity difference between Principal analysis and censored (t-test, p value)')
print(t_statistic_charity, p_value_charity)
print()


################################################
# BETWEEN Adaptive and Censored subjects 
################################################

print('BETWEEN Adaptive and Censored subjects ')

t_statistic_no_tradeoff_EDRP_censored, p_value_no_tradeoff_EDRP_censored = ttest_ind(no_tradeoff_lottery_differences_EDRP['valuation_ACPC_ASPS'], 
                                                                                     no_tradeoff_lottery_differences_censored['valuation_ACPC_ASPS'])
print('Difference of magnitude of No Tradeoff difference between Adaptive and censored (t-test, p value)')
print(t_statistic_no_tradeoff_EDRP_censored, p_value_no_tradeoff_EDRP_censored)
print()

t_statistic_self_EDRP_censored, p_value_self_EDRP_censored = ttest_ind(self_lottery_differences_EDRP['valuation_ACPS_ASPS'], self_lottery_differences_censored['valuation_ACPS_ASPS'])
print('Difference of magnitudeof Self difference between Adaptive and censored (t-test, p value)')
print(t_statistic_self_EDRP_censored, p_value_self_EDRP_censored)
print()

t_statistic_charity_EDRP_censored, p_value_charity_EDRP_censored = ttest_ind(charity_lottery_differences_EDRP['valuation_ASPC_ACPC'], charity_lottery_differences_censored['valuation_ASPC_ACPC'])
print('Difference of magnitude of Charity difference between Adaptive and censored (t-test, p value)')
print(t_statistic_charity_EDRP_censored, p_value_charity_EDRP_censored)
print()


# %%
# =============================================================================
# Simulation with sample size of Exley and Garcia
# =============================================================================

# iteration_number = 100 # Number of iterations of simulation per sample size
# sample_Exley = 57 # Exley's sample size is 57
# sample_Garcia =107 #  Garcia et al's sample size is 107

# def simulation_power_charity_coef(sample, iteration):
#     p_values = np.zeros(iteration) # variable to collect p-values for each iteration
#     for inter in range(1, iteration): # run simulation for a set number of iterations
#         # pick random subjects from our sample ("sample" number) - drawn with replacement (same subject can be drawn multiple times)    
#         subjects_drawn = np.random.choice(np.unique(data_principal['number']), sample) 
#         data_drawn = []
#         for subj in subjects_drawn:
#             # extract data from these randomly picjed subjects
#             subj_data = data_principal.loc[data_principal['number'] == subj, ['number', 'prob_option_A', 'valuation', 'charity', 'tradeoff', 'interaction']]
#             data_drawn.append(subj_data)
#         data_drawn = pd.concat(data_drawn)
        
#         try:
#             # to replicate exactly Exley's regression (not taking into account the order of case)
#             test = fixed_regression_model(data_drawn, 'valuation', ['charity', 'tradeoff', 'interaction'], 'no') 
#             coef_charity = test['P>|z|']['charity'] # extract for each sample size tested the p-value of the charity variable
#             p_values[inter] = coef_charity # collect this p-value for each iteration
            
#         except np.linalg.LinAlgError:
#             print()
#             print("Singular matrix encountered.")
#             print()
#             p_values[inter] = 1
#         except ZeroDivisionError:
#             print()
#             print("Multicollinearity encountered.")
#             print()
#             p_values[inter] = 1  
            
#     power_calculated = np.mean(p_values < 0.05) # we find the power by average significance level over iterations
#     return p_values, power_calculated


# # Power using Exley's sample size and our data

# p_val_Exley, power_Exley = simulation_power_charity_coef(sample_Exley, iteration_number)

# print()
# print()
# print('Using Exley sample size, the charity coefficient is significant for ' 
#       + str(power_Exley*100) + '% of iterations')
# print()

# # Power using Garcia's sample size and our data

# p_val_Garcia, power_Garcia = simulation_power_charity_coef(sample_Garcia, iteration_number)

# print()
# print()
# print('Using Garcia et al sample size, the charity coefficient is significant for ' 
#       + str(power_Exley*100) + '% of iterations')
# print()

# # Power using our sample size and data

# p_val_us, power_us = simulation_power_charity_coef(183, iteration_number)

# print()
# print()
# print('Using our sample size, the charity coefficient is significant for ' 
#       + str(power_us*100) + '% of iterations')
# print()


