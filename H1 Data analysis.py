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
import matplotlib.cm as cm
import statsmodels.formula.api as smf
import ast 

censure = 1 # Put 0 if include censored participants in analysis and 1 if we exclude them 
by_ind = 0 # Put 0 if no display of individual plots and 1 if display 
attention_type = 'relative' # relative for % of total time and 'absolute' for raw time

path = '/Users/carolinepioger/Desktop/pretest vincent' # change to yours :)

# Get dataframes
data = pd.read_csv(path + '/dataset.csv' )
data_autre = pd.read_csv(path + '/criterion info data.csv')
survey = pd.read_csv(path + '/survey data.csv')

data_for_plot = data

# Remove (or not) participants with censored values in part 2
exclude_participants = data_autre.loc[data_autre['censored_calibration'] == 1, 'id'] 

if censure == 1: 
    data_for_plot = data_for_plot.drop(data_for_plot[data_for_plot['id'].isin(exclude_participants) == True].index)
else: 
    data_for_plot = data_for_plot

# Convert order of cases in string 
for i in range(len(data_for_plot)):
    data_for_plot['order of cases'][i] = ast.literal_eval(data_for_plot['order of cases'][i])
    

# %%
# =============================================================================
# VISUALISE DATA 
# =============================================================================



# Get different cases

ASPS = data_for_plot[(data_for_plot['charity'] == 0) & (data_for_plot['tradeoff'] == 0)]
ACPC = data_for_plot[(data_for_plot['charity'] == 1) & (data_for_plot['tradeoff'] == 0)]
ASPC = data_for_plot[(data_for_plot['charity'] == 1) & (data_for_plot['tradeoff'] == 1)]
ACPS = data_for_plot[(data_for_plot['charity'] == 0) & (data_for_plot['tradeoff'] == 1)]

average_valuation_ASPS = ASPS.groupby('prob_option_A')['valuation'].median()
average_valuation_ACPC = ACPC.groupby('prob_option_A')['valuation'].median()
average_valuation_ACPS = ACPS.groupby('prob_option_A')['valuation'].median()
average_valuation_ASPC = ASPC.groupby('prob_option_A')['valuation'].median()

data_for_plot_2 = data_for_plot
data_for_plot_2['first case'] = [data_for_plot_2['order of cases'][i][0] for i in range(len(data_for_plot_2))]
not_first_case = data_for_plot_2.loc[data_for_plot_2['first case'] != data_for_plot_2['case']] 
data_for_plot_2 = data_for_plot_2.drop(not_first_case.index)

ASPS_between = data_for_plot_2[(data_for_plot_2['charity'] == 0) & (data_for_plot_2['tradeoff'] == 0)]
ACPC_between = data_for_plot_2[(data_for_plot_2['charity'] == 1) & (data_for_plot_2['tradeoff'] == 0)]
ASPC_between = data_for_plot_2[(data_for_plot_2['charity'] == 1) & (data_for_plot_2['tradeoff'] == 1)]
ACPS_between = data_for_plot_2[(data_for_plot_2['charity'] == 0) & (data_for_plot_2['tradeoff'] == 1)]

# Plot No Tradeoff Context (Replication Exley)

plt.plot(average_valuation_ASPS.index, average_valuation_ASPS, label='ASPS', color='blue', marker='o', linestyle='-')
plt.plot(average_valuation_ACPC.index, average_valuation_ACPC, label='ACPC', color='red', marker='o', linestyle='-')


x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)
plt.plot(x_fit, y_fit, color='grey', label='Expected value')

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Valuation as % of Riskless Lottery')
plt.title('(median) Results for No Tradeoff Context')
plt.grid(True)
plt.legend()

plt.show()

# Plot Tradeoff Context (Replication Exley)

plt.plot(average_valuation_ACPS.index, average_valuation_ACPS, label='ACPS', color='blue', marker='o', linestyle='-')
plt.plot(average_valuation_ASPC.index, average_valuation_ASPC, label='ASPC', color='red', marker='o', linestyle='-')


x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)
plt.plot(x_fit, y_fit, color='grey', label='Expected value')

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Valuation as % of Riskless Lottery')
plt.title('(median) Results for Tradeoff Context ')
plt.grid(True)
plt.legend()

plt.show()

# Plot Self Lottery Valuation

plt.plot(average_valuation_ASPS.index, average_valuation_ASPS, label='ASPS', color='green', marker='o', linestyle='-')
plt.plot(average_valuation_ACPS.index, average_valuation_ACPS, label='ACPS', color='orange', marker='o', linestyle='-')


x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)
plt.plot(x_fit, y_fit, color='grey', label='Expected value')

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Valuation as % of Riskless Lottery')
plt.title('(median) Results for Self Lottery Valuation')
plt.grid(True)
plt.legend()

plt.show()

# Plot Charity Lottery Valuation

plt.plot(average_valuation_ASPC.index, average_valuation_ASPC, label='ASPC', color='green', marker='o', linestyle='-')
plt.plot(average_valuation_ACPC.index, average_valuation_ACPC, label='ACPC', color='orange', marker='o', linestyle='-')


x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)
plt.plot(x_fit, y_fit, color='grey', label='Expected value')

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Valuation as % of Riskless Lottery')
plt.title('(median) Results for Charity Lottery Valuation')
plt.grid(True)
plt.legend()

plt.show()

# Plot Valuation for each participant ### TO REMOVE FOR REAL DATA 

if by_ind == 1: 
    for i in range(1, data['number'].nunique()+1):
        ASPS_ind = ASPS.loc[ASPS['number'] == i, ['prob_option_A', 'valuation']] 
        ASPS_ind = ASPS_ind.sort_values(by=['prob_option_A'])
        ACPC_ind = ACPC.loc[ACPC['number'] == i, ['prob_option_A', 'valuation']] 
        ACPC_ind = ACPC_ind.sort_values(by=['prob_option_A'])
        ASPC_ind = ASPC.loc[ASPC['number'] == i, ['prob_option_A', 'valuation']]
        ASPC_ind = ASPC_ind.sort_values(by=['prob_option_A'])
        ACPS_ind = ACPS.loc[ACPS['number'] == i, ['prob_option_A', 'valuation']] 
        ACPS_ind = ACPS_ind.sort_values(by=['prob_option_A'])
                   
        fig, axs = plt.subplots(2, 2)
        fig.suptitle('Individual ' + str(i))
        
        axs[0, 0].plot(ASPS_ind['prob_option_A'], ASPS_ind['valuation'], label='ASPS', color='blue')
        axs[0, 0].plot(ACPC_ind['prob_option_A'], ACPC_ind['valuation'], label='ACPC', color='red')
        axs[0, 0].legend()
        axs[0, 0].plot(x_fit, y_fit, color='grey', label='Expected value')
        axs[0, 0].tick_params(left = False, right = False)
        axs[0, 0].grid(True)
        axs[0, 0].set_title('No Tradeoff Context')
        
        axs[0, 1].plot(ACPS_ind['prob_option_A'], ACPS_ind['valuation'],label='ACPS', color='blue')
        axs[0, 1].plot(ASPC_ind['prob_option_A'], ASPC_ind['valuation'], label='ASPC', color='red')
        axs[0, 1].legend()
        axs[0, 1].plot(x_fit, y_fit, color='grey', label='Expected value')
        axs[0, 1].tick_params(left = False, right = False)
        axs[0, 1].grid(True)
        axs[0, 1].set_title('Tradeoff Context')
        
        axs[1, 0].plot(ASPS_ind['prob_option_A'], ASPS_ind['valuation'], label='ASPS', color='orange')
        axs[1, 0].plot(ACPS_ind['prob_option_A'], ACPS_ind['valuation'], label='ACPS', color='green')
        axs[1, 0].legend()
        axs[1, 0].plot(x_fit, y_fit, color='grey', label='Expected value')
        axs[1, 0].tick_params(left = False, right = False)
        axs[1, 0].grid(True)
        axs[1, 0].set_title('Self Lottery Valuation')
        
        axs[1, 1].plot(ACPC_ind['prob_option_A'], ACPC_ind['valuation'], label='ACPC', color='orange')
        axs[1, 1].plot(ASPC_ind['prob_option_A'], ASPC_ind['valuation'], label='ASPC', color='green')
        axs[1, 1].legend()
        axs[1, 1].plot(x_fit, y_fit, color='grey', label='Expected value')
        axs[1, 1].tick_params(left = False, right = False)
        axs[1, 1].grid(True)
        axs[1, 1].set_title('Charity Lottery Valuation')
        
        for ax in axs.flat:
            ax.label_outer()
        plt.show()
else:
    pass


# %%
# =============================================================================
# ANALYSE DATA 
# =============================================================================

########### Plot ditribution of participant-specific X values 
plt.hist(data_autre['charity_calibration'], bins=20) 
plt.xlabel('Participant-specific X')
plt.ylabel('Frequency')
plt.title('Distribution of X')
plt.show()

######## EXLEY REGRESSION 

data_for_analysis = pd.concat([ASPS, ACPC, ASPC, ACPS], ignore_index=True)

# Add fixed effects
dummy_ind = pd.get_dummies(data_for_analysis['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob = pd.get_dummies(data_for_analysis['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
data_for_analysis = pd.concat([data_for_analysis, dummy_ind, dummy_prob], axis=1)

# Add controls 
data_for_analysis = data_for_analysis.merge(survey, on='id', how='left')
control_variables = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + ['NEP_' + str(i) for i in range(1, 16)] + 
                 ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]

# Create the design matrix and dependent variable
X = data_for_analysis[['charity', 'tradeoff', 'interaction'] + list(dummy_ind.columns) + list(dummy_prob.columns)]
# X = data_for_analysis[['charity', 'tradeoff', 'interaction'] + list(dummy_ind.columns)]
X = pd.concat([X, data_for_analysis[control_variables]], axis=1)
X = sm.add_constant(X, has_constant='add') # add a first column full of ones to account for intercept of regression
y = data_for_analysis['valuation']

# Fit the regression model using Ordinary Least Squares
model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis['number']}) # cluster at individual level
print(model.summary())



# control_variables_2 = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + 
#                  ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]
# X_2 = data_for_analysis[['charity', 'tradeoff', 'interaction'] + list(dummy_ind.columns)]
# X_2 = pd.concat([X_2, data_for_analysis[control_variables_2]], axis=1)
# X_2 = sm.add_constant(X_2, has_constant='add') # add a first column full of ones to account for intercept of regression

# model_X2 = sm.OLS(y, X_2).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis['number']}) # cluster at individual level
# print(model_X2.summary())

md = smf.mixedlm("valuation ~ charity + tradeoff + interaction", data_for_analysis, groups=data_for_analysis["number"])
mdf = md.fit()
print(mdf.summary())    


