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
attention_type = 'absolute' # relative for % of total time and 'absolute' for raw time

path = '/Users/carolinepioger/Desktop/pretest vincent' # change to yours :)

# Get dataframes
data = pd.read_csv(path + '/dataset.csv' )
data_autre = pd.read_csv(path + '/criterion info data.csv')
survey = pd.read_csv(path + '/survey data.csv')

data_for_plot = data

# Remove (or not) participants with censored values in part 2
exclude_participants = data_for_plot.loc[data_for_plot['censored_calibration'] == 1, 'id'] 

if censure == 1: 
    data_for_plot = data_for_plot.drop(data_for_plot[data_for_plot['id'].isin(exclude_participants) == True].index)
    data_for_plot = data_for_plot.reset_index(drop=True)
else: 
    data_for_plot = data_for_plot

# Convert order of cases in string 
for i in range(len(data_for_plot)):
    data_for_plot['order of cases'][i] = ast.literal_eval(data_for_plot['order of cases'][i])
    
# Get different cases

ASPS = data_for_plot[(data_for_plot['charity'] == 0) & (data_for_plot['tradeoff'] == 0)]
ACPC = data_for_plot[(data_for_plot['charity'] == 1) & (data_for_plot['tradeoff'] == 0)]
ASPC = data_for_plot[(data_for_plot['charity'] == 1) & (data_for_plot['tradeoff'] == 1)]
ACPS = data_for_plot[(data_for_plot['charity'] == 0) & (data_for_plot['tradeoff'] == 1)]

valuation_ASPS = ASPS.groupby('prob_option_A')['valuation']
valuation_ACPS = ACPS.groupby('prob_option_A')['valuation']
valuation_ACPC = ACPC.groupby('prob_option_A')['valuation']
valuation_ASPC = ASPC.groupby('prob_option_A')['valuation']

valuations_all = [valuation_ASPS, valuation_ACPS, valuation_ACPC, valuation_ASPC]

median_valuation_ASPS = valuation_ASPS.median()
median_valuation_ACPS = valuation_ACPS.median()
median_valuation_ACPC = valuation_ACPC.median()
median_valuation_ASPC = valuation_ASPC.median()


# Difference data
self_lottery = pd.concat([ASPS, ACPS], ignore_index = True)
charity_lottery = pd.concat([ACPC, ASPC], ignore_index=True)

self_lottery_differences = pd.DataFrame(columns=['number', 'prob_option_A'])

for i in self_lottery['number'].unique():
    individual = self_lottery.loc[self_lottery['number'] == i, ['case', 'prob_option_A', 'valuation']] 
    individual_difference = individual.pivot(index='prob_option_A', columns='case', values='valuation')
    individual_difference['valuation_ACPS_ASPS'] = individual_difference['ACPS'] - individual_difference['ASPS']
    individual_difference['number'] = i
    individual_difference.reset_index(inplace=True)
    # individual_difference.columns = individual_difference.columns.droplevel(1)
    self_lottery_differences = pd.concat([self_lottery_differences, individual_difference[['number', 'prob_option_A', 'valuation_ACPS_ASPS']]], ignore_index=True)

charity_lottery_differences = pd.DataFrame(columns=['number', 'prob_option_A'])

for i in charity_lottery['number'].unique():
    individual = charity_lottery.loc[charity_lottery['number'] == i, ['case', 'prob_option_A', 'valuation']] 
    individual_difference = individual.pivot(index='prob_option_A', columns='case', values='valuation')
    individual_difference['valuation_ASPC_ACPC'] = individual_difference['ASPC'] - individual_difference['ACPC']
    individual_difference['number'] = i
    individual_difference.reset_index(inplace=True)
    # individual_difference.columns = individual_difference.columns.droplevel(1)
    charity_lottery_differences = pd.concat([charity_lottery_differences, individual_difference[['number', 'prob_option_A', 'valuation_ASPC_ACPC']]], ignore_index=True)

# MEANS 
mean_valuation_ASPS = valuation_ASPS.mean()
mean_valuation_ACPC = valuation_ACPC.mean()
mean_valuation_ACPS = valuation_ACPS.mean()
mean_valuation_ASPC = valuation_ASPC.mean()

mean_valuations = [mean_valuation_ASPS.mean(), mean_valuation_ACPS.mean(), mean_valuation_ACPC.mean(), mean_valuation_ASPC.mean()]

########### Plot ditribution of participant-specific X values 
plt.hist(data_autre['charity_calibration'], bins=20) 
plt.xlabel('Participant-specific X')
plt.ylabel('Frequency')
plt.title('Distribution of X')
plt.show()

# %%
# =============================================================================
# VISUALISE DATA 
# =============================================================================

# Plot No Tradeoff Context (Replication Exley)

plt.plot(valuation_ASPS.mean().index, valuation_ASPS.mean(), label='YSPS', color='blue', marker='o', linestyle='-')
plt.plot(valuation_ACPC.mean().index, valuation_ACPC.mean(), label='YCPC', color='red', marker='o', linestyle='-')

x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)
plt.plot(x_fit, y_fit, color='grey', label='Valeur attendue')
# plt.plot(x_fit, y_fit, color='grey', label='Expected value')

plt.xlabel('Probabilité P du résultat non nul')
plt.ylabel('Valuations moyennes en %')
plt.title('Résultats dans le contexte sans compromis (PILOT)')
# plt.xlabel('Probability P of Non-Zero Payment')
# plt.ylabel('Valuation (median) as % of Riskless Lottery')
# plt.title('Results for Tradeoff Context ')
plt.grid(True)
plt.legend()
plt.savefig('No Tradeoff H1 PILOT.png', dpi=1200)
plt.show()

# Plot Tradeoff Context (Replication Exley)

plt.plot(valuation_ACPS.mean().index, valuation_ACPS.mean(), label='YCPS', color='blue', marker='o', linestyle='-')
plt.plot(valuation_ASPC.mean().index, valuation_ASPC.mean(), label='YSPC', color='red', marker='o', linestyle='-')


x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)
plt.plot(x_fit, y_fit, color='grey', label='Valeur attendue')
# plt.plot(x_fit, y_fit, color='grey', label='Expected value')

plt.xlabel('Probabilité P du résultat non nul')
plt.ylabel('Valuations moyennes en %')
plt.title('Résultats dans le contexte avec compromis (PILOT)')
# plt.xlabel('Probability P of Non-Zero Payment')
# plt.ylabel('Valuation as % of Riskless Lottery')
# plt.title('(median) Results for Tradeoff Context ')
plt.grid(True)
plt.legend()
plt.savefig('Tradeoff H1 PILOT.png', dpi=1200)
plt.show()

# Plot Self Lottery Valuation

plt.plot(valuation_ASPS.mean().index, valuation_ASPS.mean(), label='YSPS', color='green', marker='o', linestyle='-')
plt.plot(valuation_ACPS.mean().index, valuation_ACPS.mean(), label='YCPS', color='orange', marker='o', linestyle='-')


x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)
plt.plot(x_fit, y_fit, color='grey', label='Valeur attendue')
# plt.plot(x_fit, y_fit, color='grey', label='Expected value')

plt.xlabel('Probabilité P du résultat non nul')
plt.ylabel('Valuations moyennes en %')
plt.title('Résultats pour la loterie pour soi (PILOT)')
# plt.xlabel('Probability P of Non-Zero Payment')
# plt.ylabel('Valuation as % of Riskless Lottery')
# plt.title('(median) Results for Self Lottery Valuation')
plt.grid(True)
plt.legend()
plt.savefig('Self Lottery H1 PILOT.png', dpi=1200)
plt.show()

# Plot Charity Lottery Valuation

plt.plot(valuation_ASPC.mean().index, valuation_ASPC.mean(), label='YSPC', color='green', marker='o', linestyle='-')
plt.plot(valuation_ACPC.mean().index, valuation_ACPC.mean(), label='YCPC', color='orange', marker='o', linestyle='-')


x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)
plt.plot(x_fit, y_fit, color='grey', label='Valeur attendue')
# plt.plot(x_fit, y_fit, color='grey', label='Expected value')

plt.xlabel('Probabilité P du résultat non nul')
plt.ylabel('Valuations moyennes en %')
plt.title('Résultats pour la loterie pour la charité (PILOT)')
# plt.xlabel('Probability P of Non-Zero Payment')
# plt.ylabel('Valuation as % of Riskless Lottery')
# plt.title('(median) Results for Charity Lottery Valuation')
plt.grid(True)
plt.legend()
plt.savefig('Charity Lottery H1 PILOT.png', dpi=1200)
plt.show()

# Plot all Valuations
offset = 0.015

errors_per_prob = [valuation_ASPS.std(), valuation_ACPS.std(), valuation_ACPC.std(), valuation_ASPC.std()]
errors_per_prob_mean = [errors_per_prob[i].mean() for i in range(len(errors_per_prob))]
overall_errors = np.mean(errors_per_prob_mean)

plt.errorbar(valuation_ASPS.mean().index - offset, valuation_ASPS.mean(), valuation_ASPS.std(), ecolor = 'black', fmt='none', alpha=0.7)
plt.plot(valuation_ASPS.mean().index - offset, valuation_ASPS.mean(), label='$Y^{S}(P^{S})$', color='blue', marker='o', linestyle='-')

plt.errorbar(valuation_ACPS.mean().index - offset/2, valuation_ACPS.mean(), valuation_ACPS.std(), ecolor = 'black', fmt='none', alpha=0.7)
plt.plot(valuation_ACPS.mean().index - offset/2, valuation_ACPS.mean(), label='$Y^{C}(P^{S})$', color='dodgerblue', marker='o', linestyle='-')

plt.errorbar(valuation_ACPC.mean().index + offset/2, valuation_ACPC.mean(), valuation_ACPC.std(), ecolor = 'black', fmt='none', alpha=0.7)
plt.plot(valuation_ACPC.mean().index + offset/2, valuation_ACPC.mean(), label='$Y^{C}(P^{C})$', color='green', marker='o', linestyle='-')

plt.errorbar(valuation_ASPC.mean().index + offset, valuation_ASPC.mean(), valuation_ASPC.std(), ecolor = 'black', fmt='none', alpha=0.7)
plt.plot(valuation_ASPC.mean().index + offset, valuation_ASPC.mean(), label='$Y^{S}(P^{C})$', color='limegreen', marker='o', linestyle='-')

x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)
plt.plot(x_fit, y_fit, color='grey', label='Valeur attendue')
# plt.plot(x_fit, y_fit, color='grey', label='Expected value')

plt.xlabel('Probabilité P du résultat non nul')
plt.ylabel('Valuations moyennes en %')
plt.title('Résultats pour toutes les loteries (pilote)')
# plt.xlabel('Probability P of Non-Zero Payment')
# plt.ylabel('Valuation as % of Riskless Lottery')
# plt.title('(median) Results for Charity Lottery Valuation')
plt.grid(True)
plt.legend()
plt.savefig('All Lottery H1 PILOT.png', dpi=1200)
plt.show()

# Valuation without differentiation of probabilities

error_valuation = [np.std(ASPS['valuation']), np.std(ACPS['valuation']), 
                  np.std(ACPC['valuation']), np.std(ASPC['valuation'])]

plt.bar(['$Y^{S}(P^{S})$', '$Y^{C}(P^{S})$', '$Y^{C}(P^{C})$', '$Y^{S}(P^{C})$'], mean_valuations, color = ['blue', 'dodgerblue', 'green', 'limegreen']) 
plt.errorbar(['$Y^{S}(P^{S})$', '$Y^{C}(P^{S})$', '$Y^{C}(P^{C})$', '$Y^{S}(P^{C})$'], mean_valuations, error_valuation, ecolor = 'black', fmt='none', alpha=0.7)
plt.xlabel('Cas')
plt.ylabel('Moyenne de Valuations en %')
plt.title('Valuation par cas, probabilités confondues (pilote)')
plt.savefig('Bar all Lottery H1 PILOT.png', dpi=1200)
plt.show()


# Plot the difference of valuation 

plt.bar(['Self ($Y^{C}(P^{S})-Y^{S}(P^{S})$)', 'Charity ($Y^{S}(P^{C})-Y^{C}(P^{C})$)'], 
        [self_lottery_differences['valuation_ACPS_ASPS'].mean(), charity_lottery_differences['valuation_ASPC_ACPC'].mean()], 
        color = ['lightskyblue', 'lightgreen']) 
plt.errorbar(['Self ($Y^{C}(P^{S})-Y^{S}(P^{S})$)', 'Charity ($Y^{S}(P^{C})-Y^{C}(P^{C})$)'], 
              [self_lottery_differences['valuation_ACPS_ASPS'].mean(), charity_lottery_differences['valuation_ASPC_ACPC'].mean()], 
              [np.std(self_lottery_differences['valuation_ACPS_ASPS']), np.std(charity_lottery_differences['valuation_ASPC_ACPC'])], ecolor = 'black', fmt='none', alpha=0.7)
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Type de loterie')
plt.ylabel('Difference de valuations (avec - sans compro) en %')
plt.title('Difference de valuation, probabilités confondues (pilote)')
plt.savefig('Bar diff type Lottery H1 PILOT.png', dpi=1200)
plt.show()
 

# Plot Valuation for each participant ### TO REMOVE FOR REAL DATA 

if by_ind == 1: 
    for i in data_for_plot['number'].unique():
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
        
        axs[1, 0].plot(ASPS_ind['prob_option_A'], ASPS_ind['valuation'], label='ASPS', color='green')
        axs[1, 0].plot(ACPS_ind['prob_option_A'], ACPS_ind['valuation'], label='ACPS', color='orange')
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

# Categorisation 

EDRP = np.array[]

for i in data_for_plot['number'].unique():
    self_diff = self_lottery_differences.loc[self_lottery_differences['number'] == i,['valuation_ACPS_ASPS']].mean()
    charity_diff = charity_lottery_differences.loc[charity_lottery_differences['number'] == i,['valuation_ASPC_ACPC']].mean()

    if 
    

self_lottery_differences['valuation_ACPS_ASPS'].mean()
charity_lottery_differences['valuation_ASPC_ACPC'].mean()


# %%
# =============================================================================
# ANALYSE DATA 
# =============================================================================

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


