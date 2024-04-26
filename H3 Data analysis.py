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



# Plot ATTENTION VS VALUATION (between)

all_attention_between_pred = pd.concat([ASPS_between, ACPC_between, ACPS_between, ASPC_between])

# color for ind (dwell time)
individuals = all_attention_between_pred['number'].unique()
colors = cm.rainbow(np.linspace(0, 1, len(individuals)))
individual_color_map = dict(zip(individuals, colors))
all_attention_between_pred['color'] = all_attention_between_pred['number'].map(individual_color_map)

plt.scatter(all_attention_between_pred['dwell_time'], all_attention_between_pred['valuation'], c=all_attention_between_pred['color'])

plt.xlabel('Dwell time')
plt.ylabel('Valuation')
plt.title('(Between-subj) ALL Valuation x Dwell time (' +str(attention_type) +')')
plt.grid(True)
plt.show()

# color for ind (valuation)
individuals = all_attention_between_pred['number'].unique()
colors = cm.rainbow(np.linspace(0, 1, len(individuals)))
individual_color_map = dict(zip(individuals, colors))
all_attention_between_pred['color'] = all_attention_between_pred['number'].map(individual_color_map)

plt.scatter(all_attention_between_pred['valuation'], all_attention_between_pred['dwell_time'], c=all_attention_between_pred['color'])

plt.xlabel('Valuation')
plt.ylabel('Dwell time')
plt.title('(Between-subj) ALL Valuation x Dwell time (' +str(attention_type) +')')
plt.grid(True)
plt.show()

# color for case
individuals = all_attention_between_pred['case'].unique()
colors = cm.rainbow(np.linspace(0, 1, len(individuals)))
individual_color_map = dict(zip(individuals, colors))
all_attention_between_pred['color'] = all_attention_between_pred['case'].map(individual_color_map)

plt.scatter(all_attention_between_pred['valuation'], all_attention_between_pred['dwell_time'], c=all_attention_between_pred['color'])

plt.xlabel('Valuation')
plt.ylabel('Dwell time')
plt.title('(Between-subj) ALL Valuation x Dwell time (' +str(attention_type) +')')
plt.grid(True) # ADD WHICH COLOR IS WHICH CASE 
plt.show()

# Across conditions

# for i in [ASPS_between, ACPC_between, ACPS_between, ASPC_between]:
    
#     attention_pred = i
#     individuals = attention_pred['number'].unique()
#     colors = cm.rainbow(np.linspace(0, 1, len(individuals)))
#     individual_color_map = dict(zip(individuals, colors))
#     attention_pred['color'] = attention_pred['number'].map(individual_color_map)

#     plt.scatter(attention_pred['dwell_time'], attention_pred['valuation'], c=attention_pred['color'])

#     plt.xlabel('Dwell time')
#     plt.ylabel('Valuation')
#     plt.title('(Between-subj) ' +str(attention_pred['case'].iloc[0][:4])+ ' Valuation x Dwell time (' +str(attention_type) +') ' + str(['with', 'without'][censure]) + ' censored partic')
#     plt.grid(True)
#     plt.show() 


# for i in [ASPS_between, ACPC_between, ACPS_between, ASPC_between]:
    
#     attention_pred = i
#     individuals = attention_pred['number'].unique()
#     colors = cm.rainbow(np.linspace(0, 1, len(individuals)))
#     individual_color_map = dict(zip(individuals, colors))
#     attention_pred['color'] = attention_pred['number'].map(individual_color_map)

#     plt.scatter(attention_pred['valuation'], attention_pred['dwell_time'], c=attention_pred['color'])

#     plt.xlabel('Valuation')
#     plt.ylabel('Dwell time')
#     plt.title('(Between-subj) ' +str(attention_pred['case'].iloc[0][:4])+ ' Dwell time x Valuation (' +str(attention_type) +') ' + str(['with', 'without'][censure]) + ' censored partic')
#     plt.grid(True)
#     plt.show()


# %%
# =============================================================================
# ANALYSE DATA 
# =============================================================================

data_for_analysis = pd.concat([ASPS, ACPC, ASPC, ACPS], ignore_index=True)

###### ATTENTION AS PREDICTOR
# data_for_analysis_between['case_id']=[['ASPS', 'ACPC','ASPC', 'ACPS']. index(data_for_analysis_between['case'][i]) for i in range(len(data_for_analysis_between))]

# # GOGOGOGO 
# md_4 = smf.mixedlm("valuation ~ charity*dwell_time + tradeoff*dwell_time + interaction*dwell_time", data_for_analysis, groups=data_for_analysis["number"])
# mdf_4 = md_4.fit()
# print(mdf_4.summary())

# md_5 = smf.mixedlm("valuation ~ charity*dwell_time + tradeoff*dwell_time + interaction*dwell_time", data_for_analysis_between, groups=data_for_analysis_between["case_id"])
# mdf_5 = md_5.fit()
# print(mdf_5.summary())

# md_6 = smf.mixedlm("valuation ~ dwell_time*case_id", data_for_analysis_between, groups=data_for_analysis_between["case_id"])
# mdf_6 = md_6.fit()
# print(mdf_6.summary())

# md_7 = smf.mixedlm("valuation ~ dwell_time", ASPC_between, groups=ASPC_between["prob_option_A"])
# mdf_7 = md_7.fit()
# print(mdf_7.summary())

md_8 = smf.mixedlm("valuation ~ dwell_time", data_for_analysis, groups=data_for_analysis["number"])
mdf_8 = md_8.fit()
print(mdf_8.summary())


# DIFFERENCES of valuations and dwell time 
self_lottery = pd.concat([ASPS, ACPS], ignore_index = True)
charity_lottery = pd.concat([ACPC, ASPC], ignore_index=True)

self_lottery_differences = pd.DataFrame(columns=['number', 'prob_option_A'])

for i in self_lottery['number'].unique():
    individual = self_lottery.loc[self_lottery['number'] == i, ['case', 'prob_option_A', 'valuation', 'dwell_time']] 
    individual_difference = individual.pivot(index='prob_option_A', columns='case', values=['valuation', 'dwell_time'])
    individual_difference['valuation_ACPS_ASPS'] = individual_difference['valuation']['ACPS'] - individual_difference['valuation']['ASPS']
    individual_difference['dwell_time_ACPS_ASPS'] = individual_difference['dwell_time']['ACPS'] - individual_difference['dwell_time']['ASPS']
    individual_difference['number'] = i
    individual_difference.reset_index(inplace=True)
    individual_difference.columns = individual_difference.columns.droplevel(1)
    self_lottery_differences = pd.concat([self_lottery_differences, individual_difference[['number', 'prob_option_A', 'valuation_ACPS_ASPS', 'dwell_time_ACPS_ASPS']]], ignore_index=True)

charity_lottery_differences = pd.DataFrame(columns=['number', 'prob_option_A'])

for i in charity_lottery['number'].unique():
    individual = charity_lottery.loc[charity_lottery['number'] == i, ['case', 'prob_option_A', 'valuation', 'dwell_time']] 
    individual_difference = individual.pivot(index='prob_option_A', columns='case', values=['valuation', 'dwell_time'])
    individual_difference['valuation_ASPC_ACPC'] = individual_difference['valuation']['ASPC'] - individual_difference['valuation']['ACPC']
    individual_difference['dwell_time_ASPC_ACPC'] = individual_difference['dwell_time']['ASPC'] - individual_difference['dwell_time']['ACPC']
    individual_difference['number'] = i
    individual_difference.reset_index(inplace=True)
    individual_difference.columns = individual_difference.columns.droplevel(1)
    charity_lottery_differences = pd.concat([charity_lottery_differences, individual_difference[['number', 'prob_option_A', 'valuation_ASPC_ACPC', 'dwell_time_ASPC_ACPC']]], ignore_index=True)



# self_lottery = pd.concat([ASPS, ACPS], ignore_index = True)
# charity_lottery = pd.concat([ACPC, ASPC], ignore_index=True)

# self_lottery_differences = pd.DataFrame(columns=['number', 'prob_option_A', 'valuation_ASPS_ACPS', 'dwell_time_ASPS_ACPS'])

# for i in data_for_plot['number'].unique():
#     individual = self_lottery.loc[self_lottery['number'] == i, ['case', 'prob_option_A', 'valuation', 'dwell_time']] 
#     individual_difference = individual.pivot(index='prob_option_A', columns='case', values=['valuation', 'dwell_time'])
#     individual_difference['valuation_ASPS_ACPS'] = individual_difference['valuation']['ASPS'] - individual_difference['valuation']['ACPS']
#     individual_difference['dwell_time_ASPS_ACPS'] = individual_difference['dwell_time']['ASPS'] - individual_difference['dwell_time']['ACPS']
#     individual_difference['number'] = i
#     individual_difference.reset_index(inplace=True)
#     individual_difference.columns = individual_difference.columns.droplevel(1)
#     self_lottery_differences = pd.concat([self_lottery_differences, individual_difference[['number', 'prob_option_A', 'valuation_ASPS_ACPS', 'dwell_time_ASPS_ACPS']]], ignore_index=True)

# charity_lottery_differences = pd.DataFrame(columns=['number', 'prob_option_A', 'valuation_ACPC_ASPC', 'dwell_time_ACPC_ASPC'])

# for i in data_for_plot['number'].unique():
#     individual = charity_lottery.loc[charity_lottery['number'] == i, ['case', 'prob_option_A', 'valuation', 'dwell_time']] 
#     individual_difference = individual.pivot(index='prob_option_A', columns='case', values=['valuation', 'dwell_time'])
#     individual_difference['valuation_ACPC_ASPC'] = individual_difference['valuation']['ACPC'] - individual_difference['valuation']['ASPC']
#     individual_difference['dwell_time_ACPC_ASPC'] = individual_difference['dwell_time']['ACPC'] - individual_difference['dwell_time']['ASPC']
#     individual_difference['number'] = i
#     individual_difference.reset_index(inplace=True)
#     individual_difference.columns = individual_difference.columns.droplevel(1)
#     charity_lottery_differences = pd.concat([charity_lottery_differences, individual_difference[['number', 'prob_option_A', 'valuation_ACPC_ASPC', 'dwell_time_ACPC_ASPC']]], ignore_index=True)

md_self = smf.mixedlm("valuation_ACPS_ASPS ~ dwell_time_ACPS_ASPS", self_lottery_differences, groups=self_lottery_differences["number"])
mdf_self = md_self.fit()
print(mdf_self.summary())

md_charity = smf.mixedlm("valuation_ASPC_ACPC ~ dwell_time_ASPC_ACPC", charity_lottery_differences, groups=charity_lottery_differences["number"])
mdf_charity = md_charity.fit()
print(mdf_charity.summary())

# analysis
dummy_ind = pd.get_dummies(self_lottery_differences['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob = pd.get_dummies(self_lottery_differences['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
self_lottery_differences = pd.concat([self_lottery_differences, dummy_ind, dummy_prob], axis=1)
# self_lottery_differences = self_lottery_differences.merge(survey, on='number', how='left')

X_pred_self = self_lottery_differences[['dwell_time_ASPS_ACPS'] + list(dummy_ind.columns) + list(dummy_prob.columns)]
# X_pred_self = pd.concat([X_pred_self, X_pred_self[control_variables]], axis=1)
X_pred_self = sm.add_constant(X_pred_self, has_constant='add') # add a first column full of ones to account for intercept of regression
y_pred_self = self_lottery_differences['valuation_ASPS_ACPS']
model_pred_self = sm.OLS(y_pred_self, X_pred_self).fit(cov_type='cluster', cov_kwds={'groups': self_lottery_differences['number']}) # cluster at individual level
print(model_pred_self.summary())


dummy_ind = pd.get_dummies(charity_lottery_differences['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob = pd.get_dummies(charity_lottery_differences['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
charity_lottery_differences = pd.concat([charity_lottery_differences, dummy_ind, dummy_prob], axis=1)
# charity_lottery_differences = charity_lottery_differences.merge(survey, on='number', how='left')

X_pred_charity = charity_lottery_differences[['dwell_time_ACPC_ASPC'] + list(dummy_ind.columns) + list(dummy_prob.columns)]
# X_pred_charity = pd.concat([X_pred_charity, X_pred_charity[control_variables]], axis=1)
X_pred_charity = sm.add_constant(X_pred_charity, has_constant='add') # add a first column full of ones to account for intercept of regression
y_pred_charity = charity_lottery_differences['valuation_ACPC_ASPC']
model_pred_charity = sm.OLS(y_pred_charity, X_pred_charity).fit(cov_type='cluster', cov_kwds={'groups': charity_lottery_differences['number']}) # cluster at individual level
print(model_pred_charity.summary())


## PLOTS 
individuals = self_lottery_differences['number'].unique()
colors = cm.rainbow(np.linspace(0, 1, len(individuals)))
individual_color_map = dict(zip(individuals, colors))
self_lottery_differences['color'] = self_lottery_differences['number'].map(individual_color_map)

plt.scatter(self_lottery_differences['valuation_ASPS_ACPS'], self_lottery_differences['dwell_time_ASPS_ACPS'], c=self_lottery_differences['color'])

plt.xlabel('Valuation')
plt.ylabel('Dwell time')
plt.title('(Within-subj) Diff SELF Valuation x Dwell time (' +str(attention_type) +')')
plt.grid(True)
plt.show()


individuals = charity_lottery_differences['number'].unique()
colors = cm.rainbow(np.linspace(0, 1, len(individuals)))
individual_color_map = dict(zip(individuals, colors))
charity_lottery_differences['color'] = charity_lottery_differences['number'].map(individual_color_map)

plt.scatter(charity_lottery_differences['valuation_ACPC_ASPC'], charity_lottery_differences['dwell_time_ACPC_ASPC'], c=charity_lottery_differences['color'])

plt.xlabel('Valuation')
plt.ylabel('Dwell time')
plt.title('(Within-subj) Diff CHARITY Valuation x Dwell time (' +str(attention_type) +')')
plt.grid(True)
plt.show()



plt.scatter(self_lottery_differences['prob_option_A'], self_lottery_differences['valuation_ASPS_ACPS'], c='red', label='self')
plt.scatter(charity_lottery_differences['prob_option_A'], charity_lottery_differences['valuation_ACPC_ASPC'], c='green', label='charity')

plt.xlabel('Prob')
plt.ylabel('Valuation diff')
plt.title('(Within-subj) Diff Valuation')
plt.grid(True)
plt.legend()
plt.show()

self_lottery_differences_grouped = self_lottery_differences.groupby('prob_option_A')['valuation_ASPS_ACPS'].median()
charity_lottery_differences_grouped = charity_lottery_differences.groupby('prob_option_A')['valuation_ACPC_ASPC'].median()

plt.scatter(self_lottery_differences_grouped.index, self_lottery_differences_grouped, c='red', label='self')
plt.scatter(charity_lottery_differences_grouped.index, charity_lottery_differences_grouped, c='green', label='charity')

plt.xlabel('Prob')
plt.ylabel('Valuation diff')
plt.title('(Within-subj) Diff Valuation')
plt.grid(True)
plt.legend()
plt.show()

# data_for_analysis_between_attention = data_for_analysis_between

# data_for_analysis_between_attention['charity_atten'] = data_for_analysis_between_attention['charity'] * data_for_analysis_between_attention['dwell_time']
# data_for_analysis_between_attention['tradeoff_atten'] = data_for_analysis_between_attention['tradeoff'] * data_for_analysis_between_attention['dwell_time']
# data_for_analysis_between_attention['interaction_atten'] = data_for_analysis_between_attention['interaction'] * data_for_analysis_between_attention['dwell_time']

# X_between_pred = data_for_analysis_between[['charity_atten', 'tradeoff_atten', 'interaction_atten'] + list(dummy_ind.columns) + list(dummy_prob.columns)]
# X_between_pred = sm.add_constant(X_between_pred) # add a first column full of ones to account for intercept of regression
# y_between_pred = data_for_analysis_between['valuation']

# model_4 = sm.OLS(y_between_pred, X_between_pred).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis_between['number']}) # cluster at individual level

# print(model_4.summary())



# plot dwell time 

# for i in [ASPS_between, ACPC_between, ACPS_between, ASPC_between]:
#     attention_case = i
#     bins = np.linspace(0, 50, 50)
#     plt.hist([attention_case['dwell_time_absolute'], attention_case['dwell_time_relative'], attention_case['total_time_spent_s']], bins, label = ['absolute', 'relative', 'total'])
#     plt.legend()
#     plt.title('(Between-subj) ' +str(attention_case['case'].iloc[0][:4]))
#     plt.show()

# for i in [ASPS, ACPC, ACPS, ASPC]:
#     attention_case = i
#     bins = np.linspace(0, 50, 50)
#     plt.hist([attention_case['dwell_time_absolute'], attention_case['dwell_time_relative'], attention_case['total_time_spent_s']], bins, label = ['absolute', 'relative', 'total'])
#     plt.legend()
#     plt.title('(Within-subj) ' +str(attention_case['case'].iloc[0][:4]))
#     plt.show()


# dwell_ind = data_for_analysis.loc[data_for_analysis['number'] == 1, ['case', 'prob_option_A', 'valuation', 'dwell_time_relative', 'dwell_time_absolute']] 


#### EXPLORATOIRE??? 
# data_for_analysis_between['interaction_atten'] = data_for_analysis_between['dwell_time'] * data_for_analysis['frequency'] 
# X_between_pred = data_for_analysis_between[['dwell_time', 'frequency', 'interaction_atten'] + list(dummy_ind.columns) + list(dummy_prob.columns)]

# data_for_analysis['interaction_atten'] = data_for_analysis['dwell_time'] * data_for_analysis['frequency'] 

# X_2 = data_for_analysis[['dwell_time', 'frequency', 'interaction_atten', 'prob_option_A', 'number']]

# X_2 = sm.add_constant(X_2) # add a first column full of ones to account for intercept of regression


# X_2['prob_option_A'] = X_2['prob_option_A'].astype(float) # so that everything is float for regression model
# X_2['number'] = X_2['number'].astype(float) # so that everything is float for regression model


# dummy_cases = pd.get_dummies(data_for_analysis_between['case'], dtype=int)      
# dummy_cases.drop(columns=['ASPS'], inplace=True) # force ASPS to be reference 
# data_for_analysis_between = pd.concat([data_for_analysis_between, dummy_cases], axis=1)

