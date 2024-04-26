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
import matplotlib.cm as cm
import statsmodels.formula.api as smf
import ast 

censure = 1 # Put 0 if include censored participants in analysis and 1 if we exclude them 
by_ind = 0 # Put 0 if no display of individual plots and 1 if display 
attention_type = 'relative' # relative for % of total time and 'absolute' for raw time
outliers = 1 # Put 0 if include outliers in analysis and 1 if we exclude them 

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

# Remove outliers? 

dwell_mean = data_for_plot['dwell_time'].mean()
dwell_std = np.std(data_for_plot['dwell_time'])

outliers_data = data_for_plot[(data_for_plot['dwell_time'] < dwell_mean - 4 * dwell_std)
                         | (data_for_plot['dwell_time'] > dwell_mean + 4 * dwell_std)]
if outliers ==1:
    data_for_plot = data_for_plot.drop(outliers_data.index)
    data_for_plot = data_for_plot.reset_index(drop=True)
else: 
    pass 


# Get different cases

ASPS = data_for_plot[(data_for_plot['charity'] == 0) & (data_for_plot['tradeoff'] == 0)]
ACPC = data_for_plot[(data_for_plot['charity'] == 1) & (data_for_plot['tradeoff'] == 0)]
ASPC = data_for_plot[(data_for_plot['charity'] == 1) & (data_for_plot['tradeoff'] == 1)]
ACPS = data_for_plot[(data_for_plot['charity'] == 0) & (data_for_plot['tradeoff'] == 1)]

attention_ASPS = ASPS.groupby('prob_option_A')['dwell_time']
attention_ACPC = ACPC.groupby('prob_option_A')['dwell_time']
attention_ACPS = ACPS.groupby('prob_option_A')['dwell_time']
attention_ASPC = ASPC.groupby('prob_option_A')['dwell_time']

mean_attention_ASPS = attention_ASPS.mean()
mean_attention_ACPC = attention_ACPC.mean()
mean_attention_ACPS = attention_ACPS.mean()
mean_attention_ASPC = attention_ASPC.mean()

mean_attentions = [mean_attention_ASPS.mean(), mean_attention_ACPS.mean(), 
                   mean_attention_ACPC.mean(), mean_attention_ASPC.mean()]

data_for_plot_2 = data_for_plot
data_for_plot_2['first case'] = [data_for_plot_2['order of cases'][i][0] for i in range(len(data_for_plot_2))]
not_first_case = data_for_plot_2.loc[data_for_plot_2['first case'] != data_for_plot_2['case']] 
data_for_plot_2 = data_for_plot_2.drop(not_first_case.index)

ASPS_between = data_for_plot_2[(data_for_plot_2['charity'] == 0) & (data_for_plot_2['tradeoff'] == 0)]
ACPC_between = data_for_plot_2[(data_for_plot_2['charity'] == 1) & (data_for_plot_2['tradeoff'] == 0)]
ASPC_between = data_for_plot_2[(data_for_plot_2['charity'] == 1) & (data_for_plot_2['tradeoff'] == 1)]
ACPS_between = data_for_plot_2[(data_for_plot_2['charity'] == 0) & (data_for_plot_2['tradeoff'] == 1)]


# %%
# =============================================================================
# VISUALISE DATA 
# =============================================================================

# Plot Attention (WITHIN-SUBJECT)

# Plot all attention, without differentiating probabilities

error_attention = [np.std(ASPS['dwell_time']), np.std(ACPS['dwell_time']), 
                  np.std(ACPC['dwell_time']), np.std(ASPC['dwell_time'])]

plt.bar(['ASPS', 'ACPS', 'ACPC', 'ASPC'], mean_attentions, color = ['blue', 'red', 'green', 'orange']) 
plt.errorbar(['ASPS', 'ACPS', 'ACPC', 'ASPC'], mean_attentions, error_attention, ecolor = 'black', fmt='none')
plt.xlabel('Cas')
plt.ylabel('Moyenne Attention en s')
plt.title('Attention par cas, probabilités confondues (PILOT)')
plt.savefig('Bar all Lottery H2 PILOT.png', dpi=1200)
plt.show()

# Histo attention 


plt.hist(data_for_plot['dwell_time'], bins=50)
plt.xlabel('Frequence')
plt.ylabel('Attention en s')
plt.title('Histo attention')
plt.show()

average_attention_ASPS = ASPS.groupby('prob_option_A')['dwell_time'].median()
average_attention_ACPC = ACPC.groupby('prob_option_A')['dwell_time'].median()
average_attention_ACPS = ACPS.groupby('prob_option_A')['dwell_time'].median()
average_attention_ASPC = ASPC.groupby('prob_option_A')['dwell_time'].median()

all_attention = pd.concat([average_attention_ASPS, average_attention_ACPC, average_attention_ACPS, average_attention_ASPC])
all_attention = all_attention.groupby('prob_option_A').median()

plt.plot(average_attention_ASPS.index, average_attention_ASPS, label='ASPS', color='blue', marker='o', linestyle='-')
plt.plot(average_attention_ACPS.index, average_attention_ACPS, label='ACPS', color='red', marker='o', linestyle='-')
plt.plot(average_attention_ASPC.index, average_attention_ASPC, label='ASPC', color='orange', marker='o', linestyle='-')
plt.plot(average_attention_ACPC.index, average_attention_ACPC, label='ACPC', color='green', marker='o', linestyle='-')

x_fit = np.linspace(0, 1, num = 10)

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Dwell time of urn (' +str(attention_type) +')')
plt.title('(Within-subj) Median Attentional processes (' +str(attention_type) +')')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(all_attention.index, all_attention, marker='o', linestyle='-')

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Dwell time of urn (' +str(attention_type) +')')
plt.title('(Within-subj) Median across conditions  (' +str(attention_type) +')')
plt.grid(True)
plt.show()


if by_ind == 1: 
    for i in range(1, data_for_plot['number'].nunique()+1):
        ASPS_att_ind = ASPS.loc[ASPS['number'] == i, ['prob_option_A', 'dwell_time']] 
        ASPS_att_ind = ASPS_att_ind.sort_values(by=['prob_option_A'])
        ACPC_att_ind = ACPC.loc[ACPC['number'] == i, ['prob_option_A', 'dwell_time']] 
        ACPC_att_ind = ACPC_att_ind.sort_values(by=['prob_option_A'])
        ASPC_att_ind = ASPC.loc[ASPC['number'] == i, ['prob_option_A', 'dwell_time']]
        ASPC_att_ind = ASPC_att_ind.sort_values(by=['prob_option_A'])
        ACPS_att_ind = ACPS.loc[ACPS['number'] == i, ['prob_option_A', 'dwell_time']] 
        ACPS_att_ind = ACPS_att_ind.sort_values(by=['prob_option_A'])
        
        plt.plot(ASPS_att_ind['prob_option_A'], ASPS_att_ind['dwell_time'], label='ASPS', color='blue', marker='o', linestyle='-')
        plt.plot(ACPS_att_ind['prob_option_A'], ACPS_att_ind['dwell_time'], label='ACPS', color='red', marker='o', linestyle='-')
        plt.plot(ASPC_att_ind['prob_option_A'], ASPC_att_ind['dwell_time'], label='ASPC', color='orange', marker='o', linestyle='-')
        plt.plot(ACPC_att_ind['prob_option_A'], ACPC_att_ind['dwell_time'], label='ACPC', color='green', marker='o', linestyle='-')
        plt.title('Individual (' +str(attention_type) +') ' + str(i))
        plt.grid(True)
        plt.legend()
        plt.show()
else: 
    pass

# Plot Attention (BETWEEN-SUBJECT)

average_attention_ASPS_between = ASPS_between.groupby('prob_option_A')['dwell_time'].median()
average_attention_ACPC_between = ACPC_between.groupby('prob_option_A')['dwell_time'].median()
average_attention_ACPS_between = ACPS_between.groupby('prob_option_A')['dwell_time'].median()
average_attention_ASPC_between = ASPC_between.groupby('prob_option_A')['dwell_time'].median()

all_attention_between = pd.concat([average_attention_ASPS_between, average_attention_ACPC_between, 
                                   average_attention_ACPS_between, average_attention_ASPC_between])
all_attention_between = all_attention_between.groupby('prob_option_A').median()

plt.plot(average_attention_ASPS_between.index, average_attention_ASPS_between, label='ASPS', color='blue', marker='o', linestyle='-')
plt.plot(average_attention_ACPS_between.index, average_attention_ACPS_between, label='ACPS', color='red', marker='o', linestyle='-')
plt.plot(average_attention_ASPC_between.index, average_attention_ASPC_between, label='ASPC', color='orange', marker='o', linestyle='-')
plt.plot(average_attention_ACPC_between.index, average_attention_ACPC_between, label='ACPC', color='green', marker='o', linestyle='-')

x_fit = np.linspace(0, 1, num = 10)

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Dwell time of urn (' +str(attention_type) +')')
plt.title('(Between-subj) Median Attentional processes (' +str(attention_type) +')')
plt.grid(True)
plt.legend()

plt.show()


plt.plot(all_attention_between.index, all_attention_between, marker='o', linestyle='-')

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Dwell time of urn (' +str(attention_type) +')')
plt.title('(Between-subj) Median across conditions (' +str(attention_type) +')')
plt.grid(True)
plt.show()


# %%
# =============================================================================
# ANALYSE DATA 
# =============================================================================

data_for_analysis = pd.concat([ASPS, ACPC, ASPC, ACPS], ignore_index=True)


######## ATTENTION REGRESSION (WITHIN)


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

# Same process but now dwell_time as dependent variable
y_2 = data_for_analysis['dwell_time']
model_2 = sm.OLS(y_2, X).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis['number']}) # cluster at individual level
print(model_2.summary())


md_2 = smf.mixedlm("dwell_time ~ charity + tradeoff + interaction", data_for_analysis, groups=data_for_analysis["number"])
mdf_2 = md_2.fit()
print(mdf_2.summary())

######## ATTENTION REGRESSION (BETWEEN)

# Same as above but only use the first case of each individual 
data_for_analysis_between = pd.concat([ASPS_between, ACPC_between, ASPC_between, ACPS_between], ignore_index=True)

# Add fixed effects
dummy_prob = pd.get_dummies(data_for_analysis_between['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
data_for_analysis_between = pd.concat([data_for_analysis_between, dummy_prob], axis=1)
data_for_analysis_between = data_for_analysis_between.merge(survey, on='id', how='left')

# Create the design matrix and dependent variable
X_between = data_for_analysis_between[['charity', 'tradeoff', 'interaction'] + list(dummy_prob.columns)]
X_between = pd.concat([X_between, data_for_analysis_between[control_variables]], axis=1) # add controls
X_between = sm.add_constant(X_between, has_constant='add') # add a first column full of ones to account for intercept of regression 
y_between = data_for_analysis_between['dwell_time']

model_3 = sm.OLS(y_between, X_between).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis_between['prob_option_A']}) 

print(model_3.summary())

# no controls
X_between_2 = data_for_analysis_between[['charity', 'tradeoff', 'interaction'] + list(dummy_prob.columns)]
X_between_2 = sm.add_constant(X_between_2, has_constant='add') 
model_3_2 = sm.OLS(y_between, X_between_2).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis_between['prob_option_A']}) 
print(model_3_2.summary())

# explo controls

# control_variables_2 = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + 
#                   ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]
# X_between_3 = data_for_analysis_between[['charity', 'tradeoff', 'interaction'] + list(dummy_prob.columns)]
# X_between_3 = pd.concat([X_between_3, data_for_analysis_between[control_variables_2]], axis=1) # add controls
# X_between_3 = sm.add_constant(X_between_3, has_constant='add') 
# model_3_3 = sm.OLS(y_between, X_between_3).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis_between['number']}) # cluster at individual level
# print(model_3_3.summary())

md_3 = smf.mixedlm("dwell_time ~ charity + tradeoff + interaction", data_for_analysis_between, groups=data_for_analysis_between["prob_option_A"])
mdf_3 = md_3.fit()
print(mdf_3.summary())
