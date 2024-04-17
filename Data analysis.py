#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:50:10 2024

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

average_valuation_ASPS = ASPS.groupby('prob_option_A')['valuation'].mean()
average_valuation_ACPC = ACPC.groupby('prob_option_A')['valuation'].mean()
average_valuation_ACPS = ACPS.groupby('prob_option_A')['valuation'].mean()
average_valuation_ASPC = ASPC.groupby('prob_option_A')['valuation'].mean()

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
plt.title('(Mean) Results for No Tradeoff Context')
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
plt.title('(Mean) Results for Tradeoff Context ')
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
plt.title('(Mean) Results for Self Lottery Valuation')
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
plt.title('(Mean) Results for Charity Lottery Valuation')
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


# Plot Attention (WITHIN-SUBJECT)

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
    for i in range(1, data['number'].nunique()+1):
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


######## ATTENTION REGRESSION (WITHIN)

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

self_lottery_differences = pd.DataFrame(columns=['number', 'prob_option_A', 'valuation_ASPS_ACPS', 'dwell_time_ASPS_ACPS'])

for i in range(1, self_lottery['number'].nunique()+1):
    individual = self_lottery.loc[self_lottery['number'] == i, ['case', 'prob_option_A', 'valuation', 'dwell_time']] 
    individual_difference = individual.pivot(index='prob_option_A', columns='case', values=['valuation', 'dwell_time'])
    individual_difference['valuation_ASPS_ACPS'] = individual_difference['valuation']['ASPS'] - individual_difference['valuation']['ACPS']
    individual_difference['dwell_time_ASPS_ACPS'] = individual_difference['dwell_time']['ASPS'] - individual_difference['dwell_time']['ACPS']
    individual_difference['number'] = i
    individual_difference.reset_index(inplace=True)
    individual_difference.columns = individual_difference.columns.droplevel(1)
    self_lottery_differences = pd.concat([self_lottery_differences, individual_difference[['number', 'prob_option_A', 'valuation_ASPS_ACPS', 'dwell_time_ASPS_ACPS']]], ignore_index=True)

charity_lottery_differences = pd.DataFrame(columns=['number', 'prob_option_A', 'valuation_ACPC_ASPC', 'dwell_time_ACPC_ASPC'])

for i in range(1, charity_lottery['number'].nunique()+1):
    individual = charity_lottery.loc[charity_lottery['number'] == i, ['case', 'prob_option_A', 'valuation', 'dwell_time']] 
    individual_difference = individual.pivot(index='prob_option_A', columns='case', values=['valuation', 'dwell_time'])
    individual_difference['valuation_ACPC_ASPC'] = individual_difference['valuation']['ACPC'] - individual_difference['valuation']['ASPC']
    individual_difference['dwell_time_ACPC_ASPC'] = individual_difference['dwell_time']['ACPC'] - individual_difference['dwell_time']['ASPC']
    individual_difference['number'] = i
    individual_difference.reset_index(inplace=True)
    individual_difference.columns = individual_difference.columns.droplevel(1)
    charity_lottery_differences = pd.concat([charity_lottery_differences, individual_difference[['number', 'prob_option_A', 'valuation_ACPC_ASPC', 'dwell_time_ACPC_ASPC']]], ignore_index=True)

md_self = smf.mixedlm("valuation_ASPS_ACPS ~ dwell_time_ASPS_ACPS", self_lottery_differences, groups=self_lottery_differences["number"])
mdf_self = md_self.fit()
print(mdf_self.summary())

md_charity = smf.mixedlm("valuation_ACPC_ASPC ~ dwell_time_ACPC_ASPC", charity_lottery_differences, groups=charity_lottery_differences["number"])
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

# X_between_pred = data_for_analysis_between[['dwell_time'] + list(dummy_cases.columns) + list(dummy_ind.columns) + list(dummy_prob.columns)]
# X_between_pred = sm.add_constant(X_between_pred) # add a first column full of ones to account for intercept of regression
# y_between_pred = data_for_analysis_between['valuation']

# model_4 = sm.OLS(y_between_pred, X_between_pred).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis_between['number']}) # cluster at individual level

# print(model_4.summary())

