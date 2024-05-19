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
MSP_excl = 1 # Put 0 if include MSP calib in analysis and 1 if we exclude them 
by_ind = 0 # Put 0 if no display of individual plots and 1 if display 
attention_type = 'absolute' # relative for % of total time and 'absolute' for raw time
outliers = 1 # Put 0 if include outliers in analysis and 1 if we exclude them 

path = '/Users/carolinepioger/Desktop/ALL collection' # change to yours :)

# Get dataframes
data = pd.read_csv(path + '/dataset.csv' )
data_autre = pd.read_csv(path + '/criterion info data.csv')
survey = pd.read_csv(path + '/survey data.csv')


for i in range(len(data_autre)):
    if data_autre['censored_calibration'][i] == 'MSP':
        pass
    elif isinstance(data_autre['censored_calibration'][i], str):
        data_autre['censored_calibration'][i] = ast.literal_eval(data_autre['censored_calibration'][i])

########
# data_autre['censored_calibration'].value_counts()
########
data_autre_principal = data_autre.loc[data_autre['censored_calibration'] == 0]
data_autre_principal = data_autre_principal.reset_index(drop=True)
 
data_autre_censored = data_autre.loc[data_autre['censored_calibration'] == 1] 
data_autre_censored = data_autre_censored.reset_index(drop=True)


# Remove (or not) participants with censored values in part 2
exclude_participants = data_autre.loc[data_autre['censored_calibration'] == 1, 'id'] 

if censure == 1: 
    data = data.drop(data[data['id'].isin(exclude_participants) == True].index)
    data = data.reset_index(drop=True)
else: 
    data = data


# Remove participants with mutliple switchoint (MSP) in part 2

exclude_participants_2 = data_autre.loc[data_autre['censored_calibration'] == 'MSP', 'id'] 

if MSP_excl == 1: 
    data = data.drop(data[data['id'].isin(exclude_participants_2) == True].index)
    data = data.reset_index(drop=True)
else: 
    data = data

data_for_plot = data

# Convert order of cases in string 
for i in range(len(data_for_plot)):
    data_for_plot['order of cases'][i] = ast.literal_eval(data_for_plot['order of cases'][i])
    

# %%
# =============================================================================
# REMOVING OUTLIERS
# =============================================================================

# Remove outliers

dwell_mean = data_for_plot['dwell_time'].mean()
dwell_std = np.std(data_for_plot['dwell_time'])
outliers_data = data_for_plot[(data_for_plot['dwell_time'] < dwell_mean - 3 * dwell_std)
                         | (data_for_plot['dwell_time'] > dwell_mean + 3 * dwell_std)]

dwell_mean_total = data_for_plot['total_time_spent_s'].mean()
dwell_std_total = np.std(data_for_plot['total_time_spent_s'])
outliers_data_total = data_for_plot[(data_for_plot['total_time_spent_s'] < dwell_mean_total - 3 * dwell_std_total)
                         | (data_for_plot['total_time_spent_s'] > dwell_mean_total + 3 * dwell_std_total)]

dwell_mean_relative = data_for_plot['dwell_time_relative'].mean()
dwell_std_relative = np.std(data_for_plot['dwell_time_relative'])
outliers_data_relative = data_for_plot[(data_for_plot['dwell_time_relative'] < dwell_mean_relative - 3 * dwell_std_relative)
                         | (data_for_plot['dwell_time_relative'] > dwell_mean_relative + 3 * dwell_std_relative)]

outliers_all = np.union1d(outliers_data.index, np.union1d(outliers_data_total.index, outliers_data_relative.index))

# Remove outliers and associated data 

associated_outliers = []

for index in outliers_all:
    outlier_row = data_for_plot.iloc[index]

    if outlier_row['case'] == 'ASPS':
        corresponding_row = data_for_plot[
            (data_for_plot['case'] == 'ACPS') &
            (data_for_plot['number'] == outlier_row['number']) & 
            (data_for_plot['prob_option_A'] == outlier_row['prob_option_A'])
        ]
    elif outlier_row['case'] == 'ACPS' :
        corresponding_row = data_for_plot[
            (data_for_plot['case'] == 'ASPS') &
            (data_for_plot['number'] == outlier_row['number']) & 
            (data_for_plot['prob_option_A'] == outlier_row['prob_option_A'])
        ]
    elif outlier_row['case'] == 'ACPC' :
        corresponding_row = data_for_plot[
            (data_for_plot['case'] == 'ASPC') &
            (data_for_plot['number'] == outlier_row['number']) & 
            (data_for_plot['prob_option_A'] == outlier_row['prob_option_A'])
        ]
    elif outlier_row['case'] == 'ASPC':
        corresponding_row = data_for_plot[
            (data_for_plot['case'] == 'ACPC') &
            (data_for_plot['number'] == outlier_row['number']) & 
            (data_for_plot['prob_option_A'] == outlier_row['prob_option_A'])
        ]

    associated_outliers.append(corresponding_row.index[0])


remove_all = np.union1d(associated_outliers,outliers_all)

data_for_plot = data_for_plot.drop(remove_all)
data_for_plot = data_for_plot.reset_index(drop=True)



# %%
# =============================================================================
# GET ALL DATA
# =============================================================================

# Get different cases

ASPS = data_for_plot[(data_for_plot['charity'] == 0) & (data_for_plot['tradeoff'] == 0)]
ACPC = data_for_plot[(data_for_plot['charity'] == 1) & (data_for_plot['tradeoff'] == 0)]
ASPC = data_for_plot[(data_for_plot['charity'] == 1) & (data_for_plot['tradeoff'] == 1)]
ACPS = data_for_plot[(data_for_plot['charity'] == 0) & (data_for_plot['tradeoff'] == 1)]



# Get attention values

attention_ASPS = ASPS.groupby('prob_option_A')['dwell_time_relative']
attention_ACPC = ACPC.groupby('prob_option_A')['dwell_time_relative']
attention_ACPS = ACPS.groupby('prob_option_A')['dwell_time_relative']
attention_ASPC = ASPC.groupby('prob_option_A')['dwell_time_relative']

mean_attention_ASPS = attention_ASPS.mean()
mean_attention_ACPC = attention_ACPC.mean()
mean_attention_ACPS = attention_ACPS.mean()
mean_attention_ASPC = attention_ASPC.mean()

mean_attentions = [mean_attention_ASPS.mean(), mean_attention_ACPS.mean(), 
                   mean_attention_ACPC.mean(), mean_attention_ASPC.mean()]


first_case = data_for_plot[data_for_plot['case_order']==1]
second_case = data_for_plot[data_for_plot['case_order']==2]
third_case = data_for_plot[data_for_plot['case_order']==3]
fourth_case = data_for_plot[data_for_plot['case_order']==4]


# %%
# =============================================================================
# Get valuation and attention differences specific to H3
# =============================================================================


# Difference data valuation ( /!/ inverse difference of H1 and H2 /!/ )
self_lottery = pd.concat([ASPS, ACPS], ignore_index = True)
charity_lottery = pd.concat([ACPC, ASPC], ignore_index=True)

self_lottery_differences_all = pd.DataFrame(columns=['number', 'prob_option_A'])

for i in self_lottery['number'].unique():
    individual = self_lottery.loc[self_lottery['number'] == i, ['case', 'prob_option_A', 'valuation', 'dwell_time_relative']] 
    individual_difference = individual.pivot(index='prob_option_A', columns='case')
    individual_difference['valuation_ASPS_ACPS'] = individual_difference['valuation']['ASPS'] - individual_difference['valuation']['ACPS']
    individual_difference['dwell_time_ASPS_ACPS'] = individual_difference['dwell_time_relative']['ASPS'] - individual_difference['dwell_time_relative']['ACPS']
    individual_difference['number'] = i
    individual_difference.reset_index(inplace=True)
    individual_difference.columns = individual_difference.columns.droplevel(1)
    self_lottery_differences_all = pd.concat([self_lottery_differences_all, individual_difference[['number', 'prob_option_A', 'valuation_ASPS_ACPS', 'dwell_time_ASPS_ACPS']]], ignore_index=True)

charity_lottery_differences_all = pd.DataFrame(columns=['number', 'prob_option_A'])

for i in charity_lottery['number'].unique():
    individual = charity_lottery.loc[charity_lottery['number'] == i, ['case', 'prob_option_A', 'valuation', 'dwell_time_relative']] 
    individual_difference = individual.pivot(index='prob_option_A', columns='case')
    individual_difference['valuation_ACPC_ASPC'] = individual_difference['valuation']['ACPC'] - individual_difference['valuation']['ASPC']
    individual_difference['dwell_time_ACPC_ASPC'] = individual_difference['dwell_time_relative']['ACPC'] - individual_difference['dwell_time_relative']['ASPC']
    individual_difference['number'] = i
    individual_difference.reset_index(inplace=True)
    individual_difference.columns = individual_difference.columns.droplevel(1)
    charity_lottery_differences_all = pd.concat([charity_lottery_differences_all, individual_difference[['number', 'prob_option_A', 'valuation_ACPC_ASPC', 'dwell_time_ACPC_ASPC']]], ignore_index=True)

# # Get differences attention

# self_lottery_attention = pd.concat([ASPS, ACPS], ignore_index = True)
# charity_lottery_attention = pd.concat([ACPC, ASPC], ignore_index=True)

# self_lottery_differences_attention = pd.DataFrame(columns=['number', 'prob_option_A'])

# for i in self_lottery_attention['number'].unique():
#     individual = self_lottery_attention.loc[self_lottery_attention['number'] == i, ['case', 'prob_option_A', 'dwell_time_relative']] 
#     individual_difference = individual.pivot(index='prob_option_A', columns='case', values='dwell_time_relative')
#     individual_difference['dwell_time_ASPS_ACPS'] = individual_difference['ACPS'] - individual_difference['ASPS']
#     individual_difference['number'] = i
#     individual_difference.reset_index(inplace=True)
#     # individual_difference.columns = individual_difference.columns.droplevel(1)
#     self_lottery_differences_attention = pd.concat([self_lottery_differences_attention, individual_difference[['number', 'prob_option_A', 'dwell_time_ASPS_ACPS']]], ignore_index=True)

# charity_lottery_differences_attention = pd.DataFrame(columns=['number', 'prob_option_A'])

# for i in charity_lottery_attention['number'].unique():
#     individual = charity_lottery_attention.loc[charity_lottery_attention['number'] == i, ['case', 'prob_option_A', 'dwell_time_relative']] 
#     individual_difference = individual.pivot(index='prob_option_A', columns='case', values='dwell_time_relative')
#     individual_difference['dwell_time_ACPC_ASPC'] = individual_difference['ASPC'] - individual_difference['ACPC']
#     individual_difference['number'] = i
#     individual_difference.reset_index(inplace=True)
#     # individual_difference.columns = individual_difference.columns.droplevel(1)
#     charity_lottery_differences_attention = pd.concat([charity_lottery_differences_attention, individual_difference[['number', 'prob_option_A', 'dwell_time_ACPC_ASPC']]], ignore_index=True)



# %%
# =============================================================================
# Categorisation Excuse-driven risk preferences
# =============================================================================


EDRP_self = []
EDRP_charity = []

altruistic_self = []
altruistic_charity = []

for i in data_for_plot['number'].unique():
    self_diff = self_lottery_differences_all.loc[self_lottery_differences_all['number'] == i,['valuation_ASPS_ACPS']].mean() # mean across probabilities
    charity_diff = charity_lottery_differences_all.loc[charity_lottery_differences_all['number'] == i,['valuation_ACPC_ASPC']].mean() # mean across probabilities

    if self_diff.item() > 5 :
        EDRP_self.append(i)
    elif self_diff.item() < - 5 :
        altruistic_self.append(i)
    if charity_diff.item() < - 5 :
        EDRP_charity.append(i)
    if charity_diff.item() > 5 :
        altruistic_charity.append(i)
    
EDRP_total = np.intersect1d(EDRP_self, EDRP_charity)

altruistic_total = np.intersect1d(altruistic_self, altruistic_charity)

no_EDRP = np.setdiff1d(data_for_plot['number'].unique(), np.union1d(EDRP_total, altruistic_total))

plt.bar(['Self', 'Charity'], [len(EDRP_self), len(EDRP_charity)], color = ['lightskyblue', 'lightgreen']) 
plt.bar(['Self', 'Charity'], [len(EDRP_total), len(EDRP_total)], color = ['palegoldenrod', 'palegoldenrod'], label ='Both') 
plt.xlabel('Type of Excuse-driven risk preference')
plt.ylabel('Number of people')
plt.title('Number of Excuse-driven participants')
plt.legend()
plt.show()

X_EDRP_total = data_autre_principal[data_autre_principal['number'].isin(EDRP_total)]
data_X_EDRP_total = data_for_plot[data_for_plot['number'].isin(EDRP_total)]

X_else_EDRP_total = data_autre_principal[~data_autre_principal['number'].isin(EDRP_total)]
data_else_EDRP = data_for_plot[~data_for_plot['number'].isin(data_X_EDRP_total['number'])]

X_no_EDRP_total = data_autre_principal[data_autre_principal['number'].isin(no_EDRP)]
data_no_EDRP = data_for_plot[data_for_plot['number'].isin(no_EDRP)]

X_altruistic = data_autre_principal[data_autre_principal['number'].isin(altruistic_total)]
data_altruistic = data_for_plot[data_for_plot['number'].isin(altruistic_total)]

# self_lottery_difference_EDRP = self_lottery_differences_all[self_lottery_differences_all['number'].isin(EDRP_total)]
# charity_lottery_differences_EDRP = charity_lottery_differences[charity_lottery_differences['number'].isin(EDRP_total)]


ASPS_EDRP = data_X_EDRP_total[(data_X_EDRP_total['charity'] == 0) & (data_X_EDRP_total['tradeoff'] == 0)]
ACPC_EDRP = data_X_EDRP_total[(data_X_EDRP_total['charity'] == 1) & (data_X_EDRP_total['tradeoff'] == 0)]
ASPC_EDRP = data_X_EDRP_total[(data_X_EDRP_total['charity'] == 1) & (data_X_EDRP_total['tradeoff'] == 1)]
ACPS_EDRP = data_X_EDRP_total[(data_X_EDRP_total['charity'] == 0) & (data_X_EDRP_total['tradeoff'] == 1)]

ASPS_no_EDRP = data_no_EDRP[(data_no_EDRP['charity'] == 0) & (data_no_EDRP['tradeoff'] == 0)]
ACPC_no_EDRP = data_no_EDRP[(data_no_EDRP['charity'] == 1) & (data_no_EDRP['tradeoff'] == 0)]
ASPC_no_EDRP = data_no_EDRP[(data_no_EDRP['charity'] == 1) & (data_no_EDRP['tradeoff'] == 1)]
ACPS_no_EDRP = data_no_EDRP[(data_no_EDRP['charity'] == 0) & (data_no_EDRP['tradeoff'] == 1)]

ASPS_altruistic = data_altruistic[(data_altruistic['charity'] == 0) & (data_altruistic['tradeoff'] == 0)]
ACPC_altruistic = data_altruistic[(data_altruistic['charity'] == 1) & (data_altruistic['tradeoff'] == 0)]
ASPC_altruistic = data_altruistic[(data_altruistic['charity'] == 1) & (data_altruistic['tradeoff'] == 1)]
ACPS_altruistic = data_altruistic[(data_altruistic['charity'] == 0) & (data_altruistic['tradeoff'] == 1)]

self_lottery_differences_all_EDRP = self_lottery_differences_all[self_lottery_differences_all['number'].isin(EDRP_total)]
charity_lottery_differences_all_EDRP = charity_lottery_differences_all[charity_lottery_differences_all['number'].isin(EDRP_total)]

# self_lottery_differences_no_EDRP = self_lottery_differences_all[self_lottery_differences_all['number'].isin(no_EDRP)]
# charity_lottery_differences_attention_no_EDRP = charity_lottery_differences_attention[charity_lottery_differences_attention['number'].isin(no_EDRP)]

# self_lottery_differences_attention_altruistic = self_lottery_differences_attention[self_lottery_differences_attention['number'].isin(altruistic_total)]
# charity_lottery_differences_attention_altruistic = charity_lottery_differences_attention[charity_lottery_differences_attention['number'].isin(altruistic_total)]


# %%
# =============================================================================
# VISUALISE DATA 
# =============================================================================


data_for_analysis = pd.concat([ASPS, ACPC, ASPC, ACPS], ignore_index=True)
data_for_analysis_EDRP = pd.concat([ASPS_EDRP, ACPC_EDRP, ASPC_EDRP, ACPS_EDRP], ignore_index=True)

# # Get different cases

# ASPS = data_for_plot[(data_for_plot['charity'] == 0) & (data_for_plot['tradeoff'] == 0)]
# ACPC = data_for_plot[(data_for_plot['charity'] == 1) & (data_for_plot['tradeoff'] == 0)]
# ASPC = data_for_plot[(data_for_plot['charity'] == 1) & (data_for_plot['tradeoff'] == 1)]
# ACPS = data_for_plot[(data_for_plot['charity'] == 0) & (data_for_plot['tradeoff'] == 1)]

# average_valuation_ASPS = ASPS.groupby('prob_option_A')['valuation'].median()
# average_valuation_ACPC = ACPC.groupby('prob_option_A')['valuation'].median()
# average_valuation_ACPS = ACPS.groupby('prob_option_A')['valuation'].median()
# average_valuation_ASPC = ASPC.groupby('prob_option_A')['valuation'].median()

individuals = data_for_analysis['number'].unique()
colors = cm.rainbow(np.linspace(0, 1, len(individuals)))
individual_color_map = dict(zip(individuals, colors))
data_for_analysis['color'] = data_for_analysis['number'].map(individual_color_map)

plt.scatter(data_for_analysis['dwell_time_relative'], data_for_analysis['valuation'],  c=data_for_analysis['color'])

plt.xlabel('Dwell time')
plt.ylabel('Valuation')
plt.title('(Within-subj) Diff SELF Valuation x Dwell time (' +str(attention_type) +')')
plt.grid(True)
plt.show()


individuals = data_X_EDRP_total['number'].unique()
colors = cm.rainbow(np.linspace(0, 1, len(individuals)))
individual_color_map = dict(zip(individuals, colors))
data_X_EDRP_total['color'] = data_X_EDRP_total['number'].map(individual_color_map)

plt.scatter(data_X_EDRP_total['dwell_time_relative'], data_X_EDRP_total['valuation'],  c=data_X_EDRP_total['color'])

plt.xlabel('Dwell time')
plt.ylabel('Valuation')
plt.title('EDRP Diff SELF Valuation x Dwell time (' +str(attention_type) +')')
plt.grid(True)
plt.show()


# %%
# =============================================================================
# ANALYSE DATA 
# =============================================================================


# No categorisation by lottery type 

md_8 = smf.mixedlm("valuation ~ dwell_time_relative", data_for_analysis, groups=data_for_analysis["number"])
mdf_8 = md_8.fit()
print(mdf_8.summary())

md_8 = smf.mixedlm("valuation ~ dwell_time_relative", data_for_analysis_EDRP, groups=data_for_analysis_EDRP["number"])
mdf_8 = md_8.fit()
print(mdf_8.summary())




# self_lottery_differences_all = self_lottery_differences_all.dropna()
# charity_lottery_differences_all = charity_lottery_differences_all.dropna()

# ALL 

md_self = smf.mixedlm("valuation_ASPS_ACPS ~ dwell_time_ASPS_ACPS", self_lottery_differences_all, groups=self_lottery_differences_all["number"])
mdf_self = md_self.fit()
print(mdf_self.summary())

md_charity = smf.mixedlm("valuation_ACPC_ASPC ~ dwell_time_ACPC_ASPC", charity_lottery_differences_all, groups=charity_lottery_differences_all["number"])
mdf_charity = md_charity.fit()
print(mdf_charity.summary())


# EDRP 
# self_lottery_differences_all_EDRP = self_lottery_differences_all_EDRP.dropna()
# charity_lottery_differences_all_EDRP = charity_lottery_differences_all_EDRP.dropna()

md_self_EDRP = smf.mixedlm("valuation_ASPS_ACPS ~ dwell_time_ASPS_ACPS", self_lottery_differences_all_EDRP, groups=self_lottery_differences_all_EDRP["number"])
mdf_self_EDRP = md_self_EDRP.fit()
print(mdf_self_EDRP.summary())

md_charity_EDRP = smf.mixedlm("valuation_ACPC_ASPC ~ dwell_time_ACPC_ASPC", charity_lottery_differences_all_EDRP, groups=charity_lottery_differences_all_EDRP["number"])
mdf_charity_EDRP = md_charity_EDRP.fit()
print(mdf_charity_EDRP.summary())







# analysis
dummy_ind = pd.get_dummies(self_lottery_differences_all['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob = pd.get_dummies(self_lottery_differences_all['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
self_lottery_differences_all = pd.concat([self_lottery_differences_all, dummy_ind, dummy_prob], axis=1)
# self_lottery_differences_all = self_lottery_differences_all.merge(survey, on='number', how='left')

X_pred_self = self_lottery_differences_all[['dwell_time_ASPS_ACPS'] + list(dummy_ind.columns) + list(dummy_prob.columns)]
# X_pred_self = pd.concat([X_pred_self, X_pred_self[control_variables]], axis=1)
X_pred_self = sm.add_constant(X_pred_self, has_constant='add') # add a first column full of ones to account for intercept of regression
y_pred_self = self_lottery_differences_all['valuation_ASPS_ACPS']
model_pred_self = sm.OLS(y_pred_self, X_pred_self).fit(cov_type='cluster', cov_kwds={'groups': self_lottery_differences_all['number']}) # cluster at individual level
print(model_pred_self.summary())


dummy_ind = pd.get_dummies(charity_lottery_differences_all['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob = pd.get_dummies(charity_lottery_differences_all['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
charity_lottery_differences_all = pd.concat([charity_lottery_differences_all, dummy_ind, dummy_prob], axis=1)
# charity_lottery_differences = charity_lottery_differences.merge(survey, on='number', how='left')

X_pred_charity = charity_lottery_differences_all[['dwell_time_ACPC_ASPC'] + list(dummy_ind.columns) + list(dummy_prob.columns)]
# X_pred_charity = pd.concat([X_pred_charity, X_pred_charity[control_variables]], axis=1)
X_pred_charity = sm.add_constant(X_pred_charity, has_constant='add') # add a first column full of ones to account for intercept of regression
y_pred_charity = charity_lottery_differences_all['valuation_ACPC_ASPC']
model_pred_charity = sm.OLS(y_pred_charity, X_pred_charity).fit(cov_type='cluster', cov_kwds={'groups': charity_lottery_differences_all['number']}) # cluster at individual level
print(model_pred_charity.summary())



# EDRP

dummy_ind_EDRP = pd.get_dummies(self_lottery_differences_all_EDRP['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob_EDRP = pd.get_dummies(self_lottery_differences_all_EDRP['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
self_lottery_differences_all_EDRP = pd.concat([self_lottery_differences_all_EDRP, dummy_ind_EDRP, dummy_prob_EDRP], axis=1)
# self_lottery_differences_all_EDRP = self_lottery_differences_all_EDRP.merge(survey, on='number', how='left')

X_pred_self_EDRP = self_lottery_differences_all_EDRP[['dwell_time_ASPS_ACPS'] + list(dummy_ind_EDRP.columns) + list(dummy_prob_EDRP.columns)]
# X_pred_self = pd.concat([X_pred_self_EDRP, X_pred_self[control_variables]], axis=1)
X_pred_self_EDRP = sm.add_constant(X_pred_self_EDRP, has_constant='add') # add a first column full of ones to account for intercept of regression
y_pred_self_EDRP = self_lottery_differences_all_EDRP['valuation_ASPS_ACPS']
model_pred_self_EDRP = sm.OLS(y_pred_self_EDRP, X_pred_self_EDRP).fit(cov_type='cluster', cov_kwds={'groups': self_lottery_differences_all_EDRP['number']}) # cluster at individual level
print(model_pred_self_EDRP.summary())


dummy_ind_EDRP = pd.get_dummies(charity_lottery_differences_all_EDRP['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob_EDRP = pd.get_dummies(charity_lottery_differences_all_EDRP['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
charity_lottery_differences_all_EDRP = pd.concat([charity_lottery_differences_all_EDRP, dummy_ind_EDRP, dummy_prob_EDRP], axis=1)
# charity_lottery_differences_all_EDRP = charity_lottery_differences_all_EDRP.merge(survey, on='number', how='left')

X_pred_charity_EDRP = charity_lottery_differences_all_EDRP[['dwell_time_ACPC_ASPC'] + list(dummy_ind_EDRP.columns) + list(dummy_prob_EDRP.columns)]
# X_pred_charity = pd.concat([X_pred_charity_EDRP, X_pred_charity[control_variables]], axis=1)
X_pred_charity_EDRP = sm.add_constant(X_pred_charity_EDRP, has_constant='add') # add a first column full of ones to account for intercept of regression
y_pred_charity_EDRP = charity_lottery_differences_all_EDRP['valuation_ACPC_ASPC']
model_pred_charity_EDRP = sm.OLS(y_pred_charity_EDRP, X_pred_charity_EDRP).fit(cov_type='cluster', cov_kwds={'groups': charity_lottery_differences_all_EDRP['number']}) # cluster at individual level
print(model_pred_charity_EDRP.summary())




## PLOTS 
# individuals = self_lottery_differences_all['number'].unique()
# colors = cm.rainbow(np.linspace(0, 1, len(individuals)))
# individual_color_map = dict(zip(individuals, colors))
# self_lottery_differences_all['color'] = self_lottery_differences_all['number'].map(individual_color_map)

# plt.scatter(self_lottery_differences_all['valuation_ASPS_ACPS'], self_lottery_differences_all['dwell_time_ASPS_ACPS'], c=self_lottery_differences_all['color'])

# plt.xlabel('Valuation')
# plt.ylabel('Dwell time')
# plt.title('(Within-subj) Diff SELF Valuation x Dwell time (' +str(attention_type) +')')
# plt.grid(True)
# plt.show()


# individuals = charity_lottery_differences_all['number'].unique()
# colors = cm.rainbow(np.linspace(0, 1, len(individuals)))
# individual_color_map = dict(zip(individuals, colors))
# charity_lottery_differences_all['color'] = charity_lottery_differences_all['number'].map(individual_color_map)

# plt.scatter(charity_lottery_differences_all['valuation_ACPC_ASPC'], charity_lottery_differences_all['dwell_time_ACPC_ASPC'], c=charity_lottery_differences_all['color'])

# plt.xlabel('Valuation')
# plt.ylabel('Dwell time')
# plt.title('(Within-subj) Diff CHARITY Valuation x Dwell time (' +str(attention_type) +')')
# plt.grid(True)
# plt.show()



# plt.scatter(self_lottery_differences_all['prob_option_A'], self_lottery_differences_all['valuation_ASPS_ACPS'], c='red', label='self')
# plt.scatter(charity_lottery_differences_all['prob_option_A'], charity_lottery_differences_all['valuation_ACPC_ASPC'], c='green', label='charity')

# plt.xlabel('Prob')
# plt.ylabel('Valuation diff')
# plt.title('(Within-subj) Diff Valuation')
# plt.grid(True)
# plt.legend()
# plt.show()

# self_lottery_differences_grouped = self_lottery_differences_all.groupby('prob_option_A')['valuation_ASPS_ACPS'].median()
# charity_lottery_differences_grouped = charity_lottery_differences_all.groupby('prob_option_A')['valuation_ACPC_ASPC'].median()

# plt.scatter(self_lottery_differences_grouped.index, self_lottery_differences_grouped, c='red', label='self')
# plt.scatter(charity_lottery_differences_grouped.index, charity_lottery_differences_grouped, c='green', label='charity')

# plt.xlabel('Prob')
# plt.ylabel('Valuation diff')
# plt.title('(Within-subj) Diff Valuation')
# plt.grid(True)
# plt.legend()
# plt.show()



