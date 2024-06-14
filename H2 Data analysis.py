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
from scipy.stats import ttest_ind
from matplotlib.patches import Patch
import ast 

threshold_EDRP = 2.1

path = '/Users/carolinepioger/Desktop/ALL collection' # change to yours :)

# Get dataframes
data = pd.read_csv(path + '/dataset.csv' )
data_autre = pd.read_csv(path + '/criterion info data.csv')
survey = pd.read_csv(path + '/survey data.csv')

  

# %%
# =============================================================================
# REMOVING OUTLIERS
# =============================================================================

# Remove outliers

dwell_mean = data['dwell_time_absolute'].mean()
dwell_std = np.std(data['dwell_time_absolute'])
outliers_data = data[(data['dwell_time_absolute'] < dwell_mean - 3 * dwell_std)
                         | (data['dwell_time_absolute'] > dwell_mean + 3 * dwell_std)]

dwell_mean_total = data['total_time_spent_s'].mean()
dwell_std_total = np.std(data['total_time_spent_s'])
outliers_data_total = data[(data['total_time_spent_s'] < dwell_mean_total - 3 * dwell_std_total)
                         | (data['total_time_spent_s'] > dwell_mean_total + 3 * dwell_std_total)]

dwell_mean_relative = data['dwell_time_relative'].mean()
dwell_std_relative = np.std(data['dwell_time_relative'])
outliers_data_relative = data[(data['dwell_time_relative'] < dwell_mean_relative - 3 * dwell_std_relative)
                         | (data['dwell_time_relative'] > dwell_mean_relative + 3 * dwell_std_relative)]

outliers_all = np.union1d(outliers_data.index, np.union1d(outliers_data_total.index, outliers_data_relative.index))

# Remove outliers and associated data 

associated_outliers = []

for index in outliers_all:
    outlier_row = data.iloc[index]

    if outlier_row['case'] == 'ASPS':
        corresponding_row = data[
            (data['case'] == 'ACPS') &
            (data['number'] == outlier_row['number']) & 
            (data['prob_option_A'] == outlier_row['prob_option_A'])
        ]
    elif outlier_row['case'] == 'ACPS' :
        corresponding_row = data[
            (data['case'] == 'ASPS') &
            (data['number'] == outlier_row['number']) & 
            (data['prob_option_A'] == outlier_row['prob_option_A'])
        ]
    elif outlier_row['case'] == 'ACPC' :
        corresponding_row = data[
            (data['case'] == 'ASPC') &
            (data['number'] == outlier_row['number']) & 
            (data['prob_option_A'] == outlier_row['prob_option_A'])
        ]
    elif outlier_row['case'] == 'ASPC':
        corresponding_row = data[
            (data['case'] == 'ACPC') &
            (data['number'] == outlier_row['number']) & 
            (data['prob_option_A'] == outlier_row['prob_option_A'])
        ]

    associated_outliers.append(corresponding_row.index[0])


remove_all = np.union1d(associated_outliers,outliers_all)

data = data.drop(remove_all)
data = data.reset_index(drop=True)


# %%
# =============================================================================
# Remove participants with criteria 
# =============================================================================


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


# Remove participants with censored values in part 2
exclude_participants = data_autre.loc[data_autre['censored_calibration'] == 1, 'id'] 
data_censored = data[data['id'].isin(exclude_participants) == True]

data = data.drop(data[data['id'].isin(exclude_participants) == True].index)
data = data.reset_index(drop=True)


# Remove participants with mutliple switchoint (MSP) in part 2

exclude_participants_2 = data_autre.loc[data_autre['censored_calibration'] == 'MSP', 'id'] 

data = data.drop(data[data['id'].isin(exclude_participants_2) == True].index)
data = data.reset_index(drop=True)


data_for_plot = data

# Convert order of cases in string 
for i in range(len(data_for_plot)):
    data_for_plot['order of cases'][i] = ast.literal_eval(data_for_plot['order of cases'][i])

# %%
# =============================================================================
# GET ALL DATA
# =============================================================================


# Get different cases

ASPS = data_for_plot[(data_for_plot['charity'] == 0) & (data_for_plot['tradeoff'] == 0)]
ACPC = data_for_plot[(data_for_plot['charity'] == 1) & (data_for_plot['tradeoff'] == 0)]
ASPC = data_for_plot[(data_for_plot['charity'] == 1) & (data_for_plot['tradeoff'] == 1)]
ACPS = data_for_plot[(data_for_plot['charity'] == 0) & (data_for_plot['tradeoff'] == 1)]


# Difference data for VALUATION
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


# Get attention values

# attention_ASPS = ASPS.groupby('prob_option_A')['dwell_time']
# attention_ACPC = ACPC.groupby('prob_option_A')['dwell_time']
# attention_ACPS = ACPS.groupby('prob_option_A')['dwell_time']
# attention_ASPC = ASPC.groupby('prob_option_A')['dwell_time']

# mean_attention_ASPS = attention_ASPS.mean()
# mean_attention_ACPC = attention_ACPC.mean()
# mean_attention_ACPS = attention_ACPS.mean()
# mean_attention_ASPC = attention_ASPC.mean()

# mean_attentions = [mean_attention_ASPS.mean(), mean_attention_ACPS.mean(), 
#                    mean_attention_ACPC.mean(), mean_attention_ASPC.mean()]

attention_per_proba = data_for_plot.groupby('prob_option_A')['dwell_time_relative']

first_case = data_for_plot[data_for_plot['case_order']==1]
second_case = data_for_plot[data_for_plot['case_order']==2]
third_case = data_for_plot[data_for_plot['case_order']==3]
fourth_case = data_for_plot[data_for_plot['case_order']==4]

plt.bar(['first', 'second', 'third', 'fourth'], [first_case['dwell_time_relative'].mean(), second_case['dwell_time_relative'].mean(), 
                                               third_case['dwell_time_relative'].mean(), fourth_case['dwell_time_relative'].mean()], 
        color = ['dimgray', 'darkgray', 'silver', 'lightgrey']) 
plt.errorbar(['first', 'second', 'third', 'fourth'], 
             [first_case['dwell_time_relative'].mean(), second_case['dwell_time_relative'].mean(), third_case['dwell_time_relative'].mean(), fourth_case['dwell_time_relative'].mean()], 
              [first_case['dwell_time_relative'].std(), second_case['dwell_time_relative'].std(), third_case['dwell_time_relative'].std(), fourth_case['dwell_time_relative'].std()], 
              ecolor = 'black', fmt='none', alpha=0.7, label = 'std ind level')
plt.xlabel('Case order')
plt.ylabel('Mean attention in %')
plt.title('Mean attention per case order')
plt.savefig('Attention case order H2.png', dpi=1200)
plt.show()

# ACROSS CONDITIONS

plt.bar(['0.05', '0.1', '0.25', '0.5', '0.75', '0.9', '0.95'], attention_per_proba.mean(), 
        color = ['darkgoldenrod', 'goldenrod', 'gold', 'khaki', 'beige', 'papayawhip', 'peachpuff']) 
plt.errorbar(['0.05', '0.1', '0.25', '0.5', '0.75', '0.9', '0.95'], attention_per_proba.mean(), 
             attention_per_proba.std(), 
             ecolor = 'black', fmt='none', alpha=0.5, label='std')
plt.xlabel('Probability')
plt.ylabel('Mean attention in %')
plt.title('Mean attention per probability for all')
plt.savefig('Attention probability H2.png', dpi=1200)
plt.show()



# Get differences for ATTENTION 

self_lottery_attention = pd.concat([ASPS, ACPS], ignore_index = True)
charity_lottery_attention = pd.concat([ACPC, ASPC], ignore_index=True)
no_tradeoff_lottery_attention = pd.concat([ASPS, ACPC], ignore_index=True)

self_lottery_differences_attention = pd.DataFrame(columns=['number', 'prob_option_A'])

for i in self_lottery_attention['number'].unique():
    individual = self_lottery_attention.loc[self_lottery_attention['number'] == i, ['case', 'prob_option_A', 'dwell_time_relative']] 
    individual_difference = individual.pivot(index='prob_option_A', columns='case', values='dwell_time_relative')
    individual_difference['dwell_time_ACPS_ASPS'] = individual_difference['ACPS'] - individual_difference['ASPS']
    individual_difference['number'] = i
    individual_difference.reset_index(inplace=True)
    # individual_difference.columns = individual_difference.columns.droplevel(1)
    self_lottery_differences_attention = pd.concat([self_lottery_differences_attention, individual_difference[['number', 'prob_option_A', 'dwell_time_ACPS_ASPS']]], ignore_index=True)

charity_lottery_differences_attention = pd.DataFrame(columns=['number', 'prob_option_A'])

for i in charity_lottery_attention['number'].unique():
    individual = charity_lottery_attention.loc[charity_lottery_attention['number'] == i, ['case', 'prob_option_A', 'dwell_time_relative']] 
    individual_difference = individual.pivot(index='prob_option_A', columns='case', values='dwell_time_relative')
    individual_difference['dwell_time_ASPC_ACPC'] = individual_difference['ASPC'] - individual_difference['ACPC']
    individual_difference['number'] = i
    individual_difference.reset_index(inplace=True)
    # individual_difference.columns = individual_difference.columns.droplevel(1)
    charity_lottery_differences_attention = pd.concat([charity_lottery_differences_attention, individual_difference[['number', 'prob_option_A', 'dwell_time_ASPC_ACPC']]], ignore_index=True)

no_tradeoff_lottery_differences_attention = pd.DataFrame(columns=['number', 'prob_option_A'])

for i in no_tradeoff_lottery_attention['number'].unique():
    individual = no_tradeoff_lottery_attention.loc[no_tradeoff_lottery_attention['number'] == i, ['case', 'prob_option_A', 'dwell_time_relative']] 
    individual_difference = individual.pivot(index='prob_option_A', columns='case', values='dwell_time_relative')
    try: 
        individual_difference['dwell_time_ACPC_ASPS'] = individual_difference['ACPC'] - individual_difference['ASPS']
        individual_difference['number'] = i
        individual_difference.reset_index(inplace=True)
        # individual_difference.columns = individual_difference.columns.droplevel(1)
        no_tradeoff_lottery_differences_attention = pd.concat([no_tradeoff_lottery_differences_attention, individual_difference[['number', 'prob_option_A', 'dwell_time_ACPC_ASPS']]], ignore_index=True)
    except KeyError: # since we don't remove for ACPC vs ASPS, sometimes it may give error
        pass


# %%
# =============================================================================
# Categorisation Excuse-driven risk preferences
# =============================================================================


EDRP_self = []
EDRP_charity = []

altruistic_self = []
altruistic_charity = []

for i in data_for_plot['number'].unique():
    self_diff = self_lottery_differences.loc[self_lottery_differences['number'] == i,['valuation_ACPS_ASPS']].mean() # mean across probabilities
    charity_diff = charity_lottery_differences.loc[charity_lottery_differences['number'] == i,['valuation_ASPC_ACPC']].mean() # mean across probabilities

    if self_diff.item() > threshold_EDRP :
        EDRP_self.append(i)
    elif self_diff.item() < - threshold_EDRP :
        altruistic_self.append(i)
    if charity_diff.item() < - threshold_EDRP :
        EDRP_charity.append(i)
    if charity_diff.item() > threshold_EDRP :
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

self_lottery_difference_EDRP = self_lottery_differences[self_lottery_differences['number'].isin(EDRP_total)]
charity_lottery_differences_EDRP = charity_lottery_differences[charity_lottery_differences['number'].isin(EDRP_total)]


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

self_lottery_differences_attention_EDRP = self_lottery_differences_attention[self_lottery_differences_attention['number'].isin(EDRP_total)]
charity_lottery_differences_attention_EDRP = charity_lottery_differences_attention[charity_lottery_differences_attention['number'].isin(EDRP_total)]
no_tradeoff_lottery_differences_attention_EDRP = no_tradeoff_lottery_differences_attention[no_tradeoff_lottery_differences_attention['number'].isin(EDRP_total)]

self_lottery_differences_attention_no_EDRP = self_lottery_differences_attention[self_lottery_differences_attention['number'].isin(no_EDRP)]
charity_lottery_differences_attention_no_EDRP = charity_lottery_differences_attention[charity_lottery_differences_attention['number'].isin(no_EDRP)]
no_tradeoff_lottery_differences_attention_no_EDRP = no_tradeoff_lottery_differences_attention[no_tradeoff_lottery_differences_attention['number'].isin(no_EDRP)]

self_lottery_differences_attention_altruistic = self_lottery_differences_attention[self_lottery_differences_attention['number'].isin(altruistic_total)]
charity_lottery_differences_attention_altruistic = charity_lottery_differences_attention[charity_lottery_differences_attention['number'].isin(altruistic_total)]
no_tradeoff_lottery_differences_attention_altruistic = no_tradeoff_lottery_differences_attention[no_tradeoff_lottery_differences_attention['number'].isin(altruistic_total)]


# %%
# =============================================================================
# Data for censored participants
# =============================================================================

ASPS_censored = data_censored[(data_censored['charity'] == 0) & (data_censored['tradeoff'] == 0)]
ACPC_censored = data_censored[(data_censored['charity'] == 1) & (data_censored['tradeoff'] == 0)]
ASPC_censored = data_censored[(data_censored['charity'] == 1) & (data_censored['tradeoff'] == 1)]
ACPS_censored = data_censored[(data_censored['charity'] == 0) & (data_censored['tradeoff'] == 1)]


# Difference data
self_lottery_attention_censored = pd.concat([ASPS_censored, ACPS_censored], ignore_index = True)
charity_lottery_attention_censored = pd.concat([ACPC_censored, ASPC_censored], ignore_index=True)
no_tradeoff_lottery_attention_censored = pd.concat([ASPS_censored, ACPC_censored], ignore_index=True)

self_lottery_differences_attention_censored = pd.DataFrame(columns=['number', 'prob_option_A'])

for i in self_lottery_attention_censored['number'].unique():
    individual = self_lottery_attention_censored.loc[self_lottery_attention_censored['number'] == i, ['case', 'prob_option_A', 'dwell_time_relative']] 
    individual_difference = individual.pivot(index='prob_option_A', columns='case', values='dwell_time_relative')
    individual_difference['dwell_time_ACPS_ASPS'] = individual_difference['ACPS'] - individual_difference['ASPS']
    individual_difference['number'] = i
    individual_difference.reset_index(inplace=True)
    # individual_difference.columns = individual_difference.columns.droplevel(1)
    self_lottery_differences_attention_censored = pd.concat([self_lottery_differences_attention_censored, individual_difference[['number', 'prob_option_A', 'dwell_time_ACPS_ASPS']]], ignore_index=True)

charity_lottery_differences_attention_censored = pd.DataFrame(columns=['number', 'prob_option_A'])

for i in charity_lottery_attention_censored['number'].unique():
    individual = charity_lottery_attention_censored.loc[charity_lottery_attention_censored['number'] == i, ['case', 'prob_option_A', 'dwell_time_relative']] 
    individual_difference = individual.pivot(index='prob_option_A', columns='case', values='dwell_time_relative')
    individual_difference['dwell_time_ASPC_ACPC'] = individual_difference['ASPC'] - individual_difference['ACPC']
    individual_difference['number'] = i
    individual_difference.reset_index(inplace=True)
    # individual_difference.columns = individual_difference.columns.droplevel(1)
    charity_lottery_differences_attention_censored = pd.concat([charity_lottery_differences_attention_censored, individual_difference[['number', 'prob_option_A', 'dwell_time_ASPC_ACPC']]], ignore_index=True)

no_tradeoff_lottery_differences_attention_censored = pd.DataFrame(columns=['number', 'prob_option_A'])

for i in no_tradeoff_lottery_attention_censored['number'].unique():
    individual = no_tradeoff_lottery_attention_censored.loc[no_tradeoff_lottery_attention_censored['number'] == i, ['case', 'prob_option_A', 'dwell_time_relative']] 
    individual_difference = individual.pivot(index='prob_option_A', columns='case', values='dwell_time_relative')
    individual_difference['dwell_time_ACPC_ASPS'] = individual_difference['ACPC'] - individual_difference['ASPS']
    individual_difference['number'] = i
    individual_difference.reset_index(inplace=True)
    # individual_difference.columns = individual_difference.columns.droplevel(1)
    no_tradeoff_lottery_differences_attention_censored = pd.concat([no_tradeoff_lottery_differences_attention_censored, individual_difference[['number', 'prob_option_A', 'dwell_time_ACPC_ASPS']]], ignore_index=True)

# EDRP + CENSORED
no_tradeoff_lottery_differences_attention_ALL = pd.concat([no_tradeoff_lottery_differences_attention_EDRP, no_tradeoff_lottery_differences_attention_censored], ignore_index=True)
self_lottery_differences_attention_ALL = pd.concat([self_lottery_differences_attention_EDRP, self_lottery_differences_attention_censored], ignore_index=True)
charity_lottery_differences_attention_ALL = pd.concat([charity_lottery_differences_attention_EDRP, charity_lottery_differences_attention_censored], ignore_index=True)


attention_per_proba_censored = data_censored.groupby('prob_option_A')['dwell_time_relative']

plt.bar(['0.05', '0.1', '0.25', '0.5', '0.75', '0.9', '0.95'], attention_per_proba_censored.mean(), 
        color = ['darkgoldenrod', 'goldenrod', 'gold', 'khaki', 'beige', 'papayawhip', 'peachpuff']) 
plt.xlabel('Probability')
plt.ylabel('Mean atention time in %')
plt.title('Mean attention per probability for Censored')
plt.savefig('Attention probability CENSORED H2.png', dpi=1200)
plt.show()


plt.bar(['$A^{C}(P^{C})-A^{S}(P^{S}$)', '$A^{C}(P^{S})-A^{S}(P^{S})$', '$A^{S}(P^{C})-A^{C}(P^{C})$'], 
        [no_tradeoff_lottery_differences_attention_censored['dwell_time_ACPC_ASPS'].mean(), 
          self_lottery_differences_attention_censored['dwell_time_ACPS_ASPS'].mean(), 
          charity_lottery_differences_attention_censored['dwell_time_ASPC_ACPC'].mean()], 
        color = ['bisque', 'lightskyblue', 'lightgreen']) 
plt.errorbar(['$A^{C}(P^{C})-A^{S}(P^{S}$)', '$A^{C}(P^{S})-A^{S}(P^{S})$', '$A^{S}(P^{C})-A^{C}(P^{C})$'], 
              [no_tradeoff_lottery_differences_attention_censored['dwell_time_ACPC_ASPS'].mean(), self_lottery_differences_attention_censored['dwell_time_ACPS_ASPS'].mean(), 
               charity_lottery_differences_attention_censored['dwell_time_ASPC_ACPC'].mean()], 
              [0.507, 0.611, 0.633], ecolor = 'black', fmt='none', alpha=0.7)
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery type')
plt.ylabel('Difference in attention (trad - no trad) in %')
plt.title('Difference in attention across probabilities for Censored')
plt.savefig('Bar diff type Lottery CENSORED H2.png', dpi=1200)
plt.show()

plt.bar(['$A^{C}(P^{C})-A^{S}(P^{S}$)', '$A^{C}(P^{S})-A^{S}(P^{S})$', '$A^{S}(P^{C})-A^{C}(P^{C})$'], 
        [no_tradeoff_lottery_differences_attention_ALL['dwell_time_ACPC_ASPS'].mean(), 
          self_lottery_differences_attention_ALL['dwell_time_ACPS_ASPS'].mean(), 
          charity_lottery_differences_attention_ALL['dwell_time_ASPC_ACPC'].mean()], 
        color = ['bisque', 'lightskyblue', 'lightgreen']) 
plt.errorbar(['$A^{C}(P^{C})-A^{S}(P^{S}$)', '$A^{C}(P^{S})-A^{S}(P^{S})$', '$A^{S}(P^{C})-A^{C}(P^{C})$'], 
              [no_tradeoff_lottery_differences_attention_ALL['dwell_time_ACPC_ASPS'].mean(), 
               self_lottery_differences_attention_ALL['dwell_time_ACPS_ASPS'].mean(), 
               charity_lottery_differences_attention_ALL['dwell_time_ASPC_ACPC'].mean()], 
              [0.340, 0.407, 0.461], ecolor = 'black', fmt='none', alpha=0.7)
plt.axhline(y=0, color='grey', linestyle='--')
plt.ylim(-2.75, 0.5)
plt.xlabel('Lottery type')
plt.ylabel('Difference in attention (trad - no trad) in %')
plt.title('Difference in attention across probabilities for EDRP and Censored')
plt.savefig('Bar diff type Lottery EDRP + CENSORED H2.png', dpi=1200)
plt.show()


# %%
# =============================================================================
# VISUALISE DATA 
# =============================================================================

# Plot Attention relative

# error_attention_relative = [np.std(ASPS['dwell_time_relative']), np.std(ACPS['dwell_time_relative']), 
#                   np.std(ACPC['dwell_time_relative']), np.std(ASPC['dwell_time_relative'])]

# mean_attentions_relative = [ASPS.groupby('prob_option_A')['dwell_time_relative'].mean().mean(), 
#                             ACPS.groupby('prob_option_A')['dwell_time_relative'].mean().mean(),
#                             ACPC.groupby('prob_option_A')['dwell_time_relative'].mean().mean(),
#                             ASPC.groupby('prob_option_A')['dwell_time_relative'].mean().mean()]

# plt.bar(['$A^{S}(P^{S})$', '$A^{C}(P^{S})$', '$A^{C}(P^{C})$', '$A^{S}(P^{C})$'], mean_attentions_relative, color = ['blue', 'dodgerblue', 'green', 'limegreen']) 
# plt.errorbar(['$A^{S}(P^{S})$', '$A^{C}(P^{S})$', '$A^{C}(P^{C})$', '$A^{S}(P^{C})$'], mean_attentions_relative, error_attention_relative, ecolor = 'black', fmt='none', alpha=0.5, label='std')
# plt.xlabel('Case')
# plt.ylabel('Mean attention in s')
# plt.title('Attention per case, across probabilities')
# plt.legend()
# plt.savefig('Bar all Lottery H2.png', dpi=1200)
# plt.show()


# EDRP
# error_attention_relative_EDRP = [np.std(ASPS_EDRP['dwell_time_relative']), np.std(ACPS_EDRP['dwell_time_relative']), 
#                   np.std(ACPC_EDRP['dwell_time_relative']), np.std(ASPC_EDRP['dwell_time_relative'])]

# mean_attentions_relative_EDRP = [ASPS_EDRP.groupby('prob_option_A')['dwell_time_relative'].mean().mean(), 
#                             ACPS_EDRP.groupby('prob_option_A')['dwell_time_relative'].mean().mean(),
#                             ACPC_EDRP.groupby('prob_option_A')['dwell_time_relative'].mean().mean(),
#                             ASPC_EDRP.groupby('prob_option_A')['dwell_time_relative'].mean().mean()]

# plt.bar(['$A^{S}(P^{S})$', '$A^{C}(P^{S})$', '$A^{C}(P^{C})$', '$A^{S}(P^{C})$'], mean_attentions_relative_EDRP, color = ['blue', 'dodgerblue', 'green', 'limegreen']) 
# plt.errorbar(['$A^{S}(P^{S})$', '$A^{C}(P^{S})$', '$A^{C}(P^{C})$', '$A^{S}(P^{C})$'], mean_attentions_relative_EDRP, error_attention_relative_EDRP, ecolor = 'black', fmt='none', alpha=0.5, label='std')
# plt.xlabel('Case')
# plt.ylabel('Mean attention in s')
# plt.title('Attention per case, across probabilities for EDRP subjects')
# plt.legend()
# plt.savefig('Bar all Lottery EDRP H2.png', dpi=1200)
# plt.show()



# Plot the difference of attention 

plt.bar(['$A^{C}(P^{C})-A^{S}(P^{S}$)','$A^{C}(P^{S})-A^{S}(P^{S})$', '$A^{S}(P^{C})-A^{C}(P^{C})$'], 
        [no_tradeoff_lottery_differences_attention['dwell_time_ACPC_ASPS'].mean(), 
         self_lottery_differences_attention['dwell_time_ACPS_ASPS'].mean(), 
         charity_lottery_differences_attention['dwell_time_ASPC_ACPC'].mean()], 
        color = ['bisque', 'lightskyblue', 'lightgreen']) 
plt.errorbar(['$A^{C}(P^{C})-A^{S}(P^{S}$)','$A^{C}(P^{S})-A^{S}(P^{S})$', '$A^{S}(P^{C})-A^{C}(P^{C})$'], 
              [no_tradeoff_lottery_differences_attention['dwell_time_ACPC_ASPS'].mean(), self_lottery_differences_attention['dwell_time_ACPS_ASPS'].mean(), charity_lottery_differences_attention['dwell_time_ASPC_ACPC'].mean()], 
              [0.343, 0.322, 0.4015], ecolor = 'black', fmt='none', alpha=0.7, label = 'std ind level')
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery type')
plt.ylabel('Difference in attention (trad - no trad) in %')
plt.title('Difference in attention across probabilities H2')
plt.legend()
plt.savefig('Bar diff type Lottery H2.png', dpi=1200)
plt.show()


 
# ERDP

plt.bar(['$A^{C}(P^{C})-A^{S}(P^{S}$)', '$A^{C}(P^{S})-A^{S}(P^{S})$', '$A^{S}(P^{C})-A^{C}(P^{C})$'], 
        [no_tradeoff_lottery_differences_attention_EDRP['dwell_time_ACPC_ASPS'].mean(), 
         self_lottery_differences_attention_EDRP['dwell_time_ACPS_ASPS'].mean(), 
         charity_lottery_differences_attention_EDRP['dwell_time_ASPC_ACPC'].mean()], 
        color = ['bisque', 'lightskyblue', 'lightgreen']) 
plt.errorbar(['$A^{C}(P^{C})-A^{S}(P^{S}$)', '$A^{C}(P^{S})-A^{S}(P^{S})$', '$A^{S}(P^{C})-A^{C}(P^{C})$'], 
              [no_tradeoff_lottery_differences_attention_EDRP['dwell_time_ACPC_ASPS'].mean(), self_lottery_differences_attention_EDRP['dwell_time_ACPS_ASPS'].mean(), charity_lottery_differences_attention_EDRP['dwell_time_ASPC_ACPC'].mean()], 
              [0.513, 0.565, 0.7405], ecolor = 'black', fmt='none', alpha=0.7, label = 'std ind level')
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery type')
plt.ylabel('Difference in attention (trad - no trad) in %')
plt.title('Difference in attention for EDRP subjects H2')
plt.legend()
plt.savefig('Bar diff type Lottery EDRP H2.png', dpi=1200)
plt.show()

# altruistic 

plt.bar(['$A^{C}(P^{C})-A^{S}(P^{S}$)', '$A^{C}(P^{S})-A^{S}(P^{S})$', '$A^{S}(P^{C})-A^{C}(P^{C})$'], 
        [no_tradeoff_lottery_differences_attention_altruistic['dwell_time_ACPC_ASPS'].mean(), 
         self_lottery_differences_attention_altruistic['dwell_time_ACPS_ASPS'].mean(), 
         charity_lottery_differences_attention_altruistic['dwell_time_ASPC_ACPC'].mean()], 
        color = ['bisque', 'lightskyblue', 'lightgreen']) 
plt.errorbar(['$A^{C}(P^{C})-A^{S}(P^{S}$)', '$A^{C}(P^{S})-A^{S}(P^{S})$', '$A^{S}(P^{C})-A^{C}(P^{C})$'], 
              [no_tradeoff_lottery_differences_attention_altruistic['dwell_time_ACPC_ASPS'].mean(), self_lottery_differences_attention_altruistic['dwell_time_ACPS_ASPS'].mean(), charity_lottery_differences_attention_altruistic['dwell_time_ASPC_ACPC'].mean()], 
              [0.723, 0.675, 0.786], ecolor = 'black', fmt='none', alpha=0.7, label = 'std ind level')
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery type')
plt.ylabel('Difference in attention (trad - no trad) in %')
plt.title('Difference in attention for altruistic subjects H2')
plt.legend()
plt.savefig('Bar diff type Lottery Altruistic H2.png', dpi=1200)
plt.show()

# EDRP VS CENSORED
lottery_types = ['$A^{C}(P^{C})-A^{S}(P^{S}$)', '$A^{C}(P^{S})-A^{S}(P^{S})$', '$A^{S}(P^{C})-A^{C}(P^{C})$']

EDRP_means = [
    no_tradeoff_lottery_differences_attention_EDRP['dwell_time_ACPC_ASPS'].mean(),
    self_lottery_differences_attention_EDRP['dwell_time_ACPS_ASPS'].mean(),
    charity_lottery_differences_attention_EDRP['dwell_time_ASPC_ACPC'].mean()
]
EDRP_errors = [0.513, 0.565, 0.7405]

censored_means = [
    no_tradeoff_lottery_differences_attention_censored['dwell_time_ACPC_ASPS'].mean(),
    self_lottery_differences_attention_censored['dwell_time_ACPS_ASPS'].mean(),
    charity_lottery_differences_attention_censored['dwell_time_ASPC_ACPC'].mean()
]
censored_errors = [0.507, 0.611, 0.633]

x = np.arange(len(lottery_types))
width = 0.35

plt.bar(x - width/2, EDRP_means, width, yerr=EDRP_errors, capsize=5, color=['bisque', 'lightskyblue', 'lightgreen'], label='Adaptive')
plt.bar(x + width/2, censored_means, width, yerr=censored_errors, capsize=5, color=['bisque', 'lightskyblue', 'lightgreen'], hatch="//", label='Censored')
plt.xlabel('Lottery type')
plt.ylabel('Difference in attention (trad - no trad) in %')
plt.title('Difference in attention for Adaptive and Censored subjects H2')
plt.xticks(x, lottery_types)
plt.axhline(y=0, color='grey', linestyle='--')
proxy_artists = [
    Patch(facecolor='white', edgecolor='black', label='Adaptive'),
    Patch(facecolor='white', edgecolor='black', hatch="//", label='Censored')
]
plt.ylim(-5, 1.25)
plt.legend(handles=proxy_artists)
plt.savefig('Merged Attention Adapted and Censored.png', dpi=1200)
plt.show()


# NO ERDP

# plt.bar(['Self ($A^{C}(P^{S})-A^{S}(P^{S})$)', 'Charity ($A^{S}(P^{C})-A^{C}(P^{C})$)'], 
#         [self_lottery_differences_attention_no_EDRP['dwell_time_ACPS_ASPS'].mean(), charity_lottery_differences_attention_no_EDRP['dwell_time_ASPC_ACPC'].mean()], 
#         color = ['lightskyblue', 'lightgreen']) 
# plt.errorbar(['Self ($A^{C}(P^{S})-A^{S}(P^{S})$)', 'Charity ($A^{S}(P^{C})-A^{C}(P^{C})$)'], 
#               [self_lottery_differences_attention_no_EDRP['dwell_time_ACPS_ASPS'].mean(), charity_lottery_differences_attention_no_EDRP['dwell_time_ASPC_ACPC'].mean()], 
#               [0.615, 0.787], ecolor = 'black', fmt='none', alpha=0.7)
# # plt.errorbar(['Self ($Y^{C}(P^{S})-Y^{S}(P^{S})$)', 'Charity ($Y^{S}(P^{C})-Y^{C}(P^{C})$)'], 
# #               [self_lottery_differences_attention_EDRP['dwell_time_ACPS_ASPS'].mean(), charity_lottery_differences_attention_EDRP['dwell_time_ASPC_ACPC'].mean()], 
# #               [np.std(self_lottery_differences_attention_EDRP['dwell_time_ACPS_ASPS']), np.std(charity_lottery_differences_attention_EDRP['dwell_time_ASPC_ACPC'])], ecolor = 'black', fmt='none', alpha=0.7)
# plt.axhline(y=0, color='grey', linestyle='--')
# plt.xlabel('Lottery type')
# plt.ylabel('Difference in attention (trad - no trad) in %')
# plt.title('Difference in attention for NO EDRP subjects H2')
# plt.savefig('Bar diff type Lottery NO EDRP H2.png', dpi=1200)
# plt.show()


# Histo attention 


# plt.hist(data_for_plot['dwell_time'], bins=50)
# plt.xlabel('Attention en s')
# plt.ylabel('Frequence')
# plt.title('Histo attention')
# plt.show()

# plt.hist(data_for_plot['dwell_time_relative'], bins=50)
# plt.xlabel('Attention en s')
# plt.ylabel('Frequence')
# plt.title('Histo attention')
# plt.show()

# plt.hist(data_for_plot['total_time_spent_s'], bins=50)
# plt.xlabel('Attention en s')
# plt.ylabel('Frequence')
# plt.title('Histo attention')
# plt.show()

# ALL 
# average_attention_ASPS = ASPS.groupby('prob_option_A')['dwell_time_relative'].mean()
# average_attention_ACPC = ACPC.groupby('prob_option_A')['dwell_time_relative'].mean()
# average_attention_ACPS = ACPS.groupby('prob_option_A')['dwell_time_relative'].mean()
# average_attention_ASPC = ASPC.groupby('prob_option_A')['dwell_time_relative'].mean()

# all_attention = pd.concat([average_attention_ASPS, average_attention_ACPC, average_attention_ACPS, average_attention_ASPC])
# all_attention = all_attention.groupby('prob_option_A').median()

# plt.plot(average_attention_ASPS.index, average_attention_ASPS, label='$A^{S}(P^{S})$', color='blue', marker='o', linestyle='-')
# plt.plot(average_attention_ACPS.index, average_attention_ACPS, label='$A^{C}(P^{S})$', color='dodgerblue', marker='o', linestyle='-')
# plt.plot(average_attention_ACPC.index, average_attention_ACPC, label='$A^{C}(P^{C})$', color='green', marker='o', linestyle='-')
# plt.plot(average_attention_ASPC.index, average_attention_ASPC, label='$A^{S}(P^{C})$', color='limegreen', marker='o', linestyle='-')

# plt.xlabel('Probability P of Non-Zero Payment')
# plt.ylabel('Mean Attention in %')
# plt.title('Mean Attention per probability')
# plt.grid(True)
# plt.legend()
# plt.savefig('Proba attention diff ALL H2.png', dpi=1200)
# plt.show()




# # EDRP  
# offset = 0.02

# average_attention_ASPS_EDRP = ASPS_EDRP.groupby('prob_option_A')['dwell_time_relative']
# average_attention_ACPC_EDRP = ACPC_EDRP.groupby('prob_option_A')['dwell_time_relative']
# average_attention_ACPS_EDRP = ACPS_EDRP.groupby('prob_option_A')['dwell_time_relative']
# average_attention_ASPC_EDRP = ASPC_EDRP.groupby('prob_option_A')['dwell_time_relative']

# # all_attention_EDRP = pd.concat([average_attention_ASPS_EDRP, average_attention_ACPC_EDRP, average_attention_ACPS_EDRP, average_attention_ASPC_EDRP])
# # all_attention_EDRP = all_attention_EDRP.groupby('prob_option_A').median()

# plt.errorbar(average_attention_ASPS_EDRP.mean().index - offset, average_attention_ASPS_EDRP.mean(), average_attention_ASPS_EDRP.std(), ecolor = 'black', fmt='none', alpha=0.5, label='std')
# plt.plot(average_attention_ASPS_EDRP.mean().index- offset, average_attention_ASPS_EDRP.mean(), label='$A^{S}(P^{S})$', color='blue', marker='o', linestyle='-')

# plt.errorbar(average_attention_ACPS_EDRP.mean().index - offset/2, average_attention_ACPS_EDRP.mean(), average_attention_ACPS_EDRP.std(), ecolor = 'black', fmt='none', alpha=0.5)
# plt.plot(average_attention_ACPS_EDRP.mean().index- offset/2, average_attention_ACPS_EDRP.mean(), label='$A^{C}(P^{S})$', color='dodgerblue', marker='o', linestyle='-')

# plt.errorbar(average_attention_ACPC_EDRP.mean().index + offset/2, average_attention_ACPC_EDRP.mean(), average_attention_ACPC_EDRP.std(), ecolor = 'black', fmt='none', alpha=0.5)
# plt.plot(average_attention_ACPC_EDRP.mean().index+ offset/2, average_attention_ACPC_EDRP.mean(), label='$A^{C}(P^{C})$', color='green', marker='o', linestyle='-')

# plt.errorbar(average_attention_ASPC_EDRP.mean().index + offset, average_attention_ASPC_EDRP.mean(), average_attention_ASPC_EDRP.std(), ecolor = 'black', fmt='none', alpha=0.5)
# plt.plot(average_attention_ASPC_EDRP.mean().index+ offset, average_attention_ASPC_EDRP.mean(), label='$A^{S}(P^{C})$', color='limegreen', marker='o', linestyle='-')

# plt.xlabel('Probability P of Non-Zero Payment')
# plt.ylabel('Mean Attention in %')
# plt.title('Mean Attention per probability EDRP')
# plt.grid(True)
# plt.legend()
# plt.savefig('Proba attention diff EDRP H2.png', dpi=1200)
# plt.show()

# # Censored  
# average_attention_ASPS_censored = ASPS_censored.groupby('prob_option_A')['dwell_time_relative']
# average_attention_ACPC_censored = ACPC_censored.groupby('prob_option_A')['dwell_time_relative']
# average_attention_ACPS_censored = ACPS_censored.groupby('prob_option_A')['dwell_time_relative']
# average_attention_ASPC_censored = ASPC_censored.groupby('prob_option_A')['dwell_time_relative']

# # all_attention_censored = pd.concat([average_attention_ASPS_censored, average_attention_ACPC_censored, average_attention_ACPS_censored, average_attention_ASPC_censored])
# # all_attention_censored = all_attention_censored.groupby('prob_option_A').median()

# plt.errorbar(average_attention_ASPS_censored.mean().index - offset, average_attention_ASPS_censored.mean(), average_attention_ASPS_censored.std(), ecolor = 'black', fmt='none', alpha=0.5, label='std')
# plt.plot(average_attention_ASPS_censored.mean().index- offset, average_attention_ASPS_censored.mean(), label='$A^{S}(P^{S})$', color='blue', marker='o', linestyle='-')

# plt.errorbar(average_attention_ACPS_censored.mean().index - offset/2, average_attention_ACPS_censored.mean(), average_attention_ACPS_censored.std(), ecolor = 'black', fmt='none', alpha=0.5)
# plt.plot(average_attention_ACPS_censored.mean().index- offset/2, average_attention_ACPS_censored.mean(), label='$A^{C}(P^{S})$', color='dodgerblue', marker='o', linestyle='-')

# plt.errorbar(average_attention_ACPC_censored.mean().index + offset/2, average_attention_ACPC_censored.mean(), average_attention_ACPC_censored.std(), ecolor = 'black', fmt='none', alpha=0.5)
# plt.plot(average_attention_ACPC_censored.mean().index+ offset/2, average_attention_ACPC_censored.mean(), label='$A^{C}(P^{C})$', color='green', marker='o', linestyle='-')

# plt.errorbar(average_attention_ASPC_censored.mean().index + offset, average_attention_ASPC_censored.mean(), average_attention_ASPC_censored.std(), ecolor = 'black', fmt='none', alpha=0.5)
# plt.plot(average_attention_ASPC_censored.mean().index+ offset, average_attention_ASPC_censored.mean(), label='$A^{S}(P^{C})$', color='limegreen', marker='o', linestyle='-')

# plt.xlabel('Probability P of Non-Zero Payment')
# plt.ylabel('Mean Attention in %')
# plt.title('Mean Attention per probability for Censored')
# plt.grid(True)
# plt.legend()
# plt.savefig('Proba attention diff Censored H2.png', dpi=1200)
# plt.show()



# Hist ALL difference of attention
# plt.hist([self_lottery_differences['valuation_ACPS_ASPS'], charity_lottery_differences['valuation_ASPC_ACPC']], 
#         bins = 20, color = ['lightskyblue', 'lightgreen'], label = ['Self lottery', 'Charity lottery']) 
# plt.xlabel('Difference in lottery valuation (trad - no trad)')
# plt.ylabel('Frequency')
# plt.title('Difference in valuation across probabilities')
# plt.legend()
# plt.savefig('Histo Valuation diff H1.png', dpi=1200)
# plt.show()

# diff attention across probabilities 

# ALLLL 
offset_2 = 0.02
plt.axhline(y=0, color='grey', linestyle='--')

diff_proba_self_attention = self_lottery_differences_attention.groupby('prob_option_A')['dwell_time_ACPS_ASPS']
diff_proba_charity_attention = charity_lottery_differences_attention.groupby('prob_option_A')['dwell_time_ASPC_ACPC']
diff_proba_no_tradeoff_attention = no_tradeoff_lottery_differences_attention.groupby('prob_option_A')['dwell_time_ACPC_ASPS']

plt.errorbar(diff_proba_no_tradeoff_attention.mean().index - offset_2/2, diff_proba_no_tradeoff_attention.mean(), diff_proba_no_tradeoff_attention.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_no_tradeoff_attention.mean().index - offset_2/2, diff_proba_no_tradeoff_attention.mean(), label='$A^{C}(P^{C})-A^{S}(P^{S})$', color='bisque', marker='o', linestyle='-')

plt.errorbar(diff_proba_self_attention.mean().index, diff_proba_self_attention.mean(), diff_proba_self_attention.std(), ecolor = 'black', fmt='none', alpha=0.4, label='std')
plt.plot(diff_proba_self_attention.mean().index, diff_proba_self_attention.mean(), label='$A^{C}(P^{S})-A^{S}(P^{S})$', color='dodgerblue', marker='o', linestyle='-')

plt.errorbar(diff_proba_charity_attention.mean().index + offset_2/2, diff_proba_charity_attention.mean(), diff_proba_charity_attention.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_charity_attention.mean().index + offset_2/2, diff_proba_charity_attention.mean(), label='$A^{S}(P^{C})-A^{C}(P^{C})$', color='limegreen', marker='o', linestyle='-')

x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Difference in attention (trad - no trad) in %')
plt.title('Attentional differences for principal analysis')
plt.legend()
plt.savefig('Attention across proba Lottery difference ALL H2.png', dpi=1200)
plt.show()

# EDRP
offset_2 = 0.02
plt.axhline(y=0, color='grey', linestyle='--')

diff_proba_self_attention_EDRP = self_lottery_differences_attention_EDRP.groupby('prob_option_A')['dwell_time_ACPS_ASPS']
diff_proba_charity_attention_EDRP = charity_lottery_differences_attention_EDRP.groupby('prob_option_A')['dwell_time_ASPC_ACPC']
diff_proba_no_tradeoff_attention_EDRP = no_tradeoff_lottery_differences_attention_EDRP.groupby('prob_option_A')['dwell_time_ACPC_ASPS']

plt.errorbar(diff_proba_no_tradeoff_attention_EDRP.mean().index - offset_2/2, diff_proba_no_tradeoff_attention_EDRP.mean(), diff_proba_no_tradeoff_attention_EDRP.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_no_tradeoff_attention_EDRP.mean().index - offset_2/2, diff_proba_no_tradeoff_attention_EDRP.mean(), label='$A^{C}(P^{C})-A^{S}(P^{S})$', color='bisque', marker='o', linestyle='-')

plt.errorbar(diff_proba_self_attention_EDRP.mean().index, diff_proba_self_attention_EDRP.mean(), diff_proba_self_attention_EDRP.std(), ecolor = 'black', fmt='none', alpha=0.4, label='std')
plt.plot(diff_proba_self_attention_EDRP.mean().index, diff_proba_self_attention_EDRP.mean(), label='$A^{C}(P^{S})-A^{S}(P^{S})$', color='dodgerblue', marker='o', linestyle='-')

plt.errorbar(diff_proba_charity_attention_EDRP.mean().index + offset_2/2, diff_proba_charity_attention_EDRP.mean(), diff_proba_charity_attention_EDRP.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_charity_attention_EDRP.mean().index + offset_2/2, diff_proba_charity_attention_EDRP.mean(), label='$A^{S}(P^{C})-A^{C}(P^{C})$', color='limegreen', marker='o', linestyle='-')

x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Difference in attention (trad - no trad) in %')
plt.title('Attentional differences for Excuse-driven subjects')
plt.legend()
plt.savefig('Attention across proba Lottery difference EDRP H2.png', dpi=1200)
plt.show()

# ALTRUISTIC
offset_2 = 0.02
plt.axhline(y=0, color='grey', linestyle='--')

diff_proba_self_attention_altruistic = self_lottery_differences_attention_altruistic.groupby('prob_option_A')['dwell_time_ACPS_ASPS']
diff_proba_charity_attention_altruistic = charity_lottery_differences_attention_altruistic.groupby('prob_option_A')['dwell_time_ASPC_ACPC']
diff_proba_no_tradeoff_attention_altruistic = no_tradeoff_lottery_differences_attention_altruistic.groupby('prob_option_A')['dwell_time_ACPC_ASPS']

plt.errorbar(diff_proba_no_tradeoff_attention_altruistic.mean().index - offset_2/2, diff_proba_no_tradeoff_attention_altruistic.mean(), diff_proba_no_tradeoff_attention_altruistic.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_no_tradeoff_attention_altruistic.mean().index - offset_2/2, diff_proba_no_tradeoff_attention_altruistic.mean(), label='$A^{C}(P^{C})-A^{S}(P^{S})$', color='bisque', marker='o', linestyle='-')

plt.errorbar(diff_proba_self_attention_altruistic.mean().index, diff_proba_self_attention_altruistic.mean(), diff_proba_self_attention_altruistic.std(), ecolor = 'black', fmt='none', alpha=0.4, label='std')
plt.plot(diff_proba_self_attention_altruistic.mean().index, diff_proba_self_attention_altruistic.mean(), label='$A^{C}(P^{S})-A^{S}(P^{S})$', color='dodgerblue', marker='o', linestyle='-')

plt.errorbar(diff_proba_charity_attention_altruistic.mean().index + offset_2/2, diff_proba_charity_attention_altruistic.mean(), diff_proba_charity_attention_altruistic.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_charity_attention_altruistic.mean().index + offset_2/2, diff_proba_charity_attention_altruistic.mean(), label='$A^{S}(P^{C})-A^{C}(P^{C})$', color='limegreen', marker='o', linestyle='-')

x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Difference in attention (trad - no trad) in %')
plt.title('Attentional differences for Altruistic subjects')
plt.legend()
plt.savefig('Attention across proba Lottery difference Altruistic H2.png', dpi=1200)
plt.show()


# CENSORED

offset_2 = 0.02
plt.axhline(y=0, color='grey', linestyle='--')

diff_proba_self_attention_censored = self_lottery_differences_attention_censored.groupby('prob_option_A')['dwell_time_ACPS_ASPS']
diff_proba_charity_attention_censored = charity_lottery_differences_attention_censored.groupby('prob_option_A')['dwell_time_ASPC_ACPC']
diff_proba_no_tradeoff_attention_censored = no_tradeoff_lottery_differences_attention_censored.groupby('prob_option_A')['dwell_time_ACPC_ASPS']

plt.errorbar(diff_proba_no_tradeoff_attention_censored.mean().index - offset_2/2, diff_proba_no_tradeoff_attention_censored.mean(), diff_proba_no_tradeoff_attention_censored.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_no_tradeoff_attention_censored.mean().index - offset_2/2, diff_proba_no_tradeoff_attention_censored.mean(), label='$A^{C}(P^{C})-A^{S}(P^{S})$', color='bisque', marker='o', linestyle='-')

plt.errorbar(diff_proba_self_attention_censored.mean().index, diff_proba_self_attention_censored.mean(), diff_proba_self_attention_censored.std(), ecolor = 'black', fmt='none', alpha=0.4, label='std')
plt.plot(diff_proba_self_attention_censored.mean().index, diff_proba_self_attention_censored.mean(), label='$A^{C}(P^{S})-A^{S}(P^{S})$', color='dodgerblue', marker='o', linestyle='-')

plt.errorbar(diff_proba_charity_attention_censored.mean().index + offset_2/2, diff_proba_charity_attention_censored.mean(), diff_proba_charity_attention_censored.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_charity_attention_censored.mean().index + offset_2/2, diff_proba_charity_attention_censored.mean(), label='$A^{S}(P^{C})-A^{C}(P^{C})$', color='limegreen', marker='o', linestyle='-')

x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Difference in attention (trad - no trad) in %')
plt.title('Attentional differences for Censored subjects')
plt.legend()
plt.savefig('Attention across proba Lottery difference CENSORED H2.png', dpi=1200)
plt.show()

# CENSORED + EDRP

offset_2 = 0.02
plt.axhline(y=0, color='grey', linestyle='--')

diff_proba_self_attention_ALL = self_lottery_differences_attention_ALL.groupby('prob_option_A')['dwell_time_ACPS_ASPS']
diff_proba_charity_attention_ALL = charity_lottery_differences_attention_ALL.groupby('prob_option_A')['dwell_time_ASPC_ACPC']
diff_proba_no_tradeoff_attention_ALL = no_tradeoff_lottery_differences_attention_ALL.groupby('prob_option_A')['dwell_time_ACPC_ASPS']

plt.errorbar(diff_proba_no_tradeoff_attention_ALL.mean().index - offset_2/2, diff_proba_no_tradeoff_attention_ALL.mean(), diff_proba_no_tradeoff_attention_ALL.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_no_tradeoff_attention_ALL.mean().index - offset_2/2, diff_proba_no_tradeoff_attention_ALL.mean(), label='$A^{C}(P^{C})-A^{S}(P^{S})$', color='bisque', marker='o', linestyle='-')

plt.errorbar(diff_proba_self_attention_censored.mean().index, diff_proba_self_attention_ALL.mean(), diff_proba_self_attention_ALL.std(), ecolor = 'black', fmt='none', alpha=0.4, label='std')
plt.plot(diff_proba_self_attention_censored.mean().index, diff_proba_self_attention_ALL.mean(), label='$A^{C}(P^{S})-A^{S}(P^{S})$', color='dodgerblue', marker='o', linestyle='-')

plt.errorbar(diff_proba_charity_attention_ALL.mean().index + offset_2/2, diff_proba_charity_attention_ALL.mean(), diff_proba_charity_attention_ALL.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_charity_attention_ALL.mean().index + offset_2/2, diff_proba_charity_attention_ALL.mean(), label='$A^{S}(P^{C})-A^{C}(P^{C})$', color='limegreen', marker='o', linestyle='-')

x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Difference in attention (trad - no trad) in %')
plt.title('Attentional differences for Excuse-driven and Censored subjects')
plt.legend()
plt.savefig('Attention across proba Lottery difference EDRP + CENSORED H2.png', dpi=1200)
plt.show()



# %%
# =============================================================================
# ANALYSE DATA 
# =============================================================================

data_for_analysis = pd.concat([ASPS, ACPC, ASPC, ACPS], ignore_index=True)
data_for_analysis_EDRP = pd.concat([ASPS_EDRP, ACPC_EDRP, ASPC_EDRP, ACPS_EDRP], ignore_index=True)
data_for_analysis_altruistic = pd.concat([ASPS_altruistic, ACPC_altruistic, ASPC_altruistic, ACPS_altruistic], ignore_index=True)
data_for_analysis_no_EDRP = pd.concat([ASPS_no_EDRP, ACPC_no_EDRP, ASPC_no_EDRP, ACPS_no_EDRP], ignore_index=True)

data_for_analysis_censored = pd.concat([ASPS_censored, ACPC_censored, ASPC_censored, ACPS_censored], ignore_index=True)
data_for_analysis_all_and_censored = pd.concat([ASPS, ACPC, ASPC, ACPS, ASPS_censored, ACPC_censored, ASPC_censored, ACPS_censored], ignore_index=True)
data_for_analysis_EDRP_and_censored = pd.concat([ASPS_EDRP, ACPC_EDRP, ASPC_EDRP, ACPS_EDRP, ASPS_censored, ACPC_censored, ASPC_censored, ACPS_censored], ignore_index=True)

# data_for_analysis_EDRP['dwell_time_absolute'].mean()
# data_for_analysis_censored['dwell_time_absolute'].mean()

### differences between all and censored

t_statistic_att_self, p_value_att_self = ttest_ind(self_lottery_differences_attention['dwell_time_ACPS_ASPS'], self_lottery_differences_attention_censored['dwell_time_ACPS_ASPS'])
print('t-test and p-value of Self difference between All vs censored')
print(t_statistic_att_self, p_value_att_self)
print()

t_statistic_att_charity, p_value_att_charity = ttest_ind(charity_lottery_differences_attention['dwell_time_ASPC_ACPC'], charity_lottery_differences_attention_censored['dwell_time_ASPC_ACPC'])
print('t-test and p-value of Charity difference between All vs censored')
print(t_statistic_att_charity, p_value_att_charity)
print()

### differences between EDRP and censored

t_statistic_att_notrade_2, p_value_att_notrade_2 = ttest_ind(no_tradeoff_lottery_differences_attention_EDRP.dropna()['dwell_time_ACPC_ASPS'], no_tradeoff_lottery_differences_attention_censored.dropna()['dwell_time_ACPC_ASPS'])
print('t-test and p-value of No tradeoff difference between EDRP vs censored')
print(t_statistic_att_notrade_2, p_value_att_notrade_2)
print()

t_statistic_att_self_2, p_value_att_self_2 = ttest_ind(self_lottery_differences_attention_EDRP['dwell_time_ACPS_ASPS'], self_lottery_differences_attention_censored['dwell_time_ACPS_ASPS'])
print('t-test and p-value of Self difference between EDRP vs censored')
print(t_statistic_att_self_2, p_value_att_self_2)
print()

t_statistic_att_charity_2, p_value_att_charity_2 = ttest_ind(charity_lottery_differences_attention_EDRP['dwell_time_ASPC_ACPC'], charity_lottery_differences_attention_censored['dwell_time_ASPC_ACPC'])
print('t-test and p-value of Charity difference between EDRP vs censored')
print(t_statistic_att_charity_2, p_value_att_charity_2)
print()


### differences between all and EDRP

t_statistic_att_self_3, p_value_att_self_3 = ttest_ind(self_lottery_differences_attention['dwell_time_ACPS_ASPS'], self_lottery_differences_attention_EDRP['dwell_time_ACPS_ASPS'])
print('t-test and p-value of Self difference between EDRP vs censored')
print(t_statistic_att_self_3, p_value_att_self_3)
print()

t_statistic_att_charity_3, p_value_att_charity_3 = ttest_ind(charity_lottery_differences_attention['dwell_time_ASPC_ACPC'], charity_lottery_differences_attention_EDRP['dwell_time_ASPC_ACPC'])
print('t-test and p-value of Charity difference between all vs EDRP')
print(t_statistic_att_charity_3, p_value_att_charity_3)
print()


######## ATTENTION REGRESSION


# Add fixed effects
dummy_ind = pd.get_dummies(data_for_analysis['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob = pd.get_dummies(data_for_analysis['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
data_for_analysis = pd.concat([data_for_analysis, dummy_ind, dummy_prob], axis=1)

# Add controls 
data_for_analysis = data_for_analysis.merge(survey, on='id', how='left')
control_variables = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + ['NEP_' + str(i) for i in range(1, 16)] + 
                 ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]

# Create the design matrix and dependent variable
X = data_for_analysis[['charity', 'tradeoff', 'interaction', 'case_order'] + list(dummy_ind.columns) + list(dummy_prob.columns)]
# X = data_for_analysis[['charity', 'tradeoff', 'interaction'] + list(dummy_ind.columns)]
X = pd.concat([X, data_for_analysis[control_variables]], axis=1)
X = sm.add_constant(X, has_constant='add') # add a first column full of ones to account for intercept of regression

# Same process but now dwell_time as dependent variable
y_2 = data_for_analysis['dwell_time_relative']
model_2 = sm.OLS(y_2, X).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis['number']}) # cluster at individual level
print(model_2.summary())


# EDRP

# Add fixed effects
dummy_ind_EDRP = pd.get_dummies(data_for_analysis_EDRP['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob_EDRP = pd.get_dummies(data_for_analysis_EDRP['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
data_for_analysis_EDRP = pd.concat([data_for_analysis_EDRP, dummy_ind_EDRP, dummy_prob_EDRP], axis=1)

# Add controls 
data_for_analysis_EDRP = data_for_analysis_EDRP.merge(survey, on='id', how='left')
control_variables_EDRP = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + ['NEP_' + str(i) for i in range(1, 16)] + 
                 ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]

# Create the design matrix and dependent variable
X_EDRP = data_for_analysis_EDRP[['charity', 'tradeoff', 'interaction', 'case_order'] + list(dummy_ind_EDRP.columns) + list(dummy_prob_EDRP.columns)]
# X = data_for_analysis[['charity', 'tradeoff', 'interaction'] + list(dummy_ind.columns)]
X_EDRP = pd.concat([X_EDRP, data_for_analysis_EDRP[control_variables_EDRP]], axis=1)
X_EDRP = sm.add_constant(X_EDRP, has_constant='add') # add a first column full of ones to account for intercept of regression

# Same process but now dwell_time as dependent variable
y_2_EDRP = data_for_analysis_EDRP['dwell_time_relative']
model_2_EDRP = sm.OLS(y_2_EDRP, X_EDRP).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis_EDRP['number']}) # cluster at individual level
print(model_2_EDRP.summary())


# Altruistic 

dummy_ind_altruistic = pd.get_dummies(data_for_analysis_altruistic['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob_altruistic = pd.get_dummies(data_for_analysis_altruistic['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
data_for_analysis_altruistic = pd.concat([data_for_analysis_altruistic, dummy_ind_altruistic, dummy_prob_altruistic], axis=1)

# Add controls 
data_for_analysis_altruistic = data_for_analysis_altruistic.merge(survey, on='id', how='left')
control_variables_altruistic = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + ['NEP_' + str(i) for i in range(1, 16)] + 
                 ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]

# Create the design matrix and dependent variable
X_altruistic = data_for_analysis_altruistic[['charity', 'tradeoff', 'interaction', 'case_order'] + list(dummy_ind_altruistic.columns) + list(dummy_prob_altruistic.columns)]
# X = data_for_analysis[['charity', 'tradeoff', 'interaction'] + list(dummy_ind.columns)]
X_altruistic = pd.concat([X_altruistic, data_for_analysis_altruistic[control_variables_altruistic]], axis=1)
X_altruistic = sm.add_constant(X_altruistic, has_constant='add') # add a first column full of ones to account for intercept of regression

# Same process but now dwell_time as dependent variable
y_2_altruistic = data_for_analysis_altruistic['dwell_time_relative']
model_2_altruistic = sm.OLS(y_2_altruistic, X_altruistic).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis_altruistic['number']}) # cluster at individual level
print(model_2_altruistic.summary())


# NO EDRP

# Add fixed effects
dummy_ind_no_EDRP = pd.get_dummies(data_for_analysis_no_EDRP['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob_no_EDRP = pd.get_dummies(data_for_analysis_no_EDRP['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
data_for_analysis_no_EDRP = pd.concat([data_for_analysis_no_EDRP, dummy_ind_no_EDRP, dummy_prob_no_EDRP], axis=1)

# Add controls 
data_for_analysis_no_EDRP = data_for_analysis_no_EDRP.merge(survey, on='id', how='left')
control_variables_no_EDRP = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + ['NEP_' + str(i) for i in range(1, 16)] + 
                 ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]

# Create the design matrix and dependent variable
X_no_EDRP = data_for_analysis_no_EDRP[['charity', 'tradeoff', 'interaction', 'case_order'] + list(dummy_ind_no_EDRP.columns) + list(dummy_prob_no_EDRP.columns)]
# X = data_for_analysis[['charity', 'tradeoff', 'interaction'] + list(dummy_ind.columns)]
X_no_EDRP = pd.concat([X_no_EDRP, data_for_analysis_no_EDRP[control_variables_no_EDRP]], axis=1)
X_no_EDRP = sm.add_constant(X_no_EDRP, has_constant='add') # add a first column full of ones to account for intercept of regression

# Same process but now dwell_time as dependent variable
y_2_no_EDRP = data_for_analysis_no_EDRP['dwell_time_relative']
model_2_no_EDRP = sm.OLS(y_2_no_EDRP, X_no_EDRP).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis_no_EDRP['number']}) # cluster at individual level
print(model_2_no_EDRP.summary())



# CENSORED
# Add fixed effects
dummy_ind_censored = pd.get_dummies(data_for_analysis_censored['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob_censored = pd.get_dummies(data_for_analysis_censored['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
data_for_analysis_censored = pd.concat([data_for_analysis_censored, dummy_ind_censored, dummy_prob_censored], axis=1)

# Add controls 
data_for_analysis_censored = data_for_analysis_censored.merge(survey, on='id', how='left')
control_variables_censored = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + ['NEP_' + str(i) for i in range(1, 16)] + 
                 ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]

# Create the design matrix and dependent variable
X_censored = data_for_analysis_censored[['charity', 'tradeoff', 'interaction', 'case_order'] + list(dummy_ind_censored.columns) + list(dummy_prob_censored.columns)]
# X = data_for_analysis[['charity', 'tradeoff', 'interaction'] + list(dummy_ind.columns)]
X_censored = pd.concat([X_censored, data_for_analysis_censored[control_variables_censored]], axis=1)
X_censored = sm.add_constant(X_censored, has_constant='add') # add a first column full of ones to account for intercept of regression

# Same process but now dwell_time as dependent variable
y_2_censored = data_for_analysis_censored['dwell_time_relative']
model_2_censored = sm.OLS(y_2_censored, X_censored).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis_censored['number']}) # cluster at individual level
print(model_2_censored.summary())




###########@
# All and censored
dummy_ind_all_and_censored = pd.get_dummies(data_for_analysis_all_and_censored['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob_all_and_censored = pd.get_dummies(data_for_analysis_all_and_censored['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
data_for_analysis_all_and_censored = pd.concat([data_for_analysis_all_and_censored, dummy_ind_all_and_censored, dummy_prob_all_and_censored], axis=1)

# Add controls 
data_for_analysis_all_and_censored = data_for_analysis_all_and_censored.merge(survey, on='id', how='left')
control_variables_all_and_censored = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + ['NEP_' + str(i) for i in range(1, 16)] + 
                 ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]

# Create the design matrix and dependent variable
X_all_and_censored = data_for_analysis_all_and_censored[['charity', 'tradeoff', 'interaction', 'case_order'] + list(dummy_ind_all_and_censored.columns) + list(dummy_prob_all_and_censored.columns)]
X_all_and_censored = pd.concat([X_all_and_censored, data_for_analysis_all_and_censored[control_variables_all_and_censored]], axis=1)
X_all_and_censored = sm.add_constant(X_all_and_censored, has_constant='add') # add a first column full of ones to account for intercept of regression

# Same process but now dwell_time as dependent variable
y_2_all_and_censored = data_for_analysis_all_and_censored['dwell_time_relative']
model_2_all_and_censored = sm.OLS(y_2_all_and_censored, X_all_and_censored).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis_all_and_censored['number']}) # cluster at individual level
print(model_2_all_and_censored.summary())




###########@
# EDRP and censored
dummy_ind_EDRP_and_censored = pd.get_dummies(data_for_analysis_EDRP_and_censored['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob_EDRP_and_censored = pd.get_dummies(data_for_analysis_EDRP_and_censored['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
data_for_analysis_EDRP_and_censored = pd.concat([data_for_analysis_EDRP_and_censored, dummy_ind_EDRP_and_censored, dummy_prob_EDRP_and_censored], axis=1)

# Add controls 
data_for_analysis_EDRP_and_censored = data_for_analysis_EDRP_and_censored.merge(survey, on='id', how='left')
control_variables_EDRP_and_censored = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + ['NEP_' + str(i) for i in range(1, 16)] + 
                 ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]

# Create the design matrix and dependent variable
X_EDRP_and_censored = data_for_analysis_EDRP_and_censored[['charity', 'tradeoff', 'interaction', 'case_order'] + list(dummy_ind_EDRP_and_censored.columns) + list(dummy_prob_EDRP_and_censored.columns)]
X_EDRP_and_censored = pd.concat([X_EDRP_and_censored, data_for_analysis_EDRP_and_censored[control_variables_EDRP_and_censored]], axis=1)
X_EDRP_and_censored = sm.add_constant(X_EDRP_and_censored, has_constant='add') # add a first column full of ones to account for intercept of regression

# Same process but now dwell_time as dependent variable
y_2_EDRP_and_censored = data_for_analysis_EDRP_and_censored['dwell_time_relative']
model_2_EDRP_and_censored = sm.OLS(y_2_EDRP_and_censored, X_EDRP_and_censored).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis_EDRP_and_censored['number']}) # cluster at individual level
print(model_2_EDRP_and_censored.summary())



# Mixed effects
md_2 = smf.mixedlm("dwell_time_relative ~ charity + tradeoff + interaction", data_for_analysis, groups=data_for_analysis["number"])
mdf_2 = md_2.fit()
print(mdf_2.summary())

md_3 = smf.mixedlm("dwell_time_relative ~ case_order", data_for_analysis, groups=data_for_analysis["number"])
mdf_3 = md_3.fit()
print(mdf_3.summary())

md_4 = smf.mixedlm("dwell_time_relative ~ prob_option_A", data_for_analysis, groups=data_for_analysis["number"])
mdf_4 = md_4.fit()
print(mdf_4.summary())

