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
MSP_excl = 1 # Put 0 if include MSP calib in analysis and 1 if we exclude them 
by_ind = 0 # Put 0 if no display of individual plots and 1 if display 
attention_type = 'relative' # relative for % of total time and 'absolute' for raw time
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
    

# Remove outliers? 

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


if outliers ==1:
    outliers_all = outliers_data.index.tolist() + outliers_data_total.index.tolist() + outliers_data_relative.index.tolist()
    data_for_plot = data_for_plot.drop(outliers_all)
    data_for_plot = data_for_plot.reset_index(drop=True)
else: 
    pass 

# Get different cases

ASPS = data_for_plot[(data_for_plot['charity'] == 0) & (data_for_plot['tradeoff'] == 0)]
ACPC = data_for_plot[(data_for_plot['charity'] == 1) & (data_for_plot['tradeoff'] == 0)]
ASPC = data_for_plot[(data_for_plot['charity'] == 1) & (data_for_plot['tradeoff'] == 1)]
ACPS = data_for_plot[(data_for_plot['charity'] == 0) & (data_for_plot['tradeoff'] == 1)]


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


#######
#### ENLEVER DONNEES ASSOCIEES
#######
#######

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
        color = ['black', 'dimgray', 'darkgray', 'lightgrey']) 
plt.xlabel('Case order')
plt.ylabel('Mean dwell time in %')
plt.title('Mean attention per cas order')
plt.savefig('Attention case order H2.png', dpi=1200)
plt.show()

plt.bar(['0.05', '0.1', '0.25', '0.5', '0.75', '0.9', '0.95'], attention_per_proba.mean()) 
plt.xlabel('Probability')
plt.ylabel('Mean dwell time in %')
plt.title('Mean attention per probability')
plt.savefig('Attention probability H2.png', dpi=1200)
plt.show()

# data_for_plot_2 = data_for_plot
# data_for_plot_2['first case'] = [data_for_plot_2['order of cases'][i][0] for i in range(len(data_for_plot_2))]
# not_first_case = data_for_plot_2.loc[data_for_plot_2['first case'] != data_for_plot_2['case']] 
# data_for_plot_2 = data_for_plot_2.drop(not_first_case.index)

# ASPS_between = data_for_plot_2[(data_for_plot_2['charity'] == 0) & (data_for_plot_2['tradeoff'] == 0)]
# ACPC_between = data_for_plot_2[(data_for_plot_2['charity'] == 1) & (data_for_plot_2['tradeoff'] == 0)]
# ASPC_between = data_for_plot_2[(data_for_plot_2['charity'] == 1) & (data_for_plot_2['tradeoff'] == 1)]
# ACPS_between = data_for_plot_2[(data_for_plot_2['charity'] == 0) & (data_for_plot_2['tradeoff'] == 1)]

# Get differences

self_lottery_attention = pd.concat([ASPS, ACPS], ignore_index = True)
charity_lottery_attention = pd.concat([ACPC, ASPC], ignore_index=True)

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



# %%
# =============================================================================
# Categorisation Excuse-driven risk preferences
# =============================================================================

EDRP_self = []
EDRP_charity = []

altruistic_self = []
altruistic_charity = []

for i in data_for_plot['number'].unique():
    self_diff = self_lottery_differences.loc[self_lottery_differences['number'] == i,['valuation_ACPS_ASPS']].mean()
    charity_diff = charity_lottery_differences.loc[charity_lottery_differences['number'] == i,['valuation_ASPC_ACPC']].mean()

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


X_EDRP_total = data_autre_principal[data_autre_principal['number'].isin(EDRP_total)]

data_X_EDRP_total = data_for_plot[data_for_plot['number'].isin(EDRP_total)]

X_NO_EDRP_total = data_autre_principal[~data_autre_principal['number'].isin(EDRP_total)]
data_NO_EDRP = data_for_plot[~data_for_plot['number'].isin(data_X_EDRP_total['number'])]



ASPS_EDRP = data_X_EDRP_total[(data_X_EDRP_total['charity'] == 0) & (data_X_EDRP_total['tradeoff'] == 0)]
ACPC_EDRP = data_X_EDRP_total[(data_X_EDRP_total['charity'] == 1) & (data_X_EDRP_total['tradeoff'] == 0)]
ASPC_EDRP = data_X_EDRP_total[(data_X_EDRP_total['charity'] == 1) & (data_X_EDRP_total['tradeoff'] == 1)]
ACPS_EDRP = data_X_EDRP_total[(data_X_EDRP_total['charity'] == 0) & (data_X_EDRP_total['tradeoff'] == 1)]


self_lottery_differences_attention_EDRP = self_lottery_differences_attention[self_lottery_differences_attention['number'].isin(EDRP_total)]
charity_lottery_differences_attention_EDRP = charity_lottery_differences_attention[charity_lottery_differences_attention['number'].isin(EDRP_total)]


self_lottery_differences_attention_altruistic = self_lottery_differences_attention[self_lottery_differences_attention['number'].isin(altruistic_total)]
charity_lottery_differences_attention_altruistic = charity_lottery_differences_attention[charity_lottery_differences_attention['number'].isin(altruistic_total)]


# %%
# =============================================================================
# VISUALISE DATA 
# =============================================================================

# Plot Attention (WITHIN-SUBJECT)

# Plot all attention, without differentiating probabilities

# error_attention = [np.std(ASPS['dwell_time']), np.std(ACPS['dwell_time']), 
#                   np.std(ACPC['dwell_time']), np.std(ASPC['dwell_time'])]

# plt.bar(['ASPS', 'ACPS', 'ACPC', 'ASPC'], mean_attentions, color = ['blue', 'red', 'green', 'orange']) 
# plt.errorbar(['ASPS', 'ACPS', 'ACPC', 'ASPC'], mean_attentions, error_attention, ecolor = 'black', fmt='none')
# plt.xlabel('Cas')
# plt.ylabel('Moyenne Attention en s')
# plt.title('Attention par cas, probabilités confondues')
# plt.savefig('Bar all Lottery H2.png', dpi=1200)
# plt.show()

# relative

error_attention_relative = [np.std(ASPS['dwell_time_relative']), np.std(ACPS['dwell_time_relative']), 
                  np.std(ACPC['dwell_time_relative']), np.std(ASPC['dwell_time_relative'])]

mean_attentions_relative = [ASPS.groupby('prob_option_A')['dwell_time_relative'].mean().mean(), 
                            ACPS.groupby('prob_option_A')['dwell_time_relative'].mean().mean(),
                            ACPC.groupby('prob_option_A')['dwell_time_relative'].mean().mean(),
                            ASPC.groupby('prob_option_A')['dwell_time_relative'].mean().mean()]

plt.bar(['ASPS', 'ACPS', 'ACPC', 'ASPC'], mean_attentions_relative, color = ['blue', 'red', 'green', 'orange']) 
plt.errorbar(['ASPS', 'ACPS', 'ACPC', 'ASPC'], mean_attentions_relative, error_attention_relative, ecolor = 'black', fmt='none')
plt.xlabel('Cas')
plt.ylabel('Moyenne Attention en s (relative)')
plt.title('Attention par cas, probabilités confondues')
plt.savefig('Bar all Lottery H2.png', dpi=1200)
plt.show()

# EDRP
error_attention_relative_EDRP = [np.std(ASPS_EDRP['dwell_time_relative']), np.std(ACPS_EDRP['dwell_time_relative']), 
                  np.std(ACPC_EDRP['dwell_time_relative']), np.std(ASPC_EDRP['dwell_time_relative'])]

mean_attentions_relative_EDRP = [ASPS_EDRP.groupby('prob_option_A')['dwell_time_relative'].mean().mean(), 
                            ACPS_EDRP.groupby('prob_option_A')['dwell_time_relative'].mean().mean(),
                            ACPC_EDRP.groupby('prob_option_A')['dwell_time_relative'].mean().mean(),
                            ASPC_EDRP.groupby('prob_option_A')['dwell_time_relative'].mean().mean()]

plt.bar(['ASPS', 'ACPS', 'ACPC', 'ASPC'], mean_attentions_relative_EDRP, color = ['blue', 'red', 'green', 'orange']) 
plt.errorbar(['ASPS', 'ACPS', 'ACPC', 'ASPC'], mean_attentions_relative_EDRP, error_attention_relative_EDRP, ecolor = 'black', fmt='none')
plt.xlabel('Cas')
plt.ylabel('Moyenne Attention en s (relative)')
plt.title('EDRP Attention par cas, probabilités confondues')
plt.savefig('Bar all Lottery EDRP H2.png', dpi=1200)
plt.show()



# Plot the difference of attention 

plt.bar(['Self ($A^{C}(P^{S})-A^{S}(P^{S})$)', 'Charity ($A^{S}(P^{C})-A^{C}(P^{C})$)'], 
        [self_lottery_differences_attention['dwell_time_ACPS_ASPS'].mean(), charity_lottery_differences_attention['dwell_time_ASPC_ACPC'].mean()], 
        color = ['lightskyblue', 'lightgreen']) 
plt.errorbar(['Self ($A^{C}(P^{S})-A^{S}(P^{S})$)', 'Charity ($A^{S}(P^{C})-A^{C}(P^{C})$)'], 
              [self_lottery_differences_attention['dwell_time_ACPS_ASPS'].mean(), charity_lottery_differences_attention['dwell_time_ASPC_ACPC'].mean()], 
              [0.402, 0.602], ecolor = 'black', fmt='none', alpha=0.7)
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery type')
plt.ylabel('Difference in dwell time (trad - no trad) in %')
plt.title('Difference in dwell time across probabilities H2')
plt.savefig('Bar diff type Lottery H2.png', dpi=1200)
plt.show()
 
# ERDP

plt.bar(['Self ($A^{C}(P^{S})-A^{S}(P^{S})$)', 'Charity ($A^{S}(P^{C})-A^{C}(P^{C})$)'], 
        [self_lottery_differences_attention_EDRP['dwell_time_ACPS_ASPS'].mean(), charity_lottery_differences_attention_EDRP['dwell_time_ASPC_ACPC'].mean()], 
        color = ['lightskyblue', 'lightgreen']) 
plt.errorbar(['Self ($A^{C}(P^{S})-A^{S}(P^{S})$)', 'Charity ($A^{S}(P^{C})-A^{C}(P^{C})$)'], 
              [self_lottery_differences_attention_EDRP['dwell_time_ACPS_ASPS'].mean(), charity_lottery_differences_attention_EDRP['dwell_time_ASPC_ACPC'].mean()], 
              [0.597, 1.028], ecolor = 'black', fmt='none', alpha=0.7)
# plt.errorbar(['Self ($Y^{C}(P^{S})-Y^{S}(P^{S})$)', 'Charity ($Y^{S}(P^{C})-Y^{C}(P^{C})$)'], 
#               [self_lottery_differences_attention_EDRP['dwell_time_ACPS_ASPS'].mean(), charity_lottery_differences_attention_EDRP['dwell_time_ASPC_ACPC'].mean()], 
#               [np.std(self_lottery_differences_attention_EDRP['dwell_time_ACPS_ASPS']), np.std(charity_lottery_differences_attention_EDRP['dwell_time_ASPC_ACPC'])], ecolor = 'black', fmt='none', alpha=0.7)
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery type')
plt.ylabel('Difference in dwell time (trad - no trad) in %')
plt.title('Difference in dwell time for EDRP subjects H2')
plt.savefig('Bar diff type Lottery EDRP H2.png', dpi=1200)
plt.show()

# altruistic 

plt.bar(['Self ($A^{C}(P^{S})-A^{S}(P^{S})$)', 'Charity ($A^{S}(P^{C})-A^{C}(P^{C})$)'], 
        [self_lottery_differences_attention_altruistic['dwell_time_ACPS_ASPS'].mean(), charity_lottery_differences_attention_altruistic['dwell_time_ASPC_ACPC'].mean()], 
        color = ['lightskyblue', 'lightgreen']) 
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery type')
plt.ylabel('Difference in dwell time (trad - no trad) in %')
plt.title('Difference in dwell time for altruistic subjects H2')
plt.show()


# Histo attention 


plt.hist(data_for_plot['dwell_time'], bins=50)
plt.xlabel('Attention en s')
plt.ylabel('Frequence')
plt.title('Histo attention')
plt.show()

plt.hist(data_for_plot['dwell_time_relative'], bins=50)
plt.xlabel('Attention en s')
plt.ylabel('Frequence')
plt.title('Histo attention')
plt.show()

plt.hist(data_for_plot['total_time_spent_s'], bins=50)
plt.xlabel('Attention en s')
plt.ylabel('Frequence')
plt.title('Histo attention')
plt.show()

average_attention_ASPS = ASPS.groupby('prob_option_A')['dwell_time_relative'].median()
average_attention_ACPC = ACPC.groupby('prob_option_A')['dwell_time_relative'].median()
average_attention_ACPS = ACPS.groupby('prob_option_A')['dwell_time_relative'].median()
average_attention_ASPC = ASPC.groupby('prob_option_A')['dwell_time_relative'].median()

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


# if by_ind == 1: 
#     for i in range(1, data_for_plot['number'].nunique()+1):
#         ASPS_att_ind = ASPS.loc[ASPS['number'] == i, ['prob_option_A', 'dwell_time']] 
#         ASPS_att_ind = ASPS_att_ind.sort_values(by=['prob_option_A'])
#         ACPC_att_ind = ACPC.loc[ACPC['number'] == i, ['prob_option_A', 'dwell_time']] 
#         ACPC_att_ind = ACPC_att_ind.sort_values(by=['prob_option_A'])
#         ASPC_att_ind = ASPC.loc[ASPC['number'] == i, ['prob_option_A', 'dwell_time']]
#         ASPC_att_ind = ASPC_att_ind.sort_values(by=['prob_option_A'])
#         ACPS_att_ind = ACPS.loc[ACPS['number'] == i, ['prob_option_A', 'dwell_time']] 
#         ACPS_att_ind = ACPS_att_ind.sort_values(by=['prob_option_A'])
        
#         plt.plot(ASPS_att_ind['prob_option_A'], ASPS_att_ind['dwell_time'], label='ASPS', color='blue', marker='o', linestyle='-')
#         plt.plot(ACPS_att_ind['prob_option_A'], ACPS_att_ind['dwell_time'], label='ACPS', color='red', marker='o', linestyle='-')
#         plt.plot(ASPC_att_ind['prob_option_A'], ASPC_att_ind['dwell_time'], label='ASPC', color='orange', marker='o', linestyle='-')
#         plt.plot(ACPC_att_ind['prob_option_A'], ACPC_att_ind['dwell_time'], label='ACPC', color='green', marker='o', linestyle='-')
#         plt.title('Individual (' +str(attention_type) +') ' + str(i))
#         plt.grid(True)
#         plt.legend()
#         plt.show()
# else: 
#     pass




# %%
# =============================================================================
# ANALYSE DATA 
# =============================================================================

data_for_analysis = pd.concat([ASPS, ACPC, ASPC, ACPS], ignore_index=True)
data_for_analysis_EDRP = pd.concat([ASPS_EDRP, ACPC_EDRP, ASPC_EDRP, ACPS_EDRP], ignore_index=True)

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





md_2 = smf.mixedlm("dwell_time_relative ~ charity + tradeoff + interaction", data_for_analysis, groups=data_for_analysis["number"])
mdf_2 = md_2.fit()
print(mdf_2.summary())

md_3 = smf.mixedlm("dwell_time_relative ~ case_order", data_for_analysis, groups=data_for_analysis["number"])
mdf_3 = md_3.fit()
print(mdf_3.summary())

