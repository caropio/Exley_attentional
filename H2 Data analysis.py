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

# =============================================================================
# UPLOADING DATA
# =============================================================================

# Paths information to upload data
path = '/Users/carolinepioger/Desktop/ALL collection' # change to yours :)

# Upload dataframes
data = pd.read_csv(path + '/dataset.csv' ) # pooled data for analysis
data_autre = pd.read_csv(path + '/criterion info data.csv') # participant-specific info
survey = pd.read_csv(path + '/survey data.csv') # survey information 


################################################
# Take care of string issues
################################################

# Rewrite censored_calibration from string to float 
for i in range(len(data_autre)):
    if data_autre['censored_calibration'][i] == 'MSP':
        pass
    elif isinstance(data_autre['censored_calibration'][i], str):
        data_autre['censored_calibration'][i] = ast.literal_eval(data_autre['censored_calibration'][i])

# Rewrite order of cases from string to arrays
for i in range(len(data)):
    data['order of cases'][i] = ast.literal_eval(data['order of cases'][i])
    

# %%
# =============================================================================
# REMOVING ATTENTIONAL OUTLIERS 
# =============================================================================

# We plot the histograms of attentional data before the removal of outliers
data_before_removal = data

plt.hist(data_before_removal['dwell_time_absolute'], bins=50)
plt.axvline(x=data_before_removal['dwell_time_absolute'].mean(), color='grey', linestyle='--', label = 'Mean = '+ str(np.round(data_before_removal['dwell_time_absolute'].mean(), 1)))
plt.xlabel('Attention in s')
plt.ylabel('Frequency')
plt.title('Histogram of total time spent revealing urn BEFORE removal of outliers')
plt.legend()
plt.show()

plt.hist(data_before_removal['total_time_spent_s'], bins=50)
plt.axvline(x=data_before_removal['total_time_spent_s'].mean(), color='grey', linestyle='--', label = 'Mean = '+ str(np.round(data_before_removal['total_time_spent_s'].mean(), 1)))
plt.xlabel('Attention in s')
plt.ylabel('Frequency')
plt.title('Histogram of total time spent on price list BEFORE removal of outliers')
plt.legend()
plt.show()


plt.hist(data_before_removal['dwell_time_relative'], bins=50)
plt.axvline(x=data_before_removal['dwell_time_relative'].mean(), color='grey', linestyle='--', label = 'Mean = '+ str(np.round(data_before_removal['dwell_time_relative'].mean(), 1)))
plt.xlabel('Attention in %')
plt.ylabel('Frequency')
plt.title('Histogram of attention allocation towards risk information BEFORE removal of outliers')
plt.legend()
plt.show()

# We remove outliers from attentional data (which was pre-registered) using the
# following criteria: attentional data 3 standard deviations away from the general
# attentional data for 1) total time spent revealing urn, 2) total time spent 
# on price list and 3) attention allocation towards risk information (final measure)

# 1) Data 3 std away from total time spent revealing urn - dwell_time_absolute
dwell_mean = data['dwell_time_absolute'].mean()
dwell_std = np.std(data['dwell_time_absolute'])
outliers_data = data[(data['dwell_time_absolute'] < dwell_mean - 3 * dwell_std)
                         | (data['dwell_time_absolute'] > dwell_mean + 3 * dwell_std)]

# 2) Data 3 std away from total time spent on price list - total_time_spent_s
dwell_mean_total = data['total_time_spent_s'].mean()
dwell_std_total = np.std(data['total_time_spent_s'])
outliers_data_total = data[(data['total_time_spent_s'] < dwell_mean_total - 3 * dwell_std_total)
                         | (data['total_time_spent_s'] > dwell_mean_total + 3 * dwell_std_total)]

# 3) Data 3 std away from attention allocation towards risk information - dwell_time_relative
dwell_mean_relative = data['dwell_time_relative'].mean()
dwell_std_relative = np.std(data['dwell_time_relative'])
outliers_data_relative = data[(data['dwell_time_relative'] < dwell_mean_relative - 3 * dwell_std_relative)
                         | (data['dwell_time_relative'] > dwell_mean_relative + 3 * dwell_std_relative)]

# Intersect these outlier data
outliers_all = np.union1d(outliers_data.index, np.union1d(outliers_data_total.index, outliers_data_relative.index))

# We also need to remove attentional data associated to the outliers
# Because we study attentional differences for the same lottery, we need to 
# removed the associated data from the same lottery which was rendered useless 
# without a comparator. For example if the attentional datapoint of individual 
# i for self lottery with self certain amounts for P = 0.05 was removed as an 
# outlier, we also removed the attentional datapoint of individual i for self 
# lottery with charity certain amounts for P = 0.05.

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

# Note that we remove associated data from self and charity attentional differences
# so we discard associated data from no tradeoff attention differences (since it
# isn't part of H2)

# We merge both outliers and associated data and remove it from our data
remove_all = np.union1d(associated_outliers,outliers_all)
data = data.drop(remove_all)
data = data.reset_index(drop=True)

# We plot the histograms of attentional data after the removal of outliers
plt.hist(data['dwell_time_absolute'], bins=50)
plt.axvline(x=data['dwell_time_absolute'].mean(), color='grey', linestyle='--', label = 'Mean = '+ str(np.round(data['dwell_time_absolute'].mean(), 1)))
plt.xlabel('Attention in s')
plt.ylabel('Frequency')
plt.title('Histogram of total time spent revealing urn AFTER removal of outliers')
plt.legend()
plt.show()

plt.hist(data['total_time_spent_s'], bins=50)
plt.axvline(x=data['total_time_spent_s'].mean(), color='grey', linestyle='--', label = 'Mean = '+ str(np.round(data['total_time_spent_s'].mean(), 1)))
plt.xlabel('Attention in s')
plt.ylabel('Frequency')
plt.title('Histogram of total time spent on price list AFTER removal of outliers')
plt.legend()
plt.show()


plt.hist(data['dwell_time_relative'], bins=50)
plt.axvline(x=data['dwell_time_relative'].mean(), color='grey', linestyle='--', label = 'Mean = '+ str(np.round(data['dwell_time_relative'].mean(), 1)))
plt.xlabel('Attention in %')
plt.ylabel('Frequency')
plt.title('Histogram of attention allocation towards risk information AFTER removal of outliers')
plt.legend()
plt.show()

# %%
# =============================================================================
# CATEGORISATION OF PRINCIPAL ANALYSIS AND CENSORED
# =============================================================================

# Let's get our dataframe data but only for principal analysis and censored subjects
data_principal = data

# Get id of censored participants
censored_participants = data_autre.loc[data_autre['censored_calibration'] == 1, 'id'] 

# Use their id to get dataframe (data) specifically of censored participants
data_censored = data[data['id'].isin(censored_participants) == True]
data_censored = data_censored.reset_index(drop=True)

# Remove data from censored participants (in part 2) from data_principal 
data_principal = data_principal.drop(data_principal[data_principal['id'].isin(censored_participants) == True].index)
data_principal = data_principal.reset_index(drop=True)


# Remove data from MSP participants (in part 2) from data_principal 

MSP_participants = data_autre.loc[data_autre['censored_calibration'] == 'MSP', 'id'] 

data_principal = data_principal.drop(data_principal[data_principal['id'].isin(MSP_participants) == True].index)
data_principal = data_principal.reset_index(drop=True)


# Get data (data_autre) for Principal analysis (not including participants with MSP and being censored in calibration price list)
data_autre_principal = data_autre.loc[data_autre['censored_calibration'] == 0] 
data_autre_principal = data_autre_principal.reset_index(drop=True)

# Get data (data_autre) with specifically censored participants 
data_autre_censored = data_autre.loc[data_autre['censored_calibration'] == 1] 
data_autre_censored = data_autre_censored.reset_index(drop=True)


# The dataframe data_principal gives all information for analysis 
# specifically for principal analysis and data_censored specifically for 
# censored individuals 

# The dataframe data_autre_principal gives participant-specific information 
# specifically for principal analysis and data_autre_censored specifically for 
# censored individuals


# %%
# =============================================================================
# GET DIFFERENT CATEGORIES OF DATA 
# =============================================================================

################################################
# FOR PRINCIPAL ANALYSIS
################################################

################################################
# Elicit different cases (YSPS/YCPC/etc)
################################################

ASPS_principal = data_principal[(data_principal['charity'] == 0) & (data_principal['tradeoff'] == 0)] # YSPS
ACPC_principal = data_principal[(data_principal['charity'] == 1) & (data_principal['tradeoff'] == 0)] # YCPC
ASPC_principal = data_principal[(data_principal['charity'] == 1) & (data_principal['tradeoff'] == 1)] # YSPC
ACPS_principal = data_principal[(data_principal['charity'] == 0) & (data_principal['tradeoff'] == 1)] # YCPS

# We group the attentions according to the probabilies involved in the lotteries (7 probabilies)
attention_ASPS = ASPS_principal.groupby('prob_option_A')['dwell_time_relative']
attention_ACPS = ACPS_principal.groupby('prob_option_A')['dwell_time_relative']
attention_ACPC = ACPC_principal.groupby('prob_option_A')['dwell_time_relative']
attention_ASPC = ASPC_principal.groupby('prob_option_A')['dwell_time_relative']

# We find the means of attentions for each probability (for each case) 
mean_attention_ASPS = attention_ASPS.mean()
mean_attention_ACPC = attention_ACPC.mean()
mean_attention_ACPS = attention_ACPS.mean()
mean_attention_ASPC = attention_ASPC.mean()

# We group these means together
mean_attentions = [mean_attention_ASPS.mean(), mean_attention_ACPS.mean(), 
                   mean_attention_ACPC.mean(), mean_attention_ASPC.mean()]

################################################
# Elicit data specifically checking self, charity and no tradeoff differences of H2
################################################

# Self lottery difference is ACPS-ASPS, Charity lottery difference is ASPC-ACPC
# and No Tradeoff difference is ACPC-ASPS

self_lottery_principal = pd.concat([ASPS_principal, ACPS_principal], ignore_index = True)
charity_lottery_principal = pd.concat([ACPC_principal, ASPC_principal], ignore_index=True)
no_tradeoff_lottery_principal = pd.concat([ASPS_principal, ACPC_principal], ignore_index=True)

def lottery_differences(database, var1, var2):
    lottery_differences = pd.DataFrame(columns=['number', 'prob_option_A'])
    for i in database['number'].unique():
        individual = database.loc[database['number'] == i, ['case', 'prob_option_A', 'valuation', 'dwell_time_relative']] 
        individual_difference = individual.pivot(index='prob_option_A', columns='case')
        try: 
            individual_difference[f'valuation_{var1}_{var2}'] = individual_difference['valuation'][var1] - individual_difference['valuation'][var2]
            individual_difference[f'dwell_time_{var1}_{var2}'] = individual_difference['dwell_time_relative'][var1] - individual_difference['dwell_time_relative'][var2]
            individual_difference['number'] = i
            individual_difference.reset_index(inplace=True)
            individual_difference.columns = individual_difference.columns.droplevel(1)
            lottery_differences = pd.concat([lottery_differences, individual_difference[['number', 'prob_option_A', f'valuation_{var1}_{var2}', f'dwell_time_{var1}_{var2}']]], ignore_index=True)
        except KeyError: # since we don't remove for ACPC vs ASPS, sometimes it may give error
            pass
    return lottery_differences
    # gives lottery differences for each probability for both valuation and attention

# Self lottery, charity lottery and no tradeoff differences for Principal Analysis
self_lottery_differences_principal = lottery_differences(self_lottery_principal, 'ACPS', 'ASPS') # gives YCPS-YSPS and ACPS-ASPS
charity_lottery_differences_principal = lottery_differences(charity_lottery_principal, 'ASPC', 'ACPC') # gives YSPC-YCPC and ASPC-ACPC
no_tradeoff_lottery_differences_principal = lottery_differences(no_tradeoff_lottery_principal, 'ACPC', 'ASPS') # gives YCPC-YSPS and ACPC-ASPS


################################################
# FOR CENSORED SUBJECTS
################################################

################################################
# Elicit different cases (YSPS/YCPC/etc)
################################################

ASPS_censored = data_censored[(data_censored['charity'] == 0) & (data_censored['tradeoff'] == 0)]
ACPC_censored = data_censored[(data_censored['charity'] == 1) & (data_censored['tradeoff'] == 0)]
ASPC_censored = data_censored[(data_censored['charity'] == 1) & (data_censored['tradeoff'] == 1)]
ACPS_censored = data_censored[(data_censored['charity'] == 0) & (data_censored['tradeoff'] == 1)]

# mean_attentions_censored = [ASPS_censored.groupby('prob_option_A')['dwell_time_relative'].mean().mean(), 
#                             ACPS_censored.groupby('prob_option_A')['dwell_time_relative'].mean().mean(), 
#                             ACPC_censored.groupby('prob_option_A')['dwell_time_relative'].mean().mean(), 
#                             ASPC_censored.groupby('prob_option_A')['dwell_time_relative'].mean().mean()]
# print(mean_attentions_censored)

################################################
# Elicit data specifically checking self, charity and no tradeoff differences of H1
################################################

# Self lottery difference is ACPS-ASPS, Charity lottery difference is ASPC-ACPC
# and No Tradeoff difference is ACPC-ASPS

self_lottery_censored = pd.concat([ASPS_censored, ACPS_censored], ignore_index = True)
charity_lottery_censored = pd.concat([ACPC_censored, ASPC_censored], ignore_index=True)
no_tradeoff_lottery_censored = pd.concat([ASPS_censored, ACPC_censored], ignore_index=True)

# Self lottery, charity lottery and no tradeoff differences for Censored subjects
self_lottery_differences_censored = lottery_differences(self_lottery_censored, 'ACPS', 'ASPS') # gives YCPS-YSPS and ACPS-ASPS
charity_lottery_differences_censored = lottery_differences(charity_lottery_censored, 'ASPC', 'ACPC') # gives YSPC-YCPC and ASPC-ACPC
no_tradeoff_lottery_differences_censored = lottery_differences(no_tradeoff_lottery_censored, 'ACPC', 'ASPS') # gives YCPC-YSPS and ACPC-ASPS


# %%
# =============================================================================
# CATEGORISATION OF ADAPTIVE & ALTRUISTIC SUBJECTS 
# =============================================================================

# Within principal analysis, we want to find subjects that have Excuse-driven 
# risk preferences (EDRP), which we refer to as "Adaptive" subjects
# Thus we want participants with YCPS-YSPS > 0 and YCPS-YCPC < 0 whilst 
# taking into account the no tradeoff difference YCPC-YSPS =/= 0

# We also categorise participants with risk preferences that are the opposite of H1
# so with YCPS-YSPS < 0 and YCPS-YCPC > 0 whilst also
# taking into account the no tradeoff difference YCPC-YSPS =/= 0

EDRP_self = [] # participant having YCPS-YSPS > YCPC-YSPS (Excuse-driven for self)
EDRP_charity = [] # participant having YCPS-YCPC < - (YCPC-YSPS) (Excuse-driven for charity)

altruistic_self = [] # participant having YCPS-YSPS < - (YCPC-YSPS) (Altruistic for self)
altruistic_charity = [] # participant having YCPS-YCPC > YCPC-YSPS (Altruistic for charity)

for i in data_principal['number'].unique():
    self_diff = self_lottery_differences_principal.loc[self_lottery_differences_principal['number'] == i,['valuation_ACPS_ASPS']].mean() # mean across probabilities
    charity_diff = charity_lottery_differences_principal.loc[charity_lottery_differences_principal['number'] == i,['valuation_ASPC_ACPC']].mean() # mean across probabilities
    no_trade_diff = no_tradeoff_lottery_differences_principal.loc[no_tradeoff_lottery_differences_principal['number'] == i,['valuation_ACPC_ASPS']].mean() # mean across probabilities

    if self_diff.item() > no_trade_diff.item() : # participant has YCPS-YSPS > YCPC-YSPS on average across probabilities 
        EDRP_self.append(i)
    elif self_diff.item() < - no_trade_diff.item() : # participant has YCPS-YSPS < - (YCPC-YSPS) on average across probabilities 
        altruistic_self.append(i)
    if charity_diff.item() < - no_trade_diff.item() : # participant has YSPC-YCPC < - (YCPC-YSPS) on average across probabilities 
        EDRP_charity.append(i)
    if charity_diff.item() > no_trade_diff.item() : # participant has YSPC-YCPC > YCPC-YSPS on average across probabilities 
        altruistic_charity.append(i)

EDRP_total = np.intersect1d(EDRP_self, EDRP_charity) 

data_EDRP = data_principal[data_principal['number'].isin(EDRP_total)] # data of Adaptive subjects
data_autre_EDRP = data_autre_principal[data_autre_principal['number'].isin(EDRP_total)] # data_autre of Adaptive subjects
X_EDRP_total = data_autre_principal[data_autre_principal['number'].isin(EDRP_total)] # X-values of Adaptive subjects

no_tradeoff_lottery_differences_EDRP = no_tradeoff_lottery_differences_principal[no_tradeoff_lottery_differences_principal['number'].isin(EDRP_total)] # no tradeoff diff of Adaptive subjecs
self_lottery_differences_EDRP = self_lottery_differences_principal[self_lottery_differences_principal['number'].isin(EDRP_total)] # self lottery diff of Adaptive subjecs
charity_lottery_differences_EDRP = charity_lottery_differences_principal[charity_lottery_differences_principal['number'].isin(EDRP_total)] # charity lottery diff of Adaptive subjecs

ASPS_EDRP = data_EDRP[(data_EDRP['charity'] == 0) & (data_EDRP['tradeoff'] == 0)] # YSPS for Adaptive subjects
ACPC_EDRP = data_EDRP[(data_EDRP['charity'] == 1) & (data_EDRP['tradeoff'] == 0)] # YCPC for Adaptive subjects
ASPC_EDRP = data_EDRP[(data_EDRP['charity'] == 1) & (data_EDRP['tradeoff'] == 1)] # YSPC for Adaptive subjects
ACPS_EDRP = data_EDRP[(data_EDRP['charity'] == 0) & (data_EDRP['tradeoff'] == 1)] # YCPS for Adaptive subjects


# mean_attentions_EDRP = [ASPS_EDRP.groupby('prob_option_A')['dwell_time_relative'].mean().mean(), 
#                         ACPS_EDRP.groupby('prob_option_A')['dwell_time_relative'].mean().mean(), 
#                         ACPC_EDRP.groupby('prob_option_A')['dwell_time_relative'].mean().mean(), 
#                         ASPC_EDRP.groupby('prob_option_A')['dwell_time_relative'].mean().mean()]
# print(mean_attentions_EDRP)

# Participants not being Adaptive (Principal analysis without adaptive subjects)
data_else_EDRP = data_principal[~data_principal['number'].isin(data_EDRP['number'])] # data of else than Adaptive subjects
X_else_EDRP_total = data_autre_principal[~data_autre_principal['number'].isin(EDRP_total)] # X-values of else than Adaptive subjects

# Participants being both Altruistic for self and for charity -- called Altruistic subjects
altruistic_total = np.intersect1d(altruistic_self, altruistic_charity)

data_altruistic = data_principal[data_principal['number'].isin(altruistic_total)] # data of Altruistic subjects
data_autre_altruistic = data_autre_principal[data_autre_principal['number'].isin(altruistic_total)] # data_autre of Altruistic subjects
X_altruistic = data_autre_principal[data_autre_principal['number'].isin(altruistic_total)] # X-values of Altruistic subjects

no_tradeoff_lottery_differences_altruistic = no_tradeoff_lottery_differences_principal[no_tradeoff_lottery_differences_principal['number'].isin(altruistic_total)] # no tradeoff diff of Altruistic subjecs
self_lottery_differences_altruistic = self_lottery_differences_principal[self_lottery_differences_principal['number'].isin(altruistic_total)] # self lottery diff of Altruistic subjecs
charity_lottery_differences_altruistic = charity_lottery_differences_principal[charity_lottery_differences_principal['number'].isin(altruistic_total)] # charity lottery diff of Altruistic subjecs

ASPS_altruistic = data_altruistic[(data_altruistic['charity'] == 0) & (data_altruistic['tradeoff'] == 0)] # YSPS for Altruistic subjects
ACPC_altruistic = data_altruistic[(data_altruistic['charity'] == 1) & (data_altruistic['tradeoff'] == 0)] # YCPC for Altruistic subjects
ASPC_altruistic = data_altruistic[(data_altruistic['charity'] == 1) & (data_altruistic['tradeoff'] == 1)] # YSPC for Altruistic subjects
ACPS_altruistic = data_altruistic[(data_altruistic['charity'] == 0) & (data_altruistic['tradeoff'] == 1)] # YCPS for Altruistic subjects

# Adaptive and Censored Participants combined

# ASPS_EDRP_censored = pd.concat([ASPS_EDRP, ASPS_censored], ignore_index=True)
# ACPS_EDRP_censored = pd.concat([ACPS_EDRP, ACPS_censored], ignore_index=True)
# ACPC_EDRP_censored = pd.concat([ACPC_EDRP, ACPC_censored], ignore_index=True)
# ASPC_EDRP_censored = pd.concat([ASPC_EDRP, ASPC_censored], ignore_index=True)
# mean_attentions_EDRP_censored = [ASPS_EDRP_censored.groupby('prob_option_A')['dwell_time_relative'].mean().mean(), 
#                                  ACPS_EDRP_censored.groupby('prob_option_A')['dwell_time_relative'].mean().mean(), 
#                                  ACPC_EDRP_censored.groupby('prob_option_A')['dwell_time_relative'].mean().mean(), 
#                                  ASPC_EDRP_censored.groupby('prob_option_A')['dwell_time_relative'].mean().mean()]
# print(mean_attentions_EDRP_censored)

no_tradeoff_lottery_differences_EDRP_censored = pd.concat([no_tradeoff_lottery_differences_EDRP, no_tradeoff_lottery_differences_censored], ignore_index=True)
self_lottery_differences_EDRP_censored = pd.concat([self_lottery_differences_EDRP, self_lottery_differences_censored], ignore_index=True)
charity_lottery_differences_EDRP_censored = pd.concat([charity_lottery_differences_EDRP, charity_lottery_differences_censored], ignore_index=True)

# Sample sizes
samplesize_principal = len(data_autre_principal) # sample size of Principal Analysis
samplesize_adaptive = len(data_autre_EDRP) # sample size of Adaptive subjects
samplesize_altruistic = len(data_autre_altruistic) # sample size of Altruistic subjects
samplesize_censored = len(data_autre_censored) # sample size of Censored subjects
samplesize_EDRP_censored = len(data_autre_EDRP) + len(data_autre_censored) # sample size of Adaptive and Censored subjects

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
principal_errors_att = [0.338, 0.322, 0.389]       ################# CHANGER 

# for Adaptive subjects
EDRP_means_att = [no_tradeoff_lottery_differences_EDRP['dwell_time_ACPC_ASPS'].mean(), 
              self_lottery_differences_EDRP['dwell_time_ACPS_ASPS'].mean(),
              charity_lottery_differences_EDRP['dwell_time_ASPC_ACPC'].mean()]
EDRP_errors_att = [0.531, 0.493, 0.621]               ################## CHANGER 

# for Altruistic subjects
altruistic_means_att = [no_tradeoff_lottery_differences_altruistic['dwell_time_ACPC_ASPS'].mean(), 
              self_lottery_differences_altruistic['dwell_time_ACPS_ASPS'].mean(),
              charity_lottery_differences_altruistic['dwell_time_ASPC_ACPC'].mean()]
altruistic_errors_att = [0.831, 0.825, 0.939]              ################## CHANGER 

# for Censored subjects
censored_means_att = [no_tradeoff_lottery_differences_censored['dwell_time_ACPC_ASPS'].mean(), 
                  self_lottery_differences_censored['dwell_time_ACPS_ASPS'].mean(),
                  charity_lottery_differences_censored['dwell_time_ASPC_ACPC'].mean()]
censored_errors_att = [0.547, 0.577, 0.616]        ################## CHANGER 

# for Adaptive and Censored subjects
EDRP_censored_means_att = [no_tradeoff_lottery_differences_EDRP_censored['dwell_time_ACPC_ASPS'].mean(), 
                  self_lottery_differences_EDRP_censored['dwell_time_ACPS_ASPS'].mean(),
                  charity_lottery_differences_EDRP_censored['dwell_time_ASPC_ACPC'].mean()]
EDRP_censored_errors_att = [0.382, 0.384, 0.439]        ################## CHANGER 


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
fixed_model_principal_censored = fixed_regression_model(data_for_analysis_principal_and_censored, 'dwell_time_relative', ['charity', 'tradeoff', 'interaction', 'case_order'], 'yes')
fixed_model_principal_censored.to_csv('Principal analysis and Censored Fixed regression results H2.csv')


# Adaptive and censored subjects
data_for_analysis_EDRP_and_censored = pd.concat([data_EDRP, data_censored], 
                                                     ignore_index=True) # Data specifically for Adaptive and Censored subjects 
fixed_model_EDRP_censored = fixed_regression_model(data_for_analysis_EDRP_and_censored, 'dwell_time_relative', ['charity', 'tradeoff', 'interaction', 'case_order'], 'yes')
fixed_model_EDRP_censored.to_csv('Adaptive and Censored Fixed regression results H2.csv')



