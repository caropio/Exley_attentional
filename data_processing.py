#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:58:29 2024

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
path = '/Users/carolinepioger/Desktop/EXLEY ATT' # change to yours :)

# Upload dataframes
data = pd.read_csv(path + '/Exley_attentional/data/dataset.csv' ) # pooled data for analysis
data_autre = pd.read_csv(path + '/Exley_attentional/data/criterion info data.csv') # participant-specific info
survey = pd.read_csv(path + 'Exley_attentional/data//survey data.csv') # survey information 


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

if __name__ == "__main__": # to only print when running script and not when imported
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
if __name__ == "__main__": # to only print when running script and not when imported    plt.hist(data['dwell_time_absolute'], bins=50)
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

# We group the valuations according to the probabilies involved in the lotteries (7 probabilies)
valuation_ASPS = ASPS_principal.groupby('prob_option_A')['valuation']
valuation_ACPS = ACPS_principal.groupby('prob_option_A')['valuation']
valuation_ACPC = ACPC_principal.groupby('prob_option_A')['valuation']
valuation_ASPC = ASPC_principal.groupby('prob_option_A')['valuation']

# We find the means of valuations for each probability (for each case) 
mean_valuation_ASPS = valuation_ASPS.mean()
mean_valuation_ACPC = valuation_ACPC.mean()
mean_valuation_ACPS = valuation_ACPS.mean()
mean_valuation_ASPC = valuation_ASPC.mean()

# We group these means together
mean_valuations = [mean_valuation_ASPS.mean(), mean_valuation_ACPS.mean(), 
                   mean_valuation_ACPC.mean(), mean_valuation_ASPC.mean()]

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
        individual = database.loc[database['number'] == i, ['case', 'prob_option_A', 'valuation', 'dwell_time_relative', 'dwell_time_absolute']] 
        individual_difference = individual.pivot(index='prob_option_A', columns='case')
        try: 
            individual_difference[f'valuation_{var1}_{var2}'] = individual_difference['valuation'][var1] - individual_difference['valuation'][var2]
            individual_difference[f'dwell_time_{var1}_{var2}'] = individual_difference['dwell_time_relative'][var1] - individual_difference['dwell_time_relative'][var2]
            individual_difference[f'dwell_time_absolute_{var1}_{var2}'] = individual_difference['dwell_time_absolute'][var1] - individual_difference['dwell_time_absolute'][var2]
            individual_difference['number'] = i
            individual_difference.reset_index(inplace=True)
            individual_difference.columns = individual_difference.columns.droplevel(1)
            lottery_differences = pd.concat([lottery_differences, individual_difference[['number', 'prob_option_A', f'valuation_{var1}_{var2}', f'dwell_time_{var1}_{var2}', f'dwell_time_absolute_{var1}_{var2}']]], ignore_index=True)
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
# CATEGORISATION OF ADAPTIVE & ALTRUISTIC & NORMAL SUBJECTS 
# =============================================================================

# Within principal analysis, we want to find subjects that have Excuse-driven 
# risk preferences (EDRP), which we refer to as "Adaptive" subjects
# Thus we want participants with YCPS-YSPS > 0 and YSPC-YCPC < 0 whilst 
# taking into account the no tradeoff difference YCPC-YSPS =/= 0

# We also categorise participants with risk preferences that are the opposite of H1
# so with YCPS-YSPS < 0 and YSPC-YCPC > 0 whilst also
# taking into account the no tradeoff difference YCPC-YSPS =/= 0

# We are also interested in categorising individuals who don't have differences 
# between the tradeoff and notradeoff valuations (individuals that don't adapt
# their risk preferences)

EDRP_self = [] # participant having YCPS-YSPS > YCPC-YSPS (Excuse-driven for self)
EDRP_charity = [] # participant having YSPC-YCPC < - (YCPC-YSPS) (Excuse-driven for charity)

altruistic_self = [] # participant having YCPS-YSPS < - (YCPC-YSPS) (Altruistic for self)
altruistic_charity = [] # participant having YSPC-YCPC > YCPC-YSPS (Altruistic for charity)

normal_self = [] # participants having - (YCPC-YSPS) ≤ YCPS-YSPS ≤ YCPC-YSPS 
normal_charity = []  # participants having - (YCPC-YSPS) ≤ YSPC-YCPC ≤ YCPC-YSPS 

for i in data_principal['number'].unique():
    self_diff = self_lottery_differences_principal.loc[self_lottery_differences_principal['number'] == i,['valuation_ACPS_ASPS']].mean() # mean across probabilities
    charity_diff = charity_lottery_differences_principal.loc[charity_lottery_differences_principal['number'] == i,['valuation_ASPC_ACPC']].mean() # mean across probabilities
    no_trade_diff = no_tradeoff_lottery_differences_principal.loc[no_tradeoff_lottery_differences_principal['number'] == i,['valuation_ACPC_ASPS']].mean() # mean across probabilities

    if self_diff.item() > no_trade_diff.item() : # participant has YCPS-YSPS > YCPC-YSPS on average across probabilities 
    # if self_diff.item() > 0: # if we replicate exactly Exley's categorization
        EDRP_self.append(i)
    elif self_diff.item() < - no_trade_diff.item() : # participant has YCPS-YSPS < - (YCPC-YSPS) on average across probabilities 
    # elift self_diff.item() < 0: # if we replicate exactly Exley's categorization
        altruistic_self.append(i)
    else: # participant has - (YCPC-YSPS) ≤ YCPS-YSPS ≤ YCPC-YSPS 
        normal_self.append(i)
    if charity_diff.item() < - no_trade_diff.item() : # participant has YSPC-YCPC < - (YCPC-YSPS) on average across probabilities 
    # if charity_diff.item() < 0: # if we replicate exactly Exley's categorization
        EDRP_charity.append(i)
    elif charity_diff.item() > no_trade_diff.item() : # participant has YSPC-YCPC > YCPC-YSPS on average across probabilities 
    # elif charity_diff.item() > 0: # if we replicate exactly Exley's categorization
        altruistic_charity.append(i)
    else: # participant has - (YCPC-YSPS) ≤ YSPC-YCPC ≤ YCPC-YSPS
        normal_charity.append(i)

# =============================================================================
# ADAPTIVE 
# =============================================================================

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

# =============================================================================
# ALTRUISTIC 
# =============================================================================

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

# =============================================================================
# NORMAL  
# =============================================================================

# Participants being both normal for self and for charity -- called normal subjects
normal_total = np.intersect1d(normal_self, normal_charity)

data_normal = data_principal[data_principal['number'].isin(normal_total)] # data of normal subjects
data_autre_normal = data_autre_principal[data_autre_principal['number'].isin(normal_total)] # data_autre of normal subjects
X_normal = data_autre_principal[data_autre_principal['number'].isin(normal_total)] # X-values of normal subjects

no_tradeoff_lottery_differences_normal = no_tradeoff_lottery_differences_principal[no_tradeoff_lottery_differences_principal['number'].isin(normal_total)] # no tradeoff diff of normal subjecs
self_lottery_differences_normal = self_lottery_differences_principal[self_lottery_differences_principal['number'].isin(normal_total)] # self lottery diff of normal subjecs
charity_lottery_differences_normal = charity_lottery_differences_principal[charity_lottery_differences_principal['number'].isin(normal_total)] # charity lottery diff of normal subjecs

ASPS_normal = data_normal[(data_normal['charity'] == 0) & (data_normal['tradeoff'] == 0)] # YSPS for normal subjects
ACPC_normal = data_normal[(data_normal['charity'] == 1) & (data_normal['tradeoff'] == 0)] # YCPC for normal subjects
ASPC_normal = data_normal[(data_normal['charity'] == 1) & (data_normal['tradeoff'] == 1)] # YSPC for normal subjects
ACPS_normal = data_normal[(data_normal['charity'] == 0) & (data_normal['tradeoff'] == 1)] # YCPS for normal subjects

# =============================================================================
# ELSE
# =============================================================================

# Adaptive and Censored Participants combined

data_EDRP_censored = pd.concat([data_EDRP, data_censored], ignore_index=True)

no_tradeoff_lottery_differences_EDRP_censored = pd.concat([no_tradeoff_lottery_differences_EDRP, 
                                                           no_tradeoff_lottery_differences_censored], ignore_index=True)
self_lottery_differences_EDRP_censored = pd.concat([self_lottery_differences_EDRP, 
                                                    self_lottery_differences_censored], ignore_index=True)
charity_lottery_differences_EDRP_censored = pd.concat([charity_lottery_differences_EDRP, 
                                                       charity_lottery_differences_censored], ignore_index=True)

# Principal Analysis and Censored Participants combined
no_tradeoff_lottery_differences_principal_censored = pd.concat([no_tradeoff_lottery_differences_principal, # specifically for Principal Analysis and Censored subjects 
                                                           no_tradeoff_lottery_differences_censored], ignore_index=True)
self_lottery_differences_principal_censored = pd.concat([self_lottery_differences_principal, self_lottery_differences_censored], 
                                                     ignore_index=True) # Self differences specifically for Principal Analysis and Censored subjects 
charity_lottery_differences_principal_censored = pd.concat([charity_lottery_differences_principal, charity_lottery_differences_censored], 
                                                     ignore_index=True) # Charity differences specifically for Principal Analysis and Censored subjects 


################################################
# Sample sizes
################################################

samplesize_principal = len(data_autre_principal) # sample size of Principal Analysis
samplesize_adaptive = len(data_autre_EDRP) # sample size of Adaptive subjects
samplesize_altruistic = len(data_autre_altruistic) # sample size of Altruistic subjects
samplesize_normal = len(data_autre_normal) # sample size of normal subjects
samplesize_censored = len(data_autre_censored) # sample size of Censored subjects
samplesize_EDRP_censored = len(data_autre_EDRP) + len(data_autre_censored) # sample size of Adaptive and Censored subjects
samplesize_principal_censored = len(data_autre_principal) + len(data_autre_censored) # sample size of Principal Analysis and Censored subjects


################################################
# Socio-demographic information 
################################################

if __name__ == "__main__": # to only print when running script and not when imported

# For Adaptive subjects
    survey_EDRP = pd.merge(data_EDRP[['id']], survey, on='id', how='inner')
    
    print()
    print('ADAPTIVE SUBJECTS')
    print('The mean age is ' + str(survey_EDRP['Demog_AGE'].mean()))
    print('There is ' 
          + str(round(100*len(survey_EDRP[survey_EDRP['Demog_Sex']==1])/
                      (len(survey_EDRP[survey_EDRP['Demog_Sex']==1])+len(survey_EDRP[survey_EDRP['Demog_Sex']==2])), 1))
                            + ' % of women')
    print('The mean highest education level is ' + 
          str(['A level', 'Bsci', 'Msci', 'Phd', 'RNS'][round(survey_EDRP['Demog_High_Ed_Lev'].mean())-1]))
    print()
    
    # For Altruistic subjects
    survey_altruistic = pd.merge(data_altruistic[['id']], survey, on='id', how='inner')
    
    print()
    print('ALTRUISTIC SUBJECTS')
    print('The mean age is ' + str(survey_altruistic['Demog_AGE'].mean()))
    print('There is ' 
          + str(round(100*len(survey_altruistic[survey_altruistic['Demog_Sex']==1])/
                      (len(survey_altruistic[survey_altruistic['Demog_Sex']==1])+len(survey_altruistic[survey_altruistic['Demog_Sex']==2])), 1))
                            + ' % of women')
    print('The mean highest education level is ' + 
          str(['A level', 'Bsci', 'Msci', 'Phd', 'RNS'][round(survey_altruistic['Demog_High_Ed_Lev'].mean())-1]))
    print()
    
    # Adapted and Censored subjects
    
    survey_censored = pd.merge(data_autre_censored[['id']], survey, on='id', how='inner')
    survey_adapted_censored = pd.concat([survey_EDRP, survey_censored], ignore_index = True)
    
    print()
    print('ADAPTIVE AND CENSORED  SUBJECTS')
    print('The mean age is ' + str(survey_adapted_censored['Demog_AGE'].mean()))
    print('There is ' 
          + str(round(100*len(survey_adapted_censored[survey_adapted_censored['Demog_Sex']==1])/
                      (len(survey_adapted_censored[survey_adapted_censored['Demog_Sex']==1])+len(survey_adapted_censored[survey_adapted_censored['Demog_Sex']==2])), 1))
                            + ' % of women')
    print('The mean highest education level is ' + 
          str(['A level', 'Bsci', 'Msci', 'Phd', 'RNS'][round(survey_adapted_censored['Demog_High_Ed_Lev'].mean())-1]))
    

# %%
# =============================================================================
# Participant-specific X values Analysis
# =============================================================================

################################################
# Distribution of X values
################################################

if __name__ == "__main__": # to only print when running script and not when imported

# We plot the different ditribution of participant-specific X values 

    # Distribution for all subjects
    plt.hist(data_autre['charity_calibration'], bins=20, color = 'lightcoral') 
    plt.axvline(x=data_autre['charity_calibration'].mean(), color='grey', linestyle='--', label = 'Mean = '+ str(np.round(data_autre['charity_calibration'].mean(), 1)))
    plt.axvline(x=data_autre['charity_calibration'].median(), color='gainsboro', linestyle='--', label = 'Median = '+ str(np.round(data_autre['charity_calibration'].median(), 1)))
    samplesize = len(data_autre)
    plt.text(0.15, 0.9, f'n = {samplesize}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
    plt.xlabel('Participant-specific X')
    plt.ylabel('Frequency')
    plt.title('Distribution of participant-specific X values (all subjects)')
    plt.show()
    
    # Distribution for Principal analysis 
    plt.hist(data_autre_principal['charity_calibration'], bins=20, color = 'lightcoral') 
    plt.xlabel('X values')
    plt.ylabel('Frequency')
    plt.title('Distribution of participant-specific X values (Principal Analysis)')
    plt.axvline(x=data_autre_principal['charity_calibration'].mean(), color='grey', linestyle='--', label = 'Mean = '+ str(np.round(data_autre_principal['charity_calibration'].mean(), 1)))
    plt.axvline(x=data_autre_principal['charity_calibration'].median(), color='gainsboro', linestyle='--', label = 'Median = '+ str(np.round(data_autre_principal['charity_calibration'].median(), 1)))
    plt.text(0.15, 0.9, f'n = {samplesize_principal}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
    plt.legend()
    plt.savefig('X values distribution Principal analysis.png', dpi=1200)
    plt.show()
    
    # Replication figure Exley (Distribution of Principal Analysis)
    hist, bins, _ = plt.hist(data_autre_principal['charity_calibration'], bins=16, color='black', density=True) 
    hist_percentage = hist * 100
    bar_width = np.diff(bins) * 0.8
    bin_centers = bins[:-1] + np.diff(bins) * 0.1
    plt.bar(bin_centers, hist_percentage, width=bar_width, edgecolor='black', align='center', color='black')
    plt.xlabel('X values')
    plt.ylabel('Percentage')
    plt.title('Distribution of participant-specific X values (Exley replication)')
    mean_val = np.round(data_autre_principal['charity_calibration'].mean(), 1)
    median_val = np.round(data_autre_principal['charity_calibration'].median(), 1)
    plt.text(0.27, 0.85, f'Mean = {mean_val}, Median = {median_val}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
    plt.savefig('X values distribution Principal analysis EXLEY.png', dpi=1200)
    plt.show()
    
    # Distribution for Adaptive subjects  
    plt.hist(X_EDRP_total['charity_calibration'], bins=20, color = 'lightcoral') 
    plt.xlabel('X values')
    plt.ylabel('Frequency')
    plt.title('Distribution of X values of Adaptive subjects')
    plt.axvline(x=X_EDRP_total['charity_calibration'].mean(), color='grey', linestyle='--', label = 'Mean = '+ str(np.round(X_EDRP_total['charity_calibration'].mean(), 1)))
    plt.axvline(x=X_EDRP_total['charity_calibration'].median(), color='gainsboro', linestyle='--', label = 'Median = '+ str(np.round(X_EDRP_total['charity_calibration'].median(), 1)))
    plt.text(0.15, 0.9, f'n = {samplesize_adaptive}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
    plt.legend()
    plt.savefig('X values distribution ADAPTIVE.png', dpi=1200)
    plt.show()
    
    # Distribution for Altruistic subjects  
    plt.hist(X_altruistic['charity_calibration'], bins=20, color = 'lightcoral') 
    plt.xlabel('X values')
    plt.ylabel('Frequency')
    plt.title('Distribution of X values of Altruistic subjects')
    plt.axvline(x=X_altruistic['charity_calibration'].mean(), color='grey', linestyle='--', label = 'Mean = '+ str(np.round(X_altruistic['charity_calibration'].mean(), 1)))
    plt.axvline(x=X_altruistic['charity_calibration'].median(), color='gainsboro', linestyle='--', label = 'Median = '+ str(np.round(X_altruistic['charity_calibration'].median(), 1)))
    plt.text(0.85, 0.9, f'n = {samplesize_altruistic}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
    plt.legend()
    plt.savefig('X values distribution for ALTRUISTIC.png', dpi=1200)
    plt.show()


################################################
# Comparison of X-values
################################################

# We compare the ditribution of participant-specific X values between different groups of subjects

if __name__ == "__main__": # to only print when running script and not when imported

    # BETWEEN Adaptive and Principal Analysis
    print('BETWEEN Adaptive and Principal Analysis ')
    t_statistic_X, p_value_X = ttest_ind(X_EDRP_total['charity_calibration'], data_autre_principal['charity_calibration'])
    print('Difference of X values between Adaptive and Principal Analysis (t-test, p value) : ')
    print(t_statistic_X, p_value_X)
    print()
    
    # BETWEEN Adaptive vs Altruistic
    print('BETWEEN Adaptive and Altruistic subjects ')
    t_statistic_X_2, p_value_X_2 = ttest_ind(X_EDRP_total['charity_calibration'], X_altruistic['charity_calibration'])
    print('Difference of X values between Adaptive and Altruistic subjects (t-test, p value) : ')
    print(t_statistic_X_2, p_value_X_2)
    print()
    
    # BETWEEN Altruistic vs Principal Analysis
    print('BETWEEN Altruistic and Principal Analysis ')
    t_statistic_X_3, p_value_X_3 = ttest_ind(X_altruistic['charity_calibration'], data_autre_principal['charity_calibration'])
    print('Difference of X values between Altruistic and Principal Analysis (t-test, p value) : ')
    print(t_statistic_X_3, p_value_X_3)
    print()


# %%
# =============================================================================
# Desired variables for exporting 
# =============================================================================


__all__ = [
    'survey', 'data_principal', 'data_EDRP', 'data_censored', 'data_altruistic',
    'data_normal',
    'no_tradeoff_lottery_differences_principal', 'self_lottery_differences_principal', 'charity_lottery_differences_principal', 
    'no_tradeoff_lottery_differences_censored', 'self_lottery_differences_censored', 'charity_lottery_differences_censored',
    'no_tradeoff_lottery_differences_EDRP', 'self_lottery_differences_EDRP', 'charity_lottery_differences_EDRP',
    'no_tradeoff_lottery_differences_altruistic', 'self_lottery_differences_altruistic', 'charity_lottery_differences_altruistic',
    'no_tradeoff_lottery_differences_normal', 'self_lottery_differences_normal', 'charity_lottery_differences_normal',
    'no_tradeoff_lottery_differences_EDRP_censored', 'self_lottery_differences_EDRP_censored', 'charity_lottery_differences_EDRP_censored',
    'no_tradeoff_lottery_differences_principal_censored', 'self_lottery_differences_principal_censored', 'charity_lottery_differences_principal_censored', 
    'valuation_ASPS', 'valuation_ACPS', 'valuation_ACPC', 'valuation_ASPC',
    'ASPS_principal', 'ACPS_principal', 'ACPC_principal', 'ASPC_principal', 
    'mean_valuations', 
    'samplesize_principal', 'samplesize_adaptive', 'samplesize_altruistic', 'samplesize_normal',
    'samplesize_censored', 'samplesize_EDRP_censored', 'samplesize_principal_censored'
]

