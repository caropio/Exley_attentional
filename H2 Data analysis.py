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
# REMOVING OUTLIERS 
# =============================================================================

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


# %%
# =============================================================================
# CATEGORISATION BETWEEN PRINCIPAL ANALYSIS AND CENSORED
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
    if charity_diff.item() < - no_trade_diff.item() : # participant has YCPS-YCPC < - (YCPC-YSPS) on average across probabilities 
        EDRP_charity.append(i)
    if charity_diff.item() > no_trade_diff.item() : # participant has YCPS-YCPC > YCPC-YSPS on average across probabilities 
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

no_tradeoff_lottery_differences_EDRP_censored = pd.concat([no_tradeoff_lottery_differences_EDRP, no_tradeoff_lottery_differences_censored], ignore_index=True)
self_lottery_differences_EDRP_censored = pd.concat([self_lottery_differences_EDRP, self_lottery_differences_censored], ignore_index=True)
charity_lottery_differences_EDRP_censored = pd.concat([charity_lottery_differences_EDRP, charity_lottery_differences_censored], ignore_index=True)

# Sample sizes
samplesize_principal = len(data_autre_principal) # sample size of Principal Analysis
samplesize_adaptive = len(data_autre_EDRP) # sample size of Adaptive subjects
samplesize_altruistic = len(data_autre_altruistic) # sample size of Altruistic subjects
samplesize_censored = len(data_autre_censored) # sample size of Censored subjects

# %%
# =============================================================================
# ATTENTION DATA VISUALIZATION
# =============================================================================

lottery_types_difference_attention = ['$A^{C}(P^{C})-A^{S}(P^{S})$', 
                                      '$A^{C}(P^{S})-A^{S}(P^{S})$', 
                                      '$A^{S}(P^{C})-A^{C}(P^{C})$']
x = np.arange(len(lottery_types_difference_attention))

################################################
# Attention difference
################################################

# Now we are interested in attention difference, namely ACPS-ASPS and ASPC-ACPC
# To verify for H2, we check for negative differences 



# 3 attention differences and standard errors at ind level for Principal Analysis, Adaptive, Altruistic and Censored subjects
# for Principal Analysis
principal_means_att = [no_tradeoff_lottery_differences_principal['dwell_time_ACPC_ASPS'].mean(), 
                   self_lottery_differences_principal['dwell_time_ACPS_ASPS'].mean(),
                   charity_lottery_differences_principal['dwell_time_ASPC_ACPC'].mean()]
principal_errors_att = [0.343, 0.322, 0.4015]          # CHANGER 

# for Adaptive subjects
EDRP_means_att = [no_tradeoff_lottery_differences_EDRP['dwell_time_ACPC_ASPS'].mean(), 
              self_lottery_differences_EDRP['dwell_time_ACPS_ASPS'].mean(),
              charity_lottery_differences_EDRP['dwell_time_ASPC_ACPC'].mean()]
EDRP_errors_att = [0.513, 0.565, 0.7405]                     # CHANGER 

# for Altruistic subjects
altruistic_means_att = [no_tradeoff_lottery_differences_altruistic['dwell_time_ACPC_ASPS'].mean(), 
              self_lottery_differences_altruistic['dwell_time_ACPS_ASPS'].mean(),
              charity_lottery_differences_altruistic['dwell_time_ASPC_ACPC'].mean()]
altruistic_errors_att = [0.723, 0.675, 0.786]                     # CHANGER 

# for Censored subjects
censored_means_att = [no_tradeoff_lottery_differences_censored['dwell_time_ACPC_ASPS'].mean(), 
                  self_lottery_differences_censored['dwell_time_ACPS_ASPS'].mean(),
                  charity_lottery_differences_censored['dwell_time_ASPC_ACPC'].mean()]
censored_errors_att = [0.507, 0.611, 0.633]                  # CHANGER 


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

diff_proba_self_attention = self_lottery_differences_principal.groupby('prob_option_A')['dwell_time_ACPS_ASPS']
diff_proba_charity_attention = charity_lottery_differences_principal.groupby('prob_option_A')['dwell_time_ASPC_ACPC']
diff_proba_no_tradeoff_attention = no_tradeoff_lottery_differences_principal.groupby('prob_option_A')['dwell_time_ACPC_ASPS']

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

diff_proba_self_attention_EDRP = self_lottery_differences_EDRP.groupby('prob_option_A')['dwell_time_ACPS_ASPS']
diff_proba_charity_attention_EDRP = charity_lottery_differences_EDRP.groupby('prob_option_A')['dwell_time_ASPC_ACPC']
diff_proba_no_tradeoff_attention_EDRP = no_tradeoff_lottery_differences_EDRP.groupby('prob_option_A')['dwell_time_ACPC_ASPS']

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

diff_proba_self_attention_altruistic = self_lottery_differences_altruistic.groupby('prob_option_A')['dwell_time_ACPS_ASPS']
diff_proba_charity_attention_altruistic = charity_lottery_differences_altruistic.groupby('prob_option_A')['dwell_time_ASPC_ACPC']
diff_proba_no_tradeoff_attention_altruistic = no_tradeoff_lottery_differences_altruistic.groupby('prob_option_A')['dwell_time_ACPC_ASPS']

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

diff_proba_self_attention_censored = self_lottery_differences_censored.groupby('prob_option_A')['dwell_time_ACPS_ASPS']
diff_proba_charity_attention_censored = charity_lottery_differences_censored.groupby('prob_option_A')['dwell_time_ASPC_ACPC']
diff_proba_no_tradeoff_attention_censored = no_tradeoff_lottery_differences_censored.groupby('prob_option_A')['dwell_time_ACPC_ASPS']

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

diff_proba_self_attention_ALL = self_lottery_differences_EDRP_censored.groupby('prob_option_A')['dwell_time_ACPS_ASPS']
diff_proba_charity_attention_ALL = charity_lottery_differences_EDRP_censored.groupby('prob_option_A')['dwell_time_ASPC_ACPC']
diff_proba_no_tradeoff_attention_ALL = no_tradeoff_lottery_differences_EDRP_censored.groupby('prob_option_A')['dwell_time_ACPC_ASPS']

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


attention_per_proba = data_principal.groupby('prob_option_A')['dwell_time_relative']

first_case = data_principal[data_principal['case_order']==1]
second_case = data_principal[data_principal['case_order']==2]
third_case = data_principal[data_principal['case_order']==3]
fourth_case = data_principal[data_principal['case_order']==4]

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


attention_per_proba_censored = data_censored.groupby('prob_option_A')['dwell_time_relative']

plt.bar(['0.05', '0.1', '0.25', '0.5', '0.75', '0.9', '0.95'], attention_per_proba_censored.mean(), 
        color = ['darkgoldenrod', 'goldenrod', 'gold', 'khaki', 'beige', 'papayawhip', 'peachpuff']) 
plt.xlabel('Probability')
plt.ylabel('Mean atention time in %')
plt.title('Mean attention per probability for Censored')
plt.savefig('Attention probability CENSORED H2.png', dpi=1200)
plt.show()


plt.bar(['$A^{C}(P^{C})-A^{S}(P^{S}$)', '$A^{C}(P^{S})-A^{S}(P^{S})$', '$A^{S}(P^{C})-A^{C}(P^{C})$'], 
        [no_tradeoff_lottery_differences_censored['dwell_time_ACPC_ASPS'].mean(), 
          self_lottery_differences_censored['dwell_time_ACPS_ASPS'].mean(), 
          charity_lottery_differences_censored['dwell_time_ASPC_ACPC'].mean()], 
        color = ['bisque', 'lightskyblue', 'lightgreen']) 
plt.errorbar(['$A^{C}(P^{C})-A^{S}(P^{S}$)', '$A^{C}(P^{S})-A^{S}(P^{S})$', '$A^{S}(P^{C})-A^{C}(P^{C})$'], 
              [no_tradeoff_lottery_differences_censored['dwell_time_ACPC_ASPS'].mean(), self_lottery_differences_censored['dwell_time_ACPS_ASPS'].mean(), 
               charity_lottery_differences_censored['dwell_time_ASPC_ACPC'].mean()], 
              [0.507, 0.611, 0.633], ecolor = 'black', fmt='none', alpha=0.7)
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery type')
plt.ylabel('Difference in attention (trad - no trad) in %')
plt.title('Difference in attention across probabilities for Censored')
plt.savefig('Bar diff type Lottery CENSORED H2.png', dpi=1200)
plt.show()

plt.bar(['$A^{C}(P^{C})-A^{S}(P^{S}$)', '$A^{C}(P^{S})-A^{S}(P^{S})$', '$A^{S}(P^{C})-A^{C}(P^{C})$'], 
        [no_tradeoff_lottery_differences_EDRP_censored['dwell_time_ACPC_ASPS'].mean(), 
          self_lottery_differences_EDRP_censored['dwell_time_ACPS_ASPS'].mean(), 
          charity_lottery_differences_EDRP_censored['dwell_time_ASPC_ACPC'].mean()], 
        color = ['bisque', 'lightskyblue', 'lightgreen']) 
plt.errorbar(['$A^{C}(P^{C})-A^{S}(P^{S}$)', '$A^{C}(P^{S})-A^{S}(P^{S})$', '$A^{S}(P^{C})-A^{C}(P^{C})$'], 
              [no_tradeoff_lottery_differences_EDRP_censored['dwell_time_ACPC_ASPS'].mean(), 
               self_lottery_differences_EDRP_censored['dwell_time_ACPS_ASPS'].mean(), 
               charity_lottery_differences_EDRP_censored['dwell_time_ASPC_ACPC'].mean()], 
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
# ANALYSE DATA 
# =============================================================================

data_for_analysis = pd.concat([ASPS_principal, ACPC_principal, ASPC_principal, ACPS_principal], ignore_index=True)
data_for_analysis_EDRP = pd.concat([ASPS_EDRP, ACPC_EDRP, ASPC_EDRP, ACPS_EDRP], ignore_index=True)
data_for_analysis_altruistic = pd.concat([ASPS_altruistic, ACPC_altruistic, ASPC_altruistic, ACPS_altruistic], ignore_index=True)

data_for_analysis_censored = pd.concat([ASPS_censored, ACPC_censored, ASPC_censored, ACPS_censored], ignore_index=True)
data_for_analysis_all_and_censored = pd.concat([ASPS_principal, ACPC_principal, ASPC_principal, ACPS_principal, ASPS_censored, ACPC_censored, ASPC_censored, ACPS_censored], ignore_index=True)
data_for_analysis_EDRP_and_censored = pd.concat([ASPS_EDRP, ACPC_EDRP, ASPC_EDRP, ACPS_EDRP, ASPS_censored, ACPC_censored, ASPC_censored, ACPS_censored], ignore_index=True)

# data_for_analysis_EDRP['dwell_time_absolute'].mean()
# data_for_analysis_censored['dwell_time_absolute'].mean()

### differences between all and censored




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

