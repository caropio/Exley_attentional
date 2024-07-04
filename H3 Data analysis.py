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
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
from matplotlib.patches import Patch
import matplotlib.cm as cm
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
# Elicit data specifically checking self, charity and no tradeoff differences of H3
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
self_lottery_differences_principal = lottery_differences(self_lottery_principal, 'ASPS', 'ACPS') # gives YSPS-YCPS and ASPS-ACPS
charity_lottery_differences_principal = lottery_differences(charity_lottery_principal, 'ACPC', 'ASPC') # gives YCPC-YSPC and ACPC-ASPC
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
self_lottery_differences_censored = lottery_differences(self_lottery_censored, 'ASPS', 'ACPS') # gives YSPS-YCPS and ASPS-ACPS
charity_lottery_differences_censored = lottery_differences(charity_lottery_censored, 'ACPC', 'ASPC') # gives YCPC-YSPC and ACPC-ASPC
no_tradeoff_lottery_differences_censored = lottery_differences(no_tradeoff_lottery_censored, 'ACPC', 'ASPS') # gives YCPC-YSPS and ACPC-ASPS


# %%
# =============================================================================
# CATEGORISATION OF ADAPTIVE & ALTRUISTIC SUBJECTS (!!! SPECIFIC to H3 !!!)
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

# Notice that the inequalities are differented and suited for the new representation
# of self_lottery and charity_lottery which are inversed
for i in data_principal['number'].unique():
    self_diff = self_lottery_differences_principal.loc[self_lottery_differences_principal['number'] == i,['valuation_ASPS_ACPS']].mean() # mean across probabilities
    charity_diff = charity_lottery_differences_principal.loc[charity_lottery_differences_principal['number'] == i,['valuation_ACPC_ASPC']].mean() # mean across probabilities
    no_trade_diff = no_tradeoff_lottery_differences_principal.loc[no_tradeoff_lottery_differences_principal['number'] == i,['valuation_ACPC_ASPS']].mean() # mean across probabilities

    if self_diff.item() < - no_trade_diff.item() : # participant has YSPS-YCPS < - (YCPC-YSPS) on average across probabilities 
        EDRP_self.append(i)
    elif self_diff.item() > no_trade_diff.item() : # participant has YSPS-YCPS > YCPC-YSPS on average across probabilities 
        altruistic_self.append(i)
    if charity_diff.item() > no_trade_diff.item() : # participant has YCPC-YSPC > YCPC-YSPS on average across probabilities 
        EDRP_charity.append(i)
    if charity_diff.item() < - no_trade_diff.item() : # participant has YCPC-YSPC < - (YCPC-YSPS) on average across probabilities 
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

data_EDRP_censored = pd.concat([data_EDRP, data_censored], ignore_index=True)

no_tradeoff_lottery_differences_EDRP_censored = pd.concat([no_tradeoff_lottery_differences_EDRP, 
                                                           no_tradeoff_lottery_differences_censored], ignore_index=True)
self_lottery_differences_EDRP_censored = pd.concat([self_lottery_differences_EDRP, 
                                                    self_lottery_differences_censored], ignore_index=True)
charity_lottery_differences_EDRP_censored = pd.concat([charity_lottery_differences_EDRP, 
                                                       charity_lottery_differences_censored], ignore_index=True)

# Principal Analysis and Censored Participants combined
self_lottery_differences_principal_censored = pd.concat([self_lottery_differences_principal, self_lottery_differences_censored], 
                                                     ignore_index=True) # Self differences specifically for Principal Analysis and Censored subjects 
charity_lottery_differences_principal_censored = pd.concat([charity_lottery_differences_principal, charity_lottery_differences_censored], 
                                                     ignore_index=True) # Charity differences specifically for Principal Analysis and Censored subjects 


# Sample sizes
samplesize_principal = len(data_autre_principal) # sample size of Principal Analysis
samplesize_adaptive = len(data_autre_EDRP) # sample size of Adaptive subjects
samplesize_altruistic = len(data_autre_altruistic) # sample size of Altruistic subjects
samplesize_censored = len(data_autre_censored) # sample size of Censored subjects
samplesize_EDRP_censored = len(data_autre_EDRP) + len(data_autre_censored) # sample size of Adaptive and Censored subjects
samplesize_principal_censored = len(data_autre_principal) + len(data_autre_censored) # sample size of Principal Analysis and Censored subjects



# %%
# =============================================================================
# VISUALISE CORRELATION DATA BETWEEN ATTENTION AND VALUATION
# =============================================================================

# Define function that assigns a different color for each individual
def color_by_ind(database):
    individuals = database['number'].unique()
    colors = cm.rainbow(np.linspace(0, 1, len(individuals)))
    individual_color_map = dict(zip(individuals, colors))
    return database['number'].map(individual_color_map)

# Get string version of variable name using globals()
def get_variable_name_from_globals(var):
    globals_dict = globals()
    for name, value in globals_dict.items():
        if value is var:
            return name
    return None

# Define function of plots of correlation between Attention and Valuation differences
def plot_corr_attention_valuation(database, pop, samplesize):
    if get_variable_name_from_globals(database).split('_')[0] == 'self':
        x = '_ASPS_ACPS'
    elif get_variable_name_from_globals(database).split('_')[0] == 'charity':
        x = '_ACPC_ASPC'
    
    # scatter plot with each participant having a different color 
    plt.scatter(database[f'dwell_time{x}'], database[f'valuation{x}'],  c=color_by_ind(database))
    
    # adding regression line in red
    coef = np.polyfit(database[f'dwell_time{x}'], database[f'valuation{x}'], 1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(database[f'dwell_time{x}'], poly1d_fn(database[f'dwell_time{x}']), 
                 color='red', linewidth=2, label='Regression Line')

    plt.xlabel('Attention difference in %')
    plt.ylabel('Valuation difference in %')
    plt.title('Attention vs Valuation differences ' 
              + str(get_variable_name_from_globals(database).split('_')[0]) 
              + ' / ' + str(pop) + f' n = {samplesize}')
    plt.legend()
    # plt.grid(True)
    plt.show()
  
# Plot Attention vs Valuation difference for Adaptive subjects
plot_corr_attention_valuation(self_lottery_differences_EDRP, 'Adaptive', samplesize_adaptive)
plot_corr_attention_valuation(charity_lottery_differences_EDRP, 'Adaptive', samplesize_adaptive)

# Plot Attention vs Valuation difference for Censored subjects
plot_corr_attention_valuation(self_lottery_differences_censored, 'Censored', samplesize_censored)
plot_corr_attention_valuation(charity_lottery_differences_censored, 'Censored', samplesize_censored)

# Plot Attention vs Valuation difference for both Adaptive and Censored subjects
plot_corr_attention_valuation(self_lottery_differences_EDRP_censored, 'Adaptive + Censored', samplesize_EDRP_censored)
plot_corr_attention_valuation(charity_lottery_differences_EDRP_censored, 'Adaptive + Censored', samplesize_EDRP_censored)

# Plot Attention vs Valuation difference for both Principal and Censored subjects
plot_corr_attention_valuation(self_lottery_differences_principal_censored, 'Principal + Censored', samplesize_principal_censored)
plot_corr_attention_valuation(charity_lottery_differences_principal_censored, 'Principal + Censored', samplesize_principal_censored)

# We see a general trend that there is a small correlation between attention and
# valuation, which is negative for the self lottery and positive for the charity 
# lottery. We need to verify this statistically

# %%
# =============================================================================
# ANALYSE CORRELATION DATA BETWEEN ATTENTION AND VALUATION 
# =============================================================================

################################################
# Differentiating between self and charity lottery differences
################################################

# Using linear regression between attention and valuation differences

# Adaptive subjects 
# For self lottery difference
reg_model_self_EDRP = sm.OLS(self_lottery_differences_EDRP['valuation_ASPS_ACPS'], 
                                  sm.add_constant(self_lottery_differences_EDRP['dwell_time_ASPS_ACPS'])).fit()
print(reg_model_self_EDRP.summary())

# For charity lottery difference
reg_model_charity_EDRP = sm.OLS(charity_lottery_differences_EDRP['valuation_ACPC_ASPC'], 
                                     sm.add_constant(charity_lottery_differences_EDRP['dwell_time_ACPC_ASPC'])).fit()
print(reg_model_charity_EDRP.summary())


# Censored subjects
# For self lottery difference
reg_model_self_censored = sm.OLS(self_lottery_differences_censored['valuation_ASPS_ACPS'], 
                                 sm.add_constant(self_lottery_differences_censored['dwell_time_ASPS_ACPS'])).fit()
print(reg_model_self_censored.summary())

# For charity lottery difference
reg_model_charity_censored = sm.OLS(charity_lottery_differences_censored['valuation_ACPC_ASPC'], 
                                    sm.add_constant(charity_lottery_differences_censored['dwell_time_ACPC_ASPC'])).fit()
print(reg_model_charity_censored.summary())


# Adaptive and Censored subjects
# For self lottery difference
reg_model_self_EDRP_censored = sm.OLS(self_lottery_differences_EDRP_censored['valuation_ASPS_ACPS'], 
                                      sm.add_constant(self_lottery_differences_EDRP_censored['dwell_time_ASPS_ACPS'])).fit()
print(reg_model_self_EDRP_censored.summary())

# For charity lottery difference
reg_model_charity_EDRP_censored = sm.OLS(charity_lottery_differences_EDRP_censored['valuation_ACPC_ASPC'], 
                                         sm.add_constant(charity_lottery_differences_EDRP_censored['dwell_time_ACPC_ASPC'])).fit()
print(reg_model_charity_EDRP_censored.summary())


# Principal Analysis and Censored subjects
# For self lottery difference
reg_model_self_principal_censored = sm.OLS(self_lottery_differences_principal_censored['valuation_ASPS_ACPS'], 
                                      sm.add_constant(self_lottery_differences_principal_censored['dwell_time_ASPS_ACPS'])).fit()
print(reg_model_self_principal_censored.summary())

# For charity lottery difference
reg_model_charity_principal_censored = sm.OLS(charity_lottery_differences_principal_censored['valuation_ACPC_ASPC'], 
                                         sm.add_constant(charity_lottery_differences_principal_censored['dwell_time_ACPC_ASPC'])).fit()
print(reg_model_charity_principal_censored.summary())





