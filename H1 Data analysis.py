#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 23:56:31 2024

@author: carolinepioger
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf
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

################################################
# Elicit data specifically checking self, charity and no tradeoff differences of H1
################################################

# Self lottery difference is YCPS-YSPS, Charity lottery difference is YSPC-YCPC
# and No Tradeoff difference is YCPC-YSPS

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

# Self lottery difference is YCPS-YSPS, Charity lottery difference is YSPC-YCPC
# and No Tradeoff difference is YCPC-YSPS

self_lottery_censored = pd.concat([ASPS_censored, ACPS_censored], ignore_index = True)
charity_lottery_censored = pd.concat([ACPC_censored, ASPC_censored], ignore_index=True)
no_tradeoff_lottery_censored = pd.concat([ASPS_censored, ACPC_censored], ignore_index=True)

# Self lottery, charity lottery and no tradeoff differences for Censored subjects
self_lottery_differences_censored = lottery_differences(self_lottery_censored, 'ACPS', 'ASPS') # gives YCPS-YSPS and ACPS-ASPS
charity_lottery_differences_censored = lottery_differences(charity_lottery_censored, 'ASPC', 'ACPC') # gives YSPC-YCPC and ASPC-ACPC
no_tradeoff_lottery_differences_censored = lottery_differences(no_tradeoff_lottery_censored, 'ACPC', 'ASPS') # gives YCPC-YSPS and ACPC-ASPS


# %%
# =============================================================================
# CATEGORISATION OF ADAPTIVE & ALTRUISTIC SUBJECTS and X VALUES
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
    
# Participants being both Excuse-driven for self and for charity -- called Adaptive subjects
EDRP_total = np.intersect1d(EDRP_self, EDRP_charity) 
print()
print('Excuse-driven for self : ' + str(len(EDRP_self)))
print('Excuse-driven for charity : ' + str(len(EDRP_charity)))
print('Adaptive subjects : ' + str(len(EDRP_total)))

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
print()
print('Altruistic for self : ' + str(len(altruistic_self)))
print('Altruistic for charity : ' + str(len(altruistic_charity)))
print('Altruistic subjects : ' + str(len(altruistic_total)))

data_altruistic = data_principal[data_principal['number'].isin(altruistic_total)] # data of Altruistic subjects
data_autre_altruistic = data_autre_principal[data_autre_principal['number'].isin(altruistic_total)] # data_autre of Altruistic subjects
X_altruistic = data_autre_principal[data_autre_principal['number'].isin(altruistic_total)] # X-values of Altruistic subjects

# Participants being neither Adaptive or Altruistic subjects (the rest)
no_EDRP = np.setdiff1d(data_principal['number'].unique(), np.union1d(EDRP_total, altruistic_total))
print()
print('Subjects that are neither adaptive and altruistic: ' + str(len(no_EDRP)))

data_no_EDRP = data_principal[data_principal['number'].isin(no_EDRP)] # X-values of subjects being neither Adaptive or Altruistic
X_no_EDRP_total = data_autre_principal[data_autre_principal['number'].isin(no_EDRP)] # data of subjects being neither Adaptive or Altruistic

# Sample sizes
samplesize_principal = len(data_autre_principal) # sample size of Principal Analysis
samplesize_adaptive = len(data_autre_EDRP) # sample size of Adaptive subjects
samplesize_altruistic = len(data_autre_altruistic) # sample size of Altruistic subjects
samplesize_censored = len(data_autre_censored) # sample size of Censored subjects
samplesize_EDRP_censored = len(data_autre_EDRP) + len(data_autre_censored) # sample size of Adaptive and Censored subjects

################################################
# Socio-demographic information 
################################################

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
mean_val = data_autre_principal['charity_calibration'].mean()
median_val = data_autre_principal['charity_calibration'].median()
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
# VALUATION DATA VISUALIZATION
# =============================================================================

################################################
# Valuation
################################################

lottery_types = ['$Y^{S}(P^{S})$', '$Y^{C}(P^{S})$', '$Y^{C}(P^{C})$', '$Y^{S}(P^{C})$']

# Scales of X- and Y- axis for valuation plots
x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)

# Plot Valuations in the No Tradeoff Context (Principal Analysis)
plt.figure(figsize=(5, 5))
plt.plot(x_fit, y_fit, color='grey', label='Expected value')
plt.plot(valuation_ASPS.mean().index, valuation_ASPS.mean(), label=lottery_types[0], color='blue', marker='o', linestyle='-')
plt.plot(valuation_ACPC.mean().index, valuation_ACPC.mean(), label=lottery_types[2], color='red', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Amount')
plt.ylabel('Valuation (in %)')
plt.title('Valuations for No Tradeoff Context ')
plt.grid(True)
plt.legend()
plt.savefig('No Tradeoff valuations H1.png', dpi=1200)
plt.show()

# Plot Valuations in the No Tradeoff Context (Replication Exley)
# Add the points (0, 0) and (1, 100) to replicate Exley's figures
plt.figure(figsize=(5, 5))
plt.plot(x_fit, y_fit, color='grey', label='Expected value')
valuation_ASPS_mean = valuation_ASPS.mean() if isinstance(valuation_ASPS, pd.core.groupby.SeriesGroupBy) else valuation_ASPS
valuation_ACPC_mean = valuation_ACPC.mean() if isinstance(valuation_ACPC, pd.core.groupby.SeriesGroupBy) else valuation_ACPC
valuation_ASPS_Exley = pd.concat([valuation_ASPS_mean, pd.Series({0: 0, 1: 100})]).sort_index()
valuation_ACPC_Exley = pd.concat([valuation_ACPC_mean, pd.Series({0: 0, 1: 100})]).sort_index()
plt.plot(valuation_ASPS_Exley.index, valuation_ASPS_Exley, label=lottery_types[0], color='blue', marker='o', linestyle='-')
plt.plot(valuation_ACPC_Exley.index, valuation_ACPC_Exley, label=lottery_types[2], color='red', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Valuation as % of Riskless Lottery')
plt.title('Valuations of No Tradeoff Context ')
plt.grid(True)
plt.legend()
plt.savefig('No Tradeoff valuations EXLEY H1.png', dpi=1200)
plt.show()

# Plot Valuations in the Tradeoff Context (Principal Analysis)
plt.figure(figsize=(5, 5))
plt.plot(x_fit, y_fit, color='grey', label='Expected value')
plt.plot(valuation_ACPS.mean().index, valuation_ACPS.mean(), label=lottery_types[1], color='blue', marker='o', linestyle='-')
plt.plot(valuation_ASPC.mean().index, valuation_ASPC.mean(), label=lottery_types[3], color='red', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Amount')
plt.ylabel('Valuation (in %)')
plt.title('Valuations for Tradeoff Context')
plt.grid(True)
plt.legend()
plt.savefig('Tradeoff valuations H1.png', dpi=1200)
plt.show()

# Plot Valuations in the Tradeoff Context (Replication Exley)
# Add the points (0, 0) and (1, 100) to replicate Exley's figures
plt.figure(figsize=(5, 5))
plt.plot(x_fit, y_fit, color='grey', label='Expected value')
valuation_ACPS_mean = valuation_ACPS.mean() if isinstance(valuation_ACPS, pd.core.groupby.SeriesGroupBy) else valuation_ACPS
valuation_ASPC_mean = valuation_ASPC.mean() if isinstance(valuation_ASPC, pd.core.groupby.SeriesGroupBy) else valuation_ASPC
valuation_ACPS_Exley = pd.concat([valuation_ACPS_mean, pd.Series({0: 0, 1: 100})]).sort_index()
valuation_ASPC_Exley = pd.concat([valuation_ASPC_mean, pd.Series({0: 0, 1: 100})]).sort_index()
plt.plot(valuation_ACPS_Exley.index, valuation_ACPS_Exley, label=lottery_types[1], color='blue', marker='o', linestyle='-')
plt.plot(valuation_ASPC_Exley.index, valuation_ASPC_Exley, label=lottery_types[3], color='red', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Valuation as % of Riskless Lottery')
plt.title('Valuations for Tradeoff Context')
plt.grid(True)
plt.legend()
plt.savefig('Tradeoff valuations EXLEY H1.png', dpi=1200)
plt.show()

# Plot Valuations of the Self Lottery (Principal Analysis)
plt.figure(figsize=(5, 5))
plt.plot(x_fit, y_fit, color='grey', label='Expected value')
plt.plot(valuation_ASPS.mean().index, valuation_ASPS.mean(), label=lottery_types[0], color='green', marker='o', linestyle='-')
plt.plot(valuation_ACPS.mean().index, valuation_ACPS.mean(), label=lottery_types[1], color='orange', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Amount')
plt.ylabel('Valuation (in %)')
plt.title('Valuations for Self Lottery')
plt.grid(True)
plt.legend()
plt.savefig('Self Lottery valuations H1.png', dpi=1200)
plt.show()

# Plot Valuation of the Charity Lottery (Principal Analysis)
plt.figure(figsize=(5, 5))
plt.plot(x_fit, y_fit, color='grey', label='Expected value')
plt.plot(valuation_ASPC.mean().index, valuation_ASPC.mean(), label=lottery_types[3], color='green', marker='o', linestyle='-')
plt.plot(valuation_ACPC.mean().index, valuation_ACPC.mean(), label=lottery_types[2], color='orange', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Amount')
plt.ylabel('Valuation (in %)')
plt.title('Valuations for Charity Lottery')
plt.grid(True)
plt.legend()
plt.savefig('Charity Lottery valuations H1.png', dpi=1200)
plt.show()

# Plot 4 Valuations allthogether for all probabilities (Principal Analysis)
offset = 0.015
plt.plot(x_fit, y_fit, color='grey', label='Expected value')
plt.errorbar(valuation_ASPS.mean().index - offset, valuation_ASPS.mean(), valuation_ASPS.std(), ecolor = 'black', fmt='none', alpha=0.5, label='std')
plt.plot(valuation_ASPS.mean().index - offset, valuation_ASPS.mean(), label=lottery_types[0], color='blue', marker='o', linestyle='-')
plt.errorbar(valuation_ACPS.mean().index - offset/2, valuation_ACPS.mean(), valuation_ACPS.std(), ecolor = 'black', fmt='none', alpha=0.5)
plt.plot(valuation_ACPS.mean().index - offset/2, valuation_ACPS.mean(), label=lottery_types[1], color='dodgerblue', marker='o', linestyle='-')
plt.errorbar(valuation_ACPC.mean().index + offset/2, valuation_ACPC.mean(), valuation_ACPC.std(), ecolor = 'black', fmt='none', alpha=0.5)
plt.plot(valuation_ACPC.mean().index + offset/2, valuation_ACPC.mean(), label=lottery_types[2], color='green', marker='o', linestyle='-')
plt.errorbar(valuation_ASPC.mean().index + offset, valuation_ASPC.mean(), valuation_ASPC.std(), ecolor = 'black', fmt='none', alpha=0.5)
plt.plot(valuation_ASPC.mean().index + offset, valuation_ASPC.mean(), label=lottery_types[3], color='limegreen', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Amount')
plt.ylabel('Valuation (in %)')
plt.title('4 Lottery Valuation for all probabilities')
plt.grid(True)
plt.legend()
plt.savefig('All Lottery valuations plot H1.png', dpi=1200)
plt.show()

# Plot 4 Valuations with probabilities combined (Principal Analysis)
error_valuation = [np.std(ASPS_principal['valuation']), np.std(ACPS_principal['valuation']), 
                  np.std(ACPC_principal['valuation']), np.std(ASPC_principal['valuation'])]
plt.bar(lottery_types, mean_valuations, color = ['blue', 'dodgerblue', 'green', 'limegreen']) 
plt.errorbar(lottery_types, mean_valuations, error_valuation, ecolor = 'black', fmt='none', alpha=0.7, label='std')
plt.xlabel('Case')
plt.ylabel('Valuation (in %)')
plt.title('4 Lottery Valuation with probabilities combined')
plt.legend()
plt.savefig('All Lottery valuations bar H1.png', dpi=1200)
plt.show()


################################################
# Valuation differences
################################################

# Now we are interested in valuation DIFFERENCES, namely YCPC-YSPS, YCPS-YSPS and YSPC-YCPC
# To verify for H1, we check for null, positive and negative differences respectively

lottery_types_difference = ['$Y^{C}(P^{C})-Y^{S}(P^{S})$', 
                            '$Y^{C}(P^{S})-Y^{S}(P^{S})$', 
                            '$Y^{S}(P^{C})-Y^{C}(P^{C})$']
x = np.arange(len(lottery_types_difference))
offset_2 = 0.02

# 3 valuation differences and standard errors at ind level for Principal Analysis, Adaptive and Censored subjects
# for Principal Analysis
principal_means = [no_tradeoff_lottery_differences_principal['valuation_ACPC_ASPS'].mean(),
                   self_lottery_differences_principal['valuation_ACPS_ASPS'].mean(),
                   charity_lottery_differences_principal['valuation_ASPC_ACPC'].mean()]
principal_errors = [0.778, 1.385, 1.898]           ################## CHANGER 

# for Adaptive subjects
EDRP_means = [no_tradeoff_lottery_differences_EDRP['valuation_ACPC_ASPS'].mean(), 
              self_lottery_differences_EDRP['valuation_ACPS_ASPS'].mean(),
              charity_lottery_differences_EDRP['valuation_ASPC_ACPC'].mean()]
EDRP_errors = [1.240, 2.317, 2.8548]               ################## CHANGER 

# for Censored subjects
censored_means = [no_tradeoff_lottery_differences_censored['valuation_ACPC_ASPS'].mean(), 
                  self_lottery_differences_censored['valuation_ACPS_ASPS'].mean(),
                  charity_lottery_differences_censored['valuation_ASPC_ACPC'].mean()]
censored_errors = [1.742, 3.042, 3.883]            ################## CHANGER 


# Plot 3 Valuation differences for all probabilities (Principal Analysis)
plt.axhline(y=0, color='grey', linestyle='--')
diff_proba_no_tradeoff = no_tradeoff_lottery_differences_principal.groupby('prob_option_A')['valuation_ACPC_ASPS']
diff_proba_self = self_lottery_differences_principal.groupby('prob_option_A')['valuation_ACPS_ASPS']
diff_proba_charity = charity_lottery_differences_principal.groupby('prob_option_A')['valuation_ASPC_ACPC']
plt.errorbar(diff_proba_no_tradeoff.mean().index - offset_2/2, diff_proba_no_tradeoff.mean(), diff_proba_no_tradeoff.std(), ecolor = 'black', fmt='none', alpha=0.4, label='std')
plt.plot(diff_proba_no_tradeoff.mean().index - offset_2/2, diff_proba_no_tradeoff.mean(), label=lottery_types_difference[0], color='bisque', marker='o', linestyle='-')
plt.errorbar(diff_proba_self.mean().index, diff_proba_self.mean(), diff_proba_self.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_self.mean().index, diff_proba_self.mean(), label=lottery_types_difference[1], color='dodgerblue', marker='o', linestyle='-')
plt.errorbar(diff_proba_charity.mean().index + offset_2/2, diff_proba_charity.mean(), diff_proba_charity.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_charity.mean().index + offset_2/2, diff_proba_charity.mean(), label=lottery_types_difference[2], color='limegreen', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Amount')
plt.ylabel('Valuation difference in %')
plt.title('Valuation differences for Principal Analysis')
plt.legend()
plt.savefig('All Lottery difference plot Principal H1.png', dpi=1200)
plt.show()

# Plot 3 Valuation differences for all probabilities (Censored subjects)
plt.axhline(y=0, color='grey', linestyle='--')
diff_proba_self_censored = self_lottery_differences_censored.groupby('prob_option_A')['valuation_ACPS_ASPS']
diff_proba_charity_censored = charity_lottery_differences_censored.groupby('prob_option_A')['valuation_ASPC_ACPC']
diff_proba_no_tradeoff_censored = no_tradeoff_lottery_differences_censored.groupby('prob_option_A')['valuation_ACPC_ASPS']
plt.errorbar(diff_proba_no_tradeoff_censored.mean().index - offset_2/2, diff_proba_no_tradeoff_censored.mean(), diff_proba_no_tradeoff_censored.std(), ecolor = 'black', fmt='none', alpha=0.4, label='std')
plt.plot(diff_proba_no_tradeoff_censored.mean().index - offset_2/2, diff_proba_no_tradeoff_censored.mean(), label=lottery_types_difference[0], color='bisque', marker='o', linestyle='-')
plt.errorbar(diff_proba_self_censored.mean().index, diff_proba_self_censored.mean(), diff_proba_self_censored.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_self_censored.mean().index, diff_proba_self_censored.mean(), label=lottery_types_difference[1], color='dodgerblue', marker='o', linestyle='-')
plt.errorbar(diff_proba_charity_censored.mean().index + offset_2/2, diff_proba_charity_censored.mean(), diff_proba_charity_censored.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_charity_censored.mean().index + offset_2/2, diff_proba_charity_censored.mean(), label=lottery_types_difference[2], color='limegreen', marker='o', linestyle='-')
plt.xlabel('Probability P of Non-Zero Amount')
plt.ylabel('Valuation difference in %')
plt.title('Valuation differences for Censored subjects')
plt.legend()
plt.savefig('All Lottery difference plot Censored H1.png', dpi=1200)
plt.show()

# Plot 3 Valuation differences with probabilities combined (Principal Analysis)
plt.bar(lottery_types_difference, principal_means, color = ['bisque', 'lightskyblue', 'lightgreen']) 
plt.errorbar(lottery_types_difference, principal_means, principal_errors, ecolor = 'black', fmt='none', alpha=0.7, label = 'std ind level')
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery differences')
plt.ylabel('Valuation difference in %')
plt.text(0.15, 0.9, f'n = {samplesize_principal}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
plt.legend()
plt.title('Valuation differences with probabilities combined Principal Analysis')
plt.savefig('All Lottery difference bar H1.png', dpi=1200)
plt.show()

# Plot 3 Valuation differences with probabilities combined (Adaptive Subjects)
plt.bar(lottery_types_difference, EDRP_means, color = ['bisque', 'lightskyblue', 'lightgreen']) 
plt.errorbar(lottery_types_difference, EDRP_means, EDRP_errors, ecolor = 'black', fmt='none', alpha=0.7, label = 'std ind level')
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery differences')
plt.ylabel('Valuation difference in %')
plt.text(0.15, 0.9, f'n = {samplesize_adaptive}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
plt.legend()
plt.title('Valuation differences for Adaptive subjects')
plt.savefig('Lottery differences Adaptive H1.png', dpi=1200)
plt.show()

# Plot 3 Valuation differences with probabilities combined (Censored Subjects)
plt.bar(lottery_types_difference, censored_means, color = ['bisque', 'lightskyblue', 'lightgreen']) 
plt.errorbar(lottery_types_difference, censored_means, censored_errors, ecolor = 'black', fmt='none', alpha=0.7, label = 'std ind level')
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery differences')
plt.ylabel('Valuation difference in %')
plt.text(0.15, 0.9, f'n = {samplesize_censored}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
plt.legend()
plt.title('Valuation differences for Censored subjects')
plt.savefig('Lottery differences Censored H1.png', dpi=1200)
plt.show()

# Plot Valuation differences between Principal Analysis and Censored
width = 0.35
plt.bar(x - width/2, principal_means, width, yerr=principal_errors, capsize=5, color=['bisque', 'lightskyblue', 'lightgreen'], label='Principal analysis')
plt.bar(x + width/2, censored_means, width, yerr=censored_errors, capsize=5, color=['bisque', 'lightskyblue', 'lightgreen'], hatch="//", label='Censored')
plt.xlabel('Lottery type')
plt.ylabel('Difference in valuation in %')
plt.title('Difference in valuation for Principal analysis and Censored subjects H1')
plt.xticks(x, lottery_types_difference)
plt.axhline(y=0, color='grey', linestyle='--')
proxy_artists = [Patch(facecolor='white', edgecolor='black', label=f'Principal analysis n = {samplesize_principal}'),
                 Patch(facecolor='white', edgecolor='black', hatch="//", label=f'Censored n = {samplesize_censored}')]
plt.legend(handles=proxy_artists)
plt.savefig('Merged Valuation Principal Analysis and Censored H1.png', dpi=1200)
plt.show()

# Plot Valuation differences between Adaptive and Censored
plt.bar(x - width/2, EDRP_means, width, yerr=EDRP_errors, capsize=5, color=['bisque', 'lightskyblue', 'lightgreen'], label='Adaptive')
plt.bar(x + width/2, censored_means, width, yerr=censored_errors, capsize=5, color=['bisque', 'lightskyblue', 'lightgreen'], hatch="//", label='Censored')
plt.xlabel('Lottery type')
plt.ylabel('Difference in valuation in %')
plt.title('Difference in valuation for Adaptive and Censored subjects H1')
plt.xticks(x, lottery_types_difference)
plt.axhline(y=0, color='grey', linestyle='--')
proxy_artists = [Patch(facecolor='white', edgecolor='black', label=f'Adaptive n = {samplesize_adaptive}'),
                 Patch(facecolor='white', edgecolor='black', hatch="//", label=f'Censored n = {samplesize_censored}')]
plt.legend(handles=proxy_artists)
plt.savefig('Merged Valuation Adaptive and Censored H1.png', dpi=1200)
plt.show()


# Histogram of the self and charity valuation differences of Principal Analysis
plt.hist([self_lottery_differences_principal['valuation_ACPS_ASPS'], charity_lottery_differences_principal['valuation_ASPC_ACPC']], 
        bins = 20, color = ['lightskyblue', 'lightgreen'], label = lottery_types_difference[1:3]) 
plt.xlabel('Difference in lottery valuation (trad - no trad)')
plt.ylabel('Frequency')
plt.title('Self and charity valuation differences across probabilities')
plt.legend()
plt.savefig('Histo Valuation difference Principal H1.png', dpi=1200)
plt.show()

# Histogram of the self and charity valuation differences of Adaptive subjects
plt.hist([self_lottery_differences_EDRP['valuation_ACPS_ASPS'], charity_lottery_differences_EDRP['valuation_ASPC_ACPC']], 
        bins = 20, color = ['lightskyblue', 'lightgreen'], label = lottery_types_difference[1:3]) 
plt.xlabel('Difference in lottery valuation (trad - no trad)')
plt.ylabel('Frequency')
plt.title('Self and charity valuation differences across probabilities')
plt.legend()
plt.savefig('Histo Valuation difference Adaptive H1.png', dpi=1200)
plt.show()
 

################################################
# Valuation according to case order and probability
################################################

# Check the effect of order in which case are presented on valuation of lotteries

cases = ['first', 'second', 'third', 'fourth']

first_case = data_principal[data_principal['case_order']==1] # valuation from the first case presented
second_case = data_principal[data_principal['case_order']==2] # valuation from the second case presented
third_case = data_principal[data_principal['case_order']==3] # valuation from the third case presented
fourth_case = data_principal[data_principal['case_order']==4] # valuation from the fourth case presented

case_order  = [first_case['valuation'].mean(), second_case['valuation'].mean(), 
               third_case['valuation'].mean(), fourth_case['valuation'].mean()]
case_order_std = [first_case['valuation'].std(), second_case['valuation'].std(), 
                  third_case['valuation'].std(), fourth_case['valuation'].std()]

plt.bar(cases, case_order, color = ['dimgray', 'darkgray', 'silver', 'lightgrey']) 
plt.errorbar(cases, case_order, case_order_std, ecolor = 'black', fmt='none', alpha=0.7, label = 'std')
plt.xlabel('Case order')
plt.ylabel('Valuation (in %)')
plt.title('Effect of case order on valuation (all cases combined)')
plt.savefig('Valuation case order H1.png', dpi=1200)
plt.show()

# Effect of case order on attention using mixed effects model
model_case_order = smf.mixedlm("valuation ~ case_order", data_principal, groups=data_principal["number"]).fit()
print(model_case_order.summary())

# We find that individuals generally value less lotteries in the following case
# which suggest we should control for case order in analysis 

# Check the effect of probability on valuation of lotteries 

# We group valuations according to probabilities (regardless of case and order)
valuation_per_proba = data_principal.groupby('prob_option_A')['valuation'] 
probabilities = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

plt.plot(x_fit, y_fit, color='grey', label='Expected value')
plt.plot(probabilities, valuation_per_proba.mean(), color='black', marker='o', linestyle='-')
plt.errorbar(probabilities, valuation_per_proba.mean(), valuation_per_proba.std(), ecolor = 'black', fmt='none', alpha=0.7, label = 'std')
plt.xlabel('Probability')
plt.ylabel('Valuation (in %)')
plt.title('Effect of probability on valuation (all cases combined)')
plt.savefig('Valuation probability H1.png', dpi=1200)
plt.show()

# Effect of probability on valuation using mixed effects model
model_proba = smf.mixedlm("valuation ~ prob_option_A", data_principal, groups=data_principal["number"]).fit()
print(model_proba.summary())

# We indeed find the standard empirical finding in risky decision-making that 
# the valuation is superior to expected value for small probabilities and inferior
# for high probabilities meaning that individuals are generally risk seeking for 
# small probabilities and more risk averse for high probabilities

# Importantly, we need to compare the valuation of lotteries with the same probability


# %%
# =============================================================================
# DIFFERENCES OF MAGNITUDES
# =============================================================================

# We study the magnitudes of the self, charity and no tradeoff lottery differences
# Thus we compare the absolute values of YCPS-YSPS, YSPC-YCPC and YCPC-YSPS

################################################
# Principal analysis
################################################

t_statistic_diff, p_value_diff = ttest_ind(self_lottery_differences_principal['valuation_ACPS_ASPS'].abs(), charity_lottery_differences_principal['valuation_ASPC_ACPC'].abs())
print()
print('PRINCIPAL ANALYSIS')
print('Difference of magnitude between self and charity valuation difference for principal analysis (t-test, p value):')
print(t_statistic_diff, p_value_diff)

################################################
# Censored subjects
################################################

t_statistic_diff_censored, p_value_diff_censored = ttest_ind(self_lottery_differences_censored['valuation_ACPS_ASPS'].abs(), charity_lottery_differences_censored['valuation_ASPC_ACPC'].abs())
print()
print('CENSORED SUBJECTS')
print('Difference of magnitude between self and charity valuation difference for censored subjects (t-test, p value):')
print(t_statistic_diff_censored, p_value_diff_censored)
print()

################################################
# BETWEEN Principal analysis and Censored subjects 
################################################

print('BETWEEN Principal analysis and Censored subjects ')

t_statistic_no_tradeoff, p_value_no_tradeoff = ttest_ind(no_tradeoff_lottery_differences_principal['valuation_ACPC_ASPS'], no_tradeoff_lottery_differences_censored['valuation_ACPC_ASPS'])
print('Difference of magnitude of No tradeoff difference between Principal analysis and censored (t-test, p value)')
print(t_statistic_no_tradeoff, p_value_no_tradeoff)
print()

t_statistic_self, p_value_self = ttest_ind(self_lottery_differences_principal['valuation_ACPS_ASPS'], self_lottery_differences_censored['valuation_ACPS_ASPS'])
print('Difference of magnitude of Self difference between Principal analysis and censored (t-test, p value)')
print(t_statistic_self, p_value_self)
print()

t_statistic_charity, p_value_charity = ttest_ind(charity_lottery_differences_principal['valuation_ASPC_ACPC'], charity_lottery_differences_censored['valuation_ASPC_ACPC'])
print('Difference of magnitude of Charity difference between Principal analysis and censored (t-test, p value)')
print(t_statistic_charity, p_value_charity)
print()


################################################
# BETWEEN Adaptive and Censored subjects 
################################################

print('BETWEEN Adaptive and Censored subjects ')

t_statistic_no_tradeoff_EDRP_censored, p_value_no_tradeoff_EDRP_censored = ttest_ind(no_tradeoff_lottery_differences_EDRP['valuation_ACPC_ASPS'], 
                                                                                     no_tradeoff_lottery_differences_censored['valuation_ACPC_ASPS'])
print('Difference of magnitude of No Tradeoff difference between Adaptive and censored (t-test, p value)')
print(t_statistic_no_tradeoff_EDRP_censored, p_value_no_tradeoff_EDRP_censored)
print()

t_statistic_self_EDRP_censored, p_value_self_EDRP_censored = ttest_ind(self_lottery_differences_EDRP['valuation_ACPS_ASPS'], self_lottery_differences_censored['valuation_ACPS_ASPS'])
print('Difference of magnitudeof Self difference between Adaptive and censored (t-test, p value)')
print(t_statistic_self_EDRP_censored, p_value_self_EDRP_censored)
print()

t_statistic_charity_EDRP_censored, p_value_charity_EDRP_censored = ttest_ind(charity_lottery_differences_EDRP['valuation_ASPC_ACPC'], charity_lottery_differences_censored['valuation_ASPC_ACPC'])
print('Difference of magnitude of Charity difference between Adaptive and censored (t-test, p value)')
print(t_statistic_charity_EDRP_censored, p_value_charity_EDRP_censored)
print()


# %%
# =============================================================================
# ANALYSE VALUATION DATA 
# =============================================================================

################################################
# Verifying H1 through fixed effect regression models from Exley
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
fixed_model_principal = fixed_regression_model(data_principal, 'valuation', ['charity', 'tradeoff', 'interaction', 'case_order'], 'yes')
fixed_model_principal.to_csv('Principal analysis Fixed regression results H1.csv')

# Adaptive subjects
fixed_model_EDRP = fixed_regression_model(data_EDRP, 'valuation', ['charity', 'tradeoff', 'interaction', 'case_order'], 'yes')
fixed_model_EDRP.to_csv('Adaptive Fixed regression results H1.csv')

# Censored subjects
fixed_model_censored = fixed_regression_model(data_censored, 'valuation', ['charity', 'tradeoff', 'interaction', 'case_order'], 'yes')
fixed_model_censored.to_csv('Censored Fixed regression results H1.csv')

# Principal Analysis and Censored subjects (replication of Exley)
data_for_analysis_principal_and_censored = pd.concat([data_principal, data_censored], 
                                                     ignore_index=True) # Data specifically for Principal Analysis and Censored subjects 
fixed_model_principal_and_censored = fixed_regression_model(data_for_analysis_principal_and_censored, 'valuation', ['charity', 'tradeoff', 'interaction', 'case_order'], 'yes')
fixed_model_principal_and_censored.to_csv('Principal analysis and Censored Fixed regression results H1.csv')

################################################
# Heterogeneous effects of probabilities
################################################

# Although not part of H1, we observe heterogeneous effects of probabilities in 
# the self and charity valuation difference (YCPS-YSPS and YSPC-YCPC respectively)
# More specifically, in Principal Analysis, we observe that the valuation difference 
# switches signs for high proba for the self valuation difference and for small 
# prob for the charity valuation difference (and converges to 0 for Censored subjects)

# PRINCIPAL ANALYSIS
# For the no tradeoff difference YCPC-YSPS
model_no_tradeoff_principal = fixed_regression_model(no_tradeoff_lottery_differences_principal, 'valuation_ACPC_ASPS', [], 'yes')
model_no_tradeoff_principal.to_csv('No tradeoff principal analysis Fixed regression results.csv')

# For the self lottery difference YCPS-YSPS
model_self_principal = fixed_regression_model(self_lottery_differences_principal, 'valuation_ACPS_ASPS', [], 'yes')
model_self_principal.to_csv('Self principal analysis Fixed regression results.csv')

# For the charity lottery difference YSPC-YCPC
model_charity_principal = fixed_regression_model(charity_lottery_differences_principal, 'valuation_ASPC_ACPC', [], 'yes')
model_charity_principal.to_csv('Charity principal analysis Fixed regression results.csv')


# CENSORED SUBJECTS
# For the no tradeoff difference YCPC-YSPS
model_no_tradeoff_censored = fixed_regression_model(no_tradeoff_lottery_differences_censored, 'valuation_ACPC_ASPS', [], 'yes')
model_no_tradeoff_censored.to_csv('No tradeoff censored subjects Fixed regression results.csv')

# For the self lottery difference YCPS-YSPS
model_self_censored = fixed_regression_model(self_lottery_differences_censored, 'valuation_ACPS_ASPS', [], 'yes')
model_self_censored.to_csv('Self censored subjects Fixed regression results.csv')

# For the charity lottery difference YSPC-YCPC
model_charity_censored = fixed_regression_model(charity_lottery_differences_censored, 'valuation_ASPC_ACPC', [], 'yes')
model_charity_censored.to_csv('Charity censored subjects Fixed regression results.csv')


# %%
# =============================================================================
# Simulation with sample size of Exley and Garcia
# =============================================================================

# iteration_number = 100 # Number of iterations of simulation per sample size
# sample_Exley = 57 # Exley's sample size is 57
# sample_Garcia =107 #  Garcia et al's sample size is 107

# def simulation_power_charity_coef(sample, iteration):
#     p_values = np.zeros(iteration) # variable to collect p-values for each iteration
#     for inter in range(1, iteration): # run simulation for a set number of iterations
#         # pick random subjects from our sample ("sample" number) - drawn with replacement (same subject can be drawn multiple times)    
#         subjects_drawn = np.random.choice(np.unique(data_principal['number']), sample) 
#         data_drawn = []
#         for subj in subjects_drawn:
#             # extract data from these randomly picjed subjects
#             subj_data = data_principal.loc[data_principal['number'] == subj, ['number', 'prob_option_A', 'valuation', 'charity', 'tradeoff', 'interaction']]
#             data_drawn.append(subj_data)
#         data_drawn = pd.concat(data_drawn)
        
#         try:
#             # to replicate exactly Exley's regression (not taking into account the order of case)
#             test = fixed_regression_model(data_drawn, 'valuation', ['charity', 'tradeoff', 'interaction'], 'no') 
#             coef_charity = test['P>|z|']['charity'] # extract for each sample size tested the p-value of the charity variable
#             p_values[inter] = coef_charity # collect this p-value for each iteration
            
#         except np.linalg.LinAlgError:
#             print()
#             print("Singular matrix encountered.")
#             print()
#             p_values[inter] = 1
#         except ZeroDivisionError:
#             print()
#             print("Multicollinearity encountered.")
#             print()
#             p_values[inter] = 1  
            
#     power_calculated = np.mean(p_values < 0.05) # we find the power by average significance level over iterations
#     return p_values, power_calculated


# # Power using Exley's sample size and our data

# p_val_Exley, power_Exley = simulation_power_charity_coef(sample_Exley, iteration_number)

# print()
# print()
# print('Using Exley sample size, the charity coefficient is significant for ' 
#       + str(power_Exley*100) + '% of iterations')
# print()

# # Power using Garcia's sample size and our data

# p_val_Garcia, power_Garcia = simulation_power_charity_coef(sample_Garcia, iteration_number)

# print()
# print()
# print('Using Garcia et al sample size, the charity coefficient is significant for ' 
#       + str(power_Exley*100) + '% of iterations')
# print()

# # Power using our sample size and data

# p_val_us, power_us = simulation_power_charity_coef(183, iteration_number)

# print()
# print()
# print('Using our sample size, the charity coefficient is significant for ' 
#       + str(power_us*100) + '% of iterations')
# print()


