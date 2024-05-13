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
import matplotlib.cm as cm
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
import ast 

censure = 1 # Put 0 if include censored participants in analysis and 1 if we exclude them 
MSP_excl = 1 # Put 0 if include MSP calib in analysis and 1 if we exclude them 
by_ind = 0 # Put 0 if no display of individual plots and 1 if display 
attention_type = 'absolute' # relative for % of total time and 'absolute' for raw time

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
    
# Get different cases

ASPS = data_for_plot[(data_for_plot['charity'] == 0) & (data_for_plot['tradeoff'] == 0)]
ACPC = data_for_plot[(data_for_plot['charity'] == 1) & (data_for_plot['tradeoff'] == 0)]
ASPC = data_for_plot[(data_for_plot['charity'] == 1) & (data_for_plot['tradeoff'] == 1)]
ACPS = data_for_plot[(data_for_plot['charity'] == 0) & (data_for_plot['tradeoff'] == 1)]

valuation_ASPS = ASPS.groupby('prob_option_A')['valuation']
valuation_ACPS = ACPS.groupby('prob_option_A')['valuation']
valuation_ACPC = ACPC.groupby('prob_option_A')['valuation']
valuation_ASPC = ASPC.groupby('prob_option_A')['valuation']

valuations_all = [valuation_ASPS, valuation_ACPS, valuation_ACPC, valuation_ASPC]

median_valuation_ASPS = valuation_ASPS.median()
median_valuation_ACPS = valuation_ACPS.median()
median_valuation_ACPC = valuation_ACPC.median()
median_valuation_ASPC = valuation_ASPC.median()


# Difference data
self_lottery = pd.concat([ASPS, ACPS], ignore_index = True)
charity_lottery = pd.concat([ACPC, ASPC], ignore_index=True)
no_tradeoff_lottery = pd.concat([ASPS, ACPC], ignore_index=True)

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

no_tradeoff_lottery_differences = pd.DataFrame(columns=['number', 'prob_option_A'])

for i in no_tradeoff_lottery['number'].unique():
    individual = no_tradeoff_lottery.loc[no_tradeoff_lottery['number'] == i, ['case', 'prob_option_A', 'valuation']] 
    individual_difference = individual.pivot(index='prob_option_A', columns='case', values='valuation')
    individual_difference['valuation_ACPC_ASPS'] = individual_difference['ACPC'] - individual_difference['ASPS']
    individual_difference['number'] = i
    individual_difference.reset_index(inplace=True)
    # individual_difference.columns = individual_difference.columns.droplevel(1)
    no_tradeoff_lottery_differences = pd.concat([no_tradeoff_lottery_differences, individual_difference[['number', 'prob_option_A', 'valuation_ACPC_ASPS']]], ignore_index=True)


# MEANS 
mean_valuation_ASPS = valuation_ASPS.mean()
mean_valuation_ACPC = valuation_ACPC.mean()
mean_valuation_ACPS = valuation_ACPS.mean()
mean_valuation_ASPC = valuation_ASPC.mean()

mean_valuations = [mean_valuation_ASPS.mean(), mean_valuation_ACPS.mean(), mean_valuation_ACPC.mean(), mean_valuation_ASPC.mean()]


# Difference of magnitudes
charity_lottery_differences_negated = charity_lottery_differences.copy()  
charity_lottery_differences_negated['valuation_ASPC_ACPC'] *= -1  

t_statistic_diff, p_value_diff = ttest_ind(self_lottery_differences['valuation_ACPS_ASPS'], charity_lottery_differences_negated['valuation_ASPC_ACPC'])
print()
print('Difference of magnitude between self and charity valuation difference:')
print(t_statistic_diff, p_value_diff)


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

self_lottery_difference_EDRP = self_lottery_differences[self_lottery_differences['number'].isin(EDRP_total)]
charity_lottery_differences_EDRP = charity_lottery_differences[charity_lottery_differences['number'].isin(EDRP_total)]


survey_EDRP = pd.merge(data_X_EDRP_total[['id']], survey, on='id', how='inner')
survey_altruistic = pd.merge(data_altruistic[['id']], survey, on='id', how='inner')


print('EDRP SUBJECTS')
print('The mean age is ' + str(survey_EDRP['Demog_AGE'].mean()))
print('There is ' + str(round(100*len(survey_EDRP[survey_EDRP['Demog_Sex']==1])/(len(survey_EDRP[survey_EDRP['Demog_Sex']==1])+
                                                             len(survey_EDRP[survey_EDRP['Demog_Sex']==2])), 1))
                        + ' % of women')
print('The mean highest education level is ' + 
      str(['A level', 'Bsci', 'Msci', 'Phd', 'RNS'][round(survey_EDRP['Demog_High_Ed_Lev'].mean())-1]))
print()



print('ALTRUISTIC SUBJECTS')
print('The mean age is ' + str(survey_altruistic['Demog_AGE'].mean()))
print('There is ' + str(round(100*len(survey_altruistic[survey_altruistic['Demog_Sex']==1])/(len(survey_altruistic[survey_altruistic['Demog_Sex']==1])+
                                                             len(survey_altruistic[survey_altruistic['Demog_Sex']==2])), 1))
                        + ' % of women')
print('The mean highest education level is ' + 
      str(['A level', 'Bsci', 'Msci', 'Phd', 'RNS'][round(survey_altruistic['Demog_High_Ed_Lev'].mean())-1]))
print()


# %%
# =============================================================================
# VISUALISE DATA 
# =============================================================================

x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)

# Plot No Tradeoff Context (Replication Exley)

plt.plot(x_fit, y_fit, color='grey', label='Expected value')

plt.plot(valuation_ASPS.mean().index, valuation_ASPS.mean(), label='$Y^{S}(P^{S})$', color='blue', marker='o', linestyle='-')
plt.plot(valuation_ACPC.mean().index, valuation_ACPC.mean(), label='$Y^{C}(P^{C})$', color='red', marker='o', linestyle='-')

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Valuation as % of Riskless Lottery')
plt.title('Results for No Tradeoff Context ')
plt.grid(True)
plt.legend()
plt.savefig('No Tradeoff H1.png', dpi=1200)
plt.show()

# Plot Tradeoff Context (Replication Exley)

plt.plot(x_fit, y_fit, color='grey', label='Expected value')

plt.plot(valuation_ACPS.mean().index, valuation_ACPS.mean(), label='$Y^{C}(P^{S})$', color='blue', marker='o', linestyle='-')
plt.plot(valuation_ASPC.mean().index, valuation_ASPC.mean(), label='$Y^{S}(P^{C})$', color='red', marker='o', linestyle='-')


x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Valuation as % of Riskless Lottery')
plt.title('Results for Tradeoff Context')
plt.grid(True)
plt.legend()
plt.savefig('Tradeoff H1.png', dpi=1200)
plt.show()

# Plot Self Lottery Valuation

plt.plot(x_fit, y_fit, color='grey', label='Expected value')

plt.plot(valuation_ASPS.mean().index, valuation_ASPS.mean(), label='$Y^{S}(P^{S})$', color='green', marker='o', linestyle='-')
plt.plot(valuation_ACPS.mean().index, valuation_ACPS.mean(), label='$Y^{C}(P^{S})$', color='orange', marker='o', linestyle='-')

x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Valuation as % of Riskless Lottery')
plt.title('Results for Self Lottery Valuation')
plt.grid(True)
plt.legend()
plt.savefig('Self Lottery H1.png', dpi=1200)
plt.show()

# Plot Charity Lottery Valuation

plt.plot(x_fit, y_fit, color='grey', label='Expected value')

plt.plot(valuation_ASPC.mean().index, valuation_ASPC.mean(), label='$Y^{S}(P^{C})$', color='green', marker='o', linestyle='-')
plt.plot(valuation_ACPC.mean().index, valuation_ACPC.mean(), label='$Y^{C}(P^{C})$', color='orange', marker='o', linestyle='-')


x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Valuation as % of Riskless Lottery')
plt.title('Results for Charity Lottery Valuation')
plt.grid(True)
plt.legend()
plt.savefig('Charity Lottery H1.png', dpi=1200)
plt.show()

# Plot all Valuations

offset = 0.015
plt.plot(x_fit, y_fit, color='grey', label='Expected value')

# errors_per_prob = [valuation_ASPS.std(), valuation_ACPS.std(), valuation_ACPC.std(), valuation_ASPC.std()]
# errors_per_prob_mean = [errors_per_prob[i].mean() for i in range(len(errors_per_prob))]
# overall_errors = np.mean(errors_per_prob_mean)

plt.errorbar(valuation_ASPS.mean().index - offset, valuation_ASPS.mean(), valuation_ASPS.std(), ecolor = 'black', fmt='none', alpha=0.5, label='std')
plt.plot(valuation_ASPS.mean().index - offset, valuation_ASPS.mean(), label='$Y^{S}(P^{S})$', color='blue', marker='o', linestyle='-')

plt.errorbar(valuation_ACPS.mean().index - offset/2, valuation_ACPS.mean(), valuation_ACPS.std(), ecolor = 'black', fmt='none', alpha=0.5)
plt.plot(valuation_ACPS.mean().index - offset/2, valuation_ACPS.mean(), label='$Y^{C}(P^{S})$', color='dodgerblue', marker='o', linestyle='-')

plt.errorbar(valuation_ACPC.mean().index + offset/2, valuation_ACPC.mean(), valuation_ACPC.std(), ecolor = 'black', fmt='none', alpha=0.5)
plt.plot(valuation_ACPC.mean().index + offset/2, valuation_ACPC.mean(), label='$Y^{C}(P^{C})$', color='green', marker='o', linestyle='-')

plt.errorbar(valuation_ASPC.mean().index + offset, valuation_ASPC.mean(), valuation_ASPC.std(), ecolor = 'black', fmt='none', alpha=0.5)
plt.plot(valuation_ASPC.mean().index + offset, valuation_ASPC.mean(), label='$Y^{S}(P^{C})$', color='limegreen', marker='o', linestyle='-')

x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Valuation as % of Riskless Lottery')
plt.title('Lottery Valuation for all probabilities')
plt.grid(True)
plt.legend()
plt.savefig('All Lottery H1.png', dpi=1200)
plt.show()


# Plot all Valuations differences

diff_proba_self = self_lottery_differences.groupby('prob_option_A')['valuation_ACPS_ASPS']
diff_proba_charity = charity_lottery_differences.groupby('prob_option_A')['valuation_ASPC_ACPC']

plt.errorbar(diff_proba_self.mean().index - offset/2, diff_proba_self.mean(), diff_proba_self.std(), ecolor = 'black', fmt='none', alpha=0.5, label='std')
plt.plot(diff_proba_self.mean().index - offset/2, diff_proba_self.mean(), label='$Y^{C}(P^{S})-Y^{S}(P^{S})$', color='dodgerblue', marker='o', linestyle='-')

plt.errorbar(diff_proba_charity.mean().index + offset/2, diff_proba_charity.mean(), diff_proba_charity.std(), ecolor = 'black', fmt='none', alpha=0.5)
plt.plot(diff_proba_charity.mean().index + offset/2, diff_proba_charity.mean(), label='$Y^{S}(P^{C})-Y^{C}(P^{C})$', color='limegreen', marker='o', linestyle='-')

x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Valuation as % of Riskless Lottery')
plt.title('Lottery Valuation difference for all probabilities')
plt.grid(True)
plt.legend()
plt.savefig('All Lottery difference H1.png', dpi=1200)
plt.show()


# Valuation without differentiation of probabilities

error_valuation = [np.std(ASPS['valuation']), np.std(ACPS['valuation']), 
                  np.std(ACPC['valuation']), np.std(ASPC['valuation'])]

plt.bar(['$Y^{S}(P^{S})$', '$Y^{C}(P^{S})$', '$Y^{C}(P^{C})$', '$Y^{S}(P^{C})$'], mean_valuations, color = ['blue', 'dodgerblue', 'green', 'limegreen']) 
plt.errorbar(['$Y^{S}(P^{S})$', '$Y^{C}(P^{S})$', '$Y^{C}(P^{C})$', '$Y^{S}(P^{C})$'], mean_valuations, error_valuation, ecolor = 'black', fmt='none', alpha=0.7, label='std')
plt.xlabel('Cas')
plt.ylabel('Moyenne de Valuations en %')
plt.title('Valuation par cas, probabilités confondues')
plt.legend()
plt.savefig('Bar all Lottery H1.png', dpi=1200)
plt.show()


# Plot the difference of valuation 

plt.bar(['$Y^{C}(P^{C})-Y^{S}(P^{S}$)', '$Y^{C}(P^{S})-Y^{S}(P^{S})$', '$Y^{S}(P^{C})-Y^{C}(P^{C})$'], 
        [no_tradeoff_lottery_differences['valuation_ACPC_ASPS'].mean(), self_lottery_differences['valuation_ACPS_ASPS'].mean(), charity_lottery_differences['valuation_ASPC_ACPC'].mean()], 
        color = ['bisque', 'lightskyblue', 'lightgreen']) 

plt.errorbar(['$Y^{C}(P^{C})-Y^{S}(P^{S}$)', '$Y^{C}(P^{S})-Y^{S}(P^{S})$', '$Y^{S}(P^{C})-Y^{C}(P^{C})$'], 
              [no_tradeoff_lottery_differences['valuation_ACPC_ASPS'].mean(), self_lottery_differences['valuation_ACPS_ASPS'].mean(), charity_lottery_differences['valuation_ASPC_ACPC'].mean()], 
              [0.957, 1.771, 2.383], ecolor = 'black', fmt='none', alpha=0.7, label = 'std ind level')
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery difference')
plt.ylabel('Valuation difference in %')
plt.legend()
plt.title('Valuation differences across probabilities H1')
plt.savefig('Bar diff type Lottery H1.png', dpi=1200)
plt.show()

# ALL difference of valuations
plt.hist([self_lottery_differences['valuation_ACPS_ASPS'], charity_lottery_differences['valuation_ASPC_ACPC']], 
        bins = 20, color = ['lightskyblue', 'lightgreen'], label = ['Self lottery', 'Charity lottery']) 
plt.xlabel('Difference in lottery valuation (trad - no trad)')
plt.ylabel('Frequency')
plt.title('Difference in valuation across probabilities')
plt.legend()
plt.savefig('Histo Valuation diff H1.png', dpi=1200)
plt.show()

# Difference of valuations of EDRP

plt.hist([self_lottery_difference_EDRP['valuation_ACPS_ASPS'], charity_lottery_differences_EDRP['valuation_ASPC_ACPC']], 
        bins = 20, color = ['lightskyblue', 'lightgreen'], label = ['Self lottery', 'Charity lottery']) 
plt.xlabel('Difference in lottery valuation (trad - no trad)')
plt.ylabel('Frequency')
plt.title('Difference in valuation for EDRP subjects')
plt.legend()
plt.savefig('Histo Valuation diff EDRP H1.png', dpi=1200)
plt.show()
 


# %%
# =============================================================================
# Participant-specific X values
# =============================================================================


########### Plot ditribution of participant-specific X values 

plt.hist(data_autre['charity_calibration'], bins=20) 
plt.xlabel('Participant-specific X')
plt.ylabel('Frequency')
plt.title('Distribution of X with censored individuals')
plt.show()

plt.hist(data_autre_principal['charity_calibration'], bins=20, color = 'lightcoral') 
plt.xlabel('X values')
plt.ylabel('Frequency')
plt.title('Distribution of participant-specific X values')
plt.axvline(x=data_autre_principal['charity_calibration'].mean(), color='grey', linestyle='--', label = 'Mean = '+ str(np.round(data_autre_principal['charity_calibration'].mean(), 1)))
plt.axvline(x=data_autre_principal['charity_calibration'].median(), color='gainsboro', linestyle='--', label = 'Median = '+ str(np.round(data_autre_principal['charity_calibration'].median(), 1)))
plt.legend()
plt.savefig('X values for everyone.png', dpi=1200)
plt.show()


plt.hist(X_EDRP_total['charity_calibration'], bins=20, color = 'lightcoral') 
plt.xlabel('X values')
plt.ylabel('Frequency')
plt.title('Distribution of X values of EDRP subjects')
plt.axvline(x=X_EDRP_total['charity_calibration'].mean(), color='grey', linestyle='--', label = 'Mean = '+ str(np.round(X_EDRP_total['charity_calibration'].mean(), 1)))
plt.axvline(x=X_EDRP_total['charity_calibration'].median(), color='gainsboro', linestyle='--', label = 'Median = '+ str(np.round(X_EDRP_total['charity_calibration'].median(), 1)))
plt.legend()
plt.savefig('X values for EDRP.png', dpi=1200)
plt.show()

plt.hist(X_else_EDRP_total['charity_calibration'], bins=20, color = 'lightcoral') 
plt.xlabel('X values')
plt.ylabel('Frequency')
plt.title('Distribution of X values of else EDRP subjects')
plt.axvline(x=X_else_EDRP_total['charity_calibration'].mean(), color='grey', linestyle='--', label = 'Mean = '+ str(np.round(X_else_EDRP_total['charity_calibration'].mean(), 1)))
plt.axvline(x=X_else_EDRP_total['charity_calibration'].median(), color='gainsboro', linestyle='--', label = 'Median = '+ str(np.round(X_else_EDRP_total['charity_calibration'].median(), 1)))
plt.legend()
plt.savefig('X values for else EDRP.png', dpi=1200)
plt.show()

plt.hist(X_altruistic['charity_calibration'], bins=20, color = 'lightcoral') 
plt.xlabel('X values')
plt.ylabel('Frequency')
plt.title('Distribution of X values of altruistic subjects')
plt.axvline(x=X_altruistic['charity_calibration'].mean(), color='grey', linestyle='--', label = 'Mean = '+ str(np.round(X_altruistic['charity_calibration'].mean(), 1)))
plt.axvline(x=X_altruistic['charity_calibration'].median(), color='gainsboro', linestyle='--', label = 'Median = '+ str(np.round(X_altruistic['charity_calibration'].median(), 1)))
plt.legend()
plt.savefig('X values for altruistic.png', dpi=1200)
plt.show()

plt.hist(X_no_EDRP_total['charity_calibration'], bins=20, color = 'lightcoral') 
plt.xlabel('X values')
plt.ylabel('Frequency')
plt.title('Distribution of X values of NO EDRP subjects')
plt.axvline(x=X_no_EDRP_total['charity_calibration'].mean(), color='grey', linestyle='--', label = 'Mean = '+ str(np.round(X_no_EDRP_total['charity_calibration'].mean(), 1)))
plt.axvline(x=X_no_EDRP_total['charity_calibration'].median(), color='gainsboro', linestyle='--', label = 'Median = '+ str(np.round(X_no_EDRP_total['charity_calibration'].median(), 1)))
plt.legend()
plt.savefig('X values for NO EDRP.png', dpi=1200)
plt.show()


# Comparison of X-values

# EDRP vs else EDRP
t_statistic, p_value = ttest_ind(X_EDRP_total['charity_calibration'], X_else_EDRP_total['charity_calibration'])
print('t-test and p-value of X diff between EDRP vs else EDRP')
print(t_statistic, p_value)
print()

# EDRP vs everyone
t_statistic_2, p_value_2 = ttest_ind(X_EDRP_total['charity_calibration'], data_autre_principal['charity_calibration'])
print('t-test and p-value of X diff between EDRP vs everyone')
print(t_statistic_2, p_value_2)
print()

# EDRP vs altruistic
t_statistic_3, p_value_3 = ttest_ind(X_EDRP_total['charity_calibration'], X_altruistic['charity_calibration'])
print('t-test and p-value of X diff between EDRP vs altruistic')
print(t_statistic_3, p_value_3)
print()

# EDRP vs no EDRP
t_statistic_4, p_value_4 = ttest_ind(X_EDRP_total['charity_calibration'], X_no_EDRP_total['charity_calibration'])
print('t-test and p-value of X diff between EDRP vs no EDRP')
print(t_statistic_4, p_value_4)
print()

# Altruistic vs no EDRP
t_statistic_5, p_value_5 = ttest_ind(X_altruistic['charity_calibration'], X_no_EDRP_total['charity_calibration'])
print('t-test and p-value of X diff between Altruistic vs no EDRP')
print(t_statistic_5, p_value_5)
print()

# %%
# =============================================================================
# ANALYSE DATA 
# =============================================================================

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
X = pd.concat([X, data_for_analysis[control_variables]], axis=1)
X = sm.add_constant(X, has_constant='add') # add a first column full of ones to account for intercept of regression
y = data_for_analysis['valuation']

# Fit the regression model using Ordinary Least Squares
model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis['number']}) # cluster at individual level
print(model.summary())
  


