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

threshold_EDRP = 4

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

# Notes analyses
# len(data[data['valuation']==100])/len(data['valuation']) # % censored valuations
# len(data[(data['nb_switchpoint']!=1) & (data['nb_switchpoint']!=0)])/len(data['nb_switchpoint']) # % MSP valuations
# sum(data['watching_urn_ms'].map(lambda arr: any(x <= 200 for x in arr)))/len(data['watching_urn_ms']) # % of dwell time with <=200 value


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


survey_EDRP = pd.merge(data_X_EDRP_total[['id']], survey, on='id', how='inner')
survey_altruistic = pd.merge(data_altruistic[['id']], survey, on='id', how='inner')

ASPS_EDRP = data_X_EDRP_total[(data_X_EDRP_total['charity'] == 0) & (data_X_EDRP_total['tradeoff'] == 0)]
ACPC_EDRP = data_X_EDRP_total[(data_X_EDRP_total['charity'] == 1) & (data_X_EDRP_total['tradeoff'] == 0)]
ASPC_EDRP = data_X_EDRP_total[(data_X_EDRP_total['charity'] == 1) & (data_X_EDRP_total['tradeoff'] == 1)]
ACPS_EDRP = data_X_EDRP_total[(data_X_EDRP_total['charity'] == 0) & (data_X_EDRP_total['tradeoff'] == 1)]

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
offset_2 = 0.02
plt.axhline(y=0, color='grey', linestyle='--')

diff_proba_self = self_lottery_differences.groupby('prob_option_A')['valuation_ACPS_ASPS']
diff_proba_charity = charity_lottery_differences.groupby('prob_option_A')['valuation_ASPC_ACPC']
diff_proba_no_tradeoff = no_tradeoff_lottery_differences.groupby('prob_option_A')['valuation_ACPC_ASPS']

plt.errorbar(diff_proba_no_tradeoff.mean().index - offset_2/2, diff_proba_no_tradeoff.mean(), diff_proba_no_tradeoff.std(), ecolor = 'black', fmt='none', alpha=0.4, label='std')
plt.plot(diff_proba_no_tradeoff.mean().index - offset_2/2, diff_proba_no_tradeoff.mean(), label='$Y^{C}(P^{S})-Y^{S}(P^{S})$', color='bisque', marker='o', linestyle='-')

plt.errorbar(diff_proba_self.mean().index, diff_proba_self.mean(), diff_proba_self.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_self.mean().index, diff_proba_self.mean(), label='$Y^{C}(P^{S})-Y^{S}(P^{S})$', color='dodgerblue', marker='o', linestyle='-')

plt.errorbar(diff_proba_charity.mean().index + offset_2/2, diff_proba_charity.mean(), diff_proba_charity.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_charity.mean().index + offset_2/2, diff_proba_charity.mean(), label='$Y^{S}(P^{C})-Y^{C}(P^{C})$', color='limegreen', marker='o', linestyle='-')

x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Valuation as % of Riskless Lottery')
plt.title('Lottery Valuation difference for all probabilities')
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
              [0.825, 1.456, 1.991], ecolor = 'black', fmt='none', alpha=0.7, label = 'std ind level')
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery difference')
plt.ylabel('Valuation difference in %')
plt.legend()
plt.title('Valuation differences across probabilities H1')
plt.savefig('Bar diff type Lottery H1.png', dpi=1200)
plt.show()

# Hist ALL difference of valuations
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
 
# Case order effect
first_case = data_for_plot[data_for_plot['case_order']==1]
second_case = data_for_plot[data_for_plot['case_order']==2]
third_case = data_for_plot[data_for_plot['case_order']==3]
fourth_case = data_for_plot[data_for_plot['case_order']==4]

plt.bar(['first', 'second', 'third', 'fourth'], [first_case['valuation'].mean(), second_case['valuation'].mean(), 
                                               third_case['valuation'].mean(), fourth_case['valuation'].mean()], 
        color = ['dimgray', 'darkgray', 'silver', 'lightgrey']) 
plt.errorbar(['first', 'second', 'third', 'fourth'], 
             [first_case['valuation'].mean(), second_case['valuation'].mean(), third_case['valuation'].mean(), fourth_case['valuation'].mean()], 
              [first_case['valuation'].std(), second_case['valuation'].std(), third_case['valuation'].std(), fourth_case['valuation'].std()], 
              ecolor = 'black', fmt='none', alpha=0.7, label = 'std ind level')
plt.xlabel('Case order')
plt.ylabel('Mean valuation in %')
plt.title('Mean valuation per cas order')
plt.savefig('Valuation case order H1.png', dpi=1200)
plt.show()


# Proba effect

plt.plot(x_fit, y_fit, color='grey', label='Expected value')

valuation_per_proba = data_for_plot.groupby('prob_option_A')['valuation']

plt.plot([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], valuation_per_proba.mean(), color='black', marker='o', linestyle='-')
plt.xlabel('Probability')
plt.ylabel('Mean valuation in %')
plt.title('Mean valuation per probability for all cases')
plt.savefig('Valuation probability H1.png', dpi=1200)
plt.show()


# %%
# =============================================================================
# Data for censored participants
# =============================================================================

ASPS_censored = data_censored[(data_censored['charity'] == 0) & (data_censored['tradeoff'] == 0)]
ACPC_censored = data_censored[(data_censored['charity'] == 1) & (data_censored['tradeoff'] == 0)]
ASPC_censored = data_censored[(data_censored['charity'] == 1) & (data_censored['tradeoff'] == 1)]
ACPS_censored = data_censored[(data_censored['charity'] == 0) & (data_censored['tradeoff'] == 1)]


# Difference data
self_lottery_censored = pd.concat([ASPS_censored, ACPS_censored], ignore_index = True)
charity_lottery_censored = pd.concat([ACPC_censored, ASPC_censored], ignore_index=True)
no_tradeoff_lottery_censored = pd.concat([ASPS_censored, ACPC_censored], ignore_index=True)

self_lottery_differences_censored = pd.DataFrame(columns=['number', 'prob_option_A'])

for i in self_lottery_censored['number'].unique():
    individual = self_lottery_censored.loc[self_lottery_censored['number'] == i, ['case', 'prob_option_A', 'valuation']] 
    individual_difference = individual.pivot(index='prob_option_A', columns='case', values='valuation')
    individual_difference['valuation_ACPS_ASPS'] = individual_difference['ACPS'] - individual_difference['ASPS']
    individual_difference['number'] = i
    individual_difference.reset_index(inplace=True)
    # individual_difference.columns = individual_difference.columns.droplevel(1)
    self_lottery_differences_censored = pd.concat([self_lottery_differences_censored, individual_difference[['number', 'prob_option_A', 'valuation_ACPS_ASPS']]], ignore_index=True)

charity_lottery_differences_censored = pd.DataFrame(columns=['number', 'prob_option_A'])

for i in charity_lottery_censored['number'].unique():
    individual = charity_lottery_censored.loc[charity_lottery_censored['number'] == i, ['case', 'prob_option_A', 'valuation']] 
    individual_difference = individual.pivot(index='prob_option_A', columns='case', values='valuation')
    individual_difference['valuation_ASPC_ACPC'] = individual_difference['ASPC'] - individual_difference['ACPC']
    individual_difference['number'] = i
    individual_difference.reset_index(inplace=True)
    # individual_difference.columns = individual_difference.columns.droplevel(1)
    charity_lottery_differences_censored = pd.concat([charity_lottery_differences_censored, individual_difference[['number', 'prob_option_A', 'valuation_ASPC_ACPC']]], ignore_index=True)

no_tradeoff_lottery_differences_censored = pd.DataFrame(columns=['number', 'prob_option_A'])

for i in no_tradeoff_lottery_censored['number'].unique():
    individual = no_tradeoff_lottery_censored.loc[no_tradeoff_lottery_censored['number'] == i, ['case', 'prob_option_A', 'valuation']] 
    individual_difference = individual.pivot(index='prob_option_A', columns='case', values='valuation')
    individual_difference['valuation_ACPC_ASPS'] = individual_difference['ACPC'] - individual_difference['ASPS']
    individual_difference['number'] = i
    individual_difference.reset_index(inplace=True)
    # individual_difference.columns = individual_difference.columns.droplevel(1)
    no_tradeoff_lottery_differences_censored = pd.concat([no_tradeoff_lottery_differences_censored, individual_difference[['number', 'prob_option_A', 'valuation_ACPC_ASPS']]], ignore_index=True)


plt.bar(['$Y^{C}(P^{C})-Y^{S}(P^{S}$)', '$Y^{C}(P^{S})-Y^{S}(P^{S})$', '$Y^{S}(P^{C})-Y^{C}(P^{C})$'], 
        [no_tradeoff_lottery_differences_censored['valuation_ACPC_ASPS'].mean(), self_lottery_differences_censored['valuation_ACPS_ASPS'].mean(), charity_lottery_differences_censored['valuation_ASPC_ACPC'].mean()], 
        color = ['bisque', 'lightskyblue', 'lightgreen']) 

plt.errorbar(['$Y^{C}(P^{C})-Y^{S}(P^{S}$)', '$Y^{C}(P^{S})-Y^{S}(P^{S})$', '$Y^{S}(P^{C})-Y^{C}(P^{C})$'], 
              [no_tradeoff_lottery_differences_censored['valuation_ACPC_ASPS'].mean(), self_lottery_differences_censored['valuation_ACPS_ASPS'].mean(), charity_lottery_differences_censored['valuation_ASPC_ACPC'].mean()], 
              [1.739, 3.259, 4.1035], ecolor = 'black', fmt='none', alpha=0.7, label = 'std ind level')
plt.axhline(y=0, color='grey', linestyle='--')
plt.xlabel('Lottery difference')
plt.ylabel('Valuation difference in %')
plt.legend()
plt.title('Valuation differences for Censored subjects')
plt.savefig('Bar diff type Lottery Censored H1.png', dpi=1200)
plt.show()

# Across probabilities

offset_2 = 0.02
plt.axhline(y=0, color='grey', linestyle='--')

diff_proba_self_censored = self_lottery_differences_censored.groupby('prob_option_A')['valuation_ACPS_ASPS']
diff_proba_charity_censored = charity_lottery_differences_censored.groupby('prob_option_A')['valuation_ASPC_ACPC']
diff_proba_no_tradeoff_censored = no_tradeoff_lottery_differences_censored.groupby('prob_option_A')['valuation_ACPC_ASPS']

plt.errorbar(diff_proba_no_tradeoff_censored.mean().index - offset_2/2, diff_proba_no_tradeoff_censored.mean(), diff_proba_no_tradeoff_censored.std(), ecolor = 'black', fmt='none', alpha=0.4, label='std')
plt.plot(diff_proba_no_tradeoff_censored.mean().index - offset_2/2, diff_proba_no_tradeoff_censored.mean(), label='$Y^{C}(P^{S})-Y^{S}(P^{S})$', color='bisque', marker='o', linestyle='-')

plt.errorbar(diff_proba_self_censored.mean().index, diff_proba_self_censored.mean(), diff_proba_self_censored.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_self_censored.mean().index, diff_proba_self_censored.mean(), label='$Y^{C}(P^{S})-Y^{S}(P^{S})$', color='dodgerblue', marker='o', linestyle='-')

plt.errorbar(diff_proba_charity_censored.mean().index + offset_2/2, diff_proba_charity_censored.mean(), diff_proba_charity_censored.std(), ecolor = 'black', fmt='none', alpha=0.4)
plt.plot(diff_proba_charity_censored.mean().index + offset_2/2, diff_proba_charity_censored.mean(), label='$Y^{S}(P^{C})-Y^{C}(P^{C})$', color='limegreen', marker='o', linestyle='-')

x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Valuation as % of Riskless Lottery')
plt.title('Lottery Valuation difference for all probabilities for Censored subjects')
plt.legend()
plt.savefig('All Lottery difference Censored H1.png', dpi=1200)
plt.show()

charity_lottery_differences_negated_censored = charity_lottery_differences_censored.copy()  
charity_lottery_differences_negated_censored['valuation_ASPC_ACPC'] *= -1

t_statistic_diff_censored, p_value_diff_censored = ttest_ind(self_lottery_differences_censored['valuation_ACPS_ASPS'], charity_lottery_differences_negated_censored['valuation_ASPC_ACPC'])
print()
print('Difference of magnitude between self and charity valuation difference for censored:')
print(t_statistic_diff_censored, p_value_diff_censored)

t_statistic_self, p_value_self = ttest_ind(self_lottery_differences['valuation_ACPS_ASPS'], self_lottery_differences_censored['valuation_ACPS_ASPS'])
print('t-test and p-value of Self difference between All vs censored')
print(t_statistic_self, p_value_self)
print()

t_statistic_charity, p_value_charity = ttest_ind(charity_lottery_differences['valuation_ASPC_ACPC'], charity_lottery_differences_censored['valuation_ASPC_ACPC'])
print('t-test and p-value of Charity difference between All vs censored')
print(t_statistic_charity, p_value_charity)
print()


# %%
# =============================================================================
# Participant-specific X values
# =============================================================================


########### Plot ditribution of participant-specific X values 

plt.hist(data_autre['charity_calibration'], bins=20) 
plt.xlabel('Participant-specific X')
plt.ylabel('Frequency')
plt.title('Distribution of X with censored included')
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

plt.hist(X_no_EDRP_total['charity_calibration'], bins=20, color = 'lightcoral') 
plt.xlabel('X values')
plt.ylabel('Frequency')
plt.title('Distribution of X values of No EDRP subjects')
plt.axvline(x=X_no_EDRP_total['charity_calibration'].mean(), color='grey', linestyle='--', label = 'Mean = '+ str(np.round(X_no_EDRP_total['charity_calibration'].mean(), 1)))
plt.axvline(x=X_no_EDRP_total['charity_calibration'].median(), color='gainsboro', linestyle='--', label = 'Median = '+ str(np.round(X_no_EDRP_total['charity_calibration'].median(), 1)))
plt.legend()
plt.savefig('X values for No EDRP.png', dpi=1200)
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


# Altruistic vs everyone
t_statistic_6, p_value_6 = ttest_ind(X_altruistic['charity_calibration'], data_autre_principal['charity_calibration'])
print('t-test and p-value of X diff between Altruistic vs everyone')
print(t_statistic_6, p_value_6)
print()

# %%
# =============================================================================
# ANALYSE DATA 
# =============================================================================

######## EXLEY REGRESSION 

data_for_analysis = pd.concat([ASPS, ACPC, ASPC, ACPS], ignore_index=True)
data_for_analysis_EDRP = pd.concat([ASPS_EDRP, ACPC_EDRP, ASPC_EDRP, ACPS_EDRP], ignore_index=True)
data_for_analysis_censored = pd.concat([ASPS_censored, ACPC_censored, ASPC_censored, ACPS_censored], ignore_index=True)


# Add fixed effects
dummy_ind = pd.get_dummies(data_for_analysis['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob = pd.get_dummies(data_for_analysis['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
data_for_analysis = pd.concat([data_for_analysis, dummy_ind, dummy_prob], axis=1)

# Add controls 
data_for_analysis = data_for_analysis.merge(survey, on='id', how='left')
control_variables = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + ['NEP_' + str(i) for i in range(1, 16)] + 
                 ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]

# Create the design matrix and dependent variable
# X = data_for_analysis[['charity', 'tradeoff', 'interaction'] + list(dummy_ind.columns) + list(dummy_prob.columns)]
X = data_for_analysis[['charity', 'tradeoff', 'interaction', 'case_order'] + list(dummy_ind.columns) + list(dummy_prob.columns)]
X = pd.concat([X, data_for_analysis[control_variables]], axis=1)
X = sm.add_constant(X, has_constant='add') # add a first column full of ones to account for intercept of regression
y = data_for_analysis['valuation']

# Fit the regression model using Ordinary Least Squares
model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis['number']}) # cluster at individual level
print(model.summary())


md = smf.mixedlm("valuation ~ charity + tradeoff + interaction", data_for_analysis, groups=data_for_analysis["number"])
mdf = md.fit()
print(mdf.summary())

# For EDRP 
dummy_ind_EDRP = pd.get_dummies(data_for_analysis_EDRP['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob_EDRP = pd.get_dummies(data_for_analysis_EDRP['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
data_for_analysis_EDRP = pd.concat([data_for_analysis_EDRP, dummy_ind_EDRP, dummy_prob_EDRP], axis=1)

# Add controls 
data_for_analysis_EDRP = data_for_analysis_EDRP.merge(survey, on='id', how='left')
control_variables_EDRP = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + ['NEP_' + str(i) for i in range(1, 16)] + 
                 ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]

# Create the design matrix and dependent variable
# X_EDRP = data_for_analysis_EDRP[['charity', 'tradeoff', 'interaction'] + list(dummy_ind.columns) + list(dummy_prob.columns)]
X_EDRP = data_for_analysis_EDRP[['charity', 'tradeoff', 'interaction', 'case_order'] + list(dummy_ind_EDRP.columns) + list(dummy_prob_EDRP.columns)]
X_EDRP = pd.concat([X_EDRP, data_for_analysis_EDRP[control_variables_EDRP]], axis=1)
X_EDRP = sm.add_constant(X_EDRP, has_constant='add') # add a first column full of ones to account for intercept of regression
y_EDRP = data_for_analysis_EDRP['valuation']

# Fit the regression model using Ordinary Least Squares
model_EDRP = sm.OLS(y_EDRP, X_EDRP).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis_EDRP['number']}) # cluster at individual level
print(model_EDRP.summary())



# For censored participants

dummy_ind_censored = pd.get_dummies(data_for_analysis_censored['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob_censored = pd.get_dummies(data_for_analysis_censored['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
data_for_analysis_censored = pd.concat([data_for_analysis_censored, dummy_ind_censored, dummy_prob_censored], axis=1)

# Add controls 
data_for_analysis_censored = data_for_analysis_censored.merge(survey, on='id', how='left')
control_variables_censored = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + ['NEP_' + str(i) for i in range(1, 16)] + 
                 ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]

# Create the design matrix and dependent variable
# X_censored = data_for_analysis_censored[['charity', 'tradeoff', 'interaction'] + list(dummy_ind_censored.columns) + list(dummy_prob_censored.columns)]
X_censored = data_for_analysis_censored[['charity', 'tradeoff', 'interaction', 'case_order'] + list(dummy_ind_censored.columns) + list(dummy_prob_censored.columns)]
X_censored = pd.concat([X_censored, data_for_analysis_censored[control_variables_censored]], axis=1)
X_censored = sm.add_constant(X_censored, has_constant='add') # add a first column full of ones to account for intercept of regression
y_censored = data_for_analysis_censored['valuation']

# Fit the regression model using Ordinary Least Squares
model_censored = sm.OLS(y_censored, X_censored).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis_censored['number']}) # cluster at individual level
print(model_censored.summary())



md_c = smf.mixedlm("valuation ~ charity + tradeoff + interaction", data_for_analysis_censored, groups=data_for_analysis_censored["number"])
mdf_c = md_c.fit()
print(mdf_c.summary())

# %%
# =============================================================================
# Look at differences for each probability
# =============================================================================
# FOR ALL 

# for no tradeoff
dummy_ind_proba_no_tradeoff = pd.get_dummies(no_tradeoff_lottery_differences['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob_proba_no_tradeoff = pd.get_dummies(no_tradeoff_lottery_differences['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
no_tradeoff_diff_reg = pd.concat([no_tradeoff_lottery_differences, dummy_ind_proba_no_tradeoff, dummy_prob_proba_no_tradeoff], axis=1)

# Add controls 
# no_tradeoff_diff_reg = no_tradeoff_diff_reg.merge(survey, on='number', how='left')
# control_variables = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + ['NEP_' + str(i) for i in range(1, 16)] + 
#                  ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]

# Create the design matrix and dependent variable
X_proba_no_tradeoff = no_tradeoff_diff_reg[list(dummy_ind_proba_no_tradeoff.columns) + list(dummy_prob_proba_no_tradeoff.columns)]
# X_proba_no_tradeoff = pd.concat([X_proba, no_tradeoff_lottery_differences[control_variables]], axis=1)
X_proba_no_tradeoff = sm.add_constant(X_proba_no_tradeoff, has_constant='add') # add a first column full of ones to account for intercept of regression
y_proba_no_tradeoff = no_tradeoff_diff_reg['valuation_ACPC_ASPS']

# Fit the regression model using Ordinary Least Squares
model_proba_no_tradeoff = sm.OLS(y_proba_no_tradeoff, X_proba_no_tradeoff).fit(cov_type='cluster', cov_kwds={'groups': no_tradeoff_diff_reg['number']}) # cluster at individual level
print(model_proba_no_tradeoff.summary())



# for self diff
dummy_ind_proba_self = pd.get_dummies(self_lottery_differences['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob_proba_self = pd.get_dummies(self_lottery_differences['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
self_diff_reg = pd.concat([self_lottery_differences, dummy_ind_proba_self, dummy_prob_proba_self], axis=1)

# Add controls 
# no_tradeoff_diff_reg = no_tradeoff_diff_reg.merge(survey, on='number', how='left')
# control_variables = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + ['NEP_' + str(i) for i in range(1, 16)] + 
#                  ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]

# Create the design matrix and dependent variable
X_proba_self = self_diff_reg[list(dummy_ind_proba_self.columns) + list(dummy_prob_proba_self.columns)]
X_proba_self = sm.add_constant(X_proba_self, has_constant='add') # add a first column full of ones to account for intercept of regression
y_proba_self = self_diff_reg['valuation_ACPS_ASPS']

# Fit the regression model using Ordinary Least Squares
model_proba_self = sm.OLS(y_proba_self, X_proba_self).fit(cov_type='cluster', cov_kwds={'groups': self_diff_reg['number']}) # cluster at individual level
print(model_proba_self.summary())



# for charity diff
dummy_ind_proba_charity = pd.get_dummies(charity_lottery_differences['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob_proba_charity = pd.get_dummies(charity_lottery_differences['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
charity_diff_reg = pd.concat([charity_lottery_differences, dummy_ind_proba_charity, dummy_prob_proba_charity], axis=1)

# Add controls 
# no_tradeoff_diff_reg = no_tradeoff_diff_reg.merge(survey, on='number', how='left')
# control_variables = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + ['NEP_' + str(i) for i in range(1, 16)] + 
#                  ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]

# Create the design matrix and dependent variable
X_proba_charity = charity_diff_reg[list(dummy_ind_proba_charity.columns) + list(dummy_prob_proba_charity.columns)]
X_proba_charity = sm.add_constant(X_proba_charity, has_constant='add') # add a first column full of ones to account for intercept of regression
y_proba_charity = charity_diff_reg['valuation_ASPC_ACPC']

# Fit the regression model using Ordinary Least Squares
model_proba_charity = sm.OLS(y_proba_charity, X_proba_charity).fit(cov_type='cluster', cov_kwds={'groups': charity_diff_reg['number']}) # cluster at individual level
print(model_proba_charity.summary())

###################
# FOR CENSORED


# for no tradeoff
dummy_ind_proba_no_tradeoff_censored = pd.get_dummies(no_tradeoff_lottery_differences_censored['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob_proba_no_tradeoff_censored = pd.get_dummies(no_tradeoff_lottery_differences_censored['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
no_tradeoff_diff_reg_censored = pd.concat([no_tradeoff_lottery_differences_censored, dummy_ind_proba_no_tradeoff_censored, dummy_prob_proba_no_tradeoff_censored], axis=1)

# Add controls 
# no_tradeoff_diff_reg = no_tradeoff_diff_reg.merge(survey, on='number', how='left')
# control_variables = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + ['NEP_' + str(i) for i in range(1, 16)] + 
#                  ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]

# Create the design matrix and dependent variable
X_proba_no_tradeoff_censored = no_tradeoff_diff_reg_censored[list(dummy_ind_proba_no_tradeoff_censored.columns) + list(dummy_prob_proba_no_tradeoff_censored.columns)]
X_proba_no_tradeoff_censored = sm.add_constant(X_proba_no_tradeoff_censored, has_constant='add') # add a first column full of ones to account for intercept of regression
y_proba_no_tradeoff_censored = no_tradeoff_diff_reg_censored['valuation_ACPC_ASPS']

# Fit the regression model using Ordinary Least Squares
model_proba_no_tradeoff_censored = sm.OLS(y_proba_no_tradeoff_censored, X_proba_no_tradeoff_censored).fit(cov_type='cluster', cov_kwds={'groups': no_tradeoff_diff_reg_censored['number']}) # cluster at individual level
print(model_proba_no_tradeoff_censored.summary())


# for self diff
dummy_ind_proba_self_censored = pd.get_dummies(self_lottery_differences_censored['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob_proba_self_censored = pd.get_dummies(self_lottery_differences_censored['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
self_diff_reg_censored = pd.concat([self_lottery_differences_censored, dummy_ind_proba_self_censored, dummy_prob_proba_self_censored], axis=1)

# Add controls 
# no_tradeoff_diff_reg = no_tradeoff_diff_reg.merge(survey, on='number', how='left')
# control_variables = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + ['NEP_' + str(i) for i in range(1, 16)] + 
#                  ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]

# Create the design matrix and dependent variable
X_proba_self_censored = self_diff_reg_censored[list(dummy_ind_proba_self_censored.columns) + list(dummy_prob_proba_self_censored.columns)]
X_proba_self_censored = sm.add_constant(X_proba_self_censored, has_constant='add') # add a first column full of ones to account for intercept of regression
y_proba_self_censored = self_diff_reg_censored['valuation_ACPS_ASPS']

# Fit the regression model using Ordinary Least Squares
model_proba_self_censored = sm.OLS(y_proba_self_censored, X_proba_self_censored).fit(cov_type='cluster', cov_kwds={'groups': self_diff_reg_censored['number']}) # cluster at individual level
print(model_proba_self_censored.summary())



# for charity diff
dummy_ind_proba_charity_censored = pd.get_dummies(charity_lottery_differences_censored['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
dummy_prob_proba_charity_censored = pd.get_dummies(charity_lottery_differences_censored['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
charity_diff_reg_censored = pd.concat([charity_lottery_differences_censored, dummy_ind_proba_charity_censored, dummy_prob_proba_charity_censored], axis=1)

# Add controls 
# no_tradeoff_diff_reg = no_tradeoff_diff_reg.merge(survey, on='number', how='left')
# control_variables = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + ['NEP_' + str(i) for i in range(1, 16)] + 
#                  ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]

# Create the design matrix and dependent variable
X_proba_charity_censored = charity_diff_reg_censored[list(dummy_ind_proba_charity_censored.columns) + list(dummy_prob_proba_charity_censored.columns)]
X_proba_charity_censored = sm.add_constant(X_proba_charity_censored, has_constant='add') # add a first column full of ones to account for intercept of regression
y_proba_charity_censored = charity_diff_reg_censored['valuation_ASPC_ACPC']

# Fit the regression model using Ordinary Least Squares
model_proba_charity_censored = sm.OLS(y_proba_charity_censored, X_proba_charity_censored).fit(cov_type='cluster', cov_kwds={'groups': charity_diff_reg_censored['number']}) # cluster at individual level
print(model_proba_charity_censored.summary())

# %%
# =============================================================================
# calibration bias
# =============================================================================

data_for_analysis = pd.concat([ASPS, ACPC, ASPC, ACPS], ignore_index=True)

# # Add fixed effects
# dummy_ind = pd.get_dummies(data_for_analysis['number'], drop_first=True, dtype=int)  # Dummy variable for individuals (+drop first to avoid multicollinearity)
# dummy_prob = pd.get_dummies(data_for_analysis['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
# data_for_analysis = pd.concat([data_for_analysis, dummy_ind, dummy_prob], axis=1)

# # Add controls 
# data_for_analysis = data_for_analysis.merge(survey, on='id', how='left')
# control_variables = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + ['NEP_' + str(i) for i in range(1, 16)] + 
#                  ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]

# # Create the design matrix and dependent variable
# # X = data_for_analysis[['charity', 'tradeoff', 'interaction'] + list(dummy_ind.columns) + list(dummy_prob.columns)]
# X = data_for_analysis[['charity', 'tradeoff', 'interaction', 'case_order'] + list(dummy_ind.columns) + list(dummy_prob.columns)]
# X = pd.concat([X, data_for_analysis[control_variables]], axis=1)
# X = sm.add_constant(X, has_constant='add') # add a first column full of ones to account for intercept of regression
# y = data_for_analysis['valuation']

# # Fit the regression model using Ordinary Least Squares
# model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis['number']}) # cluster at individual level
# print(model.summary())


iteration_number = 1000
sample = 107
p_values = np.zeros(iteration_number)

for inter in range(1, iteration_number):
    subjects_drawn = np.random.choice(range(1,data_for_analysis['number'].nunique()+1), sample)
    data_drawn = []
    for subj in subjects_drawn:
        subj_data = data_for_analysis.loc[data_for_analysis['number'] == subj, ['number', 'prob_option_A', 'valuation', 'charity', 'tradeoff', 'interaction']]
        data_drawn.append(subj_data)
    data_drawn = pd.concat(data_drawn)
    
    try:
        
        # dummy_ind = pd.get_dummies(data_drawn['number'], drop_first=True, dtype=int) 
        # dummy_prob = pd.get_dummies(data_drawn['prob_option_A'], drop_first=True, dtype=int) 
        # data_for_analysis = pd.concat([data_drawn, dummy_ind, dummy_prob], axis=1)
        # X = data_for_analysis[['charity', 'tradeoff', 'interaction'] + list(dummy_ind.columns) + list(dummy_prob.columns)]
        # X = sm.add_constant(X, has_constant='add') 
        # y = data_for_analysis['valuation']
        # model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis['number']}) # cluster at individual level
        # summary = model.summary()
        
        test = smf.mixedlm("valuation ~ charity + tradeoff + interaction", data_drawn, groups=data_drawn["number"])
        test = test.fit()
        summary = test.summary()
        coef_charity = summary.tables[1]['P>|z|']['charity']
        
        # coef_tradeoff = summary.tables[1].data[3][4]
        # coef_interaction = summary.tables[1].data[4][4]
        # p_values[inter] = [float(coef_tradeoff), float(coef_interaction)]
       
        coef_charity = ast.literal_eval(coef_charity)
        p_values[inter] = coef_charity
    except np.linalg.LinAlgError:
        print()
        print("Singular matrix encountered.")
        print()
        p_values[inter] = [np.nan,np.nan]
    except ZeroDivisionError:
        print()
        print("Multicollinearity encountered.")
        print()
        p_values[inter] = [np.nan,np.nan]    
        
power_calculated = np.mean(p_values < 0.05)


