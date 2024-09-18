#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:38:48 2024

@author: carolinepioger
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import ast 
import matplotlib.pyplot as plt
import time
import statsmodels.api as sm
from tqdm import tqdm
import warnings

censure = 1
outliers = 1

path = '/Users/carolinepioger/Desktop/pretest vincent' # change to yours :)

data = pd.read_csv(path + '/dataset.csv' )
data_exley = pd.read_stata('/Users/carolinepioger/Downloads/Supplementary/Data/valuation_level.dta')

# Remove (or not) participants with censored values in part 2
exclude_participants = data.loc[data['censored_calibration'] == 1, 'id'] 

if censure == 1: 
    data = data.drop(data[data['id'].isin(exclude_participants) == True].index)
    data['number'] = data.groupby('id').ngroup() + 1
else: 
    data = data

# Remove outliers? 

dwell_mean = data['dwell_time'].mean()
dwell_std = np.std(data['dwell_time'])

outliers_data = data[(data['dwell_time'] < dwell_mean - 4 * dwell_std)
                         | (data['dwell_time'] > dwell_mean + 4 * dwell_std)]
if outliers ==1:
    data = data.drop(outliers_data.index)
    data = data.reset_index(drop=True)
else: 
    pass 

# Convert dummies in integer and not string 
# for i in range(len(data)):
#     data['charity'][i] = ast.literal_eval(data['charity'][i])
#     data['tradeoff'][i] = ast.literal_eval(data['tradeoff'][i])
#     data['interaction'][i] = ast.literal_eval(data['interaction'][i])


pilot_sample = np.array(range(1, data['number'].nunique()+1))
sample_size = np.array(range(2,30))
power_needed = 0.8
alpha = 0.05
power_calculated = np.zeros((len(sample_size),2))

iteration_number = 100
loop = 0

# H1 1h20 for sample size 100 and 200 iterations 
# H2 3h for sample size 100 and 200 iterations 
# H1 9,6h for sample size 175 and 500 iterations
# H1 10h for sample size 200 and 700 iterations (without iteration prints)
# need for around 150 people fir 90 power and 140 for 80 power ? 

# %%
# =============================================================================
# POWER ANALYSIS FOR H1
# =============================================================================

start_time = time.time()

for sample in tqdm(sample_size, desc="Sample Size"): 
    p_values = np.zeros((iteration_number, 2))
    for inter in range(1, iteration_number):
        subjects_drawn = np.random.choice(range(1,data['number'].nunique()+1), sample)
        data_drawn = []
        for subj in subjects_drawn:
            subj_data = data.loc[data['number'] == subj, ['number', 'prob_option_A', 'valuation', 'charity', 'tradeoff', 'interaction']]
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
            coef_tradeoff, coef_interaction = summary.tables[1]['P>|z|'][['tradeoff', 'interaction']]
            
            # coef_tradeoff = summary.tables[1].data[3][4]
            # coef_interaction = summary.tables[1].data[4][4]
            # p_values[inter] = [float(coef_tradeoff), float(coef_interaction)]
           
            coef_tradeoff = ast.literal_eval(coef_tradeoff)
            coef_interaction = ast.literal_eval(coef_interaction)
            p_values[inter] = [coef_tradeoff,coef_interaction]
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
            
        # print()
        # print()
        # print()
        # print("Iteration " + str(inter) + " DONE")
        # print()
        # print()
        # print()
    
    power_calculated[loop, 0] = np.mean(p_values[:,0] < alpha)
    power_calculated[loop, 1] = np.mean(p_values[:,1] < alpha)
    
    loop += 1
    
    print()
    print()
    print()
    print("Sample " + str(sample) + " DONE")
    print()
    print()
    print()


end_time = time.time()
duration = end_time - start_time

print("It took " + str(duration/60) + " minutes")

plt.figure()
plt.plot(sample_size, power_calculated, label=['Tradeoff', 'Interaction'])
plt.axhline(y=power_needed, color='r', linestyle='--') 
plt.xlabel('Sample Size')
plt.ylabel('Power')
plt.title('H1 Power analysis')
plt.legend()
plt.savefig('H1 Power analysis.png', dpi=1200)
plt.show()

data_H1 = path + '/power caculated H1.csv'
power_cal = pd.DataFrame(power_calculated)
power_cal.to_csv(data_H1, index=False)


# %%
# =============================================================================
# POWER ANALYSIS FOR H2
# =============================================================================

start_time = time.time()

for sample in sample_size: 
    p_values = np.zeros((iteration_number, 2))
    for inter in range(1, iteration_number):
        subjects_drawn = np.random.choice(range(1,data['number'].nunique()+1), sample)
        data_drawn = []
        for subj in subjects_drawn:
            subj_data = data.loc[data['number'] == subj, ['number', 'prob_option_A', 'dwell_time', 'charity', 'tradeoff', 'interaction']]
            data_drawn.append(subj_data)
        data_drawn = pd.concat(data_drawn)
        
        try:
            test = smf.mixedlm("dwell_time ~ charity + tradeoff + interaction", data_drawn, groups=data_drawn["number"])
            test = test.fit()
            summary = test.summary()
            coef_tradeoff, coef_interaction = summary.tables[1]['P>|z|'][['tradeoff', 'interaction']]
            coef_tradeoff = ast.literal_eval(coef_tradeoff)
            coef_interaction = ast.literal_eval(coef_interaction)
            p_values[inter] = [coef_tradeoff,coef_interaction]
        except np.linalg.LinAlgError:
            print()
            print("Singular matrix encountered.")
            print()
            p_values[inter] = [1,1]
            
        # print()
        # print()
        # print()
        # print("Iteration " + str(inter) + " DONE")
        # print()
        # print()
        # print()
    
    power_calculated[loop, 0] = np.mean(p_values[:,0] < alpha)
    power_calculated[loop, 1] = np.mean(p_values[:,1] < alpha)
    
    loop += 1
    
    print()
    print()
    print()
    print("Sample " + str(sample) + " DONE")
    print()
    print()
    print()


### ADD CONTROLS ? 

end_time = time.time()
duration = end_time - start_time

print("It took " + str(duration/60) + " minutes")

plt.figure()
plt.plot(sample_size, power_calculated, label=['Tradeoff', 'Interaction'])
plt.axhline(y=power_needed, color='r', linestyle='--') 
plt.xlabel('Sample Size')
plt.ylabel('Power')
plt.title('H2 Power analysis')
plt.legend()
plt.savefig('H2 Power analysis.png', dpi=1200)
plt.show()



# %%
# =============================================================================
# POWER ANALYSIS FOR H3
# =============================================================================



# for sample in sample_size: 
#     p_values = np.zeros((iteration_number, 2))
#     for inter in range(1, iteration_number):
#         subjects_drawn = np.random.choice(range(1,data['number'].nunique()+1), sample)
#         data_drawn = []
#         for subj in subjects_drawn:
#             subj_data = data.loc[data['number'] == subj, ['number', 'prob_option_A', 'valuation', 'dwell_time']]
#             data_drawn.append(subj_data)
#         data_drawn = pd.concat(data_drawn)
        
#         try:
#             test = smf.mixedlm("dwell_time ~ charity + tradeoff + interaction", data_drawn, groups=data_drawn["number"])
#             test = test.fit()
#             summary = test.summary()
#             coef_tradeoff, coef_interaction = summary.tables[1]['P>|z|'][['tradeoff', 'interaction']]
#             coef_tradeoff = ast.literal_eval(coef_tradeoff)
#             coef_interaction = ast.literal_eval(coef_interaction)
#             p_values[inter] = [coef_tradeoff,coef_interaction]
#         except np.linalg.LinAlgError:
#             print()
#             print("Singular matrix encountered.")
#             print()
#             p_values[inter] = [1,1]
            
#         print()
#         print()
#         print()
#         print("Iteration " + str(inter) + " DONE")
#         print()
#         print()
#         print()
    
#     power_calculated[loop, 0] = np.mean(p_values[:,0] < alpha)
#     power_calculated[loop, 1] = np.mean(p_values[:,1] < alpha)
    
#     loop += 1
    
#     print()
#     print()
#     print()
#     print("Sample " + str(sample) + " DONE")
#     print()
#     print()
#     print()



# plt.figure()
# plt.plot(sample_size, power_calculated, label=['Tradeoff', 'Interaction'])
# plt.axhline(y=power_needed, color='r', linestyle='--') 
# plt.xlabel('Sample Size')
# plt.ylabel('Power')
# plt.title('H3 Power analysis')
# plt.legend()
# plt.savefig('H3 Power analysis.png', dpi=1200)
# plt.show()
