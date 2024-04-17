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


path = '/Users/carolinepioger/Desktop/pretest vincent' # change to yours :)

data = pd.read_csv(path + '/dataset.csv' )

# Convert dummies in integer and not string 
# for i in range(len(data)):
#     data['charity'][i] = ast.literal_eval(data['charity'][i])
#     data['tradeoff'][i] = ast.literal_eval(data['tradeoff'][i])
#     data['interaction'][i] = ast.literal_eval(data['interaction'][i])


pilot_sample = range(1, data['number'].nunique()+1) 
sample_size = range(2,100)
power_needed = 0.9
alpha = 0.05
power_calculated = np.zeros((len(sample_size),2))

iteration_number = 500
loop = 0

for sample in sample_size: 
    p_values = np.zeros((iteration_number, 2))
    for inter in range(1, iteration_number):
        subjects_drawn = np.random.choice(range(1,data['number'].nunique()+1), sample)
        data_drawn = []
        for subj in subjects_drawn:
            subj_data = data.loc[data['number'] == subj, ['number', 'prob_option_A', 'valuation', 'charity', 'tradeoff', 'interaction']]
            data_drawn.append(subj_data)
        data_drawn = pd.concat(data_drawn)
        
        try:
            test = smf.mixedlm("valuation ~ charity + tradeoff + interaction", data_drawn, groups=data_drawn["number"])
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
            
        print()
        print()
        print()
        print("Iteration " + str(inter) + " DONE")
        print()
        print()
        print()
    
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



plt.figure()
plt.plot(sample_size, power_calculated, label=['Tradeoff', 'Interaction'])
plt.axhline(y=power_needed, color='r', linestyle='--') 
plt.xlabel('Sample Size')
plt.ylabel('Power')
plt.title('Power vs. Sample Size')
plt.legend()
plt.show()