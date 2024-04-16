#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:38:48 2024

@author: carolinepioger
"""

import pandas as pd
import numpy as np

path = '/Users/carolinepioger/Desktop/pretest vincent' # change to yours :)

data = pd.read_csv(path + '/dataset.csv' )

pilot_sample = range(1, data['number'].nunique()+1) 
sample_size = range(2,10)
power = 0.8
alpha = 0.05

iteration_number = 100

for sample in sample_size: 
    p_values = np.zeros(iteration_number)
    for inter in range(1, iteration_number):
        # params_test = np.zeros((sample, 2))
        subjects_drawn = np.random.choice(range(1,data['number'].nunique()+1), sample)
        data_drawn = data.loc[data['number'] == i, ['prob_option_A', 'valuation'] for i in subjects_drawn] 
        