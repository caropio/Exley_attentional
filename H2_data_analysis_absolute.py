#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:18:21 2024

@author: carolinepioger
"""

# This script is the same as the H2_data_analysis.py one but replacing all the 
# relative dwell times (in %) by absolute ones (in s)

path = '/Users/carolinepioger/Desktop/EXLEY ATT' # change to yours

H2_relative_time = path + '/Exley_attentional/H2_data_analysis.py'

with open(H2_relative_time, 'r') as file:
    script_content = file.read()

# Make changes to the original script accordingly
script_content.replace('dwell_time_relative', 'dwell_time_absolute') # replace all the relative dwell times by absolute ones
script_content.replace('H2.csv', 'ABSOLUTE H2.csv') # replace the end of the csv files to specify the absolute context
script_content.replace('H2.png', 'ABSOLUTE H2.png') # replace the end of the png files to specify the absolute context
script_content.replace('in %', 'in s') # replace the unit in seconds 

H2_absolute_time = script_content

exec(H2_absolute_time)
