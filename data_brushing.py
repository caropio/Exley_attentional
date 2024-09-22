#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:38:56 2024

@author: carolinepioger
"""
 
 
import pandas as pd
import numpy as np
import ast 
from scipy.stats import ttest_ind

# =============================================================================
# UPLOADING RAW DATA
# =============================================================================

# Paths information to upload data
path = '/Users/carolinepioger/Desktop/EXLEY ATT/ALL collection' # change to yours :)
# path = '/Users/carolinepioger/Desktop/test' 
# dates = ['2024-03-28', '2024-03-25', '2024-04-03'] # dates of pilot data collection (to remove)
dates = ['2024-04-29','2024-04-30','2024-05-02', 
          '2024-05-14', '2024-05-15', '2024-07-12', '2024-07-25'] # different dates of data collection 

################################################
# Upload data that will be used for analysis (data)
################################################

dfs = []

for date in dates:
    df1 = pd.read_csv(path + '/EXLEY_ASPS_' + date + '.csv') # data for the YSPS case
    df2 = pd.read_csv(path + '/EXLEY_ACPC_' + date + '.csv') # data for the YCPC case
    df3 = pd.read_csv(path + '/EXLEY_ASPC_' + date + '.csv') # data for the YSPC case
    df4 = pd.read_csv(path + '/EXLEY_ACPS_' + date + '.csv') # data for the YCPS case
    dfs.extend([df1, df2, df3, df4])

concatenated_df = pd.concat(dfs, ignore_index=True) # get pooled data for all cases
concatenated_df = concatenated_df.drop(concatenated_df[
    concatenated_df['participant._current_page_name'] != 'prolific'
    ].index)                        # remove data of participants that didn't finish experiment
data = concatenated_df.sort_values(by=['participant.code']) # order data by participant

# Note that in this database each row gives the data of one price list (so there 
# are multiple rows for each participant). The database is ordered by participant

# Transform the 21 columns of choices into one summing up the choices 
data['player.choix_overview'] = data[['player.choix' + str(i) for i in range(1, 22)] # rewrite the 21 choices for valuation price list as array
                                     ].values.tolist()
data['player.prob_lottery'] = data[['player.P1', 'player.P2'] # rewrite the number of green and purple balls in probability urn 
                                   ].values.tolist()          # (probability information) respectively as array  

# Keep columns of interest
columns_mapping = {
    'participant.code': 'id', # number id associated to each participant
    'session.code': 'session',  # code of session (deployment of experiment)
    'player.association_choice': 'charity_name', # name of charity chosen by participant in part 1
    'player.x1_norm_after_correction': 'charity_calibration', # participant specific X from part 2
    'player.TREATMENT': 'case', # the case (YSPS/YCPC/YSPC/YCPS) of the presented price list of part 3
    'subsession.round_number': 'round_order', # order of presentation of price list within the case (1 to 7)
    'player.x1_outcome_vec_order': 'option_A_vector', # non-zero amount of money involved in lottery (Option A) which is either 10 or X value
    'player.prob_lottery': 'prob_option_A', # number of green and purple balls of associated probability urn
    'player.WTP_VEC': 'option_B_vector', # array of the different values of option B across the rows 
    'player.choix_overview': 'choices', # 21 choices (As and Bs) of valuation price list made across the rows 
    'player.temps_decision': 'total_time_spent_s', # time took to complete the entire price list (from the page upload until they click on next button) in seconds
    'player.temps_regard_urne': 'watching_urn_ms', # array of different times spent revealing probability urn (array of times in ms) 
    'player.user_actions': 'temporal_information' # temporal information (when urn is unmasked and choices are made) - exploratory
}

data = data.rename(columns=columns_mapping)[list(columns_mapping.values())] # renaming columns of data
data = data.reset_index(drop=True) # reset index

################################################
# Upload data which will be used to check criterions (data_autre)
################################################

data_autre = pd.concat([pd.read_csv(path + '/EXLEY_ASSO_' + date + '.csv') for date in dates]
                       , ignore_index=True) # get all data of the first part of experiment and general information
data_autre = data_autre.drop(data_autre[
    data_autre['participant._current_page_name'] != 'prolific'
    ].index) # remove data of participants that didn't finish experiment

data_autre['choix_calibration'] = data_autre[['player.choice_x_' + str(i) for i in range(1, 17)]
                                             ].values.tolist() # rewrite the 16 choices for calibration price list as array
data_autre['choix_buffer'] = data_autre[['player.choice_y_' + str(i) for i in range(1, 17)]
                                        ].values.tolist() # rewrite the 16 choices for buffer price list as array
columns_mapping = { 'participant.code': 'id', # number id associated to each participant
                   'session.code': 'session', # code of session (deployment of experiment)
                   'player.PROLIFIC_ID': 'prolific', # Prolific ID of the participant
                   'player.association_choice': 'charity_name',  # name of charity chosen by participant in part 1
                   'player.x1_norm_after_correction': 'charity_calibration', # participant specific X from part 2 
                   'choix_calibration': 'calibration_choices', # array of As and Bs chosen in calibration price list
                   'choix_buffer': 'buffer_choices', # array of As and Bs chosen in buffer price list
                   'player.corrected': 'exclusion_B_to_A'} # 1 if participant starts calibration price list with Option B (exclusion criterion)
data_autre = data_autre.rename(columns=columns_mapping)[list(columns_mapping.values())] # renaming columns of data_autre
data_autre = data_autre.reset_index(drop=True) # reset index

################################################
# Upload data to get order of cases (app_page) 
################################################

app_page = pd.concat([pd.read_csv(path + '/StartApp_' + date + '.csv') for date in dates], 
                     ignore_index=True) # data of sequence of pages presented in experiment (since the order of cases in randomized)
app_page = app_page.drop(app_page[app_page['participant._current_page_name'] != 'prolific'
                                  ].index) # remove data of participants that didn't finish experiment

app_page = app_page[['participant.code','session.code', 'player.sequence_of_apps']]
app_page = app_page.rename(columns={'participant.code':'id', # number id associated to each participant
                                    'session.code': 'session', # code of session (deployment of experiment)
                                    'player.sequence_of_apps':'case_order' # array of ordered pages presented in experiment 
                                    }) # renaming columns of app_page
app_page = app_page.reset_index(drop=True) # reset index

for i in range(len(app_page)):
    app_page['case_order'][i] = ast.literal_eval(app_page['case_order'][i])
    app_page['case_order'][i] = app_page['case_order'][i][8:12] # only keep the ordered pages of case presentation (part 3)

app_page['first case'] = np.nan
app_page['second case'] = np.nan
app_page['third case'] = np.nan
app_page['fourth case'] = np.nan
app_page['order of cases'] = np.nan
case_number = ['first', 'second', 'third', 'fourth']

# Separate each case in its associated column of order
for i in range(len(app_page)):
    for j in range(0,4):
        app_page[str(case_number[j]) + ' case'][i] = app_page['case_order'][i][j][-4:] # get the case (YSPS/etc) for each column
    app_page['order of cases'][i] = [app_page['first case'][i], app_page['second case'][i], # get the 4 ordered cases allthougher as array
                                     app_page['third case'][i], app_page['fourth case'][i]]

################################################
# Upload data to get survey information (survey)
################################################

survey = pd.concat([pd.read_csv(path + '/EXLEY_DEMOG_' + date + '.csv') for date in dates], 
                   ignore_index=True) # data from the 3 surveys of part 4 
survey = survey.drop(survey[survey['participant._current_page_name'] != 'prolific'
                            ].index) # remove data of participants that didn't finish experiment

columns_mapping = {
    'participant.code': 'id', # number id associated to each participant
    'session.code': 'session', # code of session (deployment of experiment)
    'player.PROLIFIC_ID': 'prolific', # Prolific ID of the participant
    'player.AGE': 'Demog_AGE', # Age of participant (from socio-demographic survey)
    'player.SEXE': 'Demog_Sex', # Sex of participant (from socio-demographic survey)
    'player.DISCIPLINE': 'Demog_Field', # Field of study of participant (from socio-demographic survey)
    'player.NIVEAU_ETUDE': 'Demog_High_Ed_Lev' # Highest education level of participant (from socio-demographic survey)
}

for i in range(1, 16):
    columns_mapping[f'player.QUEST_{i}'] = f'NEP_{i}' # Answer of new environmental paradigm (NEP) question

for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']:  # Question regarding how much they 1) like, 2) trust, 3) are likely to donate 
                                                        # and 4) have already donated (charity attitude survey)
    columns_mapping[f'player.CHARITY_{j}'] = f'Charity_{j}' # Answer to the charity attidue survey (see above questions) 

survey = survey.rename(columns=columns_mapping)[
    ['id', 'session', 'prolific', 'Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + 
    [f'NEP_{i}' for i in range(1, 16)] + 
    [f'Charity_{j}' for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]
][list(columns_mapping.values())] # renaming columns of survey 
                          
survey = survey.reset_index(drop=True) # reset index


# %%
# =============================================================================
# FROM RAW DATA TO CLEAN DATA
# =============================================================================

# Reconstruct option_A_vector, prob_option_A, option_B_vector and watching_urn_ms without issues (string to float)

def proba_A(array_boules):
    return array_boules[0]/np.sum(array_boules) # takes two values and gives proportion of first value 

for i in range(len(data)):
    data.loc[i, 'prob_option_A'] = proba_A(data.loc[i, 'prob_option_A']) # so the column gives the proportion of green balls in urn and thus the probability
                                                                         # of the non-zero amount (instead of the number of green and purple balls)

# Rewrite all values of option_B_vector and option_A_vector according to case (directly in float)
for i in range(len(data)):
    if data['case'][i] == 'ASPS':
        data['option_B_vector'][i] = np.round(np.linspace(0,10,21), decimals=1)
        # data.loc[i, 'option_B_vector'] = np.round(np.linspace(0, 10, 21), decimals=1)
        data['option_A_vector'][i] = [10,0]
        # data.loc[i, 'option_A_vector'] = [10, 0]
    elif data['case'][i] == 'ASPC':
        data['option_B_vector'][i] = np.round(np.linspace(0,10,21), decimals=1)
        # data.loc[i, 'option_B_vector'] = np.round(np.linspace(0, 10, 21), decimals=1)
        data['option_A_vector'][i] = [data['charity_calibration'][i],0]
        # data.loc[i, 'option_A_vector'] = [data.loc[i, 'charity_calibration'], 0]
    elif data['case'][i] == 'ACPC':
        data['option_B_vector'][i] = np.round(np.linspace(0,data['charity_calibration'][i],21), decimals=1)
        # data.loc[i, 'option_B_vector'] = np.round(np.linspace(0, data.loc[i, 'charity_calibration'], 21), decimals=1)
        data['option_A_vector'][i] = [data['charity_calibration'][i],0]
        # data.loc[i, 'option_A_vector'] = [data.loc[i, 'charity_calibration'], 0]
    elif data['case'][i] == 'ACPS': 
        data['option_B_vector'][i] = np.round(np.linspace(0,data['charity_calibration'][i],21), decimals=1)
        # data.loc[i, 'option_B_vector'] = np.round(np.linspace(0, data.loc[i, 'charity_calibration'], 21), decimals=1)
        data['option_A_vector'][i] = [10,0]
        # data.loc[i, 'option_A_vector'] = [10, 0]

# Rewrite null values as zeros and convert to float for watching_urn_ms
data['watching_urn_ms'] = data['watching_urn_ms'].apply(lambda x: str(x).replace('null', '0'))

for i in range(len(data)):
    if isinstance(data['watching_urn_ms'][i], str):
        data['watching_urn_ms'][i] = ast.literal_eval(data['watching_urn_ms'][i])


# %%
# =============================================================================
# FROM CLEAN DATA TO COMPLETE DATA
# =============================================================================

# Using the existing database, we extract all the data needed for the analysis

################################################
# Add switchpoint values 
################################################

def upper_switchpoint(array):
    switch_indices = []
    for i in range(1,len(array)):
        if array[i] != array[i - 1]:
            switch_indices.append(i+1) 
    return switch_indices # gives the index of each switching point (first new option chosen)

# Get switching points from valuation price lists (part 3)
switchpoint_values = [upper_switchpoint(data['choices'][i]) for i in range(len(data))] # get switching points for each valuation price list 
data.insert(data.columns.get_loc('choices') + 1, 'switchpoint', switchpoint_values) # add column of switching points next to column of array of choices

# Get switching points from calibration and buffer price lists (part 2)
switchpoint_values_calib = [upper_switchpoint(data_autre['calibration_choices'][i]) for i in range(len(data_autre))] # get switching points for each calibration price list
switchpoint_values_buffer = [upper_switchpoint(data_autre['buffer_choices'][i]) for i in range(len(data_autre))] # get switching points for each buffer price list
data_autre.insert(data_autre.columns.get_loc('calibration_choices') + 1, 'switchpoint_calib', switchpoint_values_calib) # add column of switching points next to column of array of choices
data_autre.insert(data_autre.columns.get_loc('buffer_choices') + 1, 'switchpoint_buffer', switchpoint_values_buffer) # add column of switching points next to column of array of choices

data_autre['censored_calibration'] = np.nan
data_autre['censored_buffer'] = np.nan

# We bring forth cases in calibration and buffer price lists where participants are censored or have multiple switching points (MSP)
# For calibration price list
for i in range(len(data_autre)):
    if data_autre['switchpoint_calib'][i] == []:
        data_autre['censored_calibration'][i] = 1   
    elif len(data_autre['switchpoint_calib'][i]) == 1:
        data_autre['censored_calibration'][i] = 0  
    elif len(data_autre['switchpoint_calib'][i]) > 1:
        data_autre['censored_calibration'][i] = 'MSP'  

# note that participants called "censored" are those who don't chose option B 
# in the calibration price list (regardless of the buffer one) thus individuals
# with data_autre['censored_calibration'] = 1 

# For buffer price list
for i in range(len(data_autre)):
    if data_autre['switchpoint_buffer'][i] == []:
        data_autre['censored_buffer'][i] = 1   
    elif len(data_autre['switchpoint_buffer'][i]) == 1:
        data_autre['censored_buffer'][i] = 0  
    elif len(data_autre['switchpoint_buffer'][i]) > 1:
        data_autre['censored_buffer'][i] = 'MSP'   

# Find participants where the switchpoint of buffer price list is strictly superior to calibration price list
def calib_buffer_choice(calib, buffer):
    if calib[0] < buffer[0]:
        return 1 # gives 1 if buffer switchpoint strictly superior to calibration's
    else: 
        return 0 # gives 0 otherwise

# We temporarily change the empty array of switching points to 30 so that the values are numerically comparable    
for i in range(len(data_autre)):
    if data_autre['switchpoint_calib'][i] == []:
        data_autre['switchpoint_calib'][i] = [30] 
for i in range(len(data_autre)):
    if data_autre['switchpoint_buffer'][i] == []:
        data_autre['switchpoint_buffer'][i] = [30] 
        
data_autre['buffer>calib']= [calib_buffer_choice(data_autre['switchpoint_calib'][i],data_autre['switchpoint_buffer'][i]) for i in range(len(data_autre))]

# Find the switching point value of the buffer price list (like the X value of calibration but of buffer)
options_buffer= np.round(np.linspace(0,30,16), decimals=1)
data_autre['buffer_X'] = np.nan
for i in range(len(data_autre)):
    if data_autre['switchpoint_buffer'][i] == [30]:
        data_autre['buffer_X'][i] = 30 # gives 30 if censored in buffer 
    else: 
        data_autre['buffer_X'][i] = options_buffer[data_autre['switchpoint_buffer'][i][0]-1] # gives the participant-specific X for the buffer price list 

# Insert censored_calibration, censored_buffer and buffer_X in database data_autre
column_order_2 = list(data_autre.columns)
column_order_2.insert(column_order_2.index('switchpoint_calib') + 1, column_order_2.pop(column_order_2.index('censored_calibration')))
column_order_2.insert(column_order_2.index('switchpoint_buffer') + 1, column_order_2.pop(column_order_2.index('censored_buffer')))
column_order_2.insert(column_order_2.index('charity_calibration') + 1, column_order_2.pop(column_order_2.index('buffer_X')))
data_autre = data_autre[column_order_2]

# Insert ensored_calibration and buffer_X in database data (for each row of participant)
data = data.merge(data_autre[['id','buffer_X']], how='left', on='id')
data = data.merge(data_autre[['id','censored_calibration']], how='left', on='id')
column_order_3 = list(data.columns)
column_order_3.insert(column_order_3.index('charity_calibration') + 1, column_order_3.pop(column_order_3.index('buffer_X')))
column_order_3.insert(column_order_3.index('buffer_X') + 1, column_order_3.pop(column_order_3.index('censored_calibration')))
data = data[column_order_3]

# Change back to empty arrays for arrays of censored calibration and buffer 
for i in range(len(data_autre)):
    if data_autre['switchpoint_calib'][i] == [30]:
        data_autre['switchpoint_calib'][i] = [] 

for i in range(len(data_autre)):
    if data_autre['switchpoint_buffer'][i] == [30]:
        data_autre['switchpoint_buffer'][i] = [] # change back to empty array

# Add column giving the number of switchpoint for each valuation price list 
data['nb_switchpoint'] = [len(data['switchpoint'][i]) for i in range(len(data))]

column_order = list(data.columns)
column_order.insert(column_order.index('switchpoint') + 1, column_order.pop(column_order.index('nb_switchpoint')))
data = data[column_order] # get the switching points after the column of switchinpoints

# Get the instances of Mutliple Switching Points (MSP) in valuation price lists 
# meaning, for each participant, the number of valuation price lists in which 
# they have multiple switching points (regardless of the number of MSP per price list)
switchpoint_counts_filtered = data[data['nb_switchpoint'] >= 2] # only count for MSP
MSP_counts = switchpoint_counts_filtered.groupby('id')['nb_switchpoint'].count().reset_index() # instances of MSP for each participant across all 28 price lists
data_autre = data_autre.merge(MSP_counts, how='left', on='id')
data_autre = data_autre.rename(columns={'nb_switchpoint': 'nb_MSP_valuation'}) # change the name of the previously added column (called nb_switchpoint as taken from data)


################################################
# Add case order of each price list
################################################

# Add information about the order of cases (from app_page) to data and data_autre    
data_autre = data_autre.merge(app_page[['id', 'order of cases']], how='left', on='id')
data = data.merge(app_page[['id', 'order of cases']], how='left', on='id')

data['case_order'] = np.nan
for i in range(len(data)):
    data['case_order'][i] = data['order of cases'][i].index(data['case'][i]) + 1 # give the order in which cases are presented (1 to 4)

column_order_1 = list(data.columns)
column_order_1.insert(column_order_1.index('case') + 1, column_order_1.pop(column_order_1.index('case_order')))
data = data[column_order_1] # get the order of cases after the column of cases

################################################
# Add valuation of each price list
################################################

# Create function giving the normalized certainty equivalent of price lists
def valuation(optionB, switchpoint):
    midpoint = (optionB[switchpoint[0]-1]+optionB[switchpoint[0]-2])/2 # get value of certainty equivalent (value between rows of switching)
    valuation =  midpoint/optionB[-1] # certainty equivalent normalized by maximal certain value (Option B) to give valuation
    return np.round(valuation, decimals = 3) # we round to take care of float problem
   
data['valuation'] = np.nan

# Add valuations for all price lists
for i in range(len(data)):
    if data['switchpoint'][i] != []:
        data['valuation'][i] = valuation(data['option_B_vector'][i],data['switchpoint'][i])    
    else:
        data['valuation'][i] = 1 # if Option B never chosen, we give a valuation of 1 (100%)

data['valuation'] = data['valuation']*100 # rewrite valuations as percentages

column_order_2 = list(data.columns)
column_order_2.insert(column_order_2.index('switchpoint') + 1, column_order_2.pop(column_order_2.index('valuation')))
data = data[column_order_2] # put the valuation column just after switchpoint column

################################################
# Add attention data of each price list
################################################

# Drop times of revealing the urn that are inferior or equal to 200 ms 
data['watching_urn_ms_corrected'] = data['watching_urn_ms'].apply(lambda arr: np.array([x for x in arr if x > 200])) 

# Get relative attention - attention towards urn normalized by total time spent on price list (in %)
# As the total time spent on price list is in seconds, we multiply it by 1000 to get ms
dwell_time_prop = [np.sum(data['watching_urn_ms_corrected'][i])/(data['total_time_spent_s'][i]*1000) for i in range(len(data))]
dwell_time_relative = [x * 100 for x in dwell_time_prop] # Get this relative attention as percentage

# Get absolute attention - attention towards urn (in s)
# As watching_urn_ms_corrected is in ms, we divide by 1000 to get s
dwell_time_absolute = [np.sum(data['watching_urn_ms_corrected'][i])/1000 for i in range(len(data))]

# Add the relative and absolute times in data (after watching_urn_ms_corrected) 
data.insert(data.columns.get_loc('watching_urn_ms_corrected') + 1, 'dwell_time_relative', dwell_time_relative)
data.insert(data.columns.get_loc('dwell_time_relative') + 1, 'dwell_time_absolute', dwell_time_absolute)

# Add information about the number of times the urn was revealed (frequency)
data['frequency'] = [len(data['watching_urn_ms_corrected'][i]) for i in range(len(data))]

# Put columns for corrected attention and frequency after column of watching_urn_ms
column_order_3 = list(data.columns)
column_order_3.insert(column_order_3.index('watching_urn_ms') + 1, column_order_3.pop(column_order_3.index('watching_urn_ms_corrected')))
column_order_3.insert(column_order_3.index('watching_urn_ms_corrected') + 1, column_order_3.pop(column_order_3.index('frequency')))
data = data[column_order_3]

################################################
# Assign a number to each participant
################################################

# Group participants by their id and assign a number for each id
data['number'] = data.groupby('id').ngroup() + 1
id_column_index = data.columns.get_loc('id')
data.insert(id_column_index + 1, 'number', data.pop('number')) # add number column after id 

# Add this number information in data_autre as well 
data_autre = data_autre.merge(data[['id', 'number']], how='left', on='id')
data_autre= data_autre.drop_duplicates(subset=['id'])

column_order_4 = list(data_autre.columns)
column_order_4.insert(column_order_4.index('id') + 1, column_order_4.pop(column_order_4.index('number')))
data_autre = data_autre[column_order_4] # add number column after id 

# Add this number information in survey as well 
survey = survey.merge(data[['id', 'number']], how='left', on='id')
survey= survey.drop_duplicates(subset=['id'])

column_order_5 = list(survey.columns)
column_order_5.insert(column_order_5.index('id') + 1, column_order_5.pop(column_order_5.index('number')))
survey = survey[column_order_5] # add number column after id 

################################################
# Get Dummy Variables for Regression model 
################################################

# Get charity and tradeoff dummies
data['charity'] = np.nan # indicator for whether the lottery is for the charity (1) or not (0)
data['tradeoff'] = np.nan # indicator for whether we are in tradeoff context (1) or not (0)

# Get dummies for all price lists 
for i in range(len(data)):
    if data['case'][i] == 'ASPS':
        data['charity'][i] = 0
        data['tradeoff'][i] = 0
    elif data['case'][i] == 'ASPC':
        data['charity'][i] = 1
        data['tradeoff'][i] = 1
    elif data['case'][i] == 'ACPC':
        data['charity'][i] = 1
        data['tradeoff'][i] = 0
    elif data['case'][i] == 'ACPS': 
        data['charity'][i] = 0
        data['tradeoff'][i] = 1

data['interaction'] = data['charity'] * data['tradeoff'] # interaction term of charity and tradeoff dummies 


################################################
# Exclusion criterion
################################################

# Using our exclusion criterion (which was pre-registered), we remove participants
# that starting from Option B in calibration price list 

exclusion_criterion = data_autre.loc[data_autre['exclusion_B_to_A'] == 1, 'id'] # participants that start with Option B in calibration price list

data = data.drop(data[data['id'].isin(exclusion_criterion) == True].index) # we remove those participants from the database data
data = data.reset_index(drop=True)

data_autre = data_autre.drop(data_autre[data_autre['id'].isin(exclusion_criterion) == True].index) # we also remove those participants from the database data_autre
data_autre = data_autre.reset_index(drop=True)

survey = survey.drop(survey[survey['id'].isin(exclusion_criterion) == True].index) # we also remove those participants from the database survey
survey = survey.reset_index(drop=True)


# %%
# =============================================================================
# SAVE DATASETS AS FILES
# =============================================================================

# Save data as "dataset.csv"
data_path = path + '/dataset.csv'
data.to_csv(data_path, index=False)
# this file pools all the data of interest for the analysis (apart from survey info)

# Save data_autre as "criterion info data.csv"
data_path_2 = path + '/criterion info data.csv'
data_autre.to_csv(data_path_2, index=False)
# this file combines all participant specific information 

# Save survey as "survey data.csv"
data_path_3 = path + '/survey data.csv'
survey.to_csv(data_path_3, index=False)
# this file gathers all of the survey information (used for controls in regressions)

# The CSV files are saved in the folder from which the raw data is taken from 


# %%
# =============================================================================
# SURVEY INFORMATION
# =============================================================================

################################################
# Socio-demographic survey
################################################

# We extract the socio-demographic information of participants for different groups
# We find the mean age, percentage of women and mean highest education level 

# For all subjects (not including ones that were excluded with above criterion)
print()
print()

print('ALL SUBJECTS')
print('The mean age is ' + str(survey['Demog_AGE'].mean()))
print('There is ' + 
      str(round(100*len(survey[survey['Demog_Sex']==1])/
                (len(survey[survey['Demog_Sex']==1])+ len(survey[survey['Demog_Sex']==2])), 1))
                        + ' % of women')
print('The mean highest education level is ' + 
      str(['A level', 'Bsci', 'Msci', 'Phd', 'RNS'][round(survey['Demog_High_Ed_Lev'].mean())-1])) # -1 since the Demog_High_Ed_Lev go from 1 to 5 and 
                                                                                                   # here we want index to have associated value in array

print()

# For principal analysis (not including subjects having MSP and being censored in calibration price list)

data_principal = data.loc[data['censored_calibration'] == 0]
data_principal = data_principal.reset_index(drop=True)
 
data_autre_principal = data_autre.loc[data_autre['censored_calibration'] == 0] 
data_autre_principal = data_autre_principal.reset_index(drop=True)

survey_principal = pd.merge(data_autre_principal[['id']], survey, on='id', how='inner')

print('Principal analyses SUBJECTS')
print('The mean age is ' + str(survey_principal['Demog_AGE'].mean()))
print('There is ' + 
      str(round(100*len(survey_principal[survey_principal['Demog_Sex']==1])/
                (len(survey_principal[survey_principal['Demog_Sex']==1])+len(survey_principal[survey_principal['Demog_Sex']==2])), 1))
                        + ' % of women')
print('The mean highest education level is ' + 
      str(['A level', 'Bsci', 'Msci', 'Phd', 'RNS'][round(survey_principal['Demog_High_Ed_Lev'].mean())-1]))
print()

# For censored subjects specifically (only subjects being censored in calibration price list)

data_censored = data.loc[data['censored_calibration'] == 0]
data_censored = data_censored.reset_index(drop=True)
                                            
data_autre_censored = data_autre.loc[data_autre['censored_calibration'] == 1] 
data_autre_censored = data_autre_censored.reset_index(drop=True)

survey_censored = pd.merge(data_autre_censored[['id']], survey, on='id', how='inner')

print('Censored  SUBJECTS')
print('The mean age is ' + str(survey_censored['Demog_AGE'].mean()))
print('There is ' + 
      str(round(100*len(survey_censored[survey_censored['Demog_Sex']==1])/
                (len(survey_censored[survey_censored['Demog_Sex']==1])+len(survey_censored[survey_censored['Demog_Sex']==2])), 1))
                        + ' % of women')
print('The mean highest education level is ' + 
      str(['A level', 'Bsci', 'Msci', 'Phd', 'RNS'][round(survey_censored['Demog_High_Ed_Lev'].mean())-1]))
print()


################################################
# Charity attitude survey
################################################

# For principal analysis (not including subjects having MSP and being censored in calibration price list)
print('Principal analyses SUBJECTS')
print('Charity like: ' + str(survey_principal['Charity_LIKE'].mean()))
print('Charity trust: ' + str(survey_principal['Charity_TRUST'].mean()))
print('Charity likelihood to donate: ' + str(survey_principal['Charity_LIKELY'].mean()))
print('Charity actual donation: ' + str(survey_principal['Charity_DONATION_DONE'].mean()))
print()

# For censored subjects specifically (only subjects being censored in calibration price list)
print('Censored  SUBJECTS')
print('Charity like: ' + str(survey_censored['Charity_LIKE'].mean()))
print('Charity trust: ' + str(survey_censored['Charity_TRUST'].mean()))
print('Charity likelihood to donate: ' + str(survey_censored['Charity_LIKELY'].mean()))
print('Charity actual donation: ' + str(survey_censored['Charity_DONATION_DONE'].mean()))
print()

# Compare results from principal analysis and censored subjects 
print('T-test between principal analysis and censored subjects: ')

t_statistic_like, p_value_like = ttest_ind(survey_principal['Charity_LIKE'], survey_censored['Charity_LIKE'])
print('Charity like: t-test ' + str(t_statistic_like) + ' and p ' + str(p_value_like))
print()

t_statistic_trust, p_value_trust = ttest_ind(survey_principal['Charity_TRUST'], survey_censored['Charity_TRUST'])
print('Charity trust: t-test ' + str(t_statistic_trust) + ' and p ' + str(p_value_trust))
print()

t_statistic_likely, p_value_likely = ttest_ind(survey_principal['Charity_LIKELY'], survey_censored['Charity_LIKELY'])
print('Charity likelihood to donate: t-test ' + str(t_statistic_likely) + ' and p ' + str(p_value_likely))
print()

t_statistic_donation, p_value_donation = ttest_ind(survey_principal['Charity_DONATION_DONE'], survey_censored['Charity_DONATION_DONE'])
print('Charity actual donation: t-test ' + str(t_statistic_donation) + ' and p ' + str(p_value_donation))
print()



# %%
# =============================================================================
# INFORMATION OF CRITERIA AND DATA TRANSFORMATION
# =============================================================================

# We extract information regarding the different criteria and data transformation (pre-registered)

print()
print('Information of criteria and data transformation:')
print()
print('For ALL DATA')
print('The percentage of censored valuations over all data : ' 
      + str(len(data[data['valuation']==100])*100/len(data['valuation']))) 

print()
print('The percentage of valuations with MSP over all data : ' 
      + str(len(data[(data['nb_switchpoint']!=1) & (data['nb_switchpoint']!=0)])*100/len(data['nb_switchpoint'])))

print()
print('The percentage of occurence of attention time inferior or equal to 200ms over all data : ' 
      + str(sum(data['watching_urn_ms'].map(lambda arr: any(x <= 200 for x in arr)))/len(data['watching_urn_ms'])*100))

# Finding total attention time inferior or equal to 200ms

attention_less_200 = data[data['watching_urn_ms'].apply(lambda arr: any(x <= 200 for x in arr))]['watching_urn_ms'].reset_index(drop=True)
less_200 = []
all_all = []
for i in range(len(attention_less_200)):
    for j in range(len(attention_less_200[i])):
        if attention_less_200[i][j] <= 200:
            less_200.append(attention_less_200[i][j]) # put all times inferior to 200ms in same array
for i in range(len(data['watching_urn_ms'])):
    for j in range(len(data['watching_urn_ms'][i])):
        all_all.append(data['watching_urn_ms'][i][j]) # put all times in same array
        
print()
print('The total attention time inferior or equal to 200ms over all data is : ' 
      + str(np.sum(less_200)) + ' ms')
print()
print('The percentage of attention time inferior or equal to 200ms over all data is : ' 
      + str(np.sum(less_200)/np.sum(all_all)*100))




# We do exactly the same thing for censored subjects data
print()
print('For CENSORED DATA')
print('The percentage of censored valuations for censored participants data: '
      + str(len(data_censored[data_censored['valuation']==100])*100/len(data_censored['valuation']))) 

print()
print('The percentage of valuations with MSP for censored participants data: '
      + str(len(data_censored[(data_censored['nb_switchpoint']!=1) & (data_censored['nb_switchpoint']!=0)])*100/len(data_censored['nb_switchpoint']))) 

print()
print('The percentage of occurence of attention time inferior or equal to 200ms  for censored participants data: '
      + str(sum(data_censored['watching_urn_ms'].map(lambda arr: any(x <= 200 for x in arr)))/len(data_censored['watching_urn_ms'])*100)) 


attention_less_200_cen = data_censored[data_censored['watching_urn_ms'].apply(lambda arr: any(x <= 200 for x in arr))]['watching_urn_ms'].reset_index(drop=True)
less_200_cen = []
all_cen = []
for i in range(len(attention_less_200_cen)):
    for j in range(len(attention_less_200_cen[i])):
        if attention_less_200_cen[i][j] <= 200:
            less_200_cen.append(attention_less_200_cen[i][j])

for i in range(len(data_censored['watching_urn_ms'])):
    for j in range(len(data_censored['watching_urn_ms'][i])):
        all_cen.append(data_censored['watching_urn_ms'][i][j])

print()
print('The total attention time inferior or equal to 200ms over censored subjects data is : ' 
      + str(np.sum(less_200_cen)) + ' ms')
print()
print('The percentage of attention time inferior or equal to 200ms over censored subjects data is : ' 
      + str(np.sum(less_200_cen)/np.sum(all_cen)*100))


