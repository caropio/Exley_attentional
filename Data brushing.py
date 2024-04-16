#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:38:56 2024

@author: carolinepioger
"""
 
 
import pandas as pd
import numpy as np
import ast 

MSP_excl = 1 # Put 0 if include MSP calib in analysis and 1 if we exclude them 
attention_type = 'absolute' # relative for % of total time and 'absolute' for raw time

# Info to find data
path = '/Users/carolinepioger/Desktop/pretest vincent' # change to yours :)
dates = ['2024-03-28', '2024-03-25', '2024-04-03']

# Data for analysis 

dfs = []

for date in dates:
    df1 = pd.read_csv(path + '/EXLEY_ASPS_' + date + '.csv')
    df2 = pd.read_csv(path + '/EXLEY_ACPC_' + date + '.csv')
    df3 = pd.read_csv(path + '/EXLEY_ASPC_' + date + '.csv')
    df4 = pd.read_csv(path + '/EXLEY_ACPS_' + date + '.csv')
    dfs.extend([df1, df2, df3, df4])

concatenated_df = pd.concat(dfs, ignore_index=True)
concatenated_df = concatenated_df.drop(concatenated_df[concatenated_df['participant._current_page_name'] != 'prolific'].index)
data = concatenated_df.sort_values(by=['participant.code'])

# Data to check criterions
data_autre = pd.concat([pd.read_csv(path + '/EXLEY_ASSO_' + date + '.csv') for date in dates], ignore_index=True)
data_autre = data_autre.drop(data_autre[data_autre['participant._current_page_name'] != 'prolific'].index)

data_autre['choix_calibration'] = data_autre[['player.choice_x_' + str(i) for i in range(1, 17)]].values.tolist()
data_autre['choix_buffer'] = data_autre[['player.choice_y_' + str(i) for i in range(1, 17)]].values.tolist()
columns_mapping = { 'participant.code': 'id', 
 'session.code': 'session', 
 'player.PROLIFIC_ID': 'prolific', 
 'player.association_choice': 'charity_name', 
 'player.x1_norm_after_correction': 'charity_calibration', 
 'choix_calibration': 'calibration_choices', 
 'choix_buffer': 'buffer_choices'}
data_autre = data_autre.rename(columns=columns_mapping)[list(columns_mapping.values())]

data_autre = data_autre.reset_index(drop=True)

# Data for between-subject analysis 
app_page = pd.concat([pd.read_csv(path + '/StartApp_' + date + '.csv') for date in dates], ignore_index=True)
app_page = app_page.drop(app_page[app_page['participant._current_page_name'] != 'prolific'].index)


app_page = app_page[['participant.code','session.code', 'player.sequence_of_apps']]
app_page = app_page.rename(columns={'participant.code':'id', 'session.code': 'session', 
                                                  'player.sequence_of_apps':'case_order'})
app_page = app_page.reset_index(drop=True)

for i in range(len(app_page)):
    app_page['case_order'][i] = ast.literal_eval(app_page['case_order'][i])
    app_page['case_order'][i] = app_page['case_order'][i][8:12]

app_page['first case'] = np.nan
app_page['second case'] = np.nan
app_page['third case'] = np.nan
app_page['fourth case'] = np.nan
app_page['order of cases'] = np.nan
case_number = ['first', 'second', 'third', 'fourth']

for i in range(len(app_page)):
    for j in range(0,4):
        app_page[str(case_number[j]) + ' case'][i] = app_page['case_order'][i][j][-4:] # get order of presentation of cases
    app_page['order of cases'][i] = [app_page['first case'][i], app_page['second case'][i], app_page['third case'][i], app_page['fourth case'][i]]

# Data from surveys 
survey = pd.concat([pd.read_csv(path + '/EXLEY_DEMOG_' + date + '.csv') for date in dates], ignore_index=True)
survey = survey.drop(survey[survey['participant._current_page_name'] != 'prolific'].index)

columns_mapping = {
    'participant.code': 'id',
    'session.code': 'session',
    'player.PROLIFIC_ID': 'prolific',
    'player.AGE': 'Demog_AGE',
    'player.SEXE': 'Demog_Sex',
    'player.DISCIPLINE': 'Demog_Field',
    'player.NIVEAU_ETUDE': 'Demog_High_Ed_Lev'
}

for i in range(1, 16):
    columns_mapping[f'player.QUEST_{i}'] = f'NEP_{i}'

for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']:
    columns_mapping[f'player.CHARITY_{j}'] = f'Charity_{j}'

survey = survey.rename(columns=columns_mapping)[
    ['id', 'session', 'prolific', 'Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + 
    [f'NEP_{i}' for i in range(1, 16)] + 
    [f'Charity_{j}' for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]
][list(columns_mapping.values())]   
                          
survey = survey.reset_index(drop=True)


# %%
# =============================================================================
# FROM RAW DATA TO CLEAN DATA
# =============================================================================

# Note that in this database each row gives the data of one table (so there 
# are multiple rows for each participant). The database is ordered by participant

# Transform the 21 columns of choices into one summing up the choices 
data['player.choix_overview'] = data[['player.choix' + str(i) for i in range(1, 22)]].values.tolist()
data['player.prob_lottery'] = data[['player.P1', 'player.P2']].values.tolist()

# Keep columns of interest
columns_mapping = {
    'participant.code': 'id', # number id associated to each participant
    'session.code': 'session',  # number of session
    'player.association_choice': 'charity_name', # charity choice from part 1
    'player.x1_norm_after_correction': 'charity_calibration', # participant specific X from part 2
    'player.TREATMENT': 'case', # case of each table of part 3
    'subsession.round_number': 'round_order', # order of presentation of tables within case
    'player.x1_outcome_vec_order': 'option_A_vector', # non-zero amount of lottery (option A)
    'player.prob_lottery': 'prob_option_A', # number of green and red balls of associated urn
    'player.WTP_VEC': 'option_B_vector', # the different values of option B along the rows 
    'player.choix_overview': 'choices', # the choices (As and Bs) made along the rows 
    'player.temps_decision': 'total_time_spent_s', # time it took to complete the entire table (and click on next button) in seconds
    'player.temps_regard_urne': 'watching_urn_ms', # dwell time of watching urn (array of times in ms) 
    'player.user_actions': 'temporal_information' # temporal information (when urn is unmasked and choices are made)
}

data = data.rename(columns=columns_mapping)[list(columns_mapping.values())]
data = data.reset_index(drop=True)

# Reconstruct option_A_vector, prob_option_A and option_B_vector without issues (string to float)

def proba_A(array_boules):
    return array_boules[0]/np.sum(array_boules) # so our column is the proportion of green balls in urn 
                                                # and thus the probability of the non-zero amount 

for i in range(len(data)):
    data.loc[i, 'prob_option_A'] = proba_A(data.loc[i, 'prob_option_A'])

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


# Reconstruct watching_urn_ms without issues (string to float)

data['watching_urn_ms'] = data['watching_urn_ms'].apply(lambda x: str(x).replace('null', '0'))

for i in range(len(data)):
    if isinstance(data['watching_urn_ms'][i], str):
        data['watching_urn_ms'][i] = ast.literal_eval(data['watching_urn_ms'][i])


# %%
# =============================================================================
# FROM CLEAN DATA TO COMPLETE DATA
# =============================================================================

# Calculate data of interest with existing database 

# Add switchpoint values for each table

def upper_switchpoint(array):
    switch_indices = []
    
    for i in range(1,len(array)):
        if array[i] != array[i - 1]:
            switch_indices.append(i+1) 
    
    return switch_indices # num du choix après switch

# for part 3
switchpoint_values = [upper_switchpoint(data['choices'][i]) for i in range(len(data))]
data.insert(data.columns.get_loc('choices') + 1, 'switchpoint', switchpoint_values)

# for part 2
switchpoint_values_calib = [upper_switchpoint(data_autre['calibration_choices'][i]) for i in range(len(data_autre))]
switchpoint_values_buffer = [upper_switchpoint(data_autre['buffer_choices'][i]) for i in range(len(data_autre))]
data_autre.insert(data_autre.columns.get_loc('calibration_choices') + 1, 'switchpoint_calib', switchpoint_values_calib)
data_autre.insert(data_autre.columns.get_loc('buffer_choices') + 1, 'switchpoint_buffer', switchpoint_values_buffer)

data_autre['censored_calibration'] = np.nan
data_autre['censored_buffer'] = np.nan

for i in range(len(data_autre)):
    if data_autre['switchpoint_calib'][i] == []:
        data_autre['censored_calibration'][i] = 1   
    elif len(data_autre['switchpoint_calib'][i]) == 1:
        data_autre['censored_calibration'][i] = 0  
    elif len(data_autre['switchpoint_calib'][i]) > 1:
        data_autre['censored_calibration'][i] = 'MSP'  

for i in range(len(data_autre)):
    if data_autre['switchpoint_buffer'][i] == []:
        data_autre['censored_buffer'][i] = 1   
    elif len(data_autre['switchpoint_buffer'][i]) == 1:
        data_autre['censored_buffer'][i] = 0  
    elif len(data_autre['switchpoint_buffer'][i]) > 1:
        data_autre['censored_buffer'][i] = 'MSP'   


def calib_buffer_choice(calib, buffer):
    if calib[0] < buffer[0]:
        return 1
    else: 
        return 0 
    
for i in range(len(data_autre)):
    if data_autre['switchpoint_calib'][i] == []:
        data_autre['switchpoint_calib'][i] = [30] # change to 30 so that it is numerically comparable
for i in range(len(data_autre)):
    if data_autre['switchpoint_buffer'][i] == []:
        data_autre['switchpoint_buffer'][i] = [30] # change to 30 so that it is numerically comparable
        
data_autre['buffer>calib']= [calib_buffer_choice(data_autre['switchpoint_calib'][i],data_autre['switchpoint_buffer'][i]) for i in range(len(data_autre))]
options_buffer= np.round(np.linspace(0,30,16), decimals=1)
data_autre['buffer_X'] = np.nan
for i in range(len(data_autre)):
    if data_autre['switchpoint_buffer'][i] == [30]:
        data_autre['buffer_X'][i] = 30
    else: 
        data_autre['buffer_X'][i] = options_buffer[data_autre['switchpoint_buffer'][i][0]-1] # get the participant-specific X for the buffer table 

column_order_2 = list(data_autre.columns)
column_order_2.insert(column_order_2.index('switchpoint_calib') + 1, column_order_2.pop(column_order_2.index('censored_calibration')))
column_order_2.insert(column_order_2.index('switchpoint_buffer') + 1, column_order_2.pop(column_order_2.index('censored_buffer')))
column_order_2.insert(column_order_2.index('charity_calibration') + 1, column_order_2.pop(column_order_2.index('buffer_X')))
data_autre = data_autre[column_order_2]

data = data.merge(data_autre[['id','buffer_X']], how='left', on='id')
data = data.merge(data_autre[['id','censored_calibration']], how='left', on='id')
column_order_3 = list(data.columns)
column_order_3.insert(column_order_3.index('charity_calibration') + 1, column_order_3.pop(column_order_3.index('buffer_X')))
column_order_3.insert(column_order_3.index('buffer_X') + 1, column_order_3.pop(column_order_3.index('censored_calibration')))
data = data[column_order_3]

for i in range(len(data_autre)):
    if data_autre['switchpoint_calib'][i] == [30]:
        data_autre['switchpoint_calib'][i] = [] # change back to empty array

for i in range(len(data_autre)):
    if data_autre['switchpoint_buffer'][i] == [30]:
        data_autre['switchpoint_buffer'][i] = [] # change back to empty array

data['nb_switchpoint'] = [len(data['switchpoint'][i]) for i in range(len(data))]

switchpoint_counts_filtered = data[data['nb_switchpoint'] >= 2] # only count MSP
MSP_counts = switchpoint_counts_filtered.groupby('id')['nb_switchpoint'].count().reset_index() # instances of MSP for each participant
data_autre = data_autre.merge(MSP_counts, how='left', on='id')
data_autre = data_autre.rename(columns={'nb_switchpoint': 'nb_MSP_valuation'})    
data_autre = data_autre.merge(app_page[['id', 'order of cases']], how='left', on='id')
data = data.merge(app_page[['id', 'order of cases']], how='left', on='id')

data['case_order'] = np.nan
for i in range(len(data)):
    data['case_order'][i] = data['order of cases'][i].index(data['case'][i]) + 1 # give the order in which cases are presented 

# Add valuation of each lottery

def valuation(optionB, switchpoint):
    midpoint = (optionB[switchpoint[0]-1]+optionB[switchpoint[0]-2])/2 
    valuation =  midpoint/optionB[-1] # midpoint divisé par maximum pour avoir pourcentage valuation
    return np.round(valuation, decimals = 3) # we round to take care of float problem
   

data['valuation'] = np.nan

for i in range(len(data)):
    if data['switchpoint'][i] != []:
        data['valuation'][i] = valuation(data['option_B_vector'][i],data['switchpoint'][i])    
    else:
        data['valuation'][i] = 1

column_order = list(data.columns)
column_order.insert(column_order.index('switchpoint') + 1, column_order.pop(column_order.index('valuation')))
column_order.insert(column_order.index('case') + 1, column_order.pop(column_order.index('case_order')))
data = data[column_order]

# Add attention data of each lottery (according to either relative or absolute) 
data['watching_urn_ms_corrected'] = data['watching_urn_ms'].apply(lambda arr: np.array([x for x in arr if x > 300])) # we drop values lower or equal to 300ms 

dwell_time_prop = [np.sum(data['watching_urn_ms_corrected'][i])/(data['total_time_spent_s'][i]*1000) for i in range(len(data))]
dwell_time_relative = [x * 100 for x in dwell_time_prop] # to get percentage
dwell_time_absolute = [np.sum(data['watching_urn_ms_corrected'][i])/1000 for i in range(len(data))]
data.insert(data.columns.get_loc('watching_urn_ms_corrected') + 1, 'dwell_time_relative', dwell_time_relative)
data.insert(data.columns.get_loc('dwell_time_relative') + 1, 'dwell_time_absolute', dwell_time_absolute)

if attention_type == 'relative':
    data.insert(data.columns.get_loc('watching_urn_ms_corrected') + 1, 'dwell_time', dwell_time_relative)
elif attention_type == 'absolute':  
    data.insert(data.columns.get_loc('watching_urn_ms_corrected') + 1, 'dwell_time', dwell_time_absolute)


data['frequency'] = [len(data['watching_urn_ms_corrected'][i]) for i in range(len(data))]
column_order_2 = list(data.columns)
column_order_2.insert(column_order_2.index('switchpoint') + 1, column_order_2.pop(column_order_2.index('nb_switchpoint')))
column_order_2.insert(column_order_2.index('watching_urn_ms') + 1, column_order_2.pop(column_order_2.index('watching_urn_ms_corrected')))
column_order_2.insert(column_order_2.index('watching_urn_ms_corrected') + 1, column_order_2.pop(column_order_2.index('dwell_time')))
column_order_2.insert(column_order_2.index('dwell_time') + 1, column_order_2.pop(column_order_2.index('frequency')))
data = data[column_order_2]


# Remove participants with MSP in part 2

exclude_participants_2 = data_autre.loc[data_autre['censored_calibration'] == 'MSP', 'id'] 

if MSP_excl == 1: 
    data = data.drop(data[data['id'].isin(exclude_participants_2) == True].index)
else: 
    data = data

data = data.reset_index(drop=True)

data['number'] = data.groupby('id').ngroup() + 1
id_column_index = data.columns.get_loc('id')
data.insert(id_column_index + 1, 'number', data.pop('number'))
    

# Save the concatenated dataset
data_path = path + '/dataset.csv'
data.to_csv(data_path, index=False)
data_path_2 = path + '/criterion info data.csv'
data_autre.to_csv(data_path_2, index=False)
data_path_3 = path + '/survey data.csv'
survey.to_csv(data_path_3, index=False)

data_path


