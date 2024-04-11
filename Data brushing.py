#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:38:56 2024

@author: carolinepioger
"""
 
 
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import ast 
import matplotlib.cm as cm
import statsmodels.formula.api as smf


censure = 0 # Put 0 if include censored participants in analysis and 1 if we exclude them 
MSP_excl = 1 # Put 0 if include MSP calib in analysis and 1 if we exclude them 
by_ind = 0 # Put 0 if no display of individual plots and 1 if display 
attention_type = 'absolute' # relative for % of total time and 'absolute' for raw time

# Info to find data
path = '/Users/carolinepioger/Desktop' # change to yours :)
file = '/pretest vincent' # to adapt
dates = ['2024-03-28', '2024-03-25', '2024-04-03']

# Data for analysis 

dfs = []

for date in dates:
    df1 = pd.read_csv(path + file + '/EXLEY_ASPS_' + date + '.csv')
    df2 = pd.read_csv(path + file + '/EXLEY_ACPC_' + date + '.csv')
    df3 = pd.read_csv(path + file + '/EXLEY_ASPC_' + date + '.csv')
    df4 = pd.read_csv(path + file + '/EXLEY_ACPS_' + date + '.csv')
    dfs.extend([df1, df2, df3, df4])

concatenated_df = pd.concat(dfs, ignore_index=True)
concatenated_df = concatenated_df.drop(concatenated_df[concatenated_df['participant._current_page_name'] != 'prolific'].index)
data = concatenated_df.sort_values(by=['participant.code'])

# Data to check criterions
data_autre = pd.concat([pd.read_csv(path + file + '/EXLEY_ASSO_' + date + '.csv') for date in dates], ignore_index=True)
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
app_page = pd.concat([pd.read_csv(path + file + '/StartApp_' + date + '.csv') for date in dates], ignore_index=True)
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
survey = pd.concat([pd.read_csv(path + file + '/EXLEY_DEMOG_' + date + '.csv') for date in dates], ignore_index=True)
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

#for part 3
switchpoint_values = [upper_switchpoint(data['choices'][i]) for i in range(len(data))]
data.insert(data.columns.get_loc('choices') + 1, 'switchpoint', switchpoint_values)

#for part 2
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
column_order_3 = list(data.columns)
column_order_3.insert(column_order_3.index('charity_calibration') + 1, column_order_3.pop(column_order_3.index('buffer_X')))
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

# if attention_type == 'relative':
#     dwell_time_prop = [np.sum(data['watching_urn_ms_corrected'][i])/(data['total_time_spent_s'][i]*1000) for i in range(len(data))]
#     dwell_time = [x * 100 for x in dwell_time_prop] # to get percentage
# elif attention_type == 'absolute':
#     dwell_time = [np.sum(data['watching_urn_ms_corrected'][i])/1000 for i in range(len(data))]
    
# data.insert(data.columns.get_loc('watching_urn_ms_corrected') + 1, 'dwell_time', dwell_time)

data['frequency'] = [len(data['watching_urn_ms_corrected'][i]) for i in range(len(data))]
column_order_2 = list(data.columns)
column_order_2.insert(column_order_2.index('switchpoint') + 1, column_order_2.pop(column_order_2.index('nb_switchpoint')))
column_order_2.insert(column_order_2.index('watching_urn_ms') + 1, column_order_2.pop(column_order_2.index('watching_urn_ms_corrected')))
column_order_2.insert(column_order_2.index('watching_urn_ms_corrected') + 1, column_order_2.pop(column_order_2.index('dwell_time')))
column_order_2.insert(column_order_2.index('dwell_time') + 1, column_order_2.pop(column_order_2.index('frequency')))
data = data[column_order_2]


# Remove (or not) participants with censored values in part 2

exclude_participants = data_autre.loc[data_autre['censored_calibration'] == 1, 'id'] 

if censure == 1: 
    data = data.drop(data[data['id'].isin(exclude_participants) == True].index)
else: 
    data = data

# Remove particiapnts with MSP in part 2

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
data_path = path + file + '/dataset.csv'
data.to_csv(data_path, index=False)
data_path_2 = path + file + '/criterion info data.csv'
data_autre.to_csv(data_path_2, index=False)

data_path

# %%
# =============================================================================
# VISUALISE DATA 
# =============================================================================

data_for_plot = data
data_for_plot['charity'] = np.nan # indicator for whether the lottery is for the charity 
data_for_plot['tradeoff'] = np.nan # indicator for whether we are in tradeoff context

for i in range(len(data_for_plot)):
    if data_for_plot['case'][i] == 'ASPS':
        data_for_plot['charity'][i] = 0
        data_for_plot['tradeoff'][i] = 0
    elif data_for_plot['case'][i] == 'ASPC':
        data_for_plot['charity'][i] = 1
        data_for_plot['tradeoff'][i] = 1
    elif data_for_plot['case'][i] == 'ACPC':
        data_for_plot['charity'][i] = 1
        data_for_plot['tradeoff'][i] = 0
    elif data_for_plot['case'][i] == 'ACPS': 
        data_for_plot['charity'][i] = 0
        data_for_plot['tradeoff'][i] = 1

ASPS = data_for_plot[(data_for_plot['charity'] == 0) & (data_for_plot['tradeoff'] == 0)]
ACPC = data_for_plot[(data_for_plot['charity'] == 1) & (data_for_plot['tradeoff'] == 0)]
ASPC = data_for_plot[(data_for_plot['charity'] == 1) & (data_for_plot['tradeoff'] == 1)]
ACPS = data_for_plot[(data_for_plot['charity'] == 0) & (data_for_plot['tradeoff'] == 1)]

average_valuation_ASPS = ASPS.groupby('prob_option_A')['valuation'].median()
average_valuation_ACPC = ACPC.groupby('prob_option_A')['valuation'].median()
average_valuation_ACPS = ACPS.groupby('prob_option_A')['valuation'].median()
average_valuation_ASPC = ASPC.groupby('prob_option_A')['valuation'].median()

data_for_plot_2 = data_for_plot
data_for_plot_2['first case'] = [data_for_plot_2['order of cases'][i][0] for i in range(len(data_for_plot_2))]
not_first_case = data_for_plot_2.loc[data_for_plot_2['first case'] != data_for_plot_2['case']] 
data_for_plot_2 = data_for_plot_2.drop(not_first_case.index)

ASPS_between = data_for_plot_2[(data_for_plot_2['charity'] == 0) & (data_for_plot_2['tradeoff'] == 0)]
ACPC_between = data_for_plot_2[(data_for_plot_2['charity'] == 1) & (data_for_plot_2['tradeoff'] == 0)]
ASPC_between = data_for_plot_2[(data_for_plot_2['charity'] == 1) & (data_for_plot_2['tradeoff'] == 1)]
ACPS_between = data_for_plot_2[(data_for_plot_2['charity'] == 0) & (data_for_plot_2['tradeoff'] == 1)]

# Plot No Tradeoff Context (Replication Exley)

plt.plot(average_valuation_ASPS.index, average_valuation_ASPS * 100, label='ASPS', color='blue', marker='o', linestyle='-')
plt.plot(average_valuation_ACPC.index, average_valuation_ACPC * 100, label='ACPC', color='red', marker='o', linestyle='-')


x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)
plt.plot(x_fit, y_fit, color='grey', label='Expected value')

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Valuation as % of Riskless Lottery')
plt.title('(Median) Results for No Tradeoff Context ' + str(['with', 'without'][censure]) + ' censored partic')
plt.grid(True)
plt.legend()

plt.show()

# Plot Tradeoff Context (Replication Exley)

plt.plot(average_valuation_ACPS.index, average_valuation_ACPS * 100, label='ACPS', color='blue', marker='o', linestyle='-')
plt.plot(average_valuation_ASPC.index, average_valuation_ASPC * 100, label='ASPC', color='red', marker='o', linestyle='-')


x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)
plt.plot(x_fit, y_fit, color='grey', label='Expected value')

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Valuation as % of Riskless Lottery')
plt.title('(Median) Results for Tradeoff Context ' + str(['with', 'without'][censure]) + ' censored partic')
plt.grid(True)
plt.legend()

plt.show()

# Plot Self Lottery Valuation

plt.plot(average_valuation_ASPS.index, average_valuation_ASPS * 100, label='ASPS', color='green', marker='o', linestyle='-')
plt.plot(average_valuation_ACPS.index, average_valuation_ACPS * 100, label='ACPS', color='orange', marker='o', linestyle='-')


x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)
plt.plot(x_fit, y_fit, color='grey', label='Expected value')

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Valuation as % of Riskless Lottery')
plt.title('(Median) Results for Self Lottery Valuation ' + str(['with', 'without'][censure]) + ' censored partic')
plt.grid(True)
plt.legend()

plt.show()

# Plot Charity Lottery Valuation

plt.plot(average_valuation_ASPC.index, average_valuation_ASPC * 100, label='ASPC', color='green', marker='o', linestyle='-')
plt.plot(average_valuation_ACPC.index, average_valuation_ACPC * 100, label='ACPC', color='orange', marker='o', linestyle='-')


x_fit = np.linspace(0, 1, num = 10)
y_fit = np.linspace(0, 100, num = 10)
plt.plot(x_fit, y_fit, color='grey', label='Expected value')

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Valuation as % of Riskless Lottery')
plt.title('(Median) Results for Charity Lottery Valuation ' + str(['with', 'without'][censure]) + ' censored partic')
plt.grid(True)
plt.legend()

plt.show()

# Plot Valuation for each participant ### TO REMOVE FOR REAL DATA 

if by_ind == 1: 
    for i in range(1, data['number'].nunique()+1):
        ASPS_ind = ASPS.loc[ASPS['number'] == i, ['prob_option_A', 'valuation']] 
        ASPS_ind = ASPS_ind.sort_values(by=['prob_option_A'])
        ACPC_ind = ACPC.loc[ACPC['number'] == i, ['prob_option_A', 'valuation']] 
        ACPC_ind = ACPC_ind.sort_values(by=['prob_option_A'])
        ASPC_ind = ASPC.loc[ASPC['number'] == i, ['prob_option_A', 'valuation']]
        ASPC_ind = ASPC_ind.sort_values(by=['prob_option_A'])
        ACPS_ind = ACPS.loc[ACPS['number'] == i, ['prob_option_A', 'valuation']] 
        ACPS_ind = ACPS_ind.sort_values(by=['prob_option_A'])
                   
        fig, axs = plt.subplots(2, 2)
        fig.suptitle('Individual ' + str(i))
        
        axs[0, 0].plot(ASPS_ind['prob_option_A'], ASPS_ind['valuation']*100, label='ASPS', color='blue')
        axs[0, 0].plot(ACPC_ind['prob_option_A'], ACPC_ind['valuation']*100, label='ACPC', color='red')
        axs[0, 0].legend()
        axs[0, 0].plot(x_fit, y_fit, color='grey', label='Expected value')
        axs[0, 0].tick_params(left = False, right = False)
        axs[0, 0].grid(True)
        axs[0, 0].set_title('No Tradeoff Context')
        
        axs[0, 1].plot(ACPS_ind['prob_option_A'], ACPS_ind['valuation']*100,label='ACPS', color='blue')
        axs[0, 1].plot(ASPC_ind['prob_option_A'], ASPC_ind['valuation']*100, label='ASPC', color='red')
        axs[0, 1].legend()
        axs[0, 1].plot(x_fit, y_fit, color='grey', label='Expected value')
        axs[0, 1].tick_params(left = False, right = False)
        axs[0, 1].grid(True)
        axs[0, 1].set_title('Tradeoff Context')
        
        axs[1, 0].plot(ASPS_ind['prob_option_A'], ASPS_ind['valuation']*100, label='ASPS', color='orange')
        axs[1, 0].plot(ACPS_ind['prob_option_A'], ACPS_ind['valuation']*100, label='ACPS', color='green')
        axs[1, 0].legend()
        axs[1, 0].plot(x_fit, y_fit, color='grey', label='Expected value')
        axs[1, 0].tick_params(left = False, right = False)
        axs[1, 0].grid(True)
        axs[1, 0].set_title('Self Lottery Valuation')
        
        axs[1, 1].plot(ACPC_ind['prob_option_A'], ACPC_ind['valuation']*100, label='ACPC', color='orange')
        axs[1, 1].plot(ASPC_ind['prob_option_A'], ASPC_ind['valuation']*100, label='ASPC', color='green')
        axs[1, 1].legend()
        axs[1, 1].plot(x_fit, y_fit, color='grey', label='Expected value')
        axs[1, 1].tick_params(left = False, right = False)
        axs[1, 1].grid(True)
        axs[1, 1].set_title('Charity Lottery Valuation')
        
        for ax in axs.flat:
            ax.label_outer()
        plt.show()
else:
    pass


# Plot Attention (WITHIN-SUBJECT)

average_attention_ASPS = ASPS.groupby('prob_option_A')['dwell_time'].median()
average_attention_ACPC = ACPC.groupby('prob_option_A')['dwell_time'].median()
average_attention_ACPS = ACPS.groupby('prob_option_A')['dwell_time'].median()
average_attention_ASPC = ASPC.groupby('prob_option_A')['dwell_time'].median()

all_attention = pd.concat([average_attention_ASPS, average_attention_ACPC, average_attention_ACPS, average_attention_ASPC])
all_attention = all_attention.groupby('prob_option_A').median()

plt.plot(average_attention_ASPS.index, average_attention_ASPS, label='ASPS', color='blue', marker='o', linestyle='-')
plt.plot(average_attention_ACPS.index, average_attention_ACPS, label='ACPS', color='red', marker='o', linestyle='-')
plt.plot(average_attention_ASPC.index, average_attention_ASPC, label='ASPC', color='orange', marker='o', linestyle='-')
plt.plot(average_attention_ACPC.index, average_attention_ACPC, label='ACPC', color='green', marker='o', linestyle='-')

x_fit = np.linspace(0, 1, num = 10)

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Dwell time of urn (' +str(attention_type) +')')
plt.title('(Within-subj) Median Attentional processes (' +str(attention_type) +') ' + str(['with', 'without'][censure]) + ' censored partic')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(all_attention.index, all_attention, marker='o', linestyle='-')

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Dwell time of urn (' +str(attention_type) +')')
plt.title('(Within-subj) Median across conditions  (' +str(attention_type) +') ' + str(['with', 'without'][censure]) + ' censored partic')
plt.grid(True)
plt.show()


if by_ind == 1: 
    for i in range(1, data['number'].nunique()+1):
        ASPS_att_ind = ASPS.loc[ASPS['number'] == i, ['prob_option_A', 'dwell_time']] 
        ASPS_att_ind = ASPS_att_ind.sort_values(by=['prob_option_A'])
        ACPC_att_ind = ACPC.loc[ACPC['number'] == i, ['prob_option_A', 'dwell_time']] 
        ACPC_att_ind = ACPC_att_ind.sort_values(by=['prob_option_A'])
        ASPC_att_ind = ASPC.loc[ASPC['number'] == i, ['prob_option_A', 'dwell_time']]
        ASPC_att_ind = ASPC_att_ind.sort_values(by=['prob_option_A'])
        ACPS_att_ind = ACPS.loc[ACPS['number'] == i, ['prob_option_A', 'dwell_time']] 
        ACPS_att_ind = ACPS_att_ind.sort_values(by=['prob_option_A'])
        
        plt.plot(ASPS_att_ind['prob_option_A'], ASPS_att_ind['dwell_time'], label='ASPS', color='blue', marker='o', linestyle='-')
        plt.plot(ACPS_att_ind['prob_option_A'], ACPS_att_ind['dwell_time'], label='ACPS', color='red', marker='o', linestyle='-')
        plt.plot(ASPC_att_ind['prob_option_A'], ASPC_att_ind['dwell_time'], label='ASPC', color='orange', marker='o', linestyle='-')
        plt.plot(ACPC_att_ind['prob_option_A'], ACPC_att_ind['dwell_time'], label='ACPC', color='green', marker='o', linestyle='-')
        plt.title('Individual (' +str(attention_type) +') ' + str(i))
        plt.grid(True)
        plt.legend()
        plt.show()
else: 
    pass

# Plot Attention (BETWEEN-SUBJECT)

average_attention_ASPS_between = ASPS_between.groupby('prob_option_A')['dwell_time'].median()
average_attention_ACPC_between = ACPC_between.groupby('prob_option_A')['dwell_time'].median()
average_attention_ACPS_between = ACPS_between.groupby('prob_option_A')['dwell_time'].median()
average_attention_ASPC_between = ASPC_between.groupby('prob_option_A')['dwell_time'].median()

all_attention_between = pd.concat([average_attention_ASPS_between, average_attention_ACPC_between, 
                                   average_attention_ACPS_between, average_attention_ASPC_between])
all_attention_between = all_attention_between.groupby('prob_option_A').median()

plt.plot(average_attention_ASPS_between.index, average_attention_ASPS_between, label='ASPS', color='blue', marker='o', linestyle='-')
plt.plot(average_attention_ACPS_between.index, average_attention_ACPS_between, label='ACPS', color='red', marker='o', linestyle='-')
plt.plot(average_attention_ASPC_between.index, average_attention_ASPC_between, label='ASPC', color='orange', marker='o', linestyle='-')
plt.plot(average_attention_ACPC_between.index, average_attention_ACPC_between, label='ACPC', color='green', marker='o', linestyle='-')

x_fit = np.linspace(0, 1, num = 10)

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Dwell time of urn (' +str(attention_type) +')')
plt.title('(Between-subj) Median Attentional processes (' +str(attention_type) +') ' + str(['with', 'without'][censure]) + ' censored partic')
plt.grid(True)
plt.legend()

plt.show()


plt.plot(all_attention_between.index, all_attention_between, marker='o', linestyle='-')

plt.xlabel('Probability P of Non-Zero Payment')
plt.ylabel('Dwell time of urn (' +str(attention_type) +')')
plt.title('(Between-subj) Median across conditions (' +str(attention_type) +') ' + str(['with', 'without'][censure]) + ' censored partic')
plt.grid(True)
plt.show()



# Plot ATTENTION VS VALUATION (between)

all_attention_between_pred = pd.concat([ASPS_between, ACPC_between, ACPS_between, ASPC_between])

# color for ind (dwell time)
individuals = all_attention_between_pred['number'].unique()
colors = cm.rainbow(np.linspace(0, 1, len(individuals)))
individual_color_map = dict(zip(individuals, colors))
all_attention_between_pred['color'] = all_attention_between_pred['number'].map(individual_color_map)

plt.scatter(all_attention_between_pred['dwell_time'], all_attention_between_pred['valuation'], c=all_attention_between_pred['color'])

plt.xlabel('Dwell time')
plt.ylabel('Valuation')
plt.title('(Between-subj) ALL Valuation x Dwell time (' +str(attention_type) +') ' + str(['with', 'without'][censure]) + ' censored partic')
plt.grid(True)
plt.show()

# color for ind (valuation)
individuals = all_attention_between_pred['number'].unique()
colors = cm.rainbow(np.linspace(0, 1, len(individuals)))
individual_color_map = dict(zip(individuals, colors))
all_attention_between_pred['color'] = all_attention_between_pred['number'].map(individual_color_map)

plt.scatter(all_attention_between_pred['valuation'], all_attention_between_pred['dwell_time'], c=all_attention_between_pred['color'])

plt.xlabel('Valuation')
plt.ylabel('Dwell time')
plt.title('(Between-subj) ALL Valuation x Dwell time (' +str(attention_type) +') ' + str(['with', 'without'][censure]) + ' censored partic')
plt.grid(True)
plt.show()

# color for case
individuals = all_attention_between_pred['case'].unique()
colors = cm.rainbow(np.linspace(0, 1, len(individuals)))
individual_color_map = dict(zip(individuals, colors))
all_attention_between_pred['color'] = all_attention_between_pred['case'].map(individual_color_map)

plt.scatter(all_attention_between_pred['valuation'], all_attention_between_pred['dwell_time'], c=all_attention_between_pred['color'])

plt.xlabel('Valuation')
plt.ylabel('Dwell time')
plt.title('(Between-subj) ALL Valuation x Dwell time (' +str(attention_type) +') ' + str(['with', 'without'][censure]) + ' censored partic')
plt.grid(True) # ADD WHICH COLOR IS WHICH CASE 
plt.show()

# Across conditions

# for i in [ASPS_between, ACPC_between, ACPS_between, ASPC_between]:
    
#     attention_pred = i
#     individuals = attention_pred['number'].unique()
#     colors = cm.rainbow(np.linspace(0, 1, len(individuals)))
#     individual_color_map = dict(zip(individuals, colors))
#     attention_pred['color'] = attention_pred['number'].map(individual_color_map)

#     plt.scatter(attention_pred['dwell_time'], attention_pred['valuation'], c=attention_pred['color'])

#     plt.xlabel('Dwell time')
#     plt.ylabel('Valuation')
#     plt.title('(Between-subj) ' +str(attention_pred['case'].iloc[0][:4])+ ' Valuation x Dwell time (' +str(attention_type) +') ' + str(['with', 'without'][censure]) + ' censored partic')
#     plt.grid(True)
#     plt.show() 


# for i in [ASPS_between, ACPC_between, ACPS_between, ASPC_between]:
    
#     attention_pred = i
#     individuals = attention_pred['number'].unique()
#     colors = cm.rainbow(np.linspace(0, 1, len(individuals)))
#     individual_color_map = dict(zip(individuals, colors))
#     attention_pred['color'] = attention_pred['number'].map(individual_color_map)

#     plt.scatter(attention_pred['valuation'], attention_pred['dwell_time'], c=attention_pred['color'])

#     plt.xlabel('Valuation')
#     plt.ylabel('Dwell time')
#     plt.title('(Between-subj) ' +str(attention_pred['case'].iloc[0][:4])+ ' Dwell time x Valuation (' +str(attention_type) +') ' + str(['with', 'without'][censure]) + ' censored partic')
#     plt.grid(True)
#     plt.show()

# %%
# =============================================================================
# ANALYSE DATA 
# =============================================================================

########### Plot ditribution of participant-specific X values 
plt.hist(data_autre['charity_calibration'], bins=20) 
plt.xlabel('Participant-specific X')
plt.ylabel('Frequency')
plt.title('Distribution of X')
plt.show()

######## EXLEY REGRESSION 

data_for_analysis = pd.concat([ASPS, ACPC, ASPC, ACPS], ignore_index=True)

data_for_analysis['valuation'] = data_for_analysis['valuation']*100 # to get percentage
data_for_analysis['interaction'] = data_for_analysis['charity'] * data_for_analysis['tradeoff'] 

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
X = data_for_analysis[['charity', 'tradeoff', 'interaction'] + list(dummy_ind.columns)]
X = pd.concat([X, data_for_analysis[control_variables]], axis=1)
X = sm.add_constant(X, has_constant='add') # add a first column full of ones to account for intercept of regression
y = data_for_analysis['valuation']

# Fit the regression model using Ordinary Least Squares
# model = sm.OLS(y, X).fit()
# model = model.get_robustcov_results(cov_type='cluster', groups=data['number'])
model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis['number']}) # cluster at individual level
print(model.summary())




# control_variables_2 = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + 
#                  ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]
# X_2 = data_for_analysis[['charity', 'tradeoff', 'interaction'] + list(dummy_ind.columns)]
# X_2 = pd.concat([X_2, data_for_analysis[control_variables_2]], axis=1)
# X_2 = sm.add_constant(X_2, has_constant='add') # add a first column full of ones to account for intercept of regression

# model_X2 = sm.OLS(y, X_2).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis['number']}) # cluster at individual level
# print(model_X2.summary())


md = smf.mixedlm("valuation ~ charity + tradeoff + interaction", data_for_analysis, groups=data_for_analysis["number"])
mdf = md.fit()
print(mdf.summary())

######## ATTENTION REGRESSION (WITHIN)

# Same process but now dwell_time as dependent variable
y_2 = data_for_analysis['dwell_time']
model_2 = sm.OLS(y_2, X).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis['number']}) # cluster at individual level


print(model_2.summary())


md_2 = smf.mixedlm("dwell_time ~ charity + tradeoff + interaction", data_for_analysis, groups=data_for_analysis["number"])
mdf_2 = md_2.fit()
print(mdf_2.summary())

######## ATTENTION REGRESSION (BETWEEN)

# Same as above but only use the first case of each individual 
data_for_analysis_between = pd.concat([ASPS_between, ACPC_between, ASPC_between, ACPS_between], ignore_index=True)

data_for_analysis_between['valuation'] = data_for_analysis_between['valuation']*100 # to get percentage
data_for_analysis_between['interaction'] = data_for_analysis_between['charity'] * data_for_analysis_between['tradeoff'] 

# Add fixed effects
dummy_prob = pd.get_dummies(data_for_analysis_between['prob_option_A'], drop_first=True, dtype=int) # Dummy variable for probabilities (+drop first to avoid multicollinearity)
data_for_analysis_between = pd.concat([data_for_analysis_between, dummy_prob], axis=1)
data_for_analysis_between = data_for_analysis_between.merge(survey, on='id', how='left')

# Create the design matrix and dependent variable
X_between = data_for_analysis_between[['charity', 'tradeoff', 'interaction'] + list(dummy_prob.columns)]
X_between = pd.concat([X_between, data_for_analysis_between[control_variables]], axis=1) # add controls
X_between = sm.add_constant(X_between, has_constant='add') # add a first column full of ones to account for intercept of regression 
y_between = data_for_analysis_between['dwell_time']

model_3 = sm.OLS(y_between, X_between).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis_between['prob_option_A']}) 

print(model_3.summary())

# no controls
X_between_2 = data_for_analysis_between[['charity', 'tradeoff', 'interaction'] + list(dummy_prob.columns)]
X_between_2 = sm.add_constant(X_between_2, has_constant='add') 
model_3_2 = sm.OLS(y_between, X_between_2).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis_between['prob_option_A']}) 
print(model_3_2.summary())

# explo controls

# control_variables_2 = [['Demog_AGE', 'Demog_Sex', 'Demog_Field', 'Demog_High_Ed_Lev'] + 
#                   ['Charity_' + str(j) for j in ['LIKE', 'TRUST', 'LIKELY', 'DONATION_DONE']]][0]
# X_between_3 = data_for_analysis_between[['charity', 'tradeoff', 'interaction'] + list(dummy_prob.columns)]
# X_between_3 = pd.concat([X_between_3, data_for_analysis_between[control_variables_2]], axis=1) # add controls
# X_between_3 = sm.add_constant(X_between_3, has_constant='add') 
# model_3_3 = sm.OLS(y_between, X_between_3).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis_between['number']}) # cluster at individual level
# print(model_3_3.summary())

md_3 = smf.mixedlm("dwell_time ~ charity + tradeoff + interaction", data_for_analysis_between, groups=data_for_analysis_between["prob_option_A"])
mdf_3 = md_3.fit()
print(mdf_3.summary())



###### ATTENTION AS PREDICTOR
data_for_analysis_between['case_id']=[['ASPS', 'ACPC','ASPC', 'ACPS']. index(data_for_analysis_between['case'][i]) for i in range(len(data_for_analysis_between))]

# GOGOGOGO 
md_4 = smf.mixedlm("valuation ~ charity*dwell_time + tradeoff*dwell_time + interaction*dwell_time", data_for_analysis, groups=data_for_analysis["number"])
mdf_4 = md_4.fit()
print(mdf_4.summary())

md_5 = smf.mixedlm("valuation ~ charity*dwell_time + tradeoff*dwell_time + interaction*dwell_time", data_for_analysis_between, groups=data_for_analysis_between["case_id"])
mdf_5 = md_5.fit()
print(mdf_5.summary())

md_6 = smf.mixedlm("valuation ~ dwell_time*case_id", data_for_analysis_between, groups=data_for_analysis_between["case_id"])
mdf_6 = md_6.fit()
print(mdf_6.summary())

md_7 = smf.mixedlm("valuation ~ dwell_time", ASPC_between, groups=ASPC_between["prob_option_A"])
mdf_7 = md_7.fit()
print(mdf_7.summary())


# DIFFERENCES 
self_lottery = pd.concat([ASPS, ACPS], ignore_index = True)
charity_lottery = pd.concat([ACPC, ASPC], ignore_index=True)

self_lottery_differences = pd.DataFrame(columns=['number', 'prob_option_A', 'valuation_ASPS_ACPS', 'dwell_time_ASPS_ACPS'])

for i in range(1, self_lottery['number'].nunique()+1):
    individual = self_lottery.loc[self_lottery['number'] == i, ['case', 'prob_option_A', 'valuation', 'dwell_time']] 
    individual_difference = individual.pivot(index='prob_option_A', columns='case', values=['valuation', 'dwell_time'])
    individual_difference['valuation_ASPS_ACPS'] = individual_difference['valuation']['ASPS'] - individual_difference['valuation']['ACPS']
    individual_difference['dwell_time_ASPS_ACPS'] = individual_difference['dwell_time']['ASPS'] - individual_difference['dwell_time']['ACPS']
    individual_difference['number'] = i
    individual_difference.reset_index(inplace=True)
    individual_difference.columns = individual_difference.columns.droplevel(1)
    self_lottery_differences = pd.concat([self_lottery_differences, individual_difference[['number', 'prob_option_A', 'valuation_ASPS_ACPS', 'dwell_time_ASPS_ACPS']]], ignore_index=True)

charity_lottery_differences = pd.DataFrame(columns=['number', 'prob_option_A', 'valuation_ACPC_ASPC', 'dwell_time_ACPC_ASPC'])

for i in range(1, charity_lottery['number'].nunique()+1):
    individual = charity_lottery.loc[charity_lottery['number'] == i, ['case', 'prob_option_A', 'valuation', 'dwell_time']] 
    individual_difference = individual.pivot(index='prob_option_A', columns='case', values=['valuation', 'dwell_time'])
    individual_difference['valuation_ACPC_ASPC'] = individual_difference['valuation']['ACPC'] - individual_difference['valuation']['ASPC']
    individual_difference['dwell_time_ACPC_ASPC'] = individual_difference['dwell_time']['ACPC'] - individual_difference['dwell_time']['ASPC']
    individual_difference['number'] = i
    individual_difference.reset_index(inplace=True)
    individual_difference.columns = individual_difference.columns.droplevel(1)
    charity_lottery_differences = pd.concat([charity_lottery_differences, individual_difference[['number', 'prob_option_A', 'valuation_ACPC_ASPC', 'dwell_time_ACPC_ASPC']]], ignore_index=True)



individuals = self_lottery_differences['number'].unique()
colors = cm.rainbow(np.linspace(0, 1, len(individuals)))
individual_color_map = dict(zip(individuals, colors))
self_lottery_differences['color'] = self_lottery_differences['number'].map(individual_color_map)

plt.scatter(self_lottery_differences['valuation_ASPS_ACPS'], self_lottery_differences['dwell_time_ASPS_ACPS'], c=self_lottery_differences['color'])

plt.xlabel('Valuation')
plt.ylabel('Dwell time')
plt.title('(Within-subj) Diff SELF Valuation x Dwell time (' +str(attention_type) +')')
plt.grid(True)
plt.show()


individuals = charity_lottery_differences['number'].unique()
colors = cm.rainbow(np.linspace(0, 1, len(individuals)))
individual_color_map = dict(zip(individuals, colors))
charity_lottery_differences['color'] = charity_lottery_differences['number'].map(individual_color_map)

plt.scatter(charity_lottery_differences['valuation_ACPC_ASPC'], charity_lottery_differences['dwell_time_ACPC_ASPC'], c=charity_lottery_differences['color'])

plt.xlabel('Valuation')
plt.ylabel('Dwell time')
plt.title('(Within-subj) Diff CHARITY Valuation x Dwell time (' +str(attention_type) +')')
plt.grid(True)
plt.show()

md_self = smf.mixedlm("valuation_ASPS_ACPS ~ dwell_time_ASPS_ACPS", self_lottery_differences, groups=self_lottery_differences["prob_option_A"])
mdf_self = md_self.fit()
print(mdf_self.summary())

md_charity = smf.mixedlm("valuation_ACPC_ASPC ~ dwell_time_ACPC_ASPC", charity_lottery_differences, groups=charity_lottery_differences["prob_option_A"])
mdf_charity = md_charity.fit()
print(mdf_charity.summary())

# data_for_analysis_between_attention = data_for_analysis_between

# data_for_analysis_between_attention['charity_atten'] = data_for_analysis_between_attention['charity'] * data_for_analysis_between_attention['dwell_time']
# data_for_analysis_between_attention['tradeoff_atten'] = data_for_analysis_between_attention['tradeoff'] * data_for_analysis_between_attention['dwell_time']
# data_for_analysis_between_attention['interaction_atten'] = data_for_analysis_between_attention['interaction'] * data_for_analysis_between_attention['dwell_time']

# X_between_pred = data_for_analysis_between[['charity_atten', 'tradeoff_atten', 'interaction_atten'] + list(dummy_ind.columns) + list(dummy_prob.columns)]
# X_between_pred = sm.add_constant(X_between_pred) # add a first column full of ones to account for intercept of regression
# y_between_pred = data_for_analysis_between['valuation']

# model_4 = sm.OLS(y_between_pred, X_between_pred).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis_between['number']}) # cluster at individual level

# print(model_4.summary())



# plot dwell time 

for i in [ASPS_between, ACPC_between, ACPS_between, ASPC_between]:
    attention_case = i
    bins = np.linspace(0, 50, 50)
    plt.hist([attention_case['dwell_time_absolute'], attention_case['dwell_time_relative'], attention_case['total_time_spent_s']], bins, label = ['absolute', 'relative', 'total'])
    plt.legend()
    plt.title('(Between-subj) ' +str(attention_case['case'].iloc[0][:4]))
    plt.show()


#### EXPLORATOIRE??? 
# data_for_analysis_between['interaction_atten'] = data_for_analysis_between['dwell_time'] * data_for_analysis['frequency'] 
# X_between_pred = data_for_analysis_between[['dwell_time', 'frequency', 'interaction_atten'] + list(dummy_ind.columns) + list(dummy_prob.columns)]

# data_for_analysis['interaction_atten'] = data_for_analysis['dwell_time'] * data_for_analysis['frequency'] 

# X_2 = data_for_analysis[['dwell_time', 'frequency', 'interaction_atten', 'prob_option_A', 'number']]

# X_2 = sm.add_constant(X_2) # add a first column full of ones to account for intercept of regression


# X_2['prob_option_A'] = X_2['prob_option_A'].astype(float) # so that everything is float for regression model
# X_2['number'] = X_2['number'].astype(float) # so that everything is float for regression model


# dummy_cases = pd.get_dummies(data_for_analysis_between['case'], dtype=int)      
# dummy_cases.drop(columns=['ASPS'], inplace=True) # force ASPS to be reference 
# data_for_analysis_between = pd.concat([data_for_analysis_between, dummy_cases], axis=1)

# X_between_pred = data_for_analysis_between[['dwell_time'] + list(dummy_cases.columns) + list(dummy_ind.columns) + list(dummy_prob.columns)]
# X_between_pred = sm.add_constant(X_between_pred) # add a first column full of ones to account for intercept of regression
# y_between_pred = data_for_analysis_between['valuation']

# model_4 = sm.OLS(y_between_pred, X_between_pred).fit(cov_type='cluster', cov_kwds={'groups': data_for_analysis_between['number']}) # cluster at individual level

# print(model_4.summary())



