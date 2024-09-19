#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:12:34 2024

@author: carolinepioger
"""

### AJOUTER UN SORTE DE CLEAR ALL 

import pandas as pd


# Info to find data
# path = '/Users/carolinepioger/Desktop/ALL collection' # change to yours :)
path = '/Users/carolinepioger/Desktop/STAGE SACHA/pretests' # change to yours :)
# dates = ['2024-04-29','2024-04-30','2024-05-02', '2024-05-14', '2024-05-15', '2024-07-12', '2024-07-25']
dates = ['2024-03-10', '2024-03-11', '2024-03-22', '2024-03-23', '2024-03-25', '2024-03-28', '2024-04-03', ]

assoc_sum = pd.concat([pd.read_csv(path + '/EXLEY_ASSO_' + date + '.csv') for date in dates], ignore_index=True)
assoc_sum = assoc_sum.drop(assoc_sum[assoc_sum['participant._current_page_name'] != 'prolific'].index)

outcome_data = pd.concat([pd.read_csv(path + '/EXLEY_RESULTAT_' + date + '.csv') for date in dates], ignore_index=True)
outcome_data = outcome_data.drop(outcome_data[outcome_data['participant._current_page_name'] != 'prolific'].index)

assoc = pd.DataFrame()
assoc['id'] = assoc_sum['participant.code'] 
assoc['charity choice'] = assoc_sum['player.association_choice']


columns_mapping = {
    'participant.code': 'id',  # number id associated to each participant
    'session.code': 'session', # number of session
    'player.PROLIFIC_ID': 'prolific id', # prolific id
    'player.ENDOWMENT': 'partic fee', # participation fee
    'player.TOTAL_PAIEMENT_SELF': 'total self', # total outcome for self
    'player.TOTAL_PAIEMENT_CHARITY': 'total charity' # total outcome for charity
}


outcome_data = outcome_data.rename(columns=columns_mapping)[list(columns_mapping.values())]
outcome_data = outcome_data.reset_index(drop=True)

self_reward = [outcome_data['total self'][i] - outcome_data['partic fee'][i] for i in range(len(outcome_data))]
outcome_data.insert(outcome_data.columns.get_loc('total self') + 1, 'self reward', self_reward)

outcome_data['total spent'] = [outcome_data['total self'][i] + outcome_data['total charity'][i] for i in range(len(outcome_data))]

outcome_data = outcome_data.merge(assoc, on='id', how='left')

outcome_data = outcome_data.drop_duplicates()

# Save the concatenated dataset
data_path = path + '/resultats_overview.csv'
outcome_data.to_csv(data_path, index=False)

data_path

# Data analysis 

print('Average bonus:')
print(outcome_data['self reward'].mean())
print()
print('Average donations:')
print(outcome_data['total charity'].mean())
print()
print('Charity choice:')
print(outcome_data['charity choice'].value_counts())
print()
print('Charity donations recap:')
print(outcome_data.groupby('charity choice')['total charity'].sum())


