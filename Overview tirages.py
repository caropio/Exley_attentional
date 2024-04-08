#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:12:34 2024

@author: carolinepioger
"""

### AJOUTER UN SORTE DE CLEAR ALL 

import pandas as pd


# Info to find data
path = '/Users/carolinepioger/Desktop' # change to yours :)
file = '/pretest vincent' # to adapt
date = '2024-03-28' # download date of first collection
date_2 = '2024-03-25' # download date of second collection
date_3 = '2024-04-03' # download date of third collection

collect_1 = pd.read_csv(path + file + '/EXLEY_RESULTAT_' + date + '.csv')
collect_2 = pd.read_csv(path + file + '/EXLEY_RESULTAT_' + date_2 + '.csv')
collect_3 = pd.read_csv(path + file + '/EXLEY_RESULTAT_' + date_3 + '.csv')

assoc_1 = pd.read_csv(path + file + '/EXLEY_ASSO_' + date + '.csv')
assoc_2 = pd.read_csv(path + file + '/EXLEY_ASSO_' + date_2 + '.csv')
assoc_3 = pd.read_csv(path + file + '/EXLEY_ASSO_' + date_3 + '.csv')

assoc_sum = pd.concat([assoc_1, assoc_2, assoc_3], ignore_index=True)
assoc_sum = assoc_sum.drop(assoc_sum[assoc_sum['participant._current_page_name'] != 'prolific'].index)

assoc = pd.DataFrame()
assoc['id'] = assoc_sum['participant.code'] 
assoc['charity choice'] = assoc_sum['player.association_choice']

# Concatenate the datasets (each case of part 3)
outcome_data = pd.concat([collect_1, collect_2, collect_3], ignore_index=True)
# outcome_data = collect_2
outcome_data = outcome_data.drop(outcome_data[outcome_data['participant._current_page_name'] != 'prolific'].index)

# Remove blank rows

blank = outcome_data.loc[pd.isna(outcome_data['player.PROLIFIC_ID']), 'player.PROLIFIC_ID'] 
outcome_data = outcome_data.drop(outcome_data[outcome_data['player.PROLIFIC_ID'].isin(blank) == True].index)
outcome_data = outcome_data.reset_index()


columns_to_keep = ['participant.code', # number id associated to each participant
                   'session.code', # number of session
                   'player.PROLIFIC_ID', # prolific id
                   'player.ENDOWMENT', # participation fee
                   'player.TOTAL_PAIEMENT_SELF',  # total outcome for self
                   'player.TOTAL_PAIEMENT_CHARITY', # total outcome for charity
                   ]


outcome_data = outcome_data[columns_to_keep]

outcome_data = outcome_data.rename(columns={'participant.code':'id', # number id associated to each participant
                                            'session.code' : 'session', # number of session
                                            'player.PROLIFIC_ID' : 'prolific id', # prolific id
                                            'player.ENDOWMENT' : 'partic fee',  # participation fee
                                            'player.TOTAL_PAIEMENT_SELF' : 'total self',  # total outcome for self
                                            'player.TOTAL_PAIEMENT_CHARITY' : 'total charity', # total outcome for charity
                                                  })

self_reward = [outcome_data['total self'][i]-outcome_data['partic fee'][i] for i in range(len(outcome_data))]
outcome_data.insert(outcome_data.columns.get_loc('total self') + 1, 'self reward', self_reward)

outcome_data['total spent'] = [outcome_data['total self'][i] + outcome_data['total charity'][i] for i in range(len(outcome_data))]

outcome_data = outcome_data.merge(assoc, on='id', how='left')

# Save the concatenated dataset
data_path = path + file + '/resultats_overview.csv'
outcome_data.to_csv(data_path, index=False)

data_path



