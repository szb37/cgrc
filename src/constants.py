"""
:Author: Balazs Szigeti <szb37@pm.me>
:Copyright: 2020, DrugNerdsLab
:License: MIT
"""

import numpy as np

estimator = 'mean'


''' Define CGRC parameter sets '''
cgrc_parameters = {
    0:{
        'cgr_values': [0.5],
        'n_cgrc_trials': 1,},
    1:{
        'cgr_values': [0.5],
        'n_cgrc_trials': 8,},
    2:{
        'cgr_values': [0.5],
        'n_cgrc_trials': 64,},
    3:{
        'cgr_values': [0.5],
        'n_cgrc_trials': 96,},
    4:{
        'cgr_values': np.linspace(0, 1, 7).tolist(),
        'n_cgrc_trials': 8,},
    5:{
        'cgr_values': np.linspace(0, 1, 9).tolist(),
        'n_cgrc_trials': 96,},
    6:{
        'cgr_values': np.linspace(0, 1, 13).tolist(), #replace with 15
        'n_cgrc_trials': 32,}, #replace with 64
    7:{
        'cgr_values': np.linspace(0, 1, 15).tolist(),
        'n_cgrc_trials': 96,},
    8:{
        'cgr_values': [0.5],
        'n_cgrc_trials': 8192,},
    9:{
        'cgr_values': np.linspace(0, 1, 11).tolist(),
        'n_cgrc_trials': 64,},
}


''' Define MD scales '''
sbmd_all = {'sbmd': ['PANAS', 'mood', 'creativity', 'WEMWB', 'SCS', 'CPS', 'energy', 'focus', 'temper', 'QIDS', 'STAIT' ]}
sbmd_acutes = {'sbmd': ['CPS', 'PANAS', 'mood', 'creativity', 'energy', 'focus', 'temper']}
sbmd_postacutes = {'sbmd': ['QIDS', 'WEMWB', 'STAIT']}
sbmd_plots = {'sbmd': ['PANAS', 'mood', 'creativity', 'energy', 'CPS', 'WEMWB']}
sbmd_test = {'sbmd': ['PANAS']}
sbmd_tmp = {'sbmd': ['creativity']}


''' Define CGRS '''
trial_cgrs={
    'sbmd':0.72,
    # AD studies with self guess
    'tads':0.64,
    'stoppd':0.55,
    'ruppats':0.67,
    # AD studies wo self guess
    'rtca':0.89,
    'cams':0.59,
    'all':0.64, # based on ADs w self guess
    'mock': 0.7,
}


guessers=['self']
respondents=['self']
