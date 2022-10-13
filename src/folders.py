"""
:Author: Balazs Szigeti <szb37@pm.me>
:Copyright: 2020, DrugNerdsLab
:License: MIT
"""

import os


""" Folders """
src  = os.path.dirname(os.path.abspath(__file__))
codebase = os.path.abspath(os.path.join(src, os.pardir))
projectroot = os.path.abspath(os.path.join(codebase, os.pardir))

test = os.path.abspath(os.path.join(codebase, 'tests'))
fixtures = os.path.abspath(os.path.join(test, 'fixtures'))
tmp_dir = os.path.join(fixtures, 'tmp')

csvs = os.path.abspath(os.path.join(projectroot, 'csvs'))
csvs_raw = os.path.abspath(os.path.join(csvs, '01_raw'))
csvs_preprocessed = os.path.abspath(os.path.join(csvs, '02_preprocessed'))
csvs_trial_data = os.path.abspath(os.path.join(csvs, '03_trial_data'))
csvs_trial_stats = os.path.abspath(os.path.join(csvs, '04_trial_stats'))
csvs_cgrc_data = os.path.abspath(os.path.join(csvs, '05_cgrc_data'))
csvs_cgrc_stats = os.path.abspath(os.path.join(csvs, '06_cgrc_stats'))
csvs_summary_tables = os.path.abspath(os.path.join(csvs, '07_summary_tables'))

plot_export = os.path.abspath(os.path.join(projectroot, 'exported_figures'))
strata_plots = os.path.abspath(os.path.join(plot_export, 'stratification'))
cgrc_plots = os.path.abspath(os.path.join(plot_export, 'CGR_curves'))
