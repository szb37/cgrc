"""
:Author: Balazs Szigeti <szb37@pm.me>
:Copyright: 2020, DrugNerdsLab
:License: MIT
"""

import src.toy_models.core as toy_models
import src.constants as constants
import src.cgrc.core as cgrc
import time

start = time.time()

""" Reproduce maunscript figures / stats"""
if False: # all_models_cgrA (reproduces model summary tables [Table 1 and Supp Table 1])
    toy_models.Controllers.run_cgrc_model_family(
        model_family_name = 'all_models',
        postfix = 'cgrA120',
        cgrc_param_set = 3,
        save_figs = False,
        n_trials = 50,
        n_patients = 120,
    )

if True: # def_models_cgrC (reproduces model CGRCs [Figure 3])
    toy_models.Controllers.run_cgrc_model_family(
        model_family_name = 'def_models',
        postfix = 'cgrC_test_new_CIplots',
        cgrc_param_set = 7, # replace wth 7
        save_figs = True,
        n_trials = 8,
        n_patients = 100,
    )

if False: # sbmd_cgrA (reproduces sbmd results table [Table 2])
    cgrc.Controllers.run_cgrc_trial(
        trial_name = 'sbmd',
        postfix = 'cgrA',
        study_scales = constants.sbmd_all,
        cgrc_param_set = 3,
        save_figs = False,
    )

if False: # sbmd_cgrC (reproduces sbmd CGRCs [Figure 4])
    cgrc.Controllers.run_cgrc_trial(
        trial_name = 'sbmd',
        postfix = 'cgrC',
        study_scales = constants.sbmd_plots,
        cgrc_param_set = 7,
        save_figs = True,
    )


""" Miscs """
if False: # test pipeline
    toy_models.Controllers.run_cgrc_model_family(
        model_family_name = 'test_models',
        postfix = 'tmp',
        cgrc_param_set = 4,
        save_figs = False,
        n_trials = 3,
        n_patients = 50,
    )

if False: # DONE sbmd_cgrC (reproduces sbmd CGRCs [Figure 4])
    cgrc.Controllers.run_cgrc_trial(
        trial_name = 'sbmd',
        postfix = 'cgrC_plots',
        study_scales = constants.sbmd_plots,
        cgrc_param_set = 7,
        save_figs = True,
    )

end = time.time()
print('\nExecution time was {} seconds (~ {} mins)'.format(
    round((end-start),1),
    round((end-start)/60,1),
))
