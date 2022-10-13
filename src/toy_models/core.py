"""
:Author: Balazs Szigeti <szb37@pm.me>
:Copyright: 2020, DrugNerdsLab
:License: MIT

Model family analyis naming convention:
analysis_name: {model_family}_cgrc{cgrc_param_set}
subanalysis_name: {model_family}_cgrc{cgrc_param_set}_{model_name}_trial{trial_id}
"""

from tqdm.contrib.itertools import product as tqdmproduct
import src.toy_models.model_defs as model_defs
from statistics import mean, median
import src.my_dataframes as mydfs
import src.constants as constants
import src.folders as folders
import src.figures as figures
import src.cgrc.core as cgrc
import src.miscs as miscs
import pandas as pd
import itertools
import random
import os


class Controllers():

    @staticmethod
    def run_cgrc_model_family(model_family_name, cgrc_param_set, postfix, n_patients, n_trials, save_figs=True):

        cgrc_parameters = constants.cgrc_parameters[cgrc_param_set]
        models = eval('model_defs.{}'.format(model_family_name))
        analysis_name = model_family_name + '_{}'.format(postfix)

        trial_data_dir, trial_stats_dir, cgrc_data_dir, cgrc_stats_dir, cgrc_plots_dir = miscs.create_analysis_dirs(
            analysis_name,
            trial_data_subdir = True,
            incl_cgrc_plots_dir = save_figs,
        )

        for model, trial_id in tqdmproduct(models, range(n_trials), desc='Model family CGRC analysis'):


            model_name = list(model.keys())[0]
            model_specs = model[model_name]
            add_columns = {'model_name':model_name, 'trial_id':trial_id}
            subanalysis_name = Helpers.get_subanalysis_name(
                analysis_name = analysis_name,
                model_name = model_name,
                trial_id = trial_id
            )

            # Generate pseudodata according to model specifications
            ToyModelsDataGenerator.get_pseudodata(
                output_dir = trial_data_dir,
                output_prefix = subanalysis_name,
                model_specs = model_specs,
                n_datapoints = n_patients,
                model_name = model_name,
                trial_id = trial_id
            )

            # Get trial stats for psuedodata
            cgrc.Controllers.get_trial_stats(
                input_dir = trial_data_dir,
                input_fname = subanalysis_name + '__trial_data.csv',
                output_dir = trial_stats_dir,
                output_prefix = subanalysis_name,
                add_columns = add_columns,
            )

            # Get CGRC from psuedodata
            cgrc.Controllers.get_cgrc_data(
                input_dir = trial_data_dir,
                input_fname = subanalysis_name + '__trial_data.csv',
                output_dir = cgrc_data_dir,
                output_prefix = subanalysis_name,
                cgrc_parameters = cgrc_parameters,
                add_columns = add_columns,
            )

            # Get psudeodata CGRC stats
            cgrc.Controllers.get_cgrc_stats(
                input_dir = cgrc_data_dir,
                input_fname = subanalysis_name + '__cgrc_data.csv',
                output_dir = cgrc_stats_dir,
                output_prefix = subanalysis_name,
                add_columns = add_columns,
            )

            # Make CGRC figues
            if save_figs:
                figures.Controllers.plot_VScgr_twinx(
                    input_dir = cgrc_stats_dir,
                    input_fname = subanalysis_name + '__cgrc_model_components.csv',
                    output_dir = cgrc_plots_dir,
                    output_prefix = subanalysis_name,
                )

        # Get summary table
        summary_df = ToyModelsAnalyis.get_model_family_summary(
            model_family_name = model_family_name,
            analysis_name = analysis_name,
            cgrc_param_set = cgrc_param_set,
        )
        print('\n', summary_df.to_string(index=False))


class ToyModelsAnalyis():
    """ Analyze family of toy models """

    @staticmethod
    def get_model_family_summary(analysis_name, model_family_name, cgrc_param_set):
        """ Construct summary table for moel family """

        assert isinstance(analysis_name, str)
        assert isinstance(model_family_name, str)
        assert isinstance(cgrc_param_set, int)

        df = mydfs.ModelFamilyResultsDf()

        trial_data = Helpers.get_concateneted_df_type(
            target_dir = os.path.join(folders.csvs_trial_data, analysis_name),
            df_type = '__trial_data')

        # Get n_patients and n_trials assuming each member of the model family has same n_patient and n_trial
        tmp_mid = trial_data.model_name.to_list()[0]
        tmp_tid = trial_data.trial_id.to_list()[0]
        n_patients = trial_data.loc[(trial_data.model_name==tmp_mid) & (trial_data.trial_id==tmp_tid)].shape[0]
        n_trials = len(trial_data.trial_id.unique().tolist())

        # Get unadjusted model components
        unadj_model_components = Helpers.get_concateneted_df_type(
            target_dir = os.path.join(folders.csvs_trial_stats, analysis_name),
            df_type = '__model_components')

        # Get adjusted model components
        cgradj_model_components = Helpers.get_concateneted_df_type(
            target_dir = os.path.join(folders.csvs_cgrc_stats, analysis_name),
            df_type = '__cgrc_model_components')

        cgradj_model_components = cgradj_model_components.loc[(cgradj_model_components.cgr==0.5)]
        assert cgradj_model_components.shape[0]>0
        cgradj_n_trials = len(cgradj_model_components.trial_id.unique().tolist())
        assert n_trials==cgradj_n_trials
        cgradj_model_names=cgradj_model_components.model_name.unique().tolist()
        unadj_model_names=unadj_model_components.model_name.unique().tolist()
        assert cgradj_model_names==unadj_model_names

        for model_name in unadj_model_components.model_name.unique().tolist():

            filtered_unadj_model_comps = unadj_model_components.loc[
                (unadj_model_components.model_name == model_name) &
                (unadj_model_components.model_type == 'without_guess') &
                (unadj_model_components.component == 'conditionAC')
            ]

            trial_model_data = trial_data.loc[(trial_data.model_name==model_name)]
            cgr = round(trial_model_data.loc[(trial_model_data.condition==trial_model_data.guess)].shape[0]/(n_trials*n_patients), 3)

            row={}
            row['model'] = model_name
            row['n_trials'] = n_trials
            row['n_patients'] = n_patients
            row['cgrc_param_set'] = cgrc_param_set
            row['cgr'] = cgr
            row['avg_trt_p'] = round(miscs.get_estimate(filtered_unadj_model_comps.p.tolist()), 3)
            row['sig_trt_rate'] = round(sum([el <=0.05 for el in filtered_unadj_model_comps.p.tolist()])/n_trials, 3)
            row['avg_trt_es'] = round(miscs.get_estimate(filtered_unadj_model_comps.est.tolist()), 3)

            filtered_cgradj_model_comps = cgradj_model_components.loc[
                (cgradj_model_components.model_name == model_name) &
                (cgradj_model_components.model_type == 'without_guess') &
                (cgradj_model_components.component == 'conditionAC')
            ]

            # Calculate average p/es aross n_cgr_trials (avg corresponds to p/es of single trial)
            trial_ids = filtered_cgradj_model_comps.trial_id.unique().tolist()
            trial_ps=[]
            trial_efs=[]
            for trial_id in trial_ids:
                tmp = filtered_cgradj_model_comps.loc[filtered_cgradj_model_comps.trial_id==trial_id]
                assert tmp.shape[0] == constants.cgrc_parameters[cgrc_param_set]['n_cgrc_trials']
                trial_ps.append(mean(tmp.p.tolist()))
                trial_efs.append(mean(tmp.est.tolist()))

            row['cgradj_avg_trt_p'] = round(miscs.get_estimate(trial_ps), 3)
            row['cgradj_sig_trt_rate'] = round(sum([p <=0.05 for p in trial_ps])/n_trials, 3)
            row['cgradj_avg_trt_es'] = round(miscs.get_estimate(trial_efs), 3)


            df = df.append(row, ignore_index=True)

        df.__class__= mydfs.ModelFamilyResultsDf
        df.set_column_types()
        df.to_csv(os.path.join(folders.csvs_summary_tables, analysis_name+'_{}__summary_table.csv'.format(constants.estimator)), index=False)

        return df


class ToyModelsDataGenerator():
    """ Function to get mock data """

    @staticmethod
    def get_pseudodata(output_dir, output_prefix, model_specs, n_datapoints, model_name=None, trial_id=None, model_type='mum', min_strata_size=4, round_digits=0):
        """ Get pseudo-data

            Args:
                n_datapoints (int): number of datapoints in each trial
                model_specs (dict): model specifictions
                model_type (str, optional): must be 'mum' or 'bum', corresponding to bengin and malicious unblinding model
                min_strata_size (int, optional): the minimum sample size of each strata
                round_digits (int, optional): generated scores rounded to
        """

        assert isinstance(n_datapoints, int)
        assert Helpers.is_valid_model_specs(model_specs)
        assert model_type in ['mum', 'bum']
        assert (min_strata_size is None) or (isinstance( min_strata_size, int))
        assert isinstance(round_digits, int)

        df = Helpers.get_TrialDatadDf(n_datapoints)

        if min_strata_size is not None:
            df = Helpers.enforce_min_strata_size(df, min_strata_size)

        for idx, row in df.iterrows():

            condition, guess, score = ToyModelsDataGenerator.get_pseudodatapoint(
                model_type = model_type,
                oc_nh = model_specs['oc_nh'],
                gs_nh = model_specs['gs_nh'],
                se = model_specs['se'],
                dte = model_specs['dte'],
                pte = model_specs['pte'],
                ate = model_specs['ate'],
                oc2gs = model_specs['oc2gs'],
                forced_condition = row.condition,
                forced_guess = row.guess,
            )

            row.condition = condition
            row.guess = guess
            row.score = round(score, round_digits)
            row.delta_score = round(score, round_digits)

        # post-processing
        if model_name is not None:
            df['model_name'] = model_name

        if trial_id is not None:
            df['trial_id'] = trial_id

        df.__class__= mydfs.TrialDataDf
        df.set_column_types()
        df.to_csv(os.path.join(output_dir, output_prefix+'__trial_data.csv'), index=False)

        return df

    @staticmethod
    def get_pseudodatapoint(model_type, oc_nh, gs_nh, se, dte, pte, ate, oc2gs, forced_condition=None, forced_guess=None):

        if model_type=='mum':
            condition, guess, score = ToyModelsDataGenerator.get_mum_pseudodatapoint(
                oc_nh, gs_nh, se, dte, pte, ate, oc2gs, forced_condition, forced_guess)
        elif model_type=='bum':
            condition, guess, score = ToyModelsDataGenerator.get_bum_pseudodatapoint(
                oc_nh, gs_nh, se, dte, pte, ate, oc2gs, forced_condition, forced_guess)
        else:
            assert False

        return condition, guess, score

    @staticmethod
    def get_mum_pseudodatapoint(oc_nh, gs_nh, se, dte, pte, ate, oc2gs, forced_condition=None, forced_guess=None):
        """ Get malicious unmasking model data point """

        assert forced_guess in [None, 'PL', 'AC']
        assert forced_condition in [None, 'PL', 'AC']

        # get conditon
        if forced_condition=='PL':
            condition = 'PL'
        elif forced_condition=='AC':
            condition = 'AC'
        elif forced_condition is None:
            condition = random.choice(['PL', 'AC'])
        else:
            assert False

        # get guess; if roll>=0.5, then guess is active
        if condition=='PL':
            roll = random.gauss(gs_nh[0], gs_nh[1])
        elif condition=='AC':
            roll = random.gauss(gs_nh[0], gs_nh[1])
            roll = min(1, roll)
            roll = max(0, roll)
            roll += random.gauss(se[0], se[1])
        else:
            assert False

        roll = min(1, roll)
        roll = max(0, roll)

        if (forced_guess is None) and (roll>=0.5):
            guess='AC'
        elif (forced_guess is None) and (roll<0.5):
            guess='PL'
        elif forced_guess is not None:
            guess=forced_guess
        else:
            assert False

        # get score
        score = random.gauss(oc_nh[0], oc_nh[1])
        if condition=='AC':
            score += random.gauss(dte[0], dte[1])
        else:
            assert condition=='PL'

        if guess=='AC':
            score += random.gauss(ate[0], ate[1])
        elif guess=='PL':
            score += random.gauss(pte[0], pte[1])
        else:
            assert False

        return condition, guess, score

    @staticmethod
    def get_bum_pseudodatapoint(oc_nh, gs_nh, se, dte, pte, ate, oc2gs, forced_condition=None, forced_guess=None):
        """ Get bengin unmasking model data point """

        assert forced_guess in [None, 'PL', 'AC']
        assert forced_condition in [None, 'PL', 'AC']

        # get condition
        if forced_condition=='PL':
            condition = 'PL'
        elif forced_condition=='AC':
            condition = 'AC'
        elif forced_condition is None:
            condition = random.choice(['PL', 'AC'])
        else:
            assert False

        # get outcome
        score = random.gauss(oc_nh[0], oc_nh[1])
        if condition=='AC':
            score += random.gauss(dte[0], dte[1])

        # get guess; if roll>=0.5, then guess is active
        roll = random.gauss(gs_nh[0], gs_nh[1])
        roll = min(1, roll)
        roll = max(0, roll)

        if condition=='AC':
            roll += random.gauss(se[0], se[1])

        if score >= oc_nh[0]: # if outcome is better than natural hitory, then, Oc contributes to guess
            roll += random.gauss(oc2gs[0], oc2gs[1])

        if (forced_guess is None) and (roll>=0.5):
            guess='AC'
        elif (forced_guess is None) and (roll<0.5):
            guess='PL'
        elif forced_guess=='PL':
            guess='PL'
        elif forced_guess=='AC':
            guess='AC'
        else:
            assert False

        return condition, guess, score


class Helpers():
    """ Helper functions """

    @staticmethod
    def get_TrialDatadDf(n, equal_sample_per_condition=True):
        """ Returns TrialDataDf with aux data filled """

        df = mydfs.TrialDataDf()

        if equal_sample_per_condition:
            n_pl = round(n/2)
            n_ac = n-n_pl
            df.condition = ['PL' for i in range(n_pl)] + ['AC' for i in range(n_ac)]

        else:
            df.condition = [None for i in range(n)]

        df.subject_id = [idx for idx in range(n)]

        df.study = 'mock'
        df.scale = 'scale1'
        df.tp = 'wk8'
        df.respondent = 'self'
        df.guesser = 'self'
        df.baseline = 0
        df.score = None
        df.delta_score = None
        df.guess = None

        df.add_univalue_columns({'guesser':'self', 'respondent':'self'})

        df['baseline'] = df['baseline'].astype('object')
        df['subject_id'] = df['subject_id'].astype('object')

        return df

    @staticmethod
    def enforce_min_strata_size(df, min_strata_size):
        """ Adds datapoints to ensure that there is enough sample in each strata for downstream
            processing
        """

        condition_idx = df.columns.get_loc('condition')
        guess_idx = df.columns.get_loc('guess')

        stratas=[
            ('PL', 'PL'),
            ('AC', 'PL'),
            ('PL', 'AC'),
            ('AC', 'AC'),
        ]

        idx=0
        for strata, sample_idx in itertools.product(stratas, range(min_strata_size)):
            df.iloc[idx , condition_idx] = strata[0]
            df.iloc[idx , guess_idx] = strata[1]
            idx+=1

        return df

    @staticmethod
    def postprocess_pseudodata_df_model_family(df, model_name, trial_id):
        """ Save psuedodata """

        df['model_name'] = model_name
        df['trial_id'] = trial_id

        df.__class__= mydfs.TrialDataDf
        df.set_column_types()

        return df

    @staticmethod
    def get_subanalysis_name(analysis_name, model_name, trial_id):
        return '{}_{}_trial{}'.format(analysis_name, model_name, trial_id)

    @staticmethod
    def get_concateneted_df_type(target_dir, df_type):

        assert df_type in [
            '__trial_data',
            '__model_components',
            '__strata_contrast',
            '__cgrc_model_components',
            '__cgrc_strata_contrast']

        # Get all trial stats
        if df_type=='__model_components':
            master_df = mydfs.ModelComponentsDf()
        elif df_type=='__strata_contrast':
            master_df = mydfs.StrataContrastDf()
        elif df_type=='__cgrc_model_components':
            master_df = mydfs.ModelComponentsDf()
            master_df.add_univalue_columns({'cgr':None, 'cgr_trial_id':None})
        elif df_type=='__cgrc_strata_contrast':
            master_df = mydfs.StrataContrastDf()
            master_df.add_univalue_columns({'cgr':None, 'cgr_trial_id':None})
        elif df_type=='__trial_data':
            master_df = mydfs.TrialDataDf()
        else:
            assert False

        master_df.add_univalue_columns({'respondent':None, 'guesser':None, 'model_name':None, 'trial_id':None})

        for fpath in [fpath for fpath in os.listdir(target_dir) if (df_type in fpath)]:
            df = pd.read_csv(os.path.join(target_dir, fpath))
            master_df = pd.concat([master_df, df], sort=False)

        if df_type=='__model_components':
            master_df.__class__= mydfs.ModelComponentsDf
        elif df_type=='__strata_contrast':
            master_df.__class__= mydfs.StrataContrastDf
        elif df_type=='__cgrc_model_components':
            master_df.__class__= mydfs.ModelComponentsDf
        elif df_type=='__cgrc_strata_contrast':
            master_df.__class__= mydfs.StrataContrastDf
        elif df_type=='__trial_data':
            master_df.__class__= mydfs.TrialDataDf
        else:
            assert False

        master_df.set_column_types()
        return master_df

    @staticmethod
    def is_valid_model_specs(model):
        """ Check if dict is a vali model parameters dictionary """

        assert isinstance(model, dict)
        assert len(model.keys()) == 7
        assert all([p in model.keys() for p in ['oc_nh', 'gs_nh', 'se', 'dte', 'pte', 'ate', 'oc2gs']])

        for model_parameter in model.values():
            assert isinstance(model_parameter, tuple)
            assert len(model_parameter)==2
            assert all([(isinstance(el, float) or isinstance(el, int)) for el in model_parameter])

        return True
