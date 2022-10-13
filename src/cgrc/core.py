"""
:Author: Balazs Szigeti <szb37@pm.me>
:Copyright: 2020, DrugNerdsLab
:License: MIT
"""

from tqdm.contrib.itertools import product as tqdmproduct
from sklearn.neighbors import KernelDensity
from itertools import product as product
from statistics import mean, median
import src.my_dataframes as mydfs
import src.constants as constants
import src.folders as folders
import src.figures as figures
import src.cgrc.stats as stats
import src.miscs as miscs
import pandas as pd
import numpy as np
import pingouin
import random
import os


class Controllers():

    @staticmethod
    def run_cgrc_trial(trial_name, postfix, cgrc_param_set, study_scales=None, add_columns=None, save_figs=True):

        assert isinstance(trial_name, str)
        assert isinstance(postfix, str)
        assert isinstance(cgrc_param_set, int)
        assert (isinstance(study_scales, dict) or (study_scales is None))
        assert (isinstance(add_columns, dict)) or (add_columns is None)

        cgrc_parameters = constants.cgrc_parameters[cgrc_param_set]
        analysis_name = trial_name + '_{}'.format(postfix)

        trial_data_dir, trial_stats_dir, cgrc_data_dir, cgrc_stats_dir, cgrc_plots_dir = miscs.create_analysis_dirs(
            analysis_name,
            trial_data_subdir = False,
            incl_cgrc_plots_dir = save_figs,
        )

        # Get trial stats
        Controllers.get_trial_stats(
            input_dir = trial_data_dir,
            input_fname = trial_name + '__trial_data.csv',
            output_dir = trial_stats_dir,
            output_prefix = analysis_name,
            study_scales = study_scales,
        )

        # Get CGRC from trial data
        Controllers.get_cgrc_data(
            input_dir = trial_data_dir,
            input_fname = trial_name + '__trial_data.csv',
            output_dir = cgrc_data_dir,
            output_prefix = analysis_name,
            cgrc_parameters = cgrc_parameters,
            study_scales = study_scales,
        )

        # Get CGRC stats
        Controllers.get_cgrc_stats(
            input_dir = cgrc_data_dir,
            input_fname = analysis_name + '__cgrc_data.csv',
            output_dir = cgrc_stats_dir,
            output_prefix = analysis_name,
            study_scales = study_scales,
        )


        # Make CGRC figues
        if save_figs:
            figures.Controllers.plot_VScgr_twinx(
                input_dir = cgrc_stats_dir,
                input_fname = analysis_name + '__cgrc_model_components.csv',
                output_dir = cgrc_plots_dir,
                output_prefix = analysis_name,
                study_scales = study_scales,
            )

        # Get summary tables
        v1_summary = Controllers.get_cgrc_comparison_table_v1(
            trial_name = trial_name,
            analysis_name = analysis_name,
            study_scales = study_scales,
            trial_data_dir = trial_data_dir,
            trial_stats_dir = trial_stats_dir,
            cgrc_data_dir = cgrc_data_dir,
            cgrc_stats_dir = cgrc_stats_dir,
        )
        print('\n',v1_summary.to_string(index=False))

        '''
        v2_summary = Controllers.get_cgrc_comparison_table_v2(
            trial_name = trial_name,
            analysis_name = analysis_name,
            study_scales = study_scales,
            cgrc_data_dir = cgrc_data_dir,
            cgrc_stats_dir = cgrc_stats_dir,
        )
        print('\n', v2_summary.to_string(index=False))
        '''

    @staticmethod
    def get_trial_stats(input_dir, input_fname, output_dir, output_prefix, study_scales=None, add_columns=None):
        ''' Wrapper function '''
        stats.Controllers.get_trial_stats(
            input_dir = input_dir,
            input_fname = input_fname,
            output_dir = output_dir,
            output_prefix = output_prefix,
            add_columns = add_columns,
            study_scales = study_scales,
        )

    @staticmethod
    def get_cgrc_data(input_dir, input_fname, output_dir, output_prefix, cgrc_parameters, study_scales=None, add_columns=None):
        ''' Wrapper function '''
        CorrectGuessRateCurve.get_cgrc_data(
            input_dir = input_dir,
            input_fname = input_fname,
            output_dir = output_dir,
            output_prefix = output_prefix,
            cgrc_parameters = cgrc_parameters,
            add_columns = add_columns,
            study_scales = study_scales,
        )

    @staticmethod
    def get_cgrc_stats(input_dir, input_fname, output_dir, output_prefix, study_scales=None, add_columns=None):
        ''' Wrapper function '''
        stats.Controllers.get_cgrc_stats(
            input_dir = input_dir,
            input_fname = input_fname,
            output_dir = output_dir,
            output_prefix = output_prefix,
            add_columns = add_columns,
            study_scales = study_scales,
        )

    @staticmethod
    def get_cgrc_comparison_table_v1(trial_name, analysis_name, trial_data_dir, trial_stats_dir, cgrc_data_dir, cgrc_stats_dir, study_scales=None,):

        assert isinstance(trial_name, str)
        assert isinstance(analysis_name, str)
        assert (isinstance(study_scales, dict) or (study_scales is None))

        df = pd.DataFrame(
            columns=[
                'scale',
                'trial_cgr',
                'trial_est',
                'trial_p',
                'trial_g',
                'cgr_cgr',
                'cgrc_est',
                'cgrc_p',
                'cgrc_g',
        ])

        trial_data = pd.read_csv(os.path.join(trial_data_dir, trial_name+'__trial_data.csv'))

        trial_stats = pd.read_csv(os.path.join(trial_stats_dir, analysis_name+'__model_components.csv'))
        trial_stats = trial_stats.loc[
            (trial_stats.model_type=='without_guess') &
            (trial_stats.component=='conditionAC')
        ]

        cgrc_data = pd.read_csv(os.path.join(cgrc_data_dir, analysis_name+'__cgrc_data.csv'))
        cgrc_data = cgrc_data.loc[
            (cgrc_data.cgr==0.5)
        ]

        cgrc_stats = pd.read_csv(os.path.join(cgrc_stats_dir, analysis_name+'__cgrc_model_components.csv'))
        cgrc_stats = cgrc_stats.loc[
            (cgrc_stats.cgr==0.5) &
            (cgrc_stats.model_type=='without_guess') &
            (cgrc_stats.component=='conditionAC')
        ]

        assert cgrc_stats.scale.unique().tolist()==trial_stats.scale.unique().tolist()

        for scale in trial_stats.scale.unique().tolist():

            tmp_trial_data = trial_data.loc[(trial_data.scale==scale)]
            tmp_trial_stats = trial_stats.loc[(trial_stats.scale==scale)]
            tmp_cgrc_data = cgrc_data.loc[(cgrc_data.scale==scale)]
            tmp_cgrc_stats = cgrc_stats.loc[(cgrc_stats.scale==scale)]

            # Compute trial effect size
            trial_hedges_g = pingouin.compute_effsize(
                tmp_trial_data.loc[tmp_trial_data.condition=='AC'].delta_score,
                tmp_trial_data.loc[tmp_trial_data.condition=='PL'].delta_score,
                eftype='hedges'
            )

            # Compute CGRC effect size
            cgrc_hedges_g=[]
            for cgr_trial_id in cgrc_data.cgr_trial_id.unique().tolist():
                cgrc_hedges_g.append(
                    pingouin.compute_effsize(
                        tmp_cgrc_data.loc[(tmp_cgrc_data.condition=='AC') & (tmp_cgrc_data.cgr_trial_id==cgr_trial_id)].delta_score,
                        tmp_cgrc_data.loc[(tmp_cgrc_data.condition=='PL') & (tmp_cgrc_data.cgr_trial_id==cgr_trial_id)].delta_score,
                        eftype='hedges'
                    ))

            # Compute trial CGR
            n_plpl = tmp_trial_data.loc[(tmp_trial_data.condition=='PL') & (tmp_trial_data.guess=='PL')].shape[0]
            n_acac = tmp_trial_data.loc[(tmp_trial_data.condition=='AC') & (tmp_trial_data.guess=='AC')].shape[0]
            trial_cgr = (n_plpl+n_acac)/tmp_trial_data.shape[0]

            # Compute CGRC CGR
            cgrc_cgrs=[]
            for cgr_trial_id in cgrc_data.cgr_trial_id.unique().tolist():
                n_plpl = tmp_cgrc_data.loc[(tmp_cgrc_data.condition=='PL') & (tmp_cgrc_data.guess=='PL') & (tmp_cgrc_data.cgr_trial_id==cgr_trial_id)].shape[0]
                n_acac = tmp_cgrc_data.loc[(tmp_cgrc_data.condition=='AC') & (tmp_cgrc_data.guess=='AC') & (tmp_cgrc_data.cgr_trial_id==cgr_trial_id)].shape[0]
                cgrc_cgrs.append((n_plpl+n_acac)/tmp_cgrc_data.loc[(tmp_cgrc_data.cgr_trial_id==cgr_trial_id)].shape[0])


            # Add results to output table
            row={}
            row['scale'] = scale
            row['trial_cgr'] = round(trial_cgr, 2)
            row['trial_est'] = '{}±{}'.format(
                round(miscs.get_estimate(tmp_trial_stats.est.tolist()), 1),
                round(miscs.get_estimate(tmp_trial_stats.se.tolist()), 1)
            )
            row['trial_p'] = round(miscs.get_estimate(tmp_trial_stats.p.tolist()), 4)
            row['trial_g'] = round(trial_hedges_g, 2)
            row['cgr_cgr'] = round(miscs.get_estimate(cgrc_cgrs), 2)
            row['cgrc_est'] = '{}±{}'.format(
                round(miscs.get_estimate(tmp_cgrc_stats.est.tolist()), 1),
                round(miscs.get_estimate(tmp_cgrc_stats.se.tolist()), 1)
            )
            row['cgrc_p'] = round(miscs.get_estimate(tmp_cgrc_stats.p.tolist()), 4)
            row['cgrc_g'] = round(miscs.get_estimate(cgrc_hedges_g), 1)
            df = df.append(row, ignore_index=True)

        df.to_csv(os.path.join(folders.csvs_summary_tables, analysis_name+'_{}__summary_table_v1.csv'.format(constants.estimator)), index=False)
        return df

    @staticmethod
    def get_cgrc_comparison_table_v2(trial_name, analysis_name, cgrc_data_dir, cgrc_stats_dir, study_scales=None,):

        assert isinstance(trial_name, str)
        assert isinstance(analysis_name, str)
        assert (isinstance(study_scales, dict) or (study_scales is None))

        df = pd.DataFrame(
            columns=[
                'scale',
                'theory_cgr',
                'measure_cgr',
                'cgrc_est',
                'cgrc_p',
                'cgrc_g',
        ])

        cgrc_data = pd.read_csv(os.path.join(cgrc_data_dir, analysis_name+'__cgrc_data.csv'))

        cgrc_stats = pd.read_csv(os.path.join(cgrc_stats_dir, analysis_name+'__cgrc_model_components.csv'))
        cgrc_stats = cgrc_stats.loc[
            (cgrc_stats.model_type=='without_guess') &
            (cgrc_stats.component=='conditionAC')
        ]

        for scale, cgr in product(cgrc_data.scale.unique().tolist(), cgrc_data.cgr.unique().tolist()):

            tmp_cgrc_data = cgrc_data.loc[(cgrc_data.scale==scale) & (cgrc_data.cgr==cgr)]
            tmp_cgrc_stats = cgrc_stats.loc[(cgrc_stats.scale==scale) & (cgrc_stats.cgr==cgr)]

            # Compute CGRC effect size
            cgrc_hedges_g=[]
            for cgr_trial_id in cgrc_data.cgr_trial_id.unique().tolist():
                cgrc_hedges_g.append(
                    pingouin.compute_effsize(
                        tmp_cgrc_data.loc[(tmp_cgrc_data.condition=='AC') & (tmp_cgrc_data.cgr_trial_id==cgr_trial_id)].delta_score,
                        tmp_cgrc_data.loc[(tmp_cgrc_data.condition=='PL') & (tmp_cgrc_data.cgr_trial_id==cgr_trial_id)].delta_score,
                        eftype='hedges'
                    ))

            # Compute CGRC CGR
            cgrc_cgrs=[]
            for cgr_trial_id in cgrc_data.cgr_trial_id.unique().tolist():
                n_plpl = tmp_cgrc_data.loc[(tmp_cgrc_data.condition=='PL') & (tmp_cgrc_data.guess=='PL') & (tmp_cgrc_data.cgr_trial_id==cgr_trial_id)].shape[0]
                n_acac = tmp_cgrc_data.loc[(tmp_cgrc_data.condition=='AC') & (tmp_cgrc_data.guess=='AC') & (tmp_cgrc_data.cgr_trial_id==cgr_trial_id)].shape[0]
                cgrc_cgrs.append((n_plpl+n_acac)/tmp_cgrc_data.loc[(tmp_cgrc_data.cgr_trial_id==cgr_trial_id)].shape[0])

            # Add results to output table
            row={}
            row['scale'] = scale
            row['theory_cgr'] = round(cgr, 2)
            row['measure_cgr'] = round(miscs.get_estimate(cgrc_cgrs), 2)
            row['cgrc_est'] = '{}±{}'.format(
                round(miscs.get_estimate(tmp_cgrc_stats.est.tolist()), 1),
                round(miscs.get_estimate(tmp_cgrc_stats.se.tolist()), 1)
            )
            row['cgrc_p'] = round(miscs.get_estimate(tmp_cgrc_stats.p.tolist()), 4)
            row['cgrc_g'] = round(miscs.get_estimate(cgrc_hedges_g), 2)
            df = df.append(row, ignore_index=True)

        df.to_csv(os.path.join(folders.csvs_summary_tables, analysis_name+'_{}__summary_table_v2.csv'.format(constants.estimator)), index=False)
        return df


class CorrectGuessRateCurve():
    """ Methods to calculate break blind curve """

    @staticmethod
    def get_cgrc_data(input_dir, input_fname, output_dir, output_prefix, cgrc_parameters, use_KDE=True, strata_sampling='all_prop', study_scales=None, add_columns=None, n_sample=None):
        """ Gets break blind curve (BBC) data
            Args:
                trial_data_df (pd.DataFrame): 'processed' type dataframe
                analysis_name (str): analysis_name for output
                study_scales (dict): defines combinations of studies/scales for which stats will be calculated
                    E.g. study_scales = {'study2':['scale1', 'scale3']}, will process scales 1 and 3 of study2
                cgr_values (list): list of CGR (break blind ratio) values; values should be 0<=float<=1
                n_cgrc_trials (int): number of psuedo-TRIAL simulated for each CGR
                strata_sampling (str, optional): method how to assign sample size between strata;
                    must be in ['all_prop', 'all_equal', 'active_equal', 'active_prop']
                ss (int, optional): sample size of each trial; if None, same sample size is used as
                    the sample size for the given study/scale pair in study_scales
                output_dir (str, optional): filepath to folder where output is written
        """

        assert isinstance(input_dir, str)
        assert isinstance(input_fname, str)
        assert isinstance(output_dir, str)
        assert isinstance(output_prefix, str)
        assert isinstance(cgrc_parameters, dict)
        assert isinstance(use_KDE, bool)
        assert strata_sampling in ['all_prop', 'all_equal', 'active_equal', 'active_prop']
        assert (isinstance(study_scales, dict) or (study_scales is None))
        assert (isinstance(add_columns, dict)) or (add_columns is None)
        assert isinstance(n_sample, int) or (n_sample is None)

        n_cgrc_trials = cgrc_parameters['n_cgrc_trials']
        cgr_values = [round(el, 5) for el in cgrc_parameters['cgr_values']] # CGR values need to be rounded, otherwise downstream R df filtering may not work

        trial_data_df = pd.read_csv(os.path.join(input_dir, input_fname))
        studies, scales, respondents, guessers = miscs.get_study_scales(
            input_df = trial_data_df,
            study_scales = study_scales)

        master_cgrc_df = mydfs.CGRCurveDf()
        master_cgrc_df.add_univalue_columns({'respondent':None})

        for study, scale, respondent, guesser in product(studies, scales, respondents, guessers):

            if guesser=='ext':
                return # CGRC only makes sense for guessing by patients

            df_filtered = trial_data_df.loc[
                (trial_data_df.guesser == guesser) &
                (trial_data_df.respondent == respondent) &
                (trial_data_df.scale == scale)]

            if study!='all':
                df_filtered = df_filtered.loc[(df_filtered.study == study)]

            if df_filtered.shape[0]==0:
                continue

            if n_sample is None:
                total_sample_size = df_filtered.shape[0]
            else:
                total_sample_size = n_sample

            if use_KDE:
                kdes = CorrectGuessRateCurve.get_kdes(df_filtered=df_filtered)

            desc='Get CGRC data ({}:{})'.format(study, scale)

            for cgr, cgr_trial_id in tqdmproduct(cgr_values, range(n_cgrc_trials), desc=desc, disable=False):

                sample_sizes = CorrectGuessRateCurve.get_strata_sample_sizes(
                    total_sample_size=total_sample_size,
                    df_filtered=df_filtered,
                    correct_guess_rate=cgr,
                    strata_sampling=strata_sampling,)

                if use_KDE:
                    cgrc_datapoint_df = CorrectGuessRateCurve.get_cgrc_datapoint_KDE(
                        df_filtered=df_filtered,
                        sample_sizes=sample_sizes,
                        kdes=kdes,
                        cgr=cgr,)
                else:
                    cgrc_datapoint_df = CorrectGuessRateCurve.get_cgrc_datapoint_distribution(
                        df_filtered=df_filtered,
                        sample_sizes=sample_sizes,
                        cgr=cgr,)


                cgrc_datapoint_df.cgr_trial_id = cgr_trial_id
                cgrc_datapoint_df.study = study
                cgrc_datapoint_df.scale = scale
                cgrc_datapoint_df['respondent'] = respondent

                master_cgrc_df = pd.concat([master_cgrc_df, cgrc_datapoint_df], sort=False)
                del cgrc_datapoint_df, sample_sizes


        master_cgrc_df.__class__= mydfs.CGRCurveDf
        master_cgrc_df.add_univalue_columns({'guesser': 'self'})
        master_cgrc_df.add_univalue_columns(add_columns)
        master_cgrc_df.set_column_types()

        master_cgrc_df.to_csv(os.path.join(output_dir, output_prefix+'__cgrc_data.csv'), index=False)

    @staticmethod
    def get_cgrc_datapoint_KDE(df_filtered, sample_sizes, cgr, kdes):
        """ Gets psuedodata for the BBC with given parameters """

        plpl_df = mydfs.CGRCurveDf()
        acpl_df = mydfs.CGRCurveDf()
        plac_df = mydfs.CGRCurveDf()
        acac_df = mydfs.CGRCurveDf()

        plpl_scores = [round(score) for score in kdes['PLPL'].sample(sample_sizes['PLPL']).reshape(1, -1).tolist()[0]]
        acpl_scores = [round(score) for score in kdes['ACPL'].sample(sample_sizes['ACPL']).reshape(1, -1).tolist()[0]]
        plac_scores = [round(score) for score in kdes['PLAC'].sample(sample_sizes['PLAC']).reshape(1, -1).tolist()[0]]
        acac_scores = [round(score) for score in kdes['ACAC'].sample(sample_sizes['ACAC']).reshape(1, -1).tolist()[0]]

        plpl_df.delta_score = plpl_scores
        acpl_df.delta_score = acpl_scores
        plac_df.delta_score = plac_scores
        acac_df.delta_score = acac_scores

        plpl_df.condition = 'PL'
        acpl_df.condition = 'AC'
        plac_df.condition = 'PL'
        acac_df.condition = 'AC'

        plpl_df.guess = 'PL'
        acpl_df.guess = 'PL'
        plac_df.guess = 'AC'
        acac_df.guess = 'AC'

        df = pd.concat([plpl_df, acpl_df, plac_df, acac_df,], sort=False)
        df.cgr = cgr
        return df

    @staticmethod
    def get_cgrc_datapoint_distribution(df_filtered, sample_sizes, cgr):
        """ Gets psuedodata for the BBC with given parameters """

        plpl_scores = [random.choice(df_filtered.loc[(df_filtered.condition=='PL') & (df_filtered.guess=='PL')].score.tolist()) for idx in range(sample_sizes['PLPL'])]
        acpl_scores = [random.choice(df_filtered.loc[(df_filtered.condition=='AC') & (df_filtered.guess=='PL')].score.tolist()) for idx in range(sample_sizes['ACPL'])]
        plac_scores = [random.choice(df_filtered.loc[(df_filtered.condition=='PL') & (df_filtered.guess=='AC')].score.tolist()) for idx in range(sample_sizes['PLAC'])]
        acac_scores = [random.choice(df_filtered.loc[(df_filtered.condition=='AC') & (df_filtered.guess=='AC')].score.tolist()) for idx in range(sample_sizes['ACAC'])]

        plpl_df = mydfs.CGRCurveDf()
        acpl_df = mydfs.CGRCurveDf()
        plac_df = mydfs.CGRCurveDf()
        acac_df = mydfs.CGRCurveDf()

        plpl_df.delta_score = plpl_scores
        acpl_df.delta_score = acpl_scores
        plac_df.delta_score = plac_scores
        acac_df.delta_score = acac_scores

        plpl_df.condition = 'PL'
        acpl_df.condition = 'AC'
        plac_df.condition = 'PL'
        acac_df.condition = 'AC'

        plpl_df.guess = 'PL'
        acpl_df.guess = 'PL'
        plac_df.guess = 'AC'
        acac_df.guess = 'AC'

        df = pd.concat([plpl_df, acpl_df, plac_df, acac_df,], sort=False)
        df.cgr = cgr
        return df

    @staticmethod
    def get_kdes(df_filtered):
        """ Get kernels for break-blind / non-break-blind scores"""

        kdes = {}

        plpl_scores = df_filtered.loc[(df_filtered.condition == 'PL') & (df_filtered.guess == 'PL'), 'delta_score']
        kdes['PLPL'] = KernelDensity(kernel='gaussian').fit(np.array(plpl_scores).reshape(-1, 1))

        acpl_scores = df_filtered.loc[(df_filtered.condition == 'AC') & (df_filtered.guess == 'PL'), 'delta_score']
        kdes['ACPL'] = KernelDensity(kernel='gaussian').fit(np.array(acpl_scores).reshape(-1, 1))

        plac_scores = df_filtered.loc[(df_filtered.condition == 'PL') & (df_filtered.guess == 'AC'), 'delta_score']
        kdes['PLAC'] = KernelDensity(kernel='gaussian').fit(np.array(plac_scores).reshape(-1, 1))

        acac_scores = df_filtered.loc[(df_filtered.condition == 'AC') & (df_filtered.guess == 'AC'), 'delta_score']
        kdes['ACAC'] = KernelDensity(kernel='gaussian').fit(np.array(acac_scores).reshape(-1, 1))

        return kdes

    @staticmethod
    def get_strata_ratio(df_filtered):
        """ Returns the ratio of each strata in df """

        strata_ratio = {'PLPL':0, 'ACPL':0, 'PLAC':0, 'ACAC':0,}
        n_all  = df_filtered.shape[0]

        ss_plpl = df_filtered.loc[(df_filtered.condition=='PL') & (df_filtered.guess=='PL')].shape[0]
        ss_acpl = df_filtered.loc[(df_filtered.condition=='AC') & (df_filtered.guess=='PL')].shape[0]
        ss_plac = df_filtered.loc[(df_filtered.condition=='PL') & (df_filtered.guess=='AC')].shape[0]
        ss_acac = df_filtered.loc[(df_filtered.condition=='AC') & (df_filtered.guess=='AC')].shape[0]

        strata_ratio['PLPL'] = round(ss_plpl/n_all, 2)
        strata_ratio['ACPL'] = round(ss_acpl/n_all, 2)
        strata_ratio['PLAC'] = round(ss_plac/n_all, 2)
        strata_ratio['ACAC'] = round(ss_acac/n_all, 2)

        return strata_ratio

    @staticmethod
    def get_strata_sample_sizes(total_sample_size, df_filtered, correct_guess_rate, strata_sampling):
        """ Get number of datapoints in each strata """

        assert strata_sampling in ['all_prop', 'all_equal', 'active_equal', 'active_prop']
        sample_sizes={}

        if strata_sampling=='all_prop':
            sample_size_bb = round(correct_guess_rate*total_sample_size) # sample_size_bb: sample size for blind breaking
            sample_size_nbb = total_sample_size-sample_size_bb # sample_size_nbb: sample size for NON blind breaking
            strata_ratio = CorrectGuessRateCurve.get_strata_ratio(df_filtered)

            # Ratio of PLPL within BB (break blind) cases
            plpl_correct_guess_rate = strata_ratio['PLPL']/(strata_ratio['PLPL']+strata_ratio['ACAC'])
            sample_sizes['PLPL'] = round(sample_size_bb*(plpl_correct_guess_rate))
            sample_sizes['ACAC'] = sample_size_bb-sample_sizes['PLPL']

            # Ratio of ACPL within NBB (non break blind) cases
            acpl_ncorrect_guess_rate = strata_ratio['ACPL']/(strata_ratio['ACPL']+strata_ratio['PLAC'])
            sample_sizes['ACPL'] = round(sample_size_nbb*(acpl_ncorrect_guess_rate))
            sample_sizes['PLAC'] = sample_size_nbb-sample_sizes['ACPL']



        elif strata_sampling=='all_equal':
            sample_size_bb = round(correct_guess_rate*total_sample_size) # sample_size_bb: sample size for blind breaking
            sample_size_nbb = total_sample_size-sample_size_bb # sample_size_nbb: sample size for NON blind breaking
            sample_sizes['PLPL'] = round(sample_size_bb/2)
            sample_sizes['ACAC'] = sample_size_bb-sample_sizes['PLPL']

            sample_sizes['ACPL'] = round(sample_size_nbb/2)
            sample_sizes['PLAC'] = sample_size_nbb-sample_sizes['ACPL']

        elif strata_sampling=='active_equal':

            acs_sample_size = round(total_sample_size/2) # sample_size of active stratas
            pls_sample_size = total_sample_size-acs_sample_size # sample_size of placebo stratas

            sample_sizes['PLPL'] = round(pls_sample_size/2)
            sample_sizes['PLAC'] = pls_sample_size-sample_sizes['PLPL']

            sample_sizes['ACAC'] = round(correct_guess_rate*acs_sample_size)
            sample_sizes['ACPL'] = acs_sample_size-sample_sizes['ACAC']

        elif strata_sampling=='active_prop':

            strata_ratio = CorrectGuessRateCurve.get_strata_ratio(df_filtered)

            acs_sample_size = round(total_sample_size*(strata_ratio['ACPL']+strata_ratio['ACAC']))
            pls_sample_size = total_sample_size-acs_sample_size

            sample_sizes['PLPL'] = round(
                pls_sample_size*(strata_ratio['PLPL']/(strata_ratio['PLPL']+strata_ratio['PLAC'])))
            sample_sizes['PLAC'] = pls_sample_size - sample_sizes['PLPL']

            sample_sizes['ACAC'] = round(correct_guess_rate*acs_sample_size)
            sample_sizes['ACPL'] = acs_sample_size-sample_sizes['ACAC']

        else:
            assert False

        return sample_sizes
