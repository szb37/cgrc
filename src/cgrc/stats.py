"""
:Author: Balazs Szigeti <szb37@pm.me>
:Copyright: 2020, DrugNerdsLab
:License: MIT

There is inconsistency in how the model and strata stats are extracted from R (TODO: sync methods):
- for strata, R output is converted to python JSON which is then parsed
- for model, R output is directly extracted form the R output
"""

import src.cgrc.covariate_selector as covariate_selector
from tqdm.contrib.itertools import product as tqdmproduct
import src.constants as constants
import src.miscs as miscs
import src.my_dataframes as mydfs
from itertools import product as product
from rpy2.robjects import r
import pandas as pd
import json
import copy
import os


r('library(lmerTest)')
r('library("emmeans")')
r('library("dplyr")')
r('library(RJSONIO)')


class Controllers():
    """ Methods to be called from the outside """

    @staticmethod
    def get_trial_stats(input_dir, input_fname, output_dir, output_prefix, study_scales=None, add_columns=None, try_covs=[]):
        """ Gets stats from processedDf and save results as CSV
            Args:
                analysis_name (str): analysis_name of input/output CSV
                study_scales (dict): key-values of studies-scales for which stats will be calculated
                    E.g. study_scales = {'study2':['scale1', 'scale3']}, will process scales 1 and 3 of study2
                    #TODO: add defualt argument where all are processed
                try_covs (list): list of strings, were each string is a column name in df; eacht will be
                    tried as a model covariate
                input_dir (str, optional): filepath to folder where input CSV is stored
                output_dir (str, optional): filepath to folder where output is written
        """

        assert isinstance(input_dir, str)
        assert isinstance(input_fname, str)
        assert isinstance(output_dir, str)
        assert isinstance(output_prefix, str)
        assert (isinstance(study_scales, dict) or (study_scales is None))
        assert (isinstance(add_columns, dict)) or (add_columns is None)
        assert isinstance(try_covs, list)

        # Initalize output
        master_model_summary_df = mydfs.ModelSummaryDf()
        master_model_components_df = mydfs.ModelComponentsDf()
        master_strata_summary_df = mydfs.StrataSummaryDf()
        master_strata_contrast_df = mydfs.StrataContrastDf()

        # Get study_scales from trial_data input
        input_trial_data_fpath = os.path.join(input_dir, input_fname).replace('\\', '/')
        trial_data_df = pd.read_csv(input_trial_data_fpath)
        studies, scales, respondents, guessers = miscs.get_study_scales(
            input_df = trial_data_df,
            study_scales = study_scales)

        # Read dataframe into R and set baseline to be PL
        Helpers.load_df_into_R_space(input_trial_data_fpath)

        desc='Get trial stats ({})'.format(input_fname)

        for study, scale, respondent, guesser in tqdmproduct(studies, scales, respondents, guessers, desc=desc, disable=False):

            Helpers.get_df_filtered(study, scale, respondent, guesser)

            try:
                all_dfs = StatsCore.get_stats(try_covs=try_covs, add_cgrc_columns=False)
                all_dfs = Helpers.add_metadata_process_trialdata(
                    all_dfs = all_dfs,
                    study=study,
                    scale=scale,
                    guesser=guesser,
                    respondent=respondent)
            except NoSelectedRows:
                continue
            finally:
                r('rm(df_filtered)')

            # Concatanate dataframes
            master_model_summary_df = pd.concat([master_model_summary_df, all_dfs['model_summary']], sort=False)
            master_model_components_df = pd.concat([master_model_components_df, all_dfs['model_components']], sort=False)
            master_strata_summary_df = pd.concat([master_strata_summary_df, all_dfs['strata_summary']], sort=False)
            master_strata_contrast_df = pd.concat([master_strata_contrast_df, all_dfs['strata_contrast']], sort=False)

        # Clear R global namespace and save results
        r('rm(list = ls())')

        master_strata_summary_df.__class__= mydfs.StrataSummaryDf
        master_strata_contrast_df.__class__= mydfs.StrataContrastDf
        master_model_components_df.__class__= mydfs.ModelComponentsDf
        master_model_summary_df.__class__= mydfs.ModelSummaryDf

        #if add_columns is not None:
        master_strata_summary_df.add_univalue_columns(add_columns)
        master_strata_contrast_df.add_univalue_columns(add_columns)
        master_model_components_df.add_univalue_columns(add_columns)
        master_model_summary_df.add_univalue_columns(add_columns)

        master_strata_summary_df.set_column_types()
        master_strata_contrast_df.set_column_types()
        master_model_components_df.set_column_types()
        master_model_summary_df.set_column_types()

        master_strata_summary_df.to_csv(os.path.join(output_dir, output_prefix+'__strata_summary.csv'), index=False)
        master_strata_contrast_df.to_csv(os.path.join(output_dir, output_prefix+'__strata_contrast.csv'), index=False)
        master_model_components_df.to_csv(os.path.join(output_dir, output_prefix+'__model_components.csv'), index=False)
        master_model_summary_df.to_csv(os.path.join(output_dir, output_prefix+'__model_summary.csv'), index=False)

    @staticmethod
    def get_cgrc_stats(input_dir, input_fname, output_dir, output_prefix, study_scales=None, add_columns=None, try_covs=[]):
        """ Gets stats from processedDf (with BBC specific columns, cgr and cgr_trial_id) and save results as CSV
            Args:
                analysis_name (str): analysis_name of input/output CSV
                study_scales (dict): defines combinations of studies/scales for which stats will be calculated
                    E.g. study_scales = {'study2':['scale1', 'scale3']}, will process scales 1 and 3 of study2
                    #TODO: add defualt argument where all are processed
                try_covs (list): list of strings, were each string is a column name in df; eacht will be
                    tried as a model covariate; covariates are not resampled for BBC, hence do not
                    include in models as default option
                input_dir (str, optional): filepath to folder where input CSV is stored
                output_dir (str, optional): filepath to folder where output is written
        """

        assert isinstance(input_dir, str)
        assert isinstance(input_fname, str)
        assert isinstance(output_dir, str)
        assert isinstance(output_prefix, str)
        assert (isinstance(study_scales, dict) or (study_scales is None))
        assert isinstance(try_covs, list)
        assert (isinstance(add_columns, dict)) or (add_columns is None)

        # Read dataframe into R and set baseline to be PL
        input_fpath = os.path.join(input_dir, input_fname).replace('\\', '/')
        Helpers.load_df_into_R_space(input_fpath)

        # Extracts cgrs and cgr_trial_ids from df
        cgrc_data_df = pd.read_csv(input_fpath)
        cgr_trial_ids = cgrc_data_df.cgr_trial_id.unique().tolist()
        cgrs = cgrc_data_df.cgr.unique().tolist()
        studies, scales, respondents, guessers = miscs.get_study_scales(
            input_df = cgrc_data_df,
            study_scales = study_scales)

        # Initalize output
        master_model_summary_df = mydfs.ModelSummaryDf()
        master_model_summary_df.add_univalue_columns({'cgr':None, 'cgr_trial_id':None})
        master_model_components_df = mydfs.ModelComponentsDf()
        master_model_components_df.add_univalue_columns({'cgr':None, 'cgr_trial_id':None})
        master_strata_summary_df = mydfs.StrataSummaryDf()
        master_strata_summary_df.add_univalue_columns({'cgr':None, 'cgr_trial_id':None})
        master_strata_contrast_df = mydfs.StrataContrastDf()
        master_strata_contrast_df.add_univalue_columns({'cgr':None, 'cgr_trial_id':None})

        desc='Get CGRC stats ({})'.format(input_fname)

        for study, scale, respondent, guesser, cgr, cgr_trial_id, in tqdmproduct(studies, scales, respondents, guessers, cgrs, cgr_trial_ids, desc=desc, disable=False):

            Helpers.get_df_filtered(study, scale, respondent, guesser, cgr, cgr_trial_id)

            try:
                all_dfs = StatsCore.get_stats(try_covs=try_covs, add_cgrc_columns=True)
                all_dfs = Helpers.add_metadata_process_cgrc(
                    all_dfs=all_dfs,
                    study=study,
                    scale=scale,
                    cgr=cgr,
                    cgr_trial_id=cgr_trial_id,
                    guesser=guesser,
                    respondent=respondent)
            except NoSelectedRows:
                continue
            finally:
                r('rm(df_filtered)')

            # Concatanate dataframes
            master_model_summary_df = pd.concat([master_model_summary_df, all_dfs['model_summary']], sort=False)
            master_model_components_df = pd.concat([master_model_components_df, all_dfs['model_components']], sort=False)
            master_strata_summary_df = pd.concat([master_strata_summary_df, all_dfs['strata_summary']], sort=False)
            master_strata_contrast_df = pd.concat([master_strata_contrast_df, all_dfs['strata_contrast']], sort=False)

        # Clear R global namespace and save results
        r('rm(list = ls())')

        master_strata_summary_df.__class__= mydfs.StrataSummaryDf
        master_strata_contrast_df.__class__= mydfs.StrataContrastDf
        master_model_components_df.__class__= mydfs.ModelComponentsDf
        master_model_summary_df.__class__= mydfs.ModelSummaryDf

        master_strata_summary_df.add_univalue_columns(add_columns)
        master_strata_contrast_df.add_univalue_columns(add_columns)
        master_model_components_df.add_univalue_columns(add_columns)
        master_model_summary_df.add_univalue_columns(add_columns)

        master_strata_summary_df.set_column_types()
        master_strata_contrast_df.set_column_types()
        master_model_components_df.set_column_types()
        master_model_summary_df.set_column_types()

        master_strata_summary_df.to_csv(os.path.join(output_dir, output_prefix + '__cgrc_strata_summary.csv'), index=False)
        master_strata_contrast_df.to_csv(os.path.join(output_dir, output_prefix + '__cgrc_strata_contrast.csv'), index=False)
        master_model_components_df.to_csv(os.path.join(output_dir, output_prefix + '__cgrc_model_components.csv'), index=False)
        master_model_summary_df.to_csv(os.path.join(output_dir, output_prefix + '__cgrc_model_summary.csv'), index=False)


class StatsCore():
    """ Methods to extract starta/model stats using R """

    @staticmethod
    def get_stats(try_covs, add_cgrc_columns):
        """ Get stats for dataframe
            Args:
                try_covs (list): list of strings, each string is a potential covariate (i.e. column name in df_filtered)
                    that will be tried as a covariate for models.
        """

        if r('nrow(df_filtered)')[0]==0:
            raise NoSelectedRows()

        model_summary, model_components = StatsCore.get_model_stats(
            try_covs=try_covs,
            add_cgrc_columns=add_cgrc_columns)


        #strata_summary, strata_contrast = StatsCore.get_strata_stats(
        #    try_covs=try_covs,
        #    add_cgrc_columns=add_cgrc_columns)

        strata_summary = mydfs.StrataSummaryDf()
        strata_summary.add_univalue_columns({'cgr':None, 'cgr_trial_id':None})
        strata_contrast = mydfs.StrataContrastDf()
        strata_contrast.add_univalue_columns({'cgr':None, 'cgr_trial_id':None})

        all_dfs = {
            'model_summary': model_summary,
            'model_components': model_components,
            'strata_summary': strata_summary,
            'strata_contrast': strata_contrast,
        }

        return all_dfs

    @staticmethod
    def get_model_stats(try_covs, add_cgrc_columns):
        """ Get component and summary dataframes

            Args:
                try_covs (list): list of strings, each string is a potential covariate (i.e. column name in df_filtered)
                    that will be tried as a covariate for models.

            Returns:
                model_summary (mydfs.ModelSummaryDf): model summary dataframe; child of pandas df
                model_components (mydfs.ModelComponentsDf): model components dataframe; child of pandas df
        """

        # Initalize output
        model_summary = mydfs.ModelSummaryDf()
        model_components = mydfs.ModelComponentsDf()
        if add_cgrc_columns:
            model_summary.add_univalue_columns({'cgr':None, 'cgr_trial_id':None})
            model_components.add_univalue_columns({'cgr':None, 'cgr_trial_id':None})

        covariate_selector.Controllers.get_adjusted_model(
            model_name='without_guess',
            try_covs=try_covs,
            enforce_covs=['condition'])
        covariate_selector.Controllers.get_adjusted_model(
            model_name='with_guess',
            try_covs=try_covs,
            enforce_covs=['condition', 'guess', 'condition*guess'])

        for model_type in ['without_guess', 'with_guess']:

            # extract model summary df
            r('model_sum = summary({})'.format(model_type))
            model_summary_fromR = Helpers.get_model_summary_stats()
            model_summary_fromR['study'] = [None]
            model_summary_fromR['scale'] = [None]
            model_summary_fromR['model_type'] = [model_type]
            model_summary_fromR['guesser'] = [None]
            model_summary = pd.concat([model_summary, pd.DataFrame.from_dict(model_summary_fromR)], sort=False)

            # extract model components df
            comps = r('rownames(model_sum$coefficients)')
            for comp in comps:

                model_components_fromR = Helpers.get_model_component_stats(comp=comp)

                if comp=='(Intercept)':
                    comp='intercept'

                model_components_fromR['study'] = [None]
                model_components_fromR['scale'] = [None]
                model_components_fromR['guesser'] = [None]
                model_components_fromR['model_type'] = [model_type]
                model_components_fromR['component'] = [comp]
                model_components = pd.concat([model_components, pd.DataFrame.from_dict(model_components_fromR)], sort=False)

            r('rm(model_sum)')

        r('rm(without_guess)')
        r('rm(with_guess)')

        model_summary.index = range(model_summary.index.shape[0])
        model_components.index = range(model_components.index.shape[0])

        return model_summary, model_components

    @staticmethod
    def get_strata_stats(try_covs, add_cgrc_columns):
        """ Get strata contrast and summary dataframes

            Args:
                try_covs (list): list of strings, each string is a potential covariate (i.e. column name in df_filtered)
                    that will be tried as a covariate for models.

            Returns:
                strata_summary (mydfs.ModelSummaryDf): strata summary dataframe; child of pandas df
                strata_contrast (mydfs.ModelComponentsDf): strata contrast dataframe; child of pandas df
        """

        # Init output
        strata_summary = mydfs.StrataSummaryDf()
        strata_contrast = mydfs.StrataContrastDf()

        if add_cgrc_columns:
            strata_summary.add_univalue_columns({'cgr':None, 'cgr_trial_id':None})
            strata_contrast.add_univalue_columns({'cgr':None, 'cgr_trial_id':None})

        # Can not calc strata stats if CGR==0 or 1
        py_df_filtered = Helpers.r2pyjson('df_filtered')
        if 'cgr' in py_df_filtered.keys():
            assert len(set(py_df_filtered['cgr']))==1
            if (0 in set(py_df_filtered['cgr'])) or (1 in set(py_df_filtered['cgr'])):
                return strata_summary, strata_contrast

        # Calc strata outputs
        covariate_selector.Controllers.get_adjusted_model(
            model_name='with_guess',
            try_covs=try_covs,
            enforce_covs=['condition', 'guess', 'condition*guess'])
        r('emmFixGuess = emmeans(with_guess, specs = pairwise ~ condition|guess)') # stratas with fixed guess
        r('emmFixCond  = emmeans(with_guess, specs = pairwise ~ guess|condition)') # stratas with fixed condition

        # Calc strata outputs - w Tukey p-value adj for multiple comparisons
        r('emm_Tukey = emmeans(with_guess, specs= c("condition", "guess"))')
        tukey_contrasts = Helpers.r2pyjson('summary(pairs(emm_Tukey))')
        tukey_contrasts = Helpers.format_Tukey_contrast(tukey_contrasts)
        r('rm(with_guess)')
        r('rm(emm_Tukey)')

        for emmName in ['emmFixCond', 'emmFixGuess']:

            # Get strata contrast numbers
            r("contrastSummary = summary({}$contrasts)".format(emmName))
            starata_contrast_fromR = Helpers.r2pyjson('contrastSummary')
            starata_contrast_fromR = Helpers.format_comparisons(starata_contrast_fromR, tukey_contrasts)
            r('rm(contrastSummary)')

            # Format strata contrast to be compatible with df
            starata_contrast_fromR['est'] = starata_contrast_fromR.pop('estimate')
            starata_contrast_fromR['se'] = starata_contrast_fromR.pop('SE')
            starata_contrast_fromR['t'] = starata_contrast_fromR.pop('t.ratio')
            starata_contrast_fromR['p'] = starata_contrast_fromR.pop('p.value')
            starata_contrast_fromR['contrast'] = starata_contrast_fromR.pop('comparison')
            starata_contrast_fromR['p_adj'] = starata_contrast_fromR.pop('adj_p')
            starata_contrast_fromR['study'] = [None, None]
            starata_contrast_fromR['scale'] = [None, None]
            starata_contrast_fromR['guesser'] = [None, None]
            strata_contrast = pd.concat([strata_contrast, pd.DataFrame.from_dict(starata_contrast_fromR)], sort=False)


            # Get strata summary
            if emmName=='emmFixCond':
                continue # No need to calculate strata stats

            r("strataSummary = summary({}$emmeans)".format(emmName))
            starata_summary_fromR = Helpers.r2pyjson('strataSummary')
            r('rm(strataSummary)')

            # Format strata summary to be compatible with df
            starata_summary_fromR['est'] = starata_summary_fromR.pop('emmean')
            starata_summary_fromR['se'] = starata_summary_fromR.pop('SE')
            starata_summary_fromR['lower_CI'] = starata_summary_fromR.pop('lower.CL')
            starata_summary_fromR['upper_CI'] = starata_summary_fromR.pop('upper.CL')
            starata_summary_fromR['study'] = [None, None, None, None]
            starata_summary_fromR['scale'] = [None, None, None, None]
            starata_summary_fromR['guesser'] = [None, None, None, None]
            starata_summary_fromR['strata'] = [
                starata_summary_fromR['condition'][0] + starata_summary_fromR['guess'][0],
                starata_summary_fromR['condition'][1] + starata_summary_fromR['guess'][1],
                starata_summary_fromR['condition'][2] + starata_summary_fromR['guess'][2],
                starata_summary_fromR['condition'][3] + starata_summary_fromR['guess'][3],
            ]
            del starata_summary_fromR['condition']
            del starata_summary_fromR['guess']

            strata_summary = pd.concat([strata_summary, pd.DataFrame.from_dict(starata_summary_fromR)], sort=False)

        r('rm(emmFixGuess)')
        r('rm(emmFixCond)')
        del strata_contrast['condition']
        del strata_contrast['guess']

        strata_contrast.index = range(strata_contrast.index.shape[0])
        strata_summary.index = range(strata_summary.index.shape[0])

        return strata_summary, strata_contrast


class Helpers():
    """ Helper functions """

    @staticmethod
    def load_df_into_R_space(input_fpath, relevel_sex=False):
        """ Loads dataframe into R global space

            Args:
                input_fpath (str): filepath to dataframe saved as CSV
                relevel_sex (bool, optional): if True, then 'sex' is converted to factorial with male as baseline
        """

        r('df = read.csv("'+input_fpath+'")')

        r('df$condition = as.factor(df$condition)')
        r('df <- within(df, condition <- relevel(condition, ref="PL"))')

        r('df$guess = as.factor(df$guess)')
        r('df <- within(df, guess <- relevel(guess, ref="PL"))')

        if relevel_sex:
            r('df$sex = as.factor(df$sex)')
            r('df <- within(df, sex <- relevel(sex, ref="M"))')

    @staticmethod
    def r2pyjson(r_var):
        """ Converts an R object to Python json

            Args:
                r_var(str): name of the variable in R global space

            Returns:
                JSON of r_var (in Python namespace)
        """

        rjson = r('toJSON({})'.format(r_var))
        return json.loads(rjson[0])

    @staticmethod
    def get_df_filtered(study, scale, respondent, guesser, cgr=None, cgr_trial_id=None):
        """ Selects subset of R dataframe.
            It is assumed that 'df' exists in R global space and it is an instance of the XYZ
            dataframe types. This functions filters df by study, scale, cgr, cgr_trial_id, respondent and guesser.
            Nothing is returned, but df_filtered is created in R global space.

            Args:
                study (str): name of the study. If 'all', then
                scale (str): name of the scale.
                respondent (str): who responded to questionnaire, must be either 'self' or 'ext'
                guesser (str): who guessed treatment, must be either 'self' or 'ext'
                cgr (float, optional): break blind ratio, ignored if None
                cgr_trial_id (int, optional): trial index if bbc_engine DF is the input, ignored if None
        """

        if (study in ['all', 'sbmd']) and (cgr is None) and (cgr_trial_id is None):
            filter_string = 'df_filtered = filter(df, scale=="{}" & respondent=="{}" & guesser=="{}")'.format(
            scale, respondent, guesser)

        elif (study in ['all', 'sbmd']) and (cgr is not None) and (cgr_trial_id is not None):
            filter_string = 'df_filtered = filter(df, scale=="{}" & cgr=={} & cgr_trial_id=={} & respondent=="{}" & guesser=="{}")'.format(
                scale, cgr, cgr_trial_id, respondent, guesser)

        elif (study not in ['all', 'sbmd']) and (cgr is None) and (cgr_trial_id is None):
            filter_string = 'df_filtered = filter(df, study=="{}" & scale=="{}" & respondent=="{}" & guesser=="{}")'.format(
                study, scale, respondent, guesser)

        elif (study not in ['all', 'sbmd']) and (cgr is not None) and (cgr_trial_id is not None):
            filter_string = 'df_filtered = filter(df, study=="{}" & scale=="{}" & cgr=={} & cgr_trial_id=={} & respondent=="{}" & guesser=="{}")'.format(
                study, scale, cgr, cgr_trial_id, respondent, guesser)

        else:
            assert False # Invalid input

        r('{}'.format(filter_string))

    @staticmethod
    def get_model_summary_stats():
        """ Returns stats for model summary """

        f     = r('model_sum$fstatistic[1]')
        df1   = r('model_sum$fstatistic[2]')
        df2   = r('model_sum$fstatistic[3]')
        adjr2 = r('model_sum$adj.r.squared')
        r('fstats = model_sum$fstatistic')
        p = r('pf(fstats[1], fstats[2], fstats[3], lower=FALSE)')

        summaryStatsDict = {'f':f[0], 'df1':df1[0], 'df2':df2[0], 'adjr2':adjr2[0], 'p':p[0]}
        return summaryStatsDict

    @staticmethod
    def get_model_component_stats(comp):
        """ Returns model component stats """

        n_row = r('which(rownames(model_sum$coefficients)=="{}")'.format(comp))

        n_col = r('which(colnames(model_sum$coefficients)=="{}")'.format("Estimate"))
        est = r('model_sum$coefficients[{},{}]'.format(n_row[0], n_col[0]))

        n_col = r('which(colnames(model_sum$coefficients)=="{}")'.format("Std. Error"))
        se = r('model_sum$coefficients[{},{}]'.format(n_row[0], n_col[0]))

        n_col = r('which(colnames(model_sum$coefficients)=="{}")'.format("t value"))
        t = r('model_sum$coefficients[{},{}]'.format(n_row[0], n_col[0]))

        n_col = r('which(colnames(model_sum$coefficients)=="{}")'.format("Pr(>|t|)"))
        p = r('model_sum$coefficients[{},{}]'.format(n_row[0], n_col[0]))

        componentStatsDict = {'est':est[0], 'se':se[0], 't':t[0], 'p':p[0]}
        return componentStatsDict

    @staticmethod
    def format_comparisons(py_contrast_dict, tukey_contrasts):
        """ Helper function to format compariosn strings """

        contrast = py_contrast_dict
        contrast['comparison'] = []
        contrast['type'] = []
        contrast['adj_p'] =[]

        # Fixed guess case
        if 'guess' in contrast.keys():

            for idx, guess in enumerate(contrast['guess']):
                condition1 = contrast['contrast'][idx][:2]
                condition2 = contrast['contrast'][idx][5:]
                assert condition1 in ['PL', 'AC']
                assert condition2 in ['PL', 'AC']

                comparison = condition1 + guess + 'vs' + condition2 + guess
                contrast['type'].append('fixGuess')
                contrast['comparison'].append(comparison)

                tukey_idx = Helpers.find_contrast_idx(tukey_contrasts, comparison)
                contrast['adj_p'].append(tukey_contrasts['p.value'][tukey_idx])

        # Fixed condition case
        if 'condition' in contrast.keys():

            for idx, condition in enumerate(contrast['condition']):
                guess1 = contrast['contrast'][idx][:2]
                guess2 = contrast['contrast'][idx][5:]
                assert guess1 in ['PL', 'AC']
                assert guess2 in ['PL', 'AC']

                comparison = condition + guess1 + 'vs' + condition + guess2
                contrast['type'].append('fixCondition')
                contrast['comparison'].append(comparison)

                tukey_idx = Helpers.find_contrast_idx(tukey_contrasts, comparison)
                contrast['adj_p'].append(tukey_contrasts['p.value'][tukey_idx])

        return contrast

    @staticmethod
    def find_contrast_idx(tukey_contrasts, comparison):
        """ Returns ID of comparison (str) within tukey_contrasts """

        strata1=comparison[:4]
        strata2=comparison[6:]
        alt_comparison = strata2+'vs'+strata1 #same comparison wit strata order switched

        id=None
        for idx, contrast in enumerate(tukey_contrasts['contrast']):
            if (contrast==comparison) or (contrast==alt_comparison):
                id=idx

        if id is None:
            assert False

        return id

    @staticmethod
    def format_Tukey_contrast(tukey_contrasts):
        """ Format Tukey contrast output """

        formatted_contrasts = copy.deepcopy(tukey_contrasts['contrast'])

        for idx, original_contrast in enumerate(tukey_contrasts['contrast']):

            if original_contrast in ['PL,PL - AC,PL', 'AC,PL - PL,PL']:
                formatted_contrasts[idx]='PLPLvsACPL'

            elif original_contrast in ['PL,PL - PL,AC', 'PL,AC - PL,PL']:
                formatted_contrasts[idx]='PLPLvsPLAC'

            elif original_contrast in ['PL,PL - AC,AC', 'AC,AC - PL,PL']:
                formatted_contrasts[idx]='PLPLvsACAC'

            elif original_contrast in ['AC,PL - PL,AC', 'PL,AC - AC,PL']:
                formatted_contrasts[idx]='ACPLvsPLAC'

            elif original_contrast in ['AC,PL - AC,AC', 'AC,AC - AC,PL']:
                formatted_contrasts[idx]='ACPLvsACAC'

            elif original_contrast in ['PL,AC - AC,AC', 'AC,AC - PL,AC']:
                formatted_contrasts[idx]='PLACvsACAC'

            else:
                assert False

        tukey_contrasts['contrast'] = formatted_contrasts
        return tukey_contrasts

    @staticmethod
    def add_metadata_process_trialdata(all_dfs, study, scale, guesser, respondent):
        """ Add metadata to model_summary, model_components, strata_summary, strata_contrast """

        all_dfs['model_summary'].study = study
        all_dfs['model_summary'].scale = scale
        all_dfs['model_summary'].guesser = guesser
        all_dfs['model_summary'].respondent = respondent

        all_dfs['model_components'].study = study
        all_dfs['model_components'].scale = scale
        all_dfs['model_components'].guesser = guesser
        all_dfs['model_components'].respondent = respondent

        all_dfs['strata_summary'].study = study
        all_dfs['strata_summary'].scale = scale
        all_dfs['strata_summary'].guesser = guesser
        all_dfs['strata_summary'].respondent = respondent

        all_dfs['strata_contrast'].study = study
        all_dfs['strata_contrast'].scale = scale
        all_dfs['strata_contrast'].guesser = guesser
        all_dfs['strata_contrast'].respondent = respondent

        return all_dfs

    @staticmethod
    def add_metadata_process_cgrc(all_dfs, study, scale, guesser, respondent, cgr, cgr_trial_id):
        """ Add metadata to model_summary, model_components, strata_summary, strata_contrast """

        all_dfs['model_summary'].study = study
        all_dfs['model_summary'].scale = scale
        all_dfs['model_summary'].cgr = cgr
        all_dfs['model_summary'].cgr_trial_id = cgr_trial_id
        all_dfs['model_summary'].guesser = guesser
        all_dfs['model_summary'].respondent = respondent

        all_dfs['model_components'].study = study
        all_dfs['model_components'].scale = scale
        all_dfs['model_components'].cgr = cgr
        all_dfs['model_components'].cgr_trial_id = cgr_trial_id
        all_dfs['model_components'].guesser = guesser
        all_dfs['model_components'].respondent = respondent

        all_dfs['strata_summary'].study = study
        all_dfs['strata_summary'].scale = scale
        all_dfs['strata_summary'].cgr = cgr
        all_dfs['strata_summary'].cgr_trial_id = cgr_trial_id
        all_dfs['strata_summary'].guesser = guesser
        all_dfs['strata_summary'].respondent = respondent

        all_dfs['strata_contrast'].study = study
        all_dfs['strata_contrast'].scale = scale
        all_dfs['strata_contrast'].cgr = cgr
        all_dfs['strata_contrast'].cgr_trial_id = cgr_trial_id
        all_dfs['strata_contrast'].respondent = respondent
        all_dfs['strata_contrast'].guesser = guesser

        return all_dfs


class NoSelectedRows(Exception):
    pass
