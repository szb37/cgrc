"""
:Author: Balazs Szigeti <szb37@pm.me>
:Copyright: 2020, DrugNerdsLab
:License: MIT

Preprocessing: format away the idiosyncrasies of each CSV to standard format
Processing: organize CSVs into stratification CSV format
"""

import src.data_transform.ad_studies_constants as ad_constants
import src.my_dataframes as mydfs
import src.constants as constants
import src.folders as folders
from statistics import mean
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
import shutil
import math
import copy
import os


class DataTransformCore():
    """ Methods to transform NIMH CSVs for downstream analysis """

    @staticmethod
    def full_process(analysis_name, study_csv_scales, output_dir=folders.csvs_trial_data):
        DataTransformCore.preprocess(study_csv_scales=study_csv_scales)
        DataTransformCore.process(
            analysis_name=analysis_name,
            study_csv_scales=study_csv_scales,
            output_dir=output_dir)

    @staticmethod
    def preprocess(study_csv_scales, output_dir=folders.csvs_preprocessed):
        """ Preprocesses CSV files to standard format """

        for study_name, outcomes in study_csv_scales.items(): # Outcomes should be called CSVs here
            print("Preprocessing study: {}".format(study_name))
            DataTransformCore.preprocess_study(study_name=study_name, outcomes=outcomes, output_dir=output_dir)

    @staticmethod
    def process(analysis_name, study_csv_scales, output_dir=folders.csvs_trial_data):
        """ Processes CSV to output format """

        # Check if target folder exists; create if not; rm files inside if it does
        #target_dir = os.path.join(folders.csvs_trial_data, analysis_name)
        #if os.path.isdir(target_dir):
        #    [os.remove(os.path.join(target_dir ,filename)) for filename in os.listdir(target_dir)]
        #else:
        #    os.mkdir(target_dir)

        df = mydfs.TrialDataDf()

        for study_name, outcomes in study_csv_scales.items(): # Outcomes should be called CSVs here
            DataTransformCore.process_study(analysis_name=analysis_name, study_name=study_name, output_dir=output_dir, outcomes=outcomes, df=df)

    @staticmethod
    def process_study(df, analysis_name, study_name, outcomes, output_dir):

        print('\nProcessing study: {}'.format(study_name).upper())

        guess_df = pd.read_csv(os.path.join(
            folders.csvs_preprocessed, study_name.lower()+'_guess_preprocessed.csv'))
        trt_df = pd.read_csv(os.path.join(
            folders.csvs_preprocessed, study_name.lower()+'_treatment_preprocessed.csv'))

        # Loop through all scores CSVs
        for outcome_filename, scales_list in outcomes.items():
            score_df = pd.read_csv(os.path.join(folders.csvs_preprocessed, outcome_filename+'_preprocessed.csv'))
            df = DataTransformCore.process_score_csv(
                df=df,
                study_name=study_name,
                outcome_filename=outcome_filename,
                score_df=score_df,
                scales_list=scales_list)


        # Process treatments and guesses
        df = DataTransformCore.process_guess_csv(df=df, guess_df=guess_df, study_name=study_name)
        df = DataTransformCore.process_treatment_csv(df=df, trt_df=trt_df, study_name=study_name)

        df = Helpers.rm_rows_WoGuessConditionScore(df=df)
        df = Helpers.calc_delta_scores(df=df)
        df = Helpers.rm_rows_WoDeltaScore(df=df)

        df = Helpers.substitute_collectId2name(df=df, study_name=study_name)
        df.set_column_types()
        df.check_assumptions()

        df.to_csv(os.path.join(output_dir, study_name+'__trial_data.csv'), index=False)

    @staticmethod
    def process_score_csv(df, study_name, outcome_filename, score_df, scales_list):
        """ Adds rows to the df based on scores """

        scales_names = Helpers.get_scale_names_str(scales_list)
        print("Processing {} scale(s) from {}".format(scales_names, outcome_filename))
        guessers = ['self', 'ext']

        for index, row in tqdm(score_df.iterrows(), total=score_df.shape[0], ncols=80):

            tp = HelperStudy.get_tps_study(row)

            for scale in scales_list:

                value = eval('row.' + scale)
                assert (
                    isinstance(value, np.float64) or
                    isinstance(value, float) or
                    isinstance(value, np.int64) or
                    isinstance(value, int)
                )
                if math.isnan(value):
                    continue

                respondent = HelperStudy.get_respondent_study(
                    study_name=study_name,
                    outcome_filename=outcome_filename,
                    scale=scale,
                    row=row)

                df = Helpers.add_TrialDataDf_rows(
                    df=df,
                    collection_id=row.collection_id,
                    subject_id=row.src_subject_id,
                    tp=tp,
                    scale=scale,
                    score=value,
                    respondent=respondent,
                    guessers=guessers,)

        return df

    @staticmethod
    def process_guess_csv(df, guess_df, study_name):
        """ Processes guess information """
        print("Processing guesses")

        tp_encoding = Helpers.get_tps_encoding(study_name=study_name)
        tp_column = tp_encoding['tp_column']
        valid_tps = tp_encoding['valid_tps']
        is_why = Helpers.check_is_why(guess_df)

        for index, row in tqdm(guess_df.iterrows(), total=guess_df.shape[0], ncols=80):

            tp = HelperStudy.get_tps_study(row)

            matched_rows = df.loc[
                (df['study'] == row.collection_id) &
                (df['tp'] == tp) &
                (df['subject_id'] == row.src_subject_id)]

            if matched_rows.shape[0] == 0:
                continue

            guess_info = HelperStudy.get_guess_study(row)
            guesser = guess_info[0]
            guess = guess_info[1]
            guess_conf = guess_info[2]

            if (guesser is None) or (guess is None):
                continue

            why_guess = None
            if is_why is True:
                why_guess = Helpers.get_why_guess(row)

            df.loc[
                (df['study'] == row.collection_id) &
                (df['tp'] == tp) &
                (df['subject_id'] == row.src_subject_id) &
                (df['guesser'] == guesser),
                'guess'] = guess

            df.loc[
                (df['study'] == row.collection_id) &
                (df['tp'] == tp) &
                (df['subject_id'] == row.src_subject_id) &
                (df['guesser'] == guesser),
                'guess_conf'] = guess_conf

            df.loc[
                (df['study'] == row.collection_id) &
                (df['tp'] == tp) &
                (df['subject_id'] == row.src_subject_id) &
                (df['guesser'] == guesser),
                'why_guess'] = why_guess

        return df

    @staticmethod
    def process_treatment_csv(df, trt_df, study_name):
        """ Places treatment to the matching subject_id rows """
        print("Processing treatment")

        valid_ids = HelperStudy.get_subject_ids_study(study_name=study_name, trt_df=trt_df)

        for id in tqdm(valid_ids, total=len(valid_ids), ncols=80):

            if math.isnan(id):
                continue

            row = trt_df.loc[trt_df['src_subject_id'] == id]
            treatment = HelperStudy.get_treatment_study(row)

            df.loc[(df['subject_id']==id), 'condition'] = treatment
            row.reset_index(drop=True, inplace=True)
            df.loc[(df['subject_id']==id), 'age'] = row.interview_age[0]
            df.loc[(df['subject_id']==id), 'sex'] = row.sex[0]

        return df


    """ Methods assocaited with preprocessing """
    @staticmethod
    def preprocess_study(study_name, outcomes, output_dir):
        """ Apply appropiate preprocessing transformation to each CSV """

        # First lets get valid subject ids
        trt_df = pd.read_csv(os.path.join(folders.csvs_raw, study_name.lower()+'_treatment.csv'))
        trt_df.drop_duplicates(inplace=True)
        trt_df['collection_id'] = trt_df['collection_id'].astype('int')
        valid_ids = HelperStudy.get_subject_ids_study(study_name=study_name, trt_df=trt_df)

        # Remove irrelevant timepoints and subjects from treatment and guess CSVs
        DataTransformCore.rm_subjects(df=trt_df, valid_ids=valid_ids)
        trt_df.to_csv(os.path.join(output_dir, study_name+'_treatment_preprocessed.csv'), index=False)

        guess_df = pd.read_csv(os.path.join(folders.csvs_raw, study_name+'_guess.csv'))
        guess_df.drop_duplicates(inplace=True)
        guess_df['collection_id'] = guess_df['collection_id'].astype('int')

        DataTransformCore.rm_subjects(df=guess_df, valid_ids=valid_ids)
        DataTransformCore.rm_tps(df=guess_df, study_name=study_name)
        guess_df.to_csv(os.path.join(output_dir, study_name+'_guess_preprocessed.csv'), index=False)

        # Remove irrelevant timepoints and subjects from score CSVs
        for outcome_filename, outcome_column in outcomes.items():
            df=pd.read_csv(os.path.join(folders.csvs_raw, outcome_filename+'.csv'))
            df.drop_duplicates(inplace=True)
            df['collection_id'] = df['collection_id'].astype('int')

            DataTransformCore.rm_subjects(df=df, valid_ids=valid_ids)
            DataTransformCore.rm_tps(df=df, study_name=study_name)

            df.to_csv(os.path.join(output_dir, outcome_filename+'_preprocessed.csv'), index=False)

        """ Apply any additional study specific preprocessing if needed """

        if study_name == 'tads':

            # Need to calc average bdi_tot
            df = pd.read_csv(os.path.join(output_dir, 'tads_bdi_preprocessed.csv'))
            DataTransformCore.avg_score(
                df=df,
                study_name='tads',
                outcome_column='bdi_tot',
                respondent_column='relationship',
                respondents_to_avg = ['mother', 'father', 'other'])
            df.to_csv(os.path.join(output_dir, 'tads_bdi_preprocessed.csv'), index=False)

        elif study_name == 'stoppd':
            pass
        elif study_name == 'rtca':
            # Remove degenerate timepoints

            for outcome_filename, outcome_column in outcomes.items():
                df = pd.read_csv(os.path.join(output_dir, outcome_filename+'_preprocessed.csv'))
                DataTransformCore.rm_degenrate_tps(
                    df=df,
                    study_name=study_name,
                    tp_column='visit',
                    valid_tps=[2,10],
                    days_since_bsl_column='daysrz',
                )
                df.to_csv(os.path.join(output_dir, outcome_filename+'_preprocessed.csv'), index=False)
        elif study_name == 'ruppats':

            # Visits are mixed up;
            # bsl is either visit=0/4
            # pep is either visit=8/12; use visitday to resolve ambiguity

            for outcome_filename, outcome_column in outcomes.items():
                df = pd.read_csv(os.path.join(output_dir, outcome_filename+'_preprocessed.csv'))

                # Re-encode all bsl to visit=0; ie rewrite visit=4 to visit=0
                df.loc[(df.visit==4), 'visit'] = 0

                # Re-encode all pep to visit=8; ie rewrite visit=12 to visit=8
                df.loc[(df.visit==12), 'visit'] = 8

                DataTransformCore.rm_degenrate_tps(
                    df=df,
                    study_name=study_name,
                    tp_column='visit',
                    valid_tps=[0,8],
                    days_since_bsl_column='visitday',
                )
                df.to_csv(os.path.join(output_dir, outcome_filename+'_preprocessed.csv'), index=False)
        elif study_name == 'cams':
            # Need to calc avearge cgas
            df = pd.read_csv(os.path.join(output_dir, 'cams_cgis_preprocessed.csv'))
            DataTransformCore.avg_score(
                study_name='cams',
                df=df,
                outcome_column='cgas',
                respondent_column='aefther',
                respondents_to_avg = [0, 1, 2]) # CGAS in cams_cgis
            df.to_csv(os.path.join(output_dir, 'cams_cgis_preprocessed.csv'), index=False)

            df = pd.read_csv(os.path.join(output_dir, 'cams_cgi_preprocessed.csv'))
            DataTransformCore.avg_score(
                study_name='cams',
                df=df,
                outcome_column='cgi_si',
                respondent_column='cgither',
                respondents_to_avg = [0, 1, 2]) # CGI_SI in cams_cgi
            df.to_csv(os.path.join(output_dir, 'cams_cgi_preprocessed.csv'), index=False)

            df = pd.read_csv(os.path.join(output_dir, 'cams_cgii_preprocessed.csv'))
            DataTransformCore.avg_score(
                study_name='cams',
                df=df,
                outcome_column='cgi_sii',
                respondent_column='cgither',
                respondents_to_avg = [0, 1, 2]) # CGI_SII in cams_cgii
            df.to_csv(os.path.join(output_dir, 'cams_cgii_preprocessed.csv'), index=False)

            df = pd.read_csv(os.path.join(output_dir, 'cams_mfq_preprocessed.csv'))
            DataTransformCore.avg_score(
                study_name='cams',
                df=df,
                outcome_column='mfqtot',
                respondent_column='respondent',
                respondents_to_avg = ['Mother', 'Father', 'Both', 'Other'])
            df.to_csv(os.path.join(output_dir, 'cams_mfq_preprocessed.csv'), index=False)

            df = pd.read_csv(os.path.join(output_dir, 'cams_stai_preprocessed.csv'))
            DataTransformCore.avg_score(
                study_name='cams',
                df=df,
                outcome_column='staittot',
                respondent_column='respondent',
                respondents_to_avg = ['Mother', 'Father', 'Both', 'Other'])
            df.to_csv(os.path.join(output_dir, 'cams_stai_preprocessed.csv'), index=False)

            df = pd.read_csv(os.path.join(output_dir, 'cams_pars_preprocessed.csv'))
            DataTransformCore.avg_score(
                study_name='cams',
                df=df,
                outcome_column='severity_totscr',
                respondent_column='version_form',
                respondents_to_avg = ['Parent', 'IE'])
            df.to_csv(os.path.join(output_dir, 'cams_pars_preprocessed.csv'), index=False)
        elif 'test_pipeline' in study_name:
            pass
        else:
            raise False # Invalid study name

    @staticmethod
    def rm_subjects(df, valid_ids):
        """ Removes rows whose TP is not either baseline or primary endpoint """

        invalid_rows_idx = df.loc[~df.src_subject_id.isin(valid_ids)].index.tolist()
        Helpers.del_rows(df, idx2del=invalid_rows_idx)

    @staticmethod
    def rm_tps(df, study_name):
        """ Removes rows whose TP is not either baseline or primary endpoint """

        tp_encoding = Helpers.get_tps_encoding(study_name)
        tp_column = tp_encoding['tp_column']
        valid_tps = tp_encoding['valid_tps']

        # Get row indecies
        rows_idx = df.index.to_list()
        valid_rows_idx = eval('df.loc[df.'+tp_column+'.isin(valid_tps)].index.tolist()')
        invalid_rows_idx = list(set(rows_idx).difference(set(valid_rows_idx)))

        Helpers.del_rows(df, idx2del=invalid_rows_idx)

    @staticmethod
    def rm_degenrate_tps(df, study_name, tp_column, valid_tps, days_since_bsl_column):
        """ Timepoint encoding is sometimes degenerate, as same numbers are used for phases 1&2.
            This code removes timepoints that are not part of phase1 (baseline to pep)
            Only works if non tps_days encoded Tps have been removed
        """

        assert valid_tps == [tp_value for tp_value in ad_constants.tps_days[study_name].keys()]

        df['delta_days'] = 9999
        dd_idx = df.columns.get_loc('delta_days')
        ids = df.src_subject_id.unique().tolist()
        idx2del = []

        for index, row in df.iterrows(): # Get Delta days where needed

            if eval('row.'+tp_column) in valid_tps:

                tp_value = eval('row.'+tp_column)
                days_since_bsl = eval('row.'+days_since_bsl_column)
                days_since_bsl_protocol = ad_constants.tps_days[study_name][tp_value]

                df.iloc[index, dd_idx] = abs(days_since_bsl - days_since_bsl_protocol)

        # Collect and del non-minimum duplicate rows
        for id, visit in itertools.product(ids, valid_tps):

            rows = df.loc[
                (df['src_subject_id'] == id) &
                (df[tp_column] == visit)
            ]

            if rows.shape[0] in [0, 1]:
                continue

            discard_rows = rows.loc[rows.delta_days != rows.delta_days.min()]

            if rows.shape[0] != discard_rows.shape[0]+1:
                assert False  # Unambigous which row is real timepoint

            for idx in discard_rows.index.tolist():
                idx2del.append(idx)

        Helpers.del_rows(df, idx2del=idx2del)

    @staticmethod
    def avg_score(df, study_name, outcome_column, respondent_column, respondents_to_avg):
        """ Gets avg score across multiple responses where column in valid_values
            (e.g. father and mother were both interviewed).
            Deletes rows where score value is missing!!
        """

        ids = df.src_subject_id.unique()
        respondents_idx = df.columns.get_loc(respondent_column)

        tp_encoding = Helpers.get_tps_encoding(study_name)
        tp_column = tp_encoding['tp_column']
        valid_tps = tp_encoding['valid_tps']

        assert 'avg_score' not in df.columns  # Can handle only one avg_score in sheet

        score_idx = df.columns.get_loc(outcome_column)  # Index of original score column
        df['avg_score'] = math.nan
        avg_idx = df.columns.get_loc('avg_score')  # Index of new avg score column

        # Copy score to avg_score where avg = original value
        for idx, row in df.iterrows():

            if eval('row.'+tp_column) not in valid_tps:  # if TP is not valid
                df.iloc[idx, avg_idx] = row.iloc[score_idx]

            if eval('row.'+respondent_column) not in respondents_to_avg:  # if respondent is not in avg
                df.iloc[idx, avg_idx] = row.iloc[score_idx]

        # Get average when needed
        for id, tp in itertools.product(ids, valid_tps):

            matching_rows = eval(
                'df.loc[ \
                    (df.src_subject_id == id) & \
                    (~df.'+outcome_column+'.isnull()) & \
                    (df.'+tp_column+' == tp) & \
                    (df.'+respondent_column+'.isin(respondents_to_avg)) \
                ]')

            if matching_rows.shape[0] == 0:
                continue

            avg_score = mean([int(value) for value in matching_rows.iloc[:, score_idx].tolist()])
            row_idx = df.index.get_loc(matching_rows.iloc[0].name)
            df.iloc[row_idx, avg_idx] = avg_score

            if matching_rows.shape[0] != 1:
                df.iloc[row_idx, respondents_idx] = 'comb'

        # Del rows wo avg_score; only skipped for rows that are included in avgs
        no_avg_rows = df.loc[df.avg_score.isnull()]
        Helpers.del_rows(df, idx2del=no_avg_rows.index.tolist())
        df.rename(columns = {'avg_score': outcome_column+'_avg'}, inplace=True)


class HelperStudy():
    """ Helper functions to extract study specific information """

    @staticmethod
    def get_guess_study(row):
        """ Returns (guesser, guess, guess_conf) based on study specific information """

        if row.collection_id == 9999:    # TESTS

            guess_conf = row.guess_conf

            if row.version_form.lower() == 'clinician':
                guesser = 'ext'
            elif row.version_form.lower() == 'patient':
                guesser = 'self'
            elif math.isnan(row.tx_guess):
                guess = None; guesser = None; guess_conf = None
            else:
                assert False

            if row.tx_guess == 'AC':
                guess = 'AC'
            elif row.tx_guess == 'PL':
                guess = 'PL'
            elif math.isnan(row.tx_guess):
                guess = None; guesser = None; guess_conf = None
            else:
                assert False



        elif row.collection_id == 2145:  # TADS; doctor's guess encoded as 'ext', but also has parents' guess

            if row.version_form == 'Adolescent':

                guesser = 'self'
                guess_conf = row.adolcert

                if row.adoltrt == 1:
                    guess = 'AC'
                elif row.adoltrt == 2:
                    guess = 'PL'
                elif math.isnan(row.adoltrt):
                    guess = None; guesser = None; guess_conf = None
                else:
                    raise False

            elif row.version_form == 'Pharmacotherapist':

                guesser = 'ext'
                guess_conf = row.txcertnt

                if row.txguess == 1:
                    guess = 'AC'
                elif row.txguess == 2:
                    guess = 'PL'
                elif math.isnan(row.txguess):
                    guess = None; guesser = None; guess_conf = None
                else:
                    raise False

            elif row.version_form == 'Parent':

                guess = None; guesser = None; guess_conf = None
        elif row.collection_id == 2278:  # STOPPD

            guesser = 'self'
            guess_conf = row.confidence

            if row.guess == 1:
                guess = 'PL'
            elif row.guess == 2:
                guess = 'AC'
            elif math.isnan(row.guess):
                guess = None; guesser = None; guess_conf = None
            else:
                assert False
        elif row.collection_id == 2151:  # RTCA

            guess_conf = None

            if row.version_form == 'patient':
                guesser = 'self'
            elif row.version_form == 'Clinician':
                guesser = 'ext'
            elif  math.isnan(row.version_form):
                guess = None; guesser = None; guess_conf = None
            else:
                assert False

            if row.ie1b  == 2:
                guess = 'AC'
            elif row.ie1b  == 4:
                guess = 'PL'
            elif (row.ie1b==5) or math.isnan(row.ie1b):
                guess = None; guesser = None; guess_conf = None
            else:
                assert False
        elif row.collection_id == 2152:  # RUPPATS; doctor's guess encoded as 'ext', but also has parents' guess

            guess_conf = None

            if row.version_form == 'subject':
                guesser = 'self'
            elif row.version_form == 'interviewer':
                guesser = 'ext'
            elif row.version_form == 'parent':
                guess = None; guesser = None; guess_conf = None
            elif math.isnan(row.version_form):
                guess = None; guesser = None; guess_conf = None
            else:
                assert False

            if row.ie1b == 2:
                guess = 'AC'
            elif row.ie1b  == 4:
                guess = 'PL'
            elif (row.ie1b==5) or math.isnan(row.ie1b):
                guess = None; guesser = None; guess_conf = None
            else:
                assert False
        elif row.collection_id == 2160:  # CAMS

            if row.pt1 in [1, 4]:
                guesser = 'ext'
                guess = 'AC'
                guess_conf = row.pt2
            elif row.pt1 == 2:
                guesser = 'ext'
                guess = 'PL'
                guess_conf = row.pt2
            else:
                guess = None; guesser = None; guess_conf = None
        else:
            assert False

        return (guesser, guess, guess_conf)

    @staticmethod
    def get_subject_ids_study(study_name, trt_df):

        if study_name == 'tads':
            return trt_df.loc[(trt_df['txcode'].isin(['PBO', 'FLX']))].src_subject_id.to_list()
        if study_name in ['test_pipeline2', 'test_pipeline3']:
            return trt_df.loc[(trt_df['txcode'].isin(['PL', 'AC']))].src_subject_id.to_list()
        elif study_name in ['stoppd', 'rtca', 'ruppats', 'cams', 'test_pipeline1']:
            return trt_df.src_subject_id.unique().tolist()
        else:
            assert False

    @staticmethod
    def get_treatment_study(row):
        """ Extracts and returns treatment """

        if row.collection_id.to_list()[0] == 9999:   # TESTS
            if row.txcode.to_list()[0] == 'PL':
                treatment = 'PL'
            elif row.txcode.to_list()[0] == 'AC':
                treatment = 'AC'
            else:
                assert False

        elif row.collection_id.to_list()[0] == 2145: # TADS
            if row.txcode.to_list()[0] == 'PBO':
                treatment = 'PL'
            elif row.txcode.to_list()[0] == 'FLX':
                treatment = 'AC'
            else:
                assert False
        elif row.collection_id.to_list()[0] == 2278: # STOPPD
            if row.tx_code.to_list()[0] == 0:
                treatment = 'PL'
            elif row.tx_code.to_list()[0] == 1:
                treatment = 'AC'
            elif row.tx_code.to_list()[0] == 2:
                treatment = 'AC'
            else:
                assert False
        elif row.collection_id.to_list()[0] == 2151: # RTCA
            if row.treatment_name.to_list()[0] == 'Placebo':
                treatment = 'PL'
            elif row.treatment_name.to_list()[0] == 'Risperidone':
                treatment = 'AC'
            else:
                assert False
        elif row.collection_id.to_list()[0] == 2152: # RUPPATS
            if row.medication_name.to_list()[0] == 'Placebo':
                treatment = 'PL'
            elif row.medication_name.to_list()[0] == 'Luvox':
                treatment = 'AC'
            else:
                assert False
        elif row.collection_id.to_list()[0] == 2160: # CAMS
            if row.txcode.to_list()[0] == 'PL':
                treatment = 'PL'
            elif row.txcode.to_list()[0] == 'AC':
                treatment = 'AC'
            else:
                assert False
        else:
            assert False # Unknown study

        return treatment

    @staticmethod
    def get_tps_study(row):
        """ Returns week encoding based on study specific information """

        if row.collection_id == 9999:    # TESTS

            if row.visit in [0, 1]:
                return 'bsl'
            elif row.visit == 5:
                return 'wk5'
            elif row.visit == 6:
                return 'wk6'
            elif row.visit == 10:
                return 'wk10'
            else:
                assert False

        elif row.collection_id == 2145:  # TADS
            if row.asstyp == 3:
                return 'bsl'
            elif row.asstyp == 4:
                return 'wk6'
            elif row.asstyp == 5:
                return 'wk12'
            else:
                assert False
        elif row.collection_id == 2278:  # STOPPD
            if row.week == 0:
                return 'bsl'
            elif row.week == 12:
                return 'wk12'
            else:
                assert False
        elif row.collection_id == 2151:  # RTCA
            if row.visit == 2:
                return 'bsl'
            elif row.visit == 10:
                return 'wk12'
            else:
                assert False
        elif row.collection_id == 2152:  # RUPPATS
            if row.visit == 0:
                return 'bsl'
            elif row.visit == 8:
                return 'wk8'
            else:
                assert False
        elif row.collection_id == 2160:  # CAMS
            if row.visit == -1:
                return 'bsl'
            elif row.visit == 12:
                return 'wk12'
            else:
                assert False
        else:
            assert False

    @staticmethod
    def get_respondent_study(study_name, outcome_filename, scale, row):
        """ Returns 'ext' if form was completed by external rater or 'self' if self-report """

        if study_name == 'tads':
            if scale in ['cgi_si', 'cgi_sii_merged', 'cdrs_14p', 'bdi_tot_avg', 'cdrs_r_b']:
                return 'ext'
            elif scale in ['rads_scr', 'anxs_tot', 'masctot', 'cdrs_14a', 'cdrs_r_a']:
                return 'self'
            else:
                assert False

        elif study_name == 'stoppd':
            if scale in ['cgi_si', 'cgi_sii', 'hamd_36', 'hamd_score_24', 'bprs_total']:
                return 'ext'
            else:
                assert False
        elif study_name == 'rtca':
            if scale in ['cgi_si', 'cgi_sii', 'cybocs_comptot', 'lin_sum_score']:
                return 'ext'
            else:
                assert False
        elif study_name == 'ruppats':
            if scale in ['cgi_si', 'cgi_sii', 'cdrs_14p', 'ham_a_score']:
                return 'ext'
            elif scale in ['anxs_tot', 'masctot', 'cdrs_14a', ]:
                return 'self'
            else:
                assert False
        elif (study_name=='cams') and (scale in ['cgi_si_avg', 'cgi_sii_avg', 'cgas_avg']):
            return 'ext'
        elif (study_name=='cams') and (scale in ['nqas', 'nqds']):
            return 'self'
        elif (study_name == 'cams') and scale=='severity_totscr_avg':
            if row.version_form=='comb':
                return 'ext'
            elif row.version_form=='Child':
                return 'self'
            else:
                assert False
        elif (study_name == 'cams') and scale in ['mfqtot_avg', 'staittot_avg']:
            if row.respondent in ['Mother', 'Father', 'Both', 'Other']:
                return 'ext'
            elif row.respondent=='Child':
                return 'self'
            else:
                assert False
        elif (study_name == 'cams') and scale == 'scared_total':
            if row.scared_version == 'Parent':
                return 'ext'
            elif row.scared_version == 'Child':
                return 'self'
            else:
                assert False
        elif (study_name == 'cams') and scale == 'matotalx':
            if row.comments_misc == 'Parent is respondent':
                return 'ext'
            elif row.comments_misc == 'Child is respondent':
                return 'self'
            else:
                assert False
        else:
            assert False


class Helpers():
    """ Helper functions """

    @staticmethod
    def calc_delta_scores(df):
        """ Gets delta from baseline scores """
        print("Get delta scores")

        for index, row in tqdm(df.iterrows(), total=df.shape[0], ncols=80):

            if (row.tp=='bsl') or (math.isnan(row.score)):
                continue

            # First try to get baseline from dataframe
            baseline_list = df.loc[
                (df.study == row.study) &
                (df.scale == row.scale) &
                (df.respondent == row.respondent) & # Could solve the splitting RUPPATS CDRS scales as well
                (df.tp == 'bsl') &
                (df.subject_id == row.subject_id)].score.unique().tolist()

            if (row.study in [2278, 2151, 2145]) and (row.scale in ['cgi_sii', 'cgi_sii_avg', 'cgi_sii_merged']):
                baseline_list = [0]

            if baseline_list==[]:
                continue # baseline not found
            elif len(baseline_list)==1:
                baseline = baseline_list[0]
            elif len(baseline_list)>1:
                assert False # mulitple baselines found
            else:
                assert False

            df.loc[index, 'delta_score'] = row.score - baseline
            df.loc[index, 'baseline'] = baseline

        return df

    @staticmethod
    def add_TrialDataDf_rows(df, collection_id, subject_id, tp, scale, score, guessers, respondent):
        """ Adds row to processed DataFrame """

        for guesser in guessers:
            df = df.append({
                'study': collection_id,
                'subject_id': subject_id,
                'age': None,
                'sex': None,
                'tp': tp,
                'scale': scale,
                'baseline': None,
                'score': score,
                'delta_score': None,
                'condition': None,
                'guess': None,
                'guess_conf': None,
                'why_guess': None,
                'guesser': guesser,
                'respondent': respondent},
                ignore_index=True)

        df.__class__= mydfs.TrialDataDf
        return df

    @staticmethod
    def get_all_scales(outcomes):

        scales=[]
        for csv_name, scale_name_list in outcomes.items():
            for scale_name in scale_name_list:
                scales.append(scale_name)

        return scales

    @staticmethod
    def get_tps_encoding(study_name):

        # Get TP encoding column and valid values
        tp_columns = list(ad_constants.tps_structure[study_name].keys())
        tp_column = [tp_column for tp_column in tp_columns]

        valid_tps = list(ad_constants.tps_structure[study_name].values())
        valid_tps = [valid_tp for valid_tp in valid_tps]

        assert len(tp_column) == 1
        assert len(valid_tps) == 1

        tp_column = tp_column[0]
        valid_tps = valid_tps[0]

        return {'tp_column': tp_column, 'valid_tps': valid_tps}

    @staticmethod
    def rm_rows_WoGuessConditionScore(df):
        """ Remove rows where guess / condition is missing """

        ind2delete_cond = df.loc[
            (df.condition.isna()) &
            (df.tp != 'bsl')
        ].index.to_list()

        ind2delete_guess = df.loc[
            (df.guess.isna()) &
            (df.tp != 'bsl')
        ].index.to_list()

        ind2delete_score = df.loc[
            (df.score.isna()) &
            (df.tp != 'bsl')
        ].index.to_list()

        ind2delete = ind2delete_cond + ind2delete_guess + ind2delete_score

        if ind2delete!=[]:
            ind2delete = list(set(ind2delete))
            df.drop(ind2delete, inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df

    @staticmethod
    def rm_rows_WoDeltaScore(df):
        """ Remove baseline rows """
        ind2delete = df.loc[(df.delta_score.isna())].index.to_list()

        if ind2delete!=[]:
            ind2delete = list(set(ind2delete))
            df.drop(ind2delete, inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df

    @staticmethod
    def substitute_collectId2name(df,  study_name):
        """ Change collection_id in study column to study acronym
            TODO: works, but clucnky; refactor
        """

        df['study'] = df['study'].astype('int')

        df.loc[(df['study'] == 2145), 'study'] = 'tads'
        df.loc[(df['study'] == 2278), 'study'] = 'stoppd'
        df.loc[(df['study'] == 2157), 'study'] = 'hypericum'
        df.loc[(df['study'] == 2151), 'study'] = 'rtca'
        df.loc[(df['study'] == 2152), 'study'] = 'ruppats'
        df.loc[(df['study'] == 2160), 'study'] = 'cams'
        df.loc[(df['study'] == 9999), 'study'] = 'test'

        df['study'] = df['study'].astype('str')

        return df

    @staticmethod
    def get_scale_names_str(scales):
        """ returns scale names as single string """

        scale_names = ''
        for scale in scales:
            scale_names += scale + ', '

        return scale_names[:-2]

    @staticmethod
    def del_rows(df, idx2del, reset_index=True):
        """ Deletes rows with index in idx2del """

        df.drop(idx2del, inplace=True)
        if reset_index is True:
            df.reset_index(drop=True, inplace=True)

    @staticmethod
    def scale_renaming(df):
        """ Renames columns to standard """

        df.loc[(df.scale=='hamd_score_24'), 'scale'] = 'hamd_24'
        df.loc[(df.scale=='bprs_total'), 'scale'] = 'bprs'

        df.loc[(df.scale=='cybocs_comptot'), 'scale'] = 'cybocs'
        df.loc[(df.scale=='lin_sum_score'), 'scale'] = 'rfrlrs'

        df.loc[(df.scale=='cgi_si_avg'), 'scale'] = 'cgi_si'
        df.loc[(df.scale=='cgi_sii_avg'), 'scale'] = 'cgi_sii'
        df.loc[(df.scale=='cgas_avg'), 'scale'] = 'cgas'
        df.loc[(df.scale=='mfqtot'), 'scale'] = 'mfq'
        df.loc[(df.scale=='severity_totscr_avg'), 'scale'] = 'pars'
        df.loc[(df.scale=='staittot_avg'), 'scale'] = 'stait'
        df.loc[(df.scale=='scared_total'), 'scale'] = 'scared'
        df.loc[(df.scale=='matotalx'), 'scale'] = 'masc'

        df.loc[(df.scale=='cgi_sii_merged'), 'scale'] = 'cgi_sii'
        df.loc[(df.scale=='anxs_tot'), 'scale'] = 'masc39'
        df.loc[(df.scale=='masctot'), 'scale'] = 'masc10'
        df.loc[(df.scale=='bdi_tot_avg'), 'scale'] = 'bdi'
        df.loc[(df.scale=='rads_scr'), 'scale'] = 'rads'

        df.loc[(df.scale=='ham_a_score'), 'scale'] = 'hama'
        df.loc[(df.scale=='anxs_tot'), 'scale'] = 'masc39'
        df.loc[(df.scale=='masctot'), 'scale'] = 'masc10'

        return df

    @staticmethod
    def concatanate_processed_csvs(analysis_name, input_dir=folders.csvs_trial_data, output_dir=folders.csvs_trial_data):
        """ Concatanate all processed csvs from input_dir into single csv

            Args:
                analysis_name (str): analysis_name of output
                input_dir (str, optional): folder from which CSVs are concatanated;
                    CSVs from subfolders are ignored
                    CSVs stratrting with test_ are ignored
                output_dir (str, optional): folder where output CSV (analysis_name+_processed.csv ) is saved

            Notes:
                Removes output CSV if already exists
        """

        target_fpath = os.path.join(output_dir, analysis_name+'_processed.csv')
        if os.path.isfile(target_fpath):
            os.remove(target_fpath)

        filenames = os.listdir(os.path.join(input_dir, analysis_name))
        csv_file_names = [filename for filename in filenames if (filename.endswith('.csv') and ('test_' not in filename))]

        for idx, csv_file_name in enumerate(csv_file_names):

            if idx==0:
                root_df=pd.read_csv(os.path.join(input_dir, analysis_name, csv_file_name))

            else:
                df=pd.read_csv(os.path.join(input_dir, analysis_name, csv_file_name))
                root_df = pd.concat([root_df, df], sort=False)

        root_df = Helpers.scale_renaming(root_df)
        root_df.to_csv(target_fpath, index=False)

    @staticmethod
    def check_is_why(df):
        """ Check if why guess info is present and if yes, extract """

        why_guess_cols=['ptleff','ptpeff','ptase','ptpse']
        isGuessWhy=[True for col in why_guess_cols if col in df.columns.to_list()]

        if isGuessWhy==[True, True, True, True]:
            return True
        elif isGuessWhy==[]:
            return False

        assert False

    @staticmethod
    def get_why_guess(row):
        """ Check if why guess info is present and if yes, extract """

        why_guess=None

        if (row.ptleff+row.ptpeff+row.ptase+row.ptpse)!=1:
            pass
        elif (row.ptase+row.ptpse)==1:
            why_guess='se'
        elif (row.ptleff+row.ptpeff)==1:
            why_guess='eff'
        else:
            pass

        return why_guess
