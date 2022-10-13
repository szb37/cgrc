"""
:Author: Balazs Szigeti <szb37@pm.me>
:Copyright: 2021, DrugNerdsLab
:License: MIT
"""

import src.data_transform.core as data_transform
import src.my_dataframes as mydfs
import src.constants as constants
import src.folders as folders
from unittest import mock
import pandas as pd
import unittest
import copy
import os


class DataTransformationUnitTests(unittest.TestCase):

    def test_del_rows(self):

        students = [
            ('jack', 34, 'Sydeny' , 'Australia'),
            ('Riti', 30, 'Delhi' , 'India' ),
            ('Vikas', 31, 'Mumbai' , 'India' ),
            ('Neelu', 32, 'Bangalore' , 'India' ),
            ('John', 30, 'New York' , 'US'),
            ('Mike', 17, 'las vegas' , 'US')
        ]

        df = pd.DataFrame(students, columns = ['name' , 'age', 'city' , 'country'])
        df_reference = copy.deepcopy(df)

        # Select and del students from Hungary
        rows = df.loc[df.country=='Hungary']
        idx2del = rows.index.tolist()
        data_transform.Helpers.del_rows(df=df, idx2del=idx2del, reset_index=True)
        self.assertTrue(df.equals(df_reference))

        # Select and del students from India - reset_index=True case
        rows = df.loc[df.country=='India']
        idx2del = rows.index.tolist()
        data_transform.Helpers.del_rows(df=df, idx2del=idx2del, reset_index=True)

        students = [
            ('jack', 34, 'Sydeny' , 'Australia'),
            ('John', 30, 'New York' , 'US'),
            ('Mike', 17, 'las vegas' , 'US')
        ]
        df_solution = pd.DataFrame(students, columns = ['name' , 'age', 'city' , 'country'])
        self.assertTrue(df.equals(df_solution))

        # Select and del students from India - reset_index=False case
        df = copy.deepcopy(df_reference)
        rows = df.loc[df.country=='India']
        idx2del = rows.index.tolist()
        data_transform.Helpers.del_rows(df=df, idx2del=idx2del, reset_index=False)

        df_solution = pd.DataFrame(students, columns = ['name' , 'age', 'city' , 'country'], index=[0,4,5])
        self.assertTrue(df.equals(df_solution))

        # Select and del students who are 30
        df = copy.deepcopy(df_reference)
        rows = df.loc[df.age==30]
        idx2del = rows.index.tolist()
        data_transform.Helpers.del_rows(df=df, idx2del=idx2del, reset_index=True)

        students = [
            ('jack', 34, 'Sydeny' , 'Australia'),
            ('Vikas', 31, 'Mumbai' , 'India' ),
            ('Neelu', 32, 'Bangalore' , 'India' ),
            ('Mike', 17, 'las vegas' , 'US')
        ]
        df_solution = pd.DataFrame(students, columns = ['name' , 'age', 'city' , 'country'])
        self.assertTrue(df.equals(df_solution))

    def test_rm_subjects(self):

        # Test case 1
        inDf = pd.read_csv(os.path.join(folders.fixtures, 'rm_subjects_input.csv'))
        refDf = pd.read_csv(os.path.join(folders.fixtures, 'rm_subjects_output1.csv'))

        valid_ids = [1001, 1002, 1003, 1111]
        data_transform.DataTransformCore.rm_subjects(valid_ids=valid_ids, df=inDf)
        self.assertTrue(inDf.equals(refDf))

        inDf = pd.read_csv(os.path.join(folders.fixtures, 'rm_subjects_input.csv'))
        refDf = pd.read_csv(os.path.join(folders.fixtures, 'rm_subjects_output2.csv'))

        valid_ids = [1005, 1006, 1007, 1008, 1009]
        data_transform.DataTransformCore.rm_subjects(valid_ids=valid_ids, df=inDf)
        self.assertTrue(inDf.equals(refDf))

    @mock.patch.object(data_transform.Helpers, 'get_tps_encoding')
    def test_rm_tps(self, mock):

        # Test case 1
        inDf = pd.read_csv(
            os.path.join(folders.fixtures, 'rm_tps_input.csv'))

        refDf = pd.read_csv(
            os.path.join(folders.fixtures, 'rm_tps_output1.csv'))

        mock.return_value = {'tp_column':'asstyp', 'valid_tps':[3,4,5]}
        data_transform.DataTransformCore.rm_tps(study_name='test', df=inDf)
        self.assertTrue(inDf.equals(refDf))

        # Test case 2
        inDf = pd.read_csv(
            os.path.join(folders.fixtures, 'rm_tps_input.csv'))

        refDf = pd.read_csv(
            os.path.join(folders.fixtures, 'rm_tps_output2.csv'))

        mock.return_value = {'tp_column':'visit', 'valid_tps':[5,7,8]}
        data_transform.DataTransformCore.rm_tps(study_name='test', df=inDf)
        self.assertTrue(inDf.equals(refDf))

    @mock.patch.object(data_transform.Helpers, 'get_tps_encoding')
    @mock.patch('src.data_transform.ad_studies_constants.tps_days', {'test':{2:0, 10:56}})
    def test_rm_degenrate_tps_case1(self, mock_tps_encoding):

        # Case 1
        inDf = pd.read_csv(os.path.join(
            folders.fixtures, 'rm_degenrate_tps_input1.csv'))
        refDf = pd.read_csv(os.path.join(
            folders.fixtures, 'rm_degenrate_tps_output1.csv'))

        mock_tps_encoding.return_value = {'tp_column':'visit', 'valid_tps':[2,10]}

        # Only works if non tps_days encoded Tps have been removed
        data_transform.DataTransformCore.rm_tps(df=inDf, study_name='test')
        data_transform.DataTransformCore.rm_degenrate_tps(
            df=inDf,
            study_name='test',
            tp_column='visit',
            valid_tps=[2,10],
            days_since_bsl_column='daysrz')

        self.assertTrue(inDf.equals(refDf))

        # Case where there are muliple visits with same delta_days
        with self.assertRaises(AssertionError):
            inDf = pd.read_csv(os.path.join(
                folders.fixtures, 'rm_degenrate_tps_input_raise.csv'))
            data_transform.DataTransformCore.rm_degenrate_tps(
                df=inDf,
                study_name='test',
                tp_column='visit',
                valid_tps=[2,10],
                days_since_bsl_column='daysrz'
            )

    @mock.patch.object(data_transform.Helpers, 'get_tps_encoding')
    @mock.patch('src.data_transform.ad_studies_constants.tps_days', {'test':{1:1, 3:3}})
    def test_rm_degenrate_tps_case2(self, mock_tps_encoding):

        inDf = pd.read_csv(os.path.join(
            folders.fixtures, 'rm_degenrate_tps_input2.csv'))
        refDf = pd.read_csv(os.path.join(
            folders.fixtures, 'rm_degenrate_tps_output2.csv'))

        mock_tps_encoding.return_value = {'tp_column':'asstype', 'valid_tps':[1,3]}

        data_transform.DataTransformCore.rm_tps(study_name='test', df=inDf)
        data_transform.DataTransformCore.rm_degenrate_tps(
            df=inDf,
            study_name='test',
            tp_column='asstype',
            valid_tps=[1,3],
            days_since_bsl_column='daysrz'
        )

        self.assertTrue(inDf.equals(refDf))

    @mock.patch.object(data_transform.Helpers, 'get_tps_encoding')
    def test_avg_score(self, mock_tps_encoding):
        """ Test peprocessing of TADS data """

        # Case 1
        inDf = pd.read_csv(os.path.join(folders.fixtures, 'avg_score_input1.csv'))
        refDf = pd.read_csv(os.path.join(folders.fixtures, 'avg_score_output1.csv'))

        mock_tps_encoding.return_value = {'tp_column':'asstyp', 'valid_tps':[1]}
        data_transform.DataTransformCore.avg_score(
            df=inDf,
            study_name='test',
            outcome_column='bdi_tot',
            respondent_column='relationship',
            respondents_to_avg=['father', 'mother'])

        self.assertTrue(inDf.equals(refDf))

        # Case 2
        inDf = pd.read_csv(os.path.join(folders.fixtures, 'avg_score_input2.csv'))
        refDf = pd.read_csv(os.path.join(folders.fixtures, 'avg_score_output2.csv'))

        mock_tps_encoding.return_value = {'tp_column':'visit', 'valid_tps':[2]}
        data_transform.DataTransformCore.avg_score(
            df=inDf,
            study_name='test',
            outcome_column='score',
            respondent_column='rel',
            respondents_to_avg=['father', 'mother', 'other'])

        self.assertTrue(inDf.equals(refDf))

        # Case 3 - some values are missing
        inDf = pd.read_csv(os.path.join(folders.fixtures, 'avg_score_input3.csv'))
        refDf = pd.read_csv(os.path.join(folders.fixtures, 'avg_score_output3.csv'))

        mock_tps_encoding.return_value = {'tp_column':'visit', 'valid_tps':[2]}
        data_transform.DataTransformCore.avg_score(
            df=inDf,
            study_name='test',
            outcome_column='score',
            respondent_column='rel',
            respondents_to_avg=['father', 'mother', 'other'])

        inDf['score'] = inDf['score'].astype('int64')
        self.assertTrue(inDf.equals(refDf))

        # Case where avg-days already exist
        with self.assertRaises(AssertionError):
            inDf = pd.read_csv(os.path.join(folders.fixtures, 'avg_score_input_raise.csv'))
            data_transform.DataTransformCore.avg_score(
                df=inDf,
                study_name='test',
                outcome_column='bdi_tot',
                respondent_column='relationship',
                respondents_to_avg=['father', 'mother'])


class DataTransformationIntegrationTests(unittest.TestCase):

    @mock.patch.object(data_transform.HelperStudy, 'get_respondent_study')
    @mock.patch('src.data_transform.ad_studies_constants.tps_structure', {'test_pipeline1':{'visit': [0, 10]}})
    def test_pipeline1(self, mock_respondent):

        mock_respondent.return_value = 'self'

        analysis_name='test_data_transform'
        data_transform.DataTransformCore.full_process(
            analysis_name=analysis_name,
            output_dir = folders.fixtures,
            study_csv_scales={'test_pipeline1':{'test_pipeline1_input': ['scale_a', 'scale_b']}})

        df = pd.read_csv(os.path.join(folders.fixtures, 'test_pipeline1__trial_data.csv'))
        refDf = pd.read_csv(os.path.join(folders.fixtures, 'test_data_tranform_pipeline1_output.csv'))

        df.__class__= mydfs.TrialDataDf
        df.set_column_types()
        refDf.__class__= mydfs.TrialDataDf
        refDf.set_column_types()

        # Ensure column order is the same and cutting why_guess_column: it reads to nan from file
        # but None is in returned object, do not care
        df.sort_index(axis=1, inplace=True)
        refDf.sort_index(axis=1, inplace=True)
        del df['why_guess']
        del refDf['why_guess']

        self.assertTrue(df.equals(refDf))

    @mock.patch.object(data_transform.HelperStudy, 'get_respondent_study')
    @mock.patch('src.data_transform.ad_studies_constants.tps_structure', {'test_pipeline2':{'visit': [1, 6]}})
    def test_pipeline2(self, mock_respondent):

        mock_respondent.return_value = 'ext'
        analysis_name='test_data_transform'

        data_transform.DataTransformCore.full_process(
            analysis_name=analysis_name,
            output_dir = folders.fixtures,
            study_csv_scales={'test_pipeline2':{'test_pipeline2_input': ['scale_a', 'scale_b']}})

        df = pd.read_csv(os.path.join(folders.fixtures, 'test_pipeline2__trial_data.csv'))
        refDf = pd.read_csv(os.path.join(folders.fixtures, 'test_data_tranform_pipeline2_output.csv'))

        df.__class__= mydfs.TrialDataDf
        df.set_column_types()
        refDf.__class__= mydfs.TrialDataDf
        refDf.set_column_types()

        # Ensure column order is the same and cutting why_guess_column: it reads to nan from file
        # but None is in returned object, do not care
        df.sort_index(axis=1, inplace=True)
        refDf.sort_index(axis=1, inplace=True)
        del df['why_guess']
        del refDf['why_guess']

        self.assertTrue(df.equals(refDf))

    @mock.patch.object(data_transform.HelperStudy, 'get_respondent_study')
    @mock.patch('src.data_transform.ad_studies_constants.tps_structure', {'test_pipeline3':{'visit': [1, 6]}})
    def test_pipeline3(self, mock_respondent):

        mock_respondent.return_value = 'ext'
        analysis_name='test_data_transform'

        data_transform.DataTransformCore.full_process(
            analysis_name=analysis_name,
            output_dir = folders.fixtures,
            study_csv_scales={'test_pipeline3':{'test_pipeline3_input': ['scale_a', 'scale_b']}})

        df = pd.read_csv(os.path.join(folders.fixtures, 'test_pipeline3__trial_data.csv'))
        refDf = pd.read_csv(os.path.join(folders.fixtures, 'test_data_tranform_pipeline3_output.csv'))

        df.__class__= mydfs.TrialDataDf
        df.set_column_types()
        refDf.__class__= mydfs.TrialDataDf
        refDf.set_column_types()

        # Ensure column order is the same and cutting why_guess_column: it reads to nan from file
        # but None is in returned object, do not care
        df.sort_index(axis=1, inplace=True)
        refDf.sort_index(axis=1, inplace=True)
        del df['why_guess']
        del refDf['why_guess']

        self.assertTrue(df.equals(refDf))
