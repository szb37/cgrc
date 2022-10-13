"""
:Author: Balazs Szigeti <szb37@pm.me>
:Copyright: 2020, DrugNerdsLab
:License: MIT
"""

import src.data_transform.core as data_transform
import src.data_transform.ad_studies_constants as ad_constants

def get_ad_processed(analysis_name, study_csv_scales=ad_constants.ad_study_csv_scales_short):
    """ Processes raw AD trials data into Processed CSV format """
    data_transform.DataTransformCore.full_process(analysis_name=analysis_name, study_csv_scales=study_csv_scales)
    data_transform.Helpers.concatanate_processed_csvs(analysis_name=analysis_name)
