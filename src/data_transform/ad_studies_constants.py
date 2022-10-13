"""
:Author: Balazs Szigeti <szb37@pm.me>
:Copyright: 2020, DrugNerdsLab
:License: MIT
"""

tps_structure = {
    'tads':{ #{timepoint defining column: [valid tp values],
        'asstyp': [3,5], #asstyp=4 is week 6
    },
    'stoppd':{
        'week': [0,12],
    },
    'rtca':{
        'visit': [2,10],
    },
    'ruppats':{
        'visit': [0,4,8,12],
    },
    'cams':{
        'visit': [-1, 12],
    },
}
tps_days = { # Only needed for studies where rm_degenrate_tps is applied
    'rtca':{ # numeric timepoint value: time in days since bsl
        2:0,
        10:56,
    },
    'ruppats':{
        0:0,
        8:56,
    },
}

ad_study_csv_scales = { #{name of csv: [column headers with scores]}
    'cams':{
        'cams_cgi':  ['cgi_si_avg'],
        'cams_cgii':  ['cgi_sii_avg'],
        'cams_mfq':  ['mfqtot_avg'],
        'cams_pars':  ['severity_totscr_avg'],
        'cams_cgis':  ['cgas_avg'],
        'cams_nassq':  ['nqas', 'nqds'],
        'cams_scared':  ['scared_total'],
        'cams_stai':  ['staittot_avg'],
        'cams_masc':  ['matotalx'],
    },
    'stoppd':{
        'stoppd_cgi':  ['cgi_si', 'cgi_sii'],
        'stoppd_hrsd': ['hamd_36',], # 'hamd_score_24'],
        'stoppd_bprs': ['bprs_total'],
    },
    'tads':{
        'tads_cdrs': ['cdrs_r_b', 'cdrs_14a', 'cdrs_14p', 'cdrs_r_a'],
        'tads_cgi':  ['cgi_si', 'cgi_sii_merged'],
        'tads_masc': ['anxs_tot', 'masctot'],
        'tads_bdi': ['bdi_tot_avg'],
        'tads_rads': ['rads_scr'],
    },
    'ruppats':{
        # if avg_score is used, be careful as tps_structure encoding is ambigous
        # see rm_degenrate_tps for reminder
        'ruppats_cdrs_a': ['cdrs_14a'],
        'ruppats_cdrs_p': ['cdrs_14p'],
        'ruppats_cgi':  ['cgi_si', 'cgi_sii'],
        'ruppats_hars': ['ham_a_score'],
        'ruppats_masc': ['anxs_tot', 'masctot'],
    },
    #'rtca':{
    #    'rtca_cgi': ['cgi_si', 'cgi_sii'],
    #    'rtca_cybocs': ['cybocs_comptot'], # obsessive compulsive behaviour scale
    #    'rtca_rfrlrs': ['lin_sum_score'], # autism scale
    #},
}
ad_study_csv_scales_short = {
    'cams':{
        'cams_cgi':  ['cgi_si_avg'],
        'cams_masc':  ['matotalx'],
    },
    'stoppd':{
        'stoppd_hrsd': ['hamd_36',], # 'hamd_score_24'],
    },
    'tads':{
        'tads_cdrs': ['cdrs_14a', 'cdrs_14p'],
    },
    'ruppats':{
        'ruppats_cdrs_p': ['cdrs_14p'],
        'ruppats_cgi':  ['cgi_si', 'cgi_sii'],
    },
}
