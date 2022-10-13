"""
:Author: Balazs Szigeti <szb37@pm.me>
:Copyright: 2020, DrugNerdsLab
:License: MIT
"""

from itertools import product as product
import matplotlib.pyplot as pyplt
import matplotlib.ticker as ticker
import src.constants as constants
import src.folders as folders
import src.miscs as miscs
import seaborn as sns
import pandas as pd
import numpy as np
import copy
import os


pyplt.rcParams.update({'font.family':'arial'})
title_fontdict={'fontsize':13, 'fontweight':'bold'}
axislabel_fontdict={'fontsize':16, 'fontweight':'bold'}
ticklabel_fontsize=14
sns.set_style("darkgrid")


class Controllers():

    def plot_test(output_dir=folders.src):

        analysis_name='AdAllEqualNoBaselineSet1'

        strata_summary = pd.read_csv(os.path.join(folders.csvs_trial_stats,  analysis_name+'__strata_summary.csv'))
        strata_summary = strata_summary.loc[
            (strata_summary.study=='ruppats') & (strata_summary.scale=='hama') & (strata_summary.guesser=='self')]

        strata_contrast = pd.read_csv(os.path.join(folders.csvs_trial_stats, analysis_name+'__strata_contrast.csv'))
        strata_contrast = strata_contrast.loc[
            (strata_contrast.study=='ruppats') & (strata_contrast.scale=='hama') & (strata_contrast.guesser=='self')]

        model_components = pd.read_csv(os.path.join(folders.csvs_trial_stats, analysis_name+'__model_components.csv'))
        model_components = model_components.loc[
            (model_components.study=='ruppats') & (model_components.scale=='hama') & (model_components.guesser=='self')]

        model_summary = pd.read_csv(os.path.join(folders.csvs_trial_stats, analysis_name+'__model_summary.csv'))
        model_summary = model_summary.loc[
            (model_summary.study=='ruppats') & (model_summary.scale=='hama') & (model_summary.guesser=='self')]

        cgrc_model_comps = pd.read_csv(os.path.join(folders.csvs_cgrc_stats, analysis_name+'__cgrc_model_comps.csv'))
        model_components = model_components.loc[
            (model_components.study=='ruppats') & (model_components.scale=='hama') & (model_components.guesser=='self')]

        title='tadaa'
        fname='test_plot_composition'

        fig = pyplt.figure(figsize=(10, 10), dpi=300)
        ax1 = fig.add_subplot(2, 2, 1) # n_rows, n_columns
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)


        Drawer.draw_empty_plot(
            ax = ax1,
            size = 5,
            title ='agyfasz1'
        )

        Drawer.draw_strata_plot(
            ax = ax2,
            title = 'agyfasz2',
            strata_summary = strata_summary,
            strata_contrast = strata_contrast,
            is_p_color = True
        )

        Drawer.draw_vsCGR_condition_p(
            ax = ax3,
            title = 'agyfasz3',
            cgrc_model_comps=cgrc_model_comps,
            cgr=0.55,
        )

        Drawer.draw_vsCGR_effect_size(
            ax = ax4,
            title = 'agyfasz4',
            cgrc_model_comps=cgrc_model_comps,
            cgr=0.85,
            add_legend=False,
        )

        Helpers.save_plot(fig, output_fname, folders.projectroot)

    def plot_strata(analysis_name, strata_summary, strata_contrast, study_scales,
        p_variant= 'p',
        is_p_color = True,
        savePNG = True,
        saveSVG = False,
        output_dir=folders.strata_plots):
        """ Creates and saves all strata plots
        Args:
            analysis_name (str): analysis_name of output files;
            strata_summary (pandas.DataFrame): strata_summary dataframe
            strata_contrast (pandas.DataFrame): strata_contrast dataframe
            study_scales (dict): defines combinations of studies/scales for which strata plots will be made
                E.g. study_scales = {'study2':['scale1', 'scale3']}, will process scales 1 and 3 of study2
            p_variant (str, optional): must be 'p' or 'p_adj';
                if p (default), then, raw p-value is used for strata comparisons
                if p_adj, then, the Tukey adjusted p-value is used for strata comparisons
            savePNG (bool, optional): if True, PNG image is saved
            saveSVG (bool, optional): if True, SVG image is saved
            output_dir (str, optional): folder where images are saved
        """

        assert isinstance(analysis_name, str)
        assert isinstance(strata_summary, pd.core.frame.DataFrame)
        assert isinstance(strata_contrast, pd.core.frame.DataFrame)
        assert isinstance(study_scales, dict)
        assert p_variant in ['p', 'p_adj']
        assert isinstance(is_p_color, bool)
        assert isinstance(savePNG, bool)
        assert isinstance(saveSVG, bool)
        assert os.path.isdir(output_dir)

        if (savePNG is False) and (saveSVG is False):
            return

        for study, scales in study_scales.items():
            print('Create strata plots - study:{}'.format(study))
            for scale, respondent, guesser in product(scales, constants.respondents, constants.guessers):

                strata_summary_filtered = strata_summary.loc[
                    (strata_summary.study==study) &
                    (strata_summary.scale==scale) &
                    (strata_summary.respondent==respondent) &
                    (strata_summary.guesser==guesser)]

                strata_contrast_filtered = strata_contrast.loc[
                    (strata_contrast.study==study) &
                    (strata_contrast.scale==scale) &
                    (strata_contrast.respondent==respondent) &
                    (strata_contrast.guesser==guesser)]

                if strata_summary_filtered.shape[0]==0:
                    assert strata_contrast_filtered.shape[0]==0
                    continue
                else:
                    assert strata_contrast_filtered.shape[0]==4

                fig = pyplt.figure(figsize=(5, 5), dpi=300)
                ax = fig.add_subplot(1, 1, 1)

                Drawer.draw_strata_plot(
                    ax = ax,
                    title = 'Study:{}; scale:{}; res:{}'.format(study, scale, respondent),
                    strata_summary = strata_summary_filtered,
                    strata_contrast = strata_contrast_filtered,
                    p_variant = p_variant,
                    is_p_color = is_p_color,
                )

                fname = analysis_name+'__'+study.upper()+'_'+scale.upper()+'_G'+guesser+'R'+respondent+'__strata_plot'
                Helpers.save_plot(
                    fig=fig,
                    output_dir=folders.strata_plots,
                    output_fname=output_fname,
                    savePNG=savePNG,
                    saveSVG=saveSVG,
                )
                pyplt.close('all')
                del fig, ax

    def plot_parameter_scaling(plot_type, analysis_namees, output_dir, fname, savePNG=True, saveSVG=False):

        assert plot_type in ['CGRvsP', 'CGRvsES', 'strata']
        assert isinstance(analysis_namees, list)
        assert os.path.isdir(output_dir)
        assert isinstance(fname, str)
        assert isinstance(savePNG, bool)
        assert isinstance(saveSVG, bool)

        if (savePNG is False) and (saveSVG is False):
            return

        fig = pyplt.figure(figsize=(20, 10), dpi=300) # width, height
        ax1 = fig.add_subplot(2, 4, 1) #n_row, n_col
        ax2 = fig.add_subplot(2, 4, 2)
        ax3 = fig.add_subplot(2, 4, 3)
        ax4 = fig.add_subplot(2, 4, 4)
        ax5 = fig.add_subplot(2, 4, 5)
        ax6 = fig.add_subplot(2, 4, 6)
        ax7 = fig.add_subplot(2, 4, 7)
        ax8 = fig.add_subplot(2, 4, 8)
        axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
        assert len(axes)>=len(analysis_namees)

        for ax, analysis_name in zip(axes, analysis_namees):

            if plot_type in ['CGRvsP', 'CGRvsES']:
                cgrc_model_comps = pd.read_csv(os.path.join(
                    folders.csvs_cgrc_stats, analysis_name+'__cgrc_model_comps.csv'))
            elif plot_type in ['strata']:
                strata_summary = pd.read_csv(os.path.join(
                    folders.csvs_trial_stats, analysis_name+'__strata_summary.csv'))
                strata_contrast = pd.read_csv(os.path.join(
                    folders.csvs_trial_stats, analysis_name+'__strata_contrast.csv'))
            else:
                assert False

            processed_df = pd.read_csv(os.path.join(folders.csvs_trial_data, analysis_name+'_processed.csv'))
            n_plpl = processed_df.loc[(processed_df.condition=='PL') & (processed_df.guess=='PL')].shape[0]
            n_acac = processed_df.loc[(processed_df.condition=='AC') & (processed_df.guess=='AC')].shape[0]
            cgr = round((n_plpl+n_acac)/processed_df.shape[0], 3)
            assert 0<=cgr<=1

            # make plots
            if plot_type=='CGRvsP':
                Drawer.draw_vsCGR_condition_p(
                    ax=ax,
                    title = analysis_name,
                    cgrc_model_comps=cgrc_model_comps,
                    cgr=cgr,
                    add_legend=True,
                )
            elif plot_type=='CGRvsES':
                Drawer.draw_vsCGR_effect_size(
                    ax=ax,
                    title = analysis_name,
                    cgrc_model_comps=cgrc_model_comps,
                    cgr=cgr,
                    add_legend=True,
                )
            elif plot_type=='strata':
                Drawer.draw_strata_plot(
                    ax = ax,
                    title = analysis_name,
                    strata_summary = strata_summary,
                    strata_contrast = strata_contrast,)
            else:
                assert False

        Helpers.save_plot(
            fig=fig,
            output_dir=output_dir,
            output_fname=fname,
            savePNG=savePNG,
            saveSVG=saveSVG,
        )
        pyplt.close('all')
        del fig, ax

    def plot_VScgr_separatex(input_dir, input_fname, output_dir, output_fname, study_scales, cgr_type='all', savePNG=True, saveSVG=False):
        """ Creates and saves combined 'CGR vs p' and 'CGR vs Effect' figures
        Args:
            cgrc_model_comps (pandas.DataFrame): model_components BBC dataframe
            analysis_name (str): analysis_name for outputs; images saved in output_dir/analysis_name/;
                target subfolder is created if it does not exist
            study_scales (dict): defines combinations of studies/scales for which strata plots will be made
                E.g. study_scales = {'study2':['scale1', 'scale3']}, will process scales 1 and 3 of study2
            savePNG (bool, optional): if True, PNG image is saved
            saveSVG (bool, optional): if True, SVG image is saved
            output_dir (str, optional): folder where images are saved
        """

        assert isinstance(study_scales, dict)
        assert cgr_type in ['all', 'active']
        assert isinstance(savePNG, bool)
        assert isinstance(saveSVG, bool)
        analysis.miscs.create_dir(output_dir)

        if (savePNG is False) and (saveSVG is False):
            return

        cgrc_model_comps = pd.read_csv(os.path.join(input_dir, input_fname))

        for study, scales in study_scales.items():
            print('Create BBC plots - study:{}'.format(study))

            if cgr_type=='all':
                original_cgr=constants.trial_cgrs[study]
            elif cgr_type=='active':
                original_cgr=constants.active_cgrs[study]
            else:
                assert False

            for scale, respondent, guesser in product(scales, constants.respondents, constants.guessers):

                cgrc_model_comps_filtered = cgrc_model_comps.loc[
                    (cgrc_model_comps.study==study) &
                    (cgrc_model_comps.scale==scale) &
                    (cgrc_model_comps.guesser==guesser) &
                    (cgrc_model_comps.respondent==respondent)
                ]

                if cgrc_model_comps_filtered.shape[0]==0:
                    continue

                fig = pyplt.figure(figsize=(10, 5), dpi=300)
                ax1 = fig.add_subplot(1, 2, 1) # n_rows, n_columns
                ax2 = fig.add_subplot(1, 2, 2, sharex=ax1)

                Drawer.draw_vsCGR_condition_p(
                    ax = ax1,
                    title = None, #'Study:{}; scale:{}; res:{}'.format(study, scale, respondent),
                    cgrc_model_comps=cgrc_model_comps_filtered,
                    cgr=original_cgr,
                    add_legend=True,
                )

                Drawer.draw_vsCGR_effect_size(
                    ax = ax2,
                    title = None, #'Study:{}; scale:{}; res:{}'.format(study, scale, respondent),
                    cgrc_model_comps=cgrc_model_comps_filtered,
                    cgr=original_cgr,
                    add_legend=True,
                )

                Helpers.save_plot(
                    fig=fig,
                    output_dir = output_dir,
                    output_fname = output_fname+'{}_{}__cgrc_plot2'.format(study, scale),
                    savePNG = savePNG,
                    saveSVG = saveSVG,
                )
                pyplt.close('all')
                del fig, ax1, ax2

    def plot_VScgr_twinx(input_dir, input_fname, output_dir, output_prefix, study_scales=None, cgr_type='all', save_figure=True):
        """ Creates and saves combined 'CGR vs p' and 'CGR vs Effect' figures
        Args:
            cgrc_model_comps (pandas.DataFrame): model_components BBC dataframe
            analysis_name (str): analysis_name for outputs; images saved in output_dir/analysis_name/;
                target subfolder is created if it does not exist
            study_scales (dict): defines combinations of studies/scales for which strata plots will be made
                E.g. study_scales = {'study2':['scale1', 'scale3']}, will process scales 1 and 3 of study2
            savePNG (bool, optional): if True, PNG image is saved
            saveSVG (bool, optional): if True, SVG image is saved
            output_dir (str, optional): folder where images are saved
        """


        assert isinstance(input_dir, str)
        assert isinstance(input_fname, str)
        assert isinstance(output_dir, str)
        assert isinstance(output_prefix, str)
        assert (isinstance(study_scales, dict) or (study_scales is None))
        assert cgr_type in ['all', 'active']
        assert isinstance(save_figure, bool)


        cgrc_model_comps = pd.read_csv(os.path.join(input_dir, input_fname))
        studies, scales, respondents, guessers = miscs.get_study_scales(
            input_df = cgrc_model_comps,
            study_scales = study_scales)

        for study, scale, respondent, guesser in product(studies, scales, respondents, guessers):

            if guesser=='ext':
                continue

            if cgr_type=='all':
                original_cgr=constants.trial_cgrs[study]
            elif cgr_type=='active':
                original_cgr=constants.active_cgrs[study]
            else:
                assert False

            cgrc_model_comps_filtered = cgrc_model_comps.loc[
                (cgrc_model_comps.study==study) &
                (cgrc_model_comps.scale==scale) &
                (cgrc_model_comps.guesser==guesser) &
                (cgrc_model_comps.respondent==respondent)
            ]

            if cgrc_model_comps_filtered.shape[0]==0:
                continue

            fig = pyplt.figure(figsize=(7, 7), dpi=300)
            ax1 = fig.add_subplot(1, 1, 1)
            ax2 = ax1.twinx()

            Drawer.draw_vsCGR(
                cgrc_model_comps=cgrc_model_comps_filtered,
                cgr = original_cgr,
                title = None,
                ax1 = ax1,
                ax2 = ax2,
                add_legend=True,
        )

            if save_figure:
                Helpers.save_plot(
                    fig=fig,
                    output_dir = output_dir,
                    output_fname = output_prefix+'_{}_{}__cgrc_plot'.format(scale, constants.estimator),
                )

            pyplt.close('all')
            del fig, ax1, ax2


class Drawer():

    def draw_empty_plot(ax, size, title):
        """ Draws empty axis """

        assert isinstance(size, int)
        labels = ['test' for el in range(size)]

        ax.set_title(title, fontdict=title_fontdict)
        ax.set_xticks(np.arange(size))
        ax.set_xticklabels(labels)

    def draw_strata_plot(ax, title, strata_summary, strata_contrast,
        is_p_color = True,
        p_variant = 'p'):
        """ Return strata strata plots in output_dir/analysis_name/

        Args:
            ax (matplotlib.axes._subplots.AxesSubplot): axes to be drawn on
            title (str): plot title, if any
            strata_summary (pd.core.frame.DataFrame): summary_df dataframe whose rows have already
                been filtered for study, scale, respondent and guesser
                # TODO: convert object to src.my_dataframes.ModelSummaryDf
            strata_contrast (pd.core.frame.DataFrame): contrast_df dataframe whose rows have already
                been filtered for study, scale, respondent and guesser
                # TODO: convert object to src.my_dataframes.ContrastSummaryDf
            p_variant (str, optional): must be 'p' or 'p_adj';
                if p (default), then, raw p-value is used for strata comparisons
                if p_adj, then, the Tukey adjusted p-value is used for strata comparisons
            is_p_color (bool, optional): if True (default), then, p-values will be colored in graph;
                red: non-sig; orange: trend; green: significant
        """

        # check inputs
        assert (title is None) or isinstance(title, str)
        assert isinstance(strata_summary, pd.core.frame.DataFrame)
        assert Helpers.is_df_filtered(strata_summary)
        assert isinstance(strata_contrast, pd.core.frame.DataFrame)
        assert Helpers.is_df_filtered(strata_contrast)
        assert isinstance(is_p_color, bool)
        assert p_variant in ['p', 'p_adj']

        # get data
        plpl = strata_summary.loc[(strata_summary.strata=='PLPL')]
        acpl = strata_summary.loc[(strata_summary.strata=='ACPL')]
        plac = strata_summary.loc[(strata_summary.strata=='PLAC')]
        acac = strata_summary.loc[(strata_summary.strata=='ACAC')]

        # make plot - draw strata errorbars
        if title is not None:
            ax.set_title(title, fontdict=title_fontdict)

        strata_labels = ['Cond:PL\nGuess:PL', 'AC\nPL', 'PL\nAC', 'AC\nAC',]
        x_pos = np.arange(4)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(strata_labels, fontdict = axislabel_fontdict)
        ax.set_ylabel('Score', fontdict = axislabel_fontdict)
        ax.tick_params(axis='y', which='major', labelsize=ticklabel_fontsize)

        lines = {'linestyle': 'None'}
        pyplt.rc('lines', **lines)

        ax.errorbar(x_pos, solid_capstyle='projecting', capsize=5, fmt='', lw=3, color='#1f77b4ff',
            y    = [plpl.est.item(), acpl.est.item(), plac.est.item(), acac.est.item()],
            yerr = [ plpl.se.item(),  acpl.se.item(),  plac.se.item(),  acac.se.item()])
        ax.scatter(
            x_pos,
            [plpl.est.item(), acpl.est.item(), plac.est.item(), acac.est.item()],
            linewidths=5,
            color='#1f77b4ff')

        # adjust image height to have space for vertical comparison bars
        y_boost, y_lvl1, y_lvl2, y_lvl3 = Helpers.adjust_strata_plot_height(ax)

        # make plot - draw vertical comparison bars
        strata_rows = strata_contrast.loc[strata_contrast.contrast.isin(['ACPLvsPLPL', 'PLPLvsACPL'])]
        p_value = eval('round(strata_rows.{}.item(),3)'.format(p_variant))
        xmax=0.346; xmin=0.046
        ax.axhline(y=y_lvl3, xmin=xmin, xmax=xmax, linestyle='-', color='black')
        ax.text(y=y_lvl3+y_boost, x=xmax-0.075, ha='left', s='p={}'.format(p_value), color=Helpers.get_p_color(p_value, is_p_color))

        strata_rows = strata_contrast.loc[strata_contrast.contrast.isin(['ACACvsPLAC', 'PLACvsACAC'])]
        p_value = eval('round(strata_rows.{}.item(),3)'.format(p_variant))
        xmax=0.953; xmin=0.653
        ax.axhline(y=y_lvl3, xmin=xmin, xmax=xmax, linestyle='-', color='black')
        ax.text(y=y_lvl3+y_boost, x=2.75, ha='right', s='p={}'.format(p_value), color=Helpers.get_p_color(p_value, is_p_color))

        strata_rows = strata_contrast.loc[strata_contrast.contrast.isin(['PLACvsPLPL', 'PLPLvsPLAC'])]
        p_value = eval('round(strata_rows.{}.item(),3)'.format(p_variant))
        xmax=0.65; xmin=0.046
        ax.axhline(y=y_lvl2, xmin=xmin, xmax=xmax, linestyle='-', color='black')
        ax.text(y=y_lvl2+y_boost, x=0.8, ha='left', s='p={}'.format(p_value), color=Helpers.get_p_color(p_value, is_p_color))

        strata_rows = strata_contrast.loc[strata_contrast.contrast.isin(['ACACvsACPL', 'ACPLvsACAC'])]
        p_value = eval('round(strata_rows.{}.item(),3)'.format(p_variant))
        xmax=0.93; xmin=0.346
        ax.axhline(y=y_lvl1, xmin=xmin, xmax=xmax, linestyle='-', color='black')
        ax.text(y=y_lvl1+y_boost, x=2, ha='left', s='p={}'.format(p_value), color=Helpers.get_p_color(p_value, is_p_color))

    def draw_vsCGR_condition_p(ax, title, cgrc_model_comps, cgr, color=None, add_legend=True):
        """ Draws CGR VS condition p-value plot
        Args:
            ax (matplotlib.axes._subplots.AxesSubplot): axes to be drawn on
            title (str): plot title, if any
            cgrc_model_comps (pd.core.frame.DataFrame): model_components_df dataframe; rows
                already filtered for study, scale, respondent and guesser
                # TODO: convert object to src.my_dataframes.ModelComponentsDf
            cgr (float): correct guess rate
            add_legend (bool, optional): add plot legend or not
        """

        # check inputs
        assert (title is None) or isinstance(title, str)
        assert isinstance(cgrc_model_comps, pd.core.frame.DataFrame)
        assert Helpers.is_df_filtered(cgrc_model_comps)
        assert isinstance(cgr, float)
        assert isinstance(add_legend, bool)

        # make plot
        if title is not None:
            ax.set_title(title, fontdict=title_fontdict)

        sns.lineplot(ax=ax, x='cgr', y='p', color=color, data=cgrc_model_comps.loc[
            (cgrc_model_comps['model_type']=='without_guess') &
            (cgrc_model_comps['component']=='conditionAC')]
        )

        ax.set_xlim([0, 1])
        ax.set_ylim([-0.05, 0.8])

        ax.set_xlabel('Correct guess rate (CGR)', fontdict = axislabel_fontdict)
        ax.set_ylabel('Treatment p-value', color=color, fontdict = axislabel_fontdict)
        ax.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)

        ax.axvline(x=cgr, color='green', ls='--') # Vertical line, original CGR
        ax.axvline(x=0.5, color='black', ls='--') # Vertical line, True CGR
        ax.axhline(y=0.05, color='red', ls='--')  # Horizontal line, significance threshold

        if add_legend:
            ax.legend(
                ['Treatment p-value', 'Original CGR', 'True blind CGR', 'p=.05 sig. threshold'],
                fontsize = 'medium')

    def draw_vsCGR_effect_size(ax, title, cgrc_model_comps, cgr, color=None, add_legend=True):
        """ Draws CGR VS effect size plot
        Args:
            ax (matplotlib.axes._subplots.AxesSubplot): axes to be drawn on
            title (str): plot title, if any
            cgrc_model_comps (pd.core.frame.DataFrame): cgrc_model_comps dataframe; rows
                already filtered for study, scale, respondent and guesser
                # TODO: convert object to src.my_dataframes.CGRCurveDf
            cgr (float): correct guess rate
            add_legend (bool, optional): add plot legend or not
        """

        # check inputs
        assert (title is None) or isinstance(title, str)
        assert isinstance(cgrc_model_comps, pd.core.frame.DataFrame) # TODO: Check conversion to DF subtype
        assert Helpers.is_df_filtered(cgrc_model_comps)
        assert isinstance(cgr, float)
        assert isinstance(add_legend, bool)

        # make plot
        if title is not None:
            ax.set_title(title, fontdict=title_fontdict)

        sns.lineplot(ax=ax, x='cgr', y='est', color=color, data=cgrc_model_comps.loc[
            (cgrc_model_comps['model_type']=='without_guess') &
            (cgrc_model_comps['component']=='conditionAC')]
        )
        ax.set_xlim([0, 1])

        ax.set_xlabel('Correct guess rate (CGR)', fontdict = axislabel_fontdict)
        ax.set_ylabel('Treatment effect', color=color, fontdict = axislabel_fontdict)
        ax.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)

        #ax.axvline(x=cgr, color='green', ls='--') # Vertical line, original CGR
        #ax.axvline(x=0.5, color='black', ls='--') # Vertical line, True CGR
        #ax.axhline(y=0, color='red', ls='--')  # Horizontal line, 0 difference

        if add_legend:
            ax.legend(
                ['Treatment effect', 'Original CGR', 'True blind CGR', 'No effect'],
                fontsize = 'medium')

    def draw_vsCGR(ax1, ax2, title, cgrc_model_comps, cgr, add_legend=True):

        assert (title is None) or isinstance(title, str)
        assert isinstance(cgrc_model_comps, pd.core.frame.DataFrame)
        assert Helpers.is_df_filtered(cgrc_model_comps)
        assert isinstance(cgr, float)
        assert isinstance(add_legend, bool)

        if title is not None:
            ax.set_title(title, fontdict=title_fontdict)

        color1='blue'
        color2='red'

        data = cgrc_model_comps.loc[
            (cgrc_model_comps['model_type']=='without_guess') &
            (cgrc_model_comps['component']=='conditionAC')]

        avg_df = Helpers.get_avg_se(data)

        if constants.estimator=='mean':
            ax2.fill_between(x = avg_df.cgr, y1 = avg_df.ci_low, y2 = avg_df.ci_high, color=color2, alpha=0.2)
            sns.lineplot(ax=ax1, x='cgr', y='p', estimator=np.mean, color=color1, data=data)
            sns.lineplot(ax=ax2, x='cgr', y='est', data=avg_df, color=color2)

        elif constants.estimator=='median':
            assert False # need to implement
            sns.lineplot(ax=ax1, x='cgr', y='p', estimator=np.median, color=color1, data=data)
            sns.lineplot(ax=ax2, x='cgr', y='est', estimator=np.median, color=color2, data=data)
        else:
            assert False


        ax1.set_xlim([0, 1])
        ax2.set_xlim([0, 1])

        ax1.set_ylim([-0.05, 0.75])

        ax2.set_ylim([round(data.est.min(),1), round(data.est.max(),1)])

        ax1.yaxis.set_major_locator(ticker.LinearLocator(9)) # originally was 9
        ax2.yaxis.set_major_locator(ticker.LinearLocator(9)) # originally was 9

        ax2.grid(False)

        ax1.set_xlabel('Correct guess rate (CGR)', fontdict = axislabel_fontdict)
        ax1.set_ylabel('Treatment p-value', color=color1, fontdict = axislabel_fontdict)
        ax2.set_ylabel('Treatment estimate', color=color2, fontdict = axislabel_fontdict)

        ax1.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)
        ax2.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)

        ax1.axvline(x=cgr, color='green', ls='--') # Vertical line, original CGR
        ax1.axvline(x=0.5, color='black', ls='--') # Vertical line, True CGR
        ax1.axhline(y=0.05, color='red', ls='--')  # Horizontal line, significance threshold

        ax1.set_ylim([-0.05, 0.75])
        ax2.set_ylim([-15, 15])

        if add_legend:
            ax1.legend(
                ['Treatment p-value', 'Treatment p-value CI', 'Original CGR', 'True blind CGR', 'Sig. threshold'], fontsize = 'medium')
            ax2.legend(
                ['Treatment estimate', 'Treatment estimate CI'], fontsize = 'medium')


class Helpers():

    @staticmethod
    def save_plot(fig, output_dir, output_fname, dpi=300):
        """ Saves plot in dir as fname """

        assert isinstance(output_fname, str)
        assert os.path.isdir(output_dir)

        fig.savefig(
            fname = os.path.join(output_dir, output_fname+'.png'),
            format='png',
            dpi=dpi,
        )

        fig.savefig(
            fname = os.path.join(output_dir, output_fname+'.svg'),
            format='svg',
            dpi=dpi,
        )

    @staticmethod
    def get_p_color(p_value , is_p_color):
        """ Returns color str appropaite for p_value
            Args:
                p_value (float): p-value
                is_p_color (bool): if False, black is returned

            Returns:
                'red' if p_value >= 0.1;
                'orange' if  0.1 > p_value > 0.05
                'green' if  0.05 >= p_value
        """

        if is_p_color is False:
            return 'black'

        if p_value >= 0.1:
            return 'red'
        elif 0.1 > p_value > 0.05:
            return 'orange'
        elif 0.05 >= p_value:
            return 'green'
        else:
            assert False

    @staticmethod
    def adjust_strata_plot_height(ax):
        """ Adds height to strata plots such thatcomparison bars fit. Also returns y-increments
            to guide p-value placements
        """

        y_low, y_high = ax.get_ylim()
        y_high_adj = y_high + 0.3*abs(y_high-y_low)
        ax.set_ylim(bottom=y_low, top=y_high_adj)

        y_lvl3 = y_high + 0.2*abs(y_high-y_low)
        y_lvl2 = y_high + 0.1*abs(y_high-y_low)
        y_lvl1 = y_high

        y_low, y_high = ax.get_ylim()
        y_boost =0.0075*abs(y_high-y_low)

        return y_boost, y_lvl1, y_lvl2, y_lvl3

    @staticmethod
    def is_df_filtered(df):
        """ Checks if values of study, scale, respondent and guesser are unque in dataframe """

        assert len(df.study.unique().tolist())==1
        assert len(df.scale.unique().tolist())==1
        assert len(df.respondent.unique().tolist())==1
        assert len(df.guesser.unique().tolist())==1
        return True

    @staticmethod
    def get_avg_se(df):

        avg_df = copy.deepcopy(df)
        del avg_df['p']
        del avg_df['est']
        del avg_df['t']
        del avg_df['se']

        avg_df['p'] = None
        avg_df['est'] = None
        avg_df['t'] = None
        avg_df['se'] = None
        avg_df['ci_low'] = None
        avg_df['ci_high'] = None
        del avg_df['cgr_trial_id']
        avg_df.drop_duplicates(inplace =True)

        for cgr in df.cgr.unique():

            tmp = df.loc[(df.cgr==cgr)]

            # Insert averages across CGR trials
            avg_df.loc[(df.cgr==cgr), 'p'] = round(tmp.p.mean(), 5)

            avg_df.loc[(df.cgr==cgr), 'est'] = round(tmp.est.mean(), 5)

            avg_df.loc[(df.cgr==cgr), 't'] = round(tmp.t.mean(), 5)

            avg_df.loc[(df.cgr==cgr), 'se'] = round(tmp.se.mean(), 5)

            avg_df.loc[(df.cgr==cgr), 'ci_low'] = round(tmp.est.mean(), 5) - 1.96*round(tmp.se.mean(), 5)

            avg_df.loc[(df.cgr==cgr), 'ci_high'] = round(tmp.est.mean(), 5) + 1.96*round(tmp.se.mean(), 5)

        avg_df['p'] = avg_df['p'].astype('float')
        avg_df['est'] = avg_df['est'].astype('float')
        avg_df['t'] = avg_df['t'].astype('float')
        avg_df['se'] = avg_df['se'].astype('float')
        avg_df['ci_low'] = avg_df['ci_low'].astype('float')
        avg_df['ci_high'] = avg_df['ci_high'].astype('float')

        return avg_df
