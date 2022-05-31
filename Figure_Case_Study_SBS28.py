# !/usr/bin/env python3

# Author: burcakotlu

# Contact: burcakotlu@eng.ucsd.edu

# Figure6
# Figure Case Study SBS28

# Lung-AdenoCA POLE-proficient
# Stomach-AdenoCA POLE-proficient

# ColoRect-AdenoCA POLE-deficient
# Uterus-AdenoCA POLE-deficient
# ESCC POLE-deficient

# copy this python file under /home/burcak/developer/python/SigProfilerTopographyAuxiliary/combined_pcawg_and_nonpcawg_figures_pdfs
# python Figure_Case_Study_SBS28.py

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as matplotlib_cm

from matplotlib import pyplot as plt
from numpy import dot
from numpy.linalg import norm

from functools import reduce
from collections import OrderedDict

from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import generate_occupancy_pdfs
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import generate_replication_time_pdfs
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import generate_processivity_pdf
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import COSMIC
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import MANUSCRIPT
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import AT_LEAST_1K_CONSRAINTS
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import PROCESSIVITY
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import plot_processivity_colorbar_vertical
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import plot_processivity_colorbar_horizontal
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import plot_processivity_legend

from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import plot_strand_bias_figures
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import TRANSCRIPTIONSTRANDBIAS
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import REPLICATIONSTRANDBIAS
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import PCAWG
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import nonPCAWG
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import MUTOGRAPHS
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import ODDS_RATIO

from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import main as strand_bias_main

from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import NUCLEOSOME_OCCUPANCY
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import EPIGENOMICS_OCCUPANCY
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import NUCLEOSOME
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import CTCF

from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import fill_cancer_type_signature_cutoff_average_probability_df
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import step1_compute_p_value
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import step2_combine_p_values
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import step3_apply_multiple_tests_correction
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import readDictionary
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import normal_combined_pcawg_nonpcawg_cancer_type_2_biosample_dict
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import TABLES
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import DATA_FILES
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import EXCEL_FILES
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import DICTIONARIES
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import fill_rows_and_columns_np_array
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import enriched
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import depleted
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import heatmap


from Figure_Case_Study_SBS4_Across_All_Tissues import figure_case_study_occupancy
from Figure_Case_Study_SBS4_Across_All_Tissues import figure_case_study_strand_coordinated_mutagenesis
from Figure_Case_Study_SBS4_Across_All_Tissues import figure_case_study_replication_time
from Figure_Case_Study_SBS4_Across_All_Tissues import plot_replication_time_legend_for_all_mutation_types
from Figure_Case_Study_SBS4_Across_All_Tissues import plot_replication_time_legend
from Figure_Case_Study_SBS4_Across_All_Tissues import figure_case_study_strand_bias

from Combined_Common import SBS
from Combined_Common import DBS
from Combined_Common import ID
from Combined_Common import natural_key

FIGURE_CASE_STUDY_SBS28 = 'FIGURE_CASE_STUDY_SBS28'

def figure_case_study_occupancy(combined_output_path,
                                plot_output_path,
                                figure_file_extension,
                                sbs_signatures,
                                dbs_signatures,
                                id_signatures,
                                cancer_types,
                                numberofSimulations,
                                figure_types,
                                cosmic_release_version,
                                pearson_spearman_correlation_cutoff,
                                dna_elements,
                                minimum_number_of_overlaps_required_for_sbs,
                                minimum_number_of_overlaps_required_for_dbs,
                                minimum_number_of_overlaps_required_for_indels,
                                number_of_mutations_required_list,
                                number_of_mutations_required_list_for_ctcf,
                                figure_case_study):

    occupancy_significance_level = 0.05
    plus_minus = 1000
    window_size = 100

    cosmic_legend = False
    cosmic_correlation_text = False
    cosmic_labels = True
    cancer_type_on_right_hand_side = False
    cosmic_fontsize_text = 45
    cosmic_fontsize_ticks = 40
    cosmic_fontsize_labels = 40
    cosmic_linewidth_plot = 10
    cosmic_title_all_cancer_types = False

    consider_both_real_and_sim_avg_overlap = True
    depleted_fold_change = 0.95
    enriched_fold_change = 1.05

    for (dna_element, occupancy_type) in dna_elements:
        # plot_occupancy_legend(plot_output_path)
        # plot_occupancy_legend(plot_output_path, num_of_cols=2)

        # plot_occupancy_legend_for_all_mutation_types(plot_output_path, mutation_types = [SBS, DBS], number_of_columns = 3)
        # plot_occupancy_legend_for_all_mutation_types(plot_output_path, mutation_types = [SBS, DBS])

        generate_occupancy_pdfs(plot_output_path,
                            combined_output_path,
                            occupancy_type,
                            dna_element,
                            cancer_types,
                            sbs_signatures,
                            id_signatures,
                            dbs_signatures,
                            numberofSimulations,
                            plus_minus,
                            plus_minus,
                            window_size,
                            consider_both_real_and_sim_avg_overlap,
                            minimum_number_of_overlaps_required_for_sbs,
                            minimum_number_of_overlaps_required_for_dbs,
                            minimum_number_of_overlaps_required_for_indels,
                            figure_types,
                            number_of_mutations_required_list,
                            number_of_mutations_required_list_for_ctcf,
                            cosmic_release_version,
                            figure_file_extension,
                            depleted_fold_change,
                            enriched_fold_change,
                            occupancy_significance_level,
                            pearson_spearman_correlation_cutoff,
                            cosmic_legend,
                            cosmic_correlation_text,
                            cosmic_labels,
                            cancer_type_on_right_hand_side,
                            cosmic_fontsize_text,
                            cosmic_fontsize_ticks,
                            cosmic_fontsize_labels,
                            cosmic_linewidth_plot,
                            cosmic_title_all_cancer_types,
                            figure_case_study)

def plot_epigenomics_heatmap_color_bar(plot_output_dir):
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_axes([0.05, 0.475, 0.9, 0.15])

    bounds = np.arange(0.25, 1.80, 0.25)
    norm = mpl.colors.Normalize(vmin=min(bounds), vmax=max(bounds))
    cbar = mpl.colorbar.ColorbarBase(ax,
                                     cmap=plt.get_cmap("seismic"),
                                     norm=norm,
                                     ticks=bounds,
                                     spacing='proportional',
                                     orientation='horizontal')

    cbar.set_label('Fold Change\n [Real mutations/Simulated Mutations]', fontsize=30, labelpad=10)
    cbar.ax.tick_params(labelsize=25)

    filename = 'epigenomics_heatmaps_seismic_color_bar.png'
    figureFile = os.path.join(plot_output_dir, filename)
    fig.savefig(figureFile, dpi=100, bbox_inches="tight")
    plt.close()


def plot_heatmap(rows,
                columns,
                avg_fold_change_np_array,
                group_name,
                df1,
                df2,
                name_for_rows,
                name_for_columns,
                significance_level,
                enriched_fold_change,
                depleted_fold_change,
                heatmap_rows_signatures_columns_cancer_types_output_path,
                figure_name,
                x_axis_labels_on_bottom = True,
                plot_title = True):

    if len(rows) > 0 or len(columns) > 0:
        dpi = 100
        squaresize = 200  # pixels

        if len(columns) <= 3:
            figwidth = 2 * len(columns) * squaresize / float(dpi)
        else:
            figwidth = len(columns) * squaresize / float(dpi)

        figheight = len(rows) * squaresize / float(dpi)

        # Set font size w.r.t. number of rows
        font_size = 10 * np.sqrt(len(avg_fold_change_np_array))
        title_font_size = font_size
        label_font_size = font_size

        # Set cell font size w.r.t. number of columns
        cell_font_size = font_size * 2 / 3

        # Figure Case Study SBS4 Topography across all cancer types
        if (figure_name) and (figure_name == 'Figure_Case_Study_SBS4'):
            title_font_size = title_font_size * 1.75
            label_font_size = font_size * 1.75
            cell_font_size = cell_font_size * 1.75

        if (figure_name) and (figure_name == 'Figure_Case_Study_Shared_Etiology'):
            title_font_size = title_font_size * 2.5
            label_font_size = font_size * 2.5
            cell_font_size = cell_font_size * 2.5

        if (figure_name) and (figure_name == 'Figure_Case_Study_SBS28_epigenomics_heatmap'):
            title_font_size = title_font_size * 3
            label_font_size = font_size * 3
            cell_font_size = cell_font_size * 3

        fig, ax = plt.subplots(1, figsize=(figwidth, figheight), dpi=dpi)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # make aspect ratio square
        ax.set_aspect(1.0)

        # Update rows labels and column labels
        rows_labels = rows
        columns_labels = columns

        rows_labels = ['Open Chromatin' if 'ATAC' in row else row for row in rows_labels]
        columns_labels = ['Open Chromatin' if 'ATAC' in column else column for column in columns_labels]

        rows_labels = [row[:-6] if '-human' in row else row for row in rows_labels]
        columns_labels = [column[:-6] if '-human' in column else column for column in columns_labels]

        print('%s --- rows:%d columns:%d figwidth:%d figheight:%d title_font_size:%d label_font_size:%d cell_font_size:%d' % (group_name, len(rows), len(columns), figwidth, figheight, title_font_size, label_font_size, cell_font_size))
        print("rows_labels:%s" % (rows_labels))
        print("columns_labels:%s" % (columns_labels))
        print('avg_fold_change_np_array.shape(%d,%d)' % (
        avg_fold_change_np_array.shape[0], avg_fold_change_np_array.shape[1]))
        print('avg_fold_change_np_array:%s' % (avg_fold_change_np_array))

        if (figure_name) and (figure_name == 'Figure_Case_Study_SBS28_nonPOLE_hypermutators'):
            rows_labels = ['SBS28\nnon-POLE hypermutators']

        if (figure_name) and (figure_name == 'Figure_Case_Study_SBS28_POLE_hypermutators'):
            rows_labels = ['SBS28\nPOLE hypermutators']

        im = heatmap(avg_fold_change_np_array,
                     rows_labels,
                     columns_labels,
                     x_axis_labels_on_bottom = x_axis_labels_on_bottom,
                     ax = ax,
                     cmap = 'seismic',
                     cbarlabel = "Fold Change [Real mutations/Simulated Mutations]",
                     vmin = 0.25,
                     vmax = 1.75,
                     fontsize = label_font_size)

        # rows [('POLE hypermutators'), ('non-POLE hypermutators') ]

        # Put text in each heatmap cell
        for row_index, row in enumerate(rows, 0):
            if row == rows[0]: # POLE deficient
                df = df1
                row = 'SBS28'
            elif row == rows[1]: # POLE proficient
                df = df2
                row = 'SBS28'
            print('#####################################################')
            for column_index, column in enumerate(columns, 0):

                if df[(df[name_for_rows] == row) & (df[name_for_columns] == column)]['avg_fold_change'].any():
                    avg_fold_change_array = df[(df[name_for_rows] == row) & (df[name_for_columns] == column)]['avg_fold_change'].values
                    q_value_array = df[(df[name_for_rows] == row) & (df[name_for_columns] == column)]['q_value'].values

                    if len(avg_fold_change_array) > 1:
                        avg_fold_change = np.mean(avg_fold_change_array)
                        q_value = 1
                        for index, avg_fold_change in enumerate(avg_fold_change_array):
                            if (enriched(avg_fold_change, enriched_fold_change) or depleted(avg_fold_change, depleted_fold_change)) and q_value_array[index] <= significance_level:
                                q_value = q_value_array[index]
                                break

                    else:
                        avg_fold_change = avg_fold_change_array[0]
                        q_value = q_value_array[0]
                    print('%s (%s, %s) [%d,%d] --> avg_fold_change:%f -- %f q_value:%f' % (group_name, row, column, row_index, column_index, avg_fold_change, avg_fold_change_np_array[row_index, column_index], q_value))

                    if q_value <= significance_level:
                        ax.text(column_index, row_index, "%.2f*" % (avg_fold_change_np_array[row_index, column_index]), ha="center", va="center", color="k", fontsize=cell_font_size)
                    else:
                        ax.text(column_index, row_index, "%.2f" % (avg_fold_change_np_array[row_index, column_index]), ha="center", va="center", color="k", fontsize=cell_font_size)

        if plot_title:
            if group_name == 'ATAC-seq':
                plt.title('Chromatin Accessibility', fontsize=title_font_size, y=1.01)
            elif group_name.endswith("-human"):
                group_name = group_name[:-6]
                plt.title('%s' % (group_name), fontsize=title_font_size, y=1.01)
            else:
                plt.title('%s' % (group_name), fontsize=title_font_size, y=1.01)

        if group_name:
            filename = '%s.png' % (group_name)
        elif figure_name:
            filename = '%s_occupancy_heatmap.png' %(figure_name)
        else:
            filename = 'occupancy_heatmap.png'

        figureFile = os.path.join(heatmap_rows_signatures_columns_cancer_types_output_path, filename)
        fig.savefig(figureFile, dpi=dpi, bbox_inches="tight")
        plt.close()


# new starts
def figure_case_study_SBS28_epigenomics_heatmap(
        combined_output_path,
        heatmaps_main_output_path,
        hm_path,
        ctcf_path,
        atac_path,
        plot_output_path,
        signatures, #  ['SBS28']
        signature_signature_type_tuples, # [('SBS28', SBS)]
        signatures_ylabels_on_the_heatmap, # [('POLE hypermutators'), ('non-POLE hypermutators') ] [('POLE-'), ('POLE+') ]
        cancer_types,
        cancer_types_pole_hypermutators,
        cancer_types_non_pole_hypermutators,
        combine_p_values_method,
        window_size,
        numberofSimulations,
        enriched_fold_change,
        depleted_fold_change,
        significance_level,
        minimum_number_of_overlaps_required_for_sbs,
        minimum_number_of_overlaps_required_for_dbs,
        minimum_number_of_overlaps_required_for_indels,
        signature_cancer_type_number_of_mutations,
        signature_cancer_type_number_of_mutations_for_ctcf,
        step1_data_ready,
        figure_type,
        cosmic_release_version,
        figure_file_extension):

    cancer_type_signature_cutoff_number_of_mutations_average_probability_df = fill_cancer_type_signature_cutoff_average_probability_df(cancer_types, combined_output_path)

    if step1_data_ready:
        # Read Step1_Signature2CancerType2Biosample2DNAElement2PValue_Dict.txt
        dictFilename = 'Step1_Signature2CancerType2Biosample2DNAElement2PValue_Dict.txt'
        dictPath = os.path.join(heatmaps_main_output_path, DICTIONARIES, dictFilename)
        step1_signature2cancer_type2biosample2dna_element2p_value_tuple_dict = readDictionary(dictPath)

    else:
        # Step1 Calculate p value using z-test
        # Epigenomics Signatures
        # Epigenomics All Mutations (SUBS, INDELS, DINUCS)
        # Nucleosome Signatures
        # Nucleosome All Mutations (SUBS, INDELS, DINUCS)
        # Complete P Value List
        #[signature, cancer_type, biosample, dna_element, avg_real_signal, avg_sim_signal, fold_change, min_sim_signal, max_sim_signal, pvalue, num_of_sims, num_of_sims_with_not_nan_avgs, real_data_avg_count, sim_avg_count, list(simulationsHorizontalMeans)]
        step1_p_value_df, step1_signature2cancer_type2biosample2dna_element2p_value_tuple_dict = step1_compute_p_value(window_size,
                                                                                               combined_output_path,
                                                                                               numberofSimulations,
                                                                                               normal_combined_pcawg_nonpcawg_cancer_type_2_biosample_dict,
                                                                                               hm_path,
                                                                                               ctcf_path,
                                                                                               atac_path,
                                                                                               cancer_type_signature_cutoff_number_of_mutations_average_probability_df,
                                                                                               signatures,
                                                                                               cancer_types,
                                                                                               heatmaps_main_output_path)

    # Step2 Combine p values using Fisher's method
    # Pool for biosamples and ENCDODE files
    # Step2: Filter Step1 rows such that fold_change is not (nan,None), p_value is not (nan,None), real_data_avg_count >= minimum_number_of_overlaps_required
    step2_combined_p_value_df, step2_signature2cancer_type2dna_element2combined_p_value_list_dict = step2_combine_p_values(
        step1_signature2cancer_type2biosample2dna_element2p_value_tuple_dict,
        heatmaps_main_output_path,
        combine_p_values_method,
        signature_signature_type_tuples,
        depleted_fold_change,
        enriched_fold_change,
        minimum_number_of_overlaps_required_for_sbs,
        minimum_number_of_overlaps_required_for_dbs,
        minimum_number_of_overlaps_required_for_indels,
        signature_cancer_type_number_of_mutations,
        signature_cancer_type_number_of_mutations_for_ctcf,
        consider_both_real_and_sim_avg_overlap = True)

    # Step3 Corrected combined p values
    # combined p value list
    # [fold_change_list,avg_fold_change,p_value_list,combined_p_value]
    step3_q_value_df, step3_signature2cancer_type2dna_element2q_value_list_dict = step3_apply_multiple_tests_correction(
        step2_signature2cancer_type2dna_element2combined_p_value_list_dict,
        heatmaps_main_output_path)

    # Plot Heatmap
    name_for_rows = 'signature'
    rows = signatures

    name_for_columns = 'dna_element'
    all_dna_elements = step3_q_value_df['dna_element'].unique()
    all_dna_elements = sorted(all_dna_elements, key=natural_key)

    # Remove open chromatin --- remove ATAC-seq
    all_dna_elements = [dna_element for dna_element in all_dna_elements if 'ATAC' not in dna_element]
    columns = all_dna_elements

    group_name = 'SBS28'

    # Consider only the cancer types in cancer_types
    pole_step3_q_value_df = step3_q_value_df[step3_q_value_df['cancer_type'].isin(cancer_types_pole_hypermutators)]
    non_pole_step3_q_value_df = step3_q_value_df[step3_q_value_df['cancer_type'].isin(cancer_types_non_pole_hypermutators)]

    pole_avg_fold_change_np_array = fill_rows_and_columns_np_array(name_for_rows, rows, name_for_columns, columns, pole_step3_q_value_df)
    non_pole_avg_fold_change_np_array = fill_rows_and_columns_np_array(name_for_rows, rows, name_for_columns, columns, non_pole_step3_q_value_df)

    SBS28_avg_fold_change_np_array = np.concatenate((pole_avg_fold_change_np_array, non_pole_avg_fold_change_np_array), axis=0)

    plot_heatmap(signatures_ylabels_on_the_heatmap,
            columns,
            SBS28_avg_fold_change_np_array,
            group_name,
            pole_step3_q_value_df,
            non_pole_step3_q_value_df,
            name_for_rows,
            name_for_columns,
            significance_level,
            enriched_fold_change,
            depleted_fold_change,
            plot_output_path,
            'Figure_Case_Study_SBS28_epigenomics_heatmap',
            plot_title=True)


# new ends


# copied from sigProfilerPlotting
def getylabels(ylabels):
	if max(ylabels) >=10**9:
		ylabels = ["{:.2e}".format(x) for x in ylabels]
	else:
		if max(ylabels)< 10**5:
			ylabels = ['{:,.2f}'.format(x/1000)+'k' for x in ylabels]
		elif max(ylabels)>= 10**5:
			ylabels = ['{:,.2f}'.format(x/(10**6))+'m' for x in ylabels]
	ylabels[0] = '0.00'
	return ylabels

# copied from sigProfilerPlotting
# modified pdf if changed into png
def plotSBS(matrix_path, output_path, project, plot_type, percentage=False, custom_text_upper=None, custom_text_middle=None, custom_text_bottom=None):
    plot_custom_text = False
    sig_probs = False
    pcawg = False
    if plot_type == '96':
        with open(matrix_path) as f:
            next(f)
            first_line = f.readline()
            first_line = first_line.strip().split()
            if first_line[0][1] == ">":
                pcawg = True
            if first_line[0][5] != "]" and first_line[0][1] != ">":
                sys.exit("The matrix does not match the correct SBS96 format. Please check you formatting and rerun this plotting function.")

        if project == 'SBS28 in POLE+':
            output_file_path = os.path.join(output_path + 'SBS_96_plots_' + 'SBS28_in_POLE_proficient' + '.png')
        else:
            output_file_path = os.path.join(output_path + 'SBS_96_plots_' + 'SBS28_in_POLE_deficient' + '.png')
        # pp = PdfPages(output_path + 'SBS_96_plots_' + project + '.pdf')

        mutations = OrderedDict()
        total_count = []
        try:
            with open(matrix_path) as f:
                first_line = f.readline()
                if pcawg:
                    samples = first_line.strip().split(",")
                    samples = samples[2:]
                else:
                    samples = first_line.strip().split("\t")
                    samples = samples[1:]

                for sample in samples:
                    mutations[sample] = OrderedDict()
                    mutations[sample]['C>A'] = OrderedDict()
                    mutations[sample]['C>G'] = OrderedDict()
                    mutations[sample]['C>T'] = OrderedDict()
                    mutations[sample]['T>A'] = OrderedDict()
                    mutations[sample]['T>C'] = OrderedDict()
                    mutations[sample]['T>G'] = OrderedDict()

                for lines in f:
                    if pcawg:
                        line = lines.strip().split(",")
                        mut_type = line[0]
                        nuc = line[1][0] + "[" + mut_type + "]" + line[1][2]
                        sample_index = 2
                    else:
                        line = lines.strip().split()
                        nuc = line[0]
                        mut_type = line[0][2:5]
                        sample_index = 1

                    for sample in samples:
                        if percentage:
                            mutCount = float(line[sample_index])
                            if mutCount < 1 and mutCount > 0:
                                sig_probs = True
                        else:
                            try:
                                mutCount = int(line[sample_index])
                            except:
                                print(
                                    "It appears that the provided matrix does not contain mutation counts.\n\tIf you have provided a signature activity matrix, please change the percentage parameter to True.\n\tOtherwise, ",
                                    end='')
                        mutations[sample][mut_type][nuc] = mutCount
                        sample_index += 1

            sample_count = 0

            for sample in mutations.keys():
                total_count = sum(sum(nuc.values()) for nuc in mutations[sample].values())
                plt.rcParams['axes.linewidth'] = 2
                plot1 = plt.figure(figsize=(43.93, 9.92))
                plt.rc('axes', edgecolor='lightgray')
                panel1 = plt.axes([0.04, 0.09, 0.95, 0.77])
                xlabels = []

                x = 0.4
                ymax = 0
                colors = [[3 / 256, 189 / 256, 239 / 256], [1 / 256, 1 / 256, 1 / 256], [228 / 256, 41 / 256, 38 / 256],
                          [203 / 256, 202 / 256, 202 / 256], [162 / 256, 207 / 256, 99 / 256],
                          [236 / 256, 199 / 256, 197 / 256]]
                i = 0
                for key in mutations[sample]:
                    for seq in mutations[sample][key]:
                        xlabels.append(seq[0] + seq[2] + seq[6])
                        if percentage:
                            if total_count > 0:
                                plt.bar(x, mutations[sample][key][seq] / total_count * 100, width=0.4, color=colors[i],
                                        align='center', zorder=1000)
                                if mutations[sample][key][seq] / total_count * 100 > ymax:
                                    ymax = mutations[sample][key][seq] / total_count * 100
                        else:
                            plt.bar(x, mutations[sample][key][seq], width=0.4, color=colors[i], align='center', zorder=1000)
                            if mutations[sample][key][seq] > ymax:
                                ymax = mutations[sample][key][seq]
                        x += 1
                    i += 1

                x = .043
                y3 = .87
                y = int(ymax * 1.25)
                y2 = y + 2
                for i in range(0, 6, 1):
                    panel1.add_patch(plt.Rectangle((x, y3), .15, .05, facecolor=colors[i], clip_on=False, transform=plt.gcf().transFigure))
                    x += .159

                yText = y3 + .06
                plt.text(.1, yText, 'C>A', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
                plt.text(.255, yText, 'C>G', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
                plt.text(.415, yText, 'C>T', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
                plt.text(.575, yText, 'T>A', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
                plt.text(.735, yText, 'T>C', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
                plt.text(.89, yText, 'T>G', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)

                if y <= 4:
                    y += 4

                while y % 4 != 0:
                    y += 1
                # ytick_offest = int(y/4)
                y = ymax / 1.025
                ytick_offest = float(y / 3)

                if percentage:
                    ylabs = [0, round(ytick_offest, 1), round(ytick_offest * 2, 1), round(ytick_offest * 3, 1),
                             round(ytick_offest * 4, 1)]
                    ylabels = [str(0), str(round(ytick_offest, 1)) + "%", str(round(ytick_offest * 2, 1)) + "%",
                               str(round(ytick_offest * 3, 1)) + "%", str(round(ytick_offest * 4, 1)) + "%"]
                else:
                    ylabs = [0, ytick_offest, ytick_offest * 2, ytick_offest * 3, ytick_offest * 4]
                    ylabels = [0, ytick_offest, ytick_offest * 2, ytick_offest * 3, ytick_offest * 4]

                labs = np.arange(0.375, 96.375, 1)

                font_label_size = 30
                if not percentage:
                    if int(ylabels[3]) >= 1000:
                        font_label_size = 20

                if percentage:
                    if len(ylabels) > 2:
                        font_label_size = 20

                if not percentage:
                    ylabels = getylabels(ylabels)

                panel1.set_xlim([0, 96])
                panel1.set_ylim([0, y])
                panel1.set_xticks(labs)
                panel1.set_yticks(ylabs)
                count = 0
                m = 0
                for i in range(0, 96, 1):
                    plt.text(i / 101 + .0415, .02, xlabels[i][0], fontsize=30, color='gray', rotation='vertical',
                             verticalalignment='center', fontname='Courier New', transform=plt.gcf().transFigure)
                    plt.text(i / 101 + .0415, .044, xlabels[i][1], fontsize=30, color=colors[m], rotation='vertical',
                             verticalalignment='center', fontname='Courier New', fontweight='bold',
                             transform=plt.gcf().transFigure)
                    plt.text(i / 101 + .0415, .071, xlabels[i][2], fontsize=30, color='gray', rotation='vertical',
                             verticalalignment='center', fontname='Courier New', transform=plt.gcf().transFigure)
                    count += 1
                    if count == 16:
                        count = 0
                        m += 1

                if sig_probs:
                    plt.text(0.045, 0.75, sample, fontsize=60, weight='bold', color='black', fontname="Arial",
                             transform=plt.gcf().transFigure)
                else:
                    plt.text(0.045, 0.75, sample, fontsize=60, weight='bold', color='black', fontname="Arial", transform=plt.gcf().transFigure)
                    if project == 'SBS28 in POLE+':
                        plt.text(0.877, 0.75, "{:,}".format(int(total_count)) + " subs",
                             # horizontalalignment='right', verticalalignment='top',
                             fontsize=60, weight='bold', color='black', fontname="Arial", transform=plt.gcf().transFigure)
                    elif project == 'SBS28 in POLE-':
                        plt.text(0.867, 0.75, "{:,}".format(int(total_count)) + " subs",
                             # horizontalalignment='right', verticalalignment='top',
                             fontsize=60, weight='bold', color='black', fontname="Arial", transform=plt.gcf().transFigure)

                custom_text_upper_plot = ''
                try:
                    custom_text_upper[sample_count]
                except:
                    custom_text_upper = False
                try:
                    custom_text_middle[sample_count]
                except:
                    custom_text_middle = False
                try:
                    custom_text_bottom[sample_count]
                except:
                    custom_text_bottom = False

                if custom_text_upper:
                    plot_custom_text = True
                    if len(custom_text_upper[sample_count]) > 40:
                        print("To add a custom text, please limit the string to <40 characters including spaces.")
                        plot_custom_text = False
                if custom_text_middle:
                    if len(custom_text_middle[sample_count]) > 40:
                        print("To add a custom text, please limit the string to <40 characters including spaces.")
                        plot_custom_text = False

                if plot_custom_text:
                    x_pos_custom = 0.98
                    if custom_text_upper and custom_text_middle:
                        custom_text_upper_plot = custom_text_upper[sample_count] + "\n" + custom_text_middle[
                            sample_count]
                        if custom_text_bottom:
                            custom_text_upper_plot += "\n" + custom_text_bottom[sample_count]

                    if custom_text_upper and not custom_text_middle:
                        custom_text_upper_plot = custom_text_upper[sample_count]
                        panel1.text(x_pos_custom, 0.78, custom_text_upper_plot, fontsize=40, weight='bold',
                                    color='black', fontname="Arial", transform=plt.gcf().transFigure, ha='right')

                    elif custom_text_upper and custom_text_middle:
                        if not custom_text_bottom:
                            panel1.text(x_pos_custom, 0.72, custom_text_upper_plot, fontsize=40, weight='bold',
                                        color='black', fontname="Arial", transform=plt.gcf().transFigure, ha='right')
                        else:
                            panel1.text(x_pos_custom, 0.68, custom_text_upper_plot, fontsize=40, weight='bold',
                                        color='black', fontname="Arial", transform=plt.gcf().transFigure, ha='right')

                    elif not custom_text_upper and custom_text_middle:
                        custom_text_upper_plot = custom_text_middle[sample_count]
                        panel1.text(x_pos_custom, 0.78, custom_text_upper_plot, fontsize=40, weight='bold',
                                    color='black', fontname="Arial", transform=plt.gcf().transFigure, ha='right')

                panel1.set_yticklabels(ylabels, fontsize=font_label_size)
                plt.gca().yaxis.grid(True)
                plt.gca().grid(which='major', axis='y', color=[0.93, 0.93, 0.93], zorder=1)
                panel1.set_xlabel('')
                panel1.set_ylabel('')

                if percentage:
                    plt.ylabel("Percentage of Single Base Substitutions", fontsize=35, fontname="Times New Roman",
                               weight='bold')
                else:
                    plt.ylabel("Number of Single Base Substitutions", fontsize=35, fontname="Times New Roman",
                               weight='bold')

                panel1.tick_params(axis='both', which='both', \
                                   bottom=False, labelbottom=False, \
                                   left=True, labelleft=True, \
                                   right=True, labelright=False, \
                                   top=False, labeltop=False, \
                                   direction='in', length=25, colors='lightgray', width=2)

                [i.set_color("black") for i in plt.gca().get_yticklabels()]
                plot1.savefig(output_file_path, dpi=100, bbox_inches="tight")
                plt.close()
                sample_count += 1
                # pp.savefig(plot1)
                # pp.close()
        except:
            print("There may be an issue with the formatting of your matrix file.")
            os.remove(output_path + 'SBS_96_plots_' + project + '.pdf')


def get_SBS28processive_group_length_list(rows_sbs_signatures,
                                     across_all_cancer_types_POLE_hypermutators_df,
                                     across_all_cancer_types_non_POLE_hypermutators_df,
                                     minimum_required_number_of_processive_groups):

    # rows_sbs_signatures = [('SBS28', 'POLE hypermutators'), ('SBS28', 'non-POLE hypermutators')]
    signatures_across_all_cancer_types = []

    for signature_tuple in rows_sbs_signatures:
        signature, POLE_status = signature_tuple
        if signature not in signatures_across_all_cancer_types:
            signatures_across_all_cancer_types.append(signature)

    pole_across_all_df = across_all_cancer_types_POLE_hypermutators_df[across_all_cancer_types_POLE_hypermutators_df['signature'].isin(signatures_across_all_cancer_types)]
    non_pole_across_all_df = across_all_cancer_types_non_POLE_hypermutators_df[across_all_cancer_types_non_POLE_hypermutators_df['signature'].isin(signatures_across_all_cancer_types)]

    pole_processsive_group_length_list = pole_across_all_df[
        (round(pole_across_all_df['radius'], 2) > 0) &
        (pole_across_all_df['avg_number_of_processive_groups'] >= minimum_required_number_of_processive_groups)]['processive_group_length'].unique()

    non_pole_processsive_group_length_list = non_pole_across_all_df[
        (round(non_pole_across_all_df['radius'], 2) > 0) &
        (non_pole_across_all_df['avg_number_of_processive_groups'] >= minimum_required_number_of_processive_groups)]['processive_group_length'].unique()

    processsive_group_length_list = list(set(pole_processsive_group_length_list).union(set(non_pole_processsive_group_length_list)))

    sorted_processsive_group_length_list = sorted(processsive_group_length_list, key=int)

    return  sorted_processsive_group_length_list



def ad_hoc_plot_SBS28_processivity_figures(plot_output_path,
                                     across_all_cancer_types_pooled_processivity_POLE_hypermutators_df,
                                     across_all_cancer_types_pooled_processivity_non_POLE_hypermutators_df,
                                     rows_sbs_signatures,
                                     rows_signatures_on_the_heatmap,
                                     minimum_required_number_of_processive_groups):

    processive_group_length_list = get_SBS28processive_group_length_list(rows_sbs_signatures,
                                                                    across_all_cancer_types_pooled_processivity_POLE_hypermutators_df,
                                                                    across_all_cancer_types_pooled_processivity_non_POLE_hypermutators_df,
                                                                    minimum_required_number_of_processive_groups)

    max_processive_group_length = max(processive_group_length_list)

    index = None
    if ((len(processive_group_length_list)>0) and (max_processive_group_length>0)):
        # Find index of max_processive_group_length in processive_group_length_array
        index = processive_group_length_list.index(max_processive_group_length)

    print("signature_list: ", rows_sbs_signatures)
    print("processive_group_length_list: ", processive_group_length_list)
    print("max_processive_group_length: ", max_processive_group_length)
    print("index of max_processive_group_length: ", index)

    width_multiply = 1.5
    height_multiply = 1.5

    fig = plt.figure(figsize=(width_multiply * max_processive_group_length, height_multiply * len(rows_sbs_signatures) ))

    fontsize = 40
    ax = plt.gca()
    ax.set_aspect(1.0)  # make aspect ratio square

    cmap = matplotlib_cm.get_cmap('YlOrRd')  # Looks better
    v_min = 2
    v_max = 20
    norm = plt.Normalize(v_min, v_max)

    #To get rid of  UserWarning: Attempting to set identical left==right results in singular transformations; automatically expanding.
    if (len(processive_group_length_list)>1):
        plt.xlim([1,index+1])
        ax.set_xticks(np.arange(0,index+2,1))
    else:
        plt.xlim([0,len(processive_group_length_list)])
        ax.set_xticks(np.arange(0,len(processive_group_length_list)+1,1))

    if (len(rows_sbs_signatures)>1):
        plt.ylim([1, len(rows_sbs_signatures)])
    else:
        plt.ylim([0, len(rows_sbs_signatures)])

    ax.set_yticks(np.arange(0, len(rows_sbs_signatures) + 1, 1))

    if (not across_all_cancer_types_pooled_processivity_POLE_hypermutators_df.empty) and (not across_all_cancer_types_pooled_processivity_non_POLE_hypermutators_df.empty):
        # Plot the circles with color
        for signature_index, signature_tuple in enumerate(rows_sbs_signatures):
            signature, pole_type = signature_tuple
            # rows_sbs_signatures = [('SBS28', 'POLE deficient'), ('SBS28', 'POLE proficient')]

            if pole_type == 'POLE deficient':
                number_of_processive_groups_column_name = 'avg_number_of_processive_groups'
                signature_processive_group_length_properties_df = across_all_cancer_types_pooled_processivity_POLE_hypermutators_df
            elif pole_type == 'POLE proficient':
                number_of_processive_groups_column_name = 'avg_number_of_processive_groups'
                signature_processive_group_length_properties_df = across_all_cancer_types_pooled_processivity_non_POLE_hypermutators_df

            for processive_group_length_index, processive_group_length in enumerate(processive_group_length_list):
                number_of_processive_groups = np.nan
                radius = np.nan
                color = np.nan

                if (signature_processive_group_length_properties_df[
                    (signature_processive_group_length_properties_df['signature'] == signature) &
                    (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)][number_of_processive_groups_column_name].values.any()):

                    number_of_processive_groups = signature_processive_group_length_properties_df[
                        (signature_processive_group_length_properties_df['signature'] == signature) &
                        (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)][number_of_processive_groups_column_name].values[0]

                if (signature_processive_group_length_properties_df[
                    (signature_processive_group_length_properties_df['signature']==signature) &
                    (signature_processive_group_length_properties_df['processive_group_length']==processive_group_length)]['radius'].values.any()):

                    radius = signature_processive_group_length_properties_df[
                        (signature_processive_group_length_properties_df['signature']==signature) &
                        (signature_processive_group_length_properties_df['processive_group_length']==processive_group_length)]['radius'].values[0]

                if (signature_processive_group_length_properties_df[
                    (signature_processive_group_length_properties_df['signature']==signature) &
                    (signature_processive_group_length_properties_df['processive_group_length']==processive_group_length)]['minus_log10_qvalue'].values.any()):

                    color = signature_processive_group_length_properties_df[
                        (signature_processive_group_length_properties_df['signature']==signature) &
                        (signature_processive_group_length_properties_df['processive_group_length']==processive_group_length)]['minus_log10_qvalue'].values[0]

                if ((not np.isnan(number_of_processive_groups)) and (number_of_processive_groups >= minimum_required_number_of_processive_groups)
                        and (not np.isnan(radius)) and (radius>0) and (not np.isnan(color))):
                    #Very important: You have to norm
                    circle = plt.Circle((processive_group_length_index + 0.5, signature_index + 0.5), radius, color=cmap(norm(color)), fill=True)
                    ax.add_artist(circle)
                elif ((not np.isnan(number_of_processive_groups)) and (number_of_processive_groups >= minimum_required_number_of_processive_groups)
                            and (not np.isnan(radius)) and (radius>0) and np.isnan(color)):
                    circle = plt.Circle((processive_group_length_index + 0.5, signature_index + 0.5), radius, color="lightgray",fill=True)
                    ax.add_artist(circle)

    ax.set_facecolor('white')
    ax.grid(color='black')

    for edge, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')

    xlabels = None
    if (index is not None):
        xlabels = processive_group_length_list[0:index+1]
    ylabels = rows_signatures_on_the_heatmap

    figures_path = os.path.join(plot_output_path, PROCESSIVITY)
    # plot colorbar in a separate figure
    plot_processivity_colorbar_vertical(figures_path, cmap, v_min, v_max)
    plot_processivity_colorbar_horizontal(figures_path, cmap, v_min, v_max)
    # plot legend in a separate figure
    plot_processivity_legend(figures_path, 30)

    # CODE GOES HERE TO CENTER X-AXIS LABELS...
    ax.set_xticklabels([])
    mticks = ax.get_xticks()

    ax.set_xticks((mticks[:-1] + mticks[1:]) / 2, minor=True)
    ax.tick_params(axis='x', which='minor', length=0, labelsize=fontsize)

    if xlabels is not None:
        ax.set_xticklabels(xlabels, minor=True)
        ax.set_xlabel('SBS28\nStrand-coordinated Mutagenesis Group Length',fontsize=fontsize, labelpad=20)
        ax.xaxis.set_label_position('top')

    ax.xaxis.set_ticks_position('top')

    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='major',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False)  # labels along the bottom edge are off

    # CODE GOES HERE TO CENTER Y-AXIS LABELS...
    ax.set_yticklabels([])
    mticks = ax.get_yticks()
    ax.set_yticks((mticks[:-1] + mticks[1:]) / 2, minor=True)
    ax.tick_params(axis='y', which='minor', length=0, labelsize=fontsize)
    ax.set_yticklabels(ylabels, minor=True) # fontsize

    ax.tick_params(
        axis='y',  # changes apply to the x-axis
        which='major',  # both major and minor ticks are affected
        left=False)  # labels along the bottom edge are off

    #create the directory if it does not exists
    filename = 'Figure_Case_Study_SBS28_Processivity.png'
    figFile = os.path.join(figures_path, filename)
    fig.savefig(figFile,dpi=100, bbox_inches="tight")

    plt.cla()
    plt.close(fig)


def figure_case_study_strand_coordinated_mutagenesis(plot_output_path,
                                                     combined_output_path,
                                                     figure_types,
                                                     cosmic_release_version,
                                                     figure_file_extension):

    rows_sbs_signatures = [('SBS28', 'POLE deficient'), ('SBS28', 'POLE proficient')]
    rows_signatures_on_the_heatmap = ['POLE-', 'POLE+']

    cancer_types_POLE_hypermutators = ['ColoRect-AdenoCA', 'Uterus-AdenoCA', 'ESCC'] # same for discreet mode and prob_mode_05
    # cancer_types_non_POLE_hypermutators = ['Lung-AdenoCA', 'Stomach-AdenoCA'] # discreet mode
    cancer_types_non_POLE_hypermutators = ['Lung-AdenoCA', 'Stomach-AdenoCA', 'Panc-AdenoCA', 'Eso-AdenoCA'] # prob_mode_05

    processivity_significance_level = 0.05
    minimum_required_processive_group_length = 2
    minimum_required_number_of_processive_groups = 2

    rows_sbs_signatures = list(reversed(rows_sbs_signatures))
    rows_signatures_on_the_heatmap = list(reversed(rows_signatures_on_the_heatmap))

    number_of_simulations = 100

    all_cancer_types_processivity_POLE_hypermutators_df, \
    across_all_cancer_types_pooled_processivity_POLE_hypermutators_df = generate_processivity_pdf(plot_output_path,
                              combined_output_path,
                              cancer_types_POLE_hypermutators,
                              number_of_simulations,
                              figure_types,
                              cosmic_release_version,
                              figure_file_extension,
                              processivity_significance_level,
                              minimum_required_processive_group_length,
                              minimum_required_number_of_processive_groups)

    all_cancer_types_processivity_non_POLE_hypermutators_df, \
    across_all_cancer_types_pooled_processivity_non_POLE_hypermutators_df = generate_processivity_pdf(plot_output_path,
                              combined_output_path,
                              cancer_types_non_POLE_hypermutators,
                              number_of_simulations,
                              figure_types,
                              cosmic_release_version,
                              figure_file_extension,
                              processivity_significance_level,
                              minimum_required_processive_group_length,
                              minimum_required_number_of_processive_groups)

    ad_hoc_plot_SBS28_processivity_figures(plot_output_path,
                                     across_all_cancer_types_pooled_processivity_POLE_hypermutators_df,
                                     across_all_cancer_types_pooled_processivity_non_POLE_hypermutators_df,
                                     rows_sbs_signatures,
                                     rows_signatures_on_the_heatmap,
                                     minimum_required_number_of_processive_groups)



def main():
    plot_mutational_context = False
    calculate_cosine_similarity = False
    occupancy = True
    replication_time = False
    strand_bias = False
    strand_coordinated_mutagenesis = False
    epigenomics_heatmap = False

    # Common parameters
    figure_types = [COSMIC, MANUSCRIPT]

    # discreet_mode input directory
    # combined_output_path = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'SigProfilerTopographyRuns', 'Combined_PCAWG_nonPCAWG_4th_iteration')

    # discreet_mode input directory
    # plot_output_path = os.path.join('/oasis', 'tscc', 'scratch', 'burcak',
    #                                 'SigProfilerTopographyRuns',
    #                                 'combined_pcawg_and_nonpcawg_figures_pdfs',
    #                                 '4th_iteration',
    #                                 'Figure_Case_Study_SBS28/')


    # prob_mode_05 input directory
    combined_output_path = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'SigProfilerTopographyRuns',
                                        'Combined_PCAWG_nonPCAWG_prob_mode_05')

    # prob_mode_05 output directory
    plot_output_path = os.path.join('/oasis', 'tscc', 'scratch', 'burcak',
                                    'SigProfilerTopographyRuns',
                                    'combined_pcawg_and_nonpcawg_figures_pdfs',
                                    'prob_mode_05',
                                    'Figure_Case_Study_SBS28/')


    os.makedirs(plot_output_path, exist_ok=True)

    # POLE+
    # POLE proficient samples
    # non-POLE hypermutators
    cancer_types = ['Lung-AdenoCA', 'Stomach-AdenoCA'] # discreet_mode
    cancer_types = ['Lung-AdenoCA', 'Stomach-AdenoCA', 'Panc-AdenoCA', 'Eso-AdenoCA'] # prob_mode_05
    mutational_profile_top_left_text = 'SBS28 in POLE+'
    # sub_figure_type = 'POLE+' # for occupancy and replication time - figures are under figures_cosmic directory
    sub_figure_type = 'SBS28 POLE+' # for strand bias - figures are under strand_bias directory

    # # POLE-
    # # POLE deficient samples
    # # POLE hypermutators
    # cancer_types = ['ColoRect-AdenoCA', 'Uterus-AdenoCA', 'ESCC']
    # mutational_profile_top_left_text = 'SBS28 in POLE-'
    # sub_figure_type = 'POLE-' # for occupancy and replication time - figures are under figures_cosmic
    # sub_figure_type = 'SBS28 POLE-' # for strand bias - figures are under strand_bias directory

    # cancer_types = ['ColoRect-AdenoCA', 'Uterus-AdenoCA', 'ESCC', 'Lung-AdenoCA', 'Stomach-AdenoCA']

    sbs_signatures = ['SBS28']
    dbs_signatures = []
    id_signatures = []

    cosmic_release_version = 'v3.2'
    figure_file_extension = 'jpg'
    numberofSimulations = 100

    if epigenomics_heatmap:
        # Figure SBS28.png is under /oasis/tscc/scratch/burcak/SigProfilerTopographyRuns/combined_pcawg_and_nonpcawg_figures_pdfs/4th_iteration/Figure_Case_Study_SBS28

        combine_p_values_method = 'fisher'
        depleted_fold_change = 0.95
        enriched_fold_change = 1.05
        significance_level = 0.05

        minimum_number_of_overlaps_required_for_sbs = 100
        minimum_number_of_overlaps_required_for_dbs = 100
        minimum_number_of_overlaps_required_for_indels = 100

        signature_cancer_type_number_of_mutations = AT_LEAST_1K_CONSRAINTS
        signature_cancer_type_number_of_mutations_for_ctcf = AT_LEAST_1K_CONSRAINTS

        step1_data_ready = True
        window_size = 100
        tissue_type = 'Normal'
        heatmaps_dir_name = "heatmaps_dna_elements_window_size_%s_%s" % (window_size, tissue_type)

        # # discreet_mode
        # heatmaps_main_output_path = os.path.join('/oasis', 'tscc', 'scratch', 'burcak', 'SigProfilerTopographyRuns',
        #                                          'combined_pcawg_and_nonpcawg_figures_pdfs', '4th_iteration',
        #                                          heatmaps_dir_name)

        # prob_mode_05
        heatmaps_main_output_path = os.path.join('/oasis', 'tscc', 'scratch', 'burcak', 'SigProfilerTopographyRuns',
                                                 'combined_pcawg_and_nonpcawg_figures_pdfs', 'prob_mode_05',
                                                 heatmaps_dir_name)


        hm_path = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'data', 'ENCODE', 'GRCh37', 'HM')
        ctcf_path = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'data', 'ENCODE', 'GRCh37', 'CTCF')
        atac_path = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'data', 'ENCODE', 'GRCh37', 'ATAC_seq')

        # Order must be consistent in these 4 lists below
        epigenomics_output_path = os.path.join(plot_output_path, 'epigenomics')

        os.makedirs(os.path.join(epigenomics_output_path, TABLES), exist_ok=True)
        os.makedirs(os.path.join(epigenomics_output_path, DATA_FILES), exist_ok=True)
        os.makedirs(os.path.join(epigenomics_output_path, EXCEL_FILES), exist_ok=True)
        os.makedirs(os.path.join(epigenomics_output_path, DICTIONARIES), exist_ok=True)

        consider_both_real_and_sim_avg_overlap = True
        sort_cancer_types = False
        remove_columns_rows_with_no_significant_result = False

        # Order must be consistent in these 4 lists below
        signatures = ['SBS28']
        signature_signature_type_tuples = [('SBS28', SBS)]
        signatures_ylabels_on_the_heatmap = [('POLE-'), ('POLE+') ]
        figure_type = MANUSCRIPT

        # # discreet_mode
        # cancer_types = ['ColoRect-AdenoCA', 'Uterus-AdenoCA', 'ESCC', 'Lung-AdenoCA', 'Stomach-AdenoCA']
        # cancer_types_pole_hypermutators = ['ColoRect-AdenoCA', 'Uterus-AdenoCA', 'ESCC']
        # cancer_types_non_pole_hypermutators = ['Lung-AdenoCA', 'Stomach-AdenoCA']

        # prob_mode_05
        cancer_types = ['ColoRect-AdenoCA', 'Uterus-AdenoCA', 'ESCC', 'Lung-AdenoCA', 'Stomach-AdenoCA', 'Panc-AdenoCA', 'Eso-AdenoCA']
        cancer_types_pole_hypermutators = ['ColoRect-AdenoCA', 'Uterus-AdenoCA', 'ESCC']
        cancer_types_non_pole_hypermutators = ['Lung-AdenoCA', 'Stomach-AdenoCA', 'Panc-AdenoCA', 'Eso-AdenoCA']

        figure_case_study_SBS28_epigenomics_heatmap(
            combined_output_path,
            heatmaps_main_output_path,
            hm_path,
            ctcf_path,
            atac_path,
            plot_output_path,
            signatures,
            signature_signature_type_tuples,
            signatures_ylabels_on_the_heatmap,
            cancer_types,
            cancer_types_pole_hypermutators,
            cancer_types_non_pole_hypermutators,
            combine_p_values_method,
            window_size,
            numberofSimulations,
            enriched_fold_change,
            depleted_fold_change,
            significance_level,
            minimum_number_of_overlaps_required_for_sbs,
            minimum_number_of_overlaps_required_for_dbs,
            minimum_number_of_overlaps_required_for_indels,
            signature_cancer_type_number_of_mutations,
            signature_cancer_type_number_of_mutations_for_ctcf,
            step1_data_ready,
            figure_type,
            cosmic_release_version,
            figure_file_extension)

    if calculate_cosine_similarity:
        discreet_mode = False

        if discreet_mode:
            pole_deficient_df = pd.read_csv(
                '/oasis/tscc/scratch/burcak/SigProfilerTopographyRuns/combined_pcawg_and_nonpcawg_figures_pdfs/4th_iteration/Figure_Case_Study_SBS28/SBS28_POLE_deficient.SBS96.all',
                sep='\t', header=0)
            pole_proficient_df = pd.read_csv(
                '/oasis/tscc/scratch/burcak/SigProfilerTopographyRuns/combined_pcawg_and_nonpcawg_figures_pdfs/4th_iteration/Figure_Case_Study_SBS28/SBS28_POLE_proficient.SBS96.all',
                sep='\t', header=0)
        else:
            pole_deficient_df = pd.read_csv(
                '/oasis/tscc/scratch/burcak/SigProfilerTopographyRuns/combined_pcawg_and_nonpcawg_figures_pdfs/prob_mode_05/Figure_Case_Study_SBS28/SBS28_POLE_deficient.SBS96.all',
                sep='\t', header=0)
            pole_proficient_df = pd.read_csv(
                '/oasis/tscc/scratch/burcak/SigProfilerTopographyRuns/combined_pcawg_and_nonpcawg_figures_pdfs/prob_mode_05/Figure_Case_Study_SBS28/SBS28_POLE_proficient.SBS96.all',
                sep='\t', header=0)


        print('pole_deficient_df.shape:', pole_deficient_df.shape, 'pole_deficient_df.columns.values:', pole_deficient_df.columns.values)
        print(type(pole_deficient_df['SBS28 in POLE-'].values), pole_deficient_df['SBS28 in POLE-'].values)
        print('pole_proficient_df.shape:', pole_proficient_df.shape, 'pole_proficient_df.columns.values:', pole_proficient_df.columns.values)
        print(type(pole_proficient_df['SBS28 in POLE+'].values), pole_proficient_df['SBS28 in POLE+'].values)
        arr1 = pole_deficient_df['SBS28 in POLE-'].values
        arr2 = pole_proficient_df['SBS28 in POLE+'].values
        cosine_similarity = dot(arr1, arr2) / (norm(arr1) * norm(arr2))
        print('Cosine similarity:', cosine_similarity)

    if plot_mutational_context:
        discreet_mode = False

        # POLE+
        # POLE proficient samples
        # non-POLE hypermutators
        # cancer_types = ['Lung-AdenoCA', 'Stomach-AdenoCA']  # discreet_mode
        # cancer_types = ['Lung-AdenoCA', 'Stomach-AdenoCA', 'Panc-AdenoCA', 'Eso-AdenoCA']  # prob_mode_05
        # mutational_profile_top_left_text = 'SBS28 in POLE+'

        # # POLE-
        # # POLE deficient samples
        # # POLE hypermutators
        cancer_types = ['ColoRect-AdenoCA', 'Uterus-AdenoCA', 'ESCC'] # same for discreet_mode and prob_mode_05
        mutational_profile_top_left_text = 'SBS28 in POLE-'

        signature = 'SBS28'

        # # discreet_mode
        # cancer_type2source_cancer_type_tuples_dict = {
        #     'ColoRect-AdenoCA': [(PCAWG, 'ColoRect-AdenoCA'), (nonPCAWG, 'ColoRect-AdenoCa')],
        #     'Uterus-AdenoCA': [(PCAWG, 'Uterus-AdenoCA')],
        #     'ESCC': [(MUTOGRAPHS, 'ESCC')],
        #     'Lung-AdenoCA': [(PCAWG, 'Lung-AdenoCA'), (nonPCAWG, 'Lung-AdenoCa')],
        #     'Stomach-AdenoCA': [(PCAWG, 'Stomach-AdenoCA'), (nonPCAWG, 'Stomach-AdenoCa')]
        # }

        # prob_mode_05
        cancer_type2source_cancer_type_tuples_dict = {
            'ColoRect-AdenoCA': [(PCAWG, 'ColoRect-AdenoCA'), (nonPCAWG, 'ColoRect-AdenoCa')],
            'Uterus-AdenoCA': [(PCAWG, 'Uterus-AdenoCA')],
            'ESCC': [(MUTOGRAPHS, 'ESCC')],
            'Lung-AdenoCA': [(PCAWG, 'Lung-AdenoCA'), (nonPCAWG, 'Lung-AdenoCa')],
            'Stomach-AdenoCA': [(PCAWG, 'Stomach-AdenoCA'), (nonPCAWG, 'Stomach-AdenoCa')],
            'Panc-AdenoCA' : [(PCAWG, 'Panc-AdenoCA'), (nonPCAWG, 'Panc-AdenoCa')],
            'Eso-AdenoCA' : [(PCAWG, 'Eso-AdenoCA'), (nonPCAWG, 'Eso-AdenoCa')]
        }

        all_df_list = []
        for cancer_type in cancer_types:
            source_cancer_type_tuples = cancer_type2source_cancer_type_tuples_dict[cancer_type]
            for source_cancer_type_tuple in source_cancer_type_tuples:
                source, source_specific_cancer_type = source_cancer_type_tuple

                if source == PCAWG:
                    matrix_file = os.path.join('/restricted/alexandrov-group/burcak/data/' +
                                               source + '/' +
                                               source_specific_cancer_type + '/filtered/output/SBS/' +
                                               source_specific_cancer_type + ".SBS96.all")

                    probabilities_file = os.path.join('/home/burcak/developer/SigProfilerTopographyRuns/' +
                                                      source + '/probabilities/' +
                                                      source_specific_cancer_type + '_sbs96_mutation_probabilities.txt')
                    matrix_mutation_type_column = 'MutationType'
                    prob_sample_column = 'Sample Names'
                    prob_mutation_type_column = 'MutationTypes'

                elif source == nonPCAWG:
                    matrix_file = os.path.join('/restricted/alexandrov-group/burcak/data/' +
                                               source + '/' +
                                               source_specific_cancer_type + '/output/SBS/' +
                                               source_specific_cancer_type + ".SBS96.all")

                    probabilities_file = os.path.join('/home/burcak/developer/SigProfilerTopographyRuns/' +
                                                      source + '/probabilities/' +
                                                      source_specific_cancer_type + '_subs_probabilities.txt')
                    matrix_mutation_type_column = 'MutationType'
                    prob_sample_column = 'Sample'
                    prob_mutation_type_column = 'Mutation'

                elif source == MUTOGRAPHS:
                    matrix_file = os.path.join('/restricted/alexandrov-group/burcak/data/' +
                                               'Mutographs_ESCC_552' + '/' +
                                               'all_samples' + '/output/SBS/' +
                                               'All_Samples_552.SBS96.all')

                    probabilities_file = os.path.join('/home','burcak','developer','SigProfilerTopographyRuns',
                                                      'Mutographs_ESCC_552','manuscript_probabilities',
                                                      'SBS288_Decomposed_Mutation_Probabilities.txt')
                    matrix_mutation_type_column = 'MutationType'
                    prob_sample_column = 'Sample Names'
                    prob_mutation_type_column = 'MutationTypes'

                topography_cutoffs_file = os.path.join('/restricted/alexandrov-group/burcak/SigProfilerTopographyRuns/Combined_PCAWG_nonPCAWG_4th_iteration' + '/' + cancer_type + '/data/Table_SBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt')
                topography_cutoffs_df = pd.read_csv(topography_cutoffs_file, sep='\t')

                if np.any(topography_cutoffs_df[topography_cutoffs_df['signature'] == signature]['cutoff'].values):
                    cutoff = topography_cutoffs_df[topography_cutoffs_df['signature'] == signature]['cutoff'].values[0]
                else:
                    print('Combined_PCAWG_nonPCAWG', cancer_type, signature, " No cutoff  is available")

                # matrix_df
                # 'MutationType' 'COAD-US_SP119755' 'COAD-US_SP16886' 'COAD-US_SP16934' ...
                matrix_df = pd.read_csv(matrix_file, sep='\t')

                # probabilities_df
                # 'Sample Names' 'MutationTypes' 'SBS1' 'SBS2' 'SBS3'  ....
                probabilities_df = pd.read_csv(probabilities_file, sep='\t')

                # Get sample names
                matrix_samples = matrix_df.columns.values[1:]

                # Fill with all samples in this source and source_specific_cancer_type
                df_list = []
                for matrix_sample in matrix_samples:
                    if source == PCAWG:
                        prob_sample = source_specific_cancer_type + '_' + matrix_sample.split('_')[1]
                    else:
                        prob_sample = matrix_sample

                    # sample_based_matrix_df:
                    # MutationType  COAD-US_SP16958
                    # A[C>A]A              463
                    # A[C>A]C              215
                    # ...
                    sample_based_matrix_df = matrix_df[[matrix_mutation_type_column, matrix_sample]]

                    # sample_based_prob_df:
                    # MutationTypes  SBS28
                    # A[C>A]A    0.0
                    # A[C>A]C    0.0
                    # ...
                    sample_based_prob_df = probabilities_df[(probabilities_df[prob_sample_column] == prob_sample)][[prob_mutation_type_column, signature]]

                    # merged_df:
                    # MutationType  COAD-US_SP16886 MutationTypes     SBS28
                    # A[C>A]A             1436       A[C>A]A  0.098285
                    # A[C>A]C             2089       A[C>A]C  0.230870
                    # A[C>A]G              225       A[C>A]G  0.300540
                    merged_df = pd.merge(sample_based_matrix_df, sample_based_prob_df, how='inner',
                                         left_on = matrix_mutation_type_column,
                                         right_on = prob_mutation_type_column)

                    # merged_df:
                    # MutationType  ColoRect-AdenoCA_SP16958
                    # A[C>A]A                         0
                    # A[C>A]C                         0
                    # A[C>A]G                         0

                    if discreet_mode:
                        # Discreet Mode: 0 or 1 * merged_df[matrix_sample]
                        merged_df.loc[(merged_df[signature] < cutoff), prob_sample] = 0
                        merged_df.loc[(merged_df[signature] >= cutoff), prob_sample] = merged_df[matrix_sample]
                    else:
                        # Probability Mode: prob * merged_df[matrix_sample]
                        # merged_df[matrix_sample] contains number of mutations
                        merged_df[prob_sample] = merged_df[signature] * merged_df[matrix_sample]

                    merged_df = merged_df[[matrix_mutation_type_column, prob_sample]]
                    merged_df[prob_sample] = merged_df[prob_sample].astype(np.int32)

                    if not merged_df.empty:
                        df_list.append(merged_df)

                # Merge all dfs in df_list
                df = reduce(lambda x, y: pd.merge(x, y, how='inner', left_on = matrix_mutation_type_column, right_on = matrix_mutation_type_column), df_list)
                column_name = source + '_' + source_specific_cancer_type + '_Samples'

                # sum all number of mutations row-based
                # df[column_name] = df.sum(axis=1)
                df[column_name] = df.iloc[:, 1:].sum(axis=1)

                # Drop all columns except the first and the last one
                # df.drop((df.columns.values[1:-1]), axis=1, inplace=True)
                df = df[[matrix_mutation_type_column, column_name]]

                number_of_mutations = df[column_name].sum()
                if number_of_mutations > 0:
                    all_df_list.append(df)

        all_df = reduce(lambda x, y: pd.merge(x, y, on=matrix_mutation_type_column), all_df_list)
        # file_name = signature + ".txt"
        # file_path = os.path.join(output_path, file_name)
        # all_df.to_csv(file_path, sep='\t', index=False, header=True)

        column_name = mutational_profile_top_left_text
        # all_df[column_name] = all_df.mean(axis=1)
        all_df[column_name] = all_df.iloc[:, 1:].mean(axis=1)

        all_df[column_name] = all_df[column_name].astype(np.int32)
        all_df.drop((all_df.columns.values[1:-1]), axis=1, inplace=True)

        if mutational_profile_top_left_text == 'SBS28 in POLE+':
            file_name = 'SBS28_POLE_proficient.SBS96.all'
        else:
            file_name = 'SBS28_POLE_deficient.SBS96.all'
        SBS96_all_path  = os.path.join(plot_output_path, file_name)
        all_df.to_csv(SBS96_all_path, sep='\t', header=True, index=False)

        plotSBS(SBS96_all_path, plot_output_path, mutational_profile_top_left_text, "96", percentage=False)

    if occupancy:
        # figures are under figures_cosmic directory

        # POLE+
        # POLE proficient samples
        # non-POLE hypermutators
        # cancer_types = ['Lung-AdenoCA', 'Stomach-AdenoCA']  # discreet_mode
        # cancer_types = ['Lung-AdenoCA', 'Stomach-AdenoCA', 'Panc-AdenoCA', 'Eso-AdenoCA']  # prob_mode_05
        # sub_figure_type = 'POLE+'  # for strand bias - figures are under strand_bias directory

        # # POLE-
        # # POLE deficient samples
        # # POLE hypermutators
        cancer_types = ['ColoRect-AdenoCA', 'Uterus-AdenoCA', 'ESCC'] # same for discreet_mode and prob_mode_05
        sub_figure_type = 'POLE-'

        dna_elements = [(CTCF, EPIGENOMICS_OCCUPANCY), (NUCLEOSOME, NUCLEOSOME_OCCUPANCY)]
        # dna_elements = [(NUCLEOSOME, NUCLEOSOME_OCCUPANCY)]
        # dna_elements = [(CTCF, EPIGENOMICS_OCCUPANCY)]

        minimum_number_of_overlaps_required_for_sbs = 100
        minimum_number_of_overlaps_required_for_dbs = 100
        minimum_number_of_overlaps_required_for_indels = 100

        number_of_mutations_required_list = [AT_LEAST_1K_CONSRAINTS]
        number_of_mutations_required_list_for_ctcf = [AT_LEAST_1K_CONSRAINTS]
        pearson_spearman_correlation_cutoff = 0.5
        figure_case_study = sub_figure_type

        # output files will be under .../figures_cosmic
        # /oasis/tscc/scratch/burcak/SigProfilerTopographyRuns/combined_pcawg_and_nonpcawg_figures_pdfs/4th_iteration/Figure_Case_Study_SBS28/occupancy/nucleosome/figures_cosmic
        figure_case_study_occupancy(combined_output_path,
                                plot_output_path,
                                figure_file_extension,
                                sbs_signatures,
                                dbs_signatures,
                                id_signatures,
                                cancer_types,
                                numberofSimulations,
                                figure_types,
                                cosmic_release_version,
                                pearson_spearman_correlation_cutoff,
                                dna_elements,
                                minimum_number_of_overlaps_required_for_sbs,
                                minimum_number_of_overlaps_required_for_dbs,
                                minimum_number_of_overlaps_required_for_indels,
                                number_of_mutations_required_list,
                                number_of_mutations_required_list_for_ctcf,
                                figure_case_study)

    if strand_bias:
        # for strand bias - figures are under strand_bias directory

        # # POLE proficient
        # # cancer_types = ['Lung-AdenoCA', 'Stomach-AdenoCA']  # discreet_mode
        # cancer_types = ['Lung-AdenoCA', 'Stomach-AdenoCA', 'Panc-AdenoCA', 'Eso-AdenoCA']  # prob_mode_05
        # sub_figure_type = 'SBS28 POLE+'

        # POLE deficient
        cancer_types = ['ColoRect-AdenoCA', 'Uterus-AdenoCA', 'ESCC'] # same for discreet_mode and prob_mode_05
        sub_figure_type = 'SBS28 POLE-'  # for strand bias - figures are under strand_bias directory

        strand_bias_output_path = os.path.join(plot_output_path, 'strand_bias')
        # We have to select both strand bias
        strand_biases = [REPLICATIONSTRANDBIAS, TRANSCRIPTIONSTRANDBIAS]
        significance_level = 0.05
        min_required_number_of_mutations_on_strands = 1000
        min_required_percentage_of_mutations_on_strands = 5
        number_of_required_mutations_for_stacked_bar_plot = 1

        signature2cancer_type_list_dict = {'SBS28': cancer_types}

        # Percentage numbers are parametric can be changed here
        percentage_numbers = [10, 20, 30, 50, 75, 100]
        percentage_strings = ['%d' % (percentage_number) + '%' for percentage_number in percentage_numbers]

        cancer_type2source_cancer_type_tuples_dict = {
            'ALL': [(nonPCAWG, 'ALL')],
            'Bladder-TCC': [(PCAWG, 'Bladder-TCC')],
            'Bone-Benign': [(PCAWG, 'Bone-Benign')],
            'Bone-Osteosarc': [(PCAWG, 'Bone-Osteosarc'), (nonPCAWG, 'Sarcoma-bone'), (nonPCAWG, 'Bone-cancer')],
            'CNS-GBM': [(PCAWG, 'CNS-GBM'), (nonPCAWG, 'CNS-GBM'), (nonPCAWG, 'CNS-Glioma-NOS')],
            'CNS-Medullo': [(PCAWG, 'CNS-Medullo'), (nonPCAWG, 'CNS-Medullo')],
            'CNS-PiloAstro': [(PCAWG, 'CNS-PiloAstro')],
            'ColoRect-AdenoCA': [(PCAWG, 'ColoRect-AdenoCA'), (nonPCAWG, 'ColoRect-AdenoCa')],
            'Ewings': [(nonPCAWG, 'Ewings')],
            'Head-SCC': [(PCAWG, 'Head-SCC')],
            'Kidney-RCC': [(PCAWG, 'Kidney-RCC'), (nonPCAWG, 'Kidney-RCC')],
            'Lung-AdenoCA': [(PCAWG, 'Lung-AdenoCA'), (nonPCAWG, 'Lung-AdenoCa')],
            'Lymph-BNHL': [(PCAWG, 'Lymph-BNHL'), (nonPCAWG, 'Lymph-BNHL')],
            'Myeloid-AML': [(PCAWG, 'Myeloid-AML'), (nonPCAWG, 'AML')],
            'Myeloid-MPN': [(PCAWG, 'Myeloid-MPN')],
            'Panc-AdenoCA': [(PCAWG, 'Panc-AdenoCA'), (nonPCAWG, 'Panc-AdenoCa')],
            'Prost-AdenoCA': [(PCAWG, 'Prost-AdenoCA'), (nonPCAWG, 'Prost-AdenoCa')],
            'SoftTissue-Leiomyo': [(PCAWG, 'SoftTissue-Leiomyo')],
            'Stomach-AdenoCA': [(PCAWG, 'Stomach-AdenoCA'), (nonPCAWG, 'Stomach-AdenoCa')],
            'Uterus-AdenoCA': [(PCAWG, 'Uterus-AdenoCA')],
            'Biliary-AdenoCA': [(PCAWG, 'Biliary-AdenoCA'), (nonPCAWG, 'Biliary-AdenoCa')],
            'Blood-CMDI': [(nonPCAWG, 'Blood-CMDI')],
            'Bone-Epith': [(PCAWG, 'Bone-Epith')],
            'Breast-Cancer': [(PCAWG, 'Breast-AdenoCA'), (PCAWG, 'Breast-DCIS'), (PCAWG, 'Breast-LobularCA'),
                              (nonPCAWG, 'Breast-cancer')],
            'CNS-LGG': [(nonPCAWG, 'CNS-LGG')],
            'CNS-Oligo': [(PCAWG, 'CNS-Oligo')],
            'Cervix-Cancer': [(PCAWG, 'Cervix-AdenoCA'), (PCAWG, 'Cervix-SCC')],
            'Eso-AdenoCA': [(PCAWG, 'Eso-AdenoCA'), (nonPCAWG, 'Eso-AdenoCa')],
            'ESCC': [(MUTOGRAPHS, 'ESCC')],
            'Eye-Melanoma': [(nonPCAWG, 'Eye-Melanoma')],
            'Kidney-ChRCC': [(PCAWG, 'Kidney-ChRCC')],
            'Liver-HCC': [(PCAWG, 'Liver-HCC'), (nonPCAWG, 'Liver-HCC')],
            'Lung-SCC': [(PCAWG, 'Lung-SCC')],
            'Lymph-CLL': [(PCAWG, 'Lymph-CLL'), (nonPCAWG, 'Lymph-CLL')],
            'Myeloid-MDS': [(PCAWG, 'Myeloid-MDS')],
            'Ovary-AdenoCA': [(PCAWG, 'Ovary-AdenoCA'), (nonPCAWG, 'Ovary-AdenoCa')],
            'Panc-Endocrine': [(PCAWG, 'Panc-Endocrine'), (nonPCAWG, 'Panc-Endocrine')],
            'Skin-Melanoma': [(PCAWG, 'Skin-Melanoma'), (nonPCAWG, 'Skin-Melanoma')],
            'SoftTissue-Liposarc': [(PCAWG, 'SoftTissue-Liposarc'), (nonPCAWG, 'Sarcoma')],
            'Thy-AdenoCA': [(PCAWG, 'Thy-AdenoCA')]}

        # POLE deficient
        # rows_sbs_signatures = [('SBS28', None),
        #                        ('SBS28', 'ColoRect-AdenoCA'),
        #                        ('SBS28', 'Uterus-AdenoCA'),
        #                        ('SBS28', 'ESCC')]
        #
        # rows_sbs_signatures_on_the_heatmap = ['SBS28\nPOLE hypermutators',
        #                                   'SBS28 ColoRect-AdenoCA',
        #                                   'SBS28 Uterus-AdenoCA',
        #                                   'SBS28 ESCC']

        # rows_sbs_signatures = [('SBS28', None),
        #                        ('SBS28', 'Lung-AdenoCA'),
        #                        ('SBS28', 'Stomach-AdenoCA'),
        #                        ('SBS28', 'Eso-AdenoCA'),
        #                        ('SBS28', 'Panc-AdenoCA')]
        #
        # rows_sbs_signatures_on_the_heatmap = ['SBS28\nnon-POLE hypermutators',
        #                                     'SBS28 Lung-AdenoCA',
        #                                     'SBS28 Stomach-AdenoCA',
        #                                     'SBS28 Eso-AdenoCA',
        #                                     'SBS28 Panc-AdenoCA']

        # Combined PCAWG nonPCAWG Strand Bias
        # Separate bar plot is plotted here for Figure Case Study SBS28
        # bar plot is plotted in COSMIC for each cancer type and COSMIC across all cancer types
        # in this function: plot_six_mutations_sbs_signatures_bar_plot_circles_across_all_tissues_and_tissue_based_together
        plot_strand_bias_figures(combined_output_path,
                                 cancer_types,
                                 sbs_signatures,
                                 dbs_signatures,
                                 id_signatures,
                                 strand_bias_output_path,
                                 strand_biases,
                                 percentage_numbers,
                                 percentage_strings,
                                 signature2cancer_type_list_dict,
                                 cancer_type2source_cancer_type_tuples_dict,
                                 figure_types,
                                 significance_level,
                                 min_required_number_of_mutations_on_strands,
                                 min_required_percentage_of_mutations_on_strands,
                                 number_of_required_mutations_for_stacked_bar_plot,
                                 cosmic_release_version,
                                 figure_file_extension,
                                 round_mode = False,
                                 discreet_mode = False,
                                 inflate_mutations_to_remove_TC_NER_effect = False,
                                 consider_only_significant_results = False,
                                 consider_also_DBS_ID_signatures = True,
                                 fold_enrichment = ODDS_RATIO,
                                 figure_case_study = sub_figure_type)

        # figure_case_study_strand_bias(combined_output_path,
        #                           plot_output_path,
        #                           cancer_types,
        #                           rows_sbs_signatures = rows_sbs_signatures,
        #                           rows_sbs_signatures_on_the_heatmap = rows_sbs_signatures_on_the_heatmap)


    if replication_time:

        # output figures are under figures_cosmic directory

        # POLE+
        # POLE proficient samples
        # non-POLE hypermutators
        # cancer_types = ['Lung-AdenoCA', 'Stomach-AdenoCA']  # discreet_mode
        # cancer_types = ['Lung-AdenoCA', 'Stomach-AdenoCA', 'Panc-AdenoCA', 'Eso-AdenoCA']  # prob_mode_05
        # sub_figure_type = 'POLE+'

        # POLE-
        # POLE deficient samples
        # POLE hypermutators
        cancer_types = ['ColoRect-AdenoCA', 'Uterus-AdenoCA', 'ESCC'] # same for discreet_mode and prob_mode_05
        sub_figure_type = 'POLE-'

        number_of_mutations_required_list = [AT_LEAST_1K_CONSRAINTS]
        cosmic_legend = False
        cosmic_signature = True
        cosmic_fontsize_text = 20
        cosmic_cancer_type_fontsize = 20/3
        cosmic_fontweight = 'semibold'
        cosmic_fontsize_labels = 10

        replication_time_significance_level = 0.05
        replication_time_slope_cutoff = 0.020
        replication_time_difference_between_min_and_max = 0.2
        replication_time_difference_between_medians = 0.135
        pearson_spearman_correlation_cutoff = 0.5

        plot_replication_time_legend(plot_output_path)
        plot_replication_time_legend(plot_output_path, num_of_cols = 2)

        plot_replication_time_legend_for_all_mutation_types(plot_output_path,
                                                            mutation_types=[SBS],
                                                            num_of_columns=2)

        plot_replication_time_legend_for_all_mutation_types(plot_output_path,
                                                            mutation_types=[SBS])

        generate_replication_time_pdfs(plot_output_path,
                                       combined_output_path,
                                       cancer_types,
                                       sbs_signatures,
                                       id_signatures,
                                       dbs_signatures,
                                       numberofSimulations,
                                       figure_types,
                                       number_of_mutations_required_list,
                                       cosmic_release_version,
                                       figure_file_extension,
                                       replication_time_significance_level,
                                       replication_time_slope_cutoff,
                                       replication_time_difference_between_min_and_max,
                                       replication_time_difference_between_medians,
                                       pearson_spearman_correlation_cutoff,
                                       cosmic_legend,
                                       cosmic_signature,
                                       cosmic_fontsize_text,
                                       cosmic_cancer_type_fontsize,
                                       cosmic_fontweight,
                                       cosmic_fontsize_labels,
                                       sub_figure_type)

    if strand_coordinated_mutagenesis:
        # output figure is under processivity
        # cancer types are defined in figure_case_study_strand_coordinated_mutagenesis
        figure_case_study_strand_coordinated_mutagenesis(plot_output_path,
                                                         combined_output_path,
                                                         figure_types,
                                                         cosmic_release_version,
                                                         figure_file_extension)