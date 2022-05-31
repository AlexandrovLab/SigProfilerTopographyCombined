# !/usr/bin/env python3

# Author: burcakotlu

# Contact: burcakotlu@eng.ucsd.edu

"""
This python file generates the plots for Figure 7.

Figure 7. Topography of non-clustered, omikli, and kataegis substitutions
across 288 whole-genome sequenced B-cell malignancies.

Occupancy: Nucleosome Occupancy of Non-clustered, omikli and kataegis substitutions
Occupancy: CTCF Occupancy of Non-clustered, omikli and kataegis substitutions
Replication Timing of Non-clustered, omikli and kataegis substitutions
Enrichment and depletions at CTCF binding sites and histone modifications.

Figure S3 figures are gathered from individual SigProfilerTopography runs.
Strand Asymmetry for Non-clustered, omikli and kataegis substitutions
Transcription Strand Asymmetry
Replication Strand Asymmetry
Genic versus Intergenic Regions

"""

import os
import numpy as np
from matplotlib import  pyplot as plt

from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import COSMIC
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import MANUSCRIPT
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import NUCLEOSOME_OCCUPANCY
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import EPIGENOMICS_OCCUPANCY
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import CTCF
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import NUCLEOSOME
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import AT_LEAST_1K_CONSRAINTS
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import generate_occupancy_pdfs
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import generate_replication_time_pdfs

from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import fill_cancer_type_signature_cutoff_average_probability_df
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import step1_compute_p_value
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import step2_combine_p_values
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import step3_apply_multiple_tests_correction
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import TABLES
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import DATA_FILES
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import EXCEL_FILES
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import DICTIONARIES
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import fill_rows_and_columns_np_array
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import enriched
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import depleted
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import heatmap
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import ALL_SUBSTITUTIONS

from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import plot_strand_bias_figures
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import TRANSCRIPTIONSTRANDBIAS
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import REPLICATIONSTRANDBIAS
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import ODDS_RATIO

from Combined_Common import SBS
from Combined_Common import natural_key

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


def plot_heatmap(rows,
                columns,
                avg_fold_change_np_array,
                group_name,
                df1,
                df2,
                df3,
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

        if (figure_name) and (figure_name == 'Figure_Case_Study_B_Cell_Malignancies_epigenomics_heatmap'):
            title_font_size = title_font_size * 2.5
            label_font_size = font_size * 2.5
            cell_font_size = cell_font_size * 2.5

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

        # rows: ['non-clustered', 'kataegis', 'omikli']

        # Put text in each heatmap cell
        for row_index, row in enumerate(rows, 0):
            if row == rows[0]: # non-clustered
                df = df1
                row = ALL_SUBSTITUTIONS
            elif row == rows[1]: # kataegis
                df = df2
                row = ALL_SUBSTITUTIONS
            elif row== rows[2]: # omikli
                df = df3
                row = ALL_SUBSTITUTIONS

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



def figure_case_study_b_cell_malignancies_epigenomics_heatmap(combined_output_path,
        epigenomics_output_path,
        hm_path,
        ctcf_path,
        atac_path,
        plot_output_path,
        signatures, #  [ALL_SUBSTITUTIONS]
        signature_signature_type_tuples, # [(ALL_SUBSTITUTIONS, SBS), (ALL_SUBSTITUTIONS, SBS), (ALL_SUBSTITUTIONS, SBS) ]
        signatures_ylabels_on_the_heatmap, # [('non-clustered'), ('omikli'), ('kataegis') ]
        cancer_types,
        cancer_types_non_clustered,
        cancer_types_omikli,
        cancer_types_kataegis,
        normal_combined_pcawg_nonpcawg_cancer_type_2_biosample_dict,
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
        signature_cancer_type_number_of_mutations_for_ctcf):

    cancer_type_signature_cutoff_number_of_mutations_average_probability_df = fill_cancer_type_signature_cutoff_average_probability_df(cancer_types,
                                                                                                                                       combined_output_path)

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
                                                                                           epigenomics_output_path)

    # Step2 Combine p values using Fisher's method
    # Pool for biosamples and ENCDODE files
    # Step2: Filter Step1 rows such that fold_change is not (nan,None), p_value is not (nan,None), real_data_avg_count >= minimum_number_of_overlaps_required
    step2_combined_p_value_df, step2_signature2cancer_type2dna_element2combined_p_value_list_dict = step2_combine_p_values(
        step1_signature2cancer_type2biosample2dna_element2p_value_tuple_dict,
        epigenomics_output_path,
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
        epigenomics_output_path)

    # Plot Heatmap
    name_for_rows = 'signature'
    rows = signatures

    name_for_columns = 'dna_element'
    all_dna_elements = step3_q_value_df['dna_element'].unique()
    all_dna_elements = sorted(all_dna_elements, key=natural_key)

    # Remove open chromatin --- remove ATAC-seq
    all_dna_elements = [dna_element for dna_element in all_dna_elements if 'ATAC' not in dna_element]
    all_dna_elements = [dna_element for dna_element in all_dna_elements if 'Nucleosome' not in dna_element]
    all_dna_elements = [dna_element for dna_element in all_dna_elements if 'H3K79me2' not in dna_element]

    columns = all_dna_elements

    group_name = 'Substitutions' # 'B cell malignancies'

    # Consider only the cancer types in cancer_types
    non_clustered_step3_q_value_df = step3_q_value_df[step3_q_value_df['cancer_type'].isin(cancer_types_non_clustered)]
    omikli_step3_q_value_df = step3_q_value_df[step3_q_value_df['cancer_type'].isin(cancer_types_omikli)]
    kataegis_step3_q_value_df = step3_q_value_df[step3_q_value_df['cancer_type'].isin(cancer_types_kataegis)]

    non_clustered_avg_fold_change_np_array = fill_rows_and_columns_np_array(name_for_rows, rows, name_for_columns,
                                                                            columns, non_clustered_step3_q_value_df)
    omikli_avg_fold_change_np_array = fill_rows_and_columns_np_array(name_for_rows, rows, name_for_columns,
                                                                     columns, omikli_step3_q_value_df)
    kataegis_avg_fold_change_np_array = fill_rows_and_columns_np_array(name_for_rows, rows, name_for_columns,
                                                                       columns, kataegis_step3_q_value_df)

    bcell_malignancies_avg_fold_change_np_array = np.concatenate((non_clustered_avg_fold_change_np_array,
                                                                  omikli_avg_fold_change_np_array,
                                                                  kataegis_avg_fold_change_np_array), axis=0)


    plot_heatmap(signatures_ylabels_on_the_heatmap,
            columns,
            bcell_malignancies_avg_fold_change_np_array,
            group_name,
            non_clustered_step3_q_value_df,
            kataegis_step3_q_value_df,
            omikli_step3_q_value_df,
            name_for_rows,
            name_for_columns,
            significance_level,
            enriched_fold_change,
            depleted_fold_change,
            plot_output_path,
            'Figure_Case_Study_B_Cell_Malignancies_epigenomics_heatmap',
            plot_title=False)



def main():

    # Figure 7
    occupancy = False
    replication_time = False
    epigenomics_heatmap = False

    # Figure S3 plots are gathered from individual SigProfilerTopography runs.

    # Common parameters
    figure_types = [COSMIC, MANUSCRIPT]
    combined_output_path = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'SigProfilerTopographyRuns',
                                        'PCAWG_nonPCAWG_lymphomas')

    plot_output_path = os.path.join('/oasis', 'tscc', 'scratch', 'burcak',
                                    'SigProfilerTopographyRuns',
                                    'combined_pcawg_and_nonpcawg_figures_pdfs',
                                    '4th_iteration',
                                    'Figure_Case_Study_B_cell_malignancies')

    os.makedirs(plot_output_path, exist_ok=True)

    figure_case_study = 'B-cell malignancies'

    sbs_signatures = []
    dbs_signatures = []
    id_signatures = []

    cosmic_release_version = 'v3.2'
    figure_file_extension = 'jpg'
    numberofSimulations = 100

    normal_combined_pcawg_nonpcawg_cancer_type_2_biosample_dict ={ 'B_cell_malignancy_kataegis': ['B-cell'],
                                                                   'B_cell_malignancy_omikli': ['B-cell'],
                                                                   'B_cell_malignancy_nonClustered': ['B-cell']}

    if epigenomics_heatmap:
        # Resulting figure is under plot_output_path which is 
        # /oasis/tscc/scratch/burcak/SigProfilerTopographyRuns/combined_pcawg_and_nonpcawg_figures_pdfs/4th_iteration/Figure_Case_Study_B_cell_malignancies
        combine_p_values_method = 'fisher'
        window_size = 100
        depleted_fold_change = 0.95
        enriched_fold_change = 1.05
        significance_level = 0.05

        minimum_number_of_overlaps_required_for_sbs = 100
        minimum_number_of_overlaps_required_for_dbs = 100
        minimum_number_of_overlaps_required_for_indels = 100

        signature_cancer_type_number_of_mutations = AT_LEAST_1K_CONSRAINTS
        signature_cancer_type_number_of_mutations_for_ctcf = AT_LEAST_1K_CONSRAINTS

        hm_path = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'data', 'ENCODE', 'GRCh37', 'HM')
        ctcf_path = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'data', 'ENCODE', 'GRCh37', 'CTCF')
        atac_path = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'data', 'ENCODE', 'GRCh37', 'ATAC_seq')

        epigenomics_output_path = os.path.join(plot_output_path, 'epigenomics')

        os.makedirs(os.path.join(epigenomics_output_path, TABLES), exist_ok=True)
        os.makedirs(os.path.join(epigenomics_output_path, DATA_FILES), exist_ok=True)
        os.makedirs(os.path.join(epigenomics_output_path, EXCEL_FILES), exist_ok=True)
        os.makedirs(os.path.join(epigenomics_output_path, DICTIONARIES), exist_ok=True)

        signatures = [ALL_SUBSTITUTIONS]

        # Order must be consistent in these 4 lists below
        signature_signature_type_tuples = [(ALL_SUBSTITUTIONS, SBS), (ALL_SUBSTITUTIONS, SBS), (ALL_SUBSTITUTIONS, SBS) ]
        signatures_ylabels_on_the_heatmap = [('Non-clustered'), ('Omikli'), ('Kataegis')]
        cancer_types = [ 'B_cell_malignancy_nonClustered', 'B_cell_malignancy_omikli', 'B_cell_malignancy_kataegis']

        cancer_types_non_clustered = ['B_cell_malignancy_nonClustered']
        cancer_types_omikli = ['B_cell_malignancy_omikli']
        cancer_types_kataegis = ['B_cell_malignancy_kataegis']

        figure_case_study_b_cell_malignancies_epigenomics_heatmap(
                combined_output_path,
                epigenomics_output_path,
                hm_path,
                ctcf_path,
                atac_path,
                plot_output_path,
                signatures,
                signature_signature_type_tuples,
                signatures_ylabels_on_the_heatmap,
                cancer_types,
                cancer_types_non_clustered,
                cancer_types_omikli,
                cancer_types_kataegis,
                normal_combined_pcawg_nonpcawg_cancer_type_2_biosample_dict,
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
                signature_cancer_type_number_of_mutations_for_ctcf)

    if replication_time:
        # Run for each cancer type one by one
        # replication time figures are under figures_cosmic

        cancer_types = ['B_cell_malignancy_kataegis']
        sub_figure_type = 'Kataegis Mutations'

        # cancer_types = ['B_cell_malignancy_omikli']
        # sub_figure_type = 'Omikli Mutations'

        # cancer_types = ['B_cell_malignancy_nonClustered']
        # sub_figure_type = 'Non-clustered Mutations'

        number_of_mutations_required_list = [AT_LEAST_1K_CONSRAINTS]
        cosmic_legend = False
        cosmic_signature = True
        cosmic_fontsize_text = 20
        cosmic_cancer_type_fontsize = 20/3
        cosmic_fontweight = 'semibold'
        cosmic_fontsize_labels = 13

        replication_time_significance_level = 0.05
        replication_time_slope_cutoff = 0.020
        replication_time_difference_between_min_and_max = 0.2
        replication_time_difference_between_medians = 0.135
        pearson_spearman_correlation_cutoff = 0.5

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


    if occupancy:
        # Current way: Used
        # run for all cancer types
        # figures are under cosmic_tissue_based_figures
        cancer_types = ['B_cell_malignancy_kataegis', 'B_cell_malignancy_omikli', 'B_cell_malignancy_nonClustered']

        # Alternative Way: Not used
        # however, you can run for each cancer and the figures will be under figures_cosmic
        # cancer_types = ['B_cell_malignancy_kataegis']
        # cancer_types = ['B_cell_malignancy_omikli']
        # cancer_types = ['B_cell_malignancy_nonClustered']

        # dna_elements = [(NUCLEOSOME, NUCLEOSOME_OCCUPANCY)]
        # dna_elements = [(CTCF, EPIGENOMICS_OCCUPANCY)]
        dna_elements = [(NUCLEOSOME, NUCLEOSOME_OCCUPANCY), (CTCF, EPIGENOMICS_OCCUPANCY)]

        minimum_number_of_overlaps_required_for_sbs = 100
        minimum_number_of_overlaps_required_for_dbs = 100
        minimum_number_of_overlaps_required_for_indels = 100

        number_of_mutations_required_list = [AT_LEAST_1K_CONSRAINTS]
        number_of_mutations_required_list_for_ctcf = [AT_LEAST_1K_CONSRAINTS]
        pearson_spearman_correlation_cutoff = 0.5

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
