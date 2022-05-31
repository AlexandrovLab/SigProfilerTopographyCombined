# !/usr/bin/env python3

# Author: burcakotlu

# Contact: burcakotlu@eng.ucsd.edu

# Figure SBS4 Across All Tissues Case Study for the Topography of Mutational Processes im Human Cancer
import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.cm as matplotlib_cm

from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import COSMIC
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import MANUSCRIPT
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import AT_LEAST_1K_CONSRAINTS
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import AT_LEAST_20K_CONSRAINTS
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import FIGURE_CASE_STUDY

from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import NUCLEOSOME_OCCUPANCY
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import EPIGENOMICS_OCCUPANCY
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import NUCLEOSOME
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import CTCF
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import OCCUPANCY
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import REPLICATION_TIME
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import PROCESSIVITY

from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import generate_occupancy_pdfs
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import generate_replication_time_pdfs
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import generate_processivity_pdf
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import plot_processivity_colorbar_vertical
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import plot_processivity_colorbar_horizontal
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import plot_processivity_legend
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import FIGURE_CASE_STUDY_SBS4_ACROSS_ALL_TISSUES
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import FIGURE_CASE_STUDY_SBS28
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import FIGURE_CASE_STUDY_SHARED_ETIOLOGY

from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import TRANSCRIPTIONSTRANDBIAS
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import REPLICATIONSTRANDBIAS
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import calculate_radius
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import plot_colorbar

from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import plot_new_dbs_and_id_signatures_figures
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import calculate_radius_add_patch

from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import LAGGING
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import LEADING
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import GENIC
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import INTERGENIC
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import TRANSCRIBED_STRAND
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import UNTRANSCRIBED_STRAND

from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import TRANSCRIBED_VERSUS_UNTRANSCRIBED
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import GENIC_VERSUS_INTERGENIC
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import LAGGING_VERSUS_LEADING
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import six_mutation_types

from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import fill_strand_bias_dfs
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import fill_strand_bias_dictionaries
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import calculate_radius_color_add_patch

from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import transcrition_strand_bias_colours
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import genic_vs_intergenic_bias_colours
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import replication_strand_bias_colours
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import strand_bias_color_bins

from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import TABLES
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import EXCEL_FILES
from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import DATA_FILES

from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures import ODDS_RATIO

from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import DICTIONARIES
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import readDictionary
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import write_dictionary_as_dataframe_step1_p_value
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import step1_compute_p_value
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import step2_combine_p_values
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import step3_apply_multiple_tests_correction
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import fill_signature2dna_element2cancer_type_list_dict
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import fill_data_array
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import heatmap_with_pie_chart
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import natural_key
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import normal_combined_pcawg_nonpcawg_cancer_type_2_biosample_dict
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import fill_cancer_type_signature_cutoff_average_probability_df
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import SBS
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import DBS
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import ID
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import prepare_array_plot_heatmap

from Combined_Common import get_signature2cancer_type_list_dict

# Nucleosome Occupancy Figures
# CTCF Occupancy Figures
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
    cosmic_title_all_cancer_types = True

    for (dna_element, occupancy_type) in dna_elements:
        plot_occupancy_legend(plot_output_path)
        plot_occupancy_legend(plot_output_path, num_of_cols=2)

        plot_occupancy_legend_for_all_mutation_types(plot_output_path,
                                                     mutation_types = [SBS, DBS],
                                                     number_of_columns = 3)

        plot_occupancy_legend_for_all_mutation_types(plot_output_path,
                                                     mutation_types = [SBS, DBS])

        consider_both_real_and_sim_avg_overlap = True
        depleted_fold_change = 0.95
        enriched_fold_change = 1.05

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


def plot_replication_time_legend_for_all_mutation_types(plot_output_path,
                                                        mutation_types = [SBS, DBS, ID],
                                                        num_of_columns = 1):
    fig, ax = plt.subplots(figsize=(8, 1))

    real_subs_rectangle = mpatches.Patch(label='Real Substitutions', edgecolor='black', facecolor='royalblue', lw=3)
    real_dinucs_rectangle = mpatches.Patch(label='Real Dinucleotides', edgecolor='black', facecolor='crimson', lw=3)
    real_indels_rectangle = mpatches.Patch(label='Real Indels', edgecolor='black', facecolor='yellowgreen', lw=3)

    legend_elements = []
    for mutation_type in mutation_types:
        if mutation_type == SBS:
            legend_elements.append(real_subs_rectangle)
        elif mutation_type == DBS:
            legend_elements.append(real_dinucs_rectangle)
        elif mutation_type == ID:
            legend_elements.append(real_indels_rectangle)
    legend_elements.append(Line2D([0], [2], linestyle="--", marker='.', lw=5, color='black', label='Simulated Mutations', markerfacecolor='black', markersize=30))


    plt.legend(handles=legend_elements, handlelength=5, ncol = num_of_columns, loc="lower center", fontsize=30)  # bbox_to_anchor=(1, 0.5),
    plt.gca().set_axis_off()

    filename = 'Replication_Time_Legend_num_of_cols_%d.png' %(num_of_columns)
    filepath = os.path.join(plot_output_path, filename)
    print(filepath)
    fig.savefig(filepath, dpi=100, bbox_inches="tight")

    plt.cla()
    plt.close(fig)


def plot_replication_time_legend(plot_output_path,
                                 num_of_cols = 1):
    fig, ax = plt.subplots(figsize=(6, 2))

    real_subs_rectangle = mpatches.Patch(label='Real Mutations', edgecolor='black', facecolor='royalblue', lw=3)
    # real_subs_rectangle = mpatches.Patch(label='Real Substitutions', edgecolor='black', facecolor='royalblue', lw=3)
    # real_dinucs_rectangle = mpatches.Patch(label='Real Dinucleotides', edgecolor='black', facecolor='crimson', lw=3)
    # real_indels_rectangle = mpatches.Patch(label='Real Indels', edgecolor='black', facecolor='yellowgreen', lw=3)

    legend_elements = [
        real_subs_rectangle,
        # real_dinucs_rectangle,
        # real_indels_rectangle,
        Line2D([0], [2], linestyle="--", marker='.', lw=5, color='black', label='Simulated Mutations', markerfacecolor='black', markersize=30)]

    plt.legend(handles=legend_elements, handlelength=5, ncol = num_of_cols, loc="lower center", fontsize=30)  # bbox_to_anchor=(1, 0.5),
    plt.gca().set_axis_off()

    filename = 'Replication_Time_Legend_num_of_cols_%d.png' %(num_of_cols)
    filepath = os.path.join(plot_output_path, filename)
    print(filepath)
    fig.savefig(filepath, dpi=100, bbox_inches="tight")

    plt.cla()
    plt.close(fig)


def plot_occupancy_legend_for_all_mutation_types(plot_output_dir,
                                                 mutation_types = [SBS, DBS, ID],
                                                 number_of_columns = 1):
    fig, ax = plt.subplots(figsize=(10, 1))

    legend_elements= []
    for mutation_type in mutation_types:
        if mutation_type == SBS:
            legend_elements.append(Line2D([0], [2], linestyle="-", lw=5, color='royalblue', label = 'Real Substitutions', markerfacecolor='royalblue', markersize=30))
        elif mutation_type == DBS:
            legend_elements.append(Line2D([0], [2], linestyle="-", lw=5, color='crimson', label = 'Real Dinucleotides', markerfacecolor='crimson', markersize=30))
        elif mutation_type == ID:
            legend_elements.append(Line2D([0], [2], linestyle="-", lw=5, color='darkgreen', label = 'Real Indels', markerfacecolor='darkgreen', markersize=30))

    legend_elements.append(Line2D([0], [2], linestyle="--", lw=5, color='gray', label = 'Simulated Mutations', markerfacecolor='gray', markersize=30))


    plt.legend(handles=legend_elements, handlelength=5, ncol = number_of_columns, loc="center", bbox_to_anchor=(0.5, 0.5), fontsize=30)
    plt.gca().set_axis_off()

    filename = 'Occupancy_Legend_%d.png' %(number_of_columns)
    filepath = os.path.join(plot_output_dir, filename)
    fig.savefig(filepath, dpi=100, bbox_inches="tight")

    plt.cla()
    plt.close(fig)



def plot_occupancy_legend(plot_output_dir,
                          num_of_cols = 1):
    fig, ax = plt.subplots(figsize=(10, 1))

    legend_elements = [
        Line2D([0], [2], linestyle="-", lw=5, color='royalblue', label='Real Mutations', markerfacecolor='royalblue', markersize=30),
        Line2D([0], [2], linestyle="--", lw=5, color='gray', label='Simulated Mutations', markerfacecolor='gray', markersize=30)]

    plt.legend(handles=legend_elements, handlelength=5, ncol=num_of_cols, loc="center", bbox_to_anchor=(0.5, 0.5), fontsize=30)
    plt.gca().set_axis_off()

    filename = 'Occupancy_Legend_num_of_cols_%d.png' %(num_of_cols)
    filepath = os.path.join(plot_output_dir, filename)
    fig.savefig(filepath, dpi=100, bbox_inches="tight")

    plt.cla()
    plt.close(fig)

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

# Replication Timing Figures
def figure_case_study_replication_time(plot_output_path,
                                   combined_output_path,
                                   cancer_types,
                                   sbs_signatures,
                                   id_signatures,
                                   dbs_signatures,
                                   numberofSimulations,
                                   figure_types,
                                   cosmic_release_version,
                                   figure_file_extension,
                                   pearson_spearman_correlation_cutoff):

    number_of_mutations_required_list = [AT_LEAST_1K_CONSRAINTS]

    replication_time_significance_level = 0.05
    replication_time_slope_cutoff = 0.020
    replication_time_difference_between_min_and_max = 0.2
    replication_time_difference_between_medians = 0.135

    cosmic_legend = False
    cosmic_signature = True
    cosmic_fontsize_text = 20
    cosmic_cancer_type_fontsize = 20
    cosmic_fontweight = 'normal'
    cosmic_fontsize_labels = 10
    sub_figure_type = FIGURE_CASE_STUDY

    # # Figure Case Study SBS4
    # plot_replication_time_legend(plot_output_path)
    # plot_replication_time_legend(plot_output_path, num_of_cols = 2)

    # Figure Case Study Shared Etiology
    plot_replication_time_legend_for_all_mutation_types(plot_output_path,
                                                        mutation_types = [SBS, DBS],
                                                        num_of_columns = 3)

    plot_replication_time_legend_for_all_mutation_types(plot_output_path,
                                                        mutation_types = [SBS, DBS])

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


def calculate_radius_color_add_patch_tissue_based(cancer_type,
                                                  strand_bias,
                                                  cmap,
                                                  norm,
                                                  df,
                                                  signature2mutation_type2strand2percent2cancertypeslist_dict,
                                                  percentage_strings,
                                                  row_sbs_signature,
                                                  row_sbs_signature_index,
                                                  mutation_type,
                                                  mutation_type_index,
                                                  top_axis):

    if strand_bias == LAGGING_VERSUS_LEADING:
        strand1 = LAGGING
        strand2 = LEADING
        strands = [LAGGING, LEADING]

    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        strand1 = GENIC
        strand2 = INTERGENIC
        strands = [GENIC, INTERGENIC]

    elif strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        strand1 = TRANSCRIBED_STRAND
        strand2 = UNTRANSCRIBED_STRAND
        strands = [TRANSCRIBED_STRAND, UNTRANSCRIBED_STRAND]

    # Calculate color using cancer_types
    color_strand1 = None
    color_strand2 = None

    for strand in strands:
        strand_values = 0
        cancer_types_at_least_10_percent = 0

        for percentage_string in percentage_strings[::-1]:
            if df[(df['cancer_type'] == cancer_type) &
                (df['signature'] == row_sbs_signature) &
                (df['mutation_type'] == mutation_type) &
                (df['significant_strand'] == strand)][percentage_string].any():

                if percentage_string == "100%":
                    strand_value = 2
                elif percentage_string == "75%":
                    strand_value = 1.75
                elif percentage_string == "50%":
                    strand_value = 1.5
                elif percentage_string == "30%":
                    strand_value = 1.3
                elif percentage_string == "20%":
                    strand_value = 1.2
                elif percentage_string == "10%":
                    strand_value = 1.1

                strand_values += strand_value
                cancer_types_at_least_10_percent += 1
                break

        if cancer_types_at_least_10_percent > 0:
            color_value = strand_values/cancer_types_at_least_10_percent
            if strand == strand1:
                color_strand1 = color_value
            elif strand == strand2:
                color_strand2 = color_value

    if color_strand2:
        color_strand2 = 2 - color_strand2

    strand1_cancer_types_percentage = None
    strand2_cancer_types_percentage = None
    percentage_string = percentage_strings[0]

    if strand1 in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type]:
        if percentage_string in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type][strand1]:
            if cancer_type in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type][strand1][percentage_string]:
                strand1_cancer_types_percentage = 100
    if strand2 in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type]:
        if percentage_string in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type][strand2]:
            if cancer_type in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type][strand2][percentage_string]:
                strand2_cancer_types_percentage = 100
    if (strand1_cancer_types_percentage is not None) and (strand2_cancer_types_percentage is None):
        radius = calculate_radius(strand1_cancer_types_percentage)
        print('DEBUGXXX', strand_bias, cancer_type, mutation_type,
              'radius:', radius,
              'color_strand1', color_strand1,
              'norm(color_strand1)', norm(color_strand1),
              'strand1_cancer_types_percentage:', strand1_cancer_types_percentage)
        if (radius > 0):
            top_axis.add_patch(plt.Circle((mutation_type_index + 0.5, row_sbs_signature_index + 0.5), radius, color = cmap(norm(color_strand1)),  fill=True))
    elif (strand2_cancer_types_percentage is not None) and (strand1_cancer_types_percentage is None):
        radius = calculate_radius(strand2_cancer_types_percentage)
        if (radius > 0):
            top_axis.add_patch(plt.Circle((mutation_type_index + 0.5, row_sbs_signature_index + 0.5), radius, color = cmap(norm(color_strand2)), fill=True))
    elif (strand1_cancer_types_percentage is not None) and (strand2_cancer_types_percentage is not None):
        radius_strand1 = calculate_radius(strand1_cancer_types_percentage)
        radius_strand2 = calculate_radius(strand2_cancer_types_percentage)
        if (radius_strand1 > radius_strand2):
            # First strand1
            top_axis.add_patch(plt.Circle((mutation_type_index + 0.5, row_sbs_signature_index + 0.5), radius_strand1, color = cmap(norm(color_strand1)), fill=True))
            # Second strand2
            if radius_strand2 > 0:
                top_axis.add_patch(plt.Circle((mutation_type_index + 0.5, row_sbs_signature_index + 0.5), radius_strand2, color = cmap(norm(color_strand2)), fill=True))
        else:
            # First strand2
            top_axis.add_patch(plt.Circle((mutation_type_index + 0.5, row_sbs_signature_index + 0.5), radius_strand2, color = cmap(norm(color_strand2)), fill=True))
            # Second strand1
            if radius_strand1 > 0:
                top_axis.add_patch(plt.Circle((mutation_type_index + 0.5, row_sbs_signature_index + 0.5), radius_strand1, color = cmap(norm(color_strand1)), fill=True))



def get_processive_group_length_list(signature_list,
                                    all_cancer_types_processivity_df,
                                    across_all_cancer_types_pooled_processivity_df,
                                    minimum_required_number_of_processive_groups):

    signatures_tissue_specific = []
    signatures_across_all_cancer_types = []

    for signature_tuple in signature_list:
        signature, cancer_type = signature_tuple
        if cancer_type:
            if signature not in signatures_tissue_specific:
                signatures_tissue_specific.append(signature)
        else:
            if signature not in signatures_across_all_cancer_types:
                signatures_across_all_cancer_types.append(signature)

    df_across_all = across_all_cancer_types_pooled_processivity_df[across_all_cancer_types_pooled_processivity_df['signature'].isin(signatures_across_all_cancer_types)]
    df_tissue_specific = all_cancer_types_processivity_df[all_cancer_types_processivity_df['signature'].isin(signatures_tissue_specific)]

    processsive_group_length_list_across_all_tissues = df_across_all[
        (round(df_across_all['radius'], 2) > 0) &
        (df_across_all['avg_number_of_processive_groups'] >= minimum_required_number_of_processive_groups)]['processive_group_length'].unique()

    processsive_group_length_list_tissue_specific = df_tissue_specific[
        (round(df_tissue_specific['radius'], 2) > 0) &
        (df_tissue_specific['number_of_processive_groups'] >= minimum_required_number_of_processive_groups)]['processive_group_length'].unique()

    processsive_group_length_list = list(set(processsive_group_length_list_across_all_tissues).union(set(processsive_group_length_list_tissue_specific)))

    sorted_processsive_group_length_list = sorted(processsive_group_length_list, key=int)

    return  sorted_processsive_group_length_list





# Plot Figure
# Plot ColorBar
# Plot Legend
def ad_hoc_plot_processivity_figures(plot_output_path,
                                     all_cancer_types_processivity_df,
                                     across_all_cancer_types_pooled_processivity_df,
                                     signature_list,
                                     rows_signatures_on_the_heatmap,
                                     minimum_required_number_of_processive_groups):

    all_cancer_types_processivity_df = all_cancer_types_processivity_df.astype(dtype={'radius': float})
    all_cancer_types_processivity_df = all_cancer_types_processivity_df.astype(dtype={'number_of_processive_groups': int})
    all_cancer_types_processivity_df = all_cancer_types_processivity_df.astype(dtype={'processive_group_length': int})

    processive_group_length_list = get_processive_group_length_list(signature_list,
                                                                    all_cancer_types_processivity_df,
                                                                    across_all_cancer_types_pooled_processivity_df,
                                                                    minimum_required_number_of_processive_groups)

    max_processive_group_length = max(processive_group_length_list)

    index = None
    if ((len(processive_group_length_list)>0) and (max_processive_group_length>0)):
        # Find index of max_processive_group_length in processive_group_length_array
        index = processive_group_length_list.index(max_processive_group_length)

    print("signature_list: ", signature_list)
    print("processive_group_length_list: ", processive_group_length_list)
    print("max_processive_group_length: ", max_processive_group_length)
    print("index of max_processive_group_length: ", index)

    width_multiply = 1.5
    height_multiply = 1.5

    fig = plt.figure(figsize=(width_multiply * max_processive_group_length, height_multiply * len(signature_list) ))

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

    if (len(signature_list)>1):
        plt.ylim([1, len(signature_list)])
    else:
        plt.ylim([0, len(signature_list)])

    ax.set_yticks(np.arange(0, len(signature_list) + 1, 1))

    if (not all_cancer_types_processivity_df.empty) and (not across_all_cancer_types_pooled_processivity_df.empty):
        # Plot the circles with color
        for signature_index, signature_tuple in enumerate(signature_list):
            signature, cancer_type = signature_tuple
            if cancer_type:
                number_of_processive_groups_column_name = 'number_of_processive_groups'
                signature_processive_group_length_properties_df = all_cancer_types_processivity_df[all_cancer_types_processivity_df['cancer_type'] == cancer_type]
            else:
                number_of_processive_groups_column_name = 'avg_number_of_processive_groups'
                signature_processive_group_length_properties_df = across_all_cancer_types_pooled_processivity_df

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
        ax.set_xlabel('Strand-coordinated Mutagenesis Group Length',fontsize=fontsize, labelpad=20)
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
    filename = 'Figure_Case_Study_Processivity.png'
    figFile = os.path.join(figures_path, filename)
    fig.savefig(figFile,dpi=100, bbox_inches="tight")

    plt.cla()
    plt.close(fig)

def ad_hoc_plot_dbs_id_signatures_strand_bias_figures(strand_bias,
                                        cmap,
                                        norm,
                                        strand_bias_output_path,
                                        percentage_strings,
                                        signature2cancer_type_list_dict,
                                        significance_level,
                                        type2strand2percent2cancertypeslist_dict,
                                        type_transcribed_versus_untranscribed_filtered_q_value_df,
                                        type_genic_versus_intergenic_filtered_q_value_df,
                                        type_lagging_versus_leading_filtered_q_value_df,
                                        rows_dbs_id_signatures,
                                        rows_dbs_id_signatures_on_the_heatmap):

    rows_dbs_id_signatures = list(reversed(rows_dbs_id_signatures))
    rows_dbs_id_signatures_on_the_heatmap = list(reversed(rows_dbs_id_signatures_on_the_heatmap))

    if len(rows_dbs_id_signatures) <= 2:
        x_ticks_labelsize = 42
        y_ticks_labelsize = 52
    elif len(rows_dbs_id_signatures) <= 3:
        x_ticks_labelsize = 40
        y_ticks_labelsize = 50
    elif len(rows_dbs_id_signatures) <= 5:
        x_ticks_labelsize = 30
        y_ticks_labelsize = 40
    else:
        x_ticks_labelsize = 17
        y_ticks_labelsize = 27

    # Plot (width,height)
    fig, ax = plt.subplots(figsize=(5 + 1.75 * len(percentage_strings), 5 + len(rows_dbs_id_signatures))) # +5 is to avoid ValueError when there is no signature to show

    # Make aspect ratio square
    ax.set_aspect(1.0)

    for percentage_string_index, percentage_string in enumerate(percentage_strings):
        for row_signature_index, row_signature_tuple in enumerate(rows_dbs_id_signatures):
            row_signature, cancer_type = row_signature_tuple
            if row_signature in type2strand2percent2cancertypeslist_dict:
                if cancer_type:
                    calculate_radius_add_patch(strand_bias,
                                    cmap,
                                    norm,
                                     signature2cancer_type_list_dict,
                                     type2strand2percent2cancertypeslist_dict,
                                     row_signature,
                                     row_signature_index,
                                     percentage_string,
                                     percentage_string_index,
                                     ax,
                                    tissue_based = cancer_type)
                else:
                    calculate_radius_add_patch(strand_bias,
                                    cmap,
                                    norm,
                                     signature2cancer_type_list_dict,
                                     type2strand2percent2cancertypeslist_dict,
                                     row_signature,
                                     row_signature_index,
                                     percentage_string,
                                     percentage_string_index,
                                     ax)

    # CODE GOES HERE TO CENTER X-AXIS LABELS...
    ax.set_xlim([0,len(percentage_strings)])
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', length=0, labelsize=x_ticks_labelsize)

    #major ticks
    ax.set_xticks(np.arange(0, len(percentage_strings), 1))
    #minor ticks
    ax.set_xticks(np.arange(0, len(percentage_strings), 1)+0.5,minor=True)
    # xticklabels_list = ['1.1', '1.2', '1.3', '1.5', '1.75', '2+']

    xticklabels_list=[]
    for percentage_string in percentage_strings:
        if percentage_string=='5%':
            xticklabels_list.append('1.05')
        elif percentage_string=='10%':
            xticklabels_list.append('1.1')
        elif percentage_string=='20%':
            xticklabels_list.append('1.2')
        elif percentage_string=='25%':
            xticklabels_list.append('1.25')
        elif percentage_string=='30%':
            xticklabels_list.append('1.3')
        elif percentage_string=='50%':
            xticklabels_list.append('1.5')
        elif percentage_string=='75%':
            xticklabels_list.append('1.75')
        elif percentage_string=='100%':
            xticklabels_list.append('2+')

    ax.set_xticklabels(xticklabels_list, minor=True, fontweight='bold', fontname='Arial', fontsize=x_ticks_labelsize)

    ax.xaxis.set_ticks_position('top')

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False)  # labels along the bottom edge are off

    # CODE GOES HERE TO CENTER Y-AXIS LABELS...
    ax.set_ylim([0,len(rows_dbs_id_signatures)])
    ax.set_yticklabels([])
    ax.tick_params(axis='y', which='both', length=0, labelsize=y_ticks_labelsize)

    #major ticks
    ax.set_yticks(np.arange(0, len(rows_dbs_id_signatures), 1))
    #minor ticks
    ax.set_yticks(np.arange(0, len(rows_dbs_id_signatures), 1)+0.5,minor=True)

    yticks = np.arange(0,len(rows_dbs_id_signatures_on_the_heatmap))
    ax.set_yticks(yticks)
    ax.set_yticklabels(rows_dbs_id_signatures_on_the_heatmap, minor=True, fontsize=y_ticks_labelsize)  # fontsize

    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False)  # labels along the bottom edge are off

    # Gridlines based on major ticks
    ax.grid(which='major', color='black')

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('black')

    if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        feature_name = 'TRANSCR_ASYM'
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        feature_name = 'GENIC_ASYM'
    elif strand_bias == LAGGING_VERSUS_LEADING:
        feature_name = 'REPLIC_ASYM'

    filename = 'DBS_ID_Signatures_%s_with_circles_%s.png' % (strand_bias, str(significance_level).replace('.','_'))

    figFile = os.path.join(strand_bias_output_path, filename)
    fig.savefig(figFile, dpi=100, bbox_inches="tight")

    plt.cla()
    plt.close(fig)



def ad_hoc_plot_sbs_signatures_strand_bias_figures(strand_bias,
                                    cmap,
                                    norm,
                                    strand_bias_output_path,
                                    percentage_strings,
                                    signature2cancer_type_list_dict,
                                    significance_level,
                                    signature2mutation_type2strand2percent2cancertypeslist_dict,
                                    signature_transcribed_versus_untranscribed_filtered_q_value_df,
                                    signature_genic_versus_intergenic_filtered_q_value_df,
                                    signature_lagging_versus_leading_filtered_q_value_df,
                                    rows_sbs_signatures,
                                    rows_sbs_signatures_on_the_heatmap):

    mutation_types = six_mutation_types

    rows_sbs_signatures = list(reversed(rows_sbs_signatures))
    rows_signatures_on_the_heatmap = list(reversed(rows_sbs_signatures_on_the_heatmap))

    xticklabels_list = []
    for percentage_string in percentage_strings:
        if percentage_string == '5%':
            xticklabels_list.append('1.05')
        elif percentage_string == '10%':
            xticklabels_list.append('1.1')
        elif percentage_string == '20%':
            xticklabels_list.append('1.2')
        elif percentage_string == '25%':
            xticklabels_list.append('1.25')
        elif percentage_string == '30%':
            xticklabels_list.append('1.3')
        elif percentage_string == '50%':
            xticklabels_list.append('1.5')
        elif percentage_string == '75%':
            xticklabels_list.append('1.75')
        elif percentage_string == '100%':
            xticklabels_list.append('2+')

    # New plot (width, height)
    fig, top_axis = plt.subplots(figsize=(5 + 1.5 * len(xticklabels_list), 10 + 1.5 * len(rows_sbs_signatures)))
    plt.rc('axes', edgecolor='lightgray')
    # Make aspect ratio square
    top_axis.set_aspect(1.0)

    # Colors are from SigProfilerPlotting tool to be consistent
    colors = [[3 / 256, 189 / 256, 239 / 256],
              [1 / 256, 1 / 256, 1 / 256],
              [228 / 256, 41 / 256, 38 / 256],
              [203 / 256, 202 / 256, 202 / 256],
              [162 / 256, 207 / 256, 99 / 256],
              [236 / 256, 199 / 256, 197 / 256]]

    # Put rectangles
    x = 0
    x_mutation_type_text = 0.2  # For mutation types text

    # Write mutation types as text
    for i in range(0, len(mutation_types), 1):
        # mutation_type
        top_axis.text(x_mutation_type_text, # text x
                      len(rows_sbs_signatures) + 1.5, # text y
                      mutation_types[i],
                      fontsize=40,
                      fontweight='semibold',
                      fontname='Arial')

        # 1st rectangle below mutation_type
        top_axis.add_patch(plt.Rectangle((x + .0415, len(rows_sbs_signatures) + 0.75),
                                         (len(percentage_strings) - (2 * .0415))/6, # width
                                         .5, # height
                                         facecolor=colors[i],
                                         clip_on=False))

        # 2nd rectangle below mutation_type
        top_axis.add_patch(plt.Rectangle((x, 0), # rectangle lower left position
                                         len(percentage_strings)/6, # width
                                         len(rows_sbs_signatures), # height
                                         facecolor=colors[i],
                                         zorder=0,
                                         alpha=0.25,
                                         edgecolor='grey'))
        x += 1
        x_mutation_type_text +=1

    # CODE GOES HERE TO CENTER X-AXIS LABELS...
    # top_axis.set_xlim([0, len(mutation_types) * len(percentage_strings)])
    top_axis.set_xlim([0, len(percentage_strings)])
    top_axis.set_xticklabels([])

    top_axis.tick_params(axis='x', which='both', length=0, labelsize=35)

    # major ticks
    top_axis.set_xticks(np.arange(0, len(percentage_strings), 1))
    # minor ticks
    top_axis.set_xticks(np.arange(0, len(percentage_strings), 1) + 0.5, minor=True)

    # No need for x tick labels
    # top_axis.set_xticklabels(xticklabels_list, minor=True, fontweight='bold', fontname='Arial', fontsize=40)

    top_axis.xaxis.set_label_position('top')
    top_axis.xaxis.set_ticks_position('top')

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False)  # labels along the bottom edge are off

    # CODE GOES HERE TO CENTER Y-AXIS LABELS...
    top_axis.set_ylim([0, len(rows_sbs_signatures)])
    top_axis.set_yticklabels([])

    top_axis.tick_params(axis='y', which='both', length=0, labelsize=40)

    # major ticks
    top_axis.set_yticks(np.arange(0, len(rows_sbs_signatures), 1))
    # minor ticks
    top_axis.set_yticks(np.arange(0, len(rows_sbs_signatures), 1) + 0.5, minor=True)
    top_axis.set_yticklabels(rows_signatures_on_the_heatmap, minor=True)

    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False)  # labels along the bottom edge are off

    # Gridlines based on major ticks
    top_axis.grid(which='major', color='black', zorder=3)

    for mutation_type_index, mutation_type in enumerate(mutation_types):
        for row_sbs_signature_index, row_sbs_signature_detail_tuple in enumerate(rows_sbs_signatures):
            row_sbs_signature, cancer_type = row_sbs_signature_detail_tuple
            # Number of cancer types is decided based on first percentage string 1.1

            if (strand_bias == LAGGING_VERSUS_LEADING):
                if row_sbs_signature in signature2mutation_type2strand2percent2cancertypeslist_dict:
                    if mutation_type in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature]:
                        if cancer_type:
                            calculate_radius_color_add_patch_tissue_based(cancer_type,
                                        strand_bias,
                                        cmap,
                                        norm,
                                        signature_lagging_versus_leading_filtered_q_value_df,
                                        signature2mutation_type2strand2percent2cancertypeslist_dict,
                                        percentage_strings,
                                        row_sbs_signature,
                                        row_sbs_signature_index,
                                        mutation_type,
                                        mutation_type_index,
                                        top_axis)
                        else:
                            calculate_radius_color_add_patch(strand_bias,
                                        cmap,
                                        norm,
                                        signature_lagging_versus_leading_filtered_q_value_df,
                                        signature2cancer_type_list_dict,
                                        signature2mutation_type2strand2percent2cancertypeslist_dict,
                                        percentage_strings,
                                        row_sbs_signature,
                                        row_sbs_signature_index,
                                        mutation_type,
                                        mutation_type_index,
                                        top_axis)

            elif (strand_bias == GENIC_VERSUS_INTERGENIC):
                if row_sbs_signature in signature2mutation_type2strand2percent2cancertypeslist_dict:
                    if mutation_type in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature]:
                        if cancer_type:
                            calculate_radius_color_add_patch_tissue_based(cancer_type,
                                                                          strand_bias,
                                                                          cmap,
                                                                          norm,
                                                                          signature_genic_versus_intergenic_filtered_q_value_df,
                                                                          signature2mutation_type2strand2percent2cancertypeslist_dict,
                                                                          percentage_strings,
                                                                          row_sbs_signature,
                                                                          row_sbs_signature_index,
                                                                          mutation_type,
                                                                          mutation_type_index,
                                                                          top_axis)
                        else:
                            calculate_radius_color_add_patch(strand_bias,
                                        cmap,
                                        norm,
                                        signature_genic_versus_intergenic_filtered_q_value_df,
                                        signature2cancer_type_list_dict,
                                        signature2mutation_type2strand2percent2cancertypeslist_dict,
                                        percentage_strings,
                                        row_sbs_signature,
                                        row_sbs_signature_index,
                                        mutation_type,
                                        mutation_type_index,
                                        top_axis)

            elif (strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED):
                if row_sbs_signature in signature2mutation_type2strand2percent2cancertypeslist_dict:
                    if mutation_type in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature]:
                        if cancer_type:
                            calculate_radius_color_add_patch_tissue_based(cancer_type,
                                                                          strand_bias,
                                                                          cmap,
                                                                          norm,
                                                                          signature_transcribed_versus_untranscribed_filtered_q_value_df,
                                                                          signature2mutation_type2strand2percent2cancertypeslist_dict,
                                                                          percentage_strings,
                                                                          row_sbs_signature,
                                                                          row_sbs_signature_index,
                                                                          mutation_type,
                                                                          mutation_type_index,
                                                                          top_axis)
                        else:
                            calculate_radius_color_add_patch(strand_bias,
                                        cmap,
                                        norm,
                                        signature_transcribed_versus_untranscribed_filtered_q_value_df,
                                        signature2cancer_type_list_dict,
                                        signature2mutation_type2strand2percent2cancertypeslist_dict,
                                        percentage_strings,
                                        row_sbs_signature,
                                        row_sbs_signature_index,
                                        mutation_type,
                                        mutation_type_index,
                                        top_axis)

    # create the directory if it does not exists
    filename = 'SBS_Signatures_%s_with_circles_%s.png' % (strand_bias, str(significance_level).replace('.','_'))
    figFile = os.path.join(strand_bias_output_path, filename)
    fig.savefig(figFile, dpi=100, bbox_inches="tight")

    plt.cla()
    plt.close(fig)


# Transcroptional Strand Bias
# Genic versus Intergenic Regions
# Replicational Strand Bias
def figure_case_study_strand_bias(combined_output_path,
                                  plot_output_path,
                                  cancer_types,
                                  rows_sbs_signatures = None,
                                  rows_sbs_signatures_on_the_heatmap = None,
                                  rows_dbs_id_signatures = None,
                                  rows_dbs_id_signatures_on_the_heatmap = None):

    strand_biases = [TRANSCRIPTIONSTRANDBIAS, REPLICATIONSTRANDBIAS]
    percentage_numbers = [10, 20, 30, 50, 75, 100]
    percentage_strings = ['%d'%(percentage_number) + '%' for percentage_number in percentage_numbers]

    signature2cancer_type_list_dict = get_signature2cancer_type_list_dict(combined_output_path , cancer_types)

    significance_level = 0.05
    min_required_number_of_mutations_on_strands = 1000
    min_required_percentage_of_mutations_on_strands = 5
    strand_bias_output_path = os.path.join(plot_output_path,'strand_bias')

    os.makedirs(os.path.join(strand_bias_output_path), exist_ok=True)
    os.makedirs(os.path.join(strand_bias_output_path, TABLES), exist_ok=True)
    os.makedirs(os.path.join(strand_bias_output_path, EXCEL_FILES), exist_ok=True)
    os.makedirs(os.path.join(strand_bias_output_path, DATA_FILES), exist_ok=True)

    inflate_mutations_to_remove_TC_NER_effect = False
    consider_only_significant_results= False
    consider_also_DBS_ID_signatures = True
    fold_enrichment = ODDS_RATIO  # ODDS_RATIO = REAL_RATIO / SIMS_RATIO


    signature_transcribed_versus_untranscribed_df, \
    signature_transcribed_versus_untranscribed_filtered_q_value_df, \
    signature_genic_versus_intergenic_df, \
    signature_genic_versus_intergenic_filtered_q_value_df, \
    signature_lagging_versus_leading_df, \
    signature_lagging_versus_leading_filtered_q_value_df, \
    type_transcribed_versus_untranscribed_df, \
    type_transcribed_versus_untranscribed_filtered_q_value_df, \
    type_genic_versus_intergenic_df, \
    type_genic_versus_intergenic_filtered_q_value_df, \
    type_lagging_versus_leading_df, \
    type_lagging_versus_leading_filtered_q_value_df = fill_strand_bias_dfs(combined_output_path,
                        cancer_types,
                        strand_biases,
                        percentage_numbers,
                        percentage_strings,
                        significance_level,
                        min_required_number_of_mutations_on_strands,
                        min_required_percentage_of_mutations_on_strands,
                        strand_bias_output_path,
                        [],
                        [],
                        inflate_mutations_to_remove_TC_NER_effect,
                        consider_only_significant_results,
                        consider_also_DBS_ID_signatures,
                        fold_enrichment)

    # combined_output_dir,
    # cancer_types,
    # strand_biases,
    # percentage_numbers,
    # percentage_strings,
    # significance_level,
    # min_required_number_of_mutations_on_strands,
    # min_required_percentage_of_mutations_on_strands,
    # strand_bias_output_dir,
    # dbs_signatures,
    # id_signatures,
    # inflate_mutations_to_remove_TC_NER_effect,
    # consider_only_significant_results,
    # consider_also_DBS_ID_signatures,
    # fold_enrichment

    signature2mutation_type2strand2percent2cancertypeslist_dict, \
    type2strand2percent2cancertypeslist_dict = fill_strand_bias_dictionaries(signature_transcribed_versus_untranscribed_filtered_q_value_df,
                                    signature_genic_versus_intergenic_filtered_q_value_df,
                                    signature_lagging_versus_leading_filtered_q_value_df,
                                    type_transcribed_versus_untranscribed_filtered_q_value_df,
                                    type_genic_versus_intergenic_filtered_q_value_df,
                                    type_lagging_versus_leading_filtered_q_value_df,
                                    percentage_strings)

    strand_bias_list = [TRANSCRIBED_VERSUS_UNTRANSCRIBED, GENIC_VERSUS_INTERGENIC, LAGGING_VERSUS_LEADING]
    for strand_bias in strand_bias_list:
        # These colours are copied from Combined_PCAWG_nonPCAWG_Strand_Bias_Figures.py
        if strand_bias == LAGGING_VERSUS_LEADING:
            strands = [LAGGING, LEADING]
            colours = replication_strand_bias_colours
            cmap = mpl.colors.ListedColormap(colours)
            norm = mpl.colors.BoundaryNorm(boundaries=strand_bias_color_bins, ncolors=len(cmap.colors))

        elif strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
            strands = [TRANSCRIBED_STRAND, UNTRANSCRIBED_STRAND]
            colours = transcrition_strand_bias_colours
            cmap = mpl.colors.ListedColormap(colours)
            norm = mpl.colors.BoundaryNorm(boundaries=strand_bias_color_bins, ncolors=len(cmap.colors))

        elif strand_bias == GENIC_VERSUS_INTERGENIC:
            strands = [GENIC, INTERGENIC]
            colours = genic_vs_intergenic_bias_colours
            cmap = mpl.colors.ListedColormap(colours)
            norm = mpl.colors.BoundaryNorm(boundaries=strand_bias_color_bins, ncolors=len(cmap.colors))

        plot_colorbar(strand_bias_output_path,
                      strand_bias,
                      strands,
                      colours,
                      sub_dir = None)

        if rows_sbs_signatures and rows_sbs_signatures_on_the_heatmap:
            ad_hoc_plot_sbs_signatures_strand_bias_figures(strand_bias,
                                        cmap,
                                        norm,
                                        strand_bias_output_path,
                                        percentage_strings,
                                        signature2cancer_type_list_dict,
                                        significance_level,
                                        signature2mutation_type2strand2percent2cancertypeslist_dict,
                                        signature_transcribed_versus_untranscribed_filtered_q_value_df,
                                        signature_genic_versus_intergenic_filtered_q_value_df,
                                        signature_lagging_versus_leading_filtered_q_value_df,
                                        rows_sbs_signatures,
                                        rows_sbs_signatures_on_the_heatmap)

        # Can I use the same function?
        if rows_dbs_id_signatures and rows_dbs_id_signatures_on_the_heatmap:
            ad_hoc_plot_dbs_id_signatures_strand_bias_figures(strand_bias,
                                        cmap,
                                        norm,
                                        strand_bias_output_path,
                                        percentage_strings,
                                        signature2cancer_type_list_dict,
                                        significance_level,
                                        type2strand2percent2cancertypeslist_dict,
                                        type_transcribed_versus_untranscribed_filtered_q_value_df,
                                        type_genic_versus_intergenic_filtered_q_value_df,
                                        type_lagging_versus_leading_filtered_q_value_df,
                                        rows_dbs_id_signatures,
                                        rows_dbs_id_signatures_on_the_heatmap)



def figure_case_study_epigenomics_heatmap(
        combined_output_path,
        heatmaps_main_output_path,
        hm_path,
        ctcf_path,
        atac_path,
        plot_output_path,
        signatures,
        signature_signature_type_tuples,
        cancer_types,
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
        step1_data_ready):

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
        consider_both_real_and_sim_avg_overlap=True)

    # Step3 Corrected combined p values
    # combined p value list
    # [fold_change_list,avg_fold_change,p_value_list,combined_p_value]
    step3_q_value_df, step3_signature2cancer_type2dna_element2q_value_list_dict = step3_apply_multiple_tests_correction(
        step2_signature2cancer_type2dna_element2combined_p_value_list_dict,
        heatmaps_main_output_path)

    # Plot Heatmap
    name_for_group_by = 'signature'
    group_name = 'SBS4'

    name_for_rows = 'dna_element'
    all_dna_elements = step3_q_value_df['dna_element'].unique()
    all_dna_elements = sorted(all_dna_elements, key=natural_key)

    # Figure 6: SBS4 case study; panel D remove "open chromatin" and "nucleosome"
    if 'Nucleosome' in all_dna_elements:
        all_dna_elements.remove('Nucleosome')
    if 'ATAC-seq' in all_dna_elements:
        all_dna_elements.remove('ATAC-seq')
    rows = all_dna_elements

    name_for_columns = 'cancer_type'
    columns = cancer_types

    grouped_df = step3_q_value_df.groupby(name_for_group_by)
    df = grouped_df.get_group(group_name)

    prepare_array_plot_heatmap(df,
                        name_for_rows,
                        rows,
                        name_for_columns,
                        columns,
                        enriched_fold_change,
                        depleted_fold_change,
                        significance_level,
                        plot_output_path,
                        group_name,
                        figure_name = 'Figure_Case_Study_SBS4')



def figure_case_study_strand_coordinated_mutagenesis(plot_output_path,
                                                         combined_output_path,
                                                         cancer_types,
                                                         figure_types,
                                                         cosmic_release_version,
                                                         figure_file_extension,
                                                         rows_sbs_signatures,
                                                         rows_signatures_on_the_heatmap):
    processivity_significance_level = 0.05
    minimum_required_processive_group_length = 2
    minimum_required_number_of_processive_groups = 2

    rows_sbs_signatures = list(reversed(rows_sbs_signatures))
    rows_signatures_on_the_heatmap = list(reversed(rows_signatures_on_the_heatmap))

    number_of_simulations = 100

    all_cancer_types_processivity_df, \
    across_all_cancer_types_pooled_processivity_df = generate_processivity_pdf(plot_output_path,
                                                                               combined_output_path,
                                                                               cancer_types,
                                                                               number_of_simulations,
                                                                               figure_types,
                                                                               cosmic_release_version,
                                                                               figure_file_extension,
                                                                               processivity_significance_level,
                                                                               minimum_required_processive_group_length,
                                                                               minimum_required_number_of_processive_groups)

    ad_hoc_plot_processivity_figures(plot_output_path,
                                     all_cancer_types_processivity_df,
                                     across_all_cancer_types_pooled_processivity_df,
                                     rows_sbs_signatures,
                                     rows_signatures_on_the_heatmap,
                                     minimum_required_number_of_processive_groups)

# Sample
# combined_output_path = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'SigProfilerTopographyRuns', 'Combined_PCAWG_nonPCAWG_4th_iteration')
# plot_output_path = os.path.join('/oasis','tscc','scratch','burcak',
#                                    'SigProfilerTopographyRuns',
#                                    'combined_pcawg_and_nonpcawg_figures_pdfs',
#                                    '4th_iteration',
#                                    'Figure_SBS4_Case_Study')

def main(input_dir, output_dir):

    occupancy = False
    epigenomics_heatmap = False
    replication_time = False
    strand_bias = True
    strand_coordinated_mutagenesis = False

    # Common parameters
    figure_types = [COSMIC]

    plot_output_path = os.path.join(output_dir, 'Figure_SBS4_Case_Study')
    os.makedirs(plot_output_path, exist_ok=True)

    sbs_signatures = ['SBS4']
    dbs_signatures = []
    id_signatures = []

    # cancer_types = ['Lung-SCC']
    cancer_types = ['Lung-AdenoCA', 'Lung-SCC', 'Head-SCC', 'Liver-HCC', 'ESCC']

    figure_file_extension = "png"
    cosmic_release_version = 'Figure_Case_Study'
    pearson_spearman_correlation_cutoff = 0.5
    numberofSimulations = 100

    if occupancy:
        dna_elements = [(NUCLEOSOME, NUCLEOSOME_OCCUPANCY), (CTCF, EPIGENOMICS_OCCUPANCY)]
        # dna_elements = [(NUCLEOSOME, NUCLEOSOME_OCCUPANCY)]
        # dna_elements = [(CTCF, EPIGENOMICS_OCCUPANCY)]

        minimum_number_of_overlaps_required_for_sbs = 100
        minimum_number_of_overlaps_required_for_dbs = 100
        minimum_number_of_overlaps_required_for_indels = 100

        number_of_mutations_required_list = [AT_LEAST_1K_CONSRAINTS]
        number_of_mutations_required_list_for_ctcf = [AT_LEAST_1K_CONSRAINTS]

        figure_case_study_occupancy(input_dir,
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
                                FIGURE_CASE_STUDY_SBS4_ACROSS_ALL_TISSUES)


    if replication_time:
        figure_case_study_replication_time(plot_output_path,
                                   input_dir,
                                   cancer_types,
                                   sbs_signatures,
                                   id_signatures,
                                   dbs_signatures,
                                   numberofSimulations,
                                   figure_types,
                                   cosmic_release_version,
                                   figure_file_extension,
                                   pearson_spearman_correlation_cutoff)

    if strand_bias:
        rows_sbs_signatures = [('SBS4', None),
                               ('SBS4', 'Lung-AdenoCA'),
                               ('SBS4', 'Lung-SCC'),
                               ('SBS4', 'Head-SCC'),
                               ('SBS4', 'Liver-HCC'),
                               ('SBS4', 'ESCC')]

        rows_sbs_signatures_on_the_heatmap = ['SBS4 (n=5)',
                                          'SBS4 Lung-AdenoCA',
                                          'SBS4 Lung-SCC',
                                          'SBS4 Head-SCC',
                                          'SBS4 Liver-HCC',
                                          'SBS4 ESCC']

        figure_case_study_strand_bias(input_dir,
                                  plot_output_path,
                                  cancer_types,
                                  rows_sbs_signatures = rows_sbs_signatures,
                                  rows_sbs_signatures_on_the_heatmap = rows_sbs_signatures_on_the_heatmap)

    if epigenomics_heatmap:
        combine_p_values_method = 'fisher'
        window_size = 100
        depleted_fold_change = 0.95
        enriched_fold_change = 1.05
        significance_level = 0.05

        minimum_number_of_overlaps_required_for_sbs = 25 # default 100, 25 for CTCF SBS4 ESCC
        minimum_number_of_overlaps_required_for_dbs = 100
        minimum_number_of_overlaps_required_for_indels = 100

        signature_cancer_type_number_of_mutations = AT_LEAST_1K_CONSRAINTS
        signature_cancer_type_number_of_mutations_for_ctcf = AT_LEAST_1K_CONSRAINTS # default AT_LEAST_20K_CONSRAINTS, AT_LEAST_1K_CONSRAINTS for CTCF SBS4 ESCC

        step1_data_ready = True
        tissue_type= 'Normal'
        heatmaps_dir_name = "heatmaps_dna_elements_window_size_%s_%s" % (window_size, tissue_type)
        heatmaps_main_output_path = os.path.join('/oasis', 'tscc', 'scratch', 'burcak', 'SigProfilerTopographyRuns',
                                              'combined_pcawg_and_nonpcawg_figures_pdfs', '4th_iteration',
                                              heatmaps_dir_name)

        hm_path = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'data', 'ENCODE', 'GRCh37', 'HM')
        ctcf_path = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'data', 'ENCODE', 'GRCh37', 'CTCF')
        atac_path = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'data', 'ENCODE', 'GRCh37', 'ATAC_seq')

        # Order must be consistent in these 4 lists below
        signatures = ['SBS4']

        signature_signature_type_tuples = [('SBS4', SBS)]

        figure_case_study_epigenomics_heatmap(input_dir,
                                            heatmaps_main_output_path,
                                            hm_path,
                                            ctcf_path,
                                            atac_path,
                                            plot_output_path,
                                            signatures,
                                            signature_signature_type_tuples,
                                            cancer_types,
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
                                            step1_data_ready)

        plot_epigenomics_heatmap_color_bar(plot_output_path)

    if strand_coordinated_mutagenesis:
        rows_sbs_signatures = [('SBS4', None),
                               ('SBS4', 'Lung-AdenoCA'),
                               ('SBS4', 'Lung-SCC'),
                               ('SBS4', 'Head-SCC'),
                               ('SBS4', 'Liver-HCC'),
                               ('SBS4', 'ESCC')]

        rows_signatures_on_the_heatmap = ['SBS4 (n=5)',
                                          'SBS4 Lung-AdenoCA',
                                          'SBS4 Lung-SCC',
                                          'SBS4 Head-SCC',
                                          'SBS4 Liver-HCC',
                                          'SBS4 ESCC']

        figure_case_study_strand_coordinated_mutagenesis(plot_output_path,
                                                         input_dir,
                                                         cancer_types,
                                                         figure_types,
                                                         cosmic_release_version,
                                                         figure_file_extension,
                                                         rows_sbs_signatures,
                                                         rows_signatures_on_the_heatmap)