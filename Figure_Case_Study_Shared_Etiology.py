# !/usr/bin/env python3

# Author: burcakotlu

# Contact: burcakotlu@eng.ucsd.edu

import os
import numpy as np

from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import COSMIC
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import MANUSCRIPT

from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import SBS
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import DBS
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import ID

from Figure_Case_Study_SBS4_Across_All_Tissues import figure_case_study_occupancy
from Figure_Case_Study_SBS4_Across_All_Tissues import figure_case_study_replication_time
from Figure_Case_Study_SBS4_Across_All_Tissues import figure_case_study_strand_bias
from Figure_Case_Study_SBS4_Across_All_Tissues import plot_epigenomics_heatmap_color_bar
from Figure_Case_Study_SBS4_Across_All_Tissues import figure_case_study_strand_coordinated_mutagenesis

from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import NUCLEOSOME_OCCUPANCY
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import EPIGENOMICS_OCCUPANCY
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import NUCLEOSOME
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import CTCF
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import H3K27ac
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import AT_LEAST_1K_CONSRAINTS
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import AT_LEAST_20K_CONSRAINTS
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import FIGURE_CASE_STUDY_SHARED_ETIOLOGY

from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import fill_cancer_type_signature_cutoff_average_probability_df
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import DICTIONARIES
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import readDictionary
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import step1_compute_p_value
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import normal_combined_pcawg_nonpcawg_cancer_type_2_biosample_dict
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import step2_combine_p_values
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import step3_apply_multiple_tests_correction
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import heatmap_with_pie_chart
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import natural_key
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import fill_signature2dna_element2cancer_type_list_dict
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import fill_data_array
from Combined_PCAWG_nonPCAWG_Heatmaps_For_DNA_Elements import prepare_array_plot_heatmap

def figure_case_study_epigenomics_heatmap(
        combined_output_path,
        heatmaps_main_output_path,
        hm_path,
        ctcf_path,
        atac_path,
        plot_output_path,
        signatures,
        signature_signature_type_tuples,
        signature_tissue_type_tuples,
        signatures_ylabels_on_the_heatmap,
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
        minimum_number_of_overlaps_required_for_sbs,
        minimum_number_of_overlaps_required_for_dbs,
        minimum_number_of_overlaps_required_for_indels,
        signature_cancer_type_number_of_mutations,
        signature_cancer_type_number_of_mutations_for_ctcf)

    # Step3 Corrected combined p values
    # combined p value list
    # [fold_change_list,avg_fold_change,p_value_list,combined_p_value]
    step3_q_value_df, step3_signature2cancer_type2dna_element2q_value_list_dict = step3_apply_multiple_tests_correction(
        step2_signature2cancer_type2dna_element2combined_p_value_list_dict,
        heatmaps_main_output_path)

    all_dna_elements = step3_q_value_df['dna_element'].unique()
    all_dna_elements = sorted(all_dna_elements, key=natural_key)

    signature2dna_element2cancer_type_list_dict, \
    considered_dna_elements, \
    considered_signatures = fill_signature2dna_element2cancer_type_list_dict(all_dna_elements,
                                                                             signatures,
                                                                             step3_q_value_df,
                                                                             enriched_fold_change,
                                                                             depleted_fold_change,
                                                                             significance_level)

    # Fill data_array w.r.t. considered_dna_elements and considered_signatures
    # We have already given the order for signatures
    # No need to reorder them
    considered_dna_elements = sorted(considered_dna_elements, key=natural_key)

    data_array = fill_data_array(considered_signatures,
                                 considered_dna_elements,
                                 signature2dna_element2cancer_type_list_dict,
                                 signature_tissue_type_tuples = signature_tissue_type_tuples)

    heatmap_with_pie_chart(data_array,
                           signatures_ylabels_on_the_heatmap,
                           considered_dna_elements,
                           "Figure_Case_Study_Shared_Etiology",
                           plot_output_path,
                           figure_type,
                           True,
                           cosmic_release_version,
                           figure_file_extension,
                           signature_tissue_type_tuples = signature_tissue_type_tuples,
                           number_of_columns_in_legend = 1)

    # Plot Heatmap
    name_for_rows='signature'
    rows = signatures

    name_for_columns='dna_element'
    all_dna_elements = step3_q_value_df['dna_element'].unique()
    all_dna_elements = sorted(all_dna_elements, key=natural_key)
    columns = all_dna_elements

    group_name = ''

    prepare_array_plot_heatmap(step3_q_value_df,
                        name_for_rows,
                        rows,
                        name_for_columns,
                        columns,
                        enriched_fold_change,
                        depleted_fold_change,
                        significance_level,
                        plot_output_path,
                        group_name,
                        figure_name = 'Figure_Case_Study_Shared_Etiology',
                        remove_columns_rows_with_no_significant_result = False)


def main():
    occupancy = True
    replication_time = False
    strand_bias = False
    epigenomics_heatmap = False
    strand_coordinated_mutagenesis = False

    # Common parameters
    figure_types = [COSMIC]
    combined_output_path = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'SigProfilerTopographyRuns', 'Combined_PCAWG_nonPCAWG_4th_iteration')
    plot_output_path = os.path.join('/oasis','tscc','scratch','burcak',
                                   'SigProfilerTopographyRuns',
                                   'combined_pcawg_and_nonpcawg_figures_pdfs',
                                   '4th_iteration',
                                   'Figure_Shared_Etiology_Case_Study')

    os.makedirs(plot_output_path, exist_ok=True)

    # Shared Etiology APOBEC
    sbs_signatures = ['SBS2', 'SBS13']
    dbs_signatures = ['DBS11']
    id_signatures = []

    figure_file_extension = "png"
    cosmic_release_version = 'Figure_Case_Study_Shared_Etiology'
    pearson_spearman_correlation_cutoff = 0.5
    numberofSimulations = 100

    # These are the 40 tissues for combined PCAWG and nonPCAWG + ESCC
    cancer_types = ['ALL', 'Bladder-TCC', 'Bone-Benign', 'Bone-Osteosarc', 'CNS-GBM', 'CNS-Medullo', 'CNS-PiloAstro',
                    'ColoRect-AdenoCA', 'Ewings', 'Head-SCC', 'Kidney-RCC', 'Lung-AdenoCA', 'Lymph-BNHL', 'Myeloid-AML',
                    'Myeloid-MPN', 'Panc-AdenoCA', 'Prost-AdenoCA', 'SoftTissue-Leiomyo', 'Stomach-AdenoCA',
                    'Uterus-AdenoCA', 'Biliary-AdenoCA', 'Blood-CMDI', 'Bone-Epith', 'Breast-Cancer', 'CNS-LGG',
                    'CNS-Oligo', 'Cervix-Cancer', 'Eso-AdenoCA', 'ESCC', 'Eye-Melanoma', 'Kidney-ChRCC', 'Liver-HCC',
                    'Lung-SCC', 'Lymph-CLL', 'Myeloid-MDS', 'Ovary-AdenoCA', 'Panc-Endocrine', 'Skin-Melanoma',
                    'SoftTissue-Liposarc', 'Thy-AdenoCA']

    if occupancy:
        # dna_elements = [(NUCLEOSOME, NUCLEOSOME_OCCUPANCY)]
        # dna_elements = [(H3K27ac, EPIGENOMICS_OCCUPANCY)]
        dna_elements = [(CTCF, EPIGENOMICS_OCCUPANCY)]

        minimum_number_of_overlaps_required_for_sbs = 100  # 25 For CTCF ESCC in SBS4 Figure, for others 100
        minimum_number_of_overlaps_required_for_dbs = 25 # 25 for DBS11 CTCF in Figure Case Study Shared Etiology
        minimum_number_of_overlaps_required_for_indels = 100

        number_of_mutations_required_list = [AT_LEAST_1K_CONSRAINTS]
        number_of_mutations_required_list_for_ctcf = [AT_LEAST_1K_CONSRAINTS]  # For DBS11 CTCF in Figure Case Study Shared Etiology

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
                                FIGURE_CASE_STUDY_SHARED_ETIOLOGY)

    if replication_time:
        figure_case_study_replication_time(plot_output_path,
                                   combined_output_path,
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
        rows_sbs_signatures = [('SBS2', None),
                               ('SBS13', None)]
        rows_sbs_signatures_on_the_heatmap = ['SBS2 (n=17)',
                                              'SBS13 (n=16)']

        rows_dbs_id_signatures = [('DBS11', None)]
        rows_dbs_id_signatures_on_the_heatmap = ['DBS11 (n=3)']

        figure_case_study_strand_bias(combined_output_path,
                                  plot_output_path,
                                  cancer_types,
                                  rows_sbs_signatures = rows_sbs_signatures,
                                  rows_sbs_signatures_on_the_heatmap = rows_sbs_signatures_on_the_heatmap,
                                  rows_dbs_id_signatures = rows_dbs_id_signatures,
                                  rows_dbs_id_signatures_on_the_heatmap = rows_dbs_id_signatures_on_the_heatmap)

    if epigenomics_heatmap:
        combine_p_values_method = 'fisher'
        window_size = 100
        depleted_fold_change = 0.95
        enriched_fold_change = 1.05
        significance_level = 0.05

        minimum_number_of_overlaps_required_for_sbs = 100
        minimum_number_of_overlaps_required_for_dbs = 100
        minimum_number_of_overlaps_required_for_indels = 100

        signature_cancer_type_number_of_mutations = AT_LEAST_1K_CONSRAINTS
        signature_cancer_type_number_of_mutations_for_ctcf = AT_LEAST_20K_CONSRAINTS

        figure_type = MANUSCRIPT

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
        signatures = ['SBS2',
                      'SBS13',
                      'DBS11']

        signature_signature_type_tuples = [('SBS2', SBS),
                                           ('SBS13', SBS),
                                           ('DBS11', DBS)]

        # Tissue Type None means Across all tissues
        # Tissue Type Not None means tissue specific
        # or write the tissue 'Lung-AdenoCA' instead of None
        signature_tissue_type_tuples = [('SBS2', None),
                                        ('SBS13', None),
                                        ('DBS11', None)]

        signatures_ylabels_on_the_heatmap = [('SBS2'),
                                             ('SBS13'),
                                             ('DBS11')]

        figure_case_study_epigenomics_heatmap(combined_output_path,
            heatmaps_main_output_path,
            hm_path,
            ctcf_path,
            atac_path,
            plot_output_path,
            signatures,
            signature_signature_type_tuples,
            signature_tissue_type_tuples,
            signatures_ylabels_on_the_heatmap,
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
            step1_data_ready,
            figure_type,
            cosmic_release_version,
            figure_file_extension)


    if strand_coordinated_mutagenesis:
        rows_sbs_signatures = [('SBS2', None),
                               ('SBS13', None)]

        rows_signatures_on_the_heatmap = ['SBS2 (n=17)',
                                          'SBS13 (n=16)']

        figure_case_study_strand_coordinated_mutagenesis(plot_output_path,
                                                         combined_output_path,
                                                         cancer_types,
                                                         figure_types,
                                                         cosmic_release_version,
                                                         figure_file_extension,
                                                         rows_sbs_signatures,
                                                         rows_signatures_on_the_heatmap)