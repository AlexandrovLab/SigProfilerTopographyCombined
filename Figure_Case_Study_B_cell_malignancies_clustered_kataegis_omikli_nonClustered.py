import os

from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import COSMIC
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import MANUSCRIPT
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import NUCLEOSOME_OCCUPANCY
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import EPIGENOMICS_OCCUPANCY
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import NUCLEOSOME
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import CTCF
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import AT_LEAST_1K_CONSRAINTS
from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity import generate_occupancy_pdfs

from SigProfilerTopography.source.commons.TopographyCommons import AGGREGATEDSUBSTITUTIONS

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


def main():
    plot_mutational_context = False # Done
    replication_time = False # Done
    occupancy = True # done
    strand_bias = False # done
    strand_coordinated_mutagenesis = False # done
    epigenomics_heatmap = True

    # Common parameters
    figure_types = [COSMIC, MANUSCRIPT]
    combined_output_path = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'SigProfilerTopographyRuns', 'PCAWG_nonPCAWG_lymphomas')

    plot_output_path = os.path.join('/oasis', 'tscc', 'scratch', 'burcak',
                                    'SigProfilerTopographyRuns',
                                    'combined_pcawg_and_nonpcawg_figures_pdfs',
                                    '4th_iteration',
                                    'Figure_Case_Study_B_cell_malignancies')

    os.makedirs(plot_output_path, exist_ok=True)

    cancer_types = ['B_cell_malignancy_clustered']
    figure_case_study = 'B-cell malignancies Clustered'

    sbs_signatures = [AGGREGATEDSUBSTITUTIONS]
    dbs_signatures = []
    id_signatures = []

    cosmic_release_version = 'v3.2'
    figure_file_extension = 'jpg'
    numberofSimulations = 100

    if occupancy:
        dna_elements = [(CTCF, EPIGENOMICS_OCCUPANCY), (NUCLEOSOME, NUCLEOSOME_OCCUPANCY)]
        # dna_elements = [(NUCLEOSOME, NUCLEOSOME_OCCUPANCY)]
        # dna_elements = [(CTCF, EPIGENOMICS_OCCUPANCY)]

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


