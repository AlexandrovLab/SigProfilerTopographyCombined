# Combined PCAWG nonPCAWG Strand Bias Bar plots and circle plot figures
# sftp the file to  /home/burcak/developer/python/SigProfilerTopographyAuxiliary/combined_pcawg_and_nonpcawg_figures_pdfs
# Run by Combined_PCAWG_nonPCAWG_Strand_Bias_Figures_Mediator.py
# output is under /oasis/tscc/scratch/burcak/SigProfilerTopographyRuns/combined_pcawg_and_nonpcawg_figures_pdfs/4th_iteration/

# CONSTRAINTS
# CONSTRAINT1 q_value <= SIGNIFICANCE_LEVEL
# CONSTRAINT2 number of mutations on strands >= MINIMUM_REQUIRED_NUMBER_OF_MUTATIONS_ON_STRANDS
# CONSTRAINT3 percentage of mutation on strands >= MINIMUM_REQUIRED_PERCENTAGE_OF_MUTATIONS_ON_STRANDS
# CONSTRAINT4 To show up in the figure there must be at least 10% difference between the strands

import os
import pandas as pd
import json
import numpy as np
import statsmodels.stats.multitest
import scipy.stats as stats
import re
import scipy
import multiprocessing

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib as mpl

from functools import reduce

from Combined_Common import cancer_type_2_NCI_Thesaurus_code_dict
from Combined_Common import signatures_attributed_to_artifacts
from Combined_Common import COSMIC_REPLICATION_STRAND_BIAS
from Combined_Common import COSMIC_TRANSCRIPTION_STRAND_BIAS
from Combined_Common import COSMIC_GENIC_VS_INTERGENIC_BIAS
from Combined_Common import deleteOldData
from Combined_Common import fill_lists
from Combined_Common import natural_key
from Combined_Common import get_signature2cancer_type_list_dict

from Combined_Common import ALTERNATIVE_OUTPUT_DIR
from Combined_Common import LYMPH_BNHL
from Combined_Common import LYMPH_BNHL_CLUSTERED
from Combined_Common import LYMPH_BNHL_NONCLUSTERED
from Combined_Common import LYMPH_CLL
from Combined_Common import LYMPH_CLL_CLUSTERED
from Combined_Common import LYMPH_BNHL_NONCLUSTERED

#turn it off with seterr
# np.seterr(divide = 'ignore')
#turn it on
np.seterr(divide = 'warn')

ONE_POINT_EIGHT = 1.08

DATA='data'
TRANSCRIPTIONSTRANDBIAS = 'transcription_strand_bias'
REPLICATIONSTRANDBIAS = 'replication_strand_bias'

# Six Mutation Types
C2A = 'C>A'
C2G = 'C>G'
C2T = 'C>T'
T2A = 'T>A'
T2C = 'T>C'
T2G = 'T>G'

six_mutation_types = [C2A, C2G, C2T, T2A, T2C, T2G]
PYRIMIDINE_C_Related_Mutation_Types = [C2A, C2G, C2T]
PYRIMIDINE_T_Related_Mutation_Types = [T2A, T2C, T2G]

PYRIMIDINE_C = 'PYRIMIDINE_C'
PYRIMIDINE_T = 'PYRIMIDINE_T'

TRANSCRIPTION_STRAND='transcription_strand'
REPLICATION_STRAND='replication_strand'

LEADING = 'Leading'
LAGGING = 'Lagging'

UNTRANSCRIBED_STRAND = 'Untranscribed' # 'UnTranscribed'
TRANSCRIBED_STRAND = 'Transcribed'
NONTRANSCRIBED_STRAND = 'Nontranscribed'  # 'NonTranscribed'

GENIC = 'Genic'
INTERGENIC = 'Intergenic'

transcription_strands = [TRANSCRIBED_STRAND, UNTRANSCRIBED_STRAND]
replication_strands = [LAGGING, LEADING]
genic_versus_intergenic_strands = [GENIC, INTERGENIC]

SBS = 'SBS'
DBS = 'DBS'
ID = 'ID'

COMBINED_PCAWG_NONPCAWG = 'combined_pcawg_nonpcawg'

P_VALUE = 'p_value'
Q_VALUE = 'q_value'
SIGNATURE='signature'
CANCER_TYPE='cancer_type'
MUTATION_TYPE='mutation_type'
TYPE = 'type'
SIGNIFICANT_STRAND='significant_strand'

# SIGNIFICANCE_LEVEL=0.01
# MINIMUM_REQUIRED_NUMBER_OF_MUTATIONS_ON_STRANDS = 1000
# MINIMUM_REQUIRED_PERCENTAGE_OF_MUTATIONS_ON_STRANDS = 5
# NUMBER_OF_REQUIRED_MUTATIONS_FOR_STACKED_BAR_PLOT = 1

ODDS_RATIO = 'Odds ratio' # ODDS_RATIO = REAL_RATIO / SIMS_RATIO
REAL_RATIO = 'Real ratio' # strand1_real / strand2_real --- order in a/b can be predefined or in the favor of higher one
SIMS_RATIO = 'Sims Ratio' # strand1_sims / strand2_sims --- order in a/b can be predefined or in the favor of higher one

TRANSCRIBED_REAL_COUNT = 'Transcribed_real_count'
UNTRANSCRIBED_REAL_COUNT = 'UnTranscribed_real_count'

TRANSCRIBED_SIMS_MEAN_COUNT = 'Transcribed_mean_sims_count'
UNTRANSCRIBED_SIMS_MEAN_COUNT = 'UnTranscribed_mean_sims_count'

NONTRANSCRIBED_REAL_COUNT = 'NonTranscribed_real_count'

LAGGING_REAL_COUNT = 'Lagging_real_count'
LEADING_REAL_COUNT = 'Leading_real_count'

LAGGING_SIMS_MEAN_COUNT = 'Lagging_mean_sims_count'
LEADING_SIMS_MEAN_COUNT = 'Leading_mean_sims_count'

GENIC_REAL_COUNT = 'genic_real_count'
INTERGENIC_REAL_COUNT = 'intergenic_real_count'

GENIC_SIMS_MEAN_COUNT = 'genic_mean_sims_count'
INTERGENIC_SIMS_MEAN_COUNT = 'intergenic_mean_sims_count'

FOLD_ENRICHMENT = 'Fold_Enrichment'
REAL_FOLD_ENRICHMENT = 'Real_Fold_Enrichment'
SIMS_FOLD_ENRICHMENT = 'Sims_Fold_Enrichment'

FC_REAL = 'FC_REAL'
FC_SIMS = 'FC_SIMS'
FC = 'FC'

TABLE_SBS_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE = "Table_SBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt"
TABLE_DBS_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE = "Table_DBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt"
TABLE_ID_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE  = "Table_ID_Signature_Cutoff_NumberofMutations_AverageProbability.txt"

SIGNATURE_MUTATION_TYPE_NAME1_VERSUS_NAME2 = 'SIGNATURE_MUTATION_TYPE_NAME1_VERSUS_NAME2'
TYPE_NAME1_VERSUS_NAME2 = 'TYPE_NAME1_VERSUS_NAME2'

LAGGING_VERSUS_LEADING='Lagging_Versus_Leading'
TRANSCRIBED_VERSUS_UNTRANSCRIBED='Transcribed_Versus_Untranscribed'
GENIC_VERSUS_INTERGENIC = 'Genic_Versus_Intergenic'

LAGGING_VERSUS_LEADING_P_VALUE='lagging_versus_leading_p_value'
TRANSCRIBED_VERSUS_UNTRANSCRIBED_P_VALUE='transcribed_versus_untranscribed_p_value'
GENIC_VERSUS_INTERGENIC_P_VALUE='genic_versus_intergenic_p_value'

LAGGING_VERSUS_LEADING_Q_VALUE='lagging_versus_leading_q_value'
TRANSCRIBED_VERSUS_UNTRANSCRIBED_Q_VALUE='transcribed_versus_untranscribed_q_value'
GENIC_VERSUS_INTERGENIC_Q_VALUE='genic_versus_intergenic_q_value'

TSCC = 'tscc'
AWS = 'aws'

EXCEL_FILES = 'excel_files'
TABLES = 'tables'
DATA_FILES = 'data_files'

COSMIC = 'Cosmic'
MANUSCRIPT = 'Manuscript'

FIGURES_COSMIC = 'figures_cosmic'
COSMIC_TISSUE_BASED_FIGURES = 'cosmic_tissue_based_figures'
FIGURES_MANUSCRIPT = 'figures_manuscript'

PCAWG = 'PCAWG'
nonPCAWG = 'nonPCAWG'
MUTOGRAPHS = 'MUTOGRAPHS'

# strand_bias_color_bins = [0, 0.25, 0.5, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.5, 1.75, 2] # 12 bins for 12 colors
strand_bias_color_bins = [0, 0.5, 0.8, 1, 1.2, 1.5, 2] # 6 bins for 6 colors

PROPORTION_OF_CANCER_TYPES_WITH_STRAND_ASYMMERTY_OF_A_SIGNATURE = 'Proportion of cancer types with strand asymmetry of a signature'
PROPORTION_OF_CANCER_TYPES_WITH_REGION_ASYMMERTY_OF_A_SIGNATURE = 'Proportion of cancer types with region asymmetry of a signature'

# 12 colors
# replication_strand_bias_colours = ['darkgoldenrod',
#                'goldenrod',
#                'darkorange',
#                'orange',
#                'gold',
#                'yellow',
#                'salmon',
#                'indianred', # cosmic
#                'brown',
#                'firebrick',
#                'maroon',
#                'darkred']

# reduced number of columns: 6 colors
replication_strand_bias_colours = [
    'goldenrod',
    'darkorange',
    'orange',
    'indianred', # cosmic
    'brown',
    'darkred'
]

# 12 colors
# transcrition_strand_bias_colours = ['darkgreen',
#                                     'green',
#                                     'darkolivegreen',
#                                     'olivedrab',
#                                     'yellowgreen',  # Cosmic
#                                     'greenyellow',
#                                     'cornflowerblue',
#                                     'royalblue',  # Cosmic
#                                     'blue',
#                                     'darkblue',
#                                     'navy',
#                                     'midnightblue'
#                                     ]

#reduced number of colors: 6 colors
transcrition_strand_bias_colours = [
                                    'darkolivegreen',
                                    'olivedrab',
                                    'yellowgreen',  # Cosmic
                                    'royalblue',  # Cosmic
                                    'blue',
                                    'darkblue'
                                    ]


# 12 colors
# genic_vs_intergenic_bias_colours = [
#                                     'dimgray',
#                                     'dimgrey',
#                                     'gray',  # Cosmic
#                                     'grey',
#                                     'darkgray',
#                                     'darkgrey',
#                                     'lightcyan',
#                                     'paleturquoise',
#                                     'aquamarine',
#                                     'turquoise',
#                                     'aqua',
#                                     'cyan' # Cosmic
#                                     ]

#reduced number of colors: 6 colors
genic_vs_intergenic_bias_colours = [
                                    'dimgray',
                                    'gray',  # Cosmic
                                    'darkgray',
                                    'aquamarine',
                                    'turquoise',
                                    'cyan' # Cosmic
                                    ]

cmap_replication_strand_bias = mpl.colors.LinearSegmentedColormap.from_list("golden2red",
                                                                            ["darkgoldenrod",
                                                                             "darkgoldenrod",
                                                                             "goldenrod",
                                                                             "goldenrod",  # Cosmic
                                                                             "darkorange",
                                                                             "orange",
                                                                             "gold",
                                                                             # "khaki",
                                                                             "white",
                                                                             # "mistyrose",
                                                                             # "lightcoral",
                                                                             # "coral",
                                                                            "salmon",
                                                                             "indianred",  # Cosmic
                                                                             "brown",
                                                                             "maroon",
                                                                             "maroon",
                                                                             "darkred",
                                                                             "darkred"])

cmap_transcription_strand_bias = mpl.colors.LinearSegmentedColormap.from_list('green2blue',
                                                                              ['darkgreen',
                                                                                # 'darkgreen',
                                                                                # 'green',
                                                                                # 'green',
                                                                                # 'forestgreen',
                                                                                'darkolivegreen',
                                                                                'darkolivegreen',
                                                                                'olivedrab',
                                                                                'yellowgreen',  # Cosmic
                                                                                'greenyellow',
                                                                                # 'palegreen',
                                                                                'white',
                                                                                # 'aliceblue',
                                                                                # 'lightsteelblue',
                                                                                # 'steelblue',
                                                                                'cornflowerblue',
                                                                                'royalblue',  # Cosmic
                                                                                'darkblue',
                                                                                'navy',
                                                                                'navy',
                                                                                'midnightblue',
                                                                                # 'midnightblue'
                                                                                ])



cmap_genic_versus_intergenic = mpl.colors.LinearSegmentedColormap.from_list("gray2cyan",
                                                                            ["dimgrey",
                                                                            "gray", # Cosmic
                                                                            "grey",
                                                                            # "silver",
                                                                            # "lightgray",
                                                                            "white",
                                                                            # 'lightcyan',
                                                                            "paleturquoise",
                                                                            "cyan",  # Cosmic
                                                                            "darkturquoise"
                                                                            # "teal",
                                                                            ])

def readDictionary(filePath):
    if (os.path.exists(filePath) and (os.path.getsize(filePath) > 0)):
        with open(filePath,'r') as json_data:
            dictionary = json.load(json_data)
        return dictionary
    else:
        # return None
        # Provide empty dictionary for not to fail for loops on None type dictionary
        return {}


def calculate_p_values(types_strand1_list,simulations_types_strand1_list,types_strand2_list,simulations_types_strand2_list):
    types_strandbias_pvalues = []

    #If there are no simulations case
    if ((simulations_types_strand1_list is None) and  (simulations_types_strand2_list is None)):
        simulations_types_strand1_list = [(x + y) / 2 for x, y in zip(types_strand1_list, types_strand2_list)]
        simulations_types_strand2_list = simulations_types_strand1_list

    for count1, count1_simulations, count2, count2_simulations in zip(types_strand1_list,
                                                                      simulations_types_strand1_list,
                                                                      types_strand2_list,
                                                                      simulations_types_strand2_list):
        #Is this true? Yes, it is correct.
        # we compare whether there is  a significance difference between the counts
        # namely, counts coming from the original data and the simulations
        contingency_table_array = [[count1, count1_simulations], [count2, count2_simulations]]

        if ((count1 < 3000000) and (count2 < 3000000) and (count1_simulations < 3000000) and (count2_simulations < 3000000)):
            oddsratio, pvalue_SBS = stats.fisher_exact(contingency_table_array)
        else:
            chi2, pvalue_SBS, dof, expected = stats.chi2_contingency(contingency_table_array)
        types_strandbias_pvalues.append(pvalue_SBS)

        print('For internal checking ---- count1:%d count2:%d count1_simulations:%d count2_simulations:%d, pvalue_SBS:%0.E' %(count1,count2,count1_simulations,count2_simulations,pvalue_SBS))

    return types_strandbias_pvalues


# Main Function
# Artifact signatures are removed only for manuscript figures
# No need to set as a parameter in main function
def main(figure_types = [MANUSCRIPT, COSMIC],
         significance_level = 0.05,
         cosmic_release_version = 'v3.2',
         figure_file_extension = 'jpg',
         min_required_number_of_mutations_on_strands = 1000,
         min_required_percentage_of_mutations_on_strands = 5,
         number_of_required_mutations_for_stacked_bar_plot = 1,
         inflate_mutations_to_remove_TC_NER_effect = True,
         consider_only_significant_results = True,
         consider_also_DBS_ID_signatures = True,
         fold_enrichment = ODDS_RATIO):

    # INPUT DIRECTORY
    # Real and simulations signals are here
    combined_output_dir = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'SigProfilerTopographyRuns','Combined_PCAWG_nonPCAWG_4th_iteration')

    # OUTPUT DIRECTORY under oasis
    strand_bias_output_dir = os.path.join('/oasis', 'tscc', 'scratch', 'burcak', 'SigProfilerTopographyRuns',
                                          'combined_pcawg_and_nonpcawg_figures_pdfs',
                                          '4th_iteration',
                                          'strand_bias')

    # CANCER TYPES
    # These are the 40 tissues for combined PCAWG and nonPCAWG + ESCC
    cancer_types=['ALL', 'Bladder-TCC', 'Bone-Benign', 'Bone-Osteosarc', 'CNS-GBM', 'CNS-Medullo', 'CNS-PiloAstro',
                  'ColoRect-AdenoCA', 'Ewings', 'Head-SCC', 'Kidney-RCC', 'Lung-AdenoCA', 'Lymph-BNHL', 'Lymph-CLL',
                  'Myeloid-AML', 'Myeloid-MPN', 'Panc-AdenoCA', 'Prost-AdenoCA', 'SoftTissue-Leiomyo', 'Stomach-AdenoCA',
                  'Uterus-AdenoCA', 'Biliary-AdenoCA', 'Blood-CMDI', 'Bone-Epith', 'Breast-Cancer', 'CNS-LGG',
                  'CNS-Oligo', 'Cervix-Cancer', 'Eso-AdenoCA', 'ESCC', 'Eye-Melanoma', 'Kidney-ChRCC', 'Liver-HCC',
                  'Lung-SCC', 'Myeloid-MDS', 'Ovary-AdenoCA', 'Panc-Endocrine', 'Skin-Melanoma',
                  'SoftTissue-Liposarc', 'Thy-AdenoCA']

    # For real run
    sbs_signatures = ['SBS1', 'SBS2', 'SBS3', 'SBS4', 'SBS5', 'SBS6', 'SBS7a', 'SBS7b', 'SBS7c', 'SBS7d', 'SBS8',
                      'SBS9', 'SBS10a', 'SBS10b', 'SBS10c', 'SBS11', 'SBS12', 'SBS13', 'SBS14', 'SBS15', 'SBS16',
                      'SBS17a', 'SBS17b', 'SBS18', 'SBS19', 'SBS20', 'SBS21', 'SBS22', 'SBS23', 'SBS24', 'SBS25',
                      'SBS26', 'SBS27', 'SBS28', 'SBS29', 'SBS30', 'SBS31', 'SBS32', 'SBS33', 'SBS34', 'SBS35',
                      'SBS36', 'SBS37', 'SBS38', 'SBS39', 'SBS40', 'SBS41', 'SBS42', 'SBS43', 'SBS44', 'SBS45',
                      'SBS46', 'SBS47', 'SBS48', 'SBS49', 'SBS50', 'SBS51', 'SBS52', 'SBS53', 'SBS54', 'SBS55',
                      'SBS56', 'SBS57', 'SBS58', 'SBS59', 'SBS60', 'SBS84', 'SBS85']

    dbs_signatures = ['DBS1', 'DBS2', 'DBS3', 'DBS4', 'DBS5', 'DBS6', 'DBS7', 'DBS8', 'DBS9', 'DBS10', 'DBS11']
    id_signatures = ['ID1', 'ID2', 'ID3', 'ID4', 'ID5', 'ID6', 'ID7', 'ID8', 'ID9', 'ID10', 'ID11', 'ID12', 'ID13',
                     'ID14', 'ID15', 'ID16', 'ID17']

    # # For testing and debugging
    # sbs_signatures = ['SBS1', 'SBS37'] # empty list is valid
    # dbs_signatures = [] # empty list is valid
    # id_signatures = [] # empty list is valid

    # Sort them
    cancer_types = sorted(cancer_types)
    sbs_signatures=sorted(sbs_signatures)
    dbs_signatures = sorted(dbs_signatures)
    id_signatures = sorted(id_signatures)

    print('len(sbs_signatures): %d \n sbs_signatures: %s' %(len(sbs_signatures),sbs_signatures))
    print('len(dbs_signatures): %d \n dbs_signatures: %s' %(len(dbs_signatures),dbs_signatures))
    print('len(id_signatures): %d \n id_signatures: %s' %(len(id_signatures),id_signatures))
    print('len(cancer_types): %d \n cancer_types: %s' %(len(cancer_types),cancer_types))

    strand_biases =[TRANSCRIPTIONSTRANDBIAS, REPLICATIONSTRANDBIAS]

    # Percentage numbers are parametric can be changed here
    percentage_numbers = [10, 20, 30, 50, 75, 100] # legacy
    # percentage_numbers = [5, 10, 20, 50, 75, 100]
    percentage_strings = ['%d'%(percentage_number) + '%' for percentage_number in percentage_numbers]

    signature2cancer_type_list_dict = get_signature2cancer_type_list_dict(combined_output_dir,cancer_types)
    print('signature2cancer_type_list_dict: ', signature2cancer_type_list_dict)

    cancer_type2source_cancer_type_tuples_dict = {
        'ALL' : [(nonPCAWG, 'ALL')],
        'Bladder-TCC':[(PCAWG, 'Bladder-TCC')],
        'Bone-Benign' : [(PCAWG, 'Bone-Benign')],
        'Bone-Osteosarc' : [(PCAWG, 'Bone-Osteosarc'), (nonPCAWG,'Sarcoma-bone'), (nonPCAWG,'Bone-cancer')],
        'CNS-GBM' : [(PCAWG, 'CNS-GBM'), (nonPCAWG,'CNS-GBM'), (nonPCAWG,'CNS-Glioma-NOS')],
        'CNS-Medullo' : [(PCAWG, 'CNS-Medullo'), (nonPCAWG, 'CNS-Medullo')],
        'CNS-PiloAstro' : [(PCAWG, 'CNS-PiloAstro')],
        'ColoRect-AdenoCA' : [(PCAWG, 'ColoRect-AdenoCA'), (nonPCAWG, 'ColoRect-AdenoCa')],
        'Ewings' : [(nonPCAWG, 'Ewings')],
        'Head-SCC' : [(PCAWG, 'Head-SCC')],
        'Kidney-RCC' : [(PCAWG, 'Kidney-RCC'), (nonPCAWG, 'Kidney-RCC')],
        'Lung-AdenoCA' : [(PCAWG, 'Lung-AdenoCA'), (nonPCAWG, 'Lung-AdenoCa')],
        'Lymph-BNHL' : [(PCAWG, 'Lymph-BNHL'), (nonPCAWG, 'Lymph-BNHL')],
        'Myeloid-AML' : [(PCAWG, 'Myeloid-AML'), (nonPCAWG, 'AML')],
        'Myeloid-MPN' : [(PCAWG, 'Myeloid-MPN')],
        'Panc-AdenoCA' : [(PCAWG, 'Panc-AdenoCA'), (nonPCAWG, 'Panc-AdenoCa')],
        'Prost-AdenoCA' : [(PCAWG, 'Prost-AdenoCA'), (nonPCAWG, 'Prost-AdenoCa')],
        'SoftTissue-Leiomyo' : [(PCAWG, 'SoftTissue-Leiomyo')],
        'Stomach-AdenoCA' : [(PCAWG, 'Stomach-AdenoCA'), (nonPCAWG, 'Stomach-AdenoCa')],
        'Uterus-AdenoCA' : [(PCAWG, 'Uterus-AdenoCA')],
        'Biliary-AdenoCA' :[(PCAWG, 'Biliary-AdenoCA'), (nonPCAWG, 'Biliary-AdenoCa')],
        'Blood-CMDI' : [(nonPCAWG, 'Blood-CMDI')],
        'Bone-Epith' : [(PCAWG, 'Bone-Epith')],
        'Breast-Cancer' : [(PCAWG, 'Breast-AdenoCA'), (PCAWG, 'Breast-DCIS'), (PCAWG, 'Breast-LobularCA'), (nonPCAWG, 'Breast-cancer')],
        'CNS-LGG' : [(nonPCAWG, 'CNS-LGG')],
        'CNS-Oligo' : [(PCAWG, 'CNS-Oligo')],
        'Cervix-Cancer' : [(PCAWG, 'Cervix-AdenoCA'), (PCAWG, 'Cervix-SCC')],
        'Eso-AdenoCA' : [(PCAWG, 'Eso-AdenoCA'), (nonPCAWG, 'Eso-AdenoCa')],
        'ESCC' : [(MUTOGRAPHS, 'ESCC')],
        'Eye-Melanoma' : [(nonPCAWG, 'Eye-Melanoma')],
        'Kidney-ChRCC' : [(PCAWG, 'Kidney-ChRCC')],
        'Liver-HCC' : [(PCAWG, 'Liver-HCC'), (nonPCAWG, 'Liver-HCC')],
        'Lung-SCC' : [(PCAWG, 'Lung-SCC')],
        'Lymph-CLL' : [(PCAWG, 'Lymph-CLL'), (nonPCAWG, 'Lymph-CLL')],
        'Myeloid-MDS' : [(PCAWG, 'Myeloid-MDS')],
        'Ovary-AdenoCA' : [(PCAWG, 'Ovary-AdenoCA'), (nonPCAWG, 'Ovary-AdenoCa')],
        'Panc-Endocrine' : [(PCAWG, 'Panc-Endocrine'), (nonPCAWG, 'Panc-Endocrine')],
        'Skin-Melanoma' : [(PCAWG, 'Skin-Melanoma'), (nonPCAWG, 'Skin-Melanoma')],
        'SoftTissue-Liposarc' : [(PCAWG, 'SoftTissue-Liposarc'), (nonPCAWG, 'Sarcoma')],
        'Thy-AdenoCA' : [(PCAWG, 'Thy-AdenoCA')]}

    plot_strand_bias_figures(combined_output_dir,
                             cancer_types,
                             sbs_signatures,
                             dbs_signatures,
                             id_signatures,
                             strand_bias_output_dir,
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
                             inflate_mutations_to_remove_TC_NER_effect,
                             consider_only_significant_results,
                             consider_also_DBS_ID_signatures,
                             fold_enrichment)


def write_my_type_dictionaries_as_dataframes(type2strand2percent2cancertypeslist_dict,signature2cancer_type_list_dict,percentage_strings,filepath):
    L = sorted([(my_type, strand,
                 b[percentage_strings[0]], len(b[percentage_strings[0]]),
                 b[percentage_strings[1]], len(b[percentage_strings[1]]),
                 b[percentage_strings[2]], len(b[percentage_strings[2]]),
                 b[percentage_strings[3]], len(b[percentage_strings[3]]),
                 b[percentage_strings[4]], len(b[percentage_strings[4]]),
                 b[percentage_strings[5]], len(b[percentage_strings[5]]),
                 signature2cancer_type_list_dict[my_type] if my_type in signature2cancer_type_list_dict else [],
                 len(signature2cancer_type_list_dict[my_type]) if my_type in signature2cancer_type_list_dict else 0)
                for my_type, a in type2strand2percent2cancertypeslist_dict.items()
                 for strand, b in a.items()])
    df = pd.DataFrame(L, columns=['my_type', 'strand', '10%', 'len(10%_cancer_types)', '20%', 'len(20%_cancer_types)', '30%', 'len(30%_cancer_types)', '50%', 'len(50%_cancer_types)', '75%', 'len(75%_cancer_types)', '100%', 'len(100%_cancer_types)',
                                  'all_cancer_types_list', 'len(all_cancer_types_list)' ])
    df.to_csv(filepath, sep='\t', header=True, index=False)
    return df

def write_signature_dictionaries_as_dataframes(signature2mutation_type2strand2percent2cancertypeslist,
                                               signature2cancer_type_list_dict,
                                               percentage_strings,
                                               filepath):

    L = sorted([(signature, mutation_type, strand,
                 c[percentage_strings[0]], len(c[percentage_strings[0]]),
                 c[percentage_strings[1]], len(c[percentage_strings[1]]),
                 c[percentage_strings[2]], len(c[percentage_strings[2]]),
                 c[percentage_strings[3]], len(c[percentage_strings[3]]),
                 c[percentage_strings[4]], len(c[percentage_strings[4]]),
                 c[percentage_strings[5]], len(c[percentage_strings[5]]), signature2cancer_type_list_dict[signature], len(signature2cancer_type_list_dict[signature]))
                for signature, a in signature2mutation_type2strand2percent2cancertypeslist.items()
                 for mutation_type, b in a.items()
                  for strand, c in b.items()])
    df = pd.DataFrame(L, columns=['signature', 'mutation_type', 'strand',
                            '10%', 'len(10%_cancer_types)',
                            '20%', 'len(20%_cancer_types)',
                            '30%', 'len(30%_cancer_types)',
                            '50%', 'len(50%_cancer_types)',
                            '75%', 'len(75%_cancer_types)',
                            '100%', 'len(100%_cancer_types)',
                            'all_cancer_types_list', 'len(all_cancer_types_list)'])
    df.to_csv(filepath, sep='\t', header=True, index=False)
    return df


# Sheet name must be less than 31 characters
def write_excel_file(df_list, sheet_list, file_name):
    writer = pd.ExcelWriter(file_name,engine='xlsxwriter')
    for dataframe, sheet in zip(df_list, sheet_list):
        dataframe.to_excel(writer, sheet_name=sheet, startrow=0 , startcol=0, index=False)
    writer.save()


def fill_strand_bias_dictionaries(signature_transcribed_versus_untranscribed_filtered_q_value_df,
                                signature_genic_versus_intergenic_filtered_q_value_df,
                                signature_lagging_versus_leading_filtered_q_value_df,
                                type_transcribed_versus_untranscribed_filtered_q_value_df,
                                type_genic_versus_intergenic_filtered_q_value_df,
                                type_lagging_versus_leading_filtered_q_value_df,
                                percentage_strings):

    # Step4 Fill this dictionary
    signature2mutation_type2strand2percent2cancertypeslist_dict = {}

    df_list = []
    if signature_transcribed_versus_untranscribed_filtered_q_value_df is not None:
        df_list.append(signature_transcribed_versus_untranscribed_filtered_q_value_df)
    if signature_genic_versus_intergenic_filtered_q_value_df is not None:
        df_list.append(signature_genic_versus_intergenic_filtered_q_value_df)
    if signature_lagging_versus_leading_filtered_q_value_df is not None:
        df_list.append(signature_lagging_versus_leading_filtered_q_value_df)

    for df in df_list:
        for index, row in df.iterrows():
            cancer_type = row[CANCER_TYPE]
            signature = row[SIGNATURE]
            mutation_type = row[MUTATION_TYPE]
            significant_strand = row[SIGNIFICANT_STRAND]

            if signature in signature2mutation_type2strand2percent2cancertypeslist_dict:
                if mutation_type in signature2mutation_type2strand2percent2cancertypeslist_dict[signature]:
                    if significant_strand in signature2mutation_type2strand2percent2cancertypeslist_dict[signature][mutation_type]:
                        for percentage_string in percentage_strings:
                            if (row[percentage_string] == 1):
                                signature2mutation_type2strand2percent2cancertypeslist_dict[signature][mutation_type][significant_strand][percentage_string].append(cancer_type)

                    else:
                        signature2mutation_type2strand2percent2cancertypeslist_dict[signature][mutation_type][significant_strand] = {}
                        for percentage_string in percentage_strings:
                            signature2mutation_type2strand2percent2cancertypeslist_dict[signature][mutation_type][significant_strand][percentage_string] = []

                        for percentage_string in percentage_strings:
                            if (row[percentage_string] == 1):
                                signature2mutation_type2strand2percent2cancertypeslist_dict[signature][mutation_type][
                                    significant_strand][percentage_string].append(cancer_type)

                else:
                    signature2mutation_type2strand2percent2cancertypeslist_dict[signature][mutation_type] = {}
                    signature2mutation_type2strand2percent2cancertypeslist_dict[signature][mutation_type][significant_strand] = {}
                    for percentage_string in percentage_strings:
                        signature2mutation_type2strand2percent2cancertypeslist_dict[signature][mutation_type][significant_strand][percentage_string] = []

                    for percentage_string in percentage_strings:
                        if (row[percentage_string] == 1):
                            signature2mutation_type2strand2percent2cancertypeslist_dict[signature][mutation_type][
                                significant_strand][percentage_string].append(cancer_type)

            else:
                signature2mutation_type2strand2percent2cancertypeslist_dict[signature] = {}
                signature2mutation_type2strand2percent2cancertypeslist_dict[signature][mutation_type] = {}
                signature2mutation_type2strand2percent2cancertypeslist_dict[signature][mutation_type][significant_strand] = {}
                for percentage_string in percentage_strings:
                    signature2mutation_type2strand2percent2cancertypeslist_dict[signature][mutation_type][significant_strand][percentage_string] = []

                for percentage_string in percentage_strings:
                    if row[percentage_string] == 1:
                        signature2mutation_type2strand2percent2cancertypeslist_dict[signature][mutation_type][
                            significant_strand][percentage_string].append(cancer_type)


    # Step4 Fill this dictionary
    type2strand2percent2cancertypeslist_dict={}

    df_list = []
    if type_transcribed_versus_untranscribed_filtered_q_value_df is not None:
        df_list.append(type_transcribed_versus_untranscribed_filtered_q_value_df)
    if type_genic_versus_intergenic_filtered_q_value_df is not None:
        df_list.append(type_genic_versus_intergenic_filtered_q_value_df)
    if type_lagging_versus_leading_filtered_q_value_df is not None:
        df_list.append(type_lagging_versus_leading_filtered_q_value_df)

    for df in df_list:
        for index, row in df.iterrows():
            cancer_type = row[CANCER_TYPE]
            my_type = row[TYPE]
            significant_strand=row[SIGNIFICANT_STRAND]

            if my_type in type2strand2percent2cancertypeslist_dict:
                if significant_strand in type2strand2percent2cancertypeslist_dict[my_type]:
                    for percentage_string in percentage_strings:
                        if row[percentage_string] == 1:
                            type2strand2percent2cancertypeslist_dict[my_type][significant_strand][
                                percentage_string].append(cancer_type)

                else:
                    type2strand2percent2cancertypeslist_dict[my_type][significant_strand]={}
                    for percentage_string in percentage_strings:
                        type2strand2percent2cancertypeslist_dict[my_type][significant_strand][percentage_string] = []

                    for percentage_string in percentage_strings:
                        if row[percentage_string] == 1:
                            type2strand2percent2cancertypeslist_dict[my_type][significant_strand][
                                percentage_string].append(cancer_type)

            else:
                type2strand2percent2cancertypeslist_dict[my_type] = {}
                type2strand2percent2cancertypeslist_dict[my_type][significant_strand] = {}
                for percentage_string in percentage_strings:
                    type2strand2percent2cancertypeslist_dict[my_type][significant_strand][percentage_string] = []

                for percentage_string in percentage_strings:
                    if row[percentage_string] == 1:
                        type2strand2percent2cancertypeslist_dict[my_type][significant_strand][
                            percentage_string].append(cancer_type)

    return signature2mutation_type2strand2percent2cancertypeslist_dict, \
           type2strand2percent2cancertypeslist_dict


def recalculate_p_values(signature_genic_versus_intergenic_df_list, type_genic_versus_intergenic_df_list):
    # recalculate p-values for inflated number of mutations on genic regions to see whether TC-NER is the only player
    # Update transcribed reals and untranscribed real value with the higher value one.
    # Update transcribed reals sims mean and untranscribed real sims mean value with the higher value one.

    # using
    # signature_genic_versus_intergenic_df_list
    # type_genic_versus_intergenic_df_list

    # Columns
    # genic_real_count
    # intergenic_real_count
    # genic_mean_sims_count
    # intergenic_mean_sims_count
    # genic_versus_intergenic_p_value
    # Transcribed_real_count
    # Transcribed_mean_sims_count
    # UnTranscribed_real_count
    # UnTranscribed_mean_sims_count
    # NonTranscribed_real_count
    # NonTranscribed_mean_sims_count

    for df_list in [signature_genic_versus_intergenic_df_list, type_genic_versus_intergenic_df_list]:
        for df in df_list:
            df.loc[df['Transcribed_real_count'] > df['UnTranscribed_real_count'], 'UnTranscribed_real_count'] = df[
                'Transcribed_real_count']
            df.loc[df['UnTranscribed_real_count'] > df['Transcribed_real_count'], 'Transcribed_real_count'] = df[
                'UnTranscribed_real_count']

            df.loc[
                df['Transcribed_mean_sims_count'] > df['UnTranscribed_mean_sims_count'], 'UnTranscribed_mean_sims_count'] = \
            df['Transcribed_mean_sims_count']
            df.loc[df['UnTranscribed_mean_sims_count'] > df['Transcribed_mean_sims_count'], 'Transcribed_mean_sims_count'] = \
            df['UnTranscribed_mean_sims_count']

            # update genic_real_count = transcribed_real_count +  untranscribed_real_count
            # update genic_sims_mean_count =  transcribed_sims_mean_count + untranscribed_sims_mean_count
            df['genic_real_count'] = df['Transcribed_real_count'] + df['UnTranscribed_real_count']
            df['genic_mean_sims_count'] = df['Transcribed_mean_sims_count'] + df['UnTranscribed_mean_sims_count']

            # update p-value
            df['genic_versus_intergenic_p_value'] = df.apply(
                lambda x: stats.fisher_exact([[x['genic_real_count'], x['genic_mean_sims_count']],
                                              [x['intergenic_real_count'], x['intergenic_mean_sims_count']]])[1], axis=1)


# For TC-NER analysis
def plot_histogram_of_fold_enrichment_in_intergenic_regions(strand_bias_output_dir,
                                                            inflate_mutations_to_remove_TC_NER_effect,
                                                            consider_only_significant_results,
                                                            consider_also_DBS_ID_signatures,
                                                            significance_level,
                                                            signature_genic_versus_intergenic_df,
                                                            type_genic_versus_intergenic_df):
    fwidth = 15
    fheight = 7

    fig = plt.figure(figsize=(fwidth, fheight), facecolor=None)
    plt.style.use('ggplot')
    ax = plt.gca()

    y = None

    # Step1
    if consider_only_significant_results:
        # remove not statistically significant ones
        sbs_df = signature_genic_versus_intergenic_df[signature_genic_versus_intergenic_df['genic_versus_intergenic_q_value'] <= significance_level]
        dbs_id_df = type_genic_versus_intergenic_df[type_genic_versus_intergenic_df['genic_versus_intergenic_q_value'] <= significance_level]
    else:
        sbs_df = signature_genic_versus_intergenic_df
        dbs_id_df = type_genic_versus_intergenic_df

    # Step2
    if consider_also_DBS_ID_signatures:
        dbs_id_df = dbs_id_df[ (dbs_id_df['type'].str.startswith('DBS')) | (dbs_id_df['type'].str.startswith('ID')) ]
        y = dbs_id_df[FOLD_ENRICHMENT].values
        y = y[~np.isnan(y)]

    # Step3
    x = sbs_df[FOLD_ENRICHMENT].values
    x = x[~np.isnan(x)]

    # Step4
    # combine x: SBS and y: DBS and ID results
    if consider_also_DBS_ID_signatures:
        z = np.append(x,y)
    else:
        z = x

    q25, q75 = np.percentile(z, [25, 75])
    bin_width = 2 * (q75 - q25) * len(z) ** (-1 / 3)
    bins = round((z.max() - z.min()) / bin_width)
    print('bin_width:',bin_width, "Freedman–Diaconis number of bins:", bins)
    plt.hist(z, bins=bins)

    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)

    ax.set_ylabel('Frequency', fontsize=40)
    ax.set_xlabel('Fold enrichment in intergenic regions', fontsize=40)
    if inflate_mutations_to_remove_TC_NER_effect:
        ax.set_title('Inflated Mutations on Genic Regions', fontsize=40)
        # ax.set_title('Corrected Number of Mutations on Genic Regions', fontsize=40)
    else:
        ax.set_title('No Inflated Mutations on Genic Regions',fontsize=40)
        # ax.set_title('No Correction on Genic Regions', fontsize=40)

    number_of_fold_enrichment = z.shape[0]
    mean = np.mean(z)
    std = np.std(z)
    info_text = 'number of fold enrichments: %d\nmean of fold enrichments: %.2f\n std of fold enrichments: %.2f' %(number_of_fold_enrichment, mean, std)
    plt.text(0.99, 0.96, info_text, verticalalignment='top', horizontalalignment='right', transform=ax.transAxes, fontsize=20)

    inflated_text = ''
    if inflate_mutations_to_remove_TC_NER_effect:
        inflated_text = 'inflated'

    if consider_also_DBS_ID_signatures:
        signatures_text = 'SBS_DBS_ID_Signatures'
    else:
        signatures_text = 'SBS_Signatures'

    significant_text = ''
    if consider_only_significant_results:
        significant_text = 'significant'

    filename = 'Fold_enrichment_in_intergenic_regions_%s_%s_%s.png' %(inflated_text, significant_text, signatures_text)
    figure_file = os.path.join(strand_bias_output_dir, filename)

    fig.savefig(figure_file, dpi=100, bbox_inches="tight")
    plt.close(fig)


# Fill element_names
# Fill p_values_list
def func_part1(df_list,
            file_path,
            element_names,
            p_values_list,
            analysis_type,
            name1_vs_name2,
            name1_vs_name2_p_value):

    df = None
    if len(df_list) > 0:
        df = pd.concat(df_list, ignore_index=True, axis=0)
        df.to_csv(file_path, sep='\t', header=True, index=False)

        for index, row in df.iterrows():
            if analysis_type == SIGNATURE_MUTATION_TYPE_NAME1_VERSUS_NAME2:
                element_name = (row[CANCER_TYPE], row[SIGNATURE], row[MUTATION_TYPE], name1_vs_name2)
            elif analysis_type == TYPE_NAME1_VERSUS_NAME2:
                element_name = (row[CANCER_TYPE], None, row[TYPE], name1_vs_name2)

            element_names.append(element_name)
            p_values_list.append(row[name1_vs_name2_p_value])

    return df

# Set q_value
# Add Significant Strand
# Set Significant Strands
# Set Percentages
# Set Fold Enrichment
def func_part2(df,
               name1_vs_name2_q_value,
               df_versus_type,
               element_names,
               all_FDR_BH_adjusted_p_values,
               percentage_strings,
               percentage_numbers,
               strand1_name,
               strand2_name,
               column1_real_name,
               column2_real_name,
               column1_sims_mean_name,
               column2_sims_mean_name,
               fold_enrichment,
               column3_real_name = None):

    if df is not None:
        df[name1_vs_name2_q_value] = np.nan

        # Set q_value
        for element_index, element_name in enumerate(element_names, 0):
            (cancer_type, signature, mutation_type, versus_type) = element_name
            q_value = all_FDR_BH_adjusted_p_values[element_index]
            print('%s %s %s %s -----  q_value: %f' % (cancer_type, signature, mutation_type, versus_type, q_value))

            if (signature is not None) and (versus_type == df_versus_type) and (SIGNATURE in df.columns.values):
                df.loc[(df[CANCER_TYPE] == cancer_type) &
                        (df[SIGNATURE] == signature) &
                        (df[MUTATION_TYPE] == mutation_type), name1_vs_name2_q_value] = q_value

            elif (signature is None) and (versus_type == df_versus_type) and (SIGNATURE not in df.columns.values):
                df.loc[(df[CANCER_TYPE] == cancer_type) &
                        (df[TYPE] == mutation_type), name1_vs_name2_q_value] = q_value

        if 'signature' in df.columns.values:
            # Add columns percentage_of_mutations_on_strands and signature_number_of_mutations_on_strands
            groupby_df = df.groupby(['cancer_type', 'signature'])

            for name, group_df in groupby_df:
                cancer_type, signature = name

                signature_number_of_mutations_on_strands = 0

                column1_real_count_value = group_df[column1_real_name].sum()
                signature_number_of_mutations_on_strands += column1_real_count_value

                column2_real_count_value = group_df[column2_real_name].sum()
                signature_number_of_mutations_on_strands += column2_real_count_value

                if column3_real_name:
                    column3_real_count_value = group_df[column3_real_name].sum()
                    signature_number_of_mutations_on_strands += column3_real_count_value

                df.loc[((df['cancer_type'] == cancer_type) & (df['signature'] == signature)),
                       'signature_number_of_mutations_on_strands'] = signature_number_of_mutations_on_strands

            if column3_real_name:
                df['percentage_of_mutations_on_strands'] = ((df[column1_real_name] + df[column2_real_name] + df[column3_real_name]) * 100) / \
                                                           df['signature_number_of_mutations_on_strands']
            else:
                df['percentage_of_mutations_on_strands'] = ((df[column1_real_name] + df[column2_real_name]) * 100) / \
                                                           df['signature_number_of_mutations_on_strands']

        # Add Significant Strand
        df[SIGNIFICANT_STRAND] = None
        for percentage_string in percentage_strings:
            df[percentage_string] = None

        # Set Significant Strands
        df.loc[(df[column1_real_name] > df[column2_real_name]), SIGNIFICANT_STRAND] = strand1_name
        df.loc[(df[column2_real_name] > df[column1_real_name]), SIGNIFICANT_STRAND] = strand2_name

        # For TC-NER analysis
        # Set Fold Enrichment
        # real_ratio = intergenic_real / genic_real # numerator/denominator order is preset
        # sim_ratio = intergenic_sim / genic_sim # numerator/denominator order is preset
        # odds_ratio = real_ratio / sim_ratio # numerator/denominator order is preset
        df.loc[ (df[column1_real_name] > 0) , REAL_FOLD_ENRICHMENT] = df[column2_real_name] / df[column1_real_name]
        df.loc[ (df[column1_sims_mean_name] > 0) , SIMS_FOLD_ENRICHMENT] = df[column2_sims_mean_name] / df[column1_sims_mean_name]
        df.loc[(df[SIMS_FOLD_ENRICHMENT] > 0), FOLD_ENRICHMENT] = df[REAL_FOLD_ENRICHMENT] / df[SIMS_FOLD_ENRICHMENT]

        if fold_enrichment == REAL_RATIO:
            # Set percentages based on real number of mutations
            for percentage_index, percentage_number in enumerate(percentage_numbers, 0):
                percentage_string = percentage_strings[percentage_index]
                # Case1: column1_real_name has more mutations. Set percentages for signature mutation_type
                df.loc[((df[column1_real_name] - df[column2_real_name]) > (df[column2_real_name] * percentage_number / 100)), percentage_string] = 1

                # Case2: column2_real_name has more mutations. Set percentages for signature mutation_type
                df.loc[((df[column2_real_name] - df[column1_real_name]) > (df[column1_real_name] * percentage_number / 100)), percentage_string] = 1

        elif fold_enrichment == ODDS_RATIO:
            # Set fold changes based on odds-ratio = real-ratio / sims-ratio
            # Case1: column1_real_name has the highest number of mutations
            df.loc[((df[column1_real_name] > df[column2_real_name]) & (df[column2_real_name] > 0)), FC_REAL] = df[column1_real_name] / df[column2_real_name]
            df.loc[((df[column1_real_name] > df[column2_real_name]) & (df[column2_sims_mean_name] > 0)), FC_SIMS] = df[column1_sims_mean_name] / df[column2_sims_mean_name]

            # Case2: column2_real_name has the highest number of mutations
            df.loc[((df[column2_real_name] > df[column1_real_name]) & (df[column1_real_name] > 0)), FC_REAL] = df[column2_real_name] / df[column1_real_name]
            df.loc[((df[column2_real_name] > df[column1_real_name]) & (df[column1_sims_mean_name] > 0)), FC_SIMS] = df[column2_sims_mean_name] / df[column1_sims_mean_name]

            # Same for both cases
            df.loc[ (df[FC_SIMS] > 0), FC] = df[FC_REAL] / df[FC_SIMS]

            for percentage_index, percentage_number in enumerate(percentage_numbers, 0):
                percentage_string = percentage_strings[percentage_index]
                # Set percentages for signature mutation_type
                fold_change = (percentage_number + 100)/100
                df.loc[ (df[FC] >= fold_change), percentage_string] = 1


# Apply constraints
# Constraint2 Minimum Required Number of Mutations on Strands
# Constraint3 Number of mutations on strands must be at least x% of the all mutations for that cancer_type, signature
def func_part3(df,
               name1_vs_name2_q_value,
               significance_level,
               min_required_number_of_mutations_on_strands,
               min_required_percentage_of_mutations_on_strands,
               column1_name,
               column2_name,
               column3_name = None):

    df_filtered_q_value_df = None

    if df is not None:
        df_filtered_q_value_df = df[df[name1_vs_name2_q_value] <= significance_level].copy()

        # Constraint2 Minimum Required Number of Mutations on Strands
        if column3_name:
            df_filtered_q_value_df = \
            df_filtered_q_value_df[
                ((df_filtered_q_value_df[column1_name] +
                  df_filtered_q_value_df[column2_name] +
                  df_filtered_q_value_df[column3_name]) >= min_required_number_of_mutations_on_strands)]
        else:
            df_filtered_q_value_df = \
            df_filtered_q_value_df[
                ((df_filtered_q_value_df[column1_name] +
                  df_filtered_q_value_df[column2_name]) >= min_required_number_of_mutations_on_strands)]

        # Constraint3 Number of mutations on strands must be at least x% of the all mutations for that cancer_type, signature
        # MINIMUM_REQUIRED_PERCENTAGE_OF_MUTATIONS_ON_STRANDS
        if ('percentage_of_mutations_on_strands' in df_filtered_q_value_df.columns.values):
            df_filtered_q_value_df = df_filtered_q_value_df[df_filtered_q_value_df['percentage_of_mutations_on_strands'] >=
                                                            min_required_percentage_of_mutations_on_strands]

    return df_filtered_q_value_df


def fill_strand_bias_dfs(combined_output_dir,
                        cancer_types,
                        strand_biases,
                        percentage_numbers,
                        percentage_strings,
                        significance_level,
                        min_required_number_of_mutations_on_strands,
                        min_required_percentage_of_mutations_on_strands,
                        strand_bias_output_dir,
                        dbs_signatures,
                        id_signatures,
                        inflate_mutations_to_remove_TC_NER_effect,
                        consider_only_significant_results,
                        consider_also_DBS_ID_signatures,
                        fold_enrichment):

    # Step1 Combine strand bias dataframes with already computed p_values coming from each project and cancer type
    signature_transcribed_versus_untranscribed_df_list = []
    signature_genic_versus_intergenic_df_list = []
    signature_lagging_versus_leading_df_list = []

    type_transcribed_versus_untranscribed_df_list = []
    type_genic_versus_intergenic_df_list = []
    type_lagging_versus_leading_df_list = []

    for cancer_type in cancer_types:
        for strand_bias in strand_biases:
            if (strand_bias == TRANSCRIPTIONSTRANDBIAS):
                signature_transcribed_versus_untranscribed_filename = 'Signature_Mutation_Type_Transcribed_Versus_Untranscribed_Strand_Table.txt'
                signature_genic_versus_intergenic_filename = 'Signature_Mutation_Type_Genic_Versus_Intergenic_Strand_Table.txt'
                type_transcribed_versus_untranscribed_filename = 'Type_Transcribed_Versus_Untranscribed_Strand_Table.txt'
                type_genic_versus_intergenic_filename = 'Type_Genic_Versus_Intergenic_Strand_Table.txt'

                if (cancer_type == LYMPH_BNHL) or (cancer_type == LYMPH_CLL):
                    # Case1 clustered
                    signature_transcribed_versus_untranscribed_p_value_df = pd.read_csv(os.path.join(os.path.join(ALTERNATIVE_OUTPUT_DIR, '%s_clustered' %(cancer_type), DATA, strand_bias, signature_transcribed_versus_untranscribed_filename)), sep='\t')
                    signature_genic_versus_untranscribed_p_value_df = pd.read_csv(os.path.join(os.path.join(ALTERNATIVE_OUTPUT_DIR, '%s_clustered' %(cancer_type), DATA, strand_bias, signature_genic_versus_intergenic_filename)), sep='\t')
                    type_transcribed_versus_untranscribed_p_value_df = pd.read_csv(os.path.join(os.path.join(ALTERNATIVE_OUTPUT_DIR, '%s_clustered' %(cancer_type), DATA, strand_bias, type_transcribed_versus_untranscribed_filename)), sep='\t')
                    type_genic_versus_untranscribed_p_value_df = pd.read_csv(os.path.join(os.path.join(ALTERNATIVE_OUTPUT_DIR, '%s_clustered' %(cancer_type), DATA, strand_bias, type_genic_versus_intergenic_filename)), sep='\t')
                    # consider only SBS37 or SBS84 or SBS85
                    signature_transcribed_versus_untranscribed_p_value_df = signature_transcribed_versus_untranscribed_p_value_df[signature_transcribed_versus_untranscribed_p_value_df['signature'].isin(['SBS37', 'SBS84', 'SBS85'])]
                    signature_genic_versus_untranscribed_p_value_df = signature_genic_versus_untranscribed_p_value_df[signature_genic_versus_untranscribed_p_value_df['signature'].isin(['SBS37', 'SBS84', 'SBS85'])]
                    type_transcribed_versus_untranscribed_p_value_df = type_transcribed_versus_untranscribed_p_value_df[type_transcribed_versus_untranscribed_p_value_df['type'].isin(['SBS37', 'SBS84', 'SBS85'])]
                    type_genic_versus_untranscribed_p_value_df = type_genic_versus_untranscribed_p_value_df[type_genic_versus_untranscribed_p_value_df['type'].isin(['SBS37', 'SBS84', 'SBS85'])]
                    # before append update cancer type
                    signature_transcribed_versus_untranscribed_p_value_df['cancer_type'] = cancer_type
                    signature_genic_versus_untranscribed_p_value_df['cancer_type'] = cancer_type
                    type_transcribed_versus_untranscribed_p_value_df['cancer_type'] = cancer_type
                    type_genic_versus_untranscribed_p_value_df['cancer_type'] = cancer_type
                    # append dataframes
                    signature_transcribed_versus_untranscribed_df_list.append(signature_transcribed_versus_untranscribed_p_value_df)
                    signature_genic_versus_intergenic_df_list.append(signature_genic_versus_untranscribed_p_value_df)
                    type_transcribed_versus_untranscribed_df_list.append(type_transcribed_versus_untranscribed_p_value_df)
                    type_genic_versus_intergenic_df_list.append(type_genic_versus_untranscribed_p_value_df)

                    # Case2 nonClustered
                    signature_transcribed_versus_untranscribed_p_value_df = pd.read_csv(os.path.join(os.path.join(ALTERNATIVE_OUTPUT_DIR, '%s_nonClustered' %(cancer_type), DATA, strand_bias, signature_transcribed_versus_untranscribed_filename)), sep='\t')
                    signature_genic_versus_untranscribed_p_value_df = pd.read_csv(os.path.join(os.path.join(ALTERNATIVE_OUTPUT_DIR, '%s_nonClustered' %(cancer_type), DATA, strand_bias, signature_genic_versus_intergenic_filename)), sep='\t')
                    type_transcribed_versus_untranscribed_p_value_df = pd.read_csv(os.path.join(os.path.join(ALTERNATIVE_OUTPUT_DIR, '%s_nonClustered' %(cancer_type), DATA, strand_bias, type_transcribed_versus_untranscribed_filename)), sep='\t')
                    type_genic_versus_untranscribed_p_value_df = pd.read_csv(os.path.join(os.path.join(ALTERNATIVE_OUTPUT_DIR, '%s_nonClustered' %(cancer_type), DATA, strand_bias, type_genic_versus_intergenic_filename)), sep='\t')
                    # filter out six_mutation_types
                    type_transcribed_versus_untranscribed_p_value_df = type_transcribed_versus_untranscribed_p_value_df[~type_transcribed_versus_untranscribed_p_value_df['type'].isin(six_mutation_types)]
                    type_genic_versus_untranscribed_p_value_df = type_genic_versus_untranscribed_p_value_df[~type_genic_versus_untranscribed_p_value_df['type'].isin(six_mutation_types)]
                    # before append update cancer type
                    signature_transcribed_versus_untranscribed_p_value_df['cancer_type'] = cancer_type
                    signature_genic_versus_untranscribed_p_value_df['cancer_type'] = cancer_type
                    type_transcribed_versus_untranscribed_p_value_df['cancer_type'] = cancer_type
                    type_genic_versus_untranscribed_p_value_df['cancer_type'] = cancer_type
                    # consider all signatures, append dataframes
                    signature_transcribed_versus_untranscribed_df_list.append(signature_transcribed_versus_untranscribed_p_value_df)
                    signature_genic_versus_intergenic_df_list.append(signature_genic_versus_untranscribed_p_value_df)
                    type_transcribed_versus_untranscribed_df_list.append(type_transcribed_versus_untranscribed_p_value_df)
                    type_genic_versus_intergenic_df_list.append(type_genic_versus_untranscribed_p_value_df)

                    # Case3 Lymph-BNHL and Lymph-CLL
                    type_transcribed_versus_untranscribed_p_value_df = pd.read_csv(os.path.join(os.path.join(combined_output_dir, cancer_type, DATA, strand_bias,type_transcribed_versus_untranscribed_filename)), sep='\t')
                    type_genic_versus_untranscribed_p_value_df = pd.read_csv(os.path.join(os.path.join(combined_output_dir, cancer_type, DATA, strand_bias,type_genic_versus_intergenic_filename)), sep='\t')
                    # consider only six mutation types, DBS and ID signatures
                    type_transcribed_versus_untranscribed_p_value_df = type_transcribed_versus_untranscribed_p_value_df[type_transcribed_versus_untranscribed_p_value_df['type'].isin(dbs_signatures + id_signatures + six_mutation_types)]
                    type_genic_versus_untranscribed_p_value_df = type_genic_versus_untranscribed_p_value_df[type_genic_versus_untranscribed_p_value_df['type'].isin(dbs_signatures + id_signatures + six_mutation_types)]
                    # append dataframes
                    type_transcribed_versus_untranscribed_df_list.append(type_transcribed_versus_untranscribed_p_value_df)
                    type_genic_versus_intergenic_df_list.append(type_genic_versus_untranscribed_p_value_df)

                else:
                    signature_transcribed_versus_untranscribed_p_value_df = pd.read_csv(os.path.join(os.path.join(combined_output_dir, cancer_type, DATA, strand_bias, signature_transcribed_versus_untranscribed_filename)), sep='\t')
                    signature_genic_versus_untranscribed_p_value_df = pd.read_csv(os.path.join(os.path.join(combined_output_dir, cancer_type, DATA, strand_bias, signature_genic_versus_intergenic_filename)), sep='\t')
                    type_transcribed_versus_untranscribed_p_value_df = pd.read_csv(os.path.join(os.path.join(combined_output_dir, cancer_type, DATA, strand_bias, type_transcribed_versus_untranscribed_filename)), sep='\t')
                    type_genic_versus_untranscribed_p_value_df = pd.read_csv(os.path.join(os.path.join(combined_output_dir, cancer_type, DATA, strand_bias, type_genic_versus_intergenic_filename)), sep='\t')

                    signature_transcribed_versus_untranscribed_df_list.append(signature_transcribed_versus_untranscribed_p_value_df)
                    signature_genic_versus_intergenic_df_list.append(signature_genic_versus_untranscribed_p_value_df)
                    type_transcribed_versus_untranscribed_df_list.append(type_transcribed_versus_untranscribed_p_value_df)
                    type_genic_versus_intergenic_df_list.append(type_genic_versus_untranscribed_p_value_df)

            elif (strand_bias == REPLICATIONSTRANDBIAS):
                signature_lagging_versus_leading_filename = 'Signature_Mutation_Type_Lagging_Versus_Leading_Strand_Table.txt'
                type_lagging_versus_leading_filename = 'Type_Lagging_Versus_Leading_Strand_Table.txt'

                if (cancer_type == LYMPH_BNHL) or (cancer_type == LYMPH_CLL):
                    # Case1 clustered
                    signature_lagging_versus_leading_p_value_df = pd.read_csv(os.path.join(os.path.join(ALTERNATIVE_OUTPUT_DIR, '%s_clustered' %(cancer_type), DATA, strand_bias, signature_lagging_versus_leading_filename)), sep='\t')
                    type_lagging_versus_leading_p_value_df = pd.read_csv(os.path.join(os.path.join(ALTERNATIVE_OUTPUT_DIR, '%s_clustered' %(cancer_type), DATA, strand_bias, type_lagging_versus_leading_filename)), sep='\t')
                    # consider only SBS37 or SBS84 or SBS85
                    signature_lagging_versus_leading_p_value_df = signature_lagging_versus_leading_p_value_df[signature_lagging_versus_leading_p_value_df['signature'].isin(['SBS37', 'SBS84', 'SBS85'])]
                    type_lagging_versus_leading_p_value_df = type_lagging_versus_leading_p_value_df[type_lagging_versus_leading_p_value_df['type'].isin(['SBS37', 'SBS84', 'SBS85'])]
                    # before append update cancer type
                    signature_lagging_versus_leading_p_value_df['cancer_type'] = cancer_type
                    type_lagging_versus_leading_p_value_df['cancer_type'] = cancer_type
                    # append dataframes
                    signature_lagging_versus_leading_df_list.append(signature_lagging_versus_leading_p_value_df)
                    type_lagging_versus_leading_df_list.append(type_lagging_versus_leading_p_value_df)

                    # nonClustered
                    signature_lagging_versus_leading_p_value_df = pd.read_csv(os.path.join(os.path.join(ALTERNATIVE_OUTPUT_DIR, '%s_nonClustered' %(cancer_type), DATA, strand_bias, signature_lagging_versus_leading_filename)), sep='\t')
                    type_lagging_versus_leading_p_value_df = pd.read_csv(os.path.join(os.path.join(ALTERNATIVE_OUTPUT_DIR, '%s_nonClustered' %(cancer_type), DATA, strand_bias, type_lagging_versus_leading_filename)), sep='\t')
                    # filter out six_mutation_types
                    type_lagging_versus_leading_p_value_df = type_lagging_versus_leading_p_value_df[~type_lagging_versus_leading_p_value_df['type'].isin(six_mutation_types)]
                    # before append update cancer type
                    signature_lagging_versus_leading_p_value_df['cancer_type'] = cancer_type
                    type_lagging_versus_leading_p_value_df['cancer_type'] = cancer_type
                    # consider all signatures, append dataframes
                    signature_lagging_versus_leading_df_list.append(signature_lagging_versus_leading_p_value_df)
                    type_lagging_versus_leading_df_list.append(type_lagging_versus_leading_p_value_df)

                    # Lymph-BNHL and Lymph-CLL
                    type_lagging_versus_leading_p_value_df = pd.read_csv(os.path.join(os.path.join(combined_output_dir, cancer_type, DATA, strand_bias, type_lagging_versus_leading_filename)), sep='\t')
                    # consider only six mutation types, DBS and ID signatures
                    type_lagging_versus_leading_p_value_df = type_lagging_versus_leading_p_value_df[type_lagging_versus_leading_p_value_df['type'].isin(dbs_signatures + id_signatures + six_mutation_types)]
                    # append dataframes
                    type_lagging_versus_leading_df_list.append(type_lagging_versus_leading_p_value_df)

                else:
                    signature_lagging_versus_leading_p_value_df = pd.read_csv(os.path.join(os.path.join(combined_output_dir, cancer_type, DATA, strand_bias, signature_lagging_versus_leading_filename)), sep='\t')
                    type_lagging_versus_leading_p_value_df = pd.read_csv(os.path.join(os.path.join(combined_output_dir, cancer_type, DATA, strand_bias, type_lagging_versus_leading_filename)), sep='\t')

                    signature_lagging_versus_leading_df_list.append(signature_lagging_versus_leading_p_value_df)
                    type_lagging_versus_leading_df_list.append(type_lagging_versus_leading_p_value_df)

    # Set filenames
    signature_transcribed_versus_untranscribed_filename = 'Signature_Mutation_Type_%s_P_Value_Table.txt' % (TRANSCRIBED_VERSUS_UNTRANSCRIBED)
    signature_genic_versus_intergenic_filename = 'Signature_Mutation_Type_%s_P_Value_Table.txt' % (GENIC_VERSUS_INTERGENIC)
    signature_lagging_versus_leading_filename = 'Signature_Mutation_Type_%s_P_Value_Table.txt' % (LAGGING_VERSUS_LEADING)

    type_transcribed_versus_untranscribed_filename = 'Type_%s_P_Value_Table.txt' % (TRANSCRIBED_VERSUS_UNTRANSCRIBED)
    type_genic_versus_intergenic_filename = 'Type_%s_P_Value_Table.txt' % (GENIC_VERSUS_INTERGENIC)
    type_lagging_versus_leading_filename = 'Type_%s_P_Value_Table.txt' % (LAGGING_VERSUS_LEADING)

    # Set filepaths
    signature_transcribed_versus_untranscribed_filepath = os.path.join(strand_bias_output_dir, TABLES, signature_transcribed_versus_untranscribed_filename)
    signature_genic_versus_intergenic_filepath = os.path.join(strand_bias_output_dir, TABLES, signature_genic_versus_intergenic_filename)
    signature_lagging_versus_leading_filepath = os.path.join(strand_bias_output_dir, TABLES, signature_lagging_versus_leading_filename)

    type_transcribed_versus_untranscribed_filepath = os.path.join(strand_bias_output_dir, TABLES, type_transcribed_versus_untranscribed_filename)
    type_genic_versus_intergenic_filepath = os.path.join(strand_bias_output_dir, TABLES, type_genic_versus_intergenic_filename)
    type_lagging_versus_leading_filepath = os.path.join(strand_bias_output_dir, TABLES, type_lagging_versus_leading_filename)

    # For TC-NER analysis
    # This extra analysis is added for the manuscript
    # It has to before func_part1
    # inflate the mutations and recalculate p values
    if inflate_mutations_to_remove_TC_NER_effect:
        recalculate_p_values(signature_genic_versus_intergenic_df_list, type_genic_versus_intergenic_df_list)

    # Step2 Compute q_value
    p_values_list = []
    element_names = []

    signature_transcribed_versus_untranscribed_df = func_part1(signature_transcribed_versus_untranscribed_df_list,
                                                           signature_transcribed_versus_untranscribed_filepath,
                                                           element_names,
                                                           p_values_list,
                                                           SIGNATURE_MUTATION_TYPE_NAME1_VERSUS_NAME2,
                                                           TRANSCRIBED_VERSUS_UNTRANSCRIBED,
                                                           TRANSCRIBED_VERSUS_UNTRANSCRIBED_P_VALUE)

    signature_genic_versus_intergenic_df = func_part1(signature_genic_versus_intergenic_df_list,
                                                signature_genic_versus_intergenic_filepath,
                                                element_names,
                                                p_values_list,
                                                SIGNATURE_MUTATION_TYPE_NAME1_VERSUS_NAME2,
                                                GENIC_VERSUS_INTERGENIC,
                                                GENIC_VERSUS_INTERGENIC_P_VALUE)

    signature_lagging_versus_leading_df = func_part1(signature_lagging_versus_leading_df_list,
                                                    signature_lagging_versus_leading_filepath,
                                                    element_names,
                                                    p_values_list,
                                                    SIGNATURE_MUTATION_TYPE_NAME1_VERSUS_NAME2,
                                                    LAGGING_VERSUS_LEADING,
                                                    LAGGING_VERSUS_LEADING_P_VALUE)

    type_transcribed_versus_untranscribed_df = func_part1(type_transcribed_versus_untranscribed_df_list,
                                                        type_transcribed_versus_untranscribed_filepath,
                                                        element_names,
                                                        p_values_list,
                                                        TYPE_NAME1_VERSUS_NAME2,
                                                        TRANSCRIBED_VERSUS_UNTRANSCRIBED,
                                                        TRANSCRIBED_VERSUS_UNTRANSCRIBED_P_VALUE)

    type_genic_versus_intergenic_df = func_part1(type_genic_versus_intergenic_df_list,
                                                type_genic_versus_intergenic_filepath,
                                                element_names,
                                                p_values_list,
                                                TYPE_NAME1_VERSUS_NAME2,
                                                GENIC_VERSUS_INTERGENIC,
                                                GENIC_VERSUS_INTERGENIC_P_VALUE)

    type_lagging_versus_leading_df = func_part1(type_lagging_versus_leading_df_list,
                                                type_lagging_versus_leading_filepath,
                                                element_names,
                                                p_values_list,
                                                TYPE_NAME1_VERSUS_NAME2,
                                                LAGGING_VERSUS_LEADING,
                                                LAGGING_VERSUS_LEADING_P_VALUE)

    print('len(p_values_list):', len(p_values_list))
    print('p_values_list:', p_values_list)

    print('len(element_names):', element_names)
    print('element_names:', element_names)

    if ((p_values_list is not None) and p_values_list and len(p_values_list) > 0):
        rejected, all_FDR_BH_adjusted_p_values, alphacSidak, alphacBonf = statsmodels.stats.multitest.multipletests(p_values_list, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
        print('(len(all_FDR_BH_adjusted_p_values): %d' % (len(all_FDR_BH_adjusted_p_values)))

        func_part2(signature_transcribed_versus_untranscribed_df,
                   TRANSCRIBED_VERSUS_UNTRANSCRIBED_Q_VALUE,
                   TRANSCRIBED_VERSUS_UNTRANSCRIBED,
                   element_names,
                   all_FDR_BH_adjusted_p_values,
                   percentage_strings,
                   percentage_numbers,
                   TRANSCRIBED_STRAND, # strand1
                   UNTRANSCRIBED_STRAND, # strand2
                   TRANSCRIBED_REAL_COUNT,
                   UNTRANSCRIBED_REAL_COUNT,
                   TRANSCRIBED_SIMS_MEAN_COUNT,
                   UNTRANSCRIBED_SIMS_MEAN_COUNT,
                   fold_enrichment,
                   column3_real_name = NONTRANSCRIBED_REAL_COUNT)

        func_part2(signature_genic_versus_intergenic_df,
                   GENIC_VERSUS_INTERGENIC_Q_VALUE,
                   GENIC_VERSUS_INTERGENIC,
                   element_names,
                   all_FDR_BH_adjusted_p_values,
                   percentage_strings,
                   percentage_numbers,
                   GENIC, # strand1
                   INTERGENIC, # strand2
                   GENIC_REAL_COUNT,
                   INTERGENIC_REAL_COUNT,
                   GENIC_SIMS_MEAN_COUNT,
                   INTERGENIC_SIMS_MEAN_COUNT,
                   fold_enrichment)

        func_part2(signature_lagging_versus_leading_df,
                   LAGGING_VERSUS_LEADING_Q_VALUE,
                   LAGGING_VERSUS_LEADING,
                   element_names,
                   all_FDR_BH_adjusted_p_values,
                   percentage_strings,
                   percentage_numbers,
                   LAGGING, # strand1
                   LEADING, # strand2
                   LAGGING_REAL_COUNT,
                   LEADING_REAL_COUNT,
                   LAGGING_SIMS_MEAN_COUNT,
                   LEADING_SIMS_MEAN_COUNT,
                   fold_enrichment)

        func_part2(type_transcribed_versus_untranscribed_df,
                   TRANSCRIBED_VERSUS_UNTRANSCRIBED_Q_VALUE,
                   TRANSCRIBED_VERSUS_UNTRANSCRIBED,
                   element_names,
                   all_FDR_BH_adjusted_p_values,
                   percentage_strings,
                   percentage_numbers,
                   TRANSCRIBED_STRAND, # strand1
                   UNTRANSCRIBED_STRAND, # strand2
                   TRANSCRIBED_REAL_COUNT,
                   UNTRANSCRIBED_REAL_COUNT,
                   TRANSCRIBED_SIMS_MEAN_COUNT,
                   UNTRANSCRIBED_SIMS_MEAN_COUNT,
                   fold_enrichment,
                   column3_real_name = NONTRANSCRIBED_REAL_COUNT)

        func_part2(type_genic_versus_intergenic_df,
                   GENIC_VERSUS_INTERGENIC_Q_VALUE,
                   GENIC_VERSUS_INTERGENIC,
                   element_names,
                   all_FDR_BH_adjusted_p_values,
                   percentage_strings,
                   percentage_numbers,
                   GENIC, # strand1
                   INTERGENIC, # strand2
                   GENIC_REAL_COUNT,
                   INTERGENIC_REAL_COUNT,
                   GENIC_SIMS_MEAN_COUNT,
                   INTERGENIC_SIMS_MEAN_COUNT,
                   fold_enrichment)

        func_part2(type_lagging_versus_leading_df,
                   LAGGING_VERSUS_LEADING_Q_VALUE,
                   LAGGING_VERSUS_LEADING,
                   element_names,
                   all_FDR_BH_adjusted_p_values,
                   percentage_strings,
                   percentage_numbers,
                   LAGGING, # strand1
                   LEADING, # strand2
                   LAGGING_REAL_COUNT,
                   LEADING_REAL_COUNT,
                   LAGGING_SIMS_MEAN_COUNT,
                   LEADING_SIMS_MEAN_COUNT,
                   fold_enrichment)

        # For TC-NER analysis
        # plot histogram of the fold enrichment in intergenic regions
        plot_histogram_of_fold_enrichment_in_intergenic_regions(strand_bias_output_dir,
                                                                inflate_mutations_to_remove_TC_NER_effect,
                                                                consider_only_significant_results,
                                                                consider_also_DBS_ID_signatures,
                                                                significance_level,
                                                                signature_genic_versus_intergenic_df,
                                                                type_genic_versus_intergenic_df)

        # Reorder columns
        if signature_transcribed_versus_untranscribed_df is not None:
            signature_transcribed_versus_untranscribed_df = signature_transcribed_versus_untranscribed_df[['cancer_type', 'signature', 'mutation_type',
                'Transcribed_real_count', 'UnTranscribed_real_count', 'NonTranscribed_real_count',
                'Transcribed_mean_sims_count', 'UnTranscribed_mean_sims_count', 'NonTranscribed_mean_sims_count',
                'transcribed_versus_untranscribed_p_value', 'transcribed_versus_untranscribed_q_value',
                'percentage_of_mutations_on_strands', 'signature_number_of_mutations_on_strands',
                SIGNIFICANT_STRAND, REAL_FOLD_ENRICHMENT, SIMS_FOLD_ENRICHMENT, FOLD_ENRICHMENT, FC_REAL, FC_SIMS, FC, percentage_strings[0], percentage_strings[1], percentage_strings[2], percentage_strings[3], percentage_strings[4], percentage_strings[5],
                'Transcribed_real_count.1', 'Transcribed_mean_sims_count.1', 'Transcribed_min_sims_count', 'Transcribed_max_sims_count', 'Transcribed_sims_count_list',
                'UnTranscribed_real_count.1', 'UnTranscribed_mean_sims_count.1', 'UnTranscribed_min_sims_count', 'UnTranscribed_max_sims_count', 'UnTranscribed_sims_count_list',
                'NonTranscribed_real_count.1', 'NonTranscribed_mean_sims_count.1', 'NonTranscribed_min_sims_count', 'NonTranscribed_max_sims_count', 'NonTranscribed_sims_count_list']]

        if signature_genic_versus_intergenic_df is not None:
            signature_genic_versus_intergenic_df = signature_genic_versus_intergenic_df[['cancer_type', 'signature', 'mutation_type',
                'genic_real_count', 'intergenic_real_count', 'genic_mean_sims_count', 'intergenic_mean_sims_count',
                'genic_versus_intergenic_p_value', 'genic_versus_intergenic_q_value',
                'percentage_of_mutations_on_strands', 'signature_number_of_mutations_on_strands',
                SIGNIFICANT_STRAND, REAL_FOLD_ENRICHMENT, SIMS_FOLD_ENRICHMENT, FOLD_ENRICHMENT, FC_REAL, FC_SIMS, FC, percentage_strings[0], percentage_strings[1], percentage_strings[2], percentage_strings[3], percentage_strings[4], percentage_strings[5],
                'Transcribed_real_count', 'Transcribed_mean_sims_count', 'Transcribed_min_sims_count', 'Transcribed_max_sims_count', 'Transcribed_sims_count_list',
                'UnTranscribed_real_count', 'UnTranscribed_mean_sims_count', 'UnTranscribed_min_sims_count', 'UnTranscribed_max_sims_count', 'UnTranscribed_sims_count_list',
                'NonTranscribed_real_count', 'NonTranscribed_mean_sims_count', 'NonTranscribed_min_sims_count', 'NonTranscribed_max_sims_count', 'NonTranscribed_sims_count_list' ]]

        if signature_lagging_versus_leading_df is not None:
            signature_lagging_versus_leading_df = signature_lagging_versus_leading_df[['cancer_type', 'signature', 'mutation_type',
                'Lagging_real_count', 'Leading_real_count', 'Lagging_mean_sims_count', 'Leading_mean_sims_count',
                'lagging_versus_leading_p_value', 'lagging_versus_leading_q_value',
                'percentage_of_mutations_on_strands', 'signature_number_of_mutations_on_strands',
                SIGNIFICANT_STRAND, REAL_FOLD_ENRICHMENT, SIMS_FOLD_ENRICHMENT, FOLD_ENRICHMENT, FC_REAL, FC_SIMS, FC, percentage_strings[0], percentage_strings[1], percentage_strings[2], percentage_strings[3], percentage_strings[4], percentage_strings[5],
                'Lagging_real_count.1', 'Lagging_mean_sims_count.1', 'Lagging_min_sims_count', 'Lagging_max_sims_count', 'Lagging_sims_count_list',
                'Leading_real_count.1', 'Leading_mean_sims_count.1', 'Leading_min_sims_count', 'Leading_max_sims_count', 'Leading_sims_count_list' ]]

        if type_transcribed_versus_untranscribed_df is not None:
            type_transcribed_versus_untranscribed_df = type_transcribed_versus_untranscribed_df[['cancer_type', 'type',
                'Transcribed_real_count', 'UnTranscribed_real_count', 'NonTranscribed_real_count',
                'Transcribed_mean_sims_count', 'UnTranscribed_mean_sims_count', 'NonTranscribed_mean_sims_count',
                'transcribed_versus_untranscribed_p_value', 'transcribed_versus_untranscribed_q_value',
                SIGNIFICANT_STRAND, REAL_FOLD_ENRICHMENT, SIMS_FOLD_ENRICHMENT, FOLD_ENRICHMENT, FC_REAL, FC_SIMS, FC, percentage_strings[0], percentage_strings[1], percentage_strings[2], percentage_strings[3], percentage_strings[4], percentage_strings[5],
                'Transcribed_real_count.1', 'Transcribed_mean_sims_count.1', 'Transcribed_min_sims_count', 'Transcribed_max_sims_count', 'Transcribed_sims_count_list',
                'UnTranscribed_real_count.1', 'UnTranscribed_mean_sims_count.1', 'UnTranscribed_min_sims_count', 'UnTranscribed_max_sims_count', 'UnTranscribed_sims_count_list',
                'NonTranscribed_real_count.1', 'NonTranscribed_mean_sims_count.1', 'NonTranscribed_min_sims_count', 'NonTranscribed_max_sims_count', 'NonTranscribed_sims_count_list']]

        if type_genic_versus_intergenic_df is not None:
            type_genic_versus_intergenic_df = type_genic_versus_intergenic_df[['cancer_type', 'type',
                'genic_real_count', 'intergenic_real_count', 'genic_mean_sims_count', 'intergenic_mean_sims_count',
                'genic_versus_intergenic_p_value', 'genic_versus_intergenic_q_value',
                SIGNIFICANT_STRAND, REAL_FOLD_ENRICHMENT, SIMS_FOLD_ENRICHMENT, FOLD_ENRICHMENT, FC_REAL, FC_SIMS, FC, percentage_strings[0], percentage_strings[1], percentage_strings[2], percentage_strings[3], percentage_strings[4], percentage_strings[5],
                'Transcribed_real_count', 'Transcribed_mean_sims_count', 'Transcribed_min_sims_count', 'Transcribed_max_sims_count', 'Transcribed_sims_count_list',
                'UnTranscribed_real_count', 'UnTranscribed_mean_sims_count', 'UnTranscribed_min_sims_count', 'UnTranscribed_max_sims_count', 'UnTranscribed_sims_count_list',
                'NonTranscribed_real_count', 'NonTranscribed_mean_sims_count', 'NonTranscribed_min_sims_count', 'NonTranscribed_max_sims_count', 'NonTranscribed_sims_count_list' ]]

        if type_lagging_versus_leading_df is not None:
            type_lagging_versus_leading_df = type_lagging_versus_leading_df[['cancer_type', 'type',
                'Lagging_real_count', 'Leading_real_count', 'Lagging_mean_sims_count', 'Leading_mean_sims_count',
                'lagging_versus_leading_p_value', 'lagging_versus_leading_q_value',
                SIGNIFICANT_STRAND, REAL_FOLD_ENRICHMENT, SIMS_FOLD_ENRICHMENT, FOLD_ENRICHMENT, FC_REAL, FC_SIMS, FC, percentage_strings[0], percentage_strings[1], percentage_strings[2], percentage_strings[3], percentage_strings[4], percentage_strings[5],
                'Lagging_real_count.1', 'Lagging_mean_sims_count.1', 'Lagging_min_sims_count', 'Lagging_max_sims_count', 'Lagging_sims_count_list',
                'Leading_real_count.1', 'Leading_mean_sims_count.1', 'Leading_min_sims_count', 'Leading_max_sims_count', 'Leading_sims_count_list' ]]


    signature_transcribed_versus_untranscribed_filtered_q_value_df = func_part3(signature_transcribed_versus_untranscribed_df,
                                                                                TRANSCRIBED_VERSUS_UNTRANSCRIBED_Q_VALUE,
                                                                                significance_level,
                                                                                min_required_number_of_mutations_on_strands,
                                                                                min_required_percentage_of_mutations_on_strands,
                                                                                TRANSCRIBED_REAL_COUNT,
                                                                                UNTRANSCRIBED_REAL_COUNT,
                                                                                column3_name=NONTRANSCRIBED_REAL_COUNT)

    signature_genic_versus_intergenic_filtered_q_value_df = func_part3(signature_genic_versus_intergenic_df,
                                                                GENIC_VERSUS_INTERGENIC_Q_VALUE,
                                                                significance_level,
                                                                min_required_number_of_mutations_on_strands,
                                                                min_required_percentage_of_mutations_on_strands,
                                                                GENIC_REAL_COUNT,
                                                                INTERGENIC_REAL_COUNT)

    signature_lagging_versus_leading_filtered_q_value_df = func_part3(signature_lagging_versus_leading_df,
                                                                    LAGGING_VERSUS_LEADING_Q_VALUE,
                                                                    significance_level,
                                                                    min_required_number_of_mutations_on_strands,
                                                                    min_required_percentage_of_mutations_on_strands,
                                                                    LAGGING_REAL_COUNT,
                                                                    LEADING_REAL_COUNT)

    type_transcribed_versus_untranscribed_filtered_q_value_df = func_part3(type_transcribed_versus_untranscribed_df,
                                                                        TRANSCRIBED_VERSUS_UNTRANSCRIBED_Q_VALUE,
                                                                        significance_level,
                                                                        min_required_number_of_mutations_on_strands,
                                                                        min_required_percentage_of_mutations_on_strands,
                                                                        TRANSCRIBED_REAL_COUNT,
                                                                        UNTRANSCRIBED_REAL_COUNT,
                                                                        column3_name=NONTRANSCRIBED_REAL_COUNT)

    type_genic_versus_intergenic_filtered_q_value_df = func_part3(type_genic_versus_intergenic_df,
                                                                GENIC_VERSUS_INTERGENIC_Q_VALUE,
                                                                significance_level,
                                                                min_required_number_of_mutations_on_strands,
                                                                min_required_percentage_of_mutations_on_strands,
                                                                GENIC_REAL_COUNT,
                                                                INTERGENIC_REAL_COUNT)

    type_lagging_versus_leading_filtered_q_value_df = func_part3(type_lagging_versus_leading_df,
                                                                LAGGING_VERSUS_LEADING_Q_VALUE,
                                                                significance_level,
                                                                min_required_number_of_mutations_on_strands,
                                                                min_required_percentage_of_mutations_on_strands,
                                                                LAGGING_REAL_COUNT,
                                                                LEADING_REAL_COUNT)



    return signature_transcribed_versus_untranscribed_df,\
           signature_transcribed_versus_untranscribed_filtered_q_value_df, \
           signature_genic_versus_intergenic_df,\
           signature_genic_versus_intergenic_filtered_q_value_df, \
           signature_lagging_versus_leading_df,\
           signature_lagging_versus_leading_filtered_q_value_df,\
           type_transcribed_versus_untranscribed_df,\
           type_transcribed_versus_untranscribed_filtered_q_value_df, \
           type_genic_versus_intergenic_df, \
           type_genic_versus_intergenic_filtered_q_value_df, \
           type_lagging_versus_leading_df,\
           type_lagging_versus_leading_filtered_q_value_df

# Call this method for strand bias figures wih percentages and circles.
def plot_strand_bias_figures(combined_output_dir,
                            cancer_types,
                            sbs_signatures,
                            dbs_signatures,
                            id_signatures,
                            strand_bias_output_dir,
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
                            inflate_mutations_to_remove_TC_NER_effect,
                            consider_only_significant_results,
                            consider_also_DBS_ID_signatures,
                            fold_enrichment,
                            figure_case_study = None):

    deleteOldData(strand_bias_output_dir)

    os.makedirs(os.path.join(strand_bias_output_dir), exist_ok=True)
    os.makedirs(os.path.join(strand_bias_output_dir, TABLES), exist_ok=True)
    os.makedirs(os.path.join(strand_bias_output_dir, EXCEL_FILES), exist_ok=True)
    os.makedirs(os.path.join(strand_bias_output_dir, DATA_FILES), exist_ok=True)

    for figure_type in figure_types:
        if figure_type == COSMIC:
            os.makedirs(os.path.join(strand_bias_output_dir, FIGURES_COSMIC), exist_ok=True)
            os.makedirs(os.path.join(strand_bias_output_dir, COSMIC_TISSUE_BASED_FIGURES), exist_ok=True)
        elif figure_type == MANUSCRIPT:
            os.makedirs(os.path.join(strand_bias_output_dir, FIGURES_MANUSCRIPT), exist_ok=True)

    # Updated for lymphoid samples
    signature_transcribed_versus_untranscribed_df,\
    signature_transcribed_versus_untranscribed_filtered_q_value_df, \
    signature_genic_versus_intergenic_df,\
    signature_genic_versus_intergenic_filtered_q_value_df, \
    signature_lagging_versus_leading_df,\
    signature_lagging_versus_leading_filtered_q_value_df, \
    type_transcribed_versus_untranscribed_df,\
    type_transcribed_versus_untranscribed_filtered_q_value_df,\
    type_genic_versus_intergenic_df,\
    type_genic_versus_intergenic_filtered_q_value_df, \
    type_lagging_versus_leading_df,\
    type_lagging_versus_leading_filtered_q_value_df = fill_strand_bias_dfs(combined_output_dir,
                                cancer_types,
                                strand_biases,
                                percentage_numbers,
                                percentage_strings,
                                significance_level,
                                min_required_number_of_mutations_on_strands,
                                min_required_percentage_of_mutations_on_strands,
                                strand_bias_output_dir,
                                dbs_signatures,
                                id_signatures,
                                inflate_mutations_to_remove_TC_NER_effect,
                                consider_only_significant_results,
                                consider_also_DBS_ID_signatures,
                                fold_enrichment)

    signature2mutation_type2strand2percent2cancertypeslist_dict, \
    type2strand2percent2cancertypeslist_dict = fill_strand_bias_dictionaries(signature_transcribed_versus_untranscribed_filtered_q_value_df,
                                signature_genic_versus_intergenic_filtered_q_value_df,
                                signature_lagging_versus_leading_filtered_q_value_df,
                                type_transcribed_versus_untranscribed_filtered_q_value_df,
                                type_genic_versus_intergenic_filtered_q_value_df,
                                type_lagging_versus_leading_filtered_q_value_df,
                                percentage_strings)

    # Consider the signatures in signature2cancer_type_list_dict
    new_signature2mutation_type2strand2percent2cancertypeslist_dict = {}
    for signature in signature2cancer_type_list_dict:
        if signature in signature2mutation_type2strand2percent2cancertypeslist_dict:
            new_signature2mutation_type2strand2percent2cancertypeslist_dict[signature] = signature2mutation_type2strand2percent2cancertypeslist_dict[signature]

    # Write these dictionaries as dataframes
    filename = 'Signature_Mutation_Type_Strand_Cancer_Types_Percentages_Table.txt'
    filepath = os.path.join(strand_bias_output_dir, TABLES, filename)
    signature_mutation_type_strand_cancer_types_percentages_df = write_signature_dictionaries_as_dataframes(new_signature2mutation_type2strand2percent2cancertypeslist_dict,
                                                                                                          signature2cancer_type_list_dict,
                                                                                                          percentage_strings,
                                                                                                          filepath)

    # Write these dictionaries as dataframes
    filename = 'Type_Strand_Cancer_Types_Percentages_Table.txt'
    filepath = os.path.join(strand_bias_output_dir, TABLES, filename)
    type_strand_cancer_types_percentages_df = write_my_type_dictionaries_as_dataframes(type2strand2percent2cancertypeslist_dict,
                                                                                       signature2cancer_type_list_dict,
                                                                                       percentage_strings,
                                                                                       filepath)

    # Write tables in excel files starts
    excel_file_name = '%s_SBS_Signatures_%s.xlsx' %(cosmic_release_version, COSMIC_TRANSCRIPTION_STRAND_BIAS)
    excel_file_path = os.path.join(strand_bias_output_dir,EXCEL_FILES, excel_file_name)
    df_list = []
    sheet_list = []
    if signature_transcribed_versus_untranscribed_df is not None:
        df_list.append(signature_transcribed_versus_untranscribed_df)
        sheet_list.append('p_value_q_value')
    if signature_transcribed_versus_untranscribed_filtered_q_value_df is not None:
        df_list.append(signature_transcribed_versus_untranscribed_filtered_q_value_df)
        sheet_list.append('filtered')
    if signature_mutation_type_strand_cancer_types_percentages_df is not None:
        df_list.append(signature_mutation_type_strand_cancer_types_percentages_df)
        sheet_list.append('all_results')
    write_excel_file(df_list, sheet_list, excel_file_path)

    excel_file_name = '%s_SBS_Signatures_%s.xlsx' %(cosmic_release_version, COSMIC_REPLICATION_STRAND_BIAS)
    excel_file_path = os.path.join(strand_bias_output_dir,EXCEL_FILES, excel_file_name)
    df_list = []
    sheet_list = []
    if signature_lagging_versus_leading_df is not None:
        df_list.append(signature_lagging_versus_leading_df)
        sheet_list.append('p_value_q_value')
    if signature_lagging_versus_leading_filtered_q_value_df is not None:
        df_list.append(signature_lagging_versus_leading_filtered_q_value_df)
        sheet_list.append('filtered')
    if signature_mutation_type_strand_cancer_types_percentages_df is not None:
        df_list.append(signature_mutation_type_strand_cancer_types_percentages_df)
        sheet_list.append('all_results')
    write_excel_file(df_list, sheet_list, excel_file_path)

    excel_file_name = '%s_SBS_Signatures_%s.xlsx' %(cosmic_release_version, COSMIC_GENIC_VS_INTERGENIC_BIAS)
    excel_file_path = os.path.join(strand_bias_output_dir,EXCEL_FILES, excel_file_name)
    df_list = []
    sheet_list = []
    if signature_genic_versus_intergenic_df is not None:
        df_list.append(signature_genic_versus_intergenic_df)
        sheet_list.append('p_value_q_value')
    if signature_genic_versus_intergenic_filtered_q_value_df is not None:
        df_list.append(signature_genic_versus_intergenic_filtered_q_value_df)
        sheet_list.append('filtered')
    if signature_mutation_type_strand_cancer_types_percentages_df is not None:
        df_list.append(signature_mutation_type_strand_cancer_types_percentages_df)
        sheet_list.append('all_results')
    write_excel_file(df_list, sheet_list, excel_file_path)

    excel_file_name = '%s_Type_%s.xlsx' %(cosmic_release_version, COSMIC_TRANSCRIPTION_STRAND_BIAS)
    excel_file_path = os.path.join(strand_bias_output_dir, EXCEL_FILES, excel_file_name)
    df_list = []
    sheet_list = []
    if type_transcribed_versus_untranscribed_df is not None:
        df_list.append(type_transcribed_versus_untranscribed_df)
        sheet_list.append('p_value_q_value')
    if type_transcribed_versus_untranscribed_filtered_q_value_df is not None:
        df_list.append(type_transcribed_versus_untranscribed_filtered_q_value_df)
        sheet_list.append('filtered')
    if type_strand_cancer_types_percentages_df is not None:
        df_list.append(type_strand_cancer_types_percentages_df)
        sheet_list.append('all_results')
    write_excel_file(df_list, sheet_list, excel_file_path)

    excel_file_name = '%s_Type_%s.xlsx' %(cosmic_release_version, COSMIC_REPLICATION_STRAND_BIAS)
    excel_file_path = os.path.join(strand_bias_output_dir, EXCEL_FILES, excel_file_name)
    df_list = []
    sheet_list = []
    if type_lagging_versus_leading_df is not None:
        df_list.append(type_lagging_versus_leading_df)
        sheet_list.append('p_value_q_value')
    if type_lagging_versus_leading_filtered_q_value_df is not None:
        df_list.append(type_lagging_versus_leading_filtered_q_value_df)
        sheet_list.append('filtered')
    if type_strand_cancer_types_percentages_df is not None:
        df_list.append(type_strand_cancer_types_percentages_df)
        sheet_list.append('all_results')
    write_excel_file(df_list, sheet_list, excel_file_path)

    excel_file_name = '%s_Type_%s.xlsx' %(cosmic_release_version, COSMIC_GENIC_VS_INTERGENIC_BIAS)
    excel_file_path = os.path.join(strand_bias_output_dir, EXCEL_FILES, excel_file_name)
    df_list = []
    sheet_list = []
    if type_genic_versus_intergenic_df is not None:
        df_list.append(type_genic_versus_intergenic_df)
        sheet_list.append('p_value_q_value')
    if type_genic_versus_intergenic_filtered_q_value_df is not None:
        df_list.append(type_genic_versus_intergenic_filtered_q_value_df)
        sheet_list.append('filtered')
    if type_strand_cancer_types_percentages_df is not None:
        df_list.append(type_strand_cancer_types_percentages_df)
        sheet_list.append('all_results')
    write_excel_file(df_list, sheet_list, excel_file_path)

    # Step5 Plot strand bias figures
    for figure_type in figure_types:
        if figure_type == MANUSCRIPT:
            # plot legends
            legend_path = os.path.join(strand_bias_output_dir, FIGURES_MANUSCRIPT)
            plot_legend(legend_path)
            plot_proportion_of_cancer_types(legend_path)

            for strand_bias in [TRANSCRIBED_VERSUS_UNTRANSCRIBED, GENIC_VERSUS_INTERGENIC, LAGGING_VERSUS_LEADING]:
                if strand_bias == LAGGING_VERSUS_LEADING:
                    strands = replication_strands
                    colours = replication_strand_bias_colours
                    cmap = mpl.colors.ListedColormap(colours)
                    norm = mpl.colors.BoundaryNorm(boundaries=strand_bias_color_bins, ncolors=len(cmap.colors))

                elif strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
                    strands = transcription_strands
                    colours = transcrition_strand_bias_colours
                    cmap = mpl.colors.ListedColormap(colours)
                    norm = mpl.colors.BoundaryNorm(boundaries=strand_bias_color_bins, ncolors=len(cmap.colors))

                elif strand_bias == GENIC_VERSUS_INTERGENIC:
                    strands = genic_versus_intergenic_strands
                    colours = genic_vs_intergenic_bias_colours
                    cmap = mpl.colors.ListedColormap(colours)
                    norm = mpl.colors.BoundaryNorm(boundaries=strand_bias_color_bins, ncolors=len(cmap.colors))

                plot_colorbar(strand_bias_output_dir, strand_bias, strands, colours)

                # Squeezed Figures (1.1, 1.2, 1.3, 1.5. 1.75. 2+ are averaged)
                plot_new_six_mutations_sbs_signatures_circle_figures(sbs_signatures,
                                strand_bias,
                                strands,
                                cmap,
                                norm,
                                strand_bias_output_dir,
                                significance_level,
                                signature2mutation_type2strand2percent2cancertypeslist_dict,
                                signature2cancer_type_list_dict,
                                percentage_strings,
                                signature_transcribed_versus_untranscribed_filtered_q_value_df,
                                signature_genic_versus_intergenic_filtered_q_value_df,
                                signature_lagging_versus_leading_filtered_q_value_df)

                plot_new_dbs_and_id_signatures_figures(DBS,
                                                   dbs_signatures,
                                                   strand_bias,
                                                   strands,
                                                   cmap,
                                                   norm,
                                                   strand_bias_output_dir,
                                                   significance_level,
                                                   type2strand2percent2cancertypeslist_dict,
                                                   signature2cancer_type_list_dict,
                                                   percentage_strings)

                plot_new_dbs_and_id_signatures_figures(ID,
                                                    id_signatures,
                                                    strand_bias,
                                                    strands,
                                                    cmap,
                                                    norm,
                                                    strand_bias_output_dir,
                                                    significance_level,
                                                    type2strand2percent2cancertypeslist_dict,
                                                    signature2cancer_type_list_dict,
                                                    percentage_strings)

                # Old version - not squeezed
                # For each mutation_type e.g.: C>A there are six fold changes [1.1, 1.2, 1.3, 1.5. 1.75. 2+]
                # plot_six_mutations_sbs_signatures_circle_figures(sbs_signatures,
                #                                 strand_bias,
                #                                 strand_bias_output_dir,
                #                                 significance_level,
                #                                 signature2mutation_type2strand2percent2cancertypeslist_dict,
                #                                 signature2cancer_type_list_dict,
                #                                 percentage_strings)

                # Old version
                # plot_dbs_and_id_signatures_figures(DBS,
                #                                    dbs_signatures,
                #                                    strand_bias,
                #                                    strand_bias_output_dir,
                #                                    significance_level,
                #                                    type2strand2percent2cancertypeslist_dict,
                #                                    signature2cancer_type_list_dict,
                #                                    percentage_strings,
                #                                    figure_type,
                #                                    cosmic_release_version,
                #                                    figure_file_extension)

                # Old version
                # plot_dbs_and_id_signatures_figures(ID,
                #                                    id_signatures,
                #                                    strand_bias,
                #                                    strand_bias_output_dir,
                #                                    significance_level,
                #                                    type2strand2percent2cancertypeslist_dict,
                #                                    signature2cancer_type_list_dict,
                #                                    percentage_strings,
                #                                    figure_type,
                #                                    cosmic_release_version,
                #                                    figure_file_extension)

        elif figure_type == COSMIC:
            for strand_bias in [TRANSCRIBED_VERSUS_UNTRANSCRIBED, GENIC_VERSUS_INTERGENIC, LAGGING_VERSUS_LEADING]:
                if (strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED) and (signature_transcribed_versus_untranscribed_df is not None):
                    signature_strand1_versus_strand2_for_bar_plot_df = combine_p_values(strand_bias,signature_transcribed_versus_untranscribed_df)
                    filename = 'signature_transcribed_versus_untranscribed_for_bar_plot_df.txt'
                    signature_strand1_versus_strand2_for_bar_plot_df.to_csv(os.path.join(strand_bias_output_dir, TABLES, filename), sep="\t", index=False)
                elif (strand_bias == LAGGING_VERSUS_LEADING) and (signature_lagging_versus_leading_df is not None):
                    signature_strand1_versus_strand2_for_bar_plot_df = combine_p_values(strand_bias,signature_lagging_versus_leading_df)
                    filename = 'signature_lagging_versus_leading_for_bar_plot_df.txt'
                    signature_strand1_versus_strand2_for_bar_plot_df.to_csv(os.path.join(strand_bias_output_dir, TABLES, filename), sep="\t", index=False)
                elif strand_bias == GENIC_VERSUS_INTERGENIC and (signature_genic_versus_intergenic_df is not None):
                    signature_strand1_versus_strand2_for_bar_plot_df = combine_p_values(strand_bias,signature_genic_versus_intergenic_df)
                    filename = 'signature_genic_versus_intergenic_for_bar_plot_df.txt'
                    signature_strand1_versus_strand2_for_bar_plot_df.to_csv(os.path.join(strand_bias_output_dir, TABLES, filename), sep="\t", index=False)

                # Parallel Version starts
                signature_tuples = []
                for sbs_signature in sbs_signatures:
                    signature_tuples.append((sbs_signature, SBS))
                for dbs_signature in dbs_signatures:
                    signature_tuples.append((dbs_signature, DBS))
                for id_signature in id_signatures:
                    signature_tuples.append((id_signature, ID))

                # Parallel version for real runs
                numofProcesses = multiprocessing.cpu_count()
                pool = multiprocessing.Pool(numofProcesses)

                for signature_tuple in signature_tuples:
                    signature, signature_type = signature_tuple
                    pool.apply_async(plot_cosmic_strand_bias_figure_in_parallel,
                                     args=(signature,
                                           signature_type,
                                           signature2cancer_type_list_dict,
                                           strand_bias,
                                           strand_bias_output_dir,
                                           signature_strand1_versus_strand2_for_bar_plot_df,
                                           signature_transcribed_versus_untranscribed_df,
                                           signature_genic_versus_intergenic_df,
                                           signature_lagging_versus_leading_df,
                                           signature_mutation_type_strand_cancer_types_percentages_df,
                                           signature2mutation_type2strand2percent2cancertypeslist_dict,
                                           type2strand2percent2cancertypeslist_dict,
                                           cancer_type2source_cancer_type_tuples_dict,
                                           percentage_strings,
                                           significance_level,
                                           number_of_required_mutations_for_stacked_bar_plot,
                                           figure_type,
                                           cosmic_release_version,
                                           figure_file_extension,
                                           figure_case_study,
                                           ),
                                     )

                pool.close()
                pool.join()
                # Parallel Version ends

                # # Sequential Version
                # for signature_tuple in signature_tuples:
                #     signature, signature_type = signature_tuple
                #     if (signature in signature2cancer_type_list_dict) and (len(signature2cancer_type_list_dict[signature]) > 0):
                #         plot_cosmic_strand_bias_figure_in_parallel(signature,
                #                                            signature_type,
                #                                            signature2cancer_type_list_dict,
                #                                            strand_bias,
                #                                            strand_bias_output_dir,
                #                                            signature_strand1_versus_strand2_for_bar_plot_df,
                #                                            signature_transcribed_versus_untranscribed_df,
                #                                            signature_genic_versus_intergenic_df,
                #                                            signature_lagging_versus_leading_df,
                #                                            signature_mutation_type_strand_cancer_types_percentages_df,
                #                                            signature2mutation_type2strand2percent2cancertypeslist_dict,
                #                                            type2strand2percent2cancertypeslist_dict,
                #                                            cancer_type2source_cancer_type_tuples_dict,
                #                                            percentage_strings,
                #                                            significance_level,
                #                                            number_of_required_mutations_for_stacked_bar_plot,
                #                                            figure_type,
                #                                            cosmic_release_version,
                #                                            figure_file_extension,
                #                                            figure_case_study)



def any_bias_to_show(signature, strand_bias,type2strand2percent2cancertypeslist_dict):
    all_cancer_type_list=[]

    if strand_bias==LAGGING_VERSUS_LEADING:
        strands=replication_strands
    elif strand_bias==TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        strands=transcription_strands
    elif strand_bias==GENIC_VERSUS_INTERGENIC:
        strands=genic_versus_intergenic_strands

    if signature in type2strand2percent2cancertypeslist_dict:
        for strand in strands:
            if strand in type2strand2percent2cancertypeslist_dict[signature]:
                for percentage in type2strand2percent2cancertypeslist_dict[signature][strand]:
                    cancer_type_list=type2strand2percent2cancertypeslist_dict[signature][strand][percentage]
                    all_cancer_type_list.extend(cancer_type_list)
    return len(all_cancer_type_list)>0



def any_bias_to_show_for_six_mutation_types(sbs_signature,strand_bias,signature2mutation_type2strand2percent2cancertypeslist_dict):
    all_cancer_type_list=[]

    if strand_bias==LAGGING_VERSUS_LEADING:
        strands=replication_strands
    elif strand_bias==TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        strands=transcription_strands
    elif strand_bias==GENIC_VERSUS_INTERGENIC:
        strands=genic_versus_intergenic_strands

    if sbs_signature in signature2mutation_type2strand2percent2cancertypeslist_dict:
        for mutation_type in signature2mutation_type2strand2percent2cancertypeslist_dict[sbs_signature]:
            for strand in strands:
                if strand in signature2mutation_type2strand2percent2cancertypeslist_dict[sbs_signature][mutation_type]:
                    for percentage in signature2mutation_type2strand2percent2cancertypeslist_dict[sbs_signature][mutation_type][strand]:
                        cancer_type_list=signature2mutation_type2strand2percent2cancertypeslist_dict[sbs_signature][mutation_type][strand][percentage]
                        all_cancer_type_list.extend(cancer_type_list)

    return len(all_cancer_type_list)>0



def calculate_radius(percentage_of_cancer_types):
    if int(percentage_of_cancer_types)==100:
        radius = (percentage_of_cancer_types / 102) / 2
    else:
        radius = (percentage_of_cancer_types / 100) / 2

    return radius

def plot_stacked_bar_plot_in_given_axis(axis,
                signature,
                strand_bias,
                strand_bias_output_dir,
                signature_strand1_versus_strand2_for_bar_plot_df,
                signature_transcribed_versus_untranscribed_df,
                signature_genic_versus_intergenic_df,
                signature_lagging_versus_leading_df,
                mutation_type_display,
                significance_level,
                number_of_required_mutations_for_stacked_bar_plot,
                tissue_based=None):

    if axis:
        box = axis.get_position()
        axis.set_position([box.x0, box.y0+0.125, box.width * 1, box.height * 1], which='both')

    mutation_types = six_mutation_types
    numberofSimulations=100
    width = 0.20

    if strand_bias==LAGGING_VERSUS_LEADING:
        strands=replication_strands
        color1='indianred'
        color2='goldenrod'
        subtitle = 'Lagging vs. Leading'
        if tissue_based:
            strand1 = "Lagging_real_count"
            strand2 = "Leading_real_count"
            strand1_sims = "Lagging_mean_sims_count"
            strand2_sims = "Leading_mean_sims_count"
            q_value_column_name = 'lagging_versus_leading_q_value'
        else:
            strand1 = "lagging_real_count_mean"
            strand2 = "leading_real_count_mean"
            strand1_sims = "lagging_sims_mean_count_mean"
            strand2_sims = "leading_sims_mean_count_mean"
            q_value_column_name = "q_value"
    elif strand_bias==TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        strands=transcription_strands
        color1='royalblue'
        color2='yellowgreen'
        subtitle = 'Transcribed vs. Untranscribed'
        if tissue_based:
            strand1 = "Transcribed_real_count"
            strand2 = "UnTranscribed_real_count"
            strand1_sims = "Transcribed_mean_sims_count"
            strand2_sims = "UnTranscribed_mean_sims_count"
            q_value_column_name = 'transcribed_versus_untranscribed_q_value'
        else:
            strand1="transcribed_real_count_mean"
            strand2="untranscribed_real_count_mean"
            strand1_sims="transcribed_sims_mean_count_mean"
            strand2_sims="untranscribed_sims_mean_count_mean"
            q_value_column_name = "q_value"
    elif strand_bias==GENIC_VERSUS_INTERGENIC:
        strands=genic_versus_intergenic_strands
        color1='cyan'
        color2='gray'
        subtitle = 'Genic vs. Intergenic'
        if tissue_based:
            strand1="genic_real_count"
            strand2="intergenic_real_count"
            strand1_sims="genic_mean_sims_count"
            strand2_sims="intergenic_mean_sims_count"
            q_value_column_name = "genic_versus_intergenic_q_value"
        else:
            strand1="genic_real_count_mean"
            strand2="intergenic_real_count_mean"
            strand1_sims="genic_sims_mean_count_mean"
            strand2_sims="intergenic_sims_mean_count_mean"
            q_value_column_name = "q_value"

    if tissue_based:
        if strand_bias == LAGGING_VERSUS_LEADING:
            groupby_df = signature_lagging_versus_leading_df.groupby(['signature','cancer_type'])
            group_df = groupby_df.get_group((signature,tissue_based))
        elif strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
            groupby_df = signature_transcribed_versus_untranscribed_df.groupby(['signature','cancer_type'])
            group_df = groupby_df.get_group((signature,tissue_based))
        elif strand_bias == GENIC_VERSUS_INTERGENIC:
            groupby_df = signature_genic_versus_intergenic_df.groupby(['signature','cancer_type'])
            group_df = groupby_df.get_group((signature,tissue_based))
    else:
        groupby_df = signature_strand1_versus_strand2_for_bar_plot_df.groupby(['signature'])
        group_df = groupby_df.get_group(signature)

    mutationtype_strand1_real_list=[]
    mutationtype_strand2_real_list=[]
    mutationtype_strand1_sims_mean_list=[]
    mutationtype_strand2_sims_mean_list=[]
    mutationtype_FDR_BH_adjusted_pvalues_list=[]
    for mutation_type in six_mutation_types:
        strand1_real_count=group_df[group_df['mutation_type']==mutation_type][strand1].values[0]
        strand2_real_count=group_df[group_df['mutation_type']==mutation_type][strand2].values[0]
        strand1_sims_count=group_df[group_df['mutation_type']==mutation_type][strand1_sims].values[0]
        strand2_sims_count=group_df[group_df['mutation_type']==mutation_type][strand2_sims].values[0]
        q_value=group_df[group_df['mutation_type']==mutation_type][q_value_column_name].values[0]
        mutationtype_FDR_BH_adjusted_pvalues_list.append(q_value)

        if (strand1_real_count >= number_of_required_mutations_for_stacked_bar_plot) or (strand2_real_count >= number_of_required_mutations_for_stacked_bar_plot):
            mutationtype_strand1_real_list.append(strand1_real_count/(strand1_real_count+strand2_real_count))
            mutationtype_strand2_real_list.append(strand2_real_count/(strand1_real_count+strand2_real_count))
        else:
            mutationtype_strand1_real_list.append(np.nan)
            mutationtype_strand2_real_list.append(np.nan)

        if (strand1_sims_count >= number_of_required_mutations_for_stacked_bar_plot) or (strand2_sims_count >= number_of_required_mutations_for_stacked_bar_plot):
            mutationtype_strand1_sims_mean_list.append(strand1_sims_count/(strand1_sims_count+strand2_sims_count))
            mutationtype_strand2_sims_mean_list.append(strand2_sims_count/(strand1_sims_count+strand2_sims_count))
        else:
            mutationtype_strand1_sims_mean_list.append(np.nan)
            mutationtype_strand2_sims_mean_list.append(np.nan)

    y_axis_label = 'Ratio of Mutations on Each Strand'
    stacked_bar_title = 'Real vs. Simulated: Odds Ratio of ' + subtitle
    plot_strand_bias_figure_with_stacked_bar_plots(strand_bias,
                             strand_bias_output_dir,
                             numberofSimulations,
                             signature,
                             len(mutation_types),
                             mutation_types,
                             y_axis_label,
                             stacked_bar_title,
                             mutationtype_strand1_real_list,
                             mutationtype_strand2_real_list,
                             mutationtype_strand1_sims_mean_list,
                             mutationtype_strand2_sims_mean_list,
                             mutation_type_display,
                             mutationtype_FDR_BH_adjusted_pvalues_list,
                             significance_level,
                             strands[0],
                             strands[1],
                             color1,
                             color2,
                             width,
                             tissue_based,
                             axis_given=axis)


def plot_bars_legend_in_given_axis(axis, strand_bias):
    box = axis.get_position()
    axis.set_position([box.x0 + 0.01, box.y0 + 0.125, box.width * 1, box.height * 1], which='both')

    # Put the legend
    if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        real_strand1_label = "Real %s" %(TRANSCRIBED_STRAND)
        real_strand2_label = "Real %s" %(UNTRANSCRIBED_STRAND)
        sims_strand1_label = "Simulated %s" %(TRANSCRIBED_STRAND)
        sims_strand2_label = "Simulated %s" %(UNTRANSCRIBED_STRAND)
        strand1_color = 'royalblue'
        strand2_color = 'yellowgreen'
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        real_strand1_label = "Real %s" %(GENIC)
        real_strand2_label = "Real %s" %(INTERGENIC)
        sims_strand1_label = "Simulated %s" %(GENIC)
        sims_strand2_label = "Simulated %s" %(INTERGENIC)
        strand1_color = 'cyan'
        strand2_color = 'gray'
    elif (strand_bias == LAGGING_VERSUS_LEADING):
        real_strand1_label = "Real %s" %(LAGGING)
        real_strand2_label = "Real %s" %(LEADING)
        sims_strand1_label = "Simulated %s" %(LAGGING)
        sims_strand2_label = "Simulated %s" %(LEADING)
        strand1_color = 'indianred'
        strand2_color = 'goldenrod'

    real_strand1_rectangle = mpatches.Patch(label=real_strand1_label, edgecolor='black', facecolor=strand1_color, lw=3)
    real_strand2_rectangle = mpatches.Patch(label=real_strand2_label, edgecolor='black', facecolor=strand2_color, lw=3)
    sims_strand1_rectangle = mpatches.Patch(label=sims_strand1_label, edgecolor='black', facecolor=strand1_color, lw=3, hatch='///')
    sims_strand2_rectangle = mpatches.Patch(label=sims_strand2_label, edgecolor='black', facecolor=strand2_color, lw=3, hatch='///')

    legend_elements = [
        real_strand1_rectangle,
        real_strand2_rectangle,
        sims_strand1_rectangle,
        sims_strand2_rectangle
        ]

    axis.set_axis_off()
    axis.legend(handles=legend_elements, ncol=1, loc='center left', fontsize=30)


def plot_bar_plot_in_given_axis(axis,
                    signature,
                    strand_bias,
                    strand_bias_output_dir,
                    signature_strand1_versus_strand2_for_bar_plot_df,
                    signature_transcribed_versus_untranscribed_df,
                    signature_genic_versus_intergenic_df,
                    signature_lagging_versus_leading_df,
                    significance_level,
                    tissue_based = None,
                    figure_case_study = None):

    if axis:
        box = axis.get_position()
        axis.set_position([box.x0, box.y0 + 0.125, box.width * 1, box.height * 1], which='both')

    mutation_types = six_mutation_types
    numberofSimulations = 100
    width = 0.20

    if strand_bias == LAGGING_VERSUS_LEADING:
        strands = replication_strands
        color1 = 'indianred'
        color2 = 'goldenrod'
        if tissue_based:
            strand1 = "Lagging_real_count"
            strand2 = "Leading_real_count"
            strand1_sims = "Lagging_mean_sims_count"
            strand2_sims = "Leading_mean_sims_count"
            q_value_column_name = 'lagging_versus_leading_q_value'
        else:
            strand1 = "lagging_real_count_mean"
            strand2 = "leading_real_count_mean"
            strand1_sims = "lagging_sims_mean_count_mean"
            strand2_sims = "leading_sims_mean_count_mean"
            q_value_column_name = 'q_value'
    elif strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        strands = transcription_strands
        color1 = 'royalblue'
        color2 = 'yellowgreen'
        if tissue_based:
            strand1 = "Transcribed_real_count"
            strand2 = "UnTranscribed_real_count"
            strand1_sims = "Transcribed_mean_sims_count"
            strand2_sims = "UnTranscribed_mean_sims_count"
            q_value_column_name = 'transcribed_versus_untranscribed_q_value'
        else:
            strand1 = "transcribed_real_count_mean"
            strand2 = "untranscribed_real_count_mean"
            strand1_sims = "transcribed_sims_mean_count_mean"
            strand2_sims = "untranscribed_sims_mean_count_mean"
            q_value_column_name = 'q_value'
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        strands = genic_versus_intergenic_strands
        color1 = 'cyan'
        color2 = 'gray'
        if tissue_based:
            strand1 = "genic_real_count"
            strand2 = "intergenic_real_count"
            strand1_sims = "genic_mean_sims_count"
            strand2_sims = "intergenic_mean_sims_count"
            q_value_column_name = "genic_versus_intergenic_q_value"
        else:
            strand1 = "genic_real_count_mean"
            strand2 = "intergenic_real_count_mean"
            strand1_sims = "genic_sims_mean_count_mean"
            strand2_sims = "intergenic_sims_mean_count_mean"
            q_value_column_name = 'q_value'

    if tissue_based:
        if strand_bias == LAGGING_VERSUS_LEADING:
            groupby_df = signature_lagging_versus_leading_df.groupby(['signature','cancer_type'])
            group_df = groupby_df.get_group((signature,tissue_based))
        elif strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
            groupby_df = signature_transcribed_versus_untranscribed_df.groupby(['signature','cancer_type'])
            group_df = groupby_df.get_group((signature,tissue_based))
        elif strand_bias == GENIC_VERSUS_INTERGENIC:
            groupby_df = signature_genic_versus_intergenic_df.groupby(['signature','cancer_type'])
            group_df = groupby_df.get_group((signature,tissue_based))
    else:
        groupby_df = signature_strand1_versus_strand2_for_bar_plot_df.groupby(['signature'])
        group_df = groupby_df.get_group(signature)

    mutationtype_strand1_real_list = []
    mutationtype_strand2_real_list = []
    mutationtype_strand1_sims_mean_list = []
    mutationtype_strand2_sims_mean_list = []
    mutationtype_FDR_BH_adjusted_pvalues_list = []

    for mutation_type in six_mutation_types:
        strand1_real_count = group_df[group_df['mutation_type'] == mutation_type][strand1].values[0]
        strand2_real_count = group_df[group_df['mutation_type'] == mutation_type][strand2].values[0]
        strand1_sims_count = group_df[group_df['mutation_type'] == mutation_type][strand1_sims].values[0]
        strand2_sims_count = group_df[group_df['mutation_type'] == mutation_type][strand2_sims].values[0]
        q_value = group_df[group_df['mutation_type'] == mutation_type][q_value_column_name].values[0]
        mutationtype_strand1_real_list.append(strand1_real_count)
        mutationtype_strand2_real_list.append(strand2_real_count)
        mutationtype_strand1_sims_mean_list.append(strand1_sims_count)
        mutationtype_strand2_sims_mean_list.append(strand2_sims_count)
        mutationtype_FDR_BH_adjusted_pvalues_list.append(q_value)

    y_axis_label = 'Number of Single Base Substitutions'
    mutation_type_display = plot_strand_bias_figure_with_bar_plots(strand_bias,
                             strand_bias_output_dir,
                             numberofSimulations,
                             signature,
                             len(mutation_types),
                             mutation_types,
                             y_axis_label,
                             mutationtype_strand1_real_list,
                             mutationtype_strand2_real_list,
                             mutationtype_strand1_sims_mean_list,
                             mutationtype_strand2_sims_mean_list,
                             mutationtype_FDR_BH_adjusted_pvalues_list,
                             significance_level,
                             strands[0],
                             strands[1],
                             color1,
                             color2,
                             width,
                             tissue_based,
                             figure_case_study,
                             axis_given = axis)

    return mutation_type_display



def plot_legend_in_given_axis(ax, strand_bias):
    # Put the legend
    if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        legend_elements = [
            Line2D([0], [0], marker='o', color='white', label=TRANSCRIBED_STRAND, markerfacecolor='royalblue',markersize=40),
            Line2D([0], [0], marker='o', color='white', label=UNTRANSCRIBED_STRAND, markerfacecolor='yellowgreen',markersize=40)]
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        legend_elements = [
            Line2D([0], [0], marker='o', color='white', label=GENIC, markerfacecolor='cyan', markersize=40),
            Line2D([0], [0], marker='o', color='white', label=INTERGENIC, markerfacecolor='gray', markersize=40)]
    elif (strand_bias == LAGGING_VERSUS_LEADING):
        legend_elements = [
            Line2D([0], [0], marker='o', color='white', label=LAGGING, markerfacecolor='indianred', markersize=40),
            Line2D([0], [0], marker='o', color='white', label=LEADING, markerfacecolor='goldenrod', markersize=40)]

    ax.set_axis_off()
    # A 2-tuple (x, y) places the corner of the legend specified by loc at x, y.
    # A 4-tuple specifies the bbox (x, y, width, height) that the legend is placed in.
    ax.legend(handles=legend_elements, ncol=len(legend_elements), bbox_to_anchor=(0, 0, 1, 1), loc='upper right', fontsize=40)  # bbox_to_anchor=(1, 2.75)
    # ax.legend(handles=legend_elements, ncol=len(legend_elements), bbox_to_anchor=(1, 2.75), loc='upper right', fontsize=40) # legacy

def plot_proportion_of_cancer_types_text_in_given_axis(ax):
    ax.set_axis_off()
    box = ax.get_position()

    ax.set_position([box.x0, box.y0 + 0.1, box.width * 1, box.height * 1], which='both')

    ax.text(box.x0 - 0.125, box.y0,
            PROPORTION_OF_CANCER_TYPES_WITH_STRAND_ASYMMERTY_OF_A_SIGNATURE,
            horizontalalignment='left',
            verticalalignment='center',
            fontsize=40,
            fontname = "Times New Roman",
            weight = 'bold',
            transform=ax.transAxes)


def plot_proportion_of_cancer_types_text_in_given_axis_new(ax, strand_bias):
    ax.set_axis_off()
    box = ax.get_position()

    if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED or strand_bias == LAGGING_VERSUS_LEADING:
        show_text = PROPORTION_OF_CANCER_TYPES_WITH_STRAND_ASYMMERTY_OF_A_SIGNATURE
    else:
        show_text = PROPORTION_OF_CANCER_TYPES_WITH_REGION_ASYMMERTY_OF_A_SIGNATURE

    ax.text(box.x0 - 0.125, box.y0,
            show_text,
            horizontalalignment='left',
            verticalalignment='center',
            fontsize=40,
            fontname = "Times New Roman",
            weight = 'bold',
            transform=ax.transAxes)


def plot_proportion_of_cancer_types_in_given_axis(ax, strand_bias, y0=None, write_text=False):
    # box = ax.get_position()
    # if y0 is None:
    #     # ax.set_position([box.x0, box.y0, box.width, box.height], which='both')  # legacy
    #     ax.set_position([box.x0, box.y0, 0.5, box.height], which='both')  # legacy
    #     # ax.set_position([box.x0 - 0.01, box.y0, box.width, box.height], which='both')  # legacy
    #     # ax.set_position([box.x0 - 0.05, box.y0, box.width, box.height], which='both')  # legacy
    # else:
    #     ax.set_position([box.x0 - 0.03, y0, box.width, box.height], which='both')
    diameter_labels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    diameter_ticklabels = [0.1, '', '', '', 0.5, '', '', '', '', 1.0]

    row_labels = ['circle']
    ax.grid(which="major", color="w", linestyle='-', linewidth=3) # original

    #Make the borders white
    plt.setp(ax.spines.values(), color='white')

    # make aspect ratio square
    ax.set_aspect(1.0)

    # ax.set_facecolor('white')
    ax.set_facecolor('lightcyan')

    for row_index, row_label in enumerate(row_labels):
        for diameter_index, diameter_label in enumerate(diameter_labels):
            circle=plt.Circle((diameter_index + 0.5, row_index + 0.5), radius=(diameter_label/(2*ONE_POINT_EIGHT)), color='gray', fill=True)
            ax.add_artist(circle)

    # CODE GOES HERE TO CENTER X-AXIS LABELS...
    ax.set_xlim([0, len(diameter_labels)])
    ax.set_xticklabels([])

    ax.tick_params(axis='x', which='both', length=0, labelsize=30)
    # major ticks
    ax.set_xticks(np.arange(0, len(diameter_labels), 1))
    # minor ticks
    ax.set_xticks(np.arange(0, len(diameter_labels), 1) + 0.5, minor=True)
    ax.set_xticklabels(diameter_ticklabels, minor=True, fontweight='bold', fontname='Arial')

    ax.xaxis.set_ticks_position('bottom')

    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='major',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False)  # labels along the bottom edge are off

    if write_text:
        if (strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED) or (strand_bias == LAGGING_VERSUS_LEADING):
            ax.set_xlabel(PROPORTION_OF_CANCER_TYPES_WITH_STRAND_ASYMMERTY_OF_A_SIGNATURE, fontsize=40, labelpad=10, fontname = "Times New Roman", weight = 'bold')
        elif strand_bias == GENIC_VERSUS_INTERGENIC:
            ax.set_xlabel(PROPORTION_OF_CANCER_TYPES_WITH_REGION_ASYMMERTY_OF_A_SIGNATURE, fontsize=40, labelpad=10, fontname = "Times New Roman", weight = 'bold')

    # ax.yaxis.set_label_position("right")
    # ax.set_ylabel('Proportion of tumor types\nwith the signature', fontsize=40, labelpad=10, rotation=0)

    # CODE GOES HERE TO CENTER Y-AXIS LABELS...
    ax.set_ylim([0, len(row_labels)])
    ax.set_yticklabels([])

    ax.tick_params(axis='y', which='minor', length=0, labelsize=30)
    # major ticks
    ax.set_yticks(np.arange(0, len(row_labels), 1))
    # minor ticks
    ax.set_yticks(np.arange(0, len(row_labels), 1) + 0.5, minor=True)
    # ax.set_yticklabels(row_labels, minor=True)  # fontsize

    ax.tick_params(
        axis='y',  # changes apply to the x-axis
        which='major',  # both major and minor ticks are affected
        left=False)  # labels along the bottom edge are off

    # Gridlines based on major ticks
    ax.grid(which='major', color='white')

# Plot proportion of cancer types only
def plot_proportion_of_cancer_types(strandbias_figures_outputDir):

    diameter_labels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    diameter_ticklabels = [0.1, '', '', '', 0.5, '', '', '', '', 1.0]

    row_labels = ['circle']

    # Make a figure and axes with dimensions as desired.
    fig, ax = plt.subplots(figsize=(8, 3))

    ax.grid(which="major", color="w", linestyle='-', linewidth=3)

    #Make the borders white
    plt.setp(ax.spines.values(), color='white')

    # make aspect ratio square
    ax.set_aspect(1.0)

    # ax.set_facecolor('white')
    ax.set_facecolor('lightcyan')

    for row_index, row_label in enumerate(row_labels):
        for diameter_index, diameter_label in enumerate(diameter_labels):
            circle=plt.Circle((diameter_index + 0.5, row_index + 0.5), radius=(diameter_label/(2*ONE_POINT_EIGHT)), color='gray', fill=True)
            ax.add_artist(circle)

    # CODE GOES HERE TO CENTER X-AXIS LABELS...
    ax.set_xlim([0, len(diameter_labels)])
    ax.set_xticklabels([])

    ax.tick_params(axis='x', which='minor', length=0, labelsize=12)
    # major ticks
    ax.set_xticks(np.arange(0, len(diameter_labels), 1))
    # minor ticks
    ax.set_xticks(np.arange(0, len(diameter_labels), 1) + 0.5, minor=True)
    ax.set_xticklabels(diameter_ticklabels, minor=True, fontweight='bold', fontname='Arial')

    ax.xaxis.set_ticks_position('bottom')

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='major',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False)  # labels along the bottom edge are off

    ax.set_xlabel('Proportion of cancer types with\nstrand asymmetry of a signature', fontsize=20, labelpad=10)

    # CODE GOES HERE TO CENTER Y-AXIS LABELS...
    ax.set_ylim([0, len(row_labels)])
    ax.set_yticklabels([])

    ax.tick_params(axis='y', which='minor', length=0, labelsize=12)
    # major ticks
    ax.set_yticks(np.arange(0, len(row_labels), 1))
    # minor ticks
    ax.set_yticks(np.arange(0, len(row_labels), 1) + 0.5, minor=True)
    # ax.set_yticklabels(row_labels, minor=True)  # fontsize

    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='major',  # both major and minor ticks are affected
        left=False)  # labels along the bottom edge are off

    # Gridlines based on major ticks
    ax.grid(which='major', color='white')

    filename = 'pcawg_proportion_of_cancer_types.png'
    figureFile = os.path.join(strandbias_figures_outputDir, filename)

    fig.savefig(figureFile, dpi=100, bbox_inches="tight")
    plt.close()


# Plot Legend only
def plot_legend(strandbias_figures_outputDir):

    strand_biases=[TRANSCRIBED_VERSUS_UNTRANSCRIBED, GENIC_VERSUS_INTERGENIC, LAGGING_VERSUS_LEADING]

    for strand_bias in strand_biases:
        fig = plt.figure(figsize=(4,1), dpi=300)
        ax = plt.gca()
        plt.axis('off')

        # Put the legend
        if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
            legend_elements = [
                Line2D([0], [0], marker='o', color='white', label='Genic: Transcribed Strand', markerfacecolor='royalblue', markersize=30),
                Line2D([0], [0], marker='o', color='white', label='Genic: Untranscribed Strand', markerfacecolor='yellowgreen', markersize=30)]
        elif strand_bias == GENIC_VERSUS_INTERGENIC:
            legend_elements = [
                Line2D([0], [0], marker='o', color='white', label='Genic Regions', markerfacecolor='cyan', markersize=30),
                Line2D([0], [0], marker='o', color='white', label='Intergenic Regions', markerfacecolor='gray', markersize=30)]
        elif (strand_bias == LAGGING_VERSUS_LEADING):
            legend_elements = [
                Line2D([0], [0], marker='o', color='white', label='Lagging Strand', markerfacecolor='indianred', markersize=30),
                Line2D([0], [0], marker='o', color='white', label='Leading Strand', markerfacecolor='goldenrod', markersize=30)]

        ax.legend(handles=legend_elements, bbox_to_anchor=(0, 0.5), loc='center left' ,fontsize = 30)

        # create the directory if it does not exists
        filename = 'Legend_%s.png' % (strand_bias)
        figFile = os.path.join(strandbias_figures_outputDir, filename)
        fig.savefig(figFile, dpi=100, bbox_inches="tight")
        # fig.tight_layout()

        plt.cla()
        plt.close(fig)


def plot_new_dbs_and_id_signatures_figures(signature_type,
                                           signatures,
                                           strand_bias,
                                           strands,
                                           cmap,
                                           norm,
                                           strand_bias_output_dir,
                                           significance_level,
                                           type2strand2percent2cancertypeslist_dict,
                                           signature2cancer_type_list_dict,
                                           percentage_strings):

    figure_dir = FIGURES_MANUSCRIPT

    rows_signatures=[]

    if strand_bias==LAGGING_VERSUS_LEADING:
        strands = replication_strands
    elif strand_bias==TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        strands = transcription_strands
    elif strand_bias==GENIC_VERSUS_INTERGENIC:
        strands = genic_versus_intergenic_strands

    # Fill rows_DBS_signatures
    # Fill rows_ID_signatures
    for signature in signatures:
        if signature  in type2strand2percent2cancertypeslist_dict:
            for strand in strands:
                if strand in type2strand2percent2cancertypeslist_dict[signature]:
                    for percentage_string in type2strand2percent2cancertypeslist_dict[signature][strand]:
                        if len(type2strand2percent2cancertypeslist_dict[signature][strand][percentage_string])>0:
                            if signature not in rows_signatures:
                                rows_signatures.append(signature)

    # No DBS and ID mutational signatures attributed to artifacts
    signatures_attributed_to_artifacts = []
    rows_signatures = list(set(rows_signatures) - set(signatures_attributed_to_artifacts))

    rows_signatures = sorted(rows_signatures, key=natural_key, reverse=True)

    rows_signatures_with_number_of_cancer_types = augment_with_number_of_cancer_types(signature_type, rows_signatures, signature2cancer_type_list_dict)

    if len(rows_signatures) <= 2:
        x_ticks_labelsize = 42
        y_ticks_labelsize = 52
    elif len(rows_signatures) <= 3:
        x_ticks_labelsize = 40
        y_ticks_labelsize = 50
    elif len(rows_signatures) <= 6:
        x_ticks_labelsize = 33
        y_ticks_labelsize = 41
    elif len(rows_signatures) <= 9:
        x_ticks_labelsize = 30
        y_ticks_labelsize = 40
    elif len(rows_signatures) <= 11:
        x_ticks_labelsize = 27
        y_ticks_labelsize = 33
    else:
        x_ticks_labelsize = 20
        y_ticks_labelsize = 30

    # Plot (width,height)
    # MANUSCRIPT
    fig, ax = plt.subplots(figsize=(5 + 1.75 * len(percentage_strings), 5 + len(rows_signatures))) # +5 is to avoid ValueError when there is no signature to show

    # set facecolor white
    ax.set_facecolor('white')

    # Make aspect ratio square
    ax.set_aspect(1.0)

    for percentage_string_index, percentage_string in enumerate(percentage_strings):
        for row_signature_index, row_signature in enumerate(rows_signatures):
            if row_signature in type2strand2percent2cancertypeslist_dict:
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

    ax.set_title('Fold Change', fontsize=y_ticks_labelsize, pad=20)

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
    ax.set_ylim([0,len(rows_signatures)])
    ax.set_yticklabels([])
    ax.tick_params(axis='y', which='both', length=0, labelsize=y_ticks_labelsize)

    #major ticks
    ax.set_yticks(np.arange(0, len(rows_signatures), 1))
    #minor ticks
    ax.set_yticks(np.arange(0, len(rows_signatures), 1)+0.5,minor=True)

    yticks = np.arange(0,len(rows_signatures_with_number_of_cancer_types))
    ax.set_yticks(yticks)
    ax.set_yticklabels(rows_signatures_with_number_of_cancer_types, minor=True, fontsize=y_ticks_labelsize)  # fontsize

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

    # v3.2_SBS1_REPLIC_ASYM_TA_C34447.jpg
    # for manuscript  signature_name = None and tissue_based = None
    filename = '%s_Signatures_%s_with_circles_%s.png' % (signature_type, strand_bias, str(significance_level).replace('.','_'))

    figFile = os.path.join(strand_bias_output_dir, figure_dir, filename)
    fig.savefig(figFile, dpi=100, bbox_inches="tight")

    plt.cla()
    plt.close(fig)


# Not used any more
# Formerly used by COSMIC and MANUSCRIPT
# Across all tissues or tissue based separately
def plot_dbs_and_id_signatures_figures(signature_type,
                                       signatures,
                                       strand_bias,
                                       strand_bias_figures_output_dir,
                                       SIGNIFICANCE_LEVEL,
                                       type2strand2percent2cancertypeslist_dict,
                                       signature2cancer_type_list_dict,
                                       percentage_strings,
                                       figure_type,
                                       cosmic_release_version,
                                       figure_file_extension,
                                       signature_name = None,
                                       tissue_based = None):

    if figure_type == MANUSCRIPT:
        x_ticks_labelsize = 35
        y_ticks_labelsize = 45
        figure_dir = FIGURES_MANUSCRIPT
    elif figure_type == COSMIC:
        x_ticks_labelsize = 35
        y_ticks_labelsize = 40
        if tissue_based:
            figure_dir = COSMIC_TISSUE_BASED_FIGURES
        else:
            figure_dir = FIGURES_COSMIC

    rows_signatures=[]

    if strand_bias==LAGGING_VERSUS_LEADING:
        strands=replication_strands
    elif strand_bias==TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        strands=transcription_strands
    elif strand_bias==GENIC_VERSUS_INTERGENIC:
        strands=genic_versus_intergenic_strands

    # Fill rows_DBS_signatures
    # Fill rows_ID_signatures
    for signature in signatures:
        if signature  in type2strand2percent2cancertypeslist_dict:
            for strand in strands:
                if strand in type2strand2percent2cancertypeslist_dict[signature]:
                    for percentage_string in type2strand2percent2cancertypeslist_dict[signature][strand]:
                        if len(type2strand2percent2cancertypeslist_dict[signature][strand][percentage_string])>0:
                            if signature not in rows_signatures:
                                rows_signatures.append(signature)

    # No DBS and ID mutational signatures attributed to artifacts
    signatures_attributed_to_artifacts = []
    rows_signatures = list(set(rows_signatures) - set(signatures_attributed_to_artifacts))

    rows_signatures = sorted(rows_signatures, key=natural_key, reverse=True)

    #This is for COSMIC
    if len(rows_signatures)==0 and signature_name:
        rows_signatures.append(signature)

    if tissue_based:
        # No need for number of cancer types having this signature
        rows_signatures_with_number_of_cancer_types = rows_signatures
    else:
        rows_signatures_with_number_of_cancer_types = augment_with_number_of_cancer_types(signature_type, rows_signatures, signature2cancer_type_list_dict)

    # Plot (width,height)
    if signature_name:
        # COSMIC
        fig = plt.figure(figsize=(5 + 1.5 * 3 * len(percentage_strings), 5 + 1.5 * len(rows_signatures)))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
        ax = plt.subplot(gs[0, :]) #yop_axis
        bottom_left_axis = plt.subplot(gs[-1, 0])
        bottom_right_axis = plt.subplot(gs[-1, -1])
        plot_proportion_of_cancer_types_in_given_axis(bottom_left_axis, bottom_left_axis.get_position().y0, write_text=True)
    else:
        # MANUSCRIPT
        fig, ax = plt.subplots(figsize=(5 + 1.75*len(percentage_strings), 5 + len(rows_signatures))) # +5 is to avoid ValueError when there is no signature to show
        # fig, ax = plt.subplots(figsize=(5 + 1.75*len(percentage_strings), len(rows_signatures)))

    # Make aspect ratio square
    ax.set_aspect(1.0)

    # Set title
    if figure_type==MANUSCRIPT:
        if strand_bias == LAGGING_VERSUS_LEADING:
            title = 'Lagging versus Leading Strand Bias'
        elif strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
            title = 'Transcribed versus Untranscribed Strand Bias'
        elif strand_bias == GENIC_VERSUS_INTERGENIC:
            title = 'Genic versus Intergenic Regions Bias'
    elif figure_type==COSMIC:
        if tissue_based:
            title = tissue_based
        else:
            if strand_bias == LAGGING_VERSUS_LEADING:
                title = '%s Lagging versus Leading Strand Bias' %(signatures[0])
            elif strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
                title = '%s Transcribed versus Untranscribed Strand Bias' %(signatures[0])
            elif strand_bias == GENIC_VERSUS_INTERGENIC:
                title = '%s Genic versus Intergenic Regions Bias' %(signatures[0])

    if figure_type==COSMIC:
        ax.text(len(percentage_strings)/2, len(rows_signatures)+1, title, horizontalalignment='center', fontsize=60, fontweight='bold', fontname='Arial')

    for percentage_diff_index, percentage_string in enumerate(percentage_strings):
        for row_signature_index, row_signature in enumerate(rows_signatures):
            if (strand_bias == LAGGING_VERSUS_LEADING):
                if row_signature in type2strand2percent2cancertypeslist_dict:
                    lagging_cancer_types_percentage = None
                    leading_cancer_types_percentage = None
                    if LAGGING in type2strand2percent2cancertypeslist_dict[row_signature]:
                        cancer_types_list = type2strand2percent2cancertypeslist_dict[row_signature][LAGGING][percentage_string]
                        all_cancer_types_list = signature2cancer_type_list_dict[row_signature]
                        if tissue_based and (tissue_based in cancer_types_list):
                            lagging_cancer_types_percentage = 100
                        elif not tissue_based:
                            lagging_cancer_types_percentage = (len(cancer_types_list) / len(all_cancer_types_list)) * 100
                    if LEADING in type2strand2percent2cancertypeslist_dict[row_signature]:
                        cancer_types_list=type2strand2percent2cancertypeslist_dict[row_signature][LEADING][percentage_string]
                        all_cancer_types_list = signature2cancer_type_list_dict[row_signature]
                        if tissue_based and (tissue_based in cancer_types_list):
                            leading_cancer_types_percentage = 100
                        elif not tissue_based:
                            leading_cancer_types_percentage = (len(cancer_types_list) / len(all_cancer_types_list)) * 100
                    if (lagging_cancer_types_percentage is not None) and (leading_cancer_types_percentage is None):
                        radius = calculate_radius(lagging_cancer_types_percentage)
                        if (radius > 0):
                            print('Plot circle at x=%d y=%d for %s %s' % (percentage_diff_index, row_signature_index, row_signature,percentage_string))
                            circle = plt.Circle((percentage_diff_index + 0.5, row_signature_index + 0.5), radius, color='indianred', fill=True)
                            ax.add_artist(circle)
                    elif (leading_cancer_types_percentage is not None) and (lagging_cancer_types_percentage is None):
                        radius = calculate_radius(leading_cancer_types_percentage)
                        if (radius > 0):
                            print('Plot circle at x=%d y=%d for %s %s' % (percentage_diff_index, row_signature_index, row_signature,percentage_string))
                            circle = plt.Circle((percentage_diff_index + 0.5, row_signature_index + 0.5), radius, color='goldenrod', fill=True)
                            ax.add_artist(circle)
                    elif (lagging_cancer_types_percentage is not None) and (leading_cancer_types_percentage is not None):
                        radius_lagging = calculate_radius(lagging_cancer_types_percentage)
                        radius_leading = calculate_radius(leading_cancer_types_percentage)
                        if (radius_lagging>radius_leading):
                            #First lagging
                            circle = plt.Circle((percentage_diff_index + 0.5, row_signature_index + 0.5), radius_lagging, color='goldenrod', fill=True)
                            ax.add_artist(circle)
                            #Second leading
                            circle = plt.Circle((percentage_diff_index + 0.5, row_signature_index + 0.5), radius_leading, color='goldenrod', fill=True)
                            ax.add_artist(circle)
                        else:
                            #First leading
                            circle = plt.Circle((percentage_diff_index + 0.5, row_signature_index + 0.5), radius_leading, color='goldenrod', fill=True)
                            ax.add_artist(circle)
                            #Second lagging
                            circle = plt.Circle((percentage_diff_index + 0.5, row_signature_index + 0.5), radius_lagging, color='goldenrod', fill=True)
                            ax.add_artist(circle)

            elif (strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED):
                if row_signature in type2strand2percent2cancertypeslist_dict:
                    transcribed_cancer_types_percentage = None
                    untranscribed_cancer_types_percentage = None
                    if TRANSCRIBED_STRAND in type2strand2percent2cancertypeslist_dict[row_signature]:
                        cancer_types_list = type2strand2percent2cancertypeslist_dict[row_signature][TRANSCRIBED_STRAND][percentage_string]
                        all_cancer_types_list = signature2cancer_type_list_dict[row_signature]
                        if tissue_based and (tissue_based in cancer_types_list):
                            transcribed_cancer_types_percentage = 100
                        elif not tissue_based:
                            transcribed_cancer_types_percentage = (len(cancer_types_list) / len(all_cancer_types_list)) * 100
                    if UNTRANSCRIBED_STRAND in type2strand2percent2cancertypeslist_dict[row_signature]:
                        cancer_types_list=type2strand2percent2cancertypeslist_dict[row_signature][UNTRANSCRIBED_STRAND][percentage_string]
                        all_cancer_types_list = signature2cancer_type_list_dict[row_signature]
                        if tissue_based and (tissue_based in cancer_types_list):
                            untranscribed_cancer_types_percentage = 100
                        elif not tissue_based:
                            untranscribed_cancer_types_percentage = (len(cancer_types_list) / len(all_cancer_types_list)) * 100
                    if (transcribed_cancer_types_percentage is not None) and (untranscribed_cancer_types_percentage is None):
                        radius = calculate_radius(transcribed_cancer_types_percentage)
                        if (radius > 0):
                            print('Plot circle at x=%d y=%d for %s %s' % (percentage_diff_index, row_signature_index, row_signature,percentage_string))
                            circle = plt.Circle((percentage_diff_index + 0.5, row_signature_index + 0.5), radius, color='royalblue', fill=True)
                            ax.add_artist(circle)
                    elif (untranscribed_cancer_types_percentage is not None) and (transcribed_cancer_types_percentage is None):
                        radius = calculate_radius(untranscribed_cancer_types_percentage)
                        if (radius > 0):
                            print('Plot circle at x=%d y=%d for %s %s' % (percentage_diff_index, row_signature_index, row_signature,percentage_string))
                            circle = plt.Circle((percentage_diff_index + 0.5, row_signature_index + 0.5), radius, color='yellowgreen', fill=True)
                            ax.add_artist(circle)
                    elif (transcribed_cancer_types_percentage is not None) and (untranscribed_cancer_types_percentage is not None):
                        radius_transcribed = calculate_radius(transcribed_cancer_types_percentage)
                        radius_untranscribed = calculate_radius(untranscribed_cancer_types_percentage)
                        if (radius_transcribed>radius_untranscribed):
                            #First transcribed
                            circle = plt.Circle((percentage_diff_index + 0.5, row_signature_index + 0.5), radius_transcribed, color='royalblue', fill=True)
                            ax.add_artist(circle)
                            #Second untranscribed
                            circle = plt.Circle((percentage_diff_index + 0.5, row_signature_index + 0.5), radius_untranscribed, color='yellowgreen', fill=True)
                            ax.add_artist(circle)
                        else:
                            #First untranscribed
                            circle = plt.Circle((percentage_diff_index + 0.5, row_signature_index + 0.5), radius_untranscribed, color='yellowgreen', fill=True)
                            ax.add_artist(circle)
                            #Second transcribed
                            circle = plt.Circle((percentage_diff_index + 0.5, row_signature_index + 0.5), radius_transcribed, color='royalblue', fill=True)
                            ax.add_artist(circle)

            elif (strand_bias == GENIC_VERSUS_INTERGENIC):
                if row_signature in type2strand2percent2cancertypeslist_dict:
                    genic_cancer_types_percentage=None
                    intergenic_cancer_types_percentage=None
                    if GENIC in type2strand2percent2cancertypeslist_dict[row_signature]:
                        cancer_types_list=type2strand2percent2cancertypeslist_dict[row_signature][GENIC][percentage_string]
                        all_cancer_types_list = signature2cancer_type_list_dict[row_signature]
                        if tissue_based and (tissue_based in cancer_types_list):
                            genic_cancer_types_percentage = 100
                        elif not tissue_based:
                            genic_cancer_types_percentage = (len(cancer_types_list) / len(all_cancer_types_list)) * 100
                    if INTERGENIC in type2strand2percent2cancertypeslist_dict[row_signature]:
                        cancer_types_list=type2strand2percent2cancertypeslist_dict[row_signature][INTERGENIC][percentage_string]
                        all_cancer_types_list = signature2cancer_type_list_dict[row_signature]
                        if tissue_based and (tissue_based in cancer_types_list):
                            intergenic_cancer_types_percentage = 100
                        elif not tissue_based:
                            intergenic_cancer_types_percentage = (len(cancer_types_list) / len(all_cancer_types_list)) * 100

                    if (genic_cancer_types_percentage is not None) and (intergenic_cancer_types_percentage is None):
                        radius = calculate_radius(genic_cancer_types_percentage)
                        if (radius > 0):
                            print('Plot circle at x=%d y=%d for %s %s' % (percentage_diff_index, row_signature_index, row_signature,percentage_string))
                            circle = plt.Circle((percentage_diff_index + 0.5, row_signature_index + 0.5), radius, color='cyan', fill=True)
                            ax.add_artist(circle)
                    elif (intergenic_cancer_types_percentage is not None) and (genic_cancer_types_percentage is None):
                        radius = calculate_radius(intergenic_cancer_types_percentage)
                        if (radius > 0):
                            print('Plot circle at x=%d y=%d for %s %s' % (percentage_diff_index, row_signature_index, row_signature,percentage_string))
                            circle = plt.Circle((percentage_diff_index + 0.5, row_signature_index + 0.5), radius, color='gray', fill=True)
                            ax.add_artist(circle)
                    elif (genic_cancer_types_percentage is not None) and (intergenic_cancer_types_percentage is not None):
                        radius_genic = calculate_radius(genic_cancer_types_percentage)
                        radius_intergenic = calculate_radius(intergenic_cancer_types_percentage)
                        if (radius_genic>radius_intergenic):
                            #First genic
                            circle = plt.Circle((percentage_diff_index + 0.5, row_signature_index + 0.5), radius_genic, color='cyan', fill=True)
                            ax.add_artist(circle)
                            #Second intergenic
                            circle = plt.Circle((percentage_diff_index + 0.5, row_signature_index + 0.5), radius_intergenic, color='gray', fill=True)
                            ax.add_artist(circle)
                        else:
                            #First untranscribed
                            circle = plt.Circle((percentage_diff_index + 0.5, row_signature_index + 0.5), radius_intergenic, color='gray', fill=True)
                            ax.add_artist(circle)
                            #Second transcribed
                            circle = plt.Circle((percentage_diff_index + 0.5, row_signature_index + 0.5), radius_genic, color='cyan', fill=True)
                            ax.add_artist(circle)

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
    ax.set_ylim([0,len(rows_signatures)])
    ax.set_yticklabels([])
    ax.tick_params(axis='y', which='both', length=0, labelsize=y_ticks_labelsize)

    #major ticks
    ax.set_yticks(np.arange(0, len(rows_signatures), 1))
    #minor ticks
    ax.set_yticks(np.arange(0, len(rows_signatures), 1)+0.5,minor=True)

    yticks = np.arange(0,len(rows_signatures_with_number_of_cancer_types))
    ax.set_yticks(yticks)
    if figure_type == COSMIC:
        ax.set_yticklabels(rows_signatures_with_number_of_cancer_types, minor=True, fontweight='bold', fontname='Times New Roman', fontsize=y_ticks_labelsize)  # fontsize
    elif figure_type == MANUSCRIPT:
        ax.set_yticklabels(rows_signatures_with_number_of_cancer_types, minor=True, fontsize=y_ticks_labelsize)  # fontsize

    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False)  # labels along the bottom edge are off

    # Put the legend
    if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        legend_elements = [
            Line2D([0], [0], marker='o', color='white', label='Genic: Transcribed Strand', markerfacecolor='royalblue', markersize=40),
            Line2D([0], [0], marker='o', color='white', label='Genic: Untranscribed Strand', markerfacecolor='yellowgreen', markersize=40)]
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        legend_elements = [
            Line2D([0], [0], marker='o', color='white', label='Genic Regions', markerfacecolor='cyan', markersize=40),
            Line2D([0], [0], marker='o', color='white', label='Intergenic Regions', markerfacecolor='gray', markersize=40)]
    elif (strand_bias == LAGGING_VERSUS_LEADING):
        legend_elements = [
            Line2D([0], [0], marker='o', color='white', label='Lagging Strand', markerfacecolor='indianred', markersize=40),
            Line2D([0], [0], marker='o', color='white', label='Leading Strand', markerfacecolor='goldenrod', markersize=40)]

    if signature_name:
        bottom_right_axis.set_axis_off()
        # bottom_right_axis.legend(handles=legend_elements, ncol=len(legend_elements), bbox_to_anchor=(1, 0.5), loc='upper right',fontsize=40)
        bottom_right_axis.legend(handles=legend_elements, ncol=1, bbox_to_anchor=(1, 0.5), loc='upper right', fontsize=40)

    # Gridlines based on major ticks
    ax.grid(which='major', color='black')

    if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        feature_name = 'TRANSCR_ASYM'
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        feature_name = 'GENIC_ASYM'
    elif strand_bias == LAGGING_VERSUS_LEADING:
        feature_name = 'REPLIC_ASYM'

    # v3.2_SBS1_REPLIC_ASYM_TA_C34447.jpg
    if figure_type == COSMIC:
        if signature_name and tissue_based:
            # v3.2_SBS1_REPLIC_ASYM_TA_C34447.jpg
            NCI_Thesaurus_code = cancer_type_2_NCI_Thesaurus_code_dict[tissue_based]
            filename = '%s_%s_%s_TA_%s.%s' % (cosmic_release_version, signature_name, feature_name, NCI_Thesaurus_code, figure_file_extension)
        elif signature_name:
            filename = '%s_%s_%s.%s' % (cosmic_release_version, signature_name, feature_name, figure_file_extension)
        else:
            filename = '%s_%s_Signatures_%s.%s' % (cosmic_release_version, signature_type, feature_name, figure_file_extension)
    elif figure_type == MANUSCRIPT:
        # for manuscript  signature_name = None and tissue_based = None
        filename = '%s_Signatures_%s_with_circles_%s.png' % (signature_type, strand_bias, str(SIGNIFICANCE_LEVEL).replace('.','_'))

    figFile = os.path.join(strand_bias_figures_output_dir, figure_dir, filename)
    fig.savefig(figFile, dpi=100, bbox_inches="tight")

    plt.cla()
    plt.close(fig)


def augment_with_number_of_cancer_types(signature_type, rows_signatures, signature2cancer_type_list_dict, new_line=False):
    rows_signatures_with_number_of_cancer_types = []

    for signature in rows_signatures:
        if new_line:
            if signature in signature2cancer_type_list_dict:
                num_of_cancer_types = "(n=%d)" %(len(signature2cancer_type_list_dict[signature]))
                signature_with_number_of_cancer_types = signature + '\n' + num_of_cancer_types
            else:
                signature_with_number_of_cancer_types = signature
            # if signature_type == SBS or signature_type == DBS:
            #     signature_with_number_of_cancer_types = f"{signature:<7}\n{num_of_cancer_types:>6}"
            # elif signature_type == ID:
            #     signature_with_number_of_cancer_types = f"{signature:<6}\n{num_of_cancer_types:>6}"
        else:
            if signature in signature2cancer_type_list_dict:
                num_of_cancer_types = "(n=%d)" %(len(signature2cancer_type_list_dict[signature]))
                signature_with_number_of_cancer_types = signature + ' ' + num_of_cancer_types
            else:
                signature_with_number_of_cancer_types = signature
            # if signature_type == SBS or signature_type == DBS:
            #     signature_with_number_of_cancer_types = f"{signature:<7}{num_of_cancer_types:>6}"
            # elif signature_type == ID:
            #     signature_with_number_of_cancer_types = f"{signature:<6}{num_of_cancer_types:>6}"
        rows_signatures_with_number_of_cancer_types.append(signature_with_number_of_cancer_types)
    return rows_signatures_with_number_of_cancer_types


# This function groups w.r.t. signature and mutation type
# Then takes mean across all cancer types having this signature and mutation type
# Combines p-values using Fisher's method
# Followed by p-value correction
def combine_p_values(strand_bias, signature_strand1_versus_strand2_df):
    if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        strand1_real_count = 'Transcribed_real_count'
        strand2_real_count = 'UnTranscribed_real_count'
        strand1_mean_sims_count = 'Transcribed_mean_sims_count'
        strand2_mean_sims_count = 'UnTranscribed_mean_sims_count'
        strand1_versus_strand2_p_value = 'transcribed_versus_untranscribed_p_value'
        strand1 = 'transcribed'
        strand2 = 'untranscribed'
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        strand1_real_count = 'genic_real_count'
        strand2_real_count = 'intergenic_real_count'
        strand1_mean_sims_count = 'genic_mean_sims_count'
        strand2_mean_sims_count = 'intergenic_mean_sims_count'
        strand1_versus_strand2_p_value = 'genic_versus_intergenic_p_value'
        strand1 = 'genic'
        strand2 = 'intergenic'
    elif strand_bias == LAGGING_VERSUS_LEADING:
        strand1_real_count='Lagging_real_count'
        strand2_real_count='Leading_real_count'
        strand1_mean_sims_count='Lagging_mean_sims_count'
        strand2_mean_sims_count='Leading_mean_sims_count'
        strand1_versus_strand2_p_value='lagging_versus_leading_p_value'
        strand1 = 'lagging'
        strand2 = 'leading'

    # new df column names
    strand1_real_count_mean = '%s_real_count_mean' %(strand1)
    strand2_real_count_mean = '%s_real_count_mean' %(strand2)
    strand1_sims_mean_count_mean = '%s_sims_mean_count_mean' %(strand1)
    strand2_sims_mean_count_mean = '%s_sims_mean_count_mean' %(strand2)
    strand1_versus_strand2_p_value_list = '%s_versus_%s_p_value_list' %(strand1,strand2)
    strand1_versus_strand2_combined_p_value = '%s_versus_%s_combined_p_value' %(strand1,strand2)
    strand1_real_count_list = '%s_real_count_list' %(strand1)
    strand2_real_count_list = '%s_real_count_list' %(strand2)
    strand1_sims_mean_count_list = '%s_sims_mean_count_list' %(strand1)
    strand2_sims_mean_count_list = '%s_sims_mean_count_list' %(strand2)

    # initialize new df
    df = pd.DataFrame(columns=['signature',
                               'mutation_type',
                               strand1_real_count_mean,
                               strand2_real_count_mean,
                               strand1_sims_mean_count_mean,
                               strand2_sims_mean_count_mean,
                               strand1_versus_strand2_p_value_list,
                               strand1_versus_strand2_combined_p_value,
                               'q_value',
                               'cancer_type_list',
                               strand1_real_count_list,
                               strand2_real_count_list,
                               strand1_sims_mean_count_list,
                               strand2_sims_mean_count_list
                               ])

    groupby_df = signature_strand1_versus_strand2_df.groupby(['signature','mutation_type'])
    p_value_list = []
    name_list = []
    for name, group_df in groupby_df:
        signature, mutation_type = name
        print(signature,mutation_type)
        strand1_real_count_mean_value = group_df[strand1_real_count].mean()
        strand2_real_count_mean_value = group_df[strand2_real_count].mean()
        strand1_sims_mean_count_mean_value = group_df[strand1_mean_sims_count].mean()
        strand2_sims_mean_count_mean_value = group_df[strand2_mean_sims_count].mean()
        strand1_versus_strand2_p_value_list_value = group_df[strand1_versus_strand2_p_value].values.tolist()
        test_statistic, combined_p_value = scipy.stats.combine_pvalues(strand1_versus_strand2_p_value_list_value, method='fisher', weights=None)
        cancer_type_list=group_df['cancer_type'].values.tolist()
        strand1_real_count_list_value = group_df[strand1_real_count].values.tolist()
        strand2_real_count_list_value = group_df[strand2_real_count].values.tolist()
        strand1_sims_mean_count_list_value = group_df[strand1_mean_sims_count].values.tolist()
        strand2_sims_mean_count_list_value = group_df[strand2_mean_sims_count].values.tolist()
        p_value_list.append(combined_p_value)
        name_list.append(name)
        df = df.append(
            {'signature':signature,
           'mutation_type':mutation_type,
           strand1_real_count_mean:strand1_real_count_mean_value,
           strand2_real_count_mean:strand2_real_count_mean_value,
           strand1_sims_mean_count_mean:strand1_sims_mean_count_mean_value,
           strand2_sims_mean_count_mean:strand2_sims_mean_count_mean_value,
           strand1_versus_strand2_p_value_list:strand1_versus_strand2_p_value_list_value,
           strand1_versus_strand2_combined_p_value:combined_p_value,
           'q_value':np.nan,
           'cancer_type_list':cancer_type_list,
           strand1_real_count_list:strand1_real_count_list_value,
           strand2_real_count_list:strand2_real_count_list_value,
           strand1_sims_mean_count_list:strand1_sims_mean_count_list_value,
           strand2_sims_mean_count_list:strand2_sims_mean_count_list_value}, ignore_index=True)
    # Correct p values
    rejected, all_FDR_BH_adjusted_p_values, alphacSidak, alphacBonf = statsmodels.stats.multitest.multipletests(p_value_list, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
    for name_index, name in enumerate(name_list, 0):
        signature, mutation_type = name
        q_value = all_FDR_BH_adjusted_p_values[name_index]
        df.loc[((df['signature']==signature) & (df['mutation_type']==mutation_type)),'q_value']=q_value
    return df

# Stacked bar plots
def plot_strand_bias_figure_with_stacked_bar_plots(strand_bias,
                                     strandbias_figures_outputDir,
                                     numberofSimulations,
                                     signature,
                                     N,
                                     x_axis_tick_labels,
                                     y_axis_label,
                                     stacked_bar_title,
                                     strand1_values,
                                     strand2_values,
                                     strand1_simulations_median_values,
                                     strand2_simulations_median_values,
                                     mutation_type_display,
                                     fdr_bh_adjusted_pvalues,
                                     significance_level,
                                     strand1Name,
                                     strand2Name,
                                     color1,
                                     color2,
                                     width,
                                     tissue_based,
                                     axis_given=None):

    # Replace np.nans with 0
    strand1_values = [0 if np.isnan(x) else x for x in strand1_values]
    strand2_values = [0 if np.isnan(x) else x for x in strand2_values]
    strand1_simulations_median_values = [0 if np.isnan(x) else x for x in strand1_simulations_median_values]
    strand2_simulations_median_values = [0 if np.isnan(x) else x for x in strand2_simulations_median_values]

    # Fill odds_ratio_list
    odds_real_list = []
    odds_sims_list = []
    for a, b in zip(strand1_values, strand2_values):
        odds_real = np.nan
        if b>0:
            odds_real = a/b
        odds_real_list.append(odds_real)

    for x, y in zip(strand1_simulations_median_values, strand2_simulations_median_values):
        odds_sims = np.nan
        if y>0:
            odds_sims = x/y
        odds_sims_list.append(odds_sims)

    odds_ratio_list = [odds_real/odds_sims if odds_sims>0 else np.nan for (odds_real, odds_sims) in zip(odds_real_list,odds_sims_list)]

    strand1_values = [x if display else 0 for x, display in zip(strand1_values, mutation_type_display)]
    strand2_values = [x if display else 0 for x, display in zip(strand2_values, mutation_type_display)]
    strand1_simulations_median_values = [x if display else 0 for x, display in zip(strand1_simulations_median_values, mutation_type_display)]
    strand2_simulations_median_values = [x if display else 0 for x, display in zip(strand2_simulations_median_values, mutation_type_display)]
    fdr_bh_adjusted_pvalues = [x if display else np.nan for x, display in zip(fdr_bh_adjusted_pvalues, mutation_type_display)]
    odds_ratio_list = [x if display else np.nan for x, display in zip(odds_ratio_list, mutation_type_display)]

    # Here we can take into difference between strand1_values and strand2_values while deciding on significance
    # the x locations for the groups
    ind = np.arange(N)
    if axis_given==None:
        fig, ax = plt.subplots(figsize=(16,10),dpi=100)
    else:
        ax=axis_given

    legend=None
    rects3=None
    rects4=None

    rects1 = ax.bar(ind, strand1_values, width=width, edgecolor='black', color=color1, zorder=1000)
    rects2 = ax.bar(ind, strand2_values, width=width, edgecolor='black', color=color2, bottom=strand1_values, zorder=1000)

    if ((strand1_simulations_median_values is not None) and strand1_simulations_median_values):
        rects3 = ax.bar(ind + width, strand1_simulations_median_values, width=width, edgecolor='black', color=color1, hatch = '///', zorder=1000)
    if ((strand2_simulations_median_values is not None) and strand2_simulations_median_values):
        rects4 = ax.bar(ind + width, strand2_simulations_median_values, width=width, edgecolor='black', color=color2, hatch = '///', bottom=strand1_simulations_median_values, zorder=1000)

    # Add some text for labels, title and axes ticks
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=35)

    ax.set_ylim(0, 1.2)
    ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=35, fontweight='bold', fontname='Arial')

    # To make the bar width not too wide
    if len(ind) < 6:
        maxn = 6
        ax.set_xlim(-0.5, maxn - 0.5)

    # Set title
    # ax.set_title(stacked_bar_title, fontsize=40,fontweight='bold')
    ax.set_title(stacked_bar_title, fontsize=40, fontname = "Times New Roman", weight = 'bold')

    # Set x tick labels
    if len(x_axis_tick_labels) > 6:
        ax.set_xticklabels(x_axis_tick_labels, fontsize=35, rotation=90, fontweight='bold', fontname='Arial')
    else:
        ax.set_xticklabels(x_axis_tick_labels, fontsize=35, fontweight='bold', fontname='Arial')

    # Set the ylabel
    # ax.set_ylabel(y_axis_label, fontsize=35, fontweight='normal')
    ax.set_ylabel(y_axis_label, fontsize=35, fontname = "Times New Roman", weight = 'bold')

    # Horizontal lines at y ticks
    ax.yaxis.grid(True)
    ax.grid(which='major', axis='y', color=[0.6, 0.6, 0.6], zorder=1)

    # Set the x axis tick locations
    if (numberofSimulations > 0):
        ax.set_xticks(ind + (width/2))
        realStrand1Name = 'Real %s' % (strand1Name)
        realStrand2Name = 'Real %s' % (strand2Name)
        simulationsStrand1Name = 'Simulated %s' % (strand1Name)
        simulationsStrand2Name = 'Simulated %s' % (strand2Name)
        # # Let's not have a legend
        # if ((rects1 is not None) and (rects2 is not None) and (rects3 is not None) and (rects4 is not None)):
        #     if ((len(rects1) > 0) and (len(rects2) > 0) and (len(rects3) > 0) and (len(rects4) > 0)):
        #         legend = ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]),
        #                            (realStrand1Name, realStrand2Name, simulationsStrand1Name, simulationsStrand2Name),prop={'size': 25}, ncol=1, loc='best')
    else:
        # Old way with no simulations
        ax.set_xticks(ind + width / 2)
        # if ((rects1 is not None) and (rects2 is not None)):
        #     if ((len(rects1) > 0) and (len(rects2) > 0)):
        #         legend = ax.legend((rects1[0], rects2[0]), (strand1Name, strand2Name), prop={'size': 25}, ncol=1, loc='upper right')

    #To make the barplot background white
    ax.set_facecolor('white')
    #To makes spines black like a rectangle with black stroke
    ax.spines["bottom"].set_color('black')
    ax.spines["left"].set_color('black')
    ax.spines["top"].set_color('black')
    ax.spines["right"].set_color('black')

    if (legend is not None):
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')

    #Add star above the bars for significant differences between the number of mutations on each strand starts
    # For each bar: Place a label
    if odds_ratio_list is not None:
        for odds_ratio, fdr_bh_adjusted_pvalue, rect1, rect2 in zip(odds_ratio_list, fdr_bh_adjusted_pvalues, rects1, rects2):
            # Get X and Y placement of label from rect.
            # y_value = max(rect1.get_height(),rect2.get_height())
            y_value = rect1.get_height() + rect2.get_height()
            x_value = rect1.get_x() + rect1.get_width()

            # Number of points between bar and label. Change to your liking.
            space = 3
            # Vertical alignment for positive values
            va = 'bottom'

            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'

            # Use Y value as label and format number with one decimal place
            label = "{:.1f}".format(y_value)

            # Create annotation
            if not np.isnan(odds_ratio):
                if ((fdr_bh_adjusted_pvalue is not None) and fdr_bh_adjusted_pvalue<=0.0001):
                    ax.annotate(
                        '%.2f ***' %(odds_ratio),  # Use `label` as label
                        (x_value, y_value),  # Place label at end of the bar
                        xytext=(0, space),  # Vertically shift label by `space`
                        textcoords="offset points",  # Interpret `xytext` as offset in points
                        ha='center',  # Horizontally center label
                        va=va,
                        fontsize=25)  # Vertically align label differently for

                elif ((fdr_bh_adjusted_pvalue is not None) and fdr_bh_adjusted_pvalue<=0.001):
                    ax.annotate(
                        '%.2f **' %(odds_ratio),  # Use `label` as label
                        (x_value, y_value),  # Place label at end of the bar
                        xytext=(0, space),  # Vertically shift label by `space`
                        textcoords="offset points",  # Interpret `xytext` as offset in points
                        ha='center',  # Horizontally center label
                        va=va,
                        fontsize=25)  # Vertically align label differently for

                elif ((fdr_bh_adjusted_pvalue is not None) and fdr_bh_adjusted_pvalue<=significance_level):
                    ax.annotate(
                        '%.2f *' %(odds_ratio),  # Use `label` as label
                        (x_value, y_value),  # Place label at end of the bar
                        xytext=(0, space),  # Vertically shift label by `space`
                        textcoords="offset points",  # Interpret `xytext` as offset in points
                        ha='center',  # Horizontally center label
                        va=va,
                        fontsize=25) # Vertically align label differently for
                else:
                    ax.annotate(
                        '%.2f' %(odds_ratio),  # Use `label` as label
                        (x_value, y_value),  # Place label at end of the bar
                        xytext=(0, space),  # Vertically shift label by `space`
                        textcoords="offset points",  # Interpret `xytext` as offset in points
                        ha='center',  # Horizontally center label
                        va=va,
                        fontsize=25) # Vertically align label differently for

    if axis_given==None:
        if tissue_based:
            filename = '%s_%s_%s_with_stacked_bars.png' %(signature,tissue_based,strand_bias)
        else:
            filename = '%s_%s_with_stacked_bars.png' %(signature,strand_bias)
        figFile = os.path.join(strandbias_figures_outputDir, filename)
        fig.savefig(figFile, dpi=100, bbox_inches="tight")

        plt.cla()
        plt.close(fig)


def plot_strand_bias_figure_with_bar_plots(strand_bias,
                                     strandbias_figures_outputDir,
                                     numberofSimulations,
                                     signature,
                                     N,
                                     x_axis_tick_labels,
                                     y_axis_label,
                                     strand1_values,
                                     strand2_values,
                                     strand1_simulations_median_values,
                                     strand2_simulations_median_values,
                                     fdr_bh_adjusted_pvalues,
                                     significance_level,
                                     strand1Name,
                                     strand2Name,
                                     color1,
                                     color2,
                                     width,
                                     tissue_based,
                                     figure_case_study,
                                     axis_given = None):

    max_strand1 = max(strand1_values)
    strand1_true_false = [True if x >= max_strand1 * 0.005 else False for x in strand1_values]
    max_strand2 = max(strand2_values)
    strand2_true_false = [True if x >= max_strand2 * 0.005 else False for x in strand2_values]
    q_values_true_false = [True if x <= significance_level else False for x in fdr_bh_adjusted_pvalues]
    mutation_type_display = [(x or y or z) for x, y, z in zip(strand1_true_false, strand2_true_false, q_values_true_false)]

    # Here we can take into difference between strand1_values and strand2_values while deciding on significance
    # the x locations for the groups
    ind = np.arange(N)
    if axis_given:
        ax = axis_given
    else:
        fig, ax = plt.subplots(figsize=(16,10),dpi=100)

    legend = None
    rects3 = None
    rects4 = None

    rects1 = ax.bar(ind, strand1_values, width=width, edgecolor='black', color=color1, zorder=1000)
    rects2 = ax.bar(ind + width, strand2_values, width=width, edgecolor='black', color=color2, zorder=1000)

    if ((strand1_simulations_median_values is not None) and strand1_simulations_median_values):
        rects3 = ax.bar(ind+ 2*width, strand1_simulations_median_values, width=width, edgecolor='black', color=color1, hatch = '///', zorder=1000)
    if ((strand2_simulations_median_values is not None) and strand2_simulations_median_values):
        rects4 = ax.bar(ind + 3*width, strand2_simulations_median_values, width=width, edgecolor='black', color=color2, hatch = '///', zorder=1000)

    # add some text for labels, title and axes ticks
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=35)

    # To make the bar width not too wide
    if len(ind) < 6:
        maxn = 6
        ax.set_xlim(-0.5, maxn - 0.5)

    # Set title
    # ax.set_title('%s vs. %s' %(strand1Name,strand2Name), fontsize=40, fontweight='bold')
    if figure_case_study:
        ax.set_title(figure_case_study, fontsize=40, fontname = "Times New Roman", weight = 'bold')
    else:
        ax.set_title('%s vs. %s' % (strand1Name, strand2Name), fontsize=40, fontname = "Times New Roman", weight = 'bold')

    # Set x tick labels
    if len(x_axis_tick_labels) > 6:
        ax.set_xticklabels(x_axis_tick_labels, fontsize=35, rotation=90, fontweight='bold', fontname='Arial')
    else:
        ax.set_xticklabels(x_axis_tick_labels, fontsize=35, fontweight='bold', fontname='Arial')

    # Set ylabel
    # ax.set_ylabel(y_axis_label, fontsize=35, fontweight='normal')
    ax.set_ylabel(y_axis_label, fontsize=35, fontname = "Times New Roman", weight = 'bold')

    # old way
    # locs = ax.get_yticks()
    # ax.set_ylim(0, locs[-1] + 5000)

    # new way
    all_y_values = strand1_values + strand2_values + strand1_simulations_median_values + strand2_simulations_median_values
    ymax = max(all_y_values)

    y = ymax/1.025
    ytick_offset = float(y / 5)
    y_ticks = [0, ytick_offset, ytick_offset * 2, ytick_offset * 3, ytick_offset * 4, ytick_offset * 5 ]
    y_tick_labels = [0, ytick_offset, ytick_offset * 2, ytick_offset * 3, ytick_offset * 4, ytick_offset * 5 ]

    y_tick_labels = ['{:,}'.format(int(x)) for x in y_tick_labels]
    if len(y_tick_labels[-1]) > 3:
        ylabels_temp = []
        if len(y_tick_labels[-1]) > 7:
            for label in y_tick_labels:
                if len(label) > 7:
                    ylabels_temp.append(label[0:-8] + "m")
                elif len(label) > 3:
                    ylabels_temp.append(label[0:-4] + "k")
                else:
                    ylabels_temp.append(label)

        else:
            for label in y_tick_labels:
                if len(label) > 3:
                    ylabels_temp.append(label[0:-4] + "k")
                else:
                    ylabels_temp.append(label)
        y_tick_labels = ylabels_temp

    ax.set_ylim([0, ymax*1.2])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=35, fontweight='bold', fontname='Arial')

    # Horizontal lines at y ticks
    ax.yaxis.grid(True)
    ax.grid(which='major', axis='y', color=[0.6, 0.6, 0.6], zorder=1)

    # Set the x axis tick locations
    if (numberofSimulations > 0):
        ax.set_xticks(ind + (3 * width) / 2)
        realStrand1Name = 'Real %s' % (strand1Name)
        realStrand2Name = 'Real %s' % (strand2Name)
        simulationsStrand1Name = 'Simulated %s' % (strand1Name)
        simulationsStrand2Name = 'Simulated %s' % (strand2Name)
        if ((rects1 is not None) and (rects2 is not None) and (rects3 is not None) and (rects4 is not None)):
            if ((len(rects1) > 0) and (len(rects2) > 0) and (len(rects3) > 0) and (len(rects4) > 0)):
                legend = ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]),
                                   (realStrand1Name, realStrand2Name, simulationsStrand1Name, simulationsStrand2Name),prop={'size': 30}, ncol=1, loc='best')
    else:
        # Old way with no simulations
        ax.set_xticks(ind + width / 2)
        if ((rects1 is not None) and (rects2 is not None)):
            if ((len(rects1) > 0) and (len(rects2) > 0)):
                legend = ax.legend((rects1[0], rects2[0]), (strand1Name, strand2Name), prop={'size': 25}, ncol=1, loc='best')

    #To make the barplot background white
    ax.set_facecolor('white')
    #To makes spines black like a rectangle with black stroke
    ax.spines["bottom"].set_color('black')
    ax.spines["left"].set_color('black')
    ax.spines["top"].set_color('black')
    ax.spines["right"].set_color('black')

    if (legend is not None):
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')

    # Add star above the bars for significant differences between the number of mutations on each strand starts
    # For each bar: Place a label
    if fdr_bh_adjusted_pvalues is not None:
        for fdr_bh_adjusted_pvalue, rect1, rect2 in zip(fdr_bh_adjusted_pvalues,rects1,rects2):
            # Get X and Y placement of label from rect.
            y_value = max(rect1.get_height(),rect2.get_height())
            x_value = rect1.get_x() + rect1.get_width()

            # Number of points between bar and label. Change to your liking.
            space = 3
            # Vertical alignment for positive values
            va = 'bottom'

            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'

            # Use Y value as label and format number with one decimal place
            label = "{:.1f}".format(y_value)

            # Create annotation
            if ((fdr_bh_adjusted_pvalue is not None) and fdr_bh_adjusted_pvalue <= 0.0001):
                ax.annotate(
                    '***',  # Use `label` as label
                    (x_value, y_value),  # Place label at end of the bar
                    xytext=(0, space),  # Vertically shift label by `space`
                    textcoords="offset points",  # Interpret `xytext` as offset in points
                    ha='center',  # Horizontally center label
                    va=va,
                    fontsize=25)  # Vertically align label differently for

            elif ((fdr_bh_adjusted_pvalue is not None) and fdr_bh_adjusted_pvalue <= 0.001):
                ax.annotate(
                    '**',  # Use `label` as label
                    (x_value, y_value),  # Place label at end of the bar
                    xytext=(0, space),  # Vertically shift label by `space`
                    textcoords="offset points",  # Interpret `xytext` as offset in points
                    ha='center',  # Horizontally center label
                    va=va,
                    fontsize=25)  # Vertically align label differently for

            elif ((fdr_bh_adjusted_pvalue is not None) and fdr_bh_adjusted_pvalue <= significance_level):
                ax.annotate(
                    '*',  # Use `label` as label
                    (x_value, y_value),  # Place label at end of the bar
                    xytext=(0, space),  # Vertically shift label by `space`
                    textcoords="offset points",  # Interpret `xytext` as offset in points
                    ha='center',  # Horizontally center label
                    va=va,
                    fontsize=25) # Vertically align label differently for

    if axis_given == None:
        if tissue_based:
            filename = '%s_%s_%s_with_bars.png' %(signature,tissue_based,strand_bias)
        else:
            filename = '%s_%s_with_bars.png' %(signature,strand_bias)
        figFile = os.path.join(strandbias_figures_outputDir, filename)
        fig.savefig(figFile, dpi=100, bbox_inches="tight")

        plt.cla()
        plt.close(fig)

    return mutation_type_display

# sigProfilerPlotting plotsBS
# Tissue-based
# Across all cancer types
def plot_real_data_strand_bias_in_given_axis(signature,
                                             df,
                                             percentage,
                                             plot_type,
                                             column_name,
                                             panel1,
                                             strand_bias,
                                             tissue_based=None):

    # box = panel1.get_position()
    # # set title
    # if strand_bias == LAGGING_VERSUS_LEADING:
    #     title = '%s Lagging versus Leading Strand Bias' %(signature)
    # elif strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
    #     title = '%s Transcribed versus Untranscribed Strand Bias' %(signature)
    # elif strand_bias == GENIC_VERSUS_INTERGENIC:
    #     title = '%s Genic versus Intergenic Strand Bias' %(signature)
    # panel1.text(6 * 3, 1 + 2.5, title, horizontalalignment='center', fontsize=60, fontweight='bold', fontname='Arial', transform=panel1.transAxes)
    # panel1.text(box.width/2, box.height, title, horizontalalignment='center', fontsize=60, fontweight='bold', fontname='Arial', transform=panel1.transAxes)

    total_count = 0
    sig_probs = False

    if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        color1 = 'royalblue'
        color2 = 'yellowgreen'
        # strand1 transcribed  T
        # strand2 untranscribed U
        strand1_letters = ['T']
        strand2_letters = ['U']
        strand1_label = 'Genic: Transcribed Strand'
        strand2_label = 'Genic: Untranscribed Strand'
        text = 'transcribed'
    elif strand_bias == LAGGING_VERSUS_LEADING:
        color1 = 'indianred'
        color2 = 'goldenrod'
        strand1_letters = ['A']
        strand2_letters = ['E']
        strand1_label = 'Lagging Strand'
        strand2_label = 'Leading Strand'
        text = 'transcribed'
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        color1 = 'cyan'
        color2 = 'gray'
        # strand1 genic  T U
        # strand2 intergenic  N
        strand1_letters = ['T', 'U']
        strand2_letters = ['N']
        strand1_label = 'Genic Regions'
        strand2_label = 'Intergenic Regions'
        text = 'genic'

    if plot_type == '192' or plot_type == '96SB' or plot_type == '384':
        x = 0.7
        ymax = 0
        colors = [[3 / 256, 189 / 256, 239 / 256],
                  [1 / 256, 1 / 256, 1 / 256],
                  [228 / 256, 41 / 256, 38 / 256],
                  [203 / 256, 202 / 256, 202 / 256],
                  [162 / 256, 207 / 256, 99 / 256],
                  [236 / 256, 199 / 256, 197 / 256]]

        xsublabels = ['C>A'] * 16 + ['C>G'] * 16 + ['C>T'] * 16 + ['T>A'] * 16 + ['T>C'] * 16 + ['T>G'] * 16

        xlabels = []
        nucleotides = ['A', 'C', 'G', 'T']
        xlabels.extend([nucleotide_left + 'C' + nucleotide_right for nucleotide_left in nucleotides for nucleotide_right in nucleotides] * 3)
        xlabels.extend([nucleotide_left + 'T' + nucleotide_right for nucleotide_left in nucleotides for nucleotide_right in nucleotides] * 3)

        # First Pass
        # Get the total_count at the start
        for i, xlabel in enumerate(xlabels):
            xsublabel = xsublabels[i]
            # E:G[C>A]T 60 # Leading
            # A:G[C>A]T 74 # Lagging
            for strand_letter in strand1_letters:
                strand1_count = df.loc[((df['MutationType'].str[2] + df['MutationType'].str[4] +
                                         df['MutationType'].str[8]) == xlabel) &
                                       (df['MutationType'].str[0] == strand_letter) &
                                       (df['MutationType'].str[4:7] == xsublabel), column_name].values[0]
                total_count += strand1_count
            for strand_letter in strand2_letters:
                strand2_count = df.loc[((df['MutationType'].str[2] + df['MutationType'].str[4] +
                                         df['MutationType'].str[8]) == xlabel) &
                                       (df['MutationType'].str[0] == strand_letter) &
                                       (df['MutationType'].str[4:7] == xsublabel), column_name].values[0]
                total_count += strand2_count

        # Second pass
        for i, xlabel in enumerate(xlabels):
            xsublabel = xsublabels[i]
            all_strand1_count = 0
            all_strand2_count = 0

            for strand_letter in strand1_letters:
                strand1_count = df.loc[((df['MutationType'].str[2] + df['MutationType'].str[4] +
                                         df['MutationType'].str[8]) == xlabel) &
                                       (df['MutationType'].str[0] == strand_letter) &
                                       (df['MutationType'].str[4:7] == xsublabel), column_name].values[0]
                all_strand1_count += strand1_count
                if strand1_count < 1 and strand1_count > 0:
                    sig_probs = True

            if percentage:
                if total_count>0:
                    if all_strand1_count / total_count * 100 > ymax:
                        ymax = all_strand1_count / total_count * 100
            else:
                if all_strand1_count > ymax:
                    ymax = all_strand1_count

            for strand_letter in strand2_letters:
                strand2_count = df.loc[((df['MutationType'].str[2] + df['MutationType'].str[4] +
                                         df['MutationType'].str[8]) == xlabel) &
                                       (df['MutationType'].str[0] == strand_letter) &
                                       (df['MutationType'].str[4:7] == xsublabel), column_name].values[0]
                all_strand2_count += strand2_count
                if strand2_count < 1 and strand2_count > 0:
                    sig_probs = True

            if percentage:
                if total_count>0 :
                    if all_strand2_count / total_count * 100 > ymax:
                        ymax = all_strand2_count / total_count * 100
            else:
                if all_strand2_count > ymax:
                    ymax = all_strand2_count

            if percentage:
                if total_count > 0:
                    strand1 = panel1.bar(x, all_strand1_count / total_count * 100, width=0.75, color=color1, align='center', zorder=1000, label=strand1_label)
                    x += 0.75
                    strand2 = panel1.bar(x, all_strand2_count / total_count * 100, width=0.75, color=color2, align='center', zorder=1000, label=strand2_label)
                    x += .2475
            else:
                strand1 = panel1.bar(x, all_strand1_count, width=0.75, color=color1, align='center', zorder=1000, label=strand1_label)
                x += 0.75
                strand2 = panel1.bar(x, all_strand2_count, width=0.75, color=color2, align='center', zorder=1000, label=strand2_label)
                x += .2475

            x += 1

        x = .0015 # .0415 - 0.04
        y3 = .87
        y = int(ymax * 1.25)
        x_plot = 0

        yText = y3 + .06 + 0.17

        panel1.text(.065, yText, 'C>A', fontsize=55, fontweight='bold', fontname='Arial', transform=panel1.transAxes)
        panel1.text(.230, yText, 'C>G', fontsize=55, fontweight='bold', fontname='Arial', transform=panel1.transAxes) # diff 0.165
        panel1.text(.395, yText, 'C>T', fontsize=55, fontweight='bold', fontname='Arial', transform=panel1.transAxes)  # diff 0.165
        panel1.text(.560, yText, 'T>A', fontsize=55, fontweight='bold', fontname='Arial', transform=panel1.transAxes) # diff 0.165
        panel1.text(.725, yText, 'T>C', fontsize=55, fontweight='bold', fontname='Arial', transform=panel1.transAxes) # diff 0.165
        panel1.text(.890, yText, 'T>G', fontsize=55, fontweight='bold', fontname='Arial', transform=panel1.transAxes) # diff 0.165

        if y <= 4:
            y += 4

        while y % 4 != 0:
            y += 1

        y = ymax / 1.025

        ytick_offest = float(y / 3)
        for i in range(0, 6, 1):
            panel1.add_patch(plt.Rectangle((x, y3 + 0.15), .164, .06, facecolor=colors[i], clip_on=False, transform=panel1.transAxes))
            panel1.add_patch(plt.Rectangle((x_plot, 0), 32, round(ytick_offest * 4, 1), facecolor=colors[i], zorder=0, alpha=0.25, edgecolor='grey')) # no transform here
            x += 0.1670
            x_plot += 32

        if percentage:
            ylabs = [0, round(ytick_offest, 1), round(ytick_offest * 2, 1), round(ytick_offest * 3, 1),
                     round(ytick_offest * 4, 1)]
            ylabels = [str(0), str(round(ytick_offest, 1)) + "%", str(round(ytick_offest * 2, 1)) + "%",
                       str(round(ytick_offest * 3, 1)) + "%", str(round(ytick_offest * 4, 1)) + "%"]
        else:
            ylabs = [0, ytick_offest, ytick_offest * 2, ytick_offest * 3, ytick_offest * 4]
            ylabels = [0, ytick_offest, ytick_offest * 2, ytick_offest * 3, ytick_offest * 4]

        labs = np.arange(0.750, 192.750, 1)

        font_label_size = 30
        if not percentage:
            if int(ylabels[3]) >= 1000:
                font_label_size = 20

        if percentage:
            if len(ylabels) > 2:
                font_label_size = 20

        if not percentage:
            ylabels = ['{:,}'.format(int(x)) for x in ylabels]
            if len(ylabels[-1]) > 3:
                ylabels_temp = []
                if len(ylabels[-1]) > 7:
                    for label in ylabels:
                        if len(label) > 7:
                            ylabels_temp.append(label[0:-8] + "m")
                        elif len(label) > 3:
                            ylabels_temp.append(label[0:-4] + "k")
                        else:
                            ylabels_temp.append(label)

                else:
                    for label in ylabels:
                        if len(label) > 3:
                            ylabels_temp.append(label[0:-4] + "k")
                        else:
                            ylabels_temp.append(label)
                ylabels = ylabels_temp

        panel1.set_xlim([0, 96])
        panel1.set_ylim([0, y])
        panel1.set_xticks(labs)
        panel1.set_yticks(ylabs)
        count = 0
        m = 0
        for i in range(0, 96, 1):
            panel1.text(i / 96 + .0015, -0.095, xlabels[i][0], fontsize=30, color='gray', rotation='vertical', verticalalignment='center', fontname='Courier New', transform=panel1.transAxes)
            panel1.text(i / 96 + .0015, -0.060, xlabels[i][1], fontsize=30, color=colors[m], rotation='vertical', verticalalignment='center', fontname='Courier New', fontweight='bold', transform=panel1.transAxes)
            panel1.text(i / 96 + .0015, -0.025, xlabels[i][2], fontsize=30, color='gray', rotation='vertical', verticalalignment='center', fontname='Courier New', transform=panel1.transAxes)
            count += 1
            if count == 16:
                count = 0
                m += 1

        if tissue_based:
            panel1_text = signature + ' ' + tissue_based
        else:
            panel1_text = signature

        # if sig_probs: # To my understanding sig_probs is true when percentage = True
        if percentage:
            panel1.text(0.0015, 0.90, panel1_text, fontsize=60, weight='bold', color='black', fontname="Arial", transform=panel1.transAxes)

        else:
            panel1.text(0.0015, 0.90, panel1_text + ": " + "{:,}".format(int(total_count)) + " " + text + " subs", fontsize=60, weight='bold', color='black', fontname="Arial", transform=panel1.transAxes) #plt

        panel1.set_yticklabels(ylabels, fontsize = 35, fontweight='bold', fontname='Arial')
        panel1.yaxis.grid(True)
        panel1.grid(which='major', axis='y', color=[0.6, 0.6, 0.6], zorder=1)  # plt

        panel1.set_xlabel('')
        panel1.set_ylabel('')
        panel1.legend(handles=[strand1, strand2], prop={'size': 30}, loc='upper right') #plt
        if percentage:
            panel1.set_ylabel("Percentage of Single Base Substitutions", fontsize=35, fontname="Times New Roman", weight='bold')
        else:
            panel1.set_ylabel("Number of Single Base Substitutions", fontsize=35, fontname="Times New Roman", weight='bold')

        panel1.tick_params(axis='both', which='both', \
                           bottom=False, labelbottom=False, \
                           left=False, labelleft=True, \
                           right=False, labelright=False, \
                           top=False, labeltop=False, \
                           direction='in', length=25, colors=[0.6, 0.6, 0.6])

        [i.set_color("black") for i in panel1.get_yticklabels()]

# if tissue_based == None Across all cancer types figure
# if tissue_based != None Tissue-based figure
def prepare_df_and_plot_real_data_in_given_axis(real_data_plot_axis,
                                 signature,
                                 signature2cancer_type_list_dict,
                                 cancer_type2source_cancer_type_tuples_dict,
                                 percentage,
                                 plot_type,
                                 strand_bias,
                                 tissue_based=None):

    # Return all_df
    all_df = pd.DataFrame()

    if tissue_based:
        cancer_type_list = [tissue_based]
    else:
        cancer_type_list = signature2cancer_type_list_dict[signature]

    # What the computations are different for transcription strand bias and replication strand bias?
    # In replication strand bias analysis, SigProfilerTopography provides number of mutations on lagging and leading strands for 96 mutation context.
    # In transcription strand bias analysis, SigProfilerTopography provides number of mutations on transcribed and untranscribed strands only for 6 mutational context.
    # Therefore to get the number of mutations on each transcriptional strand for 96 mutation context, matrix generator and many more former files are used.
    # When SigProfilerTopography provides number of mutations on transcriptional strands for 96 mutational context
    # Then we can get rid of going back to matrix generator, probabilities and cutoff files and compute in the similar way of replicational strand bias.
    if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED or strand_bias == GENIC_VERSUS_INTERGENIC:
        # Prepare df
        all_df_list = []
        for global_cancer_type in cancer_type_list:
            for source, cancer_type in cancer_type2source_cancer_type_tuples_dict[global_cancer_type]:
                print(source, cancer_type)
                # For PCAWG 'Head-SCC'
                # Comes from SigProfilerMatrixGenerator
                # sbs384_matrix_file = os.path.join('/restricted/alexandrov-group/burcak/data/PCAWG/Head-SCC/filtered/output/SBS/Head-SCC.SBS384.all')
                # Paper provides probabilities
                # probabilities_file = os.path.join('/home/burcak/developer/SigProfilerTopographyRuns/PCAWG/probabilities/Head-SCC_sbs96_mutation_probabilities.txt')
                # Comes from SigProfilerTopography
                # topography_cutoffs_file = os.path.join('/restricted/alexandrov-group/burcak/SigProfilerTopographyRuns/PCAWG/Head-SCC/data/Table_SBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt')

                # The first two conditions are written for Lymphoid samples
                if (cancer_type == LYMPH_BNHL or cancer_type == LYMPH_CLL) and (signature == 'SBS37' or signature == 'SBS84' or  signature == 'SBS85'):
                    sbs384_matrix_file = os.path.join('/home/burcak/developer/SigProfilerTopographyRuns/PCAWG_nonPCAWG_lymphomas/SBS384_Files/%s_clustered.SBS384.all' %(cancer_type))
                    probabilities_file = os.path.join('/home/burcak/developer/SigProfilerTopographyRuns/PCAWG_nonPCAWG_lymphomas/probabilities/Clustered_Probabilities.txt')
                    matrix_mutation_type_column = 'MutationType'
                    matrix_mutation_type_short_column = 'MutationTypeShort'
                    prob_sample_column = 'Sample Names'
                    prob_mutation_type_column = 'MutationTypes'
                    topography_cutoffs_file = os.path.join('/restricted/alexandrov-group/burcak/SigProfilerTopographyRuns/PCAWG_nonPCAWG_lymphomas' + '/' + cancer_type + '_clustered/data/Table_SBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt')

                elif (cancer_type == LYMPH_BNHL or cancer_type == LYMPH_CLL):
                    # get files from nonClustered
                    sbs384_matrix_file = os.path.join('/home/burcak/developer/SigProfilerTopographyRuns/PCAWG_nonPCAWG_lymphomas/SBS384_Files/%s_nonClustered.SBS384.all' %(cancer_type))
                    probabilities_file = os.path.join('/home/burcak/developer/SigProfilerTopographyRuns/PCAWG_nonPCAWG_lymphomas/probabilities/NonClustered_Probabilities.txt')
                    matrix_mutation_type_column = 'MutationType'
                    matrix_mutation_type_short_column = 'MutationTypeShort'
                    prob_sample_column = 'Sample Names'
                    prob_mutation_type_column = 'MutationTypes'
                    topography_cutoffs_file = os.path.join('/restricted/alexandrov-group/burcak/SigProfilerTopographyRuns/PCAWG_nonPCAWG_lymphomas' + '/' + cancer_type + '_nonClustered/data/Table_SBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt')

                elif source == PCAWG:
                    sbs384_matrix_file = os.path.join('/restricted/alexandrov-group/burcak/data/' + source + '/' + cancer_type + '/filtered/output/SBS/' + cancer_type + '.SBS384.all')
                    probabilities_file = os.path.join('/home/burcak/developer/SigProfilerTopographyRuns/' + source + '/probabilities/' + cancer_type + '_sbs96_mutation_probabilities.txt')
                    matrix_mutation_type_column = 'MutationType'
                    matrix_mutation_type_short_column = 'MutationTypeShort'
                    prob_sample_column = 'Sample Names'
                    prob_mutation_type_column = 'MutationTypes'
                    topography_cutoffs_file = os.path.join('/restricted/alexandrov-group/burcak/SigProfilerTopographyRuns/Combined_PCAWG_nonPCAWG_4th_iteration' + '/' + global_cancer_type + '/data/Table_SBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt')

                elif source == nonPCAWG:
                    sbs384_matrix_file = os.path.join('/restricted/alexandrov-group/burcak/data/' + source + '/' + cancer_type + '/output/SBS/' + cancer_type + '.SBS384.all')
                    if cancer_type == 'CNS-Glioma-NOS':
                        probabilities_file = os.path.join('/home/burcak/developer/SigProfilerTopographyRuns/' + source + '/probabilities/' + 'CNS-glioma-NOS_subs_probabilities.txt')
                    else:
                        probabilities_file = os.path.join('/home/burcak/developer/SigProfilerTopographyRuns/' + source + '/probabilities/' + cancer_type + '_subs_probabilities.txt')
                    matrix_mutation_type_column = 'MutationType'
                    matrix_mutation_type_short_column = 'MutationTypeShort'
                    prob_sample_column = 'Sample'
                    prob_mutation_type_column = 'Mutation'
                    topography_cutoffs_file = os.path.join('/restricted/alexandrov-group/burcak/SigProfilerTopographyRuns/Combined_PCAWG_nonPCAWG_4th_iteration' + '/' + global_cancer_type + '/data/Table_SBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt')

                elif source == MUTOGRAPHS:
                    sbs384_matrix_file = os.path.join('/restricted',  'alexandrov-group',  'burcak', 'data', 'Mutographs_ESCC_552', 'all_samples', 'output',  'SBS', 'All_Samples_552.SBS384.all')
                    probabilities_file = os.path.join('/home','burcak','developer','SigProfilerTopographyRuns','Mutographs_ESCC_552','manuscript_probabilities','SBS288_Decomposed_Mutation_Probabilities.txt')
                    matrix_mutation_type_column = 'MutationType'
                    matrix_mutation_type_short_column = 'MutationTypeShort'
                    prob_sample_column = 'Sample Names'
                    prob_mutation_type_column = 'MutationTypes'
                    topography_cutoffs_file = os.path.join('/restricted/alexandrov-group/burcak/SigProfilerTopographyRuns/Combined_PCAWG_nonPCAWG_4th_iteration' + '/' + global_cancer_type + '/data/Table_SBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt')

                matrix_df = pd.read_csv(sbs384_matrix_file, sep='\t')
                probabilities_df = pd.read_csv(probabilities_file, sep='\t')
                topography_cutoffs_df = pd.read_csv(topography_cutoffs_file, sep='\t')

                print(signature, ' ', source, ' ', cancer_type, sbs384_matrix_file)
                print(signature, ' ', source, ' ', cancer_type, probabilities_file)
                print(signature, ' Combined_PCAWG_nonPCAWG ', global_cancer_type, topography_cutoffs_file)

                matrix_samples = matrix_df.columns.values[1:]
                print(signature, ' ', source, ' ', cancer_type, ' matrix_df.columns.values.size: ', matrix_df.columns.values.size)
                print(signature, ' ', source, ' ', cancer_type, ' probabilities_df[Samples].unique().size: ', probabilities_df[prob_sample_column].unique().size)
                print(signature, ' ', 'Set difference matrix versus probabilities: ', np.setdiff1d(matrix_df.columns.values, probabilities_df[prob_sample_column].unique()))
                print(signature, ' ', 'Set difference probabilities versus matrix: ', np.setdiff1d(probabilities_df[prob_sample_column].unique(), matrix_df.columns.values))

                if np.any(topography_cutoffs_df[topography_cutoffs_df['signature'] == signature]['cutoff'].values):
                    cutoff = topography_cutoffs_df[topography_cutoffs_df['signature'] == signature]['cutoff'].values[0]
                else:
                    print('Combined_PCAWG_nonPCAWG', global_cancer_type, signature, " No cutoff  is available")

                matrix_df[matrix_mutation_type_short_column] = matrix_df[matrix_mutation_type_column].str[2:]

                df_list = []
                for matrix_sample in matrix_samples:
                    if source == PCAWG:
                        if '_' in matrix_sample:
                            prob_sample = cancer_type + '_' + matrix_sample.split('_')[1]
                        else:
                            # For Lymph-BNHL Lymph-CLL Clustered nonClustered Normal
                            prob_sample = matrix_sample
                    else:
                        prob_sample = matrix_sample
                    # sub_prob_df (96, 2) columns [prob_mutation_type_column, signature]
                    sub_prob_df = probabilities_df[(probabilities_df[prob_sample_column] == prob_sample)][[prob_mutation_type_column, signature]]
                    if sub_prob_df.shape[0] > 0:
                        if sub_prob_df.shape[0] == 192:
                            # For Lymph-BNHL or Lymph-CLL Clustered
                            # Same sample can be in kataegis and omikli probabilities files
                            # There can be 96 x 2 = 192 rows in Clustered (kataegis + omikli) probabilities
                            if (sub_prob_df.iloc[0:96:,1].sum(axis=0) > 0 and sub_prob_df.iloc[96:,1].sum(axis=0) > 0):
                                print('Information', signature, 'source:', source, 'cancer_type', cancer_type,
                                      'matrix_sample:', matrix_sample, 'prob_sample:', prob_sample,
                                      'sub_prob_df.iloc[0:96:,1].sum(axis=0):', sub_prob_df.iloc[0:96:,1].sum(axis=0),
                                      'sub_prob_df.iloc[96:,1].sum(axis=0):', sub_prob_df.iloc[96:,1].sum(axis=0))
                                if (sub_prob_df.iloc[0:96:,1].sum(axis=0) >= sub_prob_df.iloc[96:,1].sum(axis=0)):
                                    sub_prob_df = sub_prob_df.iloc[0:96, :]
                                else:
                                    sub_prob_df = sub_prob_df.iloc[96:,:]
                            elif sub_prob_df.iloc[0:96,1].sum(axis=0) > 0:
                                sub_prob_df = sub_prob_df.iloc[0:96,:]
                            elif sub_prob_df.iloc[96:,1].sum(axis=0) > 0:
                                sub_prob_df = sub_prob_df.iloc[96:,:]
                            else:
                                # both sum are zero
                                sub_prob_df = sub_prob_df.iloc[0:96,:]
                        print('##################################################################')
                        print(signature, 'source:', source, 'cancer_type', cancer_type,  'matrix_sample:', matrix_sample, 'prob_sample:', prob_sample, 'cutoff:', cutoff, 'sub_prob_df.shape:', sub_prob_df.shape, 'sub_prob_df[signature].sum(): ', sub_prob_df[signature].sum())
                        # sub_matrix_df[384 rows x 3 columns]  [mutation_type, mutation_type_short, matrix_sample]
                        sub_matrix_df = matrix_df[[matrix_mutation_type_column, matrix_mutation_type_short_column, matrix_sample]]
                        merged_df = pd.merge(sub_matrix_df, sub_prob_df, how='inner', left_on=matrix_mutation_type_short_column, right_on=prob_mutation_type_column)
                        merged_df.loc[(merged_df[signature] < cutoff), prob_sample] = 0
                        merged_df.loc[(merged_df[signature] >= cutoff), prob_sample] = merged_df[matrix_sample]
                        merged_df = merged_df[[matrix_mutation_type_column, matrix_mutation_type_short_column, prob_sample]]
                        merged_df[prob_sample] = merged_df[prob_sample].astype(np.int32)
                        # print(signature, ' ', matrix_sample, ' ', prob_sample, ' ', cutoff, ' ', sub_matrix_df.shape, '\nsub_matrix_df: ', sub_matrix_df)
                        # print(signature, ' ', matrix_sample, ' ', prob_sample, ' ', cutoff, ' ', sub_prob_df.shape  , '\nsub_prob_df: ', sub_prob_df)
                        # print(signature, ' ', matrix_sample, ' ', prob_sample, ' ', cutoff, ' ', merged_df.shape, '\nmerged_df: ', merged_df)
                        print(signature, matrix_sample, prob_sample, cutoff, 'merged_df[prob_sample].sum():', merged_df[prob_sample].sum(), 'merged_df.shape:', merged_df.shape, 'merged_df.columns.values:', merged_df.columns.values)
                        print('merged_df:', merged_df)
                        assert merged_df.shape[0] == 384, 'merged_df.shape: ' + merged_df.shape + \
                                                          ' sub_matrix_df.shape: ' + sub_matrix_df.shape + \
                                                          ' sub_prob_df.shape: ' + sub_prob_df.shape +\
                                                          ' probabilities_file: ' + probabilities_file
                        print('##################################################################')
                        # file_name = 'sub_matrix_df' + '_' + signature + '_' + source + '_' + cancer_type + ".txt"
                        # file_path = os.path.join(output_path, file_name)
                        # sub_matrix_df.to_csv(file_path, sep='\t', index=False, header=True)
                        # file_name = 'sub_prob_df' + '_' + signature + '_' + source + '_' + cancer_type + ".txt"
                        # file_path = os.path.join(output_path, file_name)
                        # sub_prob_df.to_csv(file_path, sep='\t', index=False, header=True)
                        # file_name = 'merged_df' + '_' + signature + '_' + source + '_' + cancer_type + ".txt"
                        # file_path = os.path.join(output_path, file_name)
                        # merged_df.to_csv(file_path, sep='\t', index=False, header=True)
                        if not merged_df.empty:
                            df_list.append(merged_df)


                print(signature, ' ', 'len(df_list): ', len(df_list))
                for df in df_list:
                    assert df.shape[0] == 384 ,  df.shape + ' ' + df.columns.values
                    df.set_index([matrix_mutation_type_column], inplace=True)
                df = pd.concat(df_list, axis=1)  # join='inner'
                df.reset_index(inplace=True)

                # Gives Memory error
                # df = reduce(lambda x, y: pd.merge(x, y, how='inner', left_on=matrix_mutation_type_column, right_on=matrix_mutation_type_column), df_list)

                print('df.shape:', df.shape)
                column_name = source + '_' + cancer_type + '_Samples'
                df[column_name] = df.sum(axis=1) # sum columns -> row based
                print(signature, 'Before 2 columns selection df: ', signature, source, cancer_type, 'df.shape:', df.shape, 'df.columns.values.size:', df.columns.values.size, 'df.columns.values:', df.columns.values)
                # Drop all columns except the first and the last one
                # df.drop((df.columns.values[1:-1]), axis=1, inplace=True)
                df = df[[matrix_mutation_type_column, column_name]]
                print(signature, ' ', 'After 2 columns selections df: ', signature, ' ', source, ' ', cancer_type, ' ', df.columns.values.size, ' ', df.columns.values)
                print(signature, ' ', 'df', df)
                # file_name = signature + '_' + source + '_' + cancer_type + ".txt"
                # file_path = os.path.join(output_path,file_name)
                # df.to_csv(file_path, sep='\t', index=False, header=True)
                # Decision: I do not add if all zeros. e.g.:  nonPCAWG Liver-HCC
                number_of_mutations_on_strands = df[column_name].sum()
                if number_of_mutations_on_strands > 0:
                    all_df_list.append(df)
                    # plotSBS(signature + '_' + source + '_' + cancer_type, df, percentage, plot_type, column_name)

        # Across all cancer types
        print(signature, cancer_type_list, 'len(all_df_list):', len(all_df_list))
        all_df = reduce(lambda x, y: pd.merge(x, y, on=matrix_mutation_type_column), all_df_list)
        # file_name = signature + ".txt"
        # file_path = os.path.join(output_path, file_name)
        # all_df.to_csv(file_path, sep='\t', index=False, header=True)

        column_name = 'Across_All_Cancer_Types'
        all_df[column_name] = all_df.mean(axis=1) # row-wise
        all_df.drop((all_df.columns.values[1:-1]), axis=1, inplace=True)
        # No need to write
        # all_df.to_csv(file_path, sep='\t', index=False, header=True)
    elif strand_bias == LAGGING_VERSUS_LEADING:
        df_list = []
        for cancer_type in cancer_type_list:
            if (cancer_type == LYMPH_BNHL or cancer_type == LYMPH_CLL) and (signature == 'SBS37' or signature == 'SBS84' or signature == 'SBS85'):
                # get from clustered
                df = pd.read_csv(os.path.join('/restricted/alexandrov-group/burcak/SigProfilerTopographyRuns/PCAWG_nonPCAWG_lymphomas/' + cancer_type + '_clustered' + '/data/replication_strand_bias/' + signature + '_replication_strand_bias_real_data.txt'), sep='\t', header=0)
            elif (cancer_type == LYMPH_BNHL or cancer_type == LYMPH_CLL):
                # get from nonClustered
                df = pd.read_csv(os.path.join('/restricted/alexandrov-group/burcak/SigProfilerTopographyRuns/PCAWG_nonPCAWG_lymphomas/' + cancer_type + '_nonClustered' + '/data/replication_strand_bias/' + signature + '_replication_strand_bias_real_data.txt'), sep='\t', header=0)
            else:
                df = pd.read_csv(os.path.join('/restricted/alexandrov-group/burcak/SigProfilerTopographyRuns/Combined_PCAWG_nonPCAWG_4th_iteration/' + cancer_type + '/data/replication_strand_bias/' + signature + '_replication_strand_bias_real_data.txt'), sep='\t', header=0)
            if not df.empty:
                df_list.append(df)
        merge_column_name = 'MutationType'
        # Across all cancer types
        all_df = reduce(lambda x, y: pd.merge(x, y, on=merge_column_name), df_list)
        column_name = 'Across_All_Cancer_Types'
        all_df[column_name] = all_df.mean(axis=1)
        all_df.drop((all_df.columns.values[1:-1]), axis=1, inplace=True)
        # No need to write
        # all_df.to_csv(file_path, sep='\t', index=False, header=True)

        # outputDir, jobname, DATA, strand_bias, file_name
        # file_name --> "%s_%s_real_data.txt" %(sbs_signature, strand_bias)
        # strand_bias --> replication_strand_bias
        # MutationType\tNumber_of_Mutations
        # A:A[T>A]T 185
        # E:G[T>C]T 337
        # all_df = pd.DataFrame()

    # Use the same method for plotting
    plot_real_data_strand_bias_in_given_axis(signature, all_df, percentage, plot_type, column_name, real_data_plot_axis, strand_bias, tissue_based=tissue_based)
    return all_df


def calculate_radius_add_patch_updated(strand_bias,
                               signature2mutation_type2strand2percent2cancertypeslist_dict,
                               signature2cancer_type_list_dict,
                               mutation_type,
                               mutation_type_index,
                               percentage_strings,
                               percentage_string,
                               percentage_diff_index,
                               row_sbs_signature,
                               row_sbs_signature_index,
                               tissue_based,
                               ax):

    if strand_bias == LAGGING_VERSUS_LEADING:
        strand1 = LAGGING
        strand2 = LEADING
        strand1_color = 'indianred'
        strand2_color = 'goldenrod'

    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        strand1 = GENIC
        strand2 = INTERGENIC
        strand1_color = 'cyan'
        strand2_color = 'gray'

    elif strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        strand1 = TRANSCRIBED_STRAND
        strand2 = UNTRANSCRIBED_STRAND
        strand1_color = 'royalblue'
        strand2_color = 'yellowgreen'

    if row_sbs_signature in signature2mutation_type2strand2percent2cancertypeslist_dict:
        if mutation_type in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature]:
            strand1_cancer_types_percentage = None
            strand2_cancer_types_percentage = None

            if strand1 in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type]:
                cancer_types_list = signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type][strand1][percentage_string]
                all_cancer_types_list = signature2cancer_type_list_dict[row_sbs_signature]
                if tissue_based:
                    if tissue_based in cancer_types_list:
                        strand1_cancer_types_percentage = 100
                    else:
                        strand1_cancer_types_percentage = 0
                else:
                    strand1_cancer_types_percentage = (len(cancer_types_list) / len(all_cancer_types_list)) * 100
            if strand2 in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type]:
                cancer_types_list = signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type][strand2][percentage_string]
                all_cancer_types_list = signature2cancer_type_list_dict[row_sbs_signature]
                if tissue_based:
                    if tissue_based in cancer_types_list:
                        strand2_cancer_types_percentage = 100
                    else:
                        strand2_cancer_types_percentage = 0
                else:
                    strand2_cancer_types_percentage = (len(cancer_types_list) / len(all_cancer_types_list)) * 100

            if (strand1_cancer_types_percentage is not None) and (strand2_cancer_types_percentage is None):
                radius = calculate_radius(strand1_cancer_types_percentage)
                if (radius > 0):
                    ax.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5, row_sbs_signature_index + 0.5), radius, color=strand1_color, fill=True))
            elif (strand2_cancer_types_percentage is not None) and (strand1_cancer_types_percentage is None):
                radius = calculate_radius(strand2_cancer_types_percentage)
                if (radius > 0):
                    ax.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5, row_sbs_signature_index + 0.5), radius, color=strand2_color, fill=True))
            elif (strand1_cancer_types_percentage is not None) and (strand2_cancer_types_percentage is not None):
                radius_strand1 = calculate_radius(strand1_cancer_types_percentage)
                radius_strand2 = calculate_radius(strand2_cancer_types_percentage)
                if (radius_strand1 > radius_strand2):
                    # First strand1
                    ax.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5, row_sbs_signature_index + 0.5), radius_strand1, color=strand1_color, fill=True))
                    # Second strand2
                    ax.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5, row_sbs_signature_index + 0.5), radius_strand2, color=strand2_color, fill=True))
                else:
                    # First strand2
                    ax.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5, row_sbs_signature_index + 0.5), radius_strand2, color=strand2_color, fill=True))
                    # Second strand1
                    ax.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5, row_sbs_signature_index + 0.5), radius_strand1, color=strand1_color, fill=True))


def plot_circles_in_given_axis_for_dbs_id_signatures(ax,
                                       strand_bias,
                                       percentage_strings,
                                       percentage_diff_index,
                                       percentage_string,
                                       row_signature_index,
                                       row_signature,
                                       tissue_based,
                                       type2strand2percent2cancertypeslist_dict,
                                       signature2cancer_type_list_dict):

    if strand_bias == LAGGING_VERSUS_LEADING:
        strands = replication_strands
        color1 = 'indianred' # lagging
        color2 = 'goldenrod' # leading
    elif strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        strands = transcription_strands
        color1 = 'royalblue' # Transcribed
        color2 = 'yellowgreen' # Untranscribed
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        strands = genic_versus_intergenic_strands
        color1 = 'cyan' # genic
        color2 = 'gray' # intergenic

    strand1 = strands[0] # lagging transcribed genic
    strand2 = strands[1] # leading untranscribed intergenic

    if row_signature in type2strand2percent2cancertypeslist_dict:
        strand1_cancer_types_percentage = None
        strand2_cancer_types_percentage = None
        if strand1 in type2strand2percent2cancertypeslist_dict[row_signature]:
            cancer_types_list = type2strand2percent2cancertypeslist_dict[row_signature][strand1][percentage_string]
            all_cancer_types_list = signature2cancer_type_list_dict[row_signature]
            if tissue_based and (tissue_based in cancer_types_list):
                strand1_cancer_types_percentage = 100
            elif not tissue_based:
                strand1_cancer_types_percentage = (len(cancer_types_list) / len(all_cancer_types_list)) * 100
        if strand2 in type2strand2percent2cancertypeslist_dict[row_signature]:
            cancer_types_list = type2strand2percent2cancertypeslist_dict[row_signature][strand2][percentage_string]
            all_cancer_types_list = signature2cancer_type_list_dict[row_signature]
            if tissue_based and (tissue_based in cancer_types_list):
                strand2_cancer_types_percentage = 100
            elif not tissue_based:
                strand2_cancer_types_percentage = (len(cancer_types_list) / len(all_cancer_types_list)) * 100
        if (strand1_cancer_types_percentage is not None) and (strand2_cancer_types_percentage is None):
            radius = calculate_radius(strand1_cancer_types_percentage)
            if (radius > 0):
                print('Plot circle at x=%d y=%d for %s %s' % (
                percentage_diff_index, row_signature_index, row_signature, percentage_string))
                circle = plt.Circle(
                    (row_signature_index + 0.5, len(percentage_strings) - percentage_diff_index - 0.5), radius,
                    color=color1, fill=True)
                ax.add_artist(circle)
        elif (strand2_cancer_types_percentage is not None) and (strand1_cancer_types_percentage is None):
            radius = calculate_radius(strand2_cancer_types_percentage)
            if (radius > 0):
                print('Plot circle at x=%d y=%d for %s %s' % (
                percentage_diff_index, row_signature_index, row_signature, percentage_string))
                circle = plt.Circle(
                    (row_signature_index + 0.5, len(percentage_strings) - percentage_diff_index - 0.5), radius,
                    color=color2, fill=True)
                ax.add_artist(circle)
        elif (strand1_cancer_types_percentage is not None) and (strand2_cancer_types_percentage is not None):
            radius_strand1 = calculate_radius(strand1_cancer_types_percentage)
            radius_strand2 = calculate_radius(strand2_cancer_types_percentage)
            if (radius_strand1 > radius_strand2):
                # First strand1
                circle = plt.Circle((row_signature_index + 0.5, len(percentage_strings) - percentage_diff_index - 0.5),
                    radius_strand1, color=color1, fill=True)
                ax.add_artist(circle)
                # Second strand2
                circle = plt.Circle((row_signature_index + 0.5, len(percentage_strings) - percentage_diff_index - 0.5),
                    radius_strand2, color=color2, fill=True)
                ax.add_artist(circle)
            else:
                # First strand2
                circle = plt.Circle((row_signature_index + 0.5, len(percentage_strings) - percentage_diff_index - 0.5),
                    radius_strand2, color=color2, fill=True)
                ax.add_artist(circle)
                # Second strand1
                circle = plt.Circle((row_signature_index + 0.5, len(percentage_strings) - percentage_diff_index - 0.5),
                    radius_strand1, color=color1, fill=True)
                ax.add_artist(circle)


# if tissue_based == None Across all cancer types figure
# if tissue_based != None Tissue-based figure
def plot_circles_in_given_axis(ax,
                     strand_bias,
                     percentage_strings,
                     rows_sbs_signatures, # This list contains one signature
                     mutation_types,
                     xticklabels_list,
                     sbs_signature_with_number_of_cancer_types,
                     signature2mutation_type2strand2percent2cancertypeslist_dict,
                     signature2cancer_type_list_dict,
                     tissue_based = None):

    # make aspect ratio square
    ax.set_aspect(1.0)

    # Colors are from SigProfilerPlotting tool to be consistent
    colors = [[3 / 256, 189 / 256, 239 / 256],
              [1 / 256, 1 / 256, 1 / 256],
              [228 / 256, 41 / 256, 38 / 256],
              [203 / 256, 202 / 256, 202 / 256],
              [162 / 256, 207 / 256, 99 / 256],
              [236 / 256, 199 / 256, 197 / 256]]

    # Put rectangles
    x = 0

    for i in range(0, len(mutation_types), 1):
        ax.text((x + (len(percentage_strings) / 2) - 0.75), len(rows_sbs_signatures) + 1.25, mutation_types[i],fontsize=55, fontweight='bold', fontname='Arial')
        ax.add_patch(plt.Rectangle((x + .0415, len(rows_sbs_signatures) + 0.75), len(percentage_strings) - (2 * .0445), .4,facecolor=colors[i], clip_on=False))
        ax.add_patch(plt.Rectangle((x, 0), len(percentage_strings), len(rows_sbs_signatures), facecolor=colors[i], zorder=0,alpha=0.25, edgecolor='grey'))
        x += len(percentage_strings)

    # CODE GOES HERE TO CENTER X-AXIS LABELS...
    ax.set_xlim([0, len(mutation_types) * len(percentage_strings)])
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', length=0, labelsize=35)

    # major ticks
    ax.set_xticks(np.arange(0, len(mutation_types) * len(percentage_strings), 1))
    # minor ticks
    ax.set_xticks(np.arange(0, len(mutation_types) * len(percentage_strings), 1) + 0.5, minor=True)

    ax.set_xticklabels(xticklabels_list, minor=True, fontweight='bold', fontname='Arial')
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')

    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False)  # labels along the bottom edge are off

    # CODE GOES HERE TO CENTER Y-AXIS LABELS...
    ax.set_ylim([0, len(rows_sbs_signatures)])
    ax.set_yticklabels([])
    ax.tick_params(axis='y', which='both', length=0, labelsize=40)

    # major ticks
    ax.set_yticks(np.arange(0, len(rows_sbs_signatures), 1))
    # minor ticks
    ax.set_yticks(np.arange(0, len(rows_sbs_signatures), 1) + 0.5, minor=True)
    # panel1.set_yticklabels(rows_sbs_signatures, minor=True)  # fontsize
    if tissue_based:
        ax.set_yticklabels(rows_sbs_signatures, minor=True, fontname="Times New Roman", weight='bold')  # fontsize
    else:
        ax.set_yticklabels(sbs_signature_with_number_of_cancer_types, minor=True, fontname="Times New Roman", weight='bold')  # fontsize

    ax.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False)  # labels along the bottom edge are off

    # Gridlines based on major ticks
    ax.grid(which='major', color='black', zorder=3)

    for percentage_diff_index, percentage_string in enumerate(percentage_strings):
         for mutation_type_index, mutation_type in enumerate(mutation_types):
            for row_sbs_signature_index, row_sbs_signature in enumerate(rows_sbs_signatures):
                if isinstance(row_sbs_signature, tuple) or isinstance(row_sbs_signature, list):
                    sbs_signature = row_sbs_signature[0]
                    tissue_based = row_sbs_signature[1]  # which can be either tissue or None
                    row_sbs_signature = sbs_signature

                calculate_radius_add_patch_updated(strand_bias,
                                           signature2mutation_type2strand2percent2cancertypeslist_dict,
                                           signature2cancer_type_list_dict,
                                           mutation_type,
                                           mutation_type_index,
                                           percentage_strings,
                                           percentage_string,
                                           percentage_diff_index,
                                           row_sbs_signature,
                                           row_sbs_signature_index,
                                           tissue_based,
                                           ax)


# For COSMIC DBS or ID signature across all tissues and tissue  based all together
def plot_dbs_and_id_signatures_circle_figures_across_all_tissues_and_tissue_based_together(signature_type,
                                       signatures,
                                       strand_bias,
                                       strand_bias_figures_output_dir,
                                       type2strand2percent2cancertypeslist_dict,
                                       signature2cancer_type_list_dict,
                                       percentage_strings,
                                       cosmic_release_version,
                                       figure_file_extension,
                                       signature_name = None):
    figure_dir = FIGURES_COSMIC
    x_ticks_labelsize = 60
    y_ticks_labelsize = 60
    signature_tissue_type_tuples, signatures_ylabels_on_the_heatmap = fill_lists(signatures[0], signature2cancer_type_list_dict)

    # COSMIC
    if signature_name:
        width = 20 + 2 * len(signatures_ylabels_on_the_heatmap)
        height = 20
        fig = plt.figure(figsize=(width, height))
        ax = plt.gca()
        second_legend_axis = inset_axes(ax, width=15, height=5, loc='upper left', bbox_to_anchor=(0, -0.925, 1, 0.9), bbox_transform=ax.transAxes)  # works to the left, looks better
        plot_proportion_of_cancer_types_in_given_axis(second_legend_axis, strand_bias, write_text=True)

    # Make aspect ratio square
    ax.set_aspect(1.0)

    for percentage_diff_index, percentage_string in enumerate(percentage_strings):
        for row_signature_index, signature_tissue_tuple in enumerate(signature_tissue_type_tuples):
            row_signature, tissue_based = signature_tissue_tuple
            plot_circles_in_given_axis_for_dbs_id_signatures(ax,
                                       strand_bias,
                                       percentage_strings,
                                       percentage_diff_index,
                                       percentage_string,
                                       row_signature_index,
                                       row_signature,
                                       tissue_based,
                                       type2strand2percent2cancertypeslist_dict,
                                       signature2cancer_type_list_dict)

    yticklabels_list = []
    for percentage_string in percentage_strings:
        if percentage_string=='5%':
            yticklabels_list.append('1.05')
        elif percentage_string=='10%':
            yticklabels_list.append('1.1')
        elif percentage_string=='20%':
            yticklabels_list.append('1.2')
        elif percentage_string=='25%':
            yticklabels_list.append('1.25')
        elif percentage_string=='30%':
            yticklabels_list.append('1.3')
        elif percentage_string=='50%':
            yticklabels_list.append('1.5')
        elif percentage_string=='75%':
            yticklabels_list.append('1.75')
        elif percentage_string=='100%':
            yticklabels_list.append('2+')
    yticklabels_list.reverse()

    # CODE GOES HERE TO CENTER X-AXIS LABELS...
    ax.set_xlim([0,len(signatures_ylabels_on_the_heatmap)])
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', length=0, labelsize=x_ticks_labelsize)

    #major ticks
    ax.set_xticks(np.arange(0, len(signatures_ylabels_on_the_heatmap), 1))
    #minor ticks
    ax.set_xticks(np.arange(0, len(signatures_ylabels_on_the_heatmap), 1) + 0.5, minor=True)

    # ax.set_xticklabels(signatures_ylabels_on_the_heatmap, minor=True, fontweight='bold', fontname='Arial', fontsize=x_ticks_labelsize) # legacy
    ax.set_xticklabels(signatures_ylabels_on_the_heatmap, minor=True, fontsize=x_ticks_labelsize, rotation=55, ha="left", rotation_mode="anchor")
    ax.xaxis.set_ticks_position('top')

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False)  # labels along the bottom edge are off

    # CODE GOES HERE TO CENTER Y-AXIS LABELS...
    ax.set_ylim([0,len(percentage_strings)])
    ax.set_yticklabels([])
    ax.tick_params(axis='y', which='both', length=0, labelsize=y_ticks_labelsize)

    # major ticks
    ax.set_yticks(np.arange(0, len(percentage_strings), 1) + 1)
    # minor ticks
    ax.set_yticks(np.arange(0, len(percentage_strings), 1) + 0.5, minor=True)

    yticks = np.arange(0,len(percentage_strings))
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels_list, minor=True, fontsize=y_ticks_labelsize)  # fontsize
    ax.set_ylabel('Fold\nchange', fontsize=y_ticks_labelsize, rotation=0, labelpad=100)

    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False)  # labels along the bottom edge are off

    # Put the legend
    if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        legend_elements = [
            Line2D([0], [0], marker='o', color='white', label='Genic: Transcribed Strand', markerfacecolor='royalblue', markersize=40),
            Line2D([0], [0], marker='o', color='white', label='Genic: Untranscribed Strand', markerfacecolor='yellowgreen', markersize=40)]
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        legend_elements = [
            Line2D([0], [0], marker='o', color='white', label='Genic Regions', markerfacecolor='cyan', markersize=40),
            Line2D([0], [0], marker='o', color='white', label='Intergenic Regions', markerfacecolor='gray', markersize=40)]
    elif (strand_bias == LAGGING_VERSUS_LEADING):
        legend_elements = [
            Line2D([0], [0], marker='o', color='white', label='Lagging Strand', markerfacecolor='indianred', markersize=40),
            Line2D([0], [0], marker='o', color='white', label='Leading Strand', markerfacecolor='goldenrod', markersize=40)]

    ax.legend(handles=legend_elements, ncol = 1, loc="upper left", bbox_to_anchor=(0, 0), fontsize=40) # one row

    # Gridlines based on major ticks
    ax.grid(which='major', color='black')
    # ax.grid(which="minor", color="black", linestyle='-', linewidth=2) # hm heatmaps

    if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        feature_name = 'TRANSCR_ASYM'
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        feature_name = 'GENIC_ASYM'
    elif strand_bias == LAGGING_VERSUS_LEADING:
        feature_name = 'REPLIC_ASYM'

    # v3.2_SBS1_REPLIC_ASYM_TA_C34447.jpg
    if signature_name:
        filename = '%s_%s_%s.%s' % (cosmic_release_version, signature_name, feature_name, figure_file_extension)
    else:
        filename = '%s_%s_Signatures_%s.%s' % (cosmic_release_version, signature_type, feature_name, figure_file_extension)

    figFile = os.path.join(strand_bias_figures_output_dir, figure_dir, filename)
    fig.savefig(figFile, dpi=100, bbox_inches="tight")

    plt.cla()
    plt.close(fig)


def plot_cosmic_strand_bias_figure_in_parallel(signature,
                                   signature_type,
                                   signature2cancer_type_list_dict,
                                   strand_bias,
                                   strand_bias_output_dir,
                                   signature_strand1_versus_strand2_for_bar_plot_df,
                                   signature_transcribed_versus_untranscribed_df,
                                   signature_genic_versus_intergenic_df,
                                   signature_lagging_versus_leading_df,
                                   signature_mutation_type_strand_cancer_types_percentages_df,
                                   signature2mutation_type2strand2percent2cancertypeslist_dict,
                                    type2strand2percent2cancertypeslist_dict,
                                    cancer_type2source_cancer_type_tuples_dict,
                                    percentage_strings,
                                    significance_level,
                                    number_of_required_mutations_for_stacked_bar_plot,
                                    figure_type,
                                    cosmic_release_version,
                                    figure_file_extension,
                                    figure_case_study):

    if signature_type == SBS:
        sbs_signature = signature
        # if any_bias_to_show_for_six_mutation_types(sbs_signature,strand_bias,signature2mutation_type2strand2percent2cancertypeslist_dict):
        if (sbs_signature in signature2cancer_type_list_dict) and (len(signature2cancer_type_list_dict[sbs_signature]) > 0):
            # COSMIC SBS Signature Across All Tissues and Tissue Based Together
            # For Figure Case Study SBS28 plots bar plots
            plot_six_mutations_sbs_signatures_bar_plot_circles_across_all_tissues_and_tissue_based_together(sbs_signature,
                                                                        strand_bias,
                                                                        strand_bias_output_dir,
                                                                        signature_strand1_versus_strand2_for_bar_plot_df,
                                                                        signature_transcribed_versus_untranscribed_df,
                                                                        signature_genic_versus_intergenic_df,
                                                                        signature_lagging_versus_leading_df,
                                                                        signature_mutation_type_strand_cancer_types_percentages_df,
                                                                        signature2mutation_type2strand2percent2cancertypeslist_dict,
                                                                        signature2cancer_type_list_dict,
                                                                        cancer_type2source_cancer_type_tuples_dict,
                                                                        percentage_strings,
                                                                        significance_level,
                                                                        cosmic_release_version,
                                                                        figure_file_extension,
                                                                        figure_case_study)

            # # Not used anynore
            # Cosmic SBS Signature Tissue Based
            # cancer_type_list = signature2cancer_type_list_dict[sbs_signature]
            # for tissue_based_cancer_type in cancer_type_list:
            #     plot_six_mutations_sbs_signatures_bar_plot_circles_together(sbs_signature,
            #                                                                 strand_bias,
            #                                                                 strand_bias_output_dir,
            #                                                                 signature_strand1_versus_strand2_for_bar_plot_df,
            #                                                                 signature_transcribed_versus_untranscribed_df,
            #                                                                 signature_genic_versus_intergenic_df,
            #                                                                 signature_lagging_versus_leading_df,
            #                                                                 signature_mutation_type_strand_cancer_types_percentages_df,
            #                                                                 signature2mutation_type2strand2percent2cancertypeslist_dict,
            #                                                                 signature2cancer_type_list_dict,
            #                                                                 cancer_type2source_cancer_type_tuples_dict,
            #                                                                 percentage_strings,
            #                                                                 significance_level,
            #                                                                 number_of_required_mutations_for_stacked_bar_plot,
            #                                                                 cosmic_release_version,
            #                                                                 figure_file_extension,
            #                                                                 figure_case_study,
            #                                                                 tissue_based = tissue_based_cancer_type)

            # # Not used anynore
            # # Cosmic SBS Signature Across All Cancer Types
            # plot_six_mutations_sbs_signatures_bar_plot_circles_together(sbs_signature,
            #                                                             strand_bias,
            #                                                             strand_bias_output_dir,
            #                                                             signature_strand1_versus_strand2_for_bar_plot_df,
            #                                                             signature_transcribed_versus_untranscribed_df,
            #                                                             signature_genic_versus_intergenic_df,
            #                                                             signature_lagging_versus_leading_df,
            #                                                             signature_mutation_type_strand_cancer_types_percentages_df,
            #                                                             signature2mutation_type2strand2percent2cancertypeslist_dict,
            #                                                             signature2cancer_type_list_dict,
            #                                                             cancer_type2source_cancer_type_tuples_dict,
            #                                                             percentage_strings,
            #                                                             significance_level,
            #                                                             number_of_required_mutations_for_stacked_bar_plot,
            #                                                             cosmic_release_version,
            #                                                             figure_file_extension,
            #                                                             figure_case_study)


    elif signature_type == DBS:
        dbs_signature = signature

        if dbs_signature in signature2cancer_type_list_dict:
            # Cosmic DBS Signature Across All Tissues and Tissue Based Together
            plot_dbs_and_id_signatures_circle_figures_across_all_tissues_and_tissue_based_together(DBS,
                                               [dbs_signature],
                                               strand_bias,
                                               strand_bias_output_dir,
                                               type2strand2percent2cancertypeslist_dict,
                                               signature2cancer_type_list_dict,
                                               percentage_strings,
                                               cosmic_release_version,
                                               figure_file_extension,
                                               signature_name = dbs_signature)

            # # Not used any more
            # # Cosmic DBS Signature Tissue Based
            # cancer_type_list = signature2cancer_type_list_dict[dbs_signature]
            # for tissue_based_cancer_type in cancer_type_list:
            #     plot_dbs_and_id_signatures_figures(DBS,
            #                                        [dbs_signature],
            #                                        strand_bias,
            #                                        strand_bias_output_dir,
            #                                        significance_level,
            #                                        type2strand2percent2cancertypeslist_dict,
            #                                        signature2cancer_type_list_dict,
            #                                        percentage_strings,
            #                                        figure_type,
            #                                        cosmic_release_version,
            #                                        figure_file_extension,
            #                                        signature_name = dbs_signature,
            #                                        tissue_based = tissue_based_cancer_type)
            #
            # # Not used any more
            # # Cosmic DBS Signature Across All Tissues
            # # if any_bias_to_show(dbs_signature, strand_bias,type2strand2percent2cancertypeslist_dict):
            # plot_dbs_and_id_signatures_figures(DBS,
            #                                    [dbs_signature],
            #                                    strand_bias,
            #                                    strand_bias_output_dir,
            #                                    significance_level,
            #                                    type2strand2percent2cancertypeslist_dict,
            #                                    signature2cancer_type_list_dict,
            #                                    percentage_strings,
            #                                    figure_type,
            #                                    cosmic_release_version,
            #                                    figure_file_extension,
            #                                    signature_name = dbs_signature)


    elif signature_type == ID:
        id_signature = signature

        if id_signature in signature2cancer_type_list_dict:
            # Cosmic ID Signature Across All Tissues and Tissue Based Together
            plot_dbs_and_id_signatures_circle_figures_across_all_tissues_and_tissue_based_together(ID,
                                               [id_signature],
                                               strand_bias,
                                               strand_bias_output_dir,
                                               type2strand2percent2cancertypeslist_dict,
                                               signature2cancer_type_list_dict,
                                               percentage_strings,
                                               cosmic_release_version,
                                               figure_file_extension,
                                               signature_name = id_signature)

            # not used any more
            # # Cosmic ID Signature Tissue Based
            # cancer_type_list = signature2cancer_type_list_dict[id_signature]
            # for tissue_based_cancer_type in cancer_type_list:
            #     plot_dbs_and_id_signatures_figures(ID,
            #                                        [id_signature],
            #                                        strand_bias,
            #                                        strand_bias_output_dir,
            #                                        significance_level,
            #                                        type2strand2percent2cancertypeslist_dict,
            #                                        signature2cancer_type_list_dict,
            #                                        percentage_strings,
            #                                        figure_type,
            #                                        cosmic_release_version,
            #                                        figure_file_extension,
            #                                        signature_name = id_signature,
            #                                        tissue_based = tissue_based_cancer_type)

            # not used any more
            # # Cosmic ID Signature Across All Tissues
            # # if any_bias_to_show(id_signature, strand_bias, type2strand2percent2cancertypeslist_dict):
            # plot_dbs_and_id_signatures_figures(ID,
            #                                    [id_signature],
            #                                    strand_bias,
            #                                    strand_bias_output_dir,
            #                                    significance_level,
            #                                    type2strand2percent2cancertypeslist_dict,
            #                                    signature2cancer_type_list_dict,
            #                                    percentage_strings,
            #                                    figure_type,
            #                                    cosmic_release_version,
            #                                    figure_file_extension,
            #                                    signature_name = id_signature)


# Cosmic Figure Across All Cancer Types: Across all tissues + tissue based results
def plot_six_mutations_sbs_signatures_bar_plot_circles_across_all_tissues_and_tissue_based_together(signature,
                                              strand_bias,
                                              strand_bias_output_dir,
                                              signature_strand1_versus_strand2_for_bar_plot_df,
                                              signature_transcribed_versus_untranscribed_df,
                                              signature_genic_versus_intergenic_df,
                                              signature_lagging_versus_leading_df,
                                              signature_mutation_type_strand_cancer_types_percentages_df,
                                              signature2mutation_type2strand2percent2cancertypeslist_dict,
                                              signature2cancer_type_list_dict,
                                              cancer_type2source_cancer_type_tuples_dict,
                                              percentage_strings,
                                              significance_level,
                                              cosmic_release_version,
                                              figure_file_extension,
                                              figure_case_study,
                                              tissue_based = None):

    percentage = True
    plot_type = '384'

    signature_circle_plot_df = pd.DataFrame()
    if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        data_file_name = '%s_%s_%s.txt' % (cosmic_release_version, signature, COSMIC_TRANSCRIPTION_STRAND_BIAS)
    elif strand_bias == LAGGING_VERSUS_LEADING:
        data_file_name = '%s_%s_%s.txt' % (cosmic_release_version, signature, COSMIC_REPLICATION_STRAND_BIAS)
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        data_file_name = '%s_%s_%s.txt' % (cosmic_release_version, signature, COSMIC_GENIC_VS_INTERGENIC_BIAS)
    data_file_path = os.path.join(strand_bias_output_dir, DATA_FILES, data_file_name)

    groupby_df = signature_mutation_type_strand_cancer_types_percentages_df.groupby(['signature'])

    if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        strand1 = transcription_strands[0]
        strand2 = transcription_strands[1]
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        strand1 = genic_versus_intergenic_strands[0]
        strand2 = genic_versus_intergenic_strands[1]
    elif strand_bias == LAGGING_VERSUS_LEADING:
        strand1 = replication_strands[0]
        strand2 = replication_strands[1]

    if signature in groupby_df.groups:
        signature_circle_plot_df = groupby_df.get_group(signature)
        signature_circle_plot_df = signature_circle_plot_df[(signature_circle_plot_df['strand'] == strand1) | ((signature_circle_plot_df['strand'] == strand2))]

    mutation_types = six_mutation_types

    # xticklabels_list = percentage_strings * 6
    # xticklabels_list = ['1.1', '1.2', '1.3', '1.5', '1.75', '2+'] * 6

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

    xticklabels_list = xticklabels_list * 6

    plt.rc('axes', edgecolor='lightgray')
    plt.rcParams['axes.linewidth'] = 2

    signature_tissue_type_tuples, signatures_ylabels_on_the_heatmap = fill_lists(signature, signature2cancer_type_list_dict)
    signature_tissue_type_tuples.reverse()
    signatures_ylabels_on_the_heatmap.reverse()

    # constant
    width = 60
    num_of_columns = 7

    # varying
    height = 24 + 1.65 * len(signatures_ylabels_on_the_heatmap)
    num_of_rows = 17 + len(signatures_ylabels_on_the_heatmap)

    fig = plt.figure(figsize=(width, height))  # (width, height)
    gs = gridspec.GridSpec(num_of_rows, num_of_columns, figure=fig)
    fig.subplots_adjust(hspace=0, wspace=0)

    # First row
    real_data_plot_axis = fig.add_subplot(gs[0:8, :])

    # Second row
    cirle_plot_axis = fig.add_subplot(gs[11:-1, :])

    # Third row new
    proportion_of_cancer_types_text_axis = plt.subplot(gs[-1, 0:1])
    proportion_of_cancer_types_axis = plt.subplot(gs[-1, 1:3])
    circle_plot_legend_axis = plt.subplot(gs[-1, -1])

    # Plot first row
    signature_real_data_df = prepare_df_and_plot_real_data_in_given_axis(real_data_plot_axis,
                                 signature,
                                 signature2cancer_type_list_dict,
                                 cancer_type2source_cancer_type_tuples_dict,
                                 percentage,
                                 plot_type,
                                 strand_bias,
                                 tissue_based = tissue_based)

    # Plot second row
    plot_circles_in_given_axis(cirle_plot_axis,
                             strand_bias,
                             percentage_strings,
                             signature_tissue_type_tuples,
                             mutation_types,
                             xticklabels_list,
                             signatures_ylabels_on_the_heatmap,
                             signature2mutation_type2strand2percent2cancertypeslist_dict,
                             signature2cancer_type_list_dict,
                             tissue_based = None)

    # Plot third row
    if tissue_based:
        proportion_of_cancer_types_text_axis.set_axis_off()
        proportion_of_cancer_types_axis.set_axis_off()
        plot_legend_in_given_axis(circle_plot_legend_axis, strand_bias)
    else:
        plot_proportion_of_cancer_types_text_in_given_axis_new(proportion_of_cancer_types_text_axis, strand_bias)
        plot_proportion_of_cancer_types_in_given_axis(proportion_of_cancer_types_axis, strand_bias)
        plot_legend_in_given_axis(circle_plot_legend_axis, strand_bias)

    # For Figure Case Study SBS28
    # Bar Plot in a separate figure
    # Cosmic Across All Cancer Types
    if figure_case_study:
        plot_bar_plot_in_given_axis(None,
                                    signature,
                                    strand_bias,
                                    strand_bias_output_dir,
                                    signature_strand1_versus_strand2_for_bar_plot_df,
                                    signature_transcribed_versus_untranscribed_df,
                                    signature_genic_versus_intergenic_df,
                                    signature_lagging_versus_leading_df,
                                    significance_level,
                                    tissue_based = None,
                                    figure_case_study = figure_case_study)


    # Write COSMIC data files
    signature_real_data_df.to_csv(data_file_path, sep='\t', index=False, mode='w')
    with open(data_file_path, 'a') as f:
        f.write("\n")
        signature_circle_plot_df.to_csv(f, sep='\t', index=False)

    if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        feature_name = 'TRANSCR_ASYM'
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        feature_name = 'GENIC_ASYM'
    elif strand_bias == LAGGING_VERSUS_LEADING:
        feature_name = 'REPLIC_ASYM'

    if tissue_based:
        # v3.2_SBS1_REPLIC_ASYM_TA_C34447.jpg
        NCI_Thesaurus_code = cancer_type_2_NCI_Thesaurus_code_dict[tissue_based]
        filename = '%s_%s_%s_TA_%s.%s' % (cosmic_release_version , signature, feature_name, NCI_Thesaurus_code, figure_file_extension)
        figFile = os.path.join(strand_bias_output_dir, COSMIC_TISSUE_BASED_FIGURES , filename)
        fig.savefig(figFile, dpi = 100, bbox_inches = "tight")
    else:
        filename = '%s_%s_%s.%s' % (cosmic_release_version , signature, feature_name, figure_file_extension)
        figFile = os.path.join(strand_bias_output_dir, FIGURES_COSMIC , filename)
        fig.savefig(figFile, dpi = 100, bbox_inches = "tight")

    plt.cla()
    plt.close(fig)

# Plots Cosmic Figure Tissue Based
# Formerly this function was being used for Cosmic Figure Across All Cancer Types
# now plot_six_mutations_sbs_signatures_bar_plot_circles_across_all_tissues_and_tissue_based_together
def plot_six_mutations_sbs_signatures_bar_plot_circles_together(signature,
                                              strand_bias,
                                              strand_bias_output_dir,
                                              signature_strand1_versus_strand2_for_bar_plot_df,
                                              signature_transcribed_versus_untranscribed_df,
                                              signature_genic_versus_intergenic_df,
                                              signature_lagging_versus_leading_df,
                                              signature_mutation_type_strand_cancer_types_percentages_df,
                                              signature2mutation_type2strand2percent2cancertypeslist_dict,
                                              signature2cancer_type_list_dict,
                                              cancer_type2source_cancer_type_tuples_dict,
                                              percentage_strings,
                                              significance_level,
                                              number_of_required_mutations_for_stacked_bar_plot,
                                              cosmic_release_version,
                                              figure_file_extension,
                                              figure_case_study,
                                              tissue_based = None):
    percentage = True
    plot_type = '384'

    groupby_df = signature_strand1_versus_strand2_for_bar_plot_df.groupby(['signature'])
    if signature in groupby_df.groups:
        signature_bar_plot_df = groupby_df.get_group(signature)

    groupby_df = signature_mutation_type_strand_cancer_types_percentages_df.groupby(['signature'])
    if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        strand1 = transcription_strands[0]
        strand2 = transcription_strands[1]
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        strand1 = genic_versus_intergenic_strands[0]
        strand2 = genic_versus_intergenic_strands[1]
    elif strand_bias == LAGGING_VERSUS_LEADING:
        strand1 = replication_strands[0]
        strand2 = replication_strands[1]
    if signature in groupby_df.groups:
        signature_circle_plot_df = groupby_df.get_group(signature)
        signature_circle_plot_df = signature_circle_plot_df[(signature_circle_plot_df['strand']==strand1) | ((signature_circle_plot_df['strand']==strand2))]

    mutation_types = six_mutation_types
    sbs_signature_with_number_of_cancer_types = augment_with_number_of_cancer_types(SBS, [signature], signature2cancer_type_list_dict, new_line=True)

    # xticklabels_list = percentage_strings * 6
    # xticklabels_list = ['1.1', '1.2', '1.3', '1.5', '1.75', '2+'] * 6

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

    xticklabels_list = xticklabels_list * 6

    fig = plt.figure(figsize=(60, 38))

    plt.rc('axes', edgecolor='lightgray')
    plt.rcParams['axes.linewidth'] = 2

    width = 7
    height = 4
    width_ratios = [1] * width
    gs = gridspec.GridSpec(height, width, height_ratios = [2.25, 1, 0.75, 2], width_ratios = width_ratios) # legacy
    fig.subplots_adjust(hspace=0.5, wspace=0)

    # First row
    real_data_plot_axis = plt.subplot(gs[0, :])

    # Second row
    cirle_plot_axis = plt.subplot(gs[1, :])

    # Third row
    proportion_of_cancer_types_text_axis = plt.subplot(gs[2, 0:1])
    proportion_of_cancer_types_axis = plt.subplot(gs[2, 1:3])
    circle_plot_legend_axis = plt.subplot(gs[2,-1])

    # Fourth row
    bar_plot_axis = plt.subplot(gs[3, 0:3])
    # bars_legend_axis = plt.subplot(gs[3, 3]) # newly added
    stacked_bar_plot_axis = plt.subplot(gs[3, 4:])

    # Plot first row
    signature_real_data_df = prepare_df_and_plot_real_data_in_given_axis(real_data_plot_axis,
                                 signature,
                                 signature2cancer_type_list_dict,
                                 cancer_type2source_cancer_type_tuples_dict,
                                 percentage,
                                 plot_type,
                                 strand_bias,
                                 tissue_based = tissue_based)

    # Plot second row
    # Data comes from signature2mutation_type2strand2percent2cancertypeslist_dict
    plot_circles_in_given_axis(cirle_plot_axis,
                             strand_bias,
                             percentage_strings,
                             [signature],
                             mutation_types,
                             xticklabels_list,
                             sbs_signature_with_number_of_cancer_types,
                             signature2mutation_type2strand2percent2cancertypeslist_dict,
                             signature2cancer_type_list_dict,
                             tissue_based = tissue_based)

    # Plot third row
    if tissue_based:
        proportion_of_cancer_types_text_axis.set_axis_off()
        proportion_of_cancer_types_axis.set_axis_off()
        plot_legend_in_given_axis(circle_plot_legend_axis, strand_bias)
    else:
        plot_proportion_of_cancer_types_text_in_given_axis(proportion_of_cancer_types_text_axis)
        plot_proportion_of_cancer_types_in_given_axis(proportion_of_cancer_types_axis)
        plot_legend_in_given_axis(circle_plot_legend_axis, strand_bias)

    # Plot fourth row left
    # Data comes from signature_strand1_versus_strand2_for_bar_plot_df
    mutation_type_display = plot_bar_plot_in_given_axis(bar_plot_axis,
                                signature,
                                strand_bias,
                                strand_bias_output_dir,
                                signature_strand1_versus_strand2_for_bar_plot_df,
                                signature_transcribed_versus_untranscribed_df,
                                signature_genic_versus_intergenic_df,
                                signature_lagging_versus_leading_df,
                                significance_level,
                                tissue_based = tissue_based)

    # For Figure Case Study SBS28
    # Bar Plot in a separate figure
    # Cosmic Tissue Based
    if figure_case_study:
        plot_bar_plot_in_given_axis(None,
                                    signature,
                                    strand_bias,
                                    strand_bias_output_dir,
                                    signature_strand1_versus_strand2_for_bar_plot_df,
                                    signature_transcribed_versus_untranscribed_df,
                                    signature_genic_versus_intergenic_df,
                                    signature_lagging_versus_leading_df,
                                    significance_level,
                                    tissue_based=tissue_based)
    # Plot fourth row right
    # Data comes from signature_strand1_versus_strand2_for_bar_plot_df
    plot_stacked_bar_plot_in_given_axis(stacked_bar_plot_axis,
                                signature,
                                strand_bias,
                                strand_bias_output_dir,
                                signature_strand1_versus_strand2_for_bar_plot_df,
                                signature_transcribed_versus_untranscribed_df,
                                signature_genic_versus_intergenic_df,
                                signature_lagging_versus_leading_df,
                                mutation_type_display,
                                significance_level,
                                number_of_required_mutations_for_stacked_bar_plot,
                                tissue_based = tissue_based)

    if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        feature_name = 'TRANSCR_ASYM'
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        feature_name = 'GENIC_ASYM'
    elif strand_bias == LAGGING_VERSUS_LEADING:
        feature_name = 'REPLIC_ASYM'

    if tissue_based:
        # v3.2_SBS1_REPLIC_ASYM_TA_C34447.jpg
        NCI_Thesaurus_code = cancer_type_2_NCI_Thesaurus_code_dict[tissue_based]
        filename = '%s_%s_%s_TA_%s.%s' % (cosmic_release_version , signature, feature_name, NCI_Thesaurus_code, figure_file_extension)
        figFile = os.path.join(strand_bias_output_dir, COSMIC_TISSUE_BASED_FIGURES , filename)
        fig.savefig(figFile, dpi=100, bbox_inches="tight")
    else:
        filename = '%s_%s_%s.%s' % (cosmic_release_version , signature, feature_name, figure_file_extension)
        figFile = os.path.join(strand_bias_output_dir, FIGURES_COSMIC , filename)
        fig.savefig(figFile, dpi=100, bbox_inches="tight")

    plt.cla()
    plt.close(fig)

def plot_colorbar(output_path,
                  strand_bias,
                  strands,
                  colours,
                  sub_dir = FIGURES_MANUSCRIPT):

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_axes([0.05, 0.475, 0.9, 0.15])

    bins = [0, 0.5, 0.8, 1, 1.2, 1.5, 2]
    cmap = mpl.colors.ListedColormap(colours)
    norm = mpl.colors.BoundaryNorm(boundaries=bins, ncolors=len(cmap.colors))

    cb = mpl.colorbar.ColorbarBase(ax,
                                    cmap=cmap,
                                    norm=norm,
                                    boundaries=bins,
                                    ticks=bins,
                                    spacing='proportional',
                                    orientation='horizontal')

    cb.set_ticklabels(["2+", "1.5", "1.2", "1", "1.2", "1.5", "2+"])
    cb.ax.tick_params(labelsize=25)

    if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        cb.ax.set_title('Transcriptional Strand Asymmetry', fontsize=30, pad=40) # legacy fontsize=25
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        cb.ax.set_title('Genic versus Intergenic Regions', fontsize=30, pad=40) # legacy fontsize=25
    elif strand_bias == LAGGING_VERSUS_LEADING:
        cb.ax.set_title('Replicational Strand Asymmetry', fontsize=30, pad=40)  # legacy fontsize=25

    cb.set_label("Fold change", horizontalalignment='center', rotation=0, fontsize=30)  # legacy fontsize=25

    if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        cb.ax.text(0.16, 1, strands[1], transform=cb.ax.transAxes, va='bottom', ha='center', fontsize=30)  # legacy fontsize=25
        cb.ax.text(0.87, 1, strands[0], transform=cb.ax.transAxes, va='bottom', ha='center', fontsize=30)  # legacy fontsize=25
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        cb.ax.text(0.12, 1, strands[1], transform=cb.ax.transAxes, va='bottom', ha='center', fontsize=30)  # legacy fontsize=25
        cb.ax.text(0.93, 1, strands[0], transform=cb.ax.transAxes, va='bottom', ha='center', fontsize=30)  # legacy fontsize=25
    if strand_bias == LAGGING_VERSUS_LEADING:
        cb.ax.text(0.09, 1, strands[1], transform=cb.ax.transAxes, va='bottom', ha='center', fontsize=30)  # legacy fontsize=25
        cb.ax.text(0.91, 1, strands[0], transform=cb.ax.transAxes, va='bottom', ha='center', fontsize=30)  # legacy fontsize=25

    # plt.show()
    filename = 'strand_bias_%s_color_bar.png' %(strand_bias)
    if sub_dir:
        figureFile = os.path.join(output_path, sub_dir, filename)
    else:
        figureFile = os.path.join(output_path, filename)

    fig.savefig(figureFile)
    plt.close()


def plot_colorbar_alternative(output_path, strand_bias, strands, cmap, vmin, vmax, norm):
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_axes([0.05, 0.475, 0.9, 0.15])

    # If a ListedColormap is used, the length of the bounds array must be
    # one greater than the length of the color list.  The bounds must be
    # monotonically increasing.

    bounds = np.arange(vmin, vmax+1, 0.1)
    # norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)

    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, ticks=bounds, spacing='proportional', orientation='horizontal')

    cb.ax.tick_params(labelsize=10)
    cb.set_ticks([0, 0.25, 0.5, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.5, 1.75, 2])
    cb.set_ticklabels(["2", "1.75", "1.5", "1.3", "1.2", "1.1", "1", "1.1", "1.2", "1.3", "1.5", "1.75", "2"])

    if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        cb.ax.set_title('Transcriptional Strand Bias', fontsize=25, pad=20)
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        cb.ax.set_title('Genic versus Intergenic Regions', fontsize=25, pad=20)
    elif strand_bias == LAGGING_VERSUS_LEADING:
        cb.ax.set_title('Replicational Strand Bias', fontsize=25, pad=20)

    cb.set_label("Fold change", horizontalalignment='center', rotation=0, fontsize=15)

    cb.ax.text(0.08, 1, strands[1], transform=cb.ax.transAxes, va='bottom', ha='center', fontsize=15)
    cb.ax.text(0.93, 1, strands[0], transform=cb.ax.transAxes, va='bottom', ha='center', fontsize=15)
    # plt.show()

    filename = 'strand_bias_%s_color_bar.png' %(strand_bias)
    figureFile = os.path.join(output_path, FIGURES_MANUSCRIPT, filename)

    fig.savefig(figureFile)
    plt.close()


# Called by plot_new_dbs_and_id_signatures_figures
def calculate_radius_add_patch(strand_bias,
                    cmap,
                    norm,
                    signature2cancer_type_list_dict,
                    type2strand2percent2cancertypeslist_dict,
                    row_signature,
                    row_signature_index,
                    percentage_string,
                    percentage_string_index,
                    ax,
                    tissue_based = None):

    color_value = calculate_fold_change(percentage_string)

    if strand_bias == LAGGING_VERSUS_LEADING:
        strand1 = LAGGING
        strand2 = LEADING
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        strand1 = GENIC
        strand2 = INTERGENIC
    elif strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        strand1 = TRANSCRIBED_STRAND
        strand2 = UNTRANSCRIBED_STRAND

    strand1_cancer_types_percentage = None
    strand2_cancer_types_percentage = None

    if strand1 in type2strand2percent2cancertypeslist_dict[row_signature]:
        if tissue_based:
            cancer_types_list = type2strand2percent2cancertypeslist_dict[row_signature][strand1][percentage_string]
            if tissue_based in cancer_types_list:
                strand1_cancer_types_percentage = 100
                color_strand1 = color_value
        else:
            cancer_types_list = type2strand2percent2cancertypeslist_dict[row_signature][strand1][percentage_string]
            all_cancer_types_list = signature2cancer_type_list_dict[row_signature]
            strand1_cancer_types_percentage = (len(cancer_types_list) / len(all_cancer_types_list)) * 100
            color_strand1 = color_value

    if strand2 in type2strand2percent2cancertypeslist_dict[row_signature]:
        if tissue_based:
            cancer_types_list = type2strand2percent2cancertypeslist_dict[row_signature][strand2][percentage_string]
            if tissue_based in cancer_types_list:
                strand2_cancer_types_percentage = 100
                color_strand2 = 2 - color_value
        else:
            cancer_types_list = type2strand2percent2cancertypeslist_dict[row_signature][strand2][percentage_string]
            all_cancer_types_list = signature2cancer_type_list_dict[row_signature]
            strand2_cancer_types_percentage = (len(cancer_types_list) / len(all_cancer_types_list)) * 100
            color_strand2 = 2 - color_value

    if (strand1_cancer_types_percentage is not None) and (strand2_cancer_types_percentage is None):
        radius = calculate_radius(strand1_cancer_types_percentage)
        if (radius > 0):
            print('Plot circle at x=%d y=%d for %s %s' % (percentage_string_index, row_signature_index, row_signature, percentage_string))
            circle = plt.Circle((percentage_string_index + 0.5, row_signature_index + 0.5), radius, color=cmap(norm(color_strand1)), fill=True)
            ax.add_artist(circle)
    elif (strand2_cancer_types_percentage is not None) and (strand1_cancer_types_percentage is None):
        radius = calculate_radius(strand2_cancer_types_percentage)
        if (radius > 0):
            print('Plot circle at x=%d y=%d for %s %s' % (percentage_string_index, row_signature_index, row_signature, percentage_string))
            circle = plt.Circle((percentage_string_index + 0.5, row_signature_index + 0.5), radius, color=cmap(norm(color_strand2)), fill=True)
            ax.add_artist(circle)
    elif (strand1_cancer_types_percentage is not None) and (strand2_cancer_types_percentage is not None):
        radius_strand1 = calculate_radius(strand1_cancer_types_percentage)
        radius_strand2 = calculate_radius(strand2_cancer_types_percentage)
        if (radius_strand1 > radius_strand2):
            # First strand1
            circle = plt.Circle((percentage_string_index + 0.5, row_signature_index + 0.5), radius_strand1, color=cmap(norm(color_strand1)), fill=True)
            ax.add_artist(circle)
            # Second strand2
            if radius_strand2 > 0:
                circle = plt.Circle((percentage_string_index + 0.5, row_signature_index + 0.5), radius_strand2, color=cmap(norm(color_strand2)), fill=True)
                ax.add_artist(circle)
        else:
            # First strand2
            circle = plt.Circle((percentage_string_index + 0.5, row_signature_index + 0.5), radius_strand2, color=cmap(norm(color_strand2)), fill=True)
            ax.add_artist(circle)
            # Second strand1
            if radius_strand1 > 0:
                circle = plt.Circle((percentage_string_index + 0.5, row_signature_index + 0.5), radius_strand1, color=cmap(norm(color_strand1)), fill=True)
                ax.add_artist(circle)


def calculate_fold_change(percentage_string):
    return int(percentage_string[:-1]) / 100 + 1


# Called by plot_new_six_mutations_sbs_signatures_circle_figures
# Calculate the color and radius for the given signature and mutation type for each strand
def calculate_radius_color_add_patch(strand_bias,
            cmap,
            norm,
            df,
            signature2cancer_type_list_dict,
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
    cancer_types = signature2cancer_type_list_dict[row_sbs_signature]

    color_strand1 = None
    color_strand2 = None

    for strand in strands:
        strand_values = 0
        cancer_types_at_least_10_percent = 0

        for cancer_type in cancer_types:
            for percentage_string in percentage_strings[::-1]:
                if df[(df['cancer_type'] == cancer_type) &
                    (df['signature'] == row_sbs_signature) &
                    (df['mutation_type'] == mutation_type) &
                    (df['significant_strand'] == strand)][percentage_string].any():

                    strand_value = calculate_fold_change(percentage_string)
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
        color_strand2 = 2-color_strand2

    strand1_cancer_types_percentage = None
    strand2_cancer_types_percentage = None
    percentage_string = percentage_strings[0]

    if strand1 in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type]:
        cancer_types_list = signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type][strand1][percentage_string]
        all_cancer_types_list = signature2cancer_type_list_dict[row_sbs_signature]
        strand1_cancer_types_percentage = (len(cancer_types_list) / len(all_cancer_types_list)) * 100
    if strand2 in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type]:
        cancer_types_list = signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type][strand2][percentage_string]
        all_cancer_types_list = signature2cancer_type_list_dict[row_sbs_signature]
        strand2_cancer_types_percentage = (len(cancer_types_list) / len(all_cancer_types_list)) * 100
    if (strand1_cancer_types_percentage is not None) and (strand2_cancer_types_percentage is None):
        radius = calculate_radius(strand1_cancer_types_percentage)
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


# New code for the manuscript figures
# colors SBS4
# Lung: 1.1 H&N: 1.2 Liver: 1.5 Skin: 1.2 Stomach: 1.05
# 4 cancer types >= 1.1
# Circle size: 4/5
# Color: (1.1 + 1.2 + 1.5 + 1.2)/4
# Instead of 6 circles for each mutation type
# Now there is only 1 circle for each mutation type
def plot_new_six_mutations_sbs_signatures_circle_figures(sbs_signatures,
                                                        strand_bias,
                                                        strands,
                                                        cmap,
                                                        norm,
                                                        strand_bias_output_dir,
                                                        significance_level,
                                                        signature2mutation_type2strand2percent2cancertypeslist_dict,
                                                        signature2cancer_type_list_dict,
                                                        percentage_strings,
                                                        signature_transcribed_versus_untranscribed_filtered_q_value_df,
                                                        signature_genic_versus_intergenic_filtered_q_value_df,
                                                        signature_lagging_versus_leading_filtered_q_value_df):

    mutation_types = six_mutation_types

    rows_sbs_signatures=[]

    # Fill rows_sbs_signatures
    for signature in sbs_signatures:
        if signature in signature2mutation_type2strand2percent2cancertypeslist_dict:
            for mutation_type in signature2mutation_type2strand2percent2cancertypeslist_dict[signature]:
                for strand in strands:
                    if strand in signature2mutation_type2strand2percent2cancertypeslist_dict[signature][mutation_type]:
                        for percentage_string in signature2mutation_type2strand2percent2cancertypeslist_dict[signature][mutation_type][strand]:
                            if len(signature2mutation_type2strand2percent2cancertypeslist_dict[signature][mutation_type][strand][percentage_string]) > 0:
                                if signature not in rows_sbs_signatures:
                                    rows_sbs_signatures.append(signature)

    # Remove SBS mutational signatures attributed to artifacts
    rows_sbs_signatures = list(set(rows_sbs_signatures) - set(signatures_attributed_to_artifacts))

    print('%s Before sorting: %s' %(strand_bias,rows_sbs_signatures))
    rows_sbs_signatures=sorted(rows_sbs_signatures,key=natural_key,reverse=True)
    print('%s After sorting: %s' %(strand_bias,rows_sbs_signatures))

    rows_sbs_signatures_with_number_of_cancer_types = augment_with_number_of_cancer_types(SBS, rows_sbs_signatures, signature2cancer_type_list_dict)

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

    # set facecolor white
    top_axis.set_facecolor('white')

    #make aspect ratio square
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

    # Write mutation types as text
    for i in range(0, len(mutation_types), 1):
        # mutation_type
        top_axis.text(x, # text x
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
    # top_axis.set_yticklabels(rows_sbs_signatures_with_number_of_cancer_types, minor=True, fontweight='bold', fontname='Times New Roman')  # fontsize
    top_axis.set_yticklabels(rows_sbs_signatures_with_number_of_cancer_types, minor=True)

    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False)  # labels along the bottom edge are off

    # Gridlines based on major ticks
    top_axis.grid(which='major', color='black', zorder=3)

    for mutation_type_index, mutation_type in enumerate(mutation_types):
        for row_sbs_signature_index, row_sbs_signature in enumerate(rows_sbs_signatures):
            # Number of cancer types is decided based on first percentage string 1.1

            if (strand_bias == LAGGING_VERSUS_LEADING):
                if row_sbs_signature in signature2mutation_type2strand2percent2cancertypeslist_dict:
                    if mutation_type in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature]:
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
    figFile = os.path.join(strand_bias_output_dir, FIGURES_MANUSCRIPT, filename)
    fig.savefig(figFile, dpi=100, bbox_inches="tight")

    plt.cla()
    plt.close(fig)


# legacy code for the manuscript figures
def plot_six_mutations_sbs_signatures_circle_figures(sbs_signatures,
                                              strand_bias,
                                              strand_bias_output_dir,
                                              significance_level,
                                              signature2mutation_type2strand2percent2cancertypeslist_dict,
                                              signature2cancer_type_list_dict,
                                              percentage_strings):

    mutation_types = six_mutation_types

    if strand_bias==LAGGING_VERSUS_LEADING:
        strands=replication_strands
    elif strand_bias==TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        strands=transcription_strands
    elif strand_bias==GENIC_VERSUS_INTERGENIC:
        strands=genic_versus_intergenic_strands

    rows_sbs_signatures=[]

    #Fill rows_sbs_signatures
    for signature in sbs_signatures:
        if signature in signature2mutation_type2strand2percent2cancertypeslist_dict:
            for mutation_type in signature2mutation_type2strand2percent2cancertypeslist_dict[signature]:
                for strand in strands:
                    if strand in signature2mutation_type2strand2percent2cancertypeslist_dict[signature][mutation_type]:
                        for percentage_string in signature2mutation_type2strand2percent2cancertypeslist_dict[signature][mutation_type][strand]:
                            if len(signature2mutation_type2strand2percent2cancertypeslist_dict[signature][mutation_type][strand][percentage_string])>0:
                                if signature not in rows_sbs_signatures:
                                    rows_sbs_signatures.append(signature)

    # Remove SBS mutational signatures attributed to artifacts
    rows_sbs_signatures = list(set(rows_sbs_signatures) - set(signatures_attributed_to_artifacts))

    print('%s Before sorting: %s' %(strand_bias,rows_sbs_signatures))
    rows_sbs_signatures=sorted(rows_sbs_signatures,key=natural_key,reverse=True)
    print('%s After sorting: %s' %(strand_bias,rows_sbs_signatures))

    rows_sbs_signatures_with_number_of_cancer_types = augment_with_number_of_cancer_types(SBS, rows_sbs_signatures, signature2cancer_type_list_dict)

    # xticklabels_list = percentage_strings * 6
    # xticklabels_list = ['1.1', '1.2', '1.3', '1.5', '1.75', '2+'] * 6

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

    xticklabels_list = xticklabels_list * 6

    # New plot (width,height)
    fig, top_axis = plt.subplots(figsize=(5 + 1.5 * len(xticklabels_list), 10 + 1.5 * len(rows_sbs_signatures)))
    plt.rc('axes', edgecolor='lightgray')
    #make aspect ratio square
    top_axis.set_aspect(1.0)

    # set title
    if strand_bias == LAGGING_VERSUS_LEADING:
        title = 'Lagging Strand versus Leading Strand Bias'
    elif strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        title = 'Transcribed Strand versus Untranscribed Strand Bias'
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        title = 'Genic Regions versus Intergenic Regions Bias'
    top_axis.text(len(percentage_strings) * 3, len(rows_sbs_signatures) + 2.5, title, horizontalalignment='center', fontsize=60, fontweight='bold', fontname='Arial')

    # Colors are from SigProfilerPlotting tool to be consistent
    colors = [[3 / 256, 189 / 256, 239 / 256],
              [1 / 256, 1 / 256, 1 / 256],
              [228 / 256, 41 / 256, 38 / 256],
              [203 / 256, 202 / 256, 202 / 256],
              [162 / 256, 207 / 256, 99 / 256],
              [236 / 256, 199 / 256, 197 / 256]]

    # Put rectangles
    x = 0

    for i in range(0, len(mutation_types), 1):
        top_axis.text((x + (len(percentage_strings) / 2) - 0.75), len(rows_sbs_signatures) + 1.5, mutation_types[i],fontsize=55, fontweight='bold', fontname='Arial')
        top_axis.add_patch(plt.Rectangle((x + .0415, len(rows_sbs_signatures) + 0.75), len(percentage_strings) - (2 * .0415), .5,facecolor=colors[i], clip_on=False))
        top_axis.add_patch(plt.Rectangle((x, 0), len(percentage_strings), len(rows_sbs_signatures), facecolor=colors[i], zorder=0,alpha=0.25, edgecolor='grey'))
        x += len(percentage_strings)

    # CODE GOES HERE TO CENTER X-AXIS LABELS...
    top_axis.set_xlim([0, len(mutation_types) * len(percentage_strings)])
    top_axis.set_xticklabels([])

    top_axis.tick_params(axis='x', which='both', length=0, labelsize=35)

    # major ticks
    top_axis.set_xticks(np.arange(0, len(mutation_types) * len(percentage_strings), 1))
    # minor ticks
    top_axis.set_xticks(np.arange(0, len(mutation_types) * len(percentage_strings), 1) + 0.5, minor=True)

    top_axis.set_xticklabels(xticklabels_list, minor=True, fontweight='bold', fontname='Arial', fontsize=40)

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
    # top_axis.set_yticklabels(rows_sbs_signatures_with_number_of_cancer_types, minor=True, fontweight='bold', fontname='Times New Roman')  # fontsize
    top_axis.set_yticklabels(rows_sbs_signatures_with_number_of_cancer_types, minor=True)

    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False)  # labels along the bottom edge are off

    # Gridlines based on major ticks
    top_axis.grid(which='major', color='black', zorder=3)

    # Put the legend
    if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
        legend_elements = [
            Line2D([0], [0], marker='o', color='white', label='Genic: Transcribed Strand', markerfacecolor='royalblue',markersize=40),
            Line2D([0], [0], marker='o', color='white', label='Genic: Untranscribed Strand', markerfacecolor='yellowgreen',markersize=40)]
    elif strand_bias == GENIC_VERSUS_INTERGENIC:
        legend_elements = [
            Line2D([0], [0], marker='o', color='white', label='Genic Regions', markerfacecolor='cyan', markersize=40),
            Line2D([0], [0], marker='o', color='white', label='Intergenic Regions', markerfacecolor='gray', markersize=40)]
    elif (strand_bias == LAGGING_VERSUS_LEADING):
        legend_elements = [
            Line2D([0], [0], marker='o', color='white', label='Lagging Strand', markerfacecolor='indianred', markersize=40),
            Line2D([0], [0], marker='o', color='white', label='Leading Strand', markerfacecolor='goldenrod', markersize=40)]

    top_axis.legend(handles=legend_elements, ncol=len(legend_elements), bbox_to_anchor=(0, 0, 1, 0), loc='upper right', fontsize=40)

    for percentage_diff_index, percentage_string in enumerate(percentage_strings):
         for mutation_type_index, mutation_type in enumerate(mutation_types):
            for row_sbs_signature_index, row_sbs_signature in enumerate(rows_sbs_signatures):
                if (strand_bias == LAGGING_VERSUS_LEADING):
                    if row_sbs_signature in signature2mutation_type2strand2percent2cancertypeslist_dict:
                        if mutation_type in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature]:
                            lagging_cancer_types_percentage = None
                            leading_cancer_types_percentage = None

                            if LAGGING in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type]:
                                cancer_types_list = signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type][LAGGING][percentage_string]
                                all_cancer_types_list = signature2cancer_type_list_dict[row_sbs_signature]
                                lagging_cancer_types_percentage = (len(cancer_types_list) / len(all_cancer_types_list)) * 100
                            if LEADING in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type]:
                                cancer_types_list = signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type][LEADING][percentage_string]
                                all_cancer_types_list = signature2cancer_type_list_dict[row_sbs_signature]
                                leading_cancer_types_percentage = (len(cancer_types_list) / len(all_cancer_types_list)) * 100

                            if (lagging_cancer_types_percentage is not None) and (leading_cancer_types_percentage is None):
                                radius = calculate_radius(lagging_cancer_types_percentage)
                                if (radius > 0):
                                    top_axis.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5,row_sbs_signature_index + 0.5), radius,color='indianred', fill=True))
                            elif (leading_cancer_types_percentage is not None) and (lagging_cancer_types_percentage is None):
                                radius = calculate_radius(leading_cancer_types_percentage)
                                if (radius > 0):
                                    top_axis.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5,row_sbs_signature_index + 0.5), radius,color='goldenrod', fill=True))
                            elif (lagging_cancer_types_percentage is not None) and (leading_cancer_types_percentage is not None):
                                radius_lagging = calculate_radius(lagging_cancer_types_percentage)
                                radius_leading = calculate_radius(leading_cancer_types_percentage)
                                if (radius_lagging > radius_leading):
                                    # First lagging
                                    top_axis.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5,row_sbs_signature_index + 0.5), radius_lagging,color='indianred', fill=True))
                                    # Second leading
                                    top_axis.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5,row_sbs_signature_index + 0.5), radius_leading,color='goldenrod', fill=True))
                                else:
                                    # First leading
                                    top_axis.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5,row_sbs_signature_index + 0.5), radius_leading,color='goldenrod', fill=True))
                                    # Second lagging
                                    top_axis.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5,row_sbs_signature_index + 0.5), radius_lagging,color='indianred', fill=True))

                elif (strand_bias == GENIC_VERSUS_INTERGENIC):
                    if row_sbs_signature in signature2mutation_type2strand2percent2cancertypeslist_dict:
                        if mutation_type in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature]:
                            genic_cancer_types_percentage = None
                            intergenic_cancer_types_percentage = None

                            if GENIC in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type]:
                                cancer_types_list = signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type][GENIC][percentage_string]
                                all_cancer_types_list = signature2cancer_type_list_dict[row_sbs_signature]
                                genic_cancer_types_percentage = (len(cancer_types_list) / len(all_cancer_types_list)) * 100
                            if INTERGENIC in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type]:
                                cancer_types_list = signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type][INTERGENIC][percentage_string]
                                all_cancer_types_list = signature2cancer_type_list_dict[row_sbs_signature]
                                intergenic_cancer_types_percentage = (len(cancer_types_list) / len(all_cancer_types_list)) * 100

                            if (genic_cancer_types_percentage is not None) and (intergenic_cancer_types_percentage is None):
                                radius = calculate_radius(genic_cancer_types_percentage)
                                if (radius > 0):
                                    top_axis.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5,row_sbs_signature_index + 0.5), radius, color='cyan',fill=True))
                            elif (intergenic_cancer_types_percentage is not None) and (genic_cancer_types_percentage is None):
                                radius = calculate_radius(intergenic_cancer_types_percentage)
                                if (radius > 0):
                                    top_axis.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5,row_sbs_signature_index + 0.5), radius, color='gray',fill=True))
                            elif (genic_cancer_types_percentage is not None) and (intergenic_cancer_types_percentage is not None):
                                radius_genic = calculate_radius(genic_cancer_types_percentage)
                                radius_intergenic = calculate_radius(intergenic_cancer_types_percentage)
                                if (radius_genic > radius_intergenic):
                                    # First genic
                                    top_axis.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5,row_sbs_signature_index + 0.5), radius_genic,color='cyan', fill=True))
                                    # Second intergenic
                                    top_axis.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5,row_sbs_signature_index + 0.5), radius_intergenic,color='gray', fill=True))
                                else:
                                    # First intergenic
                                    top_axis.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5,row_sbs_signature_index + 0.5), radius_intergenic,color='gray', fill=True))
                                    # Second genic
                                    top_axis.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5,row_sbs_signature_index + 0.5), radius_genic,color='cyan', fill=True))
                elif (strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED):
                    if row_sbs_signature in signature2mutation_type2strand2percent2cancertypeslist_dict:
                        if mutation_type in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature]:
                            transcribed_cancer_types_percentage = None
                            untranscribed_cancer_types_percentage = None

                            if TRANSCRIBED_STRAND in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type]:
                                cancer_types_list = signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type][TRANSCRIBED_STRAND][percentage_string]
                                all_cancer_types_list = signature2cancer_type_list_dict[row_sbs_signature]
                                transcribed_cancer_types_percentage = (len(cancer_types_list) / len(all_cancer_types_list)) * 100
                            if UNTRANSCRIBED_STRAND in signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type]:
                                cancer_types_list = signature2mutation_type2strand2percent2cancertypeslist_dict[row_sbs_signature][mutation_type][UNTRANSCRIBED_STRAND][percentage_string]
                                all_cancer_types_list = signature2cancer_type_list_dict[row_sbs_signature]
                                untranscribed_cancer_types_percentage = (len(cancer_types_list) / len(all_cancer_types_list)) * 100

                            if (transcribed_cancer_types_percentage is not None) and (untranscribed_cancer_types_percentage is None):
                                radius = calculate_radius(transcribed_cancer_types_percentage)
                                if (radius > 0):
                                    top_axis.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5,row_sbs_signature_index + 0.5), radius,color='royalblue', fill=True))
                            elif (untranscribed_cancer_types_percentage is not None) and (transcribed_cancer_types_percentage is None):
                                radius = calculate_radius(untranscribed_cancer_types_percentage)
                                if (radius > 0):
                                    top_axis.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5,row_sbs_signature_index + 0.5), radius,color='yellowgreen', fill=True))
                            elif (transcribed_cancer_types_percentage is not None) and (untranscribed_cancer_types_percentage is not None):
                                radius_transcribed = calculate_radius(transcribed_cancer_types_percentage)
                                radius_untranscribed = calculate_radius(untranscribed_cancer_types_percentage)
                                if (radius_transcribed > radius_untranscribed):
                                    # First transcribed
                                    top_axis.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5,row_sbs_signature_index + 0.5), radius_transcribed,color='royalblue', fill=True))
                                    # Second untranscribed
                                    top_axis.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5,row_sbs_signature_index + 0.5), radius_untranscribed,color='yellowgreen', fill=True))
                                else:
                                    # First untranscribed
                                    top_axis.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5,row_sbs_signature_index + 0.5), radius_untranscribed,color='yellowgreen', fill=True))
                                    # Second transcribed
                                    top_axis.add_patch(plt.Circle((mutation_type_index * len(percentage_strings) + percentage_diff_index + 0.5,row_sbs_signature_index + 0.5), radius_transcribed,color='royalblue', fill=True))

    # create the directory if it does not exists
    filename = 'SBS_Signatures_%s_with_circles_%s.png' % (strand_bias,str(significance_level).replace('.','_'))
    figFile = os.path.join(strand_bias_output_dir, FIGURES_MANUSCRIPT, filename)
    fig.savefig(figFile, dpi=100, bbox_inches="tight")

    plt.cla()
    plt.close(fig)