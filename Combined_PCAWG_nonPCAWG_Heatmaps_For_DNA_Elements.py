# !/usr/bin/env python3

# Author: burcakotlu

# Contact: burcakotlu@eng.ucsd.edu

import os
import numpy as np
import multiprocessing
import pandas as pd
import matplotlib.colors as colors

from SigProfilerTopography.source.commons.TopographyCommons import readDictionary
from SigProfilerTopography.source.commons.TopographyCommons import DATA
from SigProfilerTopography.source.commons.TopographyCommons import SIGNATUREBASED

from SigProfilerTopography.source.commons.TopographyCommons import AGGREGATEDSUBSTITUTIONS
from SigProfilerTopography.source.commons.TopographyCommons import AGGREGATEDINDELS
from SigProfilerTopography.source.commons.TopographyCommons import AGGREGATEDDINUCS

from SigProfilerTopography.source.commons.TopographyCommons import EPIGENOMICSOCCUPANCY
from SigProfilerTopography.source.commons.TopographyCommons import NUCLEOSOMEOCCUPANCY

from SigProfilerTopography.source.commons.TopographyCommons import AVERAGE_SIGNAL_ARRAY
from SigProfilerTopography.source.commons.TopographyCommons import ACCUMULATED_COUNT_ARRAY
from SigProfilerTopography.source.commons.TopographyCommons import Table_MutationType_NumberofMutations_NumberofSamples_SamplesList_Filename
from SigProfilerTopography.source.commons.TopographyCommons import calculate_pvalue_teststatistics

from SigProfilerTopography.source.plotting.OccupancyAverageSignalFigures import readData
from SigProfilerTopography.source.plotting.OccupancyAverageSignalFigures import readDataForSimulations

from Combined_Common import cancer_type_2_NCI_Thesaurus_code_dict
from Combined_Common import signatures_attributed_to_artifacts
from Combined_Common import COSMIC_HISTONE_MODIFICATIONS
from Combined_Common import deleteOldData
from Combined_Common import natural_key
from Combined_Common import fill_lists
from Combined_Common import depleted
from Combined_Common import enriched
from Combined_Common import OCCUPANCY_HEATMAP_COMMON_MULTIPLIER

from Combined_Common import LYMPH_BNHL
from Combined_Common import LYMPH_BNHL_CLUSTERED
from Combined_Common import LYMPH_BNHL_NONCLUSTERED
from Combined_Common import LYMPH_CLL
from Combined_Common import LYMPH_CLL_CLUSTERED
from Combined_Common import LYMPH_CLL_NONCLUSTERED
from Combined_Common import ALTERNATIVE_OUTPUT_DIR
from Combined_Common import get_alternative_combined_output_dir_and_cancer_type
from Combined_Common import NUMBER_OF_DECIMAL_PLACES_TO_ROUND

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

import json
import statsmodels.stats.multitest
from decimal import Decimal

import scipy
from scipy.stats import wilcoxon

COMBINE_P_VALUES_METHOD_FISHER = 'fisher'
COMBINE_P_VALUES_METHOD_STOUFFER = 'stouffer'

SBS = 'SBS'
DBS = 'DBS'
ID = 'ID'

SUBS = 'SUBS'
DINUCS = 'DINUCS'
INDELS = 'INDELS'

nucleosome_center = 1000
epigenomics_center = 1000

WEIGHTED_AVERAGE_METHOD = 'WEIGHTED_AVERAGE_METHOD'

#We will be using this fold _change method
TEST_REAL_OVER_SIM_500 = '500'
REAL_OVER_SIM_500 = '500'
REAL_OVER_SIM_100 = '100'
REAL_OVER_SIM_100_WILCOXON = 'real_over_sim_100_wilcoxon'
REAL_OVER_SIM_100_COMBINED_Q_VALUES = 'real_over_sim_100_combined_q_values'

TWO_HUNDRED_FIFTY = 250
FIFTY = 50

COLORBAR_SEISMIC = 'seismic'
COLORBAR_DISCREET = 'discreet'

NUCLEOSOME = 'Nucleosome'
NUCLEOSOME_BIOSAMPLE = "K562"

CTCF = "CTCF"

ALL_SUBSTITUTIONS = 'All Substitutions'
ALL_DINUCLEOTIDES = 'All Dinucleotides'
ALL_INDELS = 'All Indels'

NUMBER_OF_CANCER_TYPES_THAT_HAS_DEPLETED = 'NUMBER_OF_CANCER_TYPES_DEPLETED'
NUMBER_OF_CANCER_TYPES_THAT_HAS_ENRICHED = 'NUMBER_OF_CANCRER_TYPES_ENRICHED'
NUMBER_OF_CANCER_TYPES = 'NUMBER_OF_CANCRER_TYPES'

ACROSS_ALL_CANCER_TYPES = 'ACROSS_ALL_CANCER_TYPES'

ONE_HUNDRED_TWO = 102
ONE_POINT_EIGHT = 1.08

COMBINED_PCAWG_NONPCAWG = 'combined_pcawg_nonpcawg'
ALL_MUTATIONS = [ALL_SUBSTITUTIONS, ALL_INDELS, ALL_DINUCLEOTIDES]

NORMAL = 'Normal'
CANCER = 'Cancer'

TSCC = 'tscc'
AWS = 'aws'

COSMIC = 'Cosmic'
MANUSCRIPT = 'Manuscript'

FIGURES_COSMIC = 'figures_cosmic'
COSMIC_TISSUE_BASED_FIGURES = 'cosmic_tissue_based_figures'
FIGURES_MANUSCRIPT = 'figures_manuscript'

EXCEL_FILES = 'excel_files'
TABLES = 'tables'
DATA_FILES = 'data_files'
DICTIONARIES = 'dictionaries'

ENRICHED_CANCER_TYPES = 'enriched_cancer_types'
DEPLETED_CANCER_TYPES = 'depleted_cancer_types'
OTHER_CANCER_TYPES = 'other_cancer_types'

ENRICHMENT_OF_MUTATIONS = 'Enrichment of mutations'
DEPLETION_OF_MUTATIONS = 'Depletion of mutations'
NO_EFFECT = 'No effect'
NO_EFFECT_BASED_ON_EXPECTED_BY_CHANCE = "No effect based on expected by chance"

AT_LEAST_1K_CONSRAINTS = 1000
AT_LEAST_20K_CONSRAINTS = 20000

def calculate_radius(percentage_of_cancer_types):
    #To fit in a cell in the heatmap or grid
    radius = (percentage_of_cancer_types / ONE_HUNDRED_TWO) / 2
    return radius

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def heatmap(data, row_labels, col_labels, x_axis_labels_on_bottom = True, ax=None, fontsize=90, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # nans are handled here
    data = np.ma.masked_invalid(data)
    ax.patch.set(hatch='x', edgecolor='black')

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar and display (If you want to display the colorbar uncomment below)
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",fontsize=80,labelpad=25)
    # cbar.ax.tick_params(labelsize=80)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.

    if x_axis_labels_on_bottom:
        ax.set_xticklabels(col_labels, fontsize=fontsize)
        ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(row_labels, fontsize=fontsize)
    else:
        ax.set_xticklabels(col_labels, fontsize=fontsize)
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, pad=5)
        plt.setp(ax.get_xticklabels(), rotation=50, ha="left", rotation_mode="anchor")
        ax.set_yticklabels(row_labels, fontsize=fontsize)

    # Turn spines off and create white grid.
    # for edge, spine in ax.spines.items():
        # spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)

    ax.grid(which="minor", color="black", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # return im, cbar
    return im


# New way of getting encode elements associated with a biosample
def get_encode_elements_using_listdir(biosample, dir_path):
    list_of_entries = os.listdir(dir_path)
    _biosample_ = "_%s_" %(biosample)
    encode_elements = []

    for entry in list_of_entries:
        if os.path.isfile(os.path.join(dir_path,entry)) and (_biosample_ in entry):
            encode_elements.append(os.path.splitext(entry)[0])

    return encode_elements

# This will be used.
# This dictionary is very important
normal_combined_pcawg_nonpcawg_cancer_type_2_biosample_dict={'Eso-AdenoCA':['stomach', 'gastroesophageal-sphincter'],
                                                      'ESCC':['epithelial-cell-of-esophagus', 'esophagus-squamous-epithelium'],
                                                      'Lung-AdenoCA':['bronchial-epithelial-cell', 'lung', 'upper-lobe-of-left-lung'],
                                                      'Lung-SCC':['bronchial-epithelial-cell', 'lung', 'upper-lobe-of-left-lung'],
                                                      'Stomach-AdenoCA':['stomach', 'gastroesophageal-sphincter', 'stomach-smooth-muscle', 'mucosa-of-stomach', 'duodenal-mucosa'],
                                                      'Breast-Cancer':['breast-epithelium',  'mammary-epithelial-cell', 'myoepithelial-cell-of-mammary-gland',  'luminal-epithelial-cell-of-mammary-gland', 'mammary-stem-cell'],
                                                      'Liver-HCC':['liver', 'right-lobe-of-liver'],
                                                      'Prost-AdenoCA':['epithelial-cell-of-prostate',  'prostate-gland'],
                                                      'CNS-Oligo':['temporal-lobe', 'caudate-nucleus',
                                                                   'layer-of-hippocampus', 'cingulate-gyrus',
                                                                   'angular-gyrus', 'astrocyte',
                                                                   'astrocyte-of-the-cerebellum', 'brain',
                                                                   'cerebellum',
                                                                   'choroid-plexus-epithelial-cell',
                                                                   'posterior-cingulate-cortex',
                                                                   'substantia-nigra', 'germinal-matrix'],
                                                      'CNS-GBM': ['temporal-lobe', 'caudate-nucleus',
                                                                    'layer-of-hippocampus', 'cingulate-gyrus',
                                                                    'angular-gyrus', 'astrocyte',
                                                                    'astrocyte-of-the-cerebellum', 'brain',
                                                                    'cerebellum',
                                                                    'choroid-plexus-epithelial-cell',
                                                                    'posterior-cingulate-cortex',
                                                                    'substantia-nigra', 'germinal-matrix'],
                                                      'CNS-Medullo': ['temporal-lobe', 'caudate-nucleus',
                                                                    'layer-of-hippocampus', 'cingulate-gyrus',
                                                                    'angular-gyrus', 'astrocyte',
                                                                    'astrocyte-of-the-cerebellum', 'brain',
                                                                    'cerebellum',
                                                                    'choroid-plexus-epithelial-cell',
                                                                    'posterior-cingulate-cortex',
                                                                    'substantia-nigra', 'germinal-matrix'],
                                                      'CNS-PiloAstro': ['temporal-lobe', 'caudate-nucleus',
                                                                    'layer-of-hippocampus', 'cingulate-gyrus',
                                                                    'angular-gyrus', 'astrocyte',
                                                                    'astrocyte-of-the-cerebellum', 'brain',
                                                                    'cerebellum',
                                                                    'choroid-plexus-epithelial-cell',
                                                                    'posterior-cingulate-cortex',
                                                                    'substantia-nigra', 'germinal-matrix'],
                                                      'CNS-LGG': ['temporal-lobe', 'caudate-nucleus',
                                                                    'layer-of-hippocampus', 'cingulate-gyrus',
                                                                    'angular-gyrus', 'astrocyte',
                                                                    'astrocyte-of-the-cerebellum', 'brain',
                                                                    'cerebellum',
                                                                    'choroid-plexus-epithelial-cell',
                                                                    'posterior-cingulate-cortex',
                                                                    'substantia-nigra', 'germinal-matrix'],
                                                      'Ovary-AdenoCA':['ovary'],
                                                      'Uterus-AdenoCA':['uterus'],
                                                      'Cervix-Cancer':['vagina'],
                                                      'ColoRect-AdenoCA':['large-intestine', 'small-intestine', 'sigmoid-colon', 'transverse-colon', 'colonic-mucosa', 'muscle-layer-of-colon', 'mucosa-of-rectum', 'rectal-smooth-muscle-tissue'],
                                                      'Lymph-CLL':['B-cell'],
                                                      'Lymph-BNHL':['B-cell'],
                                                      'ALL':['B-cell'],
                                                      'Kidney-RCC':['kidney', 'kidney-epithelial-cell'],
                                                      'Kidney-ChRCC':['kidney', 'kidney-epithelial-cell'],
                                                      'Panc-AdenoCA':['pancreas', 'body-of-pancreas', 'endocrine-pancreas'],
                                                      'Panc-Endocrine':['pancreas', 'body-of-pancreas', 'endocrine-pancreas'],
                                                      'SoftTissue-Liposarc':['right-atrium-auricular-region', 'omental-fat-pad',
                                                                             'subcutaneous-adipose-tissue', 'subcutaneous-abdominal-adipose-tissue', 'adipose-tissue',
                                                                             'muscle-layer-of-colon', 'muscle-layer-of-duodenum', 'muscle-of-leg',  'muscle-of-trunk',
                                                                             'psoas-muscle', 'rectal-smooth-muscle-tissue',
                                                                             'skeletal-muscle-cell', 'skeletal-muscle-myoblast', 'skeletal-muscle-satellite-cell', 'skeletal-muscle-tissue',
                                                                             'stomach-smooth-muscle'],
                                                      'SoftTissue-Leiomyo':[],
                                                      'Myeloid-AML':[],
                                                      'Myeloid-MPN':[],
                                                      'Myeloid-MDS':[],
                                                      'Blood-CMDI':[],
                                                      'Skin-Melanoma':['foreskin-fibroblast', 'foreskin-melanocyte', 'lower-leg-skin'],
                                                      'Head-SCC':['thyroid-gland'],
                                                      'Thy-AdenoCA':['thyroid-gland'],
                                                      'Biliary-AdenoCA':['liver', 'right-lobe-of-liver'],
                                                      'Bone-Osteosarc':['osteoblast'],
                                                      'Bone-Benign':['osteoblast'],
                                                      'Bone-Epith':['osteoblast'],
                                                      'Ewings':['osteoblast'],
                                                      'Bladder-TCC':['urinary-bladder'],
                                                      'Eye-Melanoma':['retinal-pigment-epithelial-cell']
                                                      }


# input:  ENCFF971YPT_kidney_Normal_H3K4me3-human
# output: H3K4me3-human
# input:  ENCFF971YPT_kidney_Normal_CTCF-human
# output: CTCF-human
# input:  ENCFF971YPT_kidney_Normal_ATAC-seq
# output: ATAC-seq
def get_dna_element(dna_element):
    # return (hm.split('_')[3]).split('-')[0]
    if '_' in dna_element:
        return (dna_element.split('_')[3])
    else:
        return dna_element


# Step3 Apply  Multiple Tests Correction
# Beware step2_signature2cancer_type2dna_element2combined_p_value_list_dict
# combined p value list
# [encode_element_list, fold_change_list,avg_fold_change,p_value_list,combined_p_value]
def step3_apply_multiple_tests_correction(step2_signature2cancer_type2dna_element2combined_p_value_list_dict,heatmaps_main_output_path):
    step3_signature2cancer_type2dna_element2q_value_list_dict = {}

    all_p_values = []
    all_p_values_element_names = []
    all_FDR_BH_adjusted_p_values = None

    for signature in step2_signature2cancer_type2dna_element2combined_p_value_list_dict:
        for cancer_type in step2_signature2cancer_type2dna_element2combined_p_value_list_dict[signature]:
            for dna_element in step2_signature2cancer_type2dna_element2combined_p_value_list_dict[signature][cancer_type]:
                combined_p_value = step2_signature2cancer_type2dna_element2combined_p_value_list_dict[signature][cancer_type][dna_element][4]
                if (combined_p_value is not None) and (not np.isnan(np.array([combined_p_value], dtype=np.float)).any()) and (str(combined_p_value)!='nan'):
                    element_name = (signature, cancer_type, dna_element)
                    all_p_values.append(combined_p_value)
                    all_p_values_element_names.append(element_name)
                else:
                    print('combined_p_value is None or nan: %s %s %s %s' % (signature, cancer_type, dna_element, step2_signature2cancer_type2dna_element2combined_p_value_list_dict[signature][cancer_type][dna_element]))

    all_p_values_array = np.asarray(all_p_values)

    # If there a p_values in the array
    if(all_p_values_array.size>0):
        rejected, all_FDR_BH_adjusted_p_values, alphacSidak, alphacBonf = statsmodels.stats.multitest.multipletests(all_p_values_array, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)

    print('#######################################')
    print('len(all_p_values):%d' %len(all_p_values))

    if ((all_FDR_BH_adjusted_p_values is not None) and (all_FDR_BH_adjusted_p_values.size>0)):
        print('len(all_FDR_BH_adjusted_p_values):%d' %(len(all_FDR_BH_adjusted_p_values)))
    print('#######################################')

    for element_index, element_name in enumerate(all_p_values_element_names,0):
        signature, cancer_type, dna_element = element_name
        q_value = all_FDR_BH_adjusted_p_values[element_index]

        # combined_p_value_list
        # [fold_change_list,avg_fold_change,p_value_list,combined_p_value]
        encode_element_list = step2_signature2cancer_type2dna_element2combined_p_value_list_dict[signature][cancer_type][dna_element][0]
        fold_change_list = step2_signature2cancer_type2dna_element2combined_p_value_list_dict[signature][cancer_type][dna_element][1]
        avg_fold_change = step2_signature2cancer_type2dna_element2combined_p_value_list_dict[signature][cancer_type][dna_element][2]
        p_value_list = step2_signature2cancer_type2dna_element2combined_p_value_list_dict[signature][cancer_type][dna_element][3]
        combined_p_value = step2_signature2cancer_type2dna_element2combined_p_value_list_dict[signature][cancer_type][dna_element][4]

        # round
        fold_change_list = np.around(fold_change_list, NUMBER_OF_DECIMAL_PLACES_TO_ROUND).tolist()
        avg_fold_change = np.around(avg_fold_change, NUMBER_OF_DECIMAL_PLACES_TO_ROUND)

        if signature in step3_signature2cancer_type2dna_element2q_value_list_dict:
            if cancer_type in step3_signature2cancer_type2dna_element2q_value_list_dict[signature]:
                if dna_element in step3_signature2cancer_type2dna_element2q_value_list_dict[signature][cancer_type]:
                    print('There is a situation')
                else:
                    step3_signature2cancer_type2dna_element2q_value_list_dict[signature][cancer_type][dna_element]=[encode_element_list, fold_change_list, avg_fold_change, p_value_list, combined_p_value, q_value]
            else:
                step3_signature2cancer_type2dna_element2q_value_list_dict[signature][cancer_type] = {}
                step3_signature2cancer_type2dna_element2q_value_list_dict[signature][cancer_type][dna_element]=[encode_element_list, fold_change_list, avg_fold_change, p_value_list, combined_p_value, q_value]
        else:
            step3_signature2cancer_type2dna_element2q_value_list_dict[signature] = {}
            step3_signature2cancer_type2dna_element2q_value_list_dict[signature][cancer_type] = {}
            step3_signature2cancer_type2dna_element2q_value_list_dict[signature][cancer_type][dna_element] = [encode_element_list, fold_change_list, avg_fold_change, p_value_list, combined_p_value, q_value]

    # Write dictionary as a dataframe
    df_filename = 'Step3_Signature_CancerType_DNAElement_QValue.txt'
    filepath = os.path.join(heatmaps_main_output_path, TABLES, df_filename)
    step3_q_value_df = write_dictionary_as_dataframe_step3_q_value(step3_signature2cancer_type2dna_element2q_value_list_dict,filepath)

    return step3_q_value_df, step3_signature2cancer_type2dna_element2q_value_list_dict


def fill_cancer_type_mutation_type_number_of_mutations_df(cancer_types,combined_output_dir):
    df_list = []

    for cancer_type in cancer_types:
        mutation_type_number_of_mutations_df = pd.read_csv(os.path.join(os.path.join(combined_output_dir,cancer_type,DATA,'Table_MutationType_NumberofMutations_NumberofSamples_SamplesList.txt')),sep='\t')
        mutation_type_number_of_mutations_df.insert(0, 'cancer_type', cancer_type)
        df_list.append(mutation_type_number_of_mutations_df)

        cancer_type_mutation_type_number_of_mutations_df = pd.concat(df_list, ignore_index=True,axis=0)

    return cancer_type_mutation_type_number_of_mutations_df


# Updated for lymphoid samples
def fill_cancer_type_signature_cutoff_average_probability_df(cancer_types, combined_output_dir):
    df_list=[]

    for cancer_type in cancer_types:
        cancer_type_sbs_signature_cutoff_average_probability_df = None
        cancer_type_dbs_signature_cutoff_average_probability_df = None
        cancer_type_id_signature_cutoff_average_probability_df = None
        temp_list = []

        if cancer_type == LYMPH_BNHL or cancer_type == LYMPH_CLL:
            Table_SBS_Clustered_Signature_Cutoff_NumberofMutations_AverageProbability_path = os.path.join(ALTERNATIVE_OUTPUT_DIR, '%s_clustered' %(cancer_type), DATA, 'Table_SBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt')
            Table_SBS_NonClustered_Signature_Cutoff_NumberofMutations_AverageProbability_path = os.path.join(ALTERNATIVE_OUTPUT_DIR, '%s_nonClustered' %(cancer_type), DATA, 'Table_SBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt')

            if os.path.exists(Table_SBS_Clustered_Signature_Cutoff_NumberofMutations_AverageProbability_path):
                clustered_df = pd.read_csv(Table_SBS_Clustered_Signature_Cutoff_NumberofMutations_AverageProbability_path, sep='\t')
                clustered_df['cancer_type'] = cancer_type
                clustered_df = clustered_df[clustered_df['signature'].isin(['SBS37', 'SBS84', 'SBS85'])]
            if os.path.exists(Table_SBS_NonClustered_Signature_Cutoff_NumberofMutations_AverageProbability_path):
                nonClustered_df = pd.read_csv(Table_SBS_NonClustered_Signature_Cutoff_NumberofMutations_AverageProbability_path, sep='\t')
                nonClustered_df['cancer_type'] = cancer_type

            cancer_type_sbs_signature_cutoff_average_probability_df = pd.concat([clustered_df,nonClustered_df], axis=0)

        else:
            Table_SBS_Signature_Cutoff_NumberofMutations_AverageProbability_path = os.path.join(combined_output_dir, cancer_type, DATA, 'Table_SBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt')
            if os.path.exists(Table_SBS_Signature_Cutoff_NumberofMutations_AverageProbability_path):
                cancer_type_sbs_signature_cutoff_average_probability_df = pd.read_csv(Table_SBS_Signature_Cutoff_NumberofMutations_AverageProbability_path, sep='\t')

        Table_DBS_Signature_Cutoff_NumberofMutations_AverageProbability_path = os.path.join(combined_output_dir, cancer_type, DATA, 'Table_DBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt')
        Table_ID_Signature_Cutoff_NumberofMutations_AverageProbability_path = os.path.join(combined_output_dir, cancer_type, DATA, 'Table_ID_Signature_Cutoff_NumberofMutations_AverageProbability.txt')

        if os.path.exists(Table_DBS_Signature_Cutoff_NumberofMutations_AverageProbability_path):
            cancer_type_dbs_signature_cutoff_average_probability_df = pd.read_csv(Table_DBS_Signature_Cutoff_NumberofMutations_AverageProbability_path, sep='\t')

        if os.path.exists(Table_ID_Signature_Cutoff_NumberofMutations_AverageProbability_path):
            cancer_type_id_signature_cutoff_average_probability_df = pd.read_csv(Table_ID_Signature_Cutoff_NumberofMutations_AverageProbability_path, sep='\t')

        if cancer_type_sbs_signature_cutoff_average_probability_df is not None:
            temp_list.append(cancer_type_sbs_signature_cutoff_average_probability_df)

        if cancer_type_dbs_signature_cutoff_average_probability_df is not None:
            temp_list.append(cancer_type_dbs_signature_cutoff_average_probability_df)

        if cancer_type_id_signature_cutoff_average_probability_df is not None:
            temp_list.append(cancer_type_id_signature_cutoff_average_probability_df)

        if len(temp_list)>0:
            cancer_type_df = pd.concat(temp_list, ignore_index=True, axis=0)
            df_list.append(cancer_type_df)

    cancer_type_signature_cutoff_number_of_mutations_average_probability_df = pd.concat(df_list, ignore_index=True,axis=0)

    return cancer_type_signature_cutoff_number_of_mutations_average_probability_df


# Complete list with p value
#[signature, cancer_type, cutoff, number_of_mutations, biosample, dna_element,
# avg_real_signal, avg_sim_signal, fold_change, min_sim_signal, max_sim_signal,
# pvalue, num_of_sims, num_of_sims_with_not_nan_avgs, list(simulationsHorizontalMeans)]
def write_dictionary_as_dataframe_step1_p_value(step1_signature2CancerType2Biosample2DNAElement2PValueDict,filepath):
    L = sorted([(signature, cancer_type, complete_list[2], complete_list[3], biosample, dna_element,
                 complete_list[6], complete_list[7], complete_list[8], complete_list[9], complete_list[10],
                 complete_list[11], complete_list[12], complete_list[13], complete_list[14], complete_list[15],
                 complete_list[16])
                for signature, a in step1_signature2CancerType2Biosample2DNAElement2PValueDict.items()
                 for cancer_type, b in a.items()
                  for biosample, c in b.items()
                   for dna_element, complete_list in c.items()])
    df = pd.DataFrame(L, columns=['signature', 'cancer_type', 'cutoff', 'number_of_mutations', 'biosample', 'dna_element',
                                  'avg_real_signal','avg_simulated_signal', 'fold_change', 'min_sim_signal', 'max_sim_signal',
                                  'p_value', 'num_of_sims', 'num_of_sims_with_not_nan_avgs', 'real_data_avg_count', 'sim_avg_count', 'sim_signals'])
    df.to_csv(filepath, sep='\t', header=True, index=False)

    return df

# Combined p value
# [fold_change_list,avg_fold_change,p_value_list,combined_p_value]
def write_dictionary_as_dataframe_step2_combined_p_value(signature2cancer_type2dna_element2combined_p_value_list_dict,filepath):
    L = sorted([(signature, cancer_type, dna_element, combined_p_value_list[0], combined_p_value_list[1],combined_p_value_list[2], combined_p_value_list[3], combined_p_value_list[4])
                for signature, a in signature2cancer_type2dna_element2combined_p_value_list_dict.items()
                 for cancer_type, b in a.items()
                  for dna_element, combined_p_value_list in b.items()])
    df = pd.DataFrame(L, columns=['signature', 'cancer_type', 'dna_element', 'encode_element_list', 'fold_change_list', 'avg_fold_change' , 'p_value_list', 'combined_p_value'])
    df.to_csv(filepath, sep='\t', header=True, index=False)

    return df


# Q Value List
# [fold_change_list,avg_fold_change,p_value_list,combined_p_value,q_value]
def write_dictionary_as_dataframe_step3_q_value(step3_signature2cancer_type2dna_element2q_value_list_dict,filepath):
    L = sorted([(signature, cancer_type, dna_element, q_value_list[0], q_value_list[1], q_value_list[2], q_value_list[3], q_value_list[4], q_value_list[5])
                for signature, a in step3_signature2cancer_type2dna_element2q_value_list_dict.items()
                 for cancer_type, b in a.items()
                  for dna_element, q_value_list in b.items()])
    df = pd.DataFrame(L, columns=['signature', 'cancer_type', 'dna_element',
                                  'encode_element_list', 'fold_change_list', 'avg_fold_change', 'p_value_list', 'combined_p_value', 'q_value'])
    df.to_csv(filepath, sep='\t', header=True, index=False)

    return df


def there_is_a_result_to_show(signatureType,
                              all_dna_elements,
                            all_signatures,
                            cancer_types,
                            step3_q_value_df,
                            cancer_type_signature_cutoff_number_of_mutations_average_probability_df,
                            enriched_fold_change,
                            depleted_fold_change,
                            significance_level):

    # In COSMIC case all_signatures will contain only one signature
    signatures, \
    signature_signature_type_tuples, \
    signature_tissue_type_tuples, \
    signatures_ylabels_on_the_heatmap = fill_lists(all_signatures[0],
                                                   signatureType,
                                                   cancer_type_signature_cutoff_number_of_mutations_average_probability_df)

    # Update all_signatures
    # We want to have as many rows in heatmap as signatures
    all_signatures = signatures

    # Same for COSMIC and MANUSCRIPT
    signature2dna_element2cancer_type_list_dict, \
    considered_dna_elements, \
    considered_signatures = fill_signature2dna_element2cancer_type_list_dict(all_dna_elements,
                                                                             all_signatures,
                                                                             cancer_types,
                                                                             step3_q_value_df,
                                                                             enriched_fold_change,
                                                                             depleted_fold_change,
                                                                             significance_level)


    for signature in signature2dna_element2cancer_type_list_dict:
        for dna_element in signature2dna_element2cancer_type_list_dict[signature]:
            if len(signature2dna_element2cancer_type_list_dict[signature][dna_element][ENRICHED_CANCER_TYPES]) > 0:
                return True
            elif len(signature2dna_element2cancer_type_list_dict[signature][dna_element][DEPLETED_CANCER_TYPES]) > 0:
                return True
            elif len(signature2dna_element2cancer_type_list_dict[signature][dna_element][OTHER_CANCER_TYPES]) > 0:
                return True

    return False


def compute_fold_change_significance_plot_heatmap(combine_p_values_method,
                                                  window_size,
                                                  tissue_type,
                                                  combined_output_dir,
                                                  numberofSimulations,
                                                  hm_path,
                                                  ctcf_path,
                                                  atac_path,
                                                  heatmaps_main_output_path,
                                                  cancer_type_signature_cutoff_number_of_mutations_average_probability_df,
                                                  cancer_types,
                                                  sbs_signatures,
                                                  id_signatures,
                                                  dbs_signatures,
                                                  depleted_fold_change,
                                                  enriched_fold_change,
                                                  step1_data_ready,
                                                  significance_level,
                                                  minimum_number_of_overlaps_required_for_sbs,
                                                  minimum_number_of_overlaps_required_for_dbs,
                                                  minimum_number_of_overlaps_required_for_indels,
                                                  figure_types,
                                                  cosmic_release_version,
                                                  figure_file_extension,
                                                  signature_cancer_type_number_of_mutations,
                                                  signature_cancer_type_number_of_mutations_for_ctcf,
                                                  consider_both_real_and_sim_avg_overlap,
                                                  sort_cancer_types,
                                                  remove_columns_rows_with_no_significant_result,
                                                  heatmap_rows_signatures_columns_dna_elements):

    signatures = []
    signatures.extend(sbs_signatures)
    signatures.extend(dbs_signatures)
    signatures.extend(id_signatures)

    signature_tuples = []
    for signature in sbs_signatures:
        signature_tuples.append((signature,SBS))
    for signature in dbs_signatures:
        signature_tuples.append((signature,DBS))
    for signature in id_signatures:
        signature_tuples.append((signature,ID))

    if step1_data_ready:
        # Read Step1_Signature2CancerType2Biosample2DNAElement2PValue_Dict.txt
        dictFilename = 'Step1_Signature2CancerType2Biosample2DNAElement2PValue_Dict.txt'
        dictPath = os.path.join(heatmaps_main_output_path, DICTIONARIES, dictFilename)
        step1_signature2cancer_type2biosample2dna_element2p_value_tuple_dict = readDictionary(dictPath)

        df_filename = 'Step1_Signature_CancerType_Biosample_DNAElement_PValue.txt'
        filepath = os.path.join(heatmaps_main_output_path, TABLES, df_filename)
        step1_p_value_df = write_dictionary_as_dataframe_step1_p_value(step1_signature2cancer_type2biosample2dna_element2p_value_tuple_dict, filepath)

    else:
        # Step1 Calculate p value using z-test
        # Epigenomics Signatures
        # Epigenomics All Mutations (SUBS, INDELS, DINUCS)
        # Nucleosome Signatures
        # Nucleosome All Mutations (SUBS, INDELS, DINUCS)
        # Complete P Value List
        # [signature, cancer_type, biosample, dna_element, avg_real_signal, avg_sim_signal, fold_change, min_sim_signal, max_sim_signal, pvalue, num_of_sims, num_of_sims_with_not_nan_avgs, real_data_avg_count, sim_avg_count, list(simulationsHorizontalMeans)]
        if (tissue_type == NORMAL):
            step1_p_value_df, \
            step1_signature2cancer_type2biosample2dna_element2p_value_tuple_dict = step1_compute_p_value(window_size,
                                                                                                   combined_output_dir,
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
    step2_combined_p_value_df, \
    step2_signature2cancer_type2dna_element2combined_p_value_list_dict = step2_combine_p_values(
        step1_signature2cancer_type2biosample2dna_element2p_value_tuple_dict,
        heatmaps_main_output_path,
        combine_p_values_method,
        signature_tuples,
        depleted_fold_change,
        enriched_fold_change,
        minimum_number_of_overlaps_required_for_sbs,
        minimum_number_of_overlaps_required_for_dbs,
        minimum_number_of_overlaps_required_for_indels,
        signature_cancer_type_number_of_mutations,
        signature_cancer_type_number_of_mutations_for_ctcf,
        consider_both_real_and_sim_avg_overlap = consider_both_real_and_sim_avg_overlap)

    # Step3 Corrected combined p values
    # combined p value list
    # [fold_change_list,avg_fold_change,p_value_list,combined_p_value]
    step3_q_value_df, \
    step3_signature2cancer_type2dna_element2q_value_list_dict = step3_apply_multiple_tests_correction(step2_signature2cancer_type2dna_element2combined_p_value_list_dict,
                                                                                                      heatmaps_main_output_path)
    # FEB 11 2021
    # plot heatmap rows signatures columns cancer_types for each DNA element
    # It can use either step3_signature2cancer_type2dna_element2q_value_list_dict or step4_signature2cancer_type2dna_element2filtered_q_value_list_dict

    # Signatures are sorted here
    signatures = []
    signatures.extend(sbs_signatures[:-1])
    signatures.extend(dbs_signatures[:-1])
    signatures.extend(id_signatures[:-1])
    signatures.append(sbs_signatures[-1])
    signatures.append(dbs_signatures[-1])
    signatures.append(id_signatures[-1])

    if sort_cancer_types:
        cancer_types = sorted(cancer_types, key=natural_key)

    all_dna_elements = step3_q_value_df['dna_element'].unique()
    all_dna_elements = sorted(all_dna_elements, key=natural_key)

    # For Figures_Manuscript
    # Remove  'ATAC', 'CTCF', and 'Nucleosome' as this reads as a repetition of the previous figures/sections.
    # 'ATAC' is replaced by 'Open Chromatin' in heatmap_with_pie_chart method
    print('For information -- before remove all_dna_elements:', all_dna_elements)
    if 'CTCF-human' in all_dna_elements:
        all_dna_elements.remove('CTCF-human')
    if 'Nucleosome' in all_dna_elements:
        all_dna_elements.remove('Nucleosome')
    if 'ATAC-seq' in all_dna_elements:
        all_dna_elements.remove('ATAC-seq')
    print('For information -- after remove all_dna_elements:', all_dna_elements)

    # Uncomment in real run starts

    # Plot heatmap for each DNA element
    name_for_group_by = 'dna_element'
    name_for_rows = 'signature'
    name_for_columns = 'cancer_type'
    all_rows = signatures
    all_columns = cancer_types

    # Remove artifact signatures
    for artifact_signature in signatures_attributed_to_artifacts:
        if artifact_signature in all_rows:
            all_rows.remove(artifact_signature)

    os.makedirs(os.path.join(heatmaps_main_output_path,'heatmaps_for_each_dna_element'), exist_ok=True)
    heatmap_output_path = os.path.join(heatmaps_main_output_path,'heatmaps_for_each_dna_element')

    prepare_array_plot_heatmap_main(name_for_group_by,
                 name_for_rows,
                 name_for_columns,
                 all_rows,
                 all_columns,
                 step3_q_value_df,
                 heatmap_output_path,
                 depleted_fold_change,
                 enriched_fold_change,
                 significance_level,
                 remove_columns_rows_with_no_significant_result)

    # Plot heatmap for each signature
    name_for_group_by = 'signature'
    name_for_rows = 'dna_element'
    name_for_columns = 'cancer_type'
    all_rows = all_dna_elements
    all_columns = cancer_types

    # # For Graphical Abstract Figure
    # name_for_group_by = 'signature'
    # name_for_rows = 'cancer_type'
    # name_for_columns = 'dna_element'
    # all_rows = cancer_types
    # all_columns = all_dna_elements

    os.makedirs(os.path.join(heatmaps_main_output_path,'heatmaps_for_each_signature'), exist_ok=True)
    heatmap_output_path = os.path.join(heatmaps_main_output_path,'heatmaps_for_each_signature')

    prepare_array_plot_heatmap_main(name_for_group_by,
                 name_for_rows,
                 name_for_columns,
                 all_rows,
                 all_columns,
                 step3_q_value_df,
                 heatmap_output_path,
                 depleted_fold_change,
                 enriched_fold_change,
                 significance_level,
                 remove_columns_rows_with_no_significant_result)

    # Plot heatmap for each cancer type
    name_for_group_by='cancer_type'
    name_for_rows='dna_element'
    name_for_columns='signature'
    all_rows = all_dna_elements
    all_columns=signatures

    os.makedirs(os.path.join(heatmaps_main_output_path,'heatmaps_for_each_cancer_type'), exist_ok=True)
    heatmap_output_path = os.path.join(heatmaps_main_output_path,'heatmaps_for_each_cancer_type')

    prepare_array_plot_heatmap_main(name_for_group_by,
                 name_for_rows,
                 name_for_columns,
                 all_rows,
                 all_columns,
                 step3_q_value_df,
                 heatmap_output_path,
                 depleted_fold_change,
                 enriched_fold_change,
                 significance_level,
                 remove_columns_rows_with_no_significant_result)
    # Uncomment in real run ends

    # Plot Manuscript and Cosmic figures at the same time
    for figure_type in figure_types:
        if figure_type == MANUSCRIPT:
            # Remove signatures that are attributed to sequencing artifacts
            artifact_signatures_removed_sbs_signatures = list(set(sbs_signatures) - set(signatures_attributed_to_artifacts))
            artifact_signatures_removed_sbs_signatures = list(set(artifact_signatures_removed_sbs_signatures) - set([ALL_SUBSTITUTIONS]))

            artifact_signatures_removed_sbs_signatures = sorted(artifact_signatures_removed_sbs_signatures, key=natural_key)
            artifact_signatures_removed_sbs_signatures.append(ALL_SUBSTITUTIONS)

            if artifact_signatures_removed_sbs_signatures:
                plot_heatmaps_rows_signatures_columns_dna_elements_with_pie_charts(step3_q_value_df,
                                                                                cancer_type_signature_cutoff_number_of_mutations_average_probability_df,
                                                                                SBS,
                                                                                artifact_signatures_removed_sbs_signatures,
                                                                                all_dna_elements,
                                                                                cancer_types,
                                                                                depleted_fold_change,
                                                                                enriched_fold_change,
                                                                                significance_level,
                                                                                heatmaps_main_output_path,
                                                                                figure_type,
                                                                                cosmic_release_version,
                                                                                figure_file_extension,
                                                                                heatmap_rows_signatures_columns_dna_elements = heatmap_rows_signatures_columns_dna_elements,
                                                                                plot_legend = False)

            if dbs_signatures:
                plot_heatmaps_rows_signatures_columns_dna_elements_with_pie_charts(step3_q_value_df,
                                                                                   cancer_type_signature_cutoff_number_of_mutations_average_probability_df,
                                                                                   DBS,
                                                                                   dbs_signatures,
                                                                                   all_dna_elements,
                                                                                   cancer_types,
                                                                                   depleted_fold_change,
                                                                                   enriched_fold_change,
                                                                                   significance_level,
                                                                                   heatmaps_main_output_path,
                                                                                   figure_type,
                                                                                   cosmic_release_version,
                                                                                   figure_file_extension,
                                                                                   heatmap_rows_signatures_columns_dna_elements = heatmap_rows_signatures_columns_dna_elements,
                                                                                   plot_legend = False)

            if id_signatures:
                plot_heatmaps_rows_signatures_columns_dna_elements_with_pie_charts(step3_q_value_df,
                                                                                   cancer_type_signature_cutoff_number_of_mutations_average_probability_df,
                                                                                    ID,
                                                                                    id_signatures,
                                                                                    all_dna_elements,
                                                                                    cancer_types,
                                                                                    depleted_fold_change,
                                                                                    enriched_fold_change,
                                                                                    significance_level,
                                                                                    heatmaps_main_output_path,
                                                                                    figure_type,
                                                                                    cosmic_release_version,
                                                                                    figure_file_extension,
                                                                                    heatmap_rows_signatures_columns_dna_elements = heatmap_rows_signatures_columns_dna_elements,
                                                                                    plot_legend = False)
        elif figure_type == COSMIC:
            if sbs_signatures:
                for signature in sbs_signatures:
                    if there_is_a_result_to_show(SBS,
                            all_dna_elements,
                            [signature],
                            cancer_types,
                            step3_q_value_df,
                            cancer_type_signature_cutoff_number_of_mutations_average_probability_df,
                            enriched_fold_change,
                            depleted_fold_change,
                            significance_level):

                        signature_based_df = step3_q_value_df[(step3_q_value_df['signature'] == signature) &
                                                              (step3_q_value_df['dna_element'] != 'ATAC-seq') &
                                                              (step3_q_value_df['dna_element'] != 'CTCF-human') &
                                                              (step3_q_value_df['dna_element'] != 'Nucleosome')]

                        if len(signature_based_df) > 0:
                            # Write COSMIC data files
                            data_file_name = '%s_%s_%s.txt' % (cosmic_release_version, signature, COSMIC_HISTONE_MODIFICATIONS)
                            data_file_path = os.path.join(heatmaps_main_output_path, DATA_FILES, data_file_name)

                            with open(data_file_path, 'w') as f:
                                # header line
                                f.write("# Only cancer types with minimum 2000 mutations for SBS signatures and minimum 1000 mutations for DBS and ID signatures with average probability at least 0.75 are considered.\n")
                                signature_based_df.to_csv(f, sep='\t', index=False, mode='w')

                            plot_heatmaps_rows_signatures_columns_dna_elements_with_pie_charts(step3_q_value_df,
                                cancer_type_signature_cutoff_number_of_mutations_average_probability_df,
                                SBS,
                                [signature],
                                all_dna_elements,
                                cancer_types,
                                depleted_fold_change,
                                enriched_fold_change,
                                significance_level,
                                heatmaps_main_output_path,
                                figure_type,
                                cosmic_release_version,
                                figure_file_extension,
                                heatmap_rows_signatures_columns_dna_elements = False)

            if dbs_signatures:
                for signature in dbs_signatures:
                    if there_is_a_result_to_show(DBS,
                            all_dna_elements,
                            [signature],
                            cancer_types,
                            step3_q_value_df,
                            cancer_type_signature_cutoff_number_of_mutations_average_probability_df,
                            enriched_fold_change,
                            depleted_fold_change,
                            significance_level):

                        signature_based_df = step3_q_value_df[(step3_q_value_df['signature'] == signature) &
                                                              (step3_q_value_df['dna_element'] != 'ATAC-seq') &
                                                              (step3_q_value_df['dna_element'] != 'CTCF-human') &
                                                              (step3_q_value_df['dna_element'] != 'Nucleosome')]

                        if len(signature_based_df) > 0:
                            # Write COSMIC data files
                            data_file_name = '%s_%s_%s.txt' % (cosmic_release_version, signature, COSMIC_HISTONE_MODIFICATIONS)
                            data_file_path = os.path.join(heatmaps_main_output_path, DATA_FILES, data_file_name)

                            with open(data_file_path, 'w') as f:
                                # header line
                                f.write("# Only cancer types with minimum 2000 mutations for SBS signatures and minimum 1000 mutations for DBS and ID signatures with average probability at least 0.75 are considered.\n")
                                signature_based_df.to_csv(f, sep='\t', index=False, mode='w')

                            plot_heatmaps_rows_signatures_columns_dna_elements_with_pie_charts(step3_q_value_df,
                                cancer_type_signature_cutoff_number_of_mutations_average_probability_df,
                                DBS,
                                [signature],
                                all_dna_elements,
                                cancer_types,
                                depleted_fold_change,
                                enriched_fold_change,
                                significance_level,
                                heatmaps_main_output_path,
                                figure_type,
                                cosmic_release_version,
                                figure_file_extension,
                                heatmap_rows_signatures_columns_dna_elements = False)

            if id_signatures:
                for signature in id_signatures:
                    if there_is_a_result_to_show(ID,
                            all_dna_elements,
                            [signature],
                            cancer_types,
                            step3_q_value_df,
                            cancer_type_signature_cutoff_number_of_mutations_average_probability_df,
                            enriched_fold_change,
                            depleted_fold_change,
                            significance_level):

                        signature_based_df = step3_q_value_df[(step3_q_value_df['signature'] == signature) &
                                                              (step3_q_value_df['dna_element'] != 'ATAC-seq') &
                                                              (step3_q_value_df['dna_element'] != 'CTCF-human') &
                                                              (step3_q_value_df['dna_element'] != 'Nucleosome')]

                        if len(signature_based_df) > 0:
                            # Write COSMIC data files
                            data_file_name = '%s_%s_%s.txt' % (cosmic_release_version, signature, COSMIC_HISTONE_MODIFICATIONS)
                            data_file_path = os.path.join(heatmaps_main_output_path, DATA_FILES, data_file_name)

                            with open(data_file_path, 'w') as f:
                                # header line
                                f.write("# Only cancer types with minimum 2000 mutations for SBS signatures and minimum 1000 mutations for DBS and ID signatures with average probability at least 0.75 are considered.\n")
                                signature_based_df.to_csv(f, sep='\t', index=False, mode='w')

                            plot_heatmaps_rows_signatures_columns_dna_elements_with_pie_charts(step3_q_value_df,
                                cancer_type_signature_cutoff_number_of_mutations_average_probability_df,
                                ID,
                                [signature],
                                all_dna_elements,
                                cancer_types,
                                depleted_fold_change,
                                enriched_fold_change,
                                significance_level,
                                heatmaps_main_output_path,
                                figure_type,
                                cosmic_release_version,
                                figure_file_extension,
                                heatmap_rows_signatures_columns_dna_elements = False)


    step1_p_value_df = process_dataframes(step1_p_value_df)

    # write excel files
    excel_file_name = '%s_%s.xlsx' %(cosmic_release_version, COSMIC_HISTONE_MODIFICATIONS)
    excel_file_path = os.path.join(heatmaps_main_output_path, EXCEL_FILES, excel_file_name)
    df_list = [step1_p_value_df, step3_q_value_df]
    sheet_list = ['p_value', 'q_value']
    write_excel_file(df_list, sheet_list, excel_file_path)


# sheet name must be less than 31 characters
def write_excel_file(df_list, sheet_list, file_name):
    writer = pd.ExcelWriter(file_name,engine='xlsxwriter')
    for dataframe, sheet in zip(df_list, sheet_list):
        dataframe.to_excel(writer, sheet_name=sheet, startrow=0 , startcol=0, index=False)
    writer.save()

# Sample input_dir and output_dir
# input_dir = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'SigProfilerTopographyRuns', 'Combined_PCAWG_nonPCAWG_4th_iteration')
# output_dir = os.path.join('/oasis', 'tscc', 'scratch', 'burcak', 'SigProfilerTopographyRuns', 'combined_pcawg_and_nonpcawg_figures_pdfs', '4th_iteration')

# input_dir = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'SigProfilerTopographyRuns', 'Combined_PCAWG_nonPCAWG_prob_mode')
# output_dir = os.path.join('/oasis', 'tscc', 'scratch', 'burcak', 'SigProfilerTopographyRuns', 'combined_pcawg_and_nonpcawg_figures_pdfs', 'prob_mode')

# input_dir = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'SigProfilerTopographyRuns', 'Combined_PCAWG_nonPCAWG_prob_mode_05')
# output_dir = os.path.join('/oasis', 'tscc', 'scratch', 'burcak', 'SigProfilerTopographyRuns', 'combined_pcawg_and_nonpcawg_figures_pdfs', 'prob_mode_05')

# Main Function
# Artifact signatures are removed for manuscript
# No need to set as a parameter in main function
def main(input_dir,
         output_dir,
         tissue_type = 'Normal',
         step1_data_ready = False,
         window_size = 100,
         combine_p_values_method = 'fisher',
         depleted_fold_change = 0.95,
         enriched_fold_change = 1.05,
         significance_level = 0.05,
         consider_both_real_and_sim_avg_overlap = True,
         minimum_number_of_overlaps_required_for_sbs = 100,
         minimum_number_of_overlaps_required_for_dbs = 100,
         minimum_number_of_overlaps_required_for_indels = 100,
         signature_cancer_type_number_of_mutations = AT_LEAST_1K_CONSRAINTS,
         signature_cancer_type_number_of_mutations_for_ctcf = AT_LEAST_1K_CONSRAINTS,
         sort_cancer_types = True,
         remove_columns_rows_with_no_significant_result = True,
         heatmap_rows_signatures_columns_dna_elements = True,
         colorbar = 'seismic',
         figure_types = [MANUSCRIPT, COSMIC],
         cosmic_release_version = 'v3.2',
         figure_file_extension = 'jpg'):

    # Important
    # It helps getting the average signal files containing these dna elements in their txt filenames
    # It also fills the dictionary in epigenomics aggregated mutations using these element names.
    # Epigenomics signature based  fills using get_unique_hms() which uses get_dna_element() (columns of the heatmaps)
    # They must be consistent
    # Therefore get_dna_element() must return dna element as it is seen in encode_dna_elements_we_are_interested_in
    # encode_dna_elements_we_are_interested_in = ['H3K36me3-human', 'H3K4me1-human', 'H3K4me3-human', 'H3K27ac-human','H3K9me3-human', 'H3K27me3-human', 'H3K9ac-human', 'ATAC-seq', 'CTCF-human']

    # Real and simulations signals are here
    heatmaps_dir_name = "heatmaps_dna_elements_window_size_%s_%s" %(window_size, tissue_type)
    output_dir = os.path.join(output_dir, heatmaps_dir_name)

    if not step1_data_ready:
        deleteOldData(output_dir)
        os.makedirs(os.path.join(output_dir, TABLES), exist_ok=True)
        os.makedirs(os.path.join(output_dir, DATA_FILES), exist_ok=True)
        os.makedirs(os.path.join(output_dir, EXCEL_FILES), exist_ok=True)
        os.makedirs(os.path.join(output_dir, DICTIONARIES), exist_ok=True)

    # NUMBER OF SIMULATIONS
    numberofSimulations = 100

    # CANCER TYPES
    # These are the 40 tissues for combined PCAWG and nonPCAWG + ESCC
    cancer_types = ['ALL', 'Bladder-TCC', 'Bone-Benign', 'Bone-Osteosarc', 'CNS-GBM', 'CNS-Medullo', 'CNS-PiloAstro',
                  'ColoRect-AdenoCA', 'Ewings', 'Head-SCC', 'Kidney-RCC', 'Lung-AdenoCA', 'Lymph-BNHL', 'Myeloid-AML',
                  'Myeloid-MPN', 'Panc-AdenoCA', 'Prost-AdenoCA', 'SoftTissue-Leiomyo', 'Stomach-AdenoCA',
                  'Uterus-AdenoCA', 'Biliary-AdenoCA', 'Blood-CMDI', 'Bone-Epith', 'Breast-Cancer', 'CNS-LGG',
                  'CNS-Oligo', 'Cervix-Cancer', 'Eso-AdenoCA', 'ESCC', 'Eye-Melanoma', 'Kidney-ChRCC', 'Liver-HCC',
                  'Lung-SCC', 'Lymph-CLL', 'Myeloid-MDS', 'Ovary-AdenoCA', 'Panc-Endocrine', 'Skin-Melanoma',
                  'SoftTissue-Liposarc', 'Thy-AdenoCA']
    print('len(cancer_types): %d' %len(cancer_types))
    print('cancer_types: %s' %(cancer_types))

    # SIGNATURES
    sbs_signatures = ['SBS1', 'SBS2', 'SBS3', 'SBS4', 'SBS5', 'SBS6', 'SBS7a', 'SBS7b', 'SBS7c', 'SBS7d', 'SBS8',
                      'SBS9', 'SBS10a', 'SBS10b', 'SBS10c', 'SBS11', 'SBS12', 'SBS13', 'SBS14', 'SBS15', 'SBS16',
                      'SBS17a', 'SBS17b', 'SBS18', 'SBS19', 'SBS20', 'SBS21', 'SBS22', 'SBS23', 'SBS24', 'SBS25',
                      'SBS26', 'SBS27', 'SBS28', 'SBS29', 'SBS30', 'SBS31', 'SBS32', 'SBS33', 'SBS34', 'SBS35',
                      'SBS36', 'SBS37', 'SBS38', 'SBS39', 'SBS40', 'SBS41', 'SBS42', 'SBS43', 'SBS44', 'SBS45',
                      'SBS46', 'SBS47', 'SBS48', 'SBS49', 'SBS50', 'SBS51', 'SBS52', 'SBS53', 'SBS54', 'SBS55',
                      'SBS56', 'SBS57', 'SBS58', 'SBS59', 'SBS60', 'SBS84', 'SBS85']

    sbs_signatures = sorted(sbs_signatures, key=natural_key)
    sbs_signatures.append(ALL_SUBSTITUTIONS)

    dbs_signatures = ['DBS1', 'DBS2', 'DBS3', 'DBS4', 'DBS5', 'DBS6', 'DBS7', 'DBS8', 'DBS9', 'DBS10', 'DBS11']
    dbs_signatures.append(ALL_DINUCLEOTIDES)

    id_signatures = ['ID1', 'ID2', 'ID3', 'ID4', 'ID5', 'ID6', 'ID7', 'ID8', 'ID9', 'ID10', 'ID11', 'ID12', 'ID13',
                     'ID14', 'ID15', 'ID16', 'ID17']
    id_signatures.append(ALL_INDELS)

    hm_path = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'data', 'ENCODE', 'GRCh37', 'HM')
    ctcf_path = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'data', 'ENCODE', 'GRCh37', 'CTCF')
    atac_path = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'data', 'ENCODE', 'GRCh37', 'ATAC_seq')

    # # For testing purposes
    # # cancer_types = ['Liver-HCC']
    # sbs_signatures = ['SBS4']
    # dbs_signatures = ['DBS11']
    # id_signatures = ['ID11']
    # sbs_signatures.append(ALL_SUBSTITUTIONS)
    # dbs_signatures.append(ALL_DINUCLEOTIDES)
    # id_signatures.append(ALL_INDELS)

    print('sbs_signatures: %s' %(sbs_signatures))
    print('len(sbs_signatures): %d' %(len(sbs_signatures)))
    print('dbs_signatures: %s' %(dbs_signatures))
    print('len(dbs_signatures): %d' %(len(dbs_signatures)))
    print('id_signatures: %s' %(id_signatures))
    print('len(id_signatures): %d' %(len(id_signatures)))

    # Updated for Lymphoid samples
    cancer_type_signature_cutoff_number_of_mutations_average_probability_df = fill_cancer_type_signature_cutoff_average_probability_df(
        cancer_types,
        input_dir)

    if (cancer_type_signature_cutoff_number_of_mutations_average_probability_df is not None):
        df_file_name= 'cancer_type_signature_cutoff_average_probability_df.txt'
        df_file_path = os.path.join(output_dir, TABLES, df_file_name)
        cancer_type_signature_cutoff_number_of_mutations_average_probability_df.to_csv(df_file_path, sep='\t', header=True, index=False)

    cancer_type_mutation_type_number_of_mutations_df = fill_cancer_type_mutation_type_number_of_mutations_df(cancer_types, input_dir)

    if (cancer_type_mutation_type_number_of_mutations_df is not None):
        df_file_name = 'cancer_type_mutation_type_number_of_mutations_df.txt'
        df_file_path = os.path.join(output_dir, TABLES, df_file_name)
        cancer_type_mutation_type_number_of_mutations_df.to_csv(df_file_path, sep='\t', header=True, index=False)

    # Write excel files
    excel_file_path = os.path.join(output_dir, EXCEL_FILES,'Cancer_Types.xlsx')
    df_list = [cancer_type_signature_cutoff_number_of_mutations_average_probability_df, cancer_type_mutation_type_number_of_mutations_df]
    sheet_list = ['average_probability', 'samples']
    write_excel_file(df_list, sheet_list, excel_file_path)

    if (colorbar == COLORBAR_SEISMIC):
        plot_pcawg_heatmap_seismic_colorbar(output_dir)
    elif (colorbar == COLORBAR_DISCREET):
        plot_pcawg_heatmap_discreet_colorbar(output_dir)

    # Plot legends as separate figures
    plot_proportion_of_cancer_types_with_the_signature(output_dir)
    plot_epigenomics_heatmap_legend(output_dir)

    # Combined p value and plot heatmaps
    compute_fold_change_significance_plot_heatmap(combine_p_values_method,
                                                  window_size,
                                                  tissue_type,
                                                  input_dir,
                                                  numberofSimulations,
                                                  hm_path,
                                                  ctcf_path,
                                                  atac_path,
                                                  output_dir,
                                                  cancer_type_signature_cutoff_number_of_mutations_average_probability_df,
                                                  cancer_types,
                                                  sbs_signatures,
                                                  id_signatures,
                                                  dbs_signatures,
                                                  depleted_fold_change,
                                                  enriched_fold_change,
                                                  step1_data_ready,
                                                  significance_level,
                                                  minimum_number_of_overlaps_required_for_sbs,
                                                  minimum_number_of_overlaps_required_for_dbs,
                                                  minimum_number_of_overlaps_required_for_indels,
                                                  figure_types,
                                                  cosmic_release_version,
                                                  figure_file_extension,
                                                  signature_cancer_type_number_of_mutations,
                                                  signature_cancer_type_number_of_mutations_for_ctcf,
                                                  consider_both_real_and_sim_avg_overlap,
                                                  sort_cancer_types,
                                                  remove_columns_rows_with_no_significant_result,
                                                  heatmap_rows_signatures_columns_dna_elements)

    print('\n##########################################################')
    print('All the fold changes are computed. Heatmaps are plotted.')
    print('##########################################################\n')


def get_minimum_number_of_overlaps_required(signature,
                                            signature_tuples,
                                            minimum_number_of_overlaps_required_for_sbs,
                                            minimum_number_of_overlaps_required_for_dbs,
                                            minimum_number_of_overlaps_required_for_indels):

    # default
    minimum_number_of_overlaps_required = minimum_number_of_overlaps_required_for_sbs

    # (signature, SBS)
    # (signature, DBS)
    # (signature, ID)
    for signature_name, signature_type in signature_tuples:
        if signature_name==signature:
            if signature_type == SBS:
                minimum_number_of_overlaps_required = minimum_number_of_overlaps_required_for_sbs
            elif signature_type == DBS:
                minimum_number_of_overlaps_required = minimum_number_of_overlaps_required_for_dbs
            elif signature_type == ID:
                minimum_number_of_overlaps_required = minimum_number_of_overlaps_required_for_indels
            break

    return minimum_number_of_overlaps_required


def is_eligible(fold_change,
              p_value,
              number_of_mutations,
              number_of_mutations_required,
              consider_both_real_and_sim_avg_overlap,
              real_data_avg_count,
              sim_data_avg_count,
              depleted_fold_change,
              enriched_fold_change,
              minimum_number_of_overlaps_required):

    if consider_both_real_and_sim_avg_overlap:
        # Consider both real and sims
        if ((fold_change is not None) and (not np.isnan(np.array([fold_change], dtype=np.float)).any()) and (str(fold_change) != 'nan')) and \
                ((p_value is not None) and (not np.isnan(np.array([p_value], dtype=np.float)).any()) and (str(p_value) != 'nan')) and \
                (number_of_mutations >= number_of_mutations_required) and \
                ((real_data_avg_count is not None) and
                 (sim_data_avg_count is not None) and
                 ((real_data_avg_count >= minimum_number_of_overlaps_required) or
                  (depleted(fold_change,depleted_fold_change) and
                   (sim_data_avg_count >= minimum_number_of_overlaps_required)) or
                  (enriched(fold_change,enriched_fold_change) and
                   (sim_data_avg_count >= minimum_number_of_overlaps_required) and
                   (real_data_avg_count >= minimum_number_of_overlaps_required * OCCUPANCY_HEATMAP_COMMON_MULTIPLIER)) )) :
            return True
    else:
        # Consider only real
        if ((fold_change is not None) and (not np.isnan(np.array([fold_change], dtype=np.float)).any()) and (str(fold_change) != 'nan')) and \
                ((p_value is not None) and (not np.isnan(np.array([p_value], dtype=np.float)).any()) and (str(p_value) != 'nan')) and \
                (number_of_mutations >= number_of_mutations_required) and \
                ((real_data_avg_count is not None) and (real_data_avg_count >= minimum_number_of_overlaps_required)):
            return True

    return False

# Step2 combine p values using Fisher's method
# [fold_change_list,avg_fold_change,p_value_list,combined_p_value]
def step2_combine_p_values(step1_signature2cancer_type2biosample2dna_element2p_value_tuple_dict,
                           heatmaps_main_output_path,
                           combine_p_values_method,
                           signature_tuples,
                           depleted_fold_change,
                           enriched_fold_change,
                           minimum_number_of_overlaps_required_for_sbs,
                           minimum_number_of_overlaps_required_for_dbs,
                           minimum_number_of_overlaps_required_for_indels,
                           signature_cancer_type_number_of_mutations,
                           signature_cancer_type_number_of_mutations_for_ctcf,
                           consider_both_real_and_sim_avg_overlap = True):

    # Fill and return this dictionary
    signature2cancer_type2dna_element2combined_p_value_list_dict = {}

    # Pooling for biosample and dna_element combine q_values
    for signature in step1_signature2cancer_type2biosample2dna_element2p_value_tuple_dict:
        minimum_number_of_overlaps_required = get_minimum_number_of_overlaps_required(signature,
                                                                                      signature_tuples,
                                                                                      minimum_number_of_overlaps_required_for_sbs,
                                                                                      minimum_number_of_overlaps_required_for_dbs,
                                                                                      minimum_number_of_overlaps_required_for_indels)

        for cancer_type in step1_signature2cancer_type2biosample2dna_element2p_value_tuple_dict[signature]:
            for biosample in step1_signature2cancer_type2biosample2dna_element2p_value_tuple_dict[signature][cancer_type]:
                for encode_element in step1_signature2cancer_type2biosample2dna_element2p_value_tuple_dict[signature][cancer_type][biosample]:
                    dna_element = get_dna_element(encode_element)
                    complete_list = step1_signature2cancer_type2biosample2dna_element2p_value_tuple_dict[signature][cancer_type][biosample][encode_element]

                    # p value complete list has [signature, cancer_type, cutoff, number_of_mutations, biosample, dna_element, avg_real_signal, avg_sim_signal,
                    # fold_change, min_sim_signal, max_sim_signal, pvalue, num_of_sims, num_of_sims_with_not_nan_avgs, real_data_avg_count, list(simulationsHorizontalMeans)]
                    number_of_mutations = complete_list[3]
                    fold_change = complete_list[8]
                    p_value = complete_list[11]
                    real_data_avg_count = complete_list[14]
                    sim_data_avg_count = complete_list[15]

                    if CTCF in dna_element:
                        number_of_mutations_required = signature_cancer_type_number_of_mutations_for_ctcf
                    else:
                        number_of_mutations_required = signature_cancer_type_number_of_mutations

                    if is_eligible(fold_change,
                                 p_value,
                                 number_of_mutations,
                                 number_of_mutations_required,
                                 consider_both_real_and_sim_avg_overlap,
                                 real_data_avg_count,
                                 sim_data_avg_count,
                                 depleted_fold_change,
                                 enriched_fold_change,
                                 minimum_number_of_overlaps_required):

                        if signature in signature2cancer_type2dna_element2combined_p_value_list_dict:
                            if cancer_type in signature2cancer_type2dna_element2combined_p_value_list_dict[signature]:
                                if dna_element in signature2cancer_type2dna_element2combined_p_value_list_dict[signature][cancer_type]:
                                    list_of_lists = signature2cancer_type2dna_element2combined_p_value_list_dict[signature][cancer_type][dna_element]
                                    encode_element_list = list_of_lists[0]
                                    fold_change_list = list_of_lists[1]
                                    p_value_list = list_of_lists[2]

                                    encode_element_list.append(encode_element)
                                    fold_change_list.append(fold_change)
                                    p_value_list.append(p_value)
                                else:
                                    signature2cancer_type2dna_element2combined_p_value_list_dict[signature][cancer_type][dna_element]=[[encode_element], [fold_change],[p_value]]
                            else:
                                signature2cancer_type2dna_element2combined_p_value_list_dict[signature][cancer_type] = {}
                                signature2cancer_type2dna_element2combined_p_value_list_dict[signature][cancer_type][dna_element] = [[encode_element], [fold_change], [p_value]]
                        else:
                            signature2cancer_type2dna_element2combined_p_value_list_dict[signature] = {}
                            signature2cancer_type2dna_element2combined_p_value_list_dict[signature][cancer_type] = {}
                            signature2cancer_type2dna_element2combined_p_value_list_dict[signature][cancer_type][dna_element] = [[encode_element], [fold_change], [p_value]]

    for signature in signature2cancer_type2dna_element2combined_p_value_list_dict:
        for cancer_type in signature2cancer_type2dna_element2combined_p_value_list_dict[signature]:
            for dna_element in signature2cancer_type2dna_element2combined_p_value_list_dict[signature][cancer_type]:
                encode_element_list=signature2cancer_type2dna_element2combined_p_value_list_dict[signature][cancer_type][dna_element][0]
                fold_change_list=signature2cancer_type2dna_element2combined_p_value_list_dict[signature][cancer_type][dna_element][1]
                p_value_list=signature2cancer_type2dna_element2combined_p_value_list_dict[signature][cancer_type][dna_element][2]

                avg_fold_change = np.nanmean(fold_change_list)

                p_values_array = np.asarray(p_value_list)
                test_statistic, combined_p_value = scipy.stats.combine_pvalues(p_values_array, method=combine_p_values_method, weights=None)

                signature2cancer_type2dna_element2combined_p_value_list_dict[signature][cancer_type][dna_element]=[encode_element_list,fold_change_list,avg_fold_change,p_value_list,combined_p_value]

    # Write dictionary as a pandas dataframe
    df_filename = 'Step2_Signature_CancerType_DNAElement_CombinedPValue.txt'
    filepath = os.path.join(heatmaps_main_output_path,TABLES, df_filename)
    step2_combined_p_value_df=write_dictionary_as_dataframe_step2_combined_p_value(signature2cancer_type2dna_element2combined_p_value_list_dict,filepath)

    return step2_combined_p_value_df, signature2cancer_type2dna_element2combined_p_value_list_dict


# Fill array with avg_fold_change for the epigenomics heatmap using step3_q_value_df
def fill_rows_and_columns_np_array(name_for_rows, rows, name_for_columns, columns, df):
    # initialize a numpy array
    rows_columns_np_array = np.ones((len(rows), len(columns)))

    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(columns):
            if df[(df[name_for_rows]==row) & (df[name_for_columns]==column)]['avg_fold_change'].values.any():
                avg_fold_change_array = df[(df[name_for_rows] == row) & (df[name_for_columns] == column)]['avg_fold_change'].values
                if len(avg_fold_change_array) > 1:
                    avg_fold_change_array_mean = np.mean(avg_fold_change_array)
                    rows_columns_np_array[row_index][column_index] = round(avg_fold_change_array_mean, 2)
                else:
                    rows_columns_np_array[row_index][column_index] = round(avg_fold_change_array[0], 2)

    return rows_columns_np_array


def plot_heatmap(rows,
                columns,
                avg_fold_change_np_array,
                group_name,
                df,
                name_for_rows,
                name_for_columns,
                significance_level,
                enriched_fold_change,
                depleted_fold_change,
                heatmap_rows_signatures_columns_cancer_types_output_path,
                figure_name,
                x_axis_labels_on_bottom = True,
                plot_title = True):

    ##########################################################################
    # title_font_size = 90
    # label_font_size = 70
    # cell_font_size = 50

    # if len(rows) > len(columns):
    #     fig, ax = plt.subplots(figsize=(80, 100))
    # else:
    #     fig, ax = plt.subplots(figsize=(100, 80))
    ##########################################################################

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

        # # For Graphical Abstract Figure (3 lines)
        # title_font_size = title_font_size * 3
        # label_font_size = font_size * 3
        # cell_font_size = cell_font_size * 3

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

        if not x_axis_labels_on_bottom:
            # CTCF SBS DBS ID separately 3 heatmaps
            if len(rows_labels) == 2:
                font_size *= 5
            elif len(rows_labels) == 7:
                font_size *= 2.6
            elif len(rows_labels) == 22:
                font_size *= 1.5

            label_font_size = font_size
            cell_font_size = font_size * 2 / 3

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

        # Put text in each heatmap cell
        for row_index, row in enumerate(rows, 0):
            print('#####################################################')
            for column_index, column in enumerate(columns, 0):
                if row == 'ALL SUBSTITUTIONS':
                    row = AGGREGATEDSUBSTITUTIONS
                elif row == 'ALL DINUCLEOTIDES':
                    row = AGGREGATEDDINUCS
                elif row == 'ALL INDELS':
                    row = AGGREGATEDINDELS

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


def prepare_array_plot_heatmap(df,
                        name_for_rows,
                        rows,
                        name_for_columns,
                        columns,
                        enriched_fold_change,
                        depleted_fold_change,
                        significance_level,
                        heatmap_rows_signatures_columns_cancer_types_output_path,
                        group_name,
                        figure_name = None,
                        remove_columns_rows_with_no_significant_result = True):

    avg_fold_change_np_array = fill_rows_and_columns_np_array(name_for_rows, rows, name_for_columns, columns, df)

    if remove_columns_rows_with_no_significant_result:

        # Delete any row with no significant result
        # Significant result means (q-value <= significance_level and (avg fold change <= depleted_fold_change or avg fold change >= enriched_fold_change))
        rows_index_not_deleted = []
        for row_index, row in enumerate(rows):
            q_value_list = []
            for column in columns:
                if df[(df[name_for_rows] == row) & (df[name_for_columns] == column)]['avg_fold_change'].any():

                    avg_fold_change_array = df[(df[name_for_rows] == row) & (df[name_for_columns] == column)]['avg_fold_change'].values
                    q_value_array = df[(df[name_for_rows] == row) & (df[name_for_columns] == column)]['q_value'].values

                    assert len(avg_fold_change_array) == 1, 'There must be one avg_fold_change'
                    assert len(q_value_array) == 1, 'There must be one q_value'

                    avg_fold_change = avg_fold_change_array[0]
                    q_value = q_value_array[0]

                    if (enriched(avg_fold_change, enriched_fold_change) or depleted(avg_fold_change, depleted_fold_change)):
                        q_value_list.append(q_value)

            q_value_array = np.array(q_value_list)
            if np.any(q_value_array <= significance_level):
                rows_index_not_deleted.append(True)
            # For Manuscript Figure4
            # Do not delete rows SBS7a/b/c/d SBS11 and SBS28 even if there is no significant result to show.
            elif name_for_rows == 'signature' and name_for_columns == 'cancer_type' and group_name == 'CTCF-human' and \
                    row in ['SBS7a', 'SBS7b', 'SBS7c', 'SBS7d', 'SBS11', 'SBS28']:
                rows_index_not_deleted.append(True)
            else:
                rows_index_not_deleted.append(False)

        # Delete rows with q_value greater than significance_level
        avg_fold_change_np_array = np.array(avg_fold_change_np_array)[rows_index_not_deleted]
        rows = np.array(rows)[rows_index_not_deleted]

        # Delete any column with no significant result
        # Significant means (q-value <= significance_level and (avg fold change <= depleted_fold_change or avg fold change >= enriched_fold_change))
        columns_index_not_deleted = []
        for column in columns:
            q_value_list = []
            for row in rows:
                if df[(df[name_for_rows] == row) & (df[name_for_columns] == column)]['avg_fold_change'].any():
                    avg_fold_change_array = df[(df[name_for_rows] == row) & (df[name_for_columns] == column)]['avg_fold_change'].values
                    q_value_array = df[(df[name_for_rows] == row) & (df[name_for_columns] == column)]['q_value'].values

                    assert len(avg_fold_change_array) == 1, 'There must be one avg_fold_change'
                    assert len(q_value_array) == 1, 'There must be one q_value'

                    avg_fold_change = avg_fold_change_array[0]
                    q_value = q_value_array[0]

                    if (enriched(avg_fold_change, enriched_fold_change) or depleted(avg_fold_change, depleted_fold_change)):
                        q_value_list.append(q_value)

            q_value_array = np.array(q_value_list)
            if np.any(q_value_array <= significance_level):
                columns_index_not_deleted.append(True)
            else:
                columns_index_not_deleted.append(False)

        # Delete columns with q_value greater than significance_level
        avg_fold_change_np_array = np.array(avg_fold_change_np_array)[:, columns_index_not_deleted]
        columns = np.array(columns)[columns_index_not_deleted]

    # For Figure4 PanelA CTCF heatmap
    # One big heatmap for SBS, DBS and ID signatures each signature type are separated by empty row.
    # cancer types at top
    # figures will under
    # / oasis / tscc / scratch / burcak / SigProfilerTopographyRuns / combined_pcawg_and_nonpcawg_figures_pdfs / 4th_iteration
    # / heatmaps_dna_elements_window_size_100_Normal / heatmaps_for_each_dna_element
    if group_name == 'CTCF-human':
        # SBS signatures
        all_subs_maks = [True if i.startswith(ALL_SUBSTITUTIONS) else False for i in rows]
        sbs_signatures_mask = [True if i.startswith('SBS') else False for i in rows]

        all_substitutions_avg_fold_change_np_array =  avg_fold_change_np_array[[all_subs_maks]]
        sbs_signatures_avg_fold_change_np_array =  avg_fold_change_np_array[[sbs_signatures_mask]]

        all_sbs_array = np.concatenate((all_substitutions_avg_fold_change_np_array, sbs_signatures_avg_fold_change_np_array), axis=0)

        sbs_signatures_rows = [i for i in rows if i.startswith('SBS')]
        all_sbs_rows = [ALL_SUBSTITUTIONS] + sbs_signatures_rows

        # DBS signatures
        all_dinucs_maks = [True if i.startswith(ALL_DINUCLEOTIDES) else False for i in rows]
        dbs_signatures_mask = [True if i.startswith('DBS') else False for i in rows]

        all_dinucleotides_avg_fold_change_np_array =  avg_fold_change_np_array[[all_dinucs_maks]]
        dbs_signatures_avg_fold_change_np_array =  avg_fold_change_np_array[[dbs_signatures_mask]]

        all_dbs_array = np.concatenate((all_dinucleotides_avg_fold_change_np_array, dbs_signatures_avg_fold_change_np_array), axis=0)

        dbs_signatures_rows = [i for i in rows if i.startswith('DBS')]
        all_dbs_rows = [ALL_DINUCLEOTIDES] + dbs_signatures_rows

        # ID signatures
        all_indels_maks = [True if i.startswith(ALL_INDELS) else False for i in rows]
        id_signatures_mask = [True if i.startswith('ID') else False for i in rows]

        all_indels_avg_fold_change_np_array =  avg_fold_change_np_array[[all_indels_maks]]
        id_signatures_avg_fold_change_np_array =  avg_fold_change_np_array[[id_signatures_mask]]

        all_id_array = np.concatenate((all_indels_avg_fold_change_np_array, id_signatures_avg_fold_change_np_array), axis=0)
        id_signatures_rows = [i for i in rows if i.startswith('ID')]

        all_id_rows = [ALL_INDELS] + id_signatures_rows

        # First SBS signatures, Second DBS signature, Third ID signatures
        all_ones_array = np.ones((1,all_sbs_array.shape[1]))
        all_sbs_dbs_id_array = np.concatenate((all_sbs_array, all_ones_array, all_dbs_array, all_ones_array, all_id_array), axis=0)
        all_sbs_dbs_id_rows = all_sbs_rows + [''] + all_dbs_rows +  [''] +all_id_rows

        plot_heatmap(all_sbs_dbs_id_rows,
                columns,
                all_sbs_dbs_id_array,
                'CTCF_SBS_DBS_ID',
                df,
                name_for_rows,
                name_for_columns,
                significance_level,
                enriched_fold_change,
                depleted_fold_change,
                heatmap_rows_signatures_columns_cancer_types_output_path,
                figure_name,
                x_axis_labels_on_bottom = False,
                plot_title = False)

        plot_heatmap(all_sbs_rows,
                columns,
                all_sbs_array,
                group_name + 'SBS',
                df,
                name_for_rows,
                name_for_columns,
                significance_level,
                enriched_fold_change,
                depleted_fold_change,
                heatmap_rows_signatures_columns_cancer_types_output_path,
                figure_name,
                x_axis_labels_on_bottom = False,
                plot_title = False)

        plot_heatmap(all_dbs_rows,
                columns,
                all_dbs_array,
                group_name + 'DBS',
                df,
                name_for_rows,
                name_for_columns,
                significance_level,
                enriched_fold_change,
                depleted_fold_change,
                heatmap_rows_signatures_columns_cancer_types_output_path,
                figure_name,
                x_axis_labels_on_bottom = False,
                plot_title = False)

        plot_heatmap(all_id_rows,
                columns,
                all_id_array,
                group_name + 'ID',
                df,
                name_for_rows,
                name_for_columns,
                significance_level,
                enriched_fold_change,
                depleted_fold_change,
                heatmap_rows_signatures_columns_cancer_types_output_path,
                figure_name,
                x_axis_labels_on_bottom = False,
                plot_title = False)

    plot_heatmap(rows,
            columns,
            avg_fold_change_np_array,
            group_name,
            df,
            name_for_rows,
            name_for_columns,
            significance_level,
            enriched_fold_change,
            depleted_fold_change,
            heatmap_rows_signatures_columns_cancer_types_output_path,
            figure_name)

# Plot heatmap
# For each DNA element
# For each signature
# For each cancer type
def prepare_array_plot_heatmap_main(name_for_group_by,
                 name_for_rows,
                 name_for_columns,
                 all_rows,
                 all_columns,
                 step3_q_value_df,
                 heatmap_rows_signatures_columns_cancer_types_output_path,
                 depleted_fold_change,
                 enriched_fold_change,
                 significance_level,
                 remove_columns_rows_with_no_significant_result):

    # signature
    # cancer_type
    # dna_element
    # fold_change_list
    # avg_fold_change
    # p_value_list
    # combined_p_value
    # q_value

    grouped_df = step3_q_value_df.groupby(name_for_group_by)

    for group_name, group_df in grouped_df:
        rows = all_rows
        columns = all_columns
        prepare_array_plot_heatmap(group_df,
                        name_for_rows,
                        rows,
                        name_for_columns,
                        columns,
                        enriched_fold_change,
                        depleted_fold_change,
                        significance_level,
                        heatmap_rows_signatures_columns_cancer_types_output_path,
                        group_name,
                        remove_columns_rows_with_no_significant_result = remove_columns_rows_with_no_significant_result)



def plot_epigenomics_heatmap_legend(heatmaps_main_output_path):
    os.makedirs(os.path.join(heatmaps_main_output_path, FIGURES_MANUSCRIPT), exist_ok=True)
    heatmap_output_path=os.path.join(heatmaps_main_output_path,FIGURES_MANUSCRIPT)

    fig, ax = plt.subplots(figsize=(10, 1))
    legend_elements = [
        Line2D([0], [0], marker='o', color='white', label=ENRICHMENT_OF_MUTATIONS, markerfacecolor='indianred', markersize=30),
        Line2D([0], [0], marker='o', color='white', label=DEPLETION_OF_MUTATIONS, markerfacecolor='cornflowerblue', markersize=30),
        Line2D([0], [0], marker='o', color='white', label=NO_EFFECT_BASED_ON_EXPECTED_BY_CHANCE, markerfacecolor='silver', markersize=30)]

    # plt.legend(handles=legend_elements, ncol=len(legend_elements), loc="upper right", bbox_to_anchor=(1, 0), fontsize=30)
    plt.legend(handles=legend_elements, ncol=1, loc="upper right", bbox_to_anchor=(1, 0), fontsize=30)
    plt.gca().set_axis_off()

    filename = 'pie_chart_heatmap_legend.png'
    figureFile = os.path.join(heatmap_output_path, filename)
    fig.savefig(figureFile, dpi=100, bbox_inches="tight")
    plt.close()


def plot_proportion_of_cancer_types_with_the_signature(heatmaps_main_output_path):

    os.makedirs(os.path.join(heatmaps_main_output_path, FIGURES_MANUSCRIPT), exist_ok=True)
    heatmap_output_path=os.path.join(heatmaps_main_output_path, FIGURES_MANUSCRIPT)

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

    ax.tick_params(axis='x', which='minor', length=0, labelsize=25)
    # major ticks
    ax.set_xticks(np.arange(0, len(diameter_labels), 1))
    # minor ticks
    ax.set_xticks(np.arange(0, len(diameter_labels), 1) + 0.5, minor=True)
    ax.set_xticklabels(diameter_ticklabels, minor=True)

    ax.xaxis.set_ticks_position('bottom')

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='major',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False)  # labels along the bottom edge are off

    # ax.set_xlabel('Proportion of tumors\nwith the signature', fontsize=30,labelpad=10)
    ax.set_xlabel('Proportion of cancer types\nwith the signature', fontsize=30,labelpad=10)

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

    filename = 'proportion_of_cancer_types.png'
    figureFile = os.path.join(heatmap_output_path, filename)
    fig.savefig(figureFile, dpi=100, bbox_inches="tight")

    plt.close()

def plot_pcawg_heatmap_seismic_colorbar(heatmaps_main_output_path):
    os.makedirs(os.path.join(heatmaps_main_output_path, FIGURES_MANUSCRIPT), exist_ok=True)
    heatmap_output_path=os.path.join(heatmaps_main_output_path, FIGURES_MANUSCRIPT)

    # Make a figure and axes with dimensions as desired.
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_axes([0.05, 0.475, 0.9, 0.15])

    bounds = np.arange(0.25, 1.80, 0.25)
    norm = mpl.colors.Normalize(vmin=min(bounds), vmax=max(bounds))
    cbar = mpl.colorbar.ColorbarBase(ax, cmap=plt.get_cmap("seismic"), norm=norm, ticks=bounds, spacing='proportional',orientation='horizontal')

    cbar.set_label('Fold Change\n [Real mutations/Simulated Mutations]', fontsize=30,labelpad=10)
    cbar.ax.tick_params(labelsize=25)

    filename = 'heatmaps_seismic_color_bar.png'
    figureFile = os.path.join(heatmap_output_path, filename)
    fig.savefig(figureFile, dpi=100, bbox_inches="tight")

    plt.close()


def plot_pcawg_heatmap_discreet_colorbar(heatmaps_main_output_path):
    os.makedirs(os.path.join(heatmaps_main_output_path, FIGURES_MANUSCRIPT), exist_ok=True)
    heatmap_output_path=os.path.join(heatmaps_main_output_path, FIGURES_MANUSCRIPT)

    # Make a figure and axes with dimensions as desired.
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_axes([0.05, 0.475, 0.9, 0.15])

    # If a ListedColormap is used, the length of the bounds array must be
    # one greater than the length of the color list.  The bounds must be
    # monotonically increasing.
    cmap = mpl.colors.ListedColormap(['#131E3A', 'darkblue', 'blue', '#0080FE', '#89CFEF', 'white', 'salmon', 'indianred', 'red', 'darkred','#420D09'])
    bounds = [0.25, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.75]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    cmap.set_over('0.25')
    cmap.set_under('1.75')

    cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    boundaries=bounds,
                                    ticks=bounds,  # optional
                                    spacing='proportional',
                                    orientation='horizontal')

    # cbar.set_label('Fold Change: real over simulations',fontsize=20)
    cbar.set_label('Fold Change [Real mutations/Simulated Mutations]', fontsize=20)

    cbar.ax.tick_params(labelsize=12)

    filename = 'heatmaps_discreet_color_bar.png'
    figureFile = os.path.join(heatmap_output_path, filename)
    fig.savefig(figureFile, dpi=100, bbox_inches="tight")

    plt.close()


def heatmap_with_pie_chart(data_array,
                           row_labels, # normally rows signatures
                           column_labels, # normally columns DNA element
                           signature_type,
                           heatmap_output_path,
                           figure_type,
                           plot_legend,
                           cosmic_release_version,
                           figure_file_extension,
                           tissue_based = None,
                           signature_tissue_type_tuples = None,
                           number_of_columns_in_legend = 3,
                           heatmap_rows_signatures_columns_dna_elements = True):

    dpi = 100
    squaresize = 200  # pixels

    if len(column_labels) <= 3:
        figwidth = 2 * len(column_labels) * squaresize / float(dpi)
    else:
        figwidth = len(column_labels) * squaresize / float(dpi)

    figheight = len(row_labels) * squaresize / float(dpi)

    # Set font size w.r.t. number of rows
    # fontsize = 10 * np.sqrt(len(data_array))
    # fontsize = 2 * np.sqrt(len(row_labels)*len(column_labels))
    if signature_type == SBS:
        fontsize = 60 #58
    elif signature_type == DBS:
        fontsize = 60 #50
    else:
        fontsize = 60

    fig, ax = plt.subplots(1, figsize=(figwidth, figheight), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    print('Heatmap with pie charts --- %s num_of_rows:%d num_of_columns:%d figwidth:%d figheight:%d fontsize:%d' %(signature_type, len(row_labels), len(column_labels),  figwidth, figheight, fontsize))

    ax.set_aspect(1.0)
    ax.set_facecolor('white')

    for row_index, row in enumerate(row_labels, 0):
        for column_index, column in enumerate(column_labels, 0):
            signature_tissue_type = None
            if signature_tissue_type_tuples:
                # row can be SBS4 or SBS4 Lung-AdenoCA
                if heatmap_rows_signatures_columns_dna_elements:
                    signature_tissue_type = signature_tissue_type_tuples[row_index][1]
                else:
                    signature_tissue_type = signature_tissue_type_tuples[column_index][1]

            # Values is a tuple of 3 integer
            values = data_array[row_index, column_index]
            labels = [str(value) for value in values]
            colors = ['indianred', 'cornflowerblue', 'silver']

            # Removes 0s
            mask = [True if value>0 else False for value in values]
            values = (np.array(values)[mask]).tolist()
            labels = (np.array(labels)[mask]).tolist()
            colors = (np.array(colors)[mask]).tolist()

            wedges, text = ax.pie(values, labels = labels, labeldistance = 0.20, colors = colors, textprops = {'fontsize': 30})

            radius = 0.45
            for w in wedges:
                # We want to see the data_array[0,0] at left most top instead of left most bottom
                w.set_center((column_index, len(row_labels)-row_index-1)) # legacy
                w.set_radius(radius)
                # w.set_edgecolor('white')

            for t in text:
                if (not tissue_based) and (not signature_tissue_type):
                    x, y = t.get_position()
                    # We want to see the data_array[0,0] at left most top instead of left most bottom
                    t.set_position((x + column_index, y + len(row_labels)-row_index-1)) # legacy
                if tissue_based:
                    # Do not show 1s if the heatmap is tissue based
                    x, y = t.get_position()
                    t.set_position((x + column_index, y + len(row_labels)-row_index-1)) # legacy
                    t.set_text('')
                if signature_tissue_type:
                    x, y = t.get_position()
                    t.set_position((x + column_index, y + len(row_labels)-row_index-1)) # legacy
                    t.set_text('')

    legend_elements = [
        Line2D([0], [0], marker='o', color='white', label=ENRICHMENT_OF_MUTATIONS, markerfacecolor='indianred', markersize=30),
        Line2D([0], [0], marker='o', color='white', label=DEPLETION_OF_MUTATIONS, markerfacecolor='cornflowerblue', markersize=30),
        Line2D([0], [0], marker='o', color='white', label=NO_EFFECT, markerfacecolor='silver', markersize=30)]
    if plot_legend:
        if heatmap_rows_signatures_columns_dna_elements:
            ax.legend(handles=legend_elements, ncol = number_of_columns_in_legend, loc="upper right", bbox_to_anchor=(1, 0), fontsize=40) # one row
        else:
            # rows DNA elements, columns signatures
            # when there are a few signatures, it would be good to have legends in one column
            ax.legend(handles=legend_elements, ncol = 1, loc="upper right", bbox_to_anchor=(1, 0), fontsize=40) # one row

    # We want to show all ticks...
    ax.set_xticks(np.arange(data_array.shape[1]))
    ax.set_yticks(np.arange(data_array.shape[0]))

    # Remove ATAC from columns
    # Remove -human from columns
    if heatmap_rows_signatures_columns_dna_elements:
        column_labels = ['Open Chromatin' if 'ATAC' in row else row for row in column_labels]
        column_labels = [column[:-6] if '-human' in column else column for column in column_labels]
    else:
        row_labels = ['Open Chromatin' if 'ATAC' in row else row for row in row_labels]
        row_labels = [column[:-6] if '-human' in column else column for column in row_labels]

    # Tick labels are set since ticks are set above
    ax.set_xticklabels(column_labels, fontsize=fontsize)
    # Reverse the row labels so that left most top is the first one
    # Otherwise left most bottom is the first one
    # Do not use row_labels.reverse() since it reverse in place and return nothing
    # row_labels[::-1] returns the reversed list, do not reverse the input list
    ax.set_yticklabels(row_labels[::-1], fontsize=fontsize)

    # X axis labels at top
    ax.tick_params(left=False, top=False, bottom=False, right=False, labelbottom=False, labeltop=True, pad=5)
    plt.setp(ax.get_xticklabels(), rotation=55, ha="left", rotation_mode="anchor")

    # We want to show all minor ticks...
    ax.set_xticks(np.arange(data_array.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data_array.shape[0] + 1) - .5, minor=True)

    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlim(xmin = -0.5)
    if len(row_labels) > 1:
        ax.set_ylim(ymin = -0.5)
    elif len(row_labels) == 1:
        ax.set_ylim(ymin = -0.5,ymax = 0.5)

    if figure_type == COSMIC:
        feature_name = "HISTONE_MODS"
        # v3.2_SBS1_REPLIC_ASYM_TA_C4817.jpg
        if tissue_based:
            ax.set_title(tissue_based,fontsize=fontsize)
            NCI_Thesaurus_code = cancer_type_2_NCI_Thesaurus_code_dict[tissue_based]
            if heatmap_rows_signatures_columns_dna_elements:
                filename = '%s_%s_%s_TA_%s.%s' % (cosmic_release_version, row_labels[0].replace(' ','_'), feature_name, NCI_Thesaurus_code, figure_file_extension)
            else:
                filename = '%s_%s_%s_TA_%s.%s' % (cosmic_release_version, column_labels[0].replace(' ','_'), feature_name, NCI_Thesaurus_code, figure_file_extension)

        else:
            if heatmap_rows_signatures_columns_dna_elements:
                filename = '%s_%s_%s.%s' % (cosmic_release_version, row_labels[0].split()[0], feature_name, figure_file_extension)
            else:
                filename = '%s_%s_%s.%s' % (cosmic_release_version, column_labels[0].split()[0], feature_name, figure_file_extension)
    elif figure_type == MANUSCRIPT:
        filename = '%s_heatmap_with_pie_charts.png' %(signature_type)
    figureFile = os.path.join(heatmap_output_path, filename)
    fig.savefig(figureFile, dpi=100, bbox_inches="tight")

    plt.cla()
    plt.close(fig)


# considered_signatures can be across all tissues and tissue specific mixed
def fill_data_array(considered_signatures,
                    considered_dna_elements,
                    signature2dna_element2cancer_type_list_dict,
                    signature_tissue_type_tuples = None,
                    row_contains_signature = True):

    if row_contains_signature:
        # Rows signatures, columns dna_elements
        data_array = np.zeros((len(considered_signatures), len(considered_dna_elements)), dtype=(int, 3))
    else:
        # Rows dna_elements, columns signatures
        data_array = np.zeros((len(considered_dna_elements), len(considered_signatures)), dtype=(int, 3))

    # legacy
    for signature_index, signature in enumerate(considered_signatures,0):
        for dna_element_index, dna_element in enumerate(considered_dna_elements,0):
            enriched_cancer_types = signature2dna_element2cancer_type_list_dict[signature][dna_element][ENRICHED_CANCER_TYPES]
            depleted_cancer_types = signature2dna_element2cancer_type_list_dict[signature][dna_element][DEPLETED_CANCER_TYPES]
            other_cancer_types = signature2dna_element2cancer_type_list_dict[signature][dna_element][OTHER_CANCER_TYPES]

            if row_contains_signature:
                if signature_tissue_type_tuples:
                    tissue_type = signature_tissue_type_tuples[signature_index][1]
                    if tissue_type:
                        if tissue_type in enriched_cancer_types:
                            # Tissue specific
                            data_array[signature_index, dna_element_index] = [1, 0, 0]
                        if tissue_type in depleted_cancer_types:
                            # Tissue specific
                            data_array[signature_index, dna_element_index] = [0, 1, 0]
                        if tissue_type in other_cancer_types:
                            # Tissue specific
                            data_array[signature_index, dna_element_index] = [0, 0, 1]
                    else:
                        # Across all tissues
                        data_array[signature_index, dna_element_index] = [len(enriched_cancer_types), len(depleted_cancer_types), len(other_cancer_types)]
                else:
                    # Across all tissues
                    data_array[signature_index, dna_element_index] = [len(enriched_cancer_types), len(depleted_cancer_types), len(other_cancer_types)]
            else:
                if signature_tissue_type_tuples:
                    tissue_type = signature_tissue_type_tuples[signature_index][1]
                    if tissue_type:
                        if tissue_type in enriched_cancer_types:
                            # Tissue specific
                            data_array[dna_element_index, signature_index] = [1, 0, 0]
                        if tissue_type in depleted_cancer_types:
                            # Tissue specific
                            data_array[dna_element_index, signature_index] = [0, 1, 0]
                        if tissue_type in other_cancer_types:
                            # Tissue specific
                            data_array[dna_element_index, signature_index] = [0, 0, 1]
                    else:
                        # Across all tissues
                        data_array[dna_element_index, signature_index] = [len(enriched_cancer_types), len(depleted_cancer_types), len(other_cancer_types)]
                else:
                    # Across all tissues
                    data_array[dna_element_index, signature_index] = [len(enriched_cancer_types), len(depleted_cancer_types), len(other_cancer_types)]

    return data_array


def fill_signature2dna_element2cancer_type_list_dict(all_dna_elements,
                                                     all_signatures,
                                                     cancer_types,
                                                     step3_q_value_df,
                                                     enriched_fold_change,
                                                     depleted_fold_change,
                                                     significance_level):

    signature2dna_element2cancer_type_list_dict = {}

    # We need to decide which signatures to consider
    # We need to decide which dna_elements to consider
    # We consider any (signature, dna_element) tuple which have at least one cancer type either enriched or depleted with corrected_p_value <= significance level
    considered_dna_elements=[]
    considered_signatures=[]

    for dna_element in all_dna_elements:
        for signature in all_signatures:
            enriched_cancer_types = []
            depleted_cancer_types = []
            other_cancer_types = []

            # Is there a cancer_type either enriched or depleted with corrected_p_value <= significance level?
            # If yes consider this (dna_element,signature)
            if step3_q_value_df[(step3_q_value_df['dna_element'] == dna_element) & (step3_q_value_df['signature'] == signature)]['cancer_type'].any():
                #Get the cancer_types enriched and statistically significant
                #Get the cancer_types depleted and statistically significant
                cancer_types_array = step3_q_value_df[(step3_q_value_df['dna_element'] == dna_element) & (step3_q_value_df['signature'] == signature)]['cancer_type'].values
                for i in range(cancer_types_array.size):
                    cancer_type = cancer_types_array[i]
                    avg_fold_change = step3_q_value_df[(step3_q_value_df['dna_element'] == dna_element) & (step3_q_value_df['signature'] == signature) & (step3_q_value_df['cancer_type'] == cancer_type)]['avg_fold_change'].values[0]
                    q_value = step3_q_value_df[(step3_q_value_df['dna_element'] == dna_element) & (step3_q_value_df['signature'] == signature) & (step3_q_value_df['cancer_type']==cancer_type)]['q_value'].values[0]

                    if (cancer_type in cancer_types):
                        if (enriched(avg_fold_change, enriched_fold_change) or depleted(avg_fold_change, depleted_fold_change)) and (q_value <= significance_level):
                            if dna_element not in considered_dna_elements:
                                considered_dna_elements.append(dna_element)
                            if signature not in considered_signatures:
                                considered_signatures.append(signature)
                            if enriched(avg_fold_change, enriched_fold_change):
                                enriched_cancer_types.append(cancer_type)
                            elif depleted(avg_fold_change, depleted_fold_change):
                                depleted_cancer_types.append(cancer_type)
                        else:
                            #Not statistically significant or avg_fold_change is between (depleted_fold_change,enriched_fold_change)
                            other_cancer_types.append(cancer_type)

            print('%s %s enriched_cancer_types: %s --- depleted_cancer_types: %s --- other_cancer_types: %s' %(signature,dna_element,enriched_cancer_types,depleted_cancer_types,other_cancer_types))
            if signature in signature2dna_element2cancer_type_list_dict:
                if dna_element in signature2dna_element2cancer_type_list_dict[signature]:
                    pass  # Do nothing if already in the dictionary
                else:
                    signature2dna_element2cancer_type_list_dict[signature][dna_element]={}
                    signature2dna_element2cancer_type_list_dict[signature][dna_element][ENRICHED_CANCER_TYPES]= enriched_cancer_types
                    signature2dna_element2cancer_type_list_dict[signature][dna_element][DEPLETED_CANCER_TYPES]= depleted_cancer_types
                    signature2dna_element2cancer_type_list_dict[signature][dna_element][OTHER_CANCER_TYPES] = other_cancer_types
            else:
                signature2dna_element2cancer_type_list_dict[signature] = {}
                signature2dna_element2cancer_type_list_dict[signature][dna_element] = {}
                signature2dna_element2cancer_type_list_dict[signature][dna_element][ENRICHED_CANCER_TYPES] = enriched_cancer_types
                signature2dna_element2cancer_type_list_dict[signature][dna_element][DEPLETED_CANCER_TYPES] = depleted_cancer_types
                signature2dna_element2cancer_type_list_dict[signature][dna_element][OTHER_CANCER_TYPES] = other_cancer_types

    return signature2dna_element2cancer_type_list_dict, considered_dna_elements, considered_signatures


def fill_lists(signature, signature_type, df):
    signatures = []
    signature_signature_type_tuples = []
    signature_tissue_type_tuples = []
    signatures_ylabels_on_the_heatmap = []

    if df[df['signature'] == signature]['cancer_type'].values.any():
        cancer_types = df[df['signature'] == signature]['cancer_type'].values
        cancer_types = sorted(cancer_types, key=natural_key)

        signatures = [ signature ]
        signature_signature_type_tuples = [(signature, signature_type)]
        signature_tissue_type_tuples = [(signature, None)]
        signature_with_num_of_cancer_types = '%s (n=%d)' % (signature, len(cancer_types))

        signatures_ylabels_on_the_heatmap = [signature_with_num_of_cancer_types]

        for cancer_type in cancer_types:
            signatures.append(signature)
            signature_signature_type_tuples.append((signature, signature_type))
            signature_tissue_type_tuples.append((signature, cancer_type))
            signatures_ylabels_on_the_heatmap.append(cancer_type)

    return signatures, signature_signature_type_tuples, signature_tissue_type_tuples, signatures_ylabels_on_the_heatmap


def process_dataframes(step1_p_value_df):
    # step1_p_value_df.dtypes:
    # signature                         object
    # cancer_type                       object
    # cutoff                           float64
    # number_of_mutations                int64
    # biosample                         object
    # dna_element                       object
    # avg_real_signal                  float64
    # avg_simulated_signal             float64
    # fold_change                      float64
    # min_sim_signal                   float64
    # max_sim_signal                   float64
    # p_value                          float64
    # num_of_sims                        int64
    # num_of_sims_with_not_nan_avgs     object
    # real_data_avg_count              float64
    # sim_avg_count                    float64
    # sim_signals                       object
    # dtype: object

    # round
    step1_p_value_df['avg_real_signal'] = np.around(step1_p_value_df['avg_real_signal'], NUMBER_OF_DECIMAL_PLACES_TO_ROUND)
    step1_p_value_df['avg_simulated_signal'] = np.around(step1_p_value_df['avg_simulated_signal'], NUMBER_OF_DECIMAL_PLACES_TO_ROUND)
    step1_p_value_df['fold_change'] = np.around(step1_p_value_df['fold_change'], NUMBER_OF_DECIMAL_PLACES_TO_ROUND)
    step1_p_value_df['min_sim_signal'] = np.around(step1_p_value_df['min_sim_signal'], NUMBER_OF_DECIMAL_PLACES_TO_ROUND)
    step1_p_value_df['max_sim_signal'] = np.around(step1_p_value_df['max_sim_signal'], NUMBER_OF_DECIMAL_PLACES_TO_ROUND)
    step1_p_value_df['real_data_avg_count'] = np.around(step1_p_value_df['real_data_avg_count'], NUMBER_OF_DECIMAL_PLACES_TO_ROUND)
    step1_p_value_df['sim_avg_count'] = np.around(step1_p_value_df['sim_avg_count'], NUMBER_OF_DECIMAL_PLACES_TO_ROUND)

    # drop columns
    step1_p_value_df.drop(columns=['num_of_sims', 'num_of_sims_with_not_nan_avgs'], inplace=True)

    # rename column names
    step1_p_value_df.rename(columns={'real_data_avg_count': 'real_mutations_avg_overlaps',
                                    'sim_avg_count': 'sim_mutations_avg_overlaps'}, inplace=True)


    return step1_p_value_df


# Pie chart uses step3 results
# In the pie chart, any signature with at least one significant result is shown.
# significant mean q_value <= 0.05 and ((avg_fold_change >= 1.05) or (avg_fold_change <= 0.95))
# not significant mean q_value > 0.05 or ((avg_fold_change < 1.05) and (avg_fold_change > 0.95))
# if a signature is shown not significant results are also shown in gray.
def plot_heatmaps_rows_signatures_columns_dna_elements_with_pie_charts(step3_q_value_df,
                                                                       cancer_type_signature_cutoff_number_of_mutations_average_probability_df,
                                                                       signatureType,
                                                                       all_signatures,
                                                                       all_dna_elements,
                                                                       cancer_types,
                                                                       depleted_fold_change,
                                                                       enriched_fold_change,
                                                                       significance_level,
                                                                       heatmaps_main_output_path,
                                                                       figure_type,
                                                                       cosmic_release_version,
                                                                       figure_file_extension,
                                                                       heatmap_rows_signatures_columns_dna_elements = True,
                                                                       plot_legend = True):

    if figure_type == COSMIC:
        os.makedirs(os.path.join(heatmaps_main_output_path, FIGURES_COSMIC), exist_ok=True)
        os.makedirs(os.path.join(heatmaps_main_output_path, COSMIC_TISSUE_BASED_FIGURES), exist_ok=True)
        heatmap_output_path = os.path.join(heatmaps_main_output_path, FIGURES_COSMIC)
        cosmic_tissue_based_heatmaps_output_path = os.path.join(heatmaps_main_output_path, COSMIC_TISSUE_BASED_FIGURES)

        # In COSMIC case all_signatures will contain only one signature
        signatures, \
        signature_signature_type_tuples, \
        signature_tissue_type_tuples, \
        signatures_ylabels_on_the_heatmap = fill_lists(all_signatures[0],
                                                       signatureType,
                                                       cancer_type_signature_cutoff_number_of_mutations_average_probability_df)

        # Update all_signatures
        # We want to have as many rows in heatmap as signatures
        all_signatures = signatures

    elif figure_type == MANUSCRIPT:
        os.makedirs(os.path.join(heatmaps_main_output_path, FIGURES_MANUSCRIPT), exist_ok=True)
        heatmap_output_path = os.path.join(heatmaps_main_output_path, FIGURES_MANUSCRIPT)
        signature_tissue_type_tuples = None


    # Same for COSMIC and MANUSCRIPT
    signature2dna_element2cancer_type_list_dict, \
    considered_dna_elements, \
    considered_signatures = fill_signature2dna_element2cancer_type_list_dict(all_dna_elements,
                                                                             all_signatures,
                                                                             cancer_types,
                                                                             step3_q_value_df,
                                                                             enriched_fold_change,
                                                                             depleted_fold_change,
                                                                             significance_level)

    # Fill data_array w.r.t. considered_dna_elements and considered_signatures
    considered_signatures = sorted(considered_signatures, key=natural_key)
    considered_dna_elements = sorted(considered_dna_elements, key=natural_key)

    if figure_type == COSMIC:
        data_array = fill_data_array(signatures,
                                     all_dna_elements,
                                     signature2dna_element2cancer_type_list_dict,
                                     signature_tissue_type_tuples = signature_tissue_type_tuples,
                                     row_contains_signature = heatmap_rows_signatures_columns_dna_elements)

        if (np.sum(data_array) > 0) and (len(signatures) > 0) and (len(all_dna_elements) > 0):
            if heatmap_rows_signatures_columns_dna_elements:
                # rows signatures columns DNA elements
                heatmap_with_pie_chart(data_array,
                                       signatures_ylabels_on_the_heatmap, # legacy considered_signatures
                                       all_dna_elements,
                                       signatureType,
                                       heatmap_output_path,
                                       figure_type,
                                       plot_legend,
                                       cosmic_release_version,
                                       figure_file_extension,
                                       signature_tissue_type_tuples = signature_tissue_type_tuples)
            else:
                # rows DNA elements columns signatures
                heatmap_with_pie_chart(data_array,
                                       all_dna_elements,
                                       signatures_ylabels_on_the_heatmap, # legacy considered_signatures
                                       signatureType,
                                       heatmap_output_path,
                                       figure_type,
                                       plot_legend,
                                       cosmic_release_version,
                                       figure_file_extension,
                                       signature_tissue_type_tuples = signature_tissue_type_tuples,
                                       heatmap_rows_signatures_columns_dna_elements = heatmap_rows_signatures_columns_dna_elements)

    if figure_type == MANUSCRIPT:
        data_array = fill_data_array(considered_signatures,
                                     considered_dna_elements,
                                     signature2dna_element2cancer_type_list_dict,
                                     signature_tissue_type_tuples = signature_tissue_type_tuples)

        if len(considered_signatures) > 0 and len(considered_dna_elements) > 0:
            heatmap_with_pie_chart(data_array,
                               considered_signatures,
                               considered_dna_elements,
                               signatureType,
                               heatmap_output_path,
                               figure_type,
                               plot_legend,
                               cosmic_release_version,
                               figure_file_extension,
                               signature_tissue_type_tuples = signature_tissue_type_tuples)

    # Cancer type based Cosmic Figures
    if figure_type == COSMIC:
        tissue_based_2_data_array_dict = {}

        for signature_index, signature in enumerate(considered_signatures, 0):
            for dna_element_index, dna_element in enumerate(all_dna_elements, 0): # legacy considered_dna_elements
                enriched_cancer_types = signature2dna_element2cancer_type_list_dict[signature][dna_element][
                    ENRICHED_CANCER_TYPES]
                depleted_cancer_types = signature2dna_element2cancer_type_list_dict[signature][dna_element][
                    DEPLETED_CANCER_TYPES]
                other_cancer_types = signature2dna_element2cancer_type_list_dict[signature][dna_element][
                    OTHER_CANCER_TYPES]

                all_cancer_types = enriched_cancer_types + depleted_cancer_types + other_cancer_types
                for cancer_type in all_cancer_types:
                    enriched = 0
                    depleted = 0
                    no_effect = 0

                    if cancer_type in enriched_cancer_types:
                        enriched = 1
                    if cancer_type in depleted_cancer_types:
                        depleted = 1
                    if cancer_type in other_cancer_types:
                        no_effect = 1

                    if cancer_type not in tissue_based_2_data_array_dict:
                        tissue_based_2_data_array_dict[cancer_type] = np.zeros(
                            (len(considered_signatures), len(all_dna_elements)), dtype=(int, 3))
                        tissue_based_2_data_array_dict[cancer_type][signature_index, dna_element_index] = [enriched,
                                                                                                           depleted,
                                                                                                           no_effect]
                    else:
                        tissue_based_2_data_array_dict[cancer_type][signature_index, dna_element_index] = [enriched,
                                                                                                           depleted,
                                                                                                           no_effect]
        for tissue_type in tissue_based_2_data_array_dict:
            tissue_based_data_array = tissue_based_2_data_array_dict[tissue_type]
            heatmap_with_pie_chart(tissue_based_data_array,
                                   considered_signatures,
                                   all_dna_elements, # legacy considered_dna_elements
                                   signatureType,
                                   cosmic_tissue_based_heatmaps_output_path,
                                   figure_type,
                                   plot_legend,
                                   cosmic_release_version,
                                   figure_file_extension,
                                   tissue_based = tissue_type)


# Enrichment is done in this function.
# Always ztest
# one sample or two_sample?
# I decided to use one sample because for simulations I will get vertical vector and average of that vertical vector  must be equal to avg_simulated_signal, there is a way to self verification
# Comparing one mean with means of n simulations gives a more realistic p-value.
# In case of comparison of two samples, ztest and ttest gives either 0 or very low p-values.
def calculate_fold_change_real_over_sim(plusorMinus,
                                        main_combined_output_dir, # combined_output_dir
                                        numberofSimulations,
                                        signature,
                                        main_cancer_type, # cancer_type
                                        cutoff,
                                        number_of_mutations,
                                        biosample,
                                        dna_element,
                                        occupancy_type):

    if (occupancy_type == EPIGENOMICSOCCUPANCY):
        start = epigenomics_center-plusorMinus
        end = epigenomics_center+plusorMinus+1
    elif (occupancy_type == NUCLEOSOMEOCCUPANCY):
        start = nucleosome_center-plusorMinus
        end = nucleosome_center+plusorMinus+1

    avg_real_signal = None
    avg_sim_signal = None
    min_sim_signal = None
    max_sim_signal = None
    fold_change = None

    real_data_avg_count = None
    sim_avg_count = None

    num_of_sims_with_not_nan_avgs = None
    pvalue = None

    simulations_horizontal_means_array = None

    if dna_element == NUCLEOSOME:
        dna_element_to_be_read = None
    else:
        dna_element_to_be_read = dna_element

    # Read avg_real_data
    # SBS1_sim1_ENCFF330CCJ_osteoblast_H3K79me2-human_AverageSignalArray.txt
    if (signature == ALL_SUBSTITUTIONS):
        avg_real_data_signal_array = readData(None, None, AGGREGATEDSUBSTITUTIONS, main_combined_output_dir, main_cancer_type, occupancy_type, dna_element_to_be_read,AVERAGE_SIGNAL_ARRAY)
    elif (signature == ALL_DINUCLEOTIDES):
        avg_real_data_signal_array = readData(None, None, AGGREGATEDDINUCS, main_combined_output_dir, main_cancer_type, occupancy_type, dna_element_to_be_read, AVERAGE_SIGNAL_ARRAY)
    elif (signature == ALL_INDELS):
        avg_real_data_signal_array = readData(None, None, AGGREGATEDINDELS, main_combined_output_dir, main_cancer_type,occupancy_type, dna_element_to_be_read,AVERAGE_SIGNAL_ARRAY)
    else:
        combined_output_dir, cancer_type = get_alternative_combined_output_dir_and_cancer_type(main_combined_output_dir, main_cancer_type, signature)
        avg_real_data_signal_array = readData(None, signature, SIGNATUREBASED, combined_output_dir, cancer_type, occupancy_type, dna_element_to_be_read,AVERAGE_SIGNAL_ARRAY)

    if avg_real_data_signal_array is not None:
        # If there is nan in the list np.mean returns nan.
        try:
            avg_real_signal = np.nanmean(avg_real_data_signal_array[start:end])
        except RuntimeWarning:
            avg_real_signal = np.nan
            print('RuntimeWarning', ' avg_real_signal: ', avg_real_signal, ' avg_real_data_signal_array[start:end]: ', avg_real_data_signal_array[start:end])

    # Read real_count
    # Read accumulated_count_array
    if (signature == ALL_SUBSTITUTIONS):
        real_data_accumulated_count_array = readData(None, None, AGGREGATEDSUBSTITUTIONS, main_combined_output_dir, main_cancer_type, occupancy_type, dna_element_to_be_read, ACCUMULATED_COUNT_ARRAY)
    elif (signature == ALL_DINUCLEOTIDES):
        real_data_accumulated_count_array = readData(None, None, AGGREGATEDDINUCS, main_combined_output_dir, main_cancer_type, occupancy_type, dna_element_to_be_read, ACCUMULATED_COUNT_ARRAY)
    elif (signature == ALL_INDELS):
        real_data_accumulated_count_array = readData(None, None, AGGREGATEDINDELS, main_combined_output_dir, main_cancer_type, occupancy_type, dna_element_to_be_read, ACCUMULATED_COUNT_ARRAY)
    else:
        combined_output_dir, cancer_type = get_alternative_combined_output_dir_and_cancer_type(main_combined_output_dir, main_cancer_type, signature)
        real_data_accumulated_count_array = readData(None, signature, SIGNATUREBASED, combined_output_dir, cancer_type, occupancy_type, dna_element_to_be_read, ACCUMULATED_COUNT_ARRAY)

    if real_data_accumulated_count_array is not None:
        #If there is nan in the list np.mean returns nan.
        real_data_avg_count = np.nanmean(real_data_accumulated_count_array[start:end])

    # Get avg_sim_signal
    # Get min_sim_signal
    # Get max_sim_signal
    if (numberofSimulations > 0):
        if (signature == ALL_SUBSTITUTIONS):
            listofSimulationsSignatureBased = readDataForSimulations(None, None, AGGREGATEDSUBSTITUTIONS, main_combined_output_dir, main_cancer_type, numberofSimulations, occupancy_type, dna_element_to_be_read, AVERAGE_SIGNAL_ARRAY)
        elif (signature == ALL_DINUCLEOTIDES):
            listofSimulationsSignatureBased = readDataForSimulations(None, None, AGGREGATEDDINUCS, main_combined_output_dir, main_cancer_type, numberofSimulations, occupancy_type, dna_element_to_be_read, AVERAGE_SIGNAL_ARRAY)
        elif (signature == ALL_INDELS):
            listofSimulationsSignatureBased = readDataForSimulations(None, None, AGGREGATEDINDELS, main_combined_output_dir, main_cancer_type, numberofSimulations, occupancy_type, dna_element_to_be_read, AVERAGE_SIGNAL_ARRAY)
        else:
            combined_output_dir, cancer_type = get_alternative_combined_output_dir_and_cancer_type(main_combined_output_dir, main_cancer_type, signature)
            listofSimulationsSignatureBased = readDataForSimulations(None, signature, SIGNATUREBASED, combined_output_dir, cancer_type, numberofSimulations, occupancy_type, dna_element_to_be_read, AVERAGE_SIGNAL_ARRAY)

        if ((listofSimulationsSignatureBased is not None) and listofSimulationsSignatureBased):
            # This is the simulations data
            stackedSimulationsSignatureBased = np.vstack(listofSimulationsSignatureBased)
            (rows, cols) = stackedSimulationsSignatureBased.shape
            num_of_sims = rows

            # One sample way
            print('stackedSimulationsSignatureBased.shape', stackedSimulationsSignatureBased.shape)
            stackedSimulationsSignatureBased_of_interest = stackedSimulationsSignatureBased[:,start:end]
            print('stackedSimulationsSignatureBased_of_interest.shape', stackedSimulationsSignatureBased_of_interest.shape)

            # Get rid of rows with all nans
            stackedSimulationsSignatureBased_of_interest = stackedSimulationsSignatureBased_of_interest[~np.isnan(stackedSimulationsSignatureBased_of_interest).all(axis=1)]

            # Take mean row-wise
            simulations_horizontal_means_array = np.nanmean(stackedSimulationsSignatureBased_of_interest, axis=1)
            avg_sim_signal = np.nanmean(simulations_horizontal_means_array)
            min_sim_signal = np.nanmin(simulations_horizontal_means_array)
            max_sim_signal = np.nanmax(simulations_horizontal_means_array)
            print('simulations_horizontal_means_array.shape', simulations_horizontal_means_array.shape)

    # Get sim_avg_count
    if (numberofSimulations > 0):
        if (signature == ALL_SUBSTITUTIONS):
            listofSimulationsSignatureBasedCount = readDataForSimulations(None, None, AGGREGATEDSUBSTITUTIONS, main_combined_output_dir, main_cancer_type, numberofSimulations, occupancy_type,dna_element_to_be_read, ACCUMULATED_COUNT_ARRAY)
        elif (signature == ALL_INDELS):
            listofSimulationsSignatureBasedCount = readDataForSimulations(None, None, AGGREGATEDINDELS, main_combined_output_dir, main_cancer_type, numberofSimulations, occupancy_type,dna_element_to_be_read, ACCUMULATED_COUNT_ARRAY)
        elif (signature == ALL_DINUCLEOTIDES):
            listofSimulationsSignatureBasedCount = readDataForSimulations(None, None, AGGREGATEDDINUCS, main_combined_output_dir, main_cancer_type, numberofSimulations, occupancy_type,dna_element_to_be_read, ACCUMULATED_COUNT_ARRAY)
        else:
            combined_output_dir, cancer_type = get_alternative_combined_output_dir_and_cancer_type(main_combined_output_dir, main_cancer_type, signature)
            listofSimulationsSignatureBasedCount = readDataForSimulations(None, signature, SIGNATUREBASED, combined_output_dir, cancer_type, numberofSimulations, occupancy_type,dna_element_to_be_read, ACCUMULATED_COUNT_ARRAY)

        if ((listofSimulationsSignatureBasedCount is not None) and listofSimulationsSignatureBasedCount):

            # This is the simulations data
            stackedSimulationsSignatureBasedCount = np.vstack(listofSimulationsSignatureBasedCount)

            stackedSimulationsSignatureBasedCount_of_interest = stackedSimulationsSignatureBasedCount[:,start:end]

            # Get rid of rows with all nans
            stackedSimulationsSignatureBasedCount_of_interest = stackedSimulationsSignatureBasedCount_of_interest[~np.isnan(stackedSimulationsSignatureBasedCount_of_interest).all(axis=1)]

            # Take mean row-wise
            simulations_horizontal_count_means_array = np.nanmean(stackedSimulationsSignatureBasedCount_of_interest, axis=1)
            sim_avg_count = np.nanmean(simulations_horizontal_count_means_array)

    if (avg_real_signal is not None) and (avg_sim_signal is not None):
        try:
            fold_change = avg_real_signal / avg_sim_signal
        except ZeroDivisionError:
            fold_change = np.nan
            print('avg_real_signal:%f' % (avg_real_signal))
            print('avg_sim_signal:%f' % (avg_sim_signal))
            print('fold change:%f' % (fold_change))

        if (simulations_horizontal_means_array is not None):
            zstat, pvalue = calculate_pvalue_teststatistics(avg_real_signal, simulations_horizontal_means_array)

            print('%s %s %s  avg_real_signal:%f avg_sim_signal:%f min_sim_signal:%f max_sim_signal:%f fold_change:%f p_value: %.2E' %(signature, main_cancer_type, dna_element,avg_real_signal,avg_sim_signal,min_sim_signal,max_sim_signal, fold_change, Decimal(pvalue) ))
            print('###############################################################################################################################')

        # return [signature, main_cancer_type, cutoff, number_of_mutations, biosample, dna_element, avg_real_signal, avg_sim_signal, fold_change, min_sim_signal, max_sim_signal, pvalue, num_of_sims, num_of_sims_with_not_nan_avgs, real_data_avg_count, sim_avg_count, list(simulations_horizontal_means_array)]
        return [signature, main_cancer_type, cutoff, number_of_mutations, biosample, dna_element, avg_real_signal, avg_sim_signal, fold_change, min_sim_signal, max_sim_signal, pvalue, num_of_sims, num_of_sims_with_not_nan_avgs, real_data_avg_count, sim_avg_count, np.around(simulations_horizontal_means_array, NUMBER_OF_DECIMAL_PLACES_TO_ROUND).tolist()]
    else:
        print('%s %s %s %s nan' %(signature, main_cancer_type, biosample, dna_element))
        if (simulations_horizontal_means_array is not None):
            # return [signature, main_cancer_type, cutoff, number_of_mutations, biosample, dna_element, avg_real_signal, avg_sim_signal, fold_change, min_sim_signal, max_sim_signal, pvalue, num_of_sims, num_of_sims_with_not_nan_avgs, real_data_avg_count, sim_avg_count, list(simulations_horizontal_means_array)]
            return [signature, main_cancer_type, cutoff, number_of_mutations, biosample, dna_element, avg_real_signal, avg_sim_signal, fold_change, min_sim_signal, max_sim_signal, pvalue, num_of_sims, num_of_sims_with_not_nan_avgs, real_data_avg_count, sim_avg_count, np.around(simulations_horizontal_means_array, NUMBER_OF_DECIMAL_PLACES_TO_ROUND).tolist()]
        else:
            return [signature, main_cancer_type, cutoff, number_of_mutations, biosample, dna_element, avg_real_signal, avg_sim_signal, fold_change, min_sim_signal, max_sim_signal, pvalue, num_of_sims, num_of_sims_with_not_nan_avgs, real_data_avg_count, sim_avg_count, None]


# Step1
# Epigenomics Signatures
# Epigenomics All Mutations (SUBS, INDELS, DINUCS)
# Nucleosome Signatures
# Nucleosome All Mutations (SUBS, INDELS, DINUCS)
# Engine function
def step1_compute_p_value(window_size,
                          combined_output_dir,
                          numberofSimulations,
                          combined_pcawg_nonpcawg_cancer_type_2_biosample_dict,
                          hm_path,
                          ctcf_path,
                          atac_path,
                          cancer_type_signature_cutoff_number_of_mutations_average_probability_df,
                          signatures,
                          cancer_types,
                          heatmaps_main_output_path):

    numofProcesses = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(numofProcesses)

    signature2CancerType2Biosample2DNAElement2PValueDict = {}
    plusorMinus = int(window_size/2)

    def update_dictionary(complete_list):
        signature = complete_list[0]
        cancer_type = complete_list[1]
        biosample = complete_list[4]
        dna_element = complete_list[5]

        if signature in signature2CancerType2Biosample2DNAElement2PValueDict:
            if cancer_type in signature2CancerType2Biosample2DNAElement2PValueDict[signature]:
                if biosample in signature2CancerType2Biosample2DNAElement2PValueDict[signature][cancer_type]:
                    if dna_element in signature2CancerType2Biosample2DNAElement2PValueDict[signature][cancer_type][biosample]:
                        print('There is a problem')
                    else:
                        signature2CancerType2Biosample2DNAElement2PValueDict[signature][cancer_type][biosample][dna_element] = complete_list
                else:
                    signature2CancerType2Biosample2DNAElement2PValueDict[signature][cancer_type][biosample] = {}
                    signature2CancerType2Biosample2DNAElement2PValueDict[signature][cancer_type][biosample][dna_element] = complete_list
            else:
                signature2CancerType2Biosample2DNAElement2PValueDict[signature][cancer_type] = {}
                signature2CancerType2Biosample2DNAElement2PValueDict[signature][cancer_type][biosample] = {}
                signature2CancerType2Biosample2DNAElement2PValueDict[signature][cancer_type][biosample][dna_element] = complete_list
        else:
            signature2CancerType2Biosample2DNAElement2PValueDict[signature] = {}
            signature2CancerType2Biosample2DNAElement2PValueDict[signature][cancer_type] = {}
            signature2CancerType2Biosample2DNAElement2PValueDict[signature][cancer_type][biosample] = {}
            signature2CancerType2Biosample2DNAElement2PValueDict[signature][cancer_type][biosample][dna_element] = complete_list

    for signature in signatures:
        if signature in ALL_MUTATIONS:
            cancer_types_with_this_signature = cancer_types
        else:
            cancer_types_with_this_signature = cancer_type_signature_cutoff_number_of_mutations_average_probability_df[cancer_type_signature_cutoff_number_of_mutations_average_probability_df['signature'] == signature]['cancer_type'].unique()

        for cancer_type in cancer_types_with_this_signature:
            # cancer_type     signature       cutoff  number_of_mutations     average_probability     samples_list    len(samples_list)       len(all_samples_list)   percentage_of_samples
            # get signature cancer_type number_of_mutations
            if signature in ALL_MUTATIONS:
                mutation_type_number_of_mutations_df = pd.read_csv(os.path.join(combined_output_dir, cancer_type, DATA, Table_MutationType_NumberofMutations_NumberofSamples_SamplesList_Filename), sep='\t', header=0)
                mutation_type_number_of_mutations_df = mutation_type_number_of_mutations_df.astype(dtype={'number_of_mutations': int})

                if (signature == ALL_SUBSTITUTIONS) and (mutation_type_number_of_mutations_df[mutation_type_number_of_mutations_df['mutation_type'] == 'All']['number_of_mutations'].values.any()):
                    number_of_mutations = mutation_type_number_of_mutations_df[mutation_type_number_of_mutations_df['mutation_type'] == 'All']['number_of_mutations'].values[0]
                elif (signature == ALL_DINUCLEOTIDES) and (mutation_type_number_of_mutations_df[mutation_type_number_of_mutations_df['mutation_type'] == 'DINUCS']['number_of_mutations'].values.any()):
                    number_of_mutations = mutation_type_number_of_mutations_df[mutation_type_number_of_mutations_df['mutation_type'] == 'DINUCS']['number_of_mutations'].values[0]
                elif (signature == ALL_INDELS) and (mutation_type_number_of_mutations_df[mutation_type_number_of_mutations_df['mutation_type'] == 'INDELS']['number_of_mutations'].values.any()):
                    number_of_mutations = mutation_type_number_of_mutations_df[mutation_type_number_of_mutations_df['mutation_type'] == 'INDELS']['number_of_mutations'].values[0]
                cutoff = np.nan
            elif cancer_type_signature_cutoff_number_of_mutations_average_probability_df[
                ((cancer_type_signature_cutoff_number_of_mutations_average_probability_df['signature'] == signature)  &
                (cancer_type_signature_cutoff_number_of_mutations_average_probability_df['cancer_type'] == cancer_type))]['number_of_mutations'].values.any():

                cutoff = cancer_type_signature_cutoff_number_of_mutations_average_probability_df[
                    ((cancer_type_signature_cutoff_number_of_mutations_average_probability_df['signature'] == signature)  &
                    (cancer_type_signature_cutoff_number_of_mutations_average_probability_df['cancer_type'] == cancer_type))]['cutoff'].values[0]

                number_of_mutations = cancer_type_signature_cutoff_number_of_mutations_average_probability_df[
                    ((cancer_type_signature_cutoff_number_of_mutations_average_probability_df['signature'] == signature)  &
                    (cancer_type_signature_cutoff_number_of_mutations_average_probability_df['cancer_type'] == cancer_type))]['number_of_mutations'].values[0]

            # Epigenomics
            occupancy_type = EPIGENOMICSOCCUPANCY
            biosamples = combined_pcawg_nonpcawg_cancer_type_2_biosample_dict[cancer_type]
            for biosample in biosamples:
                hms = get_encode_elements_using_listdir(biosample, hm_path)
                ctcfs = get_encode_elements_using_listdir(biosample, ctcf_path)
                atacs = get_encode_elements_using_listdir(biosample, atac_path)

                encode_elements = []
                encode_elements.extend(hms)
                encode_elements.extend(ctcfs)
                encode_elements.extend(atacs)

                for dna_element in encode_elements:
                    # (signature, cancer_type, biosample, dna_element, avg_real_signal, avg_sim_signal, fold_change, min_sim_signal, max_sim_signal, pvalue, num_of_sims_with_not_nan_avgs, list(simulationsHorizontalMeans))
                    pool.apply_async(calculate_fold_change_real_over_sim,
                                     args=(plusorMinus, combined_output_dir, numberofSimulations, signature, cancer_type, cutoff, number_of_mutations, biosample, dna_element, occupancy_type,),
                                     callback=update_dictionary)

            # Nucleosome
            occupancy_type = NUCLEOSOMEOCCUPANCY
            biosample = NUCLEOSOME_BIOSAMPLE
            dna_element = NUCLEOSOME
            # (signature, cancer_type, biosample, dna_element, avg_real_signal, avg_sim_signal, fold_change, min_sim_signal, max_sim_signal, pvalue, num_of_sims_with_not_nan_avgs, list(simulationsHorizontalMeans))
            pool.apply_async(calculate_fold_change_real_over_sim,
                     args=(plusorMinus, combined_output_dir, numberofSimulations, signature, cancer_type, cutoff, number_of_mutations, biosample, dna_element, occupancy_type,),
                     callback=update_dictionary)

            # # Sequential  for debugging
            # update_dictionary(calculate_fold_change_real_over_sim(plusorMinus, combined_output_dir, numberofSimulations, signature, cancer_type, biosample,dna_element, occupancy_type))

    pool.close()
    pool.join()

    print('Step1 Getting p-values')
    # Complete list has
    # (signature, cancer_type, biosample, dna_element, avg_real_signal, avg_sim_signal, fold_change, min_sim_signal, max_sim_signal, pvalue, num_of_sims, num_of_sims_with_not_nan_avgs, list(simulationsHorizontalMeans))
    # It is used to speed up and bypass step1
    if (signature2CancerType2Biosample2DNAElement2PValueDict):
        dictFilename= 'Step1_Signature2CancerType2Biosample2DNAElement2PValue_Dict.txt'
        dictPath=os.path.join(heatmaps_main_output_path,DICTIONARIES,dictFilename)
        writeDictionary(dictPath, signature2CancerType2Biosample2DNAElement2PValueDict, NpEncoder)

    # Write dictionary as a dataframe
    df_filename = 'Step1_Signature_CancerType_Biosample_DNAElement_PValue.txt'
    filepath = os.path.join(heatmaps_main_output_path,TABLES, df_filename)
    step1_p_value_df = write_dictionary_as_dataframe_step1_p_value(signature2CancerType2Biosample2DNAElement2PValueDict,filepath)

    return step1_p_value_df, signature2CancerType2Biosample2DNAElement2PValueDict


def writeDictionary(dictPath, dictionary, customJSONEncoder):
    with open(dictPath, 'w') as file:
        file.write(json.dumps(dictionary, cls=customJSONEncoder))

def readDictionary(filePath):
    if (os.path.exists(filePath) and (os.path.getsize(filePath) > 0)):
        with open(filePath,'r') as json_data:
            dictionary = json.load(json_data)
        return dictionary
    else:
        # return None
        # Provide empty dictionary for not to fail for loops on None type dictionary
        return {}










