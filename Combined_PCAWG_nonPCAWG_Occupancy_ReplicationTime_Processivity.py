# !/usr/bin/env python3

# Author: burcakotlu

# Contact: burcakotlu@eng.ucsd.edu

import os
import sys
import shutil
import math
import numpy as np
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as matplotlib_cm
import matplotlib as mpl
from matplotlib.offsetbox import AnchoredText
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch, cm
from reportlab.lib import utils
from reportlab.pdfbase.pdfmetrics import stringWidth

import statsmodels.stats.multitest
import scipy
import multiprocessing

from scipy.stats import sem
from scipy.stats import spearmanr
from scipy.stats import pearsonr

from Combined_Common import cancer_type_2_NCI_Thesaurus_code_dict
from Combined_Common import signatures_attributed_to_artifacts
from Combined_Common import COSMIC_NUCLEOSOME_OCCUPANCY
from Combined_Common import COSMIC_CTCF_OCCUPANCY
from Combined_Common import COSMIC_OCCUPANCY
from Combined_Common import COSMIC_REPLICATION_TIME
from Combined_Common import COSMIC_PROCESSIVITY
from Combined_Common import deleteOldData
from Combined_Common import natural_key
from Combined_Common import fill_lists
from Combined_Common import get_signature2cancer_type_list_dict
from Combined_Common import depleted
from Combined_Common import enriched
from Combined_Common import OCCUPANCY_HEATMAP_COMMON_MULTIPLIER
from Combined_Common import NUMBER_OF_DECIMAL_PLACES_TO_ROUND

from Combined_Common import TABLE_SBS_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE
from Combined_Common import TABLE_DBS_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE
from Combined_Common import TABLE_ID_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE
from Combined_Common import TABLE_MUTATIONTYPE_NUMBEROFMUTATIONS_NUMBEROFSAMPLES_SAMPLESLIST
from Combined_Common import get_number_of_mutations_filename
from Combined_Common import get_alternative_combined_output_dir_and_cancer_type
from Combined_Common import ALTERNATIVE_OUTPUT_DIR

from Combined_Common import LYMPH_BNHL_CLUSTERED
from Combined_Common import LYMPH_BNHL_NONCLUSTERED
from Combined_Common import LYMPH_BNHL
from Combined_Common import LYMPH_CLL_CLUSTERED
from Combined_Common import LYMPH_CLL_NONCLUSTERED
from Combined_Common import LYMPH_CLL

from SigProfilerTopography.source.commons.TopographyCommons import calculate_pvalue_teststatistics

from Combined_Common import SBS
from Combined_Common import DBS
from Combined_Common import ID
from Combined_Common import SUBS
from Combined_Common import DINUCS
from Combined_Common import INDELS
from Combined_Common import AGGREGATEDDINUCS
from Combined_Common import AGGREGATEDSUBSTITUTIONS
from Combined_Common import AGGREGATEDINDELS

SIGNATUREBASED = 'signaturebased'

DATA = 'data'
FIGURE = 'figure'

NUCLEOSOME_OCCUPANCY = 'nucleosome_occupancy'
EPIGENOMICS_OCCUPANCY = 'epigenomics_occupancy'
REPLICATION_TIME = 'replication_time'
PROCESSIVITY = 'processivity'

NUCLEOSOME = "nucleosome"
CTCF = "CTCF"
ATAC_SEQ = "ATAC-seq"
H3K9ac = "H3K9ac"
H3K9me3 = "H3K9me3"
H3K27ac = "H3K27ac"
H3K27me3 = "H3K27me3"
H3K36me3 = "H3K36me3"
H3K79me2 = "H3K79me2"
H4K20me1 = "H4K20me1"
H3K4me1 = "H3K4me1"
H3K4me2 = "H3K4me2"
H3K4me3 = "H3K4me3"
H2AFZ = "H2AFZ"

OCCUPANCY = "occupancy"

AVERAGE_SIGNAL_ARRAY_TXT = "AverageSignalArray.txt"
ACCUMULATED_SIGNAL_ARRAY_TXT = "AccumulatedSignalArray.txt"
ACCUMULATED_COUNT_ARRAY_TXT = "AccumulatedCountArray.txt"

REGRESSION = 'regression'

# COMBINED OCCUPANCY
# Pearson/spearman correlation calculation for mutation start +/-500 which is 1K long

# COMBINED REPLICATION TIME
INCREASING = "INC"
FLAT = "FLAT"
DECREASING = "DEC"
UNKNOWN = "UNKNOWN"

# We consider signature and cancer type that has at least 1K mutations
# This constraint applies for occupancy and replication time analysis
AT_LEAST_1K_CONSRAINTS = 1000
AT_LEAST_20K_CONSRAINTS = 20000

CANCER_TYPE_BASED_OCCUPANCY_FIGURE = "occupancy"
ACROSS_ALL_CANCER_TYPES_OCCUPANCY_FIGURE = "across_all_cancer_types_occupancy"

PDFS = 'pdfs'
OCCUPANCY_PDF = 'across_all_tissues_and_each_tissue.pdf'
ACROSS_ALL_CANCER_TYPES = 'ACROSS_ALL_CANCER_TYPES'

ONE_POINT_EIGHT = 1.08

TSCC = 'tscc'
AWS = 'aws'

# FIGURE_SBS4_ACROSS_ALL_TISSUES = 'Figure_SBS4_Across_All_Tissues'
FIGURE_CASE_STUDY = 'Figure_Case_Study'
COSMIC = 'Cosmic'
COSMIC_TISSUE_BASED = 'Cosmic_Tissue_Based'
MANUSCRIPT = 'Manuscript'

EXCEL_FILES = 'excel_files'
TABLES = 'tables'
PDF_FILES = 'pdf_files'
DATA_FILES = 'data_files'

MANUSCRIPT_TISSUE_BASED_FIGURES = 'manuscript_tissue_based_figures'
COSMIC_TISSUE_BASED_FIGURES = 'cosmic_tissue_based_figures'
COSMIC_ACROSS_ALL_AND_TISSUE_BASED_TOGETHER = "cosmic_across_all_and_tissue_based_together"

FIGURES_COSMIC = 'figures_cosmic'
FIGURES_MANUSCRIPT = 'figures_manuscript'

FIGURE_CASE_STUDY_SBS4_ACROSS_ALL_TISSUES = 'FIGURE_CASE_STUDY_SBS4_ACROSS_ALL_TISSUES'
FIGURE_CASE_STUDY_SBS28 = 'FIGURE_CASE_STUDY_SBS28'
FIGURE_CASE_STUDY_SHARED_ETIOLOGY = 'FIGURE_CASE_STUDY_SHARED_ETIOLOGY'

STRAND_COORDINATED_MUTAGENESIS_GROUP_LENGTH = 'Strand-coordinated Mutagenesis Group Length'

def readAsFloatNumpyArray(filePath,plusorMinus):
    if os.path.exists(filePath):
        nparray = np.loadtxt(filePath, dtype=float, delimiter='\t')
        nparray = nparray[0:(plusorMinus*2)+1]
        return nparray
    else:
        return None

def readAsIntNumpyArray(filePath,plusorMinus):
    if os.path.exists(filePath):
        nparray = np.loadtxt(filePath, dtype=float, delimiter='\t')
        #nparray = np.loadtxt(filePath, dtype=int, delimiter='\t') causes ValueError: invalid literal for int() with base 10: '0.0'
        nparray = nparray[0:(plusorMinus*2)+1]
        nparray = nparray.astype(int)
        return nparray
    else:
        return None

# List contains 2D array for each DNA element file (called from cancer type based)
# List contains 2D array for each cancer type and DNA element file (called from across all cancer types)
# e.g.: If there are 3 CTCF files for a certain cancer type, there will be 3 2D arrays for that cancer type in the list
def get_simulations_means_mins_maxs(simulations_average_signal_array_list):
    rows = simulations_average_signal_array_list[0].shape[0]
    columns = simulations_average_signal_array_list[0].shape[1]

    accumulated_simulations_average_signal_array = np.zeros((rows, columns))

    # For each dna element file accumulate
    for simulations_average_signal_array in simulations_average_signal_array_list:
        # Accumulate signal arrays
        accumulated_simulations_average_signal_array += simulations_average_signal_array

    # Take average
    number_of_files = len(simulations_average_signal_array_list)
    accumulated_simulations_average_signal_array = accumulated_simulations_average_signal_array/number_of_files

    # Take column wise mean
    simulations_lows_list, \
    sims_avg_signal_array, \
    simulations_highs_list = calculate_sims_lows_means_highs(accumulated_simulations_average_signal_array)

    return sims_avg_signal_array, simulations_lows_list, simulations_highs_list

# simulations_array_list can be list of numpy arrays with 100 lists, each list contain (2001,) array
# simulations_array_list can be a numpy array with 100 rows and 2001 columns
def takeAverage(simulations_array_list):
    sims_avg_signal_array = None
    simulations_lows_list = None
    simulations_highs_list = None

    #Number of simulations >= 1
    if ((simulations_array_list is not None) and len(simulations_array_list)>=1):
        stackedSimulationAggregatedMutations = np.vstack(simulations_array_list)

        # Take column wise mean
        simulations_lows_list, \
        sims_avg_signal_array, \
        simulations_highs_list = calculate_sims_lows_means_highs(stackedSimulationAggregatedMutations)

    return  sims_avg_signal_array, simulations_lows_list, simulations_highs_list


# Plot legend only
def plot_replication_time_legend(legend_output_dir):
    fig, ax = plt.subplots(figsize=(6, 7))

    # Number of Cancer Types Replication Time
    title_increasing_flat_decreasing = 'Number of Cancer Types \n' \
                                       '\u2197 Increasing\n' \
                                       '\u2192 Flat\n' \
                                       '\u2198 Decreasing'

    anchored_text_increasing_flat_decreasing = AnchoredText(title_increasing_flat_decreasing,
                                                            frameon=False, borderpad=0, pad=0.1,
                                                            loc='upper center',
                                                            bbox_transform=plt.gca().transAxes,
                                                            prop={'fontsize': 30, 'fontweight': 'semibold'})  # bbox_to_anchor=[0, 0.5],

    ax.add_artist(anchored_text_increasing_flat_decreasing)
    plt.gca().set_axis_off()

    real_subs_rectangle = mpatches.Patch(label='Real Subs', edgecolor='black', facecolor='royalblue', lw=3)
    real_dinucs_rectangle = mpatches.Patch(label='Real Dinucs', edgecolor='black', facecolor='crimson', lw=3)
    real_indels_rectangle = mpatches.Patch(label='Real Indels', edgecolor='black', facecolor='yellowgreen', lw=3)

    legend_elements = [
        real_subs_rectangle,
        real_dinucs_rectangle,
        real_indels_rectangle,
        Line2D([0], [2], linestyle="--", marker='.', lw=5, color='black', label='Simulations', markerfacecolor='black', markersize=30)]

    plt.legend(handles=legend_elements, handlelength=5, ncol=1, loc="lower center", fontsize=30)  # bbox_to_anchor=(1, 0.5),
    plt.gca().set_axis_off()

    filename = 'Replication_Time_Legend.png'
    filepath = os.path.join(legend_output_dir, filename)
    fig.savefig(filepath, dpi=100, bbox_inches="tight")

    plt.cla()
    plt.close(fig)


# Plot Legend only
def plot_occupancy_legend(plot_output_dir, dna_element):
    fig, ax = plt.subplots(figsize=(10, 1))

    legend_elements = [
        Line2D([0], [2], linestyle="-", lw=5, color='royalblue', label='Real Subs Average Signal',
               markerfacecolor='royalblue', markersize=30),
        Line2D([0], [2], linestyle="-", lw=5, color='crimson', label='Real Dinucs Average Signal',
               markerfacecolor='crimson', markersize=30),
        Line2D([0], [2], linestyle="-", lw=5, color='darkgreen', label='Real Indels Average Signal',
               markerfacecolor='darkgreen', markersize=30),
        Line2D([0], [2], linestyle="--", lw=5, color='gray', label='Simulations Average Signal', markerfacecolor='gray',
               markersize=30)]

    plt.legend(handles=legend_elements, handlelength=5, ncol=1, loc="center", bbox_to_anchor=(0.5, 0.5), fontsize=30)
    plt.gca().set_axis_off()

    filename = 'Occupancy_Legend.png'
    filepath = os.path.join(plot_output_dir, OCCUPANCY, dna_element, FIGURES_MANUSCRIPT, filename)
    fig.savefig(filepath, dpi=100, bbox_inches="tight")

    plt.cla()
    plt.close(fig)


# COSMIC across all tissues figure
# MANUSCRIPT across all tissues figure
# across_all_tissues_simulations_signal_list is added
# fillcolor is added for simulations
def plot_occupancy_figure_across_all_cancer_types(plot_output_dir,
                                                occupancy_type,
                                                dna_element,
                                                signature,
                                                across_all_cancer_types_real_average_signal_array,
                                                across_all_cancer_types_simulations_average_signal_array_list,
                                                color,
                                                fillcolor,
                                                num_of_cancer_types_with_pearson_q_values_le_significance_level,
                                                num_of_cancer_types_with_considered_files, # extra information may confuse the user
                                                plus_minus,
                                                figure_type,
                                                cosmic_release_version,
                                                figure_file_extension,
                                                cosmic_legend = True,
                                                cosmic_correlation_text = True,
                                                cosmic_fontsize_text = 20,
                                                cosmic_fontsize_ticks = 20,
                                                cosmic_fontsize_labels = 20,
                                                cosmic_linewidth_plot = 5,
                                                cosmic_title_all_cancer_types = False,
                                                figure_case_study = None):

    min_list=[]
    max_list=[]

    if figure_type == MANUSCRIPT:
        fontsize_text = 65
        fontsize_ticks = 45
        linewidth_plot = 15
        fontsize_labels = 40
        figure_dir = FIGURES_MANUSCRIPT

    elif figure_type == COSMIC:
        fontsize_text = cosmic_fontsize_text
        fontsize_ticks = cosmic_fontsize_ticks
        fontsize_labels = cosmic_fontsize_labels
        linewidth_plot = cosmic_linewidth_plot
        figure_dir = FIGURES_COSMIC

    fwidth = 15
    fheight = 7

    fig = plt.figure(figsize=(fwidth, fheight), facecolor=None)
    plt.style.use('ggplot')

    # define margins -> size in inches / figure dimension
    left_margin = 0.95 / fwidth
    right_margin = 0.2 / fwidth
    bottom_margin = 0.5 / fheight
    top_margin = 0.25 / fheight

    # create axes
    # dimensions are calculated relative to the figure size
    x = left_margin  # horiz. position of bottom-left corner
    y = bottom_margin  # vert. position of bottom-left corner
    w = 1 - (left_margin + right_margin)  # width of axes
    h = 1 - (bottom_margin + top_margin)  # height of axes
    ax = fig.add_axes([x, y, w, h])

    # No need for these figures and we need to set labelpad
    # Define the Ylabel position
    # Location are defined in dimension relative to the figure size
    # xloc = 0.25 / fwidth
    # yloc = y + h / 2.
    # ax.yaxis.set_label_coords(xloc, yloc, transform=fig.transFigure)

    # This code makes the background white.
    ax.set_facecolor('white')

    # This code puts the edge line
    for edge_i in ['left', 'bottom', 'right', 'top']:
        ax.spines[edge_i].set_edgecolor("black")
        ax.spines[edge_i].set_linewidth(3)

    x = np.arange(-plus_minus, plus_minus+1, 1)

    listofLegends = []

    if (across_all_cancer_types_real_average_signal_array is not None):
        aggSubs = plt.plot(x, across_all_cancer_types_real_average_signal_array, color=color, label='Across All Cancer Types Average Signal', linewidth=linewidth_plot, zorder=10)
        listofLegends.append(aggSubs[0])
        min_list.append(np.nanmin(across_all_cancer_types_real_average_signal_array))
        max_list.append(np.nanmax(across_all_cancer_types_real_average_signal_array))

    sims_avg_signal_array, \
    simulations_lows_list, \
    simulations_highs_list = get_simulations_means_mins_maxs(across_all_cancer_types_simulations_average_signal_array_list)

    if (simulations_lows_list is not None and simulations_lows_list):
        min_list.append(np.nanmin(simulations_lows_list))
    if (simulations_highs_list is not None and simulations_highs_list):
        max_list.append(np.nanmax(simulations_highs_list))

    if (sims_avg_signal_array is not None):
        simulations = plt.plot(x, sims_avg_signal_array, color='gray', linestyle='--',  label='Across All Cancer Types Simulations Average Signal', linewidth=linewidth_plot) # new way
        listofLegends.append(simulations[0])

    if (simulations_lows_list is not None) and (simulations_highs_list is not None):
        plt.fill_between(x, np.array(simulations_lows_list), np.array(simulations_highs_list),facecolor=fillcolor)

    if (figure_type == COSMIC) and cosmic_legend:
        plt.legend(loc= 'lower left',handles = listofLegends, prop={'size': fontsize_text}, shadow=False, edgecolor='white', facecolor ='white')

    if (signature == AGGREGATEDSUBSTITUTIONS):
        plt.title('All Substitutions', fontsize=fontsize_text)
        # plt.title('All Substitutions:\n%d Cancer Types' %(num_of_cancer_types_with_considered_files), fontsize=fontsize_text)
        # plt.text(0.01, 0.96, 'Substitutions', verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=fontsize_text, zorder=15)
    elif (signature == AGGREGATEDDINUCS):
        plt.title('All Doublets', fontsize=fontsize_text)
        # plt.title('All Doublets:\n%d  Cancer Types' %(num_of_cancer_types_with_considered_files), fontsize=fontsize_text)
        # plt.text(0.01, 0.96, 'Dinucs', verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=fontsize_text,zorder=15)
    elif (signature == AGGREGATEDINDELS):
        plt.title('All Indels', fontsize=fontsize_text)
        # plt.title('All Indels:\n%d  Cancer Types' %(num_of_cancer_types_with_considered_files), fontsize=fontsize_text)
        # plt.text(0.01, 0.96, 'Indels', verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=fontsize_text, zorder=15)
    else:
        plt.text(0.01, 0.96, signature, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=fontsize_text, zorder=15)

    if figure_type == COSMIC:
        if num_of_cancer_types_with_considered_files == 1:
            number_of_cancer_types_text = '%d Cancer Type Considered' %(num_of_cancer_types_with_considered_files)
            text = str(num_of_cancer_types_with_pearson_q_values_le_significance_level) + '/' + str(num_of_cancer_types_with_considered_files) + ' Cancer Type With Similar Occupancy'
        elif num_of_cancer_types_with_considered_files > 1:
            number_of_cancer_types_text = '%d Cancer Types Considered' %(num_of_cancer_types_with_considered_files)
            text = str(num_of_cancer_types_with_pearson_q_values_le_significance_level) + '/' + str(num_of_cancer_types_with_considered_files) + ' Cancer Types With Similar Occupancy'

        if cosmic_title_all_cancer_types:
            if num_of_cancer_types_with_considered_files > 1:
                plt.title("All %d Cancer Types" %(num_of_cancer_types_with_considered_files), fontsize=cosmic_fontsize_text)
            else:
                plt.title("All %d Cancer Type" %(num_of_cancer_types_with_considered_files), fontsize=cosmic_fontsize_text)
        elif figure_case_study:
            # Title can be changed here
            plt.title(figure_case_study, fontsize = cosmic_fontsize_text)
        else:
            plt.text(0.99, 0.96, number_of_cancer_types_text, verticalalignment='top', horizontalalignment='right', transform=ax.transAxes,fontsize=fontsize_text, zorder=15)
        if cosmic_correlation_text:
            plt.text(0.99, 0.90, text, verticalalignment='top', horizontalalignment='right', transform=ax.transAxes,fontsize=fontsize_text, zorder=15)
    elif (figure_type == MANUSCRIPT) and (dna_element != CTCF):
        text = str(num_of_cancer_types_with_pearson_q_values_le_significance_level) + '/' +str(num_of_cancer_types_with_considered_files)
        plt.text(0.99, 0.96, text, verticalalignment='top', horizontalalignment='right', transform=ax.transAxes,fontsize=fontsize_text, zorder=15)

    # Put vertical line at x=0
    plt.axvline(x=0, color='gray', linestyle='--')

    # This code puts the tick marks
    plt.tick_params(axis='both', which='major', labelsize=fontsize_labels, width=3, length=10)
    plt.tick_params(axis='both', which='minor', labelsize=fontsize_labels, width=3, length=10)

    min_average_nucleosome_signal = np.nanmin(min_list)
    max_average_nucleosome_signal = np.nanmax(max_list)

    if occupancy_type == NUCLEOSOME_OCCUPANCY:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ymin = round(min_average_nucleosome_signal - 0.1, 2)
        ymax = round(max_average_nucleosome_signal + 0.1, 2)

        # To show less y axis tick labels
        plt.yticks(np.arange(ymin, ymax, step=0.2), fontsize=fontsize_ticks)
        plt.ylim((ymin - 0.01, ymax + 0.01))

    elif occupancy_type == EPIGENOMICS_OCCUPANCY and dna_element == CTCF:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%4d'))
        # nbins: Maximum number of intervals; one less than max number of ticks.
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=3, nbins=3)) # legacy nbins=2
        ax.yaxis.set_major_locator(MaxNLocator(3))
        # plt.locator_params(axis='y', nbins=3)
        if (np.nanmax(across_all_cancer_types_real_average_signal_array) < 100) and (np.nanmax(sims_avg_signal_array) < 100):
            ax.tick_params(axis='y', which='major', pad=15)

    elif occupancy_type == EPIGENOMICS_OCCUPANCY and dna_element == H3K27ac and figure_case_study and figure_case_study == FIGURE_CASE_STUDY_SHARED_ETIOLOGY:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # nbins: Maximum number of intervals; one less than max number of ticks.
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=3, nbins=3)) # legacy nbins=2
        ax.yaxis.set_major_locator(MaxNLocator(3))

    elif occupancy_type == EPIGENOMICS_OCCUPANCY:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # This code provides the x and y tick marks and labels
    if dna_element == NUCLEOSOME:
        ylabel_text = 'Average\n%s Signal' %dna_element.capitalize()
    else:
        ylabel_text = 'Average\n%s Signal' %dna_element
        # ylabel_text = 'Average %s Signal' %dna_element

    if figure_type == COSMIC and figure_case_study:
        tick_locations = np.arange(-plus_minus/2, plus_minus/2 + 1, step = plus_minus / 2)
        tick_labels = ['-500 bp\n\n', '0\n\nSomatic Mutations Location', '+500 bp\n\n']
        plt.xticks(tick_locations, tick_labels, fontsize=fontsize_ticks)
        plt.xlim((-plus_minus - 50, plus_minus + 50))
        ax.set_ylabel(ylabel_text, fontsize=fontsize_labels, labelpad=15)
    elif figure_type == COSMIC:
        tick_locations = np.arange(-plus_minus, plus_minus + 1, step = plus_minus / 2)
        tick_labels = ["-1000 bp\n\n5’", '-500 bp\n\n', '0\n\nSomatic Mutations Location', '+500 bp\n\n', "+1000 bp\n\n3’"]
        plt.xticks(tick_locations, tick_labels, fontsize=fontsize_ticks)
        plt.xlim((-plus_minus - 50, plus_minus + 50))
        ax.set_ylabel(ylabel_text, fontsize=fontsize_labels, labelpad=15)
    if figure_type == MANUSCRIPT and (signature == AGGREGATEDSUBSTITUTIONS or signature == AGGREGATEDDINUCS or signature == AGGREGATEDINDELS):
        # tick_locations = np.arange(-plus_minus/2, plus_minus/2 + 1, step = plus_minus / 2)
        # tick_labels = ['-500 bp\n\n', '0\n\nSomatic Mutations Location', '+500 bp\n\n']
        # plt.xticks(tick_locations, tick_labels, fontsize=fontsize_ticks)
        # plt.xlim((-plus_minus - 50, plus_minus + 50))
        plt.xticks(np.arange(-plus_minus / 2, plus_minus / 2 + 1, step=plus_minus / 2), fontsize=fontsize_ticks)
        plt.xlim((-plus_minus, plus_minus)) # legacy
        # ax.set_xlabel('Interval Around Variant (bp)', fontsize=fontsize_text-5, labelpad=15)
        ax.set_xlabel('Somatic Mutations Location', fontsize=fontsize_text-5, labelpad=15)
        ax.set_ylabel(ylabel_text, fontsize=fontsize_text-5, labelpad=15)
    elif figure_type == MANUSCRIPT:
        plt.xticks(np.arange(-plus_minus / 2, plus_minus / 2 + 1, step=plus_minus / 2), fontsize=fontsize_ticks)
        plt.xlim((-plus_minus, plus_minus)) # legacy

    # v3.2_SBS1_REPLIC_ASYM.jpg
    if dna_element == CTCF:
        feature_name = COSMIC_CTCF_OCCUPANCY
    elif dna_element == NUCLEOSOME:
        feature_name=  COSMIC_NUCLEOSOME_OCCUPANCY
    else:
        feature_name = dna_element + '_' + COSMIC_OCCUPANCY

    if figure_type == COSMIC:
        filename = '%s_%s_%s.%s' %(cosmic_release_version, signature, feature_name, figure_file_extension)
    elif figure_type == MANUSCRIPT:
        filename='%s_%s_%s.png' %(signature, dna_element, ACROSS_ALL_CANCER_TYPES_OCCUPANCY_FIGURE)

    figure_file = os.path.join(plot_output_dir, OCCUPANCY, dna_element, figure_dir, filename)

    fig.savefig(figure_file, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return sims_avg_signal_array

# Plot Cosmic Tissue Based Occupancy Figures
def plot_occupancy_figure_cosmic_tissue_based(plot_output_dir,
                                            occupancy_type,
                                            dna_element,
                                            signature,
                                            cancer_type,
                                            across_all_files_average_signal_array,
                                            across_all_files_simulations_average_signal_array_list,
                                            label,
                                            color,
                                            fillcolor,
                                            pearson_correlation,
                                            pearson_p_value,
                                            plus_or_minus,
                                            cosmic_release_version,
                                            figure_file_extension,
                                            cosmic_legend = True,
                                            cosmic_correlation_text = True,
                                            cosmic_labels= True,
                                            cancer_type_on_right_hand_side = True,
                                            cosmic_fontsize_text = 20,
                                            cosmic_fontsize_ticks = 20,
                                            cosmic_fontsize_labels = 20,
                                            cosmic_linewidth_plot = 5,
                                            figure_case_study = None):

    min_list=[]
    max_list=[]

    # fig = plt.figure(figsize=(15, 7), facecolor=None) # legacy
    # ax = plt.gca() # legacy

    fwidth = 15
    fheight = 7

    fig = plt.figure(figsize=(fwidth, fheight), facecolor=None)
    plt.style.use('ggplot')

    # define margins -> size in inches / figure dimension
    left_margin = 0.95 / fwidth
    right_margin = 0.2 / fwidth
    bottom_margin = 0.5 / fheight
    top_margin = 0.25 / fheight

    # create axes
    # dimensions are calculated relative to the figure size
    x = left_margin  # horiz. position of bottom-left corner
    y = bottom_margin  # vert. position of bottom-left corner
    w = 1 - (left_margin + right_margin)  # width of axes
    h = 1 - (bottom_margin + top_margin)  # height of axes
    ax = fig.add_axes([x, y, w, h])

    # This code makes the background white.
    ax.set_facecolor('white')

    # This code puts the edge line
    for edge_i in ['left', 'bottom','right', 'top']:
        ax.spines[edge_i].set_edgecolor("black")
        ax.spines[edge_i].set_linewidth(3)

    x = np.arange(-plus_or_minus, plus_or_minus+1, 1)

    listofLegends = []

    if (across_all_files_average_signal_array is not None):
        real_label = 'Real %s' % (label)
        aggSubs = plt.plot(x, across_all_files_average_signal_array, color=color, label=real_label, linewidth=cosmic_linewidth_plot, zorder=10)
        listofLegends.append(aggSubs[0])
        min_list.append(np.nanmin(across_all_files_average_signal_array))
        max_list.append(np.nanmax(across_all_files_average_signal_array))

    simulations_mean_average_signal_array,  \
    simulations_lows_list, \
    simulations_highs_list = get_simulations_means_mins_maxs(across_all_files_simulations_average_signal_array_list)

    if (simulations_lows_list is not None and simulations_lows_list):
        min_list.append(np.nanmin(simulations_lows_list))
    if (simulations_highs_list is not None and simulations_highs_list):
        max_list.append(np.nanmax(simulations_highs_list))

    # Plot simulations
    if (simulations_mean_average_signal_array is not None):
        sims_label = 'Simulated %s' % (label)
        simulations = plt.plot(x, simulations_mean_average_signal_array, color='gray', linestyle='--',  label=sims_label, linewidth=cosmic_linewidth_plot) # new way
        listofLegends.append(simulations[0])

    # Plot simulations lows and highs
    if (simulations_lows_list is not None) and (simulations_highs_list is not None):
        plt.fill_between(x, np.array(simulations_lows_list), np.array(simulations_highs_list), facecolor=fillcolor) # new way 2nd way

    if cosmic_legend:
        plt.legend(loc = 'lower left', handles = listofLegends, prop={'size': cosmic_fontsize_text}, shadow=False, edgecolor='white', facecolor ='white')

    # if cosmic_correlation_text:
    #     text= "Pearson\'s r =%.2f & p-value = %.2e" %(pearson_correlation,pearson_p_value)
    #     plt.text(0.99, 0.90, text, verticalalignment='top', horizontalalignment='right', transform=ax.transAxes, fontsize=cosmic_fontsize_text, zorder=15)

    if dna_element == NUCLEOSOME:
        ylabel_text = 'Average\n%s Signal' %dna_element.capitalize()
        # ylabel_text = 'Average %s Signal' %dna_element.capitalize()
    else:
        ylabel_text = 'Average\n%s Signal' %dna_element
        # ylabel_text = 'Average %s Signal' %dna_element

    if cosmic_labels:
        plt.ylabel(ylabel_text, fontsize=cosmic_fontsize_labels, labelpad=15)

    if (signature == AGGREGATEDSUBSTITUTIONS):
        if figure_case_study != 'B-cell malignancies':
            plt.text(0.01, 0.96, 'Substitutions', verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=cosmic_fontsize_text, zorder=15)
    elif (signature == AGGREGATEDDINUCS):
        plt.text(0.01, 0.96, 'Dinucs', verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=cosmic_fontsize_text,zorder=15)
    elif (signature == AGGREGATEDINDELS):
        plt.text(0.01, 0.96, 'Indels', verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=cosmic_fontsize_text, zorder=15)
    else:
        plt.text(0.01, 0.96, signature, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=cosmic_fontsize_text, zorder=15)

    if cancer_type_on_right_hand_side:
        plt.text(0.99, 0.96, cancer_type, verticalalignment='top', horizontalalignment='right', transform=ax.transAxes,fontsize=cosmic_fontsize_text, zorder=15)
    else:
        # For Figure Case Study
        if figure_case_study == 'B-cell malignancies' and cancer_type == 'B_cell_malignancy_kataegis':
            plt.title('Kataegis Mutations', fontsize=cosmic_fontsize_text)
        elif figure_case_study == 'B-cell malignancies' and cancer_type == 'B_cell_malignancy_omikli':
            plt.title('Omikli Mutations', fontsize=cosmic_fontsize_text)
        elif figure_case_study == 'B-cell malignancies' and cancer_type == 'B_cell_malignancy_nonClustered':
            plt.title('Non-clustered Mutations', fontsize=cosmic_fontsize_text)
        else:
            plt.title(cancer_type, fontsize=cosmic_fontsize_text)

    # Put vertical line at x=0
    plt.axvline(x=0, color='gray', linestyle='--')

    # This code puts the tick marks
    plt.tick_params(axis='both', which='major', labelsize=cosmic_fontsize_labels, width=3, length=10)
    plt.tick_params(axis='both', which='minor', labelsize=cosmic_fontsize_labels, width=3, length=10)

    if figure_case_study:
        # This code provides the x and y tick marks and labels
        tick_locations = np.arange(-plus_or_minus/2, plus_or_minus/2+1, step=plus_or_minus/2)
        tick_labels = ['-500 bp\n\n', '0\n\nSomatic Mutations Location', '+500 bp\n\n']
        plt.xticks(tick_locations, tick_labels, fontsize=cosmic_fontsize_ticks)
        plt.xlim((-plus_or_minus-50, plus_or_minus+50))
    else:
        # This code provides the x and y tick marks and labels
        tick_locations = np.arange(-plus_or_minus, plus_or_minus+1, step=plus_or_minus/2)
        tick_labels = ["-1000 bp\n\n5’", '-500 bp\n\n', '0\n\nSomatic Mutations Location', '+500 bp\n\n', "+1000 bp\n\n3’"]
        plt.xticks(tick_locations, tick_labels, fontsize=cosmic_fontsize_ticks)
        plt.xlim((-plus_or_minus-50, plus_or_minus+50))

    # Do not show x axis ticks and ticklabels
    # Aggregate figures and  bottom line figures with x axis labels --> comment command below
    # Rest of the figures without x axis labels --> uncomment command below
    # ax.axes.xaxis.set_visible(False)

    min_average_nucleosome_signal = np.nanmin(min_list)
    max_average_nucleosome_signal = np.nanmax(max_list)

    if occupancy_type == NUCLEOSOME_OCCUPANCY:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        if figure_case_study == 'B-cell malignancies':
            # To make ymin and ymax same for kataegis, omikli and non-clustered mutations
            ymin = 0.80 # 0.75
            ymax = 1.00
            plt.ylim((ymin - 0.02, ymax + 0.02))
            plt.yticks(np.arange(ymin, ymax, step=0.1), fontsize=cosmic_fontsize_ticks)
        else:
            ymin = round(min_average_nucleosome_signal - 0.1, 2)
            ymax = round(max_average_nucleosome_signal + 0.1, 2)
            # To show less y axis tick labels
            plt.yticks(np.arange(ymin, ymax, step=0.2), fontsize=cosmic_fontsize_ticks)
            plt.ylim((ymin - 0.01, ymax + 0.01))

    elif occupancy_type == EPIGENOMICS_OCCUPANCY and dna_element == CTCF:
        if figure_case_study == 'B-cell malignancies':
            # To make ymin and ymax same for kataegis, omikli and non-clustered mutations
            ymin = 100 # 0.75
            ymax = 180
            plt.ylim((ymin - 20, ymax + 20))
            plt.yticks(np.arange(ymin, ymax, step=40), fontsize=cosmic_fontsize_ticks)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%4d'))
        # nbins: Maximum number of intervals; one less than max number of ticks.
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=3, nbins=2))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        # plt.locator_params(axis='y', nbins=3)
        if (np.nanmax(across_all_files_average_signal_array) < 100) and (np.nanmax(simulations_mean_average_signal_array) < 100):
            ax.tick_params(axis='y', which='major', pad=15)

    elif occupancy_type == EPIGENOMICS_OCCUPANCY and dna_element == H3K27ac and figure_case_study and figure_case_study == FIGURE_CASE_STUDY_SHARED_ETIOLOGY:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # nbins: Maximum number of intervals; one less than max number of ticks.
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=3, nbins=3)) # legacy nbins=2
        ax.yaxis.set_major_locator(MaxNLocator(3))

    elif occupancy_type == EPIGENOMICS_OCCUPANCY:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # v3.2_SBS1_REPLIC_ASYM_TA_C4817.jpg
    if dna_element == CTCF:
        feature_name = COSMIC_CTCF_OCCUPANCY
    elif dna_element == NUCLEOSOME:
        feature_name= COSMIC_NUCLEOSOME_OCCUPANCY
    else:
        feature_name = dna_element + '_' + COSMIC_OCCUPANCY

    try:
        NCI_Thesaurus_code = cancer_type_2_NCI_Thesaurus_code_dict[cancer_type]
    except:
        print('There is no NCI Thesauruscode for' + cancer_type)
        NCI_Thesaurus_code = cancer_type

    filename = '%s_%s_%s_TA_%s.%s' %(cosmic_release_version, signature, feature_name, NCI_Thesaurus_code, figure_file_extension)
    figure_file = os.path.join(plot_output_dir, OCCUPANCY, dna_element, COSMIC_TISSUE_BASED_FIGURES, filename)

    fig.savefig(figure_file, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return simulations_mean_average_signal_array


# Plot occupancy for each DNA element
# Manuscript cancer type based
# Plot cancer type based occupancy with correlation and similarity metrics
# with number of overlaps
# with whether this file is considered or not
def plot_occupancy_figure_manucript_tissue_based_ENCODE_file_based(plot_output_dir,
                                            combined_output_dir,
                                            occupancy_type,
                                            dna_element,
                                            signature,
                                            cancer_type,
                                            numberofMutations,
                                            average_signal_array_file_path,
                                            signature_cancer_type_occupancy_average_signal_array,
                                            signature_cancer_type_occupancy_count_array,
                                            accumulated_average_signal_array,
                                            xlabel,
                                            ylabel,
                                            label,
                                            text,
                                            numberofSimulations,
                                            color,
                                            fillcolor,
                                            pearson_correlation,
                                            spearman_correlation,
                                            signaturebased_cutoff,
                                            average_probability,
                                            plusorMinus,
                                            start,
                                            end,
                                            consider_both_real_and_sim_avg_overlap,
                                            fold_change,
                                            depleted_fold_change,
                                            enriched_fold_change,
                                            minimum_number_of_overlaps_required,
                                            across_all_cancer_types_simulations_average_signal_array_list,
                                            across_all_files_simulations_average_signal_array_list):

    real_data_avg_count = None
    sim_data_avg_count = None
    FILE_CONSIDERED = False

    sim_average_signal_array_files = []
    sim_count_array_files = []

    min_list = []
    max_list = []

    simulations_average_signal_array = None
    simulations_average_signal_array_list = None
    simulations_mean_average_signal_array = None

    if ((signature_cancer_type_occupancy_average_signal_array is not None) and (pd.notna(signature_cancer_type_occupancy_average_signal_array).any(axis=0)) and (np.any(signature_cancer_type_occupancy_average_signal_array))):
        min_list.append(np.nanmin(signature_cancer_type_occupancy_average_signal_array))
        max_list.append(np.nanmax(signature_cancer_type_occupancy_average_signal_array))

    if ((accumulated_average_signal_array is not None) and (pd.notna(accumulated_average_signal_array).any(axis=0)) and (np.any(accumulated_average_signal_array))):
        min_list.append(np.nanmin(accumulated_average_signal_array))
        max_list.append(np.nanmax(accumulated_average_signal_array))

    title = '%s %s' % (signature,cancer_type)
    if (numberofSimulations>0):
        simulations_average_signal_array_list = []
        simulations_count_array_list = []

        # Read the simulation files w.r.t. the current folder
        if occupancy_type == NUCLEOSOME_OCCUPANCY:
            if ((signature == AGGREGATEDSUBSTITUTIONS) or (signature == AGGREGATEDINDELS) or (signature == AGGREGATEDDINUCS)):
                main_path = os.path.join(combined_output_dir, cancer_type, DATA, occupancy_type, signature)
                if os.path.exists(main_path):
                    files_list = os.listdir(main_path)
                    cancer_type_sim = "%s_sim" % (cancer_type)
                    sim_average_signal_array_files = [os.path.join(main_path, file) for file in files_list if (file.startswith(cancer_type_sim)) and (file.endswith(AVERAGE_SIGNAL_ARRAY_TXT))]
                    sim_count_array_files = [os.path.join(main_path, file) for file in files_list if (file.startswith(cancer_type_sim)) and (file.endswith(ACCUMULATED_COUNT_ARRAY_TXT))]
            else:
                main_path = os.path.join(combined_output_dir, cancer_type, DATA, occupancy_type, SIGNATUREBASED)
                if os.path.exists(main_path):
                    files_list = os.listdir(main_path)
                    signature_sim = "%s_sim" % (signature)
                    sim_average_signal_array_files = [os.path.join(main_path, file) for file in files_list if (file.startswith(signature_sim)) and (file.endswith(AVERAGE_SIGNAL_ARRAY_TXT))]
                    sim_count_array_files = [os.path.join(main_path, file) for file in files_list if (file.startswith(signature_sim)) and (file.endswith(ACCUMULATED_COUNT_ARRAY_TXT))]

        elif occupancy_type == EPIGENOMICS_OCCUPANCY:
            # Get the list of sim files for the dna_element
            file_of_interest = os.path.basename(average_signal_array_file_path)
            if cancer_type in file_of_interest:
                cancer_type_start_index = file_of_interest.find(cancer_type)
                cancer_type_end_index = cancer_type_start_index + len(cancer_type)
                file_of_interest = file_of_interest[cancer_type_end_index:]
            first_pos = file_of_interest.find('_')
            second_pos = file_of_interest.find('_', first_pos + 1)
            file_accession = None
            if first_pos >= 0 and second_pos >= 0:
                file_accession = file_of_interest[first_pos + 1: second_pos]

            if ((signature == AGGREGATEDSUBSTITUTIONS) or (signature == AGGREGATEDINDELS) or (signature == AGGREGATEDDINUCS)):
                # Liver-HCC_simXX_ENCFF665OBP_right-lobe-of-liver_Normal_CTCF-human_AverageSignalArray.txt
                main_path = os.path.join(combined_output_dir, cancer_type, DATA, occupancy_type, signature)
                if os.path.exists(main_path):
                    files_list = os.listdir(main_path)
                    cancer_type_sim = "%s_sim" % (cancer_type)

                    if file_accession:
                        sim_average_signal_array_files = [os.path.join(main_path, file) for file in files_list if
                                                          (file.startswith(cancer_type_sim)) and
                                                          (file_accession in file) and
                                                          (dna_element.upper() in file.upper()) and
                                                          (file.endswith(AVERAGE_SIGNAL_ARRAY_TXT))]
                        sim_count_array_files = [os.path.join(main_path, file) for file in files_list if
                                                 (file.startswith(cancer_type_sim)) and
                                                 (file_accession in file) and
                                                 (dna_element.upper() in file.upper()) and
                                                 (file.endswith(ACCUMULATED_COUNT_ARRAY_TXT))]


            else:
                # SBS6_simXX_ENCFF690BYG_liver_Normal_CTCF-human_AverageSignalArray.txt
                main_path = os.path.join(combined_output_dir, cancer_type, DATA, occupancy_type, SIGNATUREBASED)
                if os.path.exists(main_path):
                    files_list = os.listdir(main_path)
                    signature_sim = "%s_sim" % (signature)
                    # os.path.basename(average_signal_array_file_path) --> SBS17b_ENCFF156VNT_body-of-pancreas_Normal_CTCF-human_AverageSignalArray.txt
                    if file_accession:
                        sim_average_signal_array_files = [os.path.join(main_path, file) for file in files_list if
                                                          (file.startswith(signature_sim)) and
                                                          (file_accession in file) and
                                                          (dna_element.upper() in file.upper()) and
                                                          (file.endswith(AVERAGE_SIGNAL_ARRAY_TXT))]
                        sim_count_array_files = [os.path.join(main_path, file) for file in files_list if
                                                 (file.startswith(signature_sim)) and
                                                 (file_accession in file) and
                                                 (dna_element.upper() in file.upper()) and
                                                 (file.endswith(ACCUMULATED_COUNT_ARRAY_TXT))]


        sim_average_signal_array_files = sorted(sim_average_signal_array_files, key=natural_key)
        sim_count_array_files = sorted(sim_count_array_files, key=natural_key)

        for sim_average_signal_array_file  in sim_average_signal_array_files:
            sim_average_signal_array = readAsFloatNumpyArray(sim_average_signal_array_file, plusorMinus)
            if (sim_average_signal_array is not None):
                simulations_average_signal_array_list.append(sim_average_signal_array)

        simulations_average_signal_array = np.vstack(simulations_average_signal_array_list)

        # To consider simulations average count
        for sim_count_array_file  in sim_count_array_files:
            sim_count_array = readAsFloatNumpyArray(sim_count_array_file, plusorMinus)
            if (sim_count_array is not None):
                simulations_count_array_list.append(sim_count_array)

        # Check whether somatic mutations and DNA element have enough overlap
        if signature_cancer_type_occupancy_count_array is not None:
            # If there is nan in the list np.mean returns nan.
            real_data_avg_count = np.nanmean(signature_cancer_type_occupancy_count_array[start:end])

            # Compute sim_data_avg_count
            # This is the simulations data
            stackedSimulationsSignatureBasedCount = np.vstack(simulations_count_array_list)

            # One sample way
            stackedSimulationsSignatureBasedCount_of_interest = stackedSimulationsSignatureBasedCount[:, start:end]

            # Get rid of rows with all nans
            stackedSimulationsSignatureBasedCount_of_interest = stackedSimulationsSignatureBasedCount_of_interest[
                ~np.isnan(stackedSimulationsSignatureBasedCount_of_interest).all(axis=1)]

            # Take mean row-wise
            simulations_horizontal_count_means_array = np.nanmean(stackedSimulationsSignatureBasedCount_of_interest, axis=1)
            sim_data_avg_count = np.nanmean(simulations_horizontal_count_means_array)

            # Number of mutations is checked before this function call
            if is_eligible(fold_change,
                           consider_both_real_and_sim_avg_overlap,
                           real_data_avg_count,
                           sim_data_avg_count,
                           depleted_fold_change,
                           enriched_fold_change,
                           minimum_number_of_overlaps_required):

                across_all_cancer_types_simulations_average_signal_array_list.append(simulations_average_signal_array)
                across_all_files_simulations_average_signal_array_list.append(simulations_average_signal_array)
                FILE_CONSIDERED = True

    if ((signature_cancer_type_occupancy_average_signal_array is not None) and (pd.notna(signature_cancer_type_occupancy_average_signal_array).any(axis=0)) and (np.any(signature_cancer_type_occupancy_average_signal_array))):

        sims_avg_signal_array, simulations_lows_list, simulations_highs_list = takeAverage(simulations_average_signal_array_list)

        if (simulations_lows_list is not None and simulations_lows_list):
            min_list.append(np.nanmin(simulations_lows_list))
        if (simulations_highs_list is not None and simulations_highs_list):
            max_list.append(np.nanmax(simulations_highs_list))

        from matplotlib import rcParams
        rcParams.update({'figure.autolayout': True})

        fig = plt.figure(figsize=(20,10), facecolor=None, dpi=100)
        plt.style.use('ggplot')

        # This code makes the background white.
        ax = plt.gca()
        ax.set_facecolor('white')
        # This code puts the edge line
        for edge_i in ['left', 'bottom','right', 'top']:
            ax.spines[edge_i].set_edgecolor("black")
            ax.spines[edge_i].set_linewidth(3)

        x = np.arange(-plusorMinus ,plusorMinus+1 ,1)

        listofLegends = []

        min_average_signal = np.nanmin(min_list)
        max_average_signal = np.nanmax(max_list)

        if (signature_cancer_type_occupancy_average_signal_array is not None):
            original = plt.plot(x, signature_cancer_type_occupancy_average_signal_array, color=color, label=label,linewidth=3)
            listofLegends.append(original[0])

        if (sims_avg_signal_array is not None):
            label = 'Simulations %s' %(label)
            simulations = plt.plot(x, sims_avg_signal_array, color='gray', linestyle='--',  label=label, linewidth=3)
            listofLegends.append(simulations[0])

        if (simulations_lows_list is not None) and (simulations_highs_list is not None):
            plt.fill_between(x, np.array(simulations_lows_list), np.array(simulations_highs_list), facecolor=fillcolor)

        plt.legend(loc= 'lower left', handles=listofLegends, prop={'size': 24}, shadow=False, edgecolor='white', facecolor='white')

        # put the number of snps
        if numberofMutations:
            tobeWrittenText = "{:,}".format(numberofMutations)
        else:
            tobeWrittenText = ""

        if pearson_correlation:
            pearson_text = "pearson:%f " %(pearson_correlation)
        else:
            pearson_text = ""

        if spearman_correlation:
            spearman_text = "spearman:%f " % (spearman_correlation)
        else:
            spearman_text = ""

        tobeWrittenText = tobeWrittenText + " " + text

        if signaturebased_cutoff is not None:
            cutoff_probability_text = '%0.2f' %(signaturebased_cutoff)
            tobeWrittenText=tobeWrittenText + ', cutoff probability=' + cutoff_probability_text

        if average_probability is not None:
            average_probability_text =', avg probability=%0.2f' %average_probability
            tobeWrittenText = tobeWrittenText + average_probability_text

        tobeWrittenText = tobeWrittenText + ', ' + pearson_text

        file_of_interest = os.path.basename(average_signal_array_file_path)
        first_pos = file_of_interest.find('_')
        last_pos = file_of_interest.rfind('_')

        if FILE_CONSIDERED:
            tobeWrittenText2="Average number of overlaps: real: %d sims: %d fold_change: %.2f--> Considered" %(real_data_avg_count, sim_data_avg_count, fold_change)
        else:
            tobeWrittenText2="Average number of overlaps: real: %d sims: %d fold_change: %.2f --> NOT Considered" %(real_data_avg_count, sim_data_avg_count, fold_change)

        plt.text(0.99, 0.99, tobeWrittenText, verticalalignment='top', horizontalalignment='right', transform=ax.transAxes, fontsize=24)
        plt.text(0.99, 0.94, file_of_interest[first_pos+1:last_pos], verticalalignment='top', horizontalalignment='right', transform=ax.transAxes, fontsize=24)
        plt.text(0.99, 0.89, tobeWrittenText2, verticalalignment='top', horizontalalignment='right', transform=ax.transAxes, fontsize=24)

        # Put vertical line at x=0
        plt.axvline(x=0, color='gray', linestyle='--')

        # This code provides the x and y tick marks and labels
        plt.xticks(np.arange(-plusorMinus, plusorMinus+1, step=plusorMinus/2), fontsize=30)

        plt.xlim((-plusorMinus, plusorMinus))

        if occupancy_type == NUCLEOSOME_OCCUPANCY:
            ymin = round(min_average_signal - 0.1, 2)
            ymax = round(max_average_signal + 0.1, 2)
            print('ymin:%s ymax:%s' %(ymin,ymax))
            if ((not np.isnan(ymin)) and (not np.isnan(ymax))):
                plt.yticks(np.arange(ymin, ymax, step=0.1), fontsize=30)
                plt.ylim((ymin - 0.01, ymax + 0.01))

        # This code puts the tick marks
        plt.tick_params(axis='both', which='major', labelsize=30,width=3,length=10)
        plt.tick_params(axis='both', which='minor', labelsize=30,width=3,length=10)

        plt.title(title, fontsize=40,fontweight='bold')
        plt.xlabel(xlabel,fontsize=32,fontweight='semibold')
        plt.ylabel(ylabel,fontsize=32,fontweight='semibold')

        if occupancy_type == NUCLEOSOME_OCCUPANCY:
            filename = '%s_%s_%s.png' % (signature, cancer_type, CANCER_TYPE_BASED_OCCUPANCY_FIGURE)
        elif occupancy_type == EPIGENOMICS_OCCUPANCY:
            filename = '%s_%s_%s.png' % (signature, cancer_type, file_of_interest[first_pos+1:-4])

        figureFile = os.path.join(plot_output_dir, OCCUPANCY, dna_element, MANUSCRIPT_TISSUE_BASED_FIGURES, filename)

        fig.savefig(figureFile, dpi=100, bbox_inches="tight")
        plt.close(fig)

    if (simulations_average_signal_array is not None) and simulations_average_signal_array.any():
        simulations_mean_average_signal_array = np.nanmean(simulations_average_signal_array, axis=0) # column-wise mean

    if (simulations_mean_average_signal_array is not None) and np.any(simulations_mean_average_signal_array):
        return simulations_mean_average_signal_array
    else:
        return None

def accumulateReplicationTimeAcrossAllTissues_plot_figure_for_all_types(plot_output_dir,
                                                                        combined_output_dir,
                                                                        signature_tuples,
                                                                        cancer_types,
                                                                        numberofSimulations,
                                                                        figure_type,
                                                                        number_of_mutations_required,
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
                                                                        sub_figure_type):

    signature_cancer_type_replication_time_df_list = []

    def append_df(signature_cancer_type_df):
        signature_cancer_type_replication_time_df_list.append(signature_cancer_type_df)

    # Parallel version for real runs starts
    numofProcesses = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(numofProcesses)

    for (signature, signature_type) in signature_tuples:
        pool.apply_async(accumulate_replication_time_across_all_cancer_types_plot_figure,
                         args=(plot_output_dir,
                               combined_output_dir,
                               signature,
                               signature_type,
                               cancer_types,
                               numberofSimulations,
                               figure_type,
                               number_of_mutations_required,
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
                               sub_figure_type,),
                         callback=append_df)

    pool.close()
    pool.join()
    # Parallel version for real runs ends

    # # Sequentilal version for debugging starts
    # for (signature, signature_type) in signature_tuples:
    #     signature_cancer_type_replication_time_df = accumulate_replication_time_across_all_cancer_types_plot_figure(
    #                            plot_output_dir,
    #                            combined_output_dir,
    #                            signature,
    #                            signature_type,
    #                            cancer_types,
    #                            numberofSimulations,
    #                            figure_type,
    #                            number_of_mutations_required,
    #                            cosmic_release_version,
    #                            figure_file_extension,
    #                            replication_time_significance_level,
    #                            replication_time_slope_cutoff,
    #                            replication_time_difference_between_min_and_max,
    #                            replication_time_difference_between_medians,
    #                            pearson_spearman_correlation_cutoff,
    #                            cosmic_legend,
    #                            cosmic_signature,
    #                            cosmic_fontsize_text,
    #                            cosmic_cancer_type_fontsize,
    #                            cosmic_fontweight,
    #                            cosmic_fontsize_labels,
    #                            sub_figure_type)
    #     append_df(signature_cancer_type_replication_time_df)
    # # Sequentilal version for debugging ends

    all_signatures_replication_time_df = pd.concat(signature_cancer_type_replication_time_df_list, axis=0)
    return all_signatures_replication_time_df


# Plot figures in parallel
def accumulateOccupancyAcrossAllCancerTypes_plot_figure_for_signatures(plot_output_dir,
                                                                       combined_output_dir,
                                                                       occupancy_type,
                                                                       dna_element,
                                                                       signature_tuples,
                                                                       cancer_types,
                                                                       numberofSimulations,
                                                                       plusorMinus,
                                                                       start,
                                                                       end,
                                                                       consider_both_real_and_sim_avg_overlap,
                                                                       minimum_number_of_overlaps_required_for_sbs,
                                                                       minimum_number_of_overlaps_required_for_dbs,
                                                                       minimum_number_of_overlaps_required_for_indels,
                                                                       figure_type,
                                                                       number_of_mutations_required,
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
                                                                       figure_case_study):

    if figure_type == MANUSCRIPT:
        # plot legend to indicate that blue/red/green are real signals while gray is simulated signal
        plot_occupancy_legend(plot_output_dir, dna_element)

    df_list = []

    def append_df(signature_cancer_type_file_name_occupancy_df):
        df_list.append(signature_cancer_type_file_name_occupancy_df)

    # Parallel version for real runs starts
    numofProcesses = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(numofProcesses)

    for (signature, signature_type) in signature_tuples:
        pool.apply_async(accumulateOccupancyAcrossAllCancerTypes_plot_figure_for_signature,
                         args=(plot_output_dir,
                               combined_output_dir,
                               occupancy_type,
                               dna_element,
                               signature,
                               signature_type,
                               cancer_types,
                               numberofSimulations,
                               plusorMinus,
                               start,
                               end,
                               consider_both_real_and_sim_avg_overlap,
                               minimum_number_of_overlaps_required_for_sbs,
                               minimum_number_of_overlaps_required_for_dbs,
                               minimum_number_of_overlaps_required_for_indels,
                               figure_type,
                               number_of_mutations_required,
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
                               figure_case_study,),
                         callback = append_df)

    pool.close()
    pool.join()
    # Parallel version for real runs ends

    # # Sequential version for testing/debugging starts
    # for (signature, signature_type) in signature_tuples:
    #     signature_cancer_type_file_name_occupancy_df = accumulateOccupancyAcrossAllCancerTypes_plot_figure_for_signature(
    #                             plot_output_dir,
    #                             combined_output_dir,
    #                             occupancy_type,
    #                             dna_element,
    #                             signature,
    #                             signature_type,
    #                             cancer_types,
    #                             numberofSimulations,
    #                             plusorMinus,
    #                             start,
    #                             end,
    #                             consider_both_real_and_sim_avg_overlap,
    #                             minimum_number_of_overlaps_required_for_sbs,
    #                             minimum_number_of_overlaps_required_for_dbs,
    #                             minimum_number_of_overlaps_required_for_indels,
    #                             figure_type,
    #                             number_of_mutations_required,
    #                             cosmic_release_version,
    #                             figure_file_extension,
    #                             depleted_fold_change,
    #                             enriched_fold_change,
    #                             occupancy_significance_level,
    #                             pearson_spearman_correlation_cutoff,
    #                             cosmic_legend,
    #                             cosmic_correlation_text,
    #                             cosmic_labels,
    #                             cancer_type_on_right_hand_side,
    #                             cosmic_fontsize_text,
    #                             cosmic_fontsize_ticks,
    #                             cosmic_fontsize_labels,
    #                             cosmic_linewidth_plot,
    #                             cosmic_title_all_cancer_types,
    #                             figure_case_study)
    #     append_df(signature_cancer_type_file_name_occupancy_df)
    # # Sequential version for testing/debugging ends

    accumulated_signature_cancer_type_file_name_occupancy_df = pd.concat(df_list)

    return accumulated_signature_cancer_type_file_name_occupancy_df


def compute_sim_data_avg_signal_and_count_given_average_signal_array_file_path(combined_output_dir,
                                                                               cancer_type,
                                                                               signature,
                                                                               dna_element,
                                                                               occupancy_type,
                                                                               average_signal_array_file_path,
                                                                               plusorMinus,
                                                                               start,
                                                                               end):
    simulations_count_array_list = []
    simulations_avg_signal_array_list = []

    # Read the simulation files w.r.t. the current folder
    if occupancy_type == NUCLEOSOME_OCCUPANCY:
        if ((signature == AGGREGATEDSUBSTITUTIONS) or (signature == AGGREGATEDINDELS) or (signature == AGGREGATEDDINUCS)):
            main_path = os.path.join(combined_output_dir, cancer_type, DATA, occupancy_type, signature)
            if os.path.exists(main_path):
                files_list = os.listdir(main_path)
                cancer_type_sim = "%s_sim" % (cancer_type)
                sim_count_array_files = [os.path.join(main_path, file) for file in files_list if (file.startswith(cancer_type_sim)) and (file.endswith(ACCUMULATED_COUNT_ARRAY_TXT))]
                sim_avg_signal_array_files = [os.path.join(main_path, file) for file in files_list if (file.startswith(cancer_type_sim)) and (file.endswith(AVERAGE_SIGNAL_ARRAY_TXT))]

        else:
            main_path = os.path.join(combined_output_dir, cancer_type, DATA, occupancy_type, SIGNATUREBASED)
            if os.path.exists(main_path):
                files_list = os.listdir(main_path)
                signature_sim = "%s_sim" % (signature)
                sim_count_array_files = [os.path.join(main_path, file) for file in files_list if (file.startswith(signature_sim)) and (file.endswith(ACCUMULATED_COUNT_ARRAY_TXT))]
                sim_avg_signal_array_files = [os.path.join(main_path, file) for file in files_list if (file.startswith(signature_sim)) and (file.endswith(AVERAGE_SIGNAL_ARRAY_TXT))]

    elif occupancy_type == EPIGENOMICS_OCCUPANCY:
        # Get the list of sim files for the dna_element
        # SBS85_ENCFF900JSN_B-cell_Normal_H3K4me2-human_AverageSignalArray.txt
        # B_cell_malignancy_kataegis_ENCFF755GYS_B-cell_Normal_H3K27me3-human_AccumulatedSignalArray.txt
        # Liver-HCC_simXX_ENCFF665OBP_right-lobe-of-liver_Normal_CTCF-human_AverageSignalArray.txt
        file_of_interest = os.path.basename(average_signal_array_file_path)
        if cancer_type in file_of_interest:
            cancer_type_start_index = file_of_interest.find(cancer_type)
            cancer_type_end_index = cancer_type_start_index + len(cancer_type)
            file_of_interest = file_of_interest[cancer_type_end_index:]
        first_pos = file_of_interest.find('_')
        second_pos = file_of_interest.find('_', first_pos + 1)
        file_accession = None
        if first_pos >= 0 and second_pos >= 0:
            file_accession = file_of_interest[first_pos + 1: second_pos]
        if ((signature == AGGREGATEDSUBSTITUTIONS) or (signature == AGGREGATEDINDELS) or (signature == AGGREGATEDDINUCS)):
            # Liver-HCC_simXX_ENCFF665OBP_right-lobe-of-liver_Normal_CTCF-human_AverageSignalArray.txt
            main_path = os.path.join(combined_output_dir, cancer_type, DATA, occupancy_type, signature)
            if os.path.exists(main_path):
                files_list = os.listdir(main_path)
                cancer_type_sim = "%s_sim" % (cancer_type)
                if file_accession:
                    sim_count_array_files = [os.path.join(main_path, file) for file in files_list if (file.startswith(cancer_type_sim)) and (file_accession in file) and (dna_element.upper() in file.upper()) and (file.endswith(ACCUMULATED_COUNT_ARRAY_TXT))]
                    sim_avg_signal_array_files = [os.path.join(main_path, file) for file in files_list if (file.startswith(cancer_type_sim)) and (file_accession in file) and (dna_element.upper() in file.upper()) and (file.endswith(AVERAGE_SIGNAL_ARRAY_TXT))]

        else:
            # SBS6_simXX_ENCFF690BYG_liver_Normal_CTCF-human_AverageSignalArray.txt
            main_path = os.path.join(combined_output_dir, cancer_type, DATA, occupancy_type, SIGNATUREBASED)
            if os.path.exists(main_path):
                files_list = os.listdir(main_path)
                signature_sim = "%s_sim" % (signature)
                # os.path.basename(average_signal_array_file_path) --> SBS17b_ENCFF156VNT_body-of-pancreas_Normal_CTCF-human_AverageSignalArray.txt
                if file_accession:
                    sim_count_array_files = [os.path.join(main_path, file) for file in files_list if (file.startswith(signature_sim)) and (file_accession in file) and (dna_element.upper() in file.upper()) and (file.endswith(ACCUMULATED_COUNT_ARRAY_TXT))]
                    sim_avg_signal_array_files = [os.path.join(main_path, file) for file in files_list if (file.startswith(signature_sim)) and (file_accession in file) and (dna_element.upper() in file.upper()) and (file.endswith(AVERAGE_SIGNAL_ARRAY_TXT))]

    sim_count_array_files = sorted(sim_count_array_files, key=natural_key)
    sim_avg_signal_array_files = sorted(sim_avg_signal_array_files, key=natural_key)

    for sim_count_array_file, sim_avg_signal_array_file in zip(sim_count_array_files, sim_avg_signal_array_files):
        sim_count_array = readAsFloatNumpyArray(sim_count_array_file, plusorMinus)
        sim_avg_signal_array = readAsFloatNumpyArray(sim_avg_signal_array_file, plusorMinus)

        if (sim_count_array is not None):
            simulations_count_array_list.append(sim_count_array)
        if (sim_avg_signal_array is not None):
            simulations_avg_signal_array_list.append(sim_avg_signal_array)

    # Compute sim_data_avg_count
    # This is the simulations data
    stackedSimulationsSignatureBasedCount = np.vstack(simulations_count_array_list)

    # One sample way
    stackedSimulationsSignatureBasedCount_of_interest = stackedSimulationsSignatureBasedCount[:, start:end]

    # Get rid of rows with all nans
    stackedSimulationsSignatureBasedCount_of_interest = stackedSimulationsSignatureBasedCount_of_interest[
        ~np.isnan(stackedSimulationsSignatureBasedCount_of_interest).all(axis=1)]

    # Take mean row-wise
    simulations_horizontal_count_means_array = np.nanmean(stackedSimulationsSignatureBasedCount_of_interest, axis=1)
    sim_data_avg_count = np.nanmean(simulations_horizontal_count_means_array)

    if ((simulations_avg_signal_array_list is not None) and simulations_avg_signal_array_list):
        # This is the simulations data
        stackedSimulationsSignatureBased = np.vstack(simulations_avg_signal_array_list)
        stackedSimulationsSignatureBased_of_interest = stackedSimulationsSignatureBased[:, start:end]

        # Get rid of rows with all nans
        stackedSimulationsSignatureBased_of_interest = stackedSimulationsSignatureBased_of_interest[
            ~np.isnan(stackedSimulationsSignatureBased_of_interest).all(axis=1)]

        # Take mean row-wise
        simulations_horizontal_means_array = np.nanmean(stackedSimulationsSignatureBased_of_interest, axis=1)
        sim_data_avg_signal = np.nanmean(simulations_horizontal_means_array)

    return sim_data_avg_signal, sim_data_avg_count

def compute_fold_change(avg_real_signal, sim_data_avg_signal):
    if (avg_real_signal is not None) and (sim_data_avg_signal is not None):
        try:
            fold_change = avg_real_signal / sim_data_avg_signal
        except ZeroDivisionError:
            fold_change = np.nan

    return fold_change

def is_eligible(fold_change,
              consider_both_real_and_sim_avg_overlap,
              real_data_avg_count,
              sim_data_avg_count,
              depleted_fold_change,
              enriched_fold_change,
              minimum_number_of_overlaps_required):

    # Number of overlaps constraint is checked here
    if consider_both_real_and_sim_avg_overlap:
        if (real_data_avg_count is not None) and \
            (sim_data_avg_count is not None) and \
                ((real_data_avg_count >= minimum_number_of_overlaps_required) or \
                (depleted(fold_change, depleted_fold_change) and (
                        sim_data_avg_count >= minimum_number_of_overlaps_required)) or \
                (enriched(fold_change, enriched_fold_change) and (
                        sim_data_avg_count >= minimum_number_of_overlaps_required) and (
                         real_data_avg_count >= minimum_number_of_overlaps_required * OCCUPANCY_HEATMAP_COMMON_MULTIPLIER))):
            return True

    elif (real_data_avg_count is not None) and \
            (real_data_avg_count >= minimum_number_of_overlaps_required):
        return True

    return False


# Main engine function for combined across cancer types occupancy
# First pass: for the given signature, across all cancer types and all considered files
# Second pass: for the given signature and cancer type, across all considered files
def accumulateOccupancyAcrossAllCancerTypes_plot_figure_for_signature(plot_output_dir,
                                                                      main_combined_output_dir,
                                                                      occupancy_type,
                                                                      dna_element,
                                                                      signature,
                                                                      signature_type,
                                                                      cancer_types,
                                                                      numberofSimulations,
                                                                      plusorMinus,
                                                                      start,
                                                                      end,
                                                                      consider_both_real_and_sim_avg_overlap,
                                                                      minimum_number_of_overlaps_required_for_sbs,
                                                                      minimum_number_of_overlaps_required_for_dbs,
                                                                      minimum_number_of_overlaps_required_for_indels,
                                                                      figure_type,
                                                                      number_of_mutations_required,
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
                                                                      figure_case_study):

    signature_cancer_type_file_name_occupancy_df = pd.DataFrame(columns=["signature",
                                                                         "cancer_type",
                                                                         "file_name",
                                                                         "real_average_signal_array",
                                                                         "sims_average_signal_array",
                                                                         "spearman_corr",
                                                                         "spearman_p_value",
                                                                         "spearman_q_value",
                                                                         "pearson_corr",
                                                                         "pearson_p_value",
                                                                         "pearson_q_value",
                                                                         "cutoff",
                                                                         "number_of_mutations",
                                                                         "average_probability",
                                                                         "real_average_number_of_overlaps",
                                                                         "sims_average_number_of_overlaps",
                                                                         "avg_real_signal",
                                                                         "sim_data_avg_signal",
                                                                         "fold_change"])


    spearman_p_values = []
    pearson_p_values = []
    spearman_element_names = []
    pearson_element_names = []

    # Accumulate for all cancer types having this signature
    # Will be used in simulations of accumulated across all cancer types figure
    # Filled in the first pass
    across_all_cancer_types_average_signal_array = np.zeros((plusorMinus * 2 + 1,))

    # Filled in the second pass
    across_all_cancer_types_simulations_average_signal_array_list = []

    number_of_eligible_files = 0

    if occupancy_type == NUCLEOSOME_OCCUPANCY:
        ylabel = 'Average Nucleosome Signal'
    elif occupancy_type == EPIGENOMICS_OCCUPANCY:
        ylabel = 'Average %s Signal' %(dna_element)
    else:
        ylabel = 'Average Signal'

    if (signature_type == SBS):
        xlabel = 'Interval around single point mutation (bp)'
        label = 'Single Base Substitutions'
        text = 'subs'
        color = 'royalblue'
        fillcolor = 'lightblue'
        minimum_number_of_overlaps_required = minimum_number_of_overlaps_required_for_sbs
    elif (signature_type == DBS):
        xlabel = 'Interval around dinuc (bp)'
        label = 'Double Base Substitutions'
        text = 'dinucs'
        color = 'crimson'
        fillcolor = 'lightpink'
        minimum_number_of_overlaps_required = minimum_number_of_overlaps_required_for_dbs
    elif (signature_type == ID):
        xlabel = 'Interval around indel (bp)'
        label = 'Indels'
        text = 'indels'
        color = 'darkgreen'
        fillcolor = 'lightgreen'
        minimum_number_of_overlaps_required = minimum_number_of_overlaps_required_for_indels

    print('\n####################################')
    print('FIRST PASS STARTS for %s' %(signature))
    print('####################################\n')

    # First pass across all cancer types to accumulate signal and count arrays
    for main_cancer_type in cancer_types:
        number_of_mutations = 0

        # For Lymph-BNHL based on signature we may use either Lymph-BNHL_clustered, Lymph-BNHL_nonClustered or Lymph-BNHL
        # For Lymph-CLL based on signature we may use either Lymph-CLL_clustered, Lymph-CLL_nonClustered or Lymph-CLL
        # Use combined_output_dir and cancer_type returned by get_alternative_combined_output_dir_and_cancer_type while reading files
        # Use main_cancer_type while accumulating results in pandas dataframe or calculating p-values
        combined_output_dir, cancer_type = get_alternative_combined_output_dir_and_cancer_type(main_combined_output_dir, main_cancer_type, signature)

        if (signature == AGGREGATEDSUBSTITUTIONS) or (signature == AGGREGATEDINDELS) or (signature == AGGREGATEDDINUCS):
            number_of_mutations_filename = TABLE_MUTATIONTYPE_NUMBEROFMUTATIONS_NUMBEROFSAMPLES_SAMPLESLIST
            number_of_mutations_filepath = os.path.join(combined_output_dir,cancer_type,DATA,number_of_mutations_filename)
            number_of_mutations_df = pd.read_csv(number_of_mutations_filepath, header=0, sep='\t')
            # mutation_type   number_of_mutations     number_of_samples       samples_list
            if signature == AGGREGATEDSUBSTITUTIONS:
                if np.any(number_of_mutations_df[number_of_mutations_df['mutation_type'] == SUBS]['number_of_mutations'].values):
                    number_of_mutations = number_of_mutations_df[number_of_mutations_df['mutation_type']==SUBS]['number_of_mutations'].values[0]
            elif signature == AGGREGATEDDINUCS:
                if np.any(number_of_mutations_df[number_of_mutations_df['mutation_type'] == DINUCS]['number_of_mutations'].values):
                    number_of_mutations = number_of_mutations_df[number_of_mutations_df['mutation_type']==DINUCS]['number_of_mutations'].values[0]
            elif signature == AGGREGATEDINDELS:
                if np.any(number_of_mutations_df[number_of_mutations_df['mutation_type'] == INDELS]['number_of_mutations'].values):
                    number_of_mutations = number_of_mutations_df[number_of_mutations_df['mutation_type']==INDELS]['number_of_mutations'].values[0]

        else:
            number_of_mutations_filename = get_number_of_mutations_filename(signature_type)
            number_of_mutations_filepath = os.path.join(combined_output_dir,cancer_type,DATA,number_of_mutations_filename)
            number_of_mutations_df = pd.read_csv(number_of_mutations_filepath, header=0, sep='\t')
            if np.any(number_of_mutations_df[number_of_mutations_df['signature'] == signature]['number_of_mutations'].values):
                number_of_mutations = number_of_mutations_df[number_of_mutations_df['signature']==signature]['number_of_mutations'].values[0]

        # Number of mutations constraint is checked here
        if number_of_mutations >= number_of_mutations_required:
            accumulated_signal_array_files = []
            accumulated_count_array_files = []
            accumulated_average_signal_array_files = []

            # There is only one Nucleosome Occupancy File
            if occupancy_type == NUCLEOSOME_OCCUPANCY:
                if ((signature == AGGREGATEDSUBSTITUTIONS) or (signature == AGGREGATEDINDELS) or (signature == AGGREGATEDDINUCS)):
                    signature_cancer_type_occupancy_average_signal_array_file_name = '%s_%s' % (cancer_type, AVERAGE_SIGNAL_ARRAY_TXT)
                    signature_cancer_type_occupancy_signal_array_file_name = '%s_%s' % (cancer_type, ACCUMULATED_SIGNAL_ARRAY_TXT)
                    signature_cancer_type_occupancy_count_array_file_name = '%s_%s' % (cancer_type, ACCUMULATED_COUNT_ARRAY_TXT)

                    signature_cancer_type_occupancy_average_signal_array_file_path = os.path.join(combined_output_dir,
                                                                                        cancer_type, DATA,
                                                                                        occupancy_type,
                                                                                        signature,
                                                                                        signature_cancer_type_occupancy_average_signal_array_file_name)

                    signature_cancer_type_occupancy_signal_array_file_path = os.path.join(combined_output_dir,
                                                                                        cancer_type, DATA,
                                                                                        occupancy_type,
                                                                                        signature,
                                                                                        signature_cancer_type_occupancy_signal_array_file_name)

                    signature_cancer_type_occupancy_count_array_file_path = os.path.join(combined_output_dir,
                                                                                       cancer_type, DATA,
                                                                                       occupancy_type,
                                                                                       signature,
                                                                                       signature_cancer_type_occupancy_count_array_file_name)
                else:
                    signature_cancer_type_occupancy_average_signal_array_file_name = '%s_%s' % (signature, AVERAGE_SIGNAL_ARRAY_TXT)
                    signature_cancer_type_occupancy_signal_array_file_name = '%s_%s' % (signature, ACCUMULATED_SIGNAL_ARRAY_TXT)
                    signature_cancer_type_occupancy_count_array_file_name = '%s_%s' % (signature, ACCUMULATED_COUNT_ARRAY_TXT)

                    signature_cancer_type_occupancy_average_signal_array_file_path = os.path.join(combined_output_dir,
                                                                                        cancer_type, DATA,
                                                                                        occupancy_type,
                                                                                        SIGNATUREBASED,
                                                                                        signature_cancer_type_occupancy_average_signal_array_file_name)

                    signature_cancer_type_occupancy_signal_array_file_path = os.path.join(combined_output_dir,
                                                                                        cancer_type, DATA,
                                                                                        occupancy_type,
                                                                                        SIGNATUREBASED,
                                                                                        signature_cancer_type_occupancy_signal_array_file_name)

                    signature_cancer_type_occupancy_count_array_file_path = os.path.join(combined_output_dir,
                                                                                       cancer_type, DATA,
                                                                                       occupancy_type,
                                                                                       SIGNATUREBASED,
                                                                                       signature_cancer_type_occupancy_count_array_file_name)

                if os.path.exists(signature_cancer_type_occupancy_average_signal_array_file_path):
                    accumulated_average_signal_array_files = [signature_cancer_type_occupancy_average_signal_array_file_path]
                if os.path.exists(signature_cancer_type_occupancy_signal_array_file_path):
                    accumulated_signal_array_files = [signature_cancer_type_occupancy_signal_array_file_path]
                if os.path.exists(signature_cancer_type_occupancy_count_array_file_path):
                    accumulated_count_array_files = [signature_cancer_type_occupancy_count_array_file_path]

            # There might be more than one file for a DNA element
            elif occupancy_type == EPIGENOMICS_OCCUPANCY:
                # Get the list of files for the dna_element
                if ((signature == AGGREGATEDSUBSTITUTIONS) or (signature == AGGREGATEDINDELS) or (signature == AGGREGATEDDINUCS)):
                    # Liver-HCC_ENCFF665OBP_right-lobe-of-liver_Normal_CTCF-human_AccumulatedCountArray.txt
                    main_path = os.path.join(combined_output_dir,cancer_type, DATA, occupancy_type, signature)
                    if os.path.exists(main_path):
                        files_list = os.listdir(main_path)
                        cancer_type_ = "%s_" %(cancer_type)
                        cancer_type_sim= "%s_sim" %(cancer_type)
                        accumulated_average_signal_array_files = [os.path.join(main_path,file) for file in files_list if ((file.startswith(cancer_type_)) and (not file.startswith(cancer_type_sim)) and (dna_element.upper() in file.upper()) and (file.endswith(AVERAGE_SIGNAL_ARRAY_TXT)))]
                        accumulated_signal_array_files = [os.path.join(main_path,file) for file in files_list if ((file.startswith(cancer_type_)) and (not file.startswith(cancer_type_sim)) and (dna_element.upper() in file.upper()) and (file.endswith(ACCUMULATED_SIGNAL_ARRAY_TXT)))]
                        accumulated_count_array_files = [os.path.join(main_path,file) for file in files_list if ((file.startswith(cancer_type_)) and (not file.startswith(cancer_type_sim)) and (dna_element.upper() in file.upper()) and (file.endswith(ACCUMULATED_COUNT_ARRAY_TXT)))]
                else:
                    # SBS6_ENCFF690BYG_liver_Normal_CTCF-human_AccumulatedCountArray.txt
                    main_path = os.path.join(combined_output_dir,cancer_type, DATA,occupancy_type,SIGNATUREBASED)
                    if os.path.exists(main_path):
                        files_list = os.listdir(main_path)
                        signature_ = "%s_" %(signature)
                        signature_sim = "%s_sim" %(signature)
                        accumulated_average_signal_array_files = [os.path.join(main_path,file) for file in files_list if ((file.startswith(signature_)) and (not file.startswith(signature_sim)) and (dna_element.upper() in file.upper()) and (file.endswith(AVERAGE_SIGNAL_ARRAY_TXT)))]
                        accumulated_signal_array_files = [os.path.join(main_path,file) for file in files_list if ((file.startswith(signature_)) and (not file.startswith(signature_sim)) and (dna_element.upper() in file.upper()) and (file.endswith(ACCUMULATED_SIGNAL_ARRAY_TXT)))]
                        accumulated_count_array_files = [os.path.join(main_path,file) for file in files_list if ((file.startswith(signature_)) and (not file.startswith(signature_sim)) and (dna_element.upper() in file.upper()) and (file.endswith(ACCUMULATED_COUNT_ARRAY_TXT)))]

            if (len(accumulated_average_signal_array_files) > 0) and (len(accumulated_signal_array_files) > 0) and (len(accumulated_count_array_files) > 0):
                accumulated_average_signal_array_files = sorted(accumulated_average_signal_array_files, key=natural_key)
                accumulated_signal_array_files = sorted(accumulated_signal_array_files, key=natural_key)
                accumulated_count_array_files = sorted(accumulated_count_array_files, key=natural_key)
                for average_signal_array_file_path, signal_array_file_path, count_array_file_path in zip(accumulated_average_signal_array_files, accumulated_signal_array_files, accumulated_count_array_files):
                    if  os.path.exists(average_signal_array_file_path) and os.path.exists(signal_array_file_path) and os.path.exists(count_array_file_path):
                        signature_cancer_type_occupancy_average_signal_array = readAsFloatNumpyArray(average_signal_array_file_path, plusorMinus)
                        signature_cancer_type_occupancy_count_array = readAsIntNumpyArray(count_array_file_path, plusorMinus)
                        # Check whether somatic mutations and DNA element have enough overlap
                        if signature_cancer_type_occupancy_count_array is not None:
                            # If there is nan in the list np.mean returns nan.
                            real_data_avg_count = np.nanmean(signature_cancer_type_occupancy_count_array[start:end])

                            sim_data_avg_signal, sim_data_avg_count = compute_sim_data_avg_signal_and_count_given_average_signal_array_file_path(combined_output_dir,
                                                                                                                 cancer_type,
                                                                                                                 signature,
                                                                                                                 dna_element,
                                                                                                                 occupancy_type,
                                                                                                                 average_signal_array_file_path,
                                                                                                                 plusorMinus,
                                                                                                                 start,
                                                                                                                 end)

                            avg_real_signal = np.nanmean(signature_cancer_type_occupancy_average_signal_array[start:end])

                            fold_change = compute_fold_change(avg_real_signal, sim_data_avg_signal)

                            if is_eligible(fold_change,
                                         consider_both_real_and_sim_avg_overlap,
                                         real_data_avg_count,
                                         sim_data_avg_count,
                                         depleted_fold_change,
                                         enriched_fold_change,
                                         minimum_number_of_overlaps_required):
                                # Accumulate
                                across_all_cancer_types_average_signal_array += signature_cancer_type_occupancy_average_signal_array
                                number_of_eligible_files += 1

    # First pass ends
    # number_of_eligible_files shows the files satisfy our number of mutations and number of overlaps constraints
    if number_of_eligible_files > 0:
        # Take average
        across_all_cancer_types_average_signal_array = across_all_cancer_types_average_signal_array / number_of_eligible_files

    print('\n####################################')
    print('SECOND PASS STARTS for %s' %(signature))
    print('####################################\n')

    # Second pass: for each cancer type calculate correlation metrics and cosine similarity
    for main_cancer_type in cancer_types:

        # For Lymph-BNHL based on signature we may use either Lymph-BNHL_clustered, Lymph-BNHL_nonClustered or Lymph-BNHL
        # For Lymph-CLL based on signature we may use either Lymph-CLL_clustered, Lymph-CLL_nonClustered or Lymph-CLL
        # Use combined_output_dir and cancer_type returned by get_alternative_combined_output_dir_and_cancer_type while reading files
        # Use main_cancer_type while accumulating results in pandas dataframe or calculating p-values
        combined_output_dir, cancer_type = get_alternative_combined_output_dir_and_cancer_type(main_combined_output_dir, main_cancer_type, signature)

        number_of_mutations = None
        average_probability = None
        cutoff = None

        # For the same cancer type
        # Fill in Second Pass
        # Accumulate for all files having this signature, cancer type and dna element
        # e.g.: There can be more than one CTCF file for SBS17b, Liver-HCC and CTCF
        across_all_files_average_signal_array = np.zeros((plusorMinus * 2 + 1,))

        # For the same cancer type there can be more than one ENCODE file
        # Fill in Second Pass
        across_all_files_simulations_average_signal_array_list = []

        if (signature == AGGREGATEDSUBSTITUTIONS) or (signature == AGGREGATEDINDELS) or (signature == AGGREGATEDDINUCS):
            number_of_mutations_filename = TABLE_MUTATIONTYPE_NUMBEROFMUTATIONS_NUMBEROFSAMPLES_SAMPLESLIST
            number_of_mutations_filepath = os.path.join(combined_output_dir,cancer_type,DATA,number_of_mutations_filename)
            number_of_mutations_df = pd.read_csv(number_of_mutations_filepath, header=0, sep='\t')
            # mutation_type   number_of_mutations     number_of_samples       samples_list
            if signature == AGGREGATEDSUBSTITUTIONS:
                if np.any(number_of_mutations_df[number_of_mutations_df['mutation_type'] == SUBS]['number_of_mutations'].values):
                    number_of_mutations = number_of_mutations_df[number_of_mutations_df['mutation_type']==SUBS]['number_of_mutations'].values[0]
            elif signature == AGGREGATEDDINUCS:
                if np.any(number_of_mutations_df[number_of_mutations_df['mutation_type'] == DINUCS]['number_of_mutations'].values):
                    number_of_mutations = number_of_mutations_df[number_of_mutations_df['mutation_type']==DINUCS]['number_of_mutations'].values[0]
            elif signature == AGGREGATEDINDELS:
                if np.any(number_of_mutations_df[number_of_mutations_df['mutation_type'] == INDELS]['number_of_mutations'].values):
                    number_of_mutations = number_of_mutations_df[number_of_mutations_df['mutation_type']==INDELS]['number_of_mutations'].values[0]

        else:
            number_of_mutations_filename = get_number_of_mutations_filename(signature_type)
            number_of_mutations_filepath = os.path.join(combined_output_dir,cancer_type,DATA,number_of_mutations_filename)
            number_of_mutations_df = pd.read_csv(number_of_mutations_filepath, header=0, sep='\t')
            if np.any(number_of_mutations_df[number_of_mutations_df['signature'] == signature]['number_of_mutations'].values):
                number_of_mutations = number_of_mutations_df[number_of_mutations_df['signature'] == signature]['number_of_mutations'].values[0]
            if np.any(number_of_mutations_df[number_of_mutations_df['signature'] == signature]['average_probability'].values):
                average_probability = number_of_mutations_df[number_of_mutations_df['signature'] == signature]['average_probability'].values[0]
            if np.any(number_of_mutations_df[number_of_mutations_df['signature'] == signature]['cutoff'].values):
                cutoff = float(number_of_mutations_df[number_of_mutations_df['signature'] == signature]['cutoff'].values[0])

        if (number_of_mutations is not None) and (number_of_mutations >= number_of_mutations_required):
            average_signal_array_files = []

            print('#############################')
            print('Second Pass: %s %s ' %(signature,cancer_type))

            if occupancy_type == NUCLEOSOME_OCCUPANCY:
                if ((signature == AGGREGATEDSUBSTITUTIONS) or (signature == AGGREGATEDINDELS) or (signature == AGGREGATEDDINUCS)):
                    signature_cancer_type_occupancy_average_signal_array_filename = '%s_%s' % (cancer_type, AVERAGE_SIGNAL_ARRAY_TXT)
                    signature_cancer_type_occupancy_average_signal_array_filepath = os.path.join(combined_output_dir,cancer_type, DATA, occupancy_type,signature,signature_cancer_type_occupancy_average_signal_array_filename)

                    signature_cancer_type_occupancy_count_array_file_name = '%s_%s' % (cancer_type, ACCUMULATED_COUNT_ARRAY_TXT)
                    signature_cancer_type_occupancy_count_array_file_path = os.path.join(combined_output_dir,cancer_type, DATA,occupancy_type,signature,signature_cancer_type_occupancy_count_array_file_name)

                else:
                    signature_cancer_type_occupancy_average_signal_array_filename = '%s_%s' % (signature, AVERAGE_SIGNAL_ARRAY_TXT)
                    signature_cancer_type_occupancy_average_signal_array_filepath = os.path.join(combined_output_dir,cancer_type, DATA,occupancy_type,SIGNATUREBASED,signature_cancer_type_occupancy_average_signal_array_filename)

                    signature_cancer_type_occupancy_count_array_file_name = '%s_%s' % (signature, ACCUMULATED_COUNT_ARRAY_TXT)
                    signature_cancer_type_occupancy_count_array_file_path = os.path.join(combined_output_dir,cancer_type, DATA,occupancy_type,SIGNATUREBASED,signature_cancer_type_occupancy_count_array_file_name)

                average_signal_array_files = [signature_cancer_type_occupancy_average_signal_array_filepath]
                accumulated_count_array_files = [signature_cancer_type_occupancy_count_array_file_path]

            elif occupancy_type == EPIGENOMICS_OCCUPANCY:
                # Get the list of files for the dna_element
                if ((signature == AGGREGATEDSUBSTITUTIONS) or (signature == AGGREGATEDINDELS) or (signature == AGGREGATEDDINUCS)):
                    # Liver-HCC_ENCFF665OBP_right-lobe-of-liver_Normal_CTCF-human_AverageSignalArray.txt
                    main_path = os.path.join(combined_output_dir, cancer_type, DATA, occupancy_type, signature)
                    if os.path.exists(main_path):
                        files_list = os.listdir(main_path)
                        cancer_type_ = "%s_" %(cancer_type)
                        cancer_type_sim= "%s_sim" %(cancer_type)
                        average_signal_array_files = [os.path.join(main_path,file) for file in files_list if (file.startswith(cancer_type_)) and (not file.startswith(cancer_type_sim)) and (dna_element.upper() in file.upper()) and (file.endswith(AVERAGE_SIGNAL_ARRAY_TXT))]
                        accumulated_count_array_files = [os.path.join(main_path,file) for file in files_list if ((file.startswith(cancer_type_)) and (not file.startswith(cancer_type_sim)) and (dna_element.upper() in file.upper()) and (file.endswith(ACCUMULATED_COUNT_ARRAY_TXT)))]

                else:
                    # SBS6_ENCFF690BYG_liver_Normal_CTCF-human_AverageSignalArray.txt
                    main_path = os.path.join(combined_output_dir, cancer_type, DATA, occupancy_type, SIGNATUREBASED)
                    if os.path.exists(main_path):
                        files_list = os.listdir(main_path)
                        signature_ = "%s_" %(signature)
                        signature_sim = "%s_sim" %(signature)
                        average_signal_array_files = [os.path.join(main_path,file) for file in files_list if (file.startswith(signature_)) and (not file.startswith(signature_sim)) and (dna_element.upper() in file.upper()) and (file.endswith(AVERAGE_SIGNAL_ARRAY_TXT))]
                        accumulated_count_array_files = [os.path.join(main_path,file) for file in files_list if ((file.startswith(signature_)) and (not file.startswith(signature_sim)) and (dna_element.upper() in file.upper()) and (file.endswith(ACCUMULATED_COUNT_ARRAY_TXT)))]

            average_signal_array_files = sorted(average_signal_array_files,key=natural_key)
            accumulated_count_array_files = sorted(accumulated_count_array_files,key=natural_key)

            for average_signal_array_file_path, count_array_file_path in zip(average_signal_array_files, accumulated_count_array_files):
                signature_cancer_type_occupancy_average_signal_array = None
                signature_cancer_type_occupancy_count_array = None
                average_signal_array_file_name = None

                if (os.path.exists(average_signal_array_file_path)):
                    signature_cancer_type_occupancy_average_signal_array = readAsFloatNumpyArray(average_signal_array_file_path, plusorMinus)
                    average_signal_array_file_name = os.path.basename(average_signal_array_file_path)

                if os.path.exists(count_array_file_path):
                    signature_cancer_type_occupancy_count_array = readAsIntNumpyArray(count_array_file_path, plusorMinus)
                    # Check whether somatic mutations and DNA element have enough overlap
                    if signature_cancer_type_occupancy_count_array is not None:
                        # If there is nan in the list np.mean returns nan.
                        real_data_avg_count = np.nanmean(signature_cancer_type_occupancy_count_array[start:end])

                        sim_data_avg_signal, sim_data_avg_count = compute_sim_data_avg_signal_and_count_given_average_signal_array_file_path(
                            combined_output_dir,
                            cancer_type,
                            signature,
                            dna_element,
                            occupancy_type,
                            average_signal_array_file_path,
                            plusorMinus,
                            start,
                            end)

                        avg_real_signal = np.nanmean(signature_cancer_type_occupancy_average_signal_array[start:end])

                        fold_change = compute_fold_change(avg_real_signal, sim_data_avg_signal)

                        # Check whether somatic mutations and DNA element have enough overlaps
                        if is_eligible(fold_change,
                                       consider_both_real_and_sim_avg_overlap,
                                       real_data_avg_count,
                                       sim_data_avg_count,
                                       depleted_fold_change,
                                       enriched_fold_change,
                                       minimum_number_of_overlaps_required):

                            across_all_files_average_signal_array += signature_cancer_type_occupancy_average_signal_array

                if ((signature_cancer_type_occupancy_average_signal_array is not None) and (np.any(signature_cancer_type_occupancy_average_signal_array))):
                    print('Second Pass: %s %s %s' % (signature, cancer_type, average_signal_array_file_path))
                    spearman_corr = np.nan
                    spearman_p_value = np.nan
                    pearson_corr = np.nan
                    pearson_p_value = np.nan
                    is_eligible_flag = False


                    if is_eligible(fold_change,
                                   consider_both_real_and_sim_avg_overlap,
                                   real_data_avg_count,
                                   sim_data_avg_count,
                                   depleted_fold_change,
                                   enriched_fold_change,
                                   minimum_number_of_overlaps_required):

                        is_eligible_flag = True

                        if (not np.any(np.isnan(signature_cancer_type_occupancy_average_signal_array))):
                            # spearman_corr, spearman_p_value = spearmanr(signature_cancer_type_occupancy_average_signal_array,across_all_cancer_types_average_signal_array)
                            spearman_corr, spearman_p_value = spearmanr(
                                signature_cancer_type_occupancy_average_signal_array[500:1501],
                                across_all_cancer_types_average_signal_array[500:1501])

                        if (not np.any(np.isnan(signature_cancer_type_occupancy_average_signal_array))):
                            # pearson_corr, pearson_p_value = pearsonr(signature_cancer_type_occupancy_average_signal_array,across_all_cancer_types_average_signal_array)
                            pearson_corr, pearson_p_value = pearsonr(
                                signature_cancer_type_occupancy_average_signal_array[500:1501],
                                across_all_cancer_types_average_signal_array[500:1501])

                        if (spearman_corr and (spearman_corr >= pearson_spearman_correlation_cutoff)):
                            spearman_p_values.append(spearman_p_value)
                            spearman_element_names.append((signature, main_cancer_type, average_signal_array_file_name)) # cancer_type

                        if (pearson_corr and (pearson_corr >= pearson_spearman_correlation_cutoff)):
                            pearson_p_values.append(pearson_p_value)
                            pearson_element_names.append((signature, main_cancer_type, average_signal_array_file_name)) # cancer_type

                    # Cosine similarity is not trustworthy
                    # cos_sim = cosine_similarity([signaturebased_tissuebased_nucleosomeoccupancy_average_signal_array],[accumulated_average_signal_array])

                    # Manuscript tissue Based figure for each ENCODE file
                    # Manucscript Signature - Cancer Type - Occupancy Figure with correlations - for Manuscript pdf files
                    # Where is it plotted?
                    # Under .../SigProfilerTopographyAuxiliary/combined_pcawg_nonpcawg/occupancy/
                    # This figure has tissue based, tissue based simulations and across all tissues with pearson correlation
                    # Now we accumulate cancer_type based simulations in across_all_tissues_simulations_signal_list
                    # Accumulate simulations if number of overlaps >= minimum number of overlaps needed
                    # Accumulate simulations signal array separately
                    # Accumulate simulations count array separately
                    simulations_mean_average_signal_array = plot_occupancy_figure_manucript_tissue_based_ENCODE_file_based(plot_output_dir,
                                                                                combined_output_dir,
                                                                                occupancy_type,
                                                                                dna_element,
                                                                                signature,
                                                                                cancer_type,
                                                                                number_of_mutations,
                                                                                average_signal_array_file_path,
                                                                                signature_cancer_type_occupancy_average_signal_array,
                                                                                signature_cancer_type_occupancy_count_array,
                                                                                across_all_cancer_types_average_signal_array,
                                                                                xlabel, ylabel, label,
                                                                                text,
                                                                                numberofSimulations,
                                                                                color,
                                                                                fillcolor,
                                                                                pearson_corr,
                                                                                spearman_corr,
                                                                                cutoff,
                                                                                average_probability,
                                                                                plusorMinus,
                                                                                start,
                                                                                end,
                                                                                consider_both_real_and_sim_avg_overlap,
                                                                                fold_change,
                                                                                depleted_fold_change,
                                                                                enriched_fold_change,
                                                                                minimum_number_of_overlaps_required,
                                                                                across_all_cancer_types_simulations_average_signal_array_list,
                                                                                across_all_files_simulations_average_signal_array_list)


                    signature_cancer_type_file_name_occupancy_df = signature_cancer_type_file_name_occupancy_df.append(
                                                {"signature": signature,
                                                   "cancer_type": main_cancer_type, # Lymph-BNHL is used either results come from Lymph-BNHL_clustered, Lymph-BNHL_nonClustered or Lymph-BNHL, same for Lymph-CLL
                                                   "file_name": average_signal_array_file_name,
                                                   "real_average_signal_array" : np.around(signature_cancer_type_occupancy_average_signal_array[500:1501], NUMBER_OF_DECIMAL_PLACES_TO_ROUND).tolist(),
                                                   "sims_average_signal_array" : np.around(simulations_mean_average_signal_array[500:1501], NUMBER_OF_DECIMAL_PLACES_TO_ROUND).tolist(),
                                                   "is_eligible" : is_eligible_flag,
                                                   "spearman_corr": np.around(spearman_corr, NUMBER_OF_DECIMAL_PLACES_TO_ROUND) if spearman_corr is not None else np.nan,
                                                   "spearman_p_value": spearman_p_value,
                                                   "spearman_q_value": np.nan,
                                                   "pearson_corr": np.around(pearson_corr, NUMBER_OF_DECIMAL_PLACES_TO_ROUND) if pearson_corr is not None else np.nan,
                                                   "pearson_p_value": pearson_p_value,
                                                   "pearson_q_value": np.nan,
                                                   "cutoff": cutoff,
                                                   "number_of_mutations": number_of_mutations,
                                                   "average_probability": np.around(average_probability, NUMBER_OF_DECIMAL_PLACES_TO_ROUND) if average_probability is not None else np.nan,
                                                   "real_average_number_of_overlaps": np.around(real_data_avg_count, NUMBER_OF_DECIMAL_PLACES_TO_ROUND) if real_data_avg_count is not None else np.nan,
                                                   "sims_average_number_of_overlaps": np.around(sim_data_avg_count, NUMBER_OF_DECIMAL_PLACES_TO_ROUND) if sim_data_avg_count is not None else np.nan,
                                                   "avg_real_signal": np.around(avg_real_signal, NUMBER_OF_DECIMAL_PLACES_TO_ROUND) if avg_real_signal is not None else np.nan,
                                                   "sim_data_avg_signal": np.around(sim_data_avg_signal, NUMBER_OF_DECIMAL_PLACES_TO_ROUND) if sim_data_avg_signal is not None else np.nan,
                                                   "fold_change": np.around(fold_change, NUMBER_OF_DECIMAL_PLACES_TO_ROUND) if fold_change is not None else np.nan},
                        ignore_index=True)

        # For each cancer type in the Second Pass
        if (across_all_files_average_signal_array.any()):
            if (not np.any(np.isnan(across_all_files_average_signal_array))):

                across_all_files_average_signal_array = across_all_files_average_signal_array/len(across_all_files_simulations_average_signal_array_list)

                pearson_corr, pearson_p_value = pearsonr(across_all_files_average_signal_array[500:1501],
                                                         across_all_cancer_types_average_signal_array[500:1501])

                # Cosmic Tissue based figure for each ENCODE DNA element
                # There can be one or more files in case of e.g.: CTCF
                plot_occupancy_figure_cosmic_tissue_based(plot_output_dir,
                                                          occupancy_type,
                                                          dna_element,
                                                          signature,
                                                          main_cancer_type, # cancer_type
                                                          across_all_files_average_signal_array,
                                                          across_all_files_simulations_average_signal_array_list,
                                                          label,
                                                          color,
                                                          fillcolor,
                                                          pearson_corr,
                                                          pearson_p_value,
                                                          plusorMinus,
                                                          cosmic_release_version,
                                                          figure_file_extension,
                                                          cosmic_legend = cosmic_legend,
                                                          cosmic_correlation_text = cosmic_correlation_text,
                                                          cosmic_labels = cosmic_labels,
                                                          cancer_type_on_right_hand_side = cancer_type_on_right_hand_side,
                                                          cosmic_fontsize_text = cosmic_fontsize_text,
                                                          cosmic_fontsize_ticks = cosmic_fontsize_ticks,
                                                          cosmic_fontsize_labels = cosmic_fontsize_labels,
                                                          cosmic_linewidth_plot = cosmic_linewidth_plot,
                                                          figure_case_study = figure_case_study)

            else:
                print('There is a problem, nan value in across_all_files_average_signal_array', signature, cancer_type, dna_element)

    # Signature Based Across All Combined Tissues
    if (across_all_cancer_types_average_signal_array.any()):
        print('\n#################################################################')
        print('Second Pass ended for %s' %(signature))
        print('\n#################################################################\n')

        # Correct p-values
        spearman_p_values_array = np.asarray(spearman_p_values)
        pearson_p_values_array = np.asarray(pearson_p_values)

        # If there a p_values in the array
        if (spearman_p_values_array.size > 0):
            rejected, spearman_corrected_p_values, alphacSidak, alphacBonf = statsmodels.stats.multitest.multipletests(spearman_p_values_array, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)

        if (pearson_p_values_array.size > 0):
            rejected, pearson_corrected_p_values, alphacSidak, alphacBonf = statsmodels.stats.multitest.multipletests(pearson_p_values_array, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)

        for element_index, element_name in enumerate(spearman_element_names,0):
            (signature, cancer_type, file_name)= element_name
            spearman_q_value = spearman_corrected_p_values[element_index]
            signature_cancer_type_file_name_occupancy_df.loc[
            ((signature_cancer_type_file_name_occupancy_df['signature'] == signature) &
                (signature_cancer_type_file_name_occupancy_df['cancer_type'] == cancer_type) &
                (signature_cancer_type_file_name_occupancy_df['file_name'] == file_name)),'spearman_q_value'] = spearman_q_value

        for element_index, element_name in enumerate(pearson_element_names,0):
            (signature, cancer_type, file_name) = element_name
            pearson_q_value = pearson_corrected_p_values[element_index]
            signature_cancer_type_file_name_occupancy_df.loc[
                ((signature_cancer_type_file_name_occupancy_df['signature'] == signature) &
                (signature_cancer_type_file_name_occupancy_df['cancer_type'] == cancer_type) &
                (signature_cancer_type_file_name_occupancy_df['file_name'] == file_name)), 'pearson_q_value'] = pearson_q_value

        cancer_types_array = signature_cancer_type_file_name_occupancy_df[signature_cancer_type_file_name_occupancy_df['signature'] == signature]['cancer_type'].unique()
        num_of_cancer_types_with_considered_files = 0
        num_of_cancer_types_pearson_q_values_le_significance_level = 0

        for cancer_type in cancer_types_array:
            pearson_q_values_array = signature_cancer_type_file_name_occupancy_df[(signature_cancer_type_file_name_occupancy_df['signature'] == signature) & (signature_cancer_type_file_name_occupancy_df['cancer_type'] == cancer_type)]['pearson_q_value'].values
            # There is at least one file with corrected p-value le significance level
            if len(np.argwhere(pearson_q_values_array <= occupancy_significance_level)) > 0:
                num_of_cancer_types_pearson_q_values_le_significance_level += 1

            real_average_number_of_overlaps_array = signature_cancer_type_file_name_occupancy_df[
                (signature_cancer_type_file_name_occupancy_df['signature'] == signature) &
                (signature_cancer_type_file_name_occupancy_df['cancer_type'] == cancer_type)]['real_average_number_of_overlaps'].values

            sims_average_number_of_overlaps_array = signature_cancer_type_file_name_occupancy_df[
                (signature_cancer_type_file_name_occupancy_df['signature'] == signature) &
                (signature_cancer_type_file_name_occupancy_df['cancer_type'] == cancer_type)]['sims_average_number_of_overlaps'].values

            fold_change_array = signature_cancer_type_file_name_occupancy_df[
                (signature_cancer_type_file_name_occupancy_df['signature'] == signature) &
                (signature_cancer_type_file_name_occupancy_df['cancer_type'] == cancer_type)]['fold_change'].values

            if consider_both_real_and_sim_avg_overlap:
                for real_data_avg_count,  sim_data_avg_count, fold_change in zip(real_average_number_of_overlaps_array,
                                                                                 sims_average_number_of_overlaps_array,
                                                                                 fold_change_array):
                    if (real_data_avg_count >= minimum_number_of_overlaps_required) or\
                            (depleted(fold_change, depleted_fold_change) and
                             (sim_data_avg_count >= minimum_number_of_overlaps_required)) or\
                            (enriched(fold_change, enriched_fold_change) and
                             (sim_data_avg_count >= minimum_number_of_overlaps_required) and
                             (real_data_avg_count >= minimum_number_of_overlaps_required * OCCUPANCY_HEATMAP_COMMON_MULTIPLIER)):
                        num_of_cancer_types_with_considered_files += 1
                        break

            elif (len(np.argwhere(real_average_number_of_overlaps_array >= minimum_number_of_overlaps_required)) > 0):
                num_of_cancer_types_with_considered_files += 1

        # COSMIC Across all tissues
        # MANUSCRIPT Across all tissues
        # average nucleosome signal across all tissues having this signature with average simulations
        # Please note that we use num_of_tissues_with_pearson_q_value_le_significance_level/num_of_all_tissues in the figures
        across_all_cancer_types_sims_average_signal_array = plot_occupancy_figure_across_all_cancer_types(plot_output_dir,
                                                        occupancy_type,
                                                        dna_element,
                                                        signature,
                                                        across_all_cancer_types_average_signal_array,
                                                        across_all_cancer_types_simulations_average_signal_array_list,
                                                        color,
                                                        fillcolor,
                                                        num_of_cancer_types_pearson_q_values_le_significance_level,
                                                        num_of_cancer_types_with_considered_files,
                                                        plusorMinus,
                                                        figure_type,
                                                        cosmic_release_version,
                                                        figure_file_extension,
                                                        cosmic_legend = cosmic_legend,
                                                        cosmic_correlation_text = cosmic_correlation_text,
                                                        cosmic_fontsize_text = cosmic_fontsize_text,
                                                        cosmic_fontsize_ticks = cosmic_fontsize_ticks,
                                                        cosmic_fontsize_labels = cosmic_fontsize_labels,
                                                        cosmic_linewidth_plot = cosmic_linewidth_plot,
                                                        cosmic_title_all_cancer_types = cosmic_title_all_cancer_types,
                                                        figure_case_study = figure_case_study)


        signature_cancer_type_file_name_occupancy_df = signature_cancer_type_file_name_occupancy_df.append(
            {"signature": signature,
             "cancer_type": ACROSS_ALL_CANCER_TYPES,
             "file_name": np.nan,
             "real_average_signal_array": np.around(across_all_cancer_types_average_signal_array[500:1501], NUMBER_OF_DECIMAL_PLACES_TO_ROUND).tolist(),
             "sims_average_signal_array": np.around(across_all_cancer_types_sims_average_signal_array[500:1501], NUMBER_OF_DECIMAL_PLACES_TO_ROUND).tolist(),
             "is_eligible": np.nan,
             "spearman_corr": np.nan,
             "spearman_p_value": np.nan,
             "spearman_q_value": np.nan,
             "pearson_corr": np.nan,
             "pearson_p_value": np.nan,
             "pearson_q_value": np.nan,
             "cutoff": np.nan,
             "number_of_mutations": np.nan,
             "average_probability": np.nan,
             "real_average_number_of_overlaps": np.nan,
             "sims_average_number_of_overlaps": np.nan,
             "avg_real_signal": np.nan,
             "sim_data_avg_signal": np.nan,
             "fold_change" : np.nan
             }, ignore_index=True)

    return signature_cancer_type_file_name_occupancy_df


def fill_occupancy_pdfs(output_dir,
                        occupancy_type,
                        dna_element,
                        signature_tuples,
                        cancer_types,
                        figure_type,
                        cosmic_release_version,
                        figure_file_extension):

    # for each signature
    # center across all cancer types
    # then each cancer type left and right
    for (signature, signature_type) in signature_tuples:
        interested_file_list = []

        if figure_type == COSMIC:
            if dna_element == CTCF:
                feature_name = COSMIC_CTCF_OCCUPANCY
            elif dna_element == NUCLEOSOME:
                feature_name = COSMIC_NUCLEOSOME_OCCUPANCY
            else:
                feature_name = dna_element + '_' + COSMIC_OCCUPANCY
            filename = '%s_%s_%s.%s' %(cosmic_release_version, signature, feature_name, figure_file_extension)
            filepath = os.path.join(output_dir, OCCUPANCY, dna_element, FIGURES_COSMIC, filename)
        elif figure_type == MANUSCRIPT:
            filename = '%s_%s_%s.png' %(signature, dna_element, ACROSS_ALL_CANCER_TYPES_OCCUPANCY_FIGURE)
            filepath = os.path.join(output_dir, OCCUPANCY, dna_element, FIGURES_MANUSCRIPT, filename)
        if os.path.exists(filepath):
            interested_file_list.append(filepath)

        for cancer_type in cancer_types:
            #signature cancer_type
            if occupancy_type == NUCLEOSOME_OCCUPANCY:
                tissuebased_filename= '%s_%s_%s.png' %(signature, cancer_type, CANCER_TYPE_BASED_OCCUPANCY_FIGURE)
                tissuebased_filepath = os.path.join(output_dir, OCCUPANCY, dna_element, MANUSCRIPT_TISSUE_BASED_FIGURES, tissuebased_filename)
                #At least one file must exists for one tissue type
                if (os.path.exists(tissuebased_filepath)):
                    interested_file_list.append(tissuebased_filepath)
            elif occupancy_type == EPIGENOMICS_OCCUPANCY:
                main_path = os.path.join(output_dir, OCCUPANCY, dna_element, MANUSCRIPT_TISSUE_BASED_FIGURES)
                if os.path.exists(main_path):
                    files_list = os.listdir(main_path)
                    signature_cancer_type = "%s_%s" %(signature,cancer_type)
                    signature_cancer_type_files = [os.path.join(main_path, file) for file in files_list if
                                                      (file.startswith(signature_cancer_type)) and (dna_element.upper() in file.upper()) and ('10K' not in file) and ('5K' not in file)]
                    interested_file_list.extend(signature_cancer_type_files)

        print('Printing PDF for %s' %(signature))
        print('Printing PDF for %s len(interested_file_list):%d' %(signature,len(interested_file_list)))
        print('Printing PDF for %s interested_file_list: %s' %(signature,interested_file_list))

        # One pdf for each signature, first left image: signature across all tissue prob05 --- first right image: signature across all tissues prob09
        # other images: signature for each tissue prob05 --- signature for each tissue prob09
        print('################')
        pdffile_name = "%s_%s" % (signature,OCCUPANCY_PDF)

        pdf_file_path = os.path.join(output_dir,OCCUPANCY,dna_element, PDF_FILES, pdffile_name)
        print(pdf_file_path)
        print('################')
        c = canvas.Canvas(pdf_file_path, pagesize=letter)
        width, height = letter  # keep for later
        print('canvas letter: width=%d height=%d' % (width, height))
        # width=612 height=792

        # Center header
        c.setFont("Times-Roman", 15)
        title = 'Occupancy' + ' ' + signature

        title_width = stringWidth(title, "Times-Roman", 15)
        c.drawString((width - title_width) / 2, height - 20, title)

        # One page can take 8 images
        # For images
        c.setFont("Times-Roman", 10)
        figureCount = 0

        first_figure_x = 60
        figure_left_x = 10
        figure_right_x = 310

        y = 570

        #For nucleosome occupancy pdfs
        if (len(interested_file_list)>1):

            for file in interested_file_list:

                if (file==interested_file_list[0]):
                    figure_width = 15

                    img = utils.ImageReader(file)
                    iw, ih = img.getSize()
                    print('image: width=%d height=%d' % (iw, ih))
                    aspect = ih / float(iw)

                    # To the center
                    c.drawImage(file, first_figure_x, y, figure_width * cm, figure_width * aspect * cm)
                    figureCount = figureCount + 2
                    y = y - 180

                elif os.path.exists(file):
                    figure_width = 10

                    img = utils.ImageReader(file)
                    iw, ih = img.getSize()
                    print('image: width=%d height=%d' % (iw, ih))
                    aspect = ih / float(iw)
                    print(file)
                    figureCount = figureCount + 1

                    # To the left
                    if (figureCount % 2 == 1):
                        c.drawImage(file, figure_left_x, y, figure_width * cm, figure_width * aspect * cm)
                    # To the right
                    elif (figureCount % 2 == 0):
                        c.drawImage(file, figure_right_x, y, figure_width * cm, figure_width * aspect * cm)
                        y = y - 180
                    if (figureCount % 8 == 0):
                        c.showPage()
                        c.setFont("Times-Roman", 10)
                        y = 570
            c.save()


def getReplicationTimeBarPlot(my_type):
    if ((SBS in my_type) or (AGGREGATEDSUBSTITUTIONS in my_type)):
        barcolor='royalblue'
    elif ((ID in my_type) or (AGGREGATEDINDELS in my_type)):
        # barcolor='darkgreen'
        barcolor='yellowgreen'
    elif (DBS in my_type) or (AGGREGATEDDINUCS in my_type):
        barcolor='crimson'
    return barcolor

def getReplicationTimeSimulationFaceColor(my_type):
    if ((SBS in my_type) or (AGGREGATEDSUBSTITUTIONS in my_type)):
        facecolor='lightblue'
    elif ((ID in my_type) or (AGGREGATEDINDELS in my_type)):
        facecolor='lightgreen'
    elif (DBS in my_type) or (AGGREGATEDDINUCS in my_type):
        facecolor='lightpink'
    return facecolor


# COSMIC and MANUSCRIPT
# Tissue based
# Across all tissues
# Uses the same method
# if cancer_type == None --> Across All Cancer Types
# if cancer_type != None --> Tissue Based
def plot_replication_time_figure(output_dir,
                                normalizedMutationDensityList,
                                signature,
                                simulations_lows,
                                simulations_means,
                                simulations_highs,
                                num_of_all_tissues,
                                figure_type,
                                cosmic_release_version,
                                figure_file_extension,
                                cancer_type = None,
                                num_of_increasing = 0,
                                num_of_flat = 0,
                                num_of_decreasing = 0,
                                cosmic_legend = True,
                                cosmic_signature = True,
                                cosmic_fontsize_text = 20,
                                cosmic_cancer_type_fontsize = 20/3,
                                cosmic_fontweight = 'semibold',
                                cosmic_fontsize_labels = 10,
                                sub_figure_type = None):

    barcolor = getReplicationTimeBarPlot(signature)
    facecolor = getReplicationTimeSimulationFaceColor(signature)

    # Unicode for arrows
    # Up arrow: unicode \u2191
    # Right arrow: unicode \u2192
    # Down arrow: unicode \u2193

    # North East Arrow: \u2197
    # Right arrow: unicode \u2192
    # South East Arrow: \u2198

    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})

    if figure_type == COSMIC:
        square_a = 4 #4
        aggregated_square_a = 5
        labelsize = cosmic_fontsize_labels #10
        aggregated_labelsize = cosmic_fontsize_labels #10
        fontsize = cosmic_fontsize_text #20
        cancer_type_fontsize = cosmic_cancer_type_fontsize
        fontweight = cosmic_fontweight
        aggregated_fontsize = cosmic_fontsize_text #20
        width = 1 #1
        bar_width = 0.9 #0.9  # the width of the bars
        bar_linewidth = 1 #1
        pad = 5 #5
        spine_line_width = 1 #1
        length = 5 #5
        simulations_linewidth = 2 #2
        dpi = 300

        # COSMIC Using NorthEast Right SouthDown Arrows
        if cancer_type == None:
            if (signature == 'aggregatedsubstitutions'):
                title_signature = 'Substitutions:'
            elif (signature == 'aggregateddinucs'):
                title_signature = 'Dinucs:'
            elif (signature == 'aggregatedindels'):
                title_signature = 'Indels:'
            else:
                title_signature = signature

            if num_of_all_tissues == 1:
                title_increasing_flat_decreasing='%d Cancer type\n' \
                        '%d/%d \u2197 Increasing\n' \
                        '%d/%d \u2192 Flat\n' \
                        '%d/%d \u2198 Decreasing' % (num_of_all_tissues,num_of_increasing,num_of_all_tissues,num_of_flat, num_of_all_tissues,num_of_decreasing,num_of_all_tissues)
            else:
                title_increasing_flat_decreasing='%d Cancer types\n' \
                        '%d/%d \u2197 Increasing\n' \
                        '%d/%d \u2192 Flat\n' \
                        '%d/%d \u2198 Decreasing' % (num_of_all_tissues,num_of_increasing,num_of_all_tissues,num_of_flat, num_of_all_tissues,num_of_decreasing,num_of_all_tissues)
        else:
            title_signature = signature
            title_increasing_flat_decreasing = cancer_type

    elif figure_type == MANUSCRIPT:
        square_a = 10
        # square_a = 8 # legacy
        aggregated_square_a = 10
        labelsize=60
        aggregated_labelsize=75
        fontsize = 60
        aggregated_fontsize=75
        width=3
        bar_width = 0.9  # the width of the bars
        bar_linewidth=3
        pad=40
        spine_line_width=3
        length=10
        simulations_linewidth=5
        dpi=100

        if num_of_all_tissues == 1:
            cancer_types = 'Cancer Type'
        else:
            cancer_types = 'Cancer Types'
        # Using NorthEast Right SouthDown Arrows
        if (signature == 'aggregatedsubstitutions'):
            title = '%s\n%d %s\n \u2197: %d \u2192: %d \u2198: %d' % ('All Substitutions:', num_of_all_tissues, cancer_types, num_of_increasing, num_of_flat, num_of_decreasing) # wo circles
        elif (signature == 'aggregateddinucs'):
            title = '%s\n%d %s\n \u2197: %d \u2192: %d \u2198: %d' % ('All Doublets:', num_of_all_tissues, cancer_types, num_of_increasing, num_of_flat, num_of_decreasing) # wo circls
        elif (signature == 'aggregatedindels'):
            title = '%s\n%d %s\n \u2197: %d \u2192: %d \u2198: %d' % ('All Indels:', num_of_all_tissues, cancer_types, num_of_increasing, num_of_flat, num_of_decreasing)
        else:
            title = '%s: %d\n \u2197: %d \u2192: %d \u2198: %d' % (signature, num_of_all_tissues, num_of_increasing, num_of_flat, num_of_decreasing)

    if (signature == 'aggregatedsubstitutions') or (signature == 'aggregateddinucs') or (signature == 'aggregatedindels'):
        # Note if you decrease the figure size decrease the fontsize accordingly
        fig= plt.figure(figsize=(aggregated_square_a, aggregated_square_a), dpi=dpi)
        plt.tick_params(axis='y', which='major', labelsize=aggregated_labelsize, width=width, length=length, pad=pad)
        plt.tick_params(axis='y', which='minor', labelsize=aggregated_labelsize, width=width, length=length, pad=pad)
    else:
        fig = plt.figure(figsize=(square_a, square_a), dpi=dpi)
        plt.tick_params(axis='y', which='major', labelsize=labelsize, width=width, length=length, pad=pad)
        plt.tick_params(axis='y', which='minor', labelsize=labelsize, width=width, length=length, pad=pad)

    ax = plt.gca()

    if figure_type == COSMIC:
        if sub_figure_type:
            if cancer_type:
                plt.title(cancer_type, fontsize=fontsize, pad=10, loc='center')
            else:
                # For Figure Case Study Tobacco Smoking and Chewing
                # For Figure Case Study B cell malignancies
                plt.title(sub_figure_type, fontsize=fontsize, pad=10, loc='center')

        else:
            anchored_text_signature = AnchoredText(title_signature,
                                         frameon=False, borderpad=0, pad=0.1,
                                         loc='upper left', bbox_to_anchor=[-0.3, 1.3],
                                         bbox_transform=plt.gca().transAxes,
                                         prop={'fontsize': fontsize, 'fontweight' : fontweight})
            ax.add_artist(anchored_text_signature)

            anchored_text_increasing_flat_decreasing = AnchoredText(title_increasing_flat_decreasing,
                                         frameon=False, borderpad=0, pad=0.1,
                                         loc='upper right', bbox_to_anchor=[1, 1.3],
                                         bbox_transform=plt.gca().transAxes,
                                         prop={'fontsize': cancer_type_fontsize, 'fontweight' : fontweight})
            ax.add_artist(anchored_text_increasing_flat_decreasing)

    elif figure_type == MANUSCRIPT:
        plt.title(title, fontsize=fontsize, pad=pad, loc='center',fontweight='semibold')

        # For circle around number of cancer types
        # if num_of_all_tissues<10:
        #     bbox_to_anchor_list = [0.96, 1.42]
        # else:
        #     bbox_to_anchor_list = [1, 1.42]
        # anchored_text_number_of_cancer_types = AnchoredText(num_of_all_tissues,
        #                                                     frameon=False, borderpad=0, pad=0.1,
        #                                                     loc='upper right',
        #                                                     bbox_to_anchor=bbox_to_anchor_list,
        #                                                     bbox_transform=plt.gca().transAxes,
        #                                                     prop={'fontsize': 60,
        #                                                           'fontweight': 'semibold'})
        #
        # ax.add_artist(anchored_text_number_of_cancer_types)
        #
        # circle1 = mpatches.Circle((0.9, 1.35), radius=0.12, transform=ax.transAxes, zorder=100, fill=False, color='black',lw=8, clip_on=False)
        # ax.add_patch(circle1)

    plt.style.use('ggplot')
    # This code makes the background white.
    ax.set_facecolor('white')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(True)

    for edge_i in ['left']:
        ax.spines[edge_i].set_edgecolor("black")
        ax.spines[edge_i].set_linewidth(spine_line_width)
        # This code draws line only between [0,1]
        ax.spines[edge_i].set_bounds(0, 1)

    # Note x get labels w.r.t. the order given here, 0 means get the 0th label from  xticks
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # plt.xticks(np.arange(10),('1st', '2nd', '3rd', '4th', '5th','6th','7th','8th','9th','10th'),rotation=20)
    plt.yticks(np.arange(0, 1.01, step=0.2))
    # also works
    # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    listofLegends = []
    real = plt.bar(x, normalizedMutationDensityList, bar_width, color=barcolor, edgecolor="black", linewidth=bar_linewidth, zorder=1)
    listofLegends.append(real[0])

    if simulations_means is not None:
        sims = plt.plot(x, simulations_means, 'o--', color='black', linewidth=simulations_linewidth, zorder =2)
        listofLegends.append(sims[0])
        if (len(simulations_lows)==len(simulations_highs)):
            plt.fill_between(x, np.array(simulations_lows), np.array(simulations_highs), facecolor=facecolor, zorder=2)

    if figure_type == COSMIC:
        if cancer_type == None:
            objects = ('Across All Cancer Types Real', 'Across All Cancer Types Simulations')
        else:
            if signature.startswith('SBS') or (signature == 'aggregatedsubstitutions'):
                objects = ('Real Substitutions', 'Simulated Substitutions')
            elif signature.startswith('DBS') or (signature == 'aggregateddinucs'):
                objects = ('Real Dinucs', 'Simulated Dinucs')
            elif signature.startswith('ID') or (signature == 'aggregatedindels'):
                objects = ('Real Indels', 'Simulated Indels')

        # add the legend
        if cosmic_legend:
            plt.legend(handles = listofLegends, labels=objects, bbox_to_anchor=(-0.3, 1.18), loc='upper left', borderaxespad=0, fontsize=fontsize/3, frameon=False)

    # This code puts some extra space below 0 and above 1.0
    plt.ylim(-0.01, 1.01)

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    if figure_type == COSMIC:
        plt.xlabel('Early <--- Replication Time ---> Late', fontsize = labelsize, fontweight = 'semibold')
        plt.ylabel('\nNormalized Mutation Density', fontsize = labelsize, fontweight = 'semibold')

    # if figure_type == MANUSCRIPT and (signature == AGGREGATEDSUBSTITUTIONS or signature == AGGREGATEDDINUCS or signature == AGGREGATEDINDELS):
    #     plt.xlabel('Early <- Replication Time -> Late', fontsize=labelsize-35, fontweight='semibold')
    #     plt.ylabel('\nNormalized\nmutation density', fontsize=labelsize-20, fontweight='semibold')

    if figure_type == COSMIC:
        # v3.2_SBS1_REPLIC_TIME_TA_C4817.jpg
        if cancer_type == None:
            figure_name = '%s_%s_%s.%s' % (cosmic_release_version, signature, COSMIC_REPLICATION_TIME, figure_file_extension)
        else:
            try:
                NCI_Thesaurus_code = cancer_type_2_NCI_Thesaurus_code_dict[cancer_type]
            except:
                print('For Your Information: %s has no NCI_Thesaurus_code' %(cancer_type))
                NCI_Thesaurus_code = cancer_type
            figure_name = '%s_%s_%s_TA_%s.%s' % (cosmic_release_version, signature, COSMIC_REPLICATION_TIME, NCI_Thesaurus_code, figure_file_extension)
    elif figure_type == MANUSCRIPT:
        if cancer_type == None:
            figure_name = '%s_replication_time.png' % (signature)
        else:
            figure_name = '%s_%s_replication_time.png' % (signature, cancer_type)

    # Using preset probabilities
    if figure_type==COSMIC and cancer_type==None:
        os.makedirs(os.path.join(output_dir, REPLICATION_TIME, FIGURES_COSMIC), exist_ok=True)
        figure_file = os.path.join(output_dir, REPLICATION_TIME, FIGURES_COSMIC, figure_name)
    elif figure_type == COSMIC and cancer_type != None:
        os.makedirs(os.path.join(output_dir, REPLICATION_TIME, COSMIC_TISSUE_BASED_FIGURES), exist_ok=True)
        figure_file = os.path.join(output_dir, REPLICATION_TIME, COSMIC_TISSUE_BASED_FIGURES, figure_name)
    elif figure_type==MANUSCRIPT:
        os.makedirs(os.path.join(output_dir, REPLICATION_TIME, FIGURES_MANUSCRIPT), exist_ok=True)
        figure_file = os.path.join(output_dir, REPLICATION_TIME, FIGURES_MANUSCRIPT, figure_name)

    fig.savefig(figure_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def get_normalized_array(number_of_mutations_array, number_of_attributable_bases_array):
    mutations_density_array  = np.divide(number_of_mutations_array,number_of_attributable_bases_array)
    max_density = np.nanmax(mutations_density_array)
    if max_density > 0:
        normalized_mutations_density_array = np.divide(mutations_density_array, max_density)
    else:
        normalized_mutations_density_array = np.nan

    return normalized_mutations_density_array


def get_normalized_mutation_density_array(mutation_density_array):
    max_density = np.nanmax(mutation_density_array)
    if max_density > 0:
        normalized_mutation_density_array = np.divide(mutation_density_array, max_density)
    else:
        normalized_mutation_density_array = np.nan

    return normalized_mutation_density_array


def non_increasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

def monotonic(L):
    return non_increasing(L) or non_decreasing(L)

def calculate_sims_lows_means_highs(across_all_cancer_types_all_sims_2d_array):
    (rows, cols) = across_all_cancer_types_all_sims_2d_array.shape
    simulations_lows = []
    simulations_highs = []

    # Column-wise mean
    simulations_means = np.nanmean(across_all_cancer_types_all_sims_2d_array, axis=0)

    for col in range(cols):
        colwise_array = across_all_cancer_types_all_sims_2d_array[:, col]
        mu = np.nanmean(colwise_array)

        # Standard deviation of the sample is the degree to which individuals within the sample differ from the sample mean
        sigma = np.nanstd(colwise_array)
        start, end = scipy.stats.norm.interval(0.95, loc = mu, scale = sigma/np.sqrt(len(colwise_array)))

        # Standard error of the sample mean is an estimate of how far the sample mean is likely to be from the population mean
        # sigma = scipy.stats.sem(colwise_array) # standard error of the mean
        # start, end = scipy.stats.norm.interval(alpha=0.95, loc=mu, scale=sigma)

        simulations_lows.append(start)
        simulations_highs.append(end)

        # # Below statements resulted in hairy figures for some DBS and ID signatures
        # signal_min = np.nanmin(colwise_array)
        # signal_max = np.nanmax(colwise_array)
        # simulations_lows_list.append(signal_min)
        # simulations_highs_list.append(signal_max)

    return simulations_lows, simulations_means, simulations_highs


# normalized mutation density will be vertically stacked, column-wise mean will be computed and normalized again.
# For the given signature, accumulate replication timing across cancer types having this signature
def accumulate_replication_time_across_all_cancer_types_plot_figure(plot_output_dir,
    main_combined_output_dir,
    signature,
    signature_type,
    cancer_types,
    numberofSimulations,
    figure_type,
    number_of_mutations_required,
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
    sub_figure_type):

    signature_cancer_type_replication_time_df = pd.DataFrame(columns=["signature",
                    "cancer_type",
                    "real_number_of_mutations_array", # shape (1,10)
                    "real_normalized_mutations_density_array",  # shape (1,10)
                    "simulations_means_normalized_mutations_density_array",  # shape (1,10)
                    "spearman_corr",
                    "spearman_p_value",
                    "spearman_q_value",
                    "pearson_corr",
                    "pearson_p_value",
                    "pearson_q_value",
                    "cancer_type_slope",
                    "replication_timing_diff_btw_max_and_min",
                    "abs_replication_timing_diff_btw_medians",
                    "cancer_type_decision",
                    "cutoff",
                    "number_of_mutations",
                    "average_probability",
                    "num_of_tissues_with_spearman_ge_cutoff",
                    "num_of_spearman_q_values_le_significance_level",
                    "num_of_tissues_with_pearson_corr_ge_cutoff",
                    "num_of_tissues_with_pearson_q_value_le_significance_level",
                    "num_of_tissues_with_increasing_decision",
                    "num_of_tissues_with_flat_decision",
                    "num_of_tissues_with_decreasing_decision",
                    "num_of_all_tissues"])

    spearman_p_values = []
    spearman_element_names = []
    pearson_p_values = []
    pearson_element_names = []

    num_of_tissues_with_pearson_corr_ge_cutoff = 0
    num_of_tissues_with_pearson_q_value_le_significance_level = 0
    num_of_tissues_with_spearman_ge_cutoff = 0
    num_of_spearman_q_values_le_significance_level = 0

    num_of_all_tissues = 0

    num_of_tissues_with_slope_increasing = 0
    num_of_tissues_with_slope_flat = 0
    num_of_tissues_with_slope_decreasing = 0

    tissues_with_slope_flat = []
    tissues_with_slope_increasing = []
    tissues_with_slope_decreasing = []

    # Vertically Stack Version
    across_all_cancer_types_real_normalized_mutation_density_array = np.array([]) # vertically stacked
    across_all_cancer_types_all_simulations_normalized_mutation_density_array = np.array([]) # vertically stacked

    if (signature == AGGREGATEDSUBSTITUTIONS) or (signature == AGGREGATEDINDELS) or (signature == AGGREGATEDDINUCS):
        sub_dir = signature
    else:
        sub_dir = SIGNATUREBASED

    # First Pass
    # Consider signatures with at least number_of_mutations_required per cancer type
    # Accumulate across cancer types starts
    for main_cancer_type in cancer_types:

        # For Lymph-BNHL based on signature we may use either Lymph-BNHL_clustered, Lymph-BNHL_nonClustered or Lymph-BNHL
        # For Lymph-CLL based on signature we may use either Lymph-CLL_clustered, Lymph-CLL_nonClustered or Lymph-CLL
        # Use combined_output_dir and cancer_type returned by get_alternative_combined_output_dir_and_cancer_type while reading files
        # Use main_cancer_type while accumulating results in pandas dataframe or calculating p-values
        combined_output_dir, cancer_type = get_alternative_combined_output_dir_and_cancer_type(main_combined_output_dir, main_cancer_type, signature)

        # read cancer_type signature number_of_mutations
        # consider cancer_type signature with at least 10K number_of_mutations
        # consider cancer_type signature with at least 10K number_of_mutations
        # Table_SBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt
        # Table_DBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt
        # Table_ID_Signature_Cutoff_NumberofMutations_AverageProbability.txt
        # cancer_type     signature       cutoff  number_of_mutations     average_probability
        # Liver-HCC       SBS1    0.59    23647   0.750195729604971

        number_of_mutations = 0
        if (signature == AGGREGATEDSUBSTITUTIONS) or (signature == AGGREGATEDINDELS) or (signature == AGGREGATEDDINUCS):
            number_of_mutations_filename = TABLE_MUTATIONTYPE_NUMBEROFMUTATIONS_NUMBEROFSAMPLES_SAMPLESLIST
            number_of_mutations_filepath = os.path.join(combined_output_dir,cancer_type,DATA,number_of_mutations_filename)
            number_of_mutations_df = pd.read_csv(number_of_mutations_filepath, header=0, sep='\t')
            # mutation_type   number_of_mutations     number_of_samples       samples_list
            if signature == AGGREGATEDSUBSTITUTIONS:
                if np.any(number_of_mutations_df[number_of_mutations_df['mutation_type'] == SUBS]['number_of_mutations'].values):
                    number_of_mutations = number_of_mutations_df[number_of_mutations_df['mutation_type']==SUBS]['number_of_mutations'].values[0]
            elif signature == AGGREGATEDDINUCS:
                if np.any(number_of_mutations_df[number_of_mutations_df['mutation_type'] == DINUCS]['number_of_mutations'].values):
                    number_of_mutations = number_of_mutations_df[number_of_mutations_df['mutation_type']==DINUCS]['number_of_mutations'].values[0]
            elif signature == AGGREGATEDINDELS:
                if np.any(number_of_mutations_df[number_of_mutations_df['mutation_type'] == INDELS]['number_of_mutations'].values):
                    number_of_mutations = number_of_mutations_df[number_of_mutations_df['mutation_type']==INDELS]['number_of_mutations'].values[0]
        else:
            number_of_mutations_filename = get_number_of_mutations_filename(signature_type)
            number_of_mutations_filepath = os.path.join(combined_output_dir,cancer_type,DATA,number_of_mutations_filename)
            number_of_mutations_df = pd.read_csv(number_of_mutations_filepath, header=0, sep='\t')
            if np.any(number_of_mutations_df[number_of_mutations_df['signature'] == signature]['number_of_mutations'].values):
                number_of_mutations = number_of_mutations_df[number_of_mutations_df['signature']==signature]['number_of_mutations'].values[0]

        if number_of_mutations >= number_of_mutations_required:
            # initialized for each cancer type
            all_simulations_number_of_mutations_array = np.array([])  # vertically stacked
            all_simulations_number_of_attributable_bases_array = np.array([])  # vertically stacked
            all_simulations_normalized_mutation_density_array = np.array([])  # vertically stacked

            typebased_tissuebased_replication_time_number_of_mutations_filename = "%s_NumberofMutations.txt" %(signature)
            replication_time_number_of_attributable_bases_filename = "NumberofAttributableBases.txt"
            typebased_tissuebased_replication_time_normalized_mutation_density_filename = "%s_NormalizedMutationDensity.txt" % (signature)

            typebased_tissuebased_replication_time_number_of_mutations_filepath = os.path.join(combined_output_dir,cancer_type,DATA,REPLICATION_TIME,sub_dir,typebased_tissuebased_replication_time_number_of_mutations_filename)
            replication_time_number_of_attributable_bases_filepath = os.path.join(combined_output_dir,cancer_type,DATA,REPLICATION_TIME,replication_time_number_of_attributable_bases_filename)
            typebased_tissuebased_replication_time_normalized_mutation_densisty_filepath = os.path.join(combined_output_dir, cancer_type, DATA, REPLICATION_TIME,sub_dir,typebased_tissuebased_replication_time_normalized_mutation_density_filename)

            if (os.path.exists(typebased_tissuebased_replication_time_number_of_mutations_filepath) and
                    os.path.exists(replication_time_number_of_attributable_bases_filepath) and
                    os.path.exists(typebased_tissuebased_replication_time_normalized_mutation_densisty_filepath)):

                real_number_of_mutations_array = np.loadtxt(typebased_tissuebased_replication_time_number_of_mutations_filepath, dtype = np.float64) .astype(int) # dtype = np.int64
                real_normalized_mutation_density_array = np.loadtxt(typebased_tissuebased_replication_time_normalized_mutation_densisty_filepath, dtype=np.float64)

                if np.sum(real_number_of_mutations_array) < 2000:
                    print('FOR INFORMATION1 less than 2K', cancer_type, signature, np.sum(real_number_of_mutations_array),'real_number_of_mutations_array: ', real_number_of_mutations_array)
                    print('FOR INFORMATION1 less than 2K', cancer_type, signature, np.sum(real_normalized_mutation_density_array),'real_normalized_mutation_density_array: ', real_normalized_mutation_density_array)

                # Please note that nparray_number_of_attributable_bases is replication time dependent
                # Therefore same for all signatures, same for all real data and simulations
                # vertically stack real_normalized_mutation_density_array coming from each cancer type
                if np.any(real_number_of_mutations_array):
                    if across_all_cancer_types_real_normalized_mutation_density_array.size == 0:
                        across_all_cancer_types_real_normalized_mutation_density_array = np.expand_dims(real_normalized_mutation_density_array, axis=0)
                    else:
                        across_all_cancer_types_real_normalized_mutation_density_array = np.vstack((across_all_cancer_types_real_normalized_mutation_density_array,
                                                                                                    real_normalized_mutation_density_array))
                    num_of_all_tissues += 1

                for simNum in range(1,numberofSimulations+1):
                    typebased_tissuebased_simNumbased_replication_time_number_of_mutations_filename = "%s_sim%d_NumberofMutations.txt" % (signature,simNum)
                    replication_time_number_of_attributable_bases_filename = "NumberofAttributableBases.txt"
                    typebased_tissuebased_simNumbased_replication_time_normalized_mutation_density_filename = "%s_sim%d_NormalizedMutationDensity.txt" % (signature,simNum)

                    typebased_tissuebased_simNumbased_replication_time_number_of_mutations_filepath = os.path.join(combined_output_dir,cancer_type,DATA,REPLICATION_TIME,sub_dir,typebased_tissuebased_simNumbased_replication_time_number_of_mutations_filename)
                    replication_time_number_of_attributable_bases_filepath = os.path.join(combined_output_dir,cancer_type,DATA,REPLICATION_TIME,replication_time_number_of_attributable_bases_filename)
                    typebased_tissuebased_simNumbased_replication_time_normalized_mutation_density_filepath = os.path.join(combined_output_dir, cancer_type, DATA, REPLICATION_TIME, sub_dir,typebased_tissuebased_simNumbased_replication_time_normalized_mutation_density_filename)

                    if os.path.exists(typebased_tissuebased_simNumbased_replication_time_number_of_mutations_filepath):
                        simulation_number_of_mutations_array = np.loadtxt(typebased_tissuebased_simNumbased_replication_time_number_of_mutations_filepath, dtype = np.float64).astype(int) # dtype = np.int64
                        if all_simulations_number_of_mutations_array.size == 0:
                            all_simulations_number_of_mutations_array = simulation_number_of_mutations_array
                        else:
                            all_simulations_number_of_mutations_array = np.vstack((all_simulations_number_of_mutations_array, simulation_number_of_mutations_array))

                    if os.path.exists(replication_time_number_of_attributable_bases_filepath):
                        number_of_attributable_bases_array = np.loadtxt(replication_time_number_of_attributable_bases_filepath, dtype=np.int64)
                        if all_simulations_number_of_attributable_bases_array.size == 0:
                            all_simulations_number_of_attributable_bases_array = number_of_attributable_bases_array
                        else:
                            all_simulations_number_of_attributable_bases_array = np.vstack((all_simulations_number_of_attributable_bases_array,number_of_attributable_bases_array))

                    if os.path.exists(typebased_tissuebased_simNumbased_replication_time_normalized_mutation_density_filepath):
                        simulation_normalized_mutation_density_array = np.loadtxt(typebased_tissuebased_simNumbased_replication_time_normalized_mutation_density_filepath, dtype=np.float64)
                        if all_simulations_normalized_mutation_density_array.size == 0:
                            all_simulations_normalized_mutation_density_array = simulation_normalized_mutation_density_array
                        else:
                            all_simulations_normalized_mutation_density_array = np.vstack((all_simulations_normalized_mutation_density_array, simulation_normalized_mutation_density_array))


                # number of rows: 100 (number_of_simulations) number of cols 10
                simulations_lows, simulations_means, simulations_highs = calculate_sims_lows_means_highs(all_simulations_normalized_mutation_density_array)

                # Append row to dataframe for signature based cancer type based
                signature_cancer_type_replication_time_df = signature_cancer_type_replication_time_df.append(
                    {"signature" : signature,
                    "cancer_type" : main_cancer_type, # Accumulate under main_cancer_type, formerly it was cancer_type
                    "real_number_of_mutations_array" : real_number_of_mutations_array.tolist(),
                    "real_normalized_mutations_density_array" : np.around(real_normalized_mutation_density_array,NUMBER_OF_DECIMAL_PLACES_TO_ROUND).tolist(),
                    "simulations_means_normalized_mutations_density_array" : np.around(simulations_means,NUMBER_OF_DECIMAL_PLACES_TO_ROUND).tolist(),
                    "spearman_corr" : np.nan,
                    "spearman_p_value" : np.nan,
                    "spearman_q_value" : np.nan,
                    "pearson_corr" : np.nan,
                    "pearson_p_value" : np.nan,
                    "pearson_q_value" : np.nan,
                    "cancer_type_slope" : np.nan,
                    "replication_timing_diff_btw_max_and_min" : np.nan,
                    "abs_replication_timing_diff_btw_medians" : np.nan,
                    "cancer_type_decision" : np.nan,
                    "cutoff": np.nan,
                    "number_of_mutations" : np.nan,
                    "average_probability" : np.nan,
                    "num_of_tissues_with_spearman_ge_cutoff" : np.nan,
                    "num_of_spearman_q_values_le_significance_level" : np.nan,
                    "num_of_tissues_with_pearson_corr_ge_cutoff" : np.nan,
                    "num_of_tissues_with_pearson_q_value_le_significance_level" : np.nan,
                    "num_of_tissues_with_increasing_decision" : np.nan,
                    "num_of_tissues_with_flat_decision" : np.nan,
                    "num_of_tissues_with_decreasing_decision" : np.nan,
                    "num_of_all_tissues" : np.nan}, ignore_index=True)

                # Plot Cosmic Cancer type based figure in First Pass
                plot_replication_time_figure(plot_output_dir,
                                            real_normalized_mutation_density_array,
                                            signature,
                                            simulations_lows,
                                            simulations_means,
                                            simulations_highs,
                                            0,
                                            COSMIC,
                                            cosmic_release_version,
                                            figure_file_extension,
                                            cancer_type = main_cancer_type, # formerly it was cancer_type
                                            num_of_increasing = 0,
                                            num_of_flat = 0,
                                            num_of_decreasing = 0,
                                            cosmic_legend = cosmic_legend,
                                            cosmic_signature = cosmic_signature,
                                            cosmic_fontsize_text = cosmic_fontsize_text,
                                            cosmic_cancer_type_fontsize = cosmic_cancer_type_fontsize,
                                            cosmic_fontweight = cosmic_fontweight,
                                            cosmic_fontsize_labels = cosmic_fontsize_labels,
                                            sub_figure_type = sub_figure_type)


            # Vertically stack across cancer types
            if across_all_cancer_types_all_simulations_normalized_mutation_density_array.size == 0:
                across_all_cancer_types_all_simulations_normalized_mutation_density_array = all_simulations_normalized_mutation_density_array
            elif all_simulations_normalized_mutation_density_array.size > 0:
                across_all_cancer_types_all_simulations_normalized_mutation_density_array = np.vstack((across_all_cancer_types_all_simulations_normalized_mutation_density_array,
                                                                                                       all_simulations_normalized_mutation_density_array))

        else:
            print('FOR INFORMATION', cancer_type, signature, 'has', number_of_mutations,'mutations less than', number_of_mutations_required)

    # calculate across all cancer types real normalized mutation density
    # First column-wise mean
    # Second re-normalize
    across_all_cancer_types_real_normalized_mutation_density_array = np.nanmean(across_all_cancer_types_real_normalized_mutation_density_array, axis=0)
    across_all_cancer_types_real_normalized_mutation_density_array = get_normalized_mutation_density_array(across_all_cancer_types_real_normalized_mutation_density_array)

    # Second Pass
    # Calculate pearson correlation
    # We compare cancer_type based result with aggregated result
    for main_cancer_type in cancer_types:

        # For Lymph-BNHL based on signature we may use either Lymph-BNHL_clustered, Lymph-BNHL_nonClustered or Lymph-BNHL
        # For Lymph-CLL based on signature we may use either Lymph-CLL_clustered, Lymph-CLL_nonClustered or Lymph-CLL
        # Use combined_output_dir and cancer_type returned by get_alternative_combined_output_dir_and_cancer_type while reading files
        # Use main_cancer_type while accumulating results in pandas dataframe or calculating p-values
        combined_output_dir, cancer_type = get_alternative_combined_output_dir_and_cancer_type(main_combined_output_dir, main_cancer_type, signature)

        number_of_mutations = None
        average_probability = None
        cutoff = None
        if (signature==AGGREGATEDSUBSTITUTIONS) or (signature==AGGREGATEDINDELS) or (signature==AGGREGATEDDINUCS):
            number_of_mutations_filename = TABLE_MUTATIONTYPE_NUMBEROFMUTATIONS_NUMBEROFSAMPLES_SAMPLESLIST
            number_of_mutations_filepath = os.path.join(combined_output_dir, cancer_type, DATA, number_of_mutations_filename)
            number_of_mutations_df = pd.read_csv(number_of_mutations_filepath, header=0, sep='\t')
            # mutation_type   number_of_mutations     number_of_samples       samples_list
            if signature==AGGREGATEDSUBSTITUTIONS:
                if np.any(number_of_mutations_df[number_of_mutations_df['mutation_type'] == SUBS]['number_of_mutations'].values):
                    number_of_mutations = number_of_mutations_df[number_of_mutations_df['mutation_type']==SUBS]['number_of_mutations'].values[0]
            elif signature==AGGREGATEDDINUCS:
                if np.any(number_of_mutations_df[number_of_mutations_df['mutation_type'] == DINUCS]['number_of_mutations'].values):
                    number_of_mutations = number_of_mutations_df[number_of_mutations_df['mutation_type']==DINUCS]['number_of_mutations'].values[0]
            elif signature==AGGREGATEDINDELS:
                if np.any(number_of_mutations_df[number_of_mutations_df['mutation_type'] == INDELS]['number_of_mutations'].values):
                    number_of_mutations = number_of_mutations_df[number_of_mutations_df['mutation_type']==INDELS]['number_of_mutations'].values[0]

        else:
            number_of_mutations_filename = get_number_of_mutations_filename(signature_type)
            number_of_mutations_filepath = os.path.join(combined_output_dir,cancer_type,DATA,number_of_mutations_filename)
            number_of_mutations_df = pd.read_csv(number_of_mutations_filepath, header=0, sep='\t')
            if np.any(number_of_mutations_df[number_of_mutations_df['signature'] == signature]['number_of_mutations'].values):
                number_of_mutations = number_of_mutations_df[number_of_mutations_df['signature']==signature]['number_of_mutations'].values[0]
            if np.any(number_of_mutations_df[number_of_mutations_df['signature'] == signature]['average_probability'].values):
                average_probability = number_of_mutations_df[number_of_mutations_df['signature']==signature]['average_probability'].values[0]
            if np.any(number_of_mutations_df[number_of_mutations_df['signature'] == signature]['cutoff'].values):
                cutoff = float(number_of_mutations_df[number_of_mutations_df['signature'] == signature]['cutoff'].values[0])

        if (number_of_mutations is not None) and (number_of_mutations >= number_of_mutations_required):
            # For information
            typebased_tissuebased_replication_time_number_of_mutations_filename = "%s_NumberofMutations.txt" %(signature)
            typebased_tissuebased_replication_time_number_of_mutations_filepath = os.path.join(combined_output_dir,cancer_type,DATA,REPLICATION_TIME,sub_dir,typebased_tissuebased_replication_time_number_of_mutations_filename)

            typebased_tissuebased_replication_time_normalized_mutation_density_filename = "%s_NormalizedMutationDensity.txt" % (signature)
            typebased_tissuebased_replication_time_normalized_mutation_densisty_filepath = os.path.join(combined_output_dir, cancer_type, DATA, REPLICATION_TIME,sub_dir,typebased_tissuebased_replication_time_normalized_mutation_density_filename)

            if (os.path.exists(typebased_tissuebased_replication_time_normalized_mutation_densisty_filepath)):
                signature_cancer_type_normalized_mutation_density_array = np.loadtxt(typebased_tissuebased_replication_time_normalized_mutation_densisty_filepath, dtype=np.float64)

                if (np.any(signature_cancer_type_normalized_mutation_density_array)) and (np.any(across_all_cancer_types_real_normalized_mutation_density_array)):
                    spearman_corr, spearman_p_value = spearmanr(signature_cancer_type_normalized_mutation_density_array,
                                                                across_all_cancer_types_real_normalized_mutation_density_array)

                    x = np.array([0, 1, 2, 3, 4, 5, 6, 7 ,8, 9])
                    cancer_type_y = signature_cancer_type_normalized_mutation_density_array

                    cancer_type_m, cancer_type_b = pol = np.polyfit(x, cancer_type_y, 1)

                    # Internally set
                    # Case1
                    # Positive slope and slope greater than cutoff
                    if cancer_type_m > replication_time_slope_cutoff and monotonic(cancer_type_y):
                        num_of_tissues_with_slope_increasing += 1
                        tissues_with_slope_increasing.append(main_cancer_type) # cancer_type
                        decision = INCREASING
                        replication_timing_diff_btw_max_and_min = (np.max(cancer_type_y) - np.min(cancer_type_y))
                        abs_replication_timing_diff_btw_medians = abs(np.median(cancer_type_y[0:3]) - np.median(cancer_type_y[7:]))
                    # Case2
                    elif cancer_type_m > replication_time_slope_cutoff and \
                            (np.max(cancer_type_y) - np.min(cancer_type_y)) > replication_time_difference_between_min_and_max:
                        num_of_tissues_with_slope_increasing += 1
                        tissues_with_slope_increasing.append(main_cancer_type) # cancer_type
                        decision = INCREASING
                        replication_timing_diff_btw_max_and_min = (np.max(cancer_type_y) - np.min(cancer_type_y))
                        abs_replication_timing_diff_btw_medians = abs(np.median(cancer_type_y[0:3]) - np.median(cancer_type_y[7:]))
                    # Case3
                    # Negative Slope and abs(slope) greater than cutoff
                    elif cancer_type_m < 0 and abs(cancer_type_m) > replication_time_slope_cutoff \
                            and monotonic(cancer_type_y):
                        num_of_tissues_with_slope_decreasing += 1
                        tissues_with_slope_decreasing.append(main_cancer_type) #cancer_type
                        decision = DECREASING
                        replication_timing_diff_btw_max_and_min = (np.max(cancer_type_y) - np.min(cancer_type_y))
                        abs_replication_timing_diff_btw_medians = abs(np.median(cancer_type_y[0:3]) - np.median(cancer_type_y[7:]))
                    # Case4
                    elif cancer_type_m < 0 and abs(cancer_type_m) > replication_time_slope_cutoff \
                            and (np.max(cancer_type_y) - np.min(cancer_type_y)) > replication_time_difference_between_min_and_max:
                        num_of_tissues_with_slope_decreasing += 1
                        tissues_with_slope_decreasing.append(main_cancer_type)  # cancer_type
                        decision = DECREASING
                        replication_timing_diff_btw_max_and_min = (np.max(cancer_type_y) - np.min(cancer_type_y))
                        abs_replication_timing_diff_btw_medians = abs(np.median(cancer_type_y[0:3]) - np.median(cancer_type_y[7:]))
                    # Case5
                    # Slope less than cutoff
                    elif abs(cancer_type_m) <= replication_time_slope_cutoff:
                        num_of_tissues_with_slope_flat += 1
                        tissues_with_slope_flat.append(main_cancer_type) # cancer_type
                        decision = FLAT
                        replication_timing_diff_btw_max_and_min = (np.max(cancer_type_y) - np.min(cancer_type_y))
                        abs_replication_timing_diff_btw_medians = abs(np.median(cancer_type_y[0:3]) - np.median(cancer_type_y[7:]))
                    # Case6
                    elif abs(cancer_type_m) > replication_time_slope_cutoff \
                            and (np.max(cancer_type_y)-np.min(cancer_type_y)) <= replication_time_difference_between_min_and_max:
                        num_of_tissues_with_slope_flat += 1
                        tissues_with_slope_flat.append(main_cancer_type) # cancer_type
                        decision = FLAT
                        replication_timing_diff_btw_max_and_min = (np.max(cancer_type_y) - np.min(cancer_type_y))
                        abs_replication_timing_diff_btw_medians = abs(np.median(cancer_type_y[0:3]) - np.median(cancer_type_y[7:]))
                    elif abs(cancer_type_m) > replication_time_slope_cutoff \
                            and (abs(np.median(cancer_type_y[0:3]) - np.median(cancer_type_y[7:])) <= replication_time_difference_between_medians):
                        num_of_tissues_with_slope_flat += 1
                        tissues_with_slope_flat.append(main_cancer_type) # cancer_type
                        decision = FLAT
                        replication_timing_diff_btw_max_and_min = (np.max(cancer_type_y) - np.min(cancer_type_y))
                        abs_replication_timing_diff_btw_medians = abs(np.median(cancer_type_y[0:3]) - np.median(cancer_type_y[7:]))
                    else:
                        decision = UNKNOWN
                        replication_timing_diff_btw_max_and_min = np.nan
                        abs_replication_timing_diff_btw_medians = np.nan
                        print('FOR INFORMATION', signature, cancer_type, decision)

                    pearson_corr, pearson_p_value = pearsonr(signature_cancer_type_normalized_mutation_density_array,
                                                             across_all_cancer_types_real_normalized_mutation_density_array)

                    if (spearman_corr >= pearson_spearman_correlation_cutoff):
                        num_of_tissues_with_spearman_ge_cutoff += 1
                        spearman_p_values.append(spearman_p_value)
                        spearman_element_names.append((signature,main_cancer_type)) # cancer_type

                    if (pearson_corr >= pearson_spearman_correlation_cutoff):
                        num_of_tissues_with_pearson_corr_ge_cutoff += 1
                        pearson_p_values.append(pearson_p_value)
                        pearson_element_names.append((signature,main_cancer_type)) # cancer_type

                    # Cancer type
                    if signature_cancer_type_replication_time_df[(signature_cancer_type_replication_time_df['signature'] == signature) &
                                                              (signature_cancer_type_replication_time_df['cancer_type'] == main_cancer_type)].values.any(): # cancer_type

                        signature_cancer_type_replication_time_df.loc[((signature_cancer_type_replication_time_df['signature'] == signature) &
                                    (signature_cancer_type_replication_time_df['cancer_type'] == main_cancer_type)), 'spearman_corr' ] = spearman_corr # cancer_type

                        signature_cancer_type_replication_time_df.loc[((signature_cancer_type_replication_time_df['signature'] == signature) &
                                    (signature_cancer_type_replication_time_df['cancer_type'] == main_cancer_type)), 'spearman_p_value' ] = spearman_p_value # cancer_type

                        signature_cancer_type_replication_time_df.loc[((signature_cancer_type_replication_time_df['signature'] == signature) &
                                    (signature_cancer_type_replication_time_df['cancer_type'] == main_cancer_type)), 'pearson_corr' ] = pearson_corr # cancer_type

                        signature_cancer_type_replication_time_df.loc[((signature_cancer_type_replication_time_df['signature'] == signature) &
                                    (signature_cancer_type_replication_time_df['cancer_type'] == main_cancer_type)), 'pearson_p_value' ] = pearson_p_value # cancer_type

                        signature_cancer_type_replication_time_df.loc[((signature_cancer_type_replication_time_df['signature'] == signature) &
                                    (signature_cancer_type_replication_time_df['cancer_type'] == main_cancer_type)), 'cancer_type_slope' ] = np.around(cancer_type_m, NUMBER_OF_DECIMAL_PLACES_TO_ROUND) if cancer_type_m is not None else np.nan # cancer_type

                        signature_cancer_type_replication_time_df.loc[((signature_cancer_type_replication_time_df['signature'] == signature) &
                                    (signature_cancer_type_replication_time_df['cancer_type'] == main_cancer_type)), 'replication_timing_diff_btw_max_and_min' ] = np.around(replication_timing_diff_btw_max_and_min, NUMBER_OF_DECIMAL_PLACES_TO_ROUND) if replication_timing_diff_btw_max_and_min is not None else np.nan # cancer_type

                        signature_cancer_type_replication_time_df.loc[((signature_cancer_type_replication_time_df['signature'] == signature) &
                                    (signature_cancer_type_replication_time_df['cancer_type'] == main_cancer_type)), 'abs_replication_timing_diff_btw_medians' ] = np.around(abs_replication_timing_diff_btw_medians, NUMBER_OF_DECIMAL_PLACES_TO_ROUND) if abs_replication_timing_diff_btw_medians is not None else np.nan # cancer_type

                        signature_cancer_type_replication_time_df.loc[((signature_cancer_type_replication_time_df['signature'] == signature) &
                                    (signature_cancer_type_replication_time_df['cancer_type'] == main_cancer_type)), 'cancer_type_decision' ] = decision # cancer_type

                        signature_cancer_type_replication_time_df.loc[((signature_cancer_type_replication_time_df['signature'] == signature) &
                                    (signature_cancer_type_replication_time_df['cancer_type'] == main_cancer_type)), 'cutoff'] = cutoff # cancer_type

                        signature_cancer_type_replication_time_df.loc[((signature_cancer_type_replication_time_df['signature'] == signature) &
                                    (signature_cancer_type_replication_time_df['cancer_type'] == main_cancer_type)), 'number_of_mutations'] = number_of_mutations # cancer_type

                        signature_cancer_type_replication_time_df.loc[((signature_cancer_type_replication_time_df['signature'] == signature) &
                                    (signature_cancer_type_replication_time_df['cancer_type'] == main_cancer_type)), 'average_probability'] = np.around(average_probability, NUMBER_OF_DECIMAL_PLACES_TO_ROUND) if average_probability is not None else np.nan # cancer_type



    # Signature Based Across All Cancer Types
    if (np.any(across_all_cancer_types_real_normalized_mutation_density_array)):
        # Correct pvalues
        # Display only q-values < SIGNIFICANCE_LEVEL
        pearson_p_values_array = np.asarray(pearson_p_values)
        spearman_p_values_array = np.asarray(spearman_p_values)

        if (spearman_p_values_array.size > 0):
            rejected, spearman_corrected_p_values, alphacSidak, alphacBonf = statsmodels.stats.multitest.multipletests(spearman_p_values_array, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
            num_of_spearman_q_values_le_significance_level=len(np.argwhere(spearman_corrected_p_values <= replication_time_significance_level))

        if (pearson_p_values_array.size > 0):
            rejected, pearson_corrected_p_values, alphacSidak, alphacBonf = statsmodels.stats.multitest.multipletests(pearson_p_values_array, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
            num_of_tissues_with_pearson_q_value_le_significance_level=len(np.argwhere(pearson_corrected_p_values <= replication_time_significance_level))

        for element_index, element_name in enumerate(spearman_element_names,0):
            (signature, cancer_type) = element_name
            spearman_q_value = spearman_corrected_p_values[element_index]

            if signature_cancer_type_replication_time_df[
                (signature_cancer_type_replication_time_df['signature'] == signature) &
                (signature_cancer_type_replication_time_df['cancer_type'] == cancer_type)].values.any():
                signature_cancer_type_replication_time_df.loc[
                    ((signature_cancer_type_replication_time_df['signature'] == signature) &
                     (signature_cancer_type_replication_time_df['cancer_type'] == cancer_type)), 'spearman_q_value'] = spearman_q_value

        for element_index, element_name in enumerate(pearson_element_names,0):
            (signature, cancer_type)= element_name
            pearson_q_value = pearson_corrected_p_values[element_index]

            if signature_cancer_type_replication_time_df[
                (signature_cancer_type_replication_time_df['signature'] == signature) &
                (signature_cancer_type_replication_time_df['cancer_type'] == cancer_type)].values.any():
                signature_cancer_type_replication_time_df.loc[
                    ((signature_cancer_type_replication_time_df['signature'] == signature) &
                     (signature_cancer_type_replication_time_df['cancer_type'] == cancer_type)), 'pearson_q_value'] = pearson_q_value

        simulations_lows, \
        simulations_means, \
        simulations_highs = calculate_sims_lows_means_highs(across_all_cancer_types_all_simulations_normalized_mutation_density_array)

        # Across all cancer types
        signature_cancer_type_replication_time_df = signature_cancer_type_replication_time_df.append(
            {"signature" : signature,
            "cancer_type" : ACROSS_ALL_CANCER_TYPES,
            "real_number_of_mutations_array" : np.nan,
            "real_normalized_mutations_density_array" : np.around(across_all_cancer_types_real_normalized_mutation_density_array, NUMBER_OF_DECIMAL_PLACES_TO_ROUND).tolist(),
            "simulations_means_normalized_mutations_density_array": np.around(simulations_means, NUMBER_OF_DECIMAL_PLACES_TO_ROUND).tolist(),
            "spearman_corr" : np.nan,
            "spearman_p_value" : np.nan,
            "spearman_q_value" : np.nan,
            "pearson_corr" : np.nan,
            "pearson_p_value" : np.nan,
            "pearson_q_value" : np.nan,
            "cancer_type_slope" : np.nan,
            "replication_timing_diff_btw_max_and_min" : np.nan,
            "abs_replication_timing_diff_btw_medians": np.nan,
            "cancer_type_decision"  : np.nan,
            "cutoff": np.nan,
            "number_of_mutations" : np.nan,
            "average_probability" : np.nan,
            "num_of_tissues_with_spearman_ge_cutoff" : num_of_tissues_with_spearman_ge_cutoff,
            "num_of_spearman_q_values_le_significance_level" : num_of_spearman_q_values_le_significance_level,
            "num_of_tissues_with_pearson_corr_ge_cutoff" : num_of_tissues_with_pearson_corr_ge_cutoff,
            "num_of_tissues_with_pearson_q_value_le_significance_level" : num_of_tissues_with_pearson_q_value_le_significance_level,
            "num_of_tissues_with_increasing_decision" : num_of_tissues_with_slope_increasing,
            "num_of_tissues_with_flat_decision" : num_of_tissues_with_slope_flat,
            "num_of_tissues_with_decreasing_decision" : num_of_tissues_with_slope_decreasing,
            "num_of_all_tissues" : num_of_all_tissues}, ignore_index=True)

        # COSMIC
        # MANUSCRIPT
        # Plot the accumulated normalized mutation density figure
        # Please note that we use num_of_tissues_with_pearson_q_value_le_significance_level/num_of_all_tissues in the figures
        plot_replication_time_figure(plot_output_dir,
                                across_all_cancer_types_real_normalized_mutation_density_array,
                                signature,
                                simulations_lows,
                                simulations_means,
                                simulations_highs,
                                num_of_all_tissues,
                                figure_type,
                                cosmic_release_version,
                                figure_file_extension,
                                num_of_increasing = num_of_tissues_with_slope_increasing,
                                num_of_flat = num_of_tissues_with_slope_flat,
                                num_of_decreasing = num_of_tissues_with_slope_decreasing,
                                cosmic_legend = cosmic_legend,
                                cosmic_signature = cosmic_signature,
                                cosmic_fontsize_text = cosmic_fontsize_text,
                                cosmic_cancer_type_fontsize = cosmic_cancer_type_fontsize,
                                cosmic_fontweight = cosmic_fontweight,
                                cosmic_fontsize_labels = cosmic_fontsize_labels,
                                sub_figure_type = sub_figure_type)

    print('INFORMATION %s num_inc: %d -- cancer_types_inc: %s' %(signature, num_of_tissues_with_slope_increasing, tissues_with_slope_increasing))
    print('INFORMATION %s num_flat: %d -- cancer_types_flat: %s' % (signature, num_of_tissues_with_slope_flat, tissues_with_slope_flat))
    print('INFORMATION %s num_dec: %d -- cancer_types_dec: %s' % (signature, num_of_tissues_with_slope_decreasing, tissues_with_slope_decreasing))

    return  signature_cancer_type_replication_time_df


def fill_replication_time_figures_pdfs(plot_output_dir,
                                       combined_output_dir,
                                       signature_tuples,
                                       cancer_types,
                                       all_signatures_replication_time_df):
    # Replication Time
    path= os.path.join(plot_output_dir, REPLICATION_TIME)

    # For each type create a pdf
    # In each pdf, one figure for across all tissues and one figure for each tissue
    for (signature, signature_type) in signature_tuples:
        interested_file_list = []

        # type across_all_tissues
        figurename = '%s_replication_time.png' % (signature)
        filepath = os.path.join(path, FIGURES_MANUSCRIPT, figurename)

        if os.path.exists(filepath):
            interested_file_list.append(filepath)

        signature_with_underscore = signature + '_'

        for cancer_type in cancer_types:
            # tissue
            cancer_type_output_path = os.path.join(combined_output_dir,cancer_type, FIGURE, REPLICATION_TIME)

            # There can be a replication time figure for number of mutations all zeros.
            # This check is to avoid displaying empty replication time figure
            if (signature == AGGREGATEDSUBSTITUTIONS) or (signature == AGGREGATEDINDELS) or (signature == AGGREGATEDDINUCS):
                sub_dir = signature
            else:
                sub_dir = SIGNATUREBASED
            typebased_tissuebased_replication_time_number_of_mutations_filename = "%s_NumberofMutations.txt" % (signature)
            typebased_tissuebased_replication_time_number_of_mutations_filepath = os.path.join(combined_output_dir,cancer_type, DATA,REPLICATION_TIME,sub_dir,typebased_tissuebased_replication_time_number_of_mutations_filename)

            if (os.path.exists(typebased_tissuebased_replication_time_number_of_mutations_filepath)):
                nparray_number_of_mutations = np.loadtxt(typebased_tissuebased_replication_time_number_of_mutations_filepath) # dtype default float legacy dtype=np.int64
                if np.any(nparray_number_of_mutations) and os.path.exists(cancer_type_output_path):
                    for file in os.listdir(cancer_type_output_path):
                        if (signature_with_underscore in file) and ('_replication_time.png' in file):
                            signaturebased_cutoff_file_path = os.path.join(combined_output_dir, cancer_type, FIGURE, REPLICATION_TIME, file)

                            if (os.path.exists(signaturebased_cutoff_file_path)):
                                interested_file_list.append(signaturebased_cutoff_file_path)

        # For replication time pdfs
        if (len(interested_file_list)>0) :
            # One pdf for each type, first left image: signature across all tissue prob05 --- first right image: signature across all tissues prob09
            # other images: type for each tissue prob05 --- type for each tissue prob09
            print('################')
            pdffile_name = "%s_across_all_tissues_and_each_tissue.pdf" % (signature)
            pdf_file_path = os.path.join(path,PDF_FILES,pdffile_name)

            print(pdf_file_path)
            print('################')
            c = canvas.Canvas(pdf_file_path, pagesize=letter)
            width, height = letter  # keep for later
            print('canvas letter: width=%d height=%d' % (width, height))
            # width=612 height=792

            # Center header
            c.setFont("Times-Roman", 15)
            title = 'Replication Time' + ' ' + signature

            title_width = stringWidth(title, "Times-Roman", 15)
            c.drawString((width - title_width) / 2, height - 20, title)

            # One page can take 8 images
            # For images
            c.setFont("Times-Roman", 8)
            figureCount=0

            figure_width = 5

            figure_left_x = 40
            figure_left_label_x = figure_left_x

            figure_middle_x = 200

            figure_right_x = 300
            figure_right_label_x = figure_right_x

            y = 610
            label_y_plus = 150
            label_y_minus = -10

            ###############################################################
            for file in interested_file_list:
                # SBS1_83300_replication_time.png
                # output_dir, cancer_type, 'figure', 'all', 'replication_time', 'SBS1_83300_replication_time.png'

                # First file across all tissues
                if (file == interested_file_list[0]):
                    tissue = ''
                    pearson_corr = ''
                    img = utils.ImageReader(file)
                    iw, ih = img.getSize()
                    print('image: width=%d height=%d' % (iw, ih))
                    aspect = ih / float(iw)
                    print(file)
                    figureCount=figureCount+2
                    #To the center
                    c.drawImage(file, figure_middle_x, y, figure_width * cm, figure_width * aspect * cm)
                    y = y - 160
                # each tissue
                else:
                    indexEnd = file.index('figure')
                    indexSlashForward = file.rfind('/', 0, indexEnd-1)
                    tissue=file[indexSlashForward+1:indexEnd-1]

                    if (all_signatures_replication_time_df[
                        (all_signatures_replication_time_df['signature'] == signature) &
                        (all_signatures_replication_time_df['cancer_type'] == tissue)].values.any()):

                        pearson_corr = all_signatures_replication_time_df[
                            (all_signatures_replication_time_df['signature']==signature) &
                            (all_signatures_replication_time_df['cancer_type']==tissue)]['pearson_corr'].values[0]

                        cancer_type_slope = all_signatures_replication_time_df[
                            (all_signatures_replication_time_df['signature'] == signature) &
                            (all_signatures_replication_time_df['cancer_type'] == tissue)]['cancer_type_slope'].values[0]

                        cancer_type_decision = all_signatures_replication_time_df[
                            (all_signatures_replication_time_df['signature'] == signature) &
                            (all_signatures_replication_time_df['cancer_type'] == tissue)]['cancer_type_decision'].values[0]

                        real_number_of_mutations_array = all_signatures_replication_time_df[
                            (all_signatures_replication_time_df['signature'] == signature) &
                            (all_signatures_replication_time_df['cancer_type'] == tissue)]['real_number_of_mutations_array'].values[0]

                        real_normalized_mutations_density_array = all_signatures_replication_time_df[
                            (all_signatures_replication_time_df['signature'] == signature) &
                            (all_signatures_replication_time_df['cancer_type'] == tissue)]['real_normalized_mutations_density_array'].values[0]

                        pearson_corr = 'Pearson Corr: %.2f ' %pearson_corr
                        slope = 'Slope: %.6f' %cancer_type_slope
                        decision = '%s' %cancer_type_decision
                        real_number_of_mutations_array = "%s" %(real_number_of_mutations_array)
                        real_normalized_mutations_density_array = [round(mutation_density,2) for mutation_density in real_normalized_mutations_density_array]
                        real_normalized_mutations_density_array = "%s" % (real_normalized_mutations_density_array)

                    else:
                        pearson_corr = 'Pearson correlation is not available'
                        slope = 'Slope is not available'
                        decision = 'Decision is not available'
                        real_number_of_mutations_array = "Number of mutations not available"
                        real_normalized_mutations_density_array = "Normalized mutations density not available"

                    img = utils.ImageReader(file)
                    iw, ih = img.getSize()
                    print('image: width=%d height=%d' % (iw, ih))
                    aspect = ih / float(iw)
                    print(file)
                    figureCount = figureCount + 1

                    # To the left
                    if (figureCount % 2 == 1):
                        c.drawImage(file, figure_left_x, y, figure_width * cm, figure_width * aspect * cm)
                        c.setFont("Times-Roman", 8)
                        c.drawString(figure_left_label_x, y + label_y_plus, tissue)
                        c.drawString(figure_left_label_x+80, y + label_y_plus, pearson_corr + " " + slope + " " + decision)
                        c.drawString(figure_left_label_x-20, y + label_y_minus, real_number_of_mutations_array)
                        c.drawString(figure_left_label_x-20, y + label_y_minus + label_y_minus, real_normalized_mutations_density_array)

                    # To the right
                    elif (figureCount % 2 == 0):
                        c.drawImage(file, figure_right_x, y, figure_width * cm, figure_width * aspect * cm)
                        c.setFont("Times-Roman", 8)
                        c.drawString(figure_right_label_x, y + label_y_plus, tissue)
                        c.drawString(figure_right_label_x+80, y + label_y_plus, pearson_corr+ " " + slope + " " + decision)
                        c.drawString(figure_right_label_x, y + label_y_minus, real_number_of_mutations_array)
                        c.drawString(figure_right_label_x, y + label_y_minus + label_y_minus, real_normalized_mutations_density_array)
                        # y = y - 200
                        y = y - 190

                    #There will  be 8 figures in one page except the first page (7 figures)
                    if (figureCount%8 == 0):
                        c.showPage()
                        c.setFont("Times-Roman", 8)
                        y = 610
            c.save()



def fill_processivity_figures_pdf(plot_output_dir,interested_figure_file_list):

    path= os.path.join(plot_output_dir,PROCESSIVITY,PDF_FILES)

    # For processivity pdfs
    if (len(interested_figure_file_list)>0) :
        ###############################################################
        # One pdf for each type, first left image: signature across all tissue prob05 --- first right image: signature across all tissues prob09
        # other images: type for each tissue prob05 --- type for each tissue prob09
        print('################')
        pdffile_name = "Combined_PCAWG_nonPCAWG_All_Cancer_Types_Processivity.pdf"
        pdf_file_path = os.path.join(path,pdffile_name)
        print(pdf_file_path)
        print('################')

        c = canvas.Canvas(pdf_file_path, pagesize=letter)
        c.translate(cm, cm)
        width, height = letter  # keep for later
        print('canvas letter: width=%d height=%d' % (width, height))
        # width=612 height=792

        # Center header
        c.setFont("Times-Roman", 15)
        title = 'Processity'

        title_width = stringWidth(title, "Times-Roman", 15)
        c.drawString((width - title_width) / 2, height - 20, title)

        # One page can take 8 images
        # For images
        c.setFont("Times-Roman", 8)

        x = 1*cm
        y = height-3*cm

        for file in interested_figure_file_list:
            # /oasis/tscc/scratch/burcak/developer/python/SigProfilerTopography/SigProfilerTopography/output/PCAWG_Matlab_Clean/Breast-AdenoCA/figure/all/processivity

            indexEnd = file.index('figure')
            indexSlashForward = file.rfind('/', 0, indexEnd-1)
            tissue=file[indexSlashForward+1:indexEnd-1]

            img = utils.ImageReader(file)
            iw, ih = img.getSize()
            iw_250x = iw/250
            ih_250x = ih/250
            # For information
            # print('tissue:%s image: width=%d height=%d' % (tissue,iw, ih))
            # print('tissue:%s image: width/250=%d height/250=%d' % (tissue,iw_250x, ih_250x))
            print(file)

            if (y > ih_250x*cm+2*cm):
                y = y - ih_250x * cm - 1 * cm
                c.drawImage(file, x, y, iw_250x *cm, ih_250x*cm,preserveAspectRatio=True)

            else:
                c.showPage()
                y = height-3*cm
                y = y - ih_250x * cm - 1 * cm
                c.drawImage(file, x, y, iw_250x * cm, ih_250x * cm, preserveAspectRatio=True)

        c.save()


def write_all_tissues_combined_dataframe(plot_output_dir,
                                         interested_processivity_table_file_list,
                                         interested_cancer_types_list,
                                         cosmic_release_version,
                                         figure_file_extension,
                                         minimum_required_number_of_processive_groups):

    path = os.path.join(plot_output_dir, PROCESSIVITY)

    processivity_dfs = []

    # Vertically combine dfs
    for file_index, interested_processivity_table_file in enumerate(interested_processivity_table_file_list):
        cancer_type = interested_cancer_types_list[file_index]
        df = pd.read_csv(interested_processivity_table_file, header=0, sep='\t')

        # This column is added for Figure Case Study SBS4
        df['cancer_type'] = cancer_type

        if np.any(df.isin(['Sample']).any(axis=1).values):
            sample_row_index = df.loc[df.isin(['Sample']).any(axis=1)].index.tolist()[0]
            print(df.loc[sample_row_index])
            df = df.iloc[0:sample_row_index,:]

        # Formerly, we were appending each df as below
        # processivity_dfs.append(df)

        # Specific to lymphoid samples starts
        # We don't consider Lymph-BNHL or Lymph-CLL
        # Since, for processivity, we are not interested in aggregated mutations, DBS and ID signatures
        if cancer_type == LYMPH_BNHL_CLUSTERED or cancer_type == LYMPH_CLL_CLUSTERED:
            # take rows containing SBS37, SBS84 and SBS85 only
            df = df[df['signature'].isin(['SBS37', 'SBS84', 'SBS85'])]
            # To show results under main_cancer_type remove '_clustered' at the end
            df['cancer_type'] = cancer_type[:-10]
            processivity_dfs.append(df)
        elif cancer_type == LYMPH_BNHL_NONCLUSTERED or cancer_type == LYMPH_CLL_NONCLUSTERED:
            # consider all rows
            # To show results under main_cancer_type remove '_nonClustered' at the end
            df['cancer_type'] = cancer_type[:-13]
            processivity_dfs.append(df)
        elif cancer_type != LYMPH_BNHL and cancer_type != LYMPH_CLL:
            processivity_dfs.append(df)

        if cancer_type == LYMPH_BNHL_CLUSTERED or cancer_type == LYMPH_CLL_CLUSTERED:
            cancer_type = cancer_type[:-10]
        elif cancer_type == LYMPH_BNHL_NONCLUSTERED or cancer_type == LYMPH_BNHL_NONCLUSTERED:
            cancer_type = cancer_type[:-13]
        # Specific to lymphoid samples ends

        df = df.astype(dtype={'radius': float})
        df = df.astype(dtype={'number_of_processive_groups': int})
        df = df.astype(dtype={'processive_group_length': int})
        df = df.astype(dtype={'log10_number_of_processive_groups': float})

        max_processive_group_length = df[(round(df['radius'], 2) > 0) & (df['number_of_processive_groups'] >= minimum_required_number_of_processive_groups)]['processive_group_length'].max()
        processsive_group_length_list = df[(round(df['radius'], 2) > 0) & (df['number_of_processive_groups'] >= minimum_required_number_of_processive_groups)]['processive_group_length'].unique()
        sorted_processsive_group_length_list = sorted(processsive_group_length_list, key=int)

        signatures = df['signature'].unique()

        # COSMIC cancer type based processivity figures using df
        for signature in signatures:
            signature_list = [signature]
            plot_processivity_figure(path,
                                     signature_list,
                                     sorted_processsive_group_length_list,
                                     max_processive_group_length,
                                     df,
                                     COSMIC,
                                     cosmic_release_version,
                                     figure_file_extension,
                                     minimum_required_number_of_processive_groups,
                                     signature_name = signature,
                                     cancer_type = cancer_type)

    combined_PCAWG_nonPCAWG_processivity_df = pd.concat(processivity_dfs)

    # Write all cancer types pooled processivity_df
    pcawg_processivity_df_file_path = os.path.join(path, TABLES, 'Combined_PCAWG_nonPCAWG_All_Cancer_Types_Processivity.txt')
    combined_PCAWG_nonPCAWG_processivity_df.to_csv(pcawg_processivity_df_file_path, sep='\t', header=True, index=False)

    combined_PCAWG_nonPCAWG_processivity_df = combined_PCAWG_nonPCAWG_processivity_df.astype(dtype={'number_of_processive_groups': int})
    # combined_PCAWG_nonPCAWG_processivity_df['number_of_processive_groups'] = combined_PCAWG_nonPCAWG_processivity_df['number_of_processive_groups'].astype(int)

    return combined_PCAWG_nonPCAWG_processivity_df


def set_radius(df):
    df['radius'] = df['log10_avg_number_of_processive_groups'] / df['log10_avg_number_of_processive_groups'].max() * 0.48
    return df

def plot_processivity_mediator(processivity_output_dir,
                               signature_processive_group_length_properties_df,
                               figure_type,
                               cosmic_release_version,
                               figure_file_extension,
                               processivity_significance_level,
                               minimum_required_processive_group_length,
                               minimum_required_number_of_processive_groups):

    if figure_type == COSMIC:
        df_list=[]
        # Plot each signature one by one
        signature_array = signature_processive_group_length_properties_df['signature'].unique()
        for signature in signature_array:
            df = signature_processive_group_length_properties_df[signature_processive_group_length_properties_df['signature'] == signature].copy()
            df_list.append(plot_processivity_across_all_tissues(processivity_output_dir,
                                                                df,
                                                                figure_type,
                                                                cosmic_release_version,
                                                                figure_file_extension,
                                                                processivity_significance_level,
                                                                minimum_required_processive_group_length,
                                                                minimum_required_number_of_processive_groups,
                                                                signature_name = signature))
        return pd.concat(df_list,axis=0)

    elif figure_type == MANUSCRIPT:
        # Plot all signatures in one page summary figure
        return plot_processivity_across_all_tissues(processivity_output_dir,
                                                    signature_processive_group_length_properties_df,
                                                    figure_type,
                                                    cosmic_release_version,
                                                    figure_file_extension,
                                                    processivity_significance_level,
                                                    minimum_required_processive_group_length,
                                                    minimum_required_number_of_processive_groups)



# We are plotting one figure across all cancer types for MANUSCRIPT
# We are plotting one figure for each signature for COSMIC
def plot_processivity_across_all_tissues(processivity_output_dir,
                                        signature_processive_group_length_properties_df,
                                        figure_type,
                                        cosmic_release_version,
                                        figure_file_extension,
                                        processivity_significance_level,
                                        minimum_required_processive_group_length,
                                        minimum_required_number_of_processive_groups,
                                        signature_name = None):

    # Get the list of signatures using original
    signature_array = signature_processive_group_length_properties_df['signature'].unique()

    # Get the list of processive group lengths using original
    processive_group_length_array = signature_processive_group_length_properties_df['processive_group_length'].unique()

    # Step1 Calculate p-values
    all_p_values = []
    names_list = []

    for signature in signature_array:
        for processive_group_length in processive_group_length_array:
            if signature_processive_group_length_properties_df[(signature_processive_group_length_properties_df['signature'] == signature) &
                                                               (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)].values.any():

                observed_value = signature_processive_group_length_properties_df[
                    (signature_processive_group_length_properties_df['signature'] == signature) &
                    (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)]['avg_number_of_processive_groups'].values[0]

                expected_values = signature_processive_group_length_properties_df[
                    (signature_processive_group_length_properties_df['signature'] == signature) &
                    (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)]['expected_avg_number_of_processive_groups'].values[0]

                zscore = None

                if (not np.isnan(observed_value)) and ((len(expected_values) > 0) and np.count_nonzero(expected_values) > 0):
                    zstat, pvalue = calculate_pvalue_teststatistics(observed_value, expected_values, alternative = 'smaller')

                    # Please note
                    # If pvalue is np.nan e.g.: due to a few expected values like only one [1]
                    # Then there must be cases when you may want to manually set minus_log10_qvalue to np.inf
                    if np.isnan(pvalue):
                        signature_processive_group_length_properties_df.loc[
                            ((signature_processive_group_length_properties_df['signature'] == signature) &
                             (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'minus_log10_qvalue'] = np.inf

                    if (pvalue is not None) and (not np.isnan(pvalue)):
                        all_p_values.append(pvalue)
                        names_list.append((signature,processive_group_length))

                    avg_sims = sum(expected_values)/len(expected_values)
                    min_sims = min(expected_values)
                    max_sims = max(expected_values)
                    mean_sims = np.mean(expected_values)
                    std_sims = np.std(expected_values)

                    if (std_sims > 0):
                        zscore = (observed_value - mean_sims)/std_sims

                    signature_processive_group_length_properties_df.loc[
                        ((signature_processive_group_length_properties_df['signature'] == signature) &
                        (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'avg_sims'] = avg_sims

                    signature_processive_group_length_properties_df.loc[
                        ((signature_processive_group_length_properties_df['signature'] == signature) &
                        (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'min_sims'] = min_sims

                    signature_processive_group_length_properties_df.loc[
                        ((signature_processive_group_length_properties_df['signature'] == signature) &
                        (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'max_sims'] = max_sims

                    signature_processive_group_length_properties_df.loc[
                        ((signature_processive_group_length_properties_df['signature'] == signature) &
                        (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'mean_sims'] = mean_sims

                    signature_processive_group_length_properties_df.loc[
                        ((signature_processive_group_length_properties_df['signature'] == signature) &
                        (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'std_sims'] = std_sims

                    signature_processive_group_length_properties_df.loc[
                        ((signature_processive_group_length_properties_df['signature'] == signature) &
                        (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'pvalue'] = pvalue

                    signature_processive_group_length_properties_df.loc[
                        ((signature_processive_group_length_properties_df['signature'] == signature) &
                        (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'zscore'] = zscore

                # if expected_values all zeros p_value calculation gives error RuntimeWarning: divide by zero encountered in double_scalars
                elif (not np.isnan(observed_value)) and ((len(expected_values) > 0) and np.count_nonzero(expected_values) == 0):
                    # manually set
                    signature_processive_group_length_properties_df.loc[
                        ((signature_processive_group_length_properties_df['signature'] == signature) &
                         (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'pvalue'] = 0

                    signature_processive_group_length_properties_df.loc[
                        ((signature_processive_group_length_properties_df['signature'] == signature) &
                         (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'qvalue'] = 0

                    signature_processive_group_length_properties_df.loc[
                        ((signature_processive_group_length_properties_df['signature'] == signature) &
                         (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'minus_log10_qvalue'] = np.inf

                    signature_processive_group_length_properties_df.loc[
                        ((signature_processive_group_length_properties_df['signature'] == signature) &
                        (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'avg_sims'] = 0

                    signature_processive_group_length_properties_df.loc[
                        ((signature_processive_group_length_properties_df['signature'] == signature) &
                        (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'min_sims'] = 0

                    signature_processive_group_length_properties_df.loc[
                        ((signature_processive_group_length_properties_df['signature'] == signature) &
                        (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'max_sims'] = 0

                    signature_processive_group_length_properties_df.loc[
                        ((signature_processive_group_length_properties_df['signature'] == signature) &
                        (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'mean_sims'] = 0

                    signature_processive_group_length_properties_df.loc[
                        ((signature_processive_group_length_properties_df['signature'] == signature) &
                        (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'std_sims'] = 0

                elif (not np.isnan(observed_value)) and  (len(expected_values) == 0):
                    # manually set
                    signature_processive_group_length_properties_df.loc[
                        ((signature_processive_group_length_properties_df['signature'] == signature) &
                         (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'pvalue'] = 0

                    signature_processive_group_length_properties_df.loc[
                        ((signature_processive_group_length_properties_df['signature'] == signature) &
                         (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'qvalue'] = 0

                    signature_processive_group_length_properties_df.loc[
                        ((signature_processive_group_length_properties_df['signature'] == signature) &
                         (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'minus_log10_qvalue'] = np.inf

    all_p_values_array = np.asarray(all_p_values)
    all_FDR_BH_adjusted_p_values=None

    # Step2 FDR BH Multiple Testing Correction
    try:
        rejected, all_FDR_BH_adjusted_p_values, alphacSidak, alphacBonf = statsmodels.stats.multitest.multipletests(all_p_values_array, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
    except ZeroDivisionError:
        print('ZeroDivisionError during statsmodels.stats.multitest.multipletests')
        print('for debug ZeroDivisionError, all_p_values_array:')
        print(all_p_values_array)

    # If q_value==0 then minus_log10_q_value=np.nan since log10(0) is undefined.
    # if q_value>=SIGNIFICANCE_LEVEL then minus_log10_q_value<=2 for significance level of 0.01  and with color bar set between [2,20] will not get a color.
    # minus_log10_all_FDR_BH_adjusted_p_values=[-math.log10(q_value)  if (q_value>0 and q_value<SIGNIFICANCE_LEVEL)  else np.nan for q_value in all_FDR_BH_adjusted_p_values]
    minus_log10_all_FDR_BH_adjusted_p_values = [-np.log10(q_value)  if (q_value > 0)  else np.inf for q_value in all_FDR_BH_adjusted_p_values]

    print('#############################################')
    print('len(all_p_values):%d\nall_p_values: %s' %(len(all_p_values), all_p_values))
    print('len(names_list): ', len(names_list),' names_list: ' , names_list)
    print('len(all_FDR_BH_adjusted_p_values):%d\nall_FDR_BH_adjusted_p_values: %s' %(len(all_FDR_BH_adjusted_p_values), all_FDR_BH_adjusted_p_values))
    print('len(minus_log10_all_FDR_BH_adjusted_p_values):%d\n minus_log10_all_FDR_BH_adjusted_p_values:%s' %(len(minus_log10_all_FDR_BH_adjusted_p_values),minus_log10_all_FDR_BH_adjusted_p_values))

    # Get the corrected q values in an order
    for index, (signature, processive_group_length) in  enumerate(names_list,0):
        qvalue = all_FDR_BH_adjusted_p_values[index]
        minus_log10_qvalue = minus_log10_all_FDR_BH_adjusted_p_values[index]

        if signature_processive_group_length_properties_df[
            ((signature_processive_group_length_properties_df['signature'] == signature) &
                 (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length))].values.any():

            signature_processive_group_length_properties_df.loc[
                ((signature_processive_group_length_properties_df['signature'] == signature) &
                 (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'qvalue'] = qvalue

            signature_processive_group_length_properties_df.loc[
                ((signature_processive_group_length_properties_df['signature'] == signature) &
                 (signature_processive_group_length_properties_df['processive_group_length'] == processive_group_length)), 'minus_log10_qvalue'] = minus_log10_qvalue

    # Remove the processive group length < MINIMUM_REQUIRED_PROCESSIVE_GROUP_LENGTH
    if figure_type == MANUSCRIPT:
        # Filter1
        # Remove rows where signature is attributed to artefact signatures
        signature_processive_group_length_properties_df = signature_processive_group_length_properties_df[~signature_processive_group_length_properties_df['signature'].isin(signatures_attributed_to_artifacts)]

        # Filter2
        # Filter the rows where processive_group_length >= MINIMUM_REQUIRED_PROCESSIVE_GROUP_LENGTH
        signature_processive_group_length_properties_df = signature_processive_group_length_properties_df[
            signature_processive_group_length_properties_df['processive_group_length'] >= minimum_required_processive_group_length]

        signature_processive_group_length_properties_df['log10_avg_number_of_processive_groups'] = \
            np.log10(signature_processive_group_length_properties_df['avg_number_of_processive_groups'].replace(0, np.nan))

        # To show avg_number_of_processive_groups = 1
        if minimum_required_number_of_processive_groups == 1:
            signature_processive_group_length_properties_df.loc[(signature_processive_group_length_properties_df[
                                                                     'avg_number_of_processive_groups'] == 1), 'log10_avg_number_of_processive_groups'] = np.log10(2) / 2

        # Here we set radius
        signature_processive_group_length_properties_df = \
            signature_processive_group_length_properties_df.groupby('signature').apply(lambda df: set_radius(df))

        # Get the highest processive group length with a nonzero radius
        max_processive_group_length = signature_processive_group_length_properties_df[
            (round(signature_processive_group_length_properties_df['radius'], 2) > 0) &
            (signature_processive_group_length_properties_df['avg_number_of_processive_groups'] >= minimum_required_number_of_processive_groups) &
            (signature_processive_group_length_properties_df['qvalue'] <= processivity_significance_level)][
            'processive_group_length'].max()

        # Update sorted_processsive_group_length_list
        processsive_group_length_list = signature_processive_group_length_properties_df[
            (round(signature_processive_group_length_properties_df['radius'], 2) > 0) &
            (signature_processive_group_length_properties_df['avg_number_of_processive_groups'] >= minimum_required_number_of_processive_groups) &
            (signature_processive_group_length_properties_df['qvalue'] <= processivity_significance_level)][
            'processive_group_length'].unique()
        sorted_processsive_group_length_list = sorted(processsive_group_length_list, key=int)

        # Pay attention it has to be reverse=True for grid representation to get SBS1 on the upper left corner otherwise SBS1 shows up on the lower left corner
        # Update sorted_signature_list
        signatures_list = signature_processive_group_length_properties_df[
            (round(signature_processive_group_length_properties_df['radius'], 2) > 0) &
            (signature_processive_group_length_properties_df['avg_number_of_processive_groups'] >= minimum_required_number_of_processive_groups) &
            (signature_processive_group_length_properties_df['qvalue'] <= processivity_significance_level)]['signature'].unique()
        sorted_signature_list = sorted(signatures_list, reverse=True, key=natural_key)

    elif figure_type == COSMIC:
        # Set radius
        signature_processive_group_length_properties_df['log10_avg_number_of_processive_groups'] = \
            np.log10(signature_processive_group_length_properties_df['avg_number_of_processive_groups'].replace(0, np.nan))

        # To show avg_number_of_processive_groups=1
        if minimum_required_number_of_processive_groups == 1:
            signature_processive_group_length_properties_df.loc[(signature_processive_group_length_properties_df[
                                                                     'avg_number_of_processive_groups'] == 1), 'log10_avg_number_of_processive_groups'] = np.log10(2) / 2

        # Here we set radius
        signature_processive_group_length_properties_df = \
            signature_processive_group_length_properties_df.groupby('signature').apply(lambda df: set_radius(df))


        # Get the highest processive group length with a nonzero radius
        max_processive_group_length = signature_processive_group_length_properties_df[
            (round(signature_processive_group_length_properties_df['radius'], 2) > 0) &
            (signature_processive_group_length_properties_df['avg_number_of_processive_groups'] >= minimum_required_number_of_processive_groups)][
            'processive_group_length'].max()

        # Update sorted_processsive_group_length_list
        processsive_group_length_list = signature_processive_group_length_properties_df[
            (round(signature_processive_group_length_properties_df['radius'], 2) > 0) &
            (signature_processive_group_length_properties_df['avg_number_of_processive_groups'] >= minimum_required_number_of_processive_groups)][
            'processive_group_length'].unique()
        sorted_processsive_group_length_list = sorted(processsive_group_length_list, key=int)

        # Pay attention it has to be reverse=True for grid representation to get SBS1 on the upper left corner otherwise SBS1 shows up on the lower left corner
        # Update sorted_signature_list
        signatures_list = signature_processive_group_length_properties_df[
            (round(signature_processive_group_length_properties_df['radius'], 2) > 0) &
            (signature_processive_group_length_properties_df['avg_number_of_processive_groups'] >= minimum_required_number_of_processive_groups)][
            'signature'].unique()
        sorted_signature_list = sorted(signatures_list, reverse=True, key=natural_key)

    # Plot figure
    plot_processivity_figure(processivity_output_dir,
                             sorted_signature_list,
                             sorted_processsive_group_length_list,
                             max_processive_group_length,
                             signature_processive_group_length_properties_df,
                             figure_type,
                             cosmic_release_version,
                             figure_file_extension,
                             minimum_required_number_of_processive_groups,
                             signature_name = signature_name)

    filePath=os.path.join(processivity_output_dir, TABLES, 'Combined_PCAWG_nonPCAWG_Across_All_Cancer_Types_Pooled_Processivity.txt')
    signature_processive_group_length_properties_df.to_csv(filePath, sep='\t', header=True, index=False)

    return signature_processive_group_length_properties_df

# Cosmic Signature Tissue Based
# Cosmic Signature Across All Tissues
# Manuscript All Signatures Across All Tissues
# Uses same method
def plot_processivity_figure(path,
                             signature_list,
                             processive_group_length_list,
                             max_processive_group_length,
                             signature_processive_group_length_properties_df,
                             figure_type,
                             cosmic_release_version,
                             figure_file_extension,
                             minimum_required_number_of_processive_groups,
                             signature_name = None,
                             cancer_type = None):

    index = None
    if ((len(processive_group_length_list) > 0) and (max_processive_group_length > 0)):
        # Find index of max_processive_group_length in processive_group_length_array
        # np.where(processive_group_length_array==max_processive_group_length) returns tuple
        # [0] is the index array
        # [0][0] is the first index of this processive group length
        index = processive_group_length_list.index(max_processive_group_length)

    print("signature_list: ", signature_list)
    print("processive_group_length_list: ", processive_group_length_list)
    print("max_processive_group_length: ", max_processive_group_length)
    print("index of max_processive_group_length: ", index)

    width_multiply = 1.5
    height_multiply = 1.5

    fig = plt.figure(figsize=(width_multiply * max_processive_group_length, height_multiply * len(signature_list) ))

    if figure_type == MANUSCRIPT:
        label_fontsize = 50
        ylabel_fontsize = 60
        ax = plt.gca()
    elif figure_type == COSMIC:
        label_fontsize = 20
        ylabel_fontsize = 30

        grid = plt.GridSpec(5, 3, hspace=0, wspace=0) # legacy
        ax = fig.add_subplot(grid[0:3, :]) # legacy
        processivity_legend_ax = fig.add_subplot(grid[-1, :]) # legacy

    ax.set_aspect(1.0)  # make aspect ratio square

    cmap = matplotlib_cm.get_cmap('YlOrRd')  # Looks better
    v_min = 2
    v_max = 20
    norm = Normalize(v_min, v_max)

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

    if not signature_processive_group_length_properties_df.empty:
        # Plot the circles with color
        for signature_index, signature in enumerate(signature_list):
            for processive_group_length_index, processive_group_length in enumerate(processive_group_length_list):
                number_of_processive_groups = np.nan
                radius=np.nan
                color=np.nan

                if cancer_type:
                    number_of_processive_groups_column_name = 'number_of_processive_groups'
                else:
                    number_of_processive_groups_column_name = 'avg_number_of_processive_groups'

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

    xlabels=None
    if (index is not None):
        xlabels = processive_group_length_list[0:index+1]
    ylabels = signature_list

    if figure_type == COSMIC:
        plot_processivity_legend_in_given_axis(processivity_legend_ax, label_fontsize/2)
        # Works w.r.t. ax
        # legacy
        if len(processive_group_length_list) == 1:
            width="8%"
            bbox_to_anchor_tuple = (2.5, 0, 1, 1)
        elif len(processive_group_length_list) <= 3:
            width="4%"
            bbox_to_anchor_tuple = (1.5, 0, 1, 1)
        elif len(processive_group_length_list) == 4:
            width = "4%"
            bbox_to_anchor_tuple = (1.25, 0, 1, 1)
        elif len(processive_group_length_list) >= 20:
            width = "0.5%"
            bbox_to_anchor_tuple = (1.05, 0, 1, 1)
        elif len(processive_group_length_list) >= 13:
            width = "1%"
            bbox_to_anchor_tuple = (1.05, 0, 1, 1)
        else:
            width="2%"
            bbox_to_anchor_tuple = (1.05, 0, 1, 1)

        axins = inset_axes(ax,  # here using axis of the lowest plot
                           width=width,  # width = 5% of parent_bbox width
                           height="200%",  # height : 340% good for a (4x4) Grid
                           loc='center left',
                           bbox_to_anchor=bbox_to_anchor_tuple,
                           bbox_transform=ax.transAxes,
                           borderpad=0)

        bounds = np.arange(v_min, v_max + 1, 2)
        cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axins, ticks=bounds, spacing='proportional', orientation='vertical')
        cb.ax.set_ylabel("-log10\n(q-value)", fontsize=label_fontsize/2, labelpad=25, rotation=0)

    if figure_type == MANUSCRIPT:
        figures_manuscript_path = os.path.join(path, FIGURES_MANUSCRIPT)
        # plot colorbar in a separate figure
        plot_processivity_colorbar_horizontal(figures_manuscript_path, cmap, v_min, v_max)
        # plot legend in a separate figure
        plot_processivity_legend(figures_manuscript_path, 30)

    # CODE GOES HERE TO CENTER X-AXIS LABELS...
    ax.set_xticklabels([])
    mticks = ax.get_xticks()

    ax.set_xticks((mticks[:-1] + mticks[1:]) / 2, minor=True)
    ax.tick_params(axis='x', which='minor', length=0, labelsize=label_fontsize)

    if xlabels is not None:
        if figure_type == COSMIC:
            ax.set_xticklabels(xlabels, minor=True)
            ax.set_xlabel(STRAND_COORDINATED_MUTAGENESIS_GROUP_LENGTH, fontsize=label_fontsize, labelpad=15)
        elif figure_type == MANUSCRIPT:
            # ax.set_xticklabels(xlabels, minor=True, fontweight='bold', fontname='Times New Roman')
            # ax.set_xlabel('Processive Group Length',fontsize=label_fontsize, labelpad=20, fontweight='bold', fontname='Times New Roman')
            ax.set_xticklabels(xlabels, minor=True)
            ax.set_xlabel(STRAND_COORDINATED_MUTAGENESIS_GROUP_LENGTH,fontsize=label_fontsize, labelpad=25, fontweight='semibold')

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
    ax.tick_params(axis='y', which='minor', length=0, labelsize=ylabel_fontsize)
    if figure_type == COSMIC:
        ax.set_yticklabels(ylabels, minor=True) # fontsize
    elif figure_type == MANUSCRIPT:
        # ax.set_yticklabels(ylabels, minor=True, fontweight='bold', fontname='Times New Roman') # fontsize
        ax.set_yticklabels(ylabels, minor=True) # fontsize

    ax.tick_params(
        axis='y',  # changes apply to the x-axis
        which='major',  # both major and minor ticks are affected
        left=False)  # labels along the bottom edge are off

    # create the directory if it does not exists
    if figure_type == COSMIC:
        # v3.2_SBS1_REPLIC_ASYM_TA_C34447.jp
        if cancer_type and signature_name:
            if cancer_type in cancer_type_2_NCI_Thesaurus_code_dict:
                NCI_Thesaurus_code = cancer_type_2_NCI_Thesaurus_code_dict[cancer_type]
                filename = '%s_%s_%s_TA_%s.%s' %(cosmic_release_version, signature_name, COSMIC_PROCESSIVITY, NCI_Thesaurus_code, figure_file_extension)
            else:
                filename = '%s_%s_%s_TA_%s.%s' %(cosmic_release_version, signature_name, COSMIC_PROCESSIVITY, cancer_type, figure_file_extension)
        elif signature_name:
            filename = '%s_%s_%s.%s' %(cosmic_release_version, signature_name, COSMIC_PROCESSIVITY, figure_file_extension)
        else:
            filename = 'Combined_PCAWG_nonPCAWG_Across_All_Cancer_Types_Pooled_Processivity.png'

    elif figure_type == MANUSCRIPT:
        if cancer_type and signature_name:
            filename = '%s_%s_Combined_PCAWG_nonPCAWG_Tissue_Based_Processivity.png' %(signature_name,cancer_type)
        elif signature_name:
            filename = '%s_Combined_PCAWG_nonPCAWG_Across_All_Cancer_Types_Pooled_Processivity.png' %(signature_name)
        else:
            filename = 'Combined_PCAWG_nonPCAWG_Across_All_Cancer_Types_Pooled_Processivity.png'

    if figure_type == COSMIC and cancer_type:
        ax.set_title(cancer_type, fontsize = ylabel_fontsize)
        # anchored_text = AnchoredText(cancer_type,
        #                              frameon=False, borderpad=0, pad=0.1,
        #                              # loc='upper right', bbox_to_anchor=[0.2, -0.3], # put text in lower left corner
        #                              # loc='lower left', bbox_to_anchor=[-0.2, -0.7], # put text in lower left corner
        #                              # loc='upper right', bbox_to_anchor=[1.3, 2.15], # put text upper right corner
        #                              loc='upper right', bbox_to_anchor=[1.3, 3.15], # put text upper right corner # check it
        #                              bbox_transform=ax.transAxes,
        #                              prop={'fontsize': label_fontsize, 'fontweight':'semibold'})
        # ax.add_artist(anchored_text)

    if figure_type == COSMIC:
        if cancer_type:
            figFile = os.path.join(path, COSMIC_TISSUE_BASED_FIGURES, filename)
        else:
            figFile = os.path.join(path, FIGURES_COSMIC, filename)
    elif figure_type==MANUSCRIPT:
        figFile = os.path.join(path, FIGURES_MANUSCRIPT, filename)

    fig.savefig(figFile,dpi=100, bbox_inches="tight")

    plt.cla()
    plt.close(fig)


# Horizontal Processivity Colorbar
def plot_processivity_colorbar_horizontal(output_path, cmap, v_min, v_max):
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_axes([0.05, 0.475, 0.9, 0.15])

    # If a ListedColormap is used, the length of the bounds array must be
    # one greater than the length of the color list.  The bounds must be
    # monotonically increasing.

    bounds = np.arange(v_min, v_max+1, 2)
    norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)

    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, ticks=bounds, spacing='proportional',orientation='horizontal')
    cb.ax.tick_params(labelsize=30)
    cb.set_label("-log10 (q-value)", horizontalalignment='center', rotation=0, fontsize=40)

    filename = 'processivity_color_bar_horizontal.png'
    figureFile = os.path.join(output_path, filename)

    fig.savefig(figureFile)
    plt.close()

def plot_processivity_colorbar_vertical(output_path, cmap, v_min, v_max):
    fig = plt.figure(figsize=(4, 10))
    ax = fig.add_axes([0.05, 0.05, 0.1, 0.9])

    # If a ListedColormap is used, the length of the bounds array must be
    # one greater than the length of the color list.  The bounds must be
    # monotonically increasing.

    bounds = np.arange(v_min, v_max + 1, 2)
    norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)

    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, ticks=bounds, spacing='proportional', orientation='vertical')
    cb.ax.tick_params(labelsize=30)
    cb.set_label("-log10\n(q-value)", verticalalignment='center', rotation=0, labelpad=100, fontsize=40)

    filename = 'processivity_color_bar_vertical.png'
    figureFile = os.path.join(output_path, filename)

    fig.savefig(figureFile)
    plt.close()

# Main engine function for processivity
# input all_cancer_types_processivity_df
# output across_all_cancer_types_pooled_processivity_df
def prepare_and_plot_processivity_data_across_all_tissues(plot_output_dir,
                                                          all_cancer_types_processivity_df,
                                                          number_of_simulations,
                                                          figure_type,
                                                          cosmic_release_version,
                                                          figure_file_extension,
                                                          processivity_significance_level,
                                                          minimum_required_processive_group_length,
                                                          minimum_required_number_of_processive_groups):

    processivity_output_dir = os.path.join(plot_output_dir, PROCESSIVITY)

    signature_processive_group_length_properties_df = pd.DataFrame(columns=["signature",
                                                    "processive_group_length",
                                                    "avg_number_of_processive_groups",
                                                    "log10_avg_number_of_processive_groups",
                                                    "radius",
                                                    "avg_sims",
                                                    "min_sims",
                                                    "max_sims",
                                                    "mean_sims",
                                                    "std_sims",
                                                    "pvalue",
                                                    "qvalue",
                                                    "minus_log10_qvalue",
                                                    "zscore",
                                                    "expected_avg_number_of_processive_groups"])

    grouped_processivity_df = all_cancer_types_processivity_df.groupby(['signature','processive_group_length'])

    for name, group in grouped_processivity_df:
        signature, processive_group_length = name
        average_number_of_processive_groups = group['number_of_processive_groups'].mean()

        # expectedValues <class 'str'> is converted into <class 'list'> of integers
        # expectedValues[1] --> [
        # expectedValues[-1] --> ]

        # listofExpectedValues=[[int(x) for x in expectedValues[1:-1].strip().split(',')]  for expectedValues in group['expected_number_of_processive_groups']]
        temp_listofExpectedValues = [[int(x) for x in expectedValues[1:-1].strip().split(',')]  if len(expectedValues)>2 else [] for expectedValues in group['expected_number_of_processive_groups']]

        # There can be more than one expected_number_of_processive_groups
        # e.g. For SBS4 processive_group_length=4 there are 5 expected_number_of_processive_groups coming from 5 cancer types
        listofExpectedValues = []

        for expected_values_list in temp_listofExpectedValues:
            if len(expected_values_list) == number_of_simulations:
                listofExpectedValues.append(expected_values_list)
            elif (len(expected_values_list) < number_of_simulations) and (len(expected_values_list) > 0) and (len(temp_listofExpectedValues) > 1):
                number_of_misses = number_of_simulations - len(expected_values_list)
                list_to_be_extended = [np.nanmean(expected_values_list)] * number_of_misses
                expected_values_list.extend(list_to_be_extended)
                listofExpectedValues.append(expected_values_list)
            elif (len(expected_values_list) < number_of_simulations) and (len(expected_values_list) > 0) and (len(temp_listofExpectedValues) == 1):
                listofExpectedValues.append(expected_values_list)
            elif len(expected_values_list) == 0:
                listofExpectedValues.append([0] * number_of_simulations)

        if len(listofExpectedValues) > 0:
            stacked_expected_values = np.vstack(listofExpectedValues)

            # Take mean column-wise
            stacked_means = np.nanmean(stacked_expected_values, axis=0)
            stacked_means_list = stacked_means.tolist()

            signature_processive_group_length_properties_df = signature_processive_group_length_properties_df.append(
                {"signature": signature,
                 "processive_group_length": int(processive_group_length),
                 "avg_number_of_processive_groups": average_number_of_processive_groups,
                 "log10_avg_number_of_processive_groups": np.nan,
                 "radius": np.nan,
                 "avg_sims": np.nan,
                 "min_sims": np.nan,
                 "max_sims": np.nan,
                 "mean_sims": np.nan,
                 "std_sims": np.nan,
                 "pvalue": np.nan,
                 "qvalue": np.nan,
                 "minus_log10_qvalue": np.nan,
                 "zscore": np.nan,
                 "expected_avg_number_of_processive_groups": np.around(stacked_means_list, NUMBER_OF_DECIMAL_PLACES_TO_ROUND).tolist()}, ignore_index = True)

    return plot_processivity_mediator(processivity_output_dir,
                                      signature_processive_group_length_properties_df,
                                      figure_type,
                                      cosmic_release_version,
                                      figure_file_extension,
                                      processivity_significance_level,
                                      minimum_required_processive_group_length,
                                      minimum_required_number_of_processive_groups)

def generate_processivity_pdf(plot_output_dir,
                              combined_output_dir,
                              cancer_types,
                              number_of_simulations,
                              figure_types,
                              cosmic_release_version,
                              figure_file_extension,
                              processivity_significance_level,
                              minimum_required_processive_group_length,
                              minimum_required_number_of_processive_groups):

    deleteOldData(os.path.join(plot_output_dir, PROCESSIVITY))

    for figure_type in figure_types:
        if figure_type == COSMIC:
            os.makedirs(os.path.join(plot_output_dir, PROCESSIVITY, FIGURES_COSMIC), exist_ok=True)
            os.makedirs(os.path.join(plot_output_dir, PROCESSIVITY, COSMIC_TISSUE_BASED_FIGURES), exist_ok=True)
            os.makedirs(os.path.join(plot_output_dir, PROCESSIVITY, COSMIC_ACROSS_ALL_AND_TISSUE_BASED_TOGETHER), exist_ok=True)
        elif figure_type == MANUSCRIPT:
            os.makedirs(os.path.join(plot_output_dir, PROCESSIVITY, FIGURES_MANUSCRIPT), exist_ok=True)

    # Processivity
    os.makedirs(os.path.join(plot_output_dir, PROCESSIVITY),exist_ok=True)
    os.makedirs(os.path.join(plot_output_dir, PROCESSIVITY, TABLES),exist_ok=True)
    os.makedirs(os.path.join(plot_output_dir, PROCESSIVITY, EXCEL_FILES),exist_ok=True)
    os.makedirs(os.path.join(plot_output_dir, PROCESSIVITY, PDF_FILES),exist_ok=True)
    os.makedirs(os.path.join(plot_output_dir, PROCESSIVITY, DATA_FILES),exist_ok=True)

    # There will be one pdf for all cancer_types
    interested_figure_file_list = []
    interested_processivity_table_file_list = []
    interested_cancer_types_list = []

    for cancer_type in cancer_types:
        # Based on SigProfilerTopography output
        cancer_type_based_processivity_figure = os.path.join(combined_output_dir, cancer_type, FIGURE, PROCESSIVITY, '%s_Processivity.png' %(cancer_type))
        if os.path.exists(cancer_type_based_processivity_figure):
            interested_figure_file_list.append(cancer_type_based_processivity_figure)

        # Based on SigProfilerTopography output
        cancer_type_based_processivity_txt = os.path.join(combined_output_dir, cancer_type, FIGURE, PROCESSIVITY, TABLES, '%s_Signatures_Processivity.txt' %(cancer_type))

        if os.path.exists(cancer_type_based_processivity_txt):
            interested_processivity_table_file_list.append(cancer_type_based_processivity_txt)
            interested_cancer_types_list.append(cancer_type)

        # This part is specific to lymphoid samples
        # extra added for Lymph-BNHL and Lymph-CLL clustered starts
        # extra added for Lymph-BNHL and Lymph-CLL nonClustered starts
        if (cancer_type == LYMPH_BNHL) or (cancer_type == LYMPH_CLL):

            cancer_type_based_processivity_txt = os.path.join(ALTERNATIVE_OUTPUT_DIR, '%s_clustered' %(cancer_type) , FIGURE, PROCESSIVITY, TABLES, '%s_clustered_Signatures_Processivity.txt' %(cancer_type))
            interested_processivity_table_file_list.append(cancer_type_based_processivity_txt)
            interested_cancer_types_list.append('%s_clustered' %(cancer_type))

            cancer_type_based_processivity_txt = os.path.join(ALTERNATIVE_OUTPUT_DIR, '%s_nonClustered' %(cancer_type) , FIGURE, PROCESSIVITY, TABLES, '%s_nonClustered_Signatures_Processivity.txt' %(cancer_type))
            interested_processivity_table_file_list.append(cancer_type_based_processivity_txt)
            interested_cancer_types_list.append('%s_nonClustered' %(cancer_type))
        # extra added for Lymph-BNHL and Lymph-CLL clustered ends
        # extra added for Lymph-BNHL and Lymph-CLL nonClustered ends

    # Fill PDF file for information
    fill_processivity_figures_pdf(plot_output_dir, interested_figure_file_list)

    # Write across all cancer types processivity dataframe
    # Plot Cosmic signature -- cancer type based -- processivity figure
    all_cancer_types_processivity_df = write_all_tissues_combined_dataframe(plot_output_dir,
                                                                            interested_processivity_table_file_list,
                                                                            interested_cancer_types_list,
                                                                            cosmic_release_version,
                                                                            figure_file_extension,
                                                                            minimum_required_number_of_processive_groups)

    # Main engine function
    # Plot Cosmic Signature across all cancer types processivity figure using combined_PCAWG_nonPCAWG_processivity_df
    # Plot Manuscript All Signatures across all cancer types processivity figure using combined_PCAWG_nonPCAWG_processivity_df
    for figure_type in figure_types:
        across_all_cancer_types_pooled_processivity_df = prepare_and_plot_processivity_data_across_all_tissues(plot_output_dir,
                                                                                                               all_cancer_types_processivity_df,
                                                                                                               number_of_simulations,
                                                                                                               figure_type,
                                                                                                               cosmic_release_version,
                                                                                                               figure_file_extension,
                                                                                                               processivity_significance_level,
                                                                                                               minimum_required_processive_group_length,
                                                                                                               minimum_required_number_of_processive_groups)

    across_all_cancer_types_pooled_processivity_df['cancer_type'] = ACROSS_ALL_CANCER_TYPES

    # For Figure Case Study SBS4
    return all_cancer_types_processivity_df, across_all_cancer_types_pooled_processivity_df


def generate_replication_time_pdfs(plot_output_dir,
                                   combined_output_dir,
                                   cancer_types,
                                   all_sbs_signatures,
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
                                   sub_figure_type):

    deleteOldData(os.path.join(plot_output_dir, REPLICATION_TIME))

    for figure_type in figure_types:
        if figure_type == COSMIC:
            os.makedirs(os.path.join(plot_output_dir, REPLICATION_TIME, FIGURES_COSMIC), exist_ok=True)
        elif figure_type == MANUSCRIPT:
            os.makedirs(os.path.join(plot_output_dir, REPLICATION_TIME, FIGURES_MANUSCRIPT), exist_ok=True)
            plot_replication_time_legend(os.path.join(plot_output_dir, REPLICATION_TIME, FIGURES_MANUSCRIPT))

    os.makedirs(os.path.join(plot_output_dir, REPLICATION_TIME),exist_ok=True)
    os.makedirs(os.path.join(plot_output_dir, REPLICATION_TIME, EXCEL_FILES),exist_ok=True)
    os.makedirs(os.path.join(plot_output_dir, REPLICATION_TIME, PDF_FILES),exist_ok=True)
    os.makedirs(os.path.join(plot_output_dir, REPLICATION_TIME, DATA_FILES),exist_ok=True)

    # Main engine function
    # Accumulate data and plot accumulated COSMIC and MANUSCRIPT figure
    # Comparisons between cancer_type based result and aggregated result are done here
    # Slope cutoff is used here for DEC FLAT INC decision
    for figure_type in figure_types:
        if figure_type == MANUSCRIPT:
            # Remove artifacts
            # SBS mutational signatures attributed to artifacts
            # No DBS and ID mutational signatures attributed to artifacts
            # This works for occupancy analysis and replication time
            # Processivity walks through cancer_types
            sbs_signatures = list(set(all_sbs_signatures) - set(signatures_attributed_to_artifacts))
        elif figure_type == COSMIC:
            sbs_signatures = all_sbs_signatures

        signature_tuples = [(AGGREGATEDSUBSTITUTIONS, SBS), (AGGREGATEDDINUCS, DBS), (AGGREGATEDINDELS, ID)]
        for signature in sbs_signatures:
            signature_tuples.append((signature, SBS))
        for signature in dbs_signatures:
            signature_tuples.append((signature, DBS))
        for signature in id_signatures:
            signature_tuples.append((signature, ID))

        for number_of_mutations_required in number_of_mutations_required_list:
            all_signatures_replication_time_df = accumulateReplicationTimeAcrossAllTissues_plot_figure_for_all_types(
                plot_output_dir,
                combined_output_dir,
                signature_tuples,
                cancer_types,
                numberofSimulations,
                figure_type,
                number_of_mutations_required,
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

    # Fill PDF files for internal usage with number of mutations
    fill_replication_time_figures_pdfs(plot_output_dir,
                                       combined_output_dir,
                                       signature_tuples,
                                       cancer_types,
                                       all_signatures_replication_time_df)

    # Write Excel File
    # Same for Manuscript and Cosmic
    all_signatures_replication_time_df = all_signatures_replication_time_df[['signature',
                                                                            'cancer_type',
                                                                            'real_number_of_mutations_array',
                                                                            'real_normalized_mutations_density_array',
                                                                            'simulations_means_normalized_mutations_density_array',
                                                                            'cancer_type_slope',
                                                                            'replication_timing_diff_btw_max_and_min',
                                                                            'abs_replication_timing_diff_btw_medians',
                                                                            'cancer_type_decision',
                                                                            'cutoff',
                                                                            'number_of_mutations',
                                                                            'average_probability',
                                                                            'num_of_tissues_with_increasing_decision',
                                                                            'num_of_tissues_with_flat_decision',
                                                                            'num_of_tissues_with_decreasing_decision',
                                                                            'num_of_all_tissues']]

    excel_file_name = '%s_%s.xlsx' %(cosmic_release_version, COSMIC_REPLICATION_TIME)
    excel_file_path = os.path.join(plot_output_dir,REPLICATION_TIME,EXCEL_FILES,excel_file_name)
    df_list = [all_signatures_replication_time_df]
    sheet_list = ['Replication_Time']
    write_excel_file(df_list, sheet_list, excel_file_path)

    # Write Cosmic replication time data files
    # Signature based for across all cancer types and each cancer type
    for signature, signature_type in signature_tuples:
        data_file_name = '%s_%s_%s.txt' % (cosmic_release_version, signature, COSMIC_REPLICATION_TIME)
        data_file_path = os.path.join(plot_output_dir, REPLICATION_TIME, DATA_FILES, data_file_name)

        # signature_based_df = all_signatures_replication_time_df[ (all_signatures_replication_time_df['signature'] == signature) & (all_signatures_replication_time_df['cancer_type'] == ACROSS_ALL_CANCER_TYPES) ]
        signature_based_df = all_signatures_replication_time_df[ (all_signatures_replication_time_df['signature'] == signature) ]

        # write if there is a result to show up
        if len(signature_based_df) > 0 :
            with open(data_file_path, 'w') as f:
                # header line
                f.write("# Only cancer types with minimum 2000 mutations for SBS signatures and minimum 1000 mutations for DBS and ID signatures with average probability at least 0.75 are considered.\n")
                signature_based_df.to_csv(f, sep='\t', index=False)


def get_minimum_number_of_overlaps_required(signature,
                                            signature_tuples,
                                            minimum_number_of_overlaps_required_for_sbs,
                                            minimum_number_of_overlaps_required_for_dbs,
                                            minimum_number_of_overlaps_required_for_indels):

    minimum_number_of_overlaps_required = None

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


def fill_across_all_cancer_types_df(occupancy_df,
                                    signature_tuples,
                                    consider_both_real_and_sim_avg_overlap,
                                    depleted_fold_change,
                                    enriched_fold_change,
                                    minimum_number_of_overlaps_required_for_sbs,
                                    minimum_number_of_overlaps_required_for_dbs,
                                    minimum_number_of_overlaps_required_for_indels,
                                    occupancy_significance_level,
                                    pearson_spearman_correlation_cutoff):

    # Fill Across All Cancer Types
    # occupancy_df = pd.DataFrame(columns=["signature",
    #                                  "cancer_type",
    #                                  "file_name",
    #                                  "spearman_corr",
    #                                  "spearman_p_value",
    #                                  "spearman_q_value",
    #                                  "pearson_corr",
    #                                  "pearson_p_value",
    #                                  "pearson_q_value",
    #                                  "cutoff",
    #                                  "number_of_mutations",
    #                                  "average_probability",
    #                                  "average_number_of_overlaps"])

    across_all_occupancy_df = pd.DataFrame(columns=["signature",
                                                    "number_of_library_files",
                                                    "number_of_library_files_considered",
                                                    "num_of_all_cancer_types",
                                                    "num_of_all_cancer_types_with_considered",
                                                    "num_of_cancer_types_spearman_ge_cutoff",
                                                    "num_of_cancer_types_spearman_q_values_le_significance_level",
                                                    "num_of_cancer_types_pearson_ge_cutoff",
                                                    "num_of_cancer_types_pearson_q_values_le_significance_level"])

    signatures_array = occupancy_df['signature'].unique()
    for signature in signatures_array:
        minimum_number_of_overlaps_required = get_minimum_number_of_overlaps_required(signature,
                                                                                      signature_tuples,
                                                                                      minimum_number_of_overlaps_required_for_sbs,
                                                                                      minimum_number_of_overlaps_required_for_dbs,
                                                                                      minimum_number_of_overlaps_required_for_indels)
        cancer_types_array = occupancy_df[occupancy_df['signature'] == signature]['cancer_type'].unique()
        cancer_types_array = np.delete(cancer_types_array, np.where(cancer_types_array == ACROSS_ALL_CANCER_TYPES))

        library_files_array = occupancy_df[(occupancy_df['signature'] == signature) & (occupancy_df['cancer_type'] != ACROSS_ALL_CANCER_TYPES)]['file_name'].values
        if np.isin(library_files_array,str(np.nan)).any():
            library_files_array = np.delete(library_files_array, np.where(library_files_array == str(np.nan)))
        elif np.isin(library_files_array,np.nan).any():
            library_files_array = np.delete(library_files_array, np.where(library_files_array == np.nan))

        number_of_cancer_types = cancer_types_array.size
        number_of_library_files = library_files_array.size

        across_all_occupancy_df = across_all_occupancy_df.append(
            {"signature": signature,
             "number_of_library_files": number_of_library_files,
             "number_of_library_files_considered": 0,
             "num_of_all_cancer_types": number_of_cancer_types,
             "num_of_all_cancer_types_with_considered": 0,
             "num_of_cancer_types_spearman_ge_cutoff": 0,
             "num_of_cancer_types_spearman_q_values_le_significance_level": 0,
             "num_of_cancer_types_pearson_ge_cutoff": 0,
             "num_of_cancer_types_pearson_q_values_le_significance_level": 0}, ignore_index=True)

        for cancer_type in cancer_types_array:
            real_average_number_of_overlaps_array = occupancy_df[(occupancy_df['signature'] == signature) &
                                                                 (occupancy_df['cancer_type'] == cancer_type)]['real_average_number_of_overlaps'].values

            sims_average_number_of_overlaps_array = occupancy_df[(occupancy_df['signature'] == signature) &
                                                                 (occupancy_df['cancer_type'] == cancer_type)]['sims_average_number_of_overlaps'].values

            fold_change_array = occupancy_df[(occupancy_df['signature'] == signature) &
                                             (occupancy_df['cancer_type'] == cancer_type)]['fold_change'].values

            spearman_corr_array = occupancy_df[(occupancy_df['signature'] == signature) &
                                               (occupancy_df['cancer_type'] == cancer_type)]['spearman_corr'].values

            spearman_q_values_array = occupancy_df[(occupancy_df['signature'] == signature) &
                                                   (occupancy_df['cancer_type'] == cancer_type)]['spearman_q_value'].values

            pearson_corr_array = occupancy_df[(occupancy_df['signature'] == signature) &
                                              (occupancy_df['cancer_type'] == cancer_type)]['pearson_corr'].values

            pearson_q_values_array = occupancy_df[(occupancy_df['signature'] == signature) &
                                                  (occupancy_df['cancer_type'] == cancer_type)]['pearson_q_value'].values

            # update num_of_all_cancer_types_with_considered
            if consider_both_real_and_sim_avg_overlap:
                for real_data_avg_count, sim_data_avg_count, fold_change in zip(real_average_number_of_overlaps_array,
                                                                                sims_average_number_of_overlaps_array,
                                                                                fold_change_array):

                    if (real_data_avg_count >= minimum_number_of_overlaps_required) or \
                            (depleted(fold_change, depleted_fold_change) and (
                                    sim_data_avg_count >= minimum_number_of_overlaps_required)) or \
                            (enriched(fold_change, enriched_fold_change) and (
                                    sim_data_avg_count >= minimum_number_of_overlaps_required) and (
                                     real_data_avg_count >= minimum_number_of_overlaps_required * OCCUPANCY_HEATMAP_COMMON_MULTIPLIER)):
                        across_all_occupancy_df.loc[(across_all_occupancy_df[
                                                         'signature'] == signature), 'num_of_all_cancer_types_with_considered'] = \
                            across_all_occupancy_df[(across_all_occupancy_df['signature'] == signature)][
                                'num_of_all_cancer_types_with_considered'] + 1
                        break

            elif (len(np.argwhere(real_average_number_of_overlaps_array >= minimum_number_of_overlaps_required)) > 0):
                across_all_occupancy_df.loc[
                    (across_all_occupancy_df['signature'] == signature), 'num_of_all_cancer_types_with_considered'] = \
                    across_all_occupancy_df[(across_all_occupancy_df['signature'] == signature)][
                        'num_of_all_cancer_types_with_considered'] + 1

            # update number_of_library_files_considered
            if consider_both_real_and_sim_avg_overlap:
                for real_data_avg_count, sim_data_avg_count, fold_change in zip(real_average_number_of_overlaps_array,
                                                                                sims_average_number_of_overlaps_array,
                                                                                fold_change_array):

                    if (real_data_avg_count >= minimum_number_of_overlaps_required) or \
                            (depleted(fold_change, depleted_fold_change) and (
                                    sim_data_avg_count >= minimum_number_of_overlaps_required)) or \
                            (enriched(fold_change, enriched_fold_change) and (
                                    sim_data_avg_count >= minimum_number_of_overlaps_required) and (
                                     real_data_avg_count >= minimum_number_of_overlaps_required * OCCUPANCY_HEATMAP_COMMON_MULTIPLIER)):
                        across_all_occupancy_df.loc[(across_all_occupancy_df[
                                                         'signature'] == signature), 'number_of_library_files_considered'] = \
                            across_all_occupancy_df[(across_all_occupancy_df['signature'] == signature)][
                                'number_of_library_files_considered'] + 1

            else:
                for real_average_number_of_overlaps in real_average_number_of_overlaps_array:
                    if real_average_number_of_overlaps >= minimum_number_of_overlaps_required:
                        across_all_occupancy_df.loc[
                            (across_all_occupancy_df['signature'] == signature), 'number_of_library_files_considered'] = \
                            across_all_occupancy_df[(across_all_occupancy_df['signature'] == signature)][
                                'number_of_library_files_considered'] + 1

            # There is at least one file with corrected p-value le significance level
            if len(np.argwhere(spearman_corr_array >= pearson_spearman_correlation_cutoff)) > 0:
                across_all_occupancy_df.loc[(across_all_occupancy_df['signature'] == signature), 'num_of_cancer_types_spearman_ge_cutoff']=\
                    across_all_occupancy_df[(across_all_occupancy_df['signature'] == signature)]['num_of_cancer_types_spearman_ge_cutoff'] + 1

            if len(np.argwhere(spearman_q_values_array <= occupancy_significance_level)) > 0:
                across_all_occupancy_df.loc[(across_all_occupancy_df['signature']== signature), 'num_of_cancer_types_spearman_q_values_le_significance_level']=\
                    across_all_occupancy_df[(across_all_occupancy_df['signature'] == signature)]['num_of_cancer_types_spearman_q_values_le_significance_level'] + 1

            if len(np.argwhere(pearson_corr_array >= pearson_spearman_correlation_cutoff)) > 0:
                across_all_occupancy_df.loc[(across_all_occupancy_df['signature'] == signature), 'num_of_cancer_types_pearson_ge_cutoff']=\
                    across_all_occupancy_df[(across_all_occupancy_df['signature'] == signature)]['num_of_cancer_types_pearson_ge_cutoff'] + 1

            if len(np.argwhere(pearson_q_values_array <= occupancy_significance_level)) > 0:
                across_all_occupancy_df.loc[(across_all_occupancy_df['signature'] == signature), 'num_of_cancer_types_pearson_q_values_le_significance_level']=\
                    across_all_occupancy_df[(across_all_occupancy_df['signature'] == signature)]['num_of_cancer_types_pearson_q_values_le_significance_level'] + 1

    return across_all_occupancy_df


def generate_signature_based_occupancy_pdf(plot_output_dir,
                                        dna_elements,
                                        sbs_signatures,
                                        id_signatures,
                                        dbs_signatures,
                                       figure_file_extension):
    dir_name = 'signature_all_dna_elements_pdfs'

    deleteOldData(os.path.join(plot_output_dir, OCCUPANCY, dir_name))
    os.makedirs(os.path.join(plot_output_dir, OCCUPANCY, dir_name), exist_ok=True)
    all_signatures = []
    all_signatures.extend(sbs_signatures)
    all_signatures.extend(dbs_signatures)
    all_signatures.extend(id_signatures)
    all_signatures.extend(AGGREGATEDSUBSTITUTIONS)
    all_signatures.extend(AGGREGATEDDINUCS)
    all_signatures.extend(AGGREGATEDINDELS)

    for signature in all_signatures:
        fill_signature_all_dna_elements_pdfs(plot_output_dir,
                                             dir_name,
                                             signature,
                                             dna_elements)


def fill_signature_all_dna_elements_pdfs(output_dir, dir_name, signature, dna_elements):
    # for each signature
    interested_file_list = []
    for dna_element, dna_element_type in dna_elements:
        # SBS3_nucleosome_across_all_cancer_types_occupancy.png
        # SBS3_H3K27me3_across_all_cancer_types_occupancy.png
        filename = '%s_%s_%s.png' % (signature, dna_element, ACROSS_ALL_CANCER_TYPES_OCCUPANCY_FIGURE)
        # filepath = os.path.join(output_dir,OCCUPANCY,dna_element,FIGURES_COSMIC,filename)
        filepath = os.path.join(output_dir, OCCUPANCY, dna_element, FIGURES_MANUSCRIPT, filename)
        if os.path.exists(filepath):
            interested_file_list.append((filepath,dna_element))

    # One pdf for each signature, first left image: signature across all tissue prob05 --- first right image: signature across all tissues prob09
    # other images: signature for each tissue prob05 --- signature for each tissue prob09
    pdf_file_name = "%s_%s.pdf" % (signature,'all')
    pdf_file_path = os.path.join(output_dir, OCCUPANCY, dir_name, pdf_file_name)
    print(pdf_file_path)
    c = canvas.Canvas(pdf_file_path, pagesize=letter)
    width, height = letter  # keep for later
    print('canvas letter: width=%d height=%d' % (width, height))
    # width=612 height=792

    # Center header
    c.setFont("Times-Roman", 15)
    title = 'Occupancy' + ' ' + signature

    title_width = stringWidth(title, "Times-Roman", 15)
    c.drawString((width - title_width) / 2, height - 20, title)

    # One page can take 8 images
    # For images
    c.setFont("Times-Roman", 10)
    figureCount = 0

    first_figure_x = 60
    figure_left_x = 10
    figure_right_x = 310

    y = 570

    # For nucleosome occupancy pdfs
    if (len(interested_file_list)>1):
        for file, dna_element in interested_file_list:

            if (file==interested_file_list[0]):
                figure_width = 15

                img = utils.ImageReader(file)
                iw, ih = img.getSize()
                print('image: width=%d height=%d' % (iw, ih))
                aspect = ih / float(iw)

                # To the center
                c.drawImage(file, first_figure_x, y, figure_width * cm, figure_width * aspect * cm)
                figureCount = figureCount + 2
                y = y - 180

            elif os.path.exists(file):
                figure_width = 10

                img = utils.ImageReader(file)
                iw, ih = img.getSize()
                print('image: width=%d height=%d' % (iw, ih))
                aspect = ih / float(iw)
                print(file)
                figureCount = figureCount + 1

                # To the left
                if (figureCount % 2 == 1):
                    c.drawImage(file, figure_left_x, y, figure_width * cm, figure_width * aspect * cm)
                    c.setFont("Times-Roman", 18)
                    c.drawString(figure_left_x, y + 140, dna_element)

                # To the right
                elif (figureCount % 2 == 0):
                    c.drawImage(file, figure_right_x, y, figure_width * cm, figure_width * aspect * cm)
                    c.setFont("Times-Roman", 18)
                    c.drawString(figure_right_x, y + 140, dna_element)
                    y = y - 180

                if (figureCount % 8 == 0):
                    c.showPage()
                    c.setFont("Times-Roman", 10)
                    y = 570
        c.save()


def generate_occupancy_pdfs(plot_output_dir,
                            combined_output_dir,
                            occupancy_type,
                            dna_element,
                            cancer_types,
                            all_sbs_signatures,
                            id_signatures,
                            dbs_signatures,
                            numberofSimulations,
                            plus_minus_nucleosome,
                            plus_minus_epigenomics,
                            window_size,
                            consider_both_real_and_sim_avg_overlap,
                            minimum_number_of_overlaps_required_for_sbs,
                            minimum_number_of_overlaps_required_for_dbs,
                            minimum_number_of_overlaps_required_for_indels,
                            figure_types,
                            number_of_mutations_required_list_for_others,
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
                            figure_case_study):

    if dna_element == CTCF:
        number_of_mutations_required_list = number_of_mutations_required_list_for_ctcf
    else:
        number_of_mutations_required_list = number_of_mutations_required_list_for_others

    if occupancy_type == NUCLEOSOME_OCCUPANCY:
        plus_minus = plus_minus_nucleosome
    elif occupancy_type == EPIGENOMICS_OCCUPANCY:
        plus_minus = plus_minus_epigenomics

    start = int(plus_minus-window_size/2)
    end = int(plus_minus+window_size/2)

    deleteOldData(os.path.join(plot_output_dir, OCCUPANCY, dna_element))

    os.makedirs(os.path.join(plot_output_dir, OCCUPANCY, dna_element),exist_ok=True)
    os.makedirs(os.path.join(plot_output_dir, OCCUPANCY, dna_element, TABLES),exist_ok=True)
    os.makedirs(os.path.join(plot_output_dir, OCCUPANCY, dna_element, EXCEL_FILES),exist_ok=True)
    os.makedirs(os.path.join(plot_output_dir, OCCUPANCY, dna_element, PDF_FILES),exist_ok=True)
    os.makedirs(os.path.join(plot_output_dir, OCCUPANCY, dna_element, DATA_FILES),exist_ok=True)
    os.makedirs(os.path.join(plot_output_dir, OCCUPANCY, dna_element, MANUSCRIPT_TISSUE_BASED_FIGURES),exist_ok=True)
    os.makedirs(os.path.join(plot_output_dir, OCCUPANCY, dna_element, COSMIC_TISSUE_BASED_FIGURES),exist_ok=True)
    os.makedirs(os.path.join(plot_output_dir, OCCUPANCY, dna_element, FIGURES_COSMIC),exist_ok=True)
    os.makedirs(os.path.join(plot_output_dir, OCCUPANCY, dna_element, FIGURES_MANUSCRIPT),exist_ok=True)

    # for testing/debugging
    # signature_tuples = []

    for figure_type in figure_types:
        if figure_type == MANUSCRIPT:
            # SBS mutational signatures attributed to artifacts
            # No DBS and ID mutational signatures attributed to artifacts
            # This works for occupancy analysis and replication time
            # Processivity walks through cancer_types
            sbs_signatures = list(set(all_sbs_signatures) - set(signatures_attributed_to_artifacts))
        elif figure_type == COSMIC:
            sbs_signatures = all_sbs_signatures

        # uncomment for real run
        signature_tuples = [(AGGREGATEDSUBSTITUTIONS, SBS), (AGGREGATEDDINUCS, DBS), (AGGREGATEDINDELS, ID)]

        for signature in sbs_signatures:
            signature_tuples.append((signature, SBS))
        for signature in dbs_signatures:
            signature_tuples.append((signature, DBS))
        for signature in id_signatures:
            signature_tuples.append((signature, ID))

        for number_of_mutations_required in number_of_mutations_required_list:
            occupancy_df = accumulateOccupancyAcrossAllCancerTypes_plot_figure_for_signatures(plot_output_dir,
                                                                                              combined_output_dir,
                                                                                              occupancy_type,
                                                                                              dna_element,
                                                                                              signature_tuples,
                                                                                              cancer_types,
                                                                                              numberofSimulations,
                                                                                              plus_minus,
                                                                                              start,
                                                                                              end,
                                                                                              consider_both_real_and_sim_avg_overlap,
                                                                                              minimum_number_of_overlaps_required_for_sbs,
                                                                                              minimum_number_of_overlaps_required_for_dbs,
                                                                                              minimum_number_of_overlaps_required_for_indels,
                                                                                              figure_type,
                                                                                              number_of_mutations_required,
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

    # Write occupancy tables dataframes using latest figure_type in figure_types
    filename = "Combined_PCAWG_nonPCAWG_%s_Cancer_Type_Based.txt" % (dna_element)
    filepath = os.path.join(plot_output_dir, OCCUPANCY, dna_element, TABLES, filename)
    occupancy_df.to_csv(filepath, sep='\t', header=True, index=False)

    across_all_occupancy_df = fill_across_all_cancer_types_df(occupancy_df,
                                                              signature_tuples,
                                                              consider_both_real_and_sim_avg_overlap,
                                                              depleted_fold_change,
                                                              enriched_fold_change,
                                                              minimum_number_of_overlaps_required_for_sbs,
                                                              minimum_number_of_overlaps_required_for_dbs,
                                                              minimum_number_of_overlaps_required_for_indels,
                                                              occupancy_significance_level,
                                                              pearson_spearman_correlation_cutoff)

    filename = "Combined_PCAWG_nonPCAWG_%s_Across_All_Cancer_Types.txt" % (dna_element)
    filepath = os.path.join(plot_output_dir, OCCUPANCY, dna_element, TABLES, filename)
    across_all_occupancy_df.to_csv(filepath, sep='\t', header=True, index=False)

    if dna_element == CTCF:
        feature_name = COSMIC_CTCF_OCCUPANCY
    elif dna_element == NUCLEOSOME:
        feature_name = COSMIC_NUCLEOSOME_OCCUPANCY
    else:
        feature_name = dna_element + '_' + COSMIC_OCCUPANCY

    # Consider only some columns and reorder them
    occupancy_df = occupancy_df[["signature", "cancer_type",
                                 "cutoff", "number_of_mutations", "average_probability", "file_name",
                                 # "is_eligible",
                                 "pearson_corr", "pearson_p_value", "pearson_q_value",
                                 "real_average_number_of_overlaps", "sims_average_number_of_overlaps",
                                 "avg_real_signal", "sim_data_avg_signal", "fold_change",
                                 "real_average_signal_array", "sims_average_signal_array"]]

    # Consider only some columns and reorder them
    across_all_occupancy_df = across_all_occupancy_df[['signature',
                                                       'number_of_library_files',
                                                       'number_of_library_files_considered',
                                                       'num_of_all_cancer_types',
                                                       'num_of_all_cancer_types_with_considered',
                                                       'num_of_cancer_types_pearson_ge_cutoff',
                                                       'num_of_cancer_types_pearson_q_values_le_significance_level']]

    # Write occupancy excel files using latest figure_type in figure_types
    excel_file_name = "%s_%s.xlsx" % (cosmic_release_version, feature_name)
    excel_file_path = os.path.join(plot_output_dir, OCCUPANCY, dna_element, EXCEL_FILES, excel_file_name)
    df_list = [occupancy_df, across_all_occupancy_df]
    sheet_list = ['Cancer_Type_Based', 'Across_All_Cancer_Types']
    write_excel_file(df_list, sheet_list, excel_file_path)

    # Write COSMIC occupancy data files using latest figure_type in figure_types
    # Signature based for across all cancer types and each cancer type
    for signature, signature_type in signature_tuples:
        # v3.2_SBS1_REPLIC_ASYM_TA_C4817.jpg
        # NCI_Thesaurus_code = cancer_type_2_NCI_Thesaurus_code_dict[cancer_type]
        data_file_name = '%s_%s_%s.txt' % (cosmic_release_version, signature, feature_name)
        data_file_path = os.path.join(plot_output_dir, OCCUPANCY, dna_element, DATA_FILES, data_file_name)

        signature_based_df = occupancy_df[occupancy_df['signature'] == signature]

        # write if there is a result to show up and if there is a result for ACROSS_ALL_CANCER_TYPES
        if (len(signature_based_df) > 0) and \
                (signature_based_df[signature_based_df['cancer_type'] == ACROSS_ALL_CANCER_TYPES].values.any()):
            # header line
            with open(data_file_path, 'w') as f:
                f.write("# Only cancer types with minimum 2000 mutations for SBS signatures and minimum 1000 mutations for DBS and ID signatures with average probability at least 0.75 are considered.\n")
                signature_based_df.to_csv(f, sep='\t', index=False)

        # might be confusing to end user
        # signature_based_df = across_all_occupancy_df[across_all_occupancy_df['signature'] == signature]
        # if len(signature_based_df)>0:
        #     signature_based_df.to_csv(data_file_path, sep='\t', index=False, mode='a')



    # Write occupancy pfds using latest figure_type in figure_types
    fill_occupancy_pdfs(plot_output_dir,
                        occupancy_type,
                        dna_element,
                        signature_tuples,
                        cancer_types,
                        figure_type,
                        cosmic_release_version,
                        figure_file_extension)


# sheet name must be less than 31 characters
def write_excel_file(df_list, sheet_list, file_name):
    writer = pd.ExcelWriter(file_name,engine='xlsxwriter')
    for dataframe, sheet in zip(df_list, sheet_list):
        dataframe.to_excel(writer, sheet_name=sheet, startrow=0 , startcol=0, index=False)
    writer.save()


# copied from Figure_Case_Study_SBS4
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

# COSMIC ACROSS ALL TISSUES and TISSUE BASED TOGETHER
# Plot Figure
# Plot ColorBar
# Plot Legend
def plot_cosmic_processivity_figure_across_all_tissues_and_tissue_based_together(plot_output_path,
                                     all_cancer_types_processivity_df,
                                     across_all_cancer_types_pooled_processivity_df,
                                     cosmic_release_version,
                                     figure_file_extension,
                                     signature,
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

    if len(processive_group_length_list)>0:
        max_processive_group_length = max(processive_group_length_list)

        index = None
        if ((len(processive_group_length_list)>0) and (max_processive_group_length>0)):
            # Find index of max_processive_group_length in processive_group_length_array
            index = processive_group_length_list.index(max_processive_group_length)

        width_multiply = 2
        height_multiply = 2.5

        width = int(width_multiply * len(processive_group_length_list))
        height = int(height_multiply * len(signature_list))

        fig = plt.figure(figsize=(width, height))
        fontsize = 35

        # Initialize GridSpec
        grid = plt.GridSpec(ncols=width, nrows=height, hspace=0, wspace=0) # (width, height) (num_of_cols, num_of_rows)
        ax = fig.add_subplot(grid[:, :])  # First axis: Grid with circles
        ax.set_aspect(1.0)

        cmap = matplotlib_cm.get_cmap('YlOrRd')  # Looks better
        v_min = 2
        v_max = 20
        norm = plt.Normalize(v_min, v_max)

        # To get rid of  UserWarning: Attempting to set identical left==right results in singular transformations; automatically expanding.
        if (len(processive_group_length_list) > 1):
            plt.xlim([1,index+1])
            ax.set_xticks(np.arange(0,index+2,1))
        else:
            plt.xlim([0,len(processive_group_length_list)])
            ax.set_xticks(np.arange(0,len(processive_group_length_list)+1,1))

        if (len(signature_list) > 1):
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

        xlabels=None
        if (index is not None):
            xlabels = processive_group_length_list[0:index+1]
        ylabels = rows_signatures_on_the_heatmap

        # CODE GOES HERE TO CENTER X-AXIS LABELS...
        ax.set_xticklabels([])
        mticks = ax.get_xticks()
        ax.set_xticks((mticks[:-1] + mticks[1:]) / 2, minor=True)
        ax.tick_params(axis='x', which='minor', length=0, labelsize=fontsize)
        if xlabels is not None:
            ax.set_xticklabels(xlabels, minor=True)
            ax.set_xlabel(STRAND_COORDINATED_MUTAGENESIS_GROUP_LENGTH,fontsize=fontsize, labelpad=20)
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

        divider = make_axes_locatable(ax)

        # Legend
        legend_ax = divider.append_axes("bottom", size=0.75, pad=0.1)
        plot_processivity_legend_in_given_axis(legend_ax, fontsize / 2)

        # Color Bar
        bounds = np.arange(v_min, v_max + 1, 2)
        if len(signature_list) >= 15:
            pad = 0.02
            # width = 0.03
            pos = ax.get_position()
            height = 0.3 * (pos.ymax - pos.ymin)
            midpoint = 0.5 * (pos.ymax + pos.ymin)
            y = midpoint - height / 2
            width = 0.02 * (pos.xmax - pos.xmin)
            cax = fig.add_axes([pos.xmax + pad, y, width, height])
        else:
            cax = divider.append_axes("right", size=0.2, pad=0.1)

        cax.grid(False)
        cax.set_xticks([])
        cax.set_yticks([])
        for edge, spine in cax.spines.items():
            spine.set_visible(False)

        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=bounds, spacing='proportional', orientation='vertical')
        cb.ax.tick_params(labelsize=fontsize / 2)
        cb.set_label("-log10\n(q-value)", verticalalignment='center', rotation=0, labelpad=60, fontsize=fontsize / 2)

        figures_path = os.path.join(plot_output_path, PROCESSIVITY, COSMIC_ACROSS_ALL_AND_TISSUE_BASED_TOGETHER)
        filename = '%s_%s_%s.%s' % (cosmic_release_version, signature, COSMIC_PROCESSIVITY, figure_file_extension)
        figFile = os.path.join(figures_path, filename)
        fig.savefig(figFile, dpi=100, bbox_inches="tight")

        # plot colorbar in a separate figure
        plot_processivity_colorbar_vertical(figures_path, cmap, v_min, v_max)
        plot_processivity_colorbar_horizontal(figures_path, cmap, v_min, v_max)
        # plot legend in a separate figure
        plot_processivity_legend(figures_path, 30)

        plt.cla()
        plt.close(fig)

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
         occupancy = True,
         plot_occupancy = True,
         replication_time = True,
         processivity = True,
         figure_types = [MANUSCRIPT, COSMIC],
         number_of_mutations_required_list_for_others = [AT_LEAST_1K_CONSRAINTS],
         number_of_mutations_required_list_for_ctcf = [AT_LEAST_1K_CONSRAINTS],
         consider_both_real_and_sim_avg_overlap = True,
         minimum_number_of_overlaps_required_for_sbs = 100,
         minimum_number_of_overlaps_required_for_dbs = 100,
         minimum_number_of_overlaps_required_for_indels = 100,
         number_of_simulations = 100,
         depleted_fold_change = 0.95,
         enriched_fold_change = 1.05,
         occupancy_significance_level = 0.05,
         replication_time_significance_level = 0.05,
         replication_time_slope_cutoff = 0.020,
         replication_time_difference_between_min_and_max = 0.2,
         replication_time_difference_between_medians = 0.135,
         processivity_significance_level = 0.05,
         minimum_required_processive_group_length = 4,
         minimum_required_number_of_processive_groups = 2,
         pearson_spearman_correlation_cutoff = 0.5,
         cosmic_release_version = 'v3.2',
         figure_file_extension = 'jpg'):

    os.makedirs(output_dir, exist_ok=True)

    # For real run include alll signatures
    sbs_signatures = ['SBS1', 'SBS2', 'SBS3', 'SBS4','SBS5', 'SBS6', 'SBS7a', 'SBS7b', 'SBS7c', 'SBS7d', 'SBS8',
                      'SBS9', 'SBS10a', 'SBS10b', 'SBS10c', 'SBS11', 'SBS12', 'SBS13', 'SBS14', 'SBS15','SBS16',
                      'SBS17a', 'SBS17b', 'SBS18', 'SBS19', 'SBS20', 'SBS21','SBS22', 'SBS23', 'SBS24', 'SBS25',
                      'SBS26', 'SBS27', 'SBS28', 'SBS29', 'SBS30', 'SBS31', 'SBS32', 'SBS33', 'SBS34', 'SBS35',
                      'SBS36', 'SBS37', 'SBS38', 'SBS39', 'SBS40', 'SBS41', 'SBS42','SBS43', 'SBS44', 'SBS45',
                      'SBS46', 'SBS47', 'SBS48', 'SBS49','SBS50', 'SBS51', 'SBS52', 'SBS53', 'SBS54', 'SBS55',
                      'SBS56', 'SBS57', 'SBS58', 'SBS59', 'SBS60', 'SBS84', 'SBS85']

    dbs_signatures = ['DBS1', 'DBS2', 'DBS3', 'DBS4','DBS5', 'DBS6', 'DBS7', 'DBS8', 'DBS9', 'DBS10', 'DBS11']

    id_signatures = ['ID1','ID2', 'ID3', 'ID4', 'ID5', 'ID6', 'ID7', 'ID8', 'ID9', 'ID10','ID11', 'ID12', 'ID13',
                     'ID14', 'ID15', 'ID16', 'ID17']

    # # For testing/debugging have one signature from each type
    # sbs_signatures = ['SBS31']
    # dbs_signatures = []
    # id_signatures = []

    # These are the 40 tissues for combined PCAWG and nonPCAWG + ESCC
    cancer_types = ['ALL', 'Bladder-TCC', 'Bone-Benign', 'Bone-Osteosarc', 'CNS-GBM', 'CNS-Medullo', 'CNS-PiloAstro',
                  'ColoRect-AdenoCA', 'Ewings', 'Head-SCC', 'Kidney-RCC', 'Lung-AdenoCA', 'Myeloid-AML',
                  'Myeloid-MPN', 'Panc-AdenoCA', 'Prost-AdenoCA', 'SoftTissue-Leiomyo', 'Stomach-AdenoCA',
                  'Uterus-AdenoCA', 'Biliary-AdenoCA', 'Blood-CMDI', 'Bone-Epith', 'Breast-Cancer', 'CNS-LGG',
                  'CNS-Oligo', 'Cervix-Cancer', 'Eso-AdenoCA', 'ESCC', 'Eye-Melanoma', 'Kidney-ChRCC', 'Liver-HCC',
                  'Lung-SCC', 'Lymph-BNHL', 'Lymph-CLL', 'Myeloid-MDS', 'Ovary-AdenoCA', 'Panc-Endocrine', 'Skin-Melanoma',
                  'SoftTissue-Liposarc', 'Thy-AdenoCA']

    # # for testing purposes
    # cancer_types = ['Lymph-BNHL', 'Lymph-CLL']

    print('--- len(sbs_signatures):', len(sbs_signatures))
    print('--- sbs_signatures:', sbs_signatures)
    print('--- len(dbs_signatures):', len(dbs_signatures))
    print('--- dbs_signatures:', dbs_signatures)
    print('--- len(id_signatures):', len(id_signatures))
    print('--- id_signatures:', id_signatures)
    print('--- len(cancer_types):', len(cancer_types))
    print('--- cancer_types:', cancer_types)

    plus_minus_nucleosome = 1000
    plus_minus_epigenomics = 1000
    window_size = 100

    # For real run
    dna_elements = [(NUCLEOSOME, NUCLEOSOME_OCCUPANCY),
                    (CTCF, EPIGENOMICS_OCCUPANCY),
                    (ATAC_SEQ, EPIGENOMICS_OCCUPANCY),
                    (H3K4me1, EPIGENOMICS_OCCUPANCY),
                    (H3K4me2, EPIGENOMICS_OCCUPANCY),
                    (H3K4me3, EPIGENOMICS_OCCUPANCY),
                    (H3K9ac, EPIGENOMICS_OCCUPANCY),
                    (H3K27ac, EPIGENOMICS_OCCUPANCY),
                    (H3K36me3, EPIGENOMICS_OCCUPANCY),
                    (H3K79me2, EPIGENOMICS_OCCUPANCY),
                    (H4K20me1, EPIGENOMICS_OCCUPANCY),
                    (H2AFZ, EPIGENOMICS_OCCUPANCY),
                    (H3K9me3, EPIGENOMICS_OCCUPANCY),
                    (H3K27me3, EPIGENOMICS_OCCUPANCY)]

    # dna_elements = [(NUCLEOSOME, NUCLEOSOME_OCCUPANCY), (CTCF, EPIGENOMICS_OCCUPANCY)] # for testing/debugging
    # dna_elements = [(NUCLEOSOME, NUCLEOSOME_OCCUPANCY)] # for testing/debugging
    # dna_elements = [(CTCF, EPIGENOMICS_OCCUPANCY)] # for testing/debugging
    # dna_elements = [(H3K27ac, EPIGENOMICS_OCCUPANCY)]  # for testing/debugging

    if replication_time:
        cosmic_legend = True
        cosmic_signature = True
        cosmic_fontsize_text = 20
        cosmic_cancer_type_fontsize = 20/3
        cosmic_fontweight = 'semibold'
        cosmic_fontsize_labels = 10
        sub_figure_type = None

        generate_replication_time_pdfs(output_dir,
                                       input_dir,
                                       cancer_types,
                                       sbs_signatures,
                                       id_signatures,
                                       dbs_signatures,
                                       number_of_simulations,
                                       figure_types,
                                       number_of_mutations_required_list_for_others,
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
        cosmic_legend = True
        cosmic_correlation_text = True
        cosmic_labels = True,
        cancer_type_on_right_hand_side = True
        cosmic_fontsize_text = 20
        cosmic_fontsize_ticks = 20
        cosmic_fontsize_labels = 20
        cosmic_linewidth_plot = 5
        cosmic_title_all_cancer_types = False
        figure_case_study = None

        for (dna_element, occupancy_type) in dna_elements:
            generate_occupancy_pdfs(output_dir,
                                    input_dir,
                                    occupancy_type,
                                    dna_element,
                                    cancer_types,
                                    sbs_signatures,
                                    id_signatures,
                                    dbs_signatures,
                                    number_of_simulations,
                                    plus_minus_nucleosome,
                                    plus_minus_epigenomics,
                                    window_size,
                                    consider_both_real_and_sim_avg_overlap,
                                    minimum_number_of_overlaps_required_for_sbs,
                                    minimum_number_of_overlaps_required_for_dbs,
                                    minimum_number_of_overlaps_required_for_indels,
                                    figure_types,
                                    number_of_mutations_required_list_for_others,
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

    # No new analysis
    # This part ony generate signature based pdf for all DNA elements occupancy analysis to see all together
    if (plot_occupancy):
        generate_signature_based_occupancy_pdf(output_dir,
                                                dna_elements,
                                                sbs_signatures,
                                                id_signatures,
                                                dbs_signatures,
                                                figure_file_extension)

    if processivity:
        # MANUSCRIPT All Signatures across all tissues figure
        # COSMIC across all tissues and tissue based figures
        # signatures attributed to artifacts are handled within generate_processivity_pdf
        # in function plot_processivity_across_all_tissues
        all_cancer_types_processivity_df, across_all_cancer_types_pooled_processivity_df = generate_processivity_pdf(output_dir,
                                  input_dir,
                                  cancer_types,
                                  number_of_simulations,
                                  figure_types,
                                  cosmic_release_version,
                                  figure_file_extension,
                                  processivity_significance_level,
                                  minimum_required_processive_group_length,
                                  minimum_required_number_of_processive_groups)


        # COSMIC Across All Tissues and Tissue Based Together Figures
        signature2cancer_type_list_dict = get_signature2cancer_type_list_dict(input_dir, cancer_types)

        for signature in sbs_signatures:
            signature_tissue_type_tuples, \
            signatures_ylabels_on_the_heatmap = fill_lists(signature, signature2cancer_type_list_dict)

            signature_tissue_type_tuples = list(reversed(signature_tissue_type_tuples))
            signatures_ylabels_on_the_heatmap = list(reversed(signatures_ylabels_on_the_heatmap))

            # New Figure for COSMIC
            plot_cosmic_processivity_figure_across_all_tissues_and_tissue_based_together(output_dir,
                                             all_cancer_types_processivity_df,
                                             across_all_cancer_types_pooled_processivity_df,
                                             cosmic_release_version,
                                             figure_file_extension,
                                             signature,
                                             signature_tissue_type_tuples,
                                             signatures_ylabels_on_the_heatmap,
                                             minimum_required_number_of_processive_groups)


        # Set dtype as float for object types
        all_cancer_types_processivity_df['log10_number_of_processive_groups'] = all_cancer_types_processivity_df['log10_number_of_processive_groups'].astype(np.float64)
        all_cancer_types_processivity_df['radius'] = all_cancer_types_processivity_df['radius'].astype(np.float64)
        all_cancer_types_processivity_df['avg_sims'] = all_cancer_types_processivity_df['avg_sims'].astype(np.float64)

        # Round
        all_cancer_types_processivity_df['log10_number_of_processive_groups'] = np.around(all_cancer_types_processivity_df['log10_number_of_processive_groups'], NUMBER_OF_DECIMAL_PLACES_TO_ROUND)
        all_cancer_types_processivity_df['radius'] = np.around(all_cancer_types_processivity_df['radius'], NUMBER_OF_DECIMAL_PLACES_TO_ROUND)
        all_cancer_types_processivity_df['avg_sims'] = np.around(all_cancer_types_processivity_df['avg_sims'], NUMBER_OF_DECIMAL_PLACES_TO_ROUND)
        all_cancer_types_processivity_df['mean_sims'] = np.around(all_cancer_types_processivity_df['mean_sims'], NUMBER_OF_DECIMAL_PLACES_TO_ROUND)
        all_cancer_types_processivity_df['std_sims'] = np.around(all_cancer_types_processivity_df['std_sims'], NUMBER_OF_DECIMAL_PLACES_TO_ROUND)
        all_cancer_types_processivity_df['minus_log10_qvalue'] = np.around(all_cancer_types_processivity_df['minus_log10_qvalue'], NUMBER_OF_DECIMAL_PLACES_TO_ROUND)

        # No need to show all columns
        all_cancer_types_processivity_df = all_cancer_types_processivity_df[['signature', 'cancer_type',
                                                                             'processive_group_length',
                                                                             'number_of_processive_groups',
                                                                             'log10_number_of_processive_groups',
                                                                             'radius',
                                                                             # 'avg_sims',
                                                                             'min_sims', 'max_sims', 'mean_sims',
                                                                             'std_sims',
                                                                             'pvalue', 'qvalue', 'minus_log10_qvalue',
                                                                             # 'zscore',
                                                                             'expected_number_of_processive_groups']]

        # Round
        across_all_cancer_types_pooled_processivity_df['avg_number_of_processive_groups'] = np.around(across_all_cancer_types_pooled_processivity_df['avg_number_of_processive_groups'], NUMBER_OF_DECIMAL_PLACES_TO_ROUND)
        across_all_cancer_types_pooled_processivity_df['log10_avg_number_of_processive_groups'] = np.around(across_all_cancer_types_pooled_processivity_df['log10_avg_number_of_processive_groups'], NUMBER_OF_DECIMAL_PLACES_TO_ROUND)
        across_all_cancer_types_pooled_processivity_df['radius'] = np.around(across_all_cancer_types_pooled_processivity_df['radius'], NUMBER_OF_DECIMAL_PLACES_TO_ROUND)
        across_all_cancer_types_pooled_processivity_df['avg_sims'] = np.around(across_all_cancer_types_pooled_processivity_df['avg_sims'], NUMBER_OF_DECIMAL_PLACES_TO_ROUND)
        across_all_cancer_types_pooled_processivity_df['mean_sims'] = np.around(across_all_cancer_types_pooled_processivity_df['mean_sims'], NUMBER_OF_DECIMAL_PLACES_TO_ROUND)
        across_all_cancer_types_pooled_processivity_df['std_sims'] = np.around(across_all_cancer_types_pooled_processivity_df['std_sims'], NUMBER_OF_DECIMAL_PLACES_TO_ROUND)
        across_all_cancer_types_pooled_processivity_df['minus_log10_qvalue'] = np.around(across_all_cancer_types_pooled_processivity_df['minus_log10_qvalue'], NUMBER_OF_DECIMAL_PLACES_TO_ROUND)

        # No need to show all columns
        across_all_cancer_types_pooled_processivity_df = across_all_cancer_types_pooled_processivity_df[
            ['signature', 'cancer_type',
             'processive_group_length',
             'avg_number_of_processive_groups',
             'log10_avg_number_of_processive_groups', 'radius',
             # 'avg_sims',
             'min_sims', 'max_sims', 'mean_sims', 'std_sims',
             'pvalue', 'qvalue', 'minus_log10_qvalue',
             # 'zscore',
             'expected_avg_number_of_processive_groups']]

        # Write excel files
        excel_file_name = '%s_%s.xlsx' % (cosmic_release_version, COSMIC_PROCESSIVITY)
        excel_file_path = os.path.join(output_dir, PROCESSIVITY, EXCEL_FILES, excel_file_name)
        df_list = [all_cancer_types_processivity_df, across_all_cancer_types_pooled_processivity_df]
        sheet_list = ['Cancer_Type_Based', 'Across_All_Cancer_Types']
        write_excel_file(df_list, sheet_list, excel_file_path)

        # Write COSMIC processivity data files
        # Signature based for across all cancer types and each cancer type
        signature_array = across_all_cancer_types_pooled_processivity_df['signature'].unique()
        for signature in signature_array:
            data_file_name = '%s_%s_%s.txt' % (cosmic_release_version, signature, COSMIC_PROCESSIVITY)
            data_file_path = os.path.join(output_dir, PROCESSIVITY, DATA_FILES, data_file_name)
            if signature not in ['SBS288P']:
                signature_based_df = all_cancer_types_processivity_df[
                    (all_cancer_types_processivity_df['signature'] == signature)]
                if len(signature_based_df) > 0:
                    # header line
                    with open(data_file_path, 'w') as f:
                        f.write(
                            "# Only cancer types with minimum 2000 mutations for SBS signatures with average probability at least 0.75 are considered.\n")
                        signature_based_df.to_csv(f, sep='\t', index=False)

                signature_based_df = across_all_cancer_types_pooled_processivity_df[
                    (across_all_cancer_types_pooled_processivity_df['signature'] == signature)]
                if len(signature_based_df) > 0:
                    with open(data_file_path, 'a') as f:
                        f.write('\n')
                        signature_based_df.to_csv(f, sep='\t', index=False)


def plot_processivity_legend(path, fontsize):
    fig, ax = plt.subplots(figsize=(10, 3))
    plot_processivity_legend_in_given_axis(ax, fontsize)

    filename = 'normalized_number_of_processive_group.png'
    figureFile = os.path.join(path, filename)

    fig.savefig(figureFile)
    plt.close()


# For SigProfilerTopography Manuscript Processivity Legend
# For COSMIC website
def plot_processivity_legend_in_given_axis(ax, fontsize):
    diameter_labels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    row_labels = ['circle']
    ax.grid(which="major", color="white", linestyle='-', linewidth=3)

    # make aspect ratio square
    ax.set_aspect(1.0)

    for row_index, row_label in enumerate(row_labels):
        for diameter_index, diameter_label in enumerate(diameter_labels):
            circle=plt.Circle((diameter_index + 0.5, row_index + 0.5), radius=(diameter_label/(2*1.09)), color='gray', fill=True)
            ax.add_artist(circle)

    # CODE GOES HERE TO CENTER X-AXIS LABELS...
    ax.set_xlim([0, len(diameter_labels)])
    ax.set_xticklabels([])

    ax.tick_params(axis='x', which='minor', length=0, labelsize=12)
    # major ticks
    ax.set_xticks(np.arange(0, len(diameter_labels), 1))
    ax.xaxis.set_ticks_position('bottom')

    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='major',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False)  # labels along the bottom edge are off

    ax.set_xlabel('Normalized number of groups with specific\n length of strand-coordinated mutations', fontsize=fontsize, labelpad=5)

    # CODE GOES HERE TO CENTER Y-AXIS LABELS...
    ax.set_ylim([0, len(row_labels)])
    ax.set_yticklabels([])

    ax.tick_params(axis='y', which='minor', length=0, labelsize=12)
    # major ticks
    ax.set_yticks(np.arange(0, len(row_labels), 1))

    ax.tick_params(
        axis='y',  # changes apply to the x-axis
        which='major',  # both major and minor ticks are affected
        left=False)  # labels along the bottom edge are off

    ax.grid(which='major', color='black', linestyle='-', linewidth=1)







