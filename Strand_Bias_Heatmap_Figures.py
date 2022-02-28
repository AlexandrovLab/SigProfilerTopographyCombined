# Feb 5 2021
# Provide heatmaps using excel files or tables text files
# where rows mutational signatures, columns cancer types for Combined PCAWG and nonPCAWG analysis


import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import re

LEADING= 'Leading'
LAGGING = 'Lagging'

GENIC='Genic'
INTERGENIC='Intergenic'

UNTRANSCRIBED_STRAND = 'UnTranscribed'
TRANSCRIBED_STRAND = 'Transcribed'
NONTRANSCRIBED_STRAND = 'NonTranscribed'

transcription_strands = [TRANSCRIBED_STRAND, UNTRANSCRIBED_STRAND]
genicVersusIntergenic_strands=[GENIC, INTERGENIC]
replication_strands = [LAGGING, LEADING]

TRANSCRIPTIONSTRANDBIAS = 'transcription_strand_bias'
REPLICATIONSTRANDBIAS = 'replication_strand_bias'
GENICVERSUSINTERGENIC = 'genic_versus_intergenic'

input_dir = os.path.join("C:\\","Users","burcak","Documents","AlexandrovLab","BurcakOtlu_Papers","Topography_of_Mutational_Processes_In_Human_Cancer","AI_Figures_Combined_PCAWG_nonPCWG","4th_iteration","Figure_2_strand_bias","tables")
output_dir = os.path.join("C:\\","Users","burcak","Documents","AlexandrovLab","BurcakOtlu_Papers","Topography_of_Mutational_Processes_In_Human_Cancer")


########################################################
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
########################################################


###################################################################
def heatmap(data, row_labels, col_labels,ax=None, fontsize=None,cbar_kw={}, cbarlabel="", **kwargs):
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

    #nans are handled here
    data = np.ma.masked_invalid(data)
    ax.patch.set(hatch='x', edgecolor='black')

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",fontsize=fontsize,labelpad=25)
    cbar.ax.tick_params(labelsize=fontsize)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.

    ax.set_xticklabels(col_labels,fontsize=fontsize)
    ax.set_yticklabels(row_labels,fontsize=fontsize)

    # Let the x axes labeling appear on bottom.
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",rotation_mode="anchor")
    plt.setp(ax.get_xticklabels(), rotation=75, ha="right",rotation_mode="anchor")

    # Turn spines off and create white grid.
    # for edge, spine in ax.spines.items():
    # spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)

    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.grid(b=False, which="major")

    return im, cbar
###################################################################

################################################################################
if __name__ == "__main__":

    percent = "20%"

    type_file_path=os.path.join(input_dir,"Type_Strand_Cancer_Types_Percentages_Table.txt")
    type_df=pd.read_csv(type_file_path, sep="\t")

    print(type_df)
    print(type_df['strand'])
    print(type_df.columns.values)

    rows_np_array=type_df['my_type'].unique()
    print(rows_np_array)
    print(type(rows_np_array))
    print(len(rows_np_array))

    # np.array(rows_np_array)
    rows_types_list = sorted(rows_np_array, key=natural_key)
    columns_signatures_list=[]

    cancer_types_np_array=type_df['10%'].unique()
    print(cancer_types_np_array)
    print(cancer_types_np_array.size)


    strands_list=[transcription_strands,replication_strands, genicVersusIntergenic_strands]

    cancer_type_set=set()

    for strands_index, strands in enumerate(strands_list):
        if strands_index==0:
            strand_bias=TRANSCRIPTIONSTRANDBIAS
            colors = ["yellowgreen", "white", "royalblue"]
        elif strands_index==1:
            strand_bias=REPLICATIONSTRANDBIAS
            colors = [ "goldenrod", "white", "indianred"]
        elif strands_index==2:
            strand_bias=GENICVERSUSINTERGENIC
            colors = [ "gray", "white", "cyan"]


        for strand in strands:
            for my_type in rows_types_list:
                if type_df[(type_df['my_type']==my_type) & (type_df['strand']==strand)][percent].any():
                    cancer_types_np_array=type_df[(type_df['my_type']==my_type) & (type_df['strand']==strand)][percent].values[0]
                    cancer_types_list=eval(cancer_types_np_array)
                    for cancer_type in cancer_types_list:
                        cancer_type_set.add(cancer_type)

        # cancer_type_list=list(cancer_type_set)
        cancer_types_list = sorted(cancer_type_set, key=natural_key)
        print('cancer_types_list:%s' %cancer_types_list)

        print(rows_types_list)
        print(len(rows_types_list))

        rows_types_columns_cancer_types_np_array=np.zeros((len(rows_types_list), len(cancer_types_list)))
        print(rows_types_columns_cancer_types_np_array)

        boundaries=[-1,0,1]
        norm = matplotlib.colors.BoundaryNorm(boundaries=boundaries + [boundaries[-1]], ncolors=256)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

        rows_sbs_signatures_list=[]
        rows_dbs_signatures_list=[]
        rows_id_signatures_list=[]
        rows_six_mutation_types_list=[]

        for my_type in rows_types_list:
            if my_type.startswith('SBS'):
                rows_sbs_signatures_list.append(my_type)
            elif my_type.startswith('DBS'):
                rows_dbs_signatures_list.append(my_type)
            elif my_type.startswith('ID'):
                rows_id_signatures_list.append(my_type)
            else:
                rows_six_mutation_types_list.append(my_type)

        print(rows_sbs_signatures_list)
        print(rows_dbs_signatures_list)
        print(rows_id_signatures_list)
        print(rows_six_mutation_types_list)

        all_row_list=[rows_sbs_signatures_list,rows_dbs_signatures_list,rows_id_signatures_list,rows_six_mutation_types_list]


        #fill the numpy array
        for row_list_index, row_list in enumerate(all_row_list):
            rows_types_columns_cancer_types_np_array = np.zeros((len(row_list), len(cancer_types_list)))

            for my_type_index, my_type in enumerate(row_list):
                for strand in strands:
                    if type_df[(type_df['my_type']==my_type) & (type_df['strand']==strand)][percent].any():
                        cancer_types_np_array= type_df[(type_df['my_type']==my_type) & (type_df['strand']==strand)][percent].values[0]
                        cancer_types_np_array_list = eval(cancer_types_np_array)
                        for cancer_type in cancer_types_np_array_list:
                            cancer_type_index=cancer_types_list.index(cancer_type)
                            if strand==TRANSCRIBED_STRAND or strand==LAGGING or strand==GENIC:
                                rows_types_columns_cancer_types_np_array[my_type_index,cancer_type_index]=1
                            elif strand==UNTRANSCRIBED_STRAND or strand==LEADING or strand==INTERGENIC:
                                rows_types_columns_cancer_types_np_array[my_type_index,cancer_type_index]=-1

            if row_list_index==0:
                filename = '%s_rows_sbs_signatures_columns_cancer_types.png' %(strand_bias)
                rows_list=rows_sbs_signatures_list
            elif row_list_index==1:
                filename = '%s_rows_dbs_signatures_columns_cancer_types.png' %(strand_bias)
                rows_list=rows_dbs_signatures_list
            elif row_list_index==2:
                filename = '%s_rows_id_signatures_columns_cancer_types.png' %(strand_bias)
                rows_list=rows_id_signatures_list
            elif row_list_index==3:
                filename = '%s_rows_six_mutation_types_columns_cancer_types.png' %(strand_bias)
                rows_list=rows_six_mutation_types_list

            #Plot heatmap
            if row_list_index==0:
                fig, ax = plt.subplots(figsize=(2*len(cancer_types_list), len(row_list)))
                fontsize=40
            else:
                fig, ax = plt.subplots(figsize=(len(cancer_types_list), 1.5*len(row_list)))
                fontsize=20

            #Plot heatmap rows types columns cancer types
            heatmap(rows_types_columns_cancer_types_np_array, rows_list, cancer_types_list, ax=ax, cmap=cmap, norm=norm, vmin=-1, vmax=1, fontsize=fontsize)

            plt.title('Combined PCAWG nonPCAWG Transcription Strand Bias using %s min difference results' %(percent), fontsize=fontsize)
            # Results in big squares when array is small
            # plt.tight_layout()

            heatmaps_output_dir= os.path.join(output_dir,'strand_bias_aggregated_heatmaps')
            os.makedirs(heatmaps_output_dir, exist_ok=True)

            figureFile = os.path.join(heatmaps_output_dir, filename)
            fig.savefig(figureFile, bbox_inches='tight')
################################################################################
