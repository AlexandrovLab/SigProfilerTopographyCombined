import pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import numpy as np
from functools import reduce

DATA='data'
TABLE_SBS_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE = "Table_SBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt"
TABLE_DBS_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE = "Table_DBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt"
TABLE_ID_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE  = "Table_ID_Signature_Cutoff_NumberofMutations_AverageProbability.txt"
SIGNATURE='signature'

LAGGING_VERSUS_LEADING='Lagging_Versus_Leading'
TRANSCRIBED_VERSUS_UNTRANSCRIBED='Transcribed_Versus_Untranscribed'
GENIC_VERSUS_INTERGENIC = 'Genic_Versus_Intergenic'

percentage = True
plot_type = '384'

def plotSBS(signature, df, percentage, plot_type, column_name):
    total_count = 0
    sig_probs = False
    # plotting legacy code
    strand_biases = [TRANSCRIBED_VERSUS_UNTRANSCRIBED, GENIC_VERSUS_INTERGENIC]

    for strand_bias in strand_biases:
        if strand_bias == TRANSCRIBED_VERSUS_UNTRANSCRIBED:
            color1 = 'royalblue'
            color2 = 'yellowgreen'
            # strand1 transcribed  T
            # strand2 untranscribed U
            strand1_letters = ['T']
            strand2_letters = ['U']
            strand1_label = 'Transcribed Strand'
            strand2_label = 'Untranscribed Strand'
            text = 'transcribed'
        elif strand_bias == LAGGING_VERSUS_LEADING:
            color1 = 'indianred'
            color2 = 'goldenrod'
            # TODO
        elif strand_bias == GENIC_VERSUS_INTERGENIC:
            color1 = 'cyan'
            color2 = 'gray'
            # strand1 genic  T U
            # strand2 intergenic  N
            strand1_letters = ['T', 'U']
            strand2_letters = ['N']
            strand1_label = 'Genic'
            strand2_label = 'Intergenic'
            text = 'genic'

        if plot_type == '192' or plot_type == '96SB' or plot_type == '384':
            pp = PdfPages(output_path + signature + '_' + 'percentage' + '_' + str(percentage) + '_' + strand_bias + '.pdf')

            plt.rcParams['axes.linewidth'] = 2
            plot1 = plt.figure(figsize=(43.93, 9.92))
            plt.rc('axes', edgecolor='lightgray')
            panel1 = plt.axes([0.04, 0.09, 0.95, 0.77])

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
            xlabels.extend(
                [nucleotide_left + 'C' + nucleotide_right for nucleotide_left in nucleotides for nucleotide_right in
                 nucleotides] * 3)
            xlabels.extend(
                [nucleotide_left + 'T' + nucleotide_right for nucleotide_left in nucleotides for nucleotide_right in
                 nucleotides] * 3)

            # First Pass
            # Get the total_count at the start
            for i, xlabel in enumerate(xlabels):
                xsublabel = xsublabels[i]
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
                        # trans = plt.bar(x, transribed_count, width=0.75,color=[1/256,70/256,102/256],align='center', zorder=1000, label='Transcribed Strand')
                        strand1 = plt.bar(x, strand1_count / total_count * 100, width=0.75, color=color1,
                                          align='center', zorder=1000, label=strand1_label)
                        x += 0.75
                        # untrans = plt.bar(x, untransribed_count,width=0.75,color=[228/256,41/256,38/256],align='center', zorder=1000, label='Untranscribed Strand')
                        strand2 = plt.bar(x, strand2_count / total_count * 100, width=0.75, color=color2,
                                          align='center', zorder=1000, label=strand2_label)
                        x += .2475
                else:
                    # trans = plt.bar(x, transribed_count, width=0.75,color=[1/256,70/256,102/256],align='center', zorder=1000, label='Transcribed Strand')
                    strand1 = plt.bar(x, strand1_count, width=0.75, color=color1, align='center', zorder=1000,
                                      label=strand1_label)
                    x += 0.75
                    # untrans = plt.bar(x, untransribed_count,width=0.75,color=[228/256,41/256,38/256],align='center', zorder=1000, label='Untranscribed Strand')
                    strand2 = plt.bar(x, strand2_count, width=0.75, color=color2, align='center', zorder=1000,
                                      label=strand2_label)
                    x += .2475

                x += 1

            x = .0415
            y3 = .87
            y = int(ymax * 1.25)
            x_plot = 0

            yText = y3 + .06
            plt.text(.1, yText, 'C>A', fontsize=55, fontweight='bold', fontname='Arial',
                     transform=plt.gcf().transFigure)
            plt.text(.255, yText, 'C>G', fontsize=55, fontweight='bold', fontname='Arial',
                     transform=plt.gcf().transFigure)
            plt.text(.415, yText, 'C>T', fontsize=55, fontweight='bold', fontname='Arial',
                     transform=plt.gcf().transFigure)
            plt.text(.575, yText, 'T>A', fontsize=55, fontweight='bold', fontname='Arial',
                     transform=plt.gcf().transFigure)
            plt.text(.735, yText, 'T>C', fontsize=55, fontweight='bold', fontname='Arial',
                     transform=plt.gcf().transFigure)
            plt.text(.89, yText, 'T>G', fontsize=55, fontweight='bold', fontname='Arial',
                     transform=plt.gcf().transFigure)

            if y <= 4:
                y += 4

            while y % 4 != 0:
                y += 1

            y = ymax / 1.025

            ytick_offest = float(y / 3)
            for i in range(0, 6, 1):
                panel1.add_patch(plt.Rectangle((x, y3), .155, .05, facecolor=colors[i], clip_on=False,
                                               transform=plt.gcf().transFigure))
                panel1.add_patch(
                    plt.Rectangle((x_plot, 0), 32, round(ytick_offest * 4, 1), facecolor=colors[i], zorder=0,
                                  alpha=0.25, edgecolor='grey'))
                x += .1585
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

            # if sig_probs: # To my understanding sig_probs is true when percentage = True
            if percentage:
                plt.text(0.045, 0.75, signature, fontsize=60, weight='bold', color='black', fontname="Arial",
                         transform=plt.gcf().transFigure)
            else:
                plt.text(0.045, 0.75, signature + ": " + "{:,}".format(int(total_count)) + " " + text + " subs",
                         fontsize=60, weight='bold', color='black', fontname="Arial", transform=plt.gcf().transFigure)

            panel1.set_yticklabels(ylabels, fontsize=font_label_size)
            plt.gca().yaxis.grid(True)
            plt.gca().grid(which='major', axis='y', color=[0.6, 0.6, 0.6], zorder=1)
            panel1.set_xlabel('')
            panel1.set_ylabel('')
            plt.legend(handles=[strand1, strand2], prop={'size': 30})
            if percentage:
                plt.ylabel("Percentage of Single Base Substitutions", fontsize=35, fontname="Times New Roman",
                           weight='bold')
            else:
                plt.ylabel("Number of Single Base Substitutions", fontsize=35, fontname="Times New Roman",
                           weight='bold')

            panel1.tick_params(axis='both', which='both', \
                               bottom=False, labelbottom=False, \
                               left=False, labelleft=True, \
                               right=False, labelright=False, \
                               top=False, labeltop=False, \
                               direction='in', length=25, colors=[0.6, 0.6, 0.6])

            [i.set_color("black") for i in plt.gca().get_yticklabels()]
            pp.savefig(plot1)
            plt.close()
            pp.close()

def get_signature2cancer_type_list_dict(combined_output_dir,cancer_types):
    signature2cancer_type_list_dict={}

    for cancer_type in cancer_types:
        sbs_file_path = os.path.join(combined_output_dir, cancer_type, DATA, TABLE_SBS_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE)
        sbs_signatures_df = pd.read_csv(sbs_file_path, header=0, sep="\t")

        for signature in sbs_signatures_df[SIGNATURE].unique():
            if signature in signature2cancer_type_list_dict:
                if cancer_type not in signature2cancer_type_list_dict[signature]:
                    signature2cancer_type_list_dict[signature].append(cancer_type)
            else:
                signature2cancer_type_list_dict[signature]=[]
                signature2cancer_type_list_dict[signature].append(cancer_type)

    return signature2cancer_type_list_dict

# # For Test BMI_Samples
# signature = 'SBS4'
# output_path=os.path.join('/Users/burcakotlu/Desktop/strand_bias_figures/')
# project='burcak_test'
#
# input_file_name = 'bmi_samples_90percent_T2A_removed.SBS384.all'
# input_path = os.path.join('/Users/burcakotlu/Desktop/strand_bias_figures/')
# df = pd.read_csv(os.path.join(input_path,input_file_name), sep='\t')
# print(df.columns.values)
#
# # Sum all columns except MutationType
# # assign them to a column as the last column
# df['BMI_Samples'] = df.sum(axis=1)
#
# #Drop all columns except first and last column
# df.drop((df.columns.values[1:-1]),axis=1, inplace=True)
# df.to_csv(os.path.join(input_path,'BMI.SBS384.all'), sep='\t', index=False)
# column_name = 'BMI_Samples'

# works on terminal but not here
# def plotSBS(matrix_path, output_path, project, plot_type, percentage=False, custom_text_upper=None, custom_text_middle=None, custom_text_bottom=None)
# import sigProfilerPlotting as sigPlt
# import os
# sigPlt.plotSBS(os.path.join('/Users/burcakotlu/Desktop/strand_bias_figures/BMI.SBS384.all'), "/Users/burcakotlu/Desktop/strand_bias_figures/", "Jun23_2021", "384", percentage=True)

# For SBS4 Combined
# SBS4 Head-SCC PCAWG
# SBS4 Liver-HCC PCAWG nonPCAWG
# SBS4 Lung-AdenoCA PCAWG -- Lung-AdenoCa nonPCAWG
# SBS4 Lung-SCC PCAWG
# nonPCAWG
# /restricted/alexandrov-group/burcak/data/nonPCAWG/Liver-HCC/output/SBS/Liver-HCC.SBS384.all
# /restricted/alexandrov-group/burcak/data/nonPCAWG/Lung-AdenoCa/output/SBS/Lung-AdenoCa.SBS384.all
# PCAWG
# /restricted/alexandrov-group/burcak/data/PCAWG/Liver-HCC/filtered/output/SBS/Liver-HCC.SBS384.all
# /restricted/alexandrov-group/burcak/data/PCAWG/Lung-AdenoCA/filtered/output/SBS/Lung-AdenoCA.SBS384.all
# /restricted/alexandrov-group/burcak/data/PCAWG/Lung-SCC/filtered/output/SBS/Lung-SCC.SBS384.all
# /restricted/alexandrov-group/burcak/data/PCAWG/Head-SCC/filtered/output/SBS/Head-SCC.SBS384.all

PCAWG = 'PCAWG'
nonPCAWG = 'nonPCAWG'

cancer_types = ['ALL', 'Bladder-TCC', 'Bone-Benign', 'Bone-Osteosarc', 'CNS-GBM', 'CNS-Medullo', 'CNS-PiloAstro', 'ColoRect-AdenoCA', 'Ewings', 'Head-SCC', 'Kidney-RCC', 'Lung-AdenoCA', 'Lymph-BNHL', 'Myeloid-AML', 'Myeloid-MPN', 'Panc-AdenoCA', 'Prost-AdenoCA', 'SoftTissue-Leiomyo', 'Stomach-AdenoCA', 'Uterus-AdenoCA', 'Biliary-AdenoCA', 'Blood-CMDI', 'Bone-Epith', 'Breast-Cancer', 'CNS-LGG', 'CNS-Oligo', 'Cervix-Cancer', 'Eso-AdenoCA', 'Eye-Melanoma', 'Kidney-ChRCC', 'Liver-HCC', 'Lung-SCC','Lymph-CLL', 'Myeloid-MDS', 'Ovary-AdenoCA', 'Panc-Endocrine', 'Skin-Melanoma', 'SoftTissue-Liposarc', 'Thy-AdenoCA']
sbs_signatures = ['SBS1', 'SBS2', 'SBS3', 'SBS4', 'SBS5', 'SBS6', 'SBS7a', 'SBS7b', 'SBS7c', 'SBS7d', 'SBS8',
                  'SBS9', 'SBS10a', 'SBS10b', 'SBS11', 'SBS12', 'SBS13', 'SBS14', 'SBS15', 'SBS16', 'SBS17a',
                  'SBS17b', 'SBS18', 'SBS19', 'SBS20', 'SBS21', 'SBS22', 'SBS23', 'SBS24', 'SBS25', 'SBS26',
                  'SBS27', 'SBS28', 'SBS29', 'SBS30', 'SBS31', 'SBS32', 'SBS33', 'SBS34', 'SBS35', 'SBS36',
                  'SBS37', 'SBS38', 'SBS39', 'SBS40', 'SBS41', 'SBS42', 'SBS43', 'SBS44', 'SBS45', 'SBS46',
                  'SBS47', 'SBS48', 'SBS49', 'SBS50', 'SBS51', 'SBS52', 'SBS53', 'SBS54', 'SBS55', 'SBS56',
                  'SBS57', 'SBS58', 'SBS59', 'SBS60']

combined_output_dir = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'SigProfilerTopographyRuns','Combined_PCAWG_nonPCAWG_4th_iteration')
signature2cancer_type_list_dict = get_signature2cancer_type_list_dict(combined_output_dir, cancer_types)
print('signature2cancer_type_list_dict: ', signature2cancer_type_list_dict)

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


output_path=os.path.join('/oasis/tscc/scratch/burcak/temp/')

for signature in signature2cancer_type_list_dict:
    all_df_list = []
    for cancer_type in signature2cancer_type_list_dict[signature]:
        for source , cancer_type in cancer_type2source_cancer_type_tuples_dict[cancer_type]:
            print(source,cancer_type)
            # For PCAWG 'Head-SCC'
            # sbs384_matrix_file = os.path.join('/restricted/alexandrov-group/burcak/data/PCAWG/Head-SCC/filtered/output/SBS/Head-SCC.SBS384.all')
            # probabilities_file = os.path.join('/home/burcak/developer/SigProfilerTopographyRuns/PCAWG/probabilities/Head-SCC_sbs96_mutation_probabilities.txt')
            # topography_cutoffs_file = os.path.join('/restricted/alexandrov-group/burcak/SigProfilerTopographyRuns/PCAWG/Head-SCC/data/Table_SBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt')
            if source == PCAWG:
                sbs384_matrix_file = os.path.join('/restricted/alexandrov-group/burcak/data/' + source + '/' + cancer_type + '/filtered/output/SBS/' + cancer_type + '.SBS384.all')
                probabilities_file = os.path.join('/home/burcak/developer/SigProfilerTopographyRuns/' + source + '/probabilities/' + cancer_type + '_sbs96_mutation_probabilities.txt')
                matrix_mutation_type_column = 'MutationType'
                matrix_mutation_type_short_column = 'MutationTypeShort'
                prob_sample_column = 'Sample Names'
                prob_mutation_type_column = 'MutationTypes'
            elif source == nonPCAWG:
                sbs384_matrix_file = os.path.join('/restricted/alexandrov-group/burcak/data/' + source + '/' + cancer_type + '/output/SBS/' + cancer_type + '.SBS384.all')
                if cancer_type == 'CNS-Glioma-NOS':
                    probabilities_file = os.path.join(
                        '/home/burcak/developer/SigProfilerTopographyRuns/' + source + '/probabilities/' + 'CNS-glioma-NOS_subs_probabilities.txt')
                else:
                    probabilities_file = os.path.join('/home/burcak/developer/SigProfilerTopographyRuns/' + source + '/probabilities/' + cancer_type + '_subs_probabilities.txt')
                matrix_mutation_type_column = 'MutationType'
                matrix_mutation_type_short_column = 'MutationTypeShort'
                prob_sample_column ='Sample'
                prob_mutation_type_column = 'Mutation'

            topography_cutoffs_file = os.path.join('/restricted/alexandrov-group/burcak/SigProfilerTopographyRuns/' + source + '/' + cancer_type + '/data/Table_SBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt')

            matrix_df = pd.read_csv(sbs384_matrix_file, sep='\t')
            probabilities_df = pd.read_csv(probabilities_file, sep='\t')
            topography_cutoffs_df = pd.read_csv(topography_cutoffs_file, sep='\t')

            print(signature, ' ', source, ' ' , cancer_type, sbs384_matrix_file)
            print(signature, ' ', source, ' ' , cancer_type, probabilities_file)
            print(signature, ' ', source, ' ' , cancer_type, topography_cutoffs_file)

            matrix_samples =  matrix_df.columns.values[1:]
            print(signature, ' ', source, ' ' , cancer_type, ' matrix_df.columns.values.size: ', matrix_df.columns.values.size)
            print(signature, ' ', source, ' ' , cancer_type, ' probabilities_df[Samples].unique().size: ', probabilities_df[prob_sample_column].unique().size)
            print(signature, ' ', 'Set difference matrix versus probabilities: ', np.setdiff1d(matrix_df.columns.values, probabilities_df[prob_sample_column].unique()))
            print(signature, ' ', 'Set difference probabilities versus matrix: ', np.setdiff1d(probabilities_df[prob_sample_column].unique(), matrix_df.columns.values))

            if np.any(topography_cutoffs_df[topography_cutoffs_df['signature']==signature]['cutoff'].values):
                cutoff = topography_cutoffs_df[topography_cutoffs_df['signature']==signature]['cutoff'].values[0]

            matrix_df[matrix_mutation_type_short_column] = matrix_df[matrix_mutation_type_column].str[2:]

            df_list = []
            for matrix_sample in matrix_samples:
                if source == PCAWG:
                    prob_sample = cancer_type + '_' + matrix_sample.split('_')[1]
                else:
                    prob_sample = matrix_sample
                # sub_prob_df (96, 2) columns [prob_mutation_type_column, signature]
                sub_prob_df = probabilities_df[(probabilities_df[prob_sample_column] == prob_sample)][[prob_mutation_type_column, signature]]
                print('##################################################################')
                print(signature, ' ', matrix_sample, ' ', prob_sample, ' ', cutoff, ' sub_prob_df[signature].sum(): ', sub_prob_df[signature].sum())
                # sub_matrix_df[384 rows x 3 columns]  [mutation_type, mutation_type_short, matrix_sample]
                sub_matrix_df = matrix_df[[matrix_mutation_type_column, matrix_mutation_type_short_column, matrix_sample]]
                merged_df = pd.merge(sub_matrix_df, sub_prob_df, how='inner', left_on=matrix_mutation_type_short_column, right_on= prob_mutation_type_column)
                merged_df.loc[(merged_df[signature] < cutoff), prob_sample] = 0
                merged_df.loc[(merged_df[signature] >= cutoff), prob_sample] = merged_df[matrix_sample]
                merged_df = merged_df[[matrix_mutation_type_column, matrix_mutation_type_short_column, prob_sample]]
                merged_df[prob_sample] = merged_df[prob_sample].astype(np.int32)
                # print(signature, ' ', matrix_sample, ' ', prob_sample, ' ', cutoff, ' ', sub_matrix_df.shape, '\nsub_matrix_df: ', sub_matrix_df)
                # print(signature, ' ', matrix_sample, ' ', prob_sample, ' ', cutoff, ' ', sub_prob_df.shape  , '\nsub_prob_df: ', sub_prob_df)
                # print(signature, ' ', matrix_sample, ' ', prob_sample, ' ', cutoff, ' ', merged_df.shape, '\nmerged_df: ', merged_df)
                print(signature, ' ', matrix_sample, ' ', prob_sample, ' ', cutoff, ' merged_df[prob_sample].sum(): ', merged_df[prob_sample].sum())
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
                df_list.append(merged_df)

            print(signature, ' ', 'len(df_list): ', len(df_list))
            df = reduce(lambda x, y: pd.merge(x,y, on = matrix_mutation_type_column), df_list)
            column_name = source + '_' + cancer_type + '_Samples'
            df[column_name] = df.sum(axis=1)
            print(signature, ' ', 'Before df_merged drop columns: ', signature, ' ', source, ' ', cancer_type, ' ', df.columns.values.size, ' ', df.columns.values)
            # Drop all columns except the first and the last one
            df.drop((df.columns.values[1:-1]), axis=1, inplace=True)
            print(signature, ' ', 'After df_merged drop columns: ', signature, ' ', source, ' ', cancer_type, ' ', df.columns.values.size, ' ', df.columns.values)
            print(signature, ' ', 'df', df)
            # file_name = signature + '_' + source + '_' + cancer_type + ".txt"
            # file_path = os.path.join(output_path,file_name)
            # df.to_csv(file_path, sep='\t', index=False, header=True)
            # Decision: I do not add if all zeros. e.g.:  nonPCAWG Liver-HCC
            number_of_mutations_on_strands = df[column_name].sum()
            if number_of_mutations_on_strands>0:
                all_df_list.append(df)
                # plotSBS(signature + '_' + source + '_' + cancer_type, df, percentage, plot_type, column_name)

    # Across all cancer types
    all_df = reduce(lambda x, y: pd.merge(x,y, on = matrix_mutation_type_column), all_df_list)
    file_name = signature + ".txt"
    file_path = os.path.join(output_path, file_name)
    all_df.to_csv(file_path, sep='\t', index=False, header=True)

    column_name = 'Across_All_Cancer_Types'
    all_df[column_name] =all_df.mean(axis=1)
    all_df.drop((all_df.columns.values[1:-1]), axis=1, inplace=True)
    file_name = signature + "_across_all_cancer_types.txt"
    file_path = os.path.join(output_path, file_name)
    all_df.to_csv(file_path, sep='\t', index=False, header=True)
    plotSBS(signature, all_df, percentage, plot_type, column_name)