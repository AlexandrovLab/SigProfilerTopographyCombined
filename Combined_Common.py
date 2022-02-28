import os
import shutil
import re
import pandas as pd

DATA = 'data'
SIGNATURE='signature'

TABLE_SBS_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE = "Table_SBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt"
TABLE_DBS_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE = "Table_DBS_Signature_Cutoff_NumberofMutations_AverageProbability.txt"
TABLE_ID_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE  = "Table_ID_Signature_Cutoff_NumberofMutations_AverageProbability.txt"
TABLE_MUTATIONTYPE_NUMBEROFMUTATIONS_NUMBEROFSAMPLES_SAMPLESLIST = "Table_MutationType_NumberofMutations_NumberofSamples_SamplesList.txt"

AGGREGATEDDINUCS = 'aggregateddinucs'
AGGREGATEDSUBSTITUTIONS = 'aggregatedsubstitutions'
AGGREGATEDINDELS = 'aggregatedindels'

COSMIC_NUCLEOSOME_OCCUPANCY = 'NUCLEOSOME_OCC'
COSMIC_CTCF_OCCUPANCY = 'CTCF_OCC'
COSMIC_OCCUPANCY = 'OCC'
COSMIC_REPLICATION_TIME = 'REPLIC_TIME'
COSMIC_PROCESSIVITY = 'PROCESSIVITY'
COSMIC_REPLICATION_STRAND_BIAS = 'REPLIC_ASYM'
COSMIC_TRANSCRIPTION_STRAND_BIAS = 'TRANSCR_ASYM'
COSMIC_GENIC_VS_INTERGENIC_BIAS = 'GENIC_ASYM'
COSMIC_HISTONE_MODIFICATIONS = 'HISTONE_MODS'

SBS = 'SBS'
DBS = 'DBS'
ID = 'ID'

SUBS = 'SUBS'
DINUCS = 'DINUCS'
INDELS = 'INDELS'

OCCUPANCY_HEATMAP_COMMON_MULTIPLIER = 1/2

# This dictionary is used for cosmic figure namings
cancer_type_2_NCI_Thesaurus_code_dict = {
    'Bladder-TCC' : 'C39851',
    'Bone-Benign' : 'C4880',
    'Bone-Osteosarc' :	'C53707',
    'CNS-GBM' :	'C3058',
    'CNS-Medullo' :	'C3222',
    'CNS-PiloAstro' :	'C4047',
    'ColoRect-AdenoCA' :	'C5105',
    'Ewings' :	'C4817',
    'Head-SCC' :	'C34447',
    'Kidney-RCC' :	'C9385',
    'Lung-AdenoCA' :	'C3512',
    'Lymph-BNHL' :	'C3457',
    'Myeloid-AML' :	'C3171',
    'Myeloid-MPN' :	'C4345',
    'Panc-AdenoCA' :	'C8294',
    'Prost-AdenoCA' :	'C2919',
    'SoftTissue-Leiomyo' :	'C3157',
    'Stomach-AdenoCA' :	'C4004',
    'Uterus-AdenoCA' :	'C7359',
    'Biliary-AdenoCA' :	'C4436',
    'Blood-CMDI' :	'Blood-CMDI',
    'Bone-Epith' :	'C2947',
    'Breast-Cancer' :	'C4872',
    'CNS-LGG' :	'C132067',
    'CNS-Oligo' :	'C3288',
    'Cervix-Cancer' :	'C9039',
    'Eso-AdenoCA' :	'C4025',
    'ESCC' : 'C4024',
    'Eye-Melanoma' :	'C8562',
    'Kidney-ChRCC' :	'C4146',
    'Liver-HCC' :	'C3099',
    'Lung-SCC' :	'C3493',
    'Lymph-CLL' :	'C3163',
    'Myeloid-MDS' :	'C3247',
    'Ovary-AdenoCA' :	'C7700',
    'Panc-Endocrine' :	'C27720',
    'Skin-Melanoma' :	'C3510',
    'SoftTissue-Liposarc' :	'C3194',
    'Thy-AdenoCA' :	'C27380',
    'ALL' :	'C3167'
}

# Artifact Signatures to be removed in manucript figures
signatures_attributed_to_artifacts = ['SBS27', 'SBS43', 'SBS45', 'SBS46', 'SBS47', 'SBS48', 'SBS49',
                                      'SBS50', 'SBS51', 'SBS52', 'SBS53', 'SBS54', 'SBS55', 'SBS56',
                                      'SBS57', 'SBS58', 'SBS59', 'SBS60']

ALTERNATIVE_OUTPUT_DIR = os.path.join('/restricted', 'alexandrov-group', 'burcak', 'SigProfilerTopographyRuns', 'PCAWG_nonPCAWG_lymphomas')
LYMPH_BNHL_CLUSTERED = 'Lymph-BNHL_clustered'
LYMPH_BNHL_NONCLUSTERED = 'Lymph-BNHL_nonClustered'
LYMPH_BNHL = 'Lymph-BNHL'
LYMPH_CLL_CLUSTERED = 'Lymph-CLL_clustered'
LYMPH_CLL_NONCLUSTERED = 'Lymph-CLL_nonClustered'
LYMPH_CLL = 'Lymph-CLL'

# This function is implemented for using different Topography results for lymphoid samples
# where main_cancer_type is either Lymph-BNHL or Lymph-CLL
# for other main_cancer_type there is no change
def get_alternative_combined_output_dir_and_cancer_type(main_combined_output_dir, main_cancer_type, signature):
    if main_cancer_type == LYMPH_BNHL:
        if signature == 'SBS84' or signature == 'SBS85' or signature == 'SBS37':
            return ALTERNATIVE_OUTPUT_DIR, LYMPH_BNHL_CLUSTERED
        elif (signature == AGGREGATEDSUBSTITUTIONS) or (signature == AGGREGATEDDINUCS) or (signature == AGGREGATEDINDELS) \
                or signature.startswith('DBS') or signature.startswith('ID'):
            return main_combined_output_dir, main_cancer_type
        else :
            return ALTERNATIVE_OUTPUT_DIR, LYMPH_BNHL_NONCLUSTERED
    elif main_cancer_type == LYMPH_CLL:
        if signature == 'SBS84' or signature == 'SBS85' or signature == 'SBS37':
            return ALTERNATIVE_OUTPUT_DIR, LYMPH_CLL_CLUSTERED
        elif (signature == AGGREGATEDSUBSTITUTIONS) or (signature == AGGREGATEDDINUCS) or (signature == AGGREGATEDINDELS) \
                or signature.startswith('DBS') or signature.startswith('ID'):
            return main_combined_output_dir, main_cancer_type
        else:
            return ALTERNATIVE_OUTPUT_DIR, LYMPH_CLL_NONCLUSTERED
    else:
        return main_combined_output_dir, main_cancer_type

def deleteOldData(toBeDeletePath):
    if (os.path.exists(toBeDeletePath)):
        try:
            shutil.rmtree(toBeDeletePath)
        except OSError as e:
            print('Error: %s - %s.' % (e.filename, e.strerror))

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

# Updated for lymphoid samples
def get_signature2cancer_type_list_dict(combined_output_dir, cancer_types):
    signature2cancer_type_list_dict = {}

    for cancer_type in cancer_types:
        # Added for lymphoid samples
        if cancer_type == LYMPH_BNHL or cancer_type == LYMPH_CLL:
            sbs_file_path1 = os.path.join(ALTERNATIVE_OUTPUT_DIR, '%s_clustered' %(cancer_type), DATA, TABLE_SBS_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE)
            sbs_file_path2 = os.path.join(ALTERNATIVE_OUTPUT_DIR, '%s_nonClustered' %(cancer_type), DATA, TABLE_SBS_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE)
            dbs_file_path = os.path.join(combined_output_dir, cancer_type, DATA, TABLE_DBS_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE)
            id_file_path = os.path.join(combined_output_dir, cancer_type, DATA, TABLE_ID_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE)
            sbs_signatures_df1 = pd.read_csv(sbs_file_path1, header=0, sep="\t")
            sbs_signatures_df2 = pd.read_csv(sbs_file_path2, header=0, sep="\t")
            dbs_signatures_df = pd.read_csv(dbs_file_path, header=0, sep="\t")
            id_signatures_df = pd.read_csv(id_file_path, header=0, sep="\t")
            signature_df_list = [sbs_signatures_df1, sbs_signatures_df2, dbs_signatures_df, id_signatures_df]
        else:
            sbs_file_path = os.path.join(combined_output_dir, cancer_type, DATA, TABLE_SBS_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE)
            dbs_file_path = os.path.join(combined_output_dir, cancer_type, DATA, TABLE_DBS_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE)
            id_file_path = os.path.join(combined_output_dir, cancer_type, DATA, TABLE_ID_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE)
            sbs_signatures_df = pd.read_csv(sbs_file_path, header=0, sep="\t")
            dbs_signatures_df = pd.read_csv(dbs_file_path, header=0, sep="\t")
            id_signatures_df = pd.read_csv(id_file_path, header=0, sep="\t")
            signature_df_list = [sbs_signatures_df, dbs_signatures_df, id_signatures_df]

        for signature_df in signature_df_list:
            for signature in signature_df[SIGNATURE].unique():
                if signature in signature2cancer_type_list_dict:
                    if cancer_type not in signature2cancer_type_list_dict[signature]:
                        signature2cancer_type_list_dict[signature].append(cancer_type)
                else:
                    signature2cancer_type_list_dict[signature]=[]
                    signature2cancer_type_list_dict[signature].append(cancer_type)

    return signature2cancer_type_list_dict


def fill_lists(signature, signature2cancer_type_list_dict):
    signature_tissue_type_tuples = []
    signatures_ylabels_on_the_heatmap = []

    if signature in signature2cancer_type_list_dict:
        cancer_types = signature2cancer_type_list_dict[signature]
        cancer_types = sorted(cancer_types, key=natural_key)

        signature_tissue_type_tuples = [(signature, None)]
        signature_with_num_of_cancer_types = '%s (n=%d)' % (signature, len(cancer_types))
        signatures_ylabels_on_the_heatmap = [signature_with_num_of_cancer_types]

        for cancer_type in cancer_types:
            signature_tissue_type_tuples.append((signature, cancer_type))
            signatures_ylabels_on_the_heatmap.append(cancer_type)

    return signature_tissue_type_tuples, signatures_ylabels_on_the_heatmap


def enriched(avg_fold_change, enriched_fold_change):
 return round(avg_fold_change,2) >= enriched_fold_change

def depleted(avg_fold_change, depleted_fold_change):
 return round(avg_fold_change,2) <= depleted_fold_change


def get_number_of_mutations_filename(signature_type):
    if signature_type == SBS:
        return TABLE_SBS_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE
    elif signature_type == DBS:
        return TABLE_DBS_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE
    elif signature_type == ID:
        return TABLE_ID_SIGNATURE_CUTOFF_NUMBEROFMUTATIONS_AVERAGEPROBABILITY_FILE
    else:
        return None
