import pandas as pd
import os

chroms = list(range(1,23))
chroms.extend(['X','Y'])

cancer_types=['ALL', 'Bladder-TCC', 'Bone-Benign', 'Bone-Osteosarc', 'CNS-GBM', 'CNS-Medullo', 'CNS-PiloAstro', 'ColoRect-AdenoCA', 'Ewings', 'Head-SCC', 'Kidney-RCC', 'Lung-AdenoCA', 'Lymph-BNHL', 'Myeloid-AML', 'Myeloid-MPN', 'Panc-AdenoCA', 'Prost-AdenoCA', 'SoftTissue-Leiomyo', 'Stomach-AdenoCA', 'Uterus-AdenoCA', 'Biliary-AdenoCA', 'Blood-CMDI', 'Bone-Epith', 'Breast-Cancer', 'CNS-LGG', 'CNS-Oligo', 'Cervix-Cancer', 'Eso-AdenoCA', 'Eye-Melanoma', 'Kidney-ChRCC', 'Liver-HCC', 'Lung-SCC','Lymph-CLL', 'Myeloid-MDS', 'Ovary-AdenoCA', 'Panc-Endocrine', 'Skin-Melanoma', 'SoftTissue-Liposarc', 'Thy-AdenoCA']

for cancer_type in cancer_types:
    combined_pcawg_nonpcawg_cancer_type_path = '/restricted/alexandrov-group/burcak/SigProfilerTopographyRuns/Combined_PCAWG_nonPCAWG_4th_iteration/%s/data/chrbased' %(cancer_type)
    sbs3_all=set()
    sbs17b_all=set()
    for chrom in chroms:
        file_name  = 'chr%s_SUBS_for_topography.txt' %(chrom)
        if os.path.exists(os.path.join(combined_pcawg_nonpcawg_cancer_type_path,file_name)):
            # print(os.path.join(combined_pcawg_nonpcawg_cancer_type_path,file_name))
            df=pd.read_csv(os.path.join(combined_pcawg_nonpcawg_cancer_type_path,file_name), sep='\t',header=0)
            sbs3_chrom_based=set(df[df['SBS3']>0.5]['Sample'].values)
            sbs3_all = sbs3_all.union(sbs3_chrom_based)
            sbs17b_chrom_based=set(df[df['SBS17b']>0.5]['Sample'].values)
            sbs17b_all = sbs17b_all.union(sbs17b_chrom_based)
    print('\n', cancer_type)
    print('# of samples having SBS3', len(sbs3_all))
    print('# of samples having SBS17b', len(sbs17b_all))
    print('# of samples having SBS3 but not SBS17b ', len(sbs3_all.difference(sbs17b_all)))
    print(sbs3_all.difference(sbs17b_all))