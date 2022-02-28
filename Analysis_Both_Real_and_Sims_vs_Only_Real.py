import pandas as pd
import numpy as np
import os
df = pd.read_csv(os.path.join('/Users', 'burcakotlu', 'Downloads', 'Step1_Signature_CancerType_Biosample_DNAElement_PValue.txt'), sep='\t')

print('df.shape:', df.shape)
print('df.columns.values:', df.columns.values)

df_only_real = df[ df['real_data_avg_count'] >= 100]
print("df_only_real.shape", df_only_real.shape)

df_both_real_and_sims = df[ (df['real_data_avg_count'] >= 100) | (df['sim_avg_count'] >= 100)]
print("df_both_real_and_sims.shape", df_both_real_and_sims.shape)

print('Number of extra rows considered:', df_both_real_and_sims.shape[0]- df_only_real.shape[0])


df_interest1 = df[ (df['real_data_avg_count'] < 100) & (df['sim_avg_count'] >= 100)]
df_interest2 = df[ (df['real_data_avg_count'] < 100) & (df['sim_avg_count'] >= 100) & (df['fold_change']>1.00)]
df_interest3 = df[ (df['real_data_avg_count'] < 100) & (df['sim_avg_count'] >= 100) & (df['fold_change']<1.00)]
df_interest4 = df[ (df['real_data_avg_count'] < 100) & (df['sim_avg_count'] >= 100) & (df['fold_change'] > 1.00) & ((df['real_data_avg_count'] >= 50))]
df_interest5 = df[ (df['real_data_avg_count'] < 100) & (df['sim_avg_count'] >= 100) & (df['fold_change'] > 1.00) & ((df['real_data_avg_count'] < 50))]


print("real_data_avg_count < 100 and  sim_avg_count >= 100",  df_interest1.shape)
print("real_data_avg_count < 100 and  sim_avg_count >= 100 and fold_change > 1.00", df_interest2.shape)
print("real_data_avg_count < 100 and  sim_avg_count >= 100 and fold_change < 1.00", df_interest3.shape)
print("real_data_avg_count < 100 and  sim_avg_count >= 100 and fold_change > 1.00 and real_data_avg_count >= 50", df_interest4.shape)
print("real_data_avg_count < 100 and  sim_avg_count >= 100 and fold_change > 1.00 and real_data_avg_count < 50", df_interest5.shape)

print(df_interest2[['signature', 'cancer_type', 'biosample', 'dna_element','fold_change', 'real_data_avg_count', 'sim_avg_count']])
print(df_interest5[['real_data_avg_count']])
df_interest5.to_csv(os.path.join('/Users', 'burcakotlu', 'Downloads', 'Step1_Investigate.txt'), sep='\t')
