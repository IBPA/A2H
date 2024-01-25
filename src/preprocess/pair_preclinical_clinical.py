import sys
import pandas as pd

df_preclinical = pd.read_csv(
    '../data/preprocessed/preclinical.tsv',
    sep='\t',
    dtype=str,
    keep_default_na=False,
)

df_clinical = pd.read_csv(
    '../data/preprocessed/clinical.tsv',
    sep='\t',
    dtype=str,
    keep_default_na=False,
)

df_preclinical = df_preclinical.add_suffix(' preclinical')
df_clinical = df_clinical.add_suffix(' clinical')

# df_joined = pd.merge(
#     df_preclinical,
#     df_clinical,
#     left_on=['intervention preclinical', 'acute/sustained preclinical'],
#     right_on=['intervention clinical', 'acute/sustained clinical'],
# )

df_joined = pd.merge(
    df_preclinical,
    df_clinical,
    left_on=['intervention preclinical'],
    right_on=['intervention clinical'],
)

print(f'Joined data size: {df_joined.shape}')
df_joined.to_csv('../data/preprocessed/pairs.tsv', sep='\t', index=False)
