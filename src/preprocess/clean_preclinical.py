import sys
import pandas as pd

df = pd.read_csv(
    '../data/raw/A2H R-CDI Data - Pre-Clinical Trials.csv',
    dtype=str,
    keep_default_na=False,
)
print(f'Raw preclinical data size: {df.shape}')

df = df[df['matched'] == 'TRUE']
print(df.shape)

# column-by-column manual cleanup
df = df[df['sample group size'].str.isnumeric()]
df = df[df['outcome count'].str.isnumeric()]
df['gnotobiotic'] = df['gnotobiotic'].apply(lambda x: ''.join(x.lower().split()))
df['animal strain'] = df['animal strain'].apply(lambda x: ''.join(x.lower().split()))
df['animal sex'] = df['animal sex'].apply(lambda x: ''.join(x.lower().split()))
df['animal age value (days)'] = df['animal age value (days)'].apply(lambda x: ''.join(x.lower().split()))
df['animal age value (days)'] = df['animal age value (days)'].apply(lambda x: '' if x == 'n/a' else x)
df['animal weight value (g)'] = df['animal weight value (g)'].apply(lambda x: ''.join(x.lower().split()))
df['animal weight value (g)'] = df['animal weight value (g)'].apply(lambda x: '' if x == 'n/a' else x)
df['species'] = df['species'].apply(lambda x: ''.join(x.lower().split()))
df['mechanism_of_action'] = df['mechanism_of_action'].apply(lambda x: ''.join(x.lower().split()))
df['disease model'] = df['disease model'].apply(lambda x: ''.join(x.lower().split()))
df = df[df['success measure'] != 'development of enterocaecitis']
df['success measure'] = df['success measure'].apply(lambda x: ''.join(x.lower().split()))
df['tested dose dependently'] = df['tested dose dependently'].apply(lambda x: ''.join(x.lower().split()))
df['prophylactic / therapeutic'] = df['prophylactic / therapeutic'].apply(lambda x: ''.join(x.lower().split()))
df['ribotype'] = df['ribotype'].apply(lambda x: ''.join(x.lower().split()))
df['c diff strain'] = df['c diff strain'].apply(lambda x: ''.join(x.lower().split()))
df['c diff dose units'] = df['c diff dose units'].apply(lambda x: ''.join(x.lower().split()))
df['c diff dose units'] = df['c diff dose units'].apply(lambda x: '' if x == 'n/a' else x)
df['c diff dose vegetative/spores'] = df['c diff dose vegetative/spores'].apply(lambda x: ''.join(x.lower().split()))
df['c diff dose vegetative/spores'] = df['c diff dose vegetative/spores'].apply(lambda x: '' if x == 'n/a' else x)
df['daily dosage'] = df['daily dosage'].apply(lambda x: ''.join(x.lower().split()))
df['daily dosage'] = df['daily dosage'].apply(lambda x: '' if x == 'n/a' else x)
df['total dosage'] = df['total dosage'].apply(lambda x: ''.join(x.lower().split()))
df['total dosage'] = df['total dosage'].apply(lambda x: '' if x == 'n/a' else x)
df['dosage units'] = df['dosage units'].apply(lambda x: ''.join(x.lower().split()))
df['dosage times per day'] = df['dosage times per day'].apply(lambda x: ''.join(x.lower().split()))
df['intervention'] = df['intervention'].apply(lambda x: ''.join(x.lower().split()))
df['intervention'] = df['intervention'].apply(lambda x: '' if x == 'n/a' else x)

# create dependent variable
df['result'] = df.apply(lambda r: int(r['outcome count']) / int(r['sample group size']), axis=1)

#
columns_to_drop = [
    # 'Link',
    'ID',
    'doc_name',
    'has_clinical',
    'sample group size',
    'outcome count',
    'disease',
    'outcome direction',
    'animal age description',
    'animal weight description',
    'intervention delivery',
    # 'success measure',
    'c diff dose description',
    'dosage description',
    'comparator',
    'accute efficacy',
    'sustained efficacy',
    'comment',
    'Done?',
    'matched',
]

df.drop(columns=columns_to_drop, inplace=True)

#
print(df.shape)
df.to_csv(
    '../data/preprocessed/preclinical_temp.tsv',
    sep='\t',
    index=False,
)
