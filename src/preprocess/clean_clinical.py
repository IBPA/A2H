import sys
import pandas as pd

df = pd.read_csv(
    '../data/raw/A2H R-CDI Data - CT Data.csv',
    dtype=str,
    keep_default_na=False,
)
print(f'Raw clinical data size: {df.shape}')

df = df[df['sample group size'].str.isnumeric()]
df = df[df['sample group size'] != '0']
df['sample group size'] = pd.to_numeric(df['sample group size'])

df['measure description'] = df['measure description'].apply(lambda x: ''.join(x.lower().split()))
df = df[
    (df['measure description'] == 'participants') |
    (df['measure description'] == 'percentageofparticipants')]

def calc_responders(row) -> int:
    if row['measure description'] == 'participants':
        return int(row['outcome group size'])
    elif row['measure description'] == 'percentageofparticipants':
        percentage = float(row['outcome group size'].strip().split(' ')[0])
        return row['sample group size'] * percentage / 100
    else:
        raise RuntimeError()

df['outcome count'] = df.apply(calc_responders, axis=1)

# change directions to all positives
def reverse_negative(row) -> int:
    if row['outcome direction'] == 'positive':
        return row['outcome count']
    else:
        return row['sample group size'] - row['outcome count']

df['outcome count'] = df.apply(lambda row: reverse_negative(row), axis=1)
df['outcome direction'] = 'positive'

# replicate clinical observation with synonyms/variations
def explode_df(df, input_column, output_column, drop_set) -> pd.DataFrame:
    subset = df[df[input_column] != ''].copy()
    subset[output_column + '-match-type'] = input_column
    subset[output_column] = subset[input_column].apply(
        lambda x: [synonym for synonym in x.strip(';').split(';')])
    return subset.drop(columns=drop_set).explode(output_column)

intervention_set = ['intervention:Specific', 'intervention:Variation']
df = pd.concat(
    [explode_df(df, x, 'intervention', intervention_set) for x in intervention_set])

# column-by-column manual cleanup
df['acute/sustained'] = df['acute/sustained'].apply(lambda x: ''.join(x.lower().split()))
df['daily dosage'] = df['daily dosage'].apply(lambda x: ''.join(x.lower().split()))
df['total dosage'] = df['total dosage'].apply(lambda x: ''.join(x.lower().split()))
df['dosage value'] = df['dosage value'].apply(lambda x: ''.join(x.lower().split()))
df['dosage units'] = df['dosage units'].apply(lambda x: ''.join(x.lower().split()))
df['dosage times per day'] = df['dosage times per day'].apply(lambda x: ''.join(x.lower().split()))
df['dosage duration'] = df['dosage duration'].apply(lambda x: ''.join(x.lower().split()))
df['intervention class'] = df['intervention class'].apply(lambda x: ''.join(x.lower().split()))
df['intervention'] = df['intervention'].apply(lambda x: ''.join(x.lower().split()))
df['dose dependence'] = df['dose dependence'].apply(lambda x: ''.join(x.lower().split()))
df['prophylactic / therapeutic'] = df['prophylactic / therapeutic'].apply(lambda x: ''.join(x.lower().split()))
df['Gender'] = df['Gender'].apply(lambda x: ''.join(x.lower().split()))
df['Age Groups'] = df['Age Groups'].apply(lambda x: ''.join(x.lower().split()))
df['Phases'] = df['Phases'].apply(lambda x: ''.join(x.lower().split()))

# create dependent variable
df['result'] = df.apply(lambda r: int(r['outcome count']) / int(r['sample group size']), axis=1)

#
columns_to_drop = [
    'case ID',
    'dosage comment',
    'sample group size',
    'outcome',
    'outcome direction',
    'measure description',
    'outcome group size',
    'author conclusion',
    'conclusion method',
    'References',
    'NCT Number',
    'Title',
    'Age Description',
    # 'URL',
    'outcome count',
    'intervention-match-type',
]

df.drop(columns=columns_to_drop, inplace=True)
df.to_csv(
    '../data/preprocessed/clinical.tsv',
    sep='\t',
    index=False,
)
print('Final clinical shape: ', df.shape)
