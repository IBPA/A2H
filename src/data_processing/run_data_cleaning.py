import os

import pandas as pd
import click


@click.command()
@click.option(
    '--path-input',
    type=click.Path(exists=True),
    default="data/pairs.tsv",
    help="Path to the input data file.",
)
@click.option(
    '--path-output',
    type=click.Path(),
    default="outputs/data_processing/pairs_cleaned.tsv",
    help="Path to the output data file.",
)
def main(path_input, path_output):
    data = pd.read_csv(path_input, sep='\t')
    print(f"Original: {data.shape}")

    # Drop rows with missing `total dosage value`.
    data = data.query(
        "`total dosage preclinical`.notnull() and `total dosage clinical`.notnull()"
    )
    print(f"+ Removed rows missing total dosage values: {data.shape}")

    # Keep rows with dosage units that are dominant and convert units.
    data = data.query(
        "`dosage units preclinical` == 'mg/kg' "
        "& `dosage units clinical` in ['mg', 'g']"
    ).copy()
    data.loc[
        data['dosage units clinical'] == 'g',
        ['daily dosage clinical', 'total dosage clinical', 'dosage value clinical']
    ] *= 1000
    data = data.drop(columns=['dosage units preclinical', 'dosage units clinical'])
    print(f"+ Removed rows with unconvertible units: {data.shape}")

    # Resolve synonyms and rename values..
    data['animal strain preclinical'] = data['animal strain preclinical'].replace({
        'c57/b6': 'c57bl/6',
    })
    data.loc[data['species preclinical'] == 'pig', 'animal strain preclinical'] = 'pig'
    data['dosage times per day preclinical'] \
        = data['dosage times per day preclinical'].replace({
            'qd': 1,
            'bid': 2,
            'tid': 3,
            'qid': 4,
        })
    data['intervention'] = data['intervention clinical']
    data['dosage times per day clinical'] \
        = data['dosage times per day clinical'].replace({
            'qd': 1,
            'bid': 2,
            'tid': 3,
            'qid': 4,
        })
    data['prophylactic / therapeutic clinical'] \
        = data['prophylactic / therapeutic clinical'].replace(
            {'prophylactic': 'p', 'therapeutic': 't'}
        )

    # Generate Group IDs.
    data['group_id'] = data['Link preclinical'] + '_' + data['URL clinical']
    data = data.set_index('group_id')

    # # Generate target variables.
    # delta = 0.05
    # data['translated'] = (
    #     (data['result preclinical'] - data['result clinical']).abs() < delta
    # ).astype(int)

    # Rename or drop unused columns.
    data = data.drop(
        columns=[
            'Link preclinical',
            'Year preclinical',
            'mechanism_of_action preclinical',
            'success measure preclinical',
            'tested dose dependently preclinical',
            'intervention preclinical',
            # 'result preclinical',
            'URL clinical',
            'intervention class clinical',
            'intervention clinical',
            'dose dependence clinical',
            'Gender clinical',
            # 'result clinical',
        ]
    )
    print(f"+ Removed unused columns: {data.shape}")

    # Impute some missing values.
    data['gnotobiotic preclinical'] \
        = data['gnotobiotic preclinical'].fillna('Unknown')
    data['animal strain preclinical'] \
        = data['animal strain preclinical'].fillna('Unknown')
    data['animal sex preclinical'] \
        = data['animal sex preclinical'].fillna('Unknown')
    data['ribotype preclinical'] \
        = data['ribotype preclinical'].fillna('Unknown')
    data['c diff dose units preclinical'] \
        = data['c diff dose units preclinical'].fillna('Unknown')
    data['c diff dose vegetative/spores preclinical'] \
        = data['c diff dose vegetative/spores preclinical'].fillna('Unknown')

    if not os.path.exists(os.path.dirname(path_output)):
        os.makedirs(os.path.dirname(path_output), exist_ok=True)
    data.to_csv(path_output, sep='\t')


if __name__ == '__main__':
    main()
