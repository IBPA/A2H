import os
import pickle

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
import click

from ..utils import load_classification_data


def generate_train_test_splits(
    inputs,
    targets,
    test_size,
    random_state,
    verbose=True,
):
    """
    """
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    idxs_train, idxs_test = list(splitter.split(
        X=inputs,
        y=targets,
        groups=inputs.index,
    ))[0]

    assert set(targets.iloc[idxs_train].index.tolist()) \
        & set(targets.iloc[idxs_test].index.tolist()) \
        == set()

    if verbose:
        print("Training set distribution:")
        print(targets.iloc[idxs_train].value_counts())
        print()
        print("Testing set distribution:")
        print(targets.iloc[idxs_test].value_counts())

    return idxs_train, idxs_test


def generate_k_fold_splits(
    inputs,
    targets,
    n_folds,
    random_state,
    verbose=True,
):
    """
    """
    splitter = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )

    splits = list(splitter.split(
        X=inputs,
        y=targets,
        groups=inputs.index,
    ))

    # Make sure grouped samples are unique to each split.
    for i, (idxs_train, idxs_val) in enumerate(splits):
        targets_train, targets_val = targets.iloc[idxs_train], targets.iloc[idxs_val]
        assert set(targets_train.index.tolist()) & set(targets_val.index.tolist()) \
            == set()

        if verbose:
            print(f"Validation Fold {i}:")
            print(targets_val.value_counts())
            print()

    return splits


@click.command()
@click.argument(
    'path-save-dir',
)
@click.option(
    '--delta',
    default=0.1,
    help='Delta for classification.',
)
@click.option(
    '--subset',
    default='all',
    help='Subset of data to use.',
)
@click.option(
    '--test-size',
    default=0.2,
    help='Fraction of data to use for testing.',
)
@click.option(
    '--n-folds',
    default=5,
    help='Number of folds for cross-validation.',
)
@click.option(
    '--random-state',
    default=42,
    help='Random state for reproducibility.',
)
def main(
    path_save_dir,
    delta,
    subset,
    test_size,
    n_folds,
    random_state,
):
    inputs, targets = load_classification_data(
        delta=delta, is_sigma=True, subset=subset
    )
    inputs = inputs.drop(columns=['intervention'])

    idxs_train, idxs_test = generate_train_test_splits(
        inputs,
        targets,
        test_size=test_size,
        random_state=random_state,
    )
    inputs_train, targets_train = inputs.iloc[idxs_train], targets.iloc[idxs_train]
    inputs_test, targets_test = inputs.iloc[idxs_test], targets.iloc[idxs_test]

    k_fold_splits = generate_k_fold_splits(
        inputs_train,
        targets_train,
        n_folds=n_folds,
        random_state=random_state,
    )

    if subset == 'all':
        path_save_dir \
            = f"{path_save_dir}/cls_delta-{delta}_rs-{random_state}"
    else:
        path_save_dir \
            = f"{path_save_dir}/cls_delta-{delta}_subset-{subset}_rs-{random_state}"
    os.makedirs(path_save_dir, exist_ok=True)
    pd.concat([inputs_train, targets_train], axis=1).to_csv(
        f"{path_save_dir}/data_train.csv",
        index=False,
    )
    pd.concat([inputs_test, targets_test], axis=1).to_csv(
        f"{path_save_dir}/data_test.csv",
        index=False,
    )
    with open(f"{path_save_dir}/cv_splits.pkl", 'wb') as f:
        pickle.dump(k_fold_splits, f)


if __name__ == '__main__':
    main()
