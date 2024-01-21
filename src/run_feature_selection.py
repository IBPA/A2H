import os
import sys
import warnings
import pickle

import click

from .utils import constants, load_splits, load_best_pipeline, get_simple_pipeline

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


@click.command()
@click.argument(
    'path-cls-dir',
    type=click.Path(exists=True),
)
@click.argument(
    'path-data-dir',
    type=click.Path(exists=True),
)
def main(
    path_cls_dir,
    path_data_dir,
):
    clf_pipeline = load_best_pipeline(path_cls_dir)
    data_train, _, cv_splits = load_splits(path_data_dir)

    X_train = data_train.drop(columns=['result translation'])
    y_train = data_train['result translation']


    prep_pipeline = get_simple_pipeline(
        X_train, constants.FEATURES_CAT
    )
    prep_pipeline.fit(X_train)
    X_train_transformed = prep_pipeline.transform(X_train)

    from mlxtend.feature_selection import SequentialFeatureSelector

    sfs = SequentialFeatureSelector(
        clf_pipeline,
        k_features='parsimonious',
        forward=True,
        floating=False,
        verbose=2,
        scoring='f1',
        cv=cv_splits,
        n_jobs=-1,
    )
    sfs.fit(X_train_transformed, y_train)

    with open(f"{path_cls_dir}/classifiers/sfs.pkl", 'wb') as f:
        pickle.dump(sfs, f)


if __name__ == '__main__':
    main()
