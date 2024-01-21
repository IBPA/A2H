import pickle

import pandas as pd

from msap.utils import get_all_metrics
from ..utils import constants, load_splits, load_best_pipeline, get_simple_pipeline


def get_model_predictions(
    model,
    data_train,
    data_test,
):
    """
    """
    X_train = data_train.drop(columns=['result translation'])
    y_train = data_train['result translation']
    X_test = data_test.drop(columns=['result translation'])
    y_test = data_test['result translation']

    prep_pipeline = get_simple_pipeline(X_train, constants.FEATURES_CAT)
    prep_pipeline.fit(X_train)
    X_train_transformed = prep_pipeline.transform(X_train)
    X_test_transformed = prep_pipeline.transform(X_test)
    if sfs is not None:
        X_train_transformed = X_train_transformed[list(sfs.k_feature_names_)]
        X_test_transformed = X_test_transformed[list(sfs.k_feature_names_)]

    model.fit(X_train_transformed, y_train)
    y_pred = model.predict(X_test_transformed)
    y_score = model.predict_proba(X_test_transformed)

    return y_test, y_pred, y_score


if __name__ == '__main__':
    path_data_dir = "outputs/data_processing/splits/cls_delta-0.5_rs-42"
    path_cls_dir = "outputs/msap/cls_delta-0.5_rs-42"

    data_train, data_test, _ = load_splits(path_data_dir)
    model = load_best_pipeline(path_cls_dir)
    with open(f"{path_cls_dir}/classifiers/sfs.pkl", 'rb') as f:
        sfs = pickle.load(f)

    inputs_train = data_train.drop('result translation', axis=1)
    labels_train = data_train['result translation']
    inputs_test = data_test.drop('result translation', axis=1)
    labels_test = data_test['result translation']

    prep_pipeline = get_simple_pipeline(inputs_train, constants.FEATURES_CAT)
    prep_pipeline.fit(inputs_train)
    inputs_train = prep_pipeline.transform(inputs_train)
    inputs_test = prep_pipeline.transform(inputs_test)

    k_best = 35
    k_parsimonious = 16
    k_elbow = 7
    sfs_result = pd.DataFrame(sfs.get_metric_dict()).T
    for k in [k_best, k_parsimonious, k_elbow]:
        features = sfs_result.loc[k, 'feature_names']

        inputs_train_k = inputs_train[list(features)]
        inputs_test_k = inputs_test[list(features)]

        model.fit(inputs_train_k, labels_train)
        y_true = labels_test
        y_pred = model.predict(inputs_test_k)
        y_score = model.predict_proba(inputs_test_k)

        print(f"K = {k}")
        print(get_all_metrics(y_true, y_pred, y_score[:, 1]))
