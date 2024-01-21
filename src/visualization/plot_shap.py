import pickle

import pandas as pd
import shap
import matplotlib.pyplot as plt

from ..utils import constants, load_best_pipeline, load_splits, get_simple_pipeline


if __name__ == '__main__':
    THRESHOLD = 0.5

    path_data_dir \
        = f"outputs/data_processing/splits/cls_delta-{THRESHOLD}_rs-42"
    path_cls_dir \
        = f"outputs/msap/cls_delta-{THRESHOLD}_rs-42"

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

    inputs_train = inputs_train[list(sfs.k_feature_names_)]
    inputs_test = inputs_test[list(sfs.k_feature_names_)]

    model.fit(inputs_train, labels_train)
    model = model.named_steps['cls']
    explainer = shap.Explainer(model)
    shap_values = explainer(inputs_train)

    order = pd.read_csv(f"{path_cls_dir}/feature_importance.csv")['feature_name']
    col2num = {col: i for i, col in enumerate(model.feature_names_in_)}
    order = list(map(col2num.get, order))

    shap.plots.beeswarm(
        shap_values[:, :, 1],
        max_display=16,
        show=False,
        order=order,
    )
    plt.savefig(
        "outputs/visualization/shap.svg"
    )
    plt.close()
