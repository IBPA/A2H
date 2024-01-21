import pickle

import pandas as pd
from scipy.stats import ranksums, spearmanr

from ..utils import constants, load_best_pipeline, load_splits, get_simple_pipeline


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

    inputs_train = inputs_train[list(sfs.k_feature_names_)]
    inputs_test = inputs_test[list(sfs.k_feature_names_)]

    model.fit(inputs_train, labels_train)
    model = model.named_steps['cls']

    with open(f"{path_cls_dir}/shap_values.pkl", 'rb') as f:
        shap_values = pickle.load(f)

    # Sustained/Acute.
    shap_values = pd.DataFrame(
        shap_values[:, :, 1].values,
        columns=model.feature_names_in_,
    )
    for feature_name in [
        'ohe_0__acute/sustained preclinical_sustained',
        'ohe_19__acute/sustained clinical_sustained',
    ]:
        shap_values_s = shap_values.loc[inputs_train[feature_name] == 1, feature_name]
        shap_values_a = shap_values.loc[inputs_train[feature_name] == 0, feature_name]

        _, pvals = ranksums(shap_values_s, shap_values_a)
        print(feature_name)
        print(shap_values_s.mean())
        print(pvals)

    idxs_both_sus = inputs_train.query(
        "`ohe_0__acute/sustained preclinical_sustained` == 1 "
        "& `ohe_19__acute/sustained clinical_sustained` == 1"
    ).index
    idxs_both_acu = inputs_train.query(
        "`ohe_0__acute/sustained preclinical_sustained` == 0 "
        "& `ohe_19__acute/sustained clinical_sustained` == 0"
    ).index

    labels_both_sus = labels_train.loc[idxs_both_sus]
    labels_both_non_acu = labels_train.drop(index=idxs_both_sus)

    print(labels_both_sus.mean())
    print(labels_both_non_acu.mean())
    _, pvals = ranksums(labels_both_sus, labels_both_non_acu, alternative='less')
    print(pvals)

    # Subjuct ages.
    shap_values_p = shap_values['scaler_4__animal age value (days) preclinical']
    feature_values_p = inputs_train['scaler_4__animal age value (days) preclinical']
    print(spearmanr(shap_values_p, feature_values_p))

    shap_values_child = shap_values.loc[
        inputs_train['ohe_26__Age Groups clinical_child'] == 1,
        'ohe_26__Age Groups clinical_child',
    ]
    shap_values_non_child = shap_values.loc[
        inputs_train['ohe_26__Age Groups clinical_child'] == 0,
        'ohe_26__Age Groups clinical_child',
    ]
    _, pvals = ranksums(shap_values_child, shap_values_non_child, alternative='less')
    print(shap_values_child.mean())
    print(shap_values_non_child.mean())
    print(pvals)
