import pickle

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import shap

from ..utils import constants, load_splits, load_best_pipeline, get_simple_pipeline


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

    prep_pipeline = get_simple_pipeline(inputs_train, constants.FEATURES_CAT)
    prep_pipeline.fit(inputs_train)
    inputs_train = prep_pipeline.transform(inputs_train)
    inputs_test = prep_pipeline.transform(inputs_test)

    inputs_train = inputs_train[list(sfs.k_feature_names_)]
    inputs_test = inputs_test[list(sfs.k_feature_names_)]

    # Get RFE rankings.
    k_features = len(sfs.k_feature_names_)
    sfs_result = pd.DataFrame(sfs.get_metric_dict()).T

    result_rows = []
    for i in range(1, k_features + 1):
        feature_names = sfs_result.loc[i, 'feature_names']
        if i == 1:
            feature_names_prev = ()
        else:
            feature_names_prev = sfs_result.loc[i - 1, 'feature_names']
        feature_name = list(set(feature_names) - set(feature_names_prev))
        assert len(feature_name) == 1

        result_rows += [{
            'feature_name': feature_name[0],
            'rank_sequential_feature_selection': i,
        }]
    result = pd.DataFrame(result_rows).set_index('feature_name')

    # Get LDA rankings.
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(inputs_train, labels_train)
    result_lda = pd.DataFrame(lda.coef_[0], index=list(sfs.k_feature_names_))
    result_lda['rank'] = result_lda[0].abs().rank(ascending=False)
    result['rank_|linear_discriminant_analysis|'] = result_lda['rank'].astype(int)

    # Get PCC rankings.
    result_pcc = pd.concat(
        [inputs_train, labels_train],
        axis=1,
    ).corr()[['result translation']].drop('result translation')
    result_pcc['rank'] = result_pcc['result translation'].abs().rank(ascending=False)
    result['rank_|pearson_correlation_coefficient|'] = result_pcc['rank'].astype(int)

    # Get RF rankings.
    model.fit(inputs_train, labels_train)
    model = model.named_steps['cls']
    result_rf = pd.DataFrame(model.feature_importances_, index=model.feature_names_in_)
    result_rf['rank'] = result_rf[0].rank(ascending=False)
    result['rank_random_forest'] = result_rf['rank'].astype(int)

    # Get SHAP rankings.
    explainer = shap.Explainer(model)
    shap_values = explainer(inputs_train)
    with open(f"{path_cls_dir}/shap_values.pkl", 'wb') as f:
        pickle.dump(shap_values, f)

    result_shap = pd.DataFrame(
        np.absolute(shap_values[:, :, 1].values).mean(axis=0),
        index=model.feature_names_in_,
    )
    result_shap['rank'] = result_shap[0].rank(ascending=False)
    result['rank_|shap|'] = result_shap['rank'].astype(int)

    feature_names_cleaned = {
        'ohe_19__acute/sustained clinical_sustained': \
            'Clinical Acute/Sustained (Sustained: 1, Acute: 0)',
        'scaler_14__daily dosage preclinical': \
            'Preclinical Daily Dosage (mg)',
        'ohe_0__acute/sustained preclinical_sustained': \
            'Pre-clinical Acute/Sustained (Sustained: 1, Acute: 0)',
        'scaler_11__c diff dose value preclinical': \
            'Pre-clinical C. Diff. Dosage Value (CFU or cells)',
        'scaler_4__animal age value (days) preclinical': \
            'Pre-clinical Animal Age (days)',
        'scaler_23__dosage times per day clinical': \
            'Clinical Dosage Times Per Day',
        'ohe_8__prophylactic / therapeutic preclinical_t': \
            'Pre-clinical Treatment Type (Therapeutic: 1, Prophylactic: 0)',
        'ohe_26__Age Groups clinical_child': \
            'Clinical Age Groups (Child: 1, Others: 0)',
        'ohe_9__ribotype preclinical_1.0': \
            'Pre-clinical C. Diff. Ribotype: (001: 1, Others: 0)',
        'scaler_17__dosage times per day preclinical': \
            'Pre-clinical Dosage Times Per Day',
        'scaler_5__animal weight value (g) preclinical': \
            'Pre-clinical Animal Weight (g)',
        'scaler_16__dosage value preclinical': \
            'Pre-clinical Dosage Value (mg/kg)',
        'ohe_1__gnotobiotic preclinical_spf': \
            'Pre-clinical Gnotobiotic Model (Specific-Pathogen-Free: 1, Others: 0)',
        'ohe_2__animal strain preclinical_c57bl/6j': \
            'Pre-clinical Animal Strain (C57BL/6J: 1, Others: 0)',
        'ohe_7__disease model preclinical_kanamycin,gentamicin,colistin,metronidazole,'
        'vancomycin+clindamycin+challenge': \
            'Pre-clinical Disease Model (Mixed: 1, Others: 0)',
        'scaler_18__dosage duration preclinical': \
            'Pre-clinical Dosage Duration (days)',
    }
    result['feature_name_cleaned'] = [feature_names_cleaned[x] for x in result.index]

    result.to_csv(f"{path_cls_dir}/feature_importance.csv")
