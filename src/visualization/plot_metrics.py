import pickle

import pandas as pd
from msap.utils import get_all_metrics
import seaborn as sns
import matplotlib.pyplot as plt

from ..utils import constants, load_splits, load_best_pipeline, get_simple_pipeline


def get_baseline_metric(data, metric='f1'):
    """TODO:
    This is using the entire dataset to calculate the baseline F1 score.

    In reality, I should calculate baseline F1 score for each CV split and then average.

    This workd for now because the dataset is stratified.
    """
    counts = data['result translation'].value_counts()
    precision = counts[1] / (counts[1] + counts[0])
    recall = 1
    f1 = 2 * precision * recall / (precision + recall)

    if metric == 'f1':
        return f1
    elif metric == 'precision':
        return precision
    elif metric == 'recall':
        return recall


def get_model_predictions(
    model,
    data_train,
    data_test,
    sfs=None,
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
    # import matplotlib
    # sns.set_theme(style="whitegrid")
    # print(matplotlib.rcParams['font.family'])
    # exit()
    data_plot_rows =[]
    for i, thres in enumerate([0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]):
        path_data_dir \
            = f"outputs/data_processing/splits/cls_delta-{thres}_rs-42"
        path_cls_dir \
            = f"outputs/msap/cls_delta-{thres}_rs-42"

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

        y_test, y_pred, y_score \
            = get_model_predictions(model, data_train, data_test, sfs=sfs)
        baseline_f1 = get_baseline_metric(data_test, metric='f1')

        metrics = get_all_metrics(y_test, y_pred)

        for metric in ['precision', 'recall', 'f1']:
            data_plot_rows += [{
                r'$\delta_{threshold}$': f"{thres}" + r'$\sigma$',
                'Metric': metric.capitalize(),
                'Score': metrics[metric],
            }]
        data_plot_rows += [{
            r'$\delta_{threshold}$': f"{thres}" + r'$\sigma$',
            'Metric': 'F1' + r'$_{baseline}$',
            'Score': baseline_f1,
        }]

    data_plot = pd.DataFrame(data_plot_rows)

    sns.set_theme(style="whitegrid")
    g = sns.barplot(
        x=r'$\delta_{threshold}$',
        y='Score',
        hue='Metric',
        data=data_plot,
    )
    for bars in g.containers:
        g.bar_label(bars, fmt='%.2f')

    plt.savefig("outputs/visualization/metrics.svg")
    plt.close()
