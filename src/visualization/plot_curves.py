import pickle

from sklearn.metrics import (
    precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
)
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


def plot_pr_curve(
    y_true,
    y_score,
    baseline=None,
    label=None,
):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score[:, 1])
    aucpr = average_precision_score(y_true, y_score[:, 1])
    f1_oop = -1
    prec_oop = -1
    rec_oop = -1
    # thred_oop = -1
    for prec, rec, thred in zip(precisions, recalls, thresholds):
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > f1_oop:
            f1_oop = f1
            prec_oop = prec
            rec_oop = rec
            # thred_oop = thred

    g = sns.lineplot(
        x=recalls,
        y=precisions,
        estimator=None,
        sort=False,
        label=f"{label} (AUCPR = {aucpr:.2f})" if label is not None else None,
    )
    g.set_xlabel('Recall')
    g.set_ylabel('Precision')
    g.set_title(
        "Precision-recall Curve"
    )
    g.plot(
        [rec_oop],
        [prec_oop],
        marker='o',
        color=g.get_lines()[-1].get_color(),
    )


def plot_roc_curve(
    y_true,
    y_score,
    baseline=None,
    label=None,
):
    fprs, tprs, thresholds = roc_curve(y_true, y_score[:, 1])
    aucroc = roc_auc_score(y_true, y_score[:, 1])
    g = sns.lineplot(
        x=fprs,
        y=tprs,
        estimator=None,
        sort=False,
        label=f"{label} (AUCROC = {aucroc:.2f})" if label is not None else None,
    )
    if baseline:
        g.plot(
            [0, 1],
            [0, 1],
            linestyle='--',
            color='grey',
            label="Baseline (AUCPR = 0.50)",
        )
    g.set_xlabel('False Positive Rate')
    g.set_ylabel('True Positive Rate')
    g.set_title(
        "Receiver Operating Characteristic Curve"
    )


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")

    for thres in [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]:
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
        label = f"{thres}" + r'$\sigma$'
        plot_pr_curve(
            y_test,
            y_score,
            baseline=baseline_f1,
            label=label,
        )
        plt.legend(title=r'$\delta_{threshold}$')

    plt.savefig("outputs/visualization/pr_curves.svg")
    plt.close()

    for i, thres in enumerate([2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]):
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
        label = f"{thres}" + r'$\sigma$'
        plot_roc_curve(
            y_test,
            y_score,
            baseline=True if i == 5 else False,
            label=label,
        )
        plt.legend(title=r'$\delta_{threshold}$')
    plt.savefig("outputs/visualization/roc_curves.svg")
    plt.close()
