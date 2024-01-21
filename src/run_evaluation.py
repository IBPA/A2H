import os
import sys
import warnings
import pickle

import pandas as pd
from msap.utils import get_all_metrics
from sklearn.metrics import precision_recall_curve, roc_curve
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import seaborn as sns
import matplotlib.pyplot as plt
import click

from .utils import constants, load_splits, load_best_pipeline, get_simple_pipeline

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


def get_baseline_metric(data, metric='f1'):
    """
    This is using the entire dataset to calculate the baseline F1 score.

    In reality, I should calculate baseline F1 score for each CV split and then average.

    This works for now because the dataset is stratified.
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


def get_model_predictions(model, data_train, data_test, sfs=None):
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


def plot_grid_search_performance(gs_result, baseline_f1, title=None, path_save=None):
    """
    F1 score Boxes.
    """
    gs_result = gs_result[['cls_method', 'os_method', 'mean_test_f1']]
    gs_result = gs_result.rename(
        columns={
            'mean_test_f1': 'F1 score',
            'cls_method': 'Classifier',
            'os_method': 'Oversampling',
        }
    )
    gs_result['Oversampling'] = gs_result['Oversampling'].map({
        'none': 'None',
        'smote': 'SMOTE',
    })
    gs_result['Classifier'] = gs_result['Classifier'].map({
        'rf': 'RF',
        'ada': 'ADA',
        'mlp': 'MLP',
    })

    sns.set_theme(style='whitegrid')
    g = sns.boxplot(
        data=gs_result,
        x='Classifier',
        y='F1 score',
        hue='Oversampling',
        order=['RF', 'MLP', 'ADA'],
        hue_order=['SMOTE', 'None'],
    )
    g.axhline(baseline_f1, ls='--', color='black', label='Baseline')
    g.set_title(title)
    g.legend()
    plt.savefig(path_save)
    plt.close()


def plot_best_model_performance(
        model,
        data_train,
        data_test,
        path_save,
        sfs=None
    ):
    y_test, y_pred, y_score \
        = get_model_predictions(model, data_train, data_test, sfs=sfs)

    metrics = get_all_metrics(y_test, y_pred, y_score[:, 1])

    sns.set_theme(style='whitegrid')
    _, axs = plt.subplots(1, 2, figsize=(12, 5))

    # PR-curve.
    precisions, recalls, threasholds = precision_recall_curve(y_test, y_score[:, 1])
    f1_pr_oop = -1
    prec_oop = -1
    rec_oop = -1
    thred_oop = -1
    for prec, rec, thred in zip(precisions, recalls, threasholds):
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > f1_pr_oop:
            f1_pr_oop = f1
            prec_oop = prec
            rec_oop = rec
            thred_oop = thred
    baseline_pr = get_baseline_metric(data_test, metric='precision')

    g = sns.lineplot(
        x=recalls,
        y=precisions,
        errorbar=None,
        ax=axs[0],
    )
    g.set_xlabel('Recall')
    g.set_ylabel('Precision')
    g.set_title(
        f"Precision-recall Curve (AUCPR = {metrics['ap']:.3f})")
    g.plot(
        [0, 1],
        [baseline_pr, baseline_pr],
        ls='--',
        color='black',
        label='Baseline (Always Predict Positive)'
    )
    g.plot(
        [rec_oop],
        [prec_oop],
        marker='o',
        color='red',
        label=f"OOP (F1 = {f1_pr_oop:.2f}; Threshold = {thred_oop:.2f})",
    )
    g.set_yticks(
        [*g.get_yticks(), baseline_pr],
        [*g.get_yticklabels(), f"{baseline_pr:.2f}"],
    )
    g.legend()

    # ROC-curve.
    fprs, tprs, threasholds = roc_curve(y_test, y_score[:, 1])
    j_stat_roc_oop = -1
    fpr_oop = -1
    tpr_oop = -1
    thred_oop = -1
    for fpr, tpr, thred in zip(fprs, tprs, threasholds):
        j_stat = tpr - fpr
        if j_stat > j_stat_roc_oop:
            j_stat_roc_oop = j_stat
            fpr_oop = fpr
            tpr_oop = tpr
            thred_oop = thred

    g = sns.lineplot(
        x=fprs,
        y=tprs,
        errorbar=None,
        ax=axs[1],
    )
    g.set_xlabel('False Positive Rate')
    g.set_ylabel('True Positive Rate')
    g.set_title(
        f"Receiver Operating Characteristic Curve (AUCROC = {metrics['auroc']:.3f})")
    g.plot(
        [0, 1],
        [0, 1],
        ls='--',
        color='black',
        label='Baseline (Always Predict Positive)'
    )
    g.plot(
        [fpr_oop],
        [tpr_oop],
        marker='o',
        color='red',
        label=f"OOP (Youden's J = {j_stat_roc_oop:.2f}; Threshold = {thred_oop:.2f})",
    )
    g.legend()
    plt.savefig(path_save)
    plt.close()


def plot_best_model_sfs_performance(
    sfs,
    model,
    data_train,
    data_test,
    path_save_dir,
):
    # SFS curve.
    plot_sfs(
        sfs.get_metric_dict(),
        kind='std_dev',
        figsize=(10, 6),
    )
    idxs_visible = [1, len(sfs.subsets_)]
    for i in range(1, len(sfs.subsets_) + 1):
        if i % 5 == 0:
            idxs_visible.append(i)
    plt.xticks(
        range(1, len(sfs.subsets_) + 1),
        [
            i if i in idxs_visible else ''
            for i in range(1, len(sfs.subsets_) + 1)
        ],
    )
    plt.vlines(
        len(sfs.k_feature_names_),
        ymin=plt.gca().get_ylim()[0],
        ymax=plt.gca().get_ylim()[1],
        ls='--',
        color='red',
    )
    plt.text(
        len(sfs.k_feature_names_) * 1.02,
        plt.gca().get_ylim()[0] + 0.1,
        f"Optimal K = {len(sfs.k_feature_names_)}",
        color='red',
    )
    plt.title('Sequential Forward Selection (w. 5-fold CV s.t.d.)')
    plt.ylabel('F1 Score')
    plt.grid()
    plt.savefig(f"{path_save_dir}/best_model_sfs.svg")
    plt.close()

    # Curves.
    plot_best_model_performance(
        model,
        data_train,
        data_test,
        f"{path_save_dir}/best_model_sfs_performance_curves.svg",
        sfs=sfs,
    )


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
    cls_ = path_cls_dir.split('/')[-1]
    if cls_.split('_')[1].startswith('delta'):
        symbol = r'$\delta$'
    else:
        symbol = "N"
    value = cls_.split('_')[1].split('-')[1]

    data_train, data_test, _ = load_splits(path_data_dir)

    # Plot model selection results.
    gs_result = pd.read_csv(f"{path_cls_dir}/classifiers/grid_search_results.csv")
    plot_grid_search_performance(
        gs_result,
        baseline_f1=get_baseline_metric(data_train, metric='f1'),
        title=f"5-fold Cross Validation F1 Score ({symbol} = {value})",
        path_save=f"{path_cls_dir}/evaluation/grid_search.svg",
    )
    # # Plot test evaluation for the best model.
    best_model = load_best_pipeline(path_cls_dir)

    # Plot feature selection outcome.
    with open(f"{path_cls_dir}/classifiers/sfs.pkl", 'rb') as f:
        sfs = pickle.load(f)
    plot_best_model_sfs_performance(
        sfs,
        best_model,
        data_train,
        data_test,
        f"{path_cls_dir}/evaluation",
    )


if __name__ == '__main__':
    main()
