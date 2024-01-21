import pickle

import pandas as pd
from kneed import KneeLocator
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    DELTA = 0.5

    with open(
        "outputs/msap/"
        f"cls_delta-{DELTA}_rs-42/classifiers/sfs.pkl",
        'rb'
    ) as f:
        sfs = pickle.load(f)
    k_par, score_par = len(sfs.k_feature_idx_), sfs.k_score_

    sfs_result = pd.DataFrame(sfs.get_metric_dict()).T
    sfs_result_best = sfs_result.sort_values('avg_score', ascending=False).iloc[0]
    k_best, score_best \
        = len(sfs_result_best['feature_idx']), sfs_result_best['avg_score']

    kneedle = KneeLocator(
        range(1, len(sfs_result) + 1),
        sfs_result['avg_score'],
        S=0.1,
        curve='concave',
        direction='increasing',
    )
    k_elbow, score_elbow = kneedle.elbow, kneedle.elbow_y

    print(k_par, score_par)
    print(k_best, score_best)
    print(k_elbow, score_elbow)

    sns.set_theme(style="whitegrid")

    plot_sfs(
        sfs.get_metric_dict(),
        kind='std_dev',
        figsize=(6.4, 4.8),
        ylabel='F1 Score',
    )
    ymin, ymax = plt.gca().get_ylim()
    for k, score, method, color in zip(
        [k_best, k_par, k_elbow],
        [score_best, score_par, score_elbow],
        ['Best F1 Score', 'Parsimonious', 'Elbow'],
        ['red', 'green', 'blue'],
    ):
        plt.vlines(
            k,
            ymin=ymin,
            ymax=ymax,
            ls='--',
            color=color,
            label=f"{method} (F1 = {score:.3f})",
        )
    idxs_visible = [1, len(sfs.subsets_)] + [k_par, k_best, k_elbow]
    plt.title('Sequential Forward Selection (w. StdDev)')
    plt.xticks(
        range(1, len(sfs.subsets_) + 1),
        [
            i if i in idxs_visible else ''
            for i in range(1, len(sfs.subsets_) + 1)
        ],
    )
    plt.legend(title="Feature Selection Criteria")
    plt.savefig("outputs/visualization/rfe.svg")
