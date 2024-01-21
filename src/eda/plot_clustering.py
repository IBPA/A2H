import numpy as np
import pandas as pd
from sklearn.cluster import SpectralBiclustering
from scipy.stats import chi2_contingency, ranksums
from statsmodels.stats.multitest import fdrcorrection
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from ..utils import constants, load_classification_data, get_simple_pipeline


def run_hierarchical_clustering():
    inputs, labels = load_classification_data(delta=0.5, is_sigma=True)
    inputs = inputs.drop(columns=['intervention'])
    inputs_transformed = get_simple_pipeline(
        inputs,
        constants.FEATURES_CAT,
        scaling_method_name='minmax',
    ).fit_transform(inputs)
    inputs_transformed = inputs_transformed.reset_index(drop=True)
    labels = labels.reset_index(drop=True).rename('label')
    data = pd.concat([inputs_transformed, labels], axis=1)

    color_mapper = {
        1: sns.color_palette()[0],
        0: sns.color_palette()[1],
    }
    sns.clustermap(
        data.drop(columns=['label']),
        metric='euclidean',
        row_colors=data['label'].map(color_mapper),
        cmap='Blues',
        figsize=(10, 20),
    )
    plt.savefig("outputs/eda/hc_all.png")
    plt.close()

    sns.clustermap(
        data.query('label == 0').drop(columns=['label']),
        metric='euclidean',
        row_colors=data['label'].map(color_mapper),
        cmap='Blues',
        figsize=(10, 20),
    )
    plt.savefig("outputs/eda/hc_0.png")
    plt.close()

    sns.clustermap(
        data.query('label == 0').drop(columns=['label']),
        metric='euclidean',
        row_colors=data['label'].map(color_mapper),
        cmap='Blues',
        figsize=(10, 20),
    )
    plt.savefig("outputs/eda/hc_1.png")
    plt.close()


def run_biclustering():
    inputs, labels = load_classification_data(delta=0.5, is_sigma=True)
    inputs = inputs.drop(columns=['intervention'])
    inputs_transformed = get_simple_pipeline(
        inputs,
        constants.FEATURES_CAT,
        scaling_method_name='minmax',
    ).fit_transform(inputs)
    inputs_transformed = inputs_transformed.reset_index(drop=True)
    labels = labels.reset_index(drop=True).rename('label')

    # column_names_orig = inputs_transformed.columns
    # inputs_transformed.columns = list(range(inputs_transformed.shape[1]))

    data = pd.concat([inputs_transformed, labels], axis=1)
    color_mapper = {
        1: sns.color_palette()[0],
        0: sns.color_palette()[1],
    }

    # Biclustering.
    N_ROW_CLUSTERS = 8
    N_COL_CLUSTERS = 64

    model = SpectralBiclustering(
        n_clusters=(N_ROW_CLUSTERS, N_COL_CLUSTERS),
        method='scale',
        random_state=0,
    )
    model.fit(inputs_transformed)
    data_reordered = data.iloc[:, model.column_labels_.argsort()]
    data_reordered = pd.concat([data_reordered, data['label']], axis=1)
    data_reordered = data_reordered.iloc[model.row_labels_.argsort()]

    feature_name_map = pd.read_csv("src/eda/feature_name_map.csv")
    feature_name_map = dict(
        zip(feature_name_map['feature_name_original'], feature_name_map['feature_name'])
    )
    data_reordered.columns = [
        feature_name_map[x] if x in feature_name_map else x
        for x in list(data_reordered.columns)
    ]

    # Plot.
    g = sns.clustermap(
        data_reordered.drop(columns=['label']),
        metric='euclidean',
        row_cluster=False,
        col_cluster=False,
        row_colors=data_reordered['label'].map(color_mapper),
        cmap='Blues',
        figsize=(20, 25),
    )
    for i in range(N_ROW_CLUSTERS - 1):
        g.ax_heatmap.axhline(
            (model.row_labels_ <= i).sum(),
            color='red',
            linewidth=1,
            linestyle='--',
        )
    for i in range(N_COL_CLUSTERS - 1):
        g.ax_heatmap.axvline(
            (model.column_labels_ <= i).sum(),
            color='red',
            linewidth=1,
            linestyle='--',
        )
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_xlabel("Feature ID")
    g.ax_heatmap.set_ylabel("Preclinical-Clinical Pair ID")
    plt.savefig("outputs/eda/biclustering.png")
    plt.close()

    # Just plot the x axis labels.
    data_reordered_small = data_reordered.iloc[:2]
    g = sns.clustermap(
        data_reordered_small.drop(columns=['label']),
        metric='euclidean',
        row_cluster=False,
        col_cluster=False,
        row_colors=data_reordered_small['label'].map(color_mapper),
        cmap='Blues',
        figsize=(20, 10),
    )
    # Legend for labels.
    handles = [
        Patch(facecolor=color_mapper[label]) for label in [1, 0]
    ]
    plt.legend(
        handles,
        ['Yes', 'No'],
        title='Translation\nSuccess',
        bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure,
        loc='upper right',
    )
    plt.savefig("outputs/eda/biclustering_xticklabels.svg")
    plt.close()

    data_dump = data_reordered.copy()
    data_dump['row_label'] = np.sort(model.row_labels_)
    data_dump = data_dump.set_index(['row_label', data_dump.index])
    columns = pd.MultiIndex.from_arrays(
        [
            list(np.sort(model.column_labels_)) + [''],
            data_dump.columns
        ],
        names=['col_label', 'feature'],
    )
    data_dump.columns = columns
    print(data_dump)
    data_dump.to_csv("outputs/eda/bc_data.csv")

    # for i in range(N_ROW_CLUSTERS):
    #     print(
    #         f"{i:<1} - {len(data_dump.loc[i]):<4} - "
    #         f"{data_dump.loc[i, ('', 'label')].mean():.3f}"
    #     )

    # data_reordered_subset = data_reordered.copy()
    # data_reordered_subset['row_label'] = np.sort(model.row_labels_)
    # data_reordered_subset = data_reordered_subset.query("row_label in [3, 5]")

    # # Plot.
    # g = sns.clustermap(
    #     data_reordered_subset.drop(columns=['label', 'row_label']),
    #     metric='euclidean',
    #     row_cluster=False,
    #     col_cluster=False,
    #     row_colors=data_reordered_subset['label'].map(color_mapper),
    #     cmap='Blues',
    #     figsize=(10, 2),
    # )
    # g.ax_heatmap.axhline(
    #     (model.row_labels_ == data_reordered_subset.iloc[0]['row_label']).sum(),
    #     color='red',
    #     linewidth=1,
    #     linestyle='--',
    # )
    # for i in range(N_COL_CLUSTERS - 1):
    #     g.ax_heatmap.axvline(
    #         (model.column_labels_ <= i).sum(),
    #         color='red',
    #         linewidth=1,
    #         linestyle='--',
    #     )
    # g.ax_heatmap.set_xlabel("Feature ID")
    # g.ax_heatmap.set_ylabel("Preclinical-Clinical Pair ID")
    # plt.savefig("outputs/translation/eda/bc_subset.png")
    # plt.close()


def run_biclustering_analysis():
    data = pd.read_csv(
        "outputs/eda/bc_data.csv",
        skiprows=[2],
        index_col=[0, 1],
        header=[1],
    )

    # Best cluster.
    result_rows = []
    for feature in data.columns[:-1]:
        value_counts = data[feature].value_counts()
        if len(value_counts) == 1:
            continue
        else:
            for i in [3, 5]:
                result = {
                    'cluster': i,
                    'feature': feature,
                }
                data_cluster = data.loc[i]
                for j in data.index.get_level_values(0).unique():
                    if i == j:
                        result['method'] = 'self'
                        result[f'cluster_{j}_p'] = np.nan
                        continue

                    data_other = data.loc[j]
                    if len(value_counts) == 2:
                        result['method'] = 'chi2'
                        neg_cluster = len(data_cluster.query(f"`{feature}` == 0.0"))
                        pos_cluster = len(data_cluster.query(f"`{feature}` == 1.0"))
                        neg_other = len(data_other.query(f"`{feature}` == 0.0"))
                        pos_other = len(data_other.query(f"`{feature}` == 1.0"))
                        try:
                            _, p, _, expected = chi2_contingency([
                                [neg_cluster, pos_cluster],
                                [neg_other, pos_other],
                            ])
                            if (expected < 5).any():
                                result[f'cluster_{j}_p'] = np.nan
                            else:
                                result[f'cluster_{j}_p'] = p
                        except Exception:
                            result[f'cluster_{j}_p'] = np.nan

                    else:
                        # Numerical, so use ranksum.
                        result['method'] = 'ranksum'
                        _, p = ranksums(
                            data_cluster[feature],
                            data_other[feature],
                        )
                        result[f'cluster_{j}_p'] = p
                result_rows += [result]
    result = pd.DataFrame(result_rows)
    result = result.sort_values(['cluster'])

    # FDR.
    p_vals_df = result[result.columns[-8:]]
    p_vals = p_vals_df.values.flatten()
    p_vals = p_vals[~np.isnan(p_vals)]
    p_vals = list(fdrcorrection(p_vals, alpha=0.01)[1])

    p_vals_np = p_vals_df.copy().values
    for i in range(p_vals_np.shape[0]):
        for j in range(p_vals_np.shape[1]):
            if np.isnan(p_vals_np[i, j]):
                continue
            p_vals_np[i, j] = p_vals.pop(0)
    p_vals_df = pd.DataFrame(
        p_vals_np,
        columns=p_vals_df.columns,
        index=p_vals_df.index,
    )
    result[result.columns[-8:]] = p_vals_df

    result['significant'] = result[result.columns[-8:]].apply(
        lambda row: False if row.notna().sum() != 7 else row.max() < 0.01,
        axis=1,
    )
    result.to_csv("outputs/eda/bc_result.csv")


if __name__ == '__main__':
    run_hierarchical_clustering()
    run_biclustering()
    run_biclustering_analysis()
