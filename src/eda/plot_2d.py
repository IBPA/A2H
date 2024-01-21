import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

from ..utils import constants, load_classification_data, get_simple_pipeline


def plot_2d_tsne(
    label_name: str,
    perplexity: int = 30,
    figsize: tuple = (6.4, 4.8),
    path_save_dir: str|None = None,
) -> pd.DataFrame:
    """Plot 2D scatter plot.

    Args:
        inputs: Input data.
        labels: Labels.
        path_save: Path to save the figure. Defaults to None.

    Returns:
        2D t-SNE data.

    """
    inputs, labels = load_classification_data(delta=0.5, is_sigma=True)
    inputs = inputs.drop(columns=['intervention'])
    inputs_transformed = get_simple_pipeline(
        inputs,
        constants.FEATURES_CAT,
        scaling_method_name='standard',
    ).fit_transform(inputs)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        n_jobs=-1,
    )
    cols = ['t-SNE 1', 't-SNE 2']

    X_2d = tsne.fit_transform(inputs_transformed)
    X_2d.columns = cols
    X_2d[label_name] = labels.replace({0: 'Failure', 1: 'Success'})
    X_2d = X_2d.reset_index()

    sns.set_theme(style='whitegrid')
    plt.figure(figsize=figsize)

    # Main plot.
    sns.scatterplot(
        data=X_2d,
        x=cols[0],
        y=cols[1],
        hue=label_name,
        hue_order=['Success', 'Failure'],
        s=10,
    )
    plt.savefig(f"{path_save_dir}/tsne_main.svg")
    plt.close()

    # Cluster 1.
    (tsne_1_min, tsne_1_max), (tsne_2_min, tsne_2_max) = ((-10, 10), (-50, -40))
    X_2d_cluster_1_df = X_2d.loc[
        (X_2d['t-SNE 1'] < tsne_1_max)
        & (X_2d['t-SNE 1'] > tsne_1_min)
        & (X_2d['t-SNE 2'] < tsne_2_max)
        & (X_2d['t-SNE 2'] > tsne_2_min)
    ].copy()
    X_2d_cluster_1_df['cluster'] = 1

    plt.figure(figsize=figsize)
    g = sns.scatterplot(
        data=X_2d_cluster_1_df,
        x=cols[0],
        y=cols[1],
        hue=label_name,
        hue_order=['Success', 'Failure'],
    )
    g.set(
        xlim=(-10, 10),
        ylim=(-50, -40),
    )
    plt.savefig(f"{path_save_dir}/tsne_c1.svg")
    plt.close()

    # Cluster 2.
    (tsne_1_min, tsne_1_max), (tsne_2_min, tsne_2_max) = ((-52.5, -42.5), (87.5, 97.5))
    X_2d_cluster_2_df = X_2d.loc[
        (X_2d['t-SNE 1'] < tsne_1_max)
        & (X_2d['t-SNE 1'] > tsne_1_min)
        & (X_2d['t-SNE 2'] < tsne_2_max)
        & (X_2d['t-SNE 2'] > tsne_2_min)
    ].copy()
    X_2d_cluster_2_df['cluster'] = 2

    plt.figure(figsize=figsize)
    g = sns.scatterplot(
        data=X_2d_cluster_2_df,
        x=cols[0],
        y=cols[1],
        hue=label_name,
        hue_order=['Success', 'Failure'],
    )
    # g.set(
    #     xlim=(tsne_1_min - 5, tsne_1_max + 5),
    #     ylim=(tsne_2_min - 5, tsne_2_max + 5),
    # )
    plt.savefig(f"{path_save_dir}/tsne_c2.svg")
    plt.close()

    X_2d_clustered = pd.concat([X_2d_cluster_1_df, X_2d_cluster_2_df])
    X_2d_clustered.to_csv(f"{path_save_dir}/clustered_2d.csv")

    X_inputs_clustered = pd.concat([inputs, labels], axis=1).reset_index()
    X_inputs_clustered = X_inputs_clustered.loc[X_2d_clustered.index].copy()
    X_inputs_clustered['cluster'] = X_2d_clustered['cluster']
    X_inputs_clustered.to_csv(f"{path_save_dir}/clustered_inputs.csv")


if __name__ == '__main__':
    plot_2d_tsne(
        label_name="Translation (|" + r'$\delta$' + "| < 0.5" + r'$\sigma$' + ")",
        perplexity=10,
        figsize=(6.0, 4.8),
        path_save_dir="outputs/eda",
    )
