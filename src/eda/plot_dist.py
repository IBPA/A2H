import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..utils import load_classification_data


def plot_dist_preclinical(path_save=None):
    """Plot the distribution of survival rate in the preclinical dataset. Only the
    unique preclinical groups used in the paired dataset are included.

    Args:
        path_save: Path to save the figure. Defaults to None.

    Returns:
        None

    """
    data = pd.read_csv(
        "outputs/data_processing/pairs_cleaned.tsv", sep='\t'
    )

    # Get only the preclinical information.
    data['ID preclinical'] = data['group_id'].apply(lambda x: x.split('_')[0])
    data = data[
        [c for c in data.columns.tolist() if ' preclinical' in c]
        +
        ['intervention']
    ]
    data = data.drop_duplicates()

    sns.set_theme(style='whitegrid')
    g = sns.histplot(
        data['result preclinical'],
        bins=10,
    )
    g.set(
        xlabel='Survival Rate',
        ylabel='Count',
    )

    g2 = plt.twinx()
    sns.kdeplot(
        data['result preclinical'],
        color='red',
        legend=False,
        ax=g2,
    )
    g2.set(
        ylabel='Density',
    )

    plt.savefig(path_save)
    plt.close()


def plot_dist_clinical(path_save=None):
    """Plot the distribution of recovery rate in the clinical dataset. Only the
    unique clinical groups used in the paired dataset are included.

    Args:
        path_save: Path to save the figure. Defaults to None.

    Returns:
        None

    """
    data = pd.read_csv(
        "outputs/data_processing/pairs_cleaned.tsv", sep='\t'
    )

    # Get only the preclinical information.
    data['ID clinical'] = data['group_id'].apply(lambda x: x.split('_')[1])
    data = data[
        [c for c in data.columns.tolist() if ' clinical' in c]
        +
        ['intervention']
    ]
    data = data.drop_duplicates()

    sns.set_theme(style='whitegrid')
    g = sns.histplot(
        data['result clinical'],
        bins=10,
    )
    g.set(
        xlabel='Recovery Rate',
        ylabel='Count',
    )

    g2 = plt.twinx()
    sns.kdeplot(
        data['result clinical'],
        color='red',
        legend=False,
        ax=g2,
    )
    g2.set(
        ylabel='Density',
    )

    plt.savefig(path_save)
    plt.close()


def plot_dist_delta(path_save):
    """TODO.
    """
    data = pd.read_csv(
        "outputs/data_processing/pairs_cleaned.tsv", sep='\t'
    ).set_index('group_id')
    data['result delta'] = (data['result clinical'] - data['result preclinical'])
    # data['result delta'] = (data['result delta'] - data['result delta'].mean()) \
    #     / data['result delta'].std()

    sns.set_theme(style='whitegrid')
    g = sns.histplot(
        data['result delta'],
        bins=10,
    )
    g.set(
        # xlabel=r'$\sigma$' + ' of ' + r'$\delta$' + ' (= %Recovery - %Survival)',
        xlabel=r'$\delta$' + ' (= %Recovery - %Survival)',
        ylabel='Count',
    )

    g2 = plt.twinx()
    sns.kdeplot(
        data['result delta'],
        color='red',
        legend=False,
        ax=g2,
    )
    g2.set(
        ylabel='Density',
    )

    plt.savefig(path_save)
    plt.close()


def plot_dist_thresholds(path_save):
    """TODO.
    """
    data_plot_rows = []
    for t in [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]:
        _, labels = load_classification_data(delta=t, is_sigma=True)
        counts = labels.value_counts()
        data_plot_rows += [{
            'thres': f"{t}" + r'$\sigma$',
            'Label': 'Success',
            'Count': counts[1],
        }]
        data_plot_rows += [{
            'thres': f"{t}" + r'$\sigma$',
            'Label': 'Failure',
            'Count': counts[0],
        }]
    data_plot = pd.DataFrame(data_plot_rows)

    sns.set_theme(style='whitegrid')
    g = sns.barplot(
        data=data_plot,
        x='thres',
        y='Count',
        hue='Label',
    )
    g.set_xlabel(r'$\delta_{Threshold}$')

    plt.savefig(path_save)
    plt.close()


if __name__ == '__main__':
    os.makedirs("outputs/eda", exist_ok=True)
    plot_dist_preclinical(
        path_save="outputs/eda/dist_survival_rate.svg"
    )
    plot_dist_clinical(
        path_save="outputs/eda/dist_recovery_rate.svg"
    )
    plot_dist_delta(
        path_save="outputs/eda/dist_delta.svg"
    )
    plot_dist_thresholds(
        path_save="outputs/eda/dist_thresholds.svg"
    )
