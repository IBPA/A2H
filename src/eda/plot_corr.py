import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
)
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

from ..utils import constants
from ..utils import load_classification_data

set_config(transform_output='pandas')


def load_data_transformed():

    def _get_fs(method_name):
        """Get the scaling pipeline.

        Args:
            method_name (str): Name of the scaling method.

        Returns:
            TODO

        Raises:
            ValueError: If the method name is not recognized.

        """
        if method_name == 'standard':
            return StandardScaler()
        elif method_name == 'minmax':
            return MinMaxScaler()
        elif method_name == 'none':
            return FunctionTransformer(feature_names_out='one-to-one')
        else:
            raise ValueError(f"Unknown scaling method: {method_name}")


    def _get_ohe(categories):
        """Get the pipeline for one-hot encoding.

        Args:
            categories (list): List of categories.

        Returns:
            TODO

        """
        return OneHotEncoder(
            categories=categories,
            drop='if_binary',
            handle_unknown='ignore',
            sparse_output=False,
        )


    def get_ft_pipeline(
        data,
        scaling_method_name,
        categorical_features=[],
    ):
        """Get the pipeline for feature transformation. The feature
        transformation includes scaling for numerical and one-hot encoding for
        categorical features.

        Args:
            data (pd.DataFrame): Data to be transformed.
            scaling_method (str): Name of the scaling method.
            categorical_features (list): List of categorical features.

        Returns:
            (Pipeline): Pipeline for feature transformation.

        """
        return ColumnTransformer(
            transformers=[
                (f'ohe_{i}', _get_ohe([np.sort(data[col].unique())]), [col])
                if col in categorical_features else
                (f'scaler_{i}', _get_fs(scaling_method_name), [col])
                for i, col in enumerate(data.columns)
            ],
        )

    def _get_simple_pipeline(
        data,
        categorical_features=[]
    ):
        """Get the pipeline for mean/mode imputation.

        Args:
            data (pd.DataFrame): Data to be imputed.
            categorical_features (list): List of categorical features.

        Returns:
            (Pipeline): Pipeline for mean/mode imputation.

        """
        return Pipeline([
            ('simple', ColumnTransformer(
                transformers=[
                    (
                        f"mode_{i}",
                        SimpleImputer(strategy='most_frequent'),
                        [col],
                    ) if col in categorical_features else
                    (
                        f"mean_{i}",
                        SimpleImputer(strategy='mean'),
                        [col],
                    ) for i, col in enumerate(data.columns)
                ],
                verbose_feature_names_out=False,
            )),
        ])


    def get_mvi_pipeline(method_name, params={}):
        """Get the pipeline for missing value imputation.

        Args:
            method_name (str): Name of the missing value imputation method.
            params (dict): Parameters for the missing value imputation method.

        Returns:
            (Pipeline): Pipeline for missing value imputation.

        Raises:
            ValueError: If the method name is not supported.

        """
        if method_name == 'simple':
            return _get_simple_pipeline(**params)
        else:
            raise ValueError(f'Invalid method name: {method_name}')

    inputs, labels = load_classification_data(delta=0.5, is_sigma=True)
    inputs = inputs.drop(columns=['intervention'])
    inputs = Pipeline(
        steps=[
            ('mvi', get_mvi_pipeline(
                method_name='simple',
                params={
                    'data': inputs,
                    'categorical_features': constants.FEATURES_CAT,
                },
            )),
            ('ft', get_ft_pipeline(
                data=inputs,
                scaling_method_name='none',
                categorical_features=constants.FEATURES_CAT,
            )),
        ]
    ).fit_transform(inputs)

    return inputs, labels


def plot_corr_heatmap(method='pearson', path_save_dir=None):
    """
    """
    X, y = load_data_transformed()
    data = pd.concat([X, y], axis=1)
    corr = data.corr(method=method)

    # fig, ax = plt.subplots(figsize=(28, 28))
    # sns.heatmap(
    #     corr,
    #     vmin=-1,
    #     vmax=1,
    #     square=True,
    #     mask=np.triu(np.ones_like(corr, dtype=bool)),
    # )
    # plt.savefig(f"{path_save_dir}/corr_{method}.svg")
    # plt.close()

    if method == 'pearson':
        p_val = data.corr(method=lambda x, y: pearsonr(x, y)[1])
    elif method == 'spearman':
        p_val = data.corr(method=lambda x, y: spearmanr(x, y)[1])
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    corr_result = pd.concat(
        [corr['result translation'], p_val['result translation']], axis=1
    )
    corr_result.columns = ['corr', 'p_val']
    corr_result = corr_result.sort_values(
        by='p_val', ascending=True
    )
    corr_result.to_csv(f"{path_save_dir}/corr_result_{method}.csv")


def plot_corr_bar_top_10(method='pearson', path_save_dir=None):
    data_corr = pd.read_csv(f"{path_save_dir}/corr_result_{method}.csv", index_col=0)
    data_corr = data_corr.sort_values(
        by='corr', key=abs, ascending=False
    ).iloc[1:1 + 10]

    # data_corr.index = [
    #     'Pre-clinical Acute/Sustained (Sustained: 1, Acute: 0)',
    #     'Pre-clinical Gnotobiotic Model (Specific-Pathogen-Free: 1, Others: 0)',
    #     'Pre-clinical Gnotobiotic Model (Unknown: 1, Others: 0)',
    #     'Pre-clinical C. Diff. Strain (2009155: 1, Others: 0)',
    #     'Pre-clinical C. Diff. Ribotype (428: 1, Others: 0)',
    #     'Pre-clinical C. Diff. Strain (VA11: 1, Others: 0)',
    #     'Pre-clinical C. Diff. Strain (630: 1, Others: 0)',
    #     'Pre-clinical Disease Model (Clindamycin: 1, Others: 0)',
    #     'Pre-clinical Disease Model (Mixed: 1, Others: 0)',
    #     'Pre-clinical Dosage Value (mg/kg)'
    # ]
    # feature_names_cleaned = {
    #     'ohe_19__acute/sustained clinical_sustained': \
    #         'Clinical Acute/Sustained (Sustained: 1, Acute: 0)',
    #     'ohe_0__acute/sustained preclinical_sustained': \
    #         'Preclinical Acute/Sustained (Sustained: 1, Acute: 0)',
    #     'ohe_10__c diff strain preclinical_va11': \
    #         'Preclinical C. Diff. Strain (VA11: 1, Others: 0)',
    #     'ohe_9__ribotype preclinical_428.0': \
    #         'Preclinical C. Diff. Ribotype (428: 1, Others: 0)',
    #     'ohe_26__Age Groups clinical_child': \
    #         'Clinical Age Groups (Child: 1, Others: 0)',
    #     'ohe_10__c diff strain preclinical_2009155': \
    #         'Preclinical C. Diff. Strain (2009155: 1, Others: 0)',
    #     'ohe_1__gnotobiotic preclinical_spf': \
    #         'Preclinical Gnotobiotic Model (Specific-Pathogen-Free: 1, Others: 0)',
    #     'ohe_1__gnotobiotic preclinical_Unknown': \
    #         'Preclinical Gnotobiotic Model (Unknown: 1, Others: 0)',
    #     'ohe_10__c diff strain preclinical_630': \
    #         'Preclinical C. Diff. Strain (630: 1, Others: 0)',
    #     'scaler_17__dosage times per day preclinical': \
    #         'Preclinical Dosage Times Per Day',
    #     'scaler_15__total dosage preclinical': \
    #         'Preclinical Total Dosage',
    #     'scaler_14__daily dosage preclinical': \
    #         'Preclinical Daily Dosage (mg/kg)',
    #     'scaler_16__dosage value preclinical': \
    #         'Preclinical Dosage Value (mg/kg)',
    #     'scaler_4__animal age value (days) preclinical': \
    #         'Preclinical Animal Age (days)',
    # }

    feature_name_map = pd.read_csv("src/eda/feature_name_map.csv")
    feature_name_map = dict(
        zip(feature_name_map['feature_name_original'], feature_name_map['feature_name'])
    )
    data_corr.index = [
        feature_name_map[x] for x in data_corr.index
    ]
    data_corr['color'] = [
        'orange' if x < 0 else 'lightblue' for x in data_corr['corr']
    ]
    g = sns.barplot(
        data=data_corr,
        x=data_corr.index,
        y='corr',
        palette=data_corr['color'].tolist(),
    )
    g.set_ylabel('Correlation coefficient')
    g.set_xlabel('Feature')
    g.set_xticklabels(
        g.get_xticklabels(),
        rotation=90,
        # horizontalalignment='right',
    )
    plt.savefig(f"{path_save_dir}/corr_result_top_10_{method}.svg")
    pass


if __name__ == '__main__':
    plot_corr_heatmap(
        method='spearman',
        path_save_dir="outputs/eda"
    )
    plot_corr_bar_top_10(
        method='spearman',
        path_save_dir="outputs/eda"
    )
