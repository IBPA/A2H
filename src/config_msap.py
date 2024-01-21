# -*- coding: utf-8 -*-
"""The configuration for using MSAP.

Attributes:
    COLUMNS_CATEGORICAL (list): List of categorical columns.
    PARAMS_OD (dict): Parameters for outlier detection methods.
    PARAMS_MVI (dict): Parameters for missing value imputation methods.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

"""
from .utils import constants

# Experiment parameters.
COLUMNS_CATEGORICAL = constants.FEATURES_CAT

# Hyperparameters for the ourlier detection methods.
PARAMS_OD = {
    'none': {},
}

# Hyperparameters for the missing value imputation methods.
PARAMS_MVI = {
    'simple': {},
}

# Hyperparameters for the grid search.
PARAMS_GRID = {
    'rf': {
        'n_estimators': [100, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [8, 10, 12, 14],
        'max_features': ['log2'],
        'random_state': [None],
    },
    'ada': {
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.1, 0.5, 1.0],
        'algorithm': ['SAMME.R'],
        'random_state': [None],
    },
    'svc': {
        'C': [0.1, 1],
        'kernel': ['linear', 'rbf'],
        'probability': [True],
        'random_state': [None],
    },
    'mlp': {
        'hidden_layer_sizes': [(100,), (100, 100), (100, 100, 100)],
        'learning_rate': ['adaptive'],
        'learning_rate_init': [1e-5, 1e-4, 1e-3],
        'max_iter': [1000],
        'n_iter_no_change': [5],
        'random_state': [None],
    },
}
