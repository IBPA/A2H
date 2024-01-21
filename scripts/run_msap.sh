#!/bin/bash

PATH_DATA_DIR=outputs/data_processing/splits
PATH_OUTPUTS_DIR=outputs/msap
PATH_CONFIG=src.config_msap
COLUMN_TARGET="result translation"

CLS_METHODS=rf,ada,mlp
OD_METHODS=none
MVI_METHODS=simple
FS_METHODS=minmax,standard,none
OS_METHODS=smote,none
SCORING=f1

RANDOM_STATE=42

for delta in 0.0625 0.125 0.25 0.5 1.0 2.0; do
    python -m src.data_processing.run_data_splitting \
        outputs/data_processing/splits \
        --delta $delta \
        --test-size 0.2 \
        --n-folds 5 \
        --random-state $RANDOM_STATE

    python -m msap.run_preprocess \
        $PATH_DATA_DIR/cls_delta-${delta}_rs-${RANDOM_STATE}/data_train.csv \
        $PATH_OUTPUTS_DIR/cls_delta-${delta}_rs-${RANDOM_STATE} \
        --path-config $PATH_CONFIG \
        --column-target "$COLUMN_TARGET" \
        --od-methods $OD_METHODS \
        --mvi-methods $MVI_METHODS \
        --fs-methods $FS_METHODS \
        --random-state $RANDOM_STATE

    python -m msap.run_grid_search \
        $PATH_OUTPUTS_DIR/cls_delta-${delta}_rs-${RANDOM_STATE} \
        --path-config $PATH_CONFIG \
        --column-target "$COLUMN_TARGET" \
        --cls-methods $CLS_METHODS \
        --od-methods $OD_METHODS \
        --mvi-methods $MVI_METHODS \
        --fs-methods $FS_METHODS \
        --os-methods $OS_METHODS \
        --path-grid-search-splits $PATH_DATA_DIR/cls_delta-${delta}_rs-${RANDOM_STATE}/cv_splits.pkl \
        --grid-search-scoring $SCORING \
        --random-state $RANDOM_STATE

    python -m src.run_feature_selection \
        $PATH_OUTPUTS_DIR/cls_delta-${delta}_rs-${RANDOM_STATE} \
        $PATH_DATA_DIR/cls_delta-${delta}_rs-${RANDOM_STATE}

    python -m src.run_evaluation \
        $PATH_OUTPUTS_DIR/cls_delta-${delta}_rs-${RANDOM_STATE} \
        $PATH_DATA_DIR/cls_delta-${delta}_rs-${RANDOM_STATE}
done