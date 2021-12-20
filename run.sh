#!/bin/bash

CONDA_INSTALL_PATH="/opt/miniconda"
CONDA_BIN_PATH="$CONDA_INSTALL_PATH/bin"

PATH="$CONDA_BIN_PATH:$PATH"
CONDA_ENV_NAME=pcr

. $CONDA_INSTALL_PATH/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME

python ./datasets/ShapeNetPOVDepth.py