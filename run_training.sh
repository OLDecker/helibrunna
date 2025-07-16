#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=xlstm_test
#SBATCH --output=training_output_%j.log
#SBATCH --error=training_error_%j.log

# Load conda environment
module load devel/miniforge
mamba activate xlstm
echo $CONDA_DEFAULT_ENV

# Run the training
$CONDA_PREFIX/bin/python train.py configs/go_xlstm_classification.yaml
