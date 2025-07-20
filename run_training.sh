#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --time=11:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --job-name=xlstm_test
#SBATCH --output=logs/training_output_%j.log
#SBATCH --error=logs/training_error_%j.log

# Load conda environment
module load devel/miniforge
mamba activate xlstm
echo $CONDA_DEFAULT_ENV

# Run the training
echo "run with learningrate 1.6e-4"
$CONDA_PREFIX/bin/python train.py configs/go_xlstm_classification.yaml

