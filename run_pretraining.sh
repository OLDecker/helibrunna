#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --job-name=xlstm_pretrain
#SBATCH --output=logs/pretrain_output_%j.log
#SBATCH --error=logs/pretrain_error_%j.log

# Load conda environment
module load devel/miniforge
mamba activate xlstm
echo $CONDA_DEFAULT_ENV

# Run pre-training
echo "Starting xLSTM pre-training on UniParc data"
$CONDA_PREFIX/bin/python train.py configs/xlstm_uniparc_pretrain.yaml
