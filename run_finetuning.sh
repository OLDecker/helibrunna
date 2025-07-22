#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --job-name=xlstm_finetune
#SBATCH --output=logs/finetune_output_%j.log
#SBATCH --error=logs/finetune_error_%j.log

# Load conda environment
module load devel/miniforge
mamba activate xlstm
echo $CONDA_DEFAULT_ENV

# Run fine-tuning (requires pre-training to be completed first)
echo "Starting xLSTM fine-tuning on UniProt multi-label classification"
$CONDA_PREFIX/bin/python train.py configs/xlstm_uniprot_multilabel.yaml
