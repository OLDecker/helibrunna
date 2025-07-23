#!/bin/bash
#SBATCH --job-name=xlstm_scratch
#SBATCH --output=logs/xlstm_scratch_%j.out
#SBATCH --error=logs/xlstm_scratch_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate Conda environment
source /etc/profile.d/conda.sh
mamba activate xlstm

# Run the training from scratch script
$CONDA_PREFIX/bin/python train.py --config_file configs/xlstm_uniprot_multilabel_from_scratch.yaml
