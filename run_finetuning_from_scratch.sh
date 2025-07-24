#!/bin/bash
#SBATCH --job-name=xlstm_scratch
#SBATCH --output=logs/xlstm_scratch_%j.out
##SBATCH --error=logs/xlstm_scratch_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --partition=gpu-single

# Activate Conda environment
module load devel/miniforge
mamba activate xlstm

echo "hoehere lr von 1e-4 mit weight balacing dingi"
# Run the training from scratch script
$CONDA_PREFIX/bin/python train.py configs/xlstm_uniprot_multilabel_from_scratch.yaml

