#!/bin/bash
#SBATCH --job-name=xlstm_uniprot_bio_xlstm
#SBATCH --output=logs/xlstm_uniprot_bio_xlstm%j.out
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:A40:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --time=48:00:00
#SBATCH --mem=32gb
#SBATCH --export=NONE

# Activate Conda environment
module load devel/miniforge
mamba activate xlstm

# Run the training from scratch script
$CONDA_PREFIX/bin/python train.py configs/xlstm_uniprot_bio_xlstm.yaml

