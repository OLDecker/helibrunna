#!/bin/bash
#SBATCH --job-name=xlstm_uniparc_pretrain
#SBATCH --output=xlstm_uniparc_pretrain_%j.out
#SBATCH --error=xlstm_uniparc_pretrain_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Navigate to working directory
cd /gpfs/bwfor/work/ws/hd_il278-helibrunna/helibrunna

# Activate conda environment
source ~/.bashrc
mamba activate xlstm
echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"

# Check environment
echo "Python path: $CONDA_PREFIX/bin/python"
echo "PyTorch version: $($CONDA_PREFIX/bin/python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $($CONDA_PREFIX/bin/python -c 'import torch; print(torch.cuda.is_available())')"

# Run training with timeout for testing
timeout 1500 $CONDA_PREFIX/bin/python train.py configs/xlstm_uniparc_pretrain.yaml

echo "Training completed or timed out"
