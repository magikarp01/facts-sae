#!/bin/bash
#SBATCH --job-name=autoencoder-training
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --output=slurm_outputs/output-%j.txt
#SBATCH --error=slurm_outputs/error-%j.txt

source /opt/conda/etc/profile.d/conda.sh
conda activate unlrn

# Run the Python script with torchrun
torchrun --nnodes=10 --nproc_per_node=8 --rdzv_id=autoencoder_job --rdzv_backend=c10d --rdzv_endpoint=$(hostname):29500 train_dictionary.py
