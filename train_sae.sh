#!/bin/bash
#SBATCH --job-name=2.8b-sae
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --output=slurm_outputs/output_sae-%J.txt
#SBATCH --error=slurm_outputs/error_sae-%J.txt
#SBATCH --time=8:00:00

source /data/phillip_guo/miniconda3/etc/profile.d/conda.sh
conda activate unlrn
python train_dictionary.py