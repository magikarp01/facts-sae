#!/bin/bash
#SBATCH --job-name=70m-sae
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_outputs/output_sae-%J.txt
#SBATCH --error=slurm_outputs/error_sae-%J.txt

source /data/phillip_guo/miniconda3/etc/profile.d/conda.sh
conda activate unlrn
python train_dictionary.py