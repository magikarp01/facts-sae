#!/bin/bash
#SBATCH --job-name=70m-sae
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --output=slurm_outputs/output_sae-%J.txt
#SBATCH --error=slurm_outputs/error_sae-%J.txt

source /opt/conda/etc/profile.d/conda.sh
conda activate unlrn
python train_dictionary.py --size=70m --layer=1 --batch_size=1024 --steps=100000
