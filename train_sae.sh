#!/bin/bash
#SBATCH --job-name=2.8b-sae
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --output=slurm_outputs/output_sae-%J.txt
#SBATCH --error=slurm_outputs/error_sae-%J.txt

source /opt/conda/etc/profile.d/conda.sh
conda activate unlrn
python train_dictionary.py --size=2.8b --layer=1 --batch_size=160 --steps=400000
