#!/bin/bash
#SBATCH --job-name=2.8b-sae
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --output=slurm_outputs/output_sae-%J.txt
#SBATCH --error=slurm_outputs/error_sae-%J.txt

# source /opt/conda/etc/profile.d/conda.sh
export PATH=/mnt/rapid-shadow/home/Pguo/miniconda3/bin:$PATH
conda config --add envs_dirs /mnt/rapid-shadow/home/Pguo/miniconda3/envs
export WANDB_API_KEY=e3f0e114af46b3c098870f12c408690133379f22
conda init
conda activate unlrn
python train_dictionary.py --size=2.8b --layer=31 --batch_size=128 --steps=500000 --stream_dataset=True --resample_steps=50000