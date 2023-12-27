#!/bin/bash
#SBATCH --job-name=conda-install-1
#SBATCH --nodes=1
# #SBATCH --nodelist=tenant-ac-mats-h100-reserved-193-01
#SBATCH --tasks-per-node=1
# #SBATCH --gres=gpu:8
#SBATCH --output=slurm_outputs/output_%J.txt


# hostname
# source /opt/conda/etc/profile.d/conda.sh
export PATH=/mnt/rapid-shadow/home/Pguo/miniconda3/bin:$PATH
# conda info --envs
# conda create --name unlrn -y python=3.10

conda activate unlrn
# pip install -r /mnt/rapid-shadow/home/Pguo/facts-sae/requirements.txt
# conda install -c ipykernel
# python -m ipykernel install --user --name=unlrn
# conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
python setup_datasets.py