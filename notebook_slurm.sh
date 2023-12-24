#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=jupyter_logs/log-%J.txt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3
#SBATCH --time=4:00:00


# Start a tmux session in the background

# source ~/.bash_profile
# tmux new-session -d -s gpu_monitor 'nvidia-smi -l 1'

source /data/phillip_guo/miniconda3/etc/profile.d/conda.sh
conda activate unlrn

jupyter-notebook --no-browser --ip=0.0.0.0 --NotebookApp.kernel_name=iti_cap --NotebookApp.allow_origin_pat=https://.*vscode-cdn\.net --NotebookApp.allow_origin='*'