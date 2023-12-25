#!/bin/bash
#SBATCH --output=jupyter_logs/log-%J.txt
#SBATCH --nodes=2
# #SBATCH --tasks-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --gres=gpu:2
#SBATCH --time=4:00:00


# source /data/phillip_guo/miniconda3/etc/profile.d/conda.sh
conda activate unlrn

jupyter-notebook --no-browser --ip=0.0.0.0 --NotebookApp.kernel_name=unlrn --NotebookApp.allow_origin_pat=https://.*vscode-cdn\.net --NotebookApp.allow_origin='*'