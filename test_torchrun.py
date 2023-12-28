import torch
from torch.utils.data import DataLoader, Dataset
import torch.muliprocesing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from dictionary_learning.training import trainSAE


def ddp_setup(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'

  init_process_group("nccl", rank=rank, world_size=world_size)

class Trainer:
  def __init__(
      self,
      model, # SAE model
      activationBuffer, # ActivationBuffer that can be iterated over
      optimizer,
      

  )
# def main():
#   load_snapshot(snapshot_path)
#   initialize()
#   train()


# def setup_buffer():


# def train():
#   for batch in iter(dataset):
#     train_step(batch)

#     if should_checkpoint:
#       save_snapshot(snapshot_path)