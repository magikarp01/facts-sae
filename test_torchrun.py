import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.buffer import ActivationBuffer
from datasets import load_dataset
from dictionary_learning.training import ConstrainedAdam, sae_loss, resample_neurons

def ddp_setup():#rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"
    # torchrun will handle
    init_process_group(backend="nccl")#, rank=rank, world_size=world_size)
    # torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        # gpu_id: int,
        save_every: int,
        snapshot_path: str,

    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        # self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every

        self.epochs_run=0
        if os.path.exists(snapshot_path):
            print("loading snapshot")
            self._load_snapshot(snapshot_path)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
          
    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                # self._save_checkpoint(epoch)
                self._save_snapshot(epoch)

from transformer_lens import HookedTransformer, utils
from nnsight import LanguageModel
def load_train_objs(dataset_seed=0, buffer_size=100, BATCH_SIZE=32, lr=3e-4):
    # train_set = MyTrainDataset(2048)  # load your dataset
    device='cuda:0'
    size="70m"
    tl_model = HookedTransformer.from_pretrained(
        f'EleutherAI/pythia-{size}-deduped',
        # 'EleutherAI/pythia-1.4b-deduped',
        # 'EleutherAI/pythia-2.8b-deduped',
        device = device
    )
    tokenizer = tl_model.tokenizer

    layer = 1
    hook_pos = utils.get_act_name("mlp_out", layer)
    # model.set_use_hook_mlp_in(True)

    # model = LanguageModel(
    #     f'EleutherAI/pythia-{size}-deduped', # this can be any Huggingface model
    #     # 'EleutherAI/pythia-2.8b-deduped',
    #     device_map = device
    # )
    n_gpus = torch.cuda.device_count()
    assert n_gpus > 0
    nn_models = [
        LanguageModel(
            f'EleutherAI/pythia-{size}-deduped', # this can be any Huggingface model
            # 'EleutherAI/pythia-2.8b-deduped',
            device_map = f"cuda:{i}"
        )
        for i in range(1, n_gpus)
    ]
    nn_model = nn_models[0]
    submodule = nn_model.gpt_neox.layers[layer].mlp # layer 1 MLP
    submodule_fn = lambda model: model.gpt_neox.layers[layer].mlp
    activation_dim = tl_model.cfg.d_model # output dimension of the MLP
    dictionary_size = 16 * activation_dim
    
    ae = AutoEncoder(activation_dim, dictionary_size)

    train_dataset = load_dataset('monology/pile-uncopyrighted', split='train', streaming=True)
    dataset_shard = train_dataset.shuffle(buffer_size=buffer_size, seed=dataset_seed)

    def yield_sentences(data_split, cycle=True):
        while True:
            for example in data_split:
                text = example['text']
                yield text
            if not cycle:
                break

    # Creating an iterator for training sentences
    train_sentences = yield_sentences(dataset_shard, cycle=True)

    train_set = ActivationBuffer(
      train_sentences,
      nn_model,
      submodule,
      out_feats=activation_dim, # output dimension of the model component
      n_ctxs=1e4,
      in_batch_size=int(BATCH_SIZE*2*(n_gpus-1)), # batch size for the model
      out_batch_size=BATCH_SIZE*8*2, # batch size for the buffer
      # num_gpus=2, # number of GPUs to use
      models=nn_models,
      submodule_fn=submodule_fn,
    ) # buffer will return batches of tensors of dimension = submodule's output dimension
    optimizer = ConstrainedAdam(ae.parameters(), ae.decoder.parameters(), lr=lr)
    return train_set, ae, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)

# import torch
# from torch.utils.data import DataLoader, Dataset
# import torch.muliprocesing as mp
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group
# import os
# from dictionary_learning.training import trainSAE, ConstrainedAdam, sae_loss
# from tqdm import tqdm
# from datasets import load_dataset

# def ddp_setup(rank, world_size):
#   os.environ['MASTER_ADDR'] = 'localhost'
#   os.environ['MASTER_PORT'] = '12355'

#   init_process_group("nccl", rank=rank, world_size=world_size)


# class Trainer:
#   def __init__(
#       self,
#       activations, # ActivationBuffer that can be iterated over

#       ae, # SAE model      
#       gpu_id,
#       lr,
#       dictionary_size,
#       sparsity_penalty,
#       entropy=False,
#       steps=None, # if None, train until activations are exhausted
#       warmup_steps=1000, # linearly increase the learning rate for this many steps
#       # resample_steps=25000, # how often to resample dead neurons
#       # save_steps=None, # how often to save checkpoints
#       # save_dir=None, # directory for saving checkpoints
#       # log_steps=1000, # how often to print statistics
#       # test_steps=10000, # how often to test
#       refresh_steps=None, # how often to refresh the activations buffer, if None then will refresh by default
#   ):
#     self.gpu_id = gpu_id
#     self.ae = ae.to(gpu_id)
#     self.train_data = activations
#     self.ae = DDP(self.ae, device_ids=[gpu_id])

#     optimizer = ConstrainedAdam(ae.parameters(), ae.decoder.parameters(), lr=lr)
#     def warmup_fn(step):
#         if step < warmup_steps:
#             return step / warmup_steps
#         else:
#             return 1.
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)
    
#     for step, acts in enumerate(tqdm(activations, total=steps)):
#         # print(step)
#         if steps is not None and step >= steps:
#             break

#         if refresh_steps is not None and step % refresh_steps == 0:
#             activations.refresh()
#             print("refreshed activations by default")

#         if isinstance(acts, torch.Tensor): # typical casse
#             acts = acts.to(gpu_id)
#         elif isinstance(acts, tuple): # for cases where the autoencoder input and output are different
#             acts = tuple(a.to(gpu_id) for a in acts)

#         # print(f"{acts.shape=}")
#         optimizer.zero_grad()
#         loss = sae_loss(acts, ae, sparsity_penalty, entropy, separate=False)
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
# # def main():
# #   load_snapshot(snapshot_path)
# #   initialize()
# #   train()

# from dictionary_learning.buffer import ActivationBuffer
# def setup_buffer(model, submodule, 
#                  dataset='monology/pile-uncopyrighted', stream_dataset=True, split="train", seed=0, buffer_size=100,
#                   n_ctxs=1e4, in_batch_size=128, out_batch_size=128*16, ):
#     # seed could be rank, just draw from different parts of the dataset
#   """
#   Set up an ActivationBuffer for training.
#   """
#   buffer_dataset = load_dataset(dataset, split=split, streaming=stream_dataset)
#   dataset_shard = buffer_dataset.shuffle(buffer_size=buffer_size, seed=seed)

#   buffer = ActivationBuffer(
#     dataset_shard,
#     model,
#     submodule,
#     out_feats=activation_dim, # output dimension of the model component
#     n_ctxs=2e4,
#     in_batch_size=int(BATCH_SIZE*2*(n_gpus-1)), # batch size for the model
#     out_batch_size=BATCH_SIZE*8*2, # batch size for the buffer
#     # num_gpus=2, # number of GPUs to use
#     models=models,
#     submodule_fn=submodule_fn,
# ) # buffer will return batches of tensors of dimension = submodule's output dimension

     


# # def train():
# #   for batch in iter(dataset):
# #     train_step(batch)

# #     if should_checkpoint:
# #       save_snapshot(snapshot_path)