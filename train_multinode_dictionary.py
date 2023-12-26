import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from your_model_library import AutoEncoder, ActivationBuffer  # Replace with your actual imports
import os
import argparse

from nnsight import LanguageModel
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.training import trainSAE
from transformer_lens import HookedTransformer, utils
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import wandb
import psutil
from tqdm import tqdm
# from tqdm.notebook import tqdm as tqdm_notebook
import argparse
import os


dist.init_process_group(backend='nccl', init_method='env://')
rank = dist.get_rank()
world_size = dist.get_world_size()


# Create the parser and add arguments
parser = argparse.ArgumentParser(description='Run the model with specified layer')
parser.add_argument('--layer', type=int, default=1, help='Specify the layer number')
parser.add_argument('--size', type=str, default="70m", help='Specify the model size')
parser.add_argument('--batch_size', type=int, default=1024, help='Specify the batch size')
parser.add_argument('--steps', type=int, default=int(1e5), help='Specify the number of steps to train')
# Parse the arguments
args = parser.parse_args()

# Use the layer value from the command line argument
layer = args.layer
size = args.size
BATCH_SIZE = args.batch_size
steps = args.steps

#%%
print("starting to load model")
device='cuda:0'
tl_model = HookedTransformer.from_pretrained(
    f'EleutherAI/pythia-{size}-deduped',
    device = device
)
tokenizer = tl_model.tokenizer

hook_pos = utils.get_act_name("mlp_out", layer)
n_gpus = torch.cuda.device_count()
assert n_gpus > 0
models = [
    LanguageModel(
        f'EleutherAI/pythia-{size}-deduped', # this can be any Huggingface model
        device_map = f"cuda:{i}"
    )
    for i in range(1, n_gpus)
]
model = models[0]
submodule = model.gpt_neox.layers[layer].mlp # layer 1 MLP
submodule_fn = lambda model: model.gpt_neox.layers[layer].mlp
activation_dim = tl_model.cfg.d_model # output dimension of the MLP
dictionary_size = 16 * activation_dim

#%%

from tasks.ioi.IOITask import IOITask
from tasks.facts.SportsTask import SportsTask
from tasks.owt.OWTTask import OWTTask

ioi_task = IOITask(batch_size=BATCH_SIZE, tokenizer=tokenizer, device='cuda')
sports_task = SportsTask(batch_size=BATCH_SIZE, tokenizer=tokenizer, device='cuda')
owt_task = OWTTask(batch_size=BATCH_SIZE, tokenizer=tokenizer, device='cuda')

#%%

if rank == 0:
    wandb.init(project='facts-sae', 
            entity='philliphguo',
            config={
                    'model': f'EleutherAI/pythia-{size}-deduped',
                    'batch_size': BATCH_SIZE,
                    'layer': layer,
                    'steps': steps,
                    'world_size': world_size,
                    'rank': rank,
            }
        )


#%%
from datasets import load_dataset
import torch
from collections import defaultdict
from dataset_utils import yield_sentences, load_and_split_dataset

# Load the dataset
# train_dataset = load_dataset('wikitext', 'wikitext-103-v1', split='train[:1000000]')
train_dataset = load_dataset('Skylion007/openwebtext', split=f'train[:{int(7e6)}]')
train_dataset = load_and_split_dataset(train_dataset, rank, world_size)

# Creating an iterator for training sentences
train_sentences = yield_sentences(train_dataset, repeat=True)

# for i in range(10):
#     print(next(train_sentences))
#     print(len(tokenizer(next(train_sentences))['input_ids']))

buffer = ActivationBuffer(
    train_sentences,
    model,
    submodule,
    out_feats=activation_dim, # output dimension of the model component
    n_ctxs=1e4,
    in_batch_size=int(BATCH_SIZE*2*(n_gpus-1)), # batch size for the model
    out_batch_size=BATCH_SIZE*8*2, # batch size for the buffer
    # num_gpus=2, # number of GPUs to use
    models=models,
    submodule_fn=submodule_fn,
) # buffer will return batches of tensors of dimension = submodule's output dimension

#%%

from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.training import ConstrainedAdam, sae_loss, resample_neurons, trainSAE
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

autoencoder = AutoEncoder(activation_dim, dictionary_size).to(device)
autoencoder = DDP(autoencoder, device_ids=[rank])


# sae = AutoEncoder(activation_dim, dictionary_size).to(device)
print("starting training")
finished_sae, test_metrics = trainSAE(
    buffer,
    activation_dim,
    dictionary_size,
    lr=3e-4,
    sparsity_penalty=1e-3,
    steps=steps,
    warmup_steps=5000,
    resample_steps=30000,
    save_steps=int(1e4), 
    save_dir=f'trained_saes/{size}',
    log_steps=200,
    test_steps=200,
    device=device,
    tqdm_style=tqdm_notebook,
    use_wandb=True if rank == 0 else False,
)
wandb.finish()
#%%
