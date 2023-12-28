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
from test_sae import get_metrics
import psutil
from tqdm import tqdm
# from tqdm.notebook import tqdm as tqdm_notebook
import argparse
import os


# Create the parser and add arguments
parser = argparse.ArgumentParser(description='Run the model with specified layer')
parser.add_argument('--layer', type=int, default=1, help='Specify the layer number')
parser.add_argument('--size', type=str, default="70m", help='Specify the model size')
parser.add_argument('--batch_size', type=int, default=1024, help='Specify the batch size')
parser.add_argument('--steps', type=int, default=int(1e5), help='Specify the number of steps to train')
parser.add_argument('--stream_dataset', type=bool, default=False, help='Specify whether to stream the dataset')
# Parse the arguments
args = parser.parse_args()

# Use the layer value from the command line argument
layer = args.layer
size = args.size
BATCH_SIZE = args.batch_size
steps = args.steps
stream_dataset = args.stream_dataset

#%%
print("starting to load model")
# size = "70m"
# BATCH_SIZE = 1024 # for 70m
# BATCH_SIZE = 160 # for 2.8b

# size = "2.8b"
device='cuda:0'
tl_model = HookedTransformer.from_pretrained(
    f'EleutherAI/pythia-{size}-deduped',
    # 'EleutherAI/pythia-1.4b-deduped',
    # 'EleutherAI/pythia-2.8b-deduped',
    device = device
)
tokenizer = tl_model.tokenizer

# layer = 1
hook_pos = utils.get_act_name("mlp_out", layer)
# model.set_use_hook_mlp_in(True)

# model = LanguageModel(
#     f'EleutherAI/pythia-{size}-deduped', # this can be any Huggingface model
#     # 'EleutherAI/pythia-2.8b-deduped',
#     device_map = device
# )
n_gpus = torch.cuda.device_count()
assert n_gpus > 0
models = [
    LanguageModel(
        f'EleutherAI/pythia-{size}-deduped', # this can be any Huggingface model
        # 'EleutherAI/pythia-2.8b-deduped',
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
wandb.init(project='facts-sae', 
           entity='philliphguo',
           config={
                'model': f'EleutherAI/pythia-{size}-deduped',
                'batch_size': BATCH_SIZE,
                'layer': layer,
                'dataset': 'pile-uncopyrighted',
           }
           )


#%%
from datasets import load_dataset
import torch
from collections import defaultdict

# Load the dataset
train_dataset = load_dataset('monology/pile-uncopyrighted', split='train', streaming=stream_dataset)

def yield_sentences(data_split, cycle=False):
    while True:
        for example in data_split:
            text = example['text']
            # sentences = text.split('\n')
            # for sentence in sentences:
            #     if sentence:  # skip empty lines
            #         yield sentence
            yield text
        if not cycle:
            break

# Creating an iterator for training sentences
train_sentences = yield_sentences(train_dataset, cycle=True)

for i in range(10):
    print(next(train_sentences))

buffer = ActivationBuffer(
    train_sentences,
    model,
    submodule,
    out_feats=activation_dim, # output dimension of the model component
    n_ctxs=2e4,
    in_batch_size=int(BATCH_SIZE*2*(n_gpus-1)), # batch size for the model
    out_batch_size=BATCH_SIZE*8*2, # batch size for the buffer
    # num_gpus=2, # number of GPUs to use
    models=models,
    submodule_fn=submodule_fn,
) # buffer will return batches of tensors of dimension = submodule's output dimension

#%%

from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.training import ConstrainedAdam, sae_loss, resample_neurons

def trainSAE(
        activations, # a generator that outputs batches of activations
        activation_dim, # dimension of the activations
        dictionary_size, # size of the dictionary
        lr,
        sparsity_penalty,
        entropy=False,
        steps=None, # if None, train until activations are exhausted
        warmup_steps=1000, # linearly increase the learning rate for this many steps
        resample_steps=25000, # how often to resample dead neurons
        save_steps=None, # how often to save checkpoints
        save_dir=None, # directory for saving checkpoints
        log_steps=1000, # how often to print statistics
        test_steps=10000, # how often to test
        device='cpu',
        tqdm_style=tqdm,
        use_wandb=False
        ):
    """
    Train and return a sparse autoencoder
    """
    ae = AutoEncoder(activation_dim, dictionary_size).to(device)
    alives = torch.zeros(dictionary_size).bool().to(device) # which neurons are not dead?

    # set up optimizer and scheduler
    optimizer = ConstrainedAdam(ae.parameters(), ae.decoder.parameters(), lr=lr)
    def warmup_fn(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 1.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)
    metrics = defaultdict(list)
    print("setup sae, optimizer, scheduler")

    for step, acts in enumerate(tqdm_style(activations, total=steps)):
        # print(step)
        if steps is not None and step >= steps:
            break

        if isinstance(acts, torch.Tensor): # typical casse
            acts = acts.to(device)
        elif isinstance(acts, tuple): # for cases where the autoencoder input and output are different
            acts = tuple(a.to(device) for a in acts)

        # print(f"{acts.shape=}")
        optimizer.zero_grad()
        loss = sae_loss(acts, ae, sparsity_penalty, entropy, separate=False)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # deal with resampling neurons
        if resample_steps is not None:
            with torch.no_grad():
                if isinstance(acts, tuple):
                    in_acts = acts[0]
                else:
                    in_acts = acts
                dict_acts = ae.encode(in_acts)
                alives = torch.logical_or(alives, (dict_acts != 0).any(dim=0))
                if step % resample_steps == resample_steps // 2:
                    alives = torch.zeros(dictionary_size).bool().to(device)
                if step % resample_steps == resample_steps - 1:
                    deads = ~alives
                    if deads.sum() > 0:
                        print(f"resampling {deads.sum().item()} dead neurons")
                        resample_neurons(deads, acts, ae, optimizer)

        # logging
        if log_steps is not None and step % log_steps == 0:
            # total = torch.cuda.get_device_properties(0).total_memory
            # r = torch.cuda.memory_reserved(0)
            # a = torch.cuda.memory_allocated(0)
            # print(f"step {step} memory: {a*1e-9} allocated, {r*1e-9} reserved, {total*1e-9} total")
            max_vram = torch.cuda.max_memory_allocated() / 1e9
            cur_vram = torch.cuda.memory_allocated() / 1e9
            vram_usages = {f"cuda:{i}": torch.cuda.max_memory_allocated(f"cuda:{i}") // 1e9 for i in range(torch.cuda.device_count())}

            process = psutil.Process()
            cur_mem = process.memory_info().rss / 1e9
            print(f"step {step} max vram: {max_vram}, cur mem: {cur_mem}, vram usages: {vram_usages}")
            with torch.no_grad():
                mse_loss, sparsity_loss = sae_loss(acts, ae, sparsity_penalty, entropy, separate=True)
                print(f"step {step} MSE loss: {mse_loss}, sparsity loss: {sparsity_loss}")
                # dict_acts = ae.encode(acts)
                # print(f"step {step} % inactive: {(dict_acts == 0).all(dim=0).sum() / dict_acts.shape[-1]}")
                # if isinstance(activations, ActivationBuffer):
                #     tokens = activations.tokenized_batch().input_ids
                #     loss_orig, loss_reconst, loss_zero = reconstruction_loss(tokens, activations.model, activations.submodule, ae)
                #     print(f"step {step} reconstruction loss: {loss_orig}, {loss_reconst}, {loss_zero}")
                if use_wandb:
                    wandb.log({'mse_loss': mse_loss, 'sparsity_loss': sparsity_loss}, step=step)
                    # wandb.log({'mem_allocated': a*1e-9, 'mem_reserved': r*1e-9, 'mem_total': total*1e-9}, step=step)
                    wandb.log({'max_vram': max_vram, 'cur_mem': cur_mem}, step=step)
                    wandb.log(vram_usages, step=step)

        # testing
        if test_steps is not None and step % test_steps == 0:
            with torch.no_grad():
                test_metrics = get_metrics(tl_model, ae, hook_name=hook_pos, n_iter=1, ioi_task=ioi_task, sports_task=sports_task, owt_task=owt_task)
                print(test_metrics)
                for k, v in test_metrics.items():
                    metrics[k].append(test_metrics[k])
                if use_wandb:
                    wandb.log(test_metrics, step=step)

        # saving
        if save_steps is not None and save_dir is not None and step % save_steps == 0:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            if not os.path.exists(os.path.join(save_dir, "checkpoints")):
                os.mkdir(os.path.join(save_dir, "checkpoints"))
            torch.save(
                ae.state_dict(), 
                os.path.join(save_dir, "checkpoints", f"ae_{step}.pt")
                )

    return ae, metrics


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
    use_wandb=True
)
wandb.finish()
#%%
