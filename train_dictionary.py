
from nnsight import LanguageModel
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.training import trainSAE
from transformer_lens import HookedTransformer, utils
import torch
import os
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from test_sae import get_metrics
import wandb

model = LanguageModel(
    'EleutherAI/pythia-70m-deduped', # this can be any Huggingface model
    device_map = 'cuda:0'
)
submodule = model.gpt_neox.layers[1].mlp # layer 1 MLP
activation_dim = 512 # output dimension of the MLP
dictionary_size = 16 * activation_dim
tl_model = HookedTransformer.from_pretrained(
    'EleutherAI/pythia-70m-deduped',
    # 'EleutherAI/pythia-1.4b-deduped',
    # 'EleutherAI/pythia-2.8b-deduped',
    device = 'cuda:0'
)

# model.set_use_hook_mlp_in(True)
tokenizer = tl_model.tokenizer

device='cuda:0'

from tasks.ioi.IOITask import IOITask
from tasks.facts.SportsTask import SportsTask
from tasks.owt.OWTTask import OWTTask
BATCH_SIZE=128

ioi_task = IOITask(batch_size=BATCH_SIZE, tokenizer=tokenizer, device='cuda')
sports_task = SportsTask(batch_size=BATCH_SIZE, tokenizer=tokenizer, device='cuda')
owt_task = OWTTask(batch_size=BATCH_SIZE, tokenizer=tokenizer, device='cuda')
wandb.init(project='facts-sae', 
           entity='philliphguo',
           config={
                'training_steps': 1e7
           }
           )

# data much be an iterator that outputs strings
# data = iter([
#     'This is some example data',
#     'In real life, for training a dictionary',
#     'you would need much more data than this'
# ])

from datasets import load_dataset
import torch
from collections import defaultdict

# Load the dataset
# train_dataset = load_dataset('wikitext', 'wikitext-103-v1', split='train[:1000000]')
train_dataset = load_dataset('Skylion007/openwebtext', split='train[:100]')
def yield_sentences(data_split):
    for example in data_split:
        text = example['text']
        sentences = text.split('\n')
        for sentence in sentences:
            if sentence:  # skip empty lines
                yield sentence

# Creating an iterator for training sentences
train_sentences = yield_sentences(train_dataset)

# for i in range(10):
#     print(next(train_sentences))

buffer = ActivationBuffer(
    train_sentences,
    model,
    submodule,
    out_feats=activation_dim, # output dimension of the model component
    n_ctxs=3e3,
    in_batch_size=128, # batch size for the model
    out_batch_size=128*16, # batch size for the buffer
) # buffer will return batches of tensors of dimension = submodule's output dimension


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
        print(step)
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
            total = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            print(f"step {step} memory: {a*1e-9} allocated, {r*1e-9} reserved, {total*1e-9} total")
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
                    wandb.log({'mem_allocated': a*1e-9, 'mem_reserved': r*1e-9, 'mem_total': total*1e-9}, step=step)
        # testing
        if test_steps is not None and step % test_steps == 0:
            with torch.no_grad():
                test_metrics = get_metrics(activations, ae, device=device, n_iter=1, ioi_task=ioi_task, sports_task=sports_task, owt_task=owt_task)
                for k, v in test_metrics.items():
                    metrics[k].append(test_metrics[k])
                if use_wandb:
                    wandb.log(test_metrics, step=step)

        # saving
        if save_steps is not None and save_dir is not None and step % save_steps == 0:
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
    steps=1e8,
    warmup_steps=1000,
    resample_steps=25000,
    save_steps=1e5,
    save_dir='trained_saes',
    log_steps=1000,
    test_steps=10000,
    device=device,
    tqdm_style=tqdm,
    use_wandb=True
)
wandb.finish()