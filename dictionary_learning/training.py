"""
Training dictionaries
"""

import torch as t
from .dictionary import AutoEncoder
from .buffer import ActivationBuffer
import os
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

class ConstrainedAdam(t.optim.Adam):
    """
    A variant of Adam where some of the parameters are constrained to have unit norm.
    """
    def __init__(self, params, constrained_params, lr):
        super().__init__(params, lr=lr)
        self.constrained_params = list(constrained_params)
    
    def step(self, closure=None):
        with t.no_grad():
            for p in self.constrained_params:
                normed_p = p / p.norm(dim=0, keepdim=True)
                # project away the parallel component of the gradient
                p.grad -= (p.grad * normed_p).sum(dim=0, keepdim=True) * normed_p
        super().step(closure=closure)
        with t.no_grad():
            for p in self.constrained_params:
                # renormalize the constrained parameters
                p /= p.norm(dim=0, keepdim=True)

def entropy(p, eps=1e-8):
    p_sum = p.sum(dim=-1, keepdim=True)
    # epsilons for numerical stability    
    p_normed = p / (p_sum + eps)    
    p_log = t.log(p_normed + eps)
    ent = -(p_normed * p_log)
    
    # Zero out the entropy where p_sum is zero
    ent = t.where(p_sum > 0, ent, t.zeros_like(ent))

    return ent.sum(dim=-1).mean()


def sae_loss(activations, ae, sparsity_penalty, use_entropy=False, separate=False):
    """
    Compute the loss of an autoencoder on some activations
    If separate is True, return the MSE loss and the sparsity loss separately
    """
    if isinstance(activations, tuple): # for cases when the input to the autoencoder is not the same as the output
        in_acts, out_acts = activations
    else: # typically the input to the autoencoder is the same as the output
        in_acts = out_acts = activations
    f = ae.encode(in_acts)
    x_hat = ae.decode(f)
    mse_loss = t.nn.MSELoss()(
        out_acts, x_hat
    )
    if use_entropy:
        sparsity_loss = entropy(f)
    else:
        sparsity_loss = f.norm(p=1, dim=-1).mean()
    if separate:
        return mse_loss, sparsity_loss
    else:
        return mse_loss + sparsity_penalty * sparsity_loss
    
def resample_neurons(deads, activations, ae, optimizer):
    """
    resample dead neurons according to the following scheme:
    Reinitialize the decoder vector for each dead neuron to be an activation
    vector v from the dataset with probability proportional to ae's loss on v.
    Reinitialize all dead encoder vectors to be the mean alive encoder vector x 0.2.
    Reset the bias vectors for dead neurons to 0.
    Reset the Adam parameters for the dead neurons to their default values.
    """
    with t.no_grad():
        if isinstance(activations, tuple):
            in_acts, out_acts = activations
        else:
            in_acts = out_acts = activations
        in_acts = in_acts.reshape(-1, in_acts.shape[-1])
        out_acts = out_acts.reshape(-1, out_acts.shape[-1])

        # compute the loss for each activation vector
        losses = (out_acts - ae(in_acts)).norm(dim=-1)

        # resample decoder vectors for dead neurons
        indices = t.multinomial(losses, num_samples=deads.sum(), replacement=True)

        # decoder is meta for some reason, so try this
        # check if decoder is meta device
        if ae.decoder.weight.device != losses.device:
            print("decoder is meta before resampling")
        print(f"{ae.decoder.weight.device=}, {losses.device=}, {indices.device=}, {deads=}")
        dec_weight = ae.decoder.weight.clone()
        dec_weight[:,deads] = out_acts[indices].T
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        print(f"{dec_weight.device=}, {dec_weight=}")
        ae.decoder.weight.data = dec_weight
        if ae.decoder.weight.device != losses.device:
            print(f"decoder is meta after resampling, {ae.decoder.weight.device=}, {ae.decoder.weight=}")

        # ae.decoder.weight[:,deads] = out_acts[indices].T
        # ae.decoder.weight /= ae.decoder.weight.norm(dim=0, keepdim=True)

        # resample encoder vectors for dead neurons
        ae.encoder.weight[deads] = ae.encoder.weight[~deads].mean(dim=0) * 0.2

        # reset bias vectors for dead neurons
        ae.encoder.bias[deads] = 0.

        # reset Adam parameters for dead neurons
        state_dict = optimizer.state_dict()['state']
        # # encoder weight
        state_dict[1]['exp_avg'][deads] = 0.
        state_dict[1]['exp_avg_sq'][deads] = 0.
        # # encoder bias
        state_dict[2]['exp_avg'][deads] = 0.
        state_dict[2]['exp_avg_sq'][deads] = 0.



from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.buffer import ActivationBuffer
import torch
from collections import defaultdict
import psutil
import wandb

def trainSAE(
        activations, # a generator that outputs batches of activations
        ae, # the autoencoder to train
        # activation_dim, # dimension of the activations
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
        refresh_steps=None, # how often to refresh the activations buffer, if None then will refresh by default
        device='cpu',
        tqdm_style=tqdm,
        use_wandb=False,
        evaluation_fns=None,
        ):
    """
    Train and return a sparse autoencoder

    Args:
    refresh_steps: how often to refresh the activations buffer, if None then will refresh by default. For automatic refreshing (helpful for multinode), should probably be at most activations.n_ctxs * activations.ctx_len * (1 - activations.min_buffer) // activations.out_batch_size
    evaluation_fns: a dictionary of evaluation metric names to the functions that take a model and return a value.
    """
    # ae = AutoEncoder(activation_dim, dictionary_size).to(device)
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

        if refresh_steps is not None and step % refresh_steps == 0:
            activations.refresh()
            print("refreshed activations by default")

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
            max_vram = torch.cuda.max_memory_allocated() / 1e9
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
                # test_metrics = get_metrics(tl_model, ae, hook_name=hook_pos, n_iter=1, ioi_task=ioi_task, sports_task=sports_task, owt_task=owt_task)
                test_metrics = {}
                for name, fn in evaluation_fns.items():
                    test_metrics[name] = fn(ae)
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

# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# def train_SAE_DDP(buffer, rank, world_size, refresh_every=None, *args, **kwargs):
#     autoencoder = AutoEncoder(...).to(rank)
#     autoencoder = DDP(autoencoder, device_ids=[rank])

