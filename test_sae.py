from transformer_lens import HookedTransformer, utils
import torch
from tasks.ioi.IOITask import IOITask
from tasks.facts.SportsTask import SportsTask
from tasks.owt.OWTTask import OWTTask
from dictionary_learning.dictionary import Dictionary, AutoEncoder
from collections import defaultdict

BATCH_SIZE = 64

# sae = AutoEncoder(activation_dim, dictionary_size*4).to(device)
def apply_sae_hook(pattern, hook, sae, pre_sae_acts=None, post_sae_acts=None):
    """
    During inference time, run SAE on the output of the specified layer, and feed it back in.
    """
    if pre_sae_acts is not None:
        pre_sae_acts.append(pattern.clone().cpu())
    pattern = sae(pattern)
    if post_sae_acts is not None:
        post_sae_acts.append(pattern.clone().cpu())
    return pattern

def sae_inference_fn(tokens, model, hook_name, sae):
    return model.run_with_hooks(
        tokens,
        fwd_hooks = [
            (hook_name, lambda pattern, hook: apply_sae_hook(pattern, hook, sae,)) # pre_sae_acts, post_sae_acts))
        ]
    )

def get_metrics(model: HookedTransformer, sae: AutoEncoder, hook_name: str, n_iter=5, ioi_task=None, sports_task=None, owt_task=None):
    """
    Get task metrics for SAE by running inference with SAE on the specified layer. Takes a HookedTransformer (should be same as the one used to train SAE), a trained SAE, and the name of the layer/hook position to put sae in. Returns a dictionary of metrics.
    """
    performances = defaultdict(list)
    model_inference_fn = lambda tokens: sae_inference_fn(tokens, model, hook_name, sae)
    for iter in range(n_iter):
        # ioi_task: get test loss and test accuracy
        performances['ioi_test_loss'].append(ioi_task.get_test_loss(model_inference_fn).item())
        performances['ioi_test_accuracy'].append(ioi_task.get_test_accuracy(model_inference_fn))
        # sports_task: get test loss and test accuracy
        performances['sports_test_loss'].append(sports_task.get_test_loss(model_inference_fn).item())
        performances['sports_test_accuracy'].append(sports_task.get_test_accuracy(model_inference_fn))
        # owt_task: get test loss
        # performances['owt_test_loss'].append(owt_task.get_test_loss(model_inference_fn).item())
    averages = {k: sum(v) / len(v) for k, v in performances.items()}
    return averages