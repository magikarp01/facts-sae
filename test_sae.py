from transformer_lens import HookedTransformer, utils
import torch
from tasks.ioi.IOITask import IOITask
from tasks.facts.SportsTask import SportsTask
from tasks.owt.OWTTask import OWTTask
from dictionary_learning.dictionary import Dictionary, AutoEncoder
from collections import defaultdict

BATCH_SIZE = 64

def get_metrics(model: HookedTransformer, sae: AutoEncoder, n_iter=5, ioi_task=None, sports_task=None, owt_task=None):
    performances = defaultdict(list)
    for iter in range(n_iter):
        # ioi_task: get test loss and test accuracy
        performances['ioi_test_loss'] = ioi_task.get_test_loss(model)
        performances['ioi_test_accuracy'] = ioi_task.get_test_accuracy(model)
        # sports_task: get test loss and test accuracy
        performances['sports_test_loss'] = sports_task.get_test_loss(model)
        performances['sports_test_accuracy'] = sports_task.get_test_accuracy(model)
        # owt_task: get test loss
        performances['owt_test_loss'] = owt_task.get_test_loss(model)
    averages = {k: sum(v) / len(v) for k, v in performances.items()}
    return averages