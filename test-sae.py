from transformer_lens import HookedTransformer, utils
import torch
from tasks.ioi.IOITask import IOITask
from tasks.facts.SportsTask import SportsTask
from tasks.owt.OWTTask import OWTTask
from dictionary_learning.dictionary import Dictionary, AutoEncoder

BATCH_SIZE = 64
ioi_task = IOITask(batch_size=BATCH_SIZE, tokenizer=utils.get_tokenizer(), device='cuda')
sports_task = SportsTask(batch_size=BATCH_SIZE, tokenizer=utils.get_tokenizer(), device='cuda')
owt_task = OWTTask(batch_size=BATCH_SIZE, tokenizer=utils.get_tokenizer(), device='cuda')

def get_metrics(model: HookedTransformer, sae: AutoEncoder, n_iter=5):
    ave_
    for 
