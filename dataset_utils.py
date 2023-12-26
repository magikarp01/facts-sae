import torch
import os
from datasets import load_dataset
import torch
from collections import defaultdict

# Load the dataset
# train_dataset = load_dataset('wikitext', 'wikitext-103-v1', split='train[:1000000]')
def yield_sentences(data_split, repeat=False):
    while True:
        for example in data_split:
            text = example['text']
            sentences = text.split('\n')
            for sentence in sentences:
                if sentence:  # skip empty lines
                    yield sentence
        if not repeat:
            break

# def get_owt_iterator(size=int(7e6), repeat=False):
#     train_dataset = load_dataset('Skylion007/openwebtext', split=f'train[:{size}]')
#     # Creating an iterator for training sentences
#     return yield_sentences(train_dataset, repeat=repeat)

def load_and_split_dataset(full_dataset, rank, world_size):
    """
    For loading different parts of the dataset on different nodes
    """
    # Load the full dataset and split it based on rank
    part_length = len(full_dataset) // world_size
    start_index = rank * part_length
    end_index = start_index + part_length if rank != world_size - 1 else len(full_dataset)
    return full_dataset[start_index:end_index]

