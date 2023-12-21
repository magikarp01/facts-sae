from nnsight import LanguageModel
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.training import trainSAE

model = LanguageModel(
    'EleutherAI/pythia-70m-deduped', # this can be any Huggingface model
    device_map = 'cuda:0'
)
submodule = model.gpt_neox.layers[1].mlp # layer 1 MLP
activation_dim = 512 # output dimension of the MLP
dictionary_size = 16 * activation_dim

# data much be an iterator that outputs strings
# data = iter([
#     'This is some example data',
#     'In real life, for training a dictionary',
#     'you would need much more data than this'
# ])

from datasets import load_dataset
import torch

# Load the dataset
# train_dataset = load_dataset('wikitext', 'wikitext-103-v1', split='train[:1000000]')
train_dataset = load_dataset('Skylion007/openwebtext', split='train')
def yield_sentences(data_split):
    for example in data_split:
        text = example['text']
        sentences = text.split('\n')
        for sentence in sentences:
            if sentence:  # skip empty lines
                yield sentence

# Creating an iterator for training sentences
train_sentences = yield_sentences(train_dataset)

for i in range(10):
    print(next(train_sentences))

# buffer = ActivationBuffer(
#     train_sentences,
#     model,
#     submodule,
#     out_feats=activation_dim, # output dimension of the model component
# ) # buffer will return batches of tensors of dimension = submodule's output dimension

# # # train the sparse autoencoder (SAE)
# ae = trainSAE(
#     buffer,
#     activation_dim,
#     dictionary_size,
#     lr=3e-4,
#     sparsity_penalty=1e-3,
#     device='cuda:0'
# )
