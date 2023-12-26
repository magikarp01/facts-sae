import torch as t
import zstandard as zstd
import json
import io
from nnsight import LanguageModel
import os, psutil
import concurrent.futures
import time

"""
Implements a buffer of activations
"""

class EmptyStream(Exception):
    """
    An exception for when the data stream has been exhausted
    """
    def __init__(self):
        super().__init__()

class ActivationBuffer:
    def __init__(self, 
                 data, # generator which yields text data
                 model, # LanguageModel from which to extract activations
                 submodule, # submodule of the model from which to extract activations
                 in_feats=None,
                 out_feats=None, 
                 io='out', # can be 'in', 'out', or 'in_to_out'
                 n_ctxs=3e4, # approximate number of contexts to store in the buffer
                 ctx_len=128, # length of each context
                 in_batch_size=512, # size of batches in which to process the data when adding to buffer
                 out_batch_size=8192, # size of batches in which to return activations
                 models=None, # list of models to use in parallel to store activations
                 submodule_fn=None, # function to get the submodule from the model
                 default_device='cuda:0',
                 min_buffer=0.5, # minimum fraction of buffer to fill before refreshing
                 ):
        
        if io == 'in':
            if in_feats is None:
                try:
                    in_feats = submodule.in_features
                except:
                    raise ValueError("in_feats cannot be inferred and must be specified directly")
            self.activations = t.empty(0, in_feats).to(default_device)

        elif io == 'out':
            if out_feats is None:
                try:
                    out_feats = submodule.out_features
                except:
                    raise ValueError("out_feats cannot be inferred and must be specified directly")
            self.activations = t.empty(0, out_feats).to(default_device)
        elif io == 'in_to_out':
            if in_feats is None:
                try:
                    in_feats = submodule.in_features
                except:
                    raise ValueError("in_feats cannot be inferred and must be specified directly")
            if out_feats is None:
                try:
                    out_feats = submodule.out_features
                except:
                    raise ValueError("out_feats cannot be inferred and must be specified directly")
            self.activations_in = t.empty(0, in_feats)
            self.activations_out = t.empty(0, out_feats)
        self.read = t.zeros(0).bool()

        self.data = data
        self.model = model # assumes nnsight model is already on the device
        self.submodule = submodule
        self.io = io
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.in_batch_size = in_batch_size
        self.out_batch_size = out_batch_size

        if models is None:
            self.models = [model]
        else:
            self.models = models

        if submodule_fn is None:
            self.submodules = [submodule]
        else:
            self.submodules = [submodule_fn(model) for model in models]
            
        self.default_device = default_device
        self.min_buffer = min_buffer
        self.cycle = cycle
        if cycle:
            # clone data generator
            self.orig_data = list(data)

        # assert num_gpus <= t.cuda.device_count()
        # assert num_gpus == len(models)
        # self.num_gpus = num_gpus
        # self.models = models
        # self.models = [t.nn.Module() for _ in range(num_gpus)]
        # for i, model_copy in enumerate(self.models):
        #     model_copy.load_state_dict(model.state_dict())  # Copy the state of the original model
        #     model_copy.to(f'cuda:{i}')  # Move model to specific GPU
        #     model_copy.submodule = submodule  # Assign submodule

        # Ensure that CUDA is aware of the model's existence on the device(s)
        t.cuda.synchronize()
    
    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """
        with t.no_grad():
            # if buffer is less than half full, refresh
            if (~self.read).sum() < self.n_ctxs * self.ctx_len * self.min_buffer:
                try:
                    self.refresh()
                except EmptyStream: # if the data stream is exhausted, stop
                    raise StopIteration

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[t.randperm(len(unreads))[:self.out_batch_size]]
            self.read[idxs] = True
            if self.io in ['in', 'out']:
                return self.activations[idxs]
            else:
                return (self.activations_in[idxs], self.activations_out[idxs])
    
    def text_batch(self, batch_size=None):
        """
        Return a list of text
        """
        if batch_size is None:
            batch_size = self.in_batch_size
        return [
            next(self.data) for _ in range(batch_size)
        ]
    
    
    def tokenized_batch(self, batch_size=None):
        """
        Return a batch of tokenized inputs.
        """
        texts = self.text_batch(batch_size=batch_size)
        return self.model.tokenizer(
            texts,
            return_tensors='pt',
            max_length=self.ctx_len,
            padding=True,
            truncation=True
        )

    # def _refresh_std(self):
    #     """
    #     For when io == 'in' or 'out'
    #     """
    #     self.activations = self.activations[~self.read]

    #     while len(self.activations) < self.n_ctxs * self.ctx_len:
    #             # print(f"buffer size: {len(self.activations)}, need {self.n_ctxs * self.ctx_len}")
    #             # process = psutil.Process()
    #             # print(process.memory_info().rss // 1e9)  # in bytes 
                
    #             tokens = self.tokenized_batch()['input_ids']
    
    #             with self.model.generate(max_new_tokens=1, pad_token_id=self.model.tokenizer.pad_token_id) as generator:
    #                 with generator.invoke(tokens) as invoker:
    #                     if self.io == 'in':
    #                         hidden_states = self.submodule.input.save()
    #                     else:
    #                         hidden_states = self.submodule.output.save()
    #             hidden_states = hidden_states.value
    #             if isinstance(hidden_states, tuple):
    #                 hidden_states = hidden_states[0]
    #             hidden_states = hidden_states[tokens != self.model.tokenizer.pad_token_id]
    
    #             self.activations = t.cat([self.activations, hidden_states.to('cpu')], dim=0)
    #             self.read = t.zeros(len(self.activations)).bool()
    
    def _refresh_std(self):
        """
        For when io == 'in' or 'out'
        """
        self.activations = self.activations[~self.read]

        while len(self.activations) < self.n_ctxs * self.ctx_len:
            print(f"buffer size: {len(self.activations)}, need {self.n_ctxs * self.ctx_len}, buffer_shape: {self.activations.shape}")
            vram_usages = [t.cuda.max_memory_allocated(f"cuda:{i}") // 1e9 for i in range(t.cuda.device_count())]
            print(f"max vram usages: {vram_usages}")
            tokens = self.tokenized_batch()['input_ids']

            # Split tokens for different models
            split_tokens = t.chunk(tokens, len(self.models), dim=0)
            # print(f"{split_tokens[0].shape=}, {split_tokens[1].shape=}, {len(split_tokens)=}, {tokens.shape=}") 

            # Process each chunk on a different model in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # futures = [executor.submit(self.process_on_gpu, model, tokens_chunk, self.io) 
                #            for model, tokens_chunk in zip(self.models, split_tokens)]
                futures = [executor.submit(self.process_on_gpu, self.models[i], self.submodules[i], split_tokens[i], self.io, "cuda:0") for i in range(len(self.models))]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
            # print("starting moving to device")
            # starttime = time.time()
            # results = [result.to(self.default_device) for result in results]
            # print("finished moving to device, time elapsed: ", time.time() - starttime, " seconds")
            # Combine results
            hidden_states = t.cat(results, dim=0)
            # self.activations = t.cat([self.activations, hidden_states.to('cpu')], dim=0)
            self.activations = t.cat([self.activations, hidden_states], dim=0)
            self.read = t.zeros(len(self.activations)).bool()

    def process_on_gpu(self, model, submodule, tokens, io, move_to_device=None):
        with t.no_grad():
            with model.generate(max_new_tokens=1, pad_token_id=model.tokenizer.pad_token_id) as generator:
                with generator.invoke(tokens) as invoker:
                    if io == 'in':
                        hidden_states = submodule.input.save()
                    else:
                        hidden_states = submodule.output.save()
            hidden_states = hidden_states.value
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            if move_to_device is None:
                return hidden_states[tokens != model.tokenizer.pad_token_id]
            return hidden_states[tokens != model.tokenizer.pad_token_id].to(move_to_device)
    
    def _refresh_in_to_out(self):
        """
        For when io == 'in_to_out'
        """
        self.activations_in = self.activations_in[~self.read]
        self.activations_out = self.activations_out[~self.read]

        while len(self.activations_in) < self.n_ctxs * self.ctx_len:
                    
            tokens = self.tokenized_batch()['input_ids']

            with self.model.generate(max_new_tokens=1, pad_token_id=self.model.tokenizer.pad_token_id) as generator:
                with generator.invoke(tokens) as invoker:
                    hidden_states_in = self.submodule.input.save()
                    hidden_states_out = self.submodule.output.save()
            for i, hidden_states in enumerate([hidden_states_in, hidden_states_out]):
                hidden_states = hidden_states.value
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
                hidden_states = hidden_states[tokens != self.model.tokenizer.pad_token_id]
                if i == 0:
                    self.activations_in = t.cat([self.activations_in, hidden_states.to('cpu')], dim=0)
                else:
                    self.activations_out = t.cat([self.activations_out, hidden_states.to('cpu')], dim=0)
            self.read = t.zeros(len(self.activations_in)).bool()

    def refresh(self):
        """
        Refresh the buffer
        """
        print("refreshing buffer...")

        if self.io == 'in' or self.io == 'out':
            self._refresh_std()
        else:
            self._refresh_in_to_out()

        print('buffer refreshed...')

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()
