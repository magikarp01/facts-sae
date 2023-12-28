from datasets import load_dataset

owt = load_dataset('Skylion007/openwebtext', split=f'train', cache_dir='./hf_datasets')
wiki = load_dataset('wikipedia', '20220301.en', split=f'train', cache_dir='./hf_datasets')
pile = load_dataset('monology/pile-uncopyrighted', split='train', cache_dir='./hf_datasets')
# pile = load_dataset('hf_datasets/monology___pile-uncopyrighted', split='train')
