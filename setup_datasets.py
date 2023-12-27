from datasets import load_dataset

owt = load_dataset('Skylion007/openwebtext', split=f'train')
wiki = load_dataset('wikipedia', '20220301.en', split=f'train')
pile = load_dataset('monology/pile-uncopyrighted', split='train')
