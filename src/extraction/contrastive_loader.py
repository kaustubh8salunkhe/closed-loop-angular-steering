import torch
from datasets import load_dataset

class ContrastiveLoader:
    def __init__(self, dataset_path="json"):
        # Expects a dataset with 'benign_prompt' and 'malicious_prompt' pairs
        self.dataset = load_dataset(dataset_path, data_files={"train": "data/contrastive_pairs/pairs.json"})["train"]
    
    def get_batches(self, batch_size=16):
        for i in range(0, len(self.dataset), batch_size):
            batch = self.dataset[i:i+batch_size]
            yield batch['benign_prompt'], batch['malicious_prompt']
