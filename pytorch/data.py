import torch
import polars as pl
from torch.utils.data import Dataset, Sampler 
from torch.nn.utils.rnn import pad_sequence

def load_vocab(vocab_path):
    token2idx = {}
    idx2token = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for idx, token in enumerate(f):
            token = token.strip()
            token2idx[token] = idx
            idx2token.append(token)
    return token2idx, idx2token

def collate_fn(batch, max_length=512):
    src_batch = src_batch = [item[0][:max_length].clone().detach() for item in batch]
    tgt_batch = [item[1][:max_length].clone().detach() for item in batch]
    
    # Pad sequences to the maximum length in the batch
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)  # Pad with 0
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)  # Pad with 0

    return src_batch, tgt_batch

# Translation dataset
class TranslationDataset(Dataset):
    def __init__(self, path, pad_idx=0, start_idx=-1):
        self.data = pl.read_parquet(path)
        self.pad_idx = pad_idx
        self.start_idx = start_idx

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_data = [self.start_idx] + self.data["inputs"][idx].to_list()
        tgt_data = [self.start_idx] + self.data["targets"][idx].to_list()

        return torch.tensor(src_data, dtype=torch.long), torch.tensor(tgt_data, dtype=torch.long)

# Batching based on tokens    
class TokenSumBatchSampler(Sampler):

    def __init__(self, dataset, max_tokens):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.num_batches = self._calculate_num_batches()

    def _calculate_num_batches(self):
        num_batches = 0
        batch = []
        input_token_count = 0
        target_token_count = 0

        for idx in range(len(self.dataset)):
            src, tgt = self.dataset[idx]
            src_len = len(src)
            tgt_len = len(tgt)

            if (
                input_token_count + src_len > self.max_tokens
                or target_token_count + tgt_len > self.max_tokens
            ):
                if batch:  # Close current batch
                    num_batches += 1
                batch = []
                input_token_count = 0
                target_token_count = 0

            batch.append(idx)
            input_token_count += src_len
            target_token_count += tgt_len

        if batch:  # Account for the last batch
            num_batches += 1

        return num_batches

    def __iter__(self):
        batch = []
        input_token_count = 0
        target_token_count = 0

        for idx in range(len(self.dataset)):
            src, tgt = self.dataset[idx]
            src_len = len(src)
            tgt_len = len(tgt)

            # Check if adding the current sample violates the token sum condition
            if (
                input_token_count + src_len > self.max_tokens
                or target_token_count + tgt_len > self.max_tokens
            ):
                if batch:  # Yield current batch if it's not empty
                    yield batch
                batch = []
                input_token_count = 0
                target_token_count = 0

            batch.append(idx)
            input_token_count += src_len
            target_token_count += tgt_len

        if batch:  # Yield the last batch if it's not empty
            yield batch

    def __len__(self):
        return self.num_batches