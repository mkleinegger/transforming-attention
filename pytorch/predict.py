import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import polars as pl
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

# ==========================
# 1. Load Vocab
# ==========================
def load_vocab(vocab_path):
    token2idx = {}
    idx2token = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for idx, token in enumerate(f):
            token = token.strip()
            token2idx[token] = idx
            idx2token.append(token)
    return token2idx, idx2token

# ==========================
# 2. Translation Dataset
# ==========================
class TranslationDataset(Dataset):
    def __init__(self, path, pad_idx=0):
        self.data = pl.read_parquet(path)
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_data = self.data["inputs"][idx]
        tgt_data = self.data["targets"][idx]
        # Convert to tensors
        return (
            torch.tensor(src_data, dtype=torch.long), 
            torch.tensor(tgt_data, dtype=torch.long)
        )

# ==========================
# 3. Generate Subsequent Mask
#    (causal mask for decoding)
# ==========================
def generate_subsequent_mask(size: int) -> torch.Tensor:
    """
    Creates an upper-triangular mask of True values in the region 
    above the main diagonal. This mask prevents the decoder from 
    attending to subsequent tokens.
    """
    # shape: [1, size, size]
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).bool()
    return subsequent_mask

@torch.no_grad()
def generate(
    model: nn.Module,
    src_tokens: torch.Tensor,
    device: torch.device,
    start_idx: int,
    end_symbol: int = 1,
    pad_idx: int = 0,
    max_length: int = 100
):
    """
    Greedy decoding where we:
      1) Encode the source
      2) Iteratively decode one token at a time
    """
    model.eval()

    # Move src_tokens to device
    src_tokens = src_tokens.unsqueeze(0).to(device)  # full source, no trunc

    # -- 1) Encode --
    memory, src_mask = model.module.encode(src_tokens) if hasattr(model, 'module') \
        else model.encode(src_tokens)

    # We'll start our decoded output with a single [pad_idx] or empty
    # Since you have no start token, let's just begin with [pad_idx].
    ys = torch.tensor([[start_idx]], dtype=torch.long, device=device)  # [1, 1]

    for _ in range(max_length):
        # -- 2) Decode step --
        logits = (
            model.module.decode(ys, memory, src_mask)
            if hasattr(model, 'module')
            else model.decode(ys, memory, src_mask)
        )
        # logits shape = [batch=1, current_length, vocab_size]
        # take the last position
        next_token_logits = logits[:, -1, :]  # [1, vocab_size]

        # Greedy pick
        next_token = torch.argmax(next_token_logits, dim=-1)  # [1]

        # Append to sequence
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)  # shape [1, length+1]

        # Stop if we predicted <eos>
        if next_token.item() == end_symbol:
            break

    # ys shape = [1, dec_len]. We can remove the initial pad token if we want.
    # e.g. returning ys[:, 1:]
    return ys[:, 1:].squeeze(0)

import torch
import torch.nn.functional as F

@torch.no_grad()
def beam_search_decode(
    model: nn.Module,
    src_tokens: torch.Tensor,
    device: torch.device,
    pad_idx: int = 0,
    eos_idx: int = 1,
    beam_size: int = 4,
    max_length: int = 100,
    length_penalty_alpha: float = 0.0,
):
    """
    Single-sentence beam search decoding.
    
    Args:
        model (nn.Module): The Transformer with `encode()` and `decode()`.
        src_tokens (torch.Tensor): Shape [src_len], the source sentence token IDs.
        device (torch.device): e.g. torch.device('cuda', 0)
        pad_idx (int): Index of the PAD token.
        eos_idx (int): Index of the EOS token.
        beam_size (int): Number of beams to keep at each step.
        max_length (int): Maximum length of generated sequence (decoder steps).
        length_penalty_alpha (float): If > 0, use length normalization to score beams.

    Returns:
        torch.Tensor: The best decoding (shape [dec_len]) as token IDs.
    """
    model.eval()

    # 1) Encode the source
    src_tokens = src_tokens.unsqueeze(0).to(device)  # [1, src_len]
    memory, src_mask = model.module.encode(src_tokens) if hasattr(model, 'module') else model.encode(src_tokens)      # shapes: [1, src_len, d_model], [1, 1, 1, src_len]

    # 2) We store beams as a list of (score, sequence)
    #    We'll start with a single sequence: [pad_idx] (or <sos> if you have it)
    #    Score = 0.0 because log probability is 0 at the start.
    beams = [(0.0, torch.tensor([[pad_idx]], device=device, dtype=torch.long))]

    # We'll store completed beams (those that ended with <eos>) here
    completed_sequences = []

    for _ in range(max_length):
        new_beams = []
        # Expand each beam
        for score, seq in beams:
            # If this beam already ended with <eos>, just keep it
            if seq[0, -1].item() == eos_idx:
                completed_sequences.append((score, seq))
                # (Optionally) keep it in beams so it doesn't drop out, or skip expansions:
                new_beams.append((score, seq))
                continue

            # 3) Decode the partial sequence
            # shape => [1, current_length, vocab_size]
            logits =  (
                model.module.decode(seq, memory, src_mask)
                if hasattr(model, 'module')
                else model.decode(seq, memory, src_mask)
            )

            # 4) Take the last timestep's logits => shape [1, vocab_size]
            next_token_logits = logits[:, -1, :]
            log_probs = F.log_softmax(next_token_logits, dim=-1)  # shape [1, vocab_size]

            # 5) Get top K candidates
            top_log_probs, top_indices = torch.topk(log_probs, beam_size, dim=-1)

            # 6) Create new beams
            for i in range(beam_size):
                token_score = top_log_probs[0, i].item()   # log-prob of that token
                token_id = top_indices[0, i].item()

                new_score = score + token_score
                # If using length penalty, we apply it either now or at the end
                # e.g., normalized_score = new_score / ((seq.size(1)) ** length_penalty_alpha)

                new_seq = torch.cat(
                    [seq, torch.tensor([[token_id]], device=device)], 
                    dim=1
                )  # shape [1, seq_len+1]

                new_beams.append((new_score, new_seq))

        # 7) We now have up to (beam_size * len(beams)) new beams
        #    Sort them by score, descending
        #    (If using length penalty, sort by normalized score).
        new_beams = sorted(new_beams, key=lambda x: x[0], reverse=True)

        # 8) Keep the top `beam_size`
        beams = new_beams[:beam_size]

    # After we exit the loop, we may have some completed beams in `completed_sequences`.
    # If none completed, we fall back to whatever is in `beams`.
    all_candidates = completed_sequences + beams

    # Optionally re-sort them. If applying length normalization, do it here:
    #   length_norm_score = score / (seq_len**length_penalty_alpha)
    # For simplicity, we’ll do no length penalty or you can do:
    def final_score_func(score, seq):
        length = seq.size(1)
        if length_penalty_alpha > 0:
            return score / (length**length_penalty_alpha)
        return score

    # Re-rank final beams
    all_candidates = sorted(all_candidates, 
                            key=lambda x: final_score_func(x[0], x[1]),
                            reverse=True)

    best_score, best_seq = all_candidates[0]
    
    # best_seq => shape [1, dec_len]
    # Optionally remove the initial [pad_idx] if used as "start"
    # Return shape => [dec_len_without_pad]
    return best_seq.squeeze(0)[1:]



# ==========================
# 5. Putting it all together
# ==========================
def main():
    import sys
    
    # --------------------------
    # Load the dataset & vocab
    # --------------------------
    df_path = "../data/dataset.parquet"
    vocab_path = "../data/vocab.ende"

    translation_dataset = TranslationDataset(df_path, pad_idx=0)
    token2idx, idx2token = load_vocab(vocab_path)
    
    # Some assumptions about the special tokens:
    # We'll assume:
    #   pad_idx = 0
    #   eos_idx = 1  (since your dataset has sequences ending in 1)
    # If you have no start symbol in your data, we won't use it.

    pad_idx = 0
    eos_idx = 1

    # --------------------------
    # Build/Load Model
    # (Example hyperparams)
    # --------------------------
    src_vocab_size = len(token2idx)-1
    tgt_vocab_size = len(token2idx)-1
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 512
    dropout = 0.1
    label_smoothing = 0.1

    from transformer import Transformer
    # Suppose you have a Transformer definition somewhere:
    # from your_transformer_implementation import Transformer
    model = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    )

    # Initialize the distributed environment
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    
    import torch.distributed as dist
    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    model.to(device)

    # Wrap with DDP
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Load a checkpoint
    checkpoint_path = "./checkpoints/old/checkpoint_step_100000.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(f"Loaded checkpoint from step {checkpoint['step']}")

    # --------------------------
    # Greedy Generate from a Sample
    # --------------------------
    # For demonstration, let's pick the first sample in the dataset:
    src_sample, tgt_sample = translation_dataset[3]
    print("Source tokens:", src_sample)
    print("Target tokens:", tgt_sample)

    generated = generate(
        model=model,
        src_tokens=src_sample,
        device=device,
        end_symbol=eos_idx,
        pad_idx=pad_idx,
        start_idx=len(token2idx)-1,
        max_length=512,
    )

    #generated = beam_search_decode(
    #    model,
    #    src_tokens=src_sample,
    #    device=device,
    #    pad_idx=5,      # or your <sos> if you have one
    #    eos_idx=1,      # the end-of-sequence token in your vocab
    #    beam_size=5,    # typical beam size
    #    max_length=512,
    #    length_penalty_alpha=0.6,  # typical in "Attention Is All You Need"
    #)


    print("Generated token IDs:", generated.tolist())

    print(f"Sample source  : {[idx2token[t] for t in src_sample]}")
    print(f"Sample target : {[idx2token[t] for t in tgt_sample]}")
    print(f"Sample output : {[idx2token[t] for t in generated]}")
    print("------")

    # Possibly remove trailing end symbol (1) if desired
    # if eos_idx in generated_text:
        # If the end symbol is actually an integer '1' in idx2token, 
        # you'd do a safe removal of anything after the first occurrence
        # or just strip trailing token.
        # pass

    # Done
    return 0


if __name__ == "__main__":
    sys.exit(main())
