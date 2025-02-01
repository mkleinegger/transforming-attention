import os
import sys
import torch
import torch.nn.functional as F
import polars as pl
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformer import Transformer
from data import TranslationDataset, load_vocab

def generate_greedy(
    model,
    src_tokens,
    device,
    start_idx,
    end_symbol,
    pad_idx = 0,
    max_length = 100
):
    model.eval()

    src_tokens = src_tokens.unsqueeze(0).to(device)  # full source, no trunc
    memory, src_mask = model.module.encode(src_tokens) if hasattr(model, 'module') \
        else model.encode(src_tokens)

    ys = torch.tensor([[start_idx]], dtype=torch.long, device=device)  # [1, 1]

    for _ in range(max_length):
        logits = (
            model.module.decode(ys, memory, src_mask)
            if hasattr(model, 'module')
            else model.decode(ys, memory, src_mask)
        )

        next_token_logits = logits[:, -1, :]  # [1, vocab_size]
        next_token = torch.argmax(next_token_logits, dim=-1) # Greedy pick

        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)  # shape [1, length+1]
        if next_token.item() == end_symbol:
            break

    return ys[:, 1:].squeeze(0)

@torch.no_grad()
def generate_beam_search(
    model,
    src_tokens,
    device,
    start_idx,
    eos_idx,
    pad_idx = 0,
    beam_size = 4,
    max_length = 100,
    length_penalty_alpha = 0.0,
):
    model.eval()

    src_tokens = src_tokens.unsqueeze(0).to(device)  # [1, src_len]
    memory, src_mask = model.module.encode(src_tokens) if hasattr(model, 'module') else model.encode(src_tokens)      # shapes: [1, src_len, d_model], [1, 1, 1, src_len]

    beams = [(0.0, torch.tensor([[start_idx]], device=device, dtype=torch.long))]
    completed_sequences = []

    for _ in range(max_length):
        new_beams = []

        for score, seq in beams:
            if seq[0, -1].item() == eos_idx:
                completed_sequences.append((score, seq))
                new_beams.append((score, seq))
                continue

            logits =  (
                model.module.decode(seq, memory, src_mask)
                if hasattr(model, 'module')
                else model.decode(seq, memory, src_mask)
            )

            next_token_logits = logits[:, -1, :]
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            top_log_probs, top_indices = torch.topk(log_probs, beam_size, dim=-1)

            for i in range(beam_size):
                token_score = top_log_probs[0, i].item() 
                token_id = top_indices[0, i].item()

                new_score = score + token_score
                new_seq = torch.cat(
                    [seq, torch.tensor([[token_id]], device=device)], 
                    dim=1
                )  # shape [1, seq_len+1]

                new_beams.append((new_score, new_seq))

        new_beams = sorted(new_beams, key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_size]

    all_candidates = completed_sequences + beams

    def final_score_func(score, seq):
        length = seq.size(1)
        if length_penalty_alpha > 0:
            return score / (length**length_penalty_alpha)
        return score

    all_candidates = sorted(all_candidates, 
                            key=lambda x: final_score_func(x[0], x[1]),
                            reverse=True)
    best_score, best_seq = all_candidates[0]
    return best_seq.squeeze(0)[1:]


def main():
    df_path = "../data/dataset.parquet"
    vocab_path = "../data/vocab.ende"
    output_file = "../data/beaminference.parquet"

    token2idx, idx2token = load_vocab(vocab_path)
    translation_dataset = TranslationDataset(df_path, pad_idx=0, start_idx=len(token2idx))
    
    pad_idx = 0
    eos_idx = 1

    src_vocab_size = len(token2idx) + 1
    tgt_vocab_size = len(token2idx) + 1
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 512
    dropout = 0.1

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

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    
    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    model.to(device)

    model = DDP(model, device_ids=[rank], output_device=rank)

    checkpoint_path = "./checkpoints/checkpoint_step_90000.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(f"Loaded checkpoint from step {checkpoint['step']}")

    data = []
    i = 0
    for src_sample, tgt_sample in tqdm(translation_dataset, desc="Running inference"):
        src_sample, tgt_sample = translation_dataset[i]

        # generated = generate_greedy(
        #     model=model,
        #     src_tokens=src_sample,
        #     device=device,
        #     end_symbol=eos_idx,
        #     pad_idx=pad_idx,
        #     start_idx=len(idx2token),
        #     max_length=512,
        #)

        generated = generate_beam_search(
           model,
           src_tokens=src_sample,
           device=device,
           pad_idx=pad_idx,
           eos_idx=eos_idx,
           beam_size=5,
           max_length=512,
           start_idx=len(idx2token),
           length_penalty_alpha=0.6,
        )
        
        data.append({
            "target": tgt_sample.tolist(),
            "generated": generated.tolist()
        })
        i += 1

    df = pl.DataFrame(data)
    df.write_parquet(output_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
