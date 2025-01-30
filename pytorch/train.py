import os
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from tqdm.auto import tqdm
from torch import optim

# Replace with your own modules
from data import (
    TokenSumBatchSampler,
    TranslationDataset,
    collate_fn,
    load_vocab,
)
from transformer import Transformer
from utils import load_checkpoint, save_checkpoint


class AttentionIsAllYouNeedSchedule(optim.lr_scheduler._LRScheduler):
    """
    Custom learning rate schedule as described in the Transformer paper:
    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num*(warmup_steps^(-1.5)))
    """
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(AttentionIsAllYouNeedSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = max(1, self._step_count)
        scale = self.d_model ** -0.5
        warmup_factor = step_num ** -0.5
        linear_warmup = step_num * (self.warmup_steps ** -1.5)
        lr = scale * min(warmup_factor, linear_warmup)
        return [lr for _ in self.base_lrs]


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    criterion,
    device,
    current_step,
    total_steps,
    save_interval,
    save_dir,
    rank,
    verbose=True
):
    """
    Trains the model for an epoch or until we reach total_steps.
    Returns: updated_step, total_loss
    """
    model.train()
    epoch_loss = 0
    local_step = current_step  # track steps across epochs

    # Only rank 0 logs progress with tqdm
    if rank == 0:
        dataloader = tqdm(dataloader, desc="Training")

    for batch in dataloader:
        if local_step >= total_steps:
            break

        src, trg = batch
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        output = model(src, trg[:, :-1])  # [B, T-1, Vocab]
        output_reshape = output.contiguous().view(-1, output.shape[-1])  # [B*(T-1), V]
        trg_reshape = trg[:, 1:].contiguous().view(-1)  # [B*(T-1)]

        loss = criterion(output_reshape, trg_reshape)
        loss.backward()

        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        local_step += 1

        # Optional: log from rank 0
        if verbose and rank == 0 and (local_step % 100 == 0):
            current_lr = scheduler.get_lr()[0]
            print(
                f"[Rank {rank}] Step {local_step}/{total_steps}, "
                f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}"
            )

        # Only rank 0 saves checkpoints
        if (rank == 0) and ((local_step % save_interval == 0) or (local_step == total_steps)):
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                local_step,
                save_dir=save_dir,
                filename=f"checkpoint_step_{local_step}.pt",
            )

        if local_step >= total_steps:
            break

    return local_step, epoch_loss


def main_worker(rank, world_size, args):
    """
    Main worker function for each process (GPU). 
    Initializes distributed training, loads data, and kicks off training.
    """
    # 1. Initialize the process group
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)

    # 2. Set device for this process (one GPU per process)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # 3. Create (or load) model
    vocab = load_vocab("../data/vocab.ende")
    src_vocab_size = len(vocab[0]) + 1
    tgt_vocab_size = len(vocab[0]) + 1
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 512
    dropout = 0.1
    label_smoothing = 0.1

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

    # Move to device
    model.to(device)

    # Wrap with DistributedDataParallel
    # (find_unused_parameters=True if your forward pass has unused params)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # 4. Prepare optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = AttentionIsAllYouNeedSchedule(optimizer, d_model, warmup_steps=4000)

    # 5. Prepare dataset and sampler
    dataset = TranslationDataset("../data/dataset.parquet", start_idx=len(vocab[0]))

    # *** If you want to keep using your custom "TokenSumBatchSampler",
    # you'll need to adapt it for distributed usage. Instead, here's a simpler
    # example with DistributedSampler that just splits data among ranks.
    #
    # IMPORTANT: If your dataset is large and you're doing many steps,
    # consider an epoch-based training loop and rely on the sampler to shuffle
    # each epoch. For demonstration, we'll keep it simpler.

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=16,   # or whatever you prefer
        sampler=sampler, # use the distributed sampler
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True
    )

    # 6. Define loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)

    # 7. Load from an existing checkpoint? (Optional)
    # If you have a path, you can do something like:
    # current_step = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
    current_step = 0

    # 8. Train
    total_steps = args.total_steps
    save_interval = args.save_interval
    save_dir = args.save_dir

    # For simplicity, do a single "epoch" style pass until you hit total_steps
    # You might loop over multiple epochs in real code
    while current_step < total_steps:
        sampler.set_epoch(current_step)  # reshuffle data each "epoch"
        current_step, epoch_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            current_step=current_step,
            total_steps=total_steps,
            save_interval=save_interval,
            save_dir=save_dir,
            rank=rank,
            verbose=True
        )

    # 9. Final checkpoint (rank 0 only)
    if rank == 0:
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            current_step,
            save_dir=save_dir,
            filename=f"checkpoint_step_{current_step}.pt",
        )

    # 10. Clean up
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=100_000, help="Number of training steps total.")
    parser.add_argument("--save_interval", type=int, default=10_000, help="Interval for saving checkpoints.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    return parser.parse_args()


def main():
    """
    Spawns multiple processes for DDP training, one per GPU.
    """
    args = parse_args()

    # When using torchrun, the following env vars are automatically set:
    #   RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    # Each process runs main_worker with a different rank
    main_worker(rank, world_size, args)


if __name__ == "__main__":
    main()