import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from tqdm.auto import tqdm
from torch import optim

from data import (
    TokenSumBatchSampler,
    TranslationDataset,
    collate_fn,
    load_vocab,
)
from transformer import Transformer
from utils import save_checkpoint


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
    model.train()
    epoch_loss = 0
    local_step = current_step 

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

        if verbose and rank == 0 and (local_step % 100 == 0):
            current_lr = scheduler.get_lr()[0]
            print(
                f"[Rank {rank}] Step {local_step}/{total_steps}, "
                f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}"
            )

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

def main():
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

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

    model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = AttentionIsAllYouNeedSchedule(optimizer, d_model, warmup_steps=4000)
    dataset = TranslationDataset("../data/dataset.parquet", start_idx=len(vocab[0]))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)
    current_step = 0

    # Train
    total_steps = 100000
    save_interval = 10000
    save_dir = "checkpoints"

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

    if rank == 0:
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            current_step,
            save_dir=save_dir,
            filename=f"checkpoint_step_{current_step}.pt",
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()