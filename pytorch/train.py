import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm.auto import tqdm
import torch
from torch import optim
from pytorch.data import (
    TokenSumBatchSampler,
    TranslationDataset,
    collate_fn,
    load_vocab,
)
from pytorch.transformer import Transformer
from pytorch.utils import load_checkpoint, save_checkpoint
from torch.utils.data import DataLoader


class AttentionIsAllYouNeedSchedule(optim.lr_scheduler._LRScheduler):
    """
    Custom learning rate schedule as described in the Transformer paper:
    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):

        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(AttentionIsAllYouNeedSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = max(1, self._step_count)  # Ensure step_num >= 1
        scale = self.d_model**-0.5
        warmup_factor = step_num**-0.5
        linear_warmup = step_num * (self.warmup_steps**-1.5)
        lr = scale * min(warmup_factor, linear_warmup)

        return [lr for _ in self.base_lrs]


def train(
    model,
    dataloader,
    optimizer,
    scheduler,
    criterion,
    total_steps,
    device="cpu",
    verbose=True,
    save_dir="checkpoints",
    save_interval=1000,
    checkpoint_path=None,
):
    model = model.to(device)
    model.train()
    step = 0
    epoch_loss = 0

    # Load from checkpoint if provided
    if checkpoint_path:
        step = load_checkpoint(model, optimizer, scheduler, checkpoint_path)

    # Loop through the dataloader
    for step in tqdm(range(total_steps), desc="Training Transformer"):
        for batch in dataloader:
            # Stop training after the total number of steps
            if step >= total_steps:
                break

            src, trg = batch
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()

            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            step += 1

            # Save checkpoint at intervals
            if step % save_interval == 0 or step == total_steps:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    step,
                    save_dir=save_dir,
                    filename=f"checkpoint_step_{step}.pt",
                )

            # if verbose and step % 100 == 0:
            print(
                f"Step {step}/{total_steps}, Loss: {loss.item()}, LR: {scheduler.get_lr()[0]:.6f}"
            )

    save_checkpoint(
        model,
        optimizer,
        scheduler,
        step,
        save_dir=save_dir,
        filename=f"checkpoint_step_{step}.pt",
    )

    avg_loss = epoch_loss / total_steps
    return avg_loss


if __name__ == "__main__":
    vocab = load_vocab("../data/vocab.ende")
    dataset = TranslationDataset("../data/translate_ende_small.parquet")

    batch_tokens = 100
    dataloader = DataLoader(
        dataset,
        batch_sampler=TokenSumBatchSampler(dataset, max_tokens=batch_tokens),
        collate_fn=collate_fn,
    )

    src_vocab_size = len(vocab[0])
    tgt_vocab_size = len(vocab[0])
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
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)

    warmpup_steps = 4000
    scheduler = AttentionIsAllYouNeedSchedule(
        optimizer, d_model, warmup_steps=warmpup_steps
    )

    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=0, label_smoothing=label_smoothing
    )

    total_steps = 300000
    train(model, dataloader, optimizer, scheduler, criterion, total_steps)
