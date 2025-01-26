import os
import torch

def save_checkpoint(model, optimizer, scheduler, step, save_dir="checkpoints", filename="checkpoint.pt"):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
    }, checkpoint_path)
    print(f"Checkpoint saved at step {step} to {checkpoint_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    step = checkpoint["step"]
    print(f"Checkpoint loaded from {checkpoint_path} at step {step}")
    return step
