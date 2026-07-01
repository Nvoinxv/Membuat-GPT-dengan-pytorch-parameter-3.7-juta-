"""
Modul Utilitas untuk Penyimpanan Checkpoint, Inisialisasi, dan Penjadwalan Learning Rate.
"""

import math
import logging
import torch
import torch.nn as nn

logger = logging.getLogger("GPT_Utils")


def init_weights_xavier(model: nn.Module):
    """Inisialisasi bobot model menggunakan Xavier Uniform."""
    for name, param in model.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)


def freeze_model(model: nn.Module):
    """Membekukan seluruh parameter model agar tidak diperbarui saat propagasi balik."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: nn.Module):
    """Membuka kebekuan seluruh parameter model."""
    for param in model.parameters():
        param.requires_grad = True


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, val_loss: float = None, path: str = "checkpoint.pt"):
    """Menyimpan state model dan optimizer ke file checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss
    }
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint berhasil disimpan di: {path}")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer = None, path: str = "checkpoint.pt", device: str = "cpu") -> int:
    """Memuat state model dan optimizer dari file checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    logger.info(f"Checkpoint dimuat dari {path} (Epoch {epoch})")
    return epoch


def get_lr_cosine_schedule(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float = 1e-5) -> float:
    """
    Menghitung learning rate dengan Linear Warmup dan Cosine Decay.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step > total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
