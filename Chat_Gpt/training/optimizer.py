"""
Modul Konfigurasi Optimizer.

Mengatur parameter grouping untuk AdamW pada model GPT:
- Memberikan weight decay pada matriks bobot (Linear, Conv)
- Menonaktifkan weight decay pada parameter bias, LayerNorm, dan Embedding
"""

import logging
import torch
from torch.optim import AdamW

logger = logging.getLogger("GPT_Optimizer")


def konfigurasi_optimisasi(model: torch.nn.Module,
                           learning_rate: float = 6e-4,
                           weight_decay: float = 0.1,
                           betas: tuple = (0.9, 0.95),
                           eps: float = 1e-8) -> AdamW:
    """
    Mengkonfigurasi optimizer AdamW dengan parameter grouping yang sesuai untuk arsitektur Transformer.
    """
    decay = set()
    no_decay = set()

    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # Tangani kasus weight tying pada lm_head jika ada
    if "lm_head.weight" in param_dict:
        no_decay.add("lm_head.weight")

    # Pastikan setiap parameter masuk tepat di satu grup
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"Parameter berikut masuk ke dalam decay dan no_decay sekaligus: {inter_params}"
    
    missing_params = param_dict.keys() - union_params
    assert len(missing_params) == 0, f"Parameter berikut belum dikelompokkan ke decay/no_decay: {missing_params}"

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps)

    logger.info(f"Konfigurasi Optimizer AdamW - Parameter dengan weight decay: {len(decay)}, Tanpa weight decay: {len(no_decay)}")

    return optimizer


def mendapatkan_optimizer(model: torch.nn.Module,
                          lr: float = 1e-4,
                          weight_decay: float = 0.1,
                          betas: tuple = (0.9, 0.95)) -> AdamW:
    """Fungsi pembantu (wrapper) untuk kompatibilitas mundur."""
    return konfigurasi_optimisasi(model, learning_rate=lr, weight_decay=weight_decay, betas=betas)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 128)
            self.layer_norm = nn.LayerNorm(128)
            self.linear1 = nn.Linear(128, 256)
            self.linear2 = nn.Linear(256, 128, bias=False)

    model = DummyModel()
    optimizer = konfigurasi_optimisasi(model)
    print(f"[INFO] Pengujian konfigurasi optimizer berhasil. Learning Rate awal: {optimizer.param_groups[0]['lr']}")