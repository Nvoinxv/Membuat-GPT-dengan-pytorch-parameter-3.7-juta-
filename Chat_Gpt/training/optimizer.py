# Sebenarnya, Adam bawaan PyTorch udah cukup bagus untuk banyak kasus, tapi untuk language model
# kayak GPT, optimisasi itu butuh perlakuan khusus. Kalau pakai Adam default, semua parameter diperlakukan sama
# padahal ada jenis parameter tertentu (seperti bias, embedding, dan LayerNorm) yang tidak cocok diberi weight decay.
# Jadi, hasil training bisa kurang optimal, misalnya loss jadi lebih tinggi atau konvergensinya lebih lambat.
# Makanya gw bikin konfigurasi khusus dengan parameter grouping supaya optimisasi lebih presisi sesuai arsitektur modelnya.

import torch
from torch.optim import AdamW
import math

def konfigurasi_optimisasi(model, learning_rate=6e-4, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8):
    """
    Mengkonfigurasi optimizer GPT style dengan parameter grouping yang proper
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
    
    # Tambahkan hanya kalau ada di model
    param_dict = {pn: p for pn, p in model.named_parameters()}
    if "lm_head.weight" in param_dict:
        no_decay.add("lm_head.weight")
    
    inter_params = decay & no_decay
    union_params = decay | no_decay
    
    assert len(inter_params) == 0, f"Parameters {str(inter_params)} masuk ke decay & no_decay!"
    assert len(param_dict.keys() - union_params) == 0, f"Parameters {str(param_dict.keys() - union_params)} belum masuk decay/no_decay!"
    
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
    ]
    
    optimizer = AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps)
    
    print("Konfigurasi Optimisasi:")
    print(f"  Decay params: {len(decay)}")
    print(f"  No decay params: {len(no_decay)}")
    print(f"  Total params: {len(param_dict)}")
    
    return optimizer

# Backward fungsi kompatibilitas
def mendapatkan_optimizer(model, lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95)):
    """Wrapper untuk backward compatibility"""
    return konfigurasi_optimisasi(model, lr, weight_decay, betas)

# Test fungsi
if __name__ == "__main__":
    import torch.nn as nn
    
    # membuat dummy model
    # lebih ke testing saja
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 128)
            self.layer_norm = nn.LayerNorm(128)
            self.linear1 = nn.Linear(128, 256)
            self.linear2 = nn.Linear(256, 128, bias=False)
    
    model = DummyModel()
    optimizer = konfigurasi_optimisasi(model)
    print("Pengoptimalan test berhasil!")
    print(f"Inisial LR: {optimizer.param_groups[0]['lr']}")