"""
Modul Arsitektur Transformer GPT (Generative Pre-trained Transformer).

Implementasi arsitektur GPT dari awal menggunakan PyTorch, mencakup:
- FeedForward Neural Network (FFN) dengan aktivasi GELU
- Multi-Head Causal Self-Attention (MHA)
- Blok Transformer dengan Pre-LayerNorm
- GPT Decoder dengan Weight Tying pada Token Embedding
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("GPT_Model")


class FeedForward(nn.Module):
    """Lapisan Feed-Forward Neural Network (MLP) dengan aktivasi GELU."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """Lapisan Multi-Head Causal Self-Attention."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) harus dapat dibagi oleh n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Proyeksi Q, K, V digabungkan dalam satu lapisan Linear untuk efisiensi
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=True)
        self.c_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Buffer masker kausal (segitiga bawah) untuk mencegah atensi ke token masa depan
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Hitung Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape menjadi (B, n_heads, T, d_k)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
        
        # Terapkan masker kausal
        att = att.masked_fill(~self.mask[:, :, :T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Agregasi nilai (value)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Proyeksi output residual
        y = self.resid_dropout(self.c_proj(y))
        return y


class GPTBlock(nn.Module):
    """Blok Transformer GPT tunggal menggunakan arsitektur Pre-LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, max_seq_len=max_seq_len, dropout=dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTDecoder(nn.Module):
    """Model Language Model (LM) berbasis GPT Decoder."""

    def __init__(self,
                 vocab_size: int = 20000,
                 d_model: int = 768,
                 n_layers: int = 12,
                 n_heads: int = 12,
                 d_ff: int = None,
                 max_seq_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        # Token dan Position Embeddings
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        # Transformer Blocks
        self.h = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff, max_seq_len=max_seq_len, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Normalisasi akhir
        self.ln_f = nn.LayerNorm(d_model)

        # Inisialisasi bobot standar GPT-2
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Model GPT berhasil diinisialisasi dengan {total_params:,} parameter.")

    def _init_weights(self):
        """Inisialisasi parameter model mengikuti standar GPT-2."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

        # Skala khusus untuk proyeksi residual (c_proj) untuk menjaga varians lapisan dalam
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.max_seq_len, f"Panjang sekuens ({t}) melebihi max_seq_len ({self.max_seq_len})"

        # Posisi tensor (0, 1, ..., t-1)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.wte(idx)  # (B, T, d_model)
        pos_emb = self.wpe(pos)  # (1, T, d_model)
        x = self.drop(tok_emb + pos_emb)

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)

        # Logits menggunakan weight tying dari token embedding (self.wte.weight)
        logits = F.linear(x, self.wte.weight)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            return logits

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Menghitung jumlah parameter dalam model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """Menghasilkan token baru secara autoregresif."""
        self.eval()
        temperature = max(temperature, 1e-5)  # Mencegah division by zero

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]

            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def membuat_gpt(vocab_size: int = 20000,
                d_model: int = 128,
                n_layers: int = 6,
                n_heads: int = 4,
                d_ff: int = 512,
                max_seq_len: int = 1024,
                dropout: float = 0.1) -> GPTDecoder:
        """Fungsi pembantu untuk membuat instance model GPTDecoder."""
        return GPTDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    model = membuat_gpt()

    x = torch.randint(0, 20000, (2, 64))
    y = torch.randint(0, 20000, (2, 64))

    logits, loss = model(x, y)
    print(f"[INFO] Bentuk Input: {x.shape}")
    print(f"[INFO] Bentuk Output: {logits.shape}")
    print(f"[INFO] Loss Uji: {loss.item():.4f}")

    generated = model.generate(x[:1, :10], max_new_tokens=20)
    print(f"[INFO] Bentuk Generasi: {generated.shape}")