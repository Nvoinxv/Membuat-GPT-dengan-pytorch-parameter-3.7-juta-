# Disini gw membuat arsitektur transfomers
# Gw membuat algoritma FFN, MHA, dan LayerNorm untuk layernorm itu udah bawaan pytorch
# Kalau ffn dan MHA nya rada gw modif aja seperti GPT
# Dan juga gw tambahin decoder kek GPT style karna generate kata kata

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Bagian kelas FFN
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),  
            nn.GELU(),  
            nn.Linear(d_ff, d_model, bias=True),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

# Bagian kelas MHA
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Kombinasi QKV Projection 
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=True)
        self.c_proj = nn.Linear(d_model, d_model, bias=True)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024))
        
    def forward(self, x):
        B, T, C = x.size()
        
        # Kalkulasi QKV nya
        qkv = self.c_attn(x)  # B, T, 3*C
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # Reshape untuk multi-head
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # B, nh, T, dk
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # B, nh, T, dk
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # B, nh, T, dk
        
        # Scaled dot-product attention 
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention ke nilai
        y = att @ v  # B, nh, T, dk
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # B, T, C
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        """
        Membuat GPT block untuk menggabungkan LayerNorm, MultiHeadAttention,
        dan juga yang terakhit Feed Forward Neural Network. disini juga ada tambahan
        variabel seperti d_model, n_heads, d_ff, dan dropout.
        """
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTDecoder(nn.Module):
    def __init__(self,
                 vocab_size=20000,
                 d_model=768,
                 n_layers=12,
                 n_heads=12,
                 d_ff=None,
                 max_seq_len=1024,
                 dropout=0.1):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model  
            
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        
        # Token + Position embeddings
        self.wte = nn.Embedding(vocab_size, d_model)  # Token embeddings
        self.wpe = nn.Embedding(max_seq_len, d_model)  # Position embeddings
        self.drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.h = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # Output head - tied weights dengan token embedding
        # self.lm_head = tied to self.wte.weight
        
        self._init_weights()
        
        # Hitung total parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"GPT Model created with {total_params:,} parameters")
        
    def _init_weights(self):
        """GPT-2 style weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Standard normal with std=0.02
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
                
        # skalasi spesial init untuk residual projections
        for name, param in self.named_parameters():
            if name.endswith('c_proj.weight'):
                # 1/sqrt(2*n_layers) 
                nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * self.n_layers))
                
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.max_seq_len, f"Sequence length {t} > max_seq_len {self.max_seq_len}"
        
        # Token + Position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  
        
        tok_emb = self.wte(idx)  # (b, t, d_model)
        pos_emb = self.wpe(pos)  # (1, t, d_model)
        x = self.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.h:
            x = block(x)
            
        x = self.ln_f(x)  # (b, t, d_model)
        
        if targets is not None:
            # Training mode - compute loss
            logits = F.linear(x, self.wte.weight)  
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            # Inference mode
            logits = F.linear(x, self.wte.weight)
            return logits
    
    def get_num_params(self, non_embedding=True):
        """
        Kembalikan jumlah parameter dalam model.  
        Untuk perhitungan non-embedding (default), embedding posisi akan dikurangi.  
        Embedding token juga akan dikurangi, kecuali karena pembagian parameter, 
        parameter ini sebenarnya digunakan sebagai bobot di lapisan akhir.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Ambil urutan pengkondisian indeks idx (LongTensor dengan bentuk (b,t)) dan 
        lengkapi urutan tersebut sebanyak max_new_tokens kali, 
        dengan memberi umpan balik prediksi ke dalam model setiap kali.
        """
        for _ in range(max_new_tokens):
            # Crop idx to the last max_seq_len tokens
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            
            # Forward pass
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            #Secara opsional potong logit hanya pada k opsi teratas
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Menerapkan softmax and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Tambahkan ke urutan
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

def membuat_gpt(vocab_size=20000,
                    d_model=128,
                    n_layers=6,
                    n_heads=4,
                    d_ff=512,
                    max_seq_len=1024,
                    dropout=0.1):
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
    # Test model
    model = membuat_gpt()
    
    # Test forward pass
    x = torch.randint(0, 20000, (2, 64))
    y = torch.randint(0, 20000, (2, 64))
    
    logits, loss = model(x, y)
    print(f"Bentuk Input: {x.shape}")
    print(f"Bentuk Output: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test generarisasi
    generated = model.generate(x[:1, :10], max_new_tokens=50)
    print(f"Bentuk Generarisasi: {generated.shape}")