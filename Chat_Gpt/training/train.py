# Disini gw melakukan training model GPT gw
# Gw cuma implementasi pelatihan pelatihan
# Dan menggabungkan semua algortima yang sudah gw buat
# Kek optimizer, dataloader, bpe, utils, dan transfomers
import torch
from torch.utils.data import random_split
from tqdm import tqdm
import random
import math
import numpy as np

# Konekin pytorch ke GPU
# Jangan pakai CPU nanti bakal lambat pelatihan nya
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🚀 Device yang di gunakan: {DEVICE} ({gpu_name})")
else:
    print(f"🚀 Device yang di gunakan: {DEVICE}")

# Parameter 
VOCAB_SIZE = 20000
D_MODEL = 128    
SEQ_LEN = 128
N_LAYERS = 6         
N_HEADS = 8        
D_FF = 512      
DROPOUT = 0.15     
BATCH_SIZE = 32      
EPOCHS = 100
LR = 3e-4             
WEIGHT_DECAY = 0.001   
GRAD_CLIP = 0.8       
LABEL_SMOOTHING = 0.1 
WARMUP_RATIO = 0.15   
ACCUMULATION_STEPS = 2 
AUGMENT_PROB = 0.1
MASK_PROB = 0.05    

class BalancedTextDataSet(torch.utils.data.Dataset):
    """Dataset dengan augmentasi yang lebih agresif untuk mengurangi overfitting"""
    
    def __init__(self, token_ids, seq_len=128, vocab_size=30000, 
                 augment_prob=0.15, mask_prob=0.1, noise_prob=0.05):
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.augment_prob = augment_prob
        self.mask_prob = mask_prob
        self.noise_prob = noise_prob
        
        # Buat urutan yang saling tumpang tindih untuk cakupan yang lebih baik
        self.sequences = []
        stride = max(1, seq_len // 4)  
        for i in range(0, len(token_ids) - seq_len + 1, stride):
            self.sequences.append(token_ids[i:i + seq_len])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx].copy()
        
        # Mengaplikasikan perkalian augmentasi teknik
        if random.random() < self.augment_prob:
            sequence = self._apply_augmentation(sequence)
        
        # Mengubah input dan target jadi tensor
        x = torch.tensor(sequence[:-1], dtype=torch.long)
        y = torch.tensor(sequence[1:], dtype=torch.long)
        
        # Pastikan ada padding yang tepat jika diperlukan
        if len(x) < self.seq_len - 1:
            pad_len = (self.seq_len - 1) - len(x)
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), -1, dtype=torch.long)])
        
        return x, y
    
    def _apply_augmentation(self, sequence):
        """Terapkan berbagai teknik augmentasi"""
        sequence = sequence.copy()
        
        # Random masking
        for i in range(len(sequence)):
            if random.random() < self.mask_prob:
                sequence[i] = min(1, self.vocab_size - 1)  
        
        # Token dropout (menepatkan dengan random token)
        for i in range(len(sequence)):
            if random.random() < self.noise_prob:
                sequence[i] = random.randint(0, min(self.vocab_size - 1, max(sequence)))
        
        # Pengacakan token (tukar token yang berdekatan)
        if random.random() < 0.1 and len(sequence) > 1:
            i = random.randint(0, len(sequence) - 2)
            sequence[i], sequence[i + 1] = sequence[i + 1], sequence[i]
        
        return sequence

def pelatihan_model_gpt():
    global VOCAB_SIZE
    global VOCAB_SIZE_ORIGINAL
    print("📖 Loading data...")
    with open(r"D:\Chatgpt_Pytorch\Chat_Gpt\data\input.txt", "r", encoding="utf-8") as f:
        text_data = f.read()
    
    print(f"📊 Data loaded: {len(text_data)} karakter")
    
    # Batasi data jika terlalu besar
    if len(text_data) > 1000000:
        text_data = text_data[:1000000]
        print(f"⚠️ Data dipotong menjadi {len(text_data)} karakter")
    
    # BPE tokenization
    print("🔤 Start BPE tokenisasi...")
    from Chat_Gpt.utils.bpe import BPE
    VOCAB_SIZE_ORIGINAL = VOCAB_SIZE
    bpe = BPE(max_tokens=VOCAB_SIZE_ORIGINAL)
    bpe.fit(text_data)

    # Dapatkan actual vocab size yang terbentuk
    actual_vocab_size = bpe.get_vocab_size()
    print(f"\n✅ Vocab info:")
    print(f"   Target: {VOCAB_SIZE}")
    print(f"   Actual: {actual_vocab_size}")
    print(f"   Merges: {len(bpe.merges)}")
    
    # Update global VOCAB_SIZE jadi actual size biar konsisten
    VOCAB_SIZE = actual_vocab_size
    
    # Membersihkan token id dengan validasi yang lebih ketat
    cleaned_token_ids = []
    invalid_count = 0
    unk_count = 0
    unk_id = bpe.token_to_id.get('<UNK>', 1)

    for tid in bpe.token_ids:
        # Validasi range token ID
        if tid < 0:
            cleaned_token_ids.append(0)
            invalid_count += 1
        elif tid >= actual_vocab_size:
            # Token ID keluar range, pake UNK
            cleaned_token_ids.append(unk_id)
            invalid_count += 1
        else:
            cleaned_token_ids.append(tid)
            # Hitung berapa banyak UNK dalam data
            if tid == unk_id:
                unk_count += 1

    # Statistik tokenization
    unique_tokens = len(set(cleaned_token_ids))
    vocab_coverage = (unique_tokens / actual_vocab_size) * 100 if actual_vocab_size > 0 else 0
    unk_ratio = (unk_count / len(cleaned_token_ids)) * 100 if len(cleaned_token_ids) > 0 else 0
    
    print(f"\n📊 Statistik Tokenization:")
    print(f"   Total tokens: {len(cleaned_token_ids):,}")
    print(f"   Unique tokens: {unique_tokens:,}")
    print(f"   Cakupan kosakata: {vocab_coverage:.1f}%")
    print(f"   UNK ratio: {unk_ratio:.2f}%")
    
    if invalid_count > 0:
        print(f"⚠️ Membersihkan {invalid_count} token ID yang tidak valid")
    
    print(f"✅ Token IDs di bersihkan: {len(cleaned_token_ids)} tokens")
    print(f"✅ Ukuran kosakata digunakan: {bpe.get_vocab_size()}")
    print(f"✅ Token unik dalam data: {len(set(cleaned_token_ids))}")
    
    # Warning kalo UNK terlalu banyak
    if unk_ratio > 5.0:
        print(f"   ⚠️ UNK ratio tinggi ({unk_ratio:.1f}%)! Pertimbangkan:")
        print(f"      - Naikin data training")
        print(f"      - Naikin target vocab size")
    elif unk_ratio < 1.0:
        print(f"   ✅ UNK ratio bagus ({unk_ratio:.2f}%)!")
    
    # Warning kalo vocab coverage rendah
    if vocab_coverage < 50:
        print(f"   ⚠️ Vocab coverage rendah ({vocab_coverage:.1f}%)!")
        print(f"      Banyak token gak kepake, vocab size mungkin terlalu besar")
        print(f"      Token unik dalam data: {len(set(cleaned_token_ids))}")

    # Membuat dataset jadi lebih balance
    print("📦 Membuat balanced dataset...")
    full_dataset = BalancedTextDataSet(
        cleaned_token_ids, 
        seq_len=SEQ_LEN,
        vocab_size=actual_vocab_size,
        augment_prob=AUGMENT_PROB,
        mask_prob=MASK_PROB
    )
    
    # Split dengan ratio yang lebih balanced untuk validation
    train_ratio = 0.8  # 80% train, 20% val (lebih banyak untuk val)
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"🎯 Train: {train_size}, Val: {val_size}")
    
    # DataLoaders
    from Chat_Gpt.training.dataloader import membuat_dataloader
    train_loader = membuat_dataloader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = membuat_dataloader(val_dataset, BATCH_SIZE, shuffle=False)
    
    # Model dengan regularization lebih kuat
    print("🏗️ Inisialisasi Model...")
    from Chat_Gpt.models.transfomers import membuat_gpt
    
    # Membuat inisialisasi model
    model = membuat_gpt(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        max_seq_len=SEQ_LEN,
        dropout=DROPOUT,
    ).to(DEVICE)
    
    # Terapkan pengurangan berat pada lebih banyak parameter
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if 'bias' in name or 'norm' in name or 'embed' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    # Optimizer dengan weight decay yang lebih selektif
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=LR, betas=(0.9, 0.95))
    
    # Scheduler dengan warmup yang lebih lama
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_RATIO * total_steps)
    
    # Loss dengan label smoothing lebih kuat
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=-1, 
        label_smoothing=LABEL_SMOOTHING
    )
    
    print(f"🎯 Setup selesai!")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Langkah Warmup: {warmup_steps}")
    print(f"Ukuran Batch yang Efektif: {BATCH_SIZE * ACCUMULATION_STEPS}")
    
    # Training loop dengan monitoring yang lebih ketat
    # FIX: Define best_val_loss di awal sebelum dipake
    best_val_loss = float('inf')
    best_val_kerugian = float('inf')  # Ini yang kurang, harus di define dulu
    patience = 12  
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*50}")
        print(f"🚂 EPOCH {epoch+1}/{EPOCHS}")
        print(f"{'='*50}")
        
        # Pelatihan Language yang kita buat
        model.train()
        total_loss = 0
        accumulated_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Train E{epoch+1}")
        
        optimizer.zero_grad()
        
        for batch_idx, (x, y) in enumerate(train_pbar):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # validasi input dari jarak tertentu
            x = torch.clamp(x, 0, VOCAB_SIZE - 1)
            y = torch.clamp(y, -1, VOCAB_SIZE - 1)
            
            # Forward pass
            logits = model(x)
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            
            # Scale loss
            scaled_loss = loss / ACCUMULATION_STEPS
            scaled_loss.backward()
            accumulated_loss += loss.item()
            
            # Akumulasi gradient
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += accumulated_loss / ACCUMULATION_STEPS
                accumulated_loss = 0
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
            })
        
        rata_rata_pelatihan_kerugian = total_loss / (len(train_loader) // ACCUMULATION_STEPS)
        train_losses.append(rata_rata_pelatihan_kerugian)
        
        # Bagian validasi
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Val E{epoch+1}")
        
        with torch.no_grad():
            for x_val, y_val in val_pbar:
                x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
                x_val = torch.clamp(x_val, 0, VOCAB_SIZE - 1)
                y_val = torch.clamp(y_val, -1, VOCAB_SIZE - 1)
                
                logits = model(x_val)
                loss = criterion(logits.view(-1, VOCAB_SIZE), y_val.view(-1))
                val_loss += loss.item()
                
                val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        rata_rata_val_kerugian = val_loss / len(val_loader)
        val_losses.append(rata_rata_val_kerugian)
        
        # Kalkulasi GAP
        gap = rata_rata_pelatihan_kerugian - rata_rata_val_kerugian
        gap_kesenjangan = abs(gap / rata_rata_val_kerugian) * 100 if rata_rata_val_kerugian > 0 else 0
        
        print(f"\n📈 Hasil dari Epoch {epoch+1}:")
        print(f"   Train Loss: {rata_rata_pelatihan_kerugian:.4f}")
        print(f"   Val Loss:   {rata_rata_val_kerugian:.4f}")
        print(f"   Gap:        {gap:.4f} ({gap_kesenjangan:.1f}%)")
        
        # Gap monitoring
        if gap_kesenjangan > 30:
            print(f"   ⚠️ Overfitting yang sangat tinggi! Gap: {gap_kesenjangan:.1f}%")
            # Reduksi Learning rate jika agressive model nya
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
                print(f"   📉 Learning rate dikurangi jadi: {param_group['lr']:.6f}")
        elif gap_kesenjangan < 15:
            print(f"   ✅ Model bagus tidak overfitting! Gap: {gap_kesenjangan:.1f}%")
        
        # Logika Early Stopping
        if rata_rata_val_kerugian < best_val_kerugian:
            best_val_kerugian = rata_rata_val_kerugian
            best_val_loss = rata_rata_val_kerugian  # Update keduanya biar konsisten
            patience_counter = 0
            
            # Save best model
            from Chat_Gpt.utils.utils import save_checkpoint
            save_checkpoint(model, optimizer, epoch+1, "best_balanced_model.pt")
            print(f"   ✅ Best model saved! Val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"   ⏳ Patience counter: {patience_counter}/{patience}")
        
        # Di berhentikan jika overfitting yang berlebihan
        if gap_kesenjangan > 50:
            print(f"\n🛑 Pelatihan di berhentikan gap nya terlalu jauh (gap: {gap_kesenjangan:.1f}%)")
            break
            
        if patience_counter >= patience:
            print(f"\n🛑 Tidak ada perubahan di berhentkan! {patience} epochs")
            break
        
        # Membersihkan memori
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n🎉 Pelatihan berhasil!")
    print(f"📊 Hasil akhir:")
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    if len(train_losses) > 0 and len(val_losses) > 0:
        print(f"   Akhir gap: {train_losses[-1] - val_losses[-1]:.4f}")
    
    return model, train_losses, val_losses

# Penambahan regulasi untuk biar model gak overfit berlebihan
class DropPath(torch.nn.Module):
    """Stochastic Depth / Drop Path regularization"""
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand(
            (x.shape[0], 1, 1), dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

# Gunakan fungsi ini untuk training
if __name__ == "__main__":
    model, train_losses, val_losses = pelatihan_model_gpt()