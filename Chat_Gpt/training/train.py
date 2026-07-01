"""
Modul Pelatihan (Training) Model GPT.

Menjalankan alur pelatihan model language model secara profesional:
- Pemrosesan data teks dan tokenisasi menggunakan BPE
- Dataset Autoregresif Causal Language Modeling (tanpa korupsi masking/noise acak)
- Optimisasi parameter terstruktur (AdamW) dengan pemisahan weight decay
- Penjadwalan learning rate dengan Cosine Decay dan Linear Warmup
- Pemantauan metrik (Loss, Perplexity, Gap) dan penyimpanan checkpoint terbaik
"""

import math
import os
import time
import logging
import torch
from torch.utils.data import Dataset, random_split
from tqdm import tqdm

from Chat_Gpt.models.transfomers import membuat_gpt
from Chat_Gpt.training.optimizer import konfigurasi_optimisasi
from Chat_Gpt.training.dataloader import membuat_dataloader
from Chat_Gpt.utils.bpe import BPE
from Chat_Gpt.utils.utils import save_checkpoint, get_lr_cosine_schedule

# Konfigurasi Logging Profesional
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("GPT_Trainer")

# Pengaturan Perangkat keras
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resolusi Jalur Berkas (Paths) secara Dinamis
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_GPT_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(CHAT_GPT_DIR)

DATA_DIR = os.path.join(CHAT_GPT_DIR, "data")
CHECKPOINT_DIR = os.path.join(CHAT_GPT_DIR, "checkpoints")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "input.txt")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
VOCAB_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "vocab.json")

# Hiperparameter Model & Pelatihan (~3.7 Juta Parameter)
VOCAB_SIZE = 20000
D_MODEL = 128
SEQ_LEN = 128
N_LAYERS = 6
N_HEADS = 8
D_FF = 512
DROPOUT = 0.1

BATCH_SIZE = 32
EPOCHS = 50
MAX_LR = 6e-4
MIN_LR = 3e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_RATIO = 0.1
ACCUMULATION_STEPS = 2


class LanguageModelDataset(Dataset):
    """
    Dataset untuk Causal Language Modeling (Autoregressive).
    Memecah deret token menjadi sekuens berukuran (SEQ_LEN + 1) untuk target pergeseran 1 token:
    Input (x)  : token[0 : SEQ_LEN]
    Target (y) : token[1 : SEQ_LEN + 1]
    """

    def __init__(self, token_ids: list, seq_len: int = 128):
        self.seq_len = seq_len
        self.sequences = []

        # Langkah geser (stride) diatur setengah dari seq_len untuk memperkaya sampel latih
        stride = max(1, seq_len // 2)
        for i in range(0, len(token_ids) - seq_len, stride):
            chunk = token_ids[i : i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.sequences.append(chunk)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


def siapkan_data_dan_tokenizer(data_path: str = DEFAULT_DATA_PATH) -> tuple:
    """Memuat data teks, melatih BPE tokenizer, dan mengembalikan token IDs bersih."""
    if not os.path.exists(data_path):
        logger.warning(f"Berkas data tidak ditemukan di {data_path}. Membuat sampel korpus teks default...")
        sample_corpus = (
            "Pada suatu hari di masa depan, kecerdasan buatan telah berkembang sangat pesat. "
            "Model bahasa seperti Transformer mampu memahami struktur sintaksis dan semantik secara mendalam. "
            "Pelatihan model deep learning membutuhkan arsitektur yang tepat, optimisasi parameter yang akurat, "
            "serta penjadwalan learning rate yang teratur. Dengan arsitektur Multi-Head Attention dan Feed Forward Network, "
            "model dapat mengolah urutan kata dengan sangat efisien. "
        ) * 500
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(sample_corpus)
        logger.info(f"Berkas sampel data berhasil dibuat di: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    logger.info(f"Korpus teks dimuat. Total karakter: {len(text_data):,}")
    if len(text_data) > 2000000:
        text_data = text_data[:2000000]
        logger.info(f"Korpus dibatasi hingga 2,000,000 karakter untuk efisiensi tokenisasi.")

    bpe = BPE(max_tokens=VOCAB_SIZE)
    bpe.fit(text_data)

    # Simpan kosakata agar konsisten saat evaluasi
    bpe.save_vocab(VOCAB_SAVE_PATH)

    actual_vocab_size = bpe.get_vocab_size()
    cleaned_token_ids = [max(0, min(tid, actual_vocab_size - 1)) for tid in bpe.token_ids]

    logger.info(f"Tokenisasi Selesai - Ukuran Kosakata Aktif: {actual_vocab_size:,} | Total Token Datasets: {len(cleaned_token_ids):,}")
    return cleaned_token_ids, actual_vocab_size, bpe


def pelatihan_model_gpt():
    if DEVICE.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Perangkat komputasi aktif: {DEVICE} ({gpu_name})")
    else:
        logger.info(f"Perangkat komputasi aktif: {DEVICE}")

    token_ids, actual_vocab_size, bpe = siapkan_data_dan_tokenizer(DEFAULT_DATA_PATH)

    dataset = LanguageModelDataset(token_ids, seq_len=SEQ_LEN)
    if len(dataset) == 0:
        raise ValueError("Dataset kosong! Pastikan korpus teks lebih panjang dari SEQ_LEN.")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"Distribusi Dataset - Pembelajaran (Train): {train_size:,} sampel | Validasi (Val): {val_size:,} sampel")

    train_loader = membuat_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = membuat_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logger.info("Membangun arsitektur model GPT...")
    model = membuat_gpt(
        vocab_size=actual_vocab_size,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        max_seq_len=SEQ_LEN,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = konfigurasi_optimisasi(model, learning_rate=MAX_LR, weight_decay=WEIGHT_DECAY)

    total_steps = EPOCHS * (len(train_loader) // ACCUMULATION_STEPS)
    warmup_steps = int(WARMUP_RATIO * total_steps)
    logger.info(f"Total Langkah Pelatihan: {total_steps:,} | Langkah Warmup: {warmup_steps:,}")

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    train_losses = []
    val_losses = []
    global_step = 0

    logger.info("-" * 80)
    logger.info(f"{'EPOCH':<8} | {'TRAIN LOSS':<12} | {'VAL LOSS':<12} | {'PERPLEXITY':<12} | {'LR':<10} | {'WAKTU':<8}")
    logger.info("-" * 80)

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Perbarui learning rate berdasarkan jadwal Cosine
            lr_current = get_lr_cosine_schedule(global_step, warmup_steps, total_steps, MAX_LR, MIN_LR)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_current

            logits = model(x)
            loss = criterion(logits.view(-1, actual_vocab_size), y.view(-1))
            scaled_loss = loss / ACCUMULATION_STEPS
            scaled_loss.backward()

            total_train_loss += loss.item()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validasi
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
                logits_val = model(x_val)
                loss_val = criterion(logits_val.view(-1, actual_vocab_size), y_val.view(-1))
                total_val_loss += loss_val.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_perplexity = math.exp(min(avg_val_loss, 20.0))
        elapsed_time = time.time() - start_time

        logger.info(
            f"{epoch:<8} | {avg_train_loss:<12.4f} | {avg_val_loss:<12.4f} | "
            f"{val_perplexity:<12.2f} | {lr_current:<10.6f} | {elapsed_time:<6.1f}s"
        )

        # Simpan Checkpoint Terbaik
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss=best_val_loss, path=BEST_MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping dipicu setelah {epoch} epoch tanpa perbaikan validasi loss.")
            break

    logger.info("-" * 80)
    logger.info(f"Pelatihan Selesai! Validation Loss Terbaik: {best_val_loss:.4f} | Perplexity Terbaik: {math.exp(min(best_val_loss, 20.0)):.2f}")
    return model, train_losses, val_losses


if __name__ == "__main__":
    pelatihan_model_gpt()