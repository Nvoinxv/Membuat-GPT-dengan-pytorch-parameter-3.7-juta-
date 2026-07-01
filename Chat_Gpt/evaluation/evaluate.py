"""
Modul Evaluasi dan Generasi Teks Model GPT.

Fungsionalitas:
- Pemuatan checkpoint model terbaik dan kosakata BPE yang telah dilatih
- Pengujian perplexity pada sampel data teks
- Generasi teks autoregresif dengan top-k sampling dan kendali temperatur
- Mode interaktif untuk pengujian prompt khusus dari pengguna
"""

import math
import os
import logging
import torch
import numpy as np

from Chat_Gpt.models.transfomers import membuat_gpt
from Chat_Gpt.utils.bpe import BPE

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("GPT_Evaluator")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_GPT_DIR = os.path.dirname(CURRENT_DIR)

DATA_PATH = os.path.join(CHAT_GPT_DIR, "data", "input.txt")
CHECKPOINT_PATH = os.path.join(CHAT_GPT_DIR, "checkpoints", "best_model.pt")
VOCAB_PATH = os.path.join(CHAT_GPT_DIR, "checkpoints", "vocab.json")

# Hiperparameter Arsitektur (Harus konsisten dengan pelatihan)
VOCAB_SIZE = 20000
D_MODEL = 128
SEQ_LEN = 128
N_LAYERS = 6
N_HEADS = 8
D_FF = 512
DROPOUT = 0.1


def load_model_and_tokenizer() -> tuple:
    """Memuat model dari checkpoint terbaik beserta tokenizer BPE yang bersesuaian."""
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Berkas checkpoint model tidak ditemukan di: {CHECKPOINT_PATH}. Harap jalankan pelatihan terlebih dahulu.")

    bpe = BPE(max_tokens=VOCAB_SIZE)
    if os.path.exists(VOCAB_PATH):
        logger.info(f"Memuat kosakata BPE dari {VOCAB_PATH}...")
        bpe.load_vocab(VOCAB_PATH)
    elif os.path.exists(DATA_PATH):
        logger.warning(f"Berkas kosakata tidak ditemukan di {VOCAB_PATH}. Melatih ulang BPE berdasarkan {DATA_PATH}...")
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            text_data = f.read()[:2000000]
        bpe.fit(text_data)
        os.makedirs(os.path.dirname(VOCAB_PATH), exist_ok=True)
        bpe.save_vocab(VOCAB_PATH)
    else:
        raise FileNotFoundError("Baik berkas kosakata maupun berkas data tidak ditemukan.")

    actual_vocab_size = bpe.get_vocab_size()

    logger.info(f"Memuat bobot model dari {CHECKPOINT_PATH}...")
    model = membuat_gpt(
        vocab_size=actual_vocab_size,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        max_seq_len=SEQ_LEN,
        dropout=DROPOUT,
    ).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'Unknown')
        val_loss = checkpoint.get('val_loss', 'N/A')
        logger.info(f"Model berhasil dimuat (Epoch: {epoch}, Val Loss: {val_loss})")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Model state dict berhasil dimuat secara langsung.")

    model.eval()
    return model, bpe


def generate_text(model: torch.nn.Module, bpe: BPE, prompt: str, max_length: int = 80, temperature: float = 0.8, top_k: int = 40) -> str:
    """Menghasilkan teks berdasarkan prompt masukan menggunakan top-k sampling."""
    model.eval()
    temperature = max(temperature, 1e-5)

    tokens = bpe.encode(prompt)
    actual_vocab_size = bpe.get_vocab_size()
    tokens = [max(0, min(t, actual_vocab_size - 1)) for t in tokens]

    generated_tokens = tokens.copy()

    with torch.no_grad():
        for _ in range(max_length):
            context = generated_tokens[-SEQ_LEN:] if len(generated_tokens) > SEQ_LEN else generated_tokens
            x = torch.tensor([context], dtype=torch.long, device=DEVICE)

            logits = model(x)
            logits = logits[0, -1, :] / temperature

            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(0, top_k_indices, top_k_logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            next_token = max(0, min(next_token, actual_vocab_size - 1))

            generated_tokens.append(next_token)

    return bpe.decode(generated_tokens)


def calculate_perplexity(model: torch.nn.Module, bpe: BPE, text_sample: str) -> float:
    """Menghitung nilai perplexity model terhadap sampel teks."""
    model.eval()
    tokens = bpe.encode(text_sample)
    actual_vocab_size = bpe.get_vocab_size()
    tokens = [max(0, min(t, actual_vocab_size - 1)) for t in tokens]

    if len(tokens) < 2:
        logger.warning("Sampel teks terlalu pendek untuk evaluasi perplexity.")
        return float('inf')

    total_loss = 0.0
    count = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for i in range(1, len(tokens)):
            start = max(0, i - SEQ_LEN + 1)
            context = tokens[start:i]
            target = tokens[i]

            x = torch.tensor([context], dtype=torch.long, device=DEVICE)
            y = torch.tensor([target], dtype=torch.long, device=DEVICE)

            logits = model(x)
            loss = criterion(logits[0, -1:, :], y)
            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / max(1, count)
    return math.exp(min(avg_loss, 20.0))


def run_sample_evaluation():
    """Menjalankan evaluasi otomatis pada serangkaian prompt uji."""
    model, bpe = load_model_and_tokenizer()

    test_prompts = [
        "First Citizen:",
        "MARCIUS:",
        "Pada suatu hari",
        "Kecerdasan buatan adalah",
        "Machine learning"
    ]

    print("\n" + "=" * 70)
    print("EVALUASI GENERASI TEKS OTOMATIS")
    print("=" * 70)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Pengujian {i}/{len(test_prompts)}] Prompt: '{prompt}'")
        print("-" * 70)
        output = generate_text(model, bpe, prompt, max_length=60, temperature=0.7, top_k=40)
        print(output)
        print("-" * 70)


def run_perplexity_test():
    """Menguji nilai perplexity pada korpus data yang tersedia."""
    model, bpe = load_model_and_tokenizer()

    if not os.path.exists(DATA_PATH):
        logger.error(f"Berkas data tidak ditemukan di {DATA_PATH} untuk uji perplexity.")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    sample_length = 500
    if len(text) > sample_length:
        start_idx = np.random.randint(0, len(text) - sample_length)
        sample_text = text[start_idx : start_idx + sample_length]
    else:
        sample_text = text

    print("\n" + "=" * 70)
    print("PENGUJIAN PERPLEXITY MODEL")
    print("=" * 70)
    print(f"Cuplikan Teks Uji: '{sample_text[:120]}...'")
    ppl = calculate_perplexity(model, bpe, sample_text)
    print("-" * 70)
    print(f"Nilai Perplexity: {ppl:.2f} (Semakin rendah semakin baik)")
    print("=" * 70)


def interactive_mode():
    """Menjalankan sesi interaktif untuk pengujian prompt masukan pengguna."""
    model, bpe = load_model_and_tokenizer()

    print("\n" + "=" * 70)
    print("MODE INTERAKTIF GENERASI TEKS")
    print("Ketik 'exit' atau 'keluar' untuk mengakhiri sesi.")
    print("=" * 70)

    while True:
        try:
            prompt = input("\n[Masukan Prompt] > ").strip()
            if prompt.lower() in ["exit", "keluar", "quit"]:
                print("[INFO] Sesi interaktif diakhiri.")
                break
            if not prompt:
                print("[PERINGATAN] Prompt tidak boleh kosong.")
                continue

            print("-" * 70)
            output = generate_text(model, bpe, prompt, max_length=80, temperature=0.8, top_k=40)
            print(output)
            print("-" * 70)

        except KeyboardInterrupt:
            print("\n[INFO] Sesi diakhiri oleh pengguna.")
            break
        except Exception as e:
            logger.error(f"Terjadi kesalahan saat generasi teks: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MENU EVALUASI MODEL GPT")
    print("=" * 70)
    print("1. Pengujian dengan prompt otomatis")
    print("2. Mode interaktif (prompt khusus)")
    print("3. Pengujian perplexity model")
    print("4. Jalankan seluruh pengujian (1 & 3)")
    print("=" * 70)

    try:
        pilihan = input("Pilih menu (1-4): ").strip()
        if pilihan == "1":
            run_sample_evaluation()
        elif pilihan == "2":
            interactive_mode()
        elif pilihan == "3":
            run_perplexity_test()
        elif pilihan == "4":
            run_sample_evaluation()
            run_perplexity_test()
        else:
            print("[INFO] Pilihan tidak dikenali. Menjalankan pengujian prompt otomatis...")
            run_sample_evaluation()
    except KeyboardInterrupt:
        print("\n[INFO] Program dihentikan.")