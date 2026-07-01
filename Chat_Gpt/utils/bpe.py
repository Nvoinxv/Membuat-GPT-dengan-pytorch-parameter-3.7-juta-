"""
Modul Byte Pair Encoding (BPE) Tokenizer.

Tokenizer berlevel sub-word yang dioptimalkan untuk pelatihan model GPT, mencakup:
- Penanganan token khusus (<PAD>, <UNK>, <BOS>, <EOS>, <MASK>)
- Pelatihan berbasis frekuensi pasangan karakter (BPE Merging)
- Utilitas Encoding, Decoding, serta penyimpanan dan pemuatan kosakata (JSON)
"""

from collections import Counter, defaultdict
import json
import logging
import random
import re
from tqdm import tqdm

logger = logging.getLogger("BPE_Tokenizer")


class BPE:
    def __init__(self, max_tokens: int = 20000, special_tokens: list = None):
        self.max_tokens = max_tokens

        default_special_tokens = [
            '<PAD>',
            '<UNK>',
            '<BOS>',
            '<EOS>',
            '<MASK>',
        ]

        self.special_tokens = special_tokens or default_special_tokens

        self.vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = []
        self.token_ids = []
        self.is_trained = False

        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.token_to_id[token] = i
            self.id_to_token[i] = token

    def preprocess_text(self, text: str) -> str:
        """Membersihkan teks dan melakukan normalisasi spasi serta tanda baca."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.!?;:])', r' \1 ', text)
        text = re.sub(r'([,])', r' \1', text)
        text = re.sub(r'(["\'()])', r' \1 ', text)
        text = re.sub(r'(\d+)', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def mendapatkan_kata_tokens(self, text: str) -> list:
        """Memecah teks menjadi token berlevel karakter dengan penanda batas kata (</w>)."""
        kata = text.split()
        token_kata = []
        for word in kata:
            chars = [c for c in word] + ['</w>']
            token_kata.extend(chars)
        return token_kata

    def dapatkan_statistik(self, tokens: list) -> dict:
        """Menghitung frekuensi pasangan token yang bersebelahan."""
        pairs = defaultdict(int)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] += 1
        return pairs

    def Gabung_Token(self, tokens: list, pair: tuple) -> list:
        """Menggabungkan pasangan token yang dipilih dalam sekuens token."""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                merged = tokens[i] + tokens[i + 1]
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def fit(self, text: str):
        """Melatih tokenizer BPE berdasarkan korpus teks yang diberikan."""
        logger.info(f"Memulai pelatihan BPE dengan target maksimal kosakata: {self.max_tokens}")

        processed_text = self.preprocess_text(text)
        logger.info(f"Pra-pemrosesan selesai: {len(text):,} -> {len(processed_text):,} karakter")

        tokens = self.mendapatkan_kata_tokens(processed_text)
        logger.info(f"Jumlah token karakter awal: {len(tokens):,}")

        unique_chars = sorted(set(tokens))
        for char in unique_chars:
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char

        initial_vocab_size = len(self.vocab)
        logger.info(f"Ukuran kosakata dasar (termasuk token khusus): {initial_vocab_size}")

        # Batas target agar tidak melebihi self.max_tokens
        target_vocab = min(self.max_tokens, max(initial_vocab_size, int(len(tokens) * 0.15)))
        target_vocab = min(self.max_tokens, target_vocab)
        logger.info(f"Target ukuran kosakata penyesuaian: {target_vocab}")

        num_merges = target_vocab - len(self.vocab)
        if num_merges <= 0:
            logger.info("Ukuran kosakata dasar sudah mencapai atau melebihi target.")
        else:
            pbar = tqdm(range(num_merges), desc="BPE Merges", leave=False)
            min_freq_threshold = 2

            for _ in pbar:
                pairs = self.dapatkan_statistik(tokens)
                if not pairs:
                    break

                best_pair = max(pairs, key=pairs.get)
                best_freq = pairs[best_pair]

                if best_freq < min_freq_threshold:
                    break

                tokens = self.Gabung_Token(tokens, best_pair)
                penggabungan_token = best_pair[0] + best_pair[1]
                if penggabungan_token not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[penggabungan_token] = idx
                    self.token_to_id[penggabungan_token] = idx
                    self.id_to_token[idx] = penggabungan_token

                self.merges.append(best_pair)
                pbar.set_postfix({'vocab': len(self.vocab), 'freq': best_freq})

                if len(self.vocab) >= self.max_tokens:
                    break
            pbar.close()

        actual_vocab_size = len(self.vocab)
        unk_id = self.token_to_id.get('<UNK>', 1)
        self.token_ids = [self.token_to_id.get(t, unk_id) for t in tokens]
        self.max_tokens = actual_vocab_size
        self.is_trained = True

        logger.info(f"Pelatihan BPE Selesai - Kosakata akhir: {actual_vocab_size:,}, Jumlah penggabungan: {len(self.merges):,}")
        if len(self.token_ids) > 0:
            rasio_kompresi = len(processed_text) / len(self.token_ids)
            logger.info(f"Rasio Kompresi: {rasio_kompresi:.2f}x")

    def encode(self, text: str) -> list:
        """Mengenkodasi teks menjadi daftar token ID."""
        if not self.is_trained:
            raise ValueError("BPE belum dilatih atau dimuat! Panggil fit() atau load_vocab() terlebih dahulu.")

        processed_text = self.preprocess_text(text)
        tokens = self.mendapatkan_kata_tokens(processed_text)

        for merge in self.merges:
            tokens = self.Gabung_Token(tokens, merge)

        unk_id = self.token_to_id.get('<UNK>', 1)
        token_ids = [self.token_to_id.get(token, unk_id) for token in tokens]
        return token_ids

    def decode(self, token_ids: list) -> str:
        """Mendekodasi daftar token ID kembali menjadi teks natural."""
        if not self.is_trained:
            raise ValueError("BPE belum dilatih atau dimuat!")

        tokens = []
        unk_token = '<UNK>'
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append(unk_token)

        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([(\[])\s+', r'\1', text)
        text = re.sub(r'\s+([)\]])', r'\1', text)
        return text

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def random_mask(self, token_ids: list, mask_prob: float = 0.15) -> list:
        """Menerapkan teknik masking acak pada token ID (untuk keperluan eksperimental non-autoregresif)."""
        mask_token_id = self.token_to_id.get('<MASK>', 4)
        actual_max = len(self.vocab)
        masked_ids = []
        min_token = len(self.special_tokens)
        max_token = max(min_token, actual_max - 1)

        for token_id in token_ids:
            if random.random() < mask_prob:
                rand = random.random()
                if rand < 0.8:
                    masked_ids.append(mask_token_id)
                elif rand < 0.9 and max_token > min_token:
                    masked_ids.append(random.randint(min_token, max_token))
                else:
                    masked_ids.append(token_id)
            else:
                masked_ids.append(token_id)
        return masked_ids

    def save_vocab(self, file_path: str):
        """Menyimpan kosakata dan aturan penggabungan ke berkas JSON."""
        vocab_data = {
            'vocab': self.token_to_id,
            'merges': [(p[0], p[1]) for p in self.merges],
            'special_tokens': self.special_tokens,
            'max_tokens': self.max_tokens
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Kosakata BPE berhasil disimpan ke: {file_path}")

    def load_vocab(self, file_path: str):
        """Memuat kosakata dan aturan penggabungan dari berkas JSON."""
        with open(file_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        self.token_to_id = vocab_data['vocab']
        self.merges = [tuple(m) for m in vocab_data['merges']]
        self.special_tokens = vocab_data['special_tokens']
        self.max_tokens = vocab_data['max_tokens']

        self.vocab = self.token_to_id.copy()
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        self.is_trained = True
        logger.info(f"Kosakata BPE dimuat dari {file_path} (Ukuran: {len(self.vocab):,})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    sample_text = "Halo nama saya Nvoin. Hari ini saya membuat model GPT dengan PyTorch. Machine learning sangat menarik." * 20
    bpe = BPE(max_tokens=500)
    bpe.fit(sample_text)
    encoded = bpe.encode("Halo nama saya Nvoin")
    decoded = bpe.decode(encoded)
    print(f"[INFO] Hasil Uji Encode: {encoded}")
    print(f"[INFO] Hasil Uji Decode: '{decoded}'")