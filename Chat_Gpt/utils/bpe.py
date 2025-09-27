# BPE tokenizer yang dioptimalkan untuk GPT training
# Dengan special tokens dan preprocessing yang lebih baik

from collections import Counter, defaultdict
from tqdm import tqdm
import random
import re

class BPE:
    def __init__(self, max_tokens=200000, special_tokens=None):
        self.max_tokens = max_tokens
        
        # Special tokens yang penting untuk GPT
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
        
        # Initialize special tokens dulu biar gak bentrok sama token biasa
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.token_to_id[token] = i
            self.id_to_token[i] = token
    
    def preprocess_text(self, text):
        """Preprocessing text sebelum tokenization"""
        # Bersihkan dulu sebelum di tokenisasi nya 
        # lewat normalisasi spasi
        text = re.sub(r'\s+', ' ', text)
        
        # Tangani tanda baca umum dengan lebih hati-hati
        # biar gak ngerusak konteks kalimat
        text = re.sub(r'([.!?;:])', r' \1 ', text)
        text = re.sub(r'([,])', r' \1', text)
        
        # Tangani kutipan dan bracket
        text = re.sub(r'(["\'()])', r' \1 ', text)
        
        # Pisahin angka dari huruf biar gampang dipelajari
        text = re.sub(r'(\d+)', r' \1 ', text)
        
        # Bersihkan spasi ekstra yang kebentuk dari proses di atas
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def mendapatkan_kata_tokens(self, text):
        """Split text menjadi word-level tokens dengan subword boundaries"""
        # Pemisahan berbasis kata dengan mempertahankan batas kata
        kata = text.split()
        token_kata = []
        
        for word in kata:
            # Tambahkan penanda batas kata biar tau mana yang akhiran kata
            # pake karakter khusus yang gak bakal muncul di text biasa
            chars = [c for c in word] + ['</w>']  
            token_kata.extend(chars)
            
        return token_kata
    
    def dapatkan_statistik(self, tokens):
        """Hitung frekuensi pasangan token yang bersebelahan"""
        # Pake defaultdict biar gak perlu cek key exist atau engga
        pairs = defaultdict(int)
        
        # Loop cuma sampe len-1 biar gak index error
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pairs[pair] += 1
            
        return pairs
    
    def Gabung_Token(self, tokens, pair):
        """Gabungkan Token nya setelah melewati fungsi statistik"""
        new_tokens = []
        i = 0
        
        # Manual loop biar bisa skip 2 index sekaligus kalo nemu pair yang di merge
        while i < len(tokens):
            # Cek apakah current token dan next token adalah pair yang mau di merge
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                # Merge pair dengan gabungin string nya
                merged = tokens[i] + tokens[i+1]
                new_tokens.append(merged)
                i += 2  # Skip 2 token karena udah di merge
            else:
                new_tokens.append(tokens[i])
                i += 1
                
        return new_tokens
    
    def fit(self, text):
        """Train BPE pada text"""
        print("Starting BPE training...")
        print(f"Target ukuran vocab: {self.max_tokens}")
        
        # Preprocess text biar lebih konsisten
        processed_text = self.preprocess_text(text)
        print(f"Preprocessed {len(text)} -> {len(processed_text)} characters")
        
        # Dapatkan token awal (tingkat karakter dengan batas kata)
        tokens = self.mendapatkan_kata_tokens(processed_text)
        print(f"Inisialisasi Token: {len(tokens)}")
        
        # Bangun kosakata awal dari karakter unik
        unique_chars = sorted(set(tokens))  # Sort biar konsisten
        for char in unique_chars:
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char
        
        initial_vocab_size = len(self.vocab)
        print(f"Inisialisasi ukuran vocab: {initial_vocab_size} (including {len(self.special_tokens)} special tokens)")
        
        # Hitung realistis target vocab berdasarkan ukuran data
        # Rule of thumb: 1-3% dari total tokens bisa jadi vocab yang bagus
        realistic_target = min(self.max_tokens, max(initial_vocab_size + 1000, int(len(tokens) * 0.02)))
        print(f"Target realistis berdasarkan data: {realistic_target}")
        
        # Iterasi BPE untuk memunculkan progress tokenisasi nya
        num_merges = realistic_target - len(self.vocab)
        pbar = tqdm(range(num_merges), desc="BPE merges")
        
        # Adaptive threshold: mulai strict, makin lama makin permisif
        min_freq_threshold = 2
        no_progress_count = 0
        max_no_progress = 500  # Kalo 500 iterasi gak ada progress, turunin threshold
        
        for i in pbar:
            # Mendapatkan pasangan lewat perhitungan statistik
            pairs = self.dapatkan_statistik(tokens)
            
            if not pairs:
                print(f"\nTidak ada pasangan lagi untuk digabung pada iterasi {i}")
                break
            
            # Temukan pasangan yang paling sering muncul
            best_pair = max(pairs, key=pairs.get)
            best_freq = pairs[best_pair]
            
            # Adaptive threshold berdasarkan progress
            current_vocab_ratio = len(self.vocab) / realistic_target
            
            # Makin deket target, makin permisif
            if current_vocab_ratio < 0.5:
                min_freq_threshold = 2  # Awal masih strict
            elif current_vocab_ratio < 0.8:
                min_freq_threshold = 1  # Tengah mulai longgar
            else:
                min_freq_threshold = 1  # Akhir accept semua
            
            # Check apakah worth it di-merge
            if best_freq < min_freq_threshold:
                no_progress_count += 1
                
                # Kalo udah lama stuck, turunin threshold
                if no_progress_count >= max_no_progress:
                    if min_freq_threshold > 1:
                        min_freq_threshold = 1
                        no_progress_count = 0
                        print(f"\nTurunin threshold ke {min_freq_threshold}")
                    else:
                        # Udah mentok, stop aja
                        print(f"\nBerhenti: gak ada pair yang worth it lagi (freq={best_freq})")
                        break
                continue
            else:
                no_progress_count = 0  # Reset counter
            
            # Penggabungan Token
            # Yang kita sudah lakukan setelah mendapatkan
            # Statistik nya
            tokens = self.Gabung_Token(tokens, best_pair)
            
            # Tambahkan token gabungan ke vocabulary atau kosakata
            Penggabungan_token = best_pair[0] + best_pair[1]
            if Penggabungan_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[Penggabungan_token] = idx
                self.token_to_id[Penggabungan_token] = idx
                self.id_to_token[idx] = Penggabungan_token
            
            # Store merge rule buat dipake waktu encode
            self.merges.append(best_pair)
            
            # Update progress bar biar keliatan info penting
            pbar.set_postfix({
                'vocab': len(self.vocab),
                'freq': best_freq,
                'tokens': len(tokens),
                'ratio': f'{current_vocab_ratio:.1%}'
            })
            
            # Stop kalo udah nyampe realistic target
            if len(self.vocab) >= realistic_target:
                print(f"\n✅ Mencapai target realistis: {len(self.vocab)}")
                break
        
        pbar.close()
        
        # Convert final tokens to IDs dengan handling yang lebih baik
        self.token_ids = []
        actual_vocab_size = len(self.vocab)
        
        for token in tokens:
            if token in self.token_to_id:
                token_id = self.token_to_id[token]
                # Validasi range
                if 0 <= token_id < actual_vocab_size:
                    self.token_ids.append(token_id)
                else:
                    self.token_ids.append(self.token_to_id['<UNK>'])
            else:
                # Token gak ada di vocab, pake UNK
                self.token_ids.append(self.token_to_id['<UNK>'])
        
        # Update max_tokens jadi actual vocab size
        self.max_tokens = actual_vocab_size
        self.is_trained = True
        
        print(f"\nPelatihan BPE sudah berhasil!")
        print(f"Akhir dari ukuran vocab: {len(self.vocab)}")
        print(f"Angka dari penggabungan: {len(self.merges)}")
        print(f"Jumlah Token Akhir: {len(self.token_ids)}")
        if len(self.token_ids) > 0:
            print(f"Kompresi Rasio: {len(processed_text)/len(self.token_ids):.2f}x")
        
        # Statistik vocab usage
        unique_in_data = len(set(self.token_ids))
        print(f"Token unik dalam data: {unique_in_data}/{len(self.vocab)} ({unique_in_data/len(self.vocab)*100:.1f}%)")
        
    def encode(self, text):
        """Encode text menjadi token IDs"""
        if not self.is_trained:
            raise ValueError("BPE belum dilatih! Panggil fit() terlebih dahulu.")
        
        # Preprocess dulu biar konsisten sama waktu training
        processed_text = self.preprocess_text(text)
        tokens = self.mendapatkan_kata_tokens(processed_text)
        
        # Menerapkan penggabungan yang berlatih secara berurutan
        # harus sesuai urutan merge waktu training biar hasilnya sama
        for merge in self.merges:
            tokens = self.Gabung_Token(tokens, merge)
        
        # Convert ke IDs dengan safe handling
        token_ids = []
        unk_id = self.token_to_id.get('<UNK>', 1)
        actual_max = len(self.vocab)
        
        for token in tokens:
            if token in self.token_to_id:
                token_id = self.token_to_id[token]
                # Pastikan ID token berada dalam rentang yang valid
                if 0 <= token_id < actual_max:
                    token_ids.append(token_id)
                else:
                    token_ids.append(unk_id)
            else:
                token_ids.append(unk_id)
        
        return token_ids
    
    def decode(self, token_ids):
        """Decode token IDs kembali ke text"""
        if not self.is_trained:
            raise ValueError("BPE not trained yet!")
        
        tokens = []
        unk_token = '<UNK>'
        actual_max = len(self.vocab)
        
        for token_id in token_ids:
            # Tangani ID token yang tidak valid dengan baik
            if 0 <= token_id < actual_max and token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                # Fallback ke UNK kalo token_id invalid
                tokens.append(unk_token)
        
        # Join tokens dan clean up marker batas kata
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')  # Ganti marker batas kata jadi spasi
        
        # Bersihkan spasi berlebih yang kebentuk
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Clean up spasi sebelum tanda baca biar lebih natural
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([(\[])\s+', r'\1', text)
        text = re.sub(r'\s+([)\]])', r'\1', text)
        
        return text
    
    def get_vocab_size(self):
        """Return vocabulary size"""
        return len(self.vocab)
    
    def random_mask(self, token_ids, mask_prob=0.15):
        """Apply random masking dengan safe range validation"""
        mask_token_id = self.token_to_id.get('<MASK>', 4)
        actual_max = len(self.vocab)
        
        # Pastiin mask token id valid
        if mask_token_id >= actual_max:
            mask_token_id = self.token_to_id.get('<UNK>', 1)
        
        masked_ids = []
        min_token = len(self.special_tokens)  # Skip special tokens waktu random
        max_token = actual_max - 1
        
        for token_id in token_ids:
            # Ensure input token is valid dulu
            if token_id >= actual_max:
                token_id = max_token
            elif token_id < 0:
                token_id = 0
            
            # Tentuin apakah token ini mau di mask atau engga
            if random.random() < mask_prob:
                # Strategi masking: 80% mask, 10% random, 10% unchanged
                # ini standard nya BERT biar model lebih robust
                rand = random.random()
                
                if rand < 0.8:
                    # 80% chance: replace dengan MASK token
                    masked_ids.append(mask_token_id)
                elif rand < 0.9:
                    # 10% chance: replace dengan random token
                    if max_token > min_token:
                        random_id = random.randint(min_token, max_token)
                        masked_ids.append(random_id)
                    else:
                        # Fallback ke mask kalo vocab terlalu kecil
                        masked_ids.append(mask_token_id)
                else:
                    # 10% chance: keep original token
                    masked_ids.append(token_id)
            else:
                # Gak di mask, keep original
                masked_ids.append(token_id)
        
        return masked_ids
    
    def save_vocab(self, file_path):
        """Save vocabulary to file"""
        import json
        
        # Siapkan data yang mau di save
        vocab_data = {
            'vocab': self.token_to_id,
            'merges': [(p[0], p[1]) for p in self.merges],  # Convert tuple ke list biar bisa di serialize
            'special_tokens': self.special_tokens,
            'max_tokens': self.max_tokens
        }
        
        # Save ke JSON dengan encoding UTF-8 biar support semua karakter
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"Kosakata save ke {file_path}")
    
    def load_vocab(self, file_path):
        """Load vocabulary from file"""
        import json
        
        # Load dari JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # Restore semua data vocab
        self.token_to_id = vocab_data['vocab']
        self.merges = [tuple(m) for m in vocab_data['merges']]  # Convert list ke tuple
        self.special_tokens = vocab_data['special_tokens']
        self.max_tokens = vocab_data['max_tokens']
        
        # Membangun kembali pemetaan terbalik
        self.vocab = self.token_to_id.copy()
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.is_trained = True
        
        print(f"Vocabulary loaded from {file_path}")
        print(f"Vocab size: {len(self.vocab)}")

def test_bpe_performance(text_sample, vocab_sizes=[1000, 5000, 10000, 20000]):
    """Test BPE performance dengan different vocab sizes"""
    print("Testing BPE performance...")
    
    for vocab_size in vocab_sizes:
        print(f"\n{'='*50}")
        print(f"Testing vocab size: {vocab_size}")
        print(f"{'='*50}")
        
        # Train BPE dengan vocab size tertentu
        bpe = BPE(max_tokens=vocab_size)
        bpe.fit(text_sample)
        
        # Test encoding/decoding dengan sample kecil
        test_text = text_sample[:1000]  # First 1000 chars
        encoded = bpe.encode(test_text)
        decoded = bpe.decode(encoded)
        
        # Hitung metrics
        compression_ratio = len(test_text) / len(encoded) if len(encoded) > 0 else 0
        actual_vocab = bpe.get_vocab_size()
        vocab_usage = len(set(encoded)) / actual_vocab * 100 if actual_vocab > 0 else 0
        
        print(f"  Kompresi: {len(test_text)}/{len(encoded)} = {compression_ratio:.2f}x")
        print(f"  Utilitas Vocab: {len(set(encoded))}/{actual_vocab} = {vocab_usage:.1f}%")
        
        # Cek akurasi decode (seberapa mirip hasil decode sama original)
        if len(decoded) > 0:
            accuracy = sum(c1 == c2 for c1, c2 in zip(test_text, decoded)) / max(len(test_text), len(decoded)) * 100
            print(f"  Akurasi Decode: {accuracy:.1f}%")

if __name__ == "__main__":
    # Test dengan sample text yang lebih realistis
    sample_text = """Hello my name is Nvoin!, today im make GPT decoder with pytorch!. 
    Machine learning is awesome. I love coding in Python.
    BPE tokenization helps reduce vocabulary size while maintaining semantic meaning.
    """ * 50
    
    print("Testing performa BPE...")
    print(f"Sample panjang teks: {len(sample_text)} characters\n")
    
    # Train BPE
    bpe = BPE(max_tokens=1000)
    bpe.fit(sample_text)
    
    # Test encoding
    test_sentence = "Hello my name is Nvoin"
    encoded = bpe.encode(test_sentence)
    decoded = bpe.decode(encoded)
    
    print(f"\n{'='*50}")
    print("Encoding/Decoding Test")
    print(f"{'='*50}")
    print(f"Original: '{test_sentence}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    print(f"Match: {test_sentence.strip() == decoded.strip()}")
    
    # Test masking
    masked = bpe.random_mask(encoded, mask_prob=0.3)
    masked_decoded = bpe.decode(masked)
    print(f"\nMasked IDs: {masked}")
    print(f"Masked text: '{masked_decoded}'")
    
    print(f"\n{'='*50}")
    print("BPE test berhasil berjalan!")
    print(f"{'='*50}")