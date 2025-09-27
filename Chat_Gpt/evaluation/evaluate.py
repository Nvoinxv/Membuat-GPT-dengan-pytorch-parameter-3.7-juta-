import torch
import numpy as np
from Chat_Gpt.models.transfomers import membuat_gpt
from Chat_Gpt.utils.bpe import BPE

# Konfigurasi (harus sama dengan train.py)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 20000
D_MODEL = 128
SEQ_LEN = 128
N_LAYERS = 6
N_HEADS = 8
D_FF = 512
DROPOUT = 0.15

# Path ke checkpoint dan data
CHECKPOINT_PATH = r"D:\Chatgpt_Pytorch\best_balanced_model.pt"
DATA_PATH = r"D:\Chatgpt_Pytorch\Chat_Gpt\data\input.txt"

def load_model_and_tokenizer():
    """Load model dan tokenizer dari checkpoint"""
    print(f"🔄 Loading model dari: {CHECKPOINT_PATH}")
    
    # Inisialisasi model dengan arsitektur yang sama
    model = membuat_gpt(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        max_seq_len=SEQ_LEN,
        dropout=DROPOUT,
    ).to(DEVICE)
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'Unknown')
        print(f"✅ Model loaded dari epoch: {epoch}")
    else:
        # Jika checkpoint hanya berisi state_dict
        model.load_state_dict(checkpoint)
        print(f"✅ Model di load dengan sangat lancar")
    
    model.eval()
    
    # Load dan train tokenizer BPE dengan data yang sama
    print(f"🔤 Loading dan training BPE tokenizer...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text_data = f.read()
    
    # Batasi data sama seperti training
    if len(text_data) > 1000000:
        text_data = text_data[:1000000]
    
    bpe = BPE(max_tokens=VOCAB_SIZE)
    bpe.fit(text_data)
    
    print(f"✅ BPE tokenizer ready! Vocab size: {len(bpe.vocab)}")
    
    return model, bpe

def generate_text(model, bpe, prompt, max_length=100, temperature=0.8, top_k=50):
    """Generate text dari prompt menggunakan model"""
    model.eval()
    
    # Tokenize prompt
    tokens = bpe.encode(prompt)
    tokens = [max(0, min(t, VOCAB_SIZE - 1)) for t in tokens]  # Clamp tokens
    
    # Convert ke tensor
    input_ids = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
    
    generated_tokens = tokens.copy()
    
    print(f"\n📝 Generating text...")
    print(f"Prompt: '{prompt}'")
    print(f"{'='*60}")
    
    with torch.no_grad():
        for _ in range(max_length):
            # Ambil context window terakhir
            if len(generated_tokens) > SEQ_LEN:
                context = generated_tokens[-SEQ_LEN:]
            else:
                context = generated_tokens
            
            # Convert ke tensor
            x = torch.tensor([context], dtype=torch.long).to(DEVICE)
            x = torch.clamp(x, 0, VOCAB_SIZE - 1)
            
            # Forward pass
            logits = model(x)
            
            # Ambil logits untuk token terakhir
            logits = logits[0, -1, :] / temperature
            
            # Apply top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(0, top_k_indices, top_k_logits)
            
            # Softmax dan sampling
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Clamp token
            next_token = max(0, min(next_token, VOCAB_SIZE - 1))
            
            generated_tokens.append(next_token)
            
            # Stop jika generate token tertentu (opsional)
            # if next_token == bpe.vocab.get('<EOS>', -1):
            #     break
    
    # Decode hasil
    generated_text = bpe.decode(generated_tokens)
    
    return generated_text

def test_multiple_samples():
    """Test model dengan beberapa sample prompt"""
    
    # Load model dan tokenizer
    model, bpe = load_model_and_tokenizer()
    
    # Daftar prompt untuk testing
    test_prompts = [
        "First Citizen:",
        "MARCIUS:",
        "First Officer:",
        "Second Officer:",
        "CORIOLANUS:"
    ]
    
    print(f"\n{'='*60}")
    print(f"🎯 EVALUASI MODEL - TESTING PREDIKSI")
    print(f"{'='*60}\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'#'*60}")
        print(f"TEST SAMPLE {i}/{len(test_prompts)}")
        print(f"{'#'*60}")
        
        # Generate dengan parameter berbeda
        generated = generate_text(
            model=model,
            bpe=bpe,
            prompt=prompt,
            max_length=100,      
            temperature=0.6,     
            top_k=40           
        )
        
        print(f"\n📤 HASIL PREDIKSI:")
        print(f"{'-'*60}")
        print(generated)
        print(f"{'-'*60}\n")
    
    print(f"\n{'='*60}")
    print(f"✅ EVALUASI SELESAI!")
    print(f"{'='*60}")

def interactive_mode():
    """Mode interaktif untuk testing custom prompt"""
    
    model, bpe = load_model_and_tokenizer()
    
    print(f"\n{'='*60}")
    print(f"🎮 MODE INTERAKTIF - CUSTOM PROMPT")
    print(f"{'='*60}")
    print(f"Ketik 'exit' atau 'quit' untuk keluar\n")
    
    while True:
        try:
            prompt = input("\n💬 Masukkan prompt Anda: ")
            
            if prompt.lower() in ['exit', 'quit', 'keluar']:
                print("👋 Terima kasih! Sampai jumpa!")
                break
            
            if not prompt.strip():
                print("⚠️ Prompt tidak boleh kosong!")
                continue
            
            # Generate text
            generated = generate_text(
                model=model,
                bpe=bpe,
                prompt=prompt,
                max_length=100,
                temperature=0.8,
                top_k=50
            )
            
            print(f"\n📤 HASIL:")
            print(f"{'-'*60}")
            print(generated)
            print(f"{'-'*60}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Program dihentikan. Sampai jumpa!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def calculate_perplexity(model, bpe, text_sample):
    """Hitung perplexity model pada sample text"""
    model.eval()
    
    # Tokenize
    tokens = bpe.encode(text_sample)
    tokens = [max(0, min(t, VOCAB_SIZE - 1)) for t in tokens]
    
    if len(tokens) < 2:
        print("⚠️ Sample text terlalu pendek untuk perplexity")
        return None
    
    # Calculate loss
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for i in range(1, len(tokens)):
            # Context
            start = max(0, i - SEQ_LEN + 1)
            context = tokens[start:i]
            target = tokens[i]
            
            # Convert ke tensor
            x = torch.tensor([context], dtype=torch.long).to(DEVICE)
            y = torch.tensor([target], dtype=torch.long).to(DEVICE)
            
            x = torch.clamp(x, 0, VOCAB_SIZE - 1)
            y = torch.clamp(y, 0, VOCAB_SIZE - 1)
            
            # Forward pass
            logits = model(x)
            
            # Loss untuk token terakhir
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(logits[0, -1:, :], y)
            
            total_loss += loss.item()
            count += 1
    
    avg_loss = total_loss / count
    perplexity = np.exp(avg_loss)
    
    return perplexity

def run_perplexity_test():
    """Test perplexity pada sample dari dataset"""
    
    model, bpe = load_model_and_tokenizer()
    
    # Load sample text
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Ambil sample random
    sample_length = 500
    start_idx = np.random.randint(0, max(1, len(text) - sample_length))
    sample_text = text[start_idx:start_idx + sample_length]
    
    print(f"\n{'='*60}")
    print(f"📊 PERPLEXITY TEST")
    print(f"{'='*60}")
    print(f"\nSample text:")
    print(f"{'-'*60}")
    print(sample_text[:200] + "...")
    print(f"{'-'*60}")
    
    perplexity = calculate_perplexity(model, bpe, sample_text)
    
    if perplexity:
        print(f"\n📈 Perplexity: {perplexity:.2f}")
        print(f"   Lower is better! (<100 = good, <50 = excellent)")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"🚀 EVALUASI MODEL GPT")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"\nPilih mode:")
    print(f"1. Test dengan sample prompts otomatis")
    print(f"2. Mode interaktif (custom prompt)")
    print(f"3. Test perplexity")
    print(f"4. Jalankan semua test")
    
    try:
        choice = input("\nPilihan Anda (1-4): ").strip()
        
        if choice == "1":
            test_multiple_samples()
        elif choice == "2":
            interactive_mode()
        elif choice == "3":
            run_perplexity_test()
        elif choice == "4":
            print("\n🔄 Menjalankan semua test...\n")
            test_multiple_samples()
            run_perplexity_test()
            print("\n🎮 Melanjutkan ke mode interaktif...")
            interactive_mode()
        else:
            print("❌ Pilihan tidak valid! Menjalankan mode default...")
            test_multiple_samples()
            
    except KeyboardInterrupt:
        print("\n\n👋 Program dihentikan. Sampai jumpa!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()