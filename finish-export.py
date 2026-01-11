from unsloth import FastLanguageModel

# --- POINT THIS TO YOUR CHECKPOINT ---
# Check your 'outputs' folder. If you have 'checkpoint-60', use that.
checkpoint_path = "outputs/checkpoint-60" 

print(f"📂 Loading trained model from {checkpoint_path}...")

# 1. Load the Model & Adapters
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = checkpoint_path,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 2. Convert to GGUF (This will now work because you installed the tools!)
print("💾 Saving to GGUF format...")
model.save_pretrained_gguf("model_gnuradio", tokenizer, quantization_method = "q4_k_m")
print("✅ Done! check for 'model_gnuradio-unsloth.Q4_K_M.gguf'")
