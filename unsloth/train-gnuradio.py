from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# --- CONFIGURATION ---
max_seq_length = 2048 # Supports up to 8192 if you have a massive GPU
dtype = None # Auto-detects (Float16 or Bfloat16)
load_in_4bit = True # CRITICAL for consumer GPUs (reduces VRAM usage)

# 1. Load the Base Model
# We use the 4-bit version of DeepSeek Coder 6.7B to fit in memory
model, tokenizer = FastLanguageModel.from_pretrained(
    #model_name = "unsloth/deepseek-coder-6.7b-bnb-4bit",
    model_name = "deepseek-ai/deepseek-coder-6.7b-instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 2. Add LoRA Adapters (The "Fine-Tuning" part)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank (higher = smarter but slower/heavier). 16 is standard.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",   
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False, 
    loftq_config = None, 
)

# 3. Load & Format the Dataset
dataset = load_dataset("json", data_files="gnuradio-train.jsonl", split="train")

# We need to tell the model how to read your Alpaca format
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# 4. Train the Model
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can speed up training for short sequences
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        max_steps = 60, # <-- START SMALL! Increase to 300+ for real training
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        save_steps = 10
    ),
)

print("🚀 Starting Training...")
trainer.train()

# 5. Export for Ollama (GGUF)
print("💾 Saving to GGUF format...")
model.save_pretrained_gguf("model_gnuradio", tokenizer, quantization_method = "q4_k_m")
print("✅ Done! You can now import 'model_gnuradio-unsloth.Q4_K_M.gguf' into Ollama.")
