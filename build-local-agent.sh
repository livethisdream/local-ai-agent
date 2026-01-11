#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# --- CONFIGURATION ---
VENV_NAME="ai-env"
MODEL_NAME="custom-coder" # Renamed to be generic
GGUF_NAME="custom-model.gguf"
CHECKPOINT_DIR="outputs"
REPO_TO_TRAIN_ON="/home/nrogers/src/gnuradio" # CHANGE THIS to your target folder
BASE_MODEL="deepseek-ai/deepseek-coder-6.7b-instruct"
DATASET_FILE="training_data.jsonl" # Centralized filename variable

echo "======================================================"
echo "🚀 STARTING GENERIC AI BUILD PIPELINE"
echo "   Target: $REPO_TO_TRAIN_ON"
echo "   Base:   $BASE_MODEL"
echo "======================================================"

# ==============================================================================
# PHASE 1: ENVIRONMENT SETUP
# ==============================================================================
echo "🔧 [1/6] Setting up System & Python Environment..."

sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential git wget cmake ccache

if [ ! -d "$VENV_NAME" ]; then
    python3 -m venv $VENV_NAME
fi
source $VENV_NAME/bin/activate

# Force re-install to ensure clean state
pip install --upgrade pip
echo "📦 Installing PyTorch 2.5.1..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo "🦥 Installing Unsloth & Utils..."
pip install unsloth
pip install --no-deps "xformers<0.0.29" "trl<0.9.0" peft accelerate bitsandbytes pdfplumber huggingface_hub streamlit

if pip show torchao > /dev/null 2>&1; then
    echo "🧹 Removing conflicting 'torchao'..."
    pip uninstall -y torchao
fi

# ==============================================================================
# PHASE 2: GENERATE PYTHON SCRIPTS
# ==============================================================================
echo "📝 [2/6] Writing Python automation scripts..."

# --- Script 1: The Scraper ---
# NOTICE: We use EOF (no quotes) so bash variables expand.
cat << EOF > sdk_scraper.py
import os, glob, json
import pdfplumber

DATA_FILE = "$DATASET_FILE"
REPO_DIR = "$REPO_TO_TRAIN_ON"

# (Mock scraper logic - replace with your real recursive scraper!)
if not os.path.exists(DATA_FILE):
    print(f"⚠️ No dataset found. Creating a sample dataset from {REPO_DIR}...")
    # In a real scenario, you would walk REPO_DIR here.
    sample_data = [
        {
            "instruction": "How do I create a hello world program?",
            "input": "",
            "output": "Start with the basics."
        }
    ]
    with open(DATA_FILE, "w") as f:
        for entry in sample_data:
            json.dump(entry, f)
            f.write("\n")
    print(f"✅ Created sample {DATA_FILE}")
else:
    print(f"✅ Found existing {DATA_FILE}")
EOF

# --- Script 2: The Trainer ---
cat << EOF > train_on_data.py
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

# We inject the bash variable into the python string
model_name = "$BASE_MODEL"
max_seq_length = 2048

print(f"⬇️ Loading Model: {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
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

print("📚 Loading Dataset...")
# Use the centralized filename
dataset = load_dataset("json", data_files = "$DATASET_FILE", split = "train")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "output",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        save_steps = 20,
    ),
)

print("🔥 Starting Training...")
trainer.train()
print("✅ Training Complete.")
EOF

# --- Script 3: The Exporter ---
cat << EOF > finish_export.py
from unsloth import FastLanguageModel
import os
import glob

# Find latest checkpoint
checkpoints = glob.glob("outputs/checkpoint-*")
if not checkpoints:
    raise FileNotFoundError("No checkpoints found in outputs/")
latest_checkpoint = max(checkpoints, key=os.path.getctime)
print(f"📂 Using latest checkpoint: {latest_checkpoint}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = latest_checkpoint,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

print("💾 Converting to GGUF...")
# This will usually create 'model_sdk-unsloth.Q4_K_M.gguf'
model.save_pretrained_gguf("model_sdk", tokenizer, quantization_method = "q4_k_m")
print("✅ Export Complete.")
EOF

# ==============================================================================
# PHASE 3: EXECUTION LOOP
# ==============================================================================

echo "🕷️ [3/6] running Scraper..."
python3 sdk_scraper.py

echo "🏋️ [4/6] Running Training..."
if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A $CHECKPOINT_DIR)" ]; then
    echo "   (Outputs exist, continuing...)"
fi
# Fixed filename here:
python3 train_on_data.py

# ==============================================================================
# PHASE 4: COMPILATION (LLAMA.CPP)
# ==============================================================================
echo "🔨 [5/6] Building Converter Tools..."

LLAMA_DIR="sdk-scraper/llama.cpp" # Note: This dir name depends on where unsloth clones it
if [ -d "$LLAMA_DIR" ]; then
    cd $LLAMA_DIR
    cmake -B build
    cmake --build build --config Release -j
    cp build/bin/llama-quantize .
    cd ../..
fi

echo "💾 Running Export..."
python3 finish_export.py

# Smart Move: Find whatever GGUF was just created (ignoring the base model cache if present)
# We look for the file containing "model_sdk" since that's what we named it in finish_export.py
echo "🔄 Renaming output..."
FOUND_GGUF=$(find . -maxdepth 2 -type f -name "*model_sdk*.gguf" | head -n 1)

if [ -f "$FOUND_GGUF" ]; then
    mv "$FOUND_GGUF" $GGUF_NAME
    echo "✅ Created $GGUF_NAME"
else
    echo "⚠️  Could not find generated GGUF. Check directory listing:"
    ls -lh *.gguf
fi

# ==============================================================================
# PHASE 5: OLLAMA DEPLOYMENT
# ==============================================================================
echo "🐳 [6/6] Deploying to Ollama..."

cat << EOF > Modelfile
FROM ./$GGUF_NAME
SYSTEM """You are an expert developer trained on $REPO_TO_TRAIN_ON."""
EOF

ollama create $MODEL_NAME -f Modelfile

echo "======================================================"
echo "✅ SUCCESS! Agent '$MODEL_NAME' is ready."
echo "   Run: ollama run $MODEL_NAME"
echo "======================================================"