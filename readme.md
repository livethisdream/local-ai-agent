# 📚 Guide: Building a Local, Private SDK Expert

This guide documents the process of scraping a local SDK (e.g., GNU Radio), fine-tuning a DeepSeek-Coder model on consumer hardware (WSL/NVIDIA GPU), and deploying it as a private chat agent.

---

## 🏗️ Phase 1: Environment Setup (The Hard Part)
Getting CUDA, PyTorch, and Unsloth to play nicely on WSL is tricky. We utilize a reproducible setup script to handle version pinning (PyTorch 2.5.0 + Unsloth Stable) to avoid dependency conflicts.

* **Artifact:** `setup_env.sh` (or `build_agent.sh` for the all-in-one approach)
* **Key Action:** Creates a Python virtual environment (`ai-env`) and installs the "Holy Trinity":
    * **PyTorch 2.5.1** (Pinned for compatibility)
    * **Unsloth** (Stable PyPI release)
    * **xformers & trl** (Version matched to prevent downgrades)

---

## 🕷️ Phase 2: Data Collection
We turn raw source code and PDF documentation into a format the AI understands (JSONL).

* **Artifact:** `sdk_scraper.py`
* **Process:**
    1.  Recursive walk through the target folder (e.g., `~/gnuradio`).
    2.  Extract text from code (`.c`, `.cpp`, `.py`) and docs (`.dox`, `.rst`).
    3.  **PDF Parsing:** Uses `pdfplumber` to extract text from datasheets.
    4.  Formatting: Converts data into "Alpaca" instruction pairs (`instruction`, `input`, `output`).
* **Output:** `training_data.jsonl`

---

## 🏋️ Phase 3: The Training (Fine-Tuning)
We use **Unsloth** to fine-tune the `deepseek-coder-6.7b` model using LoRA (Low-Rank Adaptation) and 4-bit quantization. This allows training a 7B model on a single consumer GPU with <8GB VRAM.

* **Artifact:** `train_on_data.py`
* **Key Configs:**
    * `max_steps`: 60 (Test) / 300+ (Production)
    * `load_in_4bit`: True (Crucial for memory savings)
    * `batch_size`: 1 (Prevents OOM on smaller cards)
    * `gradient_accumulation`: 8 (Compensates for low batch size)
* **Output:** A checkpoint folder (e.g., `outputs/checkpoint-60`) containing the "adapter" weights.

---

## 💾 Phase 4: Export & Conversion
We must fuse the new "adapter" weights with the original base model and convert the result to GGUF format for Ollama.

* **Artifact:** `finish_export.py`
* **Critical Fix:** The conversion tool (`llama.cpp`) often fails to compile automatically via pip.
    * **Solution:** We manually compile it using `cmake` inside the `sdk-scraper/llama.cpp` directory to generate the `llama-quantize` tool.
* **Output:** `custom-model.gguf`

---

## 🐳 Phase 5: Deployment
We package the GGUF file into an Ollama model and (optionally) build a UI.

### 1. Ollama Model
* **Artifact:** `Modelfile`
* **Command:** ```bash
    ollama create custom-coder -f Modelfile
    ```

### 2. Chat Interface (Optional)
* **Artifact:** `chat_ui.py`
* **Description:** A Streamlit web app that connects to the local Ollama instance for a "ChatGPT-like" experience with persistent history.
* **Run:** `streamlit run chat_ui.py`

---

## Phase 6: RAG (Retrieval Augmented Generation)

RAG gives us Short-Term Perception. It allows the agent to "open the book" and read files that:

- Are brand new (released today).

- Were too boring/specific to train on (e.g., a 400-page datassheet).

- Are private to your specific project.

### The RAG Architecture
We are adding a "Vector Database" (ChromaDB) next to your Ollama model.

 - Ingest: We scan your PDFs/Code, chop them into small paragraphs, and save them in the database.

- Retrieve: When you ask "How do I configure a specific device?", we search the database for the 3 most relevant paragraphs.

- Generate: We paste those paragraphs into the prompt: "Context: [Paragraphs]... User Question: How do I configure...?"

#### Step 1: 
tep 1: Install the Librarian (ChromaDB)
We need two new libraries: chromadb (the database) and sentence-transformers (to convert text into search-able numbers).

Run this inside your ai-env:

```
pip install chromadb sentence-transformers
```

#### Step 2: Build the database (build-rag-db.py)

```
python3 build_rag_db.py
```

#### Step 3: Add RAG to your Chat App
This is the updated Streamlit app. It connects to both ChromaDB (to find facts) and Ollama (to write the answer). Adding the following to our chat agent script allows us to ask about topics in the RAG pipeline. 

```
import chromadb
from sentence_transformers import SentenceTransformer
.
.
.
# --- CONFIG ---
DB_PATH = "./rag_db"
COLLECTION_NAME = "gnuradio_knowledge"
.
.
.
# 2. RAG RETRIEVAL STEP
    with st.spinner("🔍 Searching local knowledge base..."):
        # Convert user question to vector
        query_vec = embedder.encode([user_input]).tolist()
        
        # Search DB for top 3 matches
        results = collection.query(query_embeddings=query_vec, n_results=3)
        
        # Extract the text
        context_text = ""
        sources = set()
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                src = results['metadatas'][0][i]['source']
                context_text += f"-- SOURCE: {os.path.basename(src)} --\n{doc}\n\n"
                sources.add(os.path.basename(src))
```


## 🧹 Final Step: The Cleanup Checklist
**Warning:** Only run these steps *after* you have confirmed your `custom-model.gguf` works.

* [ ] **Hugging Face Cache (~15GB):** `rm -rf ~/.cache/huggingface/hub`
* [ ] **Training Checkpoints (~2GB):** `rm -rf outputs/`
* [ ] **WSL Disk Compaction:**
    Deleting files in Linux does not shrink the Windows `.vhdx` file automatically.
    1.  `wsl --shutdown`
    2.  Run `diskpart` -> `select vdisk file="..."` -> `compact vdisk`

---

## 📝 To-Do List
* [x] **Setup:** Environment & Dependencies fixed.
* [x] **Data:** Scraper built for Code & PDFs.
* [x] **Training:** Model fine-tuned on SDK.
* [x] **Deploy:** Running locally in Ollama + Web UI.
* [ ] **RAG Pipeline:** Build a vector database to allow the agent to look up *new* datasheets it wasn't trained on (Retrieval Augmented Generation).