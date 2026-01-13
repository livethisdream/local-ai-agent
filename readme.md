# 📚 Guide: Building a Local, Private SDK Expert

This guide documents the process of scraping a local SDK (e.g., GNU Radio), fine-tuning a DeepSeek-Coder model on consumer hardware (WSL/NVIDIA GPU), and deploying it as a private chat agent. There are two ways to do this:

1) Build a fine-tuned model using unsloth and possibly add a RAG pipeline. This is captured in the `unsloth/` folder. This is resource intensive and training takes hours.

2) Build a RAG-only pipeline. Generation of the RAG database is much faster. This is in the `rag-only/` folder. 

# Using Unsloth to fine-tune a database and adding a RAG pipeline
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

## Observations
The fine-tuned model, `gnuradio-model`, works relatively well by itself. Adding the RAG information seems to produce a lot of noise in the answers. This led me to creating the RAG-only pipeline. 

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
* [x] **RAG Pipeline:** Build a vector database to allow the agent to look up *new* datasheets it wasn't trained on (Retrieval Augmented Generation).

# RAG Pipeline: Local Ingestion to Cloud Deployment
This guide documents the process of creating a Retrieval-Augmented Generation (RAG) agent for the GNURadio codebase. The resulting chat agent can run locally using streamlit or can be deployed to colab for sharing. 

## The Architecture:

- Ingestion (Local): We scrape the source code and PDF datasheets on a secure local machine and convert them into a Vector Database (rag_db).

- Storage (Cloud): We upload the database to Google Drive.

- Inference (Colab): We use a Google Colab GPU to run the LLM (DeepSeek-Coder) and host a web interface.

- Access (Tunnel): We use Cloudflare Tunnels to provide a temporary public URL for the team.

## Phase 1: Local Knowledge Creation
Run this on your local Linux/WSL machine where the source code lives.

1. Environment Setup: Create a lightweight Python environment for the ingestion tools.

```
#!Bash

python3 -m venv rag-env
source rag-env/bin/activate
pip install chromadb sentence-transformers pdfplumber
```

2. The Ingestion Script (ingest_data.py)
This script scans your target directory, chunks the code/PDFs, and saves them into a local ChromaDB folder.

**Key Settings:**

- `CHUNK_SIZE = 1500`: Large enough to capture full C++ functions.

- `embedder`: Uses all-MiniLM-L6-v2 (fast, efficient).

**To run the ingestion:**

```
# Update the REPO_PATH in the script to point to your target code
python3 ingest_data.py
```

**Output:** You will see a folder named rag_db/ appear. This contains the "brain" of your project.

3. **Pack for the Cloud:** Since Google Colab cannot access your hard drive, we zip the database for upload.

```
#!Bash

zip -r rag_db.zip rag_db/
```

## Phase 2a: Running Locally

Run `python3 rag_app.py` in the terminal and view the chat bot locally at http://localhost:8501/  

## Phase 2b: Cloud Hosting (Google Colab)

This is a bit more complicated. Do this in your browser in a Colab notebook, using Google Drive as your RAG db storage.  

1. Google Drive Setup
- Create a folder in your Google Drive named RAG_Project.

- Upload rag_db.zip into this folder.

2. The Deployment Notebook
- Open a new Google Colab notebook and ensure the Runtime is set to T4 GPU (Runtime > Change runtime type).

- Copy the following "Master Deployment Block" into a single code cell. This script handles:

  - Installing Ollama & Streamlit.

  - Applying the pysqlite3 fix (required for Colab compatibility).

  - Connecting to your Google Drive.

  - Launching the public tunnel.

```
#! Python

# ==========================================
# 🚀 MASTER RAG LAUNCHER
# ==========================================
import subprocess, time, sys, os, zipfile
from google.colab import drive

# --- 1. SETUP & CLEANUP ---
print("🛑 Cleaning old processes...")
!pkill -9 ollama
!pkill -9 streamlit
!pkill -9 cloudflared
!rm -rf nohup.out

print("📂 Mounting Google Drive...")
drive.mount('/content/drive')

# --- 2. RESTORE DATABASE ---
ZIP_PATH = "/content/drive/MyDrive/RAG_Project/rag_db.zip"
if os.path.exists(ZIP_PATH):
    print("📦 Unzipping Knowledge Base...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall("/content")
else:
    print("❌ ERROR: rag_db.zip not found in Drive!")

# --- 3. INSTALL DEPENDENCIES ---
print("⬇️ Installing System Tools...")
!pip install -q pysqlite3-binary chromadb sentence-transformers ollama
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64
!curl -fsSL https://ollama.com/install.sh | sh

# --- 4. START AI ENGINE ---
print("🤖 Starting Ollama & Downloading Model...")
subprocess.Popen("ollama serve", shell=True)
time.sleep(5) # Allow warm-up
!ollama pull deepseek-coder:6.7b

# --- 5. GENERATE APP (With SQLite Fix) ---
app_code = """
import pysqlite3
import sys
sys.modules['sqlite3'] = pysqlite3 # CRITICAL FIX

import streamlit as st
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
import os

DB_PATH = "/content/rag_db"
COLLECTION_NAME = "rag_knowledge"
MODEL_NAME = "deepseek-coder:6.7b"

st.set_page_config(page_title="Team RAG", layout="wide")
st.title("🚀 Shared RAG Agent")

@st.cache_resource
def load_resources():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return collection, embedder

collection, embedder = load_resources()

if "messages" not in st.session_state: st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if user_input := st.chat_input("Ask about the codebase..."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    # RAG Retrieval
    results = collection.query(query_embeddings=embedder.encode([user_input]).tolist(), n_results=3)
    context = ""
    sources = set()
    if results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            src = os.path.basename(results['metadatas'][0][i]['source'])
            context += f"-- SOURCE: {src} --\\n{doc}\\n\\n"
            sources.add(src)

    # Generation
    with st.chat_message("assistant"):
        prompt = f"Context:\\n{context}\\n\\nQuestion: {user_input}"
        resp = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}])
        full_resp = resp['message']['content']
        if sources: full_resp += "\\n\\n**📚 Sources:** " + ", ".join(sources)
        st.markdown(full_resp)
        st.session_state["messages"].append({"role": "assistant", "content": full_resp})
"""

with open("rag_app.py", "w") as f:
    f.write(app_code)

# --- 6. LAUNCH ---
print("🚀 Launching App...")
!streamlit run rag_app.py --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false > /dev/null 2>&1 &
time.sleep(5)
print("🔗 ACCESS LINK (Click below):")
!./cloudflared-linux-amd64 tunnel --url http://localhost:8501
```

## Troubleshooting & Maintenance

1. **Google Colab runtimes are ephemeral:** The URL will stop working if the browser tab is closed for too long or if the session exceeds 12 hours. The Fix is to simply open the Notebook, go to Runtime > Run All, and generate a new link for the team.

2. **"503 Tunnel Unavailable":** This means the Streamlit app crashed, usually due to a library mismatch. The fix is to ensure the pysqlite3-binary fix is included at the very top of the rag_app.py script (as shown in the master block above).

3. **Updating Knowledge:** If the source code changes significantly, you need to update the RAG database. The process is:

- Re-run `ingest_data.py` on your local machine.

- Re-zip `rag_db`.

- Replace the `rag_db.zip` file in Google Drive.

- Restart the Colab notebook.
