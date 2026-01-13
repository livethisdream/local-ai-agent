#!/bin/bash
set -e  # Exit on error

# --- CONFIGURATION ---
VENV_NAME=".rag-env"                 # Different env to keep things clean
REPO_TO_SCRAPE="/home/nrogers/src/gnuradio" # CHANGE THIS to your target
DB_PATH="./rag_db"
# We use the generic base model to compare against your fine-tuned one
BASE_MODEL="deepseek-coder:6.7b"    
APP_FILE="rag_app.py"
INGEST_FILE="ingest_data.py"

echo "======================================================"
echo "📚 STARTING PURE RAG PIPELINE SETUP"
echo "   Target: $REPO_TO_SCRAPE"
echo "   Model:  $BASE_MODEL"
echo "======================================================"

# ==============================================================================
# PHASE 1: LIGHTWEIGHT ENVIRONMENT SETUP
# ==============================================================================
echo "🔧 [1/4] Setting up Python Environment..."

sudo apt-get update
sudo apt-get install -y python3-venv build-essential

if [ ! -d "$VENV_NAME" ]; then
    python3 -m venv $VENV_NAME
fi
source $VENV_NAME/bin/activate

# We don't need the heavy ML libraries (torch/unsloth) for this!
# Just the database and UI tools.
pip install --upgrade pip
echo "📦 Installing ChromaDB, Streamlit, and PDF tools..."
pip install chromadb sentence-transformers streamlit ollama pdfplumber

# ==============================================================================
# PHASE 2: MODEL SETUP
# ==============================================================================
echo "🐳 [2/4] Pulling Base Model from Ollama..."
# We ensure the base model exists locally
ollama pull $BASE_MODEL

# ==============================================================================
# PHASE 3: GENERATE PYTHON SCRIPTS
# ==============================================================================
echo "📝 [3/4] Generating Python Scripts..."

# --- Script 1: The Ingestor (Scraper + Vector DB Builder) ---
cat << EOF > $INGEST_FILE
import os
import glob
import chromadb
import pdfplumber
from sentence_transformers import SentenceTransformer

REPO_PATH = "$REPO_TO_SCRAPE"
DB_PATH = "$DB_PATH"
COLLECTION_NAME = "rag_knowledge"

# Config for chunks
CHUNK_SIZE = 1500  # Larger chunks for better code context
OVERLAP = 200

print(f"📥 Loading Embedder (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print(f"📂 Accessing Database at {DB_PATH}...")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
# Delete old collection if re-running to avoid duplicates
try:
    chroma_client.delete_collection(COLLECTION_NAME)
except:
    pass
collection = chroma_client.create_collection(name=COLLECTION_NAME)

def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def process_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    text = ""
    try:
        if ext == ".pdf":
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    pt = page.extract_text()
                    if pt: text += pt + "\n"
        elif ext in [".py", ".cpp", ".h", ".c", ".md", ".txt", ".sh", ".grc"]:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        
        if not text: return

        chunks = chunk_text(text)
        if not chunks: return

        # Prepare batch data
        ids = [f"{os.path.basename(filepath)}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filepath} for _ in range(len(chunks))]
        embeddings = embedder.encode(chunks).tolist()
        
        collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        print(f"   ✅ Indexed: {os.path.basename(filepath)} ({len(chunks)} chunks)")
        
    except Exception as e:
        print(f"   ❌ Failed {os.path.basename(filepath)}: {str(e)[:50]}")

print(f"🚀 Scanning {REPO_PATH}...")
files = glob.glob(f"{REPO_PATH}/**/*", recursive=True)
count = 0
for f in files:
    if os.path.isfile(f) and f.endswith((".pdf", ".py", ".cpp", ".h", ".c", ".md")):
        process_file(f)
        count += 1

print(f"🎉 Indexed {count} files.")
EOF

# --- Script 2: The UI (With RAG Toggle & History) ---
cat << EOF > $APP_FILE
import streamlit as st
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
import os
import json

# Config
DB_PATH = "$DB_PATH"
COLLECTION_NAME = "rag_knowledge"
MODEL_NAME = "$BASE_MODEL"
HISTORY_FILE = "rag_only_history.json"

st.set_page_config(page_title="Pure RAG Explorer", layout="wide")
st.title("🔎 Base Model + RAG (No Fine-Tuning)")

@st.cache_resource
def load_resources():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return collection, embedder

try:
    collection, embedder = load_resources()
except Exception as e:
    st.error(f"Database not found! Did you run the ingest step? Error: {e}")
    st.stop()

# --- Memory ---
if "messages" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            st.session_state["messages"] = json.load(f)
    else:
        st.session_state["messages"] = []

def save_history():
    with open(HISTORY_FILE, "w") as f:
        json.dump(st.session_state["messages"], f)

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    use_rag = st.checkbox("Enable RAG", value=True)
    k_retrieval = st.slider("Context Chunks", 1, 5, 3)
    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        save_history()
        st.rerun()

# --- Chat Loop ---
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about the codebase..."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    save_history()

    context_text = ""
    sources = set()

    if use_rag:
        with st.spinner("Searching docs..."):
            query_vec = embedder.encode([user_input]).tolist()
            results = collection.query(query_embeddings=query_vec, n_results=k_retrieval)
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    src = os.path.basename(results['metadatas'][0][i]['source'])
                    context_text += f"-- FILE: {src} --\n{doc}\n\n"
                    sources.add(src)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_resp = ""
        
        # PROMPT ENGINEERING
        if context_text:
            sys_msg = (
                "You are a coding assistant. "
                "Use the Context below to answer the question. "
                "If the context is not useful, answer from general knowledge."
            )
            final_prompt = f"{sys_msg}\n\nCONTEXT:\n{context_text}\n\nQUESTION:\n{user_input}"
        else:
            final_prompt = user_input
            
        stream = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': final_prompt}], stream=True)
        
        for chunk in stream:
            content = chunk['message']['content']
            full_resp += content
            placeholder.markdown(full_resp + "▌")
        
        if sources:
            full_resp += "\n\n**Sources:** " + ", ".join([f"\`{s}\`" for s in sources])
            placeholder.markdown(full_resp)
            
    st.session_state["messages"].append({"role": "assistant", "content": full_resp})
    save_history()
EOF

# ==============================================================================
# PHASE 4: EXECUTION
# ==============================================================================
echo "🕷️ [4/4] Running Ingest (Scanning Files)..."
# We run the ingest script to build the DB
python3 $INGEST_FILE

echo "======================================================"
echo "✅ SETUP COMPLETE!"
echo "   To launch the app:"
echo "   1. source $VENV_NAME/bin/activate"
echo "   2. streamlit run $APP_FILE"
echo "======================================================"
