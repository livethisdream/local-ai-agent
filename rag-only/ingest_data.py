import os
import glob
import chromadb
import pdfplumber
from sentence_transformers import SentenceTransformer

REPO_PATH = "/home/nrogers/src/gnuradio"
DB_PATH = "./rag_db"
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
