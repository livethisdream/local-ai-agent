import os
import glob
import chromadb
import pdfplumber
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
REPO_PATH = "/home/nrogers/src/gnuradio/new"  # Where your data lives
DB_PATH = "./rag_db"  # Where to save the database
COLLECTION_NAME = "gnuradio_knowledge"
CHUNK_SIZE = 2000  # Characters per chunk
OVERLAP = 200  # Overlap to keep context

print("📥 Initializing Embedding Model (all-MiniLM-L6-v2)...")
# This is a small, fast model specifically for search
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print(f"📂 Creating Vector Database at {DB_PATH}...")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def process_file(filepath):
    """Extracts text based on file type."""
    ext = os.path.splitext(filepath)[1].lower()
    text = ""

    try:
        if ext == ".pdf":
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text: text += page_text + "\n"
        elif ext in [".py", ".cpp", ".h", ".c", ".md", ".txt"]:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        if not text: return

        # Chunk and Embed
        chunks = chunk_text(text)
        if not chunks: return

        # Generate unique IDs for the DB
        ids = [f"{filepath}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filepath} for _ in range(len(chunks))]

        # Batch Embed (It's faster)
        embeddings = embedder.encode(chunks).tolist()

        # Add to DB
        collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        print(f"   ✅ Indexed: {os.path.basename(filepath)} ({len(chunks)} chunks)")

    except Exception as e:
        print(f"   ❌ Failed {filepath}: {e}")


# --- MAIN LOOP ---
print(f"🚀 Scanning {REPO_PATH}...")
files = glob.glob(f"{REPO_PATH}/**/*", recursive=True)

count = 0
for f in files:
    if os.path.isfile(f):
        # Filter for relevant files only
        if f.endswith((".pdf", ".py", ".cpp", ".h", ".md")):
            process_file(f)
            count += 1

print(f"🎉 Done! Indexed {count} files into {DB_PATH}")