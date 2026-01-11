import os
import json
from pathlib import Path
import pdfplumber  # The new requirement

# --- CONFIGURATION ---
SDK_ROOT = "/home/nrogers/src/gnuradio"
OUTPUT_FILE = "gnuradio-train.jsonl"

# File types to scrape
ALLOWED_EXTENSIONS = {
    # The Core Logic
    ".c", ".cpp", ".cc", ".h", ".hpp", ".py",
    # The Build System (Crucial for understanding how OOT modules work)
    ".cmake", "CMakeLists.txt", 
    # The Documentation Source
    ".md", ".txt", ".rst", ".dox" 
}

# Add 'grc' to ignore if you don't want XML flowgraph coordinates cluttering the data
IGNORE_DIRS = {
    ".git", "__pycache__", "build", "bin", ".vscode"
}

def is_text_file(filepath):
    """Check if file is text-based (excludes binary like images, executables)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            f.read(512)
        return True
    except UnicodeDecodeError:
        return False

def extract_pdf_text(filepath):
    """Extracts text from a PDF file, page by page."""
    text_content = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    # Optional: Add page markers so the AI knows where it came from
                    text_content.append(f"--- Page {i+1} ---")
                    text_content.append(text)
        return "\n".join(text_content)
    except Exception as e:
        print(f"⚠️ PDF Error {filepath}: {e}")
        return ""

def create_dataset():
    print(f"🚀 Starting scrape of: {SDK_ROOT}")
    count = 0
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for root, dirs, files in os.walk(SDK_ROOT):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                file_path = Path(root) / file
                
                if file_path.suffix not in ALLOWED_EXTENSIONS:
                    continue

                content = ""
                
                # --- STRATEGY: Handle PDFs vs Code differently ---
                if file_path.suffix == ".pdf":
                    print(f"📄 Processing PDF: {file.name}...")
                    content = extract_pdf_text(file_path)
                
                # Handle Standard Text/Code Files
                elif is_text_file(file_path):
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                    except Exception as e:
                        print(f"⚠️ Read Error {file_path}: {e}")
                
                # Skip if empty or failed to read
                if not content or not content.strip():
                    continue

                # --- FORMATTING (Alpaca Style) ---
                relative_path = file_path.relative_to(SDK_ROOT)
                
                data_entry = {
                    "instruction": f"Provide the contents of the file '{relative_path}' from the SDK.",
                    "input": "",
                    "output": content
                }
                
                out_f.write(json.dumps(data_entry) + "\n")
                count += 1

    print(f"✅ Done! Scraped {count} files into '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    create_dataset()
