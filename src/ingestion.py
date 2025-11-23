import os

def load_documents(folder_path):
    docs = []
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                docs.append(f.read())
    return docs

def chunk_text(text, chunk_size=100, chunk_overlap=20):
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
        i += chunk_size - chunk_overlap
    return chunks

# If you want to test this file alone, you can add:
if __name__ == "__main__":
    docs_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs')
    docs = load_documents(docs_folder)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))
    print(f"Loaded {len(docs)} documents. Created {len(all_chunks)} chunks.")

import pdfplumber

def load_pdf_texts(file_paths):
    docs = []
    for path in file_paths:
        with pdfplumber.open(path) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text() + "\n"
            docs.append(full_text)
    return docs
