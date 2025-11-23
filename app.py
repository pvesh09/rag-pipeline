import streamlit as st
import io
import tempfile
import pdfplumber
import numpy as np

from src.ingestion import chunk_text
from src.embeddings import get_embedding_model, embed_texts
from src.vector_store import build_faiss_index, search_faiss_index
from src.rag_pipeline import generate_rag_answer

st.title("RAG Pipeline with Ollama - TXT + PDF Upload")

uploaded_files = st.file_uploader(
    "Upload .txt or .pdf files", accept_multiple_files=True, type=['txt', 'pdf']
)

if uploaded_files:
    docs = []
    pdf_buffers = []
    txt_docs = []

    # Separate uploaded files by type
    for file in uploaded_files:
        if file.type == "application/pdf":
            pdf_buffers.append(file)
        else:
            txt_docs.append(file.getvalue().decode("utf-8"))

    # Add text documents normally
    docs.extend(txt_docs)

    # Extract text from pdf files using pdfplumber
    for pdf_file in pdf_buffers:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        with pdfplumber.open(tmp_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            docs.append(text)

    # Chunk all documents
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))

    # Embed chunks
    model = get_embedding_model()
    embeddings = embed_texts(model, all_chunks)
    embeddings_np = np.array(embeddings).astype("float32")

    # Build vector index
    index = build_faiss_index(embeddings_np)

    # Query input
    query = st.text_input("Enter your question:")
    if query:
        query_embedding = model.encode([query])
        top_indices = search_faiss_index(index, np.array(query_embedding).astype("float32"))
        top_indices = top_indices.flatten()
        top_chunks = [all_chunks[i] for i in top_indices]

        # Display retrieved chunks
        st.subheader("Retrieved Chunks")
        for chunk in top_chunks:
            st.write(chunk)

        # Get and show LLM answer
        answer = generate_rag_answer(top_chunks, query)
        st.subheader("Generated Answer")
        st.write(answer)
