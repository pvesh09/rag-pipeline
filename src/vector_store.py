import faiss
import numpy as np

def build_faiss_index(embeddings):
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    return index

def search_faiss_index(index, query_embedding, top_k=3):
    D, I = index.search(query_embedding, top_k)
    return I[0]  # Top-k indices
