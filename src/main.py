import os
from ingestion import load_documents, chunk_text
from embeddings import get_embedding_model, embed_texts
from vector_store import build_faiss_index, search_faiss_index
from rag_pipeline import generate_rag_answer
import numpy as np

def main():
    # Compute the absolute path to the docs folder based on where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_folder = os.path.join(current_dir, '..', 'docs')
    docs = load_documents(docs_folder)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))
    if not all_chunks:
        print("No chunks created! Please add content to docs.")
        return

    model = get_embedding_model()
    embeddings = embed_texts(model, all_chunks)
    embeddings_np = np.array(embeddings).astype("float32")

    index = build_faiss_index(embeddings_np)
    user_query = input("Ask your question: ")
    query_embedding = model.encode([user_query])
    top_indices = search_faiss_index(index, np.array(query_embedding).astype("float32"))
    top_chunks = [all_chunks[i] for i in top_indices]

    print("\nTop relevant chunks:\n")
    for chunk in top_chunks:
        print(chunk)
        print("------")

    answer = generate_rag_answer(top_chunks, user_query)
    print("\n== RAG LLM Answer ==\n")
    print(answer)

if __name__ == "__main__":
    main()
