from sentence_transformers import SentenceTransformer

def get_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def embed_texts(model, texts):
    embeddings = model.encode(texts)
    return embeddings
