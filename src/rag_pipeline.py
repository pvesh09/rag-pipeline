import ollama

def generate_rag_answer(context_chunks, user_query, model_name='llama3'):
    context = "\n\n".join(context_chunks)
    system_prompt = (
        "You are an assistant that answers questions ONLY using the provided context. "
        "If the answer is not in the context, reply 'I do not know.'\n\nContext:\n"
        + context
    )
    full_prompt = f"{system_prompt}\n\nQuestion: {user_query}\nAnswer:"
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": full_prompt}]
    )
    return response['message']['content']
