from embedder import Embedder
from vectorstore import FaissVectorStore


# --- Retrieval ---
def retrieve_chunks(query, embedder, vectorstore, k=5):
    query_embedding = embedder.embed_texts([query])[0]
    results = vectorstore.search(query_embedding, k=k)
    return results


# --- Prompt construction ---
def build_prompt(query, retrieved_chunks):
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        src = chunk["meta"].get("id", "unknown")
        cid = chunk["meta"].get("chunk_id", "n/a")
        context_blocks.append(
            f"[Source {i+1} | paper={src} | chunk={cid}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a helpful research assistant.
Answer the question using ONLY the context below.
If the answer is not present in the context, say "I don't know".



Context:
{context}

Question: {query}

Answer:
"""
    return prompt


# --- Generation (OpenAI-style; keep isolated for easy swap) ---
from openai import OpenAI

client = OpenAI()


def generate_answer(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # modern, cheap, good for RAG
        messages=[
            {"role": "system", "content": "You are a grounded research assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


# --- End-to-end RAG ---
def answer_query(query, embedder, vectorstore, k=5):
    chunks = retrieve_chunks(query, embedder, vectorstore, k=k)
    prompt = build_prompt(query, chunks)
    answer = generate_answer(prompt)
    return answer, chunks
