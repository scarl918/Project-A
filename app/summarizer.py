from openai import OpenAI

client = OpenAI()


def summarize_chunks(chunks, max_chunks=8):
    """
    chunks: list of chunk dicts for ONE paper
    """
    # cap context so tokens don't explode
    chunks = chunks[:max_chunks]

    context = "\n\n".join(
        f"[Chunk {c['meta']['chunk_id']}]\n{c['text']}" for c in chunks
    )

    prompt = f"""
You are a research assistant.
Summarize the following research content into a concise, structured overview.
Only use the information provided. Do not add external facts.

Content:
{context}

Summary:
- Problem:
- Method:
- Key Findings:
- Applications:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()
