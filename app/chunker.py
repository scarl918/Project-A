def chunk_text(text, chunk_size=180, overlap=40):
    """
    Splits text into overlapping word-based chunks.
    Returns list of strings.
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def chunk_documents(docs):
    """
    docs: list of { "text": str, "meta": dict }
    returns: list of { "text": str, "meta": dict }
    """
    all_chunks = []

    for doc in docs:
        chunks = chunk_text(doc["text"], chunk_size=180, overlap=40)

        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "text": chunk,
                    "meta": {
                        **doc["meta"],
                        "chunk_id": i,
                        "start_word": i * (180 - 40),
                        "end_word": i * (180 - 40) + len(chunk.split()),
                    },
                }
            )

    return all_chunks
