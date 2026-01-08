def get_chunks_by_paper(chunks, paper_id):
    return [c for c in chunks if c["meta"]["id"] == paper_id]
