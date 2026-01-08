def format_citation(meta, index):
    title = meta.get("title", "Unknown Title")
    paper_id = meta.get("id", "unknown")
    return f"[{index}] {title} (arXiv:{paper_id})"
