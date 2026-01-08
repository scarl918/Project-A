import json


def load_arxiv_docs(path):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)
            text = f"{paper['title']}\n\n{paper['abstract']}"
            docs.append(
                {
                    "text": text,
                    "meta": {"id": paper["id"], "categories": paper["categories"]},
                }
            )
    return docs
