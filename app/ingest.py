import json


def load_arxiv_docs(path):
    docs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)

            text = (
                paper["title"].strip().replace("\n", " ")
                + "\n\n"
                + paper["abstract"].strip().replace("\n", " ")
            )

            docs.append(
                {
                    "text": text,
                    "meta": {"id": paper["id"], "categories": paper["categories"]},
                }
            )

    return docs


if __name__ == "__main__":
    docs = load_arxiv_docs("data/arxiv_sample.jsonl")
    print(f"Loaded {len(docs)} documents")
    print(docs[0]["text"][:500])
