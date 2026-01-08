import json

INPUT_FILE = "data/arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = "data/arxiv_sample.jsonl"

ALLOWED_CATEGORIES = {"cs.AI", "cs.CL", "cs.LG"}
MAX_PAPERS = 100  # adjust: 50–100 is perfect

count = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(
    OUTPUT_FILE, "w", encoding="utf-8"
) as fout:

    for line in fin:
        if count >= MAX_PAPERS:
            break

        paper = json.loads(line)

        categories = set(paper.get("categories", "").split())
        if not categories.intersection(ALLOWED_CATEGORIES):
            continue

        filtered = {
            "id": paper.get("id"),
            "title": paper.get("title"),
            "categories": paper.get("categories"),
            "abstract": paper.get("abstract"),
        }

        fout.write(json.dumps(filtered) + "\n")
        count += 1

print(f"Saved {count} papers to {OUTPUT_FILE}")
