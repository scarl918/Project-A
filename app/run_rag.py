from ingest import load_arxiv_docs
from chunker import chunk_documents
from embedder import Embedder
from vectorstore import FaissVectorStore
from rag import answer_query

DATA_PATH = "data/arxiv_sample.jsonl"

# Build index (reuse what you already did)
docs = load_arxiv_docs(DATA_PATH)
chunks = chunk_documents(docs)

embedder = Embedder()
texts = [c["text"] for c in chunks]
embeddings = embedder.embed_texts(texts)

vectorstore = FaissVectorStore(dim=embeddings.shape[1])
vectorstore.add(embeddings, chunks)

# Ask a question
query = "How is acoustic emission used in non-destructive testing?"
answer, sources = answer_query(query, embedder, vectorstore, k=5)

print("\nANSWER:\n", answer)
print("\nSOURCES:")
for s in sources:
    print("-", s["meta"]["id"], "| chunk", s["meta"]["chunk_id"])

print("\nRETRIEVED CONTEXT PREVIEW:")
for r in sources:
    print("-", r["text"][:150])
