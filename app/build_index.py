from ingest import load_arxiv_docs
from chunker import chunk_documents
from embedder import Embedder
from vectorstore import FaissVectorStore

DATA_PATH = "data/arxiv_sample.jsonl"

# 1. Load docs
docs = load_arxiv_docs(DATA_PATH)
print(f"Loaded {len(docs)} docs")

# 2. Chunk
chunks = chunk_documents(docs)
print(f"Created {len(chunks)} chunks")

# 3. Embed
embedder = Embedder()
texts = [c["text"] for c in chunks]
embeddings = embedder.embed_texts(texts)

# 4. Index
vectorstore = FaissVectorStore(dim=embeddings.shape[1])
vectorstore.add(embeddings, chunks)

print("FAISS index built successfully")
