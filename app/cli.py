from ingest import load_arxiv_docs
from chunker import chunk_documents
from embedder import Embedder
from vectorstore import FaissVectorStore
from rag import answer_query

DATA_PATH = "data/arxiv_sample.jsonl"


def build_vectorstore():
    # Load and prepare data (done once)
    docs = load_arxiv_docs(DATA_PATH)
    chunks = chunk_documents(docs)

    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(texts)

    vectorstore = FaissVectorStore(dim=embeddings.shape[1])
    vectorstore.add(embeddings, chunks)

    return embedder, vectorstore


def main():
    print("Welcome to the RAG-Powered Research Assistant!")
    print("Type 'exit' to quit.\n")

    embedder, vectorstore = build_vectorstore()

    while True:
        query = input("Enter your research question: ").strip()

        if query.lower() == "exit":
            print("Goodbye 👋")
            break

        answer, sources = answer_query(query, embedder, vectorstore, k=5)

        print("\nAnswer:")
        print(answer)

        print("\nSources:")
        for i, s in enumerate(sources, 1):
            print(f"[{i}] paper={s['meta']['id']} " f"chunk={s['meta']['chunk_id']}")

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
