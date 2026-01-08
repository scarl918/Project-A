import faiss
import numpy as np


class FaissVectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

    def add(self, embeddings, metadatas):
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    def search(self, query_embedding, k=5):
        query_embedding = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for idx in indices[0]:
            results.append(self.metadata[idx])

        return results
