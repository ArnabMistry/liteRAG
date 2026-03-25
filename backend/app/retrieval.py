import faiss
import numpy as np
import json
import os
from typing import List, Dict, Tuple

class VectorStore:
    def __init__(self, dimension: int, index_path: str = "backend/data/faiss_index.bin", metadata_path: str = "backend/data/metadata.json"):
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []

    def add_documents(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Adds embeddings and their corresponding metadata to the store.
        """
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries.")
        
        self.index.add(embeddings.astype("float32"))
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.6) -> List[Dict]:
        """
        Searches for the top-k most similar documents.
        threshold: Minimum similarity threshold (normalized).
        """
        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            # Convert L2 distance to a similarity score (approximate)
            # FAISS IndexFlatL2 returns squared L2 distance.
            # We can use exp(-dist) or similar mapping for score.
            score = 1.0 / (1.0 + dist) 
            
            if score < threshold:
                continue
                
            match = self.metadata[idx]
            results.append({
                "text": match["text"],
                "metadata": match["metadata"],
                "score": float(score),
                "rank": len(results) + 1
            })
            
        self._log_retrieval(results)
        return results

    def _log_retrieval(self, results: List[Dict]):
        """
        Logs retrieved chunks for transparency and debugging.
        """
        print(f"\n--- Scored Retrieval Log (Top-{len(results)}) ---")
        for res in results:
            meta = res['metadata']
            print(f"Rank {res['rank']} | Score: {res['score']:.4f} | Page: {meta.get('page')} | ID: {meta.get('chunk_id')}")
            print(f"Preview: {res['text'][:100]}...\n")

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)

if __name__ == "__main__":
    # Smoke test
    vs = VectorStore(dimension=384)
    dummy_embs = np.random.rand(3, 384).astype("float32")
    dummy_meta = [
        {"text": "Text A", "metadata": {"page": 1, "chunk_id": 1}},
        {"text": "Text B", "metadata": {"page": 2, "chunk_id": 2}},
        {"text": "Text C", "metadata": {"page": 3, "chunk_id": 3}},
    ]
    vs.add_documents(dummy_embs, dummy_meta)
    res = vs.search(dummy_embs[0], k=2)
    print(f"Search found {len(res)} results.")
