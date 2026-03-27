import faiss
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_INDEX_PATH = DATA_DIR / "faiss_index.bin"
DEFAULT_METADATA_PATH = DATA_DIR / "metadata.json"

class VectorStore:
    def __init__(self, dimension: int, index_path: str | os.PathLike | None = None, metadata_path: str | os.PathLike | None = None, auto_load: bool = True):
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else DEFAULT_INDEX_PATH
        self.metadata_path = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        if auto_load:
            self.load()

    def clear(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []

    def has_documents(self) -> bool:
        return self.index.ntotal > 0 and len(self.metadata) > 0

    def reset(self):
        try:
            self.clear()

            if self.index_path.exists():
                self.index_path.unlink()

            if self.metadata_path.exists():
                self.metadata_path.unlink()
        except Exception as exc:
            raise RuntimeError("Failed to reset vector store state.") from exc

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
        try:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f)
        except Exception as exc:
            raise RuntimeError("Failed to persist vector store state.") from exc

    def load(self):
        try:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
            else:
                self.index = faiss.IndexFlatL2(self.dimension)

            if self.metadata_path.exists():
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = []
        except Exception as exc:
            self.clear()
            raise RuntimeError("Failed to load vector store state.") from exc

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
