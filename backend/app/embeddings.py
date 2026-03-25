from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np

class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        model_name: Name of the sentence-transformers model.
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of strings.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generates embedding for a single string.
        """
        return self.model.encode([text], convert_to_numpy=True)[0]

if __name__ == "__main__":
    # Smoke test
    engine = EmbeddingEngine()
    print(f"Model dimensions: {engine.dimension}")
    emb = engine.generate_single_embedding("Hello world")
    print(f"Embedding shape: {emb.shape}")
