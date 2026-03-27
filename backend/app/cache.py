import json
import os
import hashlib
import math
import re
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_CACHE_PATH = DATA_DIR / "query_cache.json"

class QueryCache:
    def __init__(self, cache_path: str | os.PathLike | None = None):
        self.cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _normalize_query(self, query: str) -> str:
        normalized = query.strip().lower()
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _build_payload(self, value):
        if isinstance(value, dict):
            return value
        return {"answer": value}

    def _cosine_similarity(self, vector_a, vector_b) -> float:
        if not vector_a or not vector_b or len(vector_a) != len(vector_b):
            return 0.0

        dot = sum(a * b for a, b in zip(vector_a, vector_b))
        norm_a = math.sqrt(sum(a * a for a in vector_a))
        norm_b = math.sqrt(sum(b * b for b in vector_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get(self, query: str, query_embedding=None, similarity_threshold: float = 0.9) -> Optional[dict]:
        norm_query = self._normalize_query(query)
        # Use MD5 hash for keys to avoid issues with long strings
        query_hash = hashlib.md5(norm_query.encode()).hexdigest()
        exact_hit = self.cache.get(query_hash)
        if exact_hit is not None:
            payload = self._build_payload(exact_hit)
            payload["cache_match"] = "exact"
            return payload

        if query_embedding is None:
            return None

        embedding_list = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding
        for cached_value in self.cache.values():
            payload = self._build_payload(cached_value)
            cached_embedding = payload.get("query_embedding")
            confidence = payload.get("confidence", {})
            if confidence.get("level") == "low":
                continue
            similarity = self._cosine_similarity(embedding_list, cached_embedding)
            if similarity >= similarity_threshold:
                if payload.get("query_type") == "analytical" and similarity < 0.94:
                    continue
                payload["cache_match"] = "semantic"
                payload["cache_similarity"] = similarity
                return payload

        return None

    def set(self, query: str, response, query_embedding=None):
        norm_query = self._normalize_query(query)
        query_hash = hashlib.md5(norm_query.encode()).hexdigest()
        payload = self._build_payload(response)
        payload["normalized_query"] = norm_query
        if query_embedding is not None:
            payload["query_embedding"] = (
                query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding
            )
        self.cache[query_hash] = payload
        self._save_cache()

    def _save_cache(self):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f)

    def clear(self):
        self.cache = {}
        self._save_cache()
