import json
import os
import hashlib
from typing import Optional

class QueryCache:
    def __init__(self, cache_path: str = "backend/data/query_cache.json"):
        self.cache_path = cache_path
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _normalize_query(self, query: str) -> str:
        return query.strip().lower()

    def get(self, query: str) -> Optional[str]:
        norm_query = self._normalize_query(query)
        # Use MD5 hash for keys to avoid issues with long strings
        query_hash = hashlib.md5(norm_query.encode()).hexdigest()
        return self.cache.get(query_hash)

    def set(self, query: str, response: str):
        norm_query = self._normalize_query(query)
        query_hash = hashlib.md5(norm_query.encode()).hexdigest()
        self.cache[query_hash] = response
        self._save_cache()

    def _save_cache(self):
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)

    def clear(self):
        self.cache = {}
        self._save_cache()
