import json
import os
import hashlib
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
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f)

    def clear(self):
        self.cache = {}
        self._save_cache()
