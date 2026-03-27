import faiss
import numpy as np
import json
import os
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple
from app.reranking import HeuristicReranker, BaseReranker, build_default_reranker

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_INDEX_PATH = DATA_DIR / "faiss_index.bin"
DEFAULT_METADATA_PATH = DATA_DIR / "metadata.json"

class VectorStore:
    def __init__(self, dimension: int, index_path: str | os.PathLike | None = None, metadata_path: str | os.PathLike | None = None, auto_load: bool = True, reranker: BaseReranker | None = None):
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else DEFAULT_INDEX_PATH
        self.metadata_path = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        self.sparse_documents = []
        self.document_frequencies = {}
        self.avg_document_length = 0.0
        self.reranker = reranker or build_default_reranker()
        if auto_load:
            self.load()

    def clear(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self.sparse_documents = []
        self.document_frequencies = {}
        self.avg_document_length = 0.0

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
        self._rebuild_sparse_index()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _rebuild_sparse_index(self):
        self.sparse_documents = []
        self.document_frequencies = Counter()

        total_length = 0
        for chunk in self.metadata:
            tokens = self._tokenize(chunk.get("text", ""))
            term_counts = Counter(tokens)
            total_length += len(tokens)
            self.sparse_documents.append({
                "tokens": tokens,
                "term_counts": term_counts,
                "length": len(tokens),
            })
            self.document_frequencies.update(term_counts.keys())

        doc_count = len(self.sparse_documents)
        self.avg_document_length = (total_length / doc_count) if doc_count else 0.0

    def _make_result_key(self, match: Dict, fallback_idx: int) -> str:
        metadata = match.get("metadata", {})
        source = metadata.get("source", "unknown")
        chunk_id = metadata.get("chunk_id", fallback_idx)
        return f"{source}:{chunk_id}"

    def _dense_search(self, query_embedding: np.ndarray, k: int = 8) -> List[Dict]:
        if self.index.ntotal == 0 or not self.metadata:
            return []

        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        distances, indices = self.index.search(query_embedding, min(k, len(self.metadata)))

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx == -1:
                continue

            match = self.metadata[idx]
            dense_score = 1.0 / (1.0 + float(dist))
            results.append({
                "text": match["text"],
                "metadata": match["metadata"],
                "dense_score": dense_score,
                "vector_distance": float(dist),
                "dense_rank": rank,
                "score": dense_score,
            })

        return results

    def _keyword_search(self, query_text: str, k: int = 8) -> List[Dict]:
        if not self.sparse_documents:
            return []

        query_terms = self._tokenize(query_text)
        if not query_terms:
            return []

        k1 = 1.5
        b = 0.75
        doc_count = len(self.sparse_documents)
        scored = []

        for idx, doc in enumerate(self.sparse_documents):
            score = 0.0
            doc_length = max(doc["length"], 1)
            term_counts = doc["term_counts"]

            for term in query_terms:
                term_frequency = term_counts.get(term, 0)
                if term_frequency == 0:
                    continue

                document_frequency = self.document_frequencies.get(term, 0)
                idf = math.log(1 + ((doc_count - document_frequency + 0.5) / (document_frequency + 0.5)))
                numerator = term_frequency * (k1 + 1)
                denominator = term_frequency + k1 * (
                    1 - b + b * (doc_length / max(self.avg_document_length, 1))
                )
                score += idf * (numerator / denominator)

            if score <= 0:
                continue

            match = self.metadata[idx]
            scored.append({
                "text": match["text"],
                "metadata": match["metadata"],
                "keyword_score": float(score),
                "keyword_rank": 0,
            })

        scored.sort(key=lambda item: item["keyword_score"], reverse=True)
        top_results = scored[:k]
        max_keyword_score = top_results[0]["keyword_score"] if top_results else 1.0
        for rank, item in enumerate(top_results, start=1):
            item["keyword_score_raw"] = item["keyword_score"]
            item["keyword_score"] = item["keyword_score"] / max(max_keyword_score, 1e-9)
            item["keyword_rank"] = rank

        return top_results

    def _fuse_results_rrf(self, dense_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        merged = {}
        rrf_k = 60

        for result in dense_results:
            key = self._make_result_key(result, result["dense_rank"])
            merged[key] = {
                **result,
                "keyword_score": 0.0,
                "keyword_rank": None,
            }

        for result in keyword_results:
            key = self._make_result_key(result, result["keyword_rank"])
            if key in merged:
                merged[key]["keyword_score"] = result["keyword_score"]
                merged[key]["keyword_rank"] = result["keyword_rank"]
            else:
                merged[key] = {
                    **result,
                    "dense_score": 0.0,
                    "dense_rank": None,
                    "vector_distance": None,
                }

        for item in merged.values():
            dense_rrf = 1.0 / (rrf_k + item["dense_rank"]) if item.get("dense_rank") else 0.0
            keyword_rrf = 1.0 / (rrf_k + item["keyword_rank"]) if item.get("keyword_rank") else 0.0
            item["dense_rrf"] = dense_rrf
            item["keyword_rrf"] = keyword_rrf
            item["fusion_score"] = dense_rrf + keyword_rrf

        return list(merged.values())

    def _is_relevant_candidate(self, query_text: str, candidate: Dict) -> bool:
        query_tokens = set(self._tokenize(query_text))
        candidate_tokens = set(self._tokenize(candidate.get("text", "")))
        if not query_tokens or not candidate_tokens:
            return False

        overlap_ratio = len(query_tokens.intersection(candidate_tokens)) / max(len(query_tokens), 1)
        return (
            candidate.get("rerank_score", 0.0) >= 0.18
            or candidate.get("dense_score", 0.0) >= 0.2
            or overlap_ratio >= 0.35
        )

    def _passes_final_threshold(self, candidate: Dict, query_type: str, threshold: float, relative_floor: float) -> bool:
        dense_score = candidate.get("dense_score", 0.0)
        keyword_score = candidate.get("keyword_score", 0.0)
        rerank_score = candidate.get("rerank_score", 0.0)
        cross_encoder_score = candidate.get("cross_encoder_score", 0.0)
        heuristic_score = candidate.get("heuristic_score", 0.0)
        relative_pass = rerank_score >= relative_floor if relative_floor > 0 else False

        if query_type == "summarization":
            return (
                relative_pass
                or rerank_score >= max(threshold, 0.16)
                or (keyword_score >= 0.16 and dense_score >= 0.28)
                or heuristic_score >= 0.16
            )

        return (
            relative_pass
            or cross_encoder_score >= max(threshold, 0.25)
            or rerank_score >= max(threshold, 0.22)
            or (dense_score >= 0.42 and keyword_score >= 0.18)
        )

    def _compute_retrieval_metrics(self, query_text: str, ranked_candidates: List[Dict], final_k: int) -> Dict:
        top_k = ranked_candidates[:final_k]
        relevant_flags = [self._is_relevant_candidate(query_text, candidate) for candidate in top_k]
        precision_at_k = (sum(relevant_flags) / len(top_k)) if top_k else 0.0

        reciprocal_rank = 0.0
        for idx, candidate in enumerate(ranked_candidates, start=1):
            if self._is_relevant_candidate(query_text, candidate):
                reciprocal_rank = 1.0 / idx
                break

        return {
            "precision_at_k": round(precision_at_k, 4),
            "mrr": round(reciprocal_rank, 4),
            "relevant_in_top_k": int(sum(relevant_flags)),
        }

    def _apply_diversity(self, candidates: List[Dict], final_k: int = 3) -> List[Dict]:
        selected = []
        seen_pages = set()

        for candidate in candidates:
            candidate_tokens = set(self._tokenize(candidate.get("text", "")))
            is_redundant = False

            for existing in selected:
                existing_tokens = set(self._tokenize(existing.get("text", "")))
                union = len(candidate_tokens.union(existing_tokens))
                overlap = len(candidate_tokens.intersection(existing_tokens)) / union if union else 0.0
                if overlap > 0.7:
                    is_redundant = True
                    break

            page = candidate.get("metadata", {}).get("page")
            if page in seen_pages and len(selected) < final_k - 1:
                is_redundant = True

            if is_redundant:
                continue

            selected.append(candidate)
            if page is not None:
                seen_pages.add(page)

            if len(selected) >= final_k:
                break

        return selected

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        initial_k: int = 8,
        final_k: int = 3,
        threshold: float = 0.1,
        query_type: str = "analytical",
    ) -> Tuple[List[Dict], Dict]:
        overall_start = time.perf_counter()

        dense_start = time.perf_counter()
        dense_results = self._dense_search(query_embedding, k=max(initial_k, final_k))
        dense_latency_ms = (time.perf_counter() - dense_start) * 1000

        keyword_start = time.perf_counter()
        keyword_results = self._keyword_search(query_text, k=max(initial_k, final_k))
        keyword_latency_ms = (time.perf_counter() - keyword_start) * 1000

        fusion_start = time.perf_counter()
        fused_candidates = self._fuse_results_rrf(dense_results, keyword_results)
        fusion_latency_ms = (time.perf_counter() - fusion_start) * 1000

        rerank_start = time.perf_counter()
        try:
            reranked_candidates, rerank_telemetry = self.reranker.rerank(query_text, fused_candidates, query_type=query_type)
        except Exception as exc:
            fallback_reranker = HeuristicReranker()
            reranked_candidates, rerank_telemetry = fallback_reranker.rerank(query_text, fused_candidates, query_type=query_type)
            rerank_telemetry["fallback_reason"] = str(exc)
            rerank_telemetry["fallback_reranker"] = fallback_reranker.name
        rerank_latency_ms = (time.perf_counter() - rerank_start) * 1000

        max_rerank_score = max(
            (candidate.get("rerank_score", 0.0) for candidate in reranked_candidates),
            default=0.0,
        )
        relative_threshold_ratio = 0.6
        relative_floor = max_rerank_score * relative_threshold_ratio

        final_selection_start = time.perf_counter()
        filtered_candidates = [
            candidate for candidate in reranked_candidates
            if self._passes_final_threshold(
                candidate,
                query_type=query_type,
                threshold=threshold,
                relative_floor=relative_floor,
            )
        ]
        final_results = self._apply_diversity(filtered_candidates, final_k=final_k)
        soft_fallback_used = False
        fallback_candidates = []

        if not final_results and reranked_candidates:
            soft_fallback_used = True
            fallback_k = 1 if query_type in {"definition", "fact_lookup"} else 2
            fallback_candidates = self._apply_diversity(reranked_candidates, final_k=fallback_k)
            final_results = fallback_candidates

        final_selection_latency_ms = (time.perf_counter() - final_selection_start) * 1000
        discarded_candidates = [
            {
                "chunk_id": candidate.get("metadata", {}).get("chunk_id"),
                "page": candidate.get("metadata", {}).get("page"),
                "rerank_score": round(candidate.get("rerank_score", 0.0), 4),
            }
            for candidate in reranked_candidates
            if candidate not in final_results
        ]

        for rank, item in enumerate(final_results, start=1):
            item["rank"] = rank
            item["score"] = float(item.get("rerank_score", 0.0))

        telemetry = {
            "fusion_strategy": "rrf",
            "query_type": query_type,
            "dense_candidates": len(dense_results),
            "keyword_candidates": len(keyword_results),
            "fused_candidates": len(fused_candidates),
            "reranked_candidates": len(reranked_candidates),
            "reranked_chunk_ids": [item.get("metadata", {}).get("chunk_id") for item in reranked_candidates],
            "final_candidates": len(final_results),
            "selected_chunk_ids": [item.get("metadata", {}).get("chunk_id") for item in final_results],
            "discarded_candidates": discarded_candidates,
            "retrieval_metrics": self._compute_retrieval_metrics(query_text, reranked_candidates, final_k),
            "relative_threshold_ratio": relative_threshold_ratio,
            "relative_rerank_floor": round(relative_floor, 4),
            "soft_fallback_used": soft_fallback_used,
            "soft_fallback_chunk_ids": [
                item.get("metadata", {}).get("chunk_id") for item in fallback_candidates
            ],
            "rerank_latency_ms": round(rerank_latency_ms, 2),
            "latency_ms": {
                "dense_search": round(dense_latency_ms, 2),
                "keyword_search": round(keyword_latency_ms, 2),
                "fusion": round(fusion_latency_ms, 2),
                "rerank": round(rerank_latency_ms, 2),
                "final_selection": round(final_selection_latency_ms, 2),
                "overall": round((time.perf_counter() - overall_start) * 1000, 2),
            },
            **rerank_telemetry,
        }

        self._log_retrieval(final_results)
        return final_results, telemetry

    def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.6) -> List[Dict]:
        """
        Searches for the top-k most similar documents.
        threshold: Minimum similarity threshold (normalized).
        """
        results = [
            item for item in self._dense_search(query_embedding, k=k)
            if item.get("dense_score", 0.0) >= threshold
        ]
        for rank, item in enumerate(results, start=1):
            item["rank"] = rank
            item["score"] = float(item.get("dense_score", 0.0))

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
            self._rebuild_sparse_index()
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
