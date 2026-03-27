import re
import math
from typing import Dict, List, Tuple


class BaseReranker:
    name = "base"

    def rerank(self, query_text: str, candidates: List[Dict], query_type: str = "analytical") -> Tuple[List[Dict], Dict]:
        raise NotImplementedError

    def prepare_inputs(self, query_text: str, candidates: List[Dict]) -> List[Tuple[str, str]]:
        return [(query_text, candidate.get("text", "")) for candidate in candidates]


class HeuristicReranker(BaseReranker):
    """
    Lightweight reranker designed to be replaceable by a future cross-encoder.
    It keeps retrieval and reranking as separate stages while preserving the
    current local-only architecture.
    """
    name = "heuristic"

    def _tokenize(self, text: str) -> set:
        return set(re.findall(r"\w+", text.lower()))

    def rerank(self, query_text: str, candidates: List[Dict], query_type: str = "analytical") -> Tuple[List[Dict], Dict]:
        query_tokens = self._tokenize(query_text)
        reranked = []
        decision_log = []

        for candidate in candidates:
            candidate_tokens = self._tokenize(candidate.get("text", ""))
            overlap = len(query_tokens.intersection(candidate_tokens))
            lexical_overlap = overlap / max(len(query_tokens), 1)
            exact_phrase = 1.0 if query_text.lower() in candidate.get("text", "").lower() else 0.0

            rerank_score = (
                candidate.get("dense_score", 0.0) * 0.50
                + candidate.get("keyword_score", 0.0) * 0.10
                + candidate.get("fusion_score", 0.0) * 0.20
                + lexical_overlap * 0.15
                + exact_phrase * 0.05
            )

            candidate["rerank_score"] = float(rerank_score)
            reranked.append(candidate)
            decision_log.append({
                "chunk_id": candidate.get("metadata", {}).get("chunk_id"),
                "page": candidate.get("metadata", {}).get("page"),
                "dense_score": round(candidate.get("dense_score", 0.0), 4),
                "keyword_score": round(candidate.get("keyword_score", 0.0), 4),
                "fusion_score": round(candidate.get("fusion_score", 0.0), 4),
                "lexical_overlap": round(lexical_overlap, 4),
                "exact_phrase": exact_phrase,
                "rerank_score": round(candidate["rerank_score"], 4),
            })

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        telemetry = {
            "reranker": self.name,
            "decision_log": decision_log,
        }
        return reranked, telemetry


class CrossEncoderReranker(BaseReranker):
    """
    Stronger reranker that uses a lightweight cross-encoder when available.
    Falls back by raising at load/predict time so callers can recover cleanly.
    """

    name = "cross_encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def _sigmoid(self, value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    def _tokenize(self, text: str) -> set:
        return set(re.findall(r"\w+", text.lower()))

    def _heuristic_score(self, query_text: str, candidate: Dict) -> Tuple[float, float, float]:
        query_tokens = self._tokenize(query_text)
        candidate_tokens = self._tokenize(candidate.get("text", ""))
        overlap = len(query_tokens.intersection(candidate_tokens))
        lexical_overlap = overlap / max(len(query_tokens), 1)
        exact_phrase = 1.0 if query_text.lower() in candidate.get("text", "").lower() else 0.0
        heuristic_score = (
            candidate.get("dense_score", 0.0) * 0.50
            + candidate.get("keyword_score", 0.0) * 0.10
            + candidate.get("fusion_score", 0.0) * 0.20
            + lexical_overlap * 0.15
            + exact_phrase * 0.05
        )
        return heuristic_score, lexical_overlap, exact_phrase

    def rerank(self, query_text: str, candidates: List[Dict], query_type: str = "analytical") -> Tuple[List[Dict], Dict]:
        model = self._get_model()
        pairs = self.prepare_inputs(query_text, candidates)
        raw_scores = model.predict(pairs)

        reranked = []
        decision_log = []
        for candidate, raw_score in zip(candidates, raw_scores):
            raw_value = float(raw_score)
            normalized_score = self._sigmoid(raw_value)
            heuristic_score, lexical_overlap, exact_phrase = self._heuristic_score(query_text, candidate)

            if query_type == "summarization":
                rerank_score = (
                    heuristic_score * 0.60
                    + candidate.get("keyword_score", 0.0) * 0.20
                    + candidate.get("dense_score", 0.0) * 0.10
                    + candidate.get("fusion_score", 0.0) * 0.10
                )
            elif query_type in {"definition", "fact_lookup"}:
                rerank_score = (
                    normalized_score * 0.70
                    + heuristic_score * 0.20
                    + candidate.get("fusion_score", 0.0) * 0.10
                )
            else:
                rerank_score = (
                    normalized_score * 0.65
                    + heuristic_score * 0.25
                    + candidate.get("fusion_score", 0.0) * 0.10
                )

            candidate["cross_encoder_score"] = normalized_score
            candidate["heuristic_score"] = heuristic_score
            candidate["rerank_score"] = rerank_score
            reranked.append(candidate)
            decision_log.append({
                "chunk_id": candidate.get("metadata", {}).get("chunk_id"),
                "page": candidate.get("metadata", {}).get("page"),
                "cross_encoder_score": round(normalized_score, 4),
                "heuristic_score": round(heuristic_score, 4),
                "fusion_score": round(candidate.get("fusion_score", 0.0), 4),
                "lexical_overlap": round(lexical_overlap, 4),
                "exact_phrase": exact_phrase,
                "rerank_score": round(rerank_score, 4),
            })

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        telemetry = {
            "reranker": self.name,
            "model_name": self.model_name,
            "decision_log": decision_log,
        }
        return reranked, telemetry


def build_default_reranker() -> BaseReranker:
    try:
        return CrossEncoderReranker()
    except Exception:
        return HeuristicReranker()
