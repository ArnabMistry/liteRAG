import re
from typing import Dict, List


class QueryProcessor:
    DEFINITION_PREFIXES = ("what is", "what are", "define", "who is", "explain")
    FACT_PREFIXES = ("when", "where", "which", "how many", "how much", "name", "list")
    SUMMARY_KEYWORDS = ("summarize", "summary", "overview", "main points", "key themes", "abstract")
    ANALYTICAL_KEYWORDS = (
        "why",
        "how does",
        "analyze",
        "compare",
        "contrast",
        "implication",
        "evaluate",
        "relationship",
        "reason",
        "impact",
    )
    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "for", "in",
        "on", "and", "or", "at", "by", "with", "from", "about", "this", "that",
    }
    QUERY_EXPANSIONS = {
        "why does stoicism help": "benefits of stoicism philosophy for mental discipline and resilience",
        "stoicism": "stoicism philosophy principles discipline resilience virtue",
        "rag": "retrieval augmented generation retrieval pipeline embeddings context grounding",
    }
    QUERY_SIGNAL_WEIGHTS = {
        "definition": 0.9,
        "summarization": 0.85,
        "fact_lookup": 0.8,
        "analytical": 0.85,
    }

    def normalize_query(self, query: str) -> str:
        normalized = query.strip().lower()
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def classify_query(self, query: str) -> Dict:
        normalized = self.normalize_query(query)
        label = "analytical"
        matched_signals = []

        if any(normalized.startswith(prefix) for prefix in self.DEFINITION_PREFIXES):
            label = "definition"
            matched_signals.append("definition_prefix")
        elif any(keyword in normalized for keyword in self.SUMMARY_KEYWORDS):
            label = "summarization"
            matched_signals.append("summary_keyword")
        elif any(normalized.startswith(prefix) for prefix in self.FACT_PREFIXES):
            label = "fact_lookup"
            matched_signals.append("fact_prefix")
        elif any(keyword in normalized for keyword in self.ANALYTICAL_KEYWORDS):
            label = "analytical"
            matched_signals.append("analytical_keyword")
        elif len(normalized.split()) <= 6:
            label = "fact_lookup"
            matched_signals.append("short_query_heuristic")
        else:
            matched_signals.append("default_long_query")

        confidence = self.QUERY_SIGNAL_WEIGHTS.get(label, 0.6)
        if matched_signals == ["short_query_heuristic"]:
            confidence = 0.58
        elif matched_signals == ["default_long_query"]:
            confidence = 0.55

        return {
            "label": label,
            "confidence": confidence,
            "signals": matched_signals,
        }

    def rewrite_query(self, query: str, query_type: str) -> str:
        normalized = self.normalize_query(query)

        if normalized in self.QUERY_EXPANSIONS:
            return self.QUERY_EXPANSIONS[normalized]

        tokens = [token for token in normalized.split() if token not in self.STOPWORDS]
        if not tokens:
            return normalized

        # Keep rewriting lightweight so we preserve the user's phrasing.
        if len(tokens) <= 3:
            return " ".join(tokens)

        if query_type == "definition":
            return f"{' '.join(tokens)} meaning"

        if query_type == "fact_lookup":
            return " ".join(tokens)

        if query_type == "summarization":
            return f"{' '.join(tokens)} summary"

        return " ".join(tokens)

    def build_query_package(self, query: str) -> Dict:
        classification = self.classify_query(query)
        query_type = classification["label"]
        normalized = self.normalize_query(query)
        rewritten = self.rewrite_query(query, query_type)

        variants: List[str] = [query.strip()]
        for candidate in (normalized, rewritten):
            if candidate and candidate not in variants:
                variants.append(candidate)

        return {
            "original": query,
            "normalized": normalized,
            "rewritten": rewritten,
            "embedding_query": rewritten if normalized in self.QUERY_EXPANSIONS else query.strip(),
            "keyword_query": query.strip(),
            "query_type": query_type,
            "classification_confidence": classification["confidence"],
            "classification_signals": classification["signals"],
            "force_full_rag": classification["confidence"] < 0.65,
            "variants": variants,
            "should_prefer_direct_answer": (
                query_type in {"definition", "fact_lookup"}
                and len(normalized.split()) <= 8
                and classification["confidence"] >= 0.7
            ),
        }
