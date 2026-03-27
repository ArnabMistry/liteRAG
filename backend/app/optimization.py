import re
from typing import List, Dict

class ContextOptimizer:
    SUMMARY_CUE_TOKENS = {
        "should",
        "recommend",
        "recommended",
        "recommendations",
        "guidance",
        "compare",
        "avoid",
        "workflow",
        "evidence",
        "sources",
    }

    def __init__(self, token_limit: int = 800):
        self.token_limit = token_limit

    def _tokenize(self, text: str) -> set:
        """Simple tokenizer for word-level Jaccard similarity."""
        return set(re.findall(r'\w+', text.lower()))

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]

    def _build_sentence_groups(self, sentences: List[str]) -> List[str]:
        if not sentences:
            return []

        groups = []
        idx = 0
        while idx < len(sentences):
            current = sentences[idx]
            if idx + 1 < len(sentences):
                next_sentence = sentences[idx + 1]
                if len(current.split()) < 18 or len(next_sentence.split()) < 18:
                    groups.append(f"{current} {next_sentence}".strip())
                    idx += 2
                    continue
            groups.append(current)
            idx += 1

        return groups

    def build_context_package(self, query: str, retrieved_chunks: List[Dict], query_type: str = "analytical") -> Dict:
        query_tokens = self._tokenize(query)
        
        # 1. Extract and Score Sentences
        scored_sentences = []
        for chunk in retrieved_chunks:
            page = chunk['metadata'].get('page', '?')
            # Clean text and split into sentences
            text = chunk.get("text", "").replace("\n", " ").strip()
            sentences = self._split_sentences(text)
            sentence_groups = self._build_sentence_groups(sentences)
            
            for sentence_group in sentence_groups:
                sent_tokens = self._tokenize(sentence_group)
                score = self._jaccard_similarity(query_tokens, sent_tokens)
                
                # Boost score slightly if chunk had a high retrieval score
                chunk_score = chunk.get("score", 0.0)
                rerank_score = chunk.get("rerank_score", 0.0)
                cross_encoder_score = chunk.get("cross_encoder_score", 0.0)
                final_score = (
                    score
                    + (chunk_score * 0.12)
                    + (rerank_score * 0.18)
                    + (cross_encoder_score * 0.22)
                )
                if query_type == "summarization":
                    summary_cue_overlap = len(sent_tokens.intersection(self.SUMMARY_CUE_TOKENS)) / len(self.SUMMARY_CUE_TOKENS)
                    final_score += summary_cue_overlap * 0.35
                
                scored_sentences.append({
                    "text": sentence_group,
                    "score": final_score,
                    "tokens": sent_tokens,
                    "page": page,
                    "chunk_id": chunk['metadata'].get('chunk_id'),
                    "chunk_rank": chunk.get("rank", 999),
                })
        
        # Sort by relevance score descending
        scored_sentences.sort(key=lambda x: x["score"], reverse=True)
        
        # 2. Deduplication and Selection
        selected_sentences = []
        current_tokens = 0
        selected_pages = set()
        dynamic_relevance_floor = max(
            0.16 if query_type in {"definition", "fact_lookup"} else (0.08 if query_type == "summarization" else 0.14),
            min(0.34, scored_sentences[0]["score"] * (0.62 if query_type in {"definition", "fact_lookup"} else (0.28 if query_type == "summarization" else 0.5)))
            if scored_sentences else (0.08 if query_type == "summarization" else 0.14)
        )
        
        for item in scored_sentences:
            if item["score"] < dynamic_relevance_floor:
                continue

            # Check deduplication against already selected sentences
            is_redundant = False
            for selected in selected_sentences:
                overlap = self._jaccard_similarity(item["tokens"], selected["tokens"])
                if overlap > 0.6:  # 60% overlap means highly redundant
                    is_redundant = True
                    break
                    
            if is_redundant:
                continue
                
            # Simple token estimation: characters / 4
            est_tokens = len(item["text"]) // 4
            
            if current_tokens + est_tokens <= self.token_limit:
                selected_sentences.append(item)
                current_tokens += est_tokens
                selected_pages.add(item["page"])
            else:
                break

        best_by_chunk = {}
        if query_type == "summarization" and retrieved_chunks:
            top_chunk_ids = [
                chunk.get("metadata", {}).get("chunk_id")
                for chunk in sorted(retrieved_chunks, key=lambda chunk: chunk.get("rank", 999))[:3]
            ]
            for item in scored_sentences:
                chunk_id = item.get("chunk_id")
                existing = best_by_chunk.get(chunk_id)
                if existing is None or item["score"] > existing["score"]:
                    best_by_chunk[chunk_id] = item

            for chunk_id in top_chunk_ids:
                if chunk_id is None or any(selected.get("chunk_id") == chunk_id for selected in selected_sentences):
                    continue

                candidate = best_by_chunk.get(chunk_id)
                if not candidate:
                    continue

                est_tokens = len(candidate["text"]) // 4
                if current_tokens + est_tokens <= self.token_limit:
                    selected_sentences.append(candidate)
                    current_tokens += est_tokens
                    selected_pages.add(candidate["page"])
                    continue

                removable_index = next(
                    (
                        idx for idx, selected in enumerate(sorted(selected_sentences, key=lambda value: value["score"]))
                        if selected.get("chunk_id") not in top_chunk_ids
                    ),
                    None,
                )
                if removable_index is not None:
                    removable = sorted(selected_sentences, key=lambda value: value["score"])[removable_index]
                    selected_sentences.remove(removable)
                    current_tokens -= len(removable["text"]) // 4
                    if current_tokens + est_tokens <= self.token_limit:
                        selected_sentences.append(candidate)
                        current_tokens += est_tokens
                        selected_pages.add(candidate["page"])

        if query_type == "summarization" and len(selected_sentences) < 2:
            for item in sorted(best_by_chunk.values(), key=lambda value: (value.get("chunk_rank", 999), -value["score"])):
                if any(selected.get("chunk_id") == item.get("chunk_id") for selected in selected_sentences):
                    continue

                est_tokens = len(item["text"]) // 4
                if current_tokens + est_tokens > self.token_limit:
                    continue

                selected_sentences.append(item)
                current_tokens += est_tokens
                selected_pages.add(item["page"])
                if len(selected_sentences) >= min(3, len(retrieved_chunks)):
                    break
                
        # 3. Build compact context (sorted by page to retain logical flow)
        selected_sentences.sort(key=lambda x: (x.get("page", 0), -x.get("score", 0)))
        
        optimized_text = ""
        current_page = None
        for item in selected_sentences:
            if item["page"] != current_page:
                optimized_text += f"\n[Page {item['page']}] "
                current_page = item["page"]
            optimized_text += item["text"] + " "

        reasoning_summary = (
            f"Selected {len(selected_sentences)} high-relevance sentences "
            f"from {len(selected_pages)} pages under an estimated {current_tokens}-token budget."
        )

        return {
            "context": optimized_text.strip(),
            "selected_sentences": selected_sentences,
            "token_estimate": current_tokens,
            "pages_used": sorted(page for page in selected_pages if page is not None),
            "relevance_floor": round(dynamic_relevance_floor, 4),
            "reasoning_summary": reasoning_summary,
        }

    def optimize_context(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """
        Compresses retrieved chunks at the sentence level without using an LLM.
        Selects sentences most relevant to the query, removes redundancies,
        and builds a compact context enforcing a strict token limit.
        """
        return self.build_context_package(query, retrieved_chunks)["context"]
