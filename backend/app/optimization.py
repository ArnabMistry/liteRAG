import re
from typing import List, Dict

class ContextOptimizer:
    def __init__(self, token_limit: int = 800):
        self.token_limit = token_limit

    def _tokenize(self, text: str) -> set:
        """Simple tokenizer for word-level Jaccard similarity."""
        return set(re.findall(r'\w+', text.lower()))

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def optimize_context(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """
        Compresses retrieved chunks at the sentence level without using an LLM.
        Selects sentences most relevant to the query, removes redundancies,
        and builds a compact context enforcing a strict token limit.
        """
        query_tokens = self._tokenize(query)
        
        # 1. Extract and Score Sentences
        scored_sentences = []
        for chunk in retrieved_chunks:
            page = chunk['metadata'].get('page', '?')
            # Clean text and split into sentences
            text = chunk.get("text", "").replace("\n", " ").strip()
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]
            
            for sentence in sentences:
                sent_tokens = self._tokenize(sentence)
                score = self._jaccard_similarity(query_tokens, sent_tokens)
                
                # Boost score slightly if chunk had a high retrieval score
                chunk_score = chunk.get("score", 0.0)
                final_score = score + (chunk_score * 0.1)
                
                scored_sentences.append({
                    "text": sentence,
                    "score": final_score,
                    "tokens": sent_tokens,
                    "page": page
                })
        
        # Sort by relevance score descending
        scored_sentences.sort(key=lambda x: x["score"], reverse=True)
        
        # 2. Deduplication and Selection
        selected_sentences = []
        current_tokens = 0
        
        for item in scored_sentences:
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
            else:
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
            
        return optimized_text.strip()
