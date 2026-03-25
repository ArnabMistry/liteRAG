from typing import List, Dict

class ContextOptimizer:
    def __init__(self, token_limit: int = 1500):
        self.token_limit = token_limit

    def optimize_context(self, retrieved_chunks: List[Dict]) -> str:
        """
        Enforce a hard token limit by dropping chunks based on retrieval score.
        For simplicity, we use character-based estimation (4 chars ~= 1 token).
        """
        # Sort by score descending (though usually already sorted)
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x.get("score", 0), reverse=True)
        
        optimized_text = ""
        current_tokens = 0
        
        for chunk in sorted_chunks:
            # Simple token estimation: characters / 4
            chunk_tokens = len(chunk["text"]) // 4 
            
            if current_tokens + chunk_tokens <= self.token_limit:
                optimized_text += f"\n--- Source (Page {chunk['metadata'].get('page')}) ---\n"
                optimized_text += chunk["text"] + "\n"
                current_tokens += chunk_tokens
            else:
                # Hard cutoff: drop the rest
                break
                
        return optimized_text.strip()
