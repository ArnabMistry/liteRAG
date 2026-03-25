import re
from typing import List, Dict

class SemanticChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        chunk_size: Target words per chunk
        chunk_overlap: Overlap in words
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        # Simple word-based splitting for now
        words = text.split()
        chunks = []
        
        if not words:
            return []
            
        i = 0
        while i < len(words):
            # Take chunk_size words
            chunk_words = words[i : i + self.chunk_size]
            chunks.append(" ".join(chunk_words))
            
            # Move index forward by (chunk_size - overlap)
            if i + self.chunk_size >= len(words):
                break
            i += (self.chunk_size - self.chunk_overlap)
            
        return chunks

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Takes list of {text, metadata} and returns list of chunks with metadata.
        """
        processed_chunks = []
        chunk_id_counter = 0
        
        for doc in documents:
            text = doc["text"]
            metadata = doc["metadata"]
            
            chunks = self.split_text(text)
            
            for i, chunk_text in enumerate(chunks):
                chunk_id_counter += 1
                processed_chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_id": chunk_id_counter,
                        "sub_chunk_index": i
                    }
                })
                
        return processed_chunks

if __name__ == "__main__":
    # Smoke test
    chunker = SemanticChunker(chunk_size=10, chunk_overlap=2)
    sample_text = "This is a long sentence that should be split into multiple chunks for testing purposes."
    chunks = chunker.split_text(sample_text)
    for i, c in enumerate(chunks):
        print(f"Chunk {i}: {c}")
