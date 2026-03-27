import re
from typing import List, Dict

class SemanticChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 60):
        """
        chunk_size: Target words per chunk
        chunk_overlap: Overlap in words
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_sentences(self, paragraph: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def split_text(self, text: str) -> List[str]:
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
        if not paragraphs:
            return []

        chunks = []
        current_chunk_sentences: List[str] = []
        current_word_count = 0

        for paragraph in paragraphs:
            sentences = self._split_sentences(paragraph)
            if not sentences:
                continue

            paragraph_word_count = len(paragraph.split())
            if current_chunk_sentences and current_word_count + paragraph_word_count > self.chunk_size:
                chunks.append(" ".join(current_chunk_sentences).strip())
                overlap_words = " ".join(current_chunk_sentences).split()[-self.chunk_overlap:]
                current_chunk_sentences = [" ".join(overlap_words)] if overlap_words else []
                current_word_count = len(overlap_words)

            for sentence in sentences:
                sentence_word_count = len(sentence.split())
                if current_chunk_sentences and current_word_count + sentence_word_count > self.chunk_size:
                    chunks.append(" ".join(current_chunk_sentences).strip())
                    overlap_words = " ".join(current_chunk_sentences).split()[-self.chunk_overlap:]
                    current_chunk_sentences = [" ".join(overlap_words)] if overlap_words else []
                    current_word_count = len(overlap_words)

                current_chunk_sentences.append(sentence)
                current_word_count += sentence_word_count

        if current_chunk_sentences:
            final_chunk = " ".join(current_chunk_sentences).strip()
            if final_chunk:
                chunks.append(final_chunk)

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
