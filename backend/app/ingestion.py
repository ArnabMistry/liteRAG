import fitz  # PyMuPDF
from typing import List, Dict
import os

class PDFIngestor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.source_name = os.path.basename(file_path)

    def extract_text_with_metadata(self) -> List[Dict]:
        """
        Extracts text from PDF and returns a list of dictionaries with text and metadata.
        """
        doc = fitz.open(self.file_path)
        pages_content = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()
            
            if text:
                pages_content.append({
                    "text": text,
                    "metadata": {
                        "source": self.source_name,
                        "page": page_num + 1,
                        "total_pages": len(doc)
                    }
                })
        
        doc.close()
        return pages_content

if __name__ == "__main__":
    # Quick debug/smoke test
    import sys
    if len(sys.argv) > 1:
        ingestor = PDFIngestor(sys.argv[1])
        results = ingestor.extract_text_with_metadata()
        print(f"Extracted {len(results)} pages from {ingestor.source_name}")
        if results:
            print(f"First page preview: {results[0]['text'][:200]}...")
