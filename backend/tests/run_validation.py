import sys
import os
sys.path.append(os.getcwd())

from backend.app.ingestion import PDFIngestor
from backend.app.chunking import SemanticChunker
from backend.app.embeddings import EmbeddingEngine
from backend.app.retrieval import VectorStore
from backend.tests.eval_data import QA_PAIRS

def run_validation():
    print("--- Starting Retrieval Validation ---\n")
    
    # 1. Ingest & Chunk
    pdf_path = "backend/data/test_eval.pdf"
    if not os.path.exists(pdf_path):
        from backend.tests.create_test_pdf import create_test_pdf
        create_test_pdf(pdf_path)
        
    ingestor = PDFIngestor(pdf_path)
    docs = ingestor.extract_text_with_metadata()
    
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_documents(docs)
    
    # 2. Embed & Index
    engine = EmbeddingEngine()
    texts = [c["text"] for c in chunks]
    embeddings = engine.generate_embeddings(texts)
    
    vs = VectorStore(dimension=engine.dimension)
    vs.add_documents(embeddings, chunks)
    
    # 3. Test Retrieval
    pass_count = 0
    for test in QA_PAIRS:
        print(f"Query: {test['query']}")
        query_emb = engine.generate_single_embedding(test['query'])
        results = vs.search(query_emb, k=3, threshold=0.1) # low threshold for test
        
        if not results:
            print("FAILED: No results found.")
            continue
            
        top_res = results[0]
        found_page = top_res["metadata"]["page"]
        
        success = (found_page == test["expected_page"]) and (test["expected_text"].lower() in top_res["text"].lower())
        
        if success:
            print(f"PASSED: Found on page {found_page} with score {top_res['score']:.4f}")
            pass_count += 1
        else:
            print(f"FAILED: Expected page {test['expected_page']}, found page {found_page}")
        print("-" * 20)
        
    print(f"\nFinal Result: {pass_count}/{len(QA_PAIRS)} passed.")

if __name__ == "__main__":
    run_validation()
