import fitz

def create_test_pdf(filename="backend/data/test_eval.pdf"):
    doc = fitz.open()
    
    content = [
        ("The History of RAG", "Retrieval-Augmented Generation (RAG) was first proposed in 2020 by Lewis et al. It combines a pre-trained retriever and a pre-trained seq2seq model."),
        ("Vector Databases", "FAISS is a library for efficient similarity search and clustering of dense vectors. It was developed by Facebook AI Research."),
        ("The Role of Embeddings", "Embeddings are numerical representations of text. Models like all-MiniLM-L6-v2 map sentences to a 384-dimensional vector space."),
        ("Token Limits in LLMs", "Large Language Models have context windows. Sending too much context can be expensive and may lead to the 'lost in the middle' phenomenon."),
        ("Architecture of liteRAG", "liteRAG uses FastAPI for its backend and React for its frontend. It focuses on cost-efficiency by using local embeddings."),
        ("Advanced Chunking", "Semantic chunking ensures that text is split at logical boundaries, preserving the context needed for accurate answering."),
        ("Caching Strategies", "Exact match caching stores the hash of a normalized query. If the same query is seen again, the cached response is returned immediately."),
        ("Evaluation Metrics", "RAG systems are evaluated on retrieval precision, recall, and answer faithfulness. Low scores indicate a need for better chunking."),
        ("Cost Optimization", "To keep costs below 1 rupee per session, liteRAG utilizes local CPU-bound embedding models and efficient FAISS indexing."),
        ("The Future of AI Agents", "AI agents are moving towards more autonomous workflows, where they can search, plan, and execute tasks with minimal human intervention.")
    ]
    
    for title, text in content:
        page = doc.new_page()
        page.insert_text((50, 50), title, fontsize=20)
        page.insert_textbox((50, 100, 500, 300), text, fontsize=12)
        
    doc.save(filename)
    doc.close()
    print(f"Created {filename}")

if __name__ == "__main__":
    create_test_pdf()
