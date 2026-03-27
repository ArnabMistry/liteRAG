from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

load_dotenv(BASE_DIR / ".env")

from app.ingestion import PDFIngestor
from app.chunking import SemanticChunker
from app.embeddings import EmbeddingEngine
from app.retrieval import VectorStore
from app.optimization import ContextOptimizer
from app.cache import QueryCache
from app.generation import AnswerGenerator

app = FastAPI(title="liteRAG API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
embedding_engine = EmbeddingEngine()
try:
    vector_store = VectorStore(dimension=embedding_engine.dimension)
except RuntimeError as exc:
    print(f"Vector store load failed during startup: {exc}")
    vector_store = VectorStore(dimension=embedding_engine.dimension, auto_load=False)
optimizer = ContextOptimizer(token_limit=1500)
cache = QueryCache()
generator = AnswerGenerator()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

class QueryRequest(BaseModel):
    query: str

@app.get("/status")
async def get_status():
    is_indexed = vector_store.has_documents()
    latest_metadata = vector_store.metadata[0]["metadata"] if is_indexed and vector_store.metadata else {}

    return {
        "indexed": is_indexed,
        "chunks": len(vector_store.metadata),
        "source": latest_metadata.get("source"),
        "pages": latest_metadata.get("total_pages"),
    }

@app.post("/reset")
async def reset_session():
    try:
        vector_store.reset()
        cache.clear()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to reset session state.") from exc

    return {"message": "Session reset successfully", "indexed": False}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}.pdf"

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to save uploaded PDF.") from exc
    
    # Processing stages for progress feedback (emulated via logging for now)
    print(f"[{file_id}] Stage: Extracting text...")
    try:
        ingestor = PDFIngestor(str(file_path))
        docs = ingestor.extract_text_with_metadata()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to extract text from PDF.") from exc
    
    print(f"[{file_id}] Stage: Chunking...")
    chunker = SemanticChunker()
    chunks = chunker.chunk_documents(docs)
    
    print(f"[{file_id}] Stage: Generating embeddings...")
    try:
        texts = [c["text"] for c in chunks]
        embeddings = embedding_engine.generate_embeddings(texts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to generate embeddings.") from exc
    
    print(f"[{file_id}] Stage: Indexing...")
    try:
        vector_store.clear()
        vector_store.add_documents(embeddings, chunks)
        vector_store.save()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to index document.") from exc
    
    # Invalidate cache on new upload
    try:
        cache.clear()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to clear query cache.") from exc
    
    return {"message": "PDF processed and indexed successfully", "file_id": file_id, "pages": len(docs)}

@app.post("/query")
async def query_document(request: QueryRequest):
    query = request.query
    
    # 1. Check Cache
    try:
        cached_response = cache.get(query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to read query cache.") from exc

    if cached_response:
        print(f"Cache hit -> skipping LLM")
        return {"answer": cached_response, "cached": True}
    
    # 2. Retrieve
    try:
        query_emb = embedding_engine.generate_single_embedding(query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to generate query embedding.") from exc

    try:
        retrieved_chunks = vector_store.search(query_emb, k=3, threshold=0.1) # Reduced k to 3 for precision
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to search vector store.") from exc

    print(f"Chunks retrieved: {len(retrieved_chunks)}")
    
    if not retrieved_chunks:
        print("No relevant context -> skipping LLM")
        return {
            "answer": "I don't know the answer to this as the document does not contain relevant context.",
            "sources": [],
            "cached": False
        }
    
    # 3. Simple Query Fast-Path (No-LLM Mode)
    simple_prefixes = ("what is ", "what are ", "define ", "who is ", "explain ")
    is_simple_query = any(query.lower().startswith(p) for p in simple_prefixes) and len(query.split()) < 8

    if is_simple_query:
        print(f"Simple query '{query}' -> returning top chunk directly, skipping LLM")
        top_chunk_text = retrieved_chunks[0].get("text", "")
        # Minimal cleaning
        direct_answer = f"(Direct Extract) {top_chunk_text}"
        
        # 5. Cache & Return directly
        try:
            cache.set(query, direct_answer)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Failed to update query cache.") from exc

        return {
            "answer": direct_answer,
            "sources": [{
                "text": c.get("text", ""),
                "score": c.get("score", 0.0),
                "rank": c.get("rank", i + 1),
                **c.get("metadata", {})
            } for i, c in enumerate(retrieved_chunks)],
            "cached": False
        }

    # 4. Optimize Context
    context = optimizer.optimize_context(query, retrieved_chunks)
    
    if not context or len(context.strip()) < 10:
        print("No relevant context -> skipping LLM")
        return {
            "answer": "I don't know the answer to this as the document does not contain relevant context.",
            "sources": [],
            "cached": False
        }
        
    final_context_size = len(context) // 4
    print(f"Final context size: ~{final_context_size} tokens")
    
    # 5. Generate Answer
    print("Calling LLM")
    try:
        answer = generator.generate_answer(query, context)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to generate answer from LLM.") from exc
    
    # 5. Cache & Return
    try:
        cache.set(query, answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to update query cache.") from exc
    
    return {
        "answer": answer,
        "sources": [{
            "text": c.get("text", ""),
            "score": c.get("score", 0.0),
            "rank": c.get("rank", i + 1),
            **c.get("metadata", {})
        } for i, c in enumerate(retrieved_chunks)],
        "cached": False
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
