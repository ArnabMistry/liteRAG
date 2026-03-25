from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import uuid
import time
from dotenv import load_dotenv

load_dotenv()

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
vector_store = VectorStore(dimension=embedding_engine.dimension)
optimizer = ContextOptimizer(token_limit=1500)
cache = QueryCache()
generator = AnswerGenerator()

UPLOAD_DIR = "backend/data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Processing stages for progress feedback (emulated via logging for now)
    print(f"[{file_id}] Stage: Extracting text...")
    ingestor = PDFIngestor(file_path)
    docs = ingestor.extract_text_with_metadata()
    
    print(f"[{file_id}] Stage: Chunking...")
    chunker = SemanticChunker()
    chunks = chunker.chunk_documents(docs)
    
    print(f"[{file_id}] Stage: Generating embeddings...")
    texts = [c["text"] for c in chunks]
    embeddings = embedding_engine.generate_embeddings(texts)
    
    print(f"[{file_id}] Stage: Indexing...")
    vector_store.clear() # For simple version, we store one document at a time or reset
    vector_store.metadata = [] # Reset metadata
    vector_store.index = vector_store.index.__class__(vector_store.dimension) # Reset FAISS
    vector_store.add_documents(embeddings, chunks)
    vector_store.save()
    
    # Invalidate cache on new upload
    cache.clear()
    
    return {"message": "PDF processed and indexed successfully", "file_id": file_id, "pages": len(docs)}

@app.post("/query")
async def query_document(request: QueryRequest):
    query = request.query
    
    # 1. Check Cache
    cached_response = cache.get(query)
    if cached_response:
        print(f"Cache hit for query: {query}")
        return {"answer": cached_response, "cached": True}
    
    # 2. Retrieve
    query_emb = embedding_engine.generate_single_embedding(query)
    retrieved_chunks = vector_store.search(query_emb, k=5, threshold=0.1) # Threshold 0.1 for test
    
    # 3. Optimize Context
    context = optimizer.optimize_context(retrieved_chunks)
    
    # 4. Generate Answer
    answer = generator.generate_answer(query, context)
    
    # 5. Cache & Return
    cache.set(query, answer)
    
    return {
        "answer": answer,
        "sources": [c["metadata"] for c in retrieved_chunks],
        "cached": False
    }

# Patch VectorStore clear method since it wasn't in original
def vs_clear(self):
    import faiss
    self.index = faiss.IndexFlatL2(self.dimension)
    self.metadata = []
VectorStore.clear = vs_clear

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
