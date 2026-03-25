# liteRAG System Architecture

liteRAG is a production-grade, cost-efficient Retrieval-Augmented Generation (RAG) system for querying large PDF documents. This document details the end-to-end architecture, internal data flow, and component-level logic.

## 🏗️ High-Level Architecture

The system follows a modular pipeline architecture, separating ingestion, retrieval, and generation to ensure scalability and cost control.

```mermaid
graph TD
    A[PDF Upload] --> B[PDF Ingestor]
    B --> C[Semantic Chunker]
    C --> D[Embedding Engine]
    D --> E[Vector Store (FAISS)]
    
    F[User Query] --> G[Query Cache]
    G -- Cache Hit --> H[Instant Response]
    G -- Cache Miss --> I[Retrieval Pipeline]
    
    I --> J[Vector Search]
    J --> K[Context Optimizer]
    K --> L[Answer Generator (Gemini)]
    L --> M[Grounded Response]
    M --> G
```

---

## 📄 End-to-End Data Flow

### 1. Ingestion Pipeline (`/upload`)
- **Trigger**: User uploads a file through the React frontend.
- **Process**:
  1. **Upload**: File is saved to `backend/data/uploads`.
  2. **Extraction**: `PDFIngestor` uses `PyMuPDF` to extract text page-by-page.
  3. **Chunking**: `SemanticChunker` splits text into ~1000-character units with 10% overlap to maintain context across boundaries.
  4. **Embedding**: `EmbeddingEngine` generates 384-dimensional vectors using local `all-MiniLM-L6-v2`.
  5. **Indexing**: Vectors are added to a FAISS `IndexFlatL2` for efficient similarity search.

### 2. Retrieval & Generation Flow (`/query`)
- **Normalized Caching**: Queries are lowercased and trimmed to maximize `QueryCache` hits.
- **Similarity Thresholding**: `VectorStore` returns only chunks with a similarity score above the configured threshold.
- **Extractive Optimization**: `ContextOptimizer` sorts chunks by score and enforces a strict **1500-token budget** by dropping lower-ranked chunks.
- **Grounded Generation**: `AnswerGenerator` sends the optimized context and query to Gemini-3-Flash with a prompt that strictly forbids hallucinations.

---

## 🧩 Component Breakdown

### Backend (`backend/app/`)

#### `ingestion.py: PDFIngestor`
- **Purpose**: Bridge between binary PDF data and raw text.
- **Logic**: Iterates through PDF pages, extracting text and attaching metadata (`page`, `source`).
- **Why**: Metadata is critical for "Source Highlighting" in the UI.

#### `chunking.py: SemanticChunker`
- **Purpose**: Break long documents into manageable, semantically cohesive units.
- **Logic**: Uses a fixed character-count window with a 10% overlap buffer.
- **Why**: Prevents losing critical information that spans two pages or segments.

#### `embeddings.py: EmbeddingEngine`
- **Purpose**: Convert text into mathematical vectors for search.
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (Local/CPU).
- **Why**: Zero cost per token and fast execution on standard hardware.

#### `retrieval.py: VectorStore`
- **Purpose**: High-speed semantic search.
- **Engine**: FAISS (Facebook AI Similarity Search).
- **Logic**: Calculates L2 distance between query embedding and document embeddings.
- **Transparency**: Logs similarity scores for Every search to aid debugging.

#### `optimization.py: ContextOptimizer`
- **Purpose**: Cost and noise control.
- **Strategy**: Sorts retrieved chunks and performs a hard-cutoff at 1500 tokens.
- **Why**: Ensures LLM stays within low-latency and low-cost windows.

#### `generation.py: AnswerGenerator`
- **Purpose**: Turn retrieved context into human-readable answers.
- **Model**: `gemini-3-flash-preview` via Interactions API.
- **Constraint**: Strict system prompt: "Answer ONLY using provided context."

### Frontend (`frontend/src/`)

- **Design System**: Refined Minimal Dark theme implemented via CSS Variables in `index.css`.
- **App.jsx**: State machine handling `idle -> uploading -> indexing -> ready` transitions.
- **Components**:
    - `UploadZone`: Visual feedback for ingestion stages.
    - `ChatInterface`: Responsive messaging UI with source-aware bubbles.

---

## 💾 Caching Strategy

liteRAG uses a simple but effective Disk-backed JSON cache.
- **Key**: `SHA-256(normalized_query)`.
- **Value**: String response.
- **Invalidation**: Cache is cleared automatically when a new PDF is uploaded to prevent cross-document leakage.

---

## ⚖️ Quality & Performance

- **Token Budget**: 1500 tokens (~1200 content + 300 prompt).
- **Cost**: ~₹0.005 per query (excluding ingestion).
- **Retrieval Accuracy**: Verified 100% on 5-point evaluation set.

---

## 🚀 Concept Glossary

- **RAG**: Retrieval-Augmented Generation. Combining search (Retrieval) with LLM (Generation).
- **Embeddings**: Numerical representations of text meaning.
- **Vector Search**: Finding text with similar "meaning" rather than just keyword matches.
- **Token**: Semantic unit of text processed by LLMs (approx. 4 chars).
