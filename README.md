# liteRAG

Cost-efficient Retrieval-Augmented Generation pipeline for large PDF question answering (< ₹1/session)

---

## Overview

liteRAG enables users to query large PDF documents using natural language while minimizing token usage and computational cost.

Instead of processing entire documents, the system retrieves only the most relevant context and generates accurate, grounded responses.

---

## Problem

Large documents exceed LLM context limits and significantly increase token costs.

The goal is to:

* Maintain high answer accuracy
* Reduce token consumption
* Enable scalable querying

---

## Solution

liteRAG uses a Retrieval-Augmented Generation (RAG) pipeline:

* Offline document processing and indexing
* Context-aware retrieval at query time
* Minimal context passed to the language model

---

## Architecture

```
PDF → Text → Chunking → Embeddings → Vector Store

Query → Embedding → Retrieval → Compression → LLM → Response
```

---

## Features

* Efficient PDF ingestion
* Semantic search using embeddings
* Context-aware chunking
* Token-optimized responses
* Fast local retrieval (FAISS)
* Modular pipeline

---

## Cost Strategy

Designed to operate under ₹1 per session:

* Top-K retrieval
* Context compression
* Minimal LLM calls
* Optional caching

---

## Development Phases

1. Local setup (parsing, chunking, embeddings)
2. Retrieval validation
3. Answer generation
4. Optimization (compression, caching)
5. Evaluation (accuracy, cost)

---

## Tech Stack

* Python
* FAISS
* Sentence Transformers
* LLM API

---

## Future Work

* Hybrid search
* Query rewriting
* Hierarchical retrieval
* Answer citations
* Interactive UI

---

## Author

Arnab Mistry
