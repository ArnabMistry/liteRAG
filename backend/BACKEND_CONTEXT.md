# Backend Context for liteRAG

This document describes how the current liteRAG backend works as implemented in the repository at the time of analysis. It is intended to be a single technical reference for onboarding, debugging, and tracing request execution.

## 1. System Overview

### What the backend does

The backend is a FastAPI service that lets a client:

1. Upload a PDF with `POST /upload`
2. Extract page text from that PDF
3. Split the extracted text into overlapping chunks
4. Convert each chunk into a dense embedding with a local sentence-transformers model
5. Index the embeddings in an in-memory FAISS `IndexFlatL2`
6. Persist the FAISS index and chunk metadata to disk
7. Answer user questions with `POST /query` by retrieving relevant chunks, compressing them into a smaller context string, optionally calling Gemini, and caching answers

### High-level architecture

The system is modular but still very lightweight. The modules are:

- `backend/app/main.py`: FastAPI entry point and orchestration layer
- `backend/app/ingestion.py`: PDF text extraction
- `backend/app/chunking.py`: chunk creation
- `backend/app/embeddings.py`: embedding generation
- `backend/app/retrieval.py`: FAISS-backed vector store and search
- `backend/app/optimization.py`: non-LLM context compression
- `backend/app/cache.py`: disk-backed exact-match query cache
- `backend/app/generation.py`: Gemini call wrapper

The runtime model is simple:

- One process
- One global embedding model
- One global vector store object
- One global cache object
- One global answer generator
- One active document corpus at a time in practice

### Key design decisions visible in the code

- Retrieval uses exact FAISS L2 search with `IndexFlatL2`, not approximate ANN search.
- Embeddings come from the local `sentence-transformers` model `all-MiniLM-L6-v2`.
- Chunking is word-count based, not sentence-aware or semantic-model based.
- Query caching is exact-match after normalization with `strip().lower()`.
- A new upload clears the previous index and cache instead of merging documents.
- Query handling has a "simple query" fast path that skips the LLM and returns the top retrieved chunk directly.
- Context optimization is done heuristically with sentence-level Jaccard scoring, not with an LLM summarizer.

## 2. End-to-End Flow

### Upload Flow: `POST /upload`

The `/upload` route in `backend/app/main.py` performs these steps:

1. Validate file type.
   - The route only checks `file.filename.endswith(".pdf")`.
   - Validation is filename-based, not content-based.

2. Generate a file ID.
   - A UUID string is created with `uuid.uuid4()`.
   - The file is saved as `<uuid>.pdf`.

3. Save the uploaded file to disk.
   - Target directory: `backend/data/uploads` relative to the process working directory.
   - The directory is created at startup with `os.makedirs(..., exist_ok=True)`.

4. Extract text from the PDF.
   - `PDFIngestor(file_path)` is instantiated.
   - `extract_text_with_metadata()` opens the PDF with PyMuPDF (`fitz.open`).
   - Each page is loaded individually.
   - `page.get_text("text").strip()` is used.
   - Empty pages are skipped.
   - Output is a list of page objects with:
     - `text`
     - `metadata.source`
     - `metadata.page`
     - `metadata.total_pages`

5. Chunk the extracted pages.
   - `SemanticChunker()` is instantiated with defaults:
     - `chunk_size=500`
     - `chunk_overlap=50`
   - `chunk_documents(docs)` splits each page into chunks by words.
   - For each chunk, metadata is extended with:
     - `chunk_id`
     - `sub_chunk_index`

6. Generate embeddings.
   - `texts = [c["text"] for c in chunks]`
   - `EmbeddingEngine.generate_embeddings(texts)` returns a NumPy array.
   - The embedding model dimension is discovered at startup and passed into `VectorStore`.

7. Reset the vector store.
   - `vector_store.clear()` is called.
   - `vector_store.metadata = []` is set again explicitly.
   - `vector_store.index = vector_store.index.__class__(vector_store.dimension)` reinitializes the FAISS index.
   - This means uploads replace the previous searchable corpus rather than append to it.

8. Add documents to the vector store.
   - `vector_store.add_documents(embeddings, chunks)` adds vectors to FAISS and stores chunk payloads in memory.

9. Persist index and metadata.
   - `vector_store.save()` writes:
     - FAISS binary index to `backend/data/faiss_index.bin`
     - metadata JSON to `backend/data/metadata.json`

10. Invalidate the query cache.
    - `cache.clear()` empties the in-memory map and rewrites the cache file.

11. Return response.
    - The route returns:
      - success message
      - generated `file_id`
      - number of extracted non-empty pages

### Query Flow: `POST /query`

The `/query` route performs these steps:

1. Read the request body.
   - Request model: `QueryRequest`
   - Required field: `query: str`

2. Check the cache first.
   - `cache.get(query)` normalizes the query with `strip().lower()`.
   - The normalized query is hashed with MD5.
   - If found, the route returns immediately with:
     - `answer`
     - `cached: True`
   - Cached responses do not return `sources`.

3. Generate a query embedding.
   - `EmbeddingEngine.generate_single_embedding(query)` returns a single vector.

4. Retrieve nearest chunks.
   - `vector_store.search(query_emb, k=3, threshold=0.1)` is used.
   - The vector store runs FAISS L2 search against the current in-memory index.
   - Returned distances are converted to similarity scores using:
     - `score = 1.0 / (1.0 + dist)`
   - Any chunk below the threshold is dropped.
   - Retrieved chunks include:
     - `text`
     - `metadata`
     - `score`
     - `rank`

5. Handle empty retrieval.
   - If no chunks survive thresholding, the route returns:
     - a fallback "I don't know" style answer
     - `sources: []`
     - `cached: False`
   - In this path the LLM is not called.

6. Check the simple-query fast path.
   - A query is considered simple if:
     - it starts with one of:
       - `what is `
       - `what are `
       - `define `
       - `who is `
       - `explain `
     - and contains fewer than 8 whitespace-separated terms
   - If true:
     - the top retrieved chunk text is returned directly
     - the answer is prefixed with `(Direct Extract)`
     - the LLM is skipped
     - the result is cached
     - `sources` are returned from the retrieved chunks

7. Optimize context for the LLM.
   - `ContextOptimizer.optimize_context(query, retrieved_chunks)` scores sentences from the retrieved chunks.
   - Only selected sentences are kept under an estimated token budget.
   - The optimizer in `main.py` is initialized with `token_limit=1500`.

8. Handle empty or too-short context.
   - If the optimized context is missing or shorter than 10 stripped characters, the route returns the same no-context fallback answer and skips the LLM.

9. Call the LLM.
   - `AnswerGenerator.generate_answer(query, context)` creates a prompt and calls Gemini through `google.genai`.

10. Cache the LLM answer.
    - `cache.set(query, answer)` stores the final answer under the normalized-query MD5 key.

11. Return response with sources.
    - The response includes:
      - `answer`
      - `sources`
      - `cached: False`

## 3. File-by-File Breakdown

### `backend/app/main.py`

#### Purpose

Acts as the HTTP entry point and request orchestrator. It wires together all backend components and owns the upload and query flows.

#### Key objects created at import time

- `embedding_engine = EmbeddingEngine()`
- `vector_store = VectorStore(dimension=embedding_engine.dimension)`
- `optimizer = ContextOptimizer(token_limit=1500)`
- `cache = QueryCache()`
- `generator = AnswerGenerator()`

These are module-level singletons created when `main.py` is imported.

#### Main responsibilities

- Configure the FastAPI app
- Configure permissive CORS with `allow_origins=["*"]`
- Create the upload directory
- Define the request model `QueryRequest`
- Implement `/upload`
- Implement `/query`
- Monkey-patch a `clear` method onto `VectorStore`

#### Internal logic details

- `load_dotenv()` is called before importing application modules, so `.env` values are loaded into the process environment if present.
- `UPLOAD_DIR` is set to `backend/data/uploads`.
- The route handlers use `print(...)` for progress and tracing rather than structured logging.
- `VectorStore.clear` is not defined in `retrieval.py`; instead a replacement function `vs_clear` is attached dynamically in `main.py`.

#### Important data flow

- `/upload` sends extracted page dicts into chunking, chunk dicts into embeddings, embeddings plus chunk dicts into the vector store.
- `/query` sends the raw query through cache lookup, embedding generation, vector search, optional context optimization, optional LLM generation, then cache write-back.

#### Dependencies

- FastAPI, Pydantic, CORS middleware, dotenv
- All application modules in `backend/app`
- `shutil`, `os`, `uuid`, `time` from the standard library

### `backend/app/ingestion.py`

#### Purpose

Extracts page-level text and attaches per-page metadata from a PDF.

#### Main class

- `PDFIngestor`

#### Key methods

- `__init__(file_path: str)`
  - Stores the PDF path
  - Derives `source_name` with `os.path.basename(file_path)`

- `extract_text_with_metadata() -> List[Dict]`
  - Opens the PDF with `fitz.open`
  - Iterates through every page index
  - Calls `page.get_text("text")`
  - Strips whitespace
  - Skips empty pages
  - Returns one list item per non-empty page

#### Output structure

Each returned item is shaped like:

```python
{
    "text": "<page text>",
    "metadata": {
        "source": "<uploaded filename on disk>",
        "page": <1-based page number>,
        "total_pages": <total page count>
    }
}
```

#### Dependencies

- PyMuPDF via `fitz`
- `typing`
- `os`

### `backend/app/chunking.py`

#### Purpose

Converts page-level text objects into overlapping word-based chunks while preserving provenance metadata.

#### Main class

- `SemanticChunker`

The name implies semantic chunking, but the implementation is purely word-window based.

#### Key methods

- `__init__(chunk_size=500, chunk_overlap=50)`
  - Configures the target words per chunk and overlap size.

- `split_text(text: str) -> List[str]`
  - Splits on whitespace using `text.split()`
  - Builds windows of `chunk_size` words
  - Advances by `chunk_size - chunk_overlap`
  - Stops after the last window

- `chunk_documents(documents: List[Dict]) -> List[Dict]`
  - Loops through each page object
  - Splits its `text`
  - Extends the page metadata for each chunk
  - Adds:
    - `chunk_id`: monotonically increasing across the whole upload
    - `sub_chunk_index`: zero-based chunk index within the source page

#### Output structure

```python
{
    "text": "<chunk text>",
    "metadata": {
        "source": "<uuid>.pdf",
        "page": 5,
        "total_pages": 79,
        "chunk_id": 17,
        "sub_chunk_index": 1
    }
}
```

#### Dependencies

- `typing`
- `re` is imported but not used

### `backend/app/embeddings.py`

#### Purpose

Provides a thin wrapper around a sentence-transformers embedding model.

#### Main class

- `EmbeddingEngine`

#### Key methods

- `__init__(model_name="all-MiniLM-L6-v2")`
  - Loads the model immediately
  - Stores embedding dimensionality in `self.dimension`

- `generate_embeddings(texts: List[str]) -> np.ndarray`
  - Encodes a batch of strings
  - Returns a NumPy array

- `generate_single_embedding(text: str) -> np.ndarray`
  - Encodes a single string by wrapping it in a one-element list
  - Returns the first vector

#### Data properties

- Default model: `all-MiniLM-L6-v2`
- The model dimension is discovered dynamically; for this model it is typically 384.

#### Dependencies

- `sentence_transformers.SentenceTransformer`
- `numpy`
- `typing`

### `backend/app/retrieval.py`

#### Purpose

Stores chunk embeddings, performs nearest-neighbor search, and persists retrieval state to disk.

#### Main class

- `VectorStore`

#### Constructor behavior

`VectorStore.__init__` creates:

- `self.dimension`
- `self.index_path` defaulting to `backend/data/faiss_index.bin`
- `self.metadata_path` defaulting to `backend/data/metadata.json`
- `self.index = faiss.IndexFlatL2(dimension)`
- `self.metadata = []`

The constructor does not automatically load any saved index from disk.

#### Key methods

- `add_documents(embeddings, metadata)`
  - Verifies `embeddings.shape[0] == len(metadata)`
  - Adds `float32` vectors to FAISS
  - Extends `self.metadata` with the supplied chunk objects

- `search(query_embedding, k=5, threshold=0.6)`
  - Reshapes the query vector to `(1, dimension)`
  - Converts to `float32`
  - Runs `self.index.search(query_embedding, k)`
  - Iterates through `(distance, index)` pairs
  - Converts distance to a score with `1 / (1 + dist)`
  - Drops anything below `threshold`
  - Rebuilds a retrieval payload from `self.metadata[idx]`
  - Assigns ranks in the order retained
  - Calls `_log_retrieval(results)`

- `_log_retrieval(results)`
  - Prints score, page, chunk ID, and a 100-character preview

- `save()`
  - Writes the FAISS index to disk
  - Serializes the metadata list to JSON

- `load()`
  - Reads both files if they exist
  - Not invoked by `main.py`

#### Search result format

```python
{
    "text": "<chunk text>",
    "metadata": {...},
    "score": 0.42,
    "rank": 1
}
```

#### Dependencies

- `faiss`
- `numpy`
- `json`
- `os`
- `typing`

### `backend/app/optimization.py`

#### Purpose

Builds a smaller context string from retrieved chunks without calling an LLM.

#### Main class

- `ContextOptimizer`

#### Core algorithm

1. Tokenize the query into a lowercase word set using `re.findall(r'\w+', ...)`
2. For every retrieved chunk:
   - replace newlines with spaces
   - split the chunk into sentences using `re.split(r'(?<=[.!?])\s+', ...)`
   - ignore sentences shorter than 10 characters
3. Score each sentence by:
   - sentence-query Jaccard similarity
   - plus `chunk_score * 0.1`
4. Sort sentences by descending score
5. Deduplicate by sentence-token Jaccard overlap
   - if overlap with an already selected sentence is greater than `0.6`, skip it
6. Estimate token cost as `len(sentence_text) // 4`
7. Keep adding sentences until the configured token limit is reached
8. Resort selected sentences by:
   - ascending page number
   - descending score within a page
9. Build a context string with page headers like `[Page 7]`

#### Important behavior

- The optimizer is sentence-focused, not chunk-focused.
- It can reorder content relative to the original retrieval ranking.
- The token estimate is character-based rather than model-token based.
- Page order is favored in the final assembly to preserve rough document flow.

#### Dependencies

- `re`
- `typing`

### `backend/app/cache.py`

#### Purpose

Caches final answers on disk so repeated identical normalized queries can bypass retrieval and generation.

#### Main class

- `QueryCache`

#### Key methods

- `__init__(cache_path="backend/data/query_cache.json")`
  - Sets the cache file path
  - Loads the cache from disk immediately

- `_load_cache()`
  - Reads JSON if the file exists
  - Returns `{}` on any read/parse failure

- `_normalize_query(query)`
  - Returns `query.strip().lower()`

- `get(query)`
  - Normalizes the query
  - MD5-hashes it
  - Returns the cached string answer if present

- `set(query, response)`
  - Normalizes and hashes the query
  - Stores the response string
  - Writes the full cache back to disk

- `clear()`
  - Replaces the in-memory cache with an empty dict
  - Writes the empty cache to disk

#### Cache value format

The cache file is a JSON object:

```json
{
  "<md5-of-normalized-query>": "<answer text>"
}
```

Only answers are cached. Source chunks, timestamps, model metadata, and upload IDs are not cached.

#### Dependencies

- `json`
- `os`
- `hashlib`
- `typing`

### `backend/app/generation.py`

#### Purpose

Encapsulates the Gemini call used to turn optimized context plus user query into a grounded answer.

#### Main class

- `AnswerGenerator`

#### Key methods

- `__init__(model_name="gemini-3-flash-preview")`
  - Creates a `genai.Client()`
  - Stores the model name
  - Relies on environment configuration for authentication

- `generate_answer(query, context)`
  - Returns a fixed fallback if `context` is empty
  - Otherwise constructs a text prompt:
    - "Answer ONLY from context. If not found, say you don't know."
    - followed by `Context:`
    - followed by the context string
    - followed by `Q: <query>`
    - followed by `A:`
  - Calls `self.client.interactions.create(model=..., input=prompt)`
  - Returns `interaction.outputs[-1].text`

#### Dependencies

- `google.genai`
- `os` is imported but unused
- `typing` imports are unused

### `backend/tests/run_validation.py`

#### Purpose

A manual retrieval validation script rather than a full automated test suite.

#### What it does

1. Adds the current working directory to `sys.path`
2. Imports the backend modules directly
3. Uses `backend/data/test_eval.pdf`
4. Creates that PDF if missing by importing `backend.tests.create_test_pdf`
5. Reads QA pairs from `backend.tests.eval_data`
6. Runs ingestion, chunking, embedding, and retrieval
7. Checks whether the top result page and expected text match each test case

#### Important observation

The repository snapshot analyzed here contains `backend/tests/run_validation.py`, but not the referenced:

- `backend/tests/eval_data.py`
- `backend/tests/create_test_pdf.py`

So the validation script references support files that are not present in the current visible tree.

### `backend/requirements.txt`

Declares the main backend dependencies:

- `fastapi`
- `uvicorn`
- `python-multipart`
- `pymupdf`
- `sentence-transformers`
- `faiss-cpu`
- `google-genai`
- `numpy`
- `pydantic`
- `pytest`
- `python-dotenv`

### `backend/.env`

Contains `GOOGLE_API_KEY` in plaintext form in the repository working tree.

From an application behavior perspective:

- `main.py` calls `load_dotenv()`
- `AnswerGenerator` relies on environment-based authentication
- The backend therefore expects this variable to exist before the first Gemini call

## 4. Data Structures

### Page document structure

Produced by `PDFIngestor.extract_text_with_metadata()`:

```python
{
    "text": "<full page text>",
    "metadata": {
        "source": "<uuid>.pdf",
        "page": <1-based page>,
        "total_pages": <total pages in source PDF>
    }
}
```

### Chunk structure

Produced by `SemanticChunker.chunk_documents()`:

```python
{
    "text": "<chunk text>",
    "metadata": {
        "source": "<uuid>.pdf",
        "page": <1-based page>,
        "total_pages": <total pages>,
        "chunk_id": <global running integer>,
        "sub_chunk_index": <0-based index inside the source page>
    }
}
```

### Embedding format

- Type: NumPy array
- Dtype used for FAISS insertion: `float32`
- Batch shape: `(num_chunks, embedding_dimension)`
- Query shape before search: reshaped to `(1, embedding_dimension)`

### Retrieval result format

Produced by `VectorStore.search()`:

```python
{
    "text": "<chunk text>",
    "metadata": {
        "source": "...",
        "page": ...,
        "total_pages": ...,
        "chunk_id": ...,
        "sub_chunk_index": ...
    },
    "score": <float>,
    "rank": <1-based rank among retained results>
}
```

### FAISS index usage

- Index type: `faiss.IndexFlatL2`
- Storage mode: in-memory during runtime, binary on disk after `save()`
- Persistence path: `backend/data/faiss_index.bin` relative to the process working directory
- Matching metadata is stored separately in `backend/data/metadata.json`

The metadata list order is the lookup bridge between FAISS vector row index and original chunk payload:

- vector at row `i` in FAISS corresponds to `self.metadata[i]`

### Metadata file format

`metadata.json` is a JSON array where each element is a chunk object:

```json
[
  {
    "text": "...",
    "metadata": {
      "source": "0e697a9e-127b-437a-ad27-b88c217f4a63.pdf",
      "page": 2,
      "total_pages": 79,
      "chunk_id": 1,
      "sub_chunk_index": 0
    }
  }
]
```

### Cache format

`query_cache.json` is a JSON object:

```json
{
  "md5(normalized_query)": "cached answer string"
}
```

## 5. Retrieval System

### How similarity search works

The retrieval layer uses exact FAISS L2 distance search:

1. Convert the query embedding to shape `(1, d)` and `float32`
2. Ask FAISS for the `k` nearest vectors
3. Receive:
   - `distances`
   - `indices`
4. For each result:
   - ignore `idx == -1`
   - compute a custom similarity score from the FAISS distance
   - keep only results whose score meets the threshold

### Scoring mechanism

FAISS returns squared L2 distances for `IndexFlatL2`.

The application maps distance to a similarity-like value with:

```python
score = 1.0 / (1.0 + dist)
```

Properties of this mapping:

- distance `0` maps to score `1.0`
- larger distances move the score toward `0`
- the mapping is monotonic but not cosine similarity
- score meaning is application-defined, not a native FAISS confidence

### Threshold behavior

`VectorStore.search` defaults to `threshold=0.6`, but `/query` overrides it with `threshold=0.1`.

So in live query handling:

- top `k=3` neighbors are requested
- anything with score below `0.1` is dropped

Because the threshold is low, most practical filtering pressure comes from the top-k boundary rather than aggressive score filtering.

### Ranking logic

- FAISS returns nearest vectors in ascending distance order
- The code iterates through that order
- Retained results are appended sequentially
- `rank` is assigned as `len(results) + 1`

So the output rank is the order after threshold filtering, which generally still matches nearest-first ordering.

### Retrieval logging

Every search prints a scored retrieval log with:

- rank
- score
- page
- chunk ID
- text preview

This is useful for debugging relevance behavior manually from console output.

## 6. Context Optimization

### How context is built

`ContextOptimizer.optimize_context()` converts retrieved chunks into a compact context string.

The process is:

1. Tokenize the query
2. Split each retrieved chunk into sentences
3. Compute sentence-query Jaccard relevance
4. Add a small retrieval-score boost
5. Sort all candidate sentences by score
6. Remove highly overlapping sentences
7. Stop when the estimated token budget is reached
8. Reorder by page
9. Concatenate into one context string with page markers

### Sentence scoring

For each sentence:

- `score = Jaccard(query_tokens, sentence_tokens) + (chunk_score * 0.1)`

This means lexical overlap with the query is the primary driver, while retrieval score only nudges the ranking.

### Token limiting strategy

- Budget source: `ContextOptimizer(token_limit=1500)` in `main.py`
- Per-sentence estimate: `len(sentence_text) // 4`
- The optimizer accumulates sentence estimates until adding another one would exceed the limit
- When the next sentence would exceed the budget, selection stops immediately

### Ordering and filtering

Filtering stages:

- sentences under 10 characters are dropped
- near-duplicate sentences are dropped if token-set Jaccard overlap with a selected sentence exceeds `0.6`

Ordering behavior:

- selection order is relevance-first
- final presentation order is page-first

### Result format

The final context is a single string similar to:

```text
[Page 3] Sentence one. Sentence two.
[Page 4] Sentence three.
```

## 7. Caching System

### How queries are normalized

The cache normalizes a query with:

```python
query.strip().lower()
```

This removes leading and trailing whitespace and makes matching case-insensitive, but it does not:

- collapse repeated internal whitespace
- remove punctuation
- stem words
- canonicalize paraphrases

### Cache storage format

Storage key:

- `md5(normalized_query.encode()).hexdigest()`

Storage value:

- answer string only

Persistence:

- in-memory dict during runtime
- full JSON rewrite to `backend/data/query_cache.json` on every `set()` and `clear()`

### Invalidation logic

The cache is invalidated only in one explicit place:

- after a successful `/upload`, `cache.clear()` is called

That reflects the backend's one-document-at-a-time model. Cached answers from an older document are intentionally discarded whenever a new document is indexed.

## 8. LLM Integration

### How prompts are constructed

The prompt sent to Gemini is:

```text
Answer ONLY from context. If not found, say you don't know.

Context:
<optimized context>

Q: <user query>
A:
```

This is a plain string prompt, not a structured system/user message schema in the application code.

### When the LLM is called

The LLM is called only if all of the following are true:

- cache miss
- at least one retrieved chunk passes thresholding
- query does not match the simple-query fast path
- optimized context is non-empty and at least 10 stripped characters long

Otherwise the system either:

- serves from cache
- returns a direct extract
- returns a no-context fallback

### Current safeguards against misuse

The implemented safeguards are lightweight:

- instruction to answer only from provided context
- instruction to say "don't know" if context does not contain the answer
- no LLM call at all when retrieval returns no relevant chunks

What is not present in the current code:

- no moderation step
- no explicit prompt-injection filtering on retrieved text
- no output post-processing
- no citation enforcement beyond returning `sources` separately in the API response
- no retry or fallback model logic

## 9. Logging and Debugging

### What is currently logged

The backend uses `print(...)` statements rather than a logging framework.

Observed log points include:

- upload stage progress:
  - extracting text
  - chunking
  - generating embeddings
  - indexing
- cache hits in `/query`
- number of retrieved chunks
- no-context short-circuit behavior
- simple-query fast-path activation
- final optimized context size estimate
- "Calling LLM"
- retrieval score logs from `VectorStore._log_retrieval`

### How to trace execution

A practical trace order for a request is:

1. `main.py` route entry
2. upload/query route-level prints
3. `retrieval.py` scored retrieval log
4. cache file, metadata file, and FAISS file on disk if persistence is relevant

Useful runtime artifacts:

- uploaded PDFs under `backend/data/uploads`
- FAISS binary file
- metadata JSON
- query cache JSON

### Debugging characteristics

- The code is easy to follow because orchestration is explicit and linear.
- There is very little hidden abstraction.
- Failures from external libraries will mostly surface directly because exception handling is minimal.

## 10. Current Limitations

This section describes limitations and weak points visible in the current implementation. It is descriptive, not prescriptive.

### Single-document operating model

- `/upload` clears the existing vector store before indexing the new file.
- The backend therefore behaves as a one-upload-at-a-time system, not a multi-document knowledge base.
- Existing persisted data is overwritten by the next upload.

### Persistence is incomplete on startup

- `VectorStore.save()` exists and persists state to disk.
- `VectorStore.load()` also exists.
- `main.py` never calls `vector_store.load()`.
- As a result, persisted index/metadata files are not automatically reloaded when the API process starts.
- Query behavior after restart depends on whatever is currently in memory, which starts as an empty FAISS index and empty metadata list.

### Working-directory-sensitive file paths

- Most paths are written as `backend/data/...`.
- If the server is launched from the repository root, those paths resolve to `C:\\builds\\liteRAG\\backend\\data\\...`.
- If the server is launched from the `backend` directory as suggested in the README, those same paths resolve to `C:\\builds\\liteRAG\\backend\\backend\\data\\...`.
- The analyzed repository already contains data under `backend/backend/data`, which shows this path dependency in practice.
- This means runtime storage location depends on the current working directory.

### Monkey-patched vector-store clearing

- `VectorStore.clear` is added dynamically in `main.py` instead of being defined in `retrieval.py`.
- The `/upload` flow then also resets `metadata` and recreates the FAISS index again immediately afterward.
- Clearing behavior is therefore split and partially duplicated across files.

### Minimal file validation

- `/upload` only checks whether the filename ends with `.pdf`.
- The code does not verify MIME type or actual file content before attempting extraction.

### No defensive error handling around major external operations

- PDF parsing errors from PyMuPDF are not caught in the route.
- Embedding model failures are not caught.
- FAISS write errors are not caught.
- Gemini API errors are not caught.
- Cache JSON write failures are not caught.

### Retrieval score semantics are heuristic

- FAISS returns squared L2 distance.
- The backend converts that to a score with `1 / (1 + dist)`.
- This is a custom transformation, not a calibrated relevance probability.
- Threshold tuning therefore depends on empirical behavior rather than a standardized similarity metric.

### Chunking is simple and page-bound

- Chunking uses whitespace word windows only.
- It does not preserve sentence boundaries.
- It does not use headings, layout, section structure, or semantic boundaries.
- Chunking is performed independently per page, so content spanning page boundaries is not merged.

### Context optimizer is lexical and approximate

- Relevance is based on Jaccard overlap over token sets.
- Token budget is estimated by `characters // 4`, not model-token counting.
- Sentence splitting is regex-based and can be brittle for abbreviations or unusual formatting.
- Final ordering by page may reduce pure relevance ordering in favor of document order.

### Cache stores only answer strings

- Cached responses omit retrieval sources in the cache payload.
- When cache hits occur, the API returns only `answer` and `cached: True`.
- Cached answers are therefore detached from the source context that originally produced them.

### Simple-query path bypasses generation safeguards

- Queries matching the simple-prefix rule return the top chunk text directly.
- This path bypasses answer synthesis and any LLM-level instruction following.
- The returned text may therefore be longer, less focused, or less directly responsive than the LLM path.

### Logging is unstructured

- The service uses `print(...)` statements.
- There are no log levels, request IDs, structured fields, or centralized logger configuration.
- Traceability exists, but only as console text.

### Test surface is incomplete in the analyzed tree

- `run_validation.py` references `backend.tests.eval_data` and `backend.tests.create_test_pdf`.
- Those files were not present in the visible repository snapshot analyzed here.
- The visible automated validation surface is therefore incomplete.

### Sensitive configuration is stored in-repo

- The checked-in working tree contains a plaintext `GOOGLE_API_KEY` in `backend/.env`.
- From a backend audit perspective, this is a security-sensitive operational characteristic of the current repository state.

## 11. Operational Notes and Runtime Assumptions

### Startup assumptions

- The sentence-transformers model can be loaded successfully at import time
- FAISS is available locally
- A valid Google API key is available before generation
- The process has write access to the resolved `backend/data/...` paths

### In-memory vs persisted state

In-memory state used during requests:

- embedding model object
- FAISS index object
- metadata list
- cache dict
- Gemini client

Persisted state written by the backend:

- uploaded PDF binaries
- FAISS binary index
- metadata JSON
- query cache JSON

### Observable API behavior summary

`POST /upload`

- accepts only files whose names end with `.pdf`
- stores the uploaded file
- extracts text, chunks, embeds, indexes, persists, clears cache
- returns success message, file ID, and page count

`POST /query`

- checks cache first
- embeds query
- retrieves top 3 chunks above threshold 0.1
- returns a fallback if nothing relevant is found
- returns a direct extract for certain short definitional queries
- otherwise compresses context and calls Gemini
- caches the final answer

## 12. Source-of-Truth Summary

The current backend is a compact, linear RAG implementation optimized for simplicity over breadth. Its behavior is best understood as:

- page text extraction with PyMuPDF
- overlapping word-window chunking
- local sentence-transformer embeddings
- exact FAISS L2 retrieval
- heuristic sentence-level context compression
- optional Gemini generation
- exact-match answer caching
- single-upload corpus replacement

That combination makes the codebase easy to read and trace, but it also means system behavior depends heavily on process working directory, import-time initialization, and the currently loaded in-memory document state.
