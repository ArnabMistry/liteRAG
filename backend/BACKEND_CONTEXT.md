# Backend Context for liteRAG

This document is a code-verified description of the current backend implementation in `backend/app` and `backend/tests`. It is intended to be a single source of truth for onboarding, debugging, and reasoning about runtime behavior.

## 1. System Overview

### What the system does now

The backend is a FastAPI service that supports a single-document Retrieval-Augmented Generation workflow:

- `POST /upload` ingests one PDF, extracts page text, chunks it, embeds the chunks, indexes them in FAISS, persists the index and metadata, and clears prior query cache state.
- `POST /query` processes a natural-language question through query classification, hybrid retrieval, fusion, reranking, adaptive filtering, sentence-level context optimization, confidence scoring, and either direct extraction, Gemini generation, or refusal.
- `GET /status` reports whether an indexed corpus exists.
- `POST /reset` removes the persisted index and cache and returns the system to an empty state.

The system currently behaves as a single active-corpus assistant rather than a multi-document knowledge base. A new upload replaces the previously indexed corpus.

### Evolution summary

The backend has evolved from a simple dense-only FAISS search service into a multi-stage RAG pipeline with:

- semantic chunking that respects sentence and paragraph boundaries
- hybrid retrieval using dense search plus keyword search
- Reciprocal Rank Fusion
- a modular reranking layer with a cross-encoder primary path and heuristic fallback
- sentence-level context compression
- confidence-aware answer control flow
- exact and semantic caching
- a validation harness with labeled queries and retrieval metrics
- recall-oriented soft fallback behavior when strict selection would otherwise reject all candidates

### Key architectural principles

- Keep the architecture lightweight: FastAPI, FAISS, sentence-transformers, and Gemini remain the core stack.
- Separate retrieval stages explicitly: retrieval, fusion, reranking, filtering, and context optimization are distinct steps.
- Prefer grounded answers over free-form generation: Gemini is prompted to answer only from provided context.
- Persist state on disk: the vector index, metadata, and cache survive process restarts.
- Preserve explainability: responses include sources, pages, confidence, and reasoning summaries.
- Bias toward safety at the answer layer, but the current recall-oriented implementation now allows low-confidence fallback generation when partial context exists.

## 2. End-to-End Flow

### Upload Pipeline

`POST /upload` in `backend/app/main.py` performs the following steps:

1. Validate the uploaded filename by checking `.pdf` suffix.
2. Save the upload to `backend/data/uploads/<uuid>.pdf`.
3. Extract text with `PDFIngestor.extract_text_with_metadata()`.
4. Chunk extracted page text with `SemanticChunker.chunk_documents()`.
5. Generate embeddings with `EmbeddingEngine.generate_embeddings()`.
6. Clear the current vector store.
7. Add the new embeddings and chunk metadata to the vector store.
8. Persist FAISS index and metadata to disk.
9. Clear the query cache so old answers cannot survive across document replacement.
10. Return a success payload including generated `file_id` and page count.

### Query Pipeline

`POST /query` in `backend/app/main.py` currently behaves as:

```text
Query
 -> Query Processing
 -> Cache Check
 -> Hybrid Retrieval
 -> Fusion (RRF)
 -> Reranking
 -> Candidate Filtering
 -> Optional Soft Fallback
 -> Context Optimization
 -> Confidence Calculation
 -> LLM Decision or Direct Extract
 -> Answer Generation or Refusal
 -> Caching
```

Step-by-step:

1. Build a query package with `QueryProcessor.build_query_package()`.
2. Derive:
   - `embedding_query`: usually the raw query, except for a few hand-authored expansions
   - `keyword_query`: always the raw user query
   - `query_type`, classification confidence, and behavior flags
3. Generate a dense query embedding from `embedding_query`.
4. Check the cache:
   - exact lookup by normalized query hash
   - semantic lookup by cosine similarity over stored query embeddings
5. If no cache hit exists, call `VectorStore.hybrid_search()`:
   - dense FAISS retrieval
   - BM25-style keyword retrieval
   - RRF fusion
   - reranking
   - adaptive final filtering
   - optional soft fallback if strict filtering yields nothing
6. Compute a confidence payload from retrieved chunk scores and lexical support.
7. Build a source payload from retrieved chunks for transparency.
8. Run `ContextOptimizer.build_context_package()` to construct sentence-level context.
9. If sentence selection collapses to nothing but retrieved chunks exist, build a weak fallback context from the top retrieved chunk(s) and force low confidence.
10. Decide answer path:
    - direct extract for some simple high-confidence definition/fact queries
    - Gemini generation for grounded context
    - refusal when context is too weak or absent
11. Prefix low-confidence generated answers with explicit uncertainty text.
12. Cache the final response payload, including answer, sources, confidence, reasoning summary, pages, query type, and optionally query embedding.

## 3. Component Breakdown

### `backend/app/main.py`

Purpose:

- FastAPI entry point
- lifecycle setup for global backend components
- orchestration for upload, query, status, and reset endpoints

Key global objects:

- `EmbeddingEngine()`
- `VectorStore(dimension=embedding_engine.dimension)`
- `ContextOptimizer(token_limit=800)`
- `QueryCache()`
- `AnswerGenerator()`
- `QueryProcessor()`

Important endpoint behavior:

- `GET /status`
  - returns whether any indexed corpus exists
  - also returns chunk count and metadata from the first chunk when available
- `POST /reset`
  - calls `vector_store.reset()` and `cache.clear()`
- `POST /upload`
  - replaces the active corpus
- `POST /query`
  - runs the full retrieval and answer pipeline

Important helper functions:

- `build_confidence_payload()`
  - derives confidence from rerank scores, cross-encoder scores, lexical overlap, dense support, keyword support, and source count
- `apply_confidence_overrides()`
  - forces low confidence when soft fallback retrieval was used
- `should_skip_llm()`
  - decides when generation should be blocked
  - currently returns `False` for low-confidence context as long as minimal weak-context budget exists, which enables cautious fallback generation
- `build_sources_payload()`
  - creates structured source entries returned to the client
- `build_direct_answer()`
  - returns a direct extract for simple high-confidence queries
- `build_weak_context_package()`
  - constructs context from top retrieved chunks when sentence-level compression yields no usable sentences
- `apply_uncertainty_prefix()`
  - marks low-confidence generated answers explicitly

Data flow:

- query package -> cache -> retrieval telemetry -> confidence -> context -> decision -> response payload -> cache

### `backend/app/ingestion.py`

Purpose:

- PDF text extraction with page metadata

Implementation:

- uses PyMuPDF via `fitz`
- iterates page by page
- extracts text with `page.get_text("text")`
- skips empty pages

Returned structure:

- list of page dictionaries:
  - `text`
  - `metadata.source`
  - `metadata.page`
  - `metadata.total_pages`

### `backend/app/chunking.py`

Purpose:

- split extracted page text into chunk units that preserve more structure than fixed word windows

Implementation:

- paragraphs are split on blank lines
- paragraphs are further split into sentences
- chunks are accumulated sentence by sentence until `chunk_size` would be exceeded
- overlap is preserved by carrying the last `chunk_overlap` words into the next chunk

Defaults:

- `chunk_size = 500` words
- `chunk_overlap = 60` words

Chunk metadata added:

- `chunk_id`
- `sub_chunk_index`

This is more semantic than the original word-only chunking but still heuristic, not embedding-based topic segmentation.

### `backend/app/embeddings.py`

Purpose:

- generate dense embeddings for chunks and queries

Implementation:

- uses `SentenceTransformer("all-MiniLM-L6-v2")`
- `generate_embeddings()` returns a NumPy array for a list of texts
- `generate_single_embedding()` returns one embedding vector

Notes:

- embedding generation is synchronous
- embeddings are not normalized manually in this module

### `backend/app/retrieval.py`

Purpose:

- persistent vector store
- sparse keyword index
- hybrid retrieval orchestration

Core state:

- `self.index`: FAISS `IndexFlatL2`
- `self.metadata`: list of chunk payloads
- `self.sparse_documents`: tokenized chunk representations for keyword retrieval
- `self.document_frequencies`
- `self.avg_document_length`
- `self.reranker`

Persistence:

- default index path: `backend/data/faiss_index.bin`
- default metadata path: `backend/data/metadata.json`
- `load()` executes on initialization by default
- missing files are handled safely by creating empty in-memory state

Key methods:

- `add_documents()`
  - adds embeddings to FAISS
  - appends chunk metadata
  - rebuilds sparse keyword structures
- `clear()`
  - resets in-memory state only
- `reset()`
  - clears in-memory state and deletes persisted index and metadata files
- `has_documents()`
  - returns whether non-empty index and metadata are both present
- `_dense_search()`
  - runs FAISS L2 search
  - converts distance to score as `1 / (1 + distance)`
- `_keyword_search()`
  - implements a BM25-style scorer over tokenized chunk text
- `_fuse_results_rrf()`
  - merges dense and keyword candidates with true Reciprocal Rank Fusion
- `_apply_diversity()`
  - removes highly overlapping candidates and tries to avoid redundant page concentration
- `hybrid_search()`
  - full retrieval pipeline entry point

### `backend/app/reranking.py`

Purpose:

- modular reranking layer

Structure:

- `BaseReranker`
- `HeuristicReranker`
- `CrossEncoderReranker`
- `build_default_reranker()`

Heuristic reranker:

- computes a weighted score from:
  - dense score
  - keyword score
  - fusion score
  - lexical overlap
  - exact-phrase match

Cross-encoder reranker:

- lazy-loads `cross-encoder/ms-marco-MiniLM-L-6-v2`
- scores query-chunk pairs with `CrossEncoder.predict()`
- sigmoid-normalizes raw model scores
- blends model score with heuristic score and fusion score
- uses different blends for:
  - summarization
  - definition and fact lookup
  - analytical queries

Fallback behavior:

- `VectorStore.hybrid_search()` catches reranker failures
- reranking falls back to `HeuristicReranker`
- telemetry records fallback reason and fallback reranker name

### `backend/app/optimization.py`

Purpose:

- context compression without using an LLM

Implementation:

- split chunk text into sentences
- group adjacent short sentences for coherence
- score sentence groups against the query
- filter low-relevance sentence groups
- deduplicate similar content
- enforce an estimated token budget

Scoring inputs:

- query-to-sentence Jaccard overlap
- chunk score
- rerank score
- cross-encoder score
- summary cue token bonus for summarization queries

Behavior details:

- uses a dynamic relevance floor
- uses stricter floors for definition and fact lookup
- uses more permissive floors for summarization
- contains summarization-specific logic that tries to preserve at least one representative sentence from top chunks

Return payload:

- `context`
- `selected_sentences`
- `token_estimate`
- `pages_used`
- `relevance_floor`
- `reasoning_summary`

### `backend/app/query_processing.py`

Purpose:

- lightweight query intelligence layer

Classification:

- `definition`
- `fact_lookup`
- `summarization`
- `analytical`

Classification signals:

- prefix heuristics
- keyword heuristics
- short-query heuristic
- default long-query fallback

Classification outputs:

- label
- heuristic confidence
- matched signals

Rewriting:

- normalizes the raw query
- applies a small set of hand-authored expansions for a few known queries
- otherwise keeps rewriting intentionally lightweight
- preserves natural phrasing more than earlier versions

Query package fields:

- `original`
- `normalized`
- `rewritten`
- `embedding_query`
- `keyword_query`
- `query_type`
- `classification_confidence`
- `classification_signals`
- `force_full_rag`
- `variants`
- `should_prefer_direct_answer`

Important nuance:

- `embedding_query` is typically the raw query now
- only a small number of expansion overrides use an alternative embedding query
- `keyword_query` is always the raw user query

### `backend/app/cache.py`

Purpose:

- disk-backed exact and semantic response cache

Storage:

- default path: `backend/data/query_cache.json`

Normalization:

- lowercase
- strip punctuation
- collapse whitespace

Exact cache:

- key is MD5 hash of normalized query

Semantic cache:

- requires a query embedding at lookup time
- computes cosine similarity against stored query embeddings
- ignores cached responses whose stored confidence level is `low`
- default similarity threshold is `0.9`
- analytical answers require `>= 0.94`

Cached payloads can include:

- `answer`
- `sources`
- `confidence`
- `reasoning_summary`
- `pages_referenced`
- `query_type`
- `normalized_query`
- `query_embedding`

### `backend/app/generation.py`

Purpose:

- Gemini answer generation wrapper

Implementation:

- uses `google.genai.Client()`
- default model name: `gemini-3-flash-preview`
- creates a single prompt with:
  - a strict grounding instruction
  - the compressed context
  - the user question

Prompt safeguards:

- answer only from context
- say you do not know if context is insufficient
- avoid invented facts or interpretations
- cite page markers if useful

### `backend/app/logging_utils.py`

Purpose:

- structured JSON logging

Behavior:

- prints one JSON object per event
- includes UTC timestamp and event name
- used throughout upload, query, retrieval, context optimization, and completion paths

## 4. Retrieval System

### Hybrid retrieval design

The retrieval system has two first-stage retrieval channels:

- dense retrieval from FAISS over chunk embeddings
- keyword retrieval from a BM25-style sparse scorer

Both operate over the same chunk inventory.

### RRF implementation

Fusion uses true Reciprocal Rank Fusion:

- dense contribution: `1 / (k + dense_rank)`
- keyword contribution: `1 / (k + keyword_rank)`
- `k` is fixed at `60`
- fused score is the sum of both reciprocal rank contributions

This means:

- fusion depends on ranks, not raw score interpolation
- dense and keyword systems contribute symmetrically at fusion time

### Reranking pipeline

After fusion:

1. candidates are reranked by the configured reranker
2. the default reranker is `CrossEncoderReranker`
3. reranking can fall back to `HeuristicReranker`
4. telemetry records full decision logs

The cross-encoder is query-type aware:

- summarization does not trust the cross-encoder as heavily
- definition and fact lookup weight the cross-encoder more aggressively
- analytical queries use a blended model-plus-heuristic score

### Diversity and filtering logic

Current filtering is adaptive rather than purely absolute:

- compute max rerank score among reranked candidates
- keep candidates whose rerank score is at least `60%` of the top rerank score
- also allow candidates through with query-type specific absolute thresholds over rerank, cross-encoder, dense, or keyword evidence

Current absolute fallback rules:

- summarization can pass with:
  - rerank score
  - keyword plus dense support
  - heuristic score
- other query types can pass with:
  - relative rerank threshold
  - cross-encoder score
  - rerank score
  - dense plus keyword support

Diversity logic then:

- removes highly overlapping candidates
- discourages early overconcentration from the same page

### Soft fallback mode

If strict filtering returns no final candidates but reranked candidates exist:

- the system marks `soft_fallback_used = True`
- it selects the top `1` fallback candidate for definition and fact lookup
- it selects the top `2` fallback candidates for other query types
- confidence is later forced down to `low`

This is the main recall-oriented fallback in the current implementation.

## 5. Context Optimization

### Sentence grouping

The optimizer:

- splits chunk text into sentences
- groups adjacent short sentences together to reduce fragmentation
- preserves page markers in the final context string

### Relevance filtering

Sentence groups are scored using:

- Jaccard overlap with query tokens
- chunk-level score boost
- rerank score boost
- cross-encoder score boost
- summarization cue-token bonus for summarization queries

Then:

- sentence groups below a dynamic relevance floor are dropped
- groups with more than `0.6` token overlap against already selected groups are removed as redundant

### Token budgeting

- token budget is estimated as `len(text) // 4`
- global optimizer budget is `800`
- summarization logic may force inclusion of representative top-chunk sentences even after the first selection pass

### Weak-context fallback

If the optimized context is empty or too short but retrieval did produce candidates:

- `main.py` creates a weak fallback context from the top retrieved chunk(s)
- this fallback context is not sentence-selected
- it is marked as low-confidence by control flow

## 6. Query Intelligence Layer

### Query classification

Classification is heuristic, not model-based.

Decision patterns:

- prefix-based detection for definitions and fact lookups
- keyword detection for summarization and analytical queries
- short-query heuristic for very short queries
- default-long-query fallback otherwise

### Query rewriting

Current rewriting is intentionally lighter than earlier phases.

Behavior:

- preserve the raw query for keyword retrieval
- usually preserve the raw query for embedding generation
- apply a tiny set of hardcoded expansions only for a few recognized normalized forms
- otherwise append only minimal hints such as `meaning` or `summary`

### Adaptive behavior

The query package affects downstream control flow:

- `query_type` shapes reranking, thresholding, and context optimization
- `classification_confidence` controls `force_full_rag`
- `should_prefer_direct_answer` enables a direct extract path for some simple, high-confidence factual questions

## 7. Confidence System

### How confidence is computed

Confidence is computed in `main.py` from retrieved chunks, not from the LLM.

For non-summarization queries the combined score uses:

- max rerank score
- average rerank score
- max cross-encoder score
- score gap between top two candidates
- lexical support between query and chunk text
- consistency, defined as `1 - score_spread`
- source count factor

For summarization queries the combined score shifts toward:

- average rerank score
- dense support
- keyword support
- lexical support
- consistency
- source count factor

Returned confidence fields:

- `level`
- `score`
- `max_score`
- `avg_score`
- `max_cross_encoder_score`
- `score_gap`
- `consistency`
- `lexical_support`
- `sources`

### How confidence affects control flow

- `soft_fallback_used` forces confidence down to `low`
- direct extract requires `high` confidence and sufficiently strong cross-encoder support
- low-confidence answers are prefixed with uncertainty text
- current `should_skip_llm()` logic is permissive for low-confidence contexts if minimal weak-context token budget exists
- because of that, low-confidence retrieved contexts can still reach Gemini

This is one reason the system now has higher recall but weaker refusal behavior in the current validation results.

## 8. Caching System

### Exact vs semantic cache

Exact cache:

- normalized query -> MD5 hash -> payload

Semantic cache:

- compares query embedding to cached query embeddings with cosine similarity
- ignores low-confidence cached answers
- requires strong similarity threshold

Thresholds:

- default semantic threshold: `0.9`
- analytical semantic threshold: `0.94`

### Safety constraints

- low-confidence cached answers are skipped for semantic reuse
- cache is cleared on each upload
- cache entries are response payloads, not just raw strings

## 9. Validation & Evaluation Layer

### Dataset structure

The validation layer lives under `backend/tests` and includes:

- `create_test_pdf.py`
  - generates a deterministic 7-page synthetic PDF corpus
- `eval_data.py`
  - defines labeled evaluation cases
- `run_validation.py`
  - builds an isolated runtime, indexes the synthetic corpus, runs queries, computes metrics, and writes a report

Each evaluation case includes:

- `id`
- `category`
- `query`
- `expected_answer`
- `required_phrases`
- `forbidden_phrases`
- `relevant_pages`
- `relevant_snippets`
- `min_relevant_pages_returned`
- `should_call_llm`
- `should_succeed`

Categories currently include:

- definition
- fact lookup
- analytical
- summarization
- multi-hop
- ambiguous
- adversarial
- negative

### Evaluation pipeline

`run_validation.py`:

1. creates or refreshes the synthetic PDF
2. swaps in isolated vector store, cache, and stub generator instances
3. ingests the synthetic corpus
4. warms the reranker once to measure cold-start cost separately
5. runs retrieval telemetry and full `query_document()` execution for each case
6. computes ground-truth retrieval metrics and answer correctness
7. writes a JSON report to `backend/tests/artifacts/validation_report.json`

### Metrics

True metrics:

- `precision@k`
- `recall@k`
- `MRR`
- answer accuracy
- retrieval grounding accuracy

Proxy metrics:

- `retrieval_metrics` produced inside `VectorStore._compute_retrieval_metrics()`

Important distinction:

- proxy metrics are heuristic and based on internal overlap rules
- true metrics are computed against labeled pages or chunks from the validation set

### Current measured results

The latest rerun of `python -m backend.tests.run_validation` on April 12, 2026 produced:

- `total_cases`: `13`
- `answer_accuracy`: `0.3077`
- `retrieval_grounding_accuracy`: `0.3846`
- `avg_true_precision_at_k`: `0.6154`
- `avg_true_recall_at_k`: `0.7436`
- `avg_true_mrr`: `0.7179`
- `avg_proxy_precision_at_k`: `1.0`

Interpretation:

- retrieval recall is materially higher than answer accuracy
- proxy retrieval precision remains overly optimistic
- recent recall-oriented fallback behavior improved retrieval coverage but degraded refusal behavior and evaluation accuracy

### Validation harness caveat

`run_validation.py` performs a separate direct retrieval call before calling `query_document()`, and that direct retrieval still uses:

- `retrieval_query = rewritten or normalized or raw`
- `retrieval_text = " ".join(variants)`

`query_document()` no longer uses exactly the same query text strategy. The harness remains useful, but its preflight retrieval telemetry is not perfectly identical to the live query path.

## 10. Performance & Observability

### Latency tracking

Retrieval telemetry records per-stage latency:

- dense search
- keyword search
- fusion
- rerank
- final selection
- overall retrieval latency

Validation also records:

- end-to-end query latency
- retrieval latency
- rerank warmup latency

### Current reranker latency profile

Latest validation summary:

- `avg_rerank_latency_ms`: `96.52`
- `max_rerank_latency_ms`: `127.42`
- `reranker_warmup_ms`: `4481.12`

This reflects a significant cold-start cost for the cross-encoder and much lower steady-state latency afterward.

### Token estimation

- context token estimate is heuristic: characters divided by four
- latest validation average token estimate: `88.31`

### LLM call frequency

Latest validation summary:

- `llm_call_frequency`: `0.6154`
- `llm_calls_total`: `12`

This is elevated by the current low-confidence fallback behavior, which now allows Gemini calls in some borderline contexts that previously would have been refused.

### Observability events

Important logged events include:

- `upload_started`
- `upload_indexed`
- `query_cache_hit`
- `query_retrieval_complete`
- `retrieval_evaluation`
- `context_optimized`
- `query_skipped_llm`
- `query_completed`

## 11. Failure Modes & Safeguards

### No-context handling

If retrieval returns no candidates:

- the system returns an explicit no-context answer
- Gemini is not called

### Low-confidence behavior

Current behavior is mixed:

- low-confidence answers are explicitly labeled
- weak fallback context is allowed to reach Gemini
- the system can therefore return cautious but still content-bearing low-confidence answers

### Adversarial handling

The prompt tells Gemini to answer only from context, but there is no dedicated adversarial classifier.

Current code behavior:

- adversarial queries can retrieve adversarial-policy text from the indexed document
- once that text is retrieved, Gemini can still answer from it instead of refusing
- current validation confirms this is an active weakness

### Cache safety

- semantic cache ignores low-confidence stored responses
- analytical semantic cache hits require stronger similarity

### Persistence safety

- vector store load handles missing files safely
- reset removes index and metadata files
- cache is stored separately and cleared on upload and reset

## 12. Current Limitations

### Multi-hop reasoning gaps

- the retrieval system can find relevant multi-page evidence, but answer composition often fails to synthesize it fully
- validation shows multi-hop answer failures even when retrieval recall is strong

### Summarization weaknesses

- summarization reranking still overweights pages 4, 6, and 7 for the current synthetic benchmark
- the page containing explicit workflow recommendations is often under-selected or not preserved into final grounded output

### Latency tradeoffs

- cross-encoder cold start is expensive
- every query currently reranks candidates, so steady-state latency is dominated by the reranker rather than by FAISS or BM25

### Evaluation limitations

- the synthetic benchmark is still small
- the generated PDF corpus is deterministic and narrow
- the harness is helpful for regression testing but is not a real-world benchmark
- the harness contains a slight query-path mismatch before `query_document()` executes

### Precision vs recall tension

- recent recall-oriented changes intentionally relaxed filtering
- as a result, negative, ambiguous, and adversarial queries now perform worse in validation
- the system currently prefers partial grounded answers over outright refusal in more cases than before

### Direct-answer path inconsistency

- definition queries can still take a direct extract path when confidence is high
- fact lookup currently often falls through to Gemini even when validation expects no LLM call
- validation marks that as `unexpected_llm_usage`

### Single-corpus limitation

- upload replaces the active corpus
- there is no multi-document index, corpus namespace, or document-level filtering across many uploaded files

## Closing Summary

The current backend is a persisted, single-corpus, multi-stage RAG system with hybrid retrieval, RRF fusion, modular reranking, sentence-level context compression, confidence-aware control flow, semantic caching, and a labeled validation harness. Its strongest area is retrieval recall over a small indexed corpus. Its weakest areas are refusal behavior, summarization selection, multi-hop synthesis, and the current precision-recall balance introduced by recall-oriented soft fallback behavior.
