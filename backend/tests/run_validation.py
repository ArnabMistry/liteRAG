import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path.cwd()))
sys.path.append(str(Path("backend").resolve()))

from backend.app.cache import QueryCache
from backend.app.main import QueryRequest, query_document
from backend.app.optimization import ContextOptimizer
from backend.app.retrieval import VectorStore
import backend.app.main as app_main
from backend.tests.create_test_pdf import create_test_pdf
from backend.tests.eval_data import EVAL_DATASET


ARTIFACTS_DIR = Path("backend/tests/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = Path("backend/data/test_eval.pdf")


@dataclass
class ValidationEnvironment:
    original_vector_store: object
    original_cache: object
    original_generator: object
    llm_calls: int = 0
    reranker_warmup_ms: float = 0.0


class StubAnswerGenerator:
    def __init__(self, tracker):
        self.tracker = tracker

    def generate_answer(self, query: str, context: str) -> str:
        self.tracker.llm_calls += 1
        compact_context = " ".join(context.split())
        return f"SYNTHESIZED: {compact_context[:320]}".strip()


def ensure_test_pdf():
    create_test_pdf(PDF_PATH)


def prepare_isolated_runtime():
    env = ValidationEnvironment(
        original_vector_store=app_main.vector_store,
        original_cache=app_main.cache,
        original_generator=app_main.generator,
    )

    validation_index = ARTIFACTS_DIR / "validation_faiss_index.bin"
    validation_metadata = ARTIFACTS_DIR / "validation_metadata.json"
    validation_cache = ARTIFACTS_DIR / "validation_query_cache.json"

    for path in (validation_index, validation_metadata, validation_cache):
        if path.exists():
            path.unlink()

    vector_store = VectorStore(
        dimension=app_main.embedding_engine.dimension,
        index_path=validation_index,
        metadata_path=validation_metadata,
        auto_load=False,
    )
    cache = QueryCache(cache_path=validation_cache)
    generator = StubAnswerGenerator(env)

    app_main.vector_store = vector_store
    app_main.cache = cache
    app_main.generator = generator

    return env, vector_store


def restore_runtime(env: ValidationEnvironment):
    app_main.vector_store = env.original_vector_store
    app_main.cache = env.original_cache
    app_main.generator = env.original_generator


def ingest_validation_corpus(vector_store: VectorStore):
    ingestor = app_main.PDFIngestor(str(PDF_PATH))
    docs = ingestor.extract_text_with_metadata()
    chunker = app_main.SemanticChunker()
    chunks = chunker.chunk_documents(docs)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = app_main.embedding_engine.generate_embeddings(texts)
    vector_store.clear()
    vector_store.add_documents(embeddings, chunks)
    return docs, chunks


def warm_reranker(vector_store: VectorStore):
    warmup_query = "warm up reranker for evaluation"
    query_embedding = app_main.embedding_engine.generate_single_embedding(warmup_query)
    _, telemetry = vector_store.hybrid_search(
        query_text=warmup_query,
        query_embedding=query_embedding,
        initial_k=5,
        final_k=1,
        threshold=0.0,
        query_type="fact_lookup",
    )
    return telemetry.get("rerank_latency_ms", 0.0)


def derive_ground_truth_chunk_ids(chunks, relevant_pages, relevant_snippets):
    chunk_ids = set()
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        text = chunk.get("text", "")
        if metadata.get("page") in relevant_pages:
            chunk_ids.add(metadata.get("chunk_id"))
            continue
        for snippet in relevant_snippets:
            if snippet and snippet.lower() in text.lower():
                chunk_ids.add(metadata.get("chunk_id"))
                break
    return sorted(chunk_ids)


def compute_true_retrieval_metrics(result_sources, ground_truth_chunk_ids, ground_truth_pages, reranked_chunk_ids):
    retrieved_ids = [source.get("chunk_id") for source in result_sources]
    retrieved_pages = [source.get("page") for source in result_sources]

    relevant_by_id = [chunk_id for chunk_id in retrieved_ids if chunk_id in ground_truth_chunk_ids]
    relevant_by_page = [page for page in retrieved_pages if page in ground_truth_pages]

    precision_denominator = len(retrieved_ids) or 1
    precision_at_k = len(relevant_by_id or relevant_by_page) / precision_denominator

    total_relevant = len(ground_truth_chunk_ids) or len(ground_truth_pages)
    recall_at_k = (
        len(set(relevant_by_id)) / total_relevant
        if ground_truth_chunk_ids
        else (len(set(relevant_by_page)) / total_relevant if total_relevant else 0.0)
    )

    reciprocal_rank = 0.0
    for idx, chunk_id in enumerate(reranked_chunk_ids, start=1):
        if chunk_id in ground_truth_chunk_ids:
            reciprocal_rank = 1.0 / idx
            break

    return {
        "precision_at_k": round(precision_at_k, 4),
        "recall_at_k": round(recall_at_k, 4),
        "mrr": round(reciprocal_rank, 4),
    }


def normalize_text(text: str) -> str:
    normalized = (text or "").lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def is_refusal_answer(answer_text: str) -> bool:
    normalized_answer = normalize_text(answer_text)
    refusal_markers = (
        "don't know",
        "does not contain relevant context",
        "not enough confidence",
        "insufficient",
        "cannot determine",
    )
    return any(marker in normalized_answer for marker in refusal_markers)


def evaluate_answer_correctness(case, answer_text, returned_pages):
    normalized_answer = normalize_text(answer_text)
    required_phrases = [normalize_text(phrase) for phrase in case.get("required_phrases", [])]
    forbidden_phrases = [normalize_text(phrase) for phrase in case.get("forbidden_phrases", [])]
    relevant_pages = set(case.get("relevant_pages", []))
    returned_pages_set = set(returned_pages or [])
    min_relevant_pages = case.get("min_relevant_pages_returned", 0)

    failure_reasons = []

    if case["should_succeed"]:
        for phrase in required_phrases:
            if phrase not in normalized_answer:
                failure_reasons.append(f"missing_required_phrase:{phrase}")

        if min_relevant_pages:
            grounded_pages = len(relevant_pages.intersection(returned_pages_set))
            if grounded_pages < min_relevant_pages:
                failure_reasons.append(
                    f"insufficient_page_grounding:{grounded_pages}_of_{min_relevant_pages}"
                )

        for phrase in forbidden_phrases:
            if phrase and phrase in normalized_answer:
                failure_reasons.append(f"contains_forbidden_phrase:{phrase}")
    else:
        if not is_refusal_answer(answer_text):
            failure_reasons.append("missing_refusal")

        for phrase in forbidden_phrases:
            if phrase and phrase in normalized_answer:
                failure_reasons.append(f"contains_forbidden_phrase:{phrase}")

    return {
        "passed": len(failure_reasons) == 0,
        "failure_reasons": failure_reasons,
    }


async def run_case(case, chunks):
    query_package = app_main.query_processor.build_query_package(case["query"])
    retrieval_query = query_package["rewritten"] or query_package["normalized"] or case["query"]
    retrieval_text = " ".join(query_package["variants"])
    query_embedding = app_main.embedding_engine.generate_single_embedding(retrieval_query)

    retrieval_start = time.perf_counter()
    retrieved_chunks, telemetry = app_main.vector_store.hybrid_search(
        query_text=retrieval_text,
        query_embedding=query_embedding,
        initial_k=10,
        final_k=3,
        threshold=0.12,
        query_type=query_package["query_type"],
    )
    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

    ground_truth_chunk_ids = derive_ground_truth_chunk_ids(
        chunks,
        case["relevant_pages"],
        case["relevant_snippets"],
    )

    start = time.perf_counter()
    response = await query_document(QueryRequest(query=case["query"]))
    latency_ms = (time.perf_counter() - start) * 1000

    context_package = app_main.optimizer.build_context_package(
        case["query"],
        retrieved_chunks,
        query_type=query_package["query_type"],
    ) if retrieved_chunks else {
        "token_estimate": 0,
        "selected_sentences": [],
        "pages_used": [],
    }

    answer_evaluation = evaluate_answer_correctness(
        case,
        response.get("answer", ""),
        response.get("pages_referenced", []),
    )

    true_metrics = compute_true_retrieval_metrics(
        response.get("sources", []),
        ground_truth_chunk_ids,
        case["relevant_pages"],
        telemetry.get("reranked_chunk_ids", []),
    )

    llm_called_for_case = response.get("cached", False) is False and response.get("answer", "").startswith("SYNTHESIZED:")

    grounded_pages = len(set(case["relevant_pages"]).intersection(set(response.get("pages_referenced", []))))
    min_relevant_pages = case.get("min_relevant_pages_returned", 0)
    retrieval_passed = (
        grounded_pages >= min_relevant_pages
        if case["should_succeed"]
        else len(response.get("pages_referenced", [])) == 0
    )

    rerank_latency_ms = telemetry.get("rerank_latency_ms", 0.0)
    stage_latencies = telemetry.get("latency_ms", {})

    failure_reasons = list(answer_evaluation["failure_reasons"])
    if not retrieval_passed:
        failure_reasons.append("retrieval_grounding_failed")
    if llm_called_for_case != case["should_call_llm"]:
        failure_reasons.append("unexpected_llm_usage")

    return {
        "id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "expected_answer": case["expected_answer"],
        "answer": response.get("answer"),
        "answer_correct": answer_evaluation["passed"],
        "confidence": response.get("confidence", {}),
        "confidence_matches_correctness": (
            response.get("confidence", {}).get("level") != "high" or answer_evaluation["passed"]
        ),
        "relevant_pages": case["relevant_pages"],
        "ground_truth_chunk_ids": ground_truth_chunk_ids,
        "returned_pages": response.get("pages_referenced", []),
        "returned_chunk_ids": [source.get("chunk_id") for source in response.get("sources", [])],
        "proxy_retrieval_metrics": telemetry.get("retrieval_metrics", {}),
        "true_retrieval_metrics": true_metrics,
        "retrieval_passed": retrieval_passed,
        "grounded_pages_returned": grounded_pages,
        "min_relevant_pages_required": min_relevant_pages,
        "failure_reasons": failure_reasons,
        "rerank_latency_ms": round(rerank_latency_ms, 2),
        "retrieval_latency_ms": round(retrieval_latency_ms, 2),
        "retrieval_stage_latency_ms": stage_latencies,
        "latency_ms": round(latency_ms, 2),
        "token_estimate": context_package.get("token_estimate", 0),
        "llm_called": llm_called_for_case,
        "expected_llm_call": case["should_call_llm"],
        "cached": response.get("cached", False),
    }


def summarize_results(case_results, env: ValidationEnvironment):
    success_cases = [result for result in case_results if result["answer_correct"]]
    retrieval_success_cases = [result for result in case_results if result["retrieval_passed"]]
    true_precision = [result["true_retrieval_metrics"]["precision_at_k"] for result in case_results]
    true_recall = [result["true_retrieval_metrics"]["recall_at_k"] for result in case_results]
    true_mrr = [result["true_retrieval_metrics"]["mrr"] for result in case_results]
    proxy_precision = [result["proxy_retrieval_metrics"].get("precision_at_k", 0.0) for result in case_results]
    rerank_latencies = [result["rerank_latency_ms"] for result in case_results]

    high_confidence_errors = [
        result["id"]
        for result in case_results
        if result["confidence"].get("level") == "high" and not result["answer_correct"]
    ]
    failure_cases = [
        {
            "id": result["id"],
            "category": result["category"],
            "failure_reasons": result["failure_reasons"],
        }
        for result in case_results
        if result["failure_reasons"]
    ]

    category_summary = {}
    for result in case_results:
        category = result["category"]
        bucket = category_summary.setdefault(
            category,
            {"total": 0, "answer_passed": 0, "retrieval_passed": 0},
        )
        bucket["total"] += 1
        bucket["answer_passed"] += int(result["answer_correct"])
        bucket["retrieval_passed"] += int(result["retrieval_passed"])

    summary = {
        "total_cases": len(case_results),
        "answer_accuracy": round(len(success_cases) / max(len(case_results), 1), 4),
        "retrieval_grounding_accuracy": round(len(retrieval_success_cases) / max(len(case_results), 1), 4),
        "avg_true_precision_at_k": round(sum(true_precision) / max(len(true_precision), 1), 4),
        "avg_true_recall_at_k": round(sum(true_recall) / max(len(true_recall), 1), 4),
        "avg_true_mrr": round(sum(true_mrr) / max(len(true_mrr), 1), 4),
        "avg_proxy_precision_at_k": round(sum(proxy_precision) / max(len(proxy_precision), 1), 4),
        "avg_latency_ms": round(sum(result["latency_ms"] for result in case_results) / max(len(case_results), 1), 2),
        "avg_retrieval_latency_ms": round(sum(result["retrieval_latency_ms"] for result in case_results) / max(len(case_results), 1), 2),
        "avg_rerank_latency_ms": round(sum(rerank_latencies) / max(len(rerank_latencies), 1), 2),
        "max_rerank_latency_ms": round(max(rerank_latencies) if rerank_latencies else 0.0, 2),
        "reranker_warmup_ms": round(env.reranker_warmup_ms, 2),
        "avg_token_estimate": round(sum(result["token_estimate"] for result in case_results) / max(len(case_results), 1), 2),
        "llm_call_frequency": round(sum(1 for result in case_results if result["llm_called"]) / max(len(case_results), 1), 4),
        "llm_calls_total": env.llm_calls,
        "confidence_miscalibration_cases": high_confidence_errors,
        "failure_cases": failure_cases,
        "category_summary": category_summary,
    }
    return summary


async def main():
    print("--- Starting Ground-Truth Validation ---\n")
    ensure_test_pdf()
    env, vector_store = prepare_isolated_runtime()

    try:
        _, chunks = ingest_validation_corpus(vector_store)
        env.reranker_warmup_ms = warm_reranker(vector_store)
        case_results = []
        for case in EVAL_DATASET:
            result = await run_case(case, chunks)
            case_results.append(result)
            print(
                f"{case['id']}: correct={result['answer_correct']} "
                f"retrieval_passed={result['retrieval_passed']} "
                f"precision@k={result['true_retrieval_metrics']['precision_at_k']:.2f} "
                f"recall@k={result['true_retrieval_metrics']['recall_at_k']:.2f} "
                f"mrr={result['true_retrieval_metrics']['mrr']:.2f} "
                f"confidence={result['confidence'].get('level')} "
                f"rerank_ms={result['rerank_latency_ms']} "
                f"latency_ms={result['latency_ms']} "
                f"failures={result['failure_reasons']}"
            )

        summary = summarize_results(case_results, env)
        report = {
            "summary": summary,
            "cases": case_results,
        }

        report_path = ARTIFACTS_DIR / "validation_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print("\n--- Validation Summary ---")
        print(json.dumps(summary, indent=2))
        print(f"\nReport written to: {report_path}")
    finally:
        restore_runtime(env)


if __name__ == "__main__":
    asyncio.run(main())
