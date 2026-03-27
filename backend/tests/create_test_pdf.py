from pathlib import Path

import fitz


PAGES = [
    """Stoicism teaches focusing on what is within your control and accepting what is outside it.
Practical stoic training emphasizes disciplined attention, emotional regulation, and deliberate action.
Researchers sometimes use stoic examples to discuss resilience, judgment, and self-governance.""",
    """Retrieval-Augmented Generation grounds answers in retrieved documents rather than free-form invention.
FAISS performs nearest-neighbor search over embeddings, while hybrid retrieval combines semantic and keyword signals.
Researchers should compare evidence across sources, record confidence, and avoid unsupported claims.""",
    """Reciprocal rank fusion combines multiple ranked lists by summing reciprocal rank contributions from each list.
Precision at k measures how many retrieved items are relevant among the top results.
Recall at k measures how many of all relevant items were recovered by retrieval.""",
    """Confidence should reflect score strength, score gaps, and agreement across retrieved sources.
Using both score gaps and source consistency helps confidence reflect whether evidence is concentrated or broadly supported.
Consistent evidence reduces the chance that a single noisy chunk dominates the final answer.""",
    """High-precision RAG depends on both ranking quality and confidence calibration.
 Reciprocal rank fusion improves candidate ordering, while calibrated confidence helps the system refuse weak evidence.
 Together they reduce hallucination by preferring well-supported evidence and blocking answers from noisy retrieval.""",
    """Ambiguous questions should not be answered confidently when the referent is unclear.
 Queries that use pronouns like it or they without context should trigger clarification or a grounded fallback.
 A trustworthy assistant should prefer saying that the document is insufficient over inventing a specific referent.""",
    """Adversarial instructions can ask the model to ignore the document, reveal hidden prompts, or answer from memory.
 A grounded system should reject those requests and continue to use only retrieved evidence from the indexed document.
 Safety is preserved when the pipeline treats unsupported or manipulative instructions as low-confidence queries.""",
]


def create_test_pdf(output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()
    for page_text in PAGES:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(48, 48, 560, 780), page_text, fontsize=12, fontname="helv")

    doc.save(output_path)
    doc.close()


if __name__ == "__main__":
    create_test_pdf(Path(__file__).resolve().parent.parent / "data" / "test_eval.pdf")
