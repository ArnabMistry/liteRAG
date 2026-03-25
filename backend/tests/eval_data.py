# Evaluation QA Dataset

QA_PAIRS = [
    {
        "query": "When was RAG first proposed?",
        "expected_page": 1,
        "expected_text": "2020 by Lewis et al"
    },
    {
        "query": "Who developed FAISS?",
        "expected_page": 2,
        "expected_text": "Facebook AI Research"
    },
    {
        "query": "What is the dimension of all-MiniLM-L6-v2 embeddings?",
        "expected_page": 3,
        "expected_text": "384-dimensional"
    },
    {
        "query": "What is the 'lost in the middle' phenomenon?",
        "expected_page": 4,
        "expected_text": "too much context"
    },
    {
        "query": "What framework does liteRAG use for its backend?",
        "expected_page": 5,
        "expected_text": "FastAPI"
    }
]
