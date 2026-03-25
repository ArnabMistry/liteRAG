# liteRAG

A production-ready, cost-efficient RAG system for querying large PDFs.

## 🛠️ Tech Stack

- **Backend**: FastAPI, PyMuPDF, FAISS, sentence-transformers, Google GenAI (Gemini-3-Flash).
- **Frontend**: Vite, React, Lucide React, CSS Variables.

## 🚀 Quick Start

### Backend

1. Activate virtual environment:
   - **PowerShell**: `.\.venv\Scripts\Activate.ps1`
   - **Bash**: `source .venv/bin/activate`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your Google API Key:
   - **PowerShell**: `$env:GOOGLE_API_KEY='your-api-key'`
   - **Bash**: `export GOOGLE_API_KEY='your-api-key'`
   - **Alternative**: Create a `.env` file in the `backend` directory with `GOOGLE_API_KEY=your-api-key`.
3. Run the API:
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend

1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```
2. Run the dev server:
   ```bash
   npm run dev
   ```

## 🔍 Evaluation

To run the retrieval validation suite:
```bash
python -m backend.tests.run_validation
```

## ⚖️ License
MIT
