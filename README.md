# Policy RAG Assistant

A Retrieval-Augmented Generation (RAG) web app that answers questions about company policy documents (refund, cancellation, shipping, leave, code of conduct) using Google Gemini embeddings, a FAISS vector store, and Gemini Flash - all on the Gemini API free tier (no billing account needed).

**Live demo-ready**: deploys to Vercel as a FastAPI serverless app with a built-in chat UI.

## Architecture

```
User question
  -> Gemini embedding (gemini-embedding-001)
  -> FAISS similarity search over policy chunks (top 3)
  -> chunks injected into an engineered prompt
  -> Gemini Flash generates a grounded answer with sources
```

## Project structure

```
├── main.py                 # FastAPI app (Vercel entrypoint, chat UI + /api/ask)
├── rag_assistant.py        # CLI version (menu-driven, interactive Q&A, evaluation)
├── rag_system/
│   ├── data_prep.py        # document loading + token-aware chunking (500/50 overlap)
│   ├── embeddings.py       # OpenAI embedding generation (batched)
│   ├── vector_store.py     # FAISS index with save/load
│   ├── prompts.py          # v1/v2 prompt templates with source citation rules
│   ├── rag_pipeline.py     # orchestration: retrieve -> prompt -> answer
│   └── evaluate.py         # 8-case evaluation suite (accuracy / hallucination / clarity)
├── data/policies/          # policy documents (add your own .txt files here)
├── static/index.html       # chat UI
└── requirements.txt
```

## Deploy to Vercel (free Hobby plan)

1. Import this repo in Vercel (New Project -> Import Git Repository).
2. Vercel auto-detects the FastAPI app in `main.py`. No build settings needed.
3. Add an environment variable: `GEMINI_API_KEY` = your free Gemini API key from Google AI Studio (https://aistudio.google.com/apikey - no billing account required).
4. Deploy. The first question takes a few seconds while the policy index builds; after that it's instant.

The Hobby plan does not sleep or delete deployments - the link stays live. Monthly free quotas apply (100 GB transfer, 1M function invocations), far above demo traffic. Hobby is for personal/non-commercial use.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY
uvicorn main:app --reload          # web app at http://localhost:8000
# or the CLI:
python rag_assistant.py
```

## Notes

- Embeddings use the Gemini API instead of a local model so the app fits serverless size limits. Free tier rate limits apply; exceeding them returns a temporary rate-limit error, never a charge (https://ai.google.dev/gemini-api/docs/rate-limits).
- The vector index builds in memory at startup from `data/policies/` - no database needed.
- `refund_policy.txt`, `leave_policy.txt` and `code_of_conduct.txt` were recovered from the original repo. `cancellation_policy.txt` and `shipping_policy.txt` are NEW sample documents written for the demo (the originals were lost) - edit or replace them freely.
- Retrieval, prompt design, and evaluation approach are described in `rag_system/` docstrings.
