"""Embedding generation using Google's Gemini API (free tier).

Switched from the original sentence-transformers local model (~2 GB of
torch dependencies, too large for serverless) to Gemini embeddings.
The Gemini API free tier needs no billing account and covers both
embeddings and chat - see https://ai.google.dev/gemini-api/docs/pricing
"""

import os
from typing import List
from google import genai
from dotenv import load_dotenv

load_dotenv()


class EmbeddingGenerator:
    """Generate embeddings using Gemini's gemini-embedding-001 model.

    Why this model:
    - Free tier: no billing account required, generous demo limits
    - 3072 dimensions, strong retrieval quality
    - No heavy local dependencies (no torch/sentence-transformers)
    """

    def __init__(self, model_name: str = "gemini-embedding-001"):
        self.model_name = model_name
        self.dimension = 3072  # gemini-embedding-001 output dimension
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        result = await self.client.aio.models.embed_content(
            model=self.model_name,
            contents=text,
        )
        return list(result.embeddings[0].values)

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in one API call."""
        result = await self.client.aio.models.embed_content(
            model=self.model_name,
            contents=texts,
        )
        return [list(e.values) for e in result.embeddings]
