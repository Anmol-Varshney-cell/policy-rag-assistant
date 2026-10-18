"""Main RAG pipeline orchestration."""

import os
from typing import List, Dict
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

from rag_system.data_prep import DocumentLoader, TextChunker
from rag_system.embeddings import EmbeddingGenerator
from rag_system.vector_store import FAISSVectorStore
from rag_system.prompts import PromptTemplates

load_dotenv()

SYSTEM_MESSAGE = "You are a helpful customer support assistant."


class RAGPipeline:
    """Complete RAG pipeline for company policy Q&A."""

    def __init__(self, data_dir: str, index_dir: str = "./vector_index"):
        self.data_dir = data_dir
        self.index_dir = Path(index_dir)

        # Initialize components
        self.doc_loader = DocumentLoader(data_dir)
        self.chunker = TextChunker(chunk_size=500, overlap=50)
        self.embedder = EmbeddingGenerator()
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)

        # LLM setup (Gemini API - free tier, no billing account needed)
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")

        # Prompt templates
        self.prompts = PromptTemplates()
        self.use_improved_prompt = True  # Toggle between v1 and v2

    async def build_index(self, force_rebuild: bool = False):
        """Build or load the vector index."""
        # Check if index exists
        if self.index_dir.exists() and not force_rebuild:
            print("Loading existing vector index...")
            self.vector_store.load(str(self.index_dir))
            print(f"Loaded {self.vector_store.get_stats()['total_chunks']} chunks")
            return

        print("Building vector index...")

        # Load documents
        print("  Loading documents...")
        documents = self.doc_loader.load_documents()
        print(f"  Loaded {len(documents)} documents")

        # Chunk documents
        print("  Chunking documents...")
        chunks = self.chunker.chunk_documents(documents)
        print(f"  Created {len(chunks)} chunks")

        # Generate embeddings
        print("  Generating embeddings...")
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = await self.embedder.generate_embeddings_batch(chunk_texts)
        print(f"  Generated {len(embeddings)} embeddings")

        # Add to vector store
        print("  Building FAISS index...")
        self.vector_store.add_embeddings(embeddings, chunks)

        # Save index
        print(f"  Saving index to {self.index_dir}...")
        self.vector_store.save(str(self.index_dir))
        print("Index built successfully!")

    async def retrieve(self, question: str, k: int = 3) -> List[tuple]:
        """Retrieve top-k relevant chunks for a question."""
        # Generate query embedding
        query_embedding = await self.embedder.generate_embedding(question)

        # Search vector store
        results = self.vector_store.search(query_embedding, k=k)

        return results

    async def answer_question(self, question: str, k: int = 3, use_improved: bool = True) -> Dict:
        """Answer a question using RAG.

        Returns:
            Dict with answer, sources, and metadata
        """
        # Retrieve relevant chunks
        retrieved_chunks = await self.retrieve(question, k=k)

        # Handle case where no relevant documents found
        if not retrieved_chunks:
            return {
                'question': question,
                'answer': "I couldn't find any relevant information in our policy documents to answer your question. Please contact our support team at support@techshop.com for assistance.",
                'sources': [],
                'retrieval_scores': [],
                'prompt_version': 'N/A - No retrieval'
            }

        # Format context
        context = self.prompts.format_context(retrieved_chunks)

        # Select prompt template
        if use_improved:
            prompt = self.prompts.IMPROVED_PROMPT.format(context=context, question=question)
            version = "v2_improved"
        else:
            prompt = self.prompts.INITIAL_PROMPT.format(context=context, question=question)
            version = "v1_initial"

        # Generate answer
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_MESSAGE),
        )
        answer = response.text

        # Extract sources
        sources = [{
            'policy': meta.get('filename', 'Unknown'),
            'chunk_id': meta.get('chunk_id', -1)
        } for _, meta, _ in retrieved_chunks]

        retrieval_scores = [float(dist) for _, _, dist in retrieved_chunks]

        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'retrieval_scores': retrieval_scores,
            'prompt_version': version,
            'context_used': context
        }

    async def batch_answer(self, questions: List[str], use_improved: bool = True) -> List[Dict]:
        """Answer multiple questions."""
        results = []
        for question in questions:
            result = await self.answer_question(question, use_improved=use_improved)
            results.append(result)
        return results
