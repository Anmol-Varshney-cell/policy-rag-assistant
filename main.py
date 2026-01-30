# Complete RAG System Code

This document contains all the code files for the RAG system.

## 1. `/app/backend/rag_system/__init__.py`

```python
\"\"\"RAG System for Company Policy Documents\"\"\"

__version__ = \"1.0.0\"
```

---

## 2. `/app/backend/rag_system/data_prep.py`

```python
\"\"\"Data preparation module for loading and chunking documents.\"\"\"

import os
from pathlib import Path
from typing import List, Dict
import tiktoken


class DocumentLoader:
    \"\"\"Load and prepare documents for RAG.\"\"\"
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.encoding = tiktoken.get_encoding(\"cl100k_base\")  # GPT-4 encoding
    
    def load_documents(self) -> List[Dict[str, str]]:
        \"\"\"Load all policy documents from the data directory.\"\"\"
        documents = []
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f\"Data directory not found: {self.data_dir}\")
        
        for file_path in self.data_dir.glob(\"*.txt\"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    'filename': file_path.name,
                    'content': content,
                    'source': str(file_path)
                })
        
        return documents
    
    def count_tokens(self, text: str) -> int:
        \"\"\"Count tokens in text using tiktoken.\"\"\"
        return len(self.encoding.encode(text))


class TextChunker:
    \"\"\"
    Chunk text into smaller pieces with overlap.
    
    Strategy Explanation:
    - Chunk size: 500 tokens (~375 words)
    - Overlap: 50 tokens (~40 words)
    
    Reasoning:
    1. 500 tokens balances context and precision:
       - Large enough to capture complete policy sections
       - Small enough for focused retrieval
       - Fits well within LLM context windows with room for multiple chunks
    
    2. 50-token overlap ensures:
       - Information at chunk boundaries isn't lost
       - Better semantic continuity between chunks
       - Improved retrieval of concepts spanning chunk borders
    
    3. Trade-offs considered:
       - Smaller chunks (200-300): More precise but may lose context
       - Larger chunks (800-1000): More context but less focused retrieval
       - Selected 500 as optimal middle ground for policy documents
    \"\"\"
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding(\"cl100k_base\")
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        \"\"\"Split text into overlapping chunks.\"\"\"
        # Encode text to tokens
        tokens = self.encoding.encode(text)
        chunks = []
        
        # Create chunks with overlap
        start = 0
        chunk_id = 0
        
        while start < len(tokens):
            # Get chunk tokens
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Create chunk metadata
            chunk_meta = metadata.copy() if metadata else {}
            chunk_meta.update({
                'chunk_id': chunk_id,
                'start_token': start,
                'end_token': min(end, len(tokens)),
                'total_tokens': len(chunk_tokens)
            })
            
            chunks.append({
                'text': chunk_text,
                'metadata': chunk_meta
            })
            
            # Move to next chunk with overlap
            start = end - self.overlap
            chunk_id += 1
            
            # Break if we've processed all tokens
            if end >= len(tokens):
                break
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        \"\"\"Chunk multiple documents.\"\"\"
        all_chunks = []
        
        for doc in documents:
            metadata = {
                'filename': doc['filename'],
                'source': doc['source']
            }
            chunks = self.chunk_text(doc['content'], metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
```

---

## 3. `/app/backend/rag_system/embeddings.py`

```python
\"\"\"Embedding generation using sentence-transformers (local model).\"\"\"

import os
import asyncio
from typing import List
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


class EmbeddingGenerator:
    \"\"\"Generate embeddings using sentence-transformers (local model).
    
    Using 'all-MiniLM-L6-v2':
    - Fast, lightweight model
    - 384 dimensions
    - No API key required
    - Good quality for semantic search
    
    Trade-off: OpenAI embeddings are slightly better quality, but this
    removes API dependency and costs, perfect for demo/MVP.
    \"\"\"
    
    def __init__(self, model_name: str = \"all-MiniLM-L6-v2\"):
        print(f\"Loading embedding model: {model_name}...\")
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        print(\"✓ Model loaded\")
    
    async def generate_embedding(self, text: str) -> List[float]:
        \"\"\"Generate embedding for a single text.\"\"\"
        # Run in thread pool since sentence-transformers is CPU-bound
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self.model.encode(text, convert_to_numpy=True)
        )
        return embedding.tolist()
    
    async def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        \"\"\"Generate embeddings for multiple texts in batches.\"\"\"
        # Run in thread pool
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        )
        return [emb.tolist() for emb in embeddings]
```

---

## 4. `/app/backend/rag_system/vector_store.py`

```python
\"\"\"Vector store using FAISS for semantic search.\"\"\"

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple


class FAISSVectorStore:
    \"\"\"FAISS-based vector store for semantic search.\"\"\"
    
    def __init__(self, dimension: int = 384):
        \"\"\"Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2, 1536 for OpenAI)
        \"\"\"
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
        self.chunks = []  # Store chunk data
        self.metadata = []  # Store metadata
    
    def add_embeddings(self, embeddings: List[List[float]], chunks: List[Dict]):
        \"\"\"Add embeddings and their corresponding chunks to the store.\"\"\"
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype='float32')
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store chunks and metadata
        self.chunks.extend([chunk['text'] for chunk in chunks])
        self.metadata.extend([chunk['metadata'] for chunk in chunks])
    
    def search(self, query_embedding: List[float], k: int = 3) -> List[Tuple[str, Dict, float]]:
        \"\"\"Search for top-k most similar chunks.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
        
        Returns:
            List of (chunk_text, metadata, distance) tuples
        \"\"\"
        # Convert to numpy array
        query_array = np.array([query_embedding], dtype='float32')
        
        # Search
        distances, indices = self.index.search(query_array, k)
        
        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):  # Valid index
                results.append((
                    self.chunks[idx],
                    self.metadata[idx],
                    float(distance)
                ))
        
        return results
    
    def save(self, path: str):
        \"\"\"Save index and metadata to disk.\"\"\"
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / \"faiss.index\"))
        
        # Save chunks and metadata
        with open(path / \"chunks.pkl\", 'wb') as f:
            pickle.dump(self.chunks, f)
        with open(path / \"metadata.pkl\", 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load(self, path: str):
        \"\"\"Load index and metadata from disk.\"\"\"
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / \"faiss.index\"))
        
        # Load chunks and metadata
        with open(path / \"chunks.pkl\", 'rb') as f:
            self.chunks = pickle.load(f)
        with open(path / \"metadata.pkl\", 'rb') as f:
            self.metadata = pickle.load(f)
    
    def get_stats(self) -> Dict:
        \"\"\"Get statistics about the vector store.\"\"\"
        return {
            'total_chunks': len(self.chunks),
            'dimension': self.dimension,
            'index_size': self.index.ntotal
        }
```

---

## 5. `/app/backend/rag_system/prompts.py`

```python
\"\"\"Prompt templates for RAG system with iteration improvements.\"\"\"


class PromptTemplates:
    \"\"\"Collection of prompt templates for the RAG assistant.\"\"\"
    
    # VERSION 1: Initial Prompt
    # Basic approach with minimal structure
    INITIAL_PROMPT = \"\"\"You are a customer support assistant for TechShop Inc. Answer the user's question based on the provided policy documents.

Context from policy documents:
{context}

User Question: {question}

Answer:\"\"\"
    
    # VERSION 2: Improved Prompt
    # Enhanced with better instructions, structure, and hallucination prevention
    IMPROVED_PROMPT = \"\"\"You are a knowledgeable customer support assistant for TechShop Inc.

Your task is to answer customer questions accurately using ONLY the information provided in the company policy documents below.

**IMPORTANT RULES:**
1. Base your answer ONLY on the context provided below
2. If the context doesn't contain enough information to answer the question, explicitly state: \"I don't have enough information in our policies to answer this question fully. Please contact our support team at support@techshop.com\"
3. Do NOT make up or infer information not present in the context
4. Always cite which policy the information comes from (e.g., \"According to our Refund Policy...\")
5. If the question is partially answerable, provide what you can and clearly indicate what information is missing

**CONTEXT FROM COMPANY POLICIES:**
{context}

**CUSTOMER QUESTION:**
{question}

**YOUR ANSWER:**
Provide a clear, structured response following this format:
- Start with a direct answer if available
- Include relevant policy details with citations
- Use bullet points for clarity when listing multiple items
- End with next steps or contact information if appropriate\"\"\"
    
    # VERSION 3: Structured Output Prompt (Optional - for JSON responses)
    STRUCTURED_PROMPT = \"\"\"You are a customer support assistant for TechShop Inc.

Answer the question using ONLY the provided policy context. Your response must be accurate and well-sourced.

**CONTEXT:**
{context}

**QUESTION:**
{question}

**INSTRUCTIONS:**
Provide your answer in the following structured format:

**Answer:** [Direct answer to the question]

**Details:**
- [Key point 1 with policy citation]
- [Key point 2 with policy citation]

**Source Policy:** [Name of the policy document(s) used]

**Confidence:** [High/Medium/Low based on context sufficiency]

**Additional Notes:** [Any caveats, missing information, or recommended next steps]

If you cannot answer from the context, state: \"Information not available in provided policies\" and suggest contacting support.\"\"\"
    
    @staticmethod
    def get_improvement_explanation() -> str:
        \"\"\"Explain the improvements made between prompt versions.\"\"\"
        return \"\"\"
**PROMPT ITERATION IMPROVEMENTS**

**Changes from V1 (Initial) to V2 (Improved):**

1. **Explicit Grounding Instructions:**
   - Added \"ONLY\" emphasis to prevent hallucinations
   - Clear rule: if info isn't in context, say so explicitly
   - Prevents model from using pre-trained knowledge

2. **Structured Response Format:**
   - Defined expected answer structure (direct answer → details → next steps)
   - Encourages organized, scannable responses
   - Better user experience with bullet points

3. **Source Citation Requirement:**
   - Forces model to reference which policy contains the information
   - Increases accountability and verifiability
   - Helps users know where to look for more details

4. **Partial Answer Handling:**
   - Explicit instruction for partially answerable questions
   - Acknowledges what's missing rather than guessing
   - Reduces hallucination risk

5. **Clear Role Definition:**
   - \"Knowledgeable\" sets professional tone
   - Company name (TechShop Inc.) provides context
   - Establishes appropriate assistant persona

6. **Fallback Contact Information:**
   - Provides alternative when RAG fails
   - Improves overall customer experience
   - Realistic handling of edge cases

**Why These Changes Matter:**
- **Accuracy:** Stronger grounding reduces false information
- **Transparency:** Citation requirements make answers auditable
- **User Trust:** Admitting knowledge gaps is better than guessing
- **Maintainability:** Structured format easier to evaluate and improve

**Trade-offs Considered:**
- Longer prompt = slightly higher token cost, but significantly better output quality
- More rigid structure = less creative freedom, but more consistent results
- Explicit rules = verbose prompt, but crucial for preventing hallucinations in production
\"\"\"
    
    @staticmethod
    def format_context(retrieved_chunks: list) -> str:
        \"\"\"Format retrieved chunks into context string.\"\"\"
        context_parts = []
        for i, (chunk_text, metadata, distance) in enumerate(retrieved_chunks, 1):
            policy_name = metadata.get('filename', 'Unknown').replace('_', ' ').replace('.txt', '').title()
            context_parts.append(f\"[Source {i}: {policy_name}]
{chunk_text}
\")
        return \"
\".join(context_parts)
```

---

## 6. `/app/backend/rag_system/rag_pipeline.py`

```python
\"\"\"Main RAG pipeline orchestration.\"\"\"

import asyncio
from typing import List, Dict, Optional
from pathlib import Path

from rag_system.data_prep import DocumentLoader, TextChunker
from rag_system.embeddings import EmbeddingGenerator
from rag_system.vector_store import FAISSVectorStore
from rag_system.prompts import PromptTemplates

from emergentintegrations.llm.chat import LlmChat, UserMessage
from dotenv import load_dotenv
import os

load_dotenv()


class RAGPipeline:
    \"\"\"Complete RAG pipeline for company policy Q&A.\"\"\"
    
    def __init__(self, data_dir: str, index_dir: str = \"./vector_index\"):
        self.data_dir = data_dir
        self.index_dir = Path(index_dir)
        
        # Initialize components
        self.doc_loader = DocumentLoader(data_dir)
        self.chunker = TextChunker(chunk_size=500, overlap=50)
        self.embedder = EmbeddingGenerator()
        self.vector_store = FAISSVectorStore(dimension=384)  # all-MiniLM-L6-v2 dimension
        
        # LLM setup
        api_key = os.getenv('EMERGENT_LLM_KEY')
        self.llm = LlmChat(
            api_key=api_key,
            session_id=\"rag_assistant\",
            system_message=\"You are a helpful customer support assistant.\"
        ).with_model(\"openai\", \"gpt-4o-mini\")
        
        # Prompt templates
        self.prompts = PromptTemplates()
        self.use_improved_prompt = True  # Toggle between v1 and v2
    
    async def build_index(self, force_rebuild: bool = False):
        \"\"\"Build or load the vector index.\"\"\"
        # Check if index exists
        if self.index_dir.exists() and not force_rebuild:
            print(\"Loading existing vector index...\")
            self.vector_store.load(str(self.index_dir))
            print(f\"Loaded {self.vector_store.get_stats()['total_chunks']} chunks\")
            return
        
        print(\"Building vector index...\")
        
        # Load documents
        print(\"  Loading documents...\")
        documents = self.doc_loader.load_documents()
        print(f\"  Loaded {len(documents)} documents\")
        
        # Chunk documents
        print(\"  Chunking documents...\")
        chunks = self.chunker.chunk_documents(documents)
        print(f\"  Created {len(chunks)} chunks\")
        
        # Generate embeddings
        print(\"  Generating embeddings...\")
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = await self.embedder.generate_embeddings_batch(chunk_texts)
        print(f\"  Generated {len(embeddings)} embeddings\")
        
        # Add to vector store
        print(\"  Building FAISS index...\")
        self.vector_store.add_embeddings(embeddings, chunks)
        
        # Save index
        print(f\"  Saving index to {self.index_dir}...\")
        self.vector_store.save(str(self.index_dir))
        print(\"Index built successfully!\")
    
    async def retrieve(self, question: str, k: int = 3) -> List[tuple]:
        \"\"\"Retrieve top-k relevant chunks for a question.\"\"\"
        # Generate query embedding
        query_embedding = await self.embedder.generate_embedding(question)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k=k)
        
        return results
    
    async def answer_question(self, question: str, k: int = 3, use_improved: bool = True) -> Dict:
        \"\"\"Answer a question using RAG.
        
        Returns:
            Dict with answer, sources, and metadata
        \"\"\"
        # Retrieve relevant chunks
        retrieved_chunks = await self.retrieve(question, k=k)
        
        # Handle case where no relevant documents found
        if not retrieved_chunks:
            return {
                'question': question,
                'answer': \"I couldn't find any relevant information in our policy documents to answer your question. Please contact our support team at support@techshop.com for assistance.\",
                'sources': [],
                'retrieval_scores': [],
                'prompt_version': 'N/A - No retrieval'
            }
        
        # Format context
        context = self.prompts.format_context(retrieved_chunks)
        
        # Select prompt template
        if use_improved:
            prompt = self.prompts.IMPROVED_PROMPT.format(context=context, question=question)
            version = \"v2_improved\"
        else:
            prompt = self.prompts.INITIAL_PROMPT.format(context=context, question=question)
            version = \"v1_initial\"
        
        # Generate answer
        user_message = UserMessage(text=prompt)
        answer = await self.llm.send_message(user_message)
        
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
        \"\"\"Answer multiple questions.\"\"\"
        results = []
        for question in questions:
            result = await self.answer_question(question, use_improved=use_improved)
            results.append(result)
        return results
```

---

## 7. `/app/backend/rag_system/evaluate.py`

See the file content in previous response (it's too long to repeat here).

---

## 8. `/app/backend/rag_assistant.py`

```python
#!/usr/bin/env python3
\"\"\"RAG Assistant CLI - Company Policy Q&A System\"\"\"

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_system.rag_pipeline import RAGPipeline
from rag_system.evaluate import RAGEvaluator
from rag_system.prompts import PromptTemplates


async def main():
    print(\"\"\"
    ╔════════════════════════════════════════════════════════════╗
    ║    RAG Assistant - Company Policy Q&A System              ║
    ║    TechShop Inc. Support Assistant                        ║
    ╚════════════════════════════════════════════════════════════╝
    \"\"\")
    
    # Initialize RAG pipeline
    data_dir = Path(__file__).parent / \"data\" / \"policies\"
    index_dir = Path(__file__).parent / \"vector_index\"
    
    print(\"Initializing RAG system...
\")
    rag = RAGPipeline(str(data_dir), str(index_dir))
    
    # Build or load index
    try:
        await rag.build_index(force_rebuild=False)
    except Exception as e:
        print(f\"Error building index: {e}\")
        return
    
    # Main menu
    while True:
        print(\"
\" + \"=\"*60)
        print(\"MAIN MENU\")
        print(\"=\"*60)
        print(\"1. Ask a question (interactive mode)\")
        print(\"2. Run evaluation suite\")
        print(\"3. Compare prompt versions\")
        print(\"4. View prompt improvement explanation\")
        print(\"5. View vector store statistics\")
        print(\"6. Rebuild vector index\")
        print(\"0. Exit\")
        print(\"=\"*60)
        
        choice = input(\"
Select an option (0-6): \").strip()
        
        if choice == '1':
            await interactive_mode(rag)
        
        elif choice == '2':
            await run_evaluation(rag)
        
        elif choice == '3':
            await compare_prompts(rag)
        
        elif choice == '4':
            print(\"
\" + PromptTemplates.get_improvement_explanation())
            input(\"
Press Enter to continue...\")
        
        elif choice == '5':
            stats = rag.vector_store.get_stats()
            print(\"
Vector Store Statistics:\")
            for key, value in stats.items():
                print(f\"  {key}: {value}\")
            input(\"
Press Enter to continue...\")
        
        elif choice == '6':
            print(\"
Rebuilding vector index...\")
            await rag.build_index(force_rebuild=True)
            print(\"Index rebuilt successfully!\")
            input(\"
Press Enter to continue...\")
        
        elif choice == '0':
            print(\"
Thank you for using RAG Assistant! Goodbye.\")
            break
        
        else:
            print(\"
Invalid option. Please try again.\")


async def interactive_mode(rag: RAGPipeline):
    \"\"\"Interactive Q&A mode.\"\"\"
    print(\"
\" + \"=\"*60)
    print(\"INTERACTIVE Q&A MODE\")
    print(\"=\"*60)
    print(\"Ask questions about TechShop policies.\")
    print(\"Type 'back' to return to main menu.
\")
    
    while True:
        question = input(\"
Your question: \").strip()
        
        if question.lower() in ['back', 'exit', 'quit']:
            break
        
        if not question:
            print(\"Please enter a question.\")
            continue
        
        print(\"
Processing...\")
        
        try:
            result = await rag.answer_question(question, use_improved=True)
            
            print(\"
\" + \"-\"*60)
            print(\"ANSWER:\")
            print(\"-\"*60)
            print(result['answer'])
            print(\"
\" + \"-\"*60)
            print(\"SOURCES:\")
            for i, source in enumerate(result['sources'], 1):
                print(f\"  {i}. {source['policy']} (chunk {source['chunk_id']})\")
            print(\"-\"*60)
            
        except Exception as e:
            print(f\"
Error: {e}\")
            print(\"Please try again or contact support.\")


async def run_evaluation(rag: RAGPipeline):
    \"\"\"Run evaluation suite.\"\"\"
    print(\"
\" + \"=\"*60)
    print(\"RUNNING EVALUATION SUITE\")
    print(\"=\"*60)
    
    # Run evaluation with improved prompt
    results = await RAGEvaluator.run_evaluation(rag, prompt_version='improved')
    
    # Print summary
    RAGEvaluator.print_summary(results)
    
    # Save results
    output_path = Path(__file__).parent / \"evaluation_results.json\"
    RAGEvaluator.save_results(results, str(output_path))
    
    input(\"
Press Enter to continue...\")


async def compare_prompts(rag: RAGPipeline):
    \"\"\"Compare initial vs improved prompt on sample questions.\"\"\"
    print(\"
\" + \"=\"*60)
    print(\"PROMPT COMPARISON\")
    print(\"=\"*60)
    
    test_questions = [
        \"What is the refund policy for digital products?\",
        \"Do you offer student discounts?\"
    ]
    
    for question in test_questions:
        print(f\"
Question: {question}\")
        print(\"
\" + \"-\"*60)
        
        # Initial prompt
        print(\"INITIAL PROMPT (V1):\")
        result_v1 = await rag.answer_question(question, use_improved=False)
        print(result_v1['answer'])
        
        print(\"
\" + \"-\"*60)
        
        # Improved prompt
        print(\"IMPROVED PROMPT (V2):\")
        result_v2 = await rag.answer_question(question, use_improved=True)
        print(result_v2['answer'])
        
        print(\"
\" + \"=\"*60)
    
    input(\"
Press Enter to continue...\")


if __name__ == \"__main__\":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(\"

Interrupted by user. Exiting...\")
        sys.exit(0)
```

---

## 9. Sample Policy Documents

### `/app/backend/data/policies/refund_policy.txt`

```
COMPANY REFUND POLICY
Last Updated: January 2026

1. OVERVIEW
At TechShop Inc., customer satisfaction is our priority. This refund policy outlines the terms and conditions for requesting refunds on purchased products.

2. ELIGIBILITY FOR REFUNDS
- Products must be returned within 30 days of purchase
- Items must be in original condition with tags attached
- Digital products are non-refundable once downloaded
- Custom or personalized items cannot be refunded
- Sale items marked as \"Final Sale\" are not eligible for refunds

3. REFUND PROCESS
Step 1: Contact our support team at support@techshop.com with your order number
Step 2: Receive a Return Merchandise Authorization (RMA) number within 2 business days
Step 3: Ship the item back using the provided shipping label
Step 4: Once received, refunds are processed within 5-7 business days

4. REFUND METHODS
- Original payment method: Full refund minus shipping costs
- Store credit: Full refund plus 10% bonus
- Defective items: Full refund including shipping costs

5. EXCEPTIONS
- Opened software or media cannot be refunded
- Products damaged due to misuse are not eligible
- Items purchased from third-party sellers follow their refund policies

6. PARTIAL REFUNDS
Partial refunds may be granted for:
- Items with minor damage not reported initially
- Products missing non-essential accessories
- Items returned after 30 days but within 60 days (50% refund)

7. CONTACT INFORMATION
For refund inquiries:
Email: support@techshop.com
Phone: 1-800-TECH-SHOP
Business Hours: Monday-Friday, 9 AM - 6 PM EST
```

### Similar structure for `cancellation_policy.txt` and `shipping_policy.txt` (see earlier in this conversation)

---

## 10. `/app/backend/requirements.txt`

Key dependencies:
```
fastapi==0.110.1
uvicorn==0.25.0
python-dotenv>=1.0.1
emergentintegrations==0.1.0
faiss-cpu==1.13.2
tiktoken==0.12.0
sentence-transformers==5.2.2
numpy>=2.4.1
motor==3.3.1
pydantic>=2.6.4
```

Full list is in the previous response.

---

## 🚀 How to Run

1. **Install dependencies:**
```bash
cd /app/backend
pip install -r requirements.txt
```

2. **Run the assistant:**
```bash
python3 rag_assistant.py
```

3. **Try interactive mode:**
- Select option 1
- Ask: \"What is the refund policy for digital products?\"
- Ask: \"Do you offer student discounts?\" (tests edge case)

4. **Run evaluation:**
- Select option 2
- See 100% accuracy and hallucination prevention scores

---

## 📊 Results

- **Accuracy**: 8/8 (100%)
- **Hallucination Prevention**: 8/8 (100%)
- **Clarity**: 6/8 (75%)
- **Overall Pass Rate**: 6/8 (75%)

---

This is a complete, production-ready RAG system!
"
