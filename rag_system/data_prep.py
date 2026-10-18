"""Data preparation module for loading and chunking documents."""

import os
from pathlib import Path
from typing import List, Dict
import tiktoken


class DocumentLoader:
    """Load and prepare documents for RAG."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
    
    def load_documents(self) -> List[Dict[str, str]]:
        """Load all policy documents from the data directory."""
        documents = []
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        for file_path in self.data_dir.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    'filename': file_path.name,
                    'content': content,
                    'source': str(file_path)
                })
        
        return documents
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))


class TextChunker:
    """
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
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split text into overlapping chunks."""
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
        """Chunk multiple documents."""
        all_chunks = []
        
        for doc in documents:
            metadata = {
                'filename': doc['filename'],
                'source': doc['source']
            }
            chunks = self.chunk_text(doc['content'], metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
