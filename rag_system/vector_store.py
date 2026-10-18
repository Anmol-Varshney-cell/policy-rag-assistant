"""Vector store using FAISS for semantic search."""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple


class FAISSVectorStore:
    """FAISS-based vector store for semantic search."""
    
    def __init__(self, dimension: int = 384):
        """Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2, 1536 for OpenAI)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
        self.chunks = []  # Store chunk data
        self.metadata = []  # Store metadata
    
    def add_embeddings(self, embeddings: List[List[float]], chunks: List[Dict]):
        """Add embeddings and their corresponding chunks to the store."""
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype='float32')
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store chunks and metadata
        self.chunks.extend([chunk['text'] for chunk in chunks])
        self.metadata.extend([chunk['metadata'] for chunk in chunks])
    
    def search(self, query_embedding: List[float], k: int = 3) -> List[Tuple[str, Dict, float]]:
        """Search for top-k most similar chunks.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
        
        Returns:
            List of (chunk_text, metadata, distance) tuples
        """
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
        """Save index and metadata to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss.index"))
        
        # Save chunks and metadata
        with open(path / "chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        with open(path / "metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load(self, path: str):
        """Load index and metadata from disk."""
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "faiss.index"))
        
        # Load chunks and metadata
        with open(path / "chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
        with open(path / "metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            'total_chunks': len(self.chunks),
            'dimension': self.dimension,
            'index_size': self.index.ntotal
        }
