import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional

class SemanticMemory:
    def __init__(self, similarity_threshold: float = 0.7):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = similarity_threshold
        self.embeddings = []
        self.responses = []
        self.queries = []
        
        # Initialize FAISS index
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity with FAISS"""
        return embedding / np.linalg.norm(embedding)
    
    def search(self, query: str) -> Optional[str]:
        """Search for cached response based on semantic similarity"""
        if len(self.responses) == 0:
            return None
            
        # Get query embedding
        query_embedding = self.model.encode([query])[0]
        query_embedding = self._normalize_embedding(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.reshape(1, -1), 1)
        
        if scores[0][0] >= self.similarity_threshold:
            idx = indices[0][0]
            print(f"Cache hit! Similarity: {scores[0][0]:.3f} for query: '{self.queries[idx]}'")
            return self.responses[idx]
        
        return None
    
    def store(self, query: str, response: str) -> None:
        """Store query-response pair with embedding"""
        # Get and normalize embedding
        embedding = self.model.encode([query])[0]
        embedding = self._normalize_embedding(embedding)
        
        # Store in memory
        self.embeddings.append(embedding)
        self.responses.append(response)
        self.queries.append(query)
        
        # Add to FAISS index
        self.index.add(embedding.reshape(1, -1))
        
        print(f"Stored new entry. Total cached responses: {len(self.responses)}")
    
    def get_stats(self) -> dict:
        """Get memory statistics"""
        return {
            "total_entries": len(self.responses),
            "embedding_dimension": self.dimension,
            "similarity_threshold": self.similarity_threshold
        }