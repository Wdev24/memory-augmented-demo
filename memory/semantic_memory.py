# memory/semantic_memory.py

import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticMemory:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", threshold: float = 0.7):
        """
        - Uses cosine similarity instead of FAISS.
        - threshold: cosine similarity above which we call it a hit.
        """
        self.model = SentenceTransformer(embedding_model_name)
        self.threshold = threshold
        # Store normalized embeddings, texts, and responses
        self.embeddings = []  # list of 1D np.ndarray
        self.texts = []
        self.responses = []

    def add_to_memory(self, input_text: str, response_text: str):
        # 1) Encode and normalize
        emb = self.model.encode([input_text], normalize_embeddings=True)[0]
        self.embeddings.append(emb)
        self.texts.append(input_text)
        self.responses.append(response_text)

    def query(self, input_text: str):
        # If empty, immediate miss
        if not self.embeddings:
            return False, None

        # Encode and normalize
        query_emb = self.model.encode([input_text], normalize_embeddings=True)[0]
        # Compute cosine similarities
        sims = np.dot(self.embeddings, query_emb)
        idx = int(np.argmax(sims))
        best_sim = float(sims[idx])

        if best_sim >= self.threshold:
            return True, self.responses[idx]
        else:
            return False, None
