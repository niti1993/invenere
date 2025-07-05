# rag_engine/embedder.py

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Converts a list of text chunks into vector embeddings.
        Returns a NumPy array of shape (n_chunks, embedding_dim).
        """
        return self.model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
