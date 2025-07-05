# rag_engine/reranker.py

from sentence_transformers import CrossEncoder
from typing import List, Tuple, Dict

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        passages: List[Tuple[str, Dict]],
        top_n: int = 3,
        boost_on_heading: bool = True
    ) -> List[Tuple[str, Dict, float]]:
        """
        passages: list of (text, metadata) tuples
        Returns top_n (text, metadata, score), sorted by score descending.
        Optionally boosts passages whose heading matches query keywords.
        """
        # Standard cross-encoder scoring
        pairs = [(query, passage) for passage, _ in passages]
        scores = self.model.predict(pairs)

        # Optional: boost passages with heading match
        if boost_on_heading:
            query_keywords = set(query.lower().split())
            for idx, (_, meta) in enumerate(passages):
                heading = meta.get("heading", "")
                if heading and any(word in heading.lower() for word in query_keywords):
                    scores[idx] += 0.3  # Boost: tune as desired

        # Return sorted by (text, metadata, score)
        scored_passages = sorted(
            [(passages[i][0], passages[i][1], scores[i]) for i in range(len(passages))],
            key=lambda x: x[2],
            reverse=True
        )
        return scored_passages[:top_n]
