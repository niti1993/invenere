from rag_engine.chunker import chunk_text
from rag_engine.embedder import Embedder
from rag_engine.vector_store import VectorStore
from rag_engine.reranker import Reranker
from rag_engine.llama_interface import query_llama

def build_history_enhanced_query(query, history, history_turns=1):
    """
    Combine the user's current query with the last N Q&A for reranking.
    This helps the reranker consider both the present question and recent context.
    Only uses history items that are tuple/list with at least 2 elements.
    """
    recent_history = ""
    for item in history[-history_turns:]:
        if isinstance(item, (tuple, list)) and len(item) > 1:
            recent_history += f"Q: {item[0]}\nA: {item[1]}\n"
    if recent_history:
        return f"Conversation so far:\n{recent_history}Current question: {query}"
    else:
        return query


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.reranker = Reranker()

    def index_documents(self, documents: list[tuple[str, str, dict]]):
        """
        Accepts: list of (file_path, chunk_text, chunk_metadata)
        """
        all_chunks = []
        all_filepaths = []
        all_metadatas = []

        for file_path, chunk_text, chunk_metadata in documents:
            if not chunk_text or len(chunk_text.strip()) == 0:
                print(f"Skipping empty chunk in file: {file_path}")
                continue
            all_chunks.append(chunk_text)
            all_filepaths.append(file_path)
            all_metadatas.append(chunk_metadata if chunk_metadata else {})

        if all_chunks:
            embeddings = self.embedder.embed_chunks(all_chunks)
            self.vector_store.add(
                embeddings,
                all_chunks,
                all_filepaths,
                metadatas=all_metadatas
            )
            print(f"Indexed {len(all_chunks)} chunks in this batch.")
        else:
            print("No valid chunks to index in this batch!")

    def query(
        self,
        user_query: str,
        top_k: int = 10,
        final_k: int = 5,
        return_sources: bool = False,
        use_hybrid: bool = True,
        history: list = None,
        history_turns: int = 1
    ):
        if history is None:
            history = []

        query_embedding = self.embedder.embed_chunks([user_query])

        # Step 1: Retrieve top-k with metadata
        if use_hybrid:
            retrieved = self.vector_store.hybrid_search(
                user_query, query_embedding[0], top_k=top_k
            )
        else:
            retrieved = self.vector_store.search(query_embedding, top_k=top_k)

        # retrieved: list of (chunk_text, metadata)
        # For reranker, pass both chunk_text and metadata!
        history_enhanced_query = build_history_enhanced_query(
            user_query, history, history_turns=history_turns
        )

        reranked = self.reranker.rerank(
            history_enhanced_query,
            retrieved,
            top_n=final_k
        )
        # reranked: List[(chunk_text, metadata, score)]
        reranked_texts = [chunk for chunk, meta, score in reranked]
        reranked_sources = [meta.get('source', 'unknown') for chunk, meta, score in reranked]
        reranked_metadata = [meta for chunk, meta, score in reranked]

        # Build context string for LLM, including heading/section info
        context = "\n\n".join([
            f"[Section: {meta.get('heading', 'Unknown')}] {chunk}"
            if meta.get("heading") else chunk
            for chunk, meta in zip(reranked_texts, reranked_metadata)
        ])

        conversation_history = "\n".join([
    f"Q: {item[0]}\nA: {item[1]}"
    for item in history[-history_turns:]
    if isinstance(item, (tuple, list)) and len(item) > 1
])
        memory_block = (
            f"Previous Conversation:\n{conversation_history}\n\n"
            if conversation_history else ""
        )

        prompt = (
            "You are a helpful expert assistant. Carefully read the previous conversation and the context below. "
            "Answer the user's latest question by linking it to any relevant prior topics or examples. "
            "Use ONLY the information in the context for facts, but you may refer to previous Q&As to maintain coherence or connect ideas. "
            "If the context contains multiple relevant methods or ideas, name and compare them directly, with examples if possible. "
            "If the answer cannot be found in the context, reply with 'Not enough information in the context.'\n\n"
            f"{memory_block}"
            f"Context:\n{context}\n\n"
            f"Question: {user_query}\nDetailed Answer:"
        )

        print("\n--- Prompt passed to LLM ---")
        print(prompt)
        print("Prompt length (chars):", len(prompt))
        print("\n--- End of prompt ---")

        answer = query_llama(prompt)
        print("LLM Raw Answer:", answer)
        if return_sources:
            return answer, reranked_sources
        return answer
