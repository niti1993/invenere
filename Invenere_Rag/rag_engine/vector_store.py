# rag_engine/vector_store.py

import chromadb

class VectorStore:
    def __init__(self, collection_name="mydocs", persist_dir="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        if collection_name in [c.name for c in self.client.list_collections()]:
            self.collection = self.client.get_collection(collection_name)
        else:
            self.collection = self.client.create_collection(collection_name)
        self.ids = set(self.collection.get()["ids"])  # Avoid duplicate IDs

    def sanitize_metadata(self, meta):
        """
        Ensure all metadata values are strings (and not None).
        """
        return {str(k): ("" if v is None else str(v)) for k, v in meta.items()}

    def add(self, embeddings, chunks, filepaths, metadatas=None):
        ids = []
        documents = []
        all_metadatas = []
        if metadatas is None:
            metadatas = [{} for _ in chunks]
        for idx, (emb, chunk, filepath, meta) in enumerate(zip(embeddings, chunks, filepaths, metadatas)):
            id_str = f"{filepath}_{idx}"
            if id_str in self.ids:
                continue
            ids.append(id_str)
            documents.append(chunk)
            meta_with_source = dict(meta)
            meta_with_source["source"] = filepath
            sanitized = self.sanitize_metadata(meta_with_source)
            all_metadatas.append(sanitized)
            self.ids.add(id_str)
        batch_size = 5000
        total = len(ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_embeddings = [e.tolist() for e in embeddings[start:end]]
            batch_documents = documents[start:end]
            batch_metadatas = all_metadatas[start:end]
            batch_ids = ids[start:end]
            self.collection.add(
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

    def search(self, embedding, top_k=5):
        result = self.collection.query(
            query_embeddings=[embedding[0].tolist()],
            n_results=top_k
        )
        return list(zip(result["documents"][0], result["metadatas"][0]))

    def hybrid_search(self, user_query: str, query_embedding: list, top_k: int = 10):
        dense_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        dense_docs = list(zip(dense_results["documents"][0], dense_results["metadatas"][0]))
        sparse_results = self.collection.query(
            query_texts=[user_query],
            n_results=top_k
        )
        sparse_docs = list(zip(sparse_results["documents"][0], sparse_results["metadatas"][0]))
        seen = set()
        merged = []
        for doc, meta in dense_docs + sparse_docs:
            if doc not in seen:
                seen.add(doc)
                merged.append((doc, meta))
            if len(merged) >= top_k:
                break
        return merged
