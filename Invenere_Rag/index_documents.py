import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from rag_engine.rag_pipeline import RAGPipeline
from rag_engine.chunker import robust_chunker        # <- Import robust chunker!
from rag_engine.parser import load_documents_from_desktop

BATCH_SIZE = 200

def batch_index():
    print("Loading documents from Desktop...")
    docs = load_documents_from_desktop()
    total = len(docs)
    print(f"Found {total} files to index.")
    batches = [docs[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

    pipeline = RAGPipeline()

    for batch_num, batch in enumerate(batches, start=1):
        print(f"\nProcessing batch {batch_num}/{len(batches)}...")
        all_chunks = []
        all_filepaths = []
        all_metadatas = []

        for item in batch:
            try:
                file_path, text = item[:2]
                if text and len(text.strip()) > 0:
                    chunk_dicts = robust_chunker(text)
                    for chunk_dict in chunk_dicts:
                        all_chunks.append(chunk_dict['text'])
                        all_filepaths.append(file_path)
                        all_metadatas.append(chunk_dict.get('metadata', {}))
                else:
                    print(f"Skipped empty file: {file_path}")
            except Exception as e:
                print(f"Skipping {item}: {e}")

        if not all_chunks:
            print(f"No valid files found in batch {batch_num}, skipping batch.")
            continue

        try:
            # Update your pipeline.index_documents to accept metadatas!
            pipeline.index_documents(
                list(zip(all_filepaths, all_chunks, all_metadatas))
            )
            print(f"Batch {batch_num} complete with {len(all_chunks)} chunks.")
        except Exception as e:
            print(f"Batch {batch_num} failed: {e}")
            continue

if __name__ == "__main__":
    batch_index()
    print("\nIndexing complete! You can now run queries instantly.")
