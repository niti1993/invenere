# rag_engine/chunker.py
import nltk
nltk.download('punkt', quiet=True)

import re
from typing import List, Dict

def robust_chunker(
    text: str,
    max_length: int = 350,
    overlap: int = 50,
    heading_regex: str = r"(Section \d+|^# |\n\d+\.\s)"
) -> List[Dict]:
    """
    Robust chunker: 
    1. Split by headings (if present, using heading_regex),
    2. Fallback: sentence-based chunking (if long enough),
    3. Fallback: sliding window by characters.
    Adds metadata: heading (if any), chunk_start, chunk_end.
    Returns list of dicts: {"text": ..., "metadata": ...}
    """
    chunks = []
    used_headings = False

    # 1. Try section/heading-based chunking
    headings = [m.start() for m in re.finditer(heading_regex, text, re.MULTILINE)]
    if len(headings) > 1:
        used_headings = True
        for i, start in enumerate(headings):
            end = headings[i + 1] if i + 1 < len(headings) else len(text)
            chunk_text = text[start:end].strip()
            # Extract heading (first non-empty line)
            heading_lines = [l for l in chunk_text.splitlines() if l.strip()]
            heading = heading_lines[0] if heading_lines else ""
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "heading": heading,
                    "chunk_start": start,
                    "chunk_end": end,
                    "chunk_type": "heading"
                }
            })
        return chunks

    # 2. Fallback: sentence-based chunking (using nltk)
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        chunk = ""
        sent_idx = 0
        while sent_idx < len(sentences):
            chunk_start_idx = sent_idx
            chunk = sentences[sent_idx]
            sent_idx += 1
            # Add sentences until we hit max_length
            while sent_idx < len(sentences) and len(chunk) + len(sentences[sent_idx]) < max_length:
                chunk += " " + sentences[sent_idx]
                sent_idx += 1
            # Overlap: backtrack by N sentences if possible
            overlap_idx = max(0, sent_idx - overlap // 20)
            chunks.append({
                "text": chunk.strip(),
                "metadata": {
                    "heading": None,
                    "chunk_start_sentence": chunk_start_idx,
                    "chunk_end_sentence": sent_idx - 1,
                    "chunk_type": "sentence"
                }
            })
        if chunks:
            return chunks
    except Exception as e:
        print("Sentence-based chunking failed:", e)

    # 3. Fallback: sliding window by characters
    i = 0
    n = len(text)
    while i < n:
        chunk_text = text[i:i + max_length]
        chunks.append({
            "text": chunk_text.strip(),
            "metadata": {
                "heading": None,
                "chunk_start": i,
                "chunk_end": min(i + max_length, n),
                "chunk_type": "window"
            }
        })
        i += max_length - overlap
    return chunks

# Example function for your pipeline (for plain text docs)
def chunk_text(doc_text: str) -> List[str]:
    """
    Returns a list of just the chunk texts, for backward compatibility.
    If you want metadata, use robust_chunker directly.
    """
    chunk_dicts = robust_chunker(doc_text)
    return [d["text"] for d in chunk_dicts]
