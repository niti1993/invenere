from pathlib import Path
from typing import List, Tuple
import fitz
import docx

def extract_text_from_pdf(file_path: str) -> str:
    try: 
        doc = fitz.open(file_path)
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        print (f"⚠️ Skipping PDF due to error: {file_path} — {e}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs)

def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def extract_text(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif file_path.endswith(".txt") or file_path.endswith(".md"):
        return extract_text_from_txt(file_path)
    else:
        print(f"Unsupported file format: {file_path}")
        return ""

def load_documents_from_desktop() -> List[Tuple[str, str]]:
    desktop_path = Path.home() / "Desktop"
    supported_exts = (".pdf", ".docx", ".txt", ".md")
    documents = []

    for file_path in desktop_path.rglob("*"):
        if file_path.suffix.lower() in supported_exts and not file_path.name.startswith("~$"):
            try:
                content = extract_text(str(file_path))
                if content.strip():
                    documents.append((str(file_path), content))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return documents
