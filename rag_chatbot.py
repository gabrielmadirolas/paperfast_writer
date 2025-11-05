# rag_chatbot.py
import os
from dotenv import load_dotenv
import numpy as np
import faiss
try:
    import pymupdf
except ImportError:
    import fitz as pymupdf
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient
from typing import List, Tuple

# -------- Configuration --------
load_dotenv()
HF_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Set HF_API_TOKEN environment variable with your Hugging Face token.")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-base"

# -------- Hugging Face Clients --------
embed_client = InferenceClient(model=EMBED_MODEL, token=HF_TOKEN)
gen_client = InferenceClient(model=GEN_MODEL, token=HF_TOKEN)

# -------- File Readers --------
def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF file using PyMuPDF (new or legacy import)."""
    text = []
    with pymupdf.open(path) as pdf:
        for page in pdf:
            page_text = page.get_text("text")
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text])

def extract_text(path: str) -> str:
    ext = path.lower().split(".")[-1]
    if ext == "pdf":
        return extract_text_from_pdf(path)
    elif ext in ("docx", "doc"):
        return extract_text_from_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# -------- Chunking --------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_text(text)

# -------- Embedding & Vector Store --------
def embed_texts(texts: List[str]) -> List[np.ndarray]:
    vectors = []
    for t in texts:
        resp = embed_client(inputs=t)
        if isinstance(resp, list) and isinstance(resp[0], list):
            vec = np.array(resp[0], dtype=np.float32)
        elif isinstance(resp, list):
            vec = np.array(resp, dtype=np.float32)
        else:
            raise RuntimeError(f"Unexpected embedding response: {type(resp)}")
        vectors.append(vec)
    return vectors

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.metadatas = []

    def add(self, vectors: List[np.ndarray], metadatas: List[dict]):
        arr = np.vstack(vectors).astype("float32")
        faiss.normalize_L2(arr)
        self.index.add(arr)
        self.metadatas.extend(metadatas)

    def search(self, query_vec: np.ndarray, k: int = 5):
        q = query_vec.astype("float32").reshape(1, -1)
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            results.append((float(score), self.metadatas[idx]))
        return results

# -------- Pipeline Functions --------
def ingest_documents(paths: List[str]) -> Tuple[VectorStore, int]:
    all_chunks, metas = [], []
    for path in paths:
        text = extract_text(path)
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            metas.append({"source": os.path.basename(path), "chunk": i, "text": c})

    vectors = embed_texts(all_chunks)
    dim = vectors[0].shape[0]
    store = VectorStore(dim)
    store.add(vectors, metas)
    return store, dim

def retrieve_relevant(store: VectorStore, query: str, k: int = 5):
    qvec = embed_texts([query])[0]
    hits = store.search(qvec, k)
    context = "\n\n---\n\n".join([h[1]["text"] for h in hits])
    return context, hits

def build_prompt(context: str, query: str) -> str:
    return f"""
You are an assistant that writes academic-style papers.
Using ONLY the context below (user's notes) and general knowledge,
write a coherent academic paper addressing the query.
If information is missing, write 'not present in notes' instead of inventing it.

### CONTEXT
{context}

### USER QUERY
{query}

### ESSAY
"""

def generate_essay(prompt: str) -> str:
    resp = gen_client(inputs=prompt, parameters={"max_new_tokens": 600})
    if isinstance(resp, list):
        return resp[0].get("generated_text", "")
    if isinstance(resp, dict):
        return resp.get("generated_text", "")
    return str(resp)
