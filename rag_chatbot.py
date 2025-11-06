# rag_chatbot.py
import os
from dotenv import load_dotenv
import numpy as np
import faiss
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
import textract
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient
from typing import List, Tuple

# -------- Configuration --------
load_dotenv()
HF_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Set HF_API_TOKEN environment variable with your Hugging Face token.")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-large"

# -------- Hugging Face Clients --------
embed_client = InferenceClient(model=EMBED_MODEL, token=HF_TOKEN)
gen_client = InferenceClient(model=GEN_MODEL, token=HF_TOKEN)

# -------- Document Loaders --------
def extract_text_from_pdf(path: str) -> str:
    """Extract text from PDF using LangChain’s PyMuPDFLoader."""
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])

def extract_text_from_docx(path: str) -> str:
    """Extract text from DOCX using LangChain’s Docx2txtLoader."""
    loader = Docx2txtLoader(path)
    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])

def extract_text_from_doc(path: str) -> str:
    """Extract text from legacy DOC files using textract."""
    try:
        text = textract.process(path).decode("utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"Failed to read .doc file '{path}': {e}")
    return text

def extract_text_from_txt(path: str) -> str:
    """Extract text from plain TXT files."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text(path: str) -> str:
    """Auto-select extraction method based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    elif ext == ".doc":
        return extract_text_from_doc(path)
    elif ext == ".txt":
        return extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# -------- Splitting --------
def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "], # Add "" character if I later decide to support OCR’d PDFs or non-English text without spaces
    )
    return splitter.split_text(text)

# -------- Embedding & Vector Store --------
def embed_texts(texts: List[str]) -> List[np.ndarray]:
    vectors = []
    for t in texts:
        # Use the feature_extraction method for embeddings
        resp = embed_client.feature_extraction(text=t)
        
        # The response is already a numpy array
        if isinstance(resp, np.ndarray):
            vec = resp.astype(np.float32)
        elif isinstance(resp, list):
            vec = np.array(resp, dtype=np.float32)
        else:
            raise RuntimeError(f"Unexpected embedding response: {type(resp)}, value: {resp}")
        
        vectors.append(vec)
    
    print("Gab. len(vectors):", len(vectors))
    if len(vectors) > 0:
        print("Gab. vectors[0].shape:", vectors[0].shape)
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
    try:
        resp = gen_client.text_generation(
            prompt=prompt,
            max_new_tokens=600
        )
        return resp
    except Exception as e:
        return f"Error generating essay: {str(e)}\n\nPlease try again or use a different model."
