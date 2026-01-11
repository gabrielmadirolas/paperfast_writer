# rag_chatbot.py
import os
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from odf import text, teletype
from odf.opendocument import load
import textract
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import faiss

import base64
import tempfile
import json
import gzip
import io

from huggingface_hub import InferenceClient


# -------- Configuration --------
load_dotenv()
HF_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Set HF_API_TOKEN environment variable with your Hugging Face token.")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# GEN_MODEL = "meta-llama/Llama-3.1-405B-Instruct" # Wasn't working last time, but I wish it to be the main option
GEN_MODEL = "meta-llama/Llama-3.3-70B-Instruct" # In case the above is too much inference-consuming
FBACK_GEN_MODEL = "HuggingFaceTB/SmolLM3-3B" # To be fallback model.

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

def extract_text_from_odt(path: str) -> str:
    """Extract text from ODT files using odfpy."""
    try:  
        doc = load(path)
        all_paragraphs = doc.getElementsByType(text.P)
        text_content = []
        
        for paragraph in all_paragraphs:
            text_content.append(teletype.extractText(paragraph))
        
        return "\n".join(text_content)
    except Exception as e:
        raise RuntimeError(f"Failed to read .odt file '{path}': {e}")

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
    elif ext == ".odt":
        return extract_text_from_odt(path)
    elif ext == ".txt":
        return extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# -------- Splitting --------
def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Warning: specifying the separators affects and even destroys chunk overlap
        # separators=["\n\n", "\n", ".", " "], # Add "" character if I later decide to support OCR’d PDFs or non-English text without spaces
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

def add_documents_to_store(store: VectorStore, paths: List[str], filenames: List[str] = None) -> Tuple[int, int]:
    """
    Add new documents to an existing vector store.
    
    Args:
        store: Existing vector store
        paths: List of file paths to add
        filenames: Optional list of original filenames (if different from paths)
    """
    all_chunks, metas = [], []
    
    # If no filenames provided, extract from paths
    if filenames is None:
        filenames = [os.path.basename(p) for p in paths]
    
    for path, filename in zip(paths, filenames):
        text = extract_text(path)
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            metas.append({"source": filename, "chunk": i, "text": c})
    
    if not all_chunks:
        return 0, 0
    
    vectors = embed_texts(all_chunks)
    store.add(vectors, metas)
    
    return len(all_chunks), len(paths)

# -------- Save & Load Vector Store --------

def export_store(store: VectorStore) -> bytes:
    """Serialize whole store to gzipped JSON bytes."""
    # Serialize FAISS index to bytes
    tmp_idx = tempfile.NamedTemporaryFile(delete=False, suffix='.index')
    try:
        faiss.write_index(store.index, tmp_idx.name)
        tmp_idx.close()
        with open(tmp_idx.name, "rb") as f:
            idx_bytes = f.read()
    finally:
        os.remove(tmp_idx.name)

    # Create package
    package = {
        "index_b64": base64.b64encode(idx_bytes).decode(),
        "metadata": store.metadatas,
        "dim": store.index.d,
    }
    
    # Serialize to JSON and compress
    json_bytes = json.dumps(package, ensure_ascii=False).encode()
    return gzip.compress(json_bytes)


def import_store(blob: bytes) -> VectorStore:
    """Deserialize store from gzipped JSON bytes."""
    # Decompress and parse JSON
    json_bytes = gzip.decompress(blob)
    package = json.loads(json_bytes.decode())
    
    # Extract fields
    index_b64 = package["index_b64"]
    meta_list = package["metadata"]
    dim = package["dim"]
    
    # Decode base64 and write to temp file
    index_bytes = base64.b64decode(index_b64)
    tmp_idx = tempfile.NamedTemporaryFile(delete=False, suffix='.index')
    try:
        tmp_idx.write(index_bytes)
        tmp_idx.close()
        index = faiss.read_index(tmp_idx.name)
    finally:
        os.remove(tmp_idx.name)
    
    # Reconstruct VectorStore
    store = VectorStore(dim)
    store.index = index
    store.metadatas = meta_list
    
    return store

# -------- Pipeline Functions --------
def ingest_documents(paths: List[str], filenames: List[str] = None) -> Tuple[VectorStore, int]:
    """
    Ingest documents and create vector store.
    
    Args:
        paths: List of file paths to process
        filenames: Optional list of original filenames (if different from paths)
    """
    all_chunks, metas = [], []
    
    # If no filenames provided, extract from paths
    if filenames is None:
        filenames = [os.path.basename(p) for p in paths]
    
    for path, filename in zip(paths, filenames):
        text = extract_text(path)
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            metas.append({"source": filename, "chunk": i, "text": c})

    vectors = embed_texts(all_chunks)
    dim = vectors[0].shape[0]
    store = VectorStore(dim)
    store.add(vectors, metas)
    return store, dim

def retrieve_relevant(store: VectorStore, query: str, k: int = 5):
    qvec = embed_texts([query])[0]
    hits = store.search(qvec, k)
    #print(hits) # delete comment
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
    """Generate essay using Hugging Face InferenceClient with multiple fallback methods."""
    
    # Method 1: Try chat.completions.create (OpenAI-compatible, preferred)
    try:
        print(f"Attempting chat.completions.create with model: {GEN_MODEL}")
        
        completion = gen_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2500,
            temperature=0.7
        )
        
        # Extract the generated text
        generated_text = completion.choices[0].message.content
        # print(completion.choices[0].message) # Here for debugingging purposes, remove later
        print(f"chat.completions.create successful!")
        print(f"Generated {len(generated_text)} characters")
        
        # Handle thinking models - remove <think>...</think> tags
        import re
        cleaned_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL)
        cleaned_text = re.sub(r'<think>.*', '', cleaned_text, flags=re.DOTALL)
        cleaned_text = cleaned_text.strip()
        
        print(f"After cleaning: {len(cleaned_text)} characters")
        
        # If cleaning removed everything, return original with escaped HTML
        if not cleaned_text:
            print("Warning: Cleaning removed all text, returning original")
            return generated_text.replace('<', '&lt;').replace('>', '&gt;')
        
        return cleaned_text
            
    except Exception as e:
        print(f"chat.completions.create failed: {type(e).__name__}: {str(e)}")
        
        # Method 2: Try text_generation (for base/completion models)
        try:
            print(f"Trying text_generation with model: {GEN_MODEL}")
            
            response = gen_client.text_generation(
                prompt=prompt,
                max_new_tokens=2500,
                temperature=0.7,
                return_full_text=False
            )
            
            print(f"text_generation successful!")
            print(f"Generated {len(response)} characters")
            
            return response
                
        except Exception as e2:
            print(f"text_generation failed: {type(e2).__name__}: {str(e2)}")
            
            # Method 3: Try legacy chat_completion (last resort)
            try:
                print(f"Trying legacy chat_completion with model: {GEN_MODEL}")
                
                messages = [
                    {"role": "system", "content": "/no_think"},
                    {"role": "user", "content": prompt}
                ]
                
                response = gen_client.chat_completion(
                    messages=messages,
                    max_tokens=2500,
                    temperature=0.7
                )
                
                # Extract the generated text
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    print(f"Legacy chat_completion successful!")
                    generated_text = response.choices[0].message.content
                    
                    # Clean thinking tags here too
                    import re
                    cleaned_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL)
                    cleaned_text = re.sub(r'<think>.*', '', cleaned_text, flags=re.DOTALL)
                    cleaned_text = cleaned_text.strip()
                    
                    if not cleaned_text:
                        return generated_text.replace('<', '&lt;').replace('>', '&gt;')
                    
                    return cleaned_text
                else:
                    return str(response)
                    
            except Exception as e3:
                print(f"All methods failed:")
                print(f"1. chat.completions.create: {type(e).__name__}: {str(e)}")
                print(f"2. text_generation: {type(e2).__name__}: {str(e2)}")
                print(f"3. chat_completion: {type(e3).__name__}: {str(e3)}")
                
                return f"""Unable to generate essay using Hugging Face API.

Tried all available methods:
1. chat.completions.create: {type(e).__name__}
2. text_generation: {type(e2).__name__}
3. chat_completion: {type(e3).__name__}

The model '{GEN_MODEL}' may not be compatible with your enabled inference providers.
Check your Hugging Face Pro plan settings or try a different model."""

