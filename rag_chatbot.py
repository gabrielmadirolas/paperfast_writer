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
GEN_MODEL = "HuggingFaceTB/SmolLM3-3B"
#GEN_MODEL = "meta-llama/Llama-3.3-70B-Instruct:scaleway"

# -------- Hugging Face Clients --------
embed_client = InferenceClient(model=EMBED_MODEL, token=HF_TOKEN)
# Don't create gen_client here since we'll create it in the function
# gen_client = InferenceClient(model=GEN_MODEL, token=HF_TOKEN)

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
    #print(store) # delete comment
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

import requests
import json

def generate_essay(prompt: str) -> str:
    """Generate essay using direct API call."""
    # Use Hugging Face's serverless inference API
    API_URL = f"https://router.huggingface.co/hf-inference/{GEN_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.9,
            "do_sample": True,
            "return_full_text": False
        },
        "options": {
            "wait_for_model": True  # Wait if model is loading
        }
    }
    
    try:
        # Gab: delete from here
        r = requests.get(f"https://huggingface.co/api/models/{GEN_MODEL}")
        info = r.json()
        print(info.get("inference", "No inference API available"))
        # Gab: delete to here
        print(f"Calling API for model: {GEN_MODEL}")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 503:
            # Model is loading, wait and retry
            print("Model is loading, waiting 20 seconds...")
            import time
            time.sleep(20)
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            print(f"Error response: {response.text}")
            return f"API Error (status {response.status_code}): {response.text}"
        
        result = response.json()
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        # Parse the response
        if isinstance(result, list) and len(result) > 0:
            generated = result[0].get("generated_text", "")
            return generated if generated else str(result)
        elif isinstance(result, dict):
            if "error" in result:
                return f"API Error: {result['error']}"
            return result.get("generated_text", str(result))
        else:
            return str(result)
            
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}"
    except Exception as e:
        import traceback
        print(f"Unexpected error: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return f"Error: {type(e).__name__}: {str(e)}"


'''
import requests
import time

def generate_essay(prompt: str) -> str:
    """Generate essay using InferenceClient with configured model."""
    try:
        print(f"Attempting to use model: {GEN_MODEL}")
        client = InferenceClient(model=GEN_MODEL, token=HF_TOKEN)
        
        response = client.text_generation(
            prompt,
            max_new_tokens=500,
            temperature=0.9,
            do_sample=True
        )
        
        print(f"Success! Response type: {type(response)}")
        return response
        
    except Exception as e:
        print(f"Generation error with {GEN_MODEL}: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback to base GPT-2
        try:
            print("Trying fallback model: openai-community/gpt2")
            fallback_client = InferenceClient(model="openai-community/gpt2", token=HF_TOKEN)
            response = fallback_client.text_generation(
                prompt,
                max_new_tokens=400,
            )
            print(f"Fallback success! Response type: {type(response)}")
            return response
        except Exception as e2:
            print(f"Fallback error: {type(e2).__name__}: {str(e2)}")
            import traceback
            traceback.print_exc()
            
            return f"""Error: Unable to generate essay using Hugging Face API.

Primary model ({GEN_MODEL}) error: {type(e).__name__}: {str(e)}
Fallback model error: {type(e2).__name__}: {str(e2)}

Please check:
1. Your HF_API_TOKEN is valid
2. The Hugging Face API is accessible
3. Try again in a few minutes"""
'''