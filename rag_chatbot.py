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
GEN_MODEL = "HuggingFaceTB/SmolLM3-3B" # This one works, but only with chat_completion()
# GEN_MODEL = "moonshotai/Kimi-K2-Thinking" # This one works, but only with chat_completion()
# GEN_MODEL = "Qwen/Qwen3-4B-Thinking-2507" # This one works, but only with chat_completion()
# GEN_MODEL = "katanemo/Arch-Router-1.5B" # This one does not work
# GEN_MODEL = "meta-llama/Llama-3.3-70B-Instruct:scaleway" # This one does not work

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


# chat.completions version

def generate_essay(prompt: str) -> str:
    """Generate essay using Hugging Face InferenceClient with OpenAI-compatible API."""
    try:
        client = InferenceClient(token=HF_TOKEN)
        
        # Use the new OpenAI-compatible API (preferred method)
        print(f"Attempting chat.completions.create with model: {GEN_MODEL}")
        
        completion = client.chat.completions.create(
            model=GEN_MODEL,
            messages=[
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2500,
            temperature=0.7
        )
        
        # Extract the generated text
        generated_text = completion.choices[0].message.content
        print(f"Chat completions successful!")
        print(f"Generated {len(generated_text)} characters")
        
        # Handle thinking models - remove <think>...</think> tags
        import re
        # Remove thinking tags and their content
        cleaned_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL)
        # Also try removing if they're not closed properly
        cleaned_text = re.sub(r'<think>.*', '', cleaned_text, flags=re.DOTALL)
        cleaned_text = cleaned_text.strip()
        
        print(f"After cleaning: {len(cleaned_text)} characters")
        print(f"Cleaned preview: {cleaned_text[:200]}")
        
        # If cleaning removed everything, return original
        if not cleaned_text:
            print("Warning: Cleaning removed all text, returning original")
            # Escape HTML tags so they display
            return generated_text.replace('<', '&lt;').replace('>', '&gt;')
        
        return cleaned_text
            
    except Exception as e:
        print(f"chat.completions.create failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback to legacy chat_completion method
        try:
            print(f"Trying legacy chat_completion with model: {GEN_MODEL}")
            client = InferenceClient(token=HF_TOKEN)
            
            messages = [
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": prompt}
            ]
            
            response = client.chat_completion(
                messages=messages,
                model=GEN_MODEL,
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
                
        except Exception as e2:
            print(f"Legacy chat_completion also failed: {type(e2).__name__}: {str(e2)}")
            
            return f"""Unable to generate essay using Hugging Face API.

Errors:
- chat.completions.create: {type(e).__name__}: {str(e)}
- chat_completion: {type(e2).__name__}: {str(e2)}

The model '{GEN_MODEL}' may not be compatible with the Inference API."""


# text_generation and chat_completion version
'''
def generate_essay(prompt: str) -> str:
    """Generate essay using Hugging Face InferenceClient."""
    try:
        client = InferenceClient(model=GEN_MODEL, token=HF_TOKEN)
        
        # First try text_generation (for base/completion models)
        print(f"Attempting text_generation with model: {GEN_MODEL}")
        response = client.text_generation(
            prompt=prompt,
            max_new_tokens=2000,
            temperature=0.7,
            return_full_text=False
        )
        
        print(f"Text generation successful!")
        print("Generated text:", response)
        return response
            
    except Exception as e:
        print(f"Text generation failed: {type(e).__name__}: {str(e)}")
        
        # Fallback to chat_completion for instruction-tuned models
        try:
            print(f"Trying chat_completion with model: {GEN_MODEL}")
            client = InferenceClient(model=GEN_MODEL, token=HF_TOKEN)
            
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = client.chat_completion(
                messages=messages,
                max_tokens=2000,
                temperature=0.7
            )
            
            # Extract the generated text
            if hasattr(response, 'choices') and len(response.choices) > 0:
                print(f"Chat completion successful!")
                print("Completed chat:", response)
                return response.choices[0].message.content
            else:
                print("Completed chat:", response)
                return str(response)
                
        except Exception as e2:
            print(f"Chat completion also failed: {type(e2).__name__}: {str(e2)}")
            import traceback
            traceback.print_exc()
            
            return f"""Unable to generate essay using Hugging Face API.

Tried both text_generation and chat_completion.

Errors:
- Text generation: {type(e).__name__}: {str(e)}
- Chat completion: {type(e2).__name__}: {str(e2)}

The model '{GEN_MODEL}' may not support either method with the Inference API."""
'''


# requests version, was originally text_generation but changed to chat_completion

'''
import requests
import time

def generate_essay(prompt: str) -> str:
    """Generate essay using InferenceClient with configured model."""
    try:
        print(f"Attempting to use model: {GEN_MODEL}")
        client = InferenceClient(model=GEN_MODEL, token=HF_TOKEN)
        
        response = client.chat_completion(
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
            response = fallback_client.chat_completion(
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