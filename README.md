# 🧠 Academic Paper Generator with RAG

A **Retrieval-Augmented Generation (RAG)** application that transforms personal notes and documents into structured academic papers using open-source language models via the Hugging Face Inference API.

## 📖 Overview

This project combines document parsing, vector similarity search, and large language model generation to create a practical tool for academic writing assistance. Upload your research notes in various formats, ask a question, and receive a cohesive academic paper with proper source attribution.

## ✨ Key Features

- **Multi-format document ingestion**: PDF, DOCX, DOC, and TXT
- **Semantic search**: FAISS-powered vector similarity matching
- **Context-aware generation**: RAG pipeline ensures responses grounded in your documents
- **Source attribution**: Tracks which document chunks informed the generated content
- **Export options**: Download papers in TXT, DOCX, PDF, or Markdown
- **Free & open**: Uses Hugging Face's free inference API

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | LangChain | Document processing and RAG orchestration |
| **Vector Database** | FAISS | Fast similarity search with cosine distance |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Text vectorization (384-dim) |
| **LLM** | HuggingFaceTB/SmolLM3-3B | Academic text generation |
| **Document Parsing** | PyMuPDF, python-docx, textract | Multi-format file extraction |
| **Interface** | Gradio | Interactive web UI |
| **Language** | Python 3.12+ | Core implementation |

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/academic-paper-rag.git
cd academic-paper-rag
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Access
Create a `.env` file in the project root:
```bash
HF_API_TOKEN=your_huggingface_token_here
```

Get a free token at [Hugging Face](https://huggingface.co/settings/tokens).

### 4. Run the Application
```bash
python gradio_app.py
```

The interface will launch at `http://localhost:7860`

## 📝 Usage

1. **Upload Documents**: Click "📂 Upload your notes" and select PDF, DOCX, DOC, or TXT files
2. **Process**: Click "Embed & Process" to index your documents (creates vector embeddings)
3. **Query**: Enter your research question or essay prompt
4. **Generate**: Click "🧩 Generate Paper" to produce an academic paper
5. **Export**: Choose your format (TXT, DOCX, PDF, MD) and download

## 🏗️ Architecture

```
┌─────────────────┐
│  User Documents │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│ Text Extraction │─────>│  Chunking    │
│  (PyMuPDF, etc) │      │ (500 chars)  │
└─────────────────┘      └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │  Embedding   │
                         │  (MiniLM-L6) │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │ FAISS Index  │
                         └──────┬───────┘
                                │
         ┌──────────────────────┘
         │
    [User Query]
         │
         ▼
  ┌──────────────┐       ┌───────────────┐
  │ Query Embed  │──────>│ Similarity    │
  │              │       │ Search (top-k)│
  └──────────────┘       └───────┬───────┘
                                 │
                                 ▼
                          ┌──────────────┐
                          │ Context +    │
                          │ Query → LLM  │
                          └──────┬───────┘
                                 │
                                 ▼
                          ┌──────────────┐
                          │ Generated    │
                          │ Paper        │
                          └──────────────┘
```

## 🌐 Deployment

### Hugging Face Spaces

1. Create a new [Hugging Face Space](https://huggingface.co/spaces)
2. Select **Gradio** as the SDK
3. Upload all project files
4. Rename `gradio_app.py` → `app.py`
5. Add your token in **Settings → Repository Secrets**:
   - Key: `HF_API_TOKEN`
   - Value: `your_token_here`
6. The space will automatically deploy

### Local Deployment with Docker

```bash
docker build -t academic-paper-rag .
docker run -p 7860:7860 -e HF_API_TOKEN=your_token academic-paper-rag
```

## 🔧 Configuration

Edit `rag_chatbot.py` to customize:

```python
# Change embedding model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Change generation model
GEN_MODEL = "HuggingFaceTB/SmolLM3-3B"

# Adjust chunking parameters
chunk_size = 500      # Characters per chunk
chunk_overlap = 100   # Overlap between chunks

# Modify retrieval
k = 6  # Number of relevant chunks to retrieve
```

## 📦 Project Structure

```
academic-paper-rag/
├── gradio_app.py          # Gradio interface & export logic
├── rag_chatbot.py         # Core RAG pipeline
├── requirements.txt       # Python dependencies
├── .env                   # API tokens (gitignored)
└── README.md             # This file
```

## 🧪 Technical Details

### Vector Search
- **Algorithm**: FAISS IndexFlatIP (inner product)
- **Normalization**: L2 normalization for cosine similarity
- **Dimensionality**: 384 (from all-MiniLM-L6-v2)

### Text Processing
- **Chunking**: RecursiveCharacterTextSplitter with overlap
- **Separators**: Prioritizes paragraph → sentence → word boundaries
- **Encoding**: UTF-8 with fallback error handling

### Generation
- **Method**: `chat.completions.create()` (OpenAI-compatible API)
- **Post-processing**: Removes `<think>` tags from reasoning models
- **Fallback**: Legacy `chat_completion()` method if needed

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Support for additional file formats (EPUB, HTML)
- Advanced chunking strategies (semantic splitting)
- Multiple LLM provider support (OpenAI, Anthropic)
- Citation formatting (APA, MLA, Chicago)
- Multi-language support

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co) for free inference API
- [LangChain](https://langchain.com) for document processing utilities
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [Gradio](https://gradio.app) for rapid UI prototyping