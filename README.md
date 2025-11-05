# 🧠 Academic Paper Chatbot (RAG + Hugging Face)

### 🎯 Purpose
This project demonstrates my ability to build a **Retrieval-Augmented Generation (RAG)** application — a chatbot that reads your personal notes and produces structured academic papers using open-access **Hugging Face** models.

Recruiters and reviewers can use this repository to evaluate:
- My practical understanding of **LLM integration**, **LangChain**, and **RAG**
- My software engineering structure and clean code practices
- My use of **free and efficient** AI APIs for real-world deployment

### 💡 Motivation
I built this as a personal project to combine **information retrieval**, **document parsing**, and **academic text generation** — a valuable skillset for roles in **AI, NLP, or applied data science**.

### ⚙️ Tech Stack
| Component | Tool | Purpose |
|------------|------|----------|
| Framework | LangChain | RAG pipeline management |
| Vector Store | FAISS | Similarity search |
| LLMs | Hugging Face Inference API (Flan-T5, MiniLM) | Generation & Embedding |
| File Handling | pdfplumber, python-docx | Parsing user documents |
| UI | Gradio | Interactive web interface |
| Language | Python 3.9+ | Core logic |

### 🚀 How to Run
```bash
git clone https://github.com/yourusername/academic-paper-chatbot.git
cd academic-paper-chatbot
pip install -r requirements.txt
export HF_API_TOKEN="hf_your_token_here"
python gradio_app.py
````

### 📂 Upload Supported Files

* `.pdf`
* `.docx` / `.doc`

### 🌐 Deploy Online

You can host it easily on **Hugging Face Spaces**:

1. Create a new Space.
2. Upload these files.
3. Rename `gradio_app.py` to `app.py`.
4. Add your token in **Settings → Repository Secrets → HF_API_TOKEN**.

### 📄 License

MIT License © [Your Name]
