# gradio_app.py
import gradio as gr
import tempfile
from rag_chatbot import ingest_documents, retrieve_relevant, build_prompt, generate_essay

store = None

def process_files(files):
    """Upload and embed user files."""
    global store
    if not files:
        return "❗ Please upload at least one file."
    paths = []
    for f in files:
        # In newer Gradio versions, f is already a file path (string)
        # not a file object
        if isinstance(f, str):
            # f is already a path to the uploaded file
            paths.append(f)
        else:
            # Fallback for older Gradio versions or file objects
            suffix = "." + f.name.split(".")[-1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(f.read())
            tmp.close()
            paths.append(tmp.name)

    store, _ = ingest_documents(paths)
    return f"✅ Indexed {len(store.metadatas)} text chunks. You can now enter a query."

def generate_paper(query):
    global store
    if store is None:
        return "❗ Please upload and process your notes first."
    context, hits = retrieve_relevant(store, query, k=6)
    prompt = build_prompt(context, query)
    essay = generate_essay(prompt)
    refs = "\n".join([f"- {h[1]['source']} (chunk {h[1]['chunk']}, score={h[0]:.3f})" for h in hits])
    return f"### 📄 Generated Academic Paper\n{essay}\n\n---\n### 🔎 Sources Used\n{refs}"

with gr.Blocks(theme="soft") as app:
    gr.Markdown("## 🧠 Academic Paper Chatbot — RAG + Hugging Face API")
    gr.Markdown("Upload your personal notes (PDF/DOC/DOCX), ask a question, and generate an academic paper draft.")

    file_uploader = gr.File(file_count="multiple", label="📂 Upload your notes")
    upload_btn = gr.Button("Embed & Process")
    upload_output = gr.Markdown()
    upload_btn.click(process_files, inputs=[file_uploader], outputs=[upload_output])

    query_input = gr.Textbox(label="🎓 Your Question / Essay Prompt", lines=3, placeholder="e.g., Discuss the main theories and their methodological implications.")
    gen_btn = gr.Button("🧩 Generate Paper")
    output_md = gr.Markdown()
    gen_btn.click(generate_paper, inputs=[query_input], outputs=[output_md])

    gr.Markdown("---")
    gr.Markdown("⚙️ Powered by Hugging Face Free API · Built with LangChain, FAISS & Gradio.")

if __name__ == "__main__":
    app.launch()
