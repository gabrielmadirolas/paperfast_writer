# gradio_app.py
import gradio as gr
import tempfile
import os
from datetime import datetime
from rag_chatbot import ingest_documents, retrieve_relevant, build_prompt, generate_essay

# For file export
from docx import Document
from fpdf import FPDF
import markdown
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

store = None
last_essay = None
last_refs = None

def process_files(files):
    """Upload and embed user files."""
    global store
    if not files:
        return "❗ Please upload at least one file."
    paths = []
    for f in files:
        if isinstance(f, str):
            paths.append(f)
        else:
            suffix = "." + f.name.split(".")[-1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(f.read())
            tmp.close()
            paths.append(tmp.name)

    store, _ = ingest_documents(paths)
    return f"✅ Indexed {len(store.metadatas)} text chunks. You can now enter a query."

def generate_paper(query):
    global store, last_essay, last_refs
    if store is None:
        return "❗ Please upload and process your notes first.", None
    
    try:
        context, hits = retrieve_relevant(store, query, k=6)
        prompt = build_prompt(context, query)
        essay = generate_essay(prompt)
        
        if not essay or essay.strip() == "":
            return "❗ Generated essay is empty. Please try again with a different model or query.", None
        
        refs = "\n".join([f"- {h[1]['source']} (chunk {h[1]['chunk']}, score={h[0]:.3f})" for h in hits])
        
        # Store for export
        last_essay = essay
        last_refs = refs
        
        result = f"### 📄 Generated Academic Paper\n\n{essay}\n\n---\n\n### 🔎 Sources Used\n{refs}"
        
        return result, gr.update(visible=True)  # Show export section
        
    except Exception as e:
        print(f"Error in generate_paper: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error generating paper: {str(e)}", None

def export_paper(format_choice):
    """Export the generated paper in the selected format."""
    global last_essay, last_refs
    
    if not last_essay:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format_choice == "TXT":
        filename = f"paper_{timestamp}.txt"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("GENERATED ACADEMIC PAPER\n")
            f.write("=" * 50 + "\n\n")
            f.write(last_essay)
            f.write("\n\n" + "=" * 50 + "\n")
            f.write("SOURCES USED\n")
            f.write("=" * 50 + "\n")
            f.write(last_refs)
        return filepath
    
    elif format_choice == "DOCX":
        filename = f"paper_{timestamp}.docx"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        
        doc = Document()
        
        # Title
        title = doc.add_heading('Generated Academic Paper', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Essay content - parse markdown-style formatting
        lines = last_essay.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Handle headers
            if line.startswith('###'):
                doc.add_heading(line.replace('###', '').strip(), level=3)
            elif line.startswith('##'):
                doc.add_heading(line.replace('##', '').strip(), level=2)
            elif line.startswith('#'):
                doc.add_heading(line.replace('#', '').strip(), level=1)
            # Handle bold text
            elif line.startswith('**') and line.endswith('**'):
                p = doc.add_paragraph()
                run = p.add_run(line.strip('*'))
                run.bold = True
            else:
                # Regular paragraph
                doc.add_paragraph(line)
        
        # Sources section
        doc.add_page_break()
        doc.add_heading('Sources Used', 1)
        for ref in last_refs.split('\n'):
            if ref.strip():
                doc.add_paragraph(ref.strip('- '), style='List Bullet')
        
        doc.save(filepath)
        return filepath
    
    elif format_choice == "PDF":
        filename = f"paper_{timestamp}.pdf"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, 'Generated Academic Paper', 0, 1, 'C')
        pdf.ln(10)
        
        # Essay content
        pdf.set_font("Arial", '', 11)
        
        # Clean and process text for PDF
        clean_essay = last_essay.replace('**', '').replace('###', '').replace('##', '')
        
        # Split into lines and add to PDF
        for line in clean_essay.split('\n'):
            if line.strip():
                # Handle encoding issues
                try:
                    pdf.multi_cell(0, 6, line.strip())
                except UnicodeEncodeError:
                    # Fallback for special characters
                    clean_line = line.encode('latin-1', 'replace').decode('latin-1')
                    pdf.multi_cell(0, 6, clean_line.strip())
                pdf.ln(2)
        
        # Sources section
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, 'Sources Used', 0, 1)
        pdf.ln(5)
        
        pdf.set_font("Arial", '', 10)
        for ref in last_refs.split('\n'):
            if ref.strip():
                try:
                    pdf.multi_cell(0, 5, ref.strip())
                except UnicodeEncodeError:
                    clean_ref = ref.encode('latin-1', 'replace').decode('latin-1')
                    pdf.multi_cell(0, 5, clean_ref.strip())
                pdf.ln(2)
        
        pdf.output(filepath)
        return filepath
    
    elif format_choice == "MD":
        filename = f"paper_{timestamp}.md"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Generated Academic Paper\n\n")
            f.write(last_essay)
            f.write("\n\n---\n\n")
            f.write("## Sources Used\n\n")
            f.write(last_refs)
        return filepath

with gr.Blocks(theme="soft") as app:
    gr.Markdown("## 🧠 Academic Paper Chatbot — RAG + Hugging Face API")
    gr.Markdown("Upload your personal notes (PDF/DOC/DOCX), ask a question, and generate an academic paper draft.")

    file_uploader = gr.File(file_count="multiple", label="📂 Upload your notes")
    upload_btn = gr.Button("Embed & Process")
    upload_output = gr.Markdown()
    upload_btn.click(process_files, inputs=[file_uploader], outputs=[upload_output])

    query_input = gr.Textbox(label="🎯 Your Question / Essay Prompt", lines=3, placeholder="e.g., Discuss the main theories and their methodological implications.")
    gen_btn = gr.Button("🧩 Generate Paper")
    output_md = gr.Markdown()
    
    # Export section (initially hidden)
    with gr.Row(visible=False) as export_section:
        format_dropdown = gr.Dropdown(
            choices=["TXT", "DOCX", "PDF", "MD"],
            value="DOCX",
            label="📥 Export Format"
        )
        export_btn = gr.Button("💾 Download Paper")
    
    download_file = gr.File(label="📄 Your Paper", visible=True)
    
    gen_btn.click(generate_paper, inputs=[query_input], outputs=[output_md, export_section])
    export_btn.click(export_paper, inputs=[format_dropdown], outputs=[download_file])

    gr.Markdown("---")
    gr.Markdown("⚙️ Powered by Hugging Face Free API · Built with LangChain, FAISS & Gradio.")

if __name__ == "__main__":
    app.launch()