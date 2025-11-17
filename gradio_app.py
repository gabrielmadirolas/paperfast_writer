# gradio_app.py
import gradio as gr
import tempfile
import os
from datetime import datetime
from rag_chatbot02 import ingest_documents, retrieve_relevant, build_prompt, generate_essay, add_documents_to_store

# For file export
from docx import Document
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from docx.enum.text import WD_ALIGN_PARAGRAPH

store = None
stored_files = []  # List of (filename, filepath) tuples
last_essay = None
last_refs = None

def save_uploaded_files(files):
    """Save uploaded files and return their paths."""
    if not files:
        return []
    
    paths = []
    for f in files:
        if isinstance(f, str):
            # Make a permanent copy
            suffix = os.path.splitext(f)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            with open(f, 'rb') as src:
                tmp.write(src.read())
            tmp.close()
            paths.append((os.path.basename(f), tmp.name))
        else:
            suffix = "." + f.name.split(".")[-1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(f.read())
            tmp.close()
            paths.append((f.name, tmp.name))
    
    return paths

def process_files(files):
    """Add uploaded files to the vector store."""
    global store, stored_files
    
    if not files:
        return format_file_list(stored_files), "❗ Please upload at least one file.", None  # Added None for clearing
    
    # Save uploaded files
    new_files = save_uploaded_files(files)
    file_paths = [path for _, path in new_files]
    
    if store is None:
        # First upload - create new store
        store, _ = ingest_documents(file_paths)
        stored_files.extend(new_files)
        message = f"✅ Indexed {len(store.metadatas)} chunks from {len(new_files)} file(s)."
    else:
        # Add to existing store
        num_chunks, _ = add_documents_to_store(store, file_paths)
        stored_files.extend(new_files)
        message = f"✅ Added {num_chunks} chunks from {len(new_files)} file(s). Total: {len(store.metadatas)} chunks."
    
    return format_file_list(stored_files), message, None  # Return None to clear the File component

def format_file_list(files):
    """Format file list for display."""
    if not files:
        return "No files in store"
    return "\n".join([f"{i}. {name}" for i, (name, _) in enumerate(files)])

def remove_file(index_str):
    """Remove a specific file from the store."""
    global store, stored_files
    
    if not index_str or not stored_files:
        return format_file_list(stored_files), "❗ Invalid index or no files to remove."
    
    try:
        idx = int(index_str)
        if idx < 0 or idx >= len(stored_files):
            return format_file_list(stored_files), "❗ Index out of range."
        
        # Remove file
        name, path = stored_files.pop(idx)
        try:
            os.remove(path)
        except:
            pass
        
        # Rebuild store without this file
        if stored_files:
            remaining_paths = [path for _, path in stored_files]
            store, _ = ingest_documents(remaining_paths)
            message = f"✅ Removed '{name}'. Store rebuilt with {len(stored_files)} file(s)."
        else:
            store = None
            message = f"✅ Removed '{name}'. Store is now empty."
        
        return format_file_list(stored_files), message
        
    except ValueError:
        return format_file_list(stored_files), "❗ Please enter a valid number."

def clear_all():
    """Clear all files from the store."""
    global store, stored_files
    
    # Clean up temp files
    for _, path in stored_files:
        try:
            os.remove(path)
        except:
            pass
    
    stored_files = []
    store = None
    
    return "No files in store", "🗑️ All files cleared from store."

def generate_paper(query):
    global store, last_essay, last_refs
    
    if store is None:
        return "❗ Please upload and process your notes first.", gr.update(visible=False)
    
    try:
        context, hits = retrieve_relevant(store, query, k=6)
        prompt = build_prompt(context, query)
        essay = generate_essay(prompt)
        
        if not essay or essay.strip() == "":
            return "❗ Generated essay is empty. Please try again.", gr.update(visible=False)
        
        refs = "\n".join([f"- {h[1]['source']} (chunk {h[1]['chunk']}, score={h[0]:.3f})" for h in hits])
        
        # Store for export
        last_essay = essay
        last_refs = refs
        
        result = f"### 📄 Generated Academic Paper\n\n{essay}\n\n---\n\n### 🔎 Sources Used\n{refs}"
        
        return result, gr.update(visible=True)
        
    except Exception as e:
        print(f"Error in generate_paper: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error generating paper: {str(e)}", gr.update(visible=False)

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
        
        # Essay content
        lines = last_essay.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('###'):
                doc.add_heading(line.replace('###', '').strip(), level=3)
            elif line.startswith('##'):
                doc.add_heading(line.replace('##', '').strip(), level=2)
            elif line.startswith('#'):
                doc.add_heading(line.replace('#', '').strip(), level=1)
            elif line.startswith('**') and line.endswith('**'):
                p = doc.add_paragraph()
                run = p.add_run(line.strip('*'))
                run.bold = True
            else:
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
        pdf.set_font("Helvetica", 'B', 16)
        pdf.cell(0, 10, 'Generated Academic Paper', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(10)
        
        # Essay content
        pdf.set_font("Helvetica", '', 11)
        clean_essay = last_essay.replace('**', '').replace('###', '').replace('##', '')
        
        for line in clean_essay.split('\n'):
            if line.strip():
                pdf.multi_cell(0, 6, line.strip())
                pdf.ln(2)
        
        # Sources section
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', 14)
        pdf.cell(0, 10, 'Sources Used', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)
        
        pdf.set_font("Helvetica", '', 10)
        for ref in last_refs.split('\n'):
            if ref.strip():
                pdf.multi_cell(0, 5, ref.strip())
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
    gr.Markdown("Upload your personal notes (PDF/DOC/DOCX/ODT/TXT), ask a question, and generate an academic paper draft.")

    with gr.Row():
        with gr.Column():
            file_uploader = gr.File(file_count="multiple", label="📂 Upload files to add to store")
            add_btn = gr.Button("📥 Add to Store", variant="primary")
            
            gr.Markdown("### Files in Store")
            stored_display = gr.Textbox(value="No files in store", label="Stored Files", interactive=False, lines=8)
            
            with gr.Row():
                remove_index = gr.Textbox(placeholder="Enter index (e.g., 0, 1, 2...)", label="Remove file by index", scale=3)
                remove_btn = gr.Button("❌ Remove", scale=1)
            
            clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
            
            status_msg = gr.Markdown()

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
    
    # Wire up events
    add_btn.click(
    process_files, 
    inputs=[file_uploader], 
    outputs=[stored_display, status_msg, file_uploader]  # Add file_uploader as output
)
    remove_btn.click(remove_file, inputs=[remove_index], outputs=[stored_display, status_msg])
    clear_btn.click(clear_all, inputs=[], outputs=[stored_display, status_msg])
    gen_btn.click(generate_paper, inputs=[query_input], outputs=[output_md, export_section])
    export_btn.click(export_paper, inputs=[format_dropdown], outputs=[download_file])

    gr.Markdown("---")
    gr.Markdown("⚙️ Powered by Hugging Face Inference API · Built with LangChain, FAISS & Gradio.")

if __name__ == "__main__":
    app.launch()