import os
from pathlib import Path
import gradio as gr
import shutil
import traceback

# Ensure functions package importable
from functions.pymupdf_test import translate_pdf


def _human_hint_pages(pages_str: str, total_pages: int | None = None) -> str:
    if not pages_str.strip():
        return "All pages selected" if total_pages else "All pages"
    return f"Pages: {pages_str}"


def run_translation(pdf_file, pages, use_offline, rotation, min_blocks, ocr_lang):
    try:
        if pdf_file is None:
            return "Please upload a PDF.", None, None, None

        # Gradio gives a tempfile path for uploaded file
        input_path = pdf_file.name if hasattr(pdf_file, 'name') else pdf_file
        result = translate_pdf(
            pdf_path=input_path,
            pages=pages or "",
            use_offline=use_offline,
            rotation_offset=rotation or 0,
            min_blocks=min_blocks or None,
            debug=False,
            ocr_lang=ocr_lang or None,
            output_base=Path("output")
        )

        merged_path = result.get("merged_pdf_path")
        zip_path = result.get("zip_path")
        page_files = result.get("page_files", [])
        summary = [
            "Translation Complete!",
            f"Pages produced: {len(page_files)}",
            _human_hint_pages(pages),
            f"Mode: {'Offline' if use_offline else 'AI'}",
        ]
        return "\n".join(summary), merged_path, zip_path, page_files
    except Exception as e:
        traceback.print_exc()
        return f"Error: {e}", None, None, None


with gr.Blocks(title="PDF Translator") as demo:
    gr.Markdown(
        "# PDF Translator\nUpload a PDF, choose pages (e.g. `1;3;5-8`) and download the translated pages.")
    with gr.Row():
        pdf_input = gr.File(label="PDF", file_types=[
                            ".pdf"], file_count="single")
        pages_input = gr.Textbox(
            label="Pages (1-based; blank = all)", placeholder="e.g. 1;3;5-8")
    with gr.Accordion("Advanced Options", open=False):
        with gr.Row():
            use_offline = gr.Checkbox(
                label="Use Offline Translation", value=True)
            rotation = gr.Number(
                label="Rotation Offset (deg)", value=0, precision=0)
            min_blocks = gr.Number(
                label="Min Blocks Threshold (OCR)", value=1, precision=0)
            ocr_lang = gr.Textbox(label="OCR Language",
                                  value="jpn", max_lines=1)
    run_btn = gr.Button("Translate PDF")
    status = gr.Textbox(label="Status / Logs", lines=6)
    merged_file = gr.File(label="Merged Translated PDF")
    zip_file = gr.File(label="ZIP of Individual Pages")
    page_gallery = gr.Files(label="Individual Page PDFs")

    run_btn.click(
        fn=run_translation,
        inputs=[pdf_input, pages_input, use_offline,
                rotation, min_blocks, ocr_lang],
        outputs=[status, merged_file, zip_file, page_gallery]
    )

if __name__ == "__main__":
    # Bind to all interfaces if running on LAN; fallback to localhost
    demo.launch(server_name=os.getenv("HOST", "0.0.0.0"), server_port=int(
        os.getenv("PORT", "7860")), max_file_size="200mb")
