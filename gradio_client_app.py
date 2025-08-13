import gradio as gr
import requests
from pathlib import Path

API_URL = "http://localhost:8000"


def submit_job(pdf_file, pages, use_offline, rotation, min_blocks, ocr_lang):
    if pdf_file is None:
        return "Please upload a PDF first.", None, 0.0, None, None, None, None
    files = {"pdf": (pdf_file.name, open(
        pdf_file.name, 'rb'), 'application/pdf')}
    data = {
        'pages': pages or '',
        'use_offline': str(use_offline).lower(),
        'rotation': int(rotation or 0),
        'min_blocks': int(min_blocks or 1),
        'ocr_lang': ocr_lang or 'jpn'
    }
    r = requests.post(f"{API_URL}/jobs", files=files, data=data)
    if r.status_code != 200:
        return f"Could not submit job: {r.text}", None, 0.0, None, None, None, None
    job_id = r.json()['job_id']
    return f"Job submitted. Your Job ID is below. Keep it to check progress.", job_id, 0.0, None, None, None, None


def poll_status(job_id):
    if not job_id:
        return "Enter or submit a Job ID to see progress.", 0.0, None, None, None, None, None
    try:
        r = requests.get(f"{API_URL}/jobs/{job_id}")
        if r.status_code != 200:
            return f"Status error: {r.text}", 0.0, None, None, None, None, None
        js = r.json()
        status = js['status']
        msg_lines = []
        if status == 'queued':
            pos = js.get('queue_position')
            total_q = js.get('queue_length')
            if pos and total_q:
                msg_lines.append(
                    f"Waiting... you are #{pos} of {total_q} in line")
            else:
                msg_lines.append("Waiting in line…")
        elif status == 'processing':
            done = js.get('pages_done') or 0
            total = js.get('pages_total') or 0
            if total:
                msg_lines.append(f"Processing page {done+1} of {total}")
            else:
                msg_lines.append("Processing…")
        elif status == 'completed':
            msg_lines.append("Finished! You can download your files below.")
        elif status == 'failed':
            msg_lines.append("Job failed. See details below.")
        # Add any backend message
        if js.get('message') and status != 'queued':
            msg_lines.append(js['message'])
        merged = None
        zf = None
        if status == 'completed':
            merged = js.get('merged_pdf')
            zf = js.get('zip_file')
        prog = js.get('progress') or 0.0
        return "\n".join(msg_lines), prog, merged, zf, js.get('output_dir'), js.get('pages_done'), js.get('pages_total')
    except Exception as e:
        return f"Error: {e}", 0.0, None, None, None, None, None


def download_file(url, dest_name):
    if not url:
        return None
    r = requests.get(url)
    if r.status_code != 200:
        return None
    out_path = Path('output')/dest_name
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_bytes(r.content)
    return str(out_path)


def fetch_jobs():
    try:
        r = requests.get(f"{API_URL}/jobs")
        if r.status_code != 200:
            return []
        data = r.json().get('jobs', [])
        table = []
        for j in data:
            table.append([
                j.get('job_id'),
                j.get('status'),
                j.get('queue_position') if j.get('status') == 'queued' else '',
                j.get('queue_length') if j.get('status') == 'queued' else '',
                j.get('pages_done') if j.get('pages_done') is not None else '',
                j.get('pages_total') if j.get(
                    'pages_total') is not None else '',
                f"{int((j.get('progress') or 0)*100)}%" if j.get('progress') is not None else ''
            ])
        return table
    except Exception:
        return []


with gr.Blocks(title="PDF Translator (Queued)") as app:
    gr.Markdown('# PDF Translator (Queued)')
    gr.Markdown(
        "Upload a PDF and click Submit. Keep the Job ID shown. Use the Check Progress button to see updates.")
    with gr.Row():
        pdf_input = gr.File(label='PDF', file_types=['.pdf'])
        pages_input = gr.Textbox(label='Pages', placeholder='e.g. 1;3;5-8')
    with gr.Accordion('Advanced', open=False):
        with gr.Row():
            use_offline = gr.Checkbox(
                label='Use Offline Translation', value=True)
            rotation = gr.Number(label='Rotation Offset', value=0, precision=0)
            min_blocks = gr.Number(
                label='Min Blocks Threshold', value=1, precision=0)
            ocr_lang = gr.Textbox(label='OCR Language', value='jpn')
    submit_btn = gr.Button('Submit Job', variant='primary')
    poll_btn = gr.Button('Check Progress / Refresh')
    gr.Markdown('### All Jobs (refreshes automatically)')
    jobs_table = gr.Dataframe(headers=["Job ID", "Status", "Pos", "Queue Size",
                              "Pages Done", "Pages Total", "Progress"], interactive=False, row_count=0)

    job_id_box = gr.Textbox(
        label='Job ID (auto-filled after submit)', interactive=True)
    progress_bar = gr.Slider(label='Overall Progress',
                             minimum=0, maximum=1, value=0, interactive=False)
    status_box = gr.Textbox(label='Status & Messages', lines=4)
    simple_progress = gr.Textbox(label='Page Progress', lines=1)
    merged_file = gr.File(label='Merged PDF (Download when ready)')
    zip_file = gr.File(label='All Pages ZIP (Download when ready)')
    output_dir_box = gr.Textbox(
        label='Output Folder (Server)', interactive=False)

    submit_btn.click(
        fn=submit_job,
        inputs=[pdf_input, pages_input, use_offline,
                rotation, min_blocks, ocr_lang],
        outputs=[status_box, job_id_box, progress_bar,
                 merged_file, zip_file, output_dir_box, simple_progress]
    )

    def refresh(job_id):
        status_text, prog, merged_path, zip_path, out_dir, done, total = poll_status(
            job_id)
        merged_display = None
        zip_display = None
        page_prog_text = ''
        if total:
            pct = int((prog or 0)*100)
            page_prog_text = f"{pct}% of {total} page(s)" if pct < 100 else f"{total} pages done"
        if merged_path and merged_path.startswith('output'):
            merged_display = merged_path
        elif merged_path:
            merged_display = download_file(
                f"{API_URL}/jobs/{job_id}/download/merged", f"{job_id}_merged.pdf")
        if zip_path and zip_path.startswith('output'):
            zip_display = zip_path
        elif zip_path:
            zip_display = download_file(
                f"{API_URL}/jobs/{job_id}/download/zip", f"{job_id}_pages.zip")
        return status_text, prog, merged_display, zip_display, out_dir, page_prog_text

    # Auto refresh job list and current job progress using Gradio Timer API
    def refresh_jobs():
        return fetch_jobs()
    jobs_auto_timer = gr.Timer(5.0)
    jobs_auto_timer.tick(fn=refresh_jobs, outputs=jobs_table)

    # Also periodically refresh the selected job's progress
    progress_auto_timer = gr.Timer(4.0)
    progress_auto_timer.tick(fn=refresh, inputs=job_id_box, outputs=[status_box, progress_bar, merged_file,
                                                                     zip_file, output_dir_box, simple_progress])

    poll_btn.click(
        fn=refresh,
        inputs=[job_id_box],
        outputs=[status_box, progress_bar, merged_file,
                 zip_file, output_dir_box, simple_progress]
    )

    # Manual refresh of job list when clicking poll as well
    poll_btn.click(fn=fetch_jobs, inputs=[], outputs=[jobs_table])

if __name__ == '__main__':
    app.launch(server_name="0.0.0.0", server_port=7861)
