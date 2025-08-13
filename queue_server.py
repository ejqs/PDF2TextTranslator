import asyncio
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import threading
import time

# Reuse existing translation logic
from functions.pymupdf_test import translate_pdf
import pymupdf

app = FastAPI(title="PDF Translator Queue API")

# In-memory job store
_jobs: Dict[str, Dict[str, Any]] = {}
_queue: asyncio.Queue[str] = asyncio.Queue()
_worker_started = False
_lock = threading.Lock()

# Configuration
UPLOAD_ROOT = Path("queued_inputs")
UPLOAD_ROOT.mkdir(exist_ok=True)

OUTPUT_ROOT = Path("output")


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    queue_position: Optional[int] = None  # 1-based position if queued
    queue_length: Optional[int] = None    # total queued including this
    progress: Optional[float] = None      # 0.0 - 1.0 over pages
    message: Optional[str] = None
    output_dir: Optional[str] = None
    merged_pdf: Optional[str] = None
    zip_file: Optional[str] = None
    pages_done: Optional[int] = None
    pages_total: Optional[int] = None


class JobListResponse(BaseModel):
    jobs: List[JobStatusResponse]


async def _ensure_worker():
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    asyncio.create_task(_worker_loop())


async def _worker_loop():
    while True:
        job_id = await _queue.get()
        job = _jobs.get(job_id)
        if not job:
            _queue.task_done()
            continue
        job['status'] = 'processing'
        job['started_at'] = time.time()
        try:
            # Run translate_pdf in thread to avoid blocking event loop
            def run():
                def cb(done, total, msg):
                    # done can be fractional now
                    job['pages_done'] = int(done) if done is not None else 0
                    job['pages_total'] = total
                    if total:
                        job['progress'] = min(max(done / total, 0.0), 1.0)
                    job['message'] = msg
                result = translate_pdf(
                    pdf_path=job['input_pdf'],
                    pages=job['params']['pages'],
                    use_offline=job['params']['use_offline'],
                    rotation_offset=job['params']['rotation'],
                    min_blocks=job['params']['min_blocks'],
                    debug=False,
                    ocr_lang=job['params']['ocr_lang'],
                    output_base=OUTPUT_ROOT,
                    progress_cb=cb
                )
                job['result'] = result
                job['status'] = 'completed'
                job['progress'] = 1.0
                job['message'] = 'All pages processed'
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, run)
        except Exception as e:
            job['status'] = 'failed'
            job['message'] = str(e)
        finally:
            job['finished_at'] = time.time()
            _queue.task_done()


@app.post('/jobs', response_model=JobCreateResponse)
async def create_job(
    pdf: UploadFile = File(...),
    pages: str = Form(""),
    use_offline: bool = Form(True),
    rotation: int = Form(0),
    min_blocks: int = Form(1),
    ocr_lang: str = Form("jpn")
):
    await _ensure_worker()
    job_id = str(uuid.uuid4())
    dest = UPLOAD_ROOT / f"{job_id}.pdf"
    with dest.open('wb') as f:
        shutil.copyfileobj(pdf.file, f)
    # Determine total pages early for better UX
    try:
        doc = pymupdf.open(dest)
        total_pages = doc.page_count
        doc.close()
    except Exception:
        total_pages = None

    _jobs[job_id] = {
        'status': 'queued',
        'input_pdf': str(dest),
        'created_at': time.time(),
        'progress': 0.0,
        'pages_done': 0,
        'pages_total': total_pages,
        'params': {
            'pages': pages,
            'use_offline': use_offline,
            'rotation': rotation,
            'min_blocks': min_blocks,
            'ocr_lang': ocr_lang
        }
    }
    await _queue.put(job_id)
    return JobCreateResponse(job_id=job_id, status='queued')


@app.get('/jobs/{job_id}', response_model=JobStatusResponse)
async def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    position = None
    q_len = None
    if job['status'] == 'queued':
        queued_ids = [jid for jid, j in _jobs.items() if j['status']
                      == 'queued']
        q_len = len(queued_ids)
        if job_id in queued_ids:
            position = queued_ids.index(job_id) + 1
    result = job.get('result') or {}
    return JobStatusResponse(
        job_id=job_id,
        status=job['status'],
        queue_position=position,
        queue_length=q_len,
        progress=job.get('progress'),
        message=job.get('message'),
        output_dir=result.get('output_dir'),
        merged_pdf=result.get('merged_pdf_path'),
        zip_file=result.get('zip_path'),
        pages_done=job.get('pages_done'),
        pages_total=job.get('pages_total')
    )


@app.get('/jobs', response_model=JobListResponse)
async def list_jobs():
    items = []
    for jid in sorted(_jobs.keys()):
        job = _jobs[jid]
        result = job.get('result') or {}
        # Determine queue position details for queued jobs only
        queue_position = None
        queue_length = None
        if job['status'] == 'queued':
            queued_ids = [qid for qid,
                          qj in _jobs.items() if qj['status'] == 'queued']
            queue_length = len(queued_ids)
            if jid in queued_ids:
                queue_position = queued_ids.index(jid) + 1
        items.append(JobStatusResponse(
            job_id=jid,
            status=job['status'],
            queue_position=queue_position,
            queue_length=queue_length,
            progress=job.get('progress'),
            message=job.get('message'),
            output_dir=result.get('output_dir'),
            merged_pdf=result.get('merged_pdf_path'),
            zip_file=result.get('zip_path'),
            pages_done=job.get('pages_done'),
            pages_total=job.get('pages_total')
        ))
    return JobListResponse(jobs=items)


@app.get('/jobs/{job_id}/download/merged')
async def download_merged(job_id: str):
    job = _jobs.get(job_id)
    if not job or job.get('status') != 'completed':
        raise HTTPException(status_code=404, detail='Not ready')
    merged = job['result'].get('merged_pdf_path')
    if not merged or not Path(merged).exists():
        raise HTTPException(status_code=404, detail='Merged file missing')
    return FileResponse(merged, media_type='application/pdf', filename=f'{job_id}_merged.pdf')


@app.get('/jobs/{job_id}/download/zip')
async def download_zip(job_id: str):
    job = _jobs.get(job_id)
    if not job or job.get('status') != 'completed':
        raise HTTPException(status_code=404, detail='Not ready')
    zf = job['result'].get('zip_path')
    if not zf or not Path(zf).exists():
        raise HTTPException(status_code=404, detail='Zip file missing')
    return FileResponse(zf, media_type='application/zip', filename=f'{job_id}_pages.zip')

# Optional: root health


@app.get('/')
async def root():
    return {"service": "pdf-translator-queue", "jobs": len(_jobs)}

# To run: uv run uvicorn queue_server:app --reload --port 8000 --host 0.0.0.0
