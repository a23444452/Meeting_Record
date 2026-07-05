"""FastAPI 伺服器：REST API + SSE 進度串流 + 靜態前端。"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import httpx
import markdown as md_lib
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ..diarize import DiarizeError, assign_speakers, diarize
from ..extract import SUPPORTED_EXTENSIONS, ExtractError, extract_audio
from ..output import OutputError, load_transcript, write_summary, write_transcript
from ..summarize import LLMClient, SummarizeError, summarize
from ..templates import (
    SummaryTemplate,
    TemplateError,
    delete_user_template,
    list_all_templates,
    resolve_template,
    save_user_template,
)
from ..transcribe import TranscribeError, transcribe
from .jobs import JobBusyError, JobManager, JobState
from .settings import GuiSettings, load_settings, save_settings

KNOWN_ERRORS = (
    ExtractError,
    TranscribeError,
    DiarizeError,
    SummarizeError,
    TemplateError,
    OutputError,
)

STATIC_DIR = Path(__file__).parent / "static"
TERMINAL_STATES = {JobState.IDLE, JobState.DONE, JobState.ERROR}


class ProcessOptions(BaseModel):
    template: str = "standard_meeting"
    language: str = "zh"
    whisper_model: str = "large-v3"
    diarize: bool = False
    num_speakers: int | None = None
    llm_model: str = "qwen3:8b"
    llm_url: str = "http://localhost:11434"
    provider: str = "ollama"
    skip_summary: bool = False


class ResummarizeRequest(BaseModel):
    template: str = "standard_meeting"
    llm_model: str = "qwen3:8b"
    llm_url: str = "http://localhost:11434"
    provider: str = "ollama"


class TemplatePayload(BaseModel):
    stem: str
    template: SummaryTemplate


def _render_markdown(text: str) -> str:
    return md_lib.markdown(text, extensions=["tables", "nl2br"])


def _unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for i in range(2, 100):
        candidate = base.with_name(f"{base.name}_{i}")
        if not candidate.exists():
            return candidate
    raise OutputError(f"同名輸出目錄過多：{base}")


def _make_llm_client(url: str, model: str, provider: str) -> LLMClient:
    if provider not in ("ollama", "openai"):
        raise SummarizeError("provider 只能是 ollama 或 openai")
    return LLMClient(
        base_url=url,
        model=model,
        provider=provider,  # type: ignore[arg-type]
        api_key=os.environ.get("MEETING_RECORD_API_KEY"),
    )


def _process_runner(upload_path: Path, tmpdir: str, opts: ProcessOptions, output_dir: Path):
    """完整 pipeline 的背景執行函式。"""

    def runner(job: JobManager) -> None:
        try:
            job.set_state(JobState.EXTRACTING)
            wav_path = Path(tmpdir) / "audio.wav"
            extract_audio(upload_path, wav_path)

            job.set_state(JobState.TRANSCRIBING)
            transcript = transcribe(
                wav_path,
                model_size=opts.whisper_model,
                language=opts.language,
                on_segment=lambda seg: job.emit(
                    "segment", {"start": seg.start, "text": seg.text}
                ),
            )

            if opts.diarize:
                job.set_state(JobState.DIARIZING)
                turns = diarize(
                    wav_path,
                    hf_token=os.environ.get("HF_TOKEN"),
                    num_speakers=opts.num_speakers,
                )
                transcript = transcript.model_copy(
                    update={"segments": assign_speakers(transcript.segments, turns)}
                )
                speakers = {s.speaker for s in transcript.segments if s.speaker}
                job.emit("speakers", {"count": len(speakers)})

            transcript = transcript.model_copy(update={"source_file": upload_path.name})
            meeting_dir = _unique_dir(
                output_dir / f"{datetime.now():%Y-%m-%d}_{upload_path.stem}"
            )
            write_transcript(transcript, meeting_dir)

            if not opts.skip_summary:
                job.set_state(JobState.SUMMARIZING)
                template = resolve_template(opts.template)
                client = _make_llm_client(opts.llm_url, opts.llm_model, opts.provider)
                summary = summarize(
                    transcript,
                    template,
                    client,
                    on_progress=lambda i, n: job.emit(
                        "chunk_progress", {"current": i, "total": n}
                    ),
                )
                write_summary(summary, meeting_dir, upload_path.stem)

            job.emit("done", {"meeting": meeting_dir.name})
            job.set_state(JobState.DONE)
        except KNOWN_ERRORS as exc:
            job.emit("error", {"message": str(exc)})
            job.set_state(JobState.ERROR)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return runner


def _resummarize_runner(meeting_dir: Path, req: ResummarizeRequest):
    def runner(job: JobManager) -> None:
        try:
            job.set_state(JobState.SUMMARIZING)
            transcript = load_transcript(meeting_dir / "transcript.json")
            template = resolve_template(req.template)
            client = _make_llm_client(req.llm_url, req.llm_model, req.provider)
            summary = summarize(
                transcript,
                template,
                client,
                on_progress=lambda i, n: job.emit(
                    "chunk_progress", {"current": i, "total": n}
                ),
            )
            write_summary(summary, meeting_dir, Path(transcript.source_file).stem)
            job.emit("done", {"meeting": meeting_dir.name})
            job.set_state(JobState.DONE)
        except KNOWN_ERRORS as exc:
            job.emit("error", {"message": str(exc)})
            job.set_state(JobState.ERROR)

    return runner


def create_app(
    output_dir: Path = Path("output"),
    settings_path: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="meeting-record", docs_url=None, redoc_url=None)
    jobs = JobManager()
    settings_file = settings_path or Path("gui_settings.json")

    def meeting_dir_or_404(name: str) -> Path:
        if name != Path(name).name or name.startswith(".") or ".." in name:
            raise HTTPException(status_code=400, detail="會議名稱不合法")
        d = output_dir / name
        if not d.is_dir() or not (d / "transcript.json").exists():
            raise HTTPException(status_code=404, detail=f"找不到會議「{name}」")
        return d

    # -- 前端 --

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    # -- 任務 --

    @app.post("/api/jobs")
    def start_job(
        file: Annotated[UploadFile, File()],
        options: Annotated[str, Form()] = "{}",
    ) -> dict[str, Any]:
        try:
            opts = ProcessOptions.model_validate_json(options)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"選項格式錯誤:{exc}") from exc

        filename = Path(file.filename or "upload").name
        if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支援的檔案格式，支援:{', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )

        tmpdir = tempfile.mkdtemp(prefix="meeting-record-")
        upload_path = Path(tmpdir) / filename
        try:
            with upload_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)
            jobs.start(
                {"kind": "process", "filename": filename},
                _process_runner(upload_path, tmpdir, opts, output_dir),
            )
        except JobBusyError as exc:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise
        return {"started": True, "filename": filename}

    @app.get("/api/jobs/current")
    def current_job() -> dict[str, Any]:
        return jobs.snapshot()

    @app.get("/api/jobs/events")
    async def job_events(after: int = -1) -> StreamingResponse:
        async def stream():
            seq = after
            while True:
                events = await asyncio.to_thread(jobs.wait_events, seq, 10.0)
                for ev in events:
                    seq = ev["seq"]
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                if jobs.state in TERMINAL_STATES:
                    snap = jobs.snapshot()
                    if not snap["events"] or snap["events"][-1]["seq"] == seq:
                        yield 'event: end\ndata: {"type": "end"}\n\n'
                        return

        return StreamingResponse(stream(), media_type="text/event-stream")

    # -- 會議 --

    @app.get("/api/meetings")
    def list_meetings() -> dict[str, Any]:
        meetings = []
        if output_dir.is_dir():
            for d in output_dir.iterdir():
                if d.is_dir() and (d / "transcript.json").exists():
                    meetings.append(
                        {
                            "name": d.name,
                            "has_summary": (d / "summary.md").exists(),
                            "modified": datetime.fromtimestamp(d.stat().st_mtime).strftime(
                                "%Y-%m-%d %H:%M"
                            ),
                        }
                    )
        meetings.sort(key=lambda m: m["name"], reverse=True)
        return {"meetings": meetings}

    @app.get("/api/meetings/{name}")
    def get_meeting(name: str) -> dict[str, Any]:
        d = meeting_dir_or_404(name)
        transcript_md = (d / "transcript.md").read_text(encoding="utf-8")
        summary_path = d / "summary.md"
        summary_md = (
            summary_path.read_text(encoding="utf-8") if summary_path.exists() else None
        )
        return {
            "name": name,
            "transcript_html": _render_markdown(transcript_md),
            "summary_html": _render_markdown(summary_md) if summary_md else None,
        }

    @app.post("/api/meetings/{name}/resummarize")
    def resummarize_meeting(name: str, req: ResummarizeRequest) -> dict[str, Any]:
        d = meeting_dir_or_404(name)
        try:
            jobs.start(
                {"kind": "resummarize", "meeting": name},
                _resummarize_runner(d, req),
            )
        except JobBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"started": True, "meeting": name}

    # -- 模板 --

    @app.get("/api/templates")
    def get_templates() -> dict[str, Any]:
        items = [
            {"stem": stem, "builtin": builtin, "template": tpl.model_dump()}
            for stem, (tpl, builtin) in list_all_templates().items()
        ]
        items.sort(key=lambda t: (not t["builtin"], t["stem"]))
        return {"templates": items}

    @app.post("/api/templates")
    @app.put("/api/templates/{stem}")
    def upsert_template(payload: TemplatePayload, stem: str | None = None) -> dict[str, Any]:
        target = stem or payload.stem
        try:
            save_user_template(target, payload.template)
        except TemplateError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"saved": target}

    @app.delete("/api/templates/{stem}")
    def remove_template(stem: str) -> dict[str, Any]:
        try:
            delete_user_template(stem)
        except TemplateError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"deleted": stem}

    # -- 設定與環境狀態 --

    @app.get("/api/settings")
    def get_settings() -> GuiSettings:
        return load_settings(settings_file)

    @app.put("/api/settings")
    def put_settings(settings: GuiSettings) -> dict[str, Any]:
        save_settings(settings, settings_file)
        return {"saved": True}

    @app.get("/api/status")
    def get_status(llm_url: str = "http://localhost:11434") -> dict[str, Any]:
        ollama_ok = False
        try:
            resp = httpx.get(f"{llm_url.rstrip('/')}/api/tags", timeout=2.0)
            ollama_ok = resp.status_code == 200
        except httpx.HTTPError:
            pass
        return {
            "hf_token": bool(os.environ.get("HF_TOKEN")),
            "ollama": ollama_ok,
            "diarize_installed": _diarize_installed(),
        }

    return app


def _diarize_installed() -> bool:
    try:
        import pyannote.audio  # noqa: F401

        return True
    except ImportError:
        return False
