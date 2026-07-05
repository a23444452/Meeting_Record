"""輸出：每場會議一個資料夾，含 transcript.md / transcript.json / summary.md。"""

from __future__ import annotations

from pathlib import Path

from .models import Transcript, format_timestamp
from .summarize import render_segments


class OutputError(Exception):
    """寫出結果失敗。"""


def write_transcript(transcript: Transcript, meeting_dir: Path) -> tuple[Path, Path]:
    """寫出 transcript.md（給人看）與 transcript.json（給程式重跑摘要用）。"""
    try:
        meeting_dir.mkdir(parents=True, exist_ok=True)
        md_path = meeting_dir / "transcript.md"
        json_path = meeting_dir / "transcript.json"

        header = (
            f"# 逐字稿：{transcript.source_file}\n\n"
            f"- 長度：{format_timestamp(transcript.duration_seconds)}\n"
            f"- 語言：{transcript.language}\n"
            f"- 模型：whisper {transcript.whisper_model}\n\n"
        )
        md_path.write_text(header + render_segments(transcript.segments) + "\n", encoding="utf-8")
        json_path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
        return md_path, json_path
    except OSError as exc:
        raise OutputError(f"寫出逐字稿失敗：{exc}") from exc


def write_summary(summary_markdown: str, meeting_dir: Path, title: str) -> Path:
    try:
        meeting_dir.mkdir(parents=True, exist_ok=True)
        path = meeting_dir / "summary.md"
        path.write_text(f"# 會議紀錄：{title}\n\n{summary_markdown}\n", encoding="utf-8")
        return path
    except OSError as exc:
        raise OutputError(f"寫出摘要失敗：{exc}") from exc


def load_transcript(json_path: Path) -> Transcript:
    """從 transcript.json 讀回逐字稿（重跑摘要用）。"""
    if not json_path.exists():
        raise OutputError(f"找不到逐字稿檔：{json_path}")
    return Transcript.model_validate_json(json_path.read_text(encoding="utf-8"))
