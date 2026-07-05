"""核心資料模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Segment(BaseModel):
    """一段轉錄結果（不可變）。"""

    model_config = {"frozen": True}

    start: float = Field(ge=0, description="開始秒數")
    end: float = Field(ge=0, description="結束秒數")
    text: str
    speaker: str | None = None


class Transcript(BaseModel):
    """完整逐字稿。"""

    source_file: str
    language: str
    whisper_model: str
    duration_seconds: float
    segments: list[Segment]


def format_timestamp(seconds: float) -> str:
    """秒數 → HH:MM:SS。"""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
