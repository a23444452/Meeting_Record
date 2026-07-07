"""faster-whisper 本地轉錄。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .models import Segment, Transcript

# 提示 Whisper 輸出繁體中文（zh 預設常出簡體）
DEFAULT_INITIAL_PROMPT = "以下是繁體中文的會議逐字稿，請使用台灣用語與標點符號。"


class TranscribeError(Exception):
    """轉錄失敗。"""


def transcribe(
    audio_path: Path,
    *,
    model_size: str = "large-v3",
    language: str = "zh",
    initial_prompt: str | None = DEFAULT_INITIAL_PROMPT,
    on_segment: Callable[[Segment], None] | None = None,
) -> Transcript:
    """轉錄 16kHz WAV，回傳完整逐字稿。

    on_segment: 每轉出一段就回呼一次（用來顯示進度）。
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscribeError("faster-whisper 未安裝，請執行 uv sync") from exc

    if not audio_path.exists():
        raise TranscribeError(f"找不到音訊檔：{audio_path}")

    try:
        model = WhisperModel(model_size, device="auto", compute_type="int8")
        raw_segments, info = model.transcribe(
            str(audio_path),
            language=language,
            initial_prompt=initial_prompt,
            vad_filter=True,
            beam_size=5,
        )
    except Exception as exc:  # faster-whisper 丟出的例外型別不固定
        raise TranscribeError(f"Whisper 轉錄失敗：{exc}") from exc

    segments: list[Segment] = []
    for raw in raw_segments:
        seg = Segment(start=raw.start, end=raw.end, text=raw.text.strip())
        segments.append(seg)
        if on_segment is not None:
            on_segment(seg)

    return Transcript(
        source_file=audio_path.name,
        language=info.language,
        whisper_model=model_size,
        duration_seconds=info.duration,
        segments=segments,
    )
