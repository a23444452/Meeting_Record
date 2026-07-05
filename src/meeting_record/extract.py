"""用 ffmpeg 從影片／音訊檔抽出 Whisper 可用的 16kHz 單聲道 WAV。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".mp4", ".m4a", ".mp3", ".wav", ".webm", ".mkv", ".mov", ".flac", ".ogg", ".aiff", ".aac",
}


class ExtractError(Exception):
    """音訊抽取失敗。"""


def extract_audio(input_path: Path, output_path: Path) -> Path:
    """把任意媒體檔轉成 16kHz mono WAV，回傳輸出路徑。"""
    if not input_path.exists():
        raise ExtractError(f"找不到輸入檔：{input_path}")
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ExtractError(
            f"不支援的檔案格式 {input_path.suffix}，支援：{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if shutil.which("ffmpeg") is None:
        raise ExtractError("找不到 ffmpeg，請先安裝（macOS：brew install ffmpeg）")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(input_path),
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-y",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise ExtractError(f"ffmpeg 轉檔失敗：{exc.stderr.strip()[-500:]}") from exc
    return output_path
