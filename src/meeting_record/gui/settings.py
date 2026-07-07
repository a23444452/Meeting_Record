"""GUI 預設值持久化（gui_settings.json，不含任何秘密）。"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

DEFAULT_SETTINGS_PATH = Path("gui_settings.json")


class GuiSettings(BaseModel):
    template: str = "standard_meeting"
    language: str = "zh"
    whisper_model: str = "large-v3"
    diarize: bool = False
    num_speakers: int | None = None
    llm_model: str = "qwen3:8b"
    llm_url: str = "http://localhost:11434"
    provider: str = "ollama"


def load_settings(path: Path = DEFAULT_SETTINGS_PATH) -> GuiSettings:
    """讀取設定；檔案不存在或壞掉一律回預設值（不炸）。"""
    if not path.exists():
        return GuiSettings()
    try:
        return GuiSettings.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, json.JSONDecodeError, OSError):
        return GuiSettings()


def save_settings(settings: GuiSettings, path: Path = DEFAULT_SETTINGS_PATH) -> None:
    path.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
