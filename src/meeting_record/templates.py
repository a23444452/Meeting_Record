"""摘要模板：載入 JSON 模板並組裝 LLM prompt（格式參考 Meetily）。"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

BUILTIN_TEMPLATE_DIR = Path(__file__).parent / "templates"

SYSTEM_PROMPT = (
    "你是專業的會議記錄助理。你會收到一份含時間戳的會議逐字稿，"
    "請依照指定的章節結構撰寫會議紀錄。規則：\n"
    "- 一律使用繁體中文（台灣用語）撰寫，即使逐字稿夾雜英文或簡體字。\n"
    "- 只根據逐字稿內容撰寫，不要編造逐字稿中沒有的資訊。\n"
    "- 專有名詞、產品名、人名保留原文。\n"
    "- 直接輸出 Markdown，不要加任何前言或說明。"
)


class TemplateError(Exception):
    """模板載入或驗證失敗。"""


class TemplateSection(BaseModel):
    title: str
    instruction: str
    format: str = "paragraph"  # paragraph | list | table


class SummaryTemplate(BaseModel):
    name: str
    description: str = ""
    sections: list[TemplateSection]


def list_templates(template_dir: Path = BUILTIN_TEMPLATE_DIR) -> dict[str, SummaryTemplate]:
    """回傳 {模板檔名（不含副檔名）: 模板}。"""
    result: dict[str, SummaryTemplate] = {}
    for path in sorted(template_dir.glob("*.json")):
        result[path.stem] = load_template(path)
    return result


def load_template(path: Path) -> SummaryTemplate:
    if not path.exists():
        raise TemplateError(f"找不到模板：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return SummaryTemplate.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise TemplateError(f"模板格式錯誤（{path.name}）：{exc}") from exc


def resolve_template(name_or_path: str) -> SummaryTemplate:
    """接受內建模板名（如 standard_meeting）或 JSON 檔路徑。"""
    builtin = BUILTIN_TEMPLATE_DIR / f"{name_or_path}.json"
    if builtin.exists():
        return load_template(builtin)
    path = Path(name_or_path)
    if path.suffix == ".json":
        return load_template(path)
    available = ", ".join(list_templates())
    raise TemplateError(f"找不到模板「{name_or_path}」，內建模板：{available}")


def build_summary_prompt(template: SummaryTemplate, transcript_text: str) -> str:
    """組裝單次摘要的 user prompt。"""
    section_lines = []
    for i, sec in enumerate(template.sections, 1):
        fmt = {"paragraph": "段落", "list": "條列", "table": "表格"}.get(sec.format, sec.format)
        section_lines.append(f"{i}. 「{sec.title}」（{fmt}）：{sec.instruction}")
    sections_block = "\n".join(section_lines)
    return (
        f"請依照以下章節結構撰寫會議紀錄，每個章節用 Markdown 二級標題（##）：\n"
        f"{sections_block}\n\n"
        f"會議逐字稿：\n---\n{transcript_text}\n---"
    )


def build_chunk_prompt(transcript_chunk: str, part: int, total: int) -> str:
    """長會議分段時，先對每段做重點筆記的 prompt。"""
    return (
        f"以下是一場會議逐字稿的第 {part}/{total} 段。"
        f"請用繁體中文條列這一段的重點：討論主題、達成的決議、提到的待辦事項"
        f"（含負責人與時間戳）。只根據內容撰寫，不要編造。\n"
        f"---\n{transcript_chunk}\n---"
    )
