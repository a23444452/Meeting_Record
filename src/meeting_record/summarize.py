"""LLM 摘要：支援 Ollama 原生 API 與 OpenAI-compatible endpoint，長逐字稿走 map-reduce。"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal

import httpx

from .models import Segment, Transcript, format_timestamp
from .templates import (
    SYSTEM_PROMPT,
    SummaryTemplate,
    build_chunk_prompt,
    build_summary_prompt,
)

# 單次送進 LLM 的逐字稿字元上限，超過就分段 map-reduce
MAX_CHARS_PER_REQUEST = 12_000
REQUEST_TIMEOUT_SECONDS = 600.0

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class SummarizeError(Exception):
    """摘要生成失敗。"""


class LLMClient:
    """對 Ollama（原生 /api/chat）或 OpenAI-compatible（/v1/chat/completions）發請求。"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        provider: Literal["ollama", "openai"] = "ollama",
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider = provider
        self.api_key = api_key

    def chat(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            if self.provider == "ollama":
                text = self._chat_ollama(messages)
            else:
                text = self._chat_openai(messages)
        except httpx.ConnectError as exc:
            raise SummarizeError(
                f"連不上 LLM 服務（{self.base_url}），請確認 Ollama 已啟動或 endpoint 正確"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise SummarizeError(
                f"LLM 回應錯誤 HTTP {exc.response.status_code}：{exc.response.text[:300]}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise SummarizeError("LLM 回應逾時，長會議可考慮換較小的模型") from exc
        return _THINK_BLOCK.sub("", text).strip()

    def _chat_ollama(self, messages: list[dict[str, str]]) -> str:
        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": False,  # qwen3 等 thinking 模型直接出答案
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _chat_openai(self, messages: list[dict[str, str]]) -> str:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json={"model": self.model, "messages": messages},
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def render_segments(segments: list[Segment]) -> str:
    """把逐字稿段落轉成含時間戳的純文字。"""
    lines = []
    for seg in segments:
        prefix = f"[{format_timestamp(seg.start)}]"
        if seg.speaker:
            prefix += f" {seg.speaker}:"
        lines.append(f"{prefix} {seg.text}")
    return "\n".join(lines)


def chunk_segments(
    segments: list[Segment], max_chars: int = MAX_CHARS_PER_REQUEST
) -> list[list[Segment]]:
    """依段落邊界切塊，每塊渲染後不超過 max_chars（單段超長也自成一塊）。"""
    chunks: list[list[Segment]] = []
    current: list[Segment] = []
    current_len = 0
    for seg in segments:
        seg_len = len(seg.text) + 16  # 時間戳前綴的概估
        if current and current_len + seg_len > max_chars:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(seg)
        current_len += seg_len
    if current:
        chunks.append(current)
    return chunks


def summarize(
    transcript: Transcript,
    template: SummaryTemplate,
    client: LLMClient,
    on_progress: Callable[[int, int], None] | None = None,
) -> str:
    """生成會議紀錄 Markdown。長逐字稿先分段做筆記再彙整。"""
    if not transcript.segments:
        raise SummarizeError("逐字稿是空的，無法摘要")

    full_text = render_segments(transcript.segments)
    if len(full_text) <= MAX_CHARS_PER_REQUEST:
        return client.chat(SYSTEM_PROMPT, build_summary_prompt(template, full_text))

    chunks = chunk_segments(transcript.segments)
    notes: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        if on_progress is not None:
            on_progress(i, len(chunks))
        chunk_text = render_segments(chunk)
        notes.append(client.chat(SYSTEM_PROMPT, build_chunk_prompt(chunk_text, i, len(chunks))))

    combined_notes = "\n\n".join(
        f"（第 {i}/{len(notes)} 段的重點筆記）\n{note}" for i, note in enumerate(notes, 1)
    )
    return client.chat(SYSTEM_PROMPT, build_summary_prompt(template, combined_notes))
