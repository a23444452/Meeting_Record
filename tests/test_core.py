"""核心純邏輯測試：模板、分段、渲染、輸出 round-trip。"""

from pathlib import Path

import pytest

from meeting_record.models import Segment, Transcript, format_timestamp
from meeting_record.output import load_transcript, write_summary, write_transcript
from meeting_record.summarize import LLMClient, chunk_segments, render_segments, summarize
from meeting_record.templates import (
    TemplateError,
    build_summary_prompt,
    list_templates,
    resolve_template,
)


def make_transcript(segments: list[Segment]) -> Transcript:
    return Transcript(
        source_file="meeting.mp4",
        language="zh",
        whisper_model="large-v3",
        duration_seconds=3600.0,
        segments=segments,
    )


class TestFormatTimestamp:
    def test_basic(self):
        assert format_timestamp(0) == "00:00:00"
        assert format_timestamp(75.4) == "00:01:15"
        assert format_timestamp(3661) == "01:01:01"


class TestTemplates:
    def test_builtin_templates_load(self):
        templates = list_templates()
        assert "standard_meeting" in templates
        assert "project_sync" in templates
        for tpl in templates.values():
            assert tpl.sections, f"{tpl.name} 沒有章節"

    def test_resolve_unknown_template_raises(self):
        with pytest.raises(TemplateError, match="standard_meeting"):
            resolve_template("does_not_exist")

    def test_build_summary_prompt_contains_sections_and_transcript(self):
        tpl = resolve_template("standard_meeting")
        prompt = build_summary_prompt(tpl, "[00:00:01] 大家好")
        assert "會議摘要" in prompt
        assert "待辦事項" in prompt
        assert "[00:00:01] 大家好" in prompt


class TestRenderAndChunk:
    def test_render_segments_with_speaker(self):
        segs = [
            Segment(start=0, end=2, text="大家好", speaker="Alice"),
            Segment(start=2, end=5, text="開始開會"),
        ]
        text = render_segments(segs)
        assert "[00:00:00] Alice: 大家好" in text
        assert "[00:00:02] 開始開會" in text

    def test_chunk_respects_max_chars(self):
        segs = [Segment(start=i, end=i + 1, text="字" * 100) for i in range(50)]
        chunks = chunk_segments(segs, max_chars=1000)
        assert len(chunks) > 1
        assert sum(len(c) for c in chunks) == 50  # 沒有掉段
        for chunk in chunks:
            assert len(render_segments(chunk)) <= 1200  # 上限加前綴概估的餘裕

    def test_single_oversized_segment_still_forms_chunk(self):
        segs = [Segment(start=0, end=1, text="字" * 5000)]
        chunks = chunk_segments(segs, max_chars=1000)
        assert len(chunks) == 1


class TestOutputRoundTrip:
    def test_write_and_load_transcript(self, tmp_path: Path):
        transcript = make_transcript([Segment(start=0, end=2, text="測試內容")])
        md_path, json_path = write_transcript(transcript, tmp_path / "meeting")
        assert "測試內容" in md_path.read_text(encoding="utf-8")

        loaded = load_transcript(json_path)
        assert loaded == transcript

    def test_write_summary(self, tmp_path: Path):
        path = write_summary("## 會議摘要\n內容", tmp_path, "週會")
        content = path.read_text(encoding="utf-8")
        assert content.startswith("# 會議紀錄：週會")


class TestSummarize:
    def test_short_transcript_single_request(self, monkeypatch):
        calls = []

        def fake_chat(self, system, user):
            calls.append(user)
            return "## 會議摘要\n測試"

        monkeypatch.setattr(LLMClient, "chat", fake_chat)
        transcript = make_transcript([Segment(start=0, end=2, text="短會議")])
        result = summarize(transcript, resolve_template("standard_meeting"), LLMClient())
        assert result == "## 會議摘要\n測試"
        assert len(calls) == 1

    def test_long_transcript_map_reduce(self, monkeypatch):
        calls = []

        def fake_chat(self, system, user):
            calls.append(user)
            return "重點筆記"

        monkeypatch.setattr(LLMClient, "chat", fake_chat)
        segs = [Segment(start=i, end=i + 1, text="字" * 500) for i in range(40)]  # 遠超 12k 字
        summarize(make_transcript(segs), resolve_template("standard_meeting"), LLMClient())
        assert len(calls) > 2  # 多段筆記 + 最後彙整

    def test_empty_transcript_raises(self):
        from meeting_record.summarize import SummarizeError

        with pytest.raises(SummarizeError):
            summarize(make_transcript([]), resolve_template("standard_meeting"), LLMClient())


class TestThinkStripping:
    def test_think_block_removed(self, monkeypatch):
        def fake_post(url, **kwargs):
            class Resp:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"message": {"content": "<think>思考中…</think>## 摘要\n內容"}}

            return Resp()

        monkeypatch.setattr("meeting_record.summarize.httpx.post", fake_post)
        result = LLMClient().chat("system", "user")
        assert result == "## 摘要\n內容"
