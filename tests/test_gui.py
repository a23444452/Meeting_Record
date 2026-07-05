"""GUI 後端測試：狀態機、API、模板 CRUD、設定、路徑防護（pipeline 全 mock）。"""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from meeting_record.gui.jobs import JobBusyError, JobManager, JobState
from meeting_record.gui.server import create_app
from meeting_record.gui.settings import GuiSettings, load_settings, save_settings
from meeting_record.models import Segment, Transcript
from meeting_record.output import write_summary, write_transcript


def wait_until(cond, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


def make_meeting(output_dir: Path, name: str, with_summary: bool = True) -> Path:
    transcript = Transcript(
        source_file="m.mp4", language="zh", whisper_model="small",
        duration_seconds=10.0,
        segments=[Segment(start=0, end=2, text="測試內容")],
    )
    d = output_dir / name
    write_transcript(transcript, d)
    if with_summary:
        write_summary("## 會議摘要\n重點", d, "m")
    return d


class TestJobManager:
    def test_full_lifecycle_events(self):
        jm = JobManager()

        def runner(job):
            job.set_state(JobState.TRANSCRIBING)
            job.emit("segment", {"text": "哈囉"})
            job.emit("done", {})
            job.set_state(JobState.DONE)

        jm.start({"kind": "test"}, runner)
        assert wait_until(lambda: jm.state == JobState.DONE)
        snap = jm.snapshot()
        types = [e["type"] for e in snap["events"]]
        assert types == ["state", "segment", "done", "state"]
        assert [e["seq"] for e in snap["events"]] == [0, 1, 2, 3]

    def test_busy_rejects_second_job(self):
        jm = JobManager()
        release = []

        def slow(job):
            job.set_state(JobState.TRANSCRIBING)
            while not release:
                time.sleep(0.01)
            job.set_state(JobState.DONE)

        jm.start({}, slow)
        assert wait_until(lambda: jm.state == JobState.TRANSCRIBING)
        with pytest.raises(JobBusyError):
            jm.start({}, slow)
        release.append(True)
        assert wait_until(lambda: jm.state == JobState.DONE)
        jm.start({}, lambda job: job.set_state(JobState.DONE))  # 結束後可再啟動

    def test_wait_events_resumes_from_seq(self):
        jm = JobManager()
        jm.start({}, lambda job: (job.emit("a"), job.emit("b"), job.set_state(JobState.DONE)))
        assert wait_until(lambda: jm.state == JobState.DONE)
        events = jm.wait_events(after=0, timeout=0.1)
        assert [e["type"] for e in events] == ["b", "state"]

    def test_runner_crash_becomes_error(self):
        jm = JobManager()

        def boom(job):
            raise RuntimeError("爆了")

        jm.start({}, boom)
        assert wait_until(lambda: jm.state == JobState.ERROR)
        errors = [e for e in jm.snapshot()["events"] if e["type"] == "error"]
        assert "爆了" in errors[0]["data"]["message"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "meeting_record.templates.USER_TEMPLATE_DIR", tmp_path / "templates"
    )
    app = create_app(
        output_dir=tmp_path / "output", settings_path=tmp_path / "gui_settings.json"
    )
    return TestClient(app), tmp_path


class TestMeetingsApi:
    def test_list_and_read(self, client):
        c, tmp = client
        make_meeting(tmp / "output", "2026-07-05_weekly")
        make_meeting(tmp / "output", "2026-07-04_old", with_summary=False)

        meetings = c.get("/api/meetings").json()["meetings"]
        assert [m["name"] for m in meetings] == ["2026-07-05_weekly", "2026-07-04_old"]
        assert meetings[0]["has_summary"] and not meetings[1]["has_summary"]

        detail = c.get("/api/meetings/2026-07-05_weekly").json()
        assert "測試內容" in detail["transcript_html"]
        assert "會議摘要" in detail["summary_html"]

    def test_path_traversal_rejected(self, client):
        c, _ = client
        assert c.get("/api/meetings/..%2Fsecret").status_code in (400, 404)
        assert c.get("/api/meetings/.hidden").status_code == 400

    def test_missing_meeting_404(self, client):
        c, _ = client
        assert c.get("/api/meetings/nope").status_code == 404

    def test_resummarize_flow(self, client, monkeypatch):
        c, tmp = client
        make_meeting(tmp / "output", "2026-07-05_weekly")
        monkeypatch.setattr(
            "meeting_record.gui.server.summarize", lambda *a, **k: "## 新摘要\n內容"
        )
        resp = c.post(
            "/api/meetings/2026-07-05_weekly/resummarize", json={"template": "project_sync"}
        )
        assert resp.status_code == 200
        assert wait_until(
            lambda: c.get("/api/jobs/current").json()["state"] == "done"
        )
        summary = (tmp / "output/2026-07-05_weekly/summary.md").read_text(encoding="utf-8")
        assert "新摘要" in summary


class TestJobsApi:
    def test_upload_starts_pipeline_and_streams(self, client, monkeypatch):
        c, tmp = client
        transcript = Transcript(
            source_file="x", language="zh", whisper_model="small", duration_seconds=1.0,
            segments=[Segment(start=0, end=1, text="哈囉")],
        )

        def fake_extract(inp, outp):
            outp.write_bytes(b"wav")
            return outp

        def fake_transcribe(path, *, model_size, language, on_segment=None, **kw):
            if on_segment:
                on_segment(transcript.segments[0])
            return transcript

        monkeypatch.setattr("meeting_record.gui.server.extract_audio", fake_extract)
        monkeypatch.setattr("meeting_record.gui.server.transcribe", fake_transcribe)
        monkeypatch.setattr(
            "meeting_record.gui.server.summarize", lambda *a, **k: "## 摘要\nOK"
        )

        resp = c.post(
            "/api/jobs",
            files={"file": ("meeting.mp4", b"fake-video", "video/mp4")},
            data={"options": '{"whisper_model": "small"}'},
        )
        assert resp.status_code == 200
        assert wait_until(lambda: c.get("/api/jobs/current").json()["state"] == "done")

        snap = c.get("/api/jobs/current").json()
        types = [e["type"] for e in snap["events"]]
        assert "segment" in types and "done" in types
        done = next(e for e in snap["events"] if e["type"] == "done")
        meeting_dir = tmp / "output" / done["data"]["meeting"]
        assert (meeting_dir / "transcript.md").exists()
        assert (meeting_dir / "summary.md").exists()

    def test_unsupported_extension_400(self, client):
        c, _ = client
        resp = c.post("/api/jobs", files={"file": ("evil.exe", b"x", "application/x-msdownload")})
        assert resp.status_code == 400

    def test_error_in_pipeline_reported(self, client, monkeypatch):
        c, _ = client
        from meeting_record.extract import ExtractError

        def boom(inp, outp):
            raise ExtractError("ffmpeg 掛了")

        monkeypatch.setattr("meeting_record.gui.server.extract_audio", boom)
        c.post("/api/jobs", files={"file": ("m.mp4", b"x", "video/mp4")})
        assert wait_until(lambda: c.get("/api/jobs/current").json()["state"] == "error")
        events = c.get("/api/jobs/current").json()["events"]
        err = next(e for e in events if e["type"] == "error")
        assert "ffmpeg 掛了" in err["data"]["message"]


class TestTemplatesApi:
    PAYLOAD = {
        "stem": "team_weekly",
        "template": {
            "name": "團隊週報", "description": "測試",
            "sections": [{"title": "摘要", "instruction": "寫摘要", "format": "paragraph"}],
        },
    }

    def test_crud(self, client):
        c, _ = client
        assert c.post("/api/templates", json=self.PAYLOAD).status_code == 200
        stems = [t["stem"] for t in c.get("/api/templates").json()["templates"]]
        assert "team_weekly" in stems and "standard_meeting" in stems
        assert c.delete("/api/templates/team_weekly").status_code == 200
        stems = [t["stem"] for t in c.get("/api/templates").json()["templates"]]
        assert "team_weekly" not in stems

    def test_builtin_protected(self, client):
        c, _ = client
        overwrite = {**self.PAYLOAD, "stem": "standard_meeting"}
        assert c.post("/api/templates", json=overwrite).status_code == 400
        assert c.delete("/api/templates/standard_meeting").status_code == 400

    def test_invalid_stem_rejected(self, client):
        c, _ = client
        bad = {**self.PAYLOAD, "stem": "../evil"}
        assert c.post("/api/templates", json=bad).status_code == 400


class TestSettings:
    def test_roundtrip_via_api(self, client):
        c, tmp = client
        assert c.get("/api/settings").json()["whisper_model"] == "large-v3"
        updated = {**c.get("/api/settings").json(), "whisper_model": "small", "diarize": True}
        assert c.put("/api/settings", json=updated).status_code == 200
        again = c.get("/api/settings").json()
        assert again["whisper_model"] == "small" and again["diarize"] is True

    def test_corrupt_file_falls_back_to_defaults(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text("{not json", encoding="utf-8")
        assert load_settings(p) == GuiSettings()

    def test_save_and_load(self, tmp_path):
        p = tmp_path / "s.json"
        save_settings(GuiSettings(llm_model="qwen3:14b"), p)
        assert load_settings(p).llm_model == "qwen3:14b"
