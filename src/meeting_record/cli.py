"""meeting-record CLI：Teams 錄影檔 → 本地轉錄 → LLM 會議紀錄。"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .extract import ExtractError, extract_audio
from .models import format_timestamp
from .output import OutputError, load_transcript, write_summary, write_transcript
from .summarize import LLMClient, SummarizeError, summarize
from .templates import TemplateError, list_templates, resolve_template
from .transcribe import TranscribeError, transcribe

app = typer.Typer(help="本地會議記錄工具：轉錄 Teams 錄影並生成會議紀錄", no_args_is_help=True)
console = Console()

KNOWN_ERRORS = (ExtractError, TranscribeError, SummarizeError, TemplateError, OutputError)


def _make_client(llm_url: str, llm_model: str, provider: str, api_key: str | None) -> LLMClient:
    if provider not in ("ollama", "openai"):
        raise typer.BadParameter("provider 只能是 ollama 或 openai")
    return LLMClient(base_url=llm_url, model=llm_model, provider=provider, api_key=api_key)  # type: ignore[arg-type]


@app.command()
def process(
    input_file: Annotated[Path, typer.Argument(help="Teams 錄影或音訊檔（mp4/m4a/mp3/wav…）")],
    template: Annotated[str, typer.Option("--template", "-t", help="摘要模板名或 JSON 路徑")] = "standard_meeting",
    language: Annotated[str, typer.Option("--language", "-l", help="轉錄語言")] = "zh",
    whisper_model: Annotated[str, typer.Option(help="Whisper 模型")] = "large-v3",
    llm_model: Annotated[str, typer.Option(help="摘要用 LLM 模型")] = "qwen3:8b",
    llm_url: Annotated[str, typer.Option(help="LLM endpoint")] = "http://localhost:11434",
    provider: Annotated[str, typer.Option(help="ollama 或 openai（OpenAI-compatible）")] = "ollama",
    api_key: Annotated[str | None, typer.Option(envvar="MEETING_RECORD_API_KEY")] = None,
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("output"),
    skip_summary: Annotated[bool, typer.Option("--skip-summary", help="只轉錄不摘要")] = False,
) -> None:
    """完整流程:抽音訊 → 轉錄 → 生成會議紀錄。"""
    try:
        summary_template = None if skip_summary else resolve_template(template)

        meeting_name = f"{datetime.now():%Y-%m-%d}_{input_file.stem}"
        meeting_dir = output_dir / meeting_name

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "audio.wav"
            console.print(f"[cyan]1/3 抽取音訊[/cyan] {input_file.name} → 16kHz WAV")
            extract_audio(input_file, wav_path)

            console.print(f"[cyan]2/3 本地轉錄[/cyan] whisper {whisper_model}（首次會下載模型）")
            transcript = transcribe(
                wav_path,
                model_size=whisper_model,
                language=language,
                on_segment=lambda seg: console.print(
                    f"  [dim][{format_timestamp(seg.start)}][/dim] {seg.text}"
                ),
            )

        transcript = transcript.model_copy(update={"source_file": input_file.name})
        md_path, json_path = write_transcript(transcript, meeting_dir)
        console.print(f"[green]逐字稿完成[/green] → {md_path}")

        if skip_summary or summary_template is None:
            console.print("[yellow]已依 --skip-summary 略過摘要[/yellow]")
            return

        console.print(f"[cyan]3/3 生成會議紀錄[/cyan] {llm_model}（{provider}）")
        client = _make_client(llm_url, llm_model, provider, api_key)
        summary = summarize(
            transcript,
            summary_template,
            client,
            on_progress=lambda i, n: console.print(f"  [dim]長會議分段摘要 {i}/{n}[/dim]"),
        )
        summary_path = write_summary(summary, meeting_dir, input_file.stem)
        console.print(f"[green]會議紀錄完成[/green] → {summary_path}")
    except KNOWN_ERRORS as exc:
        console.print(f"[red]錯誤：[/red]{exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def resummarize(
    transcript_json: Annotated[Path, typer.Argument(help="既有的 transcript.json")],
    template: Annotated[str, typer.Option("--template", "-t")] = "standard_meeting",
    llm_model: Annotated[str, typer.Option(help="摘要用 LLM 模型")] = "qwen3:8b",
    llm_url: Annotated[str, typer.Option(help="LLM endpoint")] = "http://localhost:11434",
    provider: Annotated[str, typer.Option(help="ollama 或 openai")] = "ollama",
    api_key: Annotated[str | None, typer.Option(envvar="MEETING_RECORD_API_KEY")] = None,
) -> None:
    """用不同模板或模型，對既有逐字稿重新生成會議紀錄（不重跑轉錄）。"""
    try:
        transcript = load_transcript(transcript_json)
        summary_template = resolve_template(template)
        client = _make_client(llm_url, llm_model, provider, api_key)
        console.print(f"[cyan]生成會議紀錄[/cyan] {llm_model}（{provider}）")
        summary = summarize(
            transcript,
            summary_template,
            client,
            on_progress=lambda i, n: console.print(f"  [dim]長會議分段摘要 {i}/{n}[/dim]"),
        )
        summary_path = write_summary(
            summary, transcript_json.parent, Path(transcript.source_file).stem
        )
        console.print(f"[green]會議紀錄完成[/green] → {summary_path}")
    except KNOWN_ERRORS as exc:
        console.print(f"[red]錯誤：[/red]{exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def templates() -> None:
    """列出可用的摘要模板。"""
    for name, tpl in list_templates().items():
        console.print(f"[bold]{name}[/bold] — {tpl.description}")
        for sec in tpl.sections:
            console.print(f"  • {sec.title}")


if __name__ == "__main__":
    app()
