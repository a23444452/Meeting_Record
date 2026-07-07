"""講者辨識：pyannote speaker-diarization-3.1，並把講者對應到 Whisper 段落。

需要 optional extra：uv sync --extra diarize
模型是 gated（免費），需 HuggingFace token，申請步驟見 README。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from .models import Segment

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

TOKEN_HELP = (
    "pyannote 模型需要 HuggingFace token（免費）：\n"
    "  1. 註冊 https://huggingface.co 並在模型頁按「Agree and access」：\n"
    "     https://huggingface.co/pyannote/speaker-diarization-community-1\n"
    "  2. 到 https://huggingface.co/settings/tokens 建立 Read token\n"
    "  3. export HF_TOKEN=<token> 或加 --hf-token 參數"
)


class DiarizeError(Exception):
    """講者辨識失敗。"""


class SpeakerTurn(BaseModel):
    """一段講者發言區間。"""

    model_config = {"frozen": True}

    start: float
    end: float
    speaker: str


def diarize(
    audio_path: Path,
    hf_token: str | None,
    num_speakers: int | None = None,
) -> list[SpeakerTurn]:
    """對 WAV 跑講者辨識，回傳依時間排序的講者區間。"""
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizeError(
            "pyannote.audio 未安裝，請執行：uv sync --extra diarize"
        ) from exc

    if not hf_token:
        raise DiarizeError(f"缺少 HuggingFace token。\n{TOKEN_HELP}")

    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)
        if pipeline is None:
            raise DiarizeError(
                f"無法載入 {DIARIZATION_MODEL}，請確認已在模型頁同意條款。\n{TOKEN_HELP}"
            )
        if torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))

        kwargs = {"num_speakers": num_speakers} if num_speakers else {}
        result = pipeline(str(audio_path), **kwargs)
        # pyannote 4.x 回傳 DiarizeOutput，3.x 直接回傳 Annotation
        annotation = getattr(result, "speaker_diarization", result)
    except DiarizeError:
        raise
    except Exception as exc:
        raise DiarizeError(f"講者辨識失敗：{exc}") from exc

    turns = [
        SpeakerTurn(start=turn.start, end=turn.end, speaker=label)
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]
    return sorted(turns, key=lambda t: t.start)


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(segments: list[Segment], turns: list[SpeakerTurn]) -> list[Segment]:
    """把講者標籤指派給每個轉錄段落（取時間重疊最大的講者），並改名為 講者1、講者2…。

    純函式：回傳新的 Segment 列表，不動原輸入。沒有重疊的段落維持 speaker=None。
    """
    if not turns:
        return list(segments)

    label_map: dict[str, str] = {}

    def friendly(label: str) -> str:
        if label not in label_map:
            label_map[label] = f"講者{len(label_map) + 1}"
        return label_map[label]

    result: list[Segment] = []
    for seg in segments:
        best_label: str | None = None
        best_overlap = 0.0
        for turn in turns:
            if turn.start >= seg.end:
                break  # turns 已排序，後面不會再重疊
            ov = _overlap(seg.start, seg.end, turn.start, turn.end)
            if ov > best_overlap:
                best_overlap = ov
                best_label = turn.speaker
        if best_label is not None:
            result.append(seg.model_copy(update={"speaker": friendly(best_label)}))
        else:
            result.append(seg)
    return result
