"""raw JSONL → MLX chat 格式，切分 train/valid/test。"""
import json
import random
from pathlib import Path

SPLITS = {"train": 270, "valid": 20, "test": 10}


def build_message(rec: dict, system_prompt: str) -> dict:
    return {"messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rec["transcript"]},
        {"role": "assistant", "content": rec["summary"]},
    ]}


def split_records(records: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    if len(records) < sum(SPLITS.values()):
        raise ValueError(f"資料不足：{len(records)} < {sum(SPLITS.values())}")
    shuffled = sorted(records, key=lambda r: r["id"])
    random.Random(seed).shuffle(shuffled)
    out, start = {}, 0
    for name, size in SPLITS.items():
        out[name] = shuffled[start:start + size]
        start += size
    return out


def main() -> None:
    system_prompt = Path("config/system_prompt.txt").read_text(encoding="utf-8").strip()
    records = []
    for path in sorted(Path("data/raw").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    parts = split_records(records)
    outdir = Path("data/processed")
    outdir.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        with (outdir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for rec in part:
                f.write(json.dumps(build_message(rec, system_prompt), ensure_ascii=False) + "\n")
        print(f"{name}: {len(part)} 筆")


if __name__ == "__main__":
    main()
