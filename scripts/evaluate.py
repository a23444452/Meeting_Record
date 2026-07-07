"""測試集前後對照評估：基底 vs 微調模型。"""
import json
import sys
from pathlib import Path

from scripts.validate_data import (
    check_summary_structure, find_simplified_chars, parse_action_items,
)

MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MAX_TOKENS = 1024


def score_output(text: str) -> dict:
    return {
        "sections_ok": check_summary_structure(text) == [],
        "table_ok": len(parse_action_items(text)) > 0,
        "no_simplified": len(find_simplified_chars(text)) == 0,
    }


def run_inference(adapter_path: str | None, cases: list[dict], outdir: Path) -> list[dict]:
    from mlx_lm import load, generate  # 延遲 import，單元測試不需要載模型
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(MODEL, adapter_path=adapter_path)
    outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, case in enumerate(cases):
        messages = case["messages"][:2]  # system + user
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        text = generate(model, tokenizer, prompt=prompt,
                        max_tokens=MAX_TOKENS, sampler=make_sampler(temp=0.0))
        (outdir / f"case_{i:02d}.md").write_text(text, encoding="utf-8")
        results.append({"case": i, **score_output(text)})
        print(f"  case {i:02d}: {results[-1]}")
    return results


def summarize(name: str, results: list[dict]) -> dict:
    return {
        "model": name,
        "sections_ok": sum(r["sections_ok"] for r in results),
        "table_ok": sum(r["table_ok"] for r in results),
        "no_simplified": sum(r["no_simplified"] for r in results),
        "total": len(results),
    }


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    cases = [json.loads(l) for l in Path("data/processed/test.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    summaries = []
    if which in ("base", "both"):
        print("=== 基底模型 ===")
        summaries.append(summarize("base", run_inference(None, cases, Path("eval_outputs/base"))))
    if which in ("tuned", "both"):
        print("=== 微調模型 ===")
        summaries.append(summarize("tuned", run_inference("adapters", cases, Path("eval_outputs/tuned"))))
    Path("eval_outputs").mkdir(exist_ok=True)
    out = Path("eval_outputs/summary.json")
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    merged = {s["model"]: s for s in existing} | {s["model"]: s for s in summaries}
    out.write_text(json.dumps(list(merged.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
