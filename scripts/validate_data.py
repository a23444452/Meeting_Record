"""合成資料品質檢查：schema、摘要結構、簡體字、行動項目對應性。"""
import json
import re
import sys
from pathlib import Path

# 常見「簡繁不同形」的簡體字元（僅收錄簡體獨有字形，避免誤判）
# 注意：已排除 後(后) 與 據(据) —— 兩者的簡體字形恰好也是合法的正體字：
#   后：皇后／太后／后羿 等詞在正體中文中本就寫作「后」，並非簡化字。
#   据：「拮据」一詞在正體中文中固定寫作「拮据」，非簡化自「拮據」。
# 保留這兩字會在正常正體中文語料中造成誤判，故移出偵測清單。
SIMPLIFIED_CHARS = set(
    "这们对时说学习还发现问题业务动员总结经过实现应该开计记录报风险预团队"
    "协调专项设备变优认识让为产销费级检验办决进门间关点单电邮张条证论议"
    "会负责仅从众优传伟乐书买争亚产亲亿势复围观购贵资赛质连运选逻错长"
)

REQUIRED_SECTIONS = ["## 會議主旨", "## 討論要點", "## 決議事項", "## 行動項目"]
LENGTH_BANDS = {"short": (400, 1300), "medium": (1000, 2600), "long": (2000, 5000)}
REQUIRED_KEYS = {"id", "seed", "transcript", "summary"}


def find_simplified_chars(text: str) -> set[str]:
    return {c for c in text if c in SIMPLIFIED_CHARS}


def check_summary_structure(summary: str) -> list[str]:
    errors = []
    pos = -1
    for section in REQUIRED_SECTIONS:
        found = summary.find(section)
        if found == -1:
            errors.append(f"缺少段落：{section}")
        elif found < pos:
            errors.append(f"段落順序錯誤：{section}")
        else:
            pos = found
    return errors


def parse_action_items(summary: str) -> list[dict]:
    """解析行動項目表格，回傳資料列；表格語法錯誤回傳空清單。"""
    match = re.search(r"## 行動項目\s*\n(.*?)(?=\n## |\Z)", summary, re.S)
    if not match:
        return []
    rows = []
    lines = [l.strip() for l in match.group(1).splitlines() if l.strip().startswith("|")]
    for line in lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 3 or set(cells[0]) <= {"-", " "} or cells[0] == "事項":
            continue
        rows.append(dict(zip(["事項", "負責人", "期限"], cells)))
    return rows


def check_record(rec: dict) -> list[str]:
    errors = []
    missing = REQUIRED_KEYS - set(rec)
    if missing:
        return [f"缺少欄位：{sorted(missing)}"]

    lo, hi = LENGTH_BANDS.get(rec["seed"].get("length", ""), (400, 5000))
    n = len(rec["transcript"])
    if not lo <= n <= hi:
        errors.append(f"逐字稿長度 {n} 不在 {rec['seed'].get('length')} 級距 [{lo}, {hi}] 內")

    errors += check_summary_structure(rec["summary"])

    simplified = find_simplified_chars(rec["transcript"] + rec["summary"])
    if simplified:
        errors.append(f"含簡體字：{sorted(simplified)}")

    items = parse_action_items(rec["summary"])
    if "## 行動項目" in rec["summary"] and not items:
        errors.append("行動項目表格缺失或語法錯誤")
    for item in items:
        owner = item["負責人"]
        if not owner.strip():
            errors.append(f"行動項目「{item['事項']}」缺少負責人")
        elif owner != "待指派" and owner not in rec["transcript"]:
            errors.append(f"負責人「{owner}」未出現在逐字稿中")
    return errors


def main() -> None:
    raw_dir = Path("data/raw")
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        print("data/raw/ 沒有任何 JSONL 檔", file=sys.stderr)
        sys.exit(1)
    total, failed = 0, []
    seen_ids = set()
    for path in files:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                failed.append((path.name, line_no, "?", [f"JSON 解析失敗：{e}"]))
                continue
            errors = check_record(rec)
            rid = rec.get("id", "?")
            if rid in seen_ids:
                errors.append("id 重複")
            seen_ids.add(rid)
            if errors:
                failed.append((path.name, line_no, rid, errors))
    print(f"共 {total} 筆，通過 {total - len(failed)}，不合格 {len(failed)}")
    for fname, line_no, rid, errors in failed:
        print(f"  ✗ {fname}:{line_no} [{rid}] {'; '.join(errors)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
