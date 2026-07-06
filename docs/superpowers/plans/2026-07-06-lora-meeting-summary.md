# 繁中會議摘要 LoRA 微調 — 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 M4/24GB Mac 上以 MLX-LM QLoRA 微調 Qwen3-4B，產出繁中台灣用語會議摘要專用模型並部署至 Ollama。

**Architecture:** 五階段線性 pipeline：情境種子 → Claude 合成資料（300 筆）→ 驗證/切分 → QLoRA 訓練 → 前後對照評估 → fuse + Ollama 匯入。純邏輯腳本（種子、驗證、準備、評估檢查）走 TDD；訓練與轉檔屬外部工具呼叫，走煙霧測試。

**Tech Stack:** Python 3.12 + uv、mlx-lm（訓練/推論/fuse）、pytest、Ollama（部署，內建 GGUF 轉換與量化）。

**Spec:** `docs/superpowers/specs/2026-07-06-lora-meeting-summary-design.md`

**與 spec 的一個實作層偏差：** spec §5 匯出走「llama.cpp convert → llama-quantize」；本計畫改用 `ollama create --quantize q4_K_M` 直接匯入 fuse 出的 safetensors（Ollama 0.16 內建同一套 llama.cpp 轉換邏輯，少裝一套工具鏈）。若 Ollama 直接匯入失敗，備案仍為手動 llama.cpp 流程（Task 9 內附）。

---

## 檔案結構總覽

| 檔案 | 職責 |
|------|------|
| `pyproject.toml` | uv 專案定義；deps: mlx-lm、pyyaml；dev: pytest |
| `config/system_prompt.txt` | 唯一的 system prompt 來源（訓練資料與 Modelfile 共用，DRY） |
| `config/gen_prompt_template.md` | 資料生成 subagent 的 prompt 模板 |
| `config/lora_config.yaml` | mlx_lm lora 訓練設定 |
| `scripts/gen_seeds.py` | 確定性生成 300 筆情境種子 → `data/seeds/scenarios.json` |
| `scripts/validate_data.py` | 資料 schema／結構／簡體字／對應性檢查（可 import 的純函式 + CLI） |
| `scripts/prepare_data.py` | raw → MLX chat 格式，切分 270/20/10 → `data/processed/` |
| `scripts/train.sh` | 訓練入口（含煙霧模式） |
| `scripts/evaluate.py` | 基底 vs 微調模型測試集推論 + 程式化指標 |
| `scripts/export.sh` | fuse → Ollama 匯入 |
| `ollama/Modelfile` | 部署定義（SYSTEM、TEMPLATE、推論參數） |
| `tests/test_*.py` | 上述純邏輯的單元測試 |

---

### Task 1: 專案腳手架

**Files:**
- Create: `pyproject.toml`（uv 產生後修改）
- Create: 目錄骨架

- [ ] **Step 1: 初始化 uv 專案與目錄**

```bash
cd /Users/vincewang/LoRA_Model
uv init --bare --python 3.12
uv add "mlx-lm>=0.24" pyyaml
uv add --dev pytest
mkdir -p config data/seeds data/raw data/processed scripts tests ollama adapters export
touch tests/__init__.py
```

- [ ] **Step 2: 驗證環境**

Run: `uv run python -c "import mlx_lm; print(mlx_lm.__version__)"`
Expected: 印出版本號（≥0.24），無錯誤。

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock .python-version tests/__init__.py
git commit -m "chore: 初始化 uv 專案與 mlx-lm 依賴"
```

---

### Task 2: System prompt 與生成模板

**Files:**
- Create: `config/system_prompt.txt`
- Create: `config/gen_prompt_template.md`

- [ ] **Step 1: 寫入 system prompt（訓練與部署共用）**

`config/system_prompt.txt`（結尾不留空行以外的多餘空白）：

```
你是專業的會議紀錄助理。使用者會提供一份會議逐字稿，請你以繁體中文（台灣用語）輸出結構化會議紀錄，格式固定為：

## 會議主旨
（一到兩句話概括會議目的與背景）

## 討論要點
- （條列討論脈絡）

## 決議事項
- （已定案的結論；無則寫「本次會議無正式決議」）

## 行動項目
| 事項 | 負責人 | 期限 |
|------|--------|------|

規則：忠於逐字稿內容，不得捏造；行動項目無明確期限填「未定」，未指名負責人填「待指派」。
```

- [ ] **Step 2: 寫入資料生成模板** `config/gen_prompt_template.md`：

```markdown
# 會議資料生成指示（單批 10 筆）

你要生成擬真的台灣職場會議「逐字稿 + 標準摘要」配對，作為 LoRA 訓練資料。

## 輸入
下方附上 10 筆情境種子（JSON）。每筆種子含：id、industry、meeting_type、num_participants、length、noise_features。

## 逐字稿要求
1. 繁體中文台灣用語，職場口語自然（「那個」「就是說」「對啊」「欸」等贅詞適量）。
2. 發言格式：`人名：發言內容`，每行一段發言。人名用台灣常見名字（中文名或英文名混用皆可）。
3. 長度：short 約 600–1100 字、medium 約 1200–2200 字、long 約 2400–4200 字（計中文字元）。
4. 依 noise_features 加入指定雜訊（離題閒聊、互相打斷、中英夾雜、中途加入/離開等）。
5. 同批 10 筆的議題必須彼此不同，貼合 industry × meeting_type。
6. 每場會議至少要有 1 個行動項目可被歸納；至少一半的種子要包含明確的負責人與期限資訊。

## 摘要要求
嚴格遵守以下結構（與 system prompt 一致）：
`## 會議主旨`、`## 討論要點`、`## 決議事項`、`## 行動項目`（Markdown 表格：| 事項 | 負責人 | 期限 |）。
- 只寫逐字稿中真實出現的內容，不捏造。
- 無正式決議時，決議事項寫「本次會議無正式決議」。
- 行動項目無期限填「未定」、無負責人填「待指派」；負責人名字必須與逐字稿中出現的字串一致。

## 輸出
寫入指定的 JSONL 檔案（一行一筆，UTF-8）：
{"id": "...", "seed": {...原封不動...}, "transcript": "...", "summary": "..."}
完成後回報：成功筆數、每筆的 id 與議題一句話。
```

- [ ] **Step 3: Commit**

```bash
git add config/system_prompt.txt config/gen_prompt_template.md
git commit -m "feat: 新增共用 system prompt 與資料生成模板"
```

---

### Task 3: 情境種子生成器（TDD）

**Files:**
- Create: `scripts/gen_seeds.py`
- Test: `tests/test_gen_seeds.py`

- [ ] **Step 1: 寫失敗測試** `tests/test_gen_seeds.py`：

```python
from scripts.gen_seeds import generate_seeds

def test_generates_300_unique_seeds():
    seeds = generate_seeds()
    assert len(seeds) == 300
    assert len({s["id"] for s in seeds}) == 300

def test_seed_schema():
    s = generate_seeds()[0]
    assert set(s) == {"id", "industry", "meeting_type", "num_participants", "length", "noise_features"}
    assert s["length"] in {"short", "medium", "long"}
    assert 3 <= s["num_participants"] <= 8
    assert 1 <= len(s["noise_features"]) <= 3

def test_deterministic():
    assert generate_seeds() == generate_seeds()

def test_coverage():
    seeds = generate_seeds()
    assert len({s["industry"] for s in seeds}) >= 10
    assert len({s["meeting_type"] for s in seeds}) >= 8
    lengths = [s["length"] for s in seeds]
    assert lengths.count("long") >= 40  # 長逐字稿佔比不可太低
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `uv run pytest tests/test_gen_seeds.py -v`
Expected: FAIL（ModuleNotFoundError 或 ImportError）。

- [ ] **Step 3: 實作** `scripts/gen_seeds.py`：

```python
"""確定性生成 300 筆會議情境種子。"""
import json
import random
from pathlib import Path

INDUSTRIES = ["科技軟體", "電子製造", "電商零售", "醫療院所", "金融保險",
              "教育培訓", "餐飲連鎖", "行銷代理", "新創團隊", "非營利組織"]
MEETING_TYPES = ["部門週會", "專案進度檢討", "客戶需求訪談", "跨部門協調會",
                 "緊急事故處理", "產品規劃會議", "業績檢討會", "供應商洽談"]
NOISE_FEATURES = ["口語贅詞偏多", "有人離題閒聊", "發言互相打斷",
                  "中英夾雜（技術詞彙）", "有人中途加入", "有人提早離席"]
LENGTH_WEIGHTS = [("short", 0.4), ("medium", 0.4), ("long", 0.2)]


def generate_seeds(n: int = 300, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    lengths = [name for name, w in LENGTH_WEIGHTS for _ in range(int(n * w))]
    lengths += ["medium"] * (n - len(lengths))
    rng.shuffle(lengths)
    seeds = []
    for i in range(n):
        industry = INDUSTRIES[i % len(INDUSTRIES)]
        meeting_type = MEETING_TYPES[(i // len(INDUSTRIES)) % len(MEETING_TYPES)]
        seeds.append({
            "id": f"s{i + 1:03d}",
            "industry": industry,
            "meeting_type": meeting_type,
            "num_participants": rng.randint(3, 8),
            "length": lengths[i],
            "noise_features": sorted(rng.sample(NOISE_FEATURES, rng.randint(1, 3))),
        })
    return seeds


def main() -> None:
    out = Path("data/seeds/scenarios.json")
    out.write_text(json.dumps(generate_seeds(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已寫入 {out}（300 筆）")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑測試確認通過**

Run: `uv run pytest tests/test_gen_seeds.py -v`
Expected: 4 passed。
（若 import 失敗，在 `pyproject.toml` 加 `[tool.pytest.ini_options]` `pythonpath = ["."]`。）

- [ ] **Step 5: 產生種子檔並抽查**

Run: `uv run python scripts/gen_seeds.py && uv run python -c "import json; d=json.load(open('data/seeds/scenarios.json')); print(len(d), d[0], d[-1])"`
Expected: 300 筆，首尾格式正確。

- [ ] **Step 6: Commit**

```bash
git add scripts/gen_seeds.py tests/test_gen_seeds.py data/seeds/scenarios.json pyproject.toml
git commit -m "feat: 情境種子生成器（300 筆確定性矩陣）"
```

---

### Task 4: 資料驗證器（TDD）

**Files:**
- Create: `scripts/validate_data.py`
- Test: `tests/test_validate_data.py`

- [ ] **Step 1: 寫失敗測試** `tests/test_validate_data.py`：

```python
from scripts.validate_data import (
    find_simplified_chars, check_summary_structure,
    parse_action_items, check_record,
)

GOOD_SUMMARY = """## 會議主旨
討論第三季產品上線時程。

## 討論要點
- 前端進度落後兩週，需要增援。

## 決議事項
- 上線日期延後至十月十五日。

## 行動項目
| 事項 | 負責人 | 期限 |
|------|--------|------|
| 提出增援名單 | 小美 | 未定 |
"""


def make_record(**overrides):
    rec = {
        "id": "s001",
        "seed": {"id": "s001", "industry": "科技軟體", "meeting_type": "部門週會",
                 "num_participants": 4, "length": "short", "noise_features": ["口語贅詞偏多"]},
        "transcript": "主管：我們開始開會。小美：好的，那個前端的部分我說明一下。" + "大家討論了很久。" * 60,
        "summary": GOOD_SUMMARY,
    }
    rec.update(overrides)
    return rec


def test_simplified_chars_detected():
    assert find_simplified_chars("这个会议有问题") >= {"这", "会", "议", "问", "题"}
    assert find_simplified_chars("這個會議沒有問題") == set()


def test_summary_structure_ok():
    assert check_summary_structure(GOOD_SUMMARY) == []


def test_summary_structure_missing_section():
    errors = check_summary_structure(GOOD_SUMMARY.replace("## 決議事項", "## 結論"))
    assert any("決議事項" in e for e in errors)


def test_parse_action_items():
    items = parse_action_items(GOOD_SUMMARY)
    assert items == [{"事項": "提出增援名單", "負責人": "小美", "期限": "未定"}]


def test_good_record_passes():
    assert check_record(make_record()) == []


def test_owner_missing_from_transcript_fails():
    bad = GOOD_SUMMARY.replace("小美", "阿宏")
    errors = check_record(make_record(summary=bad))
    assert any("負責人" in e for e in errors)


def test_transcript_too_short_fails():
    errors = check_record(make_record(transcript="主管：散會。小美：好。"))
    assert any("長度" in e for e in errors)
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `uv run pytest tests/test_validate_data.py -v`
Expected: FAIL（ImportError）。

- [ ] **Step 3: 實作** `scripts/validate_data.py`：

```python
"""合成資料品質檢查：schema、摘要結構、簡體字、行動項目對應性。"""
import json
import re
import sys
from pathlib import Path

# 常見「簡繁不同形」的簡體字元（僅收錄簡體獨有字形，避免誤判）
SIMPLIFIED_CHARS = set(
    "这们对时说学习还发现问题业务动员总结经过实现应该开计记录报风险预团队"
    "协调专项设备变优认识让为产销费级检验办决进门间关点单电邮张条证论据议"
    "会负责仅从众优传伟乐书买争亚产亲亿势后复围观购贵资赛质连运选逻错长"
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
    match = re.search(r"## 行動項目\s*\n(.*)", summary, re.S)
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
        if owner != "待指派" and owner not in rec["transcript"]:
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
```

- [ ] **Step 4: 跑測試確認通過**

Run: `uv run pytest tests/test_validate_data.py -v`
Expected: 7 passed。

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_data.py tests/test_validate_data.py
git commit -m "feat: 合成資料驗證器（結構/簡體字/對應性檢查）"
```

---

### Task 5: 合成資料生成（主 session 協調，非派工任務）

> **注意：此任務由主 session（Claude 協調者）親自執行**——生成品質即教師模型品質，不可轉派給實作 subagent 用程式生成。

**Files:**
- Create: `data/raw/batch_01.jsonl` … `data/raw/batch_30.jsonl`

- [ ] **Step 1:** 讀取 `data/seeds/scenarios.json`，每 10 筆一批，共 30 批。
- [ ] **Step 2:** 每批派一個 subagent（Agent tool），prompt = `config/gen_prompt_template.md` 內容 + 該批 10 筆種子 JSON + 輸出路徑 `data/raw/batch_NN.jsonl`。可 3–4 批並行。
- [ ] **Step 3:** 每批完成後跑 `uv run python scripts/validate_data.py`，不合格的 id 連同錯誤原因重新派工生成（只重生不合格筆，覆寫該批檔案中對應行）。
- [ ] **Step 4:** 全部 300 筆通過驗證後：

Run: `uv run python scripts/validate_data.py`
Expected: `共 300 筆，通過 300，不合格 0`，exit code 0。

- [ ] **Step 5: Commit**

```bash
git add data/raw/
git commit -m "feat: 合成 300 筆繁中會議逐字稿與標準摘要"
```

---

### Task 6: 資料準備與切分（TDD）

**Files:**
- Create: `scripts/prepare_data.py`
- Test: `tests/test_prepare_data.py`

- [ ] **Step 1: 寫失敗測試** `tests/test_prepare_data.py`：

```python
import json
from scripts.prepare_data import build_message, split_records

SYSTEM = "測試用 system prompt"


def test_build_message_chat_format():
    rec = {"id": "s001", "transcript": "主管：開會了。", "summary": "## 會議主旨\n測試。"}
    msg = build_message(rec, SYSTEM)
    roles = [m["role"] for m in msg["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert msg["messages"][1]["content"] == "主管：開會了。"
    assert msg["messages"][2]["content"].startswith("## 會議主旨")


def test_split_sizes_and_determinism():
    records = [{"id": f"s{i:03d}"} for i in range(300)]
    a = split_records(records, seed=42)
    b = split_records(records, seed=42)
    assert [len(a["train"]), len(a["valid"]), len(a["test"])] == [270, 20, 10]
    assert a == b
    all_ids = [r["id"] for part in a.values() for r in part]
    assert len(set(all_ids)) == 300
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `uv run pytest tests/test_prepare_data.py -v`
Expected: FAIL（ImportError）。

- [ ] **Step 3: 實作** `scripts/prepare_data.py`：

```python
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
```

- [ ] **Step 4: 跑測試確認通過**

Run: `uv run pytest tests/test_prepare_data.py -v`
Expected: 2 passed。

- [ ] **Step 5: 實際執行並檢查**

Run: `uv run python scripts/prepare_data.py && wc -l data/processed/*.jsonl`
Expected: train 270 / valid 20 / test 10。

- [ ] **Step 6: Commit**

```bash
git add scripts/prepare_data.py tests/test_prepare_data.py data/processed/
git commit -m "feat: 資料轉換為 MLX chat 格式並切分 270/20/10"
```

---

### Task 7: 訓練（設定 + 煙霧 + 全量）

**Files:**
- Create: `config/lora_config.yaml`
- Create: `scripts/train.sh`

- [ ] **Step 1: 確認 mlx-lm CLI 介面**

Run: `uv run mlx_lm.lora --help 2>/dev/null || uv run python -m mlx_lm lora --help`
Expected: 印出可用參數。**以 --help 實際輸出為準**，若下方 config 欄位名不符，修 config 對齊。

- [ ] **Step 2: 寫訓練設定** `config/lora_config.yaml`：

```yaml
model: "mlx-community/Qwen3-4B-Instruct-2507-4bit"
train: true
fine_tune_type: lora
data: "data/processed"
seed: 42
num_layers: 16
batch_size: 1
iters: 600
val_batches: 20
learning_rate: 1.0e-5
steps_per_report: 10
steps_per_eval: 50
adapter_path: "adapters"
save_every: 100
max_seq_length: 4096
grad_checkpoint: true
lora_parameters:
  keys: ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
  rank: 16
  scale: 2.0
  dropout: 0.05
```

- [ ] **Step 3: 寫訓練入口** `scripts/train.sh`：

```bash
#!/bin/bash
# 用法：scripts/train.sh [smoke]  — smoke 模式只跑 10 iters 驗證 pipeline
set -euo pipefail
cd "$(dirname "$0")/.."
EXTRA=""
if [ "${1:-}" = "smoke" ]; then
  EXTRA="--iters 10 --steps-per-eval 5"
  echo "=== 煙霧測試模式（10 iters）==="
fi
uv run mlx_lm.lora -c config/lora_config.yaml $EXTRA
```

Run: `chmod +x scripts/train.sh`

- [ ] **Step 4: 基底模型下載 + 推論煙霧測試**（首跑會下載 ~2.3GB）

Run: `uv run mlx_lm.generate --model mlx-community/Qwen3-4B-Instruct-2507-4bit --prompt "用一句話介紹台北" --max-tokens 60`
Expected: 輸出通順繁中句子。

- [ ] **Step 5: 訓練煙霧測試**

Run: `scripts/train.sh smoke`
Expected: 10 iters 跑完無 OOM、`adapters/` 產生檔案、train loss 有數值。
若 OOM：`max_seq_length` 降 2048 重試。

- [ ] **Step 6: 全量訓練（背景執行，記錄 log）**

Run: `scripts/train.sh 2>&1 | tee adapters/train.log`（預估 30–60 分鐘）
Expected: valid loss 持續下降後趨平。**檢查 log：若最後 100 iters valid loss 明顯回升，記下最佳 checkpoint（`adapters/` 內每 100 iters 有存檔），評估時改用該 checkpoint。**

- [ ] **Step 7: 訓練後快速推論驗證**

Run: 取 `data/processed/test.jsonl` 第一筆的 user content 存成 `/tmp` 檔案，執行：
`uv run mlx_lm.generate --model mlx-community/Qwen3-4B-Instruct-2507-4bit --adapter-path adapters --system-prompt "$(cat config/system_prompt.txt)" --prompt "$(cat /tmp/sample_transcript.txt)" --max-tokens 800`
Expected: 輸出含四個段落的結構化摘要。

- [ ] **Step 8: Commit**

```bash
git add config/lora_config.yaml scripts/train.sh
git commit -m "feat: MLX QLoRA 訓練設定與入口腳本"
```

---

### Task 8: 評估（TDD 檢查函式 + 前後對照）

**Files:**
- Create: `scripts/evaluate.py`
- Test: `tests/test_evaluate.py`
- Create: `docs/eval_report.md`（執行產出）

- [ ] **Step 1: 寫失敗測試** `tests/test_evaluate.py`：

```python
from scripts.evaluate import score_output

GOOD = """## 會議主旨
測試會議。

## 討論要點
- 要點一。

## 決議事項
- 本次會議無正式決議

## 行動項目
| 事項 | 負責人 | 期限 |
|------|--------|------|
| 追蹤進度 | 待指派 | 未定 |
"""


def test_score_good_output():
    s = score_output(GOOD)
    assert s == {"sections_ok": True, "table_ok": True, "no_simplified": True}


def test_score_bad_output():
    s = score_output("这是一段没有结构的简体输出")
    assert s == {"sections_ok": False, "table_ok": False, "no_simplified": False}
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL（ImportError）。

- [ ] **Step 3: 實作** `scripts/evaluate.py`：

```python
"""測試集前後對照評估：基底 vs 微調模型。"""
import json
import sys
from pathlib import Path

from validate_data import check_summary_structure, find_simplified_chars, parse_action_items

MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MAX_TOKENS = 1024


def score_output(text: str) -> dict:
    return {
        "sections_ok": check_summary_structure(text) == [],
        "table_ok": len(parse_action_items(text)) > 0,
        "no_simplified": len(find_simplified_chars(text)) == 0,
    }


def run_inference(adapter_path: str | None, cases: list[dict], outdir: Path) -> list[dict]:
    from mlx_lm import load, generate  # 延遲 import，讓單元測試不需要 mlx
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
    n = len(results)
    return {
        "model": name,
        "sections_ok": sum(r["sections_ok"] for r in results),
        "table_ok": sum(r["table_ok"] for r in results),
        "no_simplified": sum(r["no_simplified"] for r in results),
        "total": n,
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
    Path("eval_outputs/summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
```

（注意：`from validate_data import …` 需在 `scripts/` 同層可 import；單元測試從專案根 import 時使用 `scripts.evaluate`，故 `evaluate.py` 頂部 import 需寫成先嘗試 `from scripts.validate_data import …` 再 fallback `from validate_data import …`，實作時以兩種執行方式都能跑為準。）

- [ ] **Step 4: 跑測試確認通過**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: 2 passed。

- [ ] **Step 5: 執行前後對照推論**

Run: `uv run python scripts/evaluate.py both 2>&1 | tee eval_outputs/run.log`
Expected: 產出 `eval_outputs/base/`、`eval_outputs/tuned/` 各 10 個 md + `summary.json`。

- [ ] **Step 6: 質性評分（主 session 執行）**

主 session 逐筆閱讀 10 組（參考摘要 vs base 輸出 vs tuned 輸出），對忠實度、要點覆蓋、行動項目正確性各評 1–5 分，寫入 `docs/eval_report.md`（含程式化指標表、質性分數表、逐筆短評、結論）。

- [ ] **Step 7: 驗收門檻檢查**

門檻（spec §5 階段四）：tuned 的 sections_ok ≥ 9/10、no_simplified = 10/10、質性平均 ≥ base。
未達標 → 依 spec 調整順序：epochs/lr → 資料品質 → 擴充資料，回 Task 7 重訓。

- [ ] **Step 8: Commit**

```bash
git add scripts/evaluate.py tests/test_evaluate.py docs/eval_report.md eval_outputs/summary.json
git commit -m "feat: 前後對照評估與驗收報告"
```

---

### Task 9: 匯出部署（fuse → Ollama）

**Files:**
- Create: `scripts/export.sh`
- Create: `ollama/Modelfile`

- [ ] **Step 1: 寫 Modelfile** `ollama/Modelfile`：

```
FROM ../export/fused

TEMPLATE """{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

SYSTEM """（此處由 export.sh 以 config/system_prompt.txt 內容填入，不手寫）"""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
PARAMETER stop <|im_end|>
```

- [ ] **Step 2: 寫匯出腳本** `scripts/export.sh`：

```bash
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1/3 fuse（合併 LoRA，de-quantize 輸出全精度 HF 權重）==="
uv run mlx_lm.fuse \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --adapter-path adapters \
  --save-path export/fused \
  --de-quantize

echo "=== 2/3 產生 Modelfile（注入 system prompt）==="
python3 - <<'EOF'
from pathlib import Path
template = Path("ollama/Modelfile").read_text(encoding="utf-8")
system = Path("config/system_prompt.txt").read_text(encoding="utf-8").strip()
import re
built = re.sub(r'SYSTEM """.*?"""', f'SYSTEM """{system}"""', template, flags=re.S)
Path("export/Modelfile.built").write_text(built, encoding="utf-8")
print("export/Modelfile.built 已產生")
EOF
# FROM 路徑在 export/ 下需改為相對於 built 檔的位置
sed -i '' 's|FROM ../export/fused|FROM ./fused|' export/Modelfile.built

echo "=== 3/3 Ollama 匯入（內建轉 GGUF + Q4_K_M 量化）==="
ollama create meeting-minutes-zh-tw --quantize q4_K_M -f export/Modelfile.built
ollama list | grep meeting-minutes
echo "完成。測試：ollama run meeting-minutes-zh-tw"
```

Run: `chmod +x scripts/export.sh`

- [ ] **Step 3: 執行匯出**

Run: `scripts/export.sh`
Expected: `ollama list` 出現 `meeting-minutes-zh-tw`。
**備案（Ollama 匯入 safetensors 失敗時）：** `brew install llama.cpp` + `git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llamacpp`，然後 `python /tmp/llamacpp/convert_hf_to_gguf.py export/fused --outfile export/model-f16.gguf`（需 `uv add --dev gguf torch` 或用其 requirements），`llama-quantize export/model-f16.gguf export/model-q4km.gguf Q4_K_M`，Modelfile 的 FROM 改指向 `./model-q4km.gguf` 再 `ollama create`。

- [ ] **Step 4: 端到端煙霧測試（全新逐字稿）**

用一段**不在訓練資料中的**全新繁中會議逐字稿（現寫約 800 字）：
Run: `ollama run meeting-minutes-zh-tw "$(cat /tmp/fresh_transcript.txt)"`
Expected: 結構化四段落輸出、無簡體字、行動項目表格正確。跑 `score_output` 驗證：
`uv run python -c "from scripts.evaluate import score_output; print(score_output(open('/tmp/ollama_out.md').read()))"`

- [ ] **Step 5: Commit**

```bash
git add scripts/export.sh ollama/Modelfile
git commit -m "feat: fuse 與 Ollama 匯出部署腳本"
```

---

### Task 10: README 與收尾

**Files:**
- Create: `README.md`

- [ ] **Step 1: 寫 README**：專案簡介、pipeline 圖（文字版）、各腳本用法（gen_seeds → validate → prepare → train → evaluate → export）、Ollama 使用方式（`ollama run meeting-minutes-zh-tw "<逐字稿>"` 與 API 範例）、評估結果摘要（引用 `docs/eval_report.md` 數字）、重訓指引。

- [ ] **Step 2: 全套測試最終確認**

Run: `uv run pytest -v`
Expected: 全數通過。

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: 專案 README 與使用說明"
```

---

## 執行注意事項

1. **Task 5 與 Task 8 Step 6 由主 session 執行**（資料生成與質性評分需要教師模型品質），其餘任務可派實作 subagent。
2. **外部 CLI 介面以 --help 為準**：mlx-lm 版本迭代快，config 欄位名若不符，以實際 `--help` 輸出修正，不要硬套。
3. **長時任務**（Task 7 Step 6 訓練、Task 9 Step 3 匯出）用背景執行 + log 監看，不要阻塞。
4. **每個 Task 結尾必 commit**，訊息用 conventional commits 繁中描述。
