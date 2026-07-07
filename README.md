# 繁中會議摘要 LoRA 微調專案

在 Apple Silicon（M4 / 24GB）本機以 MLX-LM QLoRA 微調 **Qwen3-4B**，打造繁體中文（台灣用語）會議摘要專用小模型：輸入會議逐字稿，輸出固定結構的 Markdown 會議紀錄，最終以 Ollama 部署。

## 輸出格式

模型固定輸出四段落結構：

```markdown
## 會議主旨
（一到兩句話概括會議目的與背景）

## 討論要點
- （條列討論脈絡）

## 決議事項
- （已定案的結論；無則寫「本次會議無正式決議」）

## 行動項目
| 事項 | 負責人 | 期限 |
|------|--------|------|
```

範例輸入輸出見 [docs/sample_transcript.txt](docs/sample_transcript.txt) 與 [docs/sample_output.md](docs/sample_output.md)。

## 使用方式（部署後）

```bash
# 直接對談
ollama run meeting-minutes-zh-tw "$(cat 你的逐字稿.txt)"

# API 呼叫
curl http://localhost:11434/api/generate -d '{
  "model": "meeting-minutes-zh-tw",
  "prompt": "主管：我們開始開會⋯⋯",
  "stream": false
}'
```

System prompt 已內建於模型（[ollama/Modelfile](ollama/Modelfile)），使用者只需貼上逐字稿。

## Pipeline（從零重建）

```bash
uv sync                              # 安裝依賴（mlx-lm、pyyaml、pytest）

uv run python scripts/gen_seeds.py   # 1. 生成 300 筆情境種子矩陣
# 2. 合成資料：依 config/gen_prompt_template.md 用大模型逐批生成到 data/raw/*.jsonl
uv run python scripts/validate_data.py   # 3. 驗證（結構/簡體字/負責人對應性）
uv run python scripts/prepare_data.py    # 4. 轉 MLX chat 格式，切分 270/20/10
scripts/train.sh smoke               # 5a. 煙霧測試（10 iters）
scripts/train.sh                     # 5b. 全量訓練（600 iters，約 4–5 小時）
uv run python -m scripts.evaluate both   # 6. 前後對照評估（base vs tuned）
scripts/export.sh                    # 7. fuse → GGUF → Ollama 匯入
```

## 專案結構

| 路徑 | 內容 |
|------|------|
| `config/system_prompt.txt` | 唯一 system prompt 來源（訓練與部署共用） |
| `config/gen_prompt_template.md` | 資料生成規格 |
| `config/lora_config.yaml` | MLX QLoRA 訓練超參數 |
| `scripts/gen_seeds.py` | 確定性情境種子生成器 |
| `scripts/validate_data.py` | 資料品質驗證器（可 import + CLI） |
| `scripts/prepare_data.py` | raw → MLX chat 格式 + 切分 |
| `scripts/evaluate.py` | 前後對照評估 + 程式化指標 |
| `scripts/export.sh` | fuse + Ollama 匯出 |
| `ollama/Modelfile` | 部署定義（system prompt、TEMPLATE、推論參數） |
| `data/raw/` | 300 筆合成逐字稿+摘要（已入庫） |
| `data/processed/` | train/valid/test 切分 |
| `docs/eval_report.md` | 微調前後評估報告 |
| `tests/` | 純邏輯單元測試（19 個，`uv run pytest`） |

## 訓練配置

- **基底模型**：`mlx-community/Qwen3-4B-Instruct-2507-4bit`
- **方法**：QLoRA，rank 16、scale 2.0、dropout 0.05，注意力層 q/k/v/o
- **超參數**：lr 1e-5、batch 1 + grad accum 4、600 iters（約 8 epochs）、max_seq 4096
- **資源**：訓練峰值記憶體 7.6 GB，推論 3.3 GB
- Val loss：2.809 → 1.889（穩定收斂，無過擬合）

## 訓練資料

300 筆合成會議資料，涵蓋 10 產業 × 8 會議類型的矩陣，逐字稿含口語贅詞、打斷、離題、中英夾雜等擬真雜訊，長度 short/medium/long 三級距。每筆均通過 `validate_data.py` 的結構、簡體字、行動項目負責人對應性檢查。

## 評估結果

測試集 10 筆，程式化指標（四段落結構、行動項目表格、無簡體字）微調前後皆 10/10。質性評分（忠實度、要點覆蓋、行動項目正確性）詳見 [docs/eval_report.md](docs/eval_report.md)。

**已知限制**：目前 adapter（600 iters）學到了目標的精簡風格與穩定版式，但在部分案例會出現負責人姓名誤植（如在名字前多加姓氏）與偶發用字錯誤（如「麵包」寫成「面包」）。改善方向：延長訓練、強化行動項目對齊的訓練樣本、加負責人/期限的後驗證。踩過的坑記錄於 [.claude/lessons.md](.claude/lessons.md)（Qwen3 推論需補空 think 區塊、mask_prompt 缺陷等）。

## 重訓指引

調整 `config/lora_config.yaml` 後重跑 `scripts/train.sh`。若要從既有 checkpoint 續訓：

```bash
uv run mlx_lm.lora -c config/lora_config.yaml \
  --resume-adapter-file adapters/adapters.safetensors
```

評估發現簡體字漏網時，補進 `scripts/validate_data.py` 的 `SIMPLIFIED_CHARS` 並加回歸測試。
