# 繁中會議摘要 LoRA 微調專案 — 設計文件

- **日期**：2026-07-06
- **狀態**：已核准（使用者於 brainstorming 階段核准設計方向）
- **目標機器**：Apple M4、24 GB RAM、macOS（Darwin 24.6.0）

## 1. 目標

在本機微調一個「繁體中文（台灣用語）會議摘要專用」小模型：輸入台灣職場會議逐字稿，輸出固定結構的 Markdown 會議紀錄。最終以 Ollama 模型形式部署，任何應用可透過 Ollama API 呼叫。

**成功標準**（微調後 vs 微調前的原始模型，於 10 筆保留測試集上）：

1. 結構合規率顯著提升：輸出恆含四個指定段落與正確的行動項目表格。
2. 輸出語言穩定為繁體中文台灣用語，無簡體字滲入。
3. 質性評分（忠實度、要點覆蓋、行動項目正確性）不低於微調前，且格式穩定性明顯更好。

## 2. 已確認的關鍵決策

| 項目 | 決定 | 理由 |
|------|------|------|
| 訓練框架 | MLX-LM（QLoRA） | Apple Silicon 原生 Metal 加速；24GB 上訓練 4B 模型餘裕充足；PyTorch MPS 訓練支援不穩，雲端方案違背本地初衷 |
| 基底模型 | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | 中文能力強、MLX 生態支援好；Instruct-2507 為 non-thinking 版本，避開 Qwen3 thinking 模式干擾 |
| 訓練資料 | 合成為主，約 300 筆 | 由 Claude（Fable 5）在 session 內以 subagent 批次生成；教師模型品質高、繁中台灣用語道地 |
| 部署目標 | Ollama / GGUF（Q4_K_M） | 最通用，任何 App 可透過 API 呼叫 |
| 輸出格式 | 結構化 Markdown 會議紀錄 | 實用性最高 |

## 3. 輸出格式規格

模型輸出必須恆為以下 Markdown 結構：

```markdown
## 會議主旨
（一到兩句話概括本次會議目的與背景）

## 討論要點
- （條列，每點一個完整議題的討論脈絡）

## 決議事項
- （條列，已定案的結論；無則寫「本次會議無正式決議」）

## 行動項目
| 事項 | 負責人 | 期限 |
|------|--------|------|
| … | … | … |
```

行動項目無明確期限時填「未定」；逐字稿中未指名負責人時填「待指派」。

## 4. 專案結構

```
LoRA_Model/
├── pyproject.toml            # uv 專案；deps: mlx-lm
├── config/
│   └── lora_config.yaml      # 訓練超參數（rank、lr、iters…）
├── data/
│   ├── seeds/scenarios.json  # 300 筆情境種子（程式化生成的多樣性矩陣）
│   ├── raw/                  # Claude 生成的原始資料（batch_*.jsonl）
│   └── processed/            # train.jsonl / valid.jsonl / test.jsonl
├── scripts/
│   ├── gen_seeds.py          # 生成情境種子矩陣
│   ├── validate_data.py      # 資料 schema 與品質檢查
│   ├── prepare_data.py       # 轉 MLX chat 格式 + 切分 270/20/10
│   ├── train.sh              # mlx_lm lora 訓練入口
│   ├── evaluate.py           # 測試集推論 + 結構合規程式化檢查
│   └── export.sh             # fuse → GGUF 轉換 → 量化 → ollama create
├── ollama/
│   └── Modelfile             # system prompt + 推論參數
├── adapters/                 # 訓練產出的 LoRA adapter（gitignore）
├── export/                   # fuse 與 GGUF 產出（gitignore）
└── docs/
    └── superpowers/specs/    # 本設計文件
```

## 5. Pipeline 設計（五階段）

### 階段一：資料生成

1. `gen_seeds.py` 程式化產生 300 筆**情境種子**，維度包含：
   - 產業：科技、製造、電商、醫療、金融、教育、餐飲、行銷代理、新創、非營利等
   - 會議類型：週會、專案檢討、客戶訪談、跨部門協調、緊急事故處理、產品規劃、業績檢討、供應商會議等
   - 與會人數：3–8 人
   - 逐字稿長度級距：短（約 800 字）、中（約 1500 字）、長（約 3000 字）
   - 雜訊特徵：口語贅詞、離題閒聊、互相打斷、中英夾雜（科技業情境）、會議中途有人加入/離開
2. Claude 以 subagent 分批（每批約 10 筆）依種子生成「擬真逐字稿 + 標準摘要」配對，輸出 JSONL 至 `data/raw/`。
3. 每筆資料 schema：

```json
{
  "id": "s001",
  "seed": { "...情境種子欄位..." },
  "transcript": "（逐字稿全文，含發言人標記）",
  "summary": "（符合第 3 節格式的標準摘要）"
}
```

### 階段二：資料驗證與準備

- `validate_data.py` 檢查每筆：欄位齊全、摘要四段落結構完整、行動項目表格語法正確、長度落在級距內、無簡體字、行動項目的負責人名字出現在逐字稿中（對應性抽查）。不合格者列出清單，由 Claude 重新生成該批。
- `prepare_data.py` 轉為 MLX chat 格式（`{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`），使用與 Modelfile 相同的 system prompt；隨機切分 270 train / 20 valid / 10 test（固定 random seed 保證可重現）。

### 階段三：訓練

- 指令：`mlx_lm lora --train`，設定集中於 `config/lora_config.yaml`。
- 起手超參數：LoRA rank 16、alpha 32（scale 2.0）、learning rate 1e-5、batch size 1 + gradient accumulation 4、約 600 iterations（約 8 epochs）、每 50 iters 驗證一次。
- 監看 valid loss：若持續下降則續跑，開始回升即取最佳 checkpoint（防過擬合）。

### 階段四：評估

- `evaluate.py` 對 10 筆測試集分別以**微調前基底模型**與**微調後模型**推論。
- 程式化指標：四段落齊全率、行動項目表格格式正確率、繁中純度（簡體字偵測）。
- 質性評估：Claude 逐筆對照評分（1–5 分）：忠實度（無捏造）、要點覆蓋率、行動項目正確性。
- 產出前後對比報告 `docs/eval_report.md`。
- 合格門檻：結構合規率 ≥ 9/10、無簡體字滲入、質性平均分不低於基底模型。不合格時的調整順序：先調 epochs/lr → 再檢討資料品質 → 最後才擴充資料量。

### 階段五：匯出部署

1. `mlx_lm fuse` 合併 adapter，輸出 HF safetensors（若 4-bit 權重轉檔有相容性問題，備案為 `--de-quantize` 輸出全精度再轉）。
2. llama.cpp `convert_hf_to_gguf.py` 轉 GGUF，量化 Q4_K_M。
3. `ollama/Modelfile`：`FROM` 指向 GGUF、內建 system prompt（使用者只需貼逐字稿）、設定 temperature 等推論參數。
4. `ollama create meeting-minutes-zh-tw -f ollama/Modelfile`，以 2–3 筆全新逐字稿煙霧測試。

## 6. 錯誤處理

- 生成批次不合格：`validate_data.py` 輸出不合格清單與原因，僅重生成不合格筆數，不整批重來。
- 訓練 OOM（機率低）：降 batch 為 1、縮 max sequence length（4096 → 2048）、或降 rank。
- GGUF 轉檔失敗：走 de-quantize 備案；llama.cpp 已支援 Qwen3 架構。
- Ollama 匯入後輸出異常（如模板不符）：檢查 Modelfile 的 TEMPLATE 與 Qwen3 chat template 一致性。

## 7. 測試策略（標準級）

- `validate_data.py`、`prepare_data.py`、`evaluate.py` 的核心邏輯（schema 驗證、格式檢查、簡體字偵測、切分可重現性）附 pytest 單元測試並實際執行。
- 訓練與轉檔屬外部工具呼叫，以煙霧測試驗證（小步數試跑、轉檔後實際推論一筆）。
- 端到端驗收：Ollama 部署後以全新逐字稿實測。

## 8. 不做的事（YAGNI）

- 不做語音轉文字（輸入即是文字逐字稿）。
- 不做 Web UI／API 服務層（Ollama 本身就是 API）。
- 不做多格式輸出切換（只有第 3 節的固定格式）。
- 不先做 1000 筆資料集（300 筆驗證方向後再說）。
- 不做自動化資料擴充迴圈（首版人工判斷評估結果即可）。
