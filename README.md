# meeting-record

公司內部的本地會議記錄工具：把 Teams 會議錄影（或任何音訊／影片檔）轉成**繁體中文逐字稿**與**結構化會議紀錄**，全程在本機處理，資料不出內網。

架構參考 [Meetily](https://github.com/Zackriya-Solutions/meetily)（MIT），但採事後批次處理路線：

```
Teams 錄影 (mp4) → ffmpeg 抽音訊 → faster-whisper 本地轉錄 → Ollama/內部 LLM 生成會議紀錄
                                          ↓                        ↓
                                    transcript.md/.json        summary.md
```

> **模型訓練層**：[`training/`](training/) 子專案是本工具摘要模型的 LoRA 微調 pipeline，
> 用合成的繁中會議資料微調 Qwen3-4B，產出 Ollama 模型 `meeting-minutes-zh-tw`——
> 可取代預設的 `qwen3:8b` 作為專用摘要模型。詳見 [training/README.md](training/README.md)。

## 需求

- Python 3.12+、[uv](https://docs.astral.sh/uv/)
- ffmpeg（macOS：`brew install ffmpeg`）
- [Ollama](https://ollama.com)（本地摘要）：`ollama pull qwen3:8b`
  - 或改用公司內部 OpenAI-compatible endpoint（見下方）

## 安裝

```bash
git clone <repo-url> && cd Meeting_Record
uv sync
```

## 使用

### 網頁介面（推薦給不熟終端機的同事）

```bash
uv run meeting-record gui
```

會自動開啟瀏覽器（http://127.0.0.1:8765），四個分頁：

- **處理會議**：拖放錄影檔、選模板／模型／講者辨識 → 即時顯示四階段進度與逐段轉錄
- **歷史會議**：瀏覽處理過的會議，閱讀逐字稿與摘要，一鍵換模板重新生成摘要
- **模板管理**：表單編輯自訂摘要模板，不用手寫 JSON
- **設定**：預設值與環境狀態檢查（Ollama 連線、HF token、講者辨識安裝狀態）

介面只綁定本機（127.0.0.1），資料同樣不出內網。頁面重新整理不會中斷處理中的任務。

### CLI：處理一場會議

```bash
uv run meeting-record process ~/Downloads/teams_meeting.mp4
```

輸出到 `output/<日期>_<檔名>/`：

- `transcript.md` — 含時間戳的逐字稿（給人看）
- `transcript.json` — 結構化逐字稿（重跑摘要用）
- `summary.md` — 會議紀錄（摘要／決議／待辦表格／討論重點）

常用選項：

```bash
--template project_sync      # 換模板（meeting-record templates 列出全部）
--whisper-model small        # 快速測試用；預設 large-v3 品質最好（首次下載約 3GB）
--skip-summary               # 只轉錄不摘要
--llm-model qwen3:8b         # 摘要用的 Ollama 模型
```

### 換模板或模型重新生成摘要（不重跑轉錄）

```bash
uv run meeting-record resummarize output/2026-07-05_weekly/transcript.json -t project_sync
```

注意：`resummarize` 會覆寫同資料夾的 `summary.md`。

### 使用公司內部 LLM endpoint

任何 OpenAI-compatible API 都可以：

```bash
export MEETING_RECORD_API_KEY=<key>   # 若需要
uv run meeting-record process meeting.mp4 \
  --provider openai \
  --llm-url https://llm.internal.company.com \
  --llm-model <內部模型名>
```

### 講者辨識（誰說的）

用 pyannote `speaker-diarization-community-1` 把段落標成「講者1／講者2…」，是選配功能：

```bash
uv sync --extra diarize        # 安裝（會拉 torch，約需數 GB）
```

模型免費但需要 HuggingFace token：

1. 註冊 [huggingface.co](https://huggingface.co)，並在
   [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   模型頁按「Agree and access」
2. 到 [Settings → Tokens](https://huggingface.co/settings/tokens) 建立 Read token
3. `export HF_TOKEN=<token>`

```bash
uv run meeting-record process meeting.mp4 --diarize                  # 自動判斷人數
uv run meeting-record process meeting.mp4 --diarize --num-speakers 4 # 已知人數更準
```

逐字稿會變成 `[00:01:23] 講者2: 這個由我來負責`，摘要的待辦事項也能對到發言人。

## 自訂摘要模板

模板是 JSON（格式參考 Meetily），內建於 `src/meeting_record/templates/`。也可以直接指定自己的檔案：

```bash
uv run meeting-record process meeting.mp4 -t /path/to/my_template.json
```

```json
{
  "name": "模板名稱",
  "description": "說明",
  "sections": [
    { "title": "章節標題", "instruction": "給 LLM 的指示", "format": "paragraph|list|table" }
  ]
}
```

## 中文品質建議

- 正式使用請用預設的 `large-v3`（`small`/`base` 錯字明顯較多）
- 轉錄時已帶繁體中文 initial prompt（`transcribe.py`），引導 Whisper 輸出台灣用語
- 摘要模型建議 `qwen3:8b`（中文佳、24GB RAM 可跑）；更高品質可換 `qwen3:14b`

## 開發

```bash
uv run pytest -q            # 測試
uv run ruff check src tests # lint
```

## Windows 安裝與使用指南

以下步驟在 **PowerShell**（Windows 10/11 內建）操作，指令與 macOS/Linux 的 bash 語法略有不同（用 `$env:變數` 取代 `export`）。

### 1. 安裝必要工具

**uv**（Python 套件與版本管理工具）：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> 若出現「因為這個系統上已停用指令碼執行」的錯誤，代表 PowerShell 執行原則擋住了安裝腳本；上面指令已用 `-ExecutionPolicy ByPass` 繞過，不需另外調整系統設定。

安裝完成後**重新開啟一個新的 PowerShell 視窗**，確認：

```powershell
uv --version
```

**ffmpeg**（音訊擷取用）：

```powershell
winget install ffmpeg
```

裝完同樣要開新視窗，用 `ffmpeg -version` 確認能找到指令；若找不到，多半是 PATH 還沒刷新，登出重新登入 Windows 或重開機即可。

**Ollama**（本地 LLM 摘要）：到 [ollama.com/download](https://ollama.com/download) 下載 Windows 安裝檔並執行，裝好後在 PowerShell 拉模型：

```powershell
ollama pull qwen3:8b
```

### 2. 下載專案並安裝相依套件

```powershell
git clone https://github.com/a23444452/Meeting_Record.git
cd Meeting_Record
uv sync
```

若機器沒裝 Git，也可以直接到 GitHub 頁面按「Code → Download ZIP」解壓縮後 `cd` 進去再執行 `uv sync`。

### 3. 執行

Windows 路徑通常含反斜線與空白，記得用**雙引號**包住檔案路徑：

```powershell
uv run meeting-record process "C:\Users\vince\Videos\teams_meeting.mp4"
```

其餘選項與 macOS 完全相同（`--template`、`--whisper-model`、`--diarize` 等），輸出同樣在專案目錄下的 `output\` 資料夾。

### 4. 設定環境變數（HF_TOKEN、內部 LLM API Key）

PowerShell 用 `$env:變數名` 讀寫環境變數，語法和 bash 的 `export` 不同：

**只在目前這個視窗有效**（關掉視窗就消失，適合先測試）：

```powershell
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
$env:MEETING_RECORD_API_KEY = "your-key"
```

**永久保存**（設定一次，之後每次開新視窗都會自動帶入，等同 macOS 寫進 `~/.zshrc`）：

```powershell
setx HF_TOKEN "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

`setx` 設定後**必須開新的 PowerShell 視窗**才會生效，當前視窗不會立即套用。也可以在「開始功能表」搜尋「編輯系統環境變數」用圖形介面新增，效果相同。

### 5. GPU 加速（有 NVIDIA 顯卡時，選用）

faster-whisper 會自動偵測可用的 NVIDIA GPU；沒有 GPU 或沒裝 CUDA 也完全能跑，只是用 CPU（int8）處理，長會議轉錄時間會拉長。要啟用 GPU 加速需另外安裝對應版本的 [CUDA Toolkit 與 cuDNN](https://developer.nvidia.com/cuda-downloads)，公司機器若無顯卡可略過此步驟。

### 6. 常見問題

| 現象 | 原因與解法 |
|---|---|
| `uv : 無法辨識...` | 安裝後沒開新視窗，PATH 還沒刷新；重開 PowerShell 或登出重登 |
| `ffmpeg 轉檔失敗` / 找不到 ffmpeg | 同上，或改用系統管理員身分重跑 `winget install ffmpeg` |
| PowerShell 顯示「已停用指令碼執行」 | 只在安裝 uv 那個指令用到，照上面指令加 `-ExecutionPolicy ByPass` 即可，不需更改系統原則 |
| 中文路徑或檔名亂碼 | 建議在「設定 → 時間與語言 → 語言與地區 → 系統管理語言設定」勾選「使用 Unicode UTF-8」 |
| 連不上 Ollama（`http://localhost:11434`）| 確認 Ollama 應用程式（工作列圖示）有在執行中 |

## 已知限制／後續方向

- 尚無批次資料夾處理與 Teams 自動抓檔
- 逐字稿與摘要都存在 `output/`（已 gitignore），內含會議內容請勿提交版控
