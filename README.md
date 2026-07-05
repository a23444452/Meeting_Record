# meeting-record

公司內部的本地會議記錄工具：把 Teams 會議錄影（或任何音訊／影片檔）轉成**繁體中文逐字稿**與**結構化會議紀錄**，全程在本機處理，資料不出內網。

架構參考 [Meetily](https://github.com/Zackriya-Solutions/meetily)（MIT），但採事後批次處理路線：

```
Teams 錄影 (mp4) → ffmpeg 抽音訊 → faster-whisper 本地轉錄 → Ollama/內部 LLM 生成會議紀錄
                                          ↓                        ↓
                                    transcript.md/.json        summary.md
```

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

### 處理一場會議

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

## Windows 同事部署注意

- 安裝 [uv](https://docs.astral.sh/uv/getting-started/installation/) 與 [ffmpeg](https://ffmpeg.org/download.html)（或 `winget install ffmpeg`）
- faster-whisper 在 CPU 即可跑；有 NVIDIA GPU 時裝 CUDA 版 ctranslate2 可大幅加速
- 其餘指令相同

## 已知限制／後續方向

- 尚無批次資料夾處理與 Teams 自動抓檔
- 逐字稿與摘要都存在 `output/`（已 gitignore），內含會議內容請勿提交版控
