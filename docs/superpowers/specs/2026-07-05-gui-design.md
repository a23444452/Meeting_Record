# meeting-record GUI 設計

日期：2026-07-05｜狀態：已核准（使用者確認）

## 目的

為 meeting-record CLI 提供使用者友善的本地網頁介面：選擇功能與設定、即時查看處理狀態，讓不熟終端機的同事也能使用。單人本機情境，資料不出內網。

## 技術選型

- **後端**：FastAPI + uvicorn（與現有 pipeline 同棧，直接呼叫既有函式）
- **前端**：單一靜態 `index.html`（vanilla JS + CSS），無 Node build step
- **即時進度**：Server-Sent Events（SSE），瀏覽器原生 `EventSource`
- **Markdown 渲染**：伺服器端 `markdown` 套件（離線可用，不依賴 CDN）
- 新增主依賴：`fastapi`、`uvicorn`、`python-multipart`、`markdown`

## 架構

```
src/meeting_record/gui/
├── __init__.py
├── jobs.py        # 單任務狀態機 + 事件串流
├── server.py      # FastAPI：REST API + SSE + 靜態檔
├── settings.py    # gui_settings.json 載入／儲存
└── static/
    └── index.html # 單頁前端（四分頁）
```

啟動：`meeting-record gui`（cli.py 新增指令）→ uvicorn 起 `127.0.0.1:8765` → 自動開瀏覽器。

現有 `extract/transcribe/diarize/summarize` 函式**完全不動**；GUI 是另一個呼叫入口，`transcribe` 的 `on_segment` callback 用來推逐段進度。

## 任務狀態機（jobs.py）

單人本機 → 同時只跑一個任務，第二個請求回 409。

```
idle → extracting → transcribing → (diarizing) → summarizing → done
                                                             ↘ error（任一階段失敗）
```

- `JobManager.start(params, runner)`：背景 thread 執行 pipeline；`runner` 可注入（測試用 mock）
- 事件累積在 list（帶序號），SSE 從客戶端指定的序號往後送 → 斷線重連不掉事件
- 事件型別：`state`（階段變更）、`segment`（逐段轉錄文字）、`chunk_progress`（長會議分段摘要 i/n）、`done`（含輸出路徑）、`error`（友善錯誤訊息，沿用現有 KNOWN_ERRORS）
- `resummarize` 也走同一個 JobManager（狀態只有 summarizing → done/error）

## API

| Method | Path | 說明 |
|---|---|---|
| GET | `/` | index.html |
| POST | `/api/jobs` | multipart：檔案 + 選項 → 啟動任務（執行中回 409） |
| GET | `/api/jobs/current` | 目前任務快照（含全部事件，供刷新後恢復畫面） |
| GET | `/api/jobs/events?after=N` | SSE 事件串流 |
| GET | `/api/meetings` | 列 output/ 下的會議（名稱、時間、有無摘要） |
| GET | `/api/meetings/{name}` | 逐字稿與摘要（Markdown 轉 HTML） |
| POST | `/api/meetings/{name}/resummarize` | 換模板／模型重生成摘要 |
| GET/POST | `/api/templates` | 列出（內建＋自訂）／新增自訂模板 |
| PUT/DELETE | `/api/templates/{name}` | 修改／刪除（僅自訂；內建唯讀） |
| GET/PUT | `/api/settings` | GUI 預設值（gui_settings.json，gitignored） |

路徑安全：`{name}` 一律驗證為單層目錄名（拒絕 `/`、`..`），限制在 output/ 與 templates/ 內。

## 前端（四分頁，全繁中）

1. **處理會議**：選檔上傳 + 選項（模板、Whisper 模型、講者辨識＋人數、LLM 設定，預設值來自設定頁）→ 四階段狀態列 + 轉錄逐段即時滾動 + 錯誤紅字顯示 → 完成後連到結果
2. **歷史會議**：會議列表 → 點開讀逐字稿／摘要（渲染後 HTML）→「重新生成摘要」按鈕（選模板／模型）
3. **模板管理**：卡片列表（內建標示唯讀）；表單編輯：名稱、描述、章節（標題／指示／格式），存成自訂模板 JSON 至 `templates/`
4. **設定**：各項預設值 + 環境狀態偵測（HF token 有無、Ollama 連線測試）

## 測試策略

- 後端：pytest + FastAPI TestClient，pipeline 以 mock runner 注入（不跑真 whisper/LLM）——狀態機轉移、事件序號續傳、409 併發拒絕、meetings 列表／閱讀、模板 CRUD 與內建保護、設定 roundtrip、路徑注入防護
- 端到端：啟動真實伺服器，用既有測試音訊（dialogue2.wav）走完整流程，驗證 SSE 與輸出

## 不在本次範圍

- 多人共用伺服器（任務佇列、認證）
- 批次多檔處理
- 舊會議的音訊播放器
