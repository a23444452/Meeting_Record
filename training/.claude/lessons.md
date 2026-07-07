# 專案教訓

### L1 — Qwen3-2507 微調後推論 prompt 必須補空 think 區塊（訓練/推論分布對齊）
- 情境：QLoRA 微調 Qwen3-4B-Instruct-2507（mlx-lm 0.29.1，chat 格式訓練資料）
- 症狀：微調後模型生成退化——空輸出（立即 EOS）、`<|im_start|>assistant` 前綴、`<tool_call>` 迴圈；但 val loss 正常下降（假性健康）
- 根因：Qwen3-2507 chat template 渲染**對話歷史**時，每個 assistant 內容前恆有 `<think>\n\n</think>\n\n` 空區塊（訓練序列如此），但 `add_generation_prompt=True` 的推論 prompt 只到 `<|im_start|>assistant\n`。模型學到「assistant 標頭後必接 think 區塊」，推論條件不符 → adapter 與基底分布打架
- 正確做法：所有推論路徑（evaluate.py、Ollama Modelfile TEMPLATE）在 generation prompt 尾端補 `<think>\n\n</think>\n\n`——這也是 Qwen3 官方 Ollama template 的做法
- 診斷方式：token 級比對 `apply_chat_template(msgs)`（訓練渲染）與 `apply_chat_template(msgs[:2], add_generation_prompt=True)`（推論渲染）的分歧點；checkpoint 二分定位
- 附帶：`mask_prompt: true` 在此情境下讓退化更嚴重（首次訓練 iter 600 完全崩壞），保持關閉
- 來源：2026-07-07/08，兩次全量訓練 debug（第一次報廢，第二次靠推論側修復救回）
