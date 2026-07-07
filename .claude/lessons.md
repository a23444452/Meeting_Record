# 專案教訓

### L1 — mlx-lm mask_prompt 在 chat 資料集上會造成生成退化
- 情境：QLoRA 微調 Qwen3-4B-Instruct-2507（mlx-lm 0.29.1，chat 格式訓練資料）
- 錯誤做法：config 開 `mask_prompt: true`（想只對 assistant 部分算 loss）
- 正確做法：不開 mask_prompt，用預設全序列 loss
- 原因：mlx-lm 對 chat 資料集的 prompt 遮罩對位有誤——val loss 在同樣錯位的遮罩下假性下降（1.836→0.877），但自由生成隨訓練逐步退化：iter 100 正常、iter 300/500 空輸出（立即 EOS）、iter 600 `<tool_call>` 無限迴圈。診斷方式：用官方 CLI 掛不同 checkpoint 生成做二分定位。
- 來源：2026-07-07，第一次全量訓練（600 iters, ~5h）因此報廢重訓
