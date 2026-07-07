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
