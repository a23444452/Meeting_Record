#!/bin/bash
# 匯出流程：fuse（合併 LoRA → 全精度 HF 權重）→ llama.cpp 轉 GGUF → 量化 → Ollama 匯入
# 註：Ollama 0.16 的 safetensors 匯入器不支援 Qwen3ForCausalLM，故走 llama.cpp 轉換。
# 需求：brew install llama.cpp（llama-quantize）；llama.cpp repo 的 convert_hf_to_gguf.py
set -euo pipefail
cd "$(dirname "$0")/.."

# llama.cpp repo 路徑（用於 convert_hf_to_gguf.py）。預設 repo 內 third_party/，
# 可用環境變數覆蓋：LLAMACPP_DIR=/path/to/llama.cpp scripts/export.sh
LLAMACPP="${LLAMACPP_DIR:-./third_party/llama.cpp}"
if [ ! -f "$LLAMACPP/convert_hf_to_gguf.py" ]; then
  echo "找不到 $LLAMACPP/convert_hf_to_gguf.py" >&2
  echo "請 git clone --depth 1 https://github.com/ggml-org/llama.cpp \$LLAMACPP，或設 LLAMACPP_DIR 環境變數" >&2
  exit 1
fi

if [ ! -d export/fused ]; then
  echo "=== 1/4 fuse（合併 LoRA，dequantize 輸出全精度 HF 權重）==="
  uv run mlx_lm.fuse \
    --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
    --adapter-path adapters \
    --save-path export/fused \
    --dequantize
else
  echo "=== 1/4 fuse：export/fused 已存在，略過 ==="
fi

echo "=== 2/4 llama.cpp 轉 GGUF（f16）==="
if [ ! -f export/model-f16.gguf ]; then
  uv run --with gguf --with torch --with sentencepiece \
    python "$LLAMACPP/convert_hf_to_gguf.py" export/fused \
    --outfile export/model-f16.gguf --outtype f16
else
  echo "export/model-f16.gguf 已存在，略過"
fi

echo "=== 3/4 產生 Modelfile ==="
uv run python - <<'EOF'
from pathlib import Path
template = Path("ollama/Modelfile").read_text(encoding="utf-8")
system = Path("config/system_prompt.txt").read_text(encoding="utf-8").strip()
built = template.replace("__SYSTEM_PROMPT__", system)
built = built.replace("FROM ../export/fused", "FROM ./model-f16.gguf")
Path("export/Modelfile.built").write_text(built, encoding="utf-8")
print("export/Modelfile.built 已產生")
EOF

echo "=== 4/4 匯入 Ollama（--quantize 由 Ollama 內部做 Q4_K_M，免裝 llama-quantize）==="
ollama create meeting-minutes-zh-tw --quantize q4_K_M -f export/Modelfile.built
ollama list | grep meeting-minutes
echo "完成。測試：ollama run meeting-minutes-zh-tw \"<貼上會議逐字稿>\""
