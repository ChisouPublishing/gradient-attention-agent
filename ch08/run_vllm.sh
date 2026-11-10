#!/bin/sh
# 8 章の vLLM のコードを、vLLM の CPU 版の Docker イメージ（Linux arm64）で実行する。
#   sh ch08/run_vllm.sh [モデル名] [allow_ws]   （allow_ws：本文の disable_any_whitespace を外して比べる）
# GPU は使わない。Apple Silicon の Mac では OrbStack か Docker Desktop の Linux（arm64）で動く。
# x86 の機械では、イメージの名前の -arm64 を外す。
set -e
CODE=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${1:-Qwen/Qwen2.5-0.5B-Instruct}
IMAGE=${VLLM_IMAGE:-vllm/vllm-openai-cpu:v0.33.0-arm64}
docker run --rm \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$CODE:/code" -w /code \
  -e VLLM_CPU_KVCACHE_SPACE=4 -e VLLM_LOGGING_LEVEL=WARNING \
  --entrypoint python3 "$IMAGE" ch08/run_vllm.py "$MODEL" ${2:-}
