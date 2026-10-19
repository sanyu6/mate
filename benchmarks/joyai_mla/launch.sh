#!/usr/bin/env bash
set -euo pipefail
mode=${1:?default, optimized or validate}
shift
: "${JOYAI_MODEL_PATH:?Set JOYAI_MODEL_PATH to the local model directory}"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MUSA_VISIBLE_DEVICES=${JOYAI_GPUS:-4,5}
export SGLANG_USE_MUSA_FUSED_KERNEL=1
export SGLANG_MOE_PADDING=1
export VLLM_PATCH_MUSA_CUSTOM_OPS=1
export SGLANG_MUSA_CONTIG_DEEPGEMM_TOKEN_THRESHOLD=99999999
if [[ -n ${JOYAI_SGLANG_SOURCE:-} ]]; then
  export PYTHONPATH="${JOYAI_SGLANG_SOURCE}/python${PYTHONPATH:+:${PYTHONPATH}}"
fi
exec "${JOYAI_PYTHON:-python}" "${script_dir}/run_server.py" --mla-mode "$mode" \
  --model-path "$JOYAI_MODEL_PATH" --served-model-name joyai-tp2 \
  --tp-size 2 --attention-backend fa3 --moe-runner-backend triton \
  --dtype bfloat16 --mem-fraction-static 0.85 --cuda-graph-bs 1 2 3 4 5 \
  --disable-piecewise-cuda-graph --max-prefill-tokens 32768 \
  --chunked-prefill-size -1 --num-continuous-decode-steps 2 --page-size 64 \
  --trust-remote-code --host 127.0.0.1 --port "${JOYAI_PORT:-38642}" \
  --watchdog-timeout 1200 --random-seed 801100110 "$@"
