#!/usr/bin/env bash
set -euo pipefail
# Launch one StellaVLA policy server for VLA-Arena evaluation (one process per
# GPU, bound to 10090 + GPU). The server injects the same-task context demo the
# model was trained with, resolved from the released demo pack.
#
# Usage:
#   CKPT=<ckpt.pt> DEMO_PACK=<pack dir> \
#   bash examples/VLA-Arena/eval_files/run_policy_server.sh [GPU_ID]
#
# then point eval_vla_arena.py at --args.port $((10090 + GPU_ID)).
#
# For the no-demo ablation, pass --no_demo instead of --use_context_demo.

STELLAVLA_DIR="${STELLAVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
stellavla_python="${stellavla_python:-python}"
CKPT="${CKPT:?set CKPT to the checkpoint .pt}"
DEMO_PACK="${DEMO_PACK:-${STELLAVLA_DIR}/playground/demo_packs/vla-arena}"
MAX_SUBGOALS="${MAX_SUBGOALS:-10}"

GPU=${1:-0}
PORT=$((10090 + GPU))

cd "${STELLAVLA_DIR}"
export PYTHONPATH="${STELLAVLA_DIR}:${PYTHONPATH:-}"

echo "[vla-arena server] GPU=${GPU} PORT=${PORT} PACK=${DEMO_PACK}"
CUDA_VISIBLE_DEVICES="${GPU}" "${stellavla_python}" \
  examples/VLA-Arena/eval_files/serve_stellavla.py \
  --ckpt_path "${CKPT}" \
  --port "${PORT}" \
  --use_context_demo \
  --demo_pack "${DEMO_PACK}" \
  --max_subgoals "${MAX_SUBGOALS}"
