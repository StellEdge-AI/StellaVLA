#!/bin/bash
# Dispatch to one of the three benchmarks. Each starts its own policy servers
# and simulator workers inside this container.
#
#   libero      [SUITES_CSV] [NUM_TRIALS]
#   libero-plus [OUT_DIR]
#   vla-arena   [extra args]   ARENA_SUITES=<name> runs just that suite
#   verify                  check every environment and evaluation client
#   fetch-assets            download weights + demo packs into $DATA_DIR
#   shell | bash | <cmd>    run something else
#
# Assets are expected at $DATA_DIR (default /data):
#   $DATA_DIR/checkpoints/{libero,vla-arena}/...
#   $DATA_DIR/demo_packs/{libero,vla-arena}/manifest.json
set -uo pipefail

DATA_DIR="${DATA_DIR:-/data}"
CKPT_DIR="${CKPT_DIR:-$DATA_DIR/checkpoints}"
PACK_DIR="${PACK_DIR:-$DATA_DIR/demo_packs}"
RESULTS_DIR="${RESULTS_DIR:-$DATA_DIR/results}"
HF_REPO="${HF_REPO:-StellarEdge/StellaVLA}"
cmd="${1:-help}"; shift || true

# The checkpoint config points at playground/Pretrained_models/<backbone>. When
# the backbone is mounted under $DATA_DIR instead, name it explicitly.
if [ -d "$DATA_DIR/Qwen3-VL-4B-Instruct" ] && [ -z "${STELLAVLA_BASE_VLM_OVERRIDE:-}" ]; then
  export STELLAVLA_BASE_VLM_OVERRIDE="$DATA_DIR/Qwen3-VL-4B-Instruct"
fi

need() {
  [ -e "$1" ] || { echo "ERROR: missing $1" >&2
                   echo "       run 'fetch-assets' first, or mount it into the container." >&2
                   exit 2; }
}

case "$cmd" in
  fetch-assets)
    set -e
    mkdir -p "$CKPT_DIR" "$PACK_DIR"
    "$STELLAVLA_PY" -m pip install -q -U "huggingface-hub>=0.35"
    for b in libero vla-arena; do
      "/opt/conda/envs/stellavla/bin/hf" download "$HF_REPO" --include "$b/*" --local-dir "$CKPT_DIR"
      mkdir -p "$PACK_DIR/$b"
      tar -xf "$CKPT_DIR/$b/demo/demo_pack.tar" -C "$PACK_DIR/$b" --strip-components=1
    done
    echo "Qwen3-VL backbone:"
    "/opt/conda/envs/stellavla/bin/hf" download Qwen/Qwen3-VL-4B-Instruct \
        --local-dir "$DATA_DIR/Qwen3-VL-4B-Instruct"
    # `hf download` can leave a partial tree behind without failing, and the
    # gap then surfaces much later as the policy server refusing to load.
    # Check the files the three benchmarks actually open.
    missing=0
    for f in "$CKPT_DIR/libero/checkpoints/steps_30000_pytorch_model.pt" \
             "$CKPT_DIR/vla-arena/checkpoints/steps_30000_pytorch_model.pt" \
             "$CKPT_DIR/libero/config.yaml" \
             "$CKPT_DIR/vla-arena/config.yaml" \
             "$PACK_DIR/libero/manifest.json" \
             "$PACK_DIR/vla-arena/manifest.json"; do
      [ -s "$f" ] || { echo "missing or empty: $f" >&2; missing=1; }
    done
    shards=$(ls "$DATA_DIR/Qwen3-VL-4B-Instruct"/*.safetensors 2>/dev/null | wc -l)
    if [ "$shards" -lt 2 ]; then
      echo "missing backbone weights: found $shards .safetensors, expected 2" >&2
      missing=1
    fi
    if [ "$missing" -ne 0 ]; then
      echo "fetch-assets did not complete; re-run it before evaluating" >&2
      exit 1
    fi
    echo "assets ready under $DATA_DIR"
    ;;

  libero)
    need "$CKPT_DIR/libero/checkpoints/steps_30000_pytorch_model.pt"
    need "$PACK_DIR/libero/manifest.json"
    export DEMO_PACK="$PACK_DIR/libero"
    export LIBERO_CONFIG=/app/libero_config
    export GPUS_CSV="${GPUS_CSV:-0}"
    export KV_REUSE="${KV_REUSE:-1}"
    exec bash /workspace/examples/LIBERO/eval_files/run_eval.sh \
        "$CKPT_DIR/libero/checkpoints/steps_30000_pytorch_model.pt" \
        "${1:-libero_spatial,libero_object,libero_goal,libero_10}" "${2:-50}"
    ;;

  libero-plus)
    need "$CKPT_DIR/libero/checkpoints/steps_30000_pytorch_model.pt"
    need "$PACK_DIR/libero/manifest.json"
    # The LIBERO-plus client runs in its own env; run_lp_client.sh activates it
    # through conda, so name it rather than passing an interpreter path.
    export CKPT="$CKPT_DIR/libero/checkpoints/steps_30000_pytorch_model.pt"
    export DEMO_PACK="$PACK_DIR/libero"
    export LP_CONDA_ENV=libero_plus
    export LP_GPUS="${LP_GPUS:-0}"
    export KV_REUSE="${KV_REUSE:-1}"
    exec bash /workspace/examples/LIBERO-plus/eval_files/run_lp_full.sh \
        "${1:-$RESULTS_DIR/libero_plus}"
    ;;

  vla-arena)
    need "$CKPT_DIR/vla-arena/checkpoints/steps_30000_pytorch_model.pt"
    need "$PACK_DIR/vla-arena/manifest.json"
    export CKPT="$CKPT_DIR/vla-arena/checkpoints/steps_30000_pytorch_model.pt"
    export DEMO_PACK="$PACK_DIR/vla-arena"
    export your_ckpt="$CKPT"
    export stellavla_python="$STELLAVLA_PY"
    # eval_vla_arena.sh writes to ./results relative to the repo root, which is
    # inside the image and would be lost when the container exits.
    mkdir -p "$RESULTS_DIR/vla_arena"
    ln -sfn "$RESULTS_DIR/vla_arena" /workspace/results
    cd /workspace
    if [ -n "${ARENA_SUITES:-}" ]; then
      # Spot-check path: one server, one (or a few) suites. run_parallel_eval.sh
      # splits all eleven suites over four GPUs, which is more than a check needs.
      gpu="${ARENA_GPU:-0}"; port=$((10090 + gpu))
      CUDA_VISIBLE_DEVICES="$gpu" "$STELLAVLA_PY" \
        examples/VLA-Arena/eval_files/serve_stellavla.py \
        --ckpt_path "$CKPT" --port "$port" --idle_timeout -1 \
        --use_context_demo --demo_pack "$DEMO_PACK" --max_subgoals 10 \
        > "$RESULTS_DIR/arena_server_gpu${gpu}.log" 2>&1 &
      srv=$!
      trap 'kill $srv 2>/dev/null' EXIT INT TERM
      echo "[arena] server on :$port, waiting ${SERVER_LOAD_WAIT:-420}s for the VLM to load"
      sleep "${SERVER_LOAD_WAIT:-420}"
        # The simulator renders through EGL, which ignores the server's
        # CUDA_VISIBLE_DEVICES and would otherwise put every client on GPU 0.
        export MUJOCO_EGL_DEVICE_ID="$gpu"
      exec bash examples/VLA-Arena/eval_files/eval_vla_arena.sh \
        -c "$CKPT" --port "$port" --suites "$ARENA_SUITES" "$@"
    fi
    exec bash /workspace/examples/VLA-Arena/eval_files/run_parallel_eval.sh \
        --vla-arena-env "$VLA_ARENA_ENV" -c "$CKPT" "$@"
    ;;

  verify) exec bash /workspace/docker/verify.sh ;;
  shell|bash) exec /bin/bash "$@" ;;
  help|-h|--help)
    sed -n '2,17p' "$0" | sed 's/^# \?//'
    ;;
  *) exec "$cmd" "$@" ;;
esac
