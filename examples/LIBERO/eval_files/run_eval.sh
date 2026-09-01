#!/bin/bash
# Closed-loop LIBERO evaluation of one StellaVLA checkpoint.
#
# Per GPU: one policy server (which injects the demo) plus one sim worker.
# Servers persist across suites. The policy and the simulator live in SEPARATE
# python envs — LIBERO pins older mujoco/numpy than the policy stack — so both
# interpreters are named explicitly.
#
# Usage:
#   STELLAVLA_PY=<env>/bin/python LIBERO_PY=<env>/bin/python \
#   DEMO_PACK=<pack dir> \
#   bash examples/LIBERO/eval_files/run_eval.sh <CKPT> [SUITES_CSV] [NUM_TRIALS]
#
#   CKPT        .../checkpoints/steps_30000_pytorch_model.pt
#   SUITES_CSV  default libero_spatial,libero_goal,libero_object,libero_10
#   NUM_TRIALS  default 50
#
# Other env: LIBERO_REPO, GPUS_CSV, MAX_SUBGOALS, DEMO_SEED, SERVER_LOAD_WAIT,
#            KV_REUSE, and the ablation switches documented at the bottom.

set -uo pipefail

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
LIBERO_REPO="${LIBERO_REPO:-$CODE_DIR/third_party/libero}"
LIBERO_CONFIG="${LIBERO_CONFIG:-$LIBERO_REPO/libero}"
STELLAVLA_PY="${STELLAVLA_PY:-python}"   # interpreter of the policy env
LIBERO_PY="${LIBERO_PY:-python}"         # interpreter of the simulator env

CKPT_PATH="${1:?usage: $0 <CKPT_PATH> [SUITES_CSV] [NUM_TRIALS]}"
SUITES_CSV="${2:-libero_spatial,libero_goal,libero_object,libero_10}"
NUM_TRIALS="${3:-50}"
IFS=',' read -ra SUITES <<< "$SUITES_CSV"

DEMO_PACK="${DEMO_PACK:-$CODE_DIR/playground/demo_packs/libero}"
MAX_SUBGOALS="${MAX_SUBGOALS:-10}"
DEMO_SEED="${DEMO_SEED:-42}"
SERVER_LOAD_WAIT="${SERVER_LOAD_WAIT:-300}"

# KV_REUSE=1 caches the demo-prefix KV once per episode and prefills only the
# per-step suffix (~5-7x faster VLM forward). The first call verifies against a
# full forward and falls back on mismatch.
KV_REUSE="${KV_REUSE:-0}"
KV_REUSE_FLAG=""
[ "$KV_REUSE" = "1" ] && KV_REUSE_FLAG="--kv_reuse"

# Ablations. NO_CTXDEMO serves without a demo at all; WRONG_DEMO injects another
# task's demo (content-vs-format control); DEMO_STRIP keeps only one modality of
# the demo payload.
NO_CTXDEMO="${NO_CTXDEMO:-0}"
CTXDEMO_FLAG="--use_context_demo"
[ "$NO_CTXDEMO" = "1" ] && CTXDEMO_FLAG=""
WRONG_DEMO_FLAG=""
[ "${WRONG_DEMO:-0}" = "1" ] && WRONG_DEMO_FLAG="--wrong_demo"
DEMO_STRIP="${DEMO_STRIP:-none}"
DEMO_STRIP_FLAG=""
[ "$DEMO_STRIP" != "none" ] && DEMO_STRIP_FLAG="--demo_strip $DEMO_STRIP"

# Base port for this run's servers. Override when another evaluation already
# holds the default range on the same host, or its clients will connect to that
# run's servers and report plausible but wrong numbers.
BASE_PORT="${BASE_PORT:-6500}"

if [ -n "${GPUS_CSV:-}" ]; then IFS=',' read -ra GPUS <<< "$GPUS_CSV"; else GPUS=(0 1 2 3 4 5 6 7); fi
NUM_GPUS=${#GPUS[@]}
NUM_WORKERS=$NUM_GPUS
# Cap per-server CPU threads so N servers on one node do not oversubscribe cores.
OMP_T="${OMP_T:-6}"

[ -f "$CKPT_PATH" ] || { echo "ERROR: ckpt missing: $CKPT_PATH" >&2; exit 2; }
run_root=$(echo "$CKPT_PATH" | awk -F'/(checkpoints|final_model)/' '{print $1}')
RESULTS_ROOT="$run_root/stellavla_eval_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$RESULTS_ROOT/logs"; VIDEO_DIR="$RESULTS_ROOT/videos"
mkdir -p "$LOG_DIR" "$VIDEO_DIR"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR:$LIBERO_REPO:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="$LIBERO_CONFIG"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "=== StellaVLA closed-loop LIBERO eval ==="
echo "  ckpt    : $CKPT_PATH"
echo "  suites  : ${SUITES[*]}  | trials/task: $NUM_TRIALS"
echo "  GPUs    : ${GPUS[*]}  ($NUM_WORKERS workers)"
echo "  demo    : pack=$DEMO_PACK max_subgoals=$MAX_SUBGOALS seed=$DEMO_SEED"
echo "  results : $RESULTS_ROOT"
echo "========================================="

# ── one policy server per GPU (persists across suites) ──────────────────────
SRV_PIDS=()
for i in "${!GPUS[@]}"; do
    gpu=${GPUS[$i]}; port=$((BASE_PORT + i))
    echo "GPU $gpu  policy server :$port"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$OMP_T MKL_NUM_THREADS=$OMP_T OPENBLAS_NUM_THREADS=$OMP_T NUMEXPR_NUM_THREADS=$OMP_T nohup "$STELLAVLA_PY" -u examples/LIBERO/eval_files/serve_stellavla.py \
        --ckpt_path "$CKPT_PATH" --port "$port" --idle_timeout -1 \
        $CTXDEMO_FLAG --demo_pack "$DEMO_PACK" \
        --max_subgoals "$MAX_SUBGOALS" --demo_seed "$DEMO_SEED" \
        $KV_REUSE_FLAG $WRONG_DEMO_FLAG $DEMO_STRIP_FLAG \
        > "$LOG_DIR/policy_server_gpu${gpu}.log" 2>&1 &
    SRV_PIDS+=($!)
done

cleanup() { echo "cleanup servers..."; kill "${SRV_PIDS[@]}" 2>/dev/null || true; pkill -P $$ 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "Waiting ${SERVER_LOAD_WAIT}s for $NUM_GPUS servers to load..."
sleep "$SERVER_LOAD_WAIT"
ss -lnt 2>/dev/null | awk '{print $4}' | grep -E ":$BASE_PORT|:$((BASE_PORT+1))|:$((BASE_PORT+2))" | sort -u | tee "$LOG_DIR/active_ports.txt"

# ── per suite: launch workers, wait, tally ─────────────────────────────────
for SUITE in "${SUITES[@]}"; do
    echo; echo "########## SUITE $SUITE ##########"
    PIDS=()
    for i in "${!GPUS[@]}"; do
        gpu=${GPUS[$i]}; port=$((BASE_PORT + i))
        log="$LOG_DIR/${SUITE}_w$(printf '%02d' $i).log"
        echo "  → worker $i on GPU $gpu (server :$port)"
        ( MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
            LD_LIBRARY_PATH="${EGL_LIB_DIR:-}:${LD_LIBRARY_PATH:-}" \
            CUDA_VISIBLE_DEVICES=$gpu \
            "$LIBERO_PY" examples/LIBERO/eval_files/eval_libero.py \
                --args.pretrained-path "$CKPT_PATH" \
                --args.host 127.0.0.1 --args.port "$port" \
                --args.task-suite-name "$SUITE" \
                --args.num-trials-per-task "$NUM_TRIALS" \
                --args.worker-id "$i" --args.num-workers "$NUM_WORKERS" \
                --args.video-out-path "$VIDEO_DIR/$SUITE" \
            > "$log" 2>&1 ) &
        PIDS+=($!)
        sleep 2
    done
    echo "  $NUM_WORKERS workers launched for $SUITE, waiting…"
    for pid in "${PIDS[@]}"; do wait "$pid" || echo "  worker pid=$pid failed"; done

    # The client writes per-task result files under <video_out>/../logs/<suite>.
    SUMMARY="$LOG_DIR/${SUITE}_summary.txt"; TS=0; TE=0
    for tid in $(seq 0 9); do
        ts=0; te=0
        for f in "$VIDEO_DIR/logs/$SUITE/task$(printf '%02d' $tid)_results_w"*.txt; do
            [ -f "$f" ] || continue
            line=$(tail -1 "$f"); ts=$((ts + $(echo "$line"|cut -d/ -f1))); te=$((te + $(echo "$line"|cut -d/ -f2)))
        done
        TS=$((TS+ts)); TE=$((TE+te))
        echo "  task$(printf '%02d' $tid): $ts/$te" | tee -a "$SUMMARY"
    done
    [ "$TE" -gt 0 ] && echo "TOTAL $SUITE: $TS/$TE = $(awk "BEGIN{printf \"%.1f%%\",100*$TS/$TE}")" | tee -a "$SUMMARY"
done

echo; echo "Results: $RESULTS_ROOT"
