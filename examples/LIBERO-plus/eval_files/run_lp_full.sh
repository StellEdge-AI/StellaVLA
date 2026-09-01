#!/bin/bash
# Full LIBERO-plus zero-shot evaluation on a single multi-GPU host.
#
# One policy server per GPU; the task space is cut into ~300-task shards and
# round-robined over the GPUs. A shard whose result json already exists is
# skipped, so re-running the same command resumes after a crash.
#
# Usage:
#   CODE_DIR=<repo> CKPT=<ckpt.pt> DEMO_PACK=<pack dir> \
#   STELLAVLA_PY=<env>/bin/python LIBERO_PLUS_HOME=<LIBERO-plus repo> \
#   bash examples/LIBERO-plus/eval_files/run_lp_full.sh [OUT_DIR]
#
# Other env: LP_GPUS, LP_THREADS, LP_BASE_PORT, LP_MAX_SUBGOALS, DEMO_SEED,
#            KV_REUSE, DEMO_STRIP, NO_CTXDEMO, LP_SKIP_BROKEN.
set -u
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
cd $CODE_DIR
CKPT=${CKPT:?set CKPT to the checkpoint .pt}
SERVPY=${STELLAVLA_PY:-python}
CLIENT=$CODE_DIR/examples/LIBERO-plus/eval_files/run_lp_client.sh
OUT=${1:-$CODE_DIR/results/libero_plus}
LOGDIR=$OUT/logs
mkdir -p "$LOGDIR"
GPUS=(${LP_GPUS:-0 1 2 3 4 5 6 7})
CHUNK=300
DEMO_PACK="${DEMO_PACK:-$CODE_DIR/playground/demo_packs/libero}"

ABL_FLAGS=""
[ "${KV_REUSE:-0}" = "1" ] && ABL_FLAGS="$ABL_FLAGS --kv_reuse"
[ "${DEMO_STRIP:-none}" != "none" ] && ABL_FLAGS="$ABL_FLAGS --demo_strip ${DEMO_STRIP}"
CTXDEMO_FLAG="--use_context_demo"
[ "${NO_CTXDEMO:-0}" = "1" ] && CTXDEMO_FLAG=""

# Cap per-server CPU threads: N servers on one node must stay well under nproc.
THREADS=${LP_THREADS:-8}
# Base port. run_eval.sh uses 6500 + worker index, so override this when a LIBERO
# run already holds that range on the same host — otherwise the clients here
# would connect to that run's servers and report plausible but wrong numbers.
LP_BASE_PORT=${LP_BASE_PORT:-6500}

# ── one policy server per GPU ───────────────────────────────────────────────
SERVER_PIDS=()
for g in "${GPUS[@]}"; do
  PORT=$((LP_BASE_PORT+g))
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS PYTHONPATH=$CODE_DIR $SERVPY -u examples/LIBERO/eval_files/serve_stellavla.py \
    --ckpt_path "$CKPT" --port $PORT --idle_timeout -1 $CTXDEMO_FLAG \
    --demo_pack "$DEMO_PACK" \
    --max_subgoals "${LP_MAX_SUBGOALS:-10}" --demo_seed "${DEMO_SEED:-42}" $ABL_FLAGS \
    --task_classification "${TASK_CLASSIFICATION:-${LIBERO_PLUS_HOME:-$CODE_DIR/third_party/LIBERO-plus}/libero/libero/benchmark/task_classification.json}" \
    > "$LOGDIR/server_gpu$g.log" 2>&1 &
  SERVER_PIDS+=($!)
done
echo "[lp] launched ${#GPUS[@]} servers, waiting 360s for the VLM to load..."
sleep 360

# ── build the shard list ────────────────────────────────────────────────────
# LP_SUITES restricts the run to a subset, for a spot check rather than the
# full 9,430 episodes.
if [ -n "${LP_SUITES:-}" ]; then IFS=',' read -ra SUITES <<< "$LP_SUITES"
else SUITES=(libero_10 libero_goal libero_object libero_spatial); fi
SIZE_libero_10=2519; SIZE_libero_goal=2591; SIZE_libero_object=2518; SIZE_libero_spatial=2402
SHARDS=()
for suite in "${SUITES[@]}"; do
  eval n=\$SIZE_$suite
  s=0
  while [ $s -lt $n ]; do
    e=$((s+CHUNK)); [ $e -gt $n ] && e=$n
    # libero_10 [1500, 2100) hangs the simulator. With LP_SKIP_BROKEN=1 (the
    # default, and what the reported score was measured with) those 600 tasks
    # are left out, so the run covers 9,430 of the 10,030 episodes. Set
    # LP_SKIP_BROKEN=0 to attempt them.
    if [ "${LP_SKIP_BROKEN:-1}" = "1" ] && [ "$suite" = "libero_10" ] && [ $s -ge 1500 ] && [ $s -lt 2100 ]; then
      echo "[skip] $suite [$s,$e)"; s=$e; continue
    fi
    SHARDS+=("$suite $s $e"); s=$e
  done
done
echo "[lp] ${#SHARDS[@]} shards (~$CHUNK tasks each) across ${#GPUS[@]} GPUs"

# ── round-robin shards to GPUs; sequential per GPU; skip finished shards ────
SHARD_PIDS=()
NGPU=${#GPUS[@]}
for gi in $(seq 0 $((NGPU-1))); do
  (
    GPU=${GPUS[$gi]}; PORT=$((LP_BASE_PORT+GPU)); idx=$gi
    while [ $idx -lt ${#SHARDS[@]} ]; do
      set -- ${SHARDS[$idx]}; suite=$1; st=$2; en=$3
      JSON="$LOGDIR/${suite}_${st}_to_${en}.json"
      if [ -f "$JSON" ]; then echo "[done] $suite [$st,$en)"; idx=$((idx+NGPU)); continue; fi
      echo "[lp] GPU$GPU port$PORT -> $suite [$st,$en)"
      bash "$CLIENT" "$GPU" "$PORT" "$suite" "$st" "$en" "$OUT" "$LOGDIR" \
        > "$LOGDIR/client_gpu${GPU}_${suite}_${st}_${en}.log" 2>&1 || echo "[lp] FAIL GPU$GPU $suite $st $en"
      idx=$((idx+NGPU))
    done
  ) &
  SHARD_PIDS+=($!)
done
wait "${SHARD_PIDS[@]}"
echo "[lp] all shards done $(date +%H:%M:%S)"
for p in "${SERVER_PIDS[@]}"; do kill "$p" 2>/dev/null; done
python $CODE_DIR/examples/LIBERO-plus/eval_files/aggregate_lp.py "$LOGDIR" || true
