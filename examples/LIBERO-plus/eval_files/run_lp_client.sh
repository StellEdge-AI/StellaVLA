#!/bin/bash
# One LIBERO-plus shard (a contiguous task range of one suite).
# Usage: run_lp_client.sh <GPU> <PORT> <SUITE> <START> <END> <OUT> <LOGDIR>
set -e
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate ${LP_CONDA_ENV:-libero_plus}
cd $CODE_DIR
export LIBERO_HOME=${LIBERO_PLUS_HOME:-$CODE_DIR/third_party/LIBERO-plus}
export LIBERO_CONFIG_PATH=$LIBERO_HOME/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=${EGL_LIB_DIR:-}:$LD_LIBRARY_PATH
export PYTHONPATH=$CODE_DIR:$LIBERO_HOME
CKPT=${CKPT:?set CKPT to the checkpoint .pt}
GPU=$1; PORT=$2; SUITE=$3; START=$4; END=$5; OUT=$6; LOGDIR=$7
mkdir -p "$OUT/$SUITE" "$LOGDIR"
CUDA_VISIBLE_DEVICES=$GPU python ./examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.pretrained-path "$CKPT" --args.host 127.0.0.1 --args.port "$PORT" \
  --args.task-suite-name "$SUITE" --args.num-trials-per-task 1 \
  --args.start-idx "$START" --args.end-idx "$END" \
  --args.video-out-path "$OUT/$SUITE" --args.log-path "$LOGDIR"
