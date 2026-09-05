#!/bin/bash
# run_parallel_eval.sh
#
# Automatically selects 4 free GPUs, launches 4 policy servers,
# splits 11 TASK_SUITES into 4 groups (3-3-3-2), and runs evaluations in parallel.
#
# Usage:
#   bash examples/VLA-Arena/eval_files/run_parallel_eval.sh [OPTIONS]
#
# Can be run from anywhere; paths are resolved from the script's location.

set -euo pipefail

# Resolve the directory this script lives in (eval_files/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# stellavla root is three levels up: eval_files -> VLA-Arena -> examples -> stellavla
STELLAVLA_HOME="${STELLAVLA_HOME:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

###########################################################################################
# === Configuration ===
your_ckpt=""
DEMO_PACK="${DEMO_PACK:-${STELLAVLA_HOME}/playground/demo_packs/vla-arena}"
VLA_ARENA_ENV=""          # path to VLA-Arena uv project, e.g. /path/to/VLA-Arena/env/
NUM_SERVERS=4
BASE_PORT=10090           # ports will be BASE_PORT + gpu_id
GPU_MEM_THRESHOLD=2000    # GPUs with memory usage below this (MiB) are considered free
SERVER_STARTUP_WAIT=180   # seconds to wait for each server to start

# All 11 task suites
ALL_SUITES=(
    "safety_static_obstacles"
    "safety_cautious_grasp"
    "safety_hazard_avoidance"
    "safety_state_preservation"
    "safety_dynamic_obstacles"
    "distractor_static_distractors"
    "distractor_dynamic_distractors"
    "extrapolation_preposition_combinations"
    "extrapolation_task_workflows"
    "extrapolation_unseen_objects"
    "long_horizon"
)

# Every suite gets its own server; see the scheduling block below.
###########################################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Parallel evaluation: launches policy servers on free GPUs and runs VLA-Arena
task suites in parallel.

OPTIONS:
    -c, --checkpoint PATH       Path to pretrained checkpoint (required)
    --vla-arena-env PATH        Path to VLA-Arena uv project dir (required)
                                e.g. /path/to/VLA-Arena/envs/openpi
    --stellavla-home PATH         stellavla root directory (default: auto-detected)
    --num-servers NUM            Number of parallel servers/groups (default: $NUM_SERVERS)
    --base-port PORT             Base port number (default: $BASE_PORT)
    --gpu-mem-threshold MiB      Free GPU memory threshold (default: $GPU_MEM_THRESHOLD)
    --server-wait SECONDS        Server startup timeout (default: $SERVER_STARTUP_WAIT)
    -h, --help                   Show this help message

ENVIRONMENT VARIABLES:
    STELLAVLA_HOME                 stellavla root (overridden by --stellavla-home)
    VLA_ARENA_ENV                VLA-Arena uv project dir (overridden by --vla-arena-env)
    stellavla_python               Python interpreter (default: python)

EXAMPLES:
    $0 -c /path/to/ckpt.pt --vla-arena-env /path/to/VLA-Arena/envs/openpi
    STELLAVLA_HOME=/opt/stellavla $0 -c ckpt.pt --vla-arena-env /opt/VLA-Arena/envs/openpi
EOF
}

# --- Argument Parsing ---
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint)         your_ckpt="$2"; shift 2 ;;
        --vla-arena-env)         VLA_ARENA_ENV="$2"; shift 2 ;;
        --stellavla-home)          STELLAVLA_HOME="$2"; shift 2 ;;
        --num-servers)           NUM_SERVERS="$2"; shift 2 ;;
        --base-port)             BASE_PORT="$2"; shift 2 ;;
        --gpu-mem-threshold)     GPU_MEM_THRESHOLD="$2"; shift 2 ;;
        --server-wait)           SERVER_STARTUP_WAIT="$2"; shift 2 ;;
        -h|--help)               show_usage; exit 0 ;;
        *) print_error "Unknown option: $1"; show_usage; exit 1 ;;
    esac
done

# Also accept VLA_ARENA_ENV from environment variable if not set via flag
VLA_ARENA_ENV="${VLA_ARENA_ENV:-${VLA_ARENA_ENV_DEFAULT:-}}"

# Validate required arguments
if [[ -z "$your_ckpt" ]]; then
    print_error "Checkpoint path is required (-c / --checkpoint)"
    show_usage
    exit 1
fi
if [[ -z "$VLA_ARENA_ENV" ]]; then
    print_error "VLA-Arena env path is required (--vla-arena-env or VLA_ARENA_ENV env var)"
    show_usage
    exit 1
fi
if [[ ! -d "$VLA_ARENA_ENV" ]]; then
    print_error "VLA-Arena env directory does not exist: $VLA_ARENA_ENV"
    exit 1
fi
if [[ ! -d "$STELLAVLA_HOME" ]]; then
    print_error "stellavla home directory does not exist: $STELLAVLA_HOME"
    exit 1
fi

LOG_DIR="log"
mkdir -p "$LOG_DIR"

print_info "stellavla home  : $STELLAVLA_HOME"
print_info "VLA-Arena env : $VLA_ARENA_ENV"
print_info "Checkpoint    : $your_ckpt"
print_info "Log directory : $LOG_DIR"

# ---- Step 1: Find free GPUs ----
print_info "Detecting free GPUs (memory usage < ${GPU_MEM_THRESHOLD} MiB)..."

FREE_GPUS=()
while IFS=, read -r idx mem_used; do
    idx=$(echo "$idx" | xargs)
    mem_used=$(echo "$mem_used" | xargs | sed 's/ MiB//')
    if (( mem_used < GPU_MEM_THRESHOLD )); then
        FREE_GPUS+=("$idx")
    fi
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)

if (( ${#FREE_GPUS[@]} < NUM_SERVERS )); then
    print_error "Need ${NUM_SERVERS} free GPUs but only found ${#FREE_GPUS[@]}: ${FREE_GPUS[*]}"
    print_info "All GPU memory usage:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
    exit 1
fi

# Take first N free GPUs
SELECTED_GPUS=("${FREE_GPUS[@]:0:$NUM_SERVERS}")
print_success "Selected GPUs: ${SELECTED_GPUS[*]}"

# ---- Step 2: Launch policy servers ----
export PYTHONPATH="${STELLAVLA_HOME}:${PYTHONPATH:-}"
stellavla_python="${stellavla_python:-python}"

SERVER_PIDS=()
EVAL_PIDS=()
EVAL_LOGS=()
PORTS=()
CLEANED_UP=0

cleanup() {
    if (( CLEANED_UP == 1 )); then
        return
    fi
    CLEANED_UP=1

    print_warning "Cleaning up server processes..."
    for pid in "${EVAL_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            print_info "Killed eval/client PID $pid"
        fi
    done

    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            print_info "Killed server PID $pid"
        fi
    done

    # Escalate if some processes ignore SIGTERM.
    sleep 1
    for pid in "${EVAL_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
            print_info "Force killed eval/client PID $pid"
        fi
    done
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
            print_info "Force killed server PID $pid"
        fi
    done

    wait 2>/dev/null || true
}

on_interrupt() {
    print_warning "Received interrupt signal, terminating all spawned server/client processes..."
    cleanup
    exit 130
}

trap cleanup EXIT

TASK_SUITES=("${ALL_SUITES[@]}")
trap on_interrupt INT TERM

wait_for_server() {   # <server-pid> <port> <log-file> <timeout-s>
    local pid=$1 port=$2 log_file=$3 timeout=$4 elapsed=0
    while (( elapsed < timeout )); do
        if ! kill -0 "$pid" 2>/dev/null; then return 1; fi
        if [[ -f "$log_file" ]] && grep -Eq "server listening on .*:${port}" "$log_file"; then
            return 0
        fi
        if command -v ss >/dev/null 2>&1; then
            if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(:|\])${port}$"; then
                return 0
            fi
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    return 1
}

# ---- Steps 2-5: one dedicated server per suite ----
# A server is started for exactly one suite and torn down when that suite ends.
# Sharing a server across suites was measured to change the scores: on
# extrapolation_unseen_objects L1 a shared server scored 0.60 where a dedicated
# one scored 1.00, and over all 33 cells only 10 matched the reference. With one
# server per suite all 33 cells reproduce exactly, so that is the layout here.
# NUM_SERVERS is now the number of suites evaluated concurrently; suites beyond
# that queue for the next free GPU slot.

declare -A SLOT_PID SLOT_SUITE
EVAL_FAILURES=0
DONE_SUITES=0

start_suite() {   # <suite> <slot-index>
    local suite="$1" slot="$2"
    local gpu_id=${SELECTED_GPUS[$slot]}
    local port=$((BASE_PORT + gpu_id))
    local slog="${LOG_DIR}/server_${suite}.log"
    local elog="${LOG_DIR}/eval_${suite}.log"

    (
        CUDA_VISIBLE_DEVICES=${gpu_id} ${stellavla_python} \
            "${STELLAVLA_HOME}/examples/VLA-Arena/eval_files/serve_stellavla.py" \
            --ckpt_path "${your_ckpt}" --port "${port}" --idle_timeout -1 \
            --use_context_demo --demo_pack "${DEMO_PACK}" \
            > "${slog}" 2>&1 &
        srv=$!
        trap 'kill $srv 2>/dev/null' EXIT INT TERM
        if ! wait_for_server "$srv" "$port" "$slog" "$SERVER_STARTUP_WAIT"; then
            echo "server for ${suite} failed to start" >> "${elog}"
            exit 1
        fi
        # The simulator renders through EGL, which ignores CUDA_VISIBLE_DEVICES;
        # without this every client would render on GPU 0.
        MUJOCO_EGL_DEVICE_ID="${gpu_id}" \
        uv run --project "${VLA_ARENA_ENV}" \
            bash "${SCRIPT_DIR}/eval_vla_arena.sh" \
            --checkpoint "${your_ckpt}" --port "${port}" --suites "${suite}" \
            >> "${elog}" 2>&1
    ) &
    SLOT_PID[$slot]=$!
    SLOT_SUITE[$slot]="$suite"
    print_info "Started ${suite} on GPU ${gpu_id} (port ${port})"
}

reap_slot() {     # <slot-index>; returns 0 if the slot is free
    local slot=$1 pid=${SLOT_PID[$slot]:-}
    [[ -z "$pid" ]] && return 0
    kill -0 "$pid" 2>/dev/null && return 1
    if wait "$pid"; then
        print_success "${SLOT_SUITE[$slot]} completed. Log: ${LOG_DIR}/eval_${SLOT_SUITE[$slot]}.log"
    else
        print_error "${SLOT_SUITE[$slot]} failed. Check ${LOG_DIR}/eval_${SLOT_SUITE[$slot]}.log"
        EVAL_FAILURES=$((EVAL_FAILURES + 1))
    fi
    DONE_SUITES=$((DONE_SUITES + 1))
    unset "SLOT_PID[$slot]"
    return 0
}

QUEUE=("${TASK_SUITES[@]}")
print_info "Evaluating ${#QUEUE[@]} suites, ${NUM_SERVERS} at a time (one server each)"
for suite in "${QUEUE[@]}"; do
    slot=""
    while [[ -z "$slot" ]]; do
        for s in $(seq 0 $((NUM_SERVERS - 1))); do
            if reap_slot "$s"; then slot=$s; break; fi
        done
        [[ -z "$slot" ]] && sleep 10
    done
    start_suite "$suite" "$slot"
    sleep 5
done
for s in $(seq 0 $((NUM_SERVERS - 1))); do
    while ! reap_slot "$s"; do sleep 10; done
done


# ---- Summary ----
echo ""
print_info "===== Parallel Evaluation Complete ====="
print_info "GPU slots used: ${SELECTED_GPUS[*]} (one server per suite)"
print_info "Eval failures: ${EVAL_FAILURES} / ${#TASK_SUITES[@]}"

if (( EVAL_FAILURES == 0 )); then
    print_success "All evaluations completed successfully!"
else
    print_warning "${EVAL_FAILURES} evaluation group(s) had failures. Check logs."
fi

print_info "Server logs: ${LOG_DIR}/server_*.log"
print_info "Eval logs: ${LOG_DIR}/eval_*.log"
