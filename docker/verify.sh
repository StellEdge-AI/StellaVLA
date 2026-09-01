#!/bin/bash
# Check that every environment in the image is usable: the policy stack builds,
# each simulator renders a frame through EGL, and each evaluation client
# imports. The client imports matter — they reach into the policy package for
# its config reader, so they need more than the simulator alone does.
set -u
fail=0

echo "=== policy server ==="
/opt/conda/envs/stellavla/bin/python - <<'PY' || fail=1
import sys, torch, transformers
sys.path.insert(0, "/workspace")
from stellavla.model.tools import FRAMEWORK_REGISTRY
import stellavla.model.framework
print("  torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("  transformers", transformers.__version__)
print("  frameworks", sorted(FRAMEWORK_REGISTRY._registry))
PY

render() {  # label, python, pythonpath, libero_config_path
  echo "=== $1 ==="
  PYTHONPATH="$3" LIBERO_CONFIG_PATH="$4" "$2" - <<'PY' || return 1
import os, robosuite
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(**{"bddl_file_name": bddl, "camera_heights": 256, "camera_widths": 256})
env.seed(7); obs = env.reset(); env.close()
print("  robosuite", robosuite.__version__, "| tasks", suite.n_tasks,
      "| rendered", obs["agentview_image"].shape)
PY
}
render "LIBERO simulator"      /opt/conda/envs/libero/bin/python      /app/libero      /app/libero_config          || fail=1
render "LIBERO-plus simulator" /opt/conda/envs/libero_plus/bin/python /app/LIBERO-plus /app/LIBERO-plus/libero     || fail=1

echo "=== VLA-Arena simulator ==="
/app/VLA-Arena/envs/base/.venv/bin/python -c "
import vla_arena, robosuite, mujoco
print('  robosuite', robosuite.__version__, '| mujoco', mujoco.__version__)" || fail=1

echo "=== evaluation clients ==="
client() {  # label, python, pythonpath, libero_config_path, file
  out=$(cd /workspace && PYTHONPATH="$3" LIBERO_CONFIG_PATH="$4" "$2" -c "
import importlib.util
s = importlib.util.spec_from_file_location('m', '$5')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)" 2>&1)
  if [ $? -eq 0 ]; then echo "  $1 imports"; else
    echo "  $1 FAILED"; echo "$out" | tail -4; return 1; fi
}
client "LIBERO     " /opt/conda/envs/libero/bin/python      /app/libero      /app/libero_config \
       /workspace/examples/LIBERO/eval_files/eval_libero.py || fail=1
client "LIBERO-plus" /opt/conda/envs/libero_plus/bin/python /app/LIBERO-plus /app/LIBERO-plus/libero \
       /workspace/examples/LIBERO-plus/eval_files/eval_libero.py || fail=1
client "VLA-Arena  " /app/VLA-Arena/envs/base/.venv/bin/python \
       /workspace:/app/VLA-Arena:/workspace/examples/VLA-Arena/eval_files /app/libero_config \
       /workspace/examples/VLA-Arena/eval_files/eval_vla_arena.py || fail=1

[ "$fail" = 0 ] && echo && echo "ALL CHECKS PASSED" || { echo; echo "SOME CHECKS FAILED"; exit 1; }
