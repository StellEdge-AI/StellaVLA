# Docker

One image, three simulators. LIBERO and LIBERO-Plus both install a package
called `libero` and cannot share an interpreter, and VLA-Arena needs a newer
robosuite/mujoco again, so the image carries four isolated environments and the
entrypoint pairs the right simulator with the policy server:

| environment | python | contents |
|---|---|---|
| `/opt/conda/envs/stellavla` | 3.10 | policy server — CUDA torch 2.6.0, transformers 4.57.0 |
| `/opt/conda/envs/libero` | 3.10 | LIBERO — robosuite 1.4.0, mujoco 3.2.3 |
| `/opt/conda/envs/libero_plus` | 3.10 | LIBERO-Plus — robosuite 1.4.0 + its 6.4 GB assets |
| `/app/VLA-Arena/envs/base/.venv` | 3.11 | VLA-Arena — robosuite 1.5.2, mujoco 3.9.0 |

The simulator repositories are pinned by commit, so a rebuild gets the same
scenes: LIBERO `8f1084e`, LIBERO-Plus `4976dc3`, VLA-Arena `2ddcb00`.

## Build

```bash
docker build -f docker/Dockerfile -t stellavla .
```

Useful build arguments:

| arg | default | why |
|---|---|---|
| `BASE_IMAGE` | `nvidia/cuda:12.1.1-base-ubuntu22.04` | the pip torch wheel brings its own CUDA libraries, so the `base` image is enough |
| `PIP_INDEX_URL` | PyPI | point at a mirror on a slow link |
| `MINIFORGE_URL` | GitHub release | same |
| `LIBERO_PLUS_ASSETS_URL` | HuggingFace | same |
| `SKIP_LIBERO_PLUS_ASSETS` | `0` | `1` skips the 6.4 GB download; mount the assets at `/app/LIBERO-plus/libero/libero/assets` instead |

## Assets

Weights, demo packs and the backbone (~28 GB) are not baked in. Fetch them once
into a host directory and mount it at `/data`:

```bash
mkdir -p $PWD/data
docker run --rm --gpus all -v $PWD/data:/data stellavla fetch-assets
```

That leaves `/data/checkpoints/{libero,vla-arena}/`,
`/data/demo_packs/{libero,vla-arena}/` and the Qwen3-VL backbone.

## Run

```bash
DOCKER="docker run --rm --gpus all --shm-size=16g -v $PWD/data:/data stellavla"

$DOCKER libero                     # 4 suites x 10 tasks x 50 trials
$DOCKER libero-plus                # 9,430 episodes
$DOCKER vla-arena                  # 11 suites x 3 levels
```

`GPUS_CSV` (LIBERO) and `LP_GPUS` (LIBERO-Plus) select devices; both default to
GPU 0. VLA-Arena picks free GPUs itself; `--num-servers N` sets how many run at
once, and `ARENA_SUITES=<suite>` restricts the run to one suite on `ARENA_GPU`.
Every benchmark writes under `/data`, so results survive the container.

`--gpus all` and `NVIDIA_DRIVER_CAPABILITIES=all` are both required: MuJoCo
renders through EGL, which needs the graphics capability and not just compute.

## Pinned versions

Each environment matches the versions the reference numbers were produced with.
The simulators use the CPU build of the same torch version — they never run a
model on the GPU — which keeps ~2.7 GB of CUDA wheels per environment out of
the image.

VLA-Arena's robosuite/mujoco are pinned **over** its uv lock, which resolves to
robosuite 1.5.1 / mujoco 3.5.0. That difference is not cosmetic: on the lock's
versions the overall score drops from 0.68 to 0.61. Because `uv run` re-syncs a
project to its lock, it would silently undo the pin, so the image sets
`UV_NO_SYNC=1` and the evaluation scripts call the venv interpreter directly.
Do not reintroduce `uv run` in the evaluation path.

| | torch | torchvision | transformers | mujoco | robosuite | numpy |
|---|---|---|---|---|---|---|
| stellavla | 2.6.0+cu124 | 0.21.0 | 4.57.0 | — | — | 2.2.6 |
| libero | 2.11.0+cpu | 0.26.0+cpu | — | 3.2.3 | 1.4.0 | 1.24.4 |
| libero_plus | 2.11.0+cpu | 0.26.0+cpu | 4.21.1 | 3.2.3 | 1.4.0 | 1.22.4 |
| vla_arena | from its uv lock | | | 3.9.0 † | 1.5.2 † | 1.26.4 |

† pinned over the lock, as described above.
