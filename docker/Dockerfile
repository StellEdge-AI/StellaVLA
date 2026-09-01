# One image, three simulators, one policy server.
#
# LIBERO and LIBERO-plus both install a package named `libero` and cannot share
# an interpreter, and VLA-Arena needs a different robosuite/mujoco generation
# again. So the image carries four isolated environments and the entrypoint
# picks the right pair for each benchmark:
#
#   /opt/conda/envs/stellavla    3.10  policy server (CUDA torch)
#   /opt/conda/envs/libero       3.10  LIBERO           robosuite 1.4.1
#   /opt/conda/envs/libero_plus  3.10  LIBERO-plus      robosuite 1.4.0 + assets
#   /app/VLA-Arena/envs/base/.venv  3.11  VLA-Arena     robosuite 1.5.2
#
# Weights and demo packs are NOT baked in (~19.6 GB). Mount them at /data, or
# run `docker run ... stellavla fetch-assets` once to download them.

ARG BASE_IMAGE=nvidia/cuda:12.1.1-base-ubuntu22.04
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="StellaVLA evaluation" \
      org.opencontainers.image.source="https://github.com/StellEdge-AI/StellaVLA"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYOPENGL_PLATFORM=egl \
    MUJOCO_GL=egl \
    NVIDIA_DRIVER_CAPABILITIES=all \
    PYTHONUNBUFFERED=1

# ── System dependencies ──────────────────────────────────────────────────────
# EGL/OpenGL for MuJoCo offscreen rendering; ImageMagick (libmagickwand) for the
# LIBERO-plus texture perturbations, which go through Wand.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget curl ca-certificates unzip \
        libegl1 libegl1-mesa-dev libgl1 libgl1-mesa-glx libgles2 \
        libglfw3 libosmesa6 libosmesa6-dev libglew-dev libglvnd-dev \
        libglib2.0-0 libsm6 libxext6 libxrender-dev \
        libmagickwand-dev libfontconfig1-dev libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# The NVIDIA EGL vendor file is bind-mounted read-only by nvidia-container-runtime
# on hosts whose default runtime is `nvidia`, so only write it when absent.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && { [ -s /usr/share/glvnd/egl_vendor.d/10_nvidia.json ] \
         || printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
            > /usr/share/glvnd/egl_vendor.d/10_nvidia.json; }

# ── conda ────────────────────────────────────────────────────────────────────
# MINIFORGE_URL and PIP_INDEX_URL exist so a build behind a slow or filtered
# link can point at a mirror; the defaults are the upstream sources.
ARG MINIFORGE_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
RUN wget -qO /tmp/miniforge.sh "${MINIFORGE_URL}" \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda clean -afy
ENV PATH=/opt/conda/bin:$PATH

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# `wget ... | tar` needs pipefail: under /bin/sh a failed download still exits 0
# because tar succeeds on empty input, leaving a silently empty source tree.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ── Simulator sources ────────────────────────────────────────────────────────
# Fetched before any Python environment so that changing a package list never
# re-downloads several gigabytes of scenes. Each repository is pinned by commit.
# Tarballs rather than clones: the LIBERO history carries its assets, and a
# shallow clone of it times out on slow links.
ARG LIBERO_REF=8f1084e3132a39270c3a13ebe37270a43ece2a01
ARG LIBERO_URL=https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/tar.gz/${LIBERO_REF}
RUN mkdir -p /app/libero \
    && wget -qO- "${LIBERO_URL}" | tar xz -C /app/libero --strip-components=1

ARG LIBERO_PLUS_REF=4976dc30028e805ff8094b55501d532c48fec182
ARG LIBERO_PLUS_URL=https://codeload.github.com/sylvestf/LIBERO-plus/tar.gz/${LIBERO_PLUS_REF}
RUN mkdir -p /app/LIBERO-plus \
    && wget -qO- "${LIBERO_PLUS_URL}" | tar xz -C /app/LIBERO-plus --strip-components=1

# LIBERO-plus ships its scenes and textures separately (~6.4 GB). The archive
# has a deeply nested top-level path, so extract it and move the `assets`
# directory into the package root where the BDDL files expect it.
ARG LIBERO_PLUS_ASSETS_URL=https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip
ARG SKIP_LIBERO_PLUS_ASSETS=0
RUN if [ "${SKIP_LIBERO_PLUS_ASSETS}" = "1" ]; then \
        echo "skipping LIBERO-plus assets; mount them at /app/LIBERO-plus/libero/libero/assets"; \
    else \
        wget --progress=dot:giga -O /tmp/assets.zip "${LIBERO_PLUS_ASSETS_URL}" \
        && unzip -q /tmp/assets.zip -d /tmp/assets-extracted \
        && rm /tmp/assets.zip \
        && rm -rf /app/LIBERO-plus/libero/libero/assets \
        && mv "$(find /tmp/assets-extracted -type d -name assets -print -quit)" \
              /app/LIBERO-plus/libero/libero/assets \
        && rm -rf /tmp/assets-extracted; \
    fi

# Two upstream gaps are patched here. VLA-Arena declares
# requires-python = "==3.11", which by PEP 440 matches only a bare "3.11" and
# rejects every real release, including the 3.11.15 upstream itself runs on.
# And envs/base/uv.lock is committed while the pyproject.toml beside it is not,
# so that environment cannot be resolved from a clean checkout — the manifest
# is supplied from docker/. Neither changes any pinned version.
ARG VLA_ARENA_REF=2ddcb003ca17bb850079acdb34fe0281140bd1df
ARG VLA_ARENA_URL=https://codeload.github.com/PKU-Alignment/VLA-Arena/tar.gz/${VLA_ARENA_REF}
RUN mkdir -p /app/VLA-Arena \
    && wget -qO- "${VLA_ARENA_URL}" | tar xz -C /app/VLA-Arena --strip-components=1 \
    && sed -i 's/^requires-python = "==3\.11"$/requires-python = "==3.11.*"/' /app/VLA-Arena/pyproject.toml
COPY docker/vla-arena-env-base.pyproject.toml /app/VLA-Arena/envs/base/pyproject.toml

# Both LIBERO forks call torch.load on their init-state files without
# weights_only=False. PyTorch 2.6 flipped that default to True, which rejects
# the numpy pickles those files contain, so every episode fails to start. The
# checkouts that produced the reference numbers carry this same edit locally;
# it is applied here so the image does not depend on an unrecorded local fix.
# Placed after the sources so it does not invalidate the 6.4 GB asset layer.
RUN sed -i 's/torch\.load(init_states_path)/torch.load(init_states_path, weights_only=False)/g' \
        /app/libero/libero/libero/benchmark/__init__.py \
        /app/libero/libero/lifelong/metric.py \
        /app/libero/libero/lifelong/evaluate.py \
        /app/LIBERO-plus/libero/libero/benchmark/__init__.py \
        /app/LIBERO-plus/libero/lifelong/metric.py \
        /app/LIBERO-plus/libero/lifelong/evaluate.py \
    && ! grep -rn "torch\.load(init_states_path)$" /app/libero /app/LIBERO-plus --include=*.py

# ── 1/4  policy server ───────────────────────────────────────────────────────
# Versions are the ones the published numbers were produced with. flash-attn is
# deliberately absent: the backbone probes for it and falls back to sdpa, which
# is what the reference runs used.
RUN conda create -y -n stellavla python=3.10 pip && conda clean -afy
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/conda/envs/stellavla/bin/pip install torch==2.6.0 torchvision==0.21.0 \
    && /opt/conda/envs/stellavla/bin/pip install \
        transformers==4.57.0 accelerate==1.5.2 diffusers==0.37.1 \
        qwen-vl-utils==0.0.14 safetensors==0.8.0 huggingface-hub==0.36.2 \
        numpy==2.2.6 pillow==11.3.0 omegaconf==2.3.0 rich \
        websockets==16.0 websocket-client==1.8.0 msgpack==1.1.2

# ── 2/4  LIBERO ──────────────────────────────────────────────────────────────
# `libero` has no top-level __init__.py, so the editable install registers no
# packages and the simulator is imported as a namespace package off PYTHONPATH,
# exactly as upstream intends. The install is kept for the dist metadata.
# torch/torchvision/mujoco/transformers are pinned to the versions the reference
# runs used, per environment. robomimic pulls a CUDA torch of its own, which
# would silently replace the pinned one, so the pair is reinstalled last and the
# CUDA runtime wheels it dragged in are removed — 2.7 GB per environment that
# nothing here loads. The CPU build of the same version is used because the
# simulator never runs a model on the GPU; torchvision has to move with torch or
# its compiled ops fail to register.
# The simulator never runs a model; torch is here only because robomimic
# imports it, and `rich` because the config reader the client shares with the
# policy package installs a rich logging handler at import time.
RUN conda create -y -n libero python=3.10 pip && conda clean -afy
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/conda/envs/libero/bin/pip install \
        --extra-index-url ${TORCH_INDEX_URL}/cpu torch==2.11.0+cpu \
    && /opt/conda/envs/libero/bin/pip install \
        numpy==1.24.4 mujoco==3.2.3 robosuite==1.4.1 bddl==1.0.1 gym==0.25.2 \
        robomimic==0.3.0 opencv-python==4.11.0.86 imageio==2.37.3 imageio-ffmpeg \
        h5py==3.16.0 easydict==1.13 hydra-core==1.3.2 tyro==1.0.13 \
        matplotlib scipy pyyaml termcolor future cloudpickle einops thop \
        omegaconf==2.3.0 rich websockets==16.0 msgpack==1.1.2 websocket-client==1.8.0 \
    && /opt/conda/envs/libero/bin/pip install \
        --extra-index-url ${TORCH_INDEX_URL}/cpu --force-reinstall --no-deps \
        torch==2.11.0+cpu torchvision==0.26.0+cpu \
    && /opt/conda/envs/libero/bin/pip uninstall -y $(/opt/conda/envs/libero/bin/pip list --format=freeze \
        | grep -E "^(nvidia-|triton)" | cut -d= -f1) \
    && /opt/conda/envs/libero/bin/pip install --no-deps -e /app/libero

# ── 3/4  LIBERO-plus ─────────────────────────────────────────────────────────
# The fork occupies the same `libero` package name as vanilla LIBERO, which is
# the reason for the separate environment.
RUN conda create -y -n libero_plus python=3.10 pip && conda clean -afy
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/conda/envs/libero_plus/bin/pip install \
        --extra-index-url ${TORCH_INDEX_URL}/cpu torch==2.11.0+cpu \
    && /opt/conda/envs/libero_plus/bin/pip install \
        numpy==1.22.4 mujoco==3.2.3 robosuite==1.4.0 bddl==1.0.1 gym==0.25.2 \
        robomimic==0.2.0 opencv-python==4.6.0.66 scikit-image==0.22.0 Wand==0.7.0 \
        imageio==2.37.3 imageio-ffmpeg h5py==3.16.0 easydict==1.9 \
        hydra-core==1.2.0 tyro==1.0.13 transformers==4.21.1 \
        matplotlib scipy pyyaml termcolor future cloudpickle einops thop \
        omegaconf==2.3.0 rich websockets==16.0 msgpack==1.1.2 websocket-client==1.8.0 \
    && /opt/conda/envs/libero_plus/bin/pip install \
        --extra-index-url ${TORCH_INDEX_URL}/cpu --force-reinstall --no-deps \
        torch==2.11.0+cpu torchvision==0.26.0+cpu \
    && /opt/conda/envs/libero_plus/bin/pip uninstall -y $(/opt/conda/envs/libero_plus/bin/pip list --format=freeze \
        | grep -E "^(nvidia-|triton)" | cut -d= -f1) \
    && /opt/conda/envs/libero_plus/bin/pip install --no-deps -e /app/LIBERO-plus

# ── 4/4  VLA-Arena ───────────────────────────────────────────────────────────
# Resolved from the project's own lock, against a conda 3.11 interpreter so uv
# never has to fetch one. --frozen rather than --locked: patching the
# unsatisfiable requires-python above changes the vla-arena metadata the lock
# recorded, so the consistency check can never pass. --frozen still installs
# exactly the locked set. That set covers the simulator only, so the packages
# the evaluation client needs are added on top, explicitly.
RUN conda create -y -n py311 python=3.11 pip && conda clean -afy \
    && /opt/conda/envs/stellavla/bin/pip install uv \
    && ln -sf /opt/conda/envs/stellavla/bin/uv /usr/local/bin/uv
RUN cd /app/VLA-Arena/envs/base \
    && UV_INDEX_URL=${PIP_INDEX_URL} \
       uv sync --frozen --python /opt/conda/envs/py311/bin/python \
    && uv pip install --python ./.venv/bin/python \
       websockets==16.0 msgpack==1.1.2 tyro omegaconf websocket-client==1.8.0 \
    && uv pip install --python ./.venv/bin/python \
       robosuite==1.5.2 mujoco==3.9.0 \
    && rm -rf ~/.cache/uv

# ── LIBERO path configs ──────────────────────────────────────────────────────
# LIBERO reads a config.yaml from $LIBERO_CONFIG_PATH. The two forks need
# different ones, so each gets its own directory rather than sharing ~/.libero.
# LIBERO-plus goes at <repo>/libero/config.yaml because that is where its own
# client script points LIBERO_CONFIG_PATH.
RUN mkdir -p /app/libero_config \
    && printf 'benchmark_root: /app/libero/libero/libero\nbddl_files: /app/libero/libero/libero/bddl_files\ninit_states: /app/libero/libero/libero/init_files\ndatasets: /app/libero/libero/datasets\nassets: /app/libero/libero/libero/assets\n' \
        > /app/libero_config/config.yaml \
    && printf 'benchmark_root: /app/LIBERO-plus/libero/libero\nbddl_files: /app/LIBERO-plus/libero/libero/bddl_files\ninit_states: /app/LIBERO-plus/libero/libero/init_files\ndatasets: /app/LIBERO-plus/libero/datasets\nassets: /app/LIBERO-plus/libero/libero/assets\n' \
        > /app/LIBERO-plus/libero/config.yaml

# ── Evaluation code ──────────────────────────────────────────────────────────
WORKDIR /workspace
COPY . /workspace
RUN /opt/conda/envs/stellavla/bin/pip install --no-deps -e /workspace \
    && mkdir -p /workspace/third_party /workspace/playground /data \
    && ln -sfn /app/libero /workspace/third_party/libero \
    && ln -sfn /app/LIBERO-plus /workspace/third_party/LIBERO-plus \
    && ln -sfn /app/VLA-Arena /workspace/third_party/VLA-Arena

ENV STELLAVLA_PY=/opt/conda/envs/stellavla/bin/python \
    LIBERO_PY=/opt/conda/envs/libero/bin/python \
    LIBERO_PLUS_PY=/opt/conda/envs/libero_plus/bin/python \
    VLA_ARENA_python=/app/VLA-Arena/envs/base/.venv/bin/python \
    VLA_ARENA_ENV=/app/VLA-Arena/envs/base \
    STELLAVLA_HOME=/workspace \
    STELLAVLA_DIR=/workspace \
    CONDA_SH=/opt/conda/etc/profile.d/conda.sh \
    LIBERO_REPO=/app/libero \
    LIBERO_CONFIG=/app/libero_config \
    LIBERO_PLUS_HOME=/app/LIBERO-plus \
    VLA_ARENA_HOME=/app/VLA-Arena \
    CODE_DIR=/workspace \
    DATA_DIR=/data \
    HF_HOME=/data/hf

ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["help"]
