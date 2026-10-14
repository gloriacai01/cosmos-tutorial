# Training container for Viam custom training jobs running cosmos-train (model/training.py).
#
# Viam's launcher only needs bash/python3/pip; it pip-installs the uploaded training package and
# runs `python3 -m model.training`. The pinned cosmos-framework checkout and its uv venv are baked
# in here so a job does not spend ~5 min and 12 GB re-creating them; checkpoints are still
# downloaded at job start. The venv brings its own torch==2.10.0+cu130 and nvidia-*-cu13 wheels,
# so the image carries no other Python ML stack: only the CUDA 13 userland (`base` flavour =
# cuda-compat for forward compatibility; libcuda itself comes from the host at /usr/local/nvidia).
#
# Build from the repo root (the joint-count patch is copied in):
#   docker buildx build --platform linux/amd64 -f docker/viam-training.Dockerfile \
#     -t <registry>/cosmos-train:<tag> --push .
FROM nvidia/cuda:13.0.3-base-ubuntu24.04

ARG FRAMEWORK_REV=efba2076c80648663a89a7fdbc92eb262fae39ac
ENV COSMOS_FRAMEWORK_ROOT=/opt/cosmos-framework \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    PIP_BREAK_SYSTEM_PACKAGES=1

# build-essential/python3-dev/linux-libc-dev: `uv sync` builds a few sdists from source
# (evdev via lerobot->pynput needs linux/input.h). PIP_BREAK_SYSTEM_PACKAGES: Ubuntu 24.04 marks
# the system Python externally managed; the launcher's `pip install` of the sdist needs it too.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-dev build-essential linux-libc-dev git ffmpeg libx11-dev \
    && rm -rf /var/lib/apt/lists/*
# Same uv the project lock resolves; a newer uv can change venv layout/resolution.
RUN pip install --no-cache-dir uv==0.12.23

# Same steps as cosmos_train.framework_setup.clone_framework / sync_framework. The patch only
# touches two source files of the editable checkout, so it is applied after the 12 GB sync layer
# to keep that layer cacheable across patch edits.
RUN git clone https://github.com/NVIDIA/cosmos-framework.git "$COSMOS_FRAMEWORK_ROOT" \
    && git -C "$COSMOS_FRAMEWORK_ROOT" checkout "$FRAMEWORK_REV"
RUN --mount=type=cache,target=/root/.cache/uv \
    cd "$COSMOS_FRAMEWORK_ROOT" && uv sync --all-extras --group=cu130-train
COPY src/cosmos_train/assets/cosmos_joint_count.patch /tmp/cosmos_joint_count.patch
COPY src/cosmos_train/assets/cosmos_loss_parts.patch /tmp/cosmos_loss_parts.patch
RUN git -C "$COSMOS_FRAMEWORK_ROOT" apply /tmp/cosmos_joint_count.patch \
    && git -C "$COSMOS_FRAMEWORK_ROOT" apply /tmp/cosmos_loss_parts.patch \
    && rm /tmp/cosmos_joint_count.patch /tmp/cosmos_loss_parts.patch

WORKDIR /workspace
