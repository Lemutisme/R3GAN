#!/usr/bin/env bash
set -euo pipefail

# Build/check R3GAN fused CUDA extensions before launching multi-process training.
#
# Usage:
#   1) Activate your runtime env first (example):
#      conda activate gan
#   2) Run from repo root:
#      bash scripts/build_custom_ops.sh
#
# Optional:
#   - Pass an explicit python executable as arg1:
#       bash scripts/build_custom_ops.sh /root/miniconda3/envs/gan/bin/python
#   - Override cache/build env vars if needed:
#       TORCH_EXTENSIONS_DIR=/path/to/cache bash scripts/build_custom_ops.sh
#
# Notes:
#   - This script checks ninja/nvcc visibility and compiles:
#       * bias_act_plugin
#       * upfirdn2d_plugin
#   - If this script fails, do NOT start training; fix toolchain first.

PYTHON_BIN="${1:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Keep extension cache in repo by default so it is writable/reproducible.
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${REPO_ROOT}/.cache/torch_extensions}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"

# CUDA defaults; callers can override via environment.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"
export PATH="$(dirname "${CUDACXX}"):${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "[build_custom_ops] repo: ${REPO_ROOT}"
echo "[build_custom_ops] python: ${PYTHON_BIN}"
echo "[build_custom_ops] TORCH_EXTENSIONS_DIR: ${TORCH_EXTENSIONS_DIR}"
echo "[build_custom_ops] CUDA_HOME: ${CUDA_HOME}"

"${PYTHON_BIN}" - <<'PY'
import os
import shutil
import torch
from torch.utils.cpp_extension import verify_ninja_availability

print('[build_custom_ops] torch:', torch.__version__, 'cuda:', torch.version.cuda)
print('[build_custom_ops] torch.cuda.is_available():', torch.cuda.is_available())
print('[build_custom_ops] ninja:', shutil.which('ninja'))
print('[build_custom_ops] nvcc:', shutil.which('nvcc'))
print('[build_custom_ops] TORCH_EXTENSIONS_DIR:', os.environ.get('TORCH_EXTENSIONS_DIR'))

verify_ninja_availability()

from torch_utils import custom_ops
custom_ops.verbosity = 'full'
from torch_utils.ops import bias_act, upfirdn2d

if not bias_act._init():
    raise RuntimeError('bias_act_plugin build/load failed')
if not upfirdn2d._init():
    raise RuntimeError('upfirdn2d_plugin build/load failed')

print('[build_custom_ops] custom ops build OK')
PY
