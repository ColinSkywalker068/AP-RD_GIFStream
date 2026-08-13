#!/bin/bash
#SBATCH --job-name=gif-env-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/yz11445/AP-RD_GIFStream/hpc_setup/logs/stage1_%j.out

set -euxo pipefail
source /scratch/yz11445/AP-RD_GIFStream/hpc_setup/env_common.sh

# Connectivity sanity check -- fail loudly if compute nodes have no egress.
curl -sI -m 15 https://pypi.org/ >/dev/null || {
  echo "FATAL: no internet egress from $(hostname)"; exit 1;
}

# 1) Miniforge into scratch (TUNA mirror first, official fallback).
if [ ! -x "$MAMBA_ROOT/bin/conda" ]; then
  inst="$TMPDIR/miniforge.sh"
  if [ "$TUNA_OK" = "1" ] && curl -fL -m 300 -o "$inst" \
      https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh; then
    echo "miniforge installer from TUNA"
  else
    curl -fL -o "$inst" \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  fi
  bash "$inst" -b -p "$MAMBA_ROOT"
fi
CONDA="$MAMBA_ROOT/bin/conda"

# 2) Env: python 3.10 + CUDA 12.4 toolchain (nvcc for the GPU-stage compiles).
if [ ! -x "$PY" ]; then
  "$CONDA" create -y -n GIFStream --override-channels -c "$CF_CHANNEL" python=3.10 \
    || "$CONDA" create -y -n GIFStream --override-channels -c conda-forge python=3.10
fi
"$CONDA" install -y -n GIFStream --override-channels -c "$CF_CHANNEL" \
    cuda-toolkit=12.4 cuda-nvcc=12.4 \
  || "$CONDA" install -y -n GIFStream --override-channels -c conda-forge \
    cuda-toolkit=12.4 cuda-nvcc=12.4 \
  || "$CONDA" install -y -n GIFStream -c nvidia/label/cuda-12.4.1 cuda-toolkit

"$PY" -m pip install ninja cmake pytest

# 3) torch cu124 wheels (official index; nvidia-* deps fall back to TUNA/PyPI).
"$PY" -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url "${PIP_INDEX_URL:-https://pypi.org/simple}" \
  --extra-index-url https://pypi.org/simple

# 4) CPU-safe requirements; the fused-ssim CUDA build is deferred to stage 2.
grep -v 'fused-ssim' "$REPO_ROOT/GIFStream_APRD/examples/requirements.txt" > "$TMPDIR/reqs_cpu.txt"
"$PY" -m pip install -r "$TMPDIR/reqs_cpu.txt" \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# 5) MLEntropy (pure C++, no GPU needed).
cd "$REPO_ROOT/GIFStream_APRD/third_party/MLEntropy"
mkdir -p build && cd build
"$ENV_PREFIX/bin/cmake" ../cpp -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DPYTHON_EXECUTABLE="$PY"
make -j8

"$PY" -c "import torch, torch_scatter, scipy, cv2; print('stage1 imports OK', torch.__version__)"
echo STAGE1_OK
