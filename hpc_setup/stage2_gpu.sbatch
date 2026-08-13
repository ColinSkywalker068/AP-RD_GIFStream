#!/bin/bash
#SBATCH --job-name=gif-env-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/yz11445/AP-RD_GIFStream/hpc_setup/logs/stage2_%j.out

set -euxo pipefail
source /scratch/yz11445/AP-RD_GIFStream/hpc_setup/env_common.sh

export CUDA_HOME="$ENV_PREFIX"
export PATH="$ENV_PREFIX/bin:$PATH"
# Cover A100 (8.0), L40S/Ada (8.9), H100/H200 (9.0) so one build serves
# whichever partition training lands on.
export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"
export MAX_JOBS=16

nvidia-smi
"$PY" -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
nvcc --version

# fused-ssim at the commit pinned by examples/requirements.txt
"$PY" -m pip install --no-build-isolation \
  "git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157"

# gsplat CUDA extension (editable) + gridencoder
cd "$REPO_ROOT/GIFStream_APRD"
"$PY" -m pip install -e . --no-build-isolation
"$PY" -m pip install ./third_party/gridencoder --no-build-isolation

"$PY" -c "import gsplat; from gsplat import csrc; print('gsplat csrc OK', gsplat.__version__)"

# Acceptance gate: the full repo test suite on the GPU node.
cd "$REPO_ROOT/GIFStream_APRD"
"$PY" -m pytest tests/ -q

echo STAGE2_OK
