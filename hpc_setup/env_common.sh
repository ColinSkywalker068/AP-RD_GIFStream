# Shared environment for the GIFStream env-setup jobs on torch.
# Everything lives under /scratch/yz11445 -- the real home is never written.
export SCRATCH_ROOT=/scratch/yz11445
export HOME="$SCRATCH_ROOT/.home_shim"
mkdir -p "$HOME"
export MAMBA_ROOT="$SCRATCH_ROOT/miniforge3"
export CONDA_PKGS_DIRS="$SCRATCH_ROOT/.conda_pkgs"
export CONDA_ENVS_DIRS="$MAMBA_ROOT/envs"
export PIP_CACHE_DIR="$SCRATCH_ROOT/.pip_cache"
export XDG_CACHE_HOME="$SCRATCH_ROOT/.cache"
export TMPDIR="$SCRATCH_ROOT/.tmp"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

export REPO_ROOT="$SCRATCH_ROOT/AP-RD_GIFStream"
export ENV_PREFIX="$MAMBA_ROOT/envs/GIFStream"
export PY="$ENV_PREFIX/bin/python"

# Mirror policy: check Tsinghua TUNA first, fall back to official sources.
TUNA_OK=0
if curl -sI -m 8 https://mirrors.tuna.tsinghua.edu.cn/ >/dev/null 2>&1; then
  TUNA_OK=1
fi
export TUNA_OK
if [ "$TUNA_OK" = "1" ]; then
  export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
  export PIP_EXTRA_INDEX_URL=https://pypi.org/simple
  export CF_CHANNEL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
else
  export CF_CHANNEL=conda-forge
fi
echo "TUNA_OK=$TUNA_OK CF_CHANNEL=$CF_CHANNEL"
