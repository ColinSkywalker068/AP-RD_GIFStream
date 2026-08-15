#!/bin/bash
#SBATCH --job-name=gif-container-decode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/yz11445/AP-RD_GIFStream/hpc_setup/logs/stage6_%j.out

# Build the counted five-GOP sequence containers and run the clean decoder
# over every (variant, rate) of the dev matrix.  Requires all GOP_0..GOP_4
# training runs to exist.  VARIANTS/RATES/N_KNN/EXP_TAG overridable.
set -euxo pipefail
source /scratch/yz11445/AP-RD_GIFStream/hpc_setup/env_common.sh

SCENE="${SCENE:-flame_salmon_1}"
VARIANTS="${VARIANTS:-official ap-gifstream-full}"
RATES="${RATES:-0 1 2 3}"
N_KNN="${N_KNN:-8}"
EXP_TAG="${EXP_TAG:-}"
SEED=20260715   # frozen ap_compression_seed shared by every producer
MANIFEST="$REPO_ROOT/provenance/h007_ap_gifstream_u3_patch_chain_patch10_dualkey_20260814.json"
MANIFEST_SHA=8e6894607d06cc474daeda4da1829d7cfc2624e588941d3392e0d8c43528d1a9

export CUDA_HOME="$ENV_PREFIX"
export PATH="$ENV_PREFIX/bin:$PATH"
export CPATH="$ENV_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="$ENV_PREFIX/targets/x86_64-linux/lib:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$REPO_ROOT/GIFStream_APRD"
git rev-parse HEAD | grep -q c98486632e7dafd830740b1a1692bd08c48b96e3 \
  || { echo "FATAL: tree not at official commit"; exit 1; }
"$PY" -c "import torch; assert torch.cuda.is_available()"

RESULTS="$SCRATCH_ROOT/results/h007/$SCENE"
fail=0
for variant in $VARIANTS; do
  for rate in $RATES; do
    base="$RESULTS/$variant/nknn${N_KNN}"
    tagsuf="${EXP_TAG:+_$EXP_TAG}"
    archives=()
    for gop in 0 1 2 3 4; do
      z="$base/GOP_${gop}/r${rate}${tagsuf}/compression_rank0.zip"
      test -f "$z" || { echo "MISSING $z"; fail=1; }
      archives+=("$z")
    done
    [ "$fail" = 1 ] && continue
    receipt="$base/GOP_0/r${rate}${tagsuf}/frozen_training_receipt.json"
    cfg_sha=$("$PY" -c "import json,sys; print(json.load(open('$receipt'))['training_config_sha256'])")
    seq_dir="$RESULTS/$variant/nknn${N_KNN}/sequences"
    mkdir -p "$seq_dir"
    seq_zip="$seq_dir/sequence_r${rate}${tagsuf}.zip"
    rm -f "$seq_zip"
    echo "=== BUILD sequence $variant r$rate ==="
    "$PY" examples/h007_sequence_container.py build \
      --scene "$SCENE" --method "$variant" \
      --gop-archives "${archives[@]}" \
      --output "$seq_zip" \
      --training-config-sha256 "$cfg_sha" \
      --seed "$SEED"
    echo "=== VALIDATE sequence $variant r$rate ==="
    "$PY" examples/h007_sequence_container.py validate \
      --archive "$seq_zip" --scene "$SCENE" --method "$variant"
    for gop in 0 1 2 3 4; do
      out="$base/GOP_${gop}/r${rate}${tagsuf}/clean_decode"
      rm -rf "$out"
      echo "=== CLEAN DECODE $variant r$rate GOP $gop ==="
      "$PY" examples/h007_clean_decode_gifstream.py \
        --archive "${archives[$gop]}" \
        --sequence-archive "$seq_zip" \
        --gop-id "$gop" \
        --output-dir "$out" \
        --provenance-manifest "$MANIFEST" \
        --provenance-manifest-sha256 "$MANIFEST_SHA"
      test -f "$out/clean_decode_manifest.json" || { echo "DECODE FAILED $variant r$rate g$gop"; fail=1; }
    done
  done
done

[ "$fail" = 0 ] || { echo STAGE6_INCOMPLETE; exit 1; }
find "$RESULTS" -name 'sequence_r*.zip' | sort
echo STAGE6_OK
