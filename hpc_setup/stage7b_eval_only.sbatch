#!/bin/bash
#SBATCH --job-name=gif-eval-only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/yz11445/AP-RD_GIFStream/hpc_setup/logs/eval_only_%j.out

# Re-run ONLY the 334v3 path evaluation (references and decodes must exist).
set -euxo pipefail
source /scratch/yz11445/AP-RD_GIFStream/hpc_setup/env_common.sh
export CUDA_HOME="$ENV_PREFIX"
export PATH="$ENV_PREFIX/bin:$PATH"
export CPATH="$ENV_PREFIX/targets/x86_64-linux/include"
export LD_LIBRARY_PATH="$ENV_PREFIX/targets/x86_64-linux/lib:$ENV_PREFIX/lib"
RESULTS=/scratch/yz11445/results/h007/flame_salmon_1
REFS=$RESULTS/references_v3
for rate in ${RATES:-0 1 2 3}; do
  OUT=$RESULTS/eval_v3/path_eval_dualpi_r${rate}.json
  rm -f "$OUT"
  "$PY" "$REPO_ROOT/eval_scripts/334_v3_paired_reference_path_evaluation.py" \
    --campaign "$REPO_ROOT/eval_scripts/h007_v3_campaign_flame_salmon.json" --rate "$rate" \
    --repo-root "$REPO_ROOT/GIFStream_APRD" \
    --official-reference-root "$REFS/official/r${rate}/flame_salmon_1" \
    --ap-reference-root "$REFS/ap/r${rate}" \
    --official-decode-root "$RESULTS/official/nknn8" \
    --ap-decode-root "$RESULTS/ap-gifstream-full/nknn8" \
    --output "$OUT"
done
echo EVAL_ONLY_OK
