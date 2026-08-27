#!/bin/bash
#SBATCH --job-name=gif-refs-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/yz11445/AP-RD_GIFStream/hpc_setup/logs/stage7_%j.out

# Task-1 evaluation sweep: export paired own-references for both arms, then
# run the v3 path evaluator (composed D_path + pi sweep) for every rate.
#  - official references: codec-entry re-run with --hdown_reference_export_dir
#  - AP references: v3 exporter against checkpoint + archive
# Same-arch rule applies (l40s_public, like every codec/decode artifact).
set -euxo pipefail
source /scratch/yz11445/AP-RD_GIFStream/hpc_setup/env_common.sh

SCENE="${SCENE:-flame_salmon_1}"
RATES="${RATES:-0 1 2 3}"
N_KNN="${N_KNN:-8}"
TYPE=neur3d_1
DATA_DIR="$SCRATCH_ROOT/datasets/Neur3D/$SCENE"
RESULTS="$SCRATCH_ROOT/results/h007/$SCENE"
REFS="$SCRATCH_ROOT/results/h007/$SCENE/references_v3"
EVAL_OUT="$SCRATCH_ROOT/results/h007/$SCENE/eval_v3"
CAMPAIGN="$REPO_ROOT/eval_scripts/h007_v3_campaign_flame_salmon.json"
MANIFEST="$REPO_ROOT/provenance/h007_ap_gifstream_u3_patch_chain_patch10_dualkey_20260814.json"
MANIFEST_SHA=dac0239f405b40510db566677700700df2f6ada777fd6d015b5a31e308d7407e

export CUDA_HOME="$ENV_PREFIX"
export PATH="$ENV_PREFIX/bin:$PATH"
export CPATH="$ENV_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="$ENV_PREFIX/targets/x86_64-linux/lib:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$REPO_ROOT/GIFStream_APRD"
git rev-parse HEAD | grep -q c98486632e7dafd830740b1a1692bd08c48b96e3 \
  || { echo "FATAL: tree not at official commit"; exit 1; }
mkdir -p "$EVAL_OUT"

for rate in $RATES; do
  case "$rate" in
    0) LAMBDA=0.0005 ;; 1) LAMBDA=0.001 ;; 2) LAMBDA=0.002 ;; 3) LAMBDA=0.004 ;;
    *) echo "bad rate"; exit 1 ;;
  esac

  # --- official references (codec entry + hdown export per GOP) ---
  for gop in 0 1 2 3 4; do
    BUNDLE="$REFS/official/r${rate}/$SCENE/gop_${gop}"
    [ -d "$BUNDLE" ] && { echo "official ref exists r$rate g$gop"; continue; }
    RESULT_DIR="$RESULTS/official/nknn${N_KNN}/GOP_${gop}/r${rate}"
    CKPT="$RESULT_DIR/ckpts/ckpt_29999_rank0.pt"
    RECEIPT="$RESULT_DIR/frozen_training_receipt.json"
    RECEIPT_SHA=$(sha256sum "$RECEIPT" | cut -d' ' -f1)
    "$PY" examples/simple_trainer_GIFStream.py "$TYPE" --disable_viewer \
      --data_factor 2 --render_traj_path ellipse \
      --data_dir "$DATA_DIR/" --result_dir "$RESULT_DIR" \
      --eval_steps 7000 30000 --save_steps 7000 30000 \
      --compression_sim --entropy_model_opt --rd_lambda "$LAMBDA" --rate "$rate" \
      --batch_size 1 --GOP_size 60 --knn --n_knn "$N_KNN" --start_frame "$((gop * 60))" \
      --ap_variant official \
      --ap_provenance_manifest "$MANIFEST" \
      --ap_provenance_manifest_sha256 "$MANIFEST_SHA" \
      --ckpt "$CKPT" \
      --h007_training_receipt "$RECEIPT" \
      --h007_training_receipt_sha256 "$RECEIPT_SHA" \
      --compression end2end \
      --hdown_reference_export_dir "$REFS/official/r${rate}"
  done

  # --- AP references (v3 exporter per GOP) ---
  for gop in 0 1 2 3 4; do
    BUNDLE="$REFS/ap/r${rate}/gop_${gop}"
    [ -d "$BUNDLE" ] && { echo "ap ref exists r$rate g$gop"; continue; }
    RESULT_DIR="$RESULTS/ap-gifstream-full/nknn${N_KNN}/GOP_${gop}/r${rate}"
    mkdir -p "$REFS/ap/r${rate}"
    rm -f "$REFS/ap/r${rate}/gop_${gop}.summary.json"
    "$PY" "$REPO_ROOT/eval_scripts/h007_export_ap_same_run_reference_v3.py" \
      --campaign "$CAMPAIGN" --rate "$rate" \
      --checkpoint "$RESULT_DIR/ckpts/ckpt_29999_rank0.pt" \
      --compression-archive "$RESULT_DIR/compression_rank0.zip" \
      --gop-id "$gop" \
      --output-bundle "$BUNDLE" \
      --summary "$REFS/ap/r${rate}/gop_${gop}.summary.json"
  done

  # --- v3 path evaluation for this rate ---
  OUT="$EVAL_OUT/path_eval_r${rate}.json"
  rm -f "$OUT"
  "$PY" "$REPO_ROOT/eval_scripts/334_v3_paired_reference_path_evaluation.py" \
    --campaign "$CAMPAIGN" --rate "$rate" \
    --repo-root "$REPO_ROOT/GIFStream_APRD" \
    --official-reference-root "$REFS/official/r${rate}/$SCENE" \
    --ap-reference-root "$REFS/ap/r${rate}" \
    --official-decode-root "$RESULTS/official/nknn${N_KNN}" \
    --ap-decode-root "$RESULTS/ap-gifstream-full/nknn${N_KNN}" \
    --output "$OUT"
done

ls -la "$EVAL_OUT"
echo STAGE7_OK
