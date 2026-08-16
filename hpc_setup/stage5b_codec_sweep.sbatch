#!/bin/bash
#SBATCH --job-name=gif-codec-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/yz11445/AP-RD_GIFStream/hpc_setup/logs/stage5b_%j.out

# Re-run ONLY the receipt + codec-entry phases for existing checkpoints
# (used after a manifest-bytes fix: training is untouched, archives are
# re-minted with the current canonical manifest embedded).
set -euxo pipefail
source /scratch/yz11445/AP-RD_GIFStream/hpc_setup/env_common.sh

SCENE="${SCENE:-flame_salmon_1}"
VARIANTS="${VARIANTS:-official ap-gifstream-full}"
RATES="${RATES:-0 1 2 3}"
GOPS="${GOPS:-0 1 2 3 4}"
N_KNN="${N_KNN:-8}"
EXP_TAG="${EXP_TAG:-}"
MANIFEST="$REPO_ROOT/provenance/h007_ap_gifstream_u3_patch_chain_patch10_dualkey_20260814.json"
MANIFEST_SHA=1c80343cb8286b37996345edd3a81122697c0af7d5fb4e8a2b11c9633ef9ed6b

case "$SCENE" in
  flame_salmon_1) TYPE=neur3d_1 ;;
  coffee_martini) TYPE=neur3d_2 ;;
  *) TYPE=neur3d_0 ;;
esac
DATA_DIR="$SCRATCH_ROOT/datasets/Neur3D/$SCENE"

export CUDA_HOME="$ENV_PREFIX"
export PATH="$ENV_PREFIX/bin:$PATH"
export CPATH="$ENV_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="$ENV_PREFIX/targets/x86_64-linux/lib:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$REPO_ROOT/GIFStream_APRD"
git rev-parse HEAD | grep -q c98486632e7dafd830740b1a1692bd08c48b96e3 \
  || { echo "FATAL: tree not at official commit"; exit 1; }

ok=0; fail=0
for variant in $VARIANTS; do
  for rate in $RATES; do
    case "$rate" in
      0) LAMBDA=0.0005 ;; 1) LAMBDA=0.001 ;; 2) LAMBDA=0.002 ;; 3) LAMBDA=0.004 ;;
      4) LAMBDA=0.0009 ;; 5) LAMBDA=0.00095 ;; *) echo "bad rate"; exit 1 ;;
    esac
    for gop in $GOPS; do
      RESULT_DIR="$SCRATCH_ROOT/results/h007/$SCENE/$variant/nknn${N_KNN}/GOP_${gop}/r${rate}${EXP_TAG:+_$EXP_TAG}"
      CKPT="$RESULT_DIR/ckpts/ckpt_29999_rank0.pt"
      test -f "$CKPT" || { echo "MISSING CKPT $variant r$rate g$gop"; fail=$((fail+1)); continue; }
      START_FRAME=$((gop * 60))
      COMMON_FLAGS=(
        --disable_viewer --data_factor 2 --render_traj_path ellipse
        --data_dir "$DATA_DIR/" --result_dir "$RESULT_DIR"
        --eval_steps 7000 30000 --save_steps 7000 30000
        --compression_sim --entropy_model_opt --rd_lambda "$LAMBDA" --rate "$rate"
        --batch_size 1 --GOP_size 60 --knn --n_knn "$N_KNN" --start_frame "$START_FRAME"
        --ap_variant "$variant"
        --ap_provenance_manifest "$MANIFEST"
        --ap_provenance_manifest_sha256 "$MANIFEST_SHA"
      )
      echo "=== CODEC SWEEP $variant r$rate g$gop ==="
      CONFIG_JSON="$RESULT_DIR/producer_training_config.json"
      RECEIPT="$RESULT_DIR/frozen_training_receipt.json"
      "$PY" "$REPO_ROOT/hpc_setup/make_training_config.py" --output "$CONFIG_JSON" -- \
        "$TYPE" "${COMMON_FLAGS[@]}"
      rm -f "$RECEIPT"
      "$PY" examples/h007_freeze_training_receipt.py \
        --checkpoints "$CKPT" \
        --training-config "$CONFIG_JSON" \
        --provenance-manifest "$MANIFEST" \
        --provenance-manifest-sha256 "$MANIFEST_SHA" \
        --repo-root "$REPO_ROOT/GIFStream_APRD" \
        --output "$RECEIPT"
      RECEIPT_SHA=$(sha256sum "$RECEIPT" | cut -d' ' -f1)
      SCORE_FLAGS=()
      if [ "$variant" != "official" ]; then
        SCORE_FLAGS=(--ap_score_path "$RESULT_DIR/ap_freeze/reference_scores_step20000.npz")
      fi
      # run_compression rebuilds compression/ from scratch; clear stale outputs.
      rm -rf "$RESULT_DIR/compression" "$RESULT_DIR/compression_rank0.zip" \
        "$RESULT_DIR"/compression_rank0_*.json
      if "$PY" examples/simple_trainer_GIFStream.py "$TYPE" "${COMMON_FLAGS[@]}" \
        --ckpt "$CKPT" \
        --h007_training_receipt "$RECEIPT" \
        --h007_training_receipt_sha256 "$RECEIPT_SHA" \
        --compression end2end "${SCORE_FLAGS[@]}" \
        && test -f "$RESULT_DIR/compression_rank0.zip"; then
        ok=$((ok+1))
      else
        echo "CELL FAILED $variant r$rate g$gop"; fail=$((fail+1))
      fi
    done
  done
done
echo "codec sweep: ok=$ok fail=$fail"
[ "$fail" = 0 ] || { echo STAGE5B_INCOMPLETE; exit 1; }
echo STAGE5B_OK
