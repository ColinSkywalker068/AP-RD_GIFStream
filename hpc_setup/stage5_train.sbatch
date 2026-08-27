#!/bin/bash
#SBATCH --job-name=gif-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/yz11445/AP-RD_GIFStream/hpc_setup/logs/stage5_%j.out

# One H007 training + codec entry for (SCENE, VARIANT, RATE, GOP_ID, N_KNN).
# Parameters arrive via sbatch --export (see submit_train.sh); defaults give
# the dev-scene acceptance run from ONBOARDING step 4.
set -euxo pipefail
source /scratch/yz11445/AP-RD_GIFStream/hpc_setup/env_common.sh

SCENE="${SCENE:-flame_salmon_1}"
VARIANT="${VARIANT:-ap-gifstream-full}"
RATE="${RATE:-0}"
GOP_ID="${GOP_ID:-0}"
N_KNN="${N_KNN:-8}"
EXP_TAG="${EXP_TAG:-}"
ZERO_HOP="${ZERO_HOP:-0}"

# Frozen rate grid (h007_sequence_container.FROZEN_RATE_LAMBDAS).
case "$RATE" in
  0) LAMBDA=0.0005 ;;
  1) LAMBDA=0.001 ;;
  2) LAMBDA=0.002 ;;
  3) LAMBDA=0.004 ;;
  4) LAMBDA=0.0009 ;;
  5) LAMBDA=0.00095 ;;
  *) echo "unknown rate index $RATE"; exit 1 ;;
esac

# Scene -> tyro preset (from the official multigop launch script).
case "$SCENE" in
  flame_salmon_1) TYPE=neur3d_1 ;;
  coffee_martini) TYPE=neur3d_2 ;;
  *) TYPE=neur3d_0 ;;
esac

DATA_DIR="$SCRATCH_ROOT/datasets/Neur3D/$SCENE"
RESULT_DIR="$SCRATCH_ROOT/results/h007/$SCENE/$VARIANT/nknn${N_KNN}/GOP_${GOP_ID}/r${RATE}${EXP_TAG:+_$EXP_TAG}"
MANIFEST="$REPO_ROOT/provenance/h007_ap_gifstream_u3_patch_chain_patch10_dualkey_20260814.json"
MANIFEST_SHA=dac0239f405b40510db566677700700df2f6ada777fd6d015b5a31e308d7407e
START_FRAME=$((GOP_ID * 60))

export CUDA_HOME="$ENV_PREFIX"
export PATH="$ENV_PREFIX/bin:$PATH"
export CPATH="$ENV_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="$ENV_PREFIX/targets/x86_64-linux/lib:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Preconditions: preprocessed scene, clean provenance tree at official commit.
test -d "$DATA_DIR/png" || { echo "FATAL: scene not preprocessed"; exit 1; }
test -d "$DATA_DIR/colmap_${START_FRAME}" || { echo "FATAL: missing colmap_${START_FRAME}"; exit 1; }
cd "$REPO_ROOT/GIFStream_APRD"
git rev-parse HEAD | grep -q c98486632e7dafd830740b1a1692bd08c48b96e3 \
  || { echo "FATAL: GIFStream_APRD is not at the official commit"; exit 1; }

mkdir -p "$RESULT_DIR"

COMMON_FLAGS=(
  --disable_viewer --data_factor 2 --render_traj_path ellipse
  --data_dir "$DATA_DIR/" --result_dir "$RESULT_DIR"
  --eval_steps 7000 30000 --save_steps 7000 30000
  --compression_sim --entropy_model_opt --rd_lambda "$LAMBDA" --rate "$RATE"
  --batch_size 1 --GOP_size 60 --knn --n_knn "$N_KNN" --start_frame "$START_FRAME"
  --ap_variant "$VARIANT"
  --ap_provenance_manifest "$MANIFEST"
  --ap_provenance_manifest_sha256 "$MANIFEST_SHA"
)
if [ "$ZERO_HOP" = "1" ]; then
  COMMON_FLAGS+=(--ap_zero_hop_closure)
fi

# 1) Compression-aware training (freeze happens at step 20000 for AP variants).
"$PY" examples/simple_trainer_GIFStream.py "$TYPE" "${COMMON_FLAGS[@]}"

# 2) Frozen training receipt (required by the codec entry, fail-closed).
CKPT="$RESULT_DIR/ckpts/ckpt_29999_rank0.pt"
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
echo "receipt sha256: $RECEIPT_SHA"

# 3) Codec entry: end-to-end compression from the final checkpoint.
SCORE_FLAGS=()
if [ "$VARIANT" != "official" ]; then
  SCORE_PATH="$RESULT_DIR/ap_freeze/reference_scores_step20000.npz"
  test -f "$SCORE_PATH" || { echo "FATAL: frozen score artifact missing"; exit 1; }
  SCORE_FLAGS=(--ap_score_path "$SCORE_PATH")
fi
"$PY" examples/simple_trainer_GIFStream.py "$TYPE" "${COMMON_FLAGS[@]}" \
  --ckpt "$CKPT" \
  --h007_training_receipt "$RECEIPT" \
  --h007_training_receipt_sha256 "$RECEIPT_SHA" \
  --compression end2end "${SCORE_FLAGS[@]}"

# 3) Artifact summary (the sequence container + clean decode need all 5 GOPs;
# they run as a separate step once GOP_0..GOP_4 exist).
find "$RESULT_DIR" -maxdepth 2 \( -name '*.zip' -o -name '*.npz' -o -name '*.json' -o -name 'ckpt_*' \) | sort
du -sh "$RESULT_DIR"
echo "STAGE5_OK scene=$SCENE variant=$VARIANT rate=$RATE gop=$GOP_ID n_knn=$N_KNN exp_tag=${EXP_TAG:-none} zero_hop=$ZERO_HOP"
