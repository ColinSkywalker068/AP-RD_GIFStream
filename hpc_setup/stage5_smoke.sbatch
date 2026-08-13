#!/bin/bash
#SBATCH --job-name=gif-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/yz11445/AP-RD_GIFStream/hpc_setup/logs/smoke_%j.out

# Smoke test: the REAL trainer + REAL codec entry, schedule scaled to 1/10
# (same phase ordering as the full run: densify -> entropy opt -> AP freeze
# at 2/3 max_steps -> path loss -> codec).  No mocked code paths.  Ends with
# smoke_verify.py, which interrogates every artifact and fails loudly.
set -euxo pipefail
trap 'echo "SMOKE_FAILED (see log above)"' ERR
source /scratch/yz11445/AP-RD_GIFStream/hpc_setup/env_common.sh

SCENE="${SCENE:-flame_salmon_1}"
VARIANT="${VARIANT:-ap-gifstream-full}"
RATE="${RATE:-0}"
LAMBDA=0.0005
MAX_STEPS=3000          # 30000 * 0.1 via --steps_scaler
FREEZE_STEP=2000        # keeps the frozen 2/3 ratio (real: 20000/30000)
ENTROPY_STEP=1000       # keeps the frozen 1/3 ratio (real: 10000/30000)

case "$SCENE" in
  flame_salmon_1) TYPE=neur3d_1 ;;
  coffee_martini) TYPE=neur3d_2 ;;
  *) TYPE=neur3d_0 ;;
esac

DATA_DIR="$SCRATCH_ROOT/datasets/Neur3D/$SCENE"
RESULT_DIR="$SCRATCH_ROOT/results/smoke/$SCENE/$VARIANT/r${RATE}"
MANIFEST="$REPO_ROOT/provenance/h007_ap_gifstream_u3_patch_chain_patch10_dualkey_20260814.json"
MANIFEST_SHA=59e37db23e640d59eddd4e56d497b9400735a9d1c04a3e995e0006134248565f

export CUDA_HOME="$ENV_PREFIX"
export PATH="$ENV_PREFIX/bin:$PATH"
export CPATH="$ENV_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="$ENV_PREFIX/targets/x86_64-linux/lib:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "=== SMOKE PRECONDITIONS ==="
test -d "$DATA_DIR/png" || { echo "FATAL: scene not preprocessed"; exit 1; }
test -d "$DATA_DIR/colmap_0" || { echo "FATAL: missing colmap_0"; exit 1; }
cd "$REPO_ROOT/GIFStream_APRD"
git rev-parse HEAD | grep -q c98486632e7dafd830740b1a1692bd08c48b96e3 \
  || { echo "FATAL: tree not at official commit"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -c "import torch, gsplat; from gsplat import csrc; assert torch.cuda.is_available(); print('torch+csrc OK', torch.__version__)"
rm -rf "$RESULT_DIR" && mkdir -p "$RESULT_DIR"

# Flags shared by BOTH phases: the codec entry re-verifies the training
# receipt against its own config, so the schedules must be identical.
COMMON_FLAGS=(
  --disable_viewer --data_factor 2 --render_traj_path ellipse
  --data_dir "$DATA_DIR/" --result_dir "$RESULT_DIR"
  --compression_sim --entropy_model_opt --rd_lambda "$LAMBDA" --rate "$RATE"
  --batch_size 1 --GOP_size 60 --knn --n_knn 8 --start_frame 0
  --steps_scaler 0.1
  --ap_freeze_step "$FREEZE_STEP"
  --entropy_steps.quats "$ENTROPY_STEP"
  --entropy_steps.scales "$ENTROPY_STEP"
  --entropy_steps.opacities "$ENTROPY_STEP"
  --entropy_steps.anchor_features "$ENTROPY_STEP"
  --entropy_steps.offsets "$ENTROPY_STEP"
  --entropy_steps.factors "$ENTROPY_STEP"
  --entropy_steps.time_features "$ENTROPY_STEP"
  --ap_variant "$VARIANT"
  --ap_provenance_manifest "$MANIFEST"
  --ap_provenance_manifest_sha256 "$MANIFEST_SHA"
)

echo "=== SMOKE PHASE 1: training ($MAX_STEPS steps, freeze @ $FREEZE_STEP) ==="
"$PY" examples/simple_trainer_GIFStream.py "$TYPE" "${COMMON_FLAGS[@]}"

echo "=== SMOKE PHASE 1 CHECKPOINTS ==="
ls -la "$RESULT_DIR/ckpts/"
if [ "$VARIANT" != "official" ]; then
  ls -la "$RESULT_DIR/ap_freeze/"
fi

echo "=== SMOKE PHASE 1.5: frozen training receipt ==="
CKPT="$RESULT_DIR/ckpts/ckpt_$((MAX_STEPS - 1))_rank0.pt"
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

echo "=== SMOKE PHASE 2: end2end codec entry ==="
SCORE_FLAGS=()
if [ "$VARIANT" != "official" ]; then
  SCORE_FLAGS=(--ap_score_path "$RESULT_DIR/ap_freeze/reference_scores_step${FREEZE_STEP}.npz")
fi
"$PY" examples/simple_trainer_GIFStream.py "$TYPE" "${COMMON_FLAGS[@]}" \
  --ckpt "$CKPT" \
  --h007_training_receipt "$RECEIPT" \
  --h007_training_receipt_sha256 "$RECEIPT_SHA" \
  --compression end2end "${SCORE_FLAGS[@]}"

echo "=== SMOKE PHASE 3: deep artifact verification ==="
"$PY" "$REPO_ROOT/hpc_setup/smoke_verify.py" \
  --result-dir "$RESULT_DIR" --variant "$VARIANT" \
  --freeze-step "$FREEZE_STEP" --max-steps "$MAX_STEPS" \
  --manifest-sha256 "$MANIFEST_SHA" --min-psnr 14.0

echo "SMOKE_OK scene=$SCENE variant=$VARIANT rate=$RATE"
