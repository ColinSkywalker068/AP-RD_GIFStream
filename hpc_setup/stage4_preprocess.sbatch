#!/bin/bash
#SBATCH --job-name=neur3d-preproc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/yz11445/AP-RD_GIFStream/hpc_setup/logs/stage4_%j.out

set -euxo pipefail
source /scratch/yz11445/AP-RD_GIFStream/hpc_setup/env_common.sh

CONDA="$MAMBA_ROOT/bin/conda"
DATA_ROOT="$SCRATCH_ROOT/datasets/Neur3D"
export PATH="$ENV_PREFIX/bin:$PATH"

# Tools for preprocessing: colmap (CUDA build preferred, for GPU SIFT on this
# node) and ffmpeg.  TUNA conda-forge mirror first, official fallback.
if ! command -v colmap >/dev/null || ! command -v ffmpeg >/dev/null; then
  "$CONDA" install -y -n GIFStream --override-channels -c "$CF_CHANNEL" "colmap=*=cuda*" ffmpeg \
    || "$CONDA" install -y -n GIFStream --override-channels -c conda-forge "colmap=*=cuda*" ffmpeg \
    || "$CONDA" install -y -n GIFStream --override-channels -c conda-forge colmap ffmpeg
fi
command -v colmap ffmpeg

# Pre-cache LPIPS metric weights (torch hub cache lives under the shimmed
# HOME, i.e. scratch) so training/eval nodes never stall on this download.
"$PY" - <<'PYEOF' || echo "lpips prefetch skipped (non-fatal)"
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
for net in ("vgg", "alex"):
    LPIPS(net_type=net)
print("lpips weights cached")
PYEOF

# Frame extraction (ffmpeg, 300 frames/cam) + per-GOP COLMAP for every scene.
# Uses the minimally-fixed wrapper (upstream argparse bug); the GIFStream_APRD
# tree itself is untouched.
cd "$SCRATCH_ROOT/AP-RD_GIFStream/hpc_setup"
"$PY" n3d_video_process_fixed.py --root_dir "$DATA_ROOT" --extract_frames

# Verify: every scene needs png frames per camera and one colmap_<f> dir per GOP.
fail=0
for scene in coffee_martini cook_spinach cut_roasted_beef flame_salmon_1 flame_steak sear_steak; do
  d="$DATA_ROOT/$scene"
  cams=$(find "$d/png" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  frames=$(find "$d/png" -name '*.png' 2>/dev/null | wc -l)
  gops=$(find "$d" -maxdepth 1 -type d -name 'colmap_*' 2>/dev/null | wc -l)
  echo "SCENE $scene: cam_dirs=$cams pngs=$frames colmap_gops=$gops"
  if [ "$cams" -lt 15 ] || [ "$gops" -lt 5 ]; then fail=1; fi
done
du -sh "$DATA_ROOT"
[ "$fail" = 0 ] || { echo STAGE4_INCOMPLETE; exit 1; }
echo STAGE4_OK
