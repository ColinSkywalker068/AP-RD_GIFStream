#!/bin/bash
#SBATCH --job-name=neur3d-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/yz11445/AP-RD_GIFStream/hpc_setup/logs/stage3_%j.out

set -euxo pipefail
source /scratch/yz11445/AP-RD_GIFStream/hpc_setup/env_common.sh

DATA_ROOT="$SCRATCH_ROOT/datasets/Neur3D"
ZIP_DIR="$DATA_ROOT/_zips"
mkdir -p "$ZIP_DIR"
cd "$ZIP_DIR"

# All assets of the v1.0 release (no hardcoded file list).
assets=$(curl -fsSL --retry 5 \
  https://api.github.com/repos/facebookresearch/Neural_3D_Video/releases/tags/v1.0 \
  | grep -oP '"browser_download_url":\s*"\K[^"]+')
echo "$assets"
test -n "$assets"

# 4-way parallel download with resume; skip files already complete.
echo "$assets" | xargs -n1 -P4 -I{} bash -c '
  f=$(basename "{}")
  curl -fL -C - --retry 5 --retry-delay 15 -o "$f" "{}" || { echo "FAILED $f"; exit 1; }
'

# Reassemble the flame_salmon_1 split archive into a single zip.
if [ -f flame_salmon_1_split.z01 ] && [ ! -f flame_salmon_1_merged.zip ]; then
  if command -v zip >/dev/null; then
    zip -q -s 0 flame_salmon_1_split.zip --out flame_salmon_1_merged.zip
  else
    cat flame_salmon_1_split.z0* flame_salmon_1_split.zip > flame_salmon_1_joined.zip
    zip -q -FF flame_salmon_1_joined.zip --out flame_salmon_1_merged.zip || \
      mv flame_salmon_1_joined.zip flame_salmon_1_merged.zip
  fi
fi

# Extract every scene (skip the split parts; use the merged archive instead).
for z in *.zip; do
  case "$z" in
    flame_salmon_1_split.zip|flame_salmon_1_joined.zip) continue ;;
  esac
  unzip -o -q "$z" -d "$DATA_ROOT"
done

# Report the layout: every scene directory with its camera count.
for d in "$DATA_ROOT"/*/; do
  [ -d "$d" ] || continue
  case "$d" in */_zips/) continue ;; esac
  n=$(find "$d" -maxdepth 1 -name 'cam*.mp4' | wc -l)
  p=$(ls "$d"/poses_bounds.npy 2>/dev/null | wc -l)
  echo "SCENE $(basename "$d"): ${n} cameras, poses_bounds=${p}"
done
du -sh "$DATA_ROOT" "$ZIP_DIR"
echo STAGE3_OK
