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

# 4-way parallel download; skip files that already exist (all-or-nothing per
# file: a failed curl leaves a partial file removed so the rerun refetches).
echo "$assets" | xargs -n1 -P4 -I{} bash -c '
  f=$(basename "{}")
  [ -s "$f" ] && exit 0
  curl -fL --retry 5 --retry-delay 15 -o "$f.part" "{}" && mv "$f.part" "$f" \
    || { rm -f "$f.part"; echo "FAILED $f"; exit 1; }
'

# Reassemble the flame_salmon_1 split archive into a single zip.  Prefer a
# real zip binary; fall back to plain concatenation, which unzip can process
# with its zip-bomb heuristic disabled (the standard recipe for this dataset).
if [ -f flame_salmon_1_split.z01 ] && [ ! -f flame_salmon_1_merged.zip ]; then
  if command -v zip >/dev/null; then
    zip -q -s 0 flame_salmon_1_split.zip --out flame_salmon_1_merged.zip
  else
    cat flame_salmon_1_split.z0* flame_salmon_1_split.zip > flame_salmon_1_joined.zip
    mv flame_salmon_1_joined.zip flame_salmon_1_merged.zip
  fi
fi

# Extract every scene (skip the split parts; use the merged archive instead).
# unzip exits nonzero on concatenation warnings even when extraction succeeds,
# so success is judged by the layout report below, not the exit code.
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
for z in *.zip; do
  case "$z" in
    flame_salmon_1_split.zip|flame_salmon_1_joined.zip) continue ;;
  esac
  marker="$ZIP_DIR/.extracted_$z"
  [ -f "$marker" ] && continue
  unzip -o -q "$z" -d "$DATA_ROOT" || echo "unzip exit $? for $z (warnings tolerated)"
  touch "$marker"
done

# Verify the layout: every expected scene must have cameras + poses.
fail=0
for scene in coffee_martini cook_spinach cut_roasted_beef flame_salmon_1 flame_steak sear_steak; do
  d="$DATA_ROOT/$scene"
  n=$(find "$d" -maxdepth 1 -name 'cam*.mp4' 2>/dev/null | wc -l)
  p=$(ls "$d"/poses_bounds.npy 2>/dev/null | wc -l)
  echo "SCENE $scene: ${n} cameras, poses_bounds=${p}"
  if [ "$n" -lt 15 ] || [ "$p" -ne 1 ]; then fail=1; fi
done
du -sh "$DATA_ROOT" "$ZIP_DIR"
[ "$fail" = 0 ] || { echo STAGE3_INCOMPLETE; exit 1; }
echo STAGE3_OK
