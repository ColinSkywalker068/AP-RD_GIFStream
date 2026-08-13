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

# flame_salmon_1 ships as a multi-volume zip (.z01-.z03 + .zip); unzip cannot
# process it reliably, so use the static 7zz binary, which reads split zip
# volumes natively.  Self-heal: wipe any partial extraction and redo it.
FS_DIR="$DATA_ROOT/flame_salmon_1"
fs_cams=$(find "$FS_DIR" -maxdepth 1 -name 'cam*.mp4' 2>/dev/null | wc -l)
if [ "${fs_cams:-0}" -lt 15 ]; then
  rm -rf "$FS_DIR" flame_salmon_1_merged.zip flame_salmon_1_joined.zip \
    "$ZIP_DIR/.extracted_flame_salmon_1_merged.zip"
  SEVENZ="$SCRATCH_ROOT/bin/7zz"
  if [ ! -x "$SEVENZ" ]; then
    mkdir -p "$SCRATCH_ROOT/bin"
    curl -fL --retry 5 -o "$TMPDIR/7z.tar.xz" \
      https://github.com/ip7z/7zip/releases/download/24.09/7z2409-linux-x64.tar.xz
    tar -xJf "$TMPDIR/7z.tar.xz" -C "$TMPDIR" 7zz
    mv "$TMPDIR/7zz" "$SEVENZ" && chmod +x "$SEVENZ"
  fi
  "$SEVENZ" x -y -o"$DATA_ROOT" flame_salmon_1_split.zip
fi

# Extract the single-volume scenes.
for z in *.zip; do
  case "$z" in
    flame_salmon_1_split.zip|flame_salmon_1_joined.zip|flame_salmon_1_merged.zip) continue ;;
  esac
  marker="$ZIP_DIR/.extracted_$z"
  [ -f "$marker" ] && continue
  unzip -o -q "$z" -d "$DATA_ROOT"
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
