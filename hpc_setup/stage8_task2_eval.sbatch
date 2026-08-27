#!/bin/bash
#SBATCH --job-name=gif-task2-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/yz11445/AP-RD_GIFStream/hpc_setup/logs/stage8_%j.out

# Task-2 ablation evaluation: containers + clean decodes for both closure
# arms (one-hop _c3 / zero-hop _c3zh), paired own-references via the v3
# exporter with each arm's campaign, then the D_path comparison
# (zero-hop as the AP slot, one-hop as the baseline slot) plus real-zip
# rate accounting per arm.
set -euxo pipefail
source /scratch/yz11445/AP-RD_GIFStream/hpc_setup/env_common.sh

SCENE=flame_salmon_1
RESULTS="$SCRATCH_ROOT/results/h007/$SCENE"
REFS="$RESULTS/references_task2"
EVAL_OUT="$RESULTS/eval_task2"
CAMPAIGN_ONE="$REPO_ROOT/eval_scripts/h007_v3_campaign_flame_salmon.json"
CAMPAIGN_ZERO="$REPO_ROOT/eval_scripts/h007_v3_campaign_flame_salmon_zerohop.json"

export CUDA_HOME="$ENV_PREFIX"
export PATH="$ENV_PREFIX/bin:$PATH"
export CPATH="$ENV_PREFIX/targets/x86_64-linux/include"
export LD_LIBRARY_PATH="$ENV_PREFIX/targets/x86_64-linux/lib:$ENV_PREFIX/lib"
cd "$REPO_ROOT/GIFStream_APRD"
git rev-parse HEAD | grep -q c98486632e7dafd830740b1a1692bd08c48b96e3 \
  || { echo "FATAL: tree not at official commit"; exit 1; }
mkdir -p "$EVAL_OUT"

# 1) containers + clean decodes for both arms (stage6 body, tag-scoped).
for tag in c3 c3zh; do
  VARIANTS="ap-gifstream-full" RATES="0 1 2 3" EXP_TAG="$tag" \
    bash "$REPO_ROOT/hpc_setup/stage6_container_decode.sbatch"
done

# 2) paired own-references per arm (campaign matches the arm's rule).
for spec in "c3:$CAMPAIGN_ONE" "c3zh:$CAMPAIGN_ZERO"; do
  tag="${spec%%:*}"; campaign="${spec##*:}"
  for rate in 0 1 2 3; do
    for gop in 0 1 2 3 4; do
      BUNDLE="$REFS/$tag/r${rate}/gop_${gop}"
      [ -d "$BUNDLE" ] && continue
      RESULT_DIR="$RESULTS/ap-gifstream-full/nknn8/GOP_${gop}/r${rate}_${tag}"
      mkdir -p "$REFS/$tag/r${rate}"
      rm -f "$REFS/$tag/r${rate}/gop_${gop}.summary.json"
      "$PY" "$REPO_ROOT/eval_scripts/h007_export_ap_same_run_reference_v3.py" \
        --campaign "$campaign" --rate "$rate" \
        --checkpoint "$RESULT_DIR/ckpts/ckpt_29999_rank0.pt" \
        --compression-archive "$RESULT_DIR/compression_rank0.zip" \
        --gop-id "$gop" \
        --output-bundle "$BUNDLE" \
        --summary "$REFS/$tag/r${rate}/gop_${gop}.summary.json"
    done
  done
done

# 3) D_path comparison per rate: zero-hop (AP slot) vs one-hop (baseline slot).
for rate in 0 1 2 3; do
  OUT="$EVAL_OUT/ablation_r${rate}.json"
  rm -f "$OUT"
  "$PY" "$REPO_ROOT/eval_scripts/334_v3_paired_reference_path_evaluation.py" \
    --campaign "$CAMPAIGN_ONE" --rate "$rate" \
    --repo-root "$REPO_ROOT/GIFStream_APRD" \
    --official-reference-root "$REFS/c3/r${rate}" \
    --ap-reference-root "$REFS/c3zh/r${rate}" \
    --official-decode-root "$RESULTS/ap-gifstream-full/nknn8" \
    --ap-decode-root "$RESULTS/ap-gifstream-full/nknn8" \
    --official-slot-method "ap-onehop-closure" \
    --official-slot-is-ap \
    --ap-slot-method "ap-zerohop-closure" \
    --official-decode-tag c3 \
    --ap-decode-tag c3zh \
    --output "$OUT"
done

# 4) real-zip rate accounting per arm.
"$PY" - <<'PYEOF'
import json
from pathlib import Path
base = Path("/scratch/yz11445/results/h007/flame_salmon_1/ap-gifstream-full/nknn8")
rows = []
for rate in (0, 1, 2, 3):
    entry = {"rate": rate}
    for tag in ("c3", "c3zh"):
        total = sum(
            (base / f"GOP_{g}" / f"r{rate}_{tag}" / "compression_rank0.zip").stat().st_size
            for g in range(5)
        )
        entry[tag] = total
    entry["zerohop_minus_onehop_bytes"] = entry["c3zh"] - entry["c3"]
    entry["zerohop_over_onehop_ratio"] = entry["c3zh"] / entry["c3"]
    rows.append(entry)
    print(f"r{rate}: onehop={entry['c3']}B zerohop={entry['c3zh']}B "
          f"delta={entry['zerohop_minus_onehop_bytes']:+d}B ratio={entry['zerohop_over_onehop_ratio']:.4f}")
out = Path("/scratch/yz11445/results/h007/flame_salmon_1/eval_task2/rate_parity_ablation.json")
out.write_text(json.dumps({"schema": "h007.task2_rate_parity.v1", "rates": rows}, indent=1, sort_keys=True))
PYEOF

ls -la "$EVAL_OUT"
echo STAGE8_OK
