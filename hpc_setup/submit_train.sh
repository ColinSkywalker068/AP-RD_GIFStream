#!/bin/bash
# Submit H007 training runs on torch.  Usage:
#   ./submit_train.sh                                  # dev acceptance pair:
#                                                      #   official + ap-gifstream-full,
#                                                      #   flame_salmon_1, rate 0, GOP 0
#   VARIANTS="ap-gifstream-full" RATES="0 1 2 3" GOPS="0 1 2 3 4" ./submit_train.sh
#   N_KNN=0 VARIANTS="ap-gifstream-full" ./submit_train.sh   # task-2 ablation
# Picks the account/partition with the earliest --test-only start estimate.
set -euo pipefail
cd "$(dirname "$0")"

SCENE="${SCENE:-flame_salmon_1}"
VARIANTS="${VARIANTS:-official ap-gifstream-full}"
RATES="${RATES:-0}"
GOPS="${GOPS:-0}"
N_KNN="${N_KNN:-8}"
ACCOUNT="${ACCOUNT:-torch_pr_69_general}"
PARTITIONS="${PARTITIONS:-l40s_public h200_public}"

pick_partition() {
  local best="" best_time="9999-99-99T99:99:99"
  for p in $PARTITIONS; do
    local est
    est=$(sbatch --test-only --account="$ACCOUNT" -p "$p" stage5_train.sbatch 2>&1 \
      | grep -oP 'start at \K\S+' | head -1) || true
    if [ -n "${est:-}" ] && [[ "$est" < "$best_time" ]]; then
      best_time="$est"; best="$p"
    fi
  done
  [ -n "$best" ] || { echo "no eligible partition" >&2; exit 1; }
  echo "$best"
}

PARTITION=$(pick_partition)
echo "partition: $PARTITION (account $ACCOUNT)"

for variant in $VARIANTS; do
  for rate in $RATES; do
    for gop in $GOPS; do
      jid=$(sbatch --parsable --account="$ACCOUNT" -p "$PARTITION" \
        --job-name="gif-${variant}-r${rate}-g${gop}" \
        --export=ALL,SCENE="$SCENE",VARIANT="$variant",RATE="$rate",GOP_ID="$gop",N_KNN="$N_KNN" \
        stage5_train.sbatch)
      echo "submitted $jid: scene=$SCENE variant=$variant rate=$rate gop=$gop n_knn=$N_KNN"
    done
  done
done
