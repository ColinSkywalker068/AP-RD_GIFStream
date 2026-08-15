# AP-RD / GIFStream — Implementation Progress

Status as of 2026-08-14. Covers everything done since taking over the handover
(`4c2a73e`), on both machines:

- **4090 workstation** (`asus4090`, `~/Research/AP-RD_GIFStream`) — development,
  tests, provenance authoring.
- **NYU torch HPC** (`yz11445`, `/scratch/yz11445/AP-RD_GIFStream`) — env,
  dataset, preprocessing, training runs. Nothing is written to the HPC home
  directory; jobs run with `HOME` shimmed into scratch.

The three copies (4090, GitHub `ColinSkywalker068/AP-RD_GIFStream`, torch) are
kept in lockstep on `main`; every change below is committed and pushed.

---

## 1. Task status (numbering = HANDOVER.md §6)

| Task | Status |
|---|---|
| **3 — donor dual-key** | **Code complete, tested, verified in a real run.** |
| **2 — closure ablation (`n_knn` 8→0)** | Not started; unblocked — one command (`N_KNN=0 ./submit_train.sh`) once the main matrix exists. |
| **1 — D_path evaluator** | Not started; independent, next code item. |
| **4/5/6 — writing items** | Untouched by design (paper-writing time). |
| Onboarding step 1 (env + tests green) | Done on both machines (see §3, §5). |
| Onboarding step 2 (patch-chain replay + hash check) | Done — replays bit-exactly. |
| Onboarding step 3 (drift → patch9) | Done — plus patch10 (see §4). |
| Onboarding step 4 (dev training + plausible audits) | **Done — acceptance pair complete, 45/45 verifier checks** (see §7). |

### Task 3 details

- `_lexicographic_rank` (`gsplat/compression/ap_gifstream.py`) gained an
  optional secondary key: **motion score asc → backbone importance asc →
  canonical ID**. Applied on the *paying side only* (count-swap donors and
  byte-DP demote candidates); the protected-class top-k selection never sees
  importance (outcome-blind defense preserved; a dedicated test proves the
  class boundary is unchanged with/without the signal).
- Importance = `frozen_backbone_importance()`: the backbone's own prune
  statistic (peak/mean-blended accumulated rendered opacity ÷ visit count from
  the densification strategy state), recorded before the AP freeze. No new
  learned estimator.
- Provenance threading: score artifact gained `importance_score` +
  `importance_definition` members (`h007.ap_scores.v2 → v3`; training state
  `v2 → v3`); container member set/order/dtype registries updated; audits now
  record `donor_ranking_keys`; `load_aligned_score_artifact` validates and
  replays the dual-key allocation in its reproducibility check.
- Tests: 6 new cases (score-tie demotion in both builders, dual-tie
  canonical-ID adjudication, protected-class blindness, prune-statistic
  hand-check, validation guards).

---

## 2. Defects found in the handed-over code (all fixed via the patch chain)

1. **In-place autograd bug** — `normalize_factor_semantics`
   (`h007_path_contract.py`) built its output with in-place column writes,
   invalidating `torch.maximum`'s saved input; the path-alignment loss could
   not backprop. Rewritten column-wise; forward verified bit-identical over
   2000 randomized trials. (Caught by an existing test that could never have
   passed as shipped — on any torch version.)
2. **Stale test fixture** — score-artifact fixture had 8 patch-chain hashes
   where the receipt helper expects the full chain.
3. **Hardcoded nine-stage pins** — patch-chain length was pinned as literal
   `9` in five places (container ×3, trainer compression path, clean decoder,
   certification). Now `PATCH_CHAIN_LENGTH` (= 11).
4. **Cross-device compare in the decoder** — decoded anchors live on CPU while
   AP masks are read to the GPU; patch8's closure re-verification used bare
   `torch.equal` and crashed. Now uses the module's own
   `_tensor_equal_device_agnostic`.
5. **Duplicate-anchor collisions** (two related, both stochastic — GPU
   nondeterminism decides if/when they fire, which is why the original author
   never hit them):
   - *Densification*: a grow candidate can land float-exactly on an existing
     anchor from another hierarchy level; upstream tolerated duplicates, the
     deterministic-KNN contract (and canonical voxel IDs at freeze) do not.
     The grow step now drops exact-collision candidates (applies to both study
     arms equally).
   - *Official codec decode*: the official variant's lossy anchor round-trip
     (16-bit PNG + voxel re-round) can merge two anchors — duplicates are
     inherent to that bitstream. Render-time KNN (`find_k_neighbors`) now opts
     in via `allow_duplicate_rows=True` with a deterministic candidate-index
     tie-break (provably a no-op for unique rows); identity-bearing contract
     paths stay strict. (The AP path is immune by design: patch8 makes
     identity the coordinate authority.)

Also noted: `provenance/patches/` contains macOS `._*` AppleDouble cruft from
the handover packaging (harmless, filtered by tooling; candidate for deletion),
and `dataset_process/n3d_video_process.py` has an argparse bug
(`type=bool` + `store_true`) that crashes at startup — worked around with a
minimally-fixed copy in `hpc_setup/` so the provenance tree stays untouched.

---

## 3. Test suite

**142 passed + 81 subtests, zero failures** (from 2 failures at handover), on
both the 4090 and a torch GPU node. New tests cover the dual-key mechanics and
the duplicate-tolerant KNN behavior.

---

## 4. Provenance chain (now 11 stages)

- **patch9** = the author's 2026-08-03 local drift, path-normalized to the
  inner tree.
- **patch10** = everything since the handover commit (task 3, all fixes in §2,
  schema bumps, the 11-stage change itself). Regenerated via
  `git diff 4c2a73e --relative=GIFStream_APRD` whenever the tree changes.
- Current preregistered manifest:
  `provenance/h007_ap_gifstream_u3_patch_chain_patch10_dualkey_20260814.json`
  - sha256 `8e6894607d06cc474daeda4da1829d7cfc2624e588941d3392e0d8c43528d1a9`
  - normalized tree `399b4bb1…`, 645 files
- Verified end-to-end on both machines: official GIFStream @ `c9848663` + all
  11 patches replays **bit-identically** to the working tree, and
  `verify_runtime_provenance()` accepts the manifest.
- Two operational discoveries encoded in the tooling:
  - The trainer requires the run tree to be a git checkout **at the official
    commit** with AP work as uncommitted modifications → torch's
    `GIFStream_APRD/` carries a transplanted `.git` at `c9848663`.
  - The normalized tree hash sweeps source files under
    `examples/gsplat/third_party`, so **in-tree builds poison it** (MLEntropy's
    cmake downloads pybind11 sources into `build/`). All builds are
    out-of-tree; the needed `.so`s land in `entropy_models/` (not hashed).

**Rule going forward**: any `GIFStream_APRD/` edit ⇒ regenerate patch10 +
manifest, update `MANIFEST_SHA` in `hpc_setup/stage5_*.sbatch`, re-verify.

---

## 5. Torch HPC infrastructure (`hpc_setup/`, all SLURM jobs)

| Stage | Script | Status |
|---|---|---|
| 1 | `stage1_cpu.sbatch` — miniforge + py3.10 env + CUDA 12.4 toolchain + torch 2.6.0+cu124 + requirements + MLEntropy (out-of-tree) | ✅ |
| 2 | `stage2_gpu.sbatch` — gsplat/gridencoder/fused-ssim compiled for sm 8.0/8.9/9.0 + full pytest gate on GPU | ✅ 141-green on node |
| 3 | `stage3_dataset.sbatch` — all Neur3D v1.0 assets (GitHub API-driven), split-zip via static `7zz`, per-scene verification | ✅ 6 scenes |
| 4 | `stage4_preprocess.sbatch` — conda colmap (CUDA SIFT) + ffmpeg, LPIPS weight prefetch, frame extraction + 5 per-GOP COLMAPs per scene | ✅ 30/30 GOPs |
| 5 | `stage5_train.sbatch` + `submit_train.sh` — one (scene, variant, rate, GOP, n_knn, exp_tag) run: training → frozen training receipt → end2end codec entry | ✅ proven |
| — | `stage5_smoke.sbatch` + `smoke_verify.py` — real pipeline at 1/10 schedule (phase ratios preserved) + ~45-check artifact interrogation | ✅ 45/45 |
| — | `make_training_config.py` — reproduces `producer_training_config` via the same tyro parse (preset drift-guarded) for the receipt generator | ✅ |

Conventions baked in: TUNA mirror first with official fallback; every submission
races account:partition lanes with `sbatch --test-only` (both general accounts
on `l40s_public`/`h200_public`, `tandon_advanced` on `h200/a100/h100_tandon`);
`EXP_TAG` suffixes result dirs/job names so parallel lane hedges never collide.
Practical note: **`l40s_public` backfill has beaten every A100/H200 estimate so
far** — estimates are worst-case, and hedge-submitting both lanes then
cancelling the loser has been the fastest strategy.

Storage (all under `/scratch/yz11445`): repo, `miniforge3/envs/GIFStream`,
`datasets/Neur3D/<scene>/{cam*.mp4,png/,colmap_*/}`, archives in
`datasets/neur3d_zips/`, results in `results/h007/...`, smoke in
`results/smoke/...`.

---

## 6. Smoke test (design + result)

`stage5_smoke.sbatch` runs the *real* trainer + codec entry at 1/10 schedule
with every phase ratio preserved (entropy opt at 1/3, AP freeze at 2/3 —
`--entropy_steps.*` overridden per key because `--steps_scaler` doesn't scale
them). `smoke_verify.py` then interrogates artifacts: importance statistics,
count/byte preservation invariants, an independent allocation replay demanding
bit-exact mask equality, archive `testzip` + audit-sha cross-checks,
outcome-blind feedback record, and a PSNR floor. **45/45 checks pass.** The
smoke caught defects §2.3–2.5 before they could burn full-length runs.

---

## 7. Acceptance runs (ONBOARDING step 4 — done)

Dev pair on flame_salmon_1, rate 0, GOP 0, `n_knn=8`, L40S (~25 min each):

- **official**: STAGE5_OK, full 30k training + codec entry, archive produced.
- **ap-gifstream-full**: STAGE5_OK; deep verifier on the real run: **45/45**.
  - Temporal exchange: promoted 1581 / demoted 1018 rows at exactly equal
    estimated byte mass (259,605 B both sides).
  - Whole-anchor swap: promoted = demoted = 0 (encoder already retained all
    3540 protected anchors = exactly 5% of 70,797 eligible).
  - `donor_ranking_keys = [score_asc, backbone_importance_asc, canonical_id]`
    in both audits; importance positive for 99.9% of anchors.
  - PSNR 28.4 (compressed) / 28.2 (val) at the lowest rate point.

Results: `results/h007/flame_salmon_1/{official,ap-gifstream-full}/nknn8/GOP_0/r0_l40s/`.

---

## 8. What remains

1. **Full dev matrix**: 2 variants × rates 0–3 × GOPs 0–4 (40 runs ≈ 17
   GPU-hours; one `submit_train.sh` call). AP variants never continue-train;
   each GOP is an independent run by contract.
2. **Task 2 ablation**: `N_KNN=0 VARIANTS=ap-gifstream-full ./submit_train.sh`
   against the same matrix (after checking the `retained_rows.numel() <= count`
   guard behavior at 0).
3. **Task 1**: D_path evaluator extending `eval_scripts/234_*.py`
   (w_i = 1, π = median frozen-reference motion, π ∈ 0.5×–2× sweep, clean-decode
   inputs only). Note: `eval_scripts/223_*` + the export script still pin
   `ap_training_state.v2` — they verify the author's archived patch8-era runs
   and were left untouched; new-run evaluation needs v3-aware counterparts.
4. **Sequence containers + clean decode**: `h007_sequence_container.py build`
   over each 5-GOP set, then `h007_clean_decode_gifstream.py` (stage-6 script
   to write once GOP_1–4 exist).
5. **Paper-time**: tasks 4/5/6; optional entropy-model transient diagnostic.
6. Housekeeping: push access for the torch clone if desired (currently pushed
   from the 4090), deleting the `._*` cruft in `provenance/patches/`.
