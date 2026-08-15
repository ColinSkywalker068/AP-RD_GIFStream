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

---

## 9. Pitfall catalog

Everything that actually bit us, by category, as symptom → cause → resolution.
Most of these are invisible until you hit them; all are now either fixed in
code, encoded in the `hpc_setup/` scripts, or listed here as operating lore.

### Provenance / contract pitfalls

1. **Manifest bytes must be canonical JSON, not just correct JSON.**
   Symptom: stage 6 rejects every archive ("manifest bytes are not canonical
   JSON") after 40 training runs passed. Cause: the trainer embeds the
   manifest's *raw file bytes* into each archive; the sequence container
   demands `canonical_json_bytes` formatting, but every earlier check only
   verifies the sha of whatever bytes exist — so a pretty-printed manifest
   sails through training and fails at the very last consumer. Resolution:
   canonicalize the file (content identical, new sha); officials only needed
   codec re-mint, AP cells needed retraining (below). Note: the handover's own
   patch8 manifest ends with a trailing newline and would fail the same check
   — this path had never been exercised on real archives.
2. **AP checkpoints pin the manifest sha at freeze time.** Any manifest change
   (even byte formatting) orphans every AP checkpoint: restore fails closed
   with "checkpoint AP runtime provenance differs from active preregistration".
   That check is the tamper-proofing working as designed — the fix is
   retraining, never weakening the check or editing checkpoints. Officials
   don't pin the manifest and survive with a codec-only re-run (stage 5b).
3. **In-tree builds poison the normalized tree hash.** MLEntropy's cmake
   *downloads pybind11 sources into `build/`* (148 files matching the hashed
   suffixes). Any in-tree build under `examples/gsplat/third_party` changes
   the code-tree hash and kills provenance verification. All builds go
   out-of-tree; the POST_BUILD step drops only `.so` files (not hashed) into
   the tree.
4. **The run tree must be a git checkout at the official commit** — the
   trainer literally runs `git rev-parse HEAD` and compares to `c9848663`.
   The handover ships no inner `.git`; transplant one from an official clone
   and keep the AP work as uncommitted working-tree modifications.
5. **The patch-chain length was pinned as literal `9` in five places**
   (container ×3, trainer compression path, clean decoder, certification) plus
   stage-name lists in two more. Adding patch stages means sweeping for every
   pin — now centralized as `PATCH_CHAIN_LENGTH`, but future stages must still
   update tests/fixtures and both stage-name lists.
6. **`eval_scripts/223_*` and the export script pin `ap_training_state.v2`.**
   They are frozen evaluators for the author's archived patch8-era runs and
   reject v3 runs by design; new-run evaluation needs v3-aware counterparts.
7. **AppleDouble `._*.patch` cruft** ships in `provenance/patches/` from the
   handover's macOS packaging — glob patterns must filter it or patch
   enumeration breaks.

### Handed-over code defects (fixed via patch chain)

8. **In-place autograd bug** in `normalize_factor_semantics`: clone +
   column-writes invalidated `torch.maximum`'s saved tensors — the
   path-alignment loss could never backprop, on any torch version. Rewritten
   column-wise (bit-identical forward, 2000-trial check).
9. **Stale test fixture** (8 vs 9 patch hashes) — the "all tests green"
   handover claim was false as shipped.
10. **Cross-device `torch.equal`** in the AP decompress closure check: decoded
    anchors live on CPU, AP masks are read to GPU. The module's own
    `_tensor_equal_device_agnostic` helper existed but wasn't used there.
11. **Duplicate anchors, twice.** (a) Densification: a grow candidate can land
    float-exactly on an existing anchor from a different hierarchy level —
    upstream tolerates duplicates, the deterministic-KNN/canonical-ID
    contracts don't; stochastic (GPU-nondeterministic), so it fires on some
    runs and not others. Fixed with an exact-collision filter in the grow
    step. (b) Official codec decode: the lossy anchor round-trip (16-bit PNG +
    voxel re-round) can merge two anchors — inherent to that bitstream, so
    render-time KNN opts into duplicates with a deterministic tie-break while
    identity-bearing paths stay strict.
12. **Upstream argparse crash** in `dataset_process/n3d_video_process.py`
    (`type=bool` with `action='store_true'`) — the stock preprocessing script
    cannot start; a minimally-fixed copy lives in `hpc_setup/`.

### Method behavior that looks like a bug but is not

13. **Byte-exact subset-sum infeasibility.** AP runs can die at the freeze
    with "no exact estimated-byte adjustment subset" — one of the two
    *documented* DP failure modes (HANDOVER task 4 requires reporting their
    counts). Concentrated at high rates (coarser quantization → chunkier byte
    costs; observed 2/10 at r2–r3, 0/10 at r0–r1). Feasibility is a fresh draw
    per run (GPU nondeterminism), so a plain retry is legitimate; keep all
    attempts in logs for honest reporting, and treat repeated failure of one
    cell as a finding, not a nuisance.

### Torch HPC operational pitfalls

14. **`sbatch` requires `--account=torch_pr_*`** (the default `users`
    association is invalid), and partition access is account-dependent:
    general accounts see only `*_public`; `tandon_advanced` unlocks
    `h200/a100/h100_tandon`. `QOSGrpGRES` pending reason = the account group's
    GPU quota is exhausted.
15. **`--test-only` estimates are worst-case and lane-dependent.** Every
    l40s_public job so far started via backfill within minutes despite
    multi-hour estimates, while "faster-looking" A100/H200 lanes sat in queue.
    Racing lanes at submit time — and for urgent pairs, hedge-submitting two
    lanes with distinct `EXP_TAG`s and cancelling the loser — has been the
    fastest strategy. `EXP_TAG` exists precisely so parallel hedges cannot
    collide on result paths.
16. **conda-forge CUDA hides headers under `targets/x86_64-linux/`** — torch
    extension builds need `CPATH`/`LIBRARY_PATH`/`LD_LIBRARY_PATH` pointed
    there or `cuda_runtime_api.h` is not found (the nvidia-channel layout
    doesn't have this problem).
17. **Compute nodes lack `zip`, and unzip's concat trick corrupts split
    archives silently** — it "extracted" flame_salmon with 7 of 19 cameras and
    a warning-level exit code. Split zips need a real multi-volume extractor
    (static `7zz`); verify extraction by counting content, never by exit code.
18. **Never park working files inside a dataset root** — the preprocessing
    script iterates every child of the root as a scene; a `_zips/` staging dir
    crashed it instantly.
19. **Don't rsync build dirs across machines** — a stale `CMakeCache.txt`
    pointing at the source machine's paths breaks cmake with a confusing
    error.
20. **`.nfs* Device or resource busy` tracebacks in job logs are noise**
    (NFS silly-rename during teardown of files still open) — never the actual
    failure; keep scanning for the real error.
21. **`scancel` can lag under RPC throttling** — a read-back immediately after
    cancelling may still show the jobs PENDING; verify with a fresh `squeue`
    before concluding a cancel failed (this false alarm nearly caused a
    cancel of the *correct* batch).
22. **Cluster-side schedule pins:** trainer smoke-scaling via `--steps_scaler`
    does *not* scale `ap_freeze_step` or the `entropy_steps` dict (fixed
    absolute values) — a naive scale-down silently skips entropy-model
    training entirely. Override per key (`--entropy_steps.time_features N`,
    tyro dict syntax) and keep the frozen phase ratios (1/3 entropy, 2/3
    freeze).
23. **The codec entry demands an external frozen training receipt**
    (`h007_training_receipt` + sha) binding checkpoint bytes, producer config,
    and runtime provenance; the producer config JSON must be regenerated
    through the same tyro parse (`hpc_setup/make_training_config.py`; preset
    literals drift-guarded against the trainer source).

### Multi-machine / session pitfalls

24. **GitHub identity and scopes:** the box's default `gh` login had pull-only
    access; the first Colin token lacked write scope (403 on push despite
    admin permissions). Also, `main`'s history was later rewritten for
    authorship — after any rewrite, secondary clones (torch) diverge with
    same-message/different-sha commits and need `git reset --hard origin/main`
    (verify the content diff is empty first).
25. **A backgrounded client restart can leave a live twin session** working
    the same transcript — ours independently submitted a duplicate 20-job
    batch to the same result paths. Check `ListAgents` after restarts, stand
    twins down explicitly, and treat "who owns the cluster" as
    single-writer state.
26. **SSH `ControlPersist` counts from last use**, not first auth — a busy
    control lane outlives its nominal 12h indefinitely; only idle gaps or
    network drops kill it.
