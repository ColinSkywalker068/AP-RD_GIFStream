# Task 1 — D_path evaluation results (v3 campaign, flame_salmon_1)

First complete run of the composed path metric the handover's task 1 asked
for, over the full dev matrix (official vs ap-gifstream-full × rates 0–3 ×
GOPs 0–4, campaign manifest `1c80343c…`). Produced by
`eval_scripts/334_v3_paired_reference_path_evaluation.py`; raw outputs in
`results/h007/flame_salmon_1/eval_v3/path_eval_dualpi_r{0..3}.json` on torch.

## Definition as implemented

`D_path = Σ_matched w_i · e_i + Σ_missing w_i · π`, `w_i = 1`, where `e_i` is
the per-identity MTE (mean per-frame L2 against the paired own-reference
path, 60 frames; matching by canonical-ID triple; duplicates count missing —
all verbatim from evaluator 234). π is computed from the frozen reference
only (outcome-blind) and printed per GOP; reported per identity and
bbox-diagonal-normalized; global and top-10%-motion subsets; sweep ×0.5/×1/×2.

## Finding 1 — the handover's π is degenerate on this scene (and that is fine)

"π = median motion of the frozen reference set" evaluates to **exactly 0 in
every GOP, both arms**: >50% of anchors have zero decoded motion (the codec's
factor gates zero static anchors — motion sparsity is the paper's own
premise). Under the literal definition, missing identities cost nothing.

This produces a *stronger* claim, not a weaker one: official misses
~29–32% of identities (slope of D_path in π = missing count > 0), AP misses
none (slope = 0), and AP's matched-error intercept is already smaller —
therefore **AP's D_path is below official's for every π ≥ 0**, by algebra;
the sweep merely confirms it. For a meaningful motion scale we additionally
report π_moving = median over identities that move at all
(≈ 0.08–0.28 world units per GOP). Direction is stable across all three π
definitions and all multipliers, at every rate and both subsets.

## Finding 2 — official deletes a large share of the scene's *motion*

At r0/GOP0 (representative): 29,594 of 70,614 reference identities are
missing after the official codec, and those identities carry **41.1% of the
scene's total motion mass** (948.95 of 2311.60 action units). AP restores
100% of identities at every rate/GOP (missing = 0).

## D_path table (bbox-normalized, per identity, π = moving-median ×1, equal-GOP mean)

| Rate | Subset | official | AP | ratio |
|---|---|---|---|---|
| 0 | global | 0.000179 | 0.000002 | ~90× |
| 0 | top10 | 0.001127 | 0.000015 | ~75× |
| 1 | global | 0.000154 | 0.000004 | ~39× |
| 1 | top10 | 0.001029 | 0.000041 | ~25× |
| 2 | global | 0.000133 | 0.000005 | ~27× |
| 2 | top10 | 0.000777 | 0.000051 | ~15× |
| 3 | global | 0.000122 | 0.000004 | ~31× |
| 3 | top10 | 0.000881 | 0.000041 | ~21× |

(Under the degenerate full-set-median π the same table holds with slightly
smaller official values — matched errors only — and identical direction.)

## Finding 3 — rate parity (real zip bytes, never estimates)

AP archives are larger than official at the same λ (structural: mask/identity
sidecars, protected-scope fine quantization, two-class factor streams; the
method's byte-exact invariant governs the temporal-stream exchange, not the
whole container — the author's own 223 protocol anticipated this with its
MATCHED_RATE / AP_HIGHER_RATE strata):

| Rate | official (5-GOP) | AP | overhead |
|---|---|---|---|
| 0 | 13,469,519 B | 15,712,316 B | +16.7% |
| 1 | 11,436,797 B | 13,298,303 B | +16.3% |
| 2 | 9,072,125 B | 10,434,531 B | +15.0% |
| 3 | 7,303,103 B | 8,256,741 B | +13.1% |

**Bytes-matched ladder comparison** (AP at the next rate point vs official —
AP at strictly *fewer* real bytes in every pair):

| Pair | bytes (AP vs official) | D_path global (AP vs official) |
|---|---|---|
| AP r1 vs official r0 | 13.30 MB < 13.47 MB | 0.000004 ≪ 0.000179 |
| AP r2 vs official r1 | 10.43 MB < 11.44 MB | 0.000005 ≪ 0.000154 |
| AP r3 vs official r2 | 8.26 MB < 9.07 MB | 0.000004 ≪ 0.000133 |

So the claim survives its strongest form: **at equal or fewer on-disk bytes,
AP preserves every identity and reduces D_path by 1–2 orders of magnitude.**

## Caveats / open items

- Single scene (dev protocol), single seed per cell; confirmatory scenes are
  future compute.
- Render-quality (PSNR/SSIM/LPIPS) pairing for the ladder comparison not yet
  assembled into one table (numbers exist in clean-decode manifests / eval
  stats; small aggregation task).
- The π-degeneracy finding and the "both paying-side sites" dual-key scope
  should be ratified by the original author.
- DP infeasibility tally for task-4 reporting so far: 4 infeasible draws
  across ~26 AP training attempts, all at r2–r3; r3/GOP4 the hotspot
  (~50/50 per draw).
