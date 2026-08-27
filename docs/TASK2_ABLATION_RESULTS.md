# Task 2 — one-hop closure ablation results (campaign M3, flame_salmon_1)

The handover's task 2: ablate the one-hop KNN protection closure ("protect
only 𝒫" vs "𝒫 + one-hop neighbors"), report the measured
|protected-scope| / |retained-set| ratio, and put numbers behind "the most
rate-expensive design choice in the method". Both arms retrained under one
manifest (`dac0239f…`): 20 cells each (rates 0–3 × GOPs 0–4),
`ap-gifstream-full` with `ap_zero_hop_closure` off (`_c3` dirs) vs on
(`_c3zh`). Raw outputs: `results/h007/flame_salmon_1/eval_task2/` on torch.

## Protection scope (the deliverable ratio)

| Arm | scope / retained | rule recorded in receipts |
|---|---|---|
| one-hop (default) | **11.17%** | `protected-plus-one-hop-retained-knn` |
| zero-hop (ablation) | **5.00%** (= protected fraction exactly) | `protected-only-zero-hop` |

The closure more than doubles the fine-quantization scope. Identity
retention is closure-independent by construction (count-preserving swap +
identity restoration): both arms show missing ≈ 0 at every rate.

## D_path (bbox-normalized per identity, π = moving-median ×1, equal-GOP mean)

| Rate | Subset | one-hop | zero-hop | better |
|---|---|---|---|---|
| 0 | global | 0.0000029 | 0.0000031 | one-hop |
| 0 | top10 | 0.0000284 | 0.0000312 | one-hop |
| 1 | global | 0.0000053 | 0.0000039 | zero-hop |
| 1 | top10 | 0.0000527 | 0.0000390 | zero-hop |
| 2 | global | 0.0000040 | 0.0000053 | one-hop |
| 2 | top10 | 0.0000395 | 0.0000528 | one-hop |
| 3 | global | 0.0000054 | 0.0000045 | zero-hop |
| 3 | top10 | 0.0000541 | 0.0000447 | zero-hop |

## Rate (real 5-GOP zip bytes)

| Rate | one-hop | zero-hop | delta |
|---|---|---|---|
| 0 | 14,971,043 B | 15,518,500 B | +3.7% |
| 1 | 13,601,321 B | 13,209,810 B | −2.9% |
| 2 | 10,342,588 B | 10,151,858 B | −1.8% |
| 3 | 8,368,212 B | 8,477,812 B | +1.3% |

## Honest interpretation

**The one-hop closure shows no detectable benefit at these operating
points.** D_path splits 2–2 across rates with small margins in both
directions, and the rate deltas (±4%) also alternate sign — both are within
the run-to-run noise of independently trained cells (GPU-nondeterministic
densification; one seed per cell). Since both arms preserve all identities,
the closure's only channel of effect is neighbor-feature quantization
precision for protected-path decoding — and that effect, if present, is
smaller than training noise here.

Implications for the paper (feeds tasks 4/5/6 writing):

- The closure claim should be stated as *defensive* (it guarantees the
  decoder's protected-path inputs are finely quantized) rather than
  *empirically load-bearing* on this scene; the measured cost of dropping it
  is nil-to-noise, and so is the measured benefit of keeping it.
- With one seed per cell this ablation cannot resolve effects below ~±30%
  of these small D_path values; seed replicates would be needed to tighten
  the bound. Worth discussing with the original author whether to (a) keep
  the closure with the defensive framing, (b) invest in replicates, or
  (c) simplify the method by dropping it.
- Byte-exact DP infeasibility tally across all campaigns so far (task-4
  reporting): 7 infeasible draws / ~70 AP training attempts, all at r2–r3;
  r3/GOP4 alone accounts for 4 of 8 draws at that cell (~50% per draw).

## Caveats

Single scene, single seed per cell; the noise floor claim is qualitative
(sign-alternation across rates), not a formal test. Zero-hop is faithfully
recorded end-to-end (config → receipts → codec contract → decoder
re-verification), so any future replicate campaign is one submit command per
arm.
