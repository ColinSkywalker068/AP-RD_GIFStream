#!/usr/bin/env python
"""Build same-identity Panoptic trajectory compression variants.

The generated candidates keep the original Gaussian order and tensor shape. They
only replace `means3D` with temporal keyframe/interpolation approximations, so
`panoptic_track_eval.py` can report MTE directly against the reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def interpolate_stride(means: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return means.copy()
    out = np.empty_like(means)
    keys = list(range(0, means.shape[0], stride))
    if keys[-1] != means.shape[0] - 1:
        keys.append(means.shape[0] - 1)
    for a, b in zip(keys[:-1], keys[1:]):
        start = means[a]
        end = means[b]
        denom = max(1, b - a)
        for t in range(a, b + 1):
            alpha = (t - a) / denom
            out[t] = (1.0 - alpha) * start + alpha * end
    return out


def path_masks(means: np.ndarray) -> dict[str, np.ndarray]:
    step_len = np.linalg.norm(np.diff(means, axis=0), axis=-1)
    path_len = step_len.sum(axis=0)
    top10 = path_len >= np.quantile(path_len, 0.90)
    top20 = path_len >= np.quantile(path_len, 0.80)
    return {"top10_path": top10, "top20_path": top20}


def trajectory_scores(ref: np.lib.npyio.NpzFile, means: np.ndarray) -> dict[str, np.ndarray]:
    step = np.diff(means, axis=0)
    step_len = np.linalg.norm(step, axis=-1)
    path_len = step_len.sum(axis=0)
    if means.shape[0] >= 3:
        accel = means[2:] - 2.0 * means[1:-1] + means[:-2]
        curvature = np.linalg.norm(accel, axis=-1).mean(axis=0)
    else:
        curvature = np.zeros(means.shape[1], dtype=np.float32)

    opacity = 1.0 / (1.0 + np.exp(-np.asarray(ref["logit_opacities"], dtype=np.float32)[:, 0]))
    scale = np.exp(np.asarray(ref["log_scales"], dtype=np.float32)).mean(axis=1)

    def norm(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        lo, hi = np.quantile(x, [0.01, 0.99])
        return np.clip((x - lo) / max(hi - lo, 1e-12), 0.0, 1.0)

    action = norm(path_len) * (0.25 + norm(curvature)) * (0.25 + norm(opacity))
    pathcurv = norm(path_len) * (0.25 + norm(curvature))
    image_sensitivity = (0.25 + norm(opacity)) * (0.10 + norm(scale))
    motion_sensitivity = norm(path_len) * (0.25 + norm(curvature))
    null_risk = motion_sensitivity / (image_sensitivity + 0.15)
    null_action = action * np.clip(null_risk, 0.0, 8.0)
    rng = np.random.default_rng(7)
    return {
        "path_len": path_len,
        "curvature": curvature,
        "pathcurv": pathcurv.astype(np.float32),
        "opacity": opacity,
        "scale": scale,
        "action": action.astype(np.float32),
        "null_action": null_action.astype(np.float32),
        "null_risk": null_risk.astype(np.float32),
        "random": rng.random(means.shape[1], dtype=np.float32),
    }


def stride_key_count(num_frames: int, stride: int) -> int:
    keys = list(range(0, num_frames, stride))
    if keys[-1] != num_frames - 1:
        keys.append(num_frames - 1)
    return len(keys)


def stride_stats(num_frames: int, stride_map: np.ndarray) -> dict:
    unique, counts = np.unique(stride_map, return_counts=True)
    total_keys = 0
    stats = {}
    for stride, count in zip(unique.tolist(), counts.tolist()):
        keys = stride_key_count(num_frames, int(stride))
        total_keys += keys * count
        stats[f"stride_{int(stride)}"] = {
            "gaussians": int(count),
            "keys_per_gaussian": int(keys),
        }
    return {
        "total_keyframes": int(total_keys),
        "avg_keyframes_per_gaussian": float(total_keys / max(len(stride_map), 1)),
        "stride_histogram": stats,
    }


def write_candidate(out_path: Path, means: np.ndarray) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, means3D=means.astype(np.float32, copy=False))
    return {"path": str(out_path), "bytes": int(out_path.stat().st_size)}


def make_adaptive(means: np.ndarray, fast_mask: np.ndarray, fast_stride: int, slow_stride: int) -> np.ndarray:
    fast = interpolate_stride(means[:, fast_mask], fast_stride)
    slow = interpolate_stride(means[:, ~fast_mask], slow_stride)
    out = np.empty_like(means)
    out[:, fast_mask] = fast
    out[:, ~fast_mask] = slow
    return out


def make_variable_stride(means: np.ndarray, stride_map: np.ndarray) -> np.ndarray:
    out = np.empty_like(means)
    for stride in sorted(np.unique(stride_map).tolist()):
        mask = stride_map == stride
        out[:, mask] = interpolate_stride(means[:, mask], int(stride))
    return out


def stride_map_from_score(score: np.ndarray, rules: list[tuple[float, int]], default_stride: int) -> np.ndarray:
    """Assign smaller strides to higher scores.

    Rules are cumulative top fractions, e.g. [(0.05, 2), (0.20, 4)] means top 5% gets stride 2 and next
    15% gets stride 4.
    """
    n = len(score)
    order = np.argsort(score)[::-1]
    stride_map = np.full(n, default_stride, dtype=np.int16)
    prev = 0
    for frac, stride in rules:
        end = min(n, int(round(n * frac)))
        if end > prev:
            stride_map[order[prev:end]] = stride
        prev = max(prev, end)
    return stride_map


def adaptive_stride_map(mask: np.ndarray, fast_stride: int, slow_stride: int) -> np.ndarray:
    stride_map = np.full(len(mask), slow_stride, dtype=np.int16)
    stride_map[mask] = fast_stride
    return stride_map


def fg_floor_score_top_map(
    fg_mask: np.ndarray,
    score: np.ndarray,
    top_frac: float,
    top_stride: int,
    fg_stride: int,
    bg_stride: int,
) -> np.ndarray:
    """Foreground floor plus extra protection for the highest AP-RD scores."""
    n = len(score)
    stride_map = np.full(n, bg_stride, dtype=np.int16)
    stride_map[fg_mask] = fg_stride
    top_n = min(n, int(round(n * top_frac)))
    if top_n > 0:
        order = np.argsort(score)[::-1]
        stride_map[order[:top_n]] = top_stride
    return stride_map


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="reference params.npz")
    parser.add_argument("--out-dir", required=True, help="directory for candidate npz files")
    parser.add_argument("--manifest", required=True, help="manifest JSON path")
    args = parser.parse_args()

    ref_path = Path(args.reference)
    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)
    ref = np.load(ref_path, allow_pickle=True)
    means = np.asarray(ref["means3D"], dtype=np.float32)
    fg = np.asarray(ref["seg_colors"][:, 0] > 0.5)
    masks = path_masks(means)
    scores = trajectory_scores(ref, means)
    stride_maps = {
        "adaptive_fg_k04_bg32": adaptive_stride_map(fg, 4, 32),
        "adaptive_top10_k04_else32": adaptive_stride_map(masks["top10_path"], 4, 32),
        "adaptive_top20_k08_else32": adaptive_stride_map(masks["top20_path"], 8, 32),
        "aprd_action_q05s2_q20s4_q40s8_else64": stride_map_from_score(
            scores["action"], [(0.05, 2), (0.20, 4), (0.40, 8)], 64
        ),
        "aprd_null_q05s2_q20s4_q40s8_else64": stride_map_from_score(
            scores["null_action"], [(0.05, 2), (0.20, 4), (0.40, 8)], 64
        ),
        "aprd_null_q08s2_q20s4_q40s16_else64": stride_map_from_score(
            scores["null_action"], [(0.08, 2), (0.20, 4), (0.40, 16)], 64
        ),
        "aprd_null_q10s2_q20s8_else64": stride_map_from_score(
            scores["null_action"], [(0.10, 2), (0.20, 8)], 64
        ),
        "aprd_hybrid_fg4_nulltop05s2_bg128": fg_floor_score_top_map(
            fg, scores["null_action"], 0.05, 2, 4, 128
        ),
        "aprd_hybrid_fg4_nulltop08s2_bg128": fg_floor_score_top_map(
            fg, scores["null_action"], 0.08, 2, 4, 128
        ),
        "aprd_hybrid_fg4_nulltop10s2_bg128": fg_floor_score_top_map(
            fg, scores["null_action"], 0.10, 2, 4, 128
        ),
        "aprd_hybrid_fg4_nulltop05s2_bg64": fg_floor_score_top_map(
            fg, scores["null_action"], 0.05, 2, 4, 64
        ),
    }

    specs = [
        ("interp_k04", "uniform temporal keyframe interpolation, stride 4", lambda: interpolate_stride(means, 4)),
        ("interp_k08", "uniform temporal keyframe interpolation, stride 8", lambda: interpolate_stride(means, 8)),
        ("interp_k16", "uniform temporal keyframe interpolation, stride 16", lambda: interpolate_stride(means, 16)),
        ("interp_k32", "uniform temporal keyframe interpolation, stride 32", lambda: interpolate_stride(means, 32)),
        (
            "adaptive_fg_k04_bg32",
            "foreground stride 4, background stride 32",
            lambda: make_variable_stride(means, stride_maps["adaptive_fg_k04_bg32"]),
        ),
        (
            "adaptive_top10_k04_else32",
            "top-10% path-length Gaussians stride 4, others stride 32",
            lambda: make_variable_stride(means, stride_maps["adaptive_top10_k04_else32"]),
        ),
        (
            "adaptive_top20_k08_else32",
            "top-20% path-length Gaussians stride 8, others stride 32",
            lambda: make_variable_stride(means, stride_maps["adaptive_top20_k08_else32"]),
        ),
        (
            "aprd_action_q05s2_q20s4_q40s8_else64",
            "AP-RD action score: top 5% stride 2, top 20% stride 4, top 40% stride 8, rest stride 64",
            lambda: make_variable_stride(means, stride_maps["aprd_action_q05s2_q20s4_q40s8_else64"]),
        ),
        (
            "aprd_null_q05s2_q20s4_q40s8_else64",
            "null-space AP-RD: top null-action 5% stride 2, 20% stride 4, 40% stride 8, rest stride 64",
            lambda: make_variable_stride(means, stride_maps["aprd_null_q05s2_q20s4_q40s8_else64"]),
        ),
        (
            "aprd_null_q08s2_q20s4_q40s16_else64",
            "null-space AP-RD: top null-action 8% stride 2, 20% stride 4, 40% stride 16, rest stride 64",
            lambda: make_variable_stride(means, stride_maps["aprd_null_q08s2_q20s4_q40s16_else64"]),
        ),
        (
            "aprd_null_q10s2_q20s8_else64",
            "null-space AP-RD: top null-action 10% stride 2, top 20% stride 8, rest stride 64",
            lambda: make_variable_stride(means, stride_maps["aprd_null_q10s2_q20s8_else64"]),
        ),
        (
            "aprd_hybrid_fg4_nulltop05s2_bg128",
            "hybrid AP-RD: foreground floor stride 4, top null-action 5% stride 2, background stride 128",
            lambda: make_variable_stride(means, stride_maps["aprd_hybrid_fg4_nulltop05s2_bg128"]),
        ),
        (
            "aprd_hybrid_fg4_nulltop08s2_bg128",
            "hybrid AP-RD: foreground floor stride 4, top null-action 8% stride 2, background stride 128",
            lambda: make_variable_stride(means, stride_maps["aprd_hybrid_fg4_nulltop08s2_bg128"]),
        ),
        (
            "aprd_hybrid_fg4_nulltop10s2_bg128",
            "hybrid AP-RD: foreground floor stride 4, top null-action 10% stride 2, background stride 128",
            lambda: make_variable_stride(means, stride_maps["aprd_hybrid_fg4_nulltop10s2_bg128"]),
        ),
        (
            "aprd_hybrid_fg4_nulltop05s2_bg64",
            "hybrid AP-RD: foreground floor stride 4, top null-action 5% stride 2, background stride 64",
            lambda: make_variable_stride(means, stride_maps["aprd_hybrid_fg4_nulltop05s2_bg64"]),
        ),
    ]

    entries = []
    for label, description, build in specs:
        out_path = out_dir / f"{label}.npz"
        if out_path.exists():
            info = {"path": str(out_path), "bytes": int(out_path.stat().st_size)}
            timesteps, gaussians = int(means.shape[0]), int(means.shape[1])
            print(f"{label}: reusing existing {out_path} size={info['bytes'] / 1e6:.2f}MB")
        else:
            cand = build()
            info = write_candidate(out_path, cand)
            timesteps, gaussians = int(cand.shape[0]), int(cand.shape[1])
            print(f"{label}: wrote {out_path} size={info['bytes'] / 1e6:.2f}MB")
            del cand
        rate_model = None
        if label.startswith("interp_k"):
            rate_model = stride_stats(
                means.shape[0], np.full(means.shape[1], int(label[-2:]), dtype=np.int16)
            )
        elif label in stride_maps:
            rate_model = stride_stats(means.shape[0], stride_maps[label])
        entry = {
            "label": label,
            "description": description,
            "candidate": info["path"],
            "bytes": info["bytes"],
            "size_mb": info["bytes"] / 1e6,
            "timesteps": timesteps,
            "gaussians": gaussians,
            "rate_model": rate_model,
        }
        entries.append(entry)

    manifest = {
        "reference": str(ref_path),
        "out_dir": str(out_dir),
        "foreground_fraction": float(fg.mean()),
        "top10_path_fraction": float(masks["top10_path"].mean()),
        "top20_path_fraction": float(masks["top20_path"].mean()),
        "score_summary": {
            name: {
                "q50": float(np.quantile(value, 0.50)),
                "q90": float(np.quantile(value, 0.90)),
                "q95": float(np.quantile(value, 0.95)),
                "q99": float(np.quantile(value, 0.99)),
            }
            for name, value in scores.items()
            if name in {"action", "null_action", "null_risk"}
        },
        "variants": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"WROTE {manifest_path}")
    print("PANOPTIC_TEMPORAL_VARIANTS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
