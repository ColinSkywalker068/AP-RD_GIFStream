#!/usr/bin/env python
"""Pack Panoptic trajectory variants into compact keyframe payloads.

This is not an entropy-coded production codec. It is a concrete byte-counted
payload for the current AP-RD trajectory representation: per-stride Gaussian
indices plus q16 keyframe positions. It replaces the dense reconstructed `.npz`
artifact size with a meaningful trajectory-rate axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from panoptic_make_temporal_variants import (
    adaptive_stride_map,
    fg_floor_score_top_map,
    path_masks,
    stride_map_from_score,
    stride_stats,
    trajectory_scores,
)


def key_indices(num_frames: int, stride: int) -> np.ndarray:
    keys = list(range(0, num_frames, stride))
    if keys[-1] != num_frames - 1:
        keys.append(num_frames - 1)
    return np.array(keys, dtype=np.uint16)


def quantize_q16(values: np.ndarray, lo: np.ndarray, scale: np.ndarray) -> np.ndarray:
    q = np.rint((values.astype(np.float32) - lo[None, None, :]) / scale[None, None, :])
    return np.clip(q, 0, 65535).astype(np.uint16)


def interpolation_mte_by_stride(means: np.ndarray, stride: int) -> np.ndarray:
    """Per-Gaussian mean interpolation error in meters for one stride."""
    keys = key_indices(means.shape[0], stride).astype(np.int64)
    err_sum = np.zeros(means.shape[1], dtype=np.float64)
    for ka, kb in zip(keys[:-1], keys[1:]):
        start = means[int(ka)]
        end = means[int(kb)]
        denom = max(1, int(kb - ka))
        for t in range(int(ka), int(kb) + 1):
            alpha = (t - ka) / denom
            pred = (1.0 - alpha) * start + alpha * end
            err_sum += np.linalg.norm(pred - means[t], axis=1)
    return (err_sum / means.shape[0]).astype(np.float32)


def q16_delta_varint_cost_by_stride(
    means: np.ndarray,
    stride: int,
    axis_min: np.ndarray,
    axis_scale: np.ndarray,
) -> np.ndarray:
    """Proxy per-Gaussian temporal residual code length for the APRDZ stream."""
    keys = key_indices(means.shape[0], stride)
    q = quantize_q16(means[keys], axis_min, axis_scale)
    dq = np.diff(q.astype(np.int32), axis=0)
    zz = ((dq << 1) ^ (dq >> 31)).astype(np.uint32)
    lens = np.ones(zz.shape, dtype=np.uint8)
    lens += zz >= 128
    lens += zz >= 16384
    lens += zz >= 2097152
    return lens.sum(axis=(0, 2)).astype(np.float32)


def entropy_distortion_budget_map(
    fg_mask: np.ndarray,
    means: np.ndarray,
    budget_scale: float,
    priority: np.ndarray | None = None,
    top_stride: int = 2,
    fg_stride: int = 4,
    baseline_bg_stride: int = 32,
    bg_stride: int = 128,
) -> np.ndarray:
    """Foreground floor plus marginal distortion reduction per residual-code bit.

    The spendable budget is the residual-code saving from moving background
    trajectories from the foreground-only baseline's stride 32 to AP-RD's
    stride 128. The top-up then buys stride-2 foreground trajectories greedily
    by estimated MTE reduction per additional residual byte.
    """
    axis_min = means.reshape(-1, 3).min(axis=0).astype(np.float32)
    axis_max = means.reshape(-1, 3).max(axis=0).astype(np.float32)
    axis_scale = np.maximum((axis_max - axis_min) / 65535.0, 1e-8).astype(np.float32)

    cost_top = q16_delta_varint_cost_by_stride(means, top_stride, axis_min, axis_scale)
    cost_fg = q16_delta_varint_cost_by_stride(means, fg_stride, axis_min, axis_scale)
    cost_bg_base = q16_delta_varint_cost_by_stride(means, baseline_bg_stride, axis_min, axis_scale)
    cost_bg = q16_delta_varint_cost_by_stride(means, bg_stride, axis_min, axis_scale)

    budget = float(np.maximum(cost_bg_base[~fg_mask] - cost_bg[~fg_mask], 0.0).sum())
    target_budget = max(0.0, budget_scale * budget)
    delta_cost = np.maximum(cost_top - cost_fg, 1.0)

    err_top = interpolation_mte_by_stride(means, top_stride)
    err_fg = interpolation_mte_by_stride(means, fg_stride)
    benefit = np.maximum(err_fg - err_top, 0.0)
    if priority is not None:
        priority = np.asarray(priority, dtype=np.float64)
        lo, hi = np.quantile(priority, [0.01, 0.99])
        priority_norm = np.clip((priority - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
        benefit = benefit * (0.10 + priority_norm.astype(np.float32))
    score = benefit / delta_cost

    candidates = np.flatnonzero(fg_mask & (benefit > 0.0))
    order = candidates[np.argsort(score[candidates])[::-1]]
    selected = np.zeros(means.shape[1], dtype=bool)
    spent = 0.0
    for idx in order.tolist():
        cost = float(delta_cost[idx])
        if spent + cost <= target_budget:
            selected[idx] = True
            spent += cost

    stride_map = np.full(means.shape[1], bg_stride, dtype=np.int16)
    stride_map[fg_mask] = fg_stride
    stride_map[selected] = top_stride
    return stride_map


def priority_cost_budget_map(
    fg_mask: np.ndarray,
    means: np.ndarray,
    priority: np.ndarray,
    budget_scale: float,
    top_stride: int = 2,
    fg_stride: int = 4,
    baseline_bg_stride: int = 32,
    bg_stride: int = 128,
) -> np.ndarray:
    """Foreground floor plus priority-per-residual-bit top-up."""
    axis_min = means.reshape(-1, 3).min(axis=0).astype(np.float32)
    axis_max = means.reshape(-1, 3).max(axis=0).astype(np.float32)
    axis_scale = np.maximum((axis_max - axis_min) / 65535.0, 1e-8).astype(np.float32)

    cost_top = q16_delta_varint_cost_by_stride(means, top_stride, axis_min, axis_scale)
    cost_fg = q16_delta_varint_cost_by_stride(means, fg_stride, axis_min, axis_scale)
    cost_bg_base = q16_delta_varint_cost_by_stride(means, baseline_bg_stride, axis_min, axis_scale)
    cost_bg = q16_delta_varint_cost_by_stride(means, bg_stride, axis_min, axis_scale)

    budget = float(np.maximum(cost_bg_base[~fg_mask] - cost_bg[~fg_mask], 0.0).sum())
    target_budget = max(0.0, budget_scale * budget)
    delta_cost = np.maximum(cost_top - cost_fg, 1.0)

    priority = np.asarray(priority, dtype=np.float64)
    lo, hi = np.quantile(priority, [0.01, 0.99])
    priority_norm = np.clip((priority - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    score = priority_norm / delta_cost

    candidates = np.flatnonzero(fg_mask & (priority_norm > 0.0))
    order = candidates[np.argsort(score[candidates])[::-1]]
    selected = np.zeros(means.shape[1], dtype=bool)
    spent = 0.0
    for idx in order.tolist():
        cost = float(delta_cost[idx])
        if spent + cost <= target_budget:
            selected[idx] = True
            spent += cost

    stride_map = np.full(means.shape[1], bg_stride, dtype=np.int16)
    stride_map[fg_mask] = fg_stride
    stride_map[selected] = top_stride
    return stride_map


def pack_payload(means: np.ndarray, stride_map: np.ndarray, out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    axis_min = means.reshape(-1, 3).min(axis=0).astype(np.float32)
    axis_max = means.reshape(-1, 3).max(axis=0).astype(np.float32)
    axis_scale = np.maximum((axis_max - axis_min) / 65535.0, 1e-8).astype(np.float32)

    arrays: dict[str, np.ndarray] = {
        "num_frames": np.array([means.shape[0]], dtype=np.uint16),
        "num_gaussians": np.array([means.shape[1]], dtype=np.uint32),
        "axis_min": axis_min,
        "axis_scale": axis_scale,
        "stride_values": np.array(sorted(np.unique(stride_map).tolist()), dtype=np.uint16),
    }
    total_keyframes = 0
    group_rows = []
    for stride in sorted(np.unique(stride_map).tolist()):
        stride_int = int(stride)
        gaussian_indices = np.flatnonzero(stride_map == stride_int).astype(np.uint32)
        keys = key_indices(means.shape[0], stride_int)
        vals = means[keys][:, gaussian_indices, :]
        arrays[f"indices_s{stride_int}"] = gaussian_indices
        arrays[f"keys_s{stride_int}"] = keys
        arrays[f"means_q16_s{stride_int}"] = quantize_q16(vals, axis_min, axis_scale)
        keyframes = int(len(keys) * len(gaussian_indices))
        total_keyframes += keyframes
        group_rows.append(
            {
                "stride": stride_int,
                "gaussians": int(len(gaussian_indices)),
                "keys_per_gaussian": int(len(keys)),
                "keyframes": keyframes,
            }
        )

    np.savez_compressed(out_path, **arrays)
    payload_bytes = int(out_path.stat().st_size)
    return {
        "payload_path": str(out_path),
        "payload_bytes": payload_bytes,
        "payload_mb": payload_bytes / 1e6,
        "total_keyframes": total_keyframes,
        "avg_keyframes_per_gaussian": total_keyframes / means.shape[1],
        "bits_per_keyframe_xyz": payload_bytes * 8.0 / max(total_keyframes, 1),
        "groups": group_rows,
        "axis_min": axis_min.tolist(),
        "axis_scale": axis_scale.tolist(),
    }


def build_stride_maps(ref: np.lib.npyio.NpzFile, means: np.ndarray) -> dict[str, np.ndarray]:
    fg = np.asarray(ref["seg_colors"][:, 0] > 0.5)
    scores = trajectory_scores(ref, means)
    maps = {
        "interp_k32": np.full(means.shape[1], 32, dtype=np.int16),
        "adaptive_fg_k04_bg32": adaptive_stride_map(fg, 4, 32),
        "hybrid_fg4_pathtop05s2_bg128": fg_floor_score_top_map(
            fg, scores["path_len"], 0.05, 2, 4, 128
        ),
        "hybrid_fg4_curvtop05s2_bg128": fg_floor_score_top_map(
            fg, scores["curvature"], 0.05, 2, 4, 128
        ),
        "hybrid_fg4_actiontop05s2_bg128": fg_floor_score_top_map(
            fg, scores["action"], 0.05, 2, 4, 128
        ),
        "hybrid_fg4_risktop05s2_bg128": fg_floor_score_top_map(
            fg, scores["null_risk"], 0.05, 2, 4, 128
        ),
        "aprd_hybrid_fg4_nulltop05s2_bg128": fg_floor_score_top_map(
            fg, scores["null_action"], 0.05, 2, 4, 128
        ),
        "aprd_hybrid_fg4_nulltop08s2_bg128": fg_floor_score_top_map(
            fg, scores["null_action"], 0.08, 2, 4, 128
        ),
        "top05_path_s2_else128": stride_map_from_score(scores["path_len"], [(0.05, 2)], 128),
        "top05_action_s2_else128": stride_map_from_score(scores["action"], [(0.05, 2)], 128),
        "top05_risk_s2_else128": stride_map_from_score(scores["null_risk"], [(0.05, 2)], 128),
        "top05_null_s2_else128": stride_map_from_score(scores["null_action"], [(0.05, 2)], 128),
    }
    for frac in (0.02, 0.03, 0.04, 0.05, 0.08, 0.10):
        tag = f"{int(round(frac * 100)):02d}"
        maps[f"sweep_fg4_path_top{tag}s2_bg128"] = fg_floor_score_top_map(
            fg, scores["path_len"], frac, 2, 4, 128
        )
        maps[f"sweep_fg4_curv_top{tag}s2_bg128"] = fg_floor_score_top_map(
            fg, scores["curvature"], frac, 2, 4, 128
        )
        maps[f"sweep_fg4_pathcurv_top{tag}s2_bg128"] = fg_floor_score_top_map(
            fg, scores["pathcurv"], frac, 2, 4, 128
        )
        maps[f"sweep_fg4_action_top{tag}s2_bg128"] = fg_floor_score_top_map(
            fg, scores["action"], frac, 2, 4, 128
        )
        maps[f"sweep_fg4_risk_top{tag}s2_bg128"] = fg_floor_score_top_map(
            fg, scores["null_risk"], frac, 2, 4, 128
        )
        maps[f"sweep_fg4_null_top{tag}s2_bg128"] = fg_floor_score_top_map(
            fg, scores["null_action"], frac, 2, 4, 128
        )
        maps[f"sweep_fg4_random_top{tag}s2_bg128"] = fg_floor_score_top_map(
            fg, scores["random"], frac, 2, 4, 128
        )
        maps[f"nofg_path_top{tag}s2_top30s4_else128"] = stride_map_from_score(
            scores["path_len"], [(frac, 2), (0.30, 4)], 128
        )
        maps[f"nofg_action_top{tag}s2_top30s4_else128"] = stride_map_from_score(
            scores["action"], [(frac, 2), (0.30, 4)], 128
        )
        for floor_frac in (0.20, 0.25):
            floor_tag = f"{int(round(floor_frac * 100)):02d}"
            maps[f"nofg_path_top{tag}s2_top{floor_tag}s4_else128"] = stride_map_from_score(
                scores["path_len"], [(frac, 2), (floor_frac, 4)], 128
            )
            maps[f"nofg_action_top{tag}s2_top{floor_tag}s4_else128"] = stride_map_from_score(
                scores["action"], [(frac, 2), (floor_frac, 4)], 128
            )
    for budget_scale in (0.25, 0.50, 0.75, 1.00):
        tag = f"{int(round(budget_scale * 1000)):03d}"
        maps[f"aprd_edr_b{tag}_fg4s2_bg128"] = entropy_distortion_budget_map(
            fg, means, budget_scale
        )
    for budget_scale in (0.25, 0.35, 0.50):
        tag = f"{int(round(budget_scale * 1000)):03d}"
        maps[f"aprd_pedr_b{tag}_fg4s2_bg128"] = entropy_distortion_budget_map(
            fg, means, budget_scale, priority=scores["path_len"]
        )
        maps[f"aprd_nedr_b{tag}_fg4s2_bg128"] = entropy_distortion_budget_map(
            fg, means, budget_scale, priority=scores["null_action"]
        )
    for budget_scale in (0.25, 0.35, 0.50):
        tag = f"{int(round(budget_scale * 1000)):03d}"
        maps[f"aprd_pbr_b{tag}_fg4s2_bg128"] = priority_cost_budget_map(
            fg, means, scores["path_len"], budget_scale
        )
        maps[f"aprd_nbr_b{tag}_fg4s2_bg128"] = priority_cost_budget_map(
            fg, means, scores["null_action"], budget_scale
        )
    return maps


def build_selected_stride_maps(
    ref: np.lib.npyio.NpzFile,
    means: np.ndarray,
    labels: list[str] | tuple[str, ...] | None,
) -> dict[str, np.ndarray]:
    """Build only requested maps, avoiding expensive unused allocator variants."""
    if not labels:
        return build_stride_maps(ref, means)

    wanted = set(labels)
    maps: dict[str, np.ndarray] = {}
    fg = np.asarray(ref["seg_colors"][:, 0] > 0.5)
    scores = trajectory_scores(ref, means)

    def add(label: str, make) -> None:
        if label in wanted:
            maps[label] = make()

    add("interp_k32", lambda: np.full(means.shape[1], 32, dtype=np.int16))
    add("adaptive_fg_k04_bg32", lambda: adaptive_stride_map(fg, 4, 32))
    add(
        "hybrid_fg4_pathtop05s2_bg128",
        lambda: fg_floor_score_top_map(fg, scores["path_len"], 0.05, 2, 4, 128),
    )
    add(
        "hybrid_fg4_curvtop05s2_bg128",
        lambda: fg_floor_score_top_map(fg, scores["curvature"], 0.05, 2, 4, 128),
    )
    add(
        "hybrid_fg4_actiontop05s2_bg128",
        lambda: fg_floor_score_top_map(fg, scores["action"], 0.05, 2, 4, 128),
    )
    add(
        "hybrid_fg4_risktop05s2_bg128",
        lambda: fg_floor_score_top_map(fg, scores["null_risk"], 0.05, 2, 4, 128),
    )
    add(
        "aprd_hybrid_fg4_nulltop05s2_bg128",
        lambda: fg_floor_score_top_map(fg, scores["null_action"], 0.05, 2, 4, 128),
    )
    add(
        "aprd_hybrid_fg4_nulltop08s2_bg128",
        lambda: fg_floor_score_top_map(fg, scores["null_action"], 0.08, 2, 4, 128),
    )
    add("top05_path_s2_else128", lambda: stride_map_from_score(scores["path_len"], [(0.05, 2)], 128))
    add("top05_action_s2_else128", lambda: stride_map_from_score(scores["action"], [(0.05, 2)], 128))
    add("top05_risk_s2_else128", lambda: stride_map_from_score(scores["null_risk"], [(0.05, 2)], 128))
    add("top05_null_s2_else128", lambda: stride_map_from_score(scores["null_action"], [(0.05, 2)], 128))

    for frac in (0.02, 0.03, 0.04, 0.05, 0.08, 0.10):
        tag = f"{int(round(frac * 100)):02d}"
        add(
            f"sweep_fg4_path_top{tag}s2_bg128",
            lambda frac=frac: fg_floor_score_top_map(fg, scores["path_len"], frac, 2, 4, 128),
        )
        add(
            f"sweep_fg4_curv_top{tag}s2_bg128",
            lambda frac=frac: fg_floor_score_top_map(fg, scores["curvature"], frac, 2, 4, 128),
        )
        add(
            f"sweep_fg4_pathcurv_top{tag}s2_bg128",
            lambda frac=frac: fg_floor_score_top_map(fg, scores["pathcurv"], frac, 2, 4, 128),
        )
        add(
            f"sweep_fg4_action_top{tag}s2_bg128",
            lambda frac=frac: fg_floor_score_top_map(fg, scores["action"], frac, 2, 4, 128),
        )
        add(
            f"sweep_fg4_risk_top{tag}s2_bg128",
            lambda frac=frac: fg_floor_score_top_map(fg, scores["null_risk"], frac, 2, 4, 128),
        )
        add(
            f"sweep_fg4_null_top{tag}s2_bg128",
            lambda frac=frac: fg_floor_score_top_map(fg, scores["null_action"], frac, 2, 4, 128),
        )
        add(
            f"sweep_fg4_random_top{tag}s2_bg128",
            lambda frac=frac: fg_floor_score_top_map(fg, scores["random"], frac, 2, 4, 128),
        )
        add(
            f"nofg_path_top{tag}s2_top30s4_else128",
            lambda frac=frac: stride_map_from_score(scores["path_len"], [(frac, 2), (0.30, 4)], 128),
        )
        add(
            f"nofg_action_top{tag}s2_top30s4_else128",
            lambda frac=frac: stride_map_from_score(scores["action"], [(frac, 2), (0.30, 4)], 128),
        )
        for floor_frac in (0.20, 0.25):
            floor_tag = f"{int(round(floor_frac * 100)):02d}"
            add(
                f"nofg_path_top{tag}s2_top{floor_tag}s4_else128",
                lambda frac=frac, floor_frac=floor_frac: stride_map_from_score(
                    scores["path_len"], [(frac, 2), (floor_frac, 4)], 128
                ),
            )
            add(
                f"nofg_action_top{tag}s2_top{floor_tag}s4_else128",
                lambda frac=frac, floor_frac=floor_frac: stride_map_from_score(
                    scores["action"], [(frac, 2), (floor_frac, 4)], 128
                ),
            )

    for budget_scale in (0.25, 0.50, 0.75, 1.00):
        tag = f"{int(round(budget_scale * 1000)):03d}"
        add(
            f"aprd_edr_b{tag}_fg4s2_bg128",
            lambda budget_scale=budget_scale: entropy_distortion_budget_map(fg, means, budget_scale),
        )
    for budget_scale in (0.25, 0.35, 0.50):
        tag = f"{int(round(budget_scale * 1000)):03d}"
        add(
            f"aprd_pedr_b{tag}_fg4s2_bg128",
            lambda budget_scale=budget_scale: entropy_distortion_budget_map(
                fg, means, budget_scale, priority=scores["path_len"]
            ),
        )
        add(
            f"aprd_nedr_b{tag}_fg4s2_bg128",
            lambda budget_scale=budget_scale: entropy_distortion_budget_map(
                fg, means, budget_scale, priority=scores["null_action"]
            ),
        )
        add(
            f"aprd_pbr_b{tag}_fg4s2_bg128",
            lambda budget_scale=budget_scale: priority_cost_budget_map(
                fg, means, scores["path_len"], budget_scale
            ),
        )
        add(
            f"aprd_nbr_b{tag}_fg4s2_bg128",
            lambda budget_scale=budget_scale: priority_cost_budget_map(
                fg, means, scores["null_action"], budget_scale
            ),
        )

    missing = sorted(wanted - set(maps))
    if missing:
        raise KeyError(f"unknown label(s) {missing}; known={sorted(set(build_stride_maps(ref, means)))}")
    return {label: maps[label] for label in labels}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--labels", nargs="*", default=None)
    args = parser.parse_args()

    ref_path = Path(args.reference)
    out_root = Path(args.out_root)
    payload_dir = out_root / "payloads"
    ref = np.load(ref_path, allow_pickle=True)
    means = np.asarray(ref["means3D"], dtype=np.float32)
    stride_maps = build_selected_stride_maps(ref, means, args.labels)
    selected = args.labels or list(stride_maps)

    rows = []
    for label in selected:
        if label not in stride_maps:
            raise SystemExit(f"unknown label {label}; known={sorted(stride_maps)}")
        out_path = payload_dir / f"{label}_q16_payload.npz"
        print(f"===== pack {label} =====", flush=True)
        info = pack_payload(means, stride_maps[label], out_path)
        info.update(
            {
                "label": label,
                "reference_bytes": int(ref_path.stat().st_size),
                "rate_model": stride_stats(means.shape[0], stride_maps[label]),
            }
        )
        rows.append(info)
        print(json.dumps(info, indent=2), flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    out_json = out_root / "summary_payload_bits.json"
    out_md = out_root / "summary_payload_bits.md"
    out_json.write_text(json.dumps(rows, indent=2))

    lines = [
        "# Panoptic Trajectory Payload Bit Counts",
        "",
        "| variant | payload MB | avg keys/G | total keyframes | bits/key xyz | ratio vs ref params |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['payload_mb']:.3f} | {row['avg_keyframes_per_gaussian']:.3f} | "
            f"{row['total_keyframes']} | {row['bits_per_keyframe_xyz']:.2f} | "
            f"{row['payload_bytes'] / row['reference_bytes']:.4f} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(out_md.read_text(), flush=True)
    print(f"WROTE {out_json}", flush=True)
    print(f"WROTE {out_md}", flush=True)
    print("PANOPTIC_PAYLOAD_BITS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
