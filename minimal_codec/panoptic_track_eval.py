#!/usr/bin/env python
"""MTE / delta / survival-style trajectory evaluation for PanopticSports params.

The reference and candidate must share Gaussian identities and tensor shape. This is
intended for uncompressed-vs-compressed variants derived from the same Dynamic3DGaussians
model, not for comparing independently trained models with different point counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


THRESHOLDS_CM = (1, 2, 4, 8, 16)


def summarize(name: str, err_m: np.ndarray, mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"name": name, "count": 0}
    e = err_m[:, mask]
    per_track_mean = e.mean(axis=0)
    per_track_max = e.max(axis=0)
    out = {
        "name": name,
        "count": int(mask.sum()),
        "mte_cm": float(e.mean() * 100.0),
        "median_track_mte_cm": float(np.median(per_track_mean) * 100.0),
        "p90_track_mte_cm": float(np.quantile(per_track_mean, 0.90) * 100.0),
        "p95_track_mte_cm": float(np.quantile(per_track_mean, 0.95) * 100.0),
    }
    for thr_cm in THRESHOLDS_CM:
        thr_m = thr_cm / 100.0
        out[f"delta@{thr_cm}cm"] = float((e <= thr_m).mean())
        out[f"survival@{thr_cm}cm"] = float((per_track_max <= thr_m).mean())
    return out


def trajectory_masks(ref_xyz: np.ndarray) -> dict[str, np.ndarray]:
    step = np.diff(ref_xyz, axis=0)
    step_len = np.linalg.norm(step, axis=-1)
    path_len = step_len.sum(axis=0)
    if ref_xyz.shape[0] >= 3:
        accel = ref_xyz[2:] - 2.0 * ref_xyz[1:-1] + ref_xyz[:-2]
        curvature = np.linalg.norm(accel, axis=-1).mean(axis=0)
    else:
        curvature = np.zeros(ref_xyz.shape[1], dtype=np.float32)

    top10_path = path_len >= np.quantile(path_len, 0.90)
    top20_path = path_len >= np.quantile(path_len, 0.80)
    top10_curv = curvature >= np.quantile(curvature, 0.90)
    return {
        "top10_path_len": top10_path,
        "top20_path_len": top20_path,
        "bottom80_path_len": ~top20_path,
        "top10_curvature": top10_curv,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="reference params.npz")
    parser.add_argument("--candidate", required=True, help="candidate params.npz")
    parser.add_argument("--out", required=True, help="JSON output path")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    ref_path = Path(args.reference)
    cand_path = Path(args.candidate)
    ref = np.load(ref_path, allow_pickle=True)
    cand = np.load(cand_path, allow_pickle=True)
    ref_xyz = np.asarray(ref["means3D"], dtype=np.float32)
    cand_xyz = np.asarray(cand["means3D"], dtype=np.float32)
    if ref_xyz.shape != cand_xyz.shape:
        raise SystemExit(
            f"shape mismatch: reference means3D {ref_xyz.shape} vs candidate {cand_xyz.shape}; "
            "use variants derived from the same model"
        )

    err = np.linalg.norm(cand_xyz - ref_xyz, axis=-1)
    fg = np.asarray(ref["seg_colors"][:, 0] > 0.5)
    masks = trajectory_masks(ref_xyz)
    groups = [
        summarize("all", err, np.ones(ref_xyz.shape[1], dtype=bool)),
        summarize("foreground", err, fg),
        summarize("background", err, ~fg),
    ]
    groups.extend(summarize(name, err, mask) for name, mask in masks.items())
    groups.append(summarize("foreground_top10_path_len", err, fg & masks["top10_path_len"]))

    result = {
        "label": args.label,
        "reference": str(ref_path),
        "candidate": str(cand_path),
        "timesteps": int(ref_xyz.shape[0]),
        "gaussians": int(ref_xyz.shape[1]),
        "groups": groups,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["groups"], indent=2))
    print(f"WROTE {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
