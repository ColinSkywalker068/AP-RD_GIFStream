#!/usr/bin/env python
"""Decode q16 trajectory payloads and evaluate Panoptic MTE from the payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from panoptic_track_eval import summarize, trajectory_masks


def decode_payload(payload_path: Path) -> np.ndarray:
    payload = np.load(payload_path, allow_pickle=True)
    num_frames = int(payload["num_frames"][0])
    num_gaussians = int(payload["num_gaussians"][0])
    axis_min = payload["axis_min"].astype(np.float32)
    axis_scale = payload["axis_scale"].astype(np.float32)
    out = np.empty((num_frames, num_gaussians, 3), dtype=np.float32)

    for stride in payload["stride_values"].astype(int).tolist():
        idx = payload[f"indices_s{stride}"].astype(np.int64)
        keys = payload[f"keys_s{stride}"].astype(np.int64)
        q = payload[f"means_q16_s{stride}"].astype(np.float32)
        vals = q * axis_scale[None, None, :] + axis_min[None, None, :]
        for ka, kb, va, vb in zip(keys[:-1], keys[1:], vals[:-1], vals[1:]):
            denom = max(1, int(kb - ka))
            for t in range(int(ka), int(kb) + 1):
                alpha = (t - ka) / denom
                out[t, idx, :] = (1.0 - alpha) * va + alpha * vb
    return out


def evaluate(ref_xyz: np.ndarray, cand_xyz: np.ndarray, fg: np.ndarray, label: str, payload_path: Path) -> dict:
    err = np.linalg.norm(cand_xyz - ref_xyz, axis=-1)
    masks = trajectory_masks(ref_xyz)
    groups = [
        summarize("all", err, np.ones(ref_xyz.shape[1], dtype=bool)),
        summarize("foreground", err, fg),
        summarize("background", err, ~fg),
    ]
    groups.extend(summarize(name, err, mask) for name, mask in masks.items())
    groups.append(summarize("foreground_top10_path_len", err, fg & masks["top10_path_len"]))
    return {
        "label": label,
        "payload": str(payload_path),
        "timesteps": int(ref_xyz.shape[0]),
        "gaussians": int(ref_xyz.shape[1]),
        "groups": groups,
    }


def group_by_name(result: dict) -> dict:
    return {g["name"]: g for g in result["groups"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--payload-root", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--labels", nargs="*", default=None)
    args = parser.parse_args()

    ref = np.load(args.reference, allow_pickle=True)
    ref_xyz = np.asarray(ref["means3D"], dtype=np.float32)
    fg = np.asarray(ref["seg_colors"][:, 0] > 0.5)
    payload_root = Path(args.payload_root)
    payload_dir = payload_root / "payloads"
    payload_summary_path = payload_root / "summary_payload_bits.json"
    payload_info = {}
    if payload_summary_path.exists():
        for row in json.loads(payload_summary_path.read_text()):
            payload_info[row["label"]] = row

    labels = args.labels
    if not labels:
        labels = sorted(
            p.name.removesuffix("_q16_payload.npz")
            for p in payload_dir.glob("*_q16_payload.npz")
        )

    results = []
    for label in labels:
        payload_path = payload_dir / f"{label}_q16_payload.npz"
        print(f"===== decode/eval {label} =====", flush=True)
        cand_xyz = decode_payload(payload_path)
        if cand_xyz.shape != ref_xyz.shape:
            raise SystemExit(f"shape mismatch for {label}: {cand_xyz.shape} vs {ref_xyz.shape}")
        result = evaluate(ref_xyz, cand_xyz, fg, label, payload_path)
        results.append(result)
        print(json.dumps(result["groups"], indent=2), flush=True)
        del cand_xyz

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))

    rows = []
    for result in results:
        groups = group_by_name(result)
        info = payload_info.get(result["label"], {})
        row = {
            "label": result["label"],
            "payload_mb": info.get("payload_mb"),
            "avg_keyframes_per_gaussian": info.get("avg_keyframes_per_gaussian"),
        }
        for name in ("all", "foreground", "background", "top10_path_len", "foreground_top10_path_len"):
            g = groups.get(name, {})
            row[f"{name}_mte_cm"] = g.get("mte_cm")
            row[f"{name}_survival8"] = g.get("survival@8cm")
        rows.append(row)

    lines = [
        "# Decoded Payload Panoptic MTE",
        "",
        "| variant | payload MB | avg keys/G | all MTE cm | fg MTE cm | top10 path MTE cm | fg top10 MTE cm | fg survival@8cm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(key: str) -> str:
            val = row.get(key)
            return "nan" if val is None else f"{val:.3f}"

        lines.append(
            f"| {row['label']} | {fmt('payload_mb')} | {fmt('avg_keyframes_per_gaussian')} | "
            f"{fmt('all_mte_cm')} | {fmt('foreground_mte_cm')} | "
            f"{fmt('top10_path_len_mte_cm')} | {fmt('foreground_top10_path_len_mte_cm')} | "
            f"{fmt('foreground_survival8')} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(out_md.read_text(), flush=True)
    print(f"WROTE {out_json}", flush=True)
    print(f"WROTE {out_md}", flush=True)
    print("PANOPTIC_DECODED_PAYLOAD_MTE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
