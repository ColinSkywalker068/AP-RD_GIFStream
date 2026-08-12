#!/usr/bin/env python
"""Decode APRDZ trajectory bitstreams and evaluate Panoptic MTE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from panoptic_entropy_codec import decode_bitstream
from panoptic_eval_decoded_payload_mte import evaluate, group_by_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--bitstream-root", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--labels", nargs="*", default=None)
    args = parser.parse_args()

    ref = np.load(args.reference, allow_pickle=True)
    ref_xyz = np.asarray(ref["means3D"], dtype=np.float32)
    fg = np.asarray(ref["seg_colors"][:, 0] > 0.5)
    bitstream_root = Path(args.bitstream_root)
    bitstream_dir = bitstream_root / "bitstreams"
    bitstream_summary_path = bitstream_root / "summary_entropy_bitstreams.json"
    if not bitstream_summary_path.exists():
        bitstream_summary_path = bitstream_root / "summary_rdp_entropy_bitstreams.json"
    bitstream_info = {}
    if bitstream_summary_path.exists():
        for row in json.loads(bitstream_summary_path.read_text()):
            bitstream_info[row["label"]] = row

    labels = args.labels
    if not labels:
        labels = sorted(p.stem for p in bitstream_dir.glob("*.aprdz"))

    results = []
    for label in labels:
        bitstream_path = bitstream_dir / f"{label}.aprdz"
        print(f"===== decode/eval entropy bitstream {label} =====", flush=True)
        cand_xyz = decode_bitstream(bitstream_path)
        if cand_xyz.shape != ref_xyz.shape:
            raise SystemExit(f"shape mismatch for {label}: {cand_xyz.shape} vs {ref_xyz.shape}")
        result = evaluate(ref_xyz, cand_xyz, fg, label, bitstream_path)
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
        info = bitstream_info.get(result["label"], {})
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
        "# Decoded Entropy Bitstream Panoptic MTE",
        "",
        "| variant | bitstream MB | avg keys/G | all MTE cm | fg MTE cm | top10 path MTE cm | fg top10 MTE cm | fg survival@8cm |",
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
    print("PANOPTIC_ENTROPY_BITSTREAM_MTE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
