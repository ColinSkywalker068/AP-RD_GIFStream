#!/usr/bin/env python
"""Pack Panoptic AP-RD trajectory variants into APRDZ entropy bitstreams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from panoptic_entropy_codec import pack_bitstream
from panoptic_make_temporal_variants import stride_stats
from panoptic_pack_trajectory_payloads import build_selected_stride_maps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--labels", nargs="*", default=None)
    args = parser.parse_args()

    ref_path = Path(args.reference)
    out_root = Path(args.out_root)
    bitstream_dir = out_root / "bitstreams"
    ref = np.load(ref_path, allow_pickle=True)
    means = np.asarray(ref["means3D"], dtype=np.float32)
    stride_maps = build_selected_stride_maps(ref, means, args.labels)
    selected = args.labels or list(stride_maps)

    rows = []
    for label in selected:
        if label not in stride_maps:
            raise SystemExit(f"unknown label {label}; known={sorted(stride_maps)}")
        out_path = bitstream_dir / f"{label}.aprdz"
        print(f"===== pack entropy bitstream {label} =====", flush=True)
        info = pack_bitstream(
            means,
            stride_maps[label],
            out_path,
            label=label,
            source_reference=str(ref_path),
        )
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
    out_json = out_root / "summary_entropy_bitstreams.json"
    out_md = out_root / "summary_entropy_bitstreams.md"
    out_json.write_text(json.dumps(rows, indent=2))

    lines = [
        "# Panoptic Entropy-Coded Trajectory Bitstreams",
        "",
        "| variant | bitstream MB | avg keys/G | total keyframes | bits/key xyz | ratio vs ref params |",
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
    print("PANOPTIC_ENTROPY_BITSTREAMS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
