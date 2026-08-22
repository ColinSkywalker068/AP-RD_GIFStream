#!/usr/bin/env python3
"""Real-zip rate accounting for the v3 campaign.

Reports, per (variant, rate): each GOP archive's on-disk bytes, the five-GOP
sequence bytes, and the AP-minus-official deltas — the paper's rate axis uses
exactly these file sizes (never entropy estimates).  Read-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--n-knn", type=int, default=8)
    parser.add_argument("--rates", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for rate in args.rates:
        entry = {"rate": rate}
        for variant in ("official", "ap-gifstream-full"):
            base = args.results_root / variant / f"nknn{args.n_knn}"
            gop_bytes = []
            for gop in range(5):
                z = base / f"GOP_{gop}" / f"r{rate}" / "compression_rank0.zip"
                gop_bytes.append(z.stat().st_size)
            seq = base / "sequences" / f"sequence_r{rate}.zip"
            entry[variant] = {
                "gop_archive_bytes": gop_bytes,
                "gop_total_bytes": sum(gop_bytes),
                "sequence_bytes": seq.stat().st_size if seq.is_file() else None,
            }
        official_total = entry["official"]["gop_total_bytes"]
        ap_total = entry["ap-gifstream-full"]["gop_total_bytes"]
        entry["ap_minus_official_bytes"] = ap_total - official_total
        entry["ap_over_official_ratio"] = ap_total / official_total
        rows.append(entry)
        print(
            f"r{rate}: official={official_total}B ap={ap_total}B "
            f"delta={ap_total - official_total:+d}B "
            f"ratio={ap_total / official_total:.4f}"
        )
    payload = {"schema": "h007.v3_rate_parity_summary.v1", "rates": rows}
    if args.output:
        args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
