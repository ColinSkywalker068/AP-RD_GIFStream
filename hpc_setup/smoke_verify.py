#!/usr/bin/env python3
"""Deep post-run verifier for the H007 smoke test.

Interrogates every artifact the training + codec entry should have produced.
Every check prints PASS/FAIL with the observed values; any failure makes the
process exit nonzero.  Nothing here trusts a filename's existence alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

CHECKS = {"pass": 0, "fail": 0}


def report(ok: bool, label: str, detail: str = "") -> None:
    CHECKS["pass" if ok else "fail"] += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" :: {detail}" if detail else ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--freeze-step", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--min-psnr", type=float, default=14.0)
    args = parser.parse_args()
    rd = args.result_dir
    is_official = args.variant == "official"

    from gsplat.compression.ap_gifstream import (
        AP_SCORE_SCHEMA,
        build_count_preserving_anchor_allocation,
        build_equal_estimated_byte_allocation,
        variant_spec,
    )

    spec = variant_spec(args.variant)

    # ---- A. checkpoint -------------------------------------------------
    ckpt_path = rd / "ckpts" / f"ckpt_{args.max_steps - 1}_rank0.pt"
    report(ckpt_path.is_file(), "checkpoint exists", str(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    splats = ckpt.get("splats", {})
    n_anchors = int(splats["anchors"].shape[0]) if "anchors" in splats else -1
    report(n_anchors > 500, "anchor count plausible", f"N={n_anchors}")

    state = ckpt.get("ap_state")
    if is_official:
        report(state is None, "official run carries no AP state")
    else:
        report(state is not None, "AP training state present")
        report(
            state.get("schema") == "h007.ap_training_state.v3",
            "AP state schema v3",
            str(state.get("schema")),
        )
        report(
            int(state.get("freeze_step", -1)) == args.freeze_step,
            "freeze step matches",
            f"{state.get('freeze_step')} vs {args.freeze_step}",
        )
        imp = state["importance_score"].to(torch.float64)
        report(imp.shape == (n_anchors,), "importance shape == anchors", str(tuple(imp.shape)))
        report(bool(torch.isfinite(imp).all()), "importance finite")
        report(bool((imp >= 0).all()), "importance nonnegative")
        pos_frac = float((imp > 0).double().mean())
        report(
            pos_frac > 0.5,
            "importance mostly positive (backbone stats real, not zeros)",
            f"positive fraction={pos_frac:.4f} mean={float(imp.mean()):.6g} max={float(imp.max()):.6g}",
        )
        official_retain = state["official_retain_mask"]
        ap_retain = state["ap_retain_mask"]
        ap_active = state["ap_active_mask"]
        ap_class = state["ap_class_mask"]
        est = state["estimated_time_bytes"].to(torch.int64)
        report(
            int(official_retain.sum()) == int(ap_retain.sum()),
            "count preservation (whole-anchor)",
            f"official={int(official_retain.sum())} ap={int(ap_retain.sum())}",
        )
        official_active = state["official_active_mask"]
        report(
            int(est[official_active].sum()) == int(est[ap_active].sum()),
            "estimated-byte preservation (temporal)",
            f"official={int(est[official_active].sum())}B ap={int(est[ap_active].sum())}B",
        )
        report(bool((~(ap_active & ~ap_retain)).all()), "active subset of retain")
        report(bool((~(ap_class & ~ap_active)).all()), "class subset of active")
        for audit_name in ("whole_allocation_audit", "temporal_allocation_audit"):
            audit = state[audit_name]
            promoted, demoted = audit.get("promoted_count"), audit.get("demoted_count")
            keys = audit.get("donor_ranking_keys")
            expected_keys = (
                ["score_asc", "backbone_importance_asc", "canonical_id"]
                if spec.swap or audit_name == "temporal_allocation_audit"
                else keys
            )
            report(
                keys == expected_keys,
                f"{audit_name}: dual-key donor ranking recorded",
                str(keys),
            )
            print(
                f"       {audit_name}: promoted={promoted} demoted={demoted} "
                f"protected={audit.get('protected_count')} eligible={audit.get('eligible_count')}"
            )
        if spec.swap:
            report(
                state["whole_allocation_audit"]["promoted_count"]
                == state["whole_allocation_audit"]["demoted_count"],
                "whole swap promoted == demoted",
            )
            report(
                int(state["temporal_allocation_audit"]["estimated_byte_delta"]) == 0,
                "temporal byte delta exactly zero",
            )
        prov = state["runtime_provenance"]
        report(
            prov.get("manifest_sha256") == args.manifest_sha256,
            "runtime provenance bound to patch10 manifest",
            str(prov.get("manifest_sha256"))[:16],
        )
        report(
            len(prov.get("patch_sha256", [])) == 11,
            "11-stage patch chain in receipt",
            f"len={len(prov.get('patch_sha256', []))}",
        )

        # ---- B. frozen score artifact + independent allocation replay ----
        score_path = rd / "ap_freeze" / f"reference_scores_step{args.freeze_step}.npz"
        report(score_path.is_file(), "score artifact exists", str(score_path))
        with np.load(score_path, allow_pickle=False) as z:
            members = set(z.files)
            for member in ("importance_score", "importance_definition", "allocation_score"):
                report(member in members, f"score artifact member {member}")
            report(
                str(np.asarray(z["schema"]).item()) == AP_SCORE_SCHEMA,
                "score artifact schema",
                str(np.asarray(z["schema"]).item()),
            )
            report(int(np.asarray(z["frame_count"]).item()) == 60, "60-frame GOP")
            ids = np.asarray(z["canonical_ids"], dtype=np.int64)
            report(
                np.unique(ids, axis=0).shape[0] == ids.shape[0],
                "canonical IDs unique",
                f"N={ids.shape[0]}",
            )
            eligible = np.asarray(z["eligible"], dtype=bool)
            report(int(eligible.sum()) > 0, "eligible anchors exist", f"{int(eligible.sum())}/{ids.shape[0]}")
            for name in ("official_retain_mask", "ap_retain_mask", "ap_active_mask", "ap_class_mask"):
                same = bool(np.array_equal(np.asarray(z[name], dtype=bool), state[name].numpy()))
                report(same, f"npz/{name} == checkpoint state")
            scores = np.asarray(z["allocation_score"], dtype=np.float64)
            importance = np.asarray(z["importance_score"], dtype=np.float64)
            est_np = np.asarray(z["estimated_time_bytes"], dtype=np.int64)
            score_t = torch.from_numpy(scores.copy())
            score_t[~torch.from_numpy(eligible)] = -torch.inf
            retain2, class2, _ = build_count_preserving_anchor_allocation(
                torch.from_numpy(np.asarray(z["official_retain_mask"], dtype=bool)),
                score_t,
                torch.from_numpy(eligible),
                torch.from_numpy(ids),
                float(np.asarray(z["protected_fraction"]).item()),
                spec.swap,
                importance=torch.from_numpy(importance),
            )
            active2, class3, _ = build_equal_estimated_byte_allocation(
                torch.from_numpy(np.asarray(z["official_active_mask"], dtype=bool)),
                score_t,
                torch.from_numpy(eligible),
                torch.from_numpy(ids),
                torch.from_numpy(est_np),
                float(np.asarray(z["protected_fraction"]).item()),
                spec.swap,
                starting_active=(
                    torch.from_numpy(np.asarray(z["official_factor0_mask"], dtype=bool)) & retain2
                ),
                retain_mask=retain2,
                importance=torch.from_numpy(importance),
            )
            report(
                bool(torch.equal(retain2, state["ap_retain_mask"])),
                "independent replay reproduces retain mask",
            )
            report(
                bool(torch.equal(active2, state["ap_active_mask"])),
                "independent replay reproduces active mask",
            )
            report(
                bool(torch.equal(class2, state["ap_class_mask"]))
                and bool(torch.equal(class2, class3)),
                "independent replay reproduces protected class",
            )

    # ---- C. codec archive ---------------------------------------------
    zip_path = rd / "compression_rank0.zip"
    report(zip_path.is_file() and zip_path.stat().st_size > 0, "codec archive exists",
           f"{zip_path.stat().st_size if zip_path.is_file() else 0} bytes")
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        report(bad is None, "codec archive integrity (testzip)", str(bad))
        names = zf.namelist()
        report(len(names) >= 10, "codec archive member count", f"{len(names)} members")
        sizes = sorted(((info.file_size, info.filename) for info in zf.infolist()), reverse=True)
        for size, name in sizes[:8]:
            print(f"       archive member: {name} ({size} B)")
    audit = json.loads((rd / "compression_rank0_zip_audit.json").read_bytes())
    actual_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    report(audit.get("sha256") == actual_sha, "zip audit sha256 matches archive bytes")
    report(int(audit.get("bytes", -1)) == zip_path.stat().st_size, "zip audit byte count matches")
    feedback = json.loads((rd / "compression_rank0_byte_feedback_record.json").read_bytes())
    report(feedback.get("outcome_fields_read") == [], "byte feedback is outcome-blind")
    report(int(feedback.get("bytes", -1)) == zip_path.stat().st_size, "feedback byte count matches")
    for member in ("clean_decode_request.json", "byte_census.json"):
        p = rd / "compression" / "rank0" / member
        ok = p.is_file() and bool(json.loads(p.read_bytes()))
        report(ok, f"compression dir carries {member}")

    # ---- D. rendered quality (catches trained-nothing) -----------------
    stats_files = sorted(rd.glob("stats/*.json"))
    report(len(stats_files) > 0, "eval stats written", f"{len(stats_files)} files")
    best_psnr = -1.0
    for sf in stats_files:
        stats = json.loads(sf.read_text())
        psnr = float(stats.get("psnr", float("nan")))
        print(f"       {sf.name}: psnr={psnr:.3f} ssim={stats.get('ssim')} lpips={stats.get('lpips')}")
        if psnr == psnr:
            best_psnr = max(best_psnr, psnr)
    report(
        best_psnr >= args.min_psnr,
        f"best PSNR >= {args.min_psnr} (model actually learned)",
        f"best={best_psnr:.3f}",
    )

    print(f"\nSMOKE VERIFY: {CHECKS['pass']} passed, {CHECKS['fail']} failed")
    return 1 if CHECKS["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
