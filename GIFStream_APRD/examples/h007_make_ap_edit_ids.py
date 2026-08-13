#!/usr/bin/env python3
"""Freeze exact parent-anchor IDs for the H007 recolor witness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SELECTION = "top_path_score_intersection_official_and_ap_retained"


def _ids_sha256(ids: np.ndarray) -> str:
    value = np.asarray(ids, dtype="<i8")
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-artifact", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("edit anchor count must be positive")
    score_payload = args.score_artifact.read_bytes()
    score_sha256 = hashlib.sha256(score_payload).hexdigest()
    reference_payload = args.reference_manifest.read_bytes()
    reference = json.loads(reference_payload.decode("utf-8"))
    required_reference = {
        "schema",
        "scene",
        "source_score_sha256",
        "selection",
        "selection_count",
        "selected_canonical_ids_sha256",
    }
    if set(reference) != required_reference:
        raise ValueError("edit reference manifest fields are incomplete or unexpected")
    if reference["schema"] != "h007.ap_edit_reference_manifest.v1":
        raise ValueError("unsupported edit reference manifest schema")
    if reference["scene"] != "flame_salmon_1":
        raise ValueError("edit reference manifest scene mismatch")
    if reference["source_score_sha256"] != score_sha256:
        raise ValueError("edit reference manifest score SHA-256 mismatch")
    if reference["selection"] != SELECTION or int(reference["selection_count"]) != int(
        args.count
    ):
        raise ValueError("edit reference manifest selection/count mismatch")
    with np.load(args.score_artifact, allow_pickle=False) as score:
        if str(np.asarray(score["schema"]).item()) != "h007.ap_scores.v3":
            raise ValueError("unsupported AP score schema")
        if str(np.asarray(score["scene"]).item()) != "flame_salmon_1":
            raise ValueError("U3 edit selection is development-locked to flame_salmon_1")
        ids = np.asarray(score["canonical_ids"], dtype=np.int64)
        path_score = np.asarray(score["path_score"], dtype=np.float64)
        eligible = np.asarray(score["eligible"], dtype=np.bool_)
        official_retain = np.asarray(score["official_retain_mask"], dtype=np.bool_)
        ap_retain = np.asarray(score["ap_retain_mask"], dtype=np.bool_)
        voxel_size = float(np.asarray(score["voxel_size"]).item())
    if ids.ndim != 2 or ids.shape[1] != 3:
        raise ValueError("malformed canonical IDs")
    universe = eligible & official_retain & ap_retain & np.isfinite(path_score)
    candidates = np.flatnonzero(universe)
    if candidates.size < args.count:
        raise ValueError(
            f"only {candidates.size} exactly shared retained anchors for {args.count} edits"
        )
    order = np.lexsort(
        (
            ids[candidates, 2],
            ids[candidates, 1],
            ids[candidates, 0],
            -path_score[candidates],
        )
    )
    chosen = candidates[order[: args.count]]
    chosen_ids = ids[chosen]
    if _ids_sha256(chosen_ids) != reference["selected_canonical_ids_sha256"]:
        raise ValueError("deterministic edit selection differs from preregistered ID hash")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        schema=np.asarray("h007.ap_edit_ids.v1"),
        scene=np.asarray("flame_salmon_1"),
        voxel_size=np.asarray(voxel_size, dtype=np.float64),
        canonical_ids=chosen_ids,
        source_score_sha256=np.asarray(score_sha256),
        selection=np.asarray(SELECTION),
        reference_manifest_sha256=np.asarray(
            hashlib.sha256(reference_payload).hexdigest()
        ),
        selected_canonical_ids_sha256=np.asarray(_ids_sha256(chosen_ids)),
        path_score=path_score[chosen],
    )
    print(
        json.dumps(
            {
                "schema": "h007.ap_edit_ids_build.v1",
                "output": str(args.output.resolve()),
                "count": int(chosen.size),
                "outcome_fields_read": [],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
