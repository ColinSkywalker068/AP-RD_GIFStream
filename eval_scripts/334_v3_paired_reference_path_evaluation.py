#!/usr/bin/env python3
"""V3 paired own-reference path evaluation with the composed D_path metric.

Fork of evaluator 234 for the v3 campaign (task 1 of the handover).  The
scientific core -- identity matching, per-identity MTE, top-10% action
subset, bbox normalization -- is carried over verbatim.  Differences:

* consumes the v3 campaign's artifact layout directly (reference bundles from
  the v3 exporter / H-DOWN export, candidate bundles from clean-decode
  output dirs) instead of the author's archived producer TARs;
* the AP counted-KNN call uses the current runtime API
  (``deterministic_knn_indices(..., canonical_ids=...)``); the official
  policy reproduces the Patch5 sklearn KDTree exactly, as in 234;
* adds the composed metric  ``D_path = sum(matched per-identity errors)
  + missing_count * pi``  with uniform weights, where ``pi`` is the median
  own-reference action of the frozen reference set (outcome-blind), printed
  explicitly, and swept over the campaign multipliers (0.5x/1x/2x) to show
  the conclusion direction is pi-stable.

Consumes only clean-decode products and pre-codec reference bundles; no
training-time tensors are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

OUTPUT_SCHEMA = "h007.v3_paired_reference_path_evaluation.v1"
CAMPAIGN_SCHEMA = "h007.v3_campaign.flame_salmon.v1"
OFFICIAL_KNN_POLICY = "official-patch5-sklearn-kdtree"
AP_KNN_POLICY = "ap-deterministic-canonical-id-knn"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def load_campaign(path: Path) -> dict:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if campaign.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError("unsupported campaign schema")
    return campaign


# --- runtime imports (repo root supplied on the CLI) -----------------------

def bind_runtime(repo_root: Path):
    sys.path.insert(0, str(repo_root / "examples"))
    sys.path.insert(0, str(repo_root))
    import torch  # noqa: F401
    from h007_hdown_final import load_bundle, canonical_ids  # noqa: E501
    from gsplat.compression.h007_clean_runtime import decode_anchor_paths
    from gsplat.compression.h007_path_contract import deterministic_knn_indices
    from sklearn.neighbors import KDTree
    return load_bundle, canonical_ids, decode_anchor_paths, deterministic_knn_indices, KDTree


# --- verbatim 234 core (identity resolution) --------------------------------

def _resolve_candidate_rows(
    reference_ids: Sequence[Sequence[int]],
    candidate_ids: Sequence[Sequence[int]],
) -> Tuple[list, Dict[str, int]]:
    candidate_groups: Dict[Tuple[int, int, int], list] = {}
    for index, row in enumerate(candidate_ids):
        key = tuple(int(value) for value in row)
        if len(key) != 3:
            raise ValueError("candidate ID does not have three components")
        candidate_groups.setdefault(key, []).append(index)
    duplicate_groups = {
        key: rows for key, rows in candidate_groups.items() if len(rows) > 1
    }
    candidate_rows = []
    collision_count = 0
    for row in reference_ids:
        key = tuple(int(value) for value in row)
        if len(key) != 3:
            raise ValueError("reference ID does not have three components")
        rows = candidate_groups.get(key, [])
        if len(rows) == 1:
            candidate_rows.append(rows[0])
        else:
            if len(rows) > 1:
                collision_count += 1
            candidate_rows.append(-1)
    return candidate_rows, {
        "candidate_duplicate_canonical_id_key_count": len(duplicate_groups),
        "candidate_duplicate_canonical_id_row_count": sum(
            len(rows) for rows in duplicate_groups.values()
        ),
        "candidate_duplicate_canonical_id_excess_row_count": sum(
            len(rows) - 1 for rows in duplicate_groups.values()
        ),
        "reference_identity_count_with_candidate_collision": collision_count,
    }


# --- metrics ----------------------------------------------------------------

def _subset_metrics(
    indices: np.ndarray,
    matched: np.ndarray,
    per_identity_error: np.ndarray,
    reference_action: np.ndarray,
    bbox_diagonal: float,
    pi_values: Mapping[str, float],
) -> Dict[str, Any]:
    """234's subset metrics, extended with the composed D_path per pi value.

    Legacy fields keep 234's semantics exactly (penalty = bbox diagonal);
    the ``d_path`` block implements task 1 with pi decoupled from the
    normalizer.
    """
    subset_matched = matched[indices]
    matched_errors = per_identity_error[indices][subset_matched]
    missing_count = int((~subset_matched).sum())
    count = int(indices.size)
    matched_only = float(matched_errors.mean()) if matched_errors.size else None
    matched_error_sum = float(matched_errors.sum()) if matched_errors.size else 0.0
    penalized = float(
        (matched_error_sum + missing_count * bbox_diagonal) / float(count)
    )
    action = reference_action[indices]
    action_total = float(action.sum())
    missing_action = float(action[~subset_matched].sum())
    missing_action_fraction = (
        missing_action / action_total
        if action_total > 0
        else float(missing_count / count)
    )
    d_path = {}
    for label, pi in pi_values.items():
        total = matched_error_sum + missing_count * float(pi)
        d_path[label] = {
            "pi": float(pi),
            "d_path": total,
            "d_path_per_identity": total / float(count),
            "bbox_normalized_d_path_per_identity": total / float(count) / bbox_diagonal,
        }
    return {
        "reference_identity_count": count,
        "matched_identity_count": int(subset_matched.sum()),
        "missing_identity_count": missing_count,
        "missing_identity_fraction": float(missing_count / count),
        "matched_error_sum": matched_error_sum,
        "raw_matched_only_mte": matched_only,
        "raw_penalized_mte": penalized,
        "bbox_normalized_matched_only_mte": (
            matched_only / bbox_diagonal if matched_only is not None else None
        ),
        "bbox_normalized_penalized_mte": penalized / bbox_diagonal,
        "reference_action_total": action_total,
        "missing_reference_action": missing_action,
        "missing_reference_action_fraction": missing_action_fraction,
        "d_path": d_path,
    }


def _audit_gop(
    runtime,
    reference_bundle: Path,
    candidate_bundle: Path,
    method: str,
    ap_method_name: str,
    pi_multipliers: Sequence[float],
    device_name: str,
) -> Dict[str, Any]:
    load_bundle, canonical_ids, decode_anchor_paths, deterministic_knn_indices, KDTree = runtime
    import torch

    device = torch.device(device_name)

    def knn_for(splats, config, ids):
        if not bool(config["knn"]):
            return None
        count = int(config["n_knn"])
        anchors = splats["anchors"]
        if method == ap_method_name:
            return deterministic_knn_indices(anchors, count, canonical_ids=ids)
        if count <= 0 or anchors.shape[0] <= count:
            raise ValueError("counted KNN request exceeds decoded anchor population")
        points = anchors.detach().cpu().numpy().astype(np.float64, copy=False)
        _, indices = KDTree(points).query(points, k=count + 1)
        return torch.from_numpy(indices[:, 1:].astype(np.int64, copy=False)).to(
            anchors.device
        )

    with torch.no_grad():
        _, ref_splats, ref_decoders, _, ref_config, _, _, _ = load_bundle(
            reference_bundle, device, reference=True
        )
        _, cand_splats, cand_decoders, _, cand_config, _, _, clean_manifest = load_bundle(
            candidate_bundle, device, reference=False
        )
        for key in ("scene", "start_frame", "GOP_size", "voxel_size", "knn", "n_knn"):
            if cand_config[key] != ref_config[key]:
                raise ValueError(f"candidate/reference configuration differs: {key}")
        voxel_size = float(ref_config["voxel_size"])
        reference_ids = canonical_ids(ref_splats["anchors"], voxel_size)
        if method == ap_method_name:
            manifest = json.loads(
                (candidate_bundle / "clean_decode_manifest.json").read_text()
            )
            if manifest.get("decoded_canonical_ids") != "decoded_canonical_ids.npy":
                raise ValueError("AP clean decode lacks its restored identity output")
            sidecar = candidate_bundle / "decoded_canonical_ids.npy"
            if sha256_file(sidecar) != manifest.get("decoded_canonical_ids_sha256"):
                raise ValueError("AP restored identities differ from clean-decode receipt")
            values = np.load(sidecar, allow_pickle=False)
            if (
                values.dtype != np.int64
                or values.ndim != 2
                or values.shape[1] != 3
                or values.shape[0] != int(cand_splats["anchors"].shape[0])
                or np.unique(values, axis=0).shape[0] != values.shape[0]
            ):
                raise ValueError("AP restored identity output is malformed")
            candidate_ids = torch.from_numpy(values.copy()).to(
                device=cand_splats["anchors"].device, dtype=torch.int64
            )
        else:
            candidate_ids = torch.round(
                cand_splats["anchors"] / voxel_size
            ).to(torch.int64)

        reference_paths = decode_anchor_paths(
            ref_splats, ref_decoders, ref_config,
            knn_indices=knn_for(ref_splats, ref_config, reference_ids),
        )
        candidate_paths = decode_anchor_paths(
            cand_splats, cand_decoders, cand_config,
            knn_indices=knn_for(cand_splats, cand_config, candidate_ids),
        )
        reference_ids_np = reference_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        candidate_ids_np = candidate_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        if np.unique(reference_ids_np, axis=0).shape[0] != reference_ids_np.shape[0]:
            raise ValueError("pre-codec reference identities are not unique")
        candidate_rows_list, collision = _resolve_candidate_rows(
            reference_ids_np.tolist(), candidate_ids_np.tolist()
        )
        candidate_rows = np.asarray(candidate_rows_list, dtype=np.int64)
        matched = candidate_rows >= 0
        per_identity_error = np.full(reference_ids_np.shape[0], np.nan, dtype=np.float64)
        if bool(matched.any()):
            reference_match = reference_paths[
                torch.from_numpy(np.flatnonzero(matched)).to(device)
            ]
            candidate_match = candidate_paths[
                torch.from_numpy(candidate_rows[matched]).to(device)
            ]
            per_identity_error[matched] = (
                torch.linalg.vector_norm(reference_match - candidate_match, dim=-1)
                .mean(dim=1)
                .detach().cpu().numpy().astype(np.float64, copy=False)
            )
        reference_action = (
            torch.linalg.vector_norm(
                reference_paths[:, 1:] - reference_paths[:, :-1], dim=-1
            )
            .sum(dim=1)
            .detach().cpu().numpy().astype(np.float64, copy=False)
        )
        bounds_min = ref_splats["anchors"].amin(dim=0)
        bounds_max = ref_splats["anchors"].amax(dim=0)
        bbox_diagonal = float(torch.linalg.vector_norm(bounds_max - bounds_min).item())
        if not math.isfinite(bbox_diagonal) or bbox_diagonal <= 0:
            raise ValueError("reference bounding-box penalty is invalid")

        # Task-1 pi, both readings of the handover's "motion quantity", each
        # reference-only and outcome-blind, swept over campaign multipliers:
        #   action = total own-reference trajectory length (path_score
        #            semantics: "a typical identity's entire motion");
        #   motion = mean distance of the own-reference path from the anchor
        #            (motion_score semantics).
        reference_motion = (
            torch.linalg.vector_norm(
                reference_paths - ref_splats["anchors"][:, None, :], dim=-1
            )
            .mean(dim=1)
            .detach().cpu().numpy().astype(np.float64, copy=False)
        )
        moving = reference_action[reference_action > 0]
        pi_bases = {
            "action": float(np.median(reference_action)),
            "motion": float(np.median(reference_motion)),
            # Supplementary, non-degenerate reading: the median motion of the
            # identities that move at all (the full-set medians are zero on
            # scenes where motion is sparse -- the paper's own premise).
            "moving": float(np.median(moving)) if moving.size else 0.0,
        }
        for name, value in pi_bases.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"reference {name} median is invalid")
        pi_values = {
            f"{name}_x{multiplier:g}": base * float(multiplier)
            for name, base in pi_bases.items()
            for multiplier in pi_multipliers
        }

        count = reference_ids_np.shape[0]
        all_indices = np.arange(count, dtype=np.int64)
        order = np.lexsort(
            (
                reference_ids_np[:, 2],
                reference_ids_np[:, 1],
                reference_ids_np[:, 0],
                -reference_action,
            )
        )
        top_indices = order[: max(1, int(math.ceil(0.10 * count)))]
        return {
            "gop_id": int(ref_config["start_frame"]) // 60,
            "start_frame": int(ref_config["start_frame"]),
            "reference_bbox_diagonal": bbox_diagonal,
            "legacy_missing_identity_penalty_rule": (
                "own-reference-anchor-bounding-box-diagonal"
            ),
            "d_path_pi_rule": {
                "action": "median-own-reference-total-path-length",
                "motion": "median-own-reference-mean-anchor-distance",
            },
            "d_path_pi_bases": pi_bases,
            "d_path_pi_values": {k: v for k, v in pi_values.items()},
            "d_path_weight_rule": "uniform_w_i_equals_1",
            "knn_policy": AP_KNN_POLICY if method == ap_method_name else OFFICIAL_KNN_POLICY,
            "top10_rule": (
                "top-ceil-10%-own-reference-path-length-with-canonical-ID-tie-break"
            ),
            "candidate_canonical_id_source": (
                "exact-restored-container-sidecar"
                if method == ap_method_name
                else "decoded-anchor-round-to-own-reference-voxel-size"
            ),
            "candidate_identity_resolution_rule": (
                "unique-ID direct; duplicate-ID unresolved and counted as missing"
            ),
            **collision,
            "global": _subset_metrics(
                all_indices, matched, per_identity_error,
                reference_action, bbox_diagonal, pi_values,
            ),
            "top10": _subset_metrics(
                top_indices, matched, per_identity_error,
                reference_action, bbox_diagonal, pi_values,
            ),
            "reference_bundle_manifest_sha256": sha256_file(
                reference_bundle / "reference_bundle_manifest.json"
            ),
            "clean_decode_manifest_sha256": sha256_file(
                candidate_bundle / "clean_decode_manifest.json"
            ),
            "decoded_splats_sha256": clean_manifest["decoded_splats_sha256"],
        }


def _equal_gop_mean(rows: Sequence[Mapping[str, Any]], pi_labels: Sequence[str]) -> Dict[str, Any]:
    def mean_of(path):
        values = []
        for row in rows:
            value = row
            for key in path:
                value = value[key]
            if value is None:
                return None
            values.append(float(value))
        return sum(values) / len(values)

    result: Dict[str, Any] = {}
    for subset in ("global", "top10"):
        block = {
            "raw_penalized_mte": mean_of((subset, "raw_penalized_mte")),
            "bbox_normalized_penalized_mte": mean_of((subset, "bbox_normalized_penalized_mte")),
            "missing_identity_fraction": mean_of((subset, "missing_identity_fraction")),
            "missing_reference_action_fraction": mean_of(
                (subset, "missing_reference_action_fraction")
            ),
            "d_path": {},
        }
        for label in pi_labels:
            block["d_path"][label] = {
                "d_path_per_identity": mean_of((subset, "d_path", label, "d_path_per_identity")),
                "bbox_normalized_d_path_per_identity": mean_of(
                    (subset, "d_path", label, "bbox_normalized_d_path_per_identity")
                ),
            }
        result[subset] = block
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--rate", type=int, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--official-reference-root", type=Path, required=True)
    parser.add_argument("--ap-reference-root", type=Path, required=True)
    parser.add_argument("--official-decode-root", type=Path, required=True)
    parser.add_argument("--ap-decode-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    campaign = load_campaign(args.campaign)
    if str(args.rate) not in campaign["rates"]:
        raise ValueError(f"rate {args.rate} outside campaign grid")
    gop_count = int(campaign["gop_count"])
    ap_method = str(campaign["method"])
    official_method = str(campaign["official_method"])
    multipliers = [float(m) for m in campaign["d_path"]["pi_sweep_multipliers"]]
    pi_labels = [
        f"{name}_x{m:g}"
        for name in ("action", "motion", "moving")
        for m in multipliers
    ]
    runtime = bind_runtime(args.repo_root.resolve(strict=True))

    methods: Dict[str, Any] = {}
    for method, ref_root, decode_root in (
        (official_method, args.official_reference_root, args.official_decode_root),
        (ap_method, args.ap_reference_root, args.ap_decode_root),
    ):
        rows = []
        for gop_id in range(gop_count):
            reference_bundle = ref_root / f"gop_{gop_id}"
            candidate_bundle = (
                decode_root / f"GOP_{gop_id}" / f"r{args.rate}" / "clean_decode"
            )
            print(f"[334v3] auditing {method} rate {args.rate} GOP {gop_id}", flush=True)
            row = _audit_gop(
                runtime, reference_bundle, candidate_bundle, method,
                ap_method, multipliers, args.device,
            )
            print(
                f"[334v3]   pi_bases={ {k: round(v, 6) for k, v in row['d_path_pi_bases'].items()} } "
                f"pi_values={ {k: round(v, 6) for k, v in row['d_path_pi_values'].items()} } "
                f"global_missing={row['global']['missing_identity_count']} "
                f"global_d_path(action_x1)={row['global']['d_path'].get('action_x1', {}).get('d_path')}",
                flush=True,
            )
            rows.append(row)
        methods[method] = {
            "per_gop": rows,
            "equal_gop_mean": _equal_gop_mean(rows, pi_labels),
        }

    # Paired deltas (AP minus official) and pi-sweep direction stability.
    deltas: Dict[str, Any] = {}
    direction: Dict[str, Any] = {}
    for subset in ("global", "top10"):
        deltas[subset] = {}
        signs = {}
        for label in pi_labels:
            ap_value = methods[ap_method]["equal_gop_mean"][subset]["d_path"][label][
                "bbox_normalized_d_path_per_identity"
            ]
            official_value = methods[official_method]["equal_gop_mean"][subset]["d_path"][
                label
            ]["bbox_normalized_d_path_per_identity"]
            delta = ap_value - official_value
            deltas[subset][label] = {
                "ap": ap_value,
                "official": official_value,
                "ap_minus_official": delta,
                "ap_improves": delta < 0,
            }
            signs[label] = delta < 0
        direction[subset] = {
            "ap_improves_at": signs,
            "conclusion_direction_stable_across_pi_sweep": len(set(signs.values())) == 1,
            "stable_within_action_definition": len(
                {v for k, v in signs.items() if k.startswith("action_")}
            ) == 1,
            "stable_within_motion_definition": len(
                {v for k, v in signs.items() if k.startswith("motion_")}
            ) == 1,
            "stable_within_moving_definition": len(
                {v for k, v in signs.items() if k.startswith("moving_")}
            ) == 1,
        }

    payload = {
        "schema": OUTPUT_SCHEMA,
        "campaign_sha256": sha256_file(args.campaign),
        "scene": campaign["scene"],
        "rate": args.rate,
        "rd_lambda": campaign["rates"][str(args.rate)],
        "pi_definition": campaign["d_path"]["pi_definition"],
        "pi_sweep_multipliers": multipliers,
        "methods": methods,
        "d_path_deltas_ap_minus_official": deltas,
        "pi_sweep_direction": direction,
        "consumed_inputs": "clean-decode products and pre-codec reference bundles only",
        "outcome_fields_read": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(payload))
    print(json.dumps({
        "output": str(args.output),
        "pi_sweep_direction": direction,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
