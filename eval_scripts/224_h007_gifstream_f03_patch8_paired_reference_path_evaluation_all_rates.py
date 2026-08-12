#!/usr/bin/env python3
"""Evaluate each codec against its own pre-codec identity/path reference.

This is the fresh f03 Patch8 all-valid-rate evaluator.  It preserves the
paired-own-reference metric and seven-gate path verdict from evaluator 217.
A valid immutable pair is decoded and measured even when the separately
reported practical-rate/render match is false.

The evaluator consumes an outcome-blind match manifest emitted by script 223.
That manifest binds both complete producer TARs, sequences, ordinary receipts,
and reference trees before this fresh replay's path outcome is read.  Match
eligibility is retained as a comparison stratum, never relabelled, and cannot
block path execution.  Raw MTE is reported for each method, while cross-method
reductions use bounding-box-normalized MTE because independently trained
method references need not have the same spatial scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


SCHEMA = "h007.ap_gifstream.f03_patch8_paired_reference_path_evaluation.v1"
MATCH_SCHEMA = "h007.ap_gifstream.f03_patch8_paired_reference_match.v1"
PROTOCOL_SCHEMA = "h007.ap_gifstream.f03_patch8_complete_evaluation_protocol.v1"
PATH_EXECUTION_POLICY = "always-evaluate-valid-pair-regardless-of-practical-match"
SCENE = "flame_salmon_1"
GOP_COUNT = 5
OFFICIAL_TAR_SHA256 = (
    "9a18296fd04b0b75810805099fa4f3affc362a7d12bf3c798b587859d62f303f"
)
AP_PROVENANCE_SHA256 = (
    "71e10c5e5226e68743812151b1da0e2f4009a7056454dda6fac95cb8067bb264"
)
AP_NORMALIZED_TREE_SHA256 = (
    "0f94bc919c616f30bdf8b32d1e28f8a80668bae04178aa10a0ee182def40b64f"
)
PATCH8_SHA256 = (
    "1d9d2c96e773f2e77e5bfdf15e87912402c1de207d0c0ac78a52526b93141bbb"
)
AP_TASK_ID = "DEV.flame_salmon_1.ap.f03-patch8-ownref-rep1.p0c32"

# Loaded only for a real evaluation.  Keeping these imports lazy permits the
# dependency-free self-test to run on the local orchestration host.
np: Any = None
torch: Any = None
counted_knn_indices: Any = None
decode_anchor_paths: Any = None
canonical_ids: Any = None
load_bundle: Any = None


def _load_runtime_dependencies() -> None:
    global np, torch, counted_knn_indices, decode_anchor_paths, canonical_ids
    global load_bundle
    if torch is not None:
        return
    import numpy as numpy_module
    import torch as torch_module
    from gsplat.compression.h007_clean_runtime import (
        counted_knn_indices as counted_knn_indices_function,
    )
    from gsplat.compression.h007_clean_runtime import (
        decode_anchor_paths as decode_anchor_paths_function,
    )
    from h007_hdown_final import canonical_ids as canonical_ids_function
    from h007_hdown_final import load_bundle as load_bundle_function

    np = numpy_module
    torch = torch_module
    counted_knn_indices = counted_knn_indices_function
    decode_anchor_paths = decode_anchor_paths_function
    canonical_ids = canonical_ids_function
    load_bundle = load_bundle_function


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("producer extraction destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:") as handle:
        names = set()
        for member in handle.getmembers():
            name = member.name
            relative = Path(name)
            target = (destination / relative).resolve()
            if (
                not name
                or name in names
                or relative.is_absolute()
                or ".." in relative.parts
                or "\\" in name
                or member.issym()
                or member.islnk()
                or member.isdev()
                or member.isfifo()
                or os.path.commonpath([str(root), str(target)]) != str(root)
            ):
                raise ValueError(f"unsafe producer TAR member: {name}")
            names.add(name)
        handle.extractall(destination)


def _reference_tree_digest(reference_root: Path) -> Dict[str, Any]:
    rows = []
    for path in sorted(reference_root.rglob("*")):
        if path.is_file():
            rows.append((path.relative_to(reference_root).as_posix(), path.read_bytes()))
    digest = hashlib.sha256()
    for name, payload in rows:
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return {"file_count": len(rows), "tree_sha256": digest.hexdigest()}


def _validate_producer_binding(
    match: Mapping[str, Any], label: str, archive: Path, root: Path
) -> Dict[str, Any]:
    row = match[label]
    if archive.stat().st_size != int(row["producer_tar_bytes"]) or sha256_file(
        archive
    ) != row["producer_tar_sha256"]:
        raise ValueError(f"{label} producer TAR differs from paired match")
    for name, byte_field, hash_field in (
        ("completion_marker.json", "completion_marker_bytes", "completion_marker_sha256"),
        ("ordinary_receipt.json", "ordinary_receipt_bytes", "ordinary_receipt_sha256"),
        ("sequence.zip", "sequence_bytes", "sequence_sha256"),
    ):
        path = root / name
        if (
            path.stat().st_size != int(row[byte_field])
            or sha256_file(path) != row[hash_field]
        ):
            raise ValueError(f"{label} {name} differs from paired match")
    completion = json.loads(
        (root / "completion_marker.json").read_text(encoding="utf-8")
    )
    reference_binding = _reference_tree_digest(root / "references")
    registered_reference = row["reference_bundle_binding"]
    if reference_binding != {
        "file_count": int(registered_reference["file_count"]),
        "tree_sha256": registered_reference["tree_sha256"],
    }:
        raise ValueError(f"{label} reference tree differs from paired match")
    if (
        int(completion.get("reference_bundle_file_count", -1))
        != reference_binding["file_count"]
        or completion.get("reference_bundle_tree_sha256")
        != reference_binding["tree_sha256"]
    ):
        raise ValueError(f"{label} reference tree differs from completion binding")
    return {
        **reference_binding,
        "producer_tar_sha256": row["producer_tar_sha256"],
        "completion_marker_sha256": row["completion_marker_sha256"],
        "reference_method": registered_reference["reference_method"],
        "reference_bundle_semantics": registered_reference[
            "reference_bundle_semantics"
        ],
        "reference_root_source_label": label,
    }


def _paired_reference_roots(
    official_root: Path, ap_root: Path
) -> Dict[str, Path]:
    roots = {
        "official": (official_root / "references").resolve(),
        "ap": (ap_root / "references").resolve(),
    }
    expected_parents = {
        "official": official_root.resolve(),
        "ap": ap_root.resolve(),
    }
    for label, root in roots.items():
        if (
            os.path.commonpath([str(expected_parents[label]), str(root)])
            != str(expected_parents[label])
        ):
            raise ValueError(f"{label} reference root escapes its producer")
    if roots["official"] == roots["ap"]:
        raise ValueError("official and AP reference roots must be producer-local")
    return roots


def _extract_and_clean_decode(
    sequence: Path,
    method: str,
    output_root: Path,
    decoder_root: Path,
    provenance_manifest: Path,
    provenance_manifest_sha256: str,
    device: str,
) -> Dict[int, Path]:
    decoder_root = decoder_root.resolve()
    decoder = decoder_root / "examples/h007_clean_decode_gifstream.py"
    if not decoder.is_file() or decoder.is_symlink():
        raise ValueError("registered clean-decoder root is unavailable")
    bundles = {}
    with zipfile.ZipFile(sequence, "r") as outer:
        manifest_rows = [
            info for info in outer.infolist() if info.filename == "sequence_manifest.json"
        ]
        if len(manifest_rows) != 1 or manifest_rows[0].is_dir():
            raise ValueError("sequence lacks one exact manifest")
        manifest = json.loads(outer.read(manifest_rows[0]).decode("utf-8"))
        if (
            manifest.get("scene") != SCENE
            or manifest.get("method") != method
            or not isinstance(manifest.get("gops"), list)
            or len(manifest["gops"]) != GOP_COUNT
        ):
            raise ValueError("sequence manifest identity or extent changed")
        for gop_id in range(GOP_COUNT):
            member = f"gops/gop_{gop_id}.zip"
            rows = [info for info in outer.infolist() if info.filename == member]
            if len(rows) != 1 or rows[0].is_dir():
                raise ValueError("sequence lacks one exact GOP member")
            payload = outer.read(rows[0])
            expected = manifest["gops"][gop_id]
            if expected.get("gop_id") != gop_id or expected.get("path") != member:
                raise ValueError("sequence GOP manifest order changed")
            if len(payload) != int(expected["bytes"]) or hashlib.sha256(
                payload
            ).hexdigest() != expected["sha256"]:
                raise ValueError("nested GOP differs from sequence manifest")
            inner = output_root / f"gop_{gop_id}.zip"
            inner.parent.mkdir(parents=True, exist_ok=True)
            inner.write_bytes(payload)
            bundle = output_root / f"bundle_{gop_id}"
            environment = os.environ.copy()
            existing_pythonpath = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = os.pathsep.join(
                [
                    str(decoder_root / "examples"),
                    str(decoder_root),
                    existing_pythonpath,
                ]
            ).rstrip(os.pathsep)
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(decoder),
                    "--archive",
                    str(inner),
                    "--sequence-archive",
                    str(sequence),
                    "--gop-id",
                    str(gop_id),
                    "--output-dir",
                    str(bundle),
                    "--device",
                    device,
                    "--provenance-manifest",
                    str(provenance_manifest),
                    "--provenance-manifest-sha256",
                    provenance_manifest_sha256,
                ],
                check=True,
                cwd=decoder_root,
                env=environment,
            )
            bundles[gop_id] = bundle
    return bundles


def _decoded_ids(
    method: str, bundle: Path, splats: Mapping[str, Any], voxel_size: float
) -> Any:
    if method == "ap-gifstream-full":
        manifest = json.loads(
            (bundle / "clean_decode_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("decoded_canonical_ids") != "decoded_canonical_ids.npy":
            raise ValueError("AP clean decode lacks its restored identity output")
        sidecar = bundle / "decoded_canonical_ids.npy"
        if sha256_file(sidecar) != manifest.get("decoded_canonical_ids_sha256"):
            raise ValueError("AP restored identities differ from clean-decode receipt")
        values = np.load(sidecar, allow_pickle=False)
        if values.dtype != np.int64 or values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("AP restored identity output is malformed")
        if values.shape[0] != int(splats["anchors"].shape[0]) or np.unique(
            values, axis=0
        ).shape[0] != values.shape[0]:
            raise ValueError("AP restored identity output is not one-to-one")
        return torch.from_numpy(values.copy()).to(
            device=splats["anchors"].device, dtype=torch.int64
        )
    values = torch.round(splats["anchors"] / float(voxel_size)).to(torch.int64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("conventional decoded canonical ID tensor is malformed")
    return values


def _resolve_candidate_rows(
    reference_ids: Sequence[Sequence[int]],
    candidate_ids: Sequence[Sequence[int]],
) -> Tuple[list[int], Dict[str, int]]:
    candidate_groups: Dict[Tuple[int, int, int], list[int]] = {}
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


def _subset_metrics(
    indices: Any,
    matched: Any,
    per_identity_error: Any,
    reference_action: Any,
    missing_penalty: float,
) -> Dict[str, Any]:
    subset_matched = matched[indices]
    matched_errors = per_identity_error[indices][subset_matched]
    missing_count = int((~subset_matched).sum())
    count = int(indices.size)
    matched_only = float(matched_errors.mean()) if matched_errors.size else None
    penalized = float(
        (matched_errors.sum() + missing_count * missing_penalty) / float(count)
    )
    action = reference_action[indices]
    action_total = float(action.sum())
    missing_action = float(action[~subset_matched].sum())
    missing_action_fraction = (
        missing_action / action_total
        if action_total > 0
        else float(missing_count / count)
    )
    return {
        "reference_identity_count": count,
        "matched_identity_count": int(subset_matched.sum()),
        "missing_identity_count": missing_count,
        "missing_identity_fraction": float(missing_count / count),
        "raw_matched_only_mte": matched_only,
        "raw_penalized_mte": penalized,
        "bbox_normalized_matched_only_mte": (
            matched_only / missing_penalty if matched_only is not None else None
        ),
        "bbox_normalized_penalized_mte": penalized / missing_penalty,
        "reference_action_total": action_total,
        "missing_reference_action": missing_action,
        "missing_reference_action_fraction": missing_action_fraction,
    }


def _audit_gop(
    reference_bundle: Path,
    candidate_bundle: Path,
    method: str,
    reference_source_label: str,
    device: Any,
) -> Dict[str, Any]:
    with torch.no_grad():
        (
            _,
            reference_splats,
            reference_decoders,
            _,
            reference_config,
            _,
            _,
            _,
        ) = load_bundle(reference_bundle, device, reference=True)
        (
            _,
            candidate_splats,
            candidate_decoders,
            _,
            candidate_config,
            _,
            _,
            clean_manifest,
        ) = load_bundle(candidate_bundle, device, reference=False)
        for key in ("scene", "start_frame", "GOP_size", "voxel_size"):
            if candidate_config[key] != reference_config[key]:
                raise ValueError(f"candidate/reference configuration differs: {key}")
        voxel_size = float(reference_config["voxel_size"])
        reference_ids = canonical_ids(reference_splats["anchors"], voxel_size)
        candidate_ids = _decoded_ids(
            method, candidate_bundle, candidate_splats, voxel_size
        )
        reference_paths = decode_anchor_paths(
            reference_splats,
            reference_decoders,
            reference_config,
            knn_indices=counted_knn_indices(reference_splats, reference_config),
        )
        candidate_paths = decode_anchor_paths(
            candidate_splats,
            candidate_decoders,
            candidate_config,
            knn_indices=counted_knn_indices(candidate_splats, candidate_config),
        )
        reference_ids_np = (
            reference_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        )
        candidate_ids_np = (
            candidate_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        )
        if np.unique(reference_ids_np, axis=0).shape[0] != reference_ids_np.shape[0]:
            raise ValueError("pre-codec reference identities are not unique")
        candidate_rows_list, collision = _resolve_candidate_rows(
            reference_ids_np.tolist(), candidate_ids_np.tolist()
        )
        candidate_rows = np.asarray(candidate_rows_list, dtype=np.int64)
        matched = candidate_rows >= 0
        per_identity_error = np.full(
            reference_ids_np.shape[0], np.nan, dtype=np.float64
        )
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
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
        reference_action = (
            torch.linalg.vector_norm(
                reference_paths[:, 1:] - reference_paths[:, :-1], dim=-1
            )
            .sum(dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        bounds_min = reference_splats["anchors"].amin(dim=0)
        bounds_max = reference_splats["anchors"].amax(dim=0)
        missing_penalty = float(
            torch.linalg.vector_norm(bounds_max - bounds_min).item()
        )
        if not math.isfinite(missing_penalty) or missing_penalty <= 0:
            raise ValueError("reference bounding-box penalty is invalid")
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
        top_count = max(1, int(math.ceil(0.10 * count)))
        top_indices = order[:top_count]
        return {
            "gop_id": int(reference_config["start_frame"]) // 60,
            "start_frame": int(reference_config["start_frame"]),
            "reference_source_label": reference_source_label,
            "reference_bundle_relative_path": (
                f"references/{SCENE}/gop_"
                f"{int(reference_config['start_frame']) // 60}"
            ),
            "reference_bbox_diagonal": missing_penalty,
            "missing_identity_penalty": missing_penalty,
            "missing_identity_penalty_rule": (
                "own-reference-anchor-bounding-box-diagonal"
            ),
            "normalization_rule": (
                "divide-each-gop-raw-error-by-own-reference-bbox-diagonal"
            ),
            "top10_rule": (
                "top-ceil-10%-own-reference-path-length-with-canonical-ID-tie-break"
            ),
            "candidate_canonical_id_source": (
                "exact-restored-container-sidecar"
                if method == "ap-gifstream-full"
                else "decoded-anchor-round-to-own-reference-voxel-size"
            ),
            "candidate_identity_resolution_rule": (
                "unique-ID direct; duplicate-ID unresolved and counted as missing"
            ),
            "candidate_identity_resolution_outcome_inputs": [],
            **collision,
            "global": _subset_metrics(
                all_indices,
                matched,
                per_identity_error,
                reference_action,
                missing_penalty,
            ),
            "top10": _subset_metrics(
                top_indices,
                matched,
                per_identity_error,
                reference_action,
                missing_penalty,
            ),
            "reference_bundle_census_sha256": sha256_file(
                reference_bundle / "byte_census.json"
            ),
            "reference_bundle_manifest_sha256": sha256_file(
                reference_bundle / "reference_bundle_manifest.json"
            ),
            "clean_decode_manifest_sha256": sha256_file(
                candidate_bundle / "clean_decode_manifest.json"
            ),
            "decoded_splats_sha256": clean_manifest["decoded_splats_sha256"],
        }


def _identity_weighted_aggregate(
    rows: list[Dict[str, Any]], field: str
) -> Dict[str, Any]:
    metrics = [row[field] for row in rows]
    reference_count = sum(row["reference_identity_count"] for row in metrics)
    matched_count = sum(row["matched_identity_count"] for row in metrics)
    missing_count = sum(row["missing_identity_count"] for row in metrics)
    raw_penalized_sum = sum(
        row["raw_penalized_mte"] * row["reference_identity_count"]
        for row in metrics
    )
    normalized_penalized_sum = sum(
        row["bbox_normalized_penalized_mte"]
        * row["reference_identity_count"]
        for row in metrics
    )
    raw_matched_sum = sum(
        (row["raw_matched_only_mte"] or 0.0) * row["matched_identity_count"]
        for row in metrics
    )
    normalized_matched_sum = sum(
        (row["bbox_normalized_matched_only_mte"] or 0.0)
        * row["matched_identity_count"]
        for row in metrics
    )
    action_total = sum(row["reference_action_total"] for row in metrics)
    missing_action = sum(row["missing_reference_action"] for row in metrics)
    return {
        "aggregation": "identity-count-weighted-descriptive-only",
        "reference_identity_count": reference_count,
        "matched_identity_count": matched_count,
        "missing_identity_count": missing_count,
        "missing_identity_fraction": missing_count / reference_count,
        "raw_matched_only_mte": (
            raw_matched_sum / matched_count if matched_count else None
        ),
        "raw_penalized_mte": raw_penalized_sum / reference_count,
        "bbox_normalized_matched_only_mte": (
            normalized_matched_sum / matched_count if matched_count else None
        ),
        "bbox_normalized_penalized_mte": (
            normalized_penalized_sum / reference_count
        ),
        "reference_action_total": action_total,
        "missing_reference_action": missing_action,
        "missing_reference_action_fraction": (
            missing_action / action_total
            if action_total > 0
            else missing_count / reference_count
        ),
        "mean_gop_raw_penalized_mte": sum(
            row["raw_penalized_mte"] for row in metrics
        )
        / len(metrics),
        "mean_gop_bbox_normalized_penalized_mte": sum(
            row["bbox_normalized_penalized_mte"] for row in metrics
        )
        / len(metrics),
    }


def _equal_gop_mean(rows: list[Dict[str, Any]], field: str) -> Dict[str, Any]:
    if len(rows) != GOP_COUNT:
        raise ValueError(f"equal-GOP aggregation requires exactly {GOP_COUNT} GOPs")
    metrics = [row[field] for row in rows]

    def required_mean(key: str) -> float:
        values = [float(row[key]) for row in metrics]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"non-finite equal-GOP metric: {field}.{key}")
        return sum(values) / float(GOP_COUNT)

    def optional_mean(key: str) -> tuple[float | None, int]:
        values = [
            float(row[key]) for row in metrics if row.get(key) is not None
        ]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"non-finite equal-GOP metric: {field}.{key}")
        return (
            sum(values) / float(len(values)) if values else None,
            len(values),
        )

    raw_matched, raw_matched_count = optional_mean("raw_matched_only_mte")
    normalized_matched, normalized_matched_count = optional_mean(
        "bbox_normalized_matched_only_mte"
    )
    return {
        "aggregation": "five-gop-equal-weight-primary",
        "gop_count": GOP_COUNT,
        "mean_reference_identity_count": required_mean(
            "reference_identity_count"
        ),
        "mean_matched_identity_count": required_mean("matched_identity_count"),
        "mean_missing_identity_count": required_mean("missing_identity_count"),
        "missing_identity_fraction": required_mean("missing_identity_fraction"),
        "raw_matched_only_mte": raw_matched,
        "raw_matched_only_valid_gop_count": raw_matched_count,
        "raw_penalized_mte": required_mean("raw_penalized_mte"),
        "bbox_normalized_matched_only_mte": normalized_matched,
        "bbox_normalized_matched_only_valid_gop_count": (
            normalized_matched_count
        ),
        "bbox_normalized_penalized_mte": required_mean(
            "bbox_normalized_penalized_mte"
        ),
        "mean_reference_action_total": required_mean("reference_action_total"),
        "mean_missing_reference_action": required_mean(
            "missing_reference_action"
        ),
        "missing_reference_action_fraction": required_mean(
            "missing_reference_action_fraction"
        ),
    }


def _paired_gop_deltas(
    official_rows: list[Dict[str, Any]],
    ap_rows: list[Dict[str, Any]],
) -> tuple[list[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    if len(official_rows) != GOP_COUNT or len(ap_rows) != GOP_COUNT:
        raise ValueError(f"paired comparison requires exactly {GOP_COUNT} GOPs")
    summaries = []
    counts = {
        scope: {
            "gop_count": GOP_COUNT,
            "raw_penalized_mte_improved_gop_count": 0,
            "bbox_normalized_penalized_mte_improved_gop_count": 0,
            "missing_reference_action_fraction_not_worse_gop_count": 0,
        }
        for scope in ("global", "top10")
    }
    for expected_gop, (official, ap) in enumerate(zip(official_rows, ap_rows)):
        if (
            int(official["gop_id"]) != expected_gop
            or int(ap["gop_id"]) != expected_gop
        ):
            raise ValueError("paired GOP rows are not aligned by GOP identity")
        summary: Dict[str, Any] = {"gop_id": expected_gop}
        for scope in ("global", "top10"):
            official_metrics = official[scope]
            ap_metrics = ap[scope]
            raw_delta = (
                float(ap_metrics["raw_penalized_mte"])
                - float(official_metrics["raw_penalized_mte"])
            )
            normalized_delta = (
                float(ap_metrics["bbox_normalized_penalized_mte"])
                - float(official_metrics["bbox_normalized_penalized_mte"])
            )
            action_delta = (
                float(ap_metrics["missing_reference_action_fraction"])
                - float(official_metrics["missing_reference_action_fraction"])
            )
            raw_improves = raw_delta < 0.0
            normalized_improves = normalized_delta < 0.0
            action_not_worse = action_delta <= 0.0
            counts[scope]["raw_penalized_mte_improved_gop_count"] += int(
                raw_improves
            )
            counts[scope][
                "bbox_normalized_penalized_mte_improved_gop_count"
            ] += int(normalized_improves)
            counts[scope][
                "missing_reference_action_fraction_not_worse_gop_count"
            ] += int(action_not_worse)
            summary[scope] = {
                "raw_penalized_mte_ap_minus_official": raw_delta,
                "bbox_normalized_penalized_mte_ap_minus_official": (
                    normalized_delta
                ),
                "missing_reference_action_fraction_ap_minus_official": (
                    action_delta
                ),
                "raw_penalized_mte_improves": raw_improves,
                "bbox_normalized_penalized_mte_improves": normalized_improves,
                "missing_reference_action_fraction_not_worse": action_not_worse,
            }
        summaries.append(summary)
    return summaries, counts


def _verdict_from_equal_gop(
    official_equal: Mapping[str, Mapping[str, Any]],
    ap_equal: Mapping[str, Mapping[str, Any]],
    improvement_counts: Mapping[str, Mapping[str, int]],
) -> tuple[str, Dict[str, bool]]:
    gates = {
        "global_equal_gop_mean_raw_penalized_mte_improves": (
            float(ap_equal["global"]["raw_penalized_mte"])
            < float(official_equal["global"]["raw_penalized_mte"])
        ),
        "top10_equal_gop_mean_raw_penalized_mte_improves": (
            float(ap_equal["top10"]["raw_penalized_mte"])
            < float(official_equal["top10"]["raw_penalized_mte"])
        ),
        "global_equal_gop_mean_bbox_normalized_penalized_mte_improves": (
            float(ap_equal["global"]["bbox_normalized_penalized_mte"])
            < float(
                official_equal["global"]["bbox_normalized_penalized_mte"]
            )
        ),
        "top10_equal_gop_mean_bbox_normalized_penalized_mte_improves": (
            float(ap_equal["top10"]["bbox_normalized_penalized_mte"])
            < float(
                official_equal["top10"]["bbox_normalized_penalized_mte"]
            )
        ),
        "global_equal_gop_mean_missing_reference_action_fraction_not_worse": (
            float(ap_equal["global"]["missing_reference_action_fraction"])
            <= float(
                official_equal["global"]["missing_reference_action_fraction"]
            )
        ),
        "global_bbox_normalized_penalized_mte_improves_at_least_3_of_5_gops": (
            int(
                improvement_counts["global"][
                    "bbox_normalized_penalized_mte_improved_gop_count"
                ]
            )
            >= 3
        ),
        "top10_bbox_normalized_penalized_mte_improves_at_least_3_of_5_gops": (
            int(
                improvement_counts["top10"][
                    "bbox_normalized_penalized_mte_improved_gop_count"
                ]
            )
            >= 3
        ),
    }
    return ("PASS" if all(gates.values()) else "MIXED"), gates


def _relative_reduction(official: float, ap: float) -> float | None:
    if official == 0.0:
        return None
    return (official - ap) / official


def _comparison_stratum(match: Mapping[str, Any]) -> Tuple[str, list[str]]:
    gates = match["gates"]
    failed_gates = sorted(name for name, passed in gates.items() if not passed)
    if gates["complete_rate"]:
        rate_stratum = "MATCHED_RATE"
    else:
        relative_rate = float(
            match["ordinary_deltas_ap_minus_official"]["relative_complete_rate"]
        )
        if relative_rate < 0.0:
            rate_stratum = "AP_LOWER_RATE_UNMATCHED"
        elif relative_rate > 0.0:
            rate_stratum = "AP_HIGHER_RATE_UNMATCHED"
        else:
            raise ValueError("zero rate delta cannot fail the complete-rate gate")
    return rate_stratum, failed_gates


def _claim_verdicts(
    path_metric_verdict: str, practical_match_eligible: bool
) -> Tuple[str | None, str | None]:
    if practical_match_eligible:
        return path_metric_verdict, path_metric_verdict
    return None, None


def _self_test() -> None:
    references = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    candidates = [(0, 0, 0), (1, 0, 0), (1, 0, 0)]
    rows, collision = _resolve_candidate_rows(references, candidates)
    assert rows == [0, -1, -1]
    assert collision == {
        "candidate_duplicate_canonical_id_key_count": 1,
        "candidate_duplicate_canonical_id_row_count": 2,
        "candidate_duplicate_canonical_id_excess_row_count": 1,
        "reference_identity_count_with_candidate_collision": 1,
    }
    with tempfile.TemporaryDirectory(prefix="h007_paired_ref_selftest_") as temporary:
        root = Path(temporary)
        official = root / "official"
        ap = root / "ap"
        (official / "references").mkdir(parents=True)
        (ap / "references").mkdir(parents=True)
        (official / "references/a").write_bytes(b"official")
        (ap / "references/a").write_bytes(b"ap")
        roots = _paired_reference_roots(official, ap)
        assert roots["official"].is_relative_to(official.resolve())
        assert roots["ap"].is_relative_to(ap.resolve())
        official_binding = _reference_tree_digest(roots["official"])
        ap_binding = _reference_tree_digest(roots["ap"])
        assert official_binding["tree_sha256"] != ap_binding["tree_sha256"]
        before = ap_binding
        (ap / "references/a").write_bytes(b"mutated")
        assert _reference_tree_digest(roots["ap"]) != before

    def synthetic_rows(
        identity_counts: Sequence[int],
        penalized_values: Sequence[float],
    ) -> list[Dict[str, Any]]:
        result = []
        for gop_id, (identity_count, value) in enumerate(
            zip(identity_counts, penalized_values)
        ):
            metrics = {
                "reference_identity_count": identity_count,
                "matched_identity_count": identity_count,
                "missing_identity_count": 0,
                "missing_identity_fraction": 0.0,
                "raw_matched_only_mte": value,
                "raw_penalized_mte": value,
                "bbox_normalized_matched_only_mte": value / 10.0,
                "bbox_normalized_penalized_mte": value / 10.0,
                "reference_action_total": float(identity_count),
                "missing_reference_action": 0.0,
                "missing_reference_action_fraction": 0.0,
            }
            result.append(
                {
                    "gop_id": gop_id,
                    "global": dict(metrics),
                    "top10": dict(metrics),
                }
            )
        return result

    official_a = synthetic_rows([10000, 1, 1, 1, 1], [10.0] * GOP_COUNT)
    official_b = synthetic_rows([1, 1, 1, 10000, 1], [10.0] * GOP_COUNT)
    ap_a = synthetic_rows([10000, 1, 1, 1, 1], [9.0, 9.0, 9.0, 11.0, 11.0])
    ap_b = synthetic_rows([1, 1, 1, 10000, 1], [9.0, 9.0, 9.0, 11.0, 11.0])
    official_equal_a = {
        scope: _equal_gop_mean(official_a, scope)
        for scope in ("global", "top10")
    }
    official_equal_b = {
        scope: _equal_gop_mean(official_b, scope)
        for scope in ("global", "top10")
    }
    ap_equal_a = {
        scope: _equal_gop_mean(ap_a, scope) for scope in ("global", "top10")
    }
    ap_equal_b = {
        scope: _equal_gop_mean(ap_b, scope) for scope in ("global", "top10")
    }
    _, counts_a = _paired_gop_deltas(official_a, ap_a)
    _, counts_b = _paired_gop_deltas(official_b, ap_b)
    verdict_a, gates_a = _verdict_from_equal_gop(
        official_equal_a, ap_equal_a, counts_a
    )
    verdict_b, gates_b = _verdict_from_equal_gop(
        official_equal_b, ap_equal_b, counts_b
    )
    assert official_equal_a == official_equal_b
    assert ap_equal_a == ap_equal_b
    assert counts_a == counts_b
    assert verdict_a == verdict_b == "PASS"
    assert gates_a == gates_b
    assert (
        _identity_weighted_aggregate(ap_a, "global")["raw_penalized_mte"]
        != _identity_weighted_aggregate(ap_b, "global")["raw_penalized_mte"]
    )
    unmatched = {
        "practical_match_eligible": False,
        "gates": {
            "complete_rate": False,
            "psnr": True,
            "ssim": True,
            "lpips": True,
        },
        "ordinary_deltas_ap_minus_official": {
            "relative_complete_rate": -0.019537542519726416,
        },
    }
    assert _comparison_stratum(unmatched) == (
        "AP_LOWER_RATE_UNMATCHED",
        ["complete_rate"],
    )
    assert _claim_verdicts("PASS", False) == (None, None)
    assert _claim_verdicts("PASS", True) == ("PASS", "PASS")
    print(
        json.dumps(
            {
                "schema": (
                    "h007.ap_gifstream.f03_patch8_paired_reference_evaluator_self_test.v1"
                ),
                "status": "PASS",
                "checks": [
                    "duplicate-candidate-IDs-are-unresolved",
                    "method-reference-roots-are-producer-local",
                    "reference-tree-hash-is-content-sensitive",
                    "equal-GOP-verdict-is-invariant-to-identity-count-weights",
                    "rate-ineligible-pair-remains-evaluable-and-labelled",
                    "rate-unmatched-top-level-and-matched-rate-verdicts-are-null",
                ],
            },
            sort_keys=True,
        )
    )


def _require_paths(args: argparse.Namespace) -> None:
    required = (
        "match_manifest",
        "match_manifest_sha256",
        "official_tar",
        "ap_tar",
        "work_dir",
        "official_decoder_root",
        "ap_decoder_root",
        "official_provenance_manifest",
        "official_provenance_manifest_sha256",
        "ap_provenance_manifest",
        "ap_provenance_manifest_sha256",
        "evaluation_protocol_registration",
        "evaluation_protocol_registration_sha256",
        "output",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"missing evaluation arguments: {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--match-manifest", type=Path)
    parser.add_argument("--match-manifest-sha256")
    parser.add_argument("--official-tar", type=Path)
    parser.add_argument("--ap-tar", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--official-decoder-root", type=Path)
    parser.add_argument("--ap-decoder-root", type=Path)
    parser.add_argument("--official-provenance-manifest", type=Path)
    parser.add_argument("--official-provenance-manifest-sha256")
    parser.add_argument("--ap-provenance-manifest", type=Path)
    parser.add_argument("--ap-provenance-manifest-sha256")
    parser.add_argument("--evaluation-protocol-registration", type=Path)
    parser.add_argument("--evaluation-protocol-registration-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    _require_paths(args)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("paired-reference output must be append-only")
    match_payload = args.match_manifest.read_bytes()
    if hashlib.sha256(match_payload).hexdigest() != args.match_manifest_sha256:
        raise ValueError("paired match-manifest SHA-256 mismatch")
    match = json.loads(match_payload.decode("utf-8"))
    if (
        match.get("schema") != MATCH_SCHEMA
        or type(match.get("practical_match_eligible")) is not bool
        or match.get("path_execution_authorized") is not True
        or match.get("path_execution_policy") != PATH_EXECUTION_POLICY
        or match.get("path_execution_conditioned_on_practical_match") is not False
        or match.get("path_outcomes_read") != []
    ):
        raise ValueError("paired reference evaluation is not registered")
    practical_gates = match.get("gates")
    if (
        not isinstance(practical_gates, dict)
        or set(practical_gates) != {"complete_rate", "psnr", "ssim", "lpips"}
        or any(type(value) is not bool for value in practical_gates.values())
        or match["practical_match_eligible"] is not all(
            practical_gates.values()
        )
    ):
        raise ValueError("paired practical-match gates are malformed")
    protocol_payload = args.evaluation_protocol_registration.read_bytes()
    if (
        hashlib.sha256(protocol_payload).hexdigest()
        != args.evaluation_protocol_registration_sha256
    ):
        raise ValueError("complete-evaluation protocol SHA-256 mismatch")
    protocol = json.loads(protocol_payload.decode("utf-8"))
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status")
        != "FROZEN_BEFORE_F03_PRODUCER_AND_PATH_OUTCOMES"
        or protocol.get("path_execution_policy") != PATH_EXECUTION_POLICY
        or protocol.get("path_outcomes_read_before_registration") != []
        or protocol.get("known_ap_producer_artifacts_before_registration") != []
        or protocol.get("practical_match_role")
        != "comparison-stratum-and-claim-eligibility-only"
    ):
        raise ValueError("complete-evaluation protocol is not registered")
    official_contract = protocol.get("official_comparator")
    if (
        not isinstance(official_contract, dict)
        or official_contract.get("producer_tar_sha256") != OFFICIAL_TAR_SHA256
        or match["official"]["producer_tar_sha256"] != OFFICIAL_TAR_SHA256
        or official_contract.get("runtime_role")
        != "independent-official-patch5"
    ):
        raise ValueError("official comparator contract changed")
    producer_contract = protocol.get("ap_producer_contract")
    if (
        not isinstance(producer_contract, dict)
        or producer_contract.get("task_id") != AP_TASK_ID
        or producer_contract.get("tar_basename_pattern")
        != "h007_p0c32_dev_ap_f03_patch8_ownref_rep1_{array_job_id}_4.tar"
        or producer_contract.get("producer_registration_basename")
        != "h007_ap_gifstream_f03_patch8_producer_registration_20260726.json"
        or producer_contract.get("producer_registration_bytes") != 5147
        or producer_contract.get("producer_registration_sha256")
        != "9b1e5de5752a28f135798ea0b7a85962b60e703ccfca9214338527c92af090af"
        or producer_contract.get("completion_schema")
        != "h007.ap_gifstream.f03_patch8_producer_completion.v1"
        or producer_contract.get("registration_schema")
        != "h007.ap_gifstream.f03_patch8_producer_registration.v1"
        or producer_contract.get("implementation_isolation_schema")
        != "h007.ap_gifstream.f03_patch8_implementation_isolation.v1"
        or match["ap"]["point_id"] != AP_TASK_ID
        or match["ap"]["ap_producer_array_job_id"]
        != match["ap_producer_array_job_id"]
        or match["ap"]["producer_registration"]["sha256"]
        != producer_contract["producer_registration_sha256"]
        or match["ap"]["producer_registration"][
            "scientific_configuration"
        ]
        != protocol.get("scientific_configuration")
    ):
        raise ValueError("AP producer contract changed")
    patch8_contract = protocol.get("patch8_contract")
    if (
        not isinstance(patch8_contract, dict)
        or patch8_contract.get("runtime_manifest_sha256")
        != AP_PROVENANCE_SHA256
        or patch8_contract.get("normalized_code_tree_sha256")
        != AP_NORMALIZED_TREE_SHA256
        or patch8_contract.get("patch8_sha256") != PATCH8_SHA256
        or patch8_contract.get("runtime_patch_count") != 9
        or match["ap"]["runtime_provenance_manifest_sha256"]
        != AP_PROVENANCE_SHA256
    ):
        raise ValueError("Patch8 runtime contract changed")
    practical_contract = protocol.get("practical_match")
    if (
        not isinstance(practical_contract, dict)
        or practical_contract.get("thresholds") != match["thresholds"]
        or practical_contract.get("inclusive") is not True
        or practical_contract.get("path_execution_gate") is not False
    ):
        raise ValueError("practical-match protocol changed")
    _load_runtime_dependencies()
    torch.manual_seed(20260715)
    np.random.seed(20260715)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    official_root = work / "official"
    ap_root = work / "ap"
    _safe_extract_tar(args.official_tar.resolve(), official_root)
    _safe_extract_tar(args.ap_tar.resolve(), ap_root)
    reference_bindings = {
        "official": _validate_producer_binding(
            match, "official", args.official_tar.resolve(), official_root
        ),
        "ap": _validate_producer_binding(
            match, "ap", args.ap_tar.resolve(), ap_root
        ),
    }
    reference_roots = _paired_reference_roots(official_root, ap_root)
    for label in ("official", "ap"):
        if (
            reference_bindings[label]["reference_root_source_label"] != label
            or match[label]["reference_bundle_binding"]["reference_source_label"]
            != label
        ):
            raise ValueError(f"{label} reference lineage label is not self-bound")
    decoded = {
        "official": _extract_and_clean_decode(
            official_root / "sequence.zip",
            "official",
            work / "decode_official",
            args.official_decoder_root,
            args.official_provenance_manifest,
            args.official_provenance_manifest_sha256,
            args.device,
        ),
        "ap": _extract_and_clean_decode(
            ap_root / "sequence.zip",
            "ap-gifstream-full",
            work / "decode_ap",
            args.ap_decoder_root,
            args.ap_provenance_manifest,
            args.ap_provenance_manifest_sha256,
            args.device,
        ),
    }
    device = torch.device(args.device)
    method_rows = {}
    for label, method in (("official", "official"), ("ap", "ap-gifstream-full")):
        rows = []
        for gop_id in range(GOP_COUNT):
            rows.append(
                _audit_gop(
                    reference_roots[label] / f"{SCENE}/gop_{gop_id}",
                    decoded[label][gop_id],
                    method,
                    label,
                    device,
                )
            )
        method_rows[label] = {
            "method": method,
            "reference_bundle_binding": reference_bindings[label],
            "reference_lineage_rule": (
                f"{label}-decode-versus-{label}-same-run-pre-codec-reference"
            ),
            "gops": rows,
            "collision_summary": {
                "candidate_duplicate_canonical_id_key_count": sum(
                    row["candidate_duplicate_canonical_id_key_count"] for row in rows
                ),
                "candidate_duplicate_canonical_id_row_count": sum(
                    row["candidate_duplicate_canonical_id_row_count"] for row in rows
                ),
                "candidate_duplicate_canonical_id_excess_row_count": sum(
                    row["candidate_duplicate_canonical_id_excess_row_count"]
                    for row in rows
                ),
                "reference_identity_count_with_candidate_collision": sum(
                    row["reference_identity_count_with_candidate_collision"]
                    for row in rows
                ),
                "resolution_rule": rows[0][
                    "candidate_identity_resolution_rule"
                ],
                "outcome_inputs": [],
            },
            "equal_gop_mean": {
                "aggregation": "five-gop-equal-weight-primary",
                "global": _equal_gop_mean(rows, "global"),
                "top10": _equal_gop_mean(rows, "top10"),
            },
            "descriptive_identity_weighted": {
                "aggregation": "identity-count-weighted-descriptive-only",
                "global": _identity_weighted_aggregate(rows, "global"),
                "top10": _identity_weighted_aggregate(rows, "top10"),
            },
        }
    official_equal = method_rows["official"]["equal_gop_mean"]
    ap_equal = method_rows["ap"]["equal_gop_mean"]
    official_global = official_equal["global"]
    ap_global = ap_equal["global"]
    official_top = official_equal["top10"]
    ap_top = ap_equal["top10"]
    paired_gop_deltas, improvement_counts = _paired_gop_deltas(
        method_rows["official"]["gops"], method_rows["ap"]["gops"]
    )
    equal_gop_reductions = {
        "aggregation": "five-gop-equal-weight-primary",
        "global_raw_penalized_mte_relative": _relative_reduction(
            official_global["raw_penalized_mte"],
            ap_global["raw_penalized_mte"],
        ),
        "global_bbox_normalized_penalized_mte_relative": _relative_reduction(
            official_global["bbox_normalized_penalized_mte"],
            ap_global["bbox_normalized_penalized_mte"],
        ),
        "global_bbox_normalized_matched_only_mte_relative": _relative_reduction(
            official_global["bbox_normalized_matched_only_mte"],
            ap_global["bbox_normalized_matched_only_mte"],
        )
        if official_global["bbox_normalized_matched_only_mte"] is not None
        and ap_global["bbox_normalized_matched_only_mte"] is not None
        else None,
        "top10_raw_penalized_mte_relative": _relative_reduction(
            official_top["raw_penalized_mte"],
            ap_top["raw_penalized_mte"],
        ),
        "top10_bbox_normalized_penalized_mte_relative": _relative_reduction(
            official_top["bbox_normalized_penalized_mte"],
            ap_top["bbox_normalized_penalized_mte"],
        ),
        "top10_bbox_normalized_matched_only_mte_relative": _relative_reduction(
            official_top["bbox_normalized_matched_only_mte"],
            ap_top["bbox_normalized_matched_only_mte"],
        )
        if official_top["bbox_normalized_matched_only_mte"] is not None
        and ap_top["bbox_normalized_matched_only_mte"] is not None
        else None,
    }
    official_identity = method_rows["official"][
        "descriptive_identity_weighted"
    ]
    ap_identity = method_rows["ap"]["descriptive_identity_weighted"]
    identity_weighted_reductions = {
        "aggregation": "identity-count-weighted-descriptive-only",
        "global_raw_penalized_mte_relative": _relative_reduction(
            official_identity["global"]["raw_penalized_mte"],
            ap_identity["global"]["raw_penalized_mte"],
        ),
        "global_bbox_normalized_penalized_mte_relative": _relative_reduction(
            official_identity["global"]["bbox_normalized_penalized_mte"],
            ap_identity["global"]["bbox_normalized_penalized_mte"],
        ),
        "top10_raw_penalized_mte_relative": _relative_reduction(
            official_identity["top10"]["raw_penalized_mte"],
            ap_identity["top10"]["raw_penalized_mte"],
        ),
        "top10_bbox_normalized_penalized_mte_relative": _relative_reduction(
            official_identity["top10"]["bbox_normalized_penalized_mte"],
            ap_identity["top10"]["bbox_normalized_penalized_mte"],
        ),
    }
    verdict, verdict_gates = _verdict_from_equal_gop(
        official_equal, ap_equal, improvement_counts
    )
    rate_stratum, failed_practical_match_gates = _comparison_stratum(match)
    top_level_verdict, matched_rate_claim_verdict = _claim_verdicts(
        verdict, match["practical_match_eligible"]
    )
    payload = {
        "schema": SCHEMA,
        "scene": SCENE,
        "paired_match_manifest": str(args.match_manifest.resolve()),
        "paired_match_manifest_sha256": args.match_manifest_sha256,
        "outcome_blind_match_verified_before_path_decode": True,
        "evaluation_protocol_registration_sha256": (
            args.evaluation_protocol_registration_sha256
        ),
        "path_execution_policy": PATH_EXECUTION_POLICY,
        "path_execution_was_conditioned_on_practical_match": False,
        "practical_match": {
            "eligible": match["practical_match_eligible"],
            "rate_gate_eligible": practical_gates["complete_rate"],
            "rate_stratum": rate_stratum,
            "failed_gates": failed_practical_match_gates,
            "role": "reported-comparison-stratum-not-path-execution-gate",
            "thresholds": match["thresholds"],
            "gate_comparison_policy": match["gate_comparison_policy"],
            "gates": practical_gates,
            "ordinary_deltas_ap_minus_official": (
                match["ordinary_deltas_ap_minus_official"]
            ),
            "exact_practical_match_disclosure": (
                match["exact_practical_match_disclosure"]
            ),
        },
        "reference_comparison_design": (
            "paired-own-reference: official-to-official and AP-to-AP"
        ),
        "cross_lineage_identity_matching_performed": False,
        "official": method_rows["official"],
        "ap": method_rows["ap"],
        "paired_gop_deltas": paired_gop_deltas,
        "improvement_counts": improvement_counts,
        "equal_gop_relative_reductions": equal_gop_reductions,
        "descriptive_identity_weighted_relative_reductions": (
            identity_weighted_reductions
        ),
        "verdict": top_level_verdict,
        "path_metric_verdict": verdict,
        "verdict_scope": (
            "top-level and matched-rate verdicts require all practical-match "
            "gates; path_metric_verdict always reports the unchanged path-only rule"
        ),
        "matched_rate_claim_verdict": matched_rate_claim_verdict,
        "verdict_gates": verdict_gates,
        "verdict_rule": (
            "PASS iff the five-GOP equal-weight AP mean improves raw penalized "
            "and own-bbox-normalized penalized MTE for both global and top10, "
            "the equal-GOP global missing own-reference action fraction is not "
            "worse, and normalized penalized MTE improves in at least 3/5 GOPs "
            "for both global and top10; otherwise MIXED"
        ),
        "claim_scope": (
            "one-scene-development-paired-own-reference; preserve exact rate "
            "stratum and do not relabel an unmatched result as matched-rate"
        ),
        "raw_cross_method_metric_warning": (
            "raw MTE is scale-sensitive across independently trained own-reference "
            "lineages; it is retained as an additional conservative sensitivity "
            "gate alongside bbox-normalized MTE and is never the sole verdict metric"
        ),
        "primary_aggregation_rule": "five-GOP-equal-weight",
        "identity_weighted_aggregation_role": "descriptive-only-no-verdict-effect",
        "lineage_correction": {
            "incorrect_predecessor": (
                "201_h007_gifstream_f01_final_decode_path_evaluation_collision_safe.py"
            ),
            "incorrect_predecessor_behavior": (
                "used official pre-codec reference for both independent models"
            ),
            "corrected_behavior": (
                "each decode uses the reference tree from its own producer TAR"
            ),
            "collision_policy_changed": False,
            "path_outcomes_used_to_select_references": False,
        },
        "official_runtime_provenance_manifest_sha256": (
            args.official_provenance_manifest_sha256
        ),
        "ap_runtime_provenance_manifest_sha256": (
            args.ap_provenance_manifest_sha256
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
