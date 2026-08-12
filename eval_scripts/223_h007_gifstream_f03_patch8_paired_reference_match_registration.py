#!/usr/bin/env python3
"""Build the f03 Patch8 outcome-blind paired-reference match registration.

Only producer identity, complete-container bytes, ordinary image metrics, and
the two pre-codec reference lineages are read.  No decoded-path artifact or
path outcome is accepted as input.  The resulting O_EXCL canonical JSON freezes
both producer TARs, the fresh producer registration, ordinary rate/quality, and
both reference lineages before evaluator 224 decodes any path.  The practical
rate/quality gates only label a comparison stratum and claim eligibility; they
never authorize or suppress path execution.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping


SCHEMA = "h007.ap_gifstream.f03_patch8_paired_reference_match.v1"
PRODUCER_REGISTRATION_SCHEMA = (
    "h007.ap_gifstream.f03_patch8_producer_registration.v1"
)
IMPLEMENTATION_ISOLATION_SCHEMA = (
    "h007.ap_gifstream.f03_patch8_implementation_isolation.v1"
)
PATH_EXECUTION_POLICY = "always-evaluate-valid-pair-regardless-of-practical-match"
SCENE = "flame_salmon_1"
GOP_COUNT = 5
FRAME_COUNT = 300
OFFICIAL_TAR_BYTES = 540_590_080
OFFICIAL_TAR_SHA256 = (
    "9a18296fd04b0b75810805099fa4f3affc362a7d12bf3c798b587859d62f303f"
)
OFFICIAL_SEQUENCE_SHA256 = (
    "f52254d7e3346e72f461b32d9d0ef867ca53088b0f0fabcf6463eb26486ed3ab"
)
OFFICIAL_REFERENCE_TREE_SHA256 = (
    "b4cee17d818ca0850e88af7ed440313b48ae6ad51684147e4bf009bff294a671"
)
OFFICIAL_PROVENANCE_SHA256 = (
    "0ec4848addaa53a1552ac1317e025c179d247b056ba30f62729a04b48b0e1425"
)
AP_PROVENANCE_SHA256 = (
    "71e10c5e5226e68743812151b1da0e2f4009a7056454dda6fac95cb8067bb264"
)
AP_NORMALIZED_TREE_SHA256 = (
    "0f94bc919c616f30bdf8b32d1e28f8a80668bae04178aa10a0ee182def40b64f"
)
AP_TASK_ID = "DEV.flame_salmon_1.ap.f03-patch8-ownref-rep1.p0c32"
AP_TAR_PREFIX = "h007_p0c32_dev_ap_f03_patch8_ownref_rep1_"
AP_TAR_SUFFIX = "_4.tar"
PATCH5_SHA256 = (
    "f2a38d5fb3e1cabc5eb33fa147de535bff96ea775db919bee40b7fedea0bca59"
)
PATCH6_SHA256 = (
    "1a74c0e9c7bb208113e6bfa04a599fd8ebbafed653996d44145ee497d06619cb"
)
PATCH7_SHA256 = (
    "96503defe9a91546077ce7683cfe29648d23da86efb8adab1305cffc38a32ad4"
)
PATCH8_SHA256 = (
    "1d9d2c96e773f2e77e5bfdf15e87912402c1de207d0c0ac78a52526b93141bbb"
)
THRESHOLDS = {
    "absolute_relative_complete_rate_max": 0.015,
    "psnr_ap_minus_official_min_db": -0.1,
    "ssim_ap_minus_official_min": -0.002,
    "lpips_ap_minus_official_max": 0.005,
}
GATE_ABSOLUTE_TOLERANCE = 1e-12
SCIENTIFIC_CONFIG = {
    "scene": SCENE,
    "scene_config": "neur3d_1",
    "method": "ap-gifstream-full",
    "rate": 5,
    "rd_lambda": 0.00095,
    "gop_count": GOP_COUNT,
    "gop_size": 60,
    "frame_count": FRAME_COUNT,
    "max_steps": 30000,
    "freeze_step": 29000,
    "protected_fraction": 0.01,
    "q_ap_multiplier": 0.5,
    "q_bg_multiplier": 1.3,
    "path_loss_lambda": 0.0001,
    "path_loss_every": 50,
    "random_seed": 42,
    "ap_random_seed": 11,
    "compression_seed": 20260715,
    "data_factor": 2,
    "batch_size": 1,
    "knn": True,
    "n_knn": 6,
    "compression_scaling": {
        "anchors": None,
        "scales": 0.038,
        "quats": None,
        "opacities": None,
        "anchor_features": 1.0,
        "offsets": 0.038,
        "factors": 0.0625,
        "time_features": 1.0,
    },
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def strict_json(payload: bytes, label: str) -> Dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON constant {value}")

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
        value: Dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate JSON key {key}")
            value[key] = item
        return value

    value = json.loads(
        payload.decode("utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_pairs,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _require_exact_int(value: Any, expected: int, label: str) -> int:
    if type(value) is not int or value != expected:
        raise ValueError(f"{label} differs from {expected}")
    return value


def _require_exact_float(value: Any, expected: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) != expected
    ):
        raise ValueError(f"{label} differs from {expected}")
    return float(value)


def _ap_array_job_id_from_tar_name(archive: Path) -> str:
    name = archive.name
    if not name.startswith(AP_TAR_PREFIX) or not name.endswith(AP_TAR_SUFFIX):
        raise ValueError("AP producer TAR basename is outside the frozen contract")
    value = name[len(AP_TAR_PREFIX) : -len(AP_TAR_SUFFIX)]
    if not value or not value.isascii() or not value.isdigit():
        raise ValueError("AP producer array job ID cannot be derived from TAR basename")
    return value


def _safe_tar_members(archive: Path) -> Dict[str, tarfile.TarInfo]:
    members: Dict[str, tarfile.TarInfo] = {}
    with tarfile.open(archive, "r:") as handle:
        for member in handle.getmembers():
            name = member.name
            path = PurePosixPath(name)
            if (
                not name
                or name in members
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in name
                or member.issym()
                or member.islnk()
                or member.isdev()
                or member.isfifo()
            ):
                raise ValueError(f"unsafe producer TAR member: {name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"unsupported producer TAR member: {name}")
            members[name] = member
    return members


def _read_tar_file(
    archive: Path, members: Mapping[str, tarfile.TarInfo], name: str
) -> bytes:
    member = members.get(name)
    if member is None or not member.isfile():
        raise ValueError(f"producer TAR lacks regular file {name}")
    with tarfile.open(archive, "r:") as handle:
        stream = handle.extractfile(member)
        if stream is None:
            raise ValueError(f"producer TAR member is unreadable: {name}")
        payload = stream.read()
    if len(payload) != member.size:
        raise ValueError(f"producer TAR member size changed: {name}")
    return payload


def _reference_tree_binding(
    archive: Path, members: Mapping[str, tarfile.TarInfo]
) -> Dict[str, Any]:
    rows = []
    with tarfile.open(archive, "r:") as handle:
        for name, member in sorted(members.items()):
            if not member.isfile() or not name.startswith("references/"):
                continue
            relative = name[len("references/") :]
            if not relative:
                raise ValueError("empty reference-tree member name")
            stream = handle.extractfile(member)
            if stream is None:
                raise ValueError(f"reference-tree member is unreadable: {name}")
            rows.append((relative, stream.read()))
    digest = hashlib.sha256()
    for name, payload in rows:
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return {"file_count": len(rows), "tree_sha256": digest.hexdigest()}


def _sequence_binding(payload: bytes, expected_method: str) -> Dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("sequence ZIP has duplicate names")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError(f"unsafe sequence ZIP member: {name}")
        rows = [
            info
            for info in infos
            if info.filename == "sequence_manifest.json" and not info.is_dir()
        ]
        if len(rows) != 1:
            raise ValueError("sequence ZIP lacks one exact manifest")
        manifest_payload = archive.read(rows[0])
        manifest = strict_json(manifest_payload, "sequence manifest")
        if (
            manifest.get("scene") != SCENE
            or manifest.get("method") != expected_method
            or not isinstance(manifest.get("gops"), list)
            or len(manifest["gops"]) != GOP_COUNT
        ):
            raise ValueError("sequence identity or extent changed")
        gop_bindings = []
        for gop_id in range(GOP_COUNT):
            member_name = f"gops/gop_{gop_id}.zip"
            member_rows = [
                info
                for info in infos
                if info.filename == member_name and not info.is_dir()
            ]
            if len(member_rows) != 1:
                raise ValueError(f"sequence lacks exact GOP {gop_id}")
            inner = archive.read(member_rows[0])
            row = manifest["gops"][gop_id]
            if (
                row.get("gop_id") != gop_id
                or row.get("path") != member_name
                or int(row.get("bytes", -1)) != len(inner)
                or row.get("sha256") != sha256_bytes(inner)
            ):
                raise ValueError(f"sequence GOP {gop_id} binding changed")
            gop_bindings.append(
                {
                    "gop_id": gop_id,
                    "bytes": len(inner),
                    "sha256": sha256_bytes(inner),
                }
            )
    return {
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "manifest_sha256": sha256_bytes(manifest_payload),
        "gops": gop_bindings,
    }


def _ordinary_metrics(
    payload: bytes,
    expected_method: str,
    sequence: Mapping[str, Any],
) -> Dict[str, Any]:
    receipt = strict_json(payload, f"{expected_method} ordinary receipt")
    if (
        receipt.get("schema") != "h007.ordinary_rate_quality_evaluator_receipt.v3"
        or receipt.get("scene") != SCENE
        or receipt.get("method") != expected_method
        or receipt.get("metric_protocol")
        != "ordinary_unedited_psnr_ssim_lpips_300frames.v3"
        or int(receipt.get("archive_bytes", -1)) != sequence["bytes"]
        or receipt.get("archive_sha256") != sequence["sha256"]
        or receipt.get("outcome_fields_read")
        != ["ordinary_unedited_fidelity", "real_container_accounting"]
    ):
        raise ValueError(f"{expected_method} ordinary receipt contract changed")
    frames = receipt.get("frame_metrics")
    if not isinstance(frames, list) or len(frames) != FRAME_COUNT:
        raise ValueError(f"{expected_method} ordinary receipt lacks 300 frames")
    if [row.get("frame") for row in frames] != list(range(FRAME_COUNT)):
        raise ValueError(f"{expected_method} ordinary frame order changed")
    metric_values: Dict[str, list[float]] = {"psnr": [], "ssim": [], "lpips": []}
    for row in frames:
        if not isinstance(row, dict):
            raise ValueError("ordinary frame row is not an object")
        for metric in metric_values:
            value = row.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"ordinary {metric} is not finite")
            metric_values[metric].append(float(value))
    return {
        "receipt_sha256": sha256_bytes(payload),
        "receipt_bytes": len(payload),
        "psnr": sum(metric_values["psnr"]) / FRAME_COUNT,
        "ssim": sum(metric_values["ssim"]) / FRAME_COUNT,
        "lpips": sum(metric_values["lpips"]) / FRAME_COUNT,
        "runtime_provenance_manifest_sha256": receipt[
            "runtime_provenance_manifest_sha256"
        ],
        "source_data_manifest_sha256": receipt["source_data_manifest_sha256"],
        "point_id": receipt["point_id"],
    }


def _require_ap_compression_archive_binding(
    manifest: Mapping[str, Any], expected_sha256: str, gop_id: int
) -> str:
    observed = _require_sha256(
        manifest.get("compression_archive_sha256"),
        f"AP reference gop {gop_id} compression archive SHA-256",
    )
    if observed != expected_sha256:
        raise ValueError(
            f"AP reference gop {gop_id} is bound to a different codec archive"
        )
    return observed


def _validate_ap_reference_files(
    archive: Path,
    members: Mapping[str, tarfile.TarInfo],
    gop_id: int,
    manifest: Mapping[str, Any],
    sequence_gop: Mapping[str, Any],
    receipt_sha256: str,
) -> Dict[str, Any]:
    base = f"references/{SCENE}/gop_{gop_id}/"
    expected_camera = {
        "camera_metadata/bounds.npy",
        "camera_metadata/camera_ids.npy",
        "camera_metadata/camera_keys.npy",
        "camera_metadata/camera_names.npy",
        "camera_metadata/camtoworlds.npy",
        "camera_metadata/image_sizes.npy",
        "camera_metadata/intrinsics.npy",
        "camera_metadata/transform.npy",
    }
    expected_files = expected_camera | {
        "byte_census.json",
        "decoder_config.json",
        "nets.pt",
        "reference_bundle_manifest.json",
        "reference_splats.pt",
    }
    observed_files = {
        name[len(base) :]
        for name, member in members.items()
        if name.startswith(base) and member.isfile()
    }
    if observed_files != expected_files:
        raise ValueError(f"AP reference gop {gop_id} file set changed")
    payloads = {
        relative: _read_tar_file(archive, members, base + relative)
        for relative in sorted(expected_files)
    }
    required_hashes = {
        "decoder_config.json": "decoder_config_sha256",
        "nets.pt": "reference_nets_sha256",
        "reference_splats.pt": "reference_splats_sha256",
    }
    for relative, field in required_hashes.items():
        if _require_sha256(
            manifest.get(field), f"AP reference gop {gop_id} {field}"
        ) != sha256_bytes(payloads[relative]):
            raise ValueError(
                f"AP reference gop {gop_id} {relative} differs from manifest"
            )
    _require_ap_compression_archive_binding(
        manifest, sequence_gop["sha256"], gop_id
    )
    census = strict_json(
        payloads["byte_census.json"], f"AP reference gop {gop_id} byte census"
    )
    census_relatives = sorted(expected_files - {"byte_census.json"})
    expected_rows = [
        {
            "path": relative,
            "bytes": len(payloads[relative]),
            "sha256": sha256_bytes(payloads[relative]),
        }
        for relative in census_relatives
    ]
    if (
        census.get("schema") != "h007.container_byte_census.v1"
        or census.get("root") != f"gop_{gop_id}"
        or census.get("file_count") != len(expected_rows)
        or census.get("raw_bytes") != sum(row["bytes"] for row in expected_rows)
        or census.get("files") != expected_rows
    ):
        raise ValueError(f"AP reference gop {gop_id} byte census changed")
    summary_payload = _read_tar_file(
        archive, members, f"reference_export_summaries/gop_{gop_id}.json"
    )
    summary = strict_json(
        summary_payload, f"AP reference export summary gop {gop_id}"
    )
    if (
        summary.get("schema") != "h007.ap_same_run_reference_export.v1"
        or summary.get("status") != "COMPLETED"
        or summary.get("scene") != SCENE
        or summary.get("gop_id") != gop_id
        or summary.get("source_checkpoint_sha256")
        != manifest["source_checkpoint_sha256"]
        or summary.get("source_training_receipt_sha256") != receipt_sha256
        or summary.get("reference_bundle_manifest_sha256")
        != sha256_bytes(payloads["reference_bundle_manifest.json"])
        or summary.get("byte_census_sha256")
        != sha256_bytes(payloads["byte_census.json"])
        or summary.get("candidate_inputs_read") != []
        or summary.get("outcome_fields_read") != []
        or summary.get("path_outcomes_read") != []
    ):
        raise ValueError(f"AP reference export summary gop {gop_id} changed")
    return {
        "byte_census_sha256": sha256_bytes(payloads["byte_census.json"]),
        "reference_export_summary_sha256": sha256_bytes(summary_payload),
        "compression_archive_sha256": sequence_gop["sha256"],
    }


def _validate_reference_manifests(
    archive: Path,
    members: Mapping[str, tarfile.TarInfo],
    label: str,
    sequence: Mapping[str, Any],
) -> Dict[str, Any]:
    manifest_hashes = []
    receipt_hashes = []
    checkpoint_hashes = []
    byte_census_hashes = []
    export_summary_hashes = []
    compression_archive_hashes = []
    for gop_id in range(GOP_COUNT):
        manifest_name = (
            f"references/{SCENE}/gop_{gop_id}/reference_bundle_manifest.json"
        )
        manifest_payload = _read_tar_file(archive, members, manifest_name)
        manifest = strict_json(manifest_payload, f"{label} reference gop {gop_id}")
        common = {
            "schema": "h007.hdown_reference_bundle.v1",
            "scene": SCENE,
            "gop_id": gop_id,
            "start_frame": 60 * gop_id,
            "frame_count": 60,
        }
        for key, expected in common.items():
            if manifest.get(key) != expected:
                raise ValueError(f"{label} reference gop {gop_id} {key} changed")
        for field in (
            "candidate_inputs_read",
            "outcome_fields_read",
            "path_outcomes_read",
        ):
            if manifest.get(field, []) != []:
                raise ValueError(
                    f"{label} reference gop {gop_id} reads {field}"
                )
        if label == "official":
            if manifest.get("variant") != "official_uncompressed_reference":
                raise ValueError("official reference variant changed")
        else:
            required = {
                "variant": "ap-gifstream-full_pre_codec_reference",
                "reference_bundle_semantics": (
                    "same-run-pre-codec-ap-checkpoint"
                ),
                "reference_method": "ap-gifstream-full",
                "ap_codec_schema": "h007.ap_gifstream.codec.v6",
                "ap_training_state_schema": "h007.ap_training_state.v2",
                "ap_training_receipt_schema": "h007.ap_training_receipt.v2",
            }
            for key, expected in required.items():
                if manifest.get(key) != expected:
                    raise ValueError(
                        f"AP reference gop {gop_id} {key} changed"
                    )
            path_contract = manifest.get("path_contract")
            if (
                not isinstance(path_contract, dict)
                or path_contract.get("schema")
                != "h007.ap_gifstream.path_contract.v1"
                or path_contract.get("knn_count") != 6
                or path_contract.get("knn_rule")
                != (
                    "retained-canonical-radius-complete-distance+"
                    "lexicographic-id"
                )
                or path_contract.get("dependency_rule")
                != "protected-plus-one-hop-retained-knn"
                or path_contract.get("canonical_anchor_reconstruction")
                is not True
                or float(path_contract.get("factor_protected_multiplier", -1))
                != 1.0 / 256.0
                or float(path_contract.get("factor_background_multiplier", -1))
                != 1.0 / 64.0
                or float(
                    path_contract.get(
                        "anchor_feature_protected_multiplier", -1
                    )
                )
                != 0.25
                or float(
                    path_contract.get(
                        "anchor_feature_background_multiplier", -1
                    )
                )
                != 1.0
            ):
                raise ValueError(
                    f"AP reference gop {gop_id} Patch8 path contract changed"
                )
            _require_sha256(
                path_contract.get("retained_knn_graph_sha256"),
                f"AP reference gop {gop_id} retained-KNN graph SHA-256",
            )
            runtime_provenance = manifest.get("runtime_provenance")
            patch_chain = (
                runtime_provenance.get("patch_sha256")
                if isinstance(runtime_provenance, dict)
                else None
            )
            normalized_tree = (
                runtime_provenance.get("normalized_code_tree")
                if isinstance(runtime_provenance, dict)
                else None
            )
            if (
                not isinstance(patch_chain, list)
                or len(patch_chain) != 9
                or patch_chain[-1] != PATCH8_SHA256
                or not isinstance(normalized_tree, dict)
                or normalized_tree.get("sha256")
                != AP_NORMALIZED_TREE_SHA256
            ):
                raise ValueError(
                    f"AP reference gop {gop_id} is not Patch8-provenanced"
                )
            _require_sha256(
                manifest.get("ap_training_receipt_sha256"),
                f"AP reference gop {gop_id} AP training receipt SHA-256",
            )
            receipt_name = f"training_receipts/gop_{gop_id}.json"
            receipt_payload = _read_tar_file(archive, members, receipt_name)
            receipt = strict_json(receipt_payload, f"AP training receipt gop {gop_id}")
            checkpoints = receipt.get("source_checkpoints")
            if (
                not isinstance(checkpoints, list)
                or len(checkpoints) != 1
                or not isinstance(checkpoints[0], dict)
            ):
                raise ValueError("AP training receipt checkpoint binding is malformed")
            checkpoint = checkpoints[0]
            if (
                manifest.get("source_checkpoint_sha256")
                != checkpoint.get("sha256")
                or int(manifest.get("source_checkpoint_bytes", -1))
                != int(checkpoint.get("bytes", -2))
                or manifest.get("source_training_receipt_sha256")
                != sha256_bytes(receipt_payload)
            ):
                raise ValueError(
                    f"AP reference gop {gop_id} is not bound to its training receipt"
                )
            for value, field in (
                (manifest.get("source_checkpoint_sha256"), "checkpoint SHA-256"),
                (
                    manifest.get("source_training_receipt_sha256"),
                    "training receipt SHA-256",
                ),
                (manifest.get("producer_receipt_sha256"), "producer receipt SHA-256"),
            ):
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)
                ):
                    raise ValueError(
                        f"AP reference gop {gop_id} invalid {field}"
                    )
            receipt_hashes.append(sha256_bytes(receipt_payload))
            checkpoint_hashes.append(manifest["source_checkpoint_sha256"])
            files = _validate_ap_reference_files(
                archive,
                members,
                gop_id,
                manifest,
                sequence["gops"][gop_id],
                sha256_bytes(receipt_payload),
            )
            byte_census_hashes.append(files["byte_census_sha256"])
            export_summary_hashes.append(
                files["reference_export_summary_sha256"]
            )
            compression_archive_hashes.append(
                files["compression_archive_sha256"]
            )
        manifest_hashes.append(sha256_bytes(manifest_payload))
    return {
        "reference_manifest_sha256": manifest_hashes,
        "training_receipt_sha256": receipt_hashes,
        "source_checkpoint_sha256": checkpoint_hashes,
        "byte_census_sha256": byte_census_hashes,
        "reference_export_summary_sha256": export_summary_hashes,
        "compression_archive_sha256": compression_archive_hashes,
    }


def _validate_ap_producer_registration(
    payload: bytes,
    completion: Mapping[str, Any],
) -> Dict[str, Any]:
    registration = strict_json(payload, "AP f03 Patch8 producer registration")
    if (
        registration.get("schema") != PRODUCER_REGISTRATION_SCHEMA
        or registration.get("status") != "FROZEN_BEFORE_F03_EXECUTION"
        or registration.get("task_id") != AP_TASK_ID
        or registration.get("path_outcomes_read") != []
        or registration.get("scientific_configuration") != SCIENTIFIC_CONFIG
        or completion.get("producer_registration_sha256")
        != sha256_bytes(payload)
    ):
        raise ValueError("AP f03 Patch8 producer registration changed")
    contract = registration.get("patch8_contract")
    if (
        not isinstance(contract, dict)
        or not isinstance(contract.get("manifest"), dict)
        or contract["manifest"].get("sha256") != AP_PROVENANCE_SHA256
        or not isinstance(contract.get("patch"), dict)
        or contract["patch"].get("sha256") != PATCH8_SHA256
        or contract.get("normalized_code_tree_sha256")
        != AP_NORMALIZED_TREE_SHA256
        or contract.get("runtime_patch_count") != 9
        or contract.get("ap_codec_schema") != "h007.ap_gifstream.codec.v6"
        or contract.get("ap_training_state_schema")
        != "h007.ap_training_state.v2"
        or contract.get("ap_training_receipt_schema")
        != "h007.ap_training_receipt.v2"
        or contract.get("path_contract_schema")
        != "h007.ap_gifstream.path_contract.v1"
    ):
        raise ValueError("AP f03 Patch8 producer registration contract changed")
    exporter = registration.get("reference_exporter")
    if (
        not isinstance(exporter, dict)
        or exporter.get("path") != "../scripts/h007_export_ap_same_run_reference_f03.py"
        or not isinstance(exporter.get("bytes"), int)
        or exporter["bytes"] <= 0
    ):
        raise ValueError("AP f03 reference exporter registration changed")
    _require_sha256(exporter.get("sha256"), "AP f03 reference exporter SHA-256")
    return {
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "schema": PRODUCER_REGISTRATION_SCHEMA,
        "status": registration["status"],
        "scientific_configuration": registration["scientific_configuration"],
        "patch8_contract": contract,
        "reference_exporter": exporter,
        "path_outcomes_read": [],
    }


def _validate_ap_implementation_isolation_report(
    archive: Path,
    members: Mapping[str, tarfile.TarInfo],
    completion: Mapping[str, Any],
    sequence: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = _read_tar_file(
        archive, members, "implementation_isolation_report.json"
    )
    report = strict_json(payload, "AP f03 Patch8 implementation isolation report")
    if (
        report.get("schema") != IMPLEMENTATION_ISOLATION_SCHEMA
        or report.get("status") != "FRESH_FIVE_GOP_PATCH8"
        or report.get("task_id") != AP_TASK_ID
        or report.get("scene") != SCENE
        or report.get("sequence_bytes") != sequence["bytes"]
        or report.get("sequence_sha256") != sequence["sha256"]
        or report.get("candidate_inputs_read") != []
        or report.get("outcome_fields_read") != []
        or report.get("path_outcomes_read") != []
        or report.get("producer_registration_sha256")
        != completion.get("producer_registration_sha256")
        or report.get("patch8_sha256") != PATCH8_SHA256
        or report.get("patch8_manifest_sha256") != AP_PROVENANCE_SHA256
        or report.get("normalized_code_tree_sha256")
        != AP_NORMALIZED_TREE_SHA256
        or report.get("runtime_patch_count") != 9
        or report.get("scientific_parameters_identical_to_f02") is not True
        or report.get("f02_checkpoint_codec_sequence_or_receipt_inputs") != []
        or completion.get("implementation_isolation_report_sha256")
        != sha256_bytes(payload)
    ):
        raise ValueError("AP f03 Patch8 implementation isolation report changed")
    forbidden_historical_fields = {
        "sequence_exact_match",
        "all_checkpoints_exact_match",
        "all_compression_archives_exact_match",
        "checkpoint_exact_match",
        "compression_archive_exact_match",
        "historical_sequence_sha256",
        "historical_checkpoint_sha256",
        "historical_compression_archive_sha256",
    }
    if forbidden_historical_fields.intersection(report):
        raise ValueError("implementation isolation report contains historical match")
    rows = report.get("gops")
    if not isinstance(rows, list) or len(rows) != GOP_COUNT:
        raise ValueError("implementation isolation report lacks five GOP rows")
    bindings = []
    for gop_id, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or row.get("gop_id") != gop_id
            or row.get("fresh_training") is not True
            or forbidden_historical_fields.intersection(row)
            or _require_sha256(
                row.get("compression_archive_sha256"),
                f"AP implementation-isolation gop {gop_id} archive SHA-256",
            )
            != sequence["gops"][gop_id]["sha256"]
        ):
            raise ValueError(
                f"AP implementation-isolation gop {gop_id} binding changed"
            )
        bindings.append(
            {
                "gop_id": gop_id,
                "checkpoint_sha256": _require_sha256(
                    row.get("checkpoint_sha256"),
                    f"AP implementation-isolation gop {gop_id} checkpoint SHA-256",
                ),
                "compression_archive_sha256": row["compression_archive_sha256"],
            }
        )
    return {
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "schema": IMPLEMENTATION_ISOLATION_SCHEMA,
        "role": "fresh-patch8-implementation-isolation-no-historical-output-match",
        "sequence_bytes": sequence["bytes"],
        "sequence_sha256": sequence["sha256"],
        "gops": bindings,
        "candidate_inputs_read": [],
        "outcome_fields_read": [],
        "path_outcomes_read": [],
    }


def _producer_row(
    archive: Path,
    label: str,
    expected_method: str,
    ap_producer_registration: Path | None = None,
    expected_ap_array_job_id: str | None = None,
) -> Dict[str, Any]:
    members = _safe_tar_members(archive)
    completion_payload = _read_tar_file(
        archive, members, "completion_marker.json"
    )
    sequence_payload = _read_tar_file(archive, members, "sequence.zip")
    ordinary_payload = _read_tar_file(archive, members, "ordinary_receipt.json")
    completion = strict_json(completion_payload, f"{label} completion marker")
    if (
        completion.get("schema")
        not in {
            "h007.gifstream.u3-node-local-producer-completion.v2",
            "h007.ap_gifstream.f03_patch8_producer_completion.v1",
        }
        or completion.get("final_status") != "COMPLETED"
        or int(completion.get("frame_count", -1)) != FRAME_COUNT
        or int(completion.get("gop_count", -1)) != GOP_COUNT
        or completion.get("path_outcomes_read", []) != []
    ):
        raise ValueError(f"{label} producer completion contract changed")
    sequence = _sequence_binding(sequence_payload, expected_method)
    if completion.get("sequence_sha256") != sequence["sha256"]:
        raise ValueError(f"{label} completion does not bind sequence")
    ordinary = _ordinary_metrics(ordinary_payload, expected_method, sequence)
    if completion.get("ordinary_receipt_sha256") != ordinary["receipt_sha256"]:
        raise ValueError(f"{label} completion does not bind ordinary receipt")
    reference = _reference_tree_binding(archive, members)
    if (
        int(completion.get("reference_bundle_file_count", -1))
        != reference["file_count"]
        or completion.get("reference_bundle_tree_sha256")
        != reference["tree_sha256"]
    ):
        raise ValueError(f"{label} completion does not bind reference tree")
    lineage = _validate_reference_manifests(
        archive, members, label, sequence
    )
    implementation_isolation = None
    producer_registration = None
    if label == "official":
        if (
            archive.stat().st_size != OFFICIAL_TAR_BYTES
            or sha256_file(archive) != OFFICIAL_TAR_SHA256
            or sequence["sha256"] != OFFICIAL_SEQUENCE_SHA256
            or reference["tree_sha256"] != OFFICIAL_REFERENCE_TREE_SHA256
            or ordinary["runtime_provenance_manifest_sha256"]
            != OFFICIAL_PROVENANCE_SHA256
        ):
            raise ValueError("official frozen artifact binding changed")
        reference_semantics = "same-run-pre-codec-official-checkpoint"
        reference_method = "official"
    else:
        array_job_id = _ap_array_job_id_from_tar_name(archive)
        if (
            expected_ap_array_job_id is None
            or array_job_id != expected_ap_array_job_id
        ):
            raise ValueError("AP producer array job ID differs from requested job")
        if ap_producer_registration is None:
            raise ValueError("AP producer registration is required")
        producer_registration_payload = ap_producer_registration.read_bytes()
        packaged_registration_payload = _read_tar_file(
            archive,
            members,
            "h007_ap_gifstream_f03_patch8_producer_registration_20260726.json",
        )
        if packaged_registration_payload != producer_registration_payload:
            raise ValueError(
                "packaged AP producer registration differs from frozen input"
            )
        producer_registration = _validate_ap_producer_registration(
            producer_registration_payload, completion
        )
        if (
            completion.get("schema")
            != "h007.ap_gifstream.f03_patch8_producer_completion.v1"
            or
            completion.get("reference_bundle_semantics")
            != "same-run-pre-codec-ap-checkpoint"
            or completion.get("reference_method") != "ap-gifstream-full"
            or ordinary["runtime_provenance_manifest_sha256"]
            != AP_PROVENANCE_SHA256
            or completion.get("task_id") != AP_TASK_ID
            or completion.get("slurm_array_job_id") != array_job_id
            or completion.get("runtime_manifest_sha256")
            != AP_PROVENANCE_SHA256
            or completion.get("patch5_p0closure32_sha256")
            != PATCH5_SHA256
            or completion.get("patch6_native_fillin_f01_sha256")
            != PATCH6_SHA256
            or completion.get("patch7_native_fillin_f02_sha256")
            != PATCH7_SHA256
            or completion.get("patch8_path_state_contract_sha256")
            != PATCH8_SHA256
            or completion.get("normalized_code_tree_sha256")
            != AP_NORMALIZED_TREE_SHA256
            or completion.get("runtime_patch_count") != 9
            or completion.get("rate_grid") != "f02-scientific-point-on-patch8"
            or completion.get("parallel_execution_registration_sha256")
            != producer_registration["sha256"]
            or completion.get("reference_exporter_sha256")
            != producer_registration["reference_exporter"]["sha256"]
            or completion.get("complete_evaluation_required") is not True
            or completion.get("rate_use") != "reporting-stratum-only"
            or completion.get("rate_may_truncate_path_evaluation") is not False
            or completion.get("candidate_inputs_read") != []
            or completion.get("outcome_fields_read") != []
            or ordinary["point_id"] != AP_TASK_ID
        ):
            raise ValueError("AP own-reference completion provenance changed")
        _require_exact_int(
            completion.get("slurm_array_task_id"), 4, "AP array task"
        )
        _require_exact_int(completion.get("rate_index"), 5, "AP rate index")
        _require_exact_float(completion.get("rd_lambda"), 0.00095, "AP RD lambda")
        _require_exact_float(
            completion.get("ap_q_ap_multiplier"), 0.5, "AP q-ap multiplier"
        )
        _require_exact_float(
            completion.get("ap_q_bg_multiplier"), 1.3, "AP q-bg multiplier"
        )
        _require_exact_float(
            completion.get("ap_path_loss_lambda"), 0.0001, "AP path-loss lambda"
        )
        _require_exact_int(
            completion.get("ap_path_loss_every"), 50, "AP path-loss interval"
        )
        reference_semantics = completion["reference_bundle_semantics"]
        reference_method = completion["reference_method"]
        implementation_isolation = _validate_ap_implementation_isolation_report(
            archive, members, completion, sequence
        )
        if [
            row["checkpoint_sha256"]
            for row in implementation_isolation["gops"]
        ] != lineage["source_checkpoint_sha256"]:
            raise ValueError(
                "implementation isolation checkpoints differ from own-reference lineage"
            )
    return {
        "method": expected_method,
        "producer_tar": str(archive.resolve()),
        "producer_tar_bytes": archive.stat().st_size,
        "producer_tar_sha256": sha256_file(archive),
        "completion_marker_bytes": len(completion_payload),
        "completion_marker_sha256": sha256_bytes(completion_payload),
        "ordinary_receipt_bytes": len(ordinary_payload),
        "ordinary_receipt_sha256": ordinary["receipt_sha256"],
        "sequence_bytes": sequence["bytes"],
        "sequence_sha256": sequence["sha256"],
        "sequence_manifest_sha256": sequence["manifest_sha256"],
        "sequence_gops": sequence["gops"],
        "psnr": ordinary["psnr"],
        "ssim": ordinary["ssim"],
        "lpips": ordinary["lpips"],
        "runtime_provenance_manifest_sha256": ordinary[
            "runtime_provenance_manifest_sha256"
        ],
        "source_data_manifest_sha256": ordinary["source_data_manifest_sha256"],
        "point_id": ordinary["point_id"],
        "ap_producer_array_job_id": (
            expected_ap_array_job_id if label == "ap" else None
        ),
        "producer_registration": producer_registration,
        "implementation_isolation_report": implementation_isolation,
        "reference_bundle_binding": {
            **reference,
            **lineage,
            "reference_source_label": label,
            "reference_method": reference_method,
            "reference_bundle_semantics": reference_semantics,
        },
    }


def _practical_match(
    official: Mapping[str, Any], ap: Mapping[str, Any]
) -> Dict[str, Any]:
    def at_most(value: float, limit: float) -> bool:
        return value <= limit or math.isclose(
            value, limit, rel_tol=0.0, abs_tol=GATE_ABSOLUTE_TOLERANCE
        )

    def at_least(value: float, limit: float) -> bool:
        return value >= limit or math.isclose(
            value, limit, rel_tol=0.0, abs_tol=GATE_ABSOLUTE_TOLERANCE
        )

    relative_rate = (
        float(ap["sequence_bytes"]) - float(official["sequence_bytes"])
    ) / float(official["sequence_bytes"])
    deltas = {
        "complete_sequence_bytes": (
            int(ap["sequence_bytes"]) - int(official["sequence_bytes"])
        ),
        "relative_complete_rate": relative_rate,
        "psnr_db": float(ap["psnr"]) - float(official["psnr"]),
        "ssim": float(ap["ssim"]) - float(official["ssim"]),
        "lpips": float(ap["lpips"]) - float(official["lpips"]),
    }
    gates = {
        "complete_rate": at_most(
            abs(deltas["relative_complete_rate"]),
            THRESHOLDS["absolute_relative_complete_rate_max"],
        ),
        "psnr": at_least(
            deltas["psnr_db"],
            THRESHOLDS["psnr_ap_minus_official_min_db"],
        ),
        "ssim": at_least(
            deltas["ssim"],
            THRESHOLDS["ssim_ap_minus_official_min"],
        ),
        "lpips": at_most(
            deltas["lpips"],
            THRESHOLDS["lpips_ap_minus_official_max"],
        ),
    }
    return {
        "thresholds": dict(THRESHOLDS),
        "gate_comparison_policy": {
            "inclusive": True,
            "absolute_floating_tolerance": GATE_ABSOLUTE_TOLERANCE,
            "relative_floating_tolerance": 0.0,
        },
        "gates": gates,
        "eligible": all(gates.values()),
        "deltas": deltas,
        "rate_disclosure": f"{relative_rate:+.6%}",
        "exact_disclosure": {
            "official_complete_sequence_bytes": int(official["sequence_bytes"]),
            "ap_complete_sequence_bytes": int(ap["sequence_bytes"]),
            "ap_minus_official_complete_sequence_bytes": deltas[
                "complete_sequence_bytes"
            ],
            "ap_minus_official_relative_complete_rate": relative_rate,
            "ap_minus_official_psnr_db": deltas["psnr_db"],
            "ap_minus_official_ssim": deltas["ssim"],
            "ap_minus_official_lpips": deltas["lpips"],
        },
    }


def build_registration(
    official_tar: Path,
    ap_tar: Path,
    ap_producer_registration: Path,
    expected_ap_array_job_id: str,
) -> Dict[str, Any]:
    official = _producer_row(official_tar.resolve(), "official", "official")
    ap = _producer_row(
        ap_tar.resolve(),
        "ap",
        "ap-gifstream-full",
        ap_producer_registration.resolve(),
        expected_ap_array_job_id,
    )
    if (
        official["source_data_manifest_sha256"]
        != ap["source_data_manifest_sha256"]
    ):
        raise ValueError("official and AP use different source data")
    practical = _practical_match(official, ap)
    return {
        "schema": SCHEMA,
        "scene": SCENE,
        "frame_count": FRAME_COUNT,
        "gop_count": GOP_COUNT,
        "practical_match_eligible": practical["eligible"],
        "claim_eligibility_basis": (
            "all-four-frozen-practical-match-gates-and-both-own-reference-lineages"
        ),
        "path_execution_authorized": True,
        "path_execution_policy": PATH_EXECUTION_POLICY,
        "path_execution_conditioned_on_practical_match": False,
        "ap_producer_array_job_id": expected_ap_array_job_id,
        "path_outcomes_read": [],
        "outcome_fields_read": [
            "producer_identity",
            "real_container_accounting",
            "ordinary_unedited_fidelity",
            "reference_tree_identity",
            "reference_checkpoint_lineage",
        ],
        "identity_comparison_design": (
            "official-decode-to-official-own-reference;"
            "ap-decode-to-ap-own-reference"
        ),
        "cross_lineage_identity_matching_authorized": False,
        "official": official,
        "ap": ap,
        "thresholds": practical["thresholds"],
        "gate_comparison_policy": practical["gate_comparison_policy"],
        "gates": practical["gates"],
        "ordinary_deltas_ap_minus_official": practical["deltas"],
        "rate_disclosure": practical["rate_disclosure"],
        "exact_practical_match_disclosure": practical["exact_disclosure"],
        "legacy_56_byte_gate_restored": False,
        "implementation_isolation": (
            "fresh-training-compression-and-evaluation-under-patch8;"
            "no-historical-output-match-required"
        ),
    }


def _self_test() -> None:
    assert AP_TASK_ID == "DEV.flame_salmon_1.ap.f03-patch8-ownref-rep1.p0c32"
    assert AP_TAR_PREFIX == "h007_p0c32_dev_ap_f03_patch8_ownref_rep1_"
    assert THRESHOLDS == {
        "absolute_relative_complete_rate_max": 0.015,
        "psnr_ap_minus_official_min_db": -0.1,
        "ssim_ap_minus_official_min": -0.002,
        "lpips_ap_minus_official_max": 0.005,
    }
    with tempfile.TemporaryDirectory(prefix="h007_match_selftest_") as temporary:
        root = Path(temporary)
        archive = root / "safe.tar"
        payloads = {
            "references/x/a": b"a",
            "references/x/b": b"bb",
        }
        with tarfile.open(archive, "w:") as handle:
            for name, payload in payloads.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                handle.addfile(info, io.BytesIO(payload))
        members = _safe_tar_members(archive)
        binding = _reference_tree_binding(archive, members)
        assert binding["file_count"] == 2
        assert len(binding["tree_sha256"]) == 64
        unsafe = root / "unsafe.tar"
        with tarfile.open(unsafe, "w:") as handle:
            info = tarfile.TarInfo("../escape")
            info.size = 1
            handle.addfile(info, io.BytesIO(b"x"))
        try:
            _safe_tar_members(unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe TAR member was accepted")
        official = {
            "sequence_bytes": 1_000_000,
            "psnr": 28.0,
            "ssim": 0.91,
            "lpips": 0.11,
        }
        eligible_ap = {
            "sequence_bytes": 1_014_999,
            "psnr": 27.901,
            "ssim": 0.9081,
            "lpips": 0.1149,
        }
        inclusive_boundary_ap = {
            "sequence_bytes": 1_015_000,
            "psnr": 27.9,
            "ssim": 0.908,
            "lpips": 0.115,
        }
        ineligible_ap = dict(eligible_ap, sequence_bytes=1_015_001)
        assert _practical_match(official, eligible_ap)["eligible"] is True
        assert (
            _practical_match(official, inclusive_boundary_ap)["eligible"]
            is True
        )
        ineligible = _practical_match(official, ineligible_ap)
        assert ineligible["eligible"] is False
        assert ineligible["gates"]["complete_rate"] is False
        try:
            _require_ap_compression_archive_binding(
                {"compression_archive_sha256": "a" * 64}, "b" * 64, 0
            )
        except ValueError:
            pass
        else:
            raise AssertionError("reference/sequence archive mismatch was accepted")
    print(
        json.dumps(
            {
                "schema": (
                    "h007.ap_gifstream.f03_patch8_paired_reference_match_self_test.v1"
                ),
                "status": "PASS",
                "checks": [
                    "reference-tree-hash",
                    "tar-path-traversal-rejected",
                    "practical-match-only-labels-claim-eligibility",
                    "unmatched-rate-path-execution-remains-authorized",
                    "reference-sequence-archive-mismatch-rejected",
                    "f03_patch8-identity-and-four-inclusive-gates-frozen",
                ],
                "path_outcomes_read": [],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--official-tar", type=Path)
    parser.add_argument("--ap-tar", type=Path)
    parser.add_argument("--ap-producer-registration", type=Path)
    parser.add_argument("--expected-ap-array-job-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if (
        args.official_tar is None
        or args.ap_tar is None
        or args.ap_producer_registration is None
        or args.expected_ap_array_job_id is None
        or args.output is None
    ):
        raise ValueError(
            "--official-tar, --ap-tar, --ap-producer-registration, "
            "--expected-ap-array-job-id, and --output are required"
        )
    if (
        not args.expected_ap_array_job_id.isascii()
        or not args.expected_ap_array_job_id.isdigit()
    ):
        raise ValueError("--expected-ap-array-job-id must be ASCII digits")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("paired match output must be append-only")
    registration = build_registration(
        args.official_tar,
        args.ap_tar,
        args.ap_producer_registration,
        args.expected_ap_array_job_id,
    )
    encoded = canonical_json_bytes(registration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": sha256_bytes(encoded),
                "schema": SCHEMA,
                "practical_match_eligible": registration[
                    "practical_match_eligible"
                ],
                "path_execution_authorized": True,
                "gates": registration["gates"],
                "path_outcomes_read": [],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
