#!/usr/bin/env python3
"""Export an AP pre-codec reference from a v3-campaign checkpoint.

V3 fork of h007_export_ap_same_run_reference_f03.py: identical validation
machinery, but every campaign-frozen expectation (scene, method, rate grid,
schedule, AP receipt fields, runtime patch chain) comes from an explicit
--campaign JSON instead of Patch8-era literals, and the AP training-state
schema is v3.  The original f03 exporter remains untouched for verifying the
author's archived runs.

This helper deliberately runs outside the registered GIFStream runtime.  It
does not alter training, compression, the compressed archive, or its decoder
configuration.  It reconstructs the same objects written by the in-run
H-DOWN exporter:

* ``reference_splats.pt`` from ``checkpoint["splats"]``;
* ``nets.pt`` from ``checkpoint["decoders"]`` and ``checkpoint["scaling"]``;
* ``decoder_config.json`` and ``camera_metadata/**`` byte-for-byte from the
  codec archive.

Every copied or reconstructed object is bound to the codec's frozen training
and producer receipts before the bundle is published.  F03 additionally
requires the Patch8 AP-v6 archive contract and refuses Patch7/v5 state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import torch


REFERENCE_SCHEMA = "h007.hdown_reference_bundle.v1"
CENSUS_SCHEMA = "h007.container_byte_census.v1"
TRAINING_RECEIPT_SCHEMA = "h007.gifstream_frozen_training_receipt.v1"
PRODUCER_RECEIPT_SCHEMA = "h007.gifstream_producer_receipt.v1"
DECODER_CONFIG_SCHEMA = "h007.gifstream_decoder_config.v3"
AP_TRAINING_STATE_SCHEMA = "h007.ap_training_state.v3"
AP_TRAINING_RECEIPT_SCHEMA = "h007.ap_training_receipt.v2"
AP_CODEC_SCHEMA = "h007.ap_gifstream.codec.v6"
PATH_CONTRACT_SCHEMA = "h007.ap_gifstream.path_contract.v1"
# Campaign-frozen expectations, populated by apply_campaign() from --campaign.
CAMPAIGN: dict = {}
EXPECTED_AP_RECEIPT: dict = {}
EXPECTED_PATH_LOSS_STEPS: list = []
RUNTIME_PATCH_COUNT = 0
RUNTIME_LAST_PATCH_SHA256 = ""
RUNTIME_TREE_SHA256 = ""
METHOD = ""
SCENE = ""
RATE = -1
RD_LAMBDA = float("nan")
TRAINING_STEP = 29999
GOP_SIZE = 60
GOP_COUNT = 5


def apply_campaign(campaign_path: Path, rate: int) -> None:
    """Load the campaign JSON and freeze module-level expectations."""
    global CAMPAIGN, EXPECTED_AP_RECEIPT, EXPECTED_PATH_LOSS_STEPS
    global RUNTIME_PATCH_COUNT, RUNTIME_LAST_PATCH_SHA256, RUNTIME_TREE_SHA256
    global METHOD, SCENE, RATE, RD_LAMBDA, TRAINING_STEP, GOP_SIZE, GOP_COUNT
    campaign = strict_json(read_regular(campaign_path, "campaign config"), "campaign config")
    if campaign.get("schema") != "h007.v3_campaign.flame_salmon.v1":
        raise ValueError("unsupported campaign schema")
    rates = campaign["rates"]
    if str(rate) not in rates:
        raise ValueError(f"rate {rate} is not in the campaign rate grid")
    CAMPAIGN = campaign
    EXPECTED_AP_RECEIPT = dict(campaign["expected_ap_receipt"])
    steps = campaign["path_loss_steps"]
    EXPECTED_PATH_LOSS_STEPS = list(range(int(steps["start"]), int(steps["stop"]), int(steps["step"])))
    runtime = campaign["runtime"]
    RUNTIME_PATCH_COUNT = int(runtime["patch_count"])
    RUNTIME_LAST_PATCH_SHA256 = require_sha256(runtime["last_patch_sha256"], "campaign last patch sha")
    RUNTIME_TREE_SHA256 = require_sha256(runtime["normalized_tree_sha256"], "campaign tree sha")
    METHOD = str(campaign["method"])
    SCENE = str(campaign["scene"])
    RATE = int(rate)
    RD_LAMBDA = float(rates[str(rate)])
    TRAINING_STEP = int(campaign["training_step"])
    GOP_SIZE = int(campaign["gop_size"])
    GOP_COUNT = int(campaign["gop_count"])
CAMERA_MEMBERS = (
    "bounds.npy",
    "camera_ids.npy",
    "camera_keys.npy",
    "camera_names.npy",
    "camtoworlds.npy",
    "image_sizes.npy",
    "intrinsics.npy",
    "transform.npy",
)
AP_SIDECARS = (
    "ap_class_mask.bin",
    "ap_path_input_mask.bin",
    "ap_real_row_mask.bin",
    "ap_padding_row_mask.bin",
    "ap_active_row_mask.bin",
    "ap_identity_corrections.bin",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def read_regular(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a regular file")
    opened = path.stat()
    if not stat.S_ISREG(opened.st_mode) or opened.st_size <= 0:
        raise ValueError(f"{label} is empty or non-regular")
    return path.read_bytes()


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def require_sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def require_exact_int(value: Any, expected: int, label: str) -> int:
    if type(value) is not int or value != expected:
        raise ValueError(f"{label} differs from the frozen value {expected}")
    return value


def require_exact_float(value: Any, expected: float, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} is not numeric")
    observed = float(value)
    if not math.isfinite(observed) or observed != expected:
        raise ValueError(f"{label} differs from the frozen value {expected}")
    return observed


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a mapping")
    return value


def validate_packed_mask(
    record: Mapping[str, Any],
    payload: bytes,
    expected_path: str,
    label: str,
) -> None:
    required = {"path", "count", "true_count", "bytes", "sha256", "bitorder"}
    if (
        set(record) != required
        or record.get("path") != expected_path
        or record.get("bitorder") != "little"
        or type(record.get("count")) is not int
        or type(record.get("true_count")) is not int
        or record["count"] <= 0
        or record["true_count"] < 0
        or record["true_count"] > record["count"]
        or record.get("bytes") != len(payload)
        or len(payload) != (record["count"] + 7) // 8
        or require_sha256(record.get("sha256"), f"{label} SHA-256")
            != sha256_bytes(payload)
    ):
        raise ValueError(f"{label} packed-mask metadata differs")
    bits = [
        bool(payload[index // 8] & (1 << (index % 8)))
        for index in range(record["count"])
    ]
    if sum(bits) != record["true_count"]:
        raise ValueError(f"{label} packed-mask true count differs")
    for index in range(record["count"], len(payload) * 8):
        if payload[index // 8] & (1 << (index % 8)):
            raise ValueError(f"{label} packed-mask padding bits are nonzero")


def validate_identity_corrections(
    record: Mapping[str, Any],
    payload: bytes,
) -> None:
    required = {
        "schema", "path", "row_count", "mismatch_count", "mask_bytes",
        "bytes", "sha256", "bitorder", "base", "code",
    }
    if (
        set(record) != required
        or record.get("schema") != "h007.ap_identity_corrections.v1"
        or record.get("path") != "ap_identity_corrections.bin"
        or record.get("bitorder") != "little"
        or record.get("base") != "round-decoded-anchor-div-voxel-size"
        or record.get("code") != "uint8-base3-dx-dy-dz-plus1"
        or type(record.get("row_count")) is not int
        or type(record.get("mismatch_count")) is not int
        or type(record.get("mask_bytes")) is not int
        or record["row_count"] <= 0
        or record["mismatch_count"] < 0
        or record["mismatch_count"] > record["row_count"]
        or record["mask_bytes"] != (record["row_count"] + 7) // 8
        or record.get("bytes") != len(payload)
        or len(payload) != record["mask_bytes"] + record["mismatch_count"]
        or require_sha256(
            record.get("sha256"), "AP identity-correction SHA-256"
        ) != sha256_bytes(payload)
    ):
        raise ValueError("AP identity-correction metadata differs")
    mask_record = {
        "path": "identity-mask-prefix",
        "count": record["row_count"],
        "true_count": record["mismatch_count"],
        "bytes": record["mask_bytes"],
        "sha256": sha256_bytes(payload[: record["mask_bytes"]]),
        "bitorder": "little",
    }
    validate_packed_mask(
        mask_record,
        payload[: record["mask_bytes"]],
        "identity-mask-prefix",
        "AP identity-correction mismatch mask",
    )
    codes = payload[record["mask_bytes"] :]
    if any(code > 26 or code == 13 for code in codes):
        raise ValueError("AP identity-correction stream contains an invalid code")


def validate_campaign_contract(
    checkpoint: Mapping[str, Any],
    ap_training_payload: bytes,
    codec_meta: Mapping[str, Any],
    producer: Mapping[str, Any],
    sidecar_payloads: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject any archive/checkpoint that is not the registered campaign state."""
    receipt = strict_json(ap_training_payload, "AP training receipt")
    state = require_mapping(checkpoint.get("ap_state"), "checkpoint AP state")
    checkpoint_receipt = require_mapping(
        checkpoint.get("ap_training_receipt"),
        "checkpoint AP training receipt",
    )
    if receipt.get("schema") != AP_TRAINING_RECEIPT_SCHEMA:
        raise ValueError("archive lacks the registered AP training-receipt contract")
    if state.get("schema") != AP_TRAINING_STATE_SCHEMA:
        raise ValueError("checkpoint lacks the registered AP training-state contract")
    if checkpoint_receipt.get("schema") != AP_TRAINING_RECEIPT_SCHEMA:
        raise ValueError("checkpoint lacks the registered AP training-receipt contract")
    if canonical_json_bytes(checkpoint_receipt) != ap_training_payload:
        raise ValueError("archive AP receipt differs from the fresh checkpoint")

    expected_receipt = dict(EXPECTED_AP_RECEIPT)
    for name, expected in expected_receipt.items():
        if receipt.get(name) != expected:
            raise ValueError(f"AP training receipt campaign field differs: {name}")
    expected_steps = list(EXPECTED_PATH_LOSS_STEPS)
    if (
        receipt.get("path_loss_applications") != len(expected_steps)
        or receipt.get("path_loss_steps") != expected_steps
    ):
        raise ValueError("AP path-loss application/step closure differs")
    graph_sha = require_sha256(
        receipt.get("path_knn_graph_sha256"),
        "AP retained-KNN graph SHA-256",
    )
    provenance = require_mapping(
        receipt.get("runtime_provenance"), "AP runtime provenance"
    )
    patch_chain = provenance.get("patch_sha256")
    tree = require_mapping(
        provenance.get("normalized_code_tree"),
        "AP normalized code tree",
    )
    if (
        not isinstance(patch_chain, list)
        or len(patch_chain) != RUNTIME_PATCH_COUNT
        or patch_chain[-1] != RUNTIME_LAST_PATCH_SHA256
        or tree.get("sha256") != RUNTIME_TREE_SHA256
    ):
        raise ValueError("AP receipt is not bound to the registered campaign tree")

    ap_meta = require_mapping(codec_meta.get("__ap__"), "AP codec metadata")
    if ap_meta.get("schema") != AP_CODEC_SCHEMA:
        raise ValueError("archive is not an AP codec-v6 container")
    if require_mapping(ap_meta.get("variant"), "AP variant").get("name") != METHOD:
        raise ValueError("AP codec variant differs")
    require_exact_float(
        ap_meta.get("q_ap_multiplier"),
        float(EXPECTED_AP_RECEIPT["q_ap_multiplier"]),
        "AP multiplier",
    )
    require_exact_float(
        ap_meta.get("q_bg_multiplier"),
        float(EXPECTED_AP_RECEIPT["q_bg_multiplier"]),
        "background multiplier",
    )
    contract = require_mapping(ap_meta.get("path_contract"), "AP path contract")
    expected_contract = {
        "schema": PATH_CONTRACT_SCHEMA,
        "knn_count": int(EXPECTED_AP_RECEIPT["path_knn_count"]),
        "knn_rule":
            "retained-canonical-radius-complete-distance+lexicographic-id",
        "dependency_rule": str(EXPECTED_AP_RECEIPT["path_dependency_rule"]),
        "retained_knn_graph_sha256": graph_sha,
        "canonical_anchor_reconstruction": True,
        "factor_protected_multiplier": float(EXPECTED_AP_RECEIPT["factor_protected_multiplier"]),
        "factor_background_multiplier": float(EXPECTED_AP_RECEIPT["factor_background_multiplier"]),
        "anchor_feature_protected_multiplier": float(EXPECTED_AP_RECEIPT["anchor_feature_protected_multiplier"]),
        "anchor_feature_background_multiplier": float(EXPECTED_AP_RECEIPT["anchor_feature_background_multiplier"]),
    }
    if dict(contract) != expected_contract:
        raise ValueError("AP codec-v6 retained-KNN path contract differs")

    for key, expected_path in (
        ("mask", "ap_class_mask.bin"),
        ("path_input_mask", "ap_path_input_mask.bin"),
        ("real_row_mask", "ap_real_row_mask.bin"),
        ("padding_row_mask", "ap_padding_row_mask.bin"),
        ("active_row_mask", "ap_active_row_mask.bin"),
    ):
        record = require_mapping(ap_meta.get(key), f"AP {key} metadata")
        payload = sidecar_payloads.get(expected_path)
        if payload is None:
            raise ValueError(f"AP codec-v6 {key} sidecar is missing")
        validate_packed_mask(record, payload, expected_path, f"AP {key}")
    identity = require_mapping(
        ap_meta.get("identity_corrections"),
        "AP identity corrections",
    )
    identity_payload = sidecar_payloads.get("ap_identity_corrections.bin")
    if identity_payload is None:
        raise ValueError("AP identity-correction sidecar is missing")
    validate_identity_corrections(identity, identity_payload)

    state_graph = require_sha256(
        state.get("path_knn_graph_sha256"),
        "checkpoint retained-KNN graph SHA-256",
    )
    if (
        state.get("path_contract_schema") != PATH_CONTRACT_SCHEMA
        or state_graph != graph_sha
    ):
        raise ValueError("checkpoint and archive retained-KNN contracts differ")
    masks: dict[str, torch.Tensor] = {}
    for name in ("ap_class_mask", "ap_path_input_mask", "ap_retain_mask"):
        value = state.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise ValueError(f"checkpoint AP mask is malformed: {name}")
        masks[name] = value.detach().cpu().to(dtype=torch.bool)
    if not (
        masks["ap_class_mask"].shape
        == masks["ap_path_input_mask"].shape
        == masks["ap_retain_mask"].shape
    ):
        raise ValueError("checkpoint AP path masks have different shapes")
    if torch.any(masks["ap_class_mask"] & ~masks["ap_path_input_mask"]):
        raise ValueError("protected AP rows escape the path-input closure")
    if torch.any(masks["ap_path_input_mask"] & ~masks["ap_retain_mask"]):
        raise ValueError("AP path-input closure escapes retained identities")
    canonical_ids = state.get("canonical_ids")
    if (
        not isinstance(canonical_ids, torch.Tensor)
        or canonical_ids.ndim != 2
        or canonical_ids.shape != (masks["ap_retain_mask"].numel(), 3)
    ):
        raise ValueError("checkpoint canonical anchor identities are malformed")
    runtime = require_mapping(
        receipt.get("runtime_provenance"), "AP training runtime provenance"
    )
    if not (
        state.get("runtime_provenance")
        == checkpoint_receipt.get("runtime_provenance")
        == ap_meta.get("runtime_provenance")
        == producer.get("runtime_provenance")
        == runtime
    ):
        raise ValueError(
            "checkpoint state/receipt, AP metadata, and producer runtime "
            "provenance differ"
        )
    return receipt, dict(ap_meta)


def archive_members(archive: Path) -> tuple[zipfile.ZipFile, dict[str, zipfile.ZipInfo]]:
    handle = zipfile.ZipFile(archive, "r")
    members: dict[str, zipfile.ZipInfo] = {}
    for info in handle.infolist():
        name = info.filename
        pure = PurePosixPath(name)
        if (
            not name
            or name.endswith("/")
            or pure.is_absolute()
            or ".." in pure.parts
            or "\\" in name
            or name in members
        ):
            handle.close()
            raise ValueError(f"codec archive contains an unsafe or duplicate member: {name}")
        if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
            handle.close()
            raise ValueError(f"codec archive contains a symlink: {name}")
        members[name] = info
    return handle, members


def require_archive_member(
    archive: zipfile.ZipFile,
    members: Mapping[str, zipfile.ZipInfo],
    name: str,
) -> bytes:
    info = members.get(name)
    if info is None or info.file_size <= 0:
        raise ValueError(f"codec archive member is missing or empty: {name}")
    payload = archive.read(info)
    if len(payload) != info.file_size:
        raise ValueError(f"short codec archive member read: {name}")
    return payload


def tensor_mapping_sha256(mapping: Mapping[str, Any]) -> str:
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("tensor mapping is empty or invalid")
    digest = hashlib.sha256()
    for name in sorted(mapping):
        value = mapping[name]
        if not isinstance(name, str) or not name:
            raise ValueError("tensor mapping key is invalid")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"tensor mapping value is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def file_census(root: Path, declared_root: str) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise ValueError(f"symlink forbidden in reference bundle: {path}")
        payload = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    return {
        "schema": CENSUS_SCHEMA,
        "root": declared_root,
        "file_count": len(rows),
        "raw_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }


def validate_receipts(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    checkpoint_sha256: str,
    checkpoint_bytes: int,
    training_payload: bytes,
    producer_payload: bytes,
    ap_training_payload: bytes,
    decoder_config: Mapping[str, Any],
    gop_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    training = strict_json(training_payload, "frozen training receipt")
    producer = strict_json(producer_payload, "producer receipt")
    if training.get("schema") != TRAINING_RECEIPT_SCHEMA:
        raise ValueError("frozen training receipt schema differs")
    if producer.get("schema") != PRODUCER_RECEIPT_SCHEMA:
        raise ValueError("producer receipt schema differs")
    for value, label in (
        (training.get("scene"), "training scene"),
        (producer.get("scene"), "producer scene"),
        (decoder_config.get("scene"), "decoder scene"),
    ):
        if value != SCENE:
            raise ValueError(f"{label} differs from {SCENE}")
    for value, label in (
        (training.get("variant"), "training variant"),
        (producer.get("variant"), "producer variant"),
        (decoder_config.get("variant"), "decoder variant"),
    ):
        if value != METHOD:
            raise ValueError(f"{label} differs from {METHOD}")
    require_exact_int(training.get("training_step"), TRAINING_STEP, "training step")
    require_exact_int(producer.get("training_step"), TRAINING_STEP, "producer step")
    require_exact_int(decoder_config.get("start_frame"), gop_id * GOP_SIZE, "GOP start")
    require_exact_int(decoder_config.get("GOP_size"), GOP_SIZE, "decoder GOP size")
    require_exact_int(decoder_config.get("rate"), RATE, "decoder rate")
    require_exact_float(
        producer.get("training_config", {}).get("rd_lambda"),
        RD_LAMBDA,
        "producer RD lambda",
    )
    require_exact_float(
        training.get("training_config", {}).get("rd_lambda"),
        RD_LAMBDA,
        "training RD lambda",
    )
    if training.get("state_position") != (
        "after_optimizer_entropy_and_strategy_post_backward"
    ):
        raise ValueError("training receipt state position differs")
    rows = training.get("source_checkpoints")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("training receipt does not bind exactly one checkpoint")
    row = rows[0]
    if (
        require_sha256(row.get("sha256"), "training checkpoint SHA-256")
        != checkpoint_sha256
        or row.get("bytes") != checkpoint_bytes
    ):
        raise ValueError("checkpoint bytes differ from frozen training receipt")
    if producer.get("source_checkpoints") != rows:
        raise ValueError("producer and training receipts bind different checkpoints")
    training_sha = sha256_bytes(training_payload)
    producer_sha = sha256_bytes(producer_payload)
    if decoder_config.get("training_receipt_sha256") != training_sha:
        raise ValueError("decoder config training-receipt binding differs")
    if decoder_config.get("producer_receipt_sha256") != producer_sha:
        raise ValueError("decoder config producer-receipt binding differs")
    if producer.get("training_receipt_sha256") != training_sha:
        raise ValueError("producer receipt training-receipt binding differs")
    if producer.get("ap_training_receipt_sha256") != sha256_bytes(
        ap_training_payload
    ):
        raise ValueError("producer receipt AP-training binding differs")
    if checkpoint.get("step") != TRAINING_STEP:
        raise ValueError("checkpoint step differs from frozen receipt")
    if checkpoint.get("state_position") != training.get("state_position"):
        raise ValueError("checkpoint state position differs from frozen receipt")
    if checkpoint.get("compression_sim") is not True:
        raise ValueError("checkpoint lacks the frozen compression-simulation state")
    for name in ("splats", "decoders", "scaling"):
        if name not in checkpoint:
            raise ValueError(f"checkpoint lacks {name}")
    model_state = producer.get("model_state_sha256")
    if not isinstance(model_state, dict):
        raise ValueError("producer receipt lacks model-state hashes")
    if tensor_mapping_sha256(checkpoint["splats"]) != model_state.get("splats"):
        raise ValueError("checkpoint splats differ from producer receipt")
    if tensor_mapping_sha256(checkpoint["decoders"]) != model_state.get("decoders"):
        raise ValueError("checkpoint decoders differ from producer receipt")
    scaling_payload = json.dumps(
        checkpoint["scaling"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if sha256_bytes(scaling_payload) != model_state.get("codec_scaling"):
        raise ValueError("checkpoint scaling differs from producer receipt")
    runtime_provenance = require_mapping(
        producer.get("runtime_provenance"), "producer runtime provenance"
    )
    patch_chain = runtime_provenance.get("patch_sha256")
    normalized_tree = require_mapping(
        runtime_provenance.get("normalized_code_tree"),
        "producer normalized code tree",
    )
    if (
        not isinstance(patch_chain, list)
        or len(patch_chain) != RUNTIME_PATCH_COUNT
        or patch_chain[-1] != RUNTIME_LAST_PATCH_SHA256
        or normalized_tree.get("sha256") != RUNTIME_TREE_SHA256
    ):
        raise ValueError("producer receipt is not bound to the campaign runtime")
    if training.get("runtime_provenance") != runtime_provenance:
        raise ValueError("outer training and producer runtime provenance differ")
    return training, producer


def exclusive_write(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("short exclusive write")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)


def export_reference(args: argparse.Namespace) -> dict[str, Any]:
    if args.gop_id not in range(GOP_COUNT):
        raise ValueError("GOP ID must be in 0..4")
    checkpoint_path = args.checkpoint.resolve(strict=True)
    archive_path = args.compression_archive.resolve(strict=True)
    checkpoint_bytes = checkpoint_path.stat().st_size
    checkpoint_sha = sha256_file(checkpoint_path)
    archive_sha = sha256_file(archive_path)
    archive, members = archive_members(archive_path)
    try:
        decoder_payload = require_archive_member(
            archive, members, "decoder_config.json"
        )
        training_payload = require_archive_member(
            archive, members, "training_receipt.json"
        )
        producer_payload = require_archive_member(
            archive, members, "producer_receipt.json"
        )
        ap_training_payload = require_archive_member(
            archive, members, "ap_training_receipt.json"
        )
        codec_meta_payload = require_archive_member(
            archive, members, "meta.json"
        )
        sidecar_payloads = {
            name: require_archive_member(archive, members, name)
            for name in AP_SIDECARS
        }
        camera_payloads = {
            name: require_archive_member(
                archive, members, f"camera_metadata/{name}"
            )
            for name in CAMERA_MEMBERS
        }
    finally:
        archive.close()
    decoder_config = strict_json(decoder_payload, "decoder config")
    if decoder_config.get("schema") != DECODER_CONFIG_SCHEMA:
        raise ValueError("decoder config schema differs from the registered Patch8 runtime")
    codec_meta = strict_json(codec_meta_payload, "codec metadata")
    producer_contract = strict_json(producer_payload, "producer receipt")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint is not a mapping")
    ap_training_receipt, ap_codec_meta = validate_campaign_contract(
        checkpoint,
        ap_training_payload,
        codec_meta,
        producer_contract,
        sidecar_payloads,
    )
    training, producer = validate_receipts(
        checkpoint_path,
        checkpoint,
        checkpoint_sha,
        checkpoint_bytes,
        training_payload,
        producer_payload,
        ap_training_payload,
        decoder_config,
        args.gop_id,
    )

    output = args.output_bundle.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"reference output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent)
    )
    try:
        exclusive_write(staging / "decoder_config.json", decoder_payload)
        camera_root = staging / "camera_metadata"
        camera_root.mkdir(mode=0o755)
        for name, payload in camera_payloads.items():
            exclusive_write(camera_root / name, payload)

        reference_splats = {
            name: value.detach().cpu()
            for name, value in checkpoint["splats"].items()
        }
        reference_nets = {
            "decoders": {
                name: value.detach().cpu()
                for name, value in checkpoint["decoders"].items()
            },
            "scaling": checkpoint["scaling"],
        }
        if bool(decoder_config.get("app_opt")):
            if "app_module" not in checkpoint:
                raise ValueError("appearance-enabled checkpoint lacks app_module")
            reference_nets["app_module"] = {
                name: value.detach().cpu()
                for name, value in checkpoint["app_module"].items()
            }
        torch.save(reference_nets, staging / "nets.pt")
        torch.save(reference_splats, staging / "reference_splats.pt")

        manifest = {
            "schema": REFERENCE_SCHEMA,
            "scene": SCENE,
            "gop_id": args.gop_id,
            "start_frame": args.gop_id * GOP_SIZE,
            "frame_count": GOP_SIZE,
            "camera": "cam00",
            "variant": f"{METHOD}_pre_codec_reference",
            "reference_bundle_semantics": "same-run-pre-codec-ap-checkpoint",
            "reference_method": METHOD,
            "source_checkpoint_sha256": checkpoint_sha,
            "source_checkpoint_bytes": checkpoint_bytes,
            "source_training_receipt_sha256": sha256_bytes(training_payload),
            "producer_receipt_sha256": sha256_bytes(producer_payload),
            "compression_archive_sha256": archive_sha,
            "decoder_config_sha256": sha256_bytes(decoder_payload),
            "codec_meta_sha256": sha256_bytes(codec_meta_payload),
            "ap_training_receipt_sha256": sha256_bytes(ap_training_payload),
            "ap_codec_schema": ap_codec_meta["schema"],
            "ap_training_state_schema": AP_TRAINING_STATE_SCHEMA,
            "ap_training_receipt_schema": ap_training_receipt["schema"],
            "path_contract": ap_codec_meta["path_contract"],
            "reference_nets_sha256": sha256_file(staging / "nets.pt"),
            "reference_splats_sha256": sha256_file(
                staging / "reference_splats.pt"
            ),
            "model_state_sha256": producer["model_state_sha256"],
            "training_config_sha256": training["training_config_sha256"],
            "runtime_provenance": producer["runtime_provenance"],
            "candidate_inputs_read": [],
            "outcome_fields_read": [],
            "path_outcomes_read": [],
        }
        exclusive_write(
            staging / "reference_bundle_manifest.json",
            canonical_json_bytes(manifest),
        )
        census = file_census(staging, output.name)
        exclusive_write(
            staging / "byte_census.json",
            canonical_json_bytes(census),
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--rate", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--compression-archive", type=Path, required=True)
    parser.add_argument("--gop-id", type=int, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    apply_campaign(args.campaign, args.rate)
    manifest = export_reference(args)
    summary = {
        "schema": "h007.ap_same_run_reference_export.v3",
        "status": "COMPLETED",
        "scene": manifest["scene"],
        "gop_id": manifest["gop_id"],
        "reference_bundle": str(args.output_bundle.absolute()),
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
        "source_training_receipt_sha256": manifest[
            "source_training_receipt_sha256"
        ],
        "ap_training_receipt_sha256": manifest[
            "ap_training_receipt_sha256"
        ],
        "codec_meta_sha256": manifest["codec_meta_sha256"],
        "ap_codec_schema": manifest["ap_codec_schema"],
        "path_contract": manifest["path_contract"],
        "reference_bundle_manifest_sha256": sha256_file(
            args.output_bundle / "reference_bundle_manifest.json"
        ),
        "byte_census_sha256": sha256_file(args.output_bundle / "byte_census.json"),
        "candidate_inputs_read": [],
        "outcome_fields_read": [],
        "path_outcomes_read": [],
    }
    exclusive_write(args.summary, canonical_json_bytes(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
