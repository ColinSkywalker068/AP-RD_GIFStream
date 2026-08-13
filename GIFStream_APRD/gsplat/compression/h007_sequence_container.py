"""Executable 300-frame sequence containers and frozen real-ZIP selection.

This module is deliberately stdlib-only.  It validates and counts the five
already-built 60-frame GOP archives, builds one deterministic outer ZIP, and
selects the preregistered 2,300,000-byte operating point from actual archive
sizes.  It never reads H-DOWN render or edit outcomes.
"""

from __future__ import annotations

import ast
import binascii
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import pickle
import pickletools
import re
import shutil
import stat
import struct
import tempfile
import zipfile
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SEQUENCE_SCHEMA = "h007.gifstream_sequence_container.v2"
SEQUENCE_CENSUS_SCHEMA = "h007.sequence_byte_census.v1"
SELECTION_SCHEMA = "h007.real_zip_operating_point_selection.v2"
ELIGIBILITY_SCHEMA = "h007.h_sota_operating_point_eligibility_request.v3"
ELIGIBILITY_EVIDENCE_SCHEMA = "h007.ordinary_rate_quality_evidence.v4"
EVALUATOR_RECEIPT_SCHEMA = "h007.ordinary_rate_quality_evaluator_receipt.v3"
GIFSTREAM_PAYLOAD_SCHEMA = "h007.gifstream_payload_manifest.v1"
PRODUCER_RECEIPT_SCHEMA = "h007.gifstream_producer_receipt.v1"
FROZEN_TRAINING_RECEIPT_SCHEMA = "h007.gifstream_frozen_training_receipt.v1"
DECODER_CONFIG_SCHEMA = "h007.gifstream_decoder_config.v3"
ORDINARY_EVALUATOR_RELATIVE_PATH = "examples/h007_ordinary_rate_quality.py"
SOURCE_DATA_SCHEMA = "h007.neur3d_source_data_manifest.v1"
OFFICIAL_COMMIT = "c98486632e7dafd830740b1a1692bd08c48b96e3"
CONFIRMATORY_SCENES = (
    "coffee_martini",
    "cook_spinach",
    "cut_roasted_beef",
    "flame_steak",
    "sear_steak",
)
ALL_SCENES = ("flame_salmon_1",) + CONFIRMATORY_SCENES
GOP_STARTS = (0, 60, 120, 180, 240)
TARGET_BYTES = 2_300_000
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
ORDINARY_METRIC_PROTOCOL = "ordinary_unedited_psnr_ssim_lpips_300frames.v3"
FROZEN_RATE_LAMBDAS = {
    0: 0.0005,
    1: 0.001,
    2: 0.002,
    3: 0.004,
    4: 0.0009,
    5: 0.00095,
}
FROZEN_ORDINARY_RATE_LAMBDAS = {
    index: FROZEN_RATE_LAMBDAS[index] for index in range(4)
}
MIN_ADJACENT_RATE_BYTE_FRACTION = 0.05
ELIGIBILITY_RECOMPUTATION_SCHEMA = "h007.h_sota_eligibility_recomputation.v4"
GIFSTREAM_ENTROPY_MODEL_KEYS = (
    "scales",
    "anchor_features",
    "offsets",
    "factors",
    "time_features",
)
FROZEN_SCENE_CAMERA_COUNTS = {
    "flame_salmon_1": 19,
    "coffee_martini": 18,
    "cook_spinach": 21,
    "cut_roasted_beef": 20,
    "flame_steak": 21,
    "sear_steak": 21,
}


def frozen_scene_camera_count(scene: str) -> int:
    if type(scene) is not str or scene not in FROZEN_SCENE_CAMERA_COUNTS:
        raise ValueError("scene is outside the frozen Neu3D camera census")
    return FROZEN_SCENE_CAMERA_COUNTS[scene]


def frozen_camera_names(scene: str, suffix: str = ".png") -> Tuple[str, ...]:
    if type(suffix) is not str or suffix not in {"", ".png"}:
        raise ValueError("camera-name suffix is outside the frozen contract")
    return tuple(
        f"cam{index:02d}{suffix}"
        for index in range(frozen_scene_camera_count(scene))
    )

DECODER_CONFIG_FIELDS = {
    "schema",
    "codec_family",
    "official_commit",
    "patch_chain_sha256",
    "runtime_manifest_sha256",
    "normalized_code_tree_sha256",
    "producer_receipt_sha256",
    "training_receipt_sha256",
    "payload_manifest_sha256",
    "variant",
    "scene",
    "data_factor",
    "start_frame",
    "GOP_size",
    "rate",
    "voxel_size",
    "anchor_feature_dim",
    "c_perframe",
    "entropy_channel",
    "n_offsets",
    "n_knn",
    "knn",
    "time_dim",
    "view_adaptive",
    "add_opacity_dist",
    "add_cov_dist",
    "add_color_dist",
    "app_opt",
    "app_embed_dim",
    "appearance_embedding_count",
    "packed",
    "antialiased",
    "camera_model",
    "phi",
    "test_set",
    "remove_set",
    "compression_seed",
    "warm_camera_pose_index",
    "warm_frame_index",
    "warmup_renders",
    "timed_renders",
    "clean_decode_entrypoint",
}

RUNTIME_RECEIPT_FIELDS = {
    "schema",
    "encode_seconds",
    "model_load_plus_entropy_decode_seconds",
    "peak_decode_cuda_bytes",
    "warm_render",
    "warm_render_fps",
    "ap_score_seconds",
    "outcome_fields_read",
}

CLEAN_DECODE_REQUEST_FIELDS = {
    "schema",
    "archive_only",
    "entrypoint",
    "expected_output",
    "expected_runtime_output",
    "external_shared_runtime",
}

PRODUCER_TRAINING_CONFIG_FIELDS = {
    "scene",
    "variant",
    "data_factor",
    "GOP_size",
    "rate",
    "rd_lambda",
    "max_steps",
    "random_seed",
    "compression_seed",
    "voxel_size",
    "anchor_feature_dim",
    "c_perframe",
    "entropy_channel",
    "n_offsets",
    "n_knn",
    "knn",
    "time_dim",
    "view_adaptive",
    "app_opt",
    "compression_sim",
    "entropy_model_opt",
}

DECODER_POSITIVE_INT_FIELDS = (
    "data_factor",
    "GOP_size",
    "anchor_feature_dim",
    "c_perframe",
    "entropy_channel",
    "n_offsets",
    "n_knn",
    "time_dim",
    "app_embed_dim",
    "appearance_embedding_count",
    "warmup_renders",
    "timed_renders",
)
DECODER_NONNEGATIVE_INT_FIELDS = (
    "start_frame",
    "rate",
    "compression_seed",
    "warm_camera_pose_index",
    "warm_frame_index",
)
PRODUCER_TRAINING_POSITIVE_INT_FIELDS = (
    "data_factor",
    "GOP_size",
    "max_steps",
    "anchor_feature_dim",
    "c_perframe",
    "entropy_channel",
    "n_offsets",
    "n_knn",
    "time_dim",
)
PRODUCER_TRAINING_NONNEGATIVE_INT_FIELDS = (
    "random_seed",
    "compression_seed",
)
PRODUCER_TRAINING_BOOL_FIELDS = (
    "knn",
    "view_adaptive",
    "app_opt",
    "compression_sim",
    "entropy_model_opt",
)

AP_SCORE_NPZ_MEMBERS = {
    "schema.npy",
    "scene.npy",
    "voxel_size.npy",
    "frame_count.npy",
    "variant.npy",
    "protected_fraction.npy",
    "q_ap_multiplier.npy",
    "q_bg_multiplier.npy",
    "random_seed.npy",
    "canonical_ids.npy",
    "eligible.npy",
    "path_score.npy",
    "motion_score.npy",
    "allocation_score.npy",
    "importance_score.npy",
    "estimated_time_bytes.npy",
    "official_retain_mask.npy",
    "official_factor0_mask.npy",
    "official_active_mask.npy",
    "ap_retain_mask.npy",
    "ap_active_mask.npy",
    "ap_class_mask.npy",
    "factor0_activation_value.npy",
    "factor3_activation_value.npy",
    "estimator_version.npy",
    "time_entropy_model_sha256.npy",
    "time_feature_scaling.npy",
    "time_entropy_model_frozen_after_freeze.npy",
    "runtime_manifest_sha256.npy",
    "normalized_code_tree_sha256.npy",
    "patch_chain_sha256.npy",
    "path_definition.npy",
    "motion_definition.npy",
    "importance_definition.npy",
    "estimated_byte_definition.npy",
}
AP_SCORE_NPZ_ORDER = (
    "schema.npy",
    "scene.npy",
    "voxel_size.npy",
    "frame_count.npy",
    "variant.npy",
    "protected_fraction.npy",
    "q_ap_multiplier.npy",
    "q_bg_multiplier.npy",
    "random_seed.npy",
    "canonical_ids.npy",
    "eligible.npy",
    "path_score.npy",
    "motion_score.npy",
    "allocation_score.npy",
    "importance_score.npy",
    "estimated_time_bytes.npy",
    "official_retain_mask.npy",
    "official_factor0_mask.npy",
    "official_active_mask.npy",
    "ap_retain_mask.npy",
    "ap_active_mask.npy",
    "ap_class_mask.npy",
    "factor0_activation_value.npy",
    "factor3_activation_value.npy",
    "estimator_version.npy",
    "time_entropy_model_sha256.npy",
    "time_feature_scaling.npy",
    "time_entropy_model_frozen_after_freeze.npy",
    "runtime_manifest_sha256.npy",
    "normalized_code_tree_sha256.npy",
    "patch_chain_sha256.npy",
    "path_definition.npy",
    "motion_definition.npy",
    "importance_definition.npy",
    "estimated_byte_definition.npy",
)

AP_EDIT_IDS_NPZ_MEMBERS = {
    "schema.npy",
    "scene.npy",
    "voxel_size.npy",
    "canonical_ids.npy",
    "source_score_sha256.npy",
    "selection.npy",
    "reference_manifest_sha256.npy",
    "selected_canonical_ids_sha256.npy",
    "path_score.npy",
}
AP_EDIT_IDS_NPZ_ORDER = (
    "schema.npy",
    "scene.npy",
    "voxel_size.npy",
    "canonical_ids.npy",
    "source_score_sha256.npy",
    "selection.npy",
    "reference_manifest_sha256.npy",
    "selected_canonical_ids_sha256.npy",
    "path_score.npy",
)
AP_SCORE_SCHEMA = "h007.ap_scores.v3"
AP_EDIT_IDS_SCHEMA = "h007.ap_edit_ids.v1"
AP_EDIT_SELECTION = "top_path_score_intersection_official_and_ap_retained"
AP_ESTIMATOR_VERSION = "h007.conditional_gaussian_per_row_bits.v1"
AP_PATH_DEFINITION = "sum_consecutive_euclidean_displacement"
AP_MOTION_DEFINITION = "mean_distance_from_canonical_anchor"
AP_IMPORTANCE_DEFINITION = "backbone_blended_opacity_per_visit_prune_statistic"
AP_ESTIMATED_BYTE_DEFINITION = (
    "ceil_deterministic_conditional_gaussian_bits_over_8"
)
AP_EDIT_SOURCE_VARIANT = "ap-gifstream-full"
AP_VARIANT_METADATA = {
    "random-full": {
        "name": "random-full",
        "ranking": "random",
        "swap": True,
        "quant": True,
        "action_loss": True,
    },
    "motion-full": {
        "name": "motion-full",
        "ranking": "motion",
        "swap": True,
        "quant": True,
        "action_loss": True,
    },
    "path-swap": {
        "name": "path-swap",
        "ranking": "path",
        "swap": True,
        "quant": False,
        "action_loss": False,
    },
    "path-quant": {
        "name": "path-quant",
        "ranking": "path",
        "swap": False,
        "quant": True,
        "action_loss": False,
    },
    "path-swap-quant": {
        "name": "path-swap-quant",
        "ranking": "path",
        "swap": True,
        "quant": True,
        "action_loss": False,
    },
    "ap-gifstream-full": {
        "name": "ap-gifstream-full",
        "ranking": "path",
        "swap": True,
        "quant": True,
        "action_loss": True,
    },
}
QUANTIZED_AP_VARIANTS = frozenset(
    name for name, row in AP_VARIANT_METADATA.items() if row["quant"] is True
)


def _edit_source_variant(method: str) -> str:
    if method == "official":
        return AP_EDIT_SOURCE_VARIANT
    if method not in AP_VARIANT_METADATA:
        raise ValueError("edit source method is outside the frozen AP variants")
    return method


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def expected_gifstream_nets_keys(app_opt: bool) -> set:
    keys = {"decoders", "scaling"} | {
        f"{name}_entropy_model" for name in GIFSTREAM_ENTROPY_MODEL_KEYS
    }
    if bool(app_opt):
        keys.add("app_module")
    return keys


def expected_gifstream_reference_nets_keys(app_opt: bool) -> set:
    keys = {"decoders", "scaling"}
    if bool(app_opt):
        keys.add("app_module")
    return keys


def _contiguous_stride(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    stride: List[int] = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= size
    return tuple(reversed(stride))


def _expected_nets_tensor_schema(
    config: Mapping[str, Any]
) -> Dict[str, List[Tuple[str, Tuple[int, ...]]]]:
    feature = _positive_int(config["anchor_feature_dim"], "decoder feature dimension")
    channel = _positive_int(config["c_perframe"], "decoder temporal channel count")
    entropy = _positive_int(config["entropy_channel"], "entropy channel count")
    offsets = _positive_int(config["n_offsets"], "decoder offset count")
    time_dim = _positive_int(config["time_dim"], "decoder time dimension")
    view = 3 if config["view_adaptive"] is True else 0
    app = _positive_int(config["app_embed_dim"], "appearance embedding dimension") if config["app_opt"] is True else 0
    opacity_dist = 1 if config["add_opacity_dist"] is True else 0
    cov_dist = 1 if config["add_cov_dist"] is True else 0
    color_dist = 1 if config["add_color_dist"] is True else 0

    def linear(prefix: str, hidden_input: int, output: int):
        return [
            (f"{prefix}.0.weight", (feature, hidden_input)),
            (f"{prefix}.0.bias", (feature,)),
            (f"{prefix}.2.weight", (output, feature)),
            (f"{prefix}.2.bias", (output,)),
        ]

    decoder = []
    decoder += linear(
        "mlp_opacity", feature + view + opacity_dist + channel, offsets
    )
    decoder += linear("mlp_cov", feature + view + cov_dist + channel, 7 * offsets)
    decoder += linear(
        "mlp_color", feature + view + color_dist + app + channel, 3 * offsets
    )
    decoder += linear("mlp_motion", feature + time_dim + channel, 7)

    def entropy_state(hidden: int, input_width: int, output_width: int):
        return [
            ("model.0.weight", (hidden, input_width)),
            ("model.0.bias", (hidden,)),
            ("model.2.weight", (output_width, hidden)),
            ("model.2.bias", (output_width,)),
        ]

    result = {
        "decoders": decoder,
        "scales_entropy_model": entropy_state(8, feature, 18),
        "anchor_features_entropy_model": entropy_state(
            12, 3 * entropy, 3 * entropy
        ),
        "offsets_entropy_model": entropy_state(16, feature, 9 * offsets),
        "factors_entropy_model": entropy_state(8, feature, 12),
        "time_features_entropy_model": entropy_state(
            12, 3 * channel, 3 * channel
        ),
    }
    if config["app_opt"] is True:
        result["app_module"] = [
            (
                "embeds.weight",
                (
                    _positive_int(
                        config["appearance_embedding_count"],
                        "appearance embedding count",
                    ),
                    _positive_int(
                        config["app_embed_dim"], "appearance embedding dimension"
                    ),
                ),
            )
        ]
    return result


def _expected_state_metadata(role: str) -> Dict[str, Any]:
    if role == "decoders":
        names = [""]
        for prefix, activation in (
            ("mlp_opacity", True),
            ("mlp_cov", False),
            ("mlp_color", True),
            ("mlp_motion", False),
        ):
            names.extend(
                [prefix, f"{prefix}.0", f"{prefix}.1", f"{prefix}.2"]
            )
            if activation:
                names.append(f"{prefix}.3")
    elif role == "app_module":
        names = ["", "embeds"]
    else:
        names = ["", "model", "model.0", "model.1", "model.2"]
    return {"_metadata": {name: {"version": 1} for name in names}}


def _expected_codec_scaling(rate: int) -> Dict[str, Any]:
    if type(rate) is not int or rate not in FROZEN_RATE_LAMBDAS:
        raise ValueError("codec scaling rate is outside the registered rate grid")
    scale = (0.02, 0.04, 0.06, 0.08, 0.036, 0.038)[rate]
    feature = (1, 1, 1.5, 2, 1, 1)[rate]
    return {
        "anchors": None,
        "scales": scale,
        "quats": None,
        "opacities": None,
        "anchor_features": feature,
        "offsets": scale,
        "factors": 0.0625,
        "time_features": feature,
    }


def _validate_nets_audit(
    audit: Mapping[str, Any],
    config: Mapping[str, Any],
    producer_state: Mapping[str, Any],
) -> None:
    expected_schema = _expected_nets_tensor_schema(config)
    actual_schema = audit.get("state_schema")
    actual_hashes = audit.get("state_sha256")
    if (
        not isinstance(actual_schema, dict)
        or set(actual_schema) != set(expected_schema)
        or not isinstance(actual_hashes, dict)
        or set(actual_hashes) != set(expected_schema)
    ):
        raise ValueError("counted nets.pt state roles are incomplete")
    for role, expected_rows in expected_schema.items():
        actual_rows = actual_schema[role]
        expected = [
            {
                "name": name,
                "dtype": "torch.float32",
                "shape": list(shape),
                "stride": list(_contiguous_stride(shape)),
            }
            for name, shape in sorted(expected_rows)
        ]
        if not isinstance(actual_rows, list) or len(actual_rows) != len(expected):
            raise ValueError(f"counted nets.pt state tensor count differs: {role}")
        for actual, wanted in zip(actual_rows, expected):
            if (
                not isinstance(actual, dict)
                or {key: actual.get(key) for key in wanted} != wanted
                or type(actual.get("storage_key")) is not int
            ):
                raise ValueError(f"counted nets.pt tensor schema differs: {role}")

    expected_hashes = {
        "decoders": producer_state["decoders"],
        **{
            f"{name}_entropy_model": digest
            for name, digest in producer_state["entropy_models"].items()
        },
    }
    if config["app_opt"] is True:
        expected_hashes["app_module"] = producer_state["appearance_module"]
    if actual_hashes != expected_hashes:
        raise ValueError("counted nets.pt tensor bytes differ from frozen training state")

    expected_scaling = _expected_codec_scaling(config["rate"])
    actual_scaling = audit.get("scaling")
    if type(actual_scaling) is not dict or list(actual_scaling) != list(expected_scaling):
        raise ValueError("counted nets.pt codec scaling keys/order differ")
    for name, expected_value in expected_scaling.items():
        actual_value = actual_scaling[name]
        if type(actual_value) is not type(expected_value) or actual_value != expected_value:
            raise ValueError(f"counted nets.pt codec scaling differs: {name}")
    if audit.get("scaling_sha256") != producer_state["codec_scaling"]:
        raise ValueError("counted nets.pt codec scaling differs from frozen training state")


def _reject_duplicate_object_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _strict_canonical_json(payload: bytes, label: str) -> Dict[str, Any]:
    """Parse one counted structural JSON with one byte-level representation."""

    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_object_pairs
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    if canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} bytes are not canonical JSON")
    return value


def _npy_payload_contract(payload: bytes, label: str) -> Dict[str, Any]:
    """Validate the exact NumPy v1 byte representation and reject ignored tails."""

    if len(payload) < 10 or payload[:6] != b"\x93NUMPY" or payload[6:8] != b"\x01\x00":
        raise ValueError(f"{label} is not canonical NumPy v1 data")
    header_length = struct.unpack("<H", payload[8:10])[0]
    if header_length <= 0 or header_length > 4096 or 10 + header_length > len(payload):
        raise ValueError(f"{label} NumPy header length is invalid")
    header = payload[10 : 10 + header_length]
    if not header.endswith(b"\n"):
        raise ValueError(f"{label} NumPy header lacks its terminal newline")
    try:
        header_text = header.decode("latin1")
        descriptor = ast.literal_eval(header_text.rstrip(" \n"))
    except (UnicodeDecodeError, SyntaxError, ValueError) as error:
        raise ValueError(f"{label} NumPy header is malformed") from error
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "descr",
        "fortran_order",
        "shape",
    }:
        raise ValueError(f"{label} NumPy header fields are unexpected")
    descr = descriptor["descr"]
    fortran_order = descriptor["fortran_order"]
    shape = descriptor["shape"]
    if (
        not isinstance(descr, str)
        or not isinstance(fortran_order, bool)
        or not isinstance(shape, tuple)
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape)
    ):
        raise ValueError(f"{label} NumPy descriptor is malformed")
    match = re.fullmatch(r"[<>=|]([?biufcSUVmM])(\d+)", descr)
    if match is None:
        raise ValueError(f"{label} NumPy dtype is outside the frozen scalar contract")
    kind, width_text = match.groups()
    width = int(width_text)
    if width <= 0:
        raise ValueError(f"{label} NumPy dtype width is invalid")
    itemsize = width * 4 if kind == "U" else width
    element_count = math.prod(shape) if shape else 1
    expected_size = 10 + header_length + element_count * itemsize
    if len(payload) != expected_size:
        raise ValueError(f"{label} NumPy payload has ignored or truncated bytes")
    shape_text = repr(shape)
    base = (
        "{'descr': "
        + repr(descr)
        + ", 'fortran_order': "
        + repr(fortran_order)
        + ", 'shape': "
        + shape_text
        + ", }"
    ).encode("latin1")
    canonical_header_length = ((10 + len(base) + 1 + 63) // 64) * 64 - 10
    canonical_header = base + b" " * (canonical_header_length - len(base) - 1) + b"\n"
    if header != canonical_header:
        raise ValueError(f"{label} NumPy header is not the canonical representation")
    return {
        "descr": descr,
        "fortran_order": fortran_order,
        "shape": list(shape),
        "data_bytes": element_count * itemsize,
    }


def _ap_numpy_sidecar_contract(
    payload: bytes,
    label: str,
    record: Mapping[str, Any],
    expected_shape: Sequence[int],
) -> Dict[str, Any]:
    """Bind one AP NPY sidecar to its exact int64 producer role."""

    audit = _npy_payload_contract(payload, label)
    shape = list(expected_shape)
    if (
        not isinstance(record, dict)
        or set(record) != {"path", "shape", "dtype", "bytes", "sha256"}
        or record.get("dtype") != "<i8"
        or not isinstance(record.get("shape"), list)
        or any(type(value) is not int for value in record["shape"])
        or record.get("shape") != shape
        or audit["descr"] != "<i8"
        or audit["fortran_order"] is not False
        or audit["shape"] != shape
    ):
        raise ValueError(f"{label} dtype/shape differs from its producer")
    return audit


def _camera_metadata_contract(
    handle: zipfile.ZipFile, expected_camera_names: Sequence[str]
) -> Dict[str, Any]:
    names = (
        "camera_keys",
        "intrinsics",
        "image_sizes",
        "camtoworlds",
        "camera_ids",
        "camera_names",
        "transform",
        "bounds",
    )
    payloads = {
        name: handle.read(f"camera_metadata/{name}.npy") for name in names
    }
    audits = {
        name: _npy_payload_contract(payload, f"camera_metadata/{name}.npy")
        for name, payload in payloads.items()
    }
    key_shape = audits["camera_keys"]["shape"]
    pose_shape = audits["camtoworlds"]["shape"]
    if len(key_shape) != 1 or key_shape[0] <= 0:
        raise ValueError("camera key array shape differs from its producer")
    if len(pose_shape) != 3 or pose_shape[0] <= 0 or pose_shape[1:] != [4, 4]:
        raise ValueError("camera pose array shape differs from its producer")
    key_count = key_shape[0]
    pose_count = pose_shape[0]
    frozen_names = tuple(expected_camera_names)
    expected_camera_count = len(frozen_names)
    if key_count != expected_camera_count or pose_count != expected_camera_count:
        raise ValueError("camera metadata count differs from the frozen camera grid")
    expected = {
        "camera_keys": ("<i8", [key_count]),
        "intrinsics": ("<f8", [key_count, 3, 3]),
        "image_sizes": ("<i8", [key_count, 2]),
        "camtoworlds": ("<f8", [pose_count, 4, 4]),
        "camera_ids": ("<i8", [pose_count]),
        "transform": ("<f8", [4, 4]),
    }
    for name, (descr, shape) in expected.items():
        if audits[name]["descr"] != descr or audits[name]["shape"] != shape:
            raise ValueError(f"camera metadata role differs from its producer: {name}")

    bounds_shape = audits["bounds"]["shape"]
    if (
        audits["bounds"]["descr"] != "<f8"
        or bounds_shape != [pose_count, 2]
    ):
        raise ValueError("camera bounds dtype/shape differs from its producer")

    names_audit = audits["camera_names"]
    match = re.fullmatch(r"<U([1-9][0-9]*)", names_audit["descr"])
    if match is None or names_audit["shape"] != [pose_count]:
        raise ValueError("camera-name dtype/shape differs from its producer")
    width = int(match.group(1))
    if width > 4096:
        raise ValueError("camera-name width exceeds the frozen producer bound")
    data = payloads["camera_names"][-names_audit["data_bytes"] :]
    required_width = 0
    decoded_names = []
    for row in range(pose_count):
        encoded = data[row * width * 4 : (row + 1) * width * 4]
        codepoints = struct.unpack(f"<{width}I", encoded)
        first_zero = next((index for index, value in enumerate(codepoints) if value == 0), width)
        if any(value != 0 for value in codepoints[first_zero:]):
            raise ValueError("camera-name Unicode padding is noncanonical")
        required_width = max(required_width, first_zero)
        decoded_names.append("".join(chr(value) for value in codepoints[:first_zero]))
    if required_width <= 0 or width != required_width:
        raise ValueError("camera-name Unicode width is not minimal")
    if tuple(decoded_names) != frozen_names:
        raise ValueError("camera names differ from the frozen camera grid")
    return {
        "camera_key_count": key_count,
        "camera_pose_count": pose_count,
        "camera_name_width": width,
        "bounds_shape": bounds_shape,
    }


def _npz_payload_contract(
    payload: bytes,
    label: str,
    expected_members: Optional[set] = None,
    expected_compression: Optional[int] = None,
    expected_order: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Validate an exact np.savez/np.savez_compressed archive byte-for-byte."""

    try:
        handle = zipfile.ZipFile(io.BytesIO(payload), "r")
    except zipfile.BadZipFile as error:
        raise ValueError(f"{label} is not a NumPy ZIP archive") from error
    try:
        if handle.comment:
            raise ValueError(f"{label} NumPy ZIP comment is forbidden")
        infos = handle.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or any(
            not name.endswith(".npy") or "/" in name or "\\" in name for name in names
        ):
            raise ValueError(f"{label} NumPy ZIP member names are unsafe or duplicated")
        if expected_members is not None and set(names) != set(expected_members):
            raise ValueError(f"{label} NumPy ZIP members differ from the exact schema")
        if expected_order is not None and names != list(expected_order):
            raise ValueError(f"{label} NumPy ZIP member order differs from its producer")
        if not names:
            raise ValueError(f"{label} NumPy ZIP is empty")
        compression_types = {info.compress_type for info in infos}
        if not compression_types.issubset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}) or len(
            compression_types
        ) != 1:
            raise ValueError(f"{label} NumPy ZIP compression is noncanonical")
        if expected_compression is not None and compression_types != {
            expected_compression
        }:
            raise ValueError(f"{label} NumPy ZIP compression differs from its producer")
        members = {}
        member_audits = {}
        for info in infos:
            if (
                info.date_time != ZIP_EPOCH
                or info.create_system != 3
                or info.external_attr != 0o600 << 16
                or info.extra
                or info.comment
            ):
                raise ValueError(f"{label} NumPy ZIP metadata is noncanonical")
            member = handle.read(info)
            member_audits[info.filename] = _npy_payload_contract(
                member, f"{label}/{info.filename}"
            )
            members[info.filename] = member
        rebuilt = io.BytesIO()
        compression = next(iter(compression_types))
        with zipfile.ZipFile(
            rebuilt, "w", compression=compression, allowZip64=True
        ) as output:
            for name in names:
                with output.open(name, "w", force_zip64=True) as destination:
                    destination.write(members[name])
        if rebuilt.getvalue() != payload:
            raise ValueError(f"{label} NumPy ZIP bytes are not canonical")
        return {
            "members": names,
            "compression": compression,
            "member_audits": member_audits,
        }
    finally:
        handle.close()


def _codec_npz_payload_contract(
    payload: bytes, label: str, metadata: Mapping[str, Any]
) -> Dict[str, Any]:
    audit = _npz_payload_contract(
        payload,
        label,
        {"arr.npy"},
        expected_compression=zipfile.ZIP_DEFLATED,
        expected_order=("arr.npy",),
    )
    array_audit = audit["member_audits"]["arr.npy"]
    if (
        metadata.get("dtype") != "float32"
        or array_audit["descr"] != "<f4"
        or array_audit["fortran_order"] is not False
        or array_audit["shape"] != metadata.get("shape")
    ):
        raise ValueError(f"{label} array dtype/shape differs from counted metadata")
    return audit


def _unicode_scalar_descr(value: str) -> str:
    return f"<U{max(1, len(str(value)))}"


def _npz_member_schema_contract(
    payload: bytes,
    label: str,
    expected_schema: Mapping[str, Tuple[str, Sequence[int]]],
    expected_order: Sequence[str],
) -> Dict[str, Any]:
    audit = _npz_payload_contract(
        payload,
        label,
        set(expected_schema),
        expected_compression=zipfile.ZIP_STORED,
        expected_order=expected_order,
    )
    for name, (descr, shape) in expected_schema.items():
        member = audit["member_audits"][name]
        if (
            member["descr"] != descr
            or member["fortran_order"] is not False
            or member["shape"] != list(shape)
        ):
            raise ValueError(f"{label}/{name} dtype/shape differs from its producer")
    return audit


def _ap_score_npz_contract(
    payload: bytes,
    label: str,
    *,
    anchor_count: Optional[int],
    scene: str,
    variant: str,
) -> Dict[str, Any]:
    if anchor_count is None:
        preliminary = _npz_payload_contract(
            payload,
            label,
            AP_SCORE_NPZ_MEMBERS,
            expected_compression=zipfile.ZIP_STORED,
            expected_order=AP_SCORE_NPZ_ORDER,
        )
        canonical_ids = preliminary["member_audits"]["canonical_ids.npy"]
        shape = canonical_ids["shape"]
        if canonical_ids["descr"] != "<i8" or len(shape) != 2 or shape[1:] != [3]:
            raise ValueError(f"{label} canonical-ID role is malformed")
        anchor_count = shape[0]
    count = _positive_int(anchor_count, f"{label} anchor count")
    scalar_f8 = ("<f8", ())
    scalar_i8 = ("<i8", ())
    scalar_bool = ("|b1", ())
    vector_f8 = ("<f8", (count,))
    vector_i8 = ("<i8", (count,))
    vector_bool = ("|b1", (count,))
    schema: Dict[str, Tuple[str, Sequence[int]]] = {
        "schema.npy": (_unicode_scalar_descr(AP_SCORE_SCHEMA), ()),
        "scene.npy": (_unicode_scalar_descr(scene), ()),
        "voxel_size.npy": scalar_f8,
        "frame_count.npy": scalar_i8,
        "variant.npy": (_unicode_scalar_descr(variant), ()),
        "protected_fraction.npy": scalar_f8,
        "q_ap_multiplier.npy": scalar_f8,
        "q_bg_multiplier.npy": scalar_f8,
        "random_seed.npy": scalar_i8,
        "canonical_ids.npy": ("<i8", (count, 3)),
        "eligible.npy": vector_bool,
        "path_score.npy": vector_f8,
        "motion_score.npy": vector_f8,
        "allocation_score.npy": vector_f8,
        "importance_score.npy": vector_f8,
        "estimated_time_bytes.npy": vector_i8,
        "official_retain_mask.npy": vector_bool,
        "official_factor0_mask.npy": vector_bool,
        "official_active_mask.npy": vector_bool,
        "ap_retain_mask.npy": vector_bool,
        "ap_active_mask.npy": vector_bool,
        "ap_class_mask.npy": vector_bool,
        "factor0_activation_value.npy": scalar_f8,
        "factor3_activation_value.npy": scalar_f8,
        "estimator_version.npy": (_unicode_scalar_descr(AP_ESTIMATOR_VERSION), ()),
        "time_entropy_model_sha256.npy": ("<U64", ()),
        "time_feature_scaling.npy": scalar_f8,
        "time_entropy_model_frozen_after_freeze.npy": scalar_bool,
        "runtime_manifest_sha256.npy": ("<U64", ()),
        "normalized_code_tree_sha256.npy": ("<U64", ()),
        "patch_chain_sha256.npy": ("<U64", (9,)),
        "path_definition.npy": (_unicode_scalar_descr(AP_PATH_DEFINITION), ()),
        "motion_definition.npy": (_unicode_scalar_descr(AP_MOTION_DEFINITION), ()),
        "importance_definition.npy": (
            _unicode_scalar_descr(AP_IMPORTANCE_DEFINITION),
            (),
        ),
        "estimated_byte_definition.npy": (
            _unicode_scalar_descr(AP_ESTIMATED_BYTE_DEFINITION),
            (),
        ),
    }
    return _npz_member_schema_contract(
        payload, label, schema, AP_SCORE_NPZ_ORDER
    )


def _ap_edit_ids_npz_contract(
    payload: bytes, label: str, *, edit_count: int, scene: str
) -> Dict[str, Any]:
    count = _positive_int(edit_count, f"{label} edit count")
    schema: Dict[str, Tuple[str, Sequence[int]]] = {
        "schema.npy": (_unicode_scalar_descr(AP_EDIT_IDS_SCHEMA), ()),
        "scene.npy": (_unicode_scalar_descr(scene), ()),
        "voxel_size.npy": ("<f8", ()),
        "canonical_ids.npy": ("<i8", (count, 3)),
        "source_score_sha256.npy": ("<U64", ()),
        "selection.npy": (_unicode_scalar_descr(AP_EDIT_SELECTION), ()),
        "reference_manifest_sha256.npy": ("<U64", ()),
        "selected_canonical_ids_sha256.npy": ("<U64", ()),
        "path_score.npy": ("<f8", (count,)),
    }
    return _npz_member_schema_contract(
        payload, label, schema, AP_EDIT_IDS_NPZ_ORDER
    )


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", binascii.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def _paeth_predictor(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def _canonical_png_payload(payload: bytes, label: str) -> Tuple[bytes, Dict[str, Any]]:
    """Decode PNG filtering and rebuild one exact filter-0/zlib-9 representation."""

    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"{label} is not PNG")
    position = 8
    chunks = []
    idat = []
    ihdr = None
    while position < len(payload):
        if position + 12 > len(payload):
            raise ValueError(f"{label} PNG chunk is truncated")
        length = struct.unpack(">I", payload[position : position + 4])[0]
        chunk_type = payload[position + 4 : position + 8]
        end = position + 12 + length
        if end > len(payload):
            raise ValueError(f"{label} PNG chunk length escapes the file")
        data = payload[position + 8 : position + 8 + length]
        expected_crc = struct.unpack(">I", payload[position + 8 + length : end])[0]
        if binascii.crc32(chunk_type + data) & 0xFFFFFFFF != expected_crc:
            raise ValueError(f"{label} PNG chunk CRC mismatch")
        chunks.append(chunk_type)
        if chunk_type == b"IHDR":
            ihdr = data
        elif chunk_type == b"IDAT":
            if not data:
                raise ValueError(f"{label} PNG has an empty padding IDAT chunk")
            idat.append(data)
        elif chunk_type != b"IEND":
            raise ValueError(f"{label} PNG ancillary/unknown chunks are forbidden")
        position = end
        if chunk_type == b"IEND":
            break
    if position != len(payload) or not chunks or chunks[0] != b"IHDR" or chunks[-1] != b"IEND":
        raise ValueError(f"{label} PNG has trailing bytes or invalid chunk order")
    if chunks.count(b"IHDR") != 1 or chunks.count(b"IEND") != 1 or not idat:
        raise ValueError(f"{label} PNG chunk closure is incomplete")
    if any(chunk not in {b"IHDR", b"IDAT", b"IEND"} for chunk in chunks):
        raise ValueError(f"{label} PNG chunk set is noncanonical")
    if ihdr is None or len(ihdr) != 13:
        raise ValueError(f"{label} PNG IHDR is malformed")
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", ihdr
    )
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if (
        width <= 0
        or height <= 0
        or channels is None
        or bit_depth not in {8, 16}
        or compression != 0
        or filtering != 0
        or interlace != 0
    ):
        raise ValueError(f"{label} PNG IHDR is outside the frozen noninterlaced contract")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(b"".join(idat)) + decoder.flush()
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError(f"{label} PNG deflate stream has ignored bytes")
    row_bytes = (width * channels * bit_depth + 7) // 8
    if len(raw) != height * (row_bytes + 1):
        raise ValueError(f"{label} PNG decompressed scanline size is inconsistent")
    bytes_per_pixel = max(1, (channels * bit_depth + 7) // 8)
    canonical_scanlines = bytearray()
    previous = bytearray(row_bytes)
    position = 0
    for row_index in range(height):
        filter_type = raw[position]
        if filter_type not in range(5):
            raise ValueError(f"{label} PNG row filter is invalid: {row_index}")
        filtered = raw[position + 1 : position + 1 + row_bytes]
        reconstructed = bytearray(row_bytes)
        for index, value in enumerate(filtered):
            left = reconstructed[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            above = previous[index]
            upper_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            else:
                predictor = _paeth_predictor(left, above, upper_left)
            reconstructed[index] = (value + predictor) & 0xFF
        canonical_scanlines.append(0)
        canonical_scanlines.extend(reconstructed)
        previous = reconstructed
        position += row_bytes + 1
    canonical_stream = zlib.compress(bytes(canonical_scanlines), 9)
    canonical = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", canonical_stream)
        + _png_chunk(b"IEND", b"")
    )
    return canonical, {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
        "canonical_filter": 0,
        "canonical_zlib_level": 9,
        "canonical_idat_chunks": 1,
    }


def _anchor_png_geometry(meta: Mapping[str, Any]) -> Dict[str, int]:
    anchor_meta = meta.get("anchors") if isinstance(meta, Mapping) else None
    shape = anchor_meta.get("shape") if isinstance(anchor_meta, Mapping) else None
    if (
        not isinstance(shape, list)
        or len(shape) < 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in shape
        )
    ):
        raise ValueError("GIFStream anchor shape cannot bind its PNG geometry")
    side = math.isqrt(shape[0])
    channels = math.prod(shape[1:])
    color_type = {1: 0, 2: 4, 3: 2, 4: 6}.get(channels)
    if side * side != shape[0] or color_type is None:
        raise ValueError("GIFStream anchor shape is outside the frozen PNG layout")
    return {
        "width": side,
        "height": side,
        "bit_depth": 8,
        "color_type": color_type,
    }


def _png_payload_contract(
    payload: bytes,
    label: str,
    expected_geometry: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    """Require one canonical PNG byte representation for the decoded pixels."""

    canonical, audit = _canonical_png_payload(payload, label)
    if payload != canonical:
        raise ValueError(f"{label} PNG bytes are not the canonical pixel representation")
    if expected_geometry is not None and any(
        audit.get(key) != value for key, value in expected_geometry.items()
    ):
        raise ValueError(f"{label} PNG geometry/type differs from its producer metadata")
    return audit


def canonicalize_gifstream_png_payloads(root: Path) -> List[Dict[str, Any]]:
    """Atomically normalize codec PNGs before any payload hash or decode is recorded."""

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("GIFStream PNG payload root is unavailable or a symlink")
    meta_path = root / "meta.json"
    if meta_path.is_symlink() or not meta_path.is_file():
        raise ValueError("GIFStream metadata cannot bind its PNG payloads")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("GIFStream metadata cannot bind its PNG payloads") from error
    expected_geometry = _anchor_png_geometry(meta)
    png_paths = sorted(root.glob("*.png"))
    if {path.name for path in png_paths} != {"anchors_l.png", "anchors_u.png"}:
        raise ValueError("GIFStream anchor PNG member set is not exact")
    rows = []
    for path in png_paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"GIFStream PNG payload is unavailable or a symlink: {path.name}")
        original = path.read_bytes()
        canonical, audit = _canonical_png_payload(original, path.name)
        if any(audit.get(key) != value for key, value in expected_geometry.items()):
            raise ValueError(
                f"{path.name} PNG geometry/type differs from its producer metadata"
            )
        if original != canonical:
            staging = path.with_name(path.name + ".h007-canonical.staging")
            if staging.exists() or staging.is_symlink():
                raise ValueError(f"GIFStream PNG canonical staging already exists: {path.name}")
            with staging.open("xb") as handle:
                handle.write(canonical)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(staging, path)
        _png_payload_contract(path.read_bytes(), path.name, expected_geometry)
        rows.append(
            {
                "path": path.name,
                "source_sha256": sha256_bytes(original),
                "canonical_sha256": sha256_file(path),
                "source_bytes": len(original),
                "canonical_bytes": path.stat().st_size,
                **audit,
            }
        )
    return rows


def _entropy_stream_contract(payload: bytes, label: str) -> None:
    if len(payload) < 4:
        raise ValueError(f"{label} entropy stream is truncated")
    declared = struct.unpack(">I", payload[:4])[0]
    if declared < 8 or declared % 4 or len(payload) != 4 + declared:
        raise ValueError(f"{label} entropy stream has ignored or truncated bytes")


def _packed_bool_mask_contract(
    payload: bytes, count: Any, true_count: Any, label: str
) -> None:
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or isinstance(true_count, bool)
        or not isinstance(true_count, int)
        or true_count < 0
        or true_count > count
    ):
        raise ValueError(f"{label} packed-mask dimensions are invalid")
    expected_bytes = (count + 7) // 8
    if len(payload) != expected_bytes:
        raise ValueError(f"{label} packed-mask byte extent is noncanonical")
    if count % 8 and payload and payload[-1] & ~((1 << (count % 8)) - 1):
        raise ValueError(f"{label} packed-mask tail bits are nonzero")
    if sum(bin(value).count("1") for value in payload) != true_count:
        raise ValueError(f"{label} packed-mask true count differs from its bytes")


def _identity_correction_payload_contract(
    payload: bytes, record: Mapping[str, Any], label: str
) -> None:
    required = {
        "schema",
        "path",
        "row_count",
        "mismatch_count",
        "mask_bytes",
        "bytes",
        "sha256",
        "bitorder",
        "base",
        "code",
    }
    if (
        not isinstance(record, dict)
        or set(record) != required
        or record.get("schema") != "h007.ap_identity_corrections.v1"
        or record.get("path") != "ap_identity_corrections.bin"
        or record.get("bitorder") != "little"
        or record.get("base") != "round-decoded-anchor-div-voxel-size"
        or record.get("code") != "uint8-base3-dx-dy-dz-plus1"
    ):
        raise ValueError(f"{label} metadata are malformed")
    rows = _positive_int(record.get("row_count"), f"{label} row count")
    mismatches = _nonnegative_int(
        record.get("mismatch_count"), f"{label} mismatch count"
    )
    mask_bytes = _positive_int(record.get("mask_bytes"), f"{label} mask bytes")
    if (
        mismatches > rows
        or mask_bytes != (rows + 7) // 8
        or record.get("bytes") != len(payload)
        or len(payload) != mask_bytes + mismatches
        or sha256_bytes(payload) != record.get("sha256")
    ):
        raise ValueError(f"{label} byte extent/binding is invalid")
    _packed_bool_mask_contract(
        payload[:mask_bytes], rows, mismatches, f"{label} mismatch mask"
    )
    if any(value > 26 or value == 13 for value in payload[mask_bytes:]):
        raise ValueError(f"{label} contains an invalid base-3 correction code")


def _canonical_torch_zip_bytes(root: str, members: Mapping[str, bytes]) -> bytes:
    fixed_prefix = [
        f"{root}/data.pkl",
        f"{root}/.format_version",
        f"{root}/.storage_alignment",
        f"{root}/byteorder",
    ]
    data_prefix = f"{root}/data/"
    data_names = sorted(
        (name for name in members if name.startswith(data_prefix)),
        key=lambda name: int(name[len(data_prefix) :]),
    )
    fixed_suffix = [f"{root}/version", f"{root}/.data/serialization_id"]
    ordered = [name for name in fixed_prefix if name in members] + data_names + [
        name for name in fixed_suffix if name in members
    ]
    if set(ordered) != set(members) or len(ordered) > 65535:
        raise ValueError("torch ZIP member order cannot be canonicalized")
    output = bytearray()
    central = []
    for name in ordered:
        encoded = name.encode("utf-8")
        member = members[name]
        offset = len(output)
        padding = (-(offset + 30 + len(encoded) + 4)) % 64
        extra = b"FB" + struct.pack("<H", padding) + b"Z" * padding
        crc = binascii.crc32(member) & 0xFFFFFFFF
        if max(offset, len(member), len(encoded), len(extra)) >= 2**32:
            raise ValueError("torch ZIP exceeds the frozen non-ZIP64 contract")
        output.extend(
            struct.pack(
                "<IHHHHHIIIHH",
                0x04034B50,
                20,
                0,
                0,
                0,
                0,
                crc,
                len(member),
                len(member),
                len(encoded),
                len(extra),
            )
        )
        output.extend(encoded)
        output.extend(extra)
        output.extend(member)
        central.append((encoded, member, crc, offset))
    central_offset = len(output)
    for encoded, member, crc, offset in central:
        output.extend(
            struct.pack(
                "<IHHHHHHIIIHHHHHII",
                0x02014B50,
                0x0314,
                20,
                0,
                0,
                0,
                0,
                crc,
                len(member),
                len(member),
                len(encoded),
                0,
                0,
                0,
                0,
                0,
                offset,
            )
        )
        output.extend(encoded)
    central_size = len(output) - central_offset
    if max(central_offset, central_size, len(output)) >= 2**32:
        raise ValueError("torch ZIP central directory exceeds the frozen contract")
    output.extend(
        struct.pack(
            "<IHHHHIIH",
            0x06054B50,
            0,
            0,
            len(central),
            len(central),
            central_size,
            central_offset,
            0,
        )
    )
    return bytes(output)


def _canonical_pickle_opcode_contract(payload: bytes, label: str) -> None:
    """Require the shortest frozen protocol-2 representation for scalar opcodes."""

    operations = list(pickletools.genops(payload))
    allowed = {
        "PROTO",
        "STOP",
        "EMPTY_DICT",
        "MARK",
        "BINUNICODE",
        "BINPUT",
        "LONG_BINPUT",
        "BINGET",
        "LONG_BINGET",
        "GLOBAL",
        "BININT1",
        "BININT2",
        "BININT",
        "LONG1",
        "LONG4",
        "NONE",
        "BINFLOAT",
        "REDUCE",
        "EMPTY_TUPLE",
        "TUPLE",
        "TUPLE1",
        "TUPLE2",
        "TUPLE3",
        "BINPERSID",
        "NEWFALSE",
        "NEWTRUE",
        "SETITEM",
        "SETITEMS",
        "BUILD",
    }
    if (
        not operations
        or operations[0][0].name != "PROTO"
        or operations[0][1] != 2
        or sum(operation.name == "PROTO" for operation, _, _ in operations) != 1
        or any(operation.name not in allowed for operation, _, _ in operations)
    ):
        raise ValueError(f"{label} torch pickle opcode set/protocol is noncanonical")

    def canonical_integer(value: int) -> bytes:
        if 0 <= value <= 0xFF:
            return pickle.BININT1 + bytes([value])
        if 0 <= value <= 0xFFFF:
            return pickle.BININT2 + struct.pack("<H", value)
        if -(2**31) <= value < 2**31:
            return pickle.BININT + struct.pack("<i", value)
        encoded = pickle.encode_long(value)
        if len(encoded) <= 0xFF:
            return pickle.LONG1 + bytes([len(encoded)]) + encoded
        return pickle.LONG4 + struct.pack("<I", len(encoded)) + encoded

    for index, (operation, argument, position) in enumerate(operations):
        end = operations[index + 1][2] if index + 1 < len(operations) else len(payload)
        encoded = payload[position:end]
        if operation.name in {"BININT1", "BININT2", "BININT", "LONG1", "LONG4"}:
            if encoded != canonical_integer(argument):
                raise ValueError(f"{label} torch pickle integer encoding is nonminimal")
        elif operation.name in {"BINPUT", "LONG_BINPUT"}:
            expected = (
                pickle.BINPUT + bytes([argument])
                if 0 <= argument <= 0xFF
                else pickle.LONG_BINPUT + struct.pack("<I", argument)
            )
            if encoded != expected:
                raise ValueError(f"{label} torch pickle memo definition is nonminimal")
        elif operation.name in {"BINGET", "LONG_BINGET"}:
            expected = (
                pickle.BINGET + bytes([argument])
                if 0 <= argument <= 0xFF
                else pickle.LONG_BINGET + struct.pack("<I", argument)
            )
            if encoded != expected:
                raise ValueError(f"{label} torch pickle memo reference is nonminimal")


def _canonical_torch_save_payload(
    payload: bytes, label: str, expected_app_opt: Optional[bool] = None
) -> Tuple[bytes, Dict[str, Any]]:
    """Validate torch storage semantics and rebuild one exact ZIP/pickle form."""

    eocd = payload.rfind(b"PK\x05\x06")
    if eocd < 0 or eocd + 22 != len(payload) or payload[eocd + 20 : eocd + 22] != b"\0\0":
        raise ValueError(f"{label} torch ZIP has a comment or ignored trailing bytes")
    try:
        handle = zipfile.ZipFile(io.BytesIO(payload), "r")
    except zipfile.BadZipFile as error:
        raise ValueError(f"{label} is not a torch.save ZIP") from error
    try:
        if handle.comment:
            raise ValueError(f"{label} torch ZIP comment is forbidden")
        infos = handle.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or not names:
            raise ValueError(f"{label} torch ZIP members are empty or duplicated")
        roots = {name.split("/", 1)[0] for name in names if "/" in name}
        if roots != {"nets"}:
            raise ValueError(f"{label} torch ZIP root differs from the frozen nets.pt producer")
        root = "nets"
        required = {
            f"{root}/data.pkl",
            f"{root}/byteorder",
            f"{root}/version",
            f"{root}/.data/serialization_id",
        }
        if not required.issubset(names):
            raise ValueError(f"{label} torch ZIP lacks its pickle/version closure")
        allowed_fixed = required
        data_indices = []
        members = {}
        for info in infos:
            name = info.filename
            if (
                info.comment
                or info.extra
                or info.compress_type != zipfile.ZIP_STORED
                or "\\" in name
                or ".." in Path(name).parts
            ):
                raise ValueError(f"{label} torch ZIP member metadata is noncanonical")
            if name not in allowed_fixed:
                prefix = f"{root}/data/"
                suffix = name[len(prefix) :] if name.startswith(prefix) else ""
                if not suffix.isdigit() or str(int(suffix)) != suffix:
                    raise ValueError(f"{label} torch ZIP has an unmanaged record")
                data_indices.append(int(suffix))
            members[name] = handle.read(info)
        if not data_indices or sorted(data_indices) != list(range(max(data_indices) + 1)):
            raise ValueError(f"{label} torch ZIP storage records are not contiguous")

        if members[f"{root}/version"] != b"3\n":
            raise ValueError(f"{label} torch ZIP version differs from the frozen producer")
        if members[f"{root}/byteorder"] != b"little":
            raise ValueError(f"{label} torch ZIP byteorder differs from the frozen producer")
        if not re.fullmatch(
            rb"[0-9]{40}", members[f"{root}/.data/serialization_id"]
        ):
            raise ValueError(f"{label} torch ZIP serialization ID width is noncanonical")

        alignment = 64
        for info in infos:
            offset = info.header_offset
            if payload[offset : offset + 4] != b"PK\x03\x04" or offset + 30 > len(payload):
                raise ValueError(f"{label} torch ZIP local header is malformed")
            filename_length, extra_length = struct.unpack(
                "<HH", payload[offset + 26 : offset + 30]
            )
            filename_start = offset + 30
            filename_end = filename_start + filename_length
            extra_end = filename_end + extra_length
            if (
                payload[filename_start:filename_end] != info.filename.encode("utf-8")
                or extra_end > len(payload)
            ):
                raise ValueError(f"{label} torch ZIP local member name is malformed")
            padding_length = (-(offset + 30 + filename_length + 4)) % alignment
            expected_extra = b"FB" + struct.pack("<H", padding_length) + b"Z" * padding_length
            if payload[filename_end:extra_end] != expected_extra:
                raise ValueError(f"{label} torch ZIP local alignment extra is noncanonical")

        pickle_payload = members[f"{root}/data.pkl"]
        if not pickle_payload or len(pickle_payload) > 32 * 1024 * 1024:
            raise ValueError(f"{label} torch pickle size is outside the frozen producer contract")
        try:
            operations = list(pickletools.genops(pickle_payload))
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError(f"{label} torch pickle is malformed") from error
        if not operations or operations[-1][0].name != "STOP" or operations[-1][2] + 1 != len(
            pickle_payload
        ):
            raise ValueError(f"{label} torch pickle has ignored trailing bytes")
        if any(operation.name in {"POP", "POP_MARK", "DUP"} for operation, _, _ in operations):
            raise ValueError(f"{label} torch pickle contains a semantic no-op channel")
        pickle_payload = pickletools.optimize(pickle_payload)
        _canonical_pickle_opcode_contract(pickle_payload, label)
        members[f"{root}/data.pkl"] = pickle_payload

        storage_expectations: Dict[int, Any] = {}
        storage_uses: Dict[int, int] = {}

        class _StorageTypeAudit:
            def __init__(self, name: str) -> None:
                self.name = name
                self.width = 4

        class _StorageAudit:
            def __init__(
                self, key: int, location: str, numel: int, storage_type: str
            ) -> None:
                self.key = key
                self.location = location
                self.numel = numel
                self.storage_type = storage_type
                self.width = 4

        class _OrderedStateAudit(dict):
            def __init__(self) -> None:
                super().__init__()
                self.state = None

            def __setstate__(self, state: Any) -> None:
                if self.state is not None:
                    raise ValueError("torch OrderedDict state is assigned more than once")
                self.state = state

        class _TensorAudit:
            def __init__(
                self, storage: Any, shape: Tuple[int, ...], stride: Tuple[int, ...]
            ) -> None:
                self.storage = storage
                self.shape = shape
                self.stride = stride

        def _rebuild_tensor_v2(
            storage: Any,
            storage_offset: Any,
            shape: Any,
            stride: Any,
            requires_grad: Any,
            backward_hooks: Any,
            metadata: Any = None,
        ) -> Any:
            if not isinstance(storage, _StorageAudit):
                raise ValueError("torch tensor does not reference an audited storage")
            if type(storage_offset) is not int or storage_offset != 0:
                raise ValueError("torch tensor storage offset is not canonical zero")
            if (
                not isinstance(shape, tuple)
                or not shape
                or any(type(value) is not int or value <= 0 for value in shape)
                or not isinstance(stride, tuple)
                or any(type(value) is not int or value <= 0 for value in stride)
                or len(stride) != len(shape)
                or stride != _contiguous_stride(shape)
            ):
                raise ValueError("torch tensor shape/stride is not exact contiguous form")
            if requires_grad is not False:
                raise ValueError("torch state tensor unexpectedly requires gradients")
            if (
                not isinstance(backward_hooks, _OrderedStateAudit)
                or backward_hooks
                or backward_hooks.state is not None
                or metadata is not None
            ):
                raise ValueError("torch tensor rebuild metadata is not canonical")
            numel = math.prod(shape)
            if numel != storage.numel:
                raise ValueError("torch tensor shape differs from its complete storage")
            storage_uses[storage.key] = storage_uses.get(storage.key, 0) + 1
            if storage_uses[storage.key] != 1:
                raise ValueError("torch storage alias/view sharing is forbidden")
            return _TensorAudit(storage, shape, stride)


        class _StorageAuditUnpickler(pickle._Unpickler):
            dispatch = pickle._Unpickler.dispatch.copy()

            def find_class(self, module: str, name: str) -> Any:
                if (module, name) == ("collections", "OrderedDict"):
                    return _OrderedStateAudit
                if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                    return _rebuild_tensor_v2
                if module in {"torch", "torch.storage"} and name == "FloatStorage":
                    return _StorageTypeAudit(name)
                raise ValueError(f"unsupported torch pickle global: {module}.{name}")

            def persistent_load(self, persistent_id: Any) -> Any:
                if (
                    not isinstance(persistent_id, tuple)
                    or len(persistent_id) != 5
                    or persistent_id[0] != "storage"
                    or not isinstance(persistent_id[1], _StorageTypeAudit)
                    or not isinstance(persistent_id[2], str)
                    or not persistent_id[2].isdigit()
                    or str(int(persistent_id[2])) != persistent_id[2]
                    or persistent_id[3] != "cuda:0"
                    or isinstance(persistent_id[4], bool)
                    or not isinstance(persistent_id[4], int)
                    or persistent_id[4] <= 0
                ):
                    raise ValueError("unsupported torch persistent ID")
                key = int(persistent_id[2])
                storage = _StorageAudit(
                    key,
                    persistent_id[3],
                    persistent_id[4],
                    persistent_id[1].name,
                )
                if key in storage_expectations:
                    raise ValueError("torch storage key is referenced more than once")
                storage_expectations[key] = storage
                return storage

            @staticmethod
            def _assign_unique(target: Any, key: Any, value: Any) -> None:
                if isinstance(target, dict):
                    if key in target:
                        raise ValueError("torch dictionary key is assigned more than once")
                    target[key] = value
                    return
                target[key] = value

            def load_setitem(self) -> None:
                value = self.stack.pop()
                key = self.stack.pop()
                self._assign_unique(self.stack[-1], key, value)

            dispatch[pickle.SETITEM[0]] = load_setitem

            def load_setitems(self) -> None:
                items = self.pop_mark()
                if len(items) < 4 or len(items) % 2:
                    raise ValueError("torch SETITEMS batch is noncanonical")
                target = self.stack[-1]
                for index in range(0, len(items), 2):
                    self._assign_unique(target, items[index], items[index + 1])

            dispatch[pickle.SETITEMS[0]] = load_setitems

            def load_tuple(self) -> None:
                items = self.pop_mark()
                if len(items) < 4:
                    raise ValueError("torch tuple uses a nonminimal opcode")
                self.append(tuple(items))

            dispatch[pickle.TUPLE[0]] = load_tuple

            def load_stop(self) -> None:
                if len(self.stack) != 1 or self.metastack:
                    raise ValueError("torch pickle leaves ignored stack values")
                raise pickle._Stop(self.stack.pop())

            dispatch[pickle.STOP[0]] = load_stop

        try:
            loaded_root = _StorageAuditUnpickler(io.BytesIO(pickle_payload)).load()
        except (pickle.UnpicklingError, EOFError, AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"{label} torch pickle storage closure is malformed") from error
        if type(loaded_root) is not dict or not loaded_root:
            raise ValueError(f"{label} torch pickle root is not the frozen model dictionary")
        root_keys = set(loaded_root)
        allowed_root_sets = {
            frozenset(expected_gifstream_nets_keys(False)),
            frozenset(expected_gifstream_nets_keys(True)),
        }
        if expected_app_opt is None:
            if frozenset(root_keys) not in allowed_root_sets:
                raise ValueError(f"{label} torch model dictionary keys are not exact")
        elif root_keys != expected_gifstream_nets_keys(bool(expected_app_opt)):
            raise ValueError(f"{label} torch model dictionary differs from app_opt")
        ordered_roles = [
            "decoders",
            "scales_entropy_model",
            "anchor_features_entropy_model",
            "offsets_entropy_model",
            "factors_entropy_model",
            "time_features_entropy_model",
            "scaling",
        ]
        if "app_module" in root_keys:
            ordered_roles.append("app_module")
        if list(loaded_root) != ordered_roles:
            raise ValueError(f"{label} torch top-level model order is not exact")

        state_hashes: Dict[str, str] = {}
        state_schemas: Dict[str, List[Dict[str, Any]]] = {}
        for role in ordered_roles:
            if role == "scaling":
                continue
            state = loaded_root[role]
            if (
                not isinstance(state, _OrderedStateAudit)
                or not state
                or state.state != _expected_state_metadata(role)
            ):
                raise ValueError(f"{label} torch state role is malformed: {role}")
            digest = hashlib.sha256()
            schema_rows = []
            for name in sorted(state):
                tensor = state[name]
                if type(name) is not str or not name or not isinstance(tensor, _TensorAudit):
                    raise ValueError(f"{label} torch state tensor role is malformed: {role}")
                raw = members[f"{root}/data/{tensor.storage.key}"]
                header = canonical_json_bytes(
                    {
                        "name": name,
                        "dtype": "torch.float32",
                        "shape": list(tensor.shape),
                    }
                )
                digest.update(len(header).to_bytes(8, "little"))
                digest.update(header)
                digest.update(raw)
                schema_rows.append(
                    {
                        "name": name,
                        "dtype": "torch.float32",
                        "shape": list(tensor.shape),
                        "stride": list(tensor.stride),
                        "storage_key": tensor.storage.key,
                    }
                )
            state_hashes[role] = digest.hexdigest()
            state_schemas[role] = schema_rows

        scaling = loaded_root["scaling"]
        scaling_order = [
            "anchors",
            "scales",
            "quats",
            "opacities",
            "anchor_features",
            "offsets",
            "factors",
            "time_features",
        ]
        if type(scaling) is not dict or list(scaling) != scaling_order:
            raise ValueError(f"{label} torch codec scaling dictionary is not exact")
        for name, value in scaling.items():
            if value is not None and (
                type(value) not in {int, float}
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{label} torch codec scaling value is invalid: {name}")

        if set(storage_expectations) != set(data_indices) or set(storage_uses) != set(
            data_indices
        ):
            raise ValueError(f"{label} torch ZIP has unreferenced or missing storage records")
        for index, storage in storage_expectations.items():
            if len(members[f"{root}/data/{index}"]) != storage.width * storage.numel:
                raise ValueError(f"{label} torch storage byte extent differs from its pickle")

        serialization_name = f"{root}/.data/serialization_id"
        serialization_digest = hashlib.sha256()
        for name in sorted(name for name in members if name != serialization_name):
            encoded = name.encode("utf-8")
            value = members[name]
            serialization_digest.update(len(encoded).to_bytes(8, "little"))
            serialization_digest.update(encoded)
            serialization_digest.update(len(value).to_bytes(8, "little"))
            serialization_digest.update(value)
        members[serialization_name] = (
            f"{int.from_bytes(serialization_digest.digest(), 'big') % (10 ** 40):040d}"
        ).encode("ascii")
        canonical = _canonical_torch_zip_bytes(root, members)
        return canonical, {
            "root": root,
            "storage_count": len(storage_expectations),
            "top_level_keys": list(loaded_root),
            "state_sha256": state_hashes,
            "state_schema": state_schemas,
            "scaling": scaling,
            "scaling_sha256": sha256_bytes(canonical_json_bytes(scaling)),
            "serialization_id": members[serialization_name].decode("ascii"),
            "pickle_bytes": len(pickle_payload),
            "archive_bytes": len(canonical),
        }
    finally:
        handle.close()


def _torch_save_zip_contract(
    payload: bytes, label: str, expected_app_opt: Optional[bool] = None
) -> Dict[str, Any]:
    canonical, audit = _canonical_torch_save_payload(payload, label, expected_app_opt)
    if payload != canonical:
        raise ValueError(f"{label} torch ZIP/pickle bytes are not canonical")
    return audit


def canonicalize_gifstream_torch_payload(path: Path, app_opt: bool) -> Dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("GIFStream torch payload is unavailable or a symlink")
    original = path.read_bytes()
    canonical, audit = _canonical_torch_save_payload(original, path.name, bool(app_opt))
    if original != canonical:
        staging = path.with_name(path.name + ".h007-canonical.staging")
        if staging.exists() or staging.is_symlink():
            raise ValueError("GIFStream torch canonical staging already exists")
        with staging.open("xb") as handle:
            handle.write(canonical)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
    _torch_save_zip_contract(path.read_bytes(), path.name, bool(app_opt))
    return {
        "path": path.name,
        "source_sha256": sha256_bytes(original),
        "canonical_sha256": sha256_file(path),
        "source_bytes": len(original),
        "canonical_bytes": path.stat().st_size,
        **audit,
    }


def _canonical_zip_bytes(members: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as handle:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            handle.writestr(
                info,
                members[name],
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return output.getvalue()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} is not a lowercase SHA-256")
    text = value
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return text


def _nonempty_string(value: Any, label: str) -> str:
    if type(value) is not str or not value or len(value) > 4096:
        raise ValueError(f"{label} is not an exact bounded JSON string")
    return value


def _safe_zip_rows(path: Path) -> Tuple[zipfile.ZipFile, Dict[str, zipfile.ZipInfo]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"ZIP archive is unavailable or a symlink: {path}")
    handle = zipfile.ZipFile(path, "r")
    rows: Dict[str, zipfile.ZipInfo] = {}
    try:
        for info in handle.infolist():
            name = info.filename
            if name in rows:
                raise ValueError(f"duplicate ZIP member: {name}")
            if name.startswith("/") or ".." in Path(name).parts or "\\" in name:
                raise ValueError(f"unsafe ZIP member: {name}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"symlink ZIP member is forbidden: {name}")
            if not info.is_dir():
                rows[name] = info
        bad = handle.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC failure: {bad}")
    except Exception:
        handle.close()
        raise
    return handle, rows


def _read_zip_json(handle: zipfile.ZipFile, rows: Mapping[str, zipfile.ZipInfo], name: str) -> Dict[str, Any]:
    if name not in rows:
        raise ValueError(f"ZIP lacks required member: {name}")
    return _strict_canonical_json(handle.read(name), f"ZIP JSON member {name}")


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} is not a positive integer")
    number = value
    if number <= 0:
        raise ValueError(f"{label} is not a positive integer")
    return number


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} is not a nonnegative integer")
    number = value
    if number < 0:
        raise ValueError(f"{label} is not a nonnegative integer")
    return number


def _frozen_rate_index(value: Any, label: str) -> int:
    allowed = {str(index): index for index in FROZEN_RATE_LAMBDAS}
    if type(value) is not str or value not in allowed:
        raise ValueError(f"{label} is not a canonical frozen rate string")
    return allowed[value]


def _validate_producer_training_config_types(
    config: Mapping[str, Any], label: str
) -> Dict[str, Any]:
    if not isinstance(config, dict) or set(config) != PRODUCER_TRAINING_CONFIG_FIELDS:
        raise ValueError(f"{label} fields are incomplete or unexpected")
    positive = {
        name: _positive_int(config[name], f"{label} {name}")
        for name in PRODUCER_TRAINING_POSITIVE_INT_FIELDS
    }
    nonnegative = {
        name: _nonnegative_int(config[name], f"{label} {name}")
        for name in PRODUCER_TRAINING_NONNEGATIVE_INT_FIELDS
    }
    if any(type(config[name]) is not bool for name in PRODUCER_TRAINING_BOOL_FIELDS):
        raise ValueError(f"{label} boolean controls are invalid")
    rate_index = _frozen_rate_index(config["rate"], f"{label} rate")
    rd_lambda = _finite_number(config["rd_lambda"], f"{label} RD lambda", positive=True)
    if abs(rd_lambda - FROZEN_RATE_LAMBDAS[rate_index]) > 1e-15:
        raise ValueError(f"{label} rate/RD-lambda differs from the frozen grid")
    _finite_number(config["voxel_size"], f"{label} voxel size", positive=True)
    return {
        "max_steps": positive["max_steps"],
        "rate_index": rate_index,
        "rd_lambda": rd_lambda,
        "compression_seed": nonnegative["compression_seed"],
    }


def _validate_decoder_config_discrete_types(config: Mapping[str, Any]) -> None:
    for name in DECODER_POSITIVE_INT_FIELDS:
        _positive_int(config[name], f"decoder config {name}")
    for name in DECODER_NONNEGATIVE_INT_FIELDS:
        _nonnegative_int(config[name], f"decoder config {name}")
    if config["rate"] not in FROZEN_RATE_LAMBDAS:
        raise ValueError("decoder config rate is outside the registered rate grid")
    camera_count = frozen_scene_camera_count(config["scene"])
    if config["warm_camera_pose_index"] >= camera_count:
        raise ValueError("decoder config warm camera pose is outside the frozen camera grid")
    if config["warm_frame_index"] >= config["GOP_size"]:
        raise ValueError("decoder config warm frame is outside the GOP")
    _finite_number(config["voxel_size"], "decoder config voxel size", positive=True)
    _finite_number(config["phi"], "decoder config phi", positive=True)


def validate_frozen_training_receipt_contract(
    receipt: Mapping[str, Any],
    *,
    expected_scene: str,
    expected_variant: str,
    expected_training_config: Mapping[str, Any],
    expected_runtime_provenance: Mapping[str, Any],
    expected_source_checkpoints: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Validate the shared official/AP checkpoint receipt contract."""

    required = {
        "schema",
        "official_commit",
        "scene",
        "variant",
        "training_step",
        "state_position",
        "training_config",
        "training_config_sha256",
        "source_checkpoints",
        "model_state_sha256",
        "ap_training_receipt_sha256",
        "runtime_provenance",
        "outcome_fields_read",
    }
    training_config = dict(expected_training_config)
    checkpoint_rows = [dict(row) for row in expected_source_checkpoints]
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise ValueError("frozen training receipt fields are incomplete or unexpected")
    training_types = _validate_producer_training_config_types(
        training_config, "frozen producer training config"
    )
    max_steps = training_types["max_steps"]
    if (
        receipt["schema"] != FROZEN_TRAINING_RECEIPT_SCHEMA
        or receipt["official_commit"] != OFFICIAL_COMMIT
        or receipt["scene"] != expected_scene
        or receipt["variant"] != expected_variant
        or _nonnegative_int(receipt["training_step"], "training step") != max_steps - 1
        or receipt["state_position"]
        != "after_optimizer_entropy_and_strategy_post_backward"
        or canonical_json_bytes(receipt["training_config"])
        != canonical_json_bytes(training_config)
        or receipt["training_config_sha256"]
        != sha256_bytes(canonical_json_bytes(training_config))
        or receipt["source_checkpoints"] != checkpoint_rows
        or receipt["runtime_provenance"] != dict(expected_runtime_provenance)
        or receipt["outcome_fields_read"] != []
    ):
        raise ValueError("frozen training receipt identity/config/runtime mismatch")
    if not checkpoint_rows:
        raise ValueError("frozen training receipt checkpoint grid is empty")
    for row in checkpoint_rows:
        if (
            set(row) != {"path", "bytes", "sha256"}
            or not Path(
                _nonempty_string(row["path"], "source-checkpoint path")
            ).is_absolute()
            or _positive_int(row["bytes"], "frozen source-checkpoint bytes") <= 0
        ):
            raise ValueError("frozen training receipt checkpoint row is malformed")
        _require_sha256(row["sha256"], "frozen source checkpoint SHA-256")

    state = receipt["model_state_sha256"]
    if not isinstance(state, dict) or set(state) != {
        "splats",
        "decoders",
        "entropy_models",
        "codec_scaling",
        "appearance_module",
    }:
        raise ValueError("frozen training receipt model-state closure is incomplete")
    for name in ("splats", "decoders", "codec_scaling"):
        _require_sha256(state[name], f"frozen training receipt {name} SHA-256")
    entropy = state["entropy_models"]
    if not isinstance(entropy, dict) or set(entropy) != {
        "scales",
        "offsets",
        "anchor_features",
        "factors",
        "time_features",
    }:
        raise ValueError("frozen training receipt entropy-state closure is incomplete")
    for name, digest in entropy.items():
        if not name:
            raise ValueError("frozen training receipt has an empty entropy-model name")
        _require_sha256(digest, f"frozen entropy model {name} SHA-256")
    app_opt = bool(training_config.get("app_opt", False))
    appearance = state["appearance_module"]
    if app_opt:
        _require_sha256(appearance, "frozen appearance-module SHA-256")
    elif appearance is not None:
        raise ValueError("appearance-disabled frozen receipt declares appearance state")
    ap_receipt = receipt["ap_training_receipt_sha256"]
    if expected_variant == "official":
        if ap_receipt is not None:
            raise ValueError("official frozen training receipt declares AP training state")
    else:
        _require_sha256(ap_receipt, "frozen AP checkpoint receipt SHA-256")
    return dict(receipt)


def validate_ap_training_receipt_binding(
    *,
    method: str,
    producer_sha256: Any,
    frozen_sha256: Any,
    counted_payload: Optional[bytes],
) -> Optional[str]:
    """Require one exact AP receipt hash across frozen, active and counted state."""

    if method == "official":
        if (
            producer_sha256 is not None
            or frozen_sha256 is not None
            or counted_payload is not None
        ):
            raise ValueError("official codec unexpectedly declares AP training state")
        return None
    producer = _require_sha256(
        producer_sha256, "producer AP training receipt SHA-256"
    )
    frozen = _require_sha256(
        frozen_sha256, "frozen AP training receipt SHA-256"
    )
    if producer != frozen:
        raise ValueError("producer/frozen AP training receipt mismatch")
    if counted_payload is None or sha256_bytes(counted_payload) != producer:
        raise ValueError("counted AP training receipt binding mismatch")
    return producer


def _validate_ap_training_receipt_contract(
    payload: bytes,
    *,
    method: str,
    expected_sha256: str,
    runtime_provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    receipt = _strict_canonical_json(payload, "counted AP training receipt")
    required = {
        "schema",
        "scene",
        "variant",
        "freeze_step",
        "score_sha256",
        "ap_score_seconds",
        "protected_fraction",
        "q_ap_multiplier",
        "q_bg_multiplier",
        "random_seed",
        "estimator_version",
        "time_entropy_model_sha256",
        "time_feature_scaling",
        "time_entropy_model_frozen_after_freeze",
        "factor_membership_columns_frozen",
        "path_contract_schema",
        "path_dependency_rule",
        "path_knn_count",
        "path_knn_graph_sha256",
        "factor_protected_multiplier",
        "factor_background_multiplier",
        "anchor_feature_protected_multiplier",
        "anchor_feature_background_multiplier",
        "path_loss_required",
        "path_loss_lambda",
        "path_loss_every",
        "path_loss_applications",
        "path_loss_steps",
        "simulation_mask_rule",
        "estimated_byte_rule",
        "path_loss_reference",
        "runtime_provenance",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise ValueError("counted AP training receipt fields are incomplete or unexpected")
    if (
        receipt["schema"] != "h007.ap_training_receipt.v2"
        or receipt["variant"] != method
        or receipt["runtime_provenance"] != dict(runtime_provenance)
        or receipt["time_entropy_model_frozen_after_freeze"] is not True
        or receipt["factor_membership_columns_frozen"] != [0, 3]
        or any(
            type(value) is not int
            for value in receipt["factor_membership_columns_frozen"]
        )
        or type(receipt["path_loss_required"]) is not bool
        or receipt["simulation_mask_rule"]
        != "quantized_factor3_gt0_and_factor0_gt0"
        or receipt["estimated_byte_rule"] != "exact_frozen_integer_subset_sum"
        or receipt["path_loss_reference"]
        != "current_raw_full_graph_vs_simulated_retained_graph_on_retained_rows"
        or receipt["path_contract_schema"]
        != "h007.ap_gifstream.path_contract.v1"
        or receipt["path_dependency_rule"]
        != "protected-plus-one-hop-retained-knn"
        or sha256_bytes(payload) != _require_sha256(
            expected_sha256, "counted AP training receipt SHA-256"
        )
    ):
        raise ValueError("counted AP training receipt identity/provenance mismatch")
    _require_sha256(receipt["score_sha256"], "AP score artifact SHA-256")
    _require_sha256(
        receipt["time_entropy_model_sha256"], "AP entropy model SHA-256"
    )
    for name in (
        "ap_score_seconds",
        "protected_fraction",
        "q_ap_multiplier",
        "q_bg_multiplier",
        "time_feature_scaling",
    ):
        _finite_number(receipt[name], f"AP receipt {name}", positive=True)
    _nonnegative_int(receipt["freeze_step"], "AP freeze step")
    _nonnegative_int(receipt["random_seed"], "AP random seed")
    _positive_int(receipt["path_knn_count"], "AP path KNN count")
    _require_sha256(
        receipt["path_knn_graph_sha256"], "AP retained-KNN graph SHA-256"
    )
    for name in (
        "factor_protected_multiplier",
        "factor_background_multiplier",
        "anchor_feature_protected_multiplier",
        "anchor_feature_background_multiplier",
    ):
        _finite_number(receipt[name], f"AP receipt {name}", positive=True)
    _finite_number(
        receipt["path_loss_lambda"], "AP path-loss lambda", nonnegative=True
    )
    _positive_int(receipt["path_loss_every"], "AP path-loss interval")
    path_steps = receipt["path_loss_steps"]
    applications = _nonnegative_int(
        receipt["path_loss_applications"], "AP path-loss application count"
    )
    if (
        not isinstance(path_steps, list)
        or len(path_steps) != applications
        or len(path_steps) > 30_000
        or any(isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in path_steps)
    ):
        raise ValueError("counted AP path-loss step closure is invalid")
    return receipt


def validate_ap_seed_quantizer_closure(
    ap_meta: Mapping[str, Any],
    ap_training: Mapping[str, Any],
    *,
    compression_seed: int,
) -> None:
    """Bind actual allocation metadata to the frozen AP training controls."""

    meta_q_ap = _finite_number(
        ap_meta.get("q_ap_multiplier"), "AP metadata q_ap_multiplier", positive=True
    )
    meta_q_bg = _finite_number(
        ap_meta.get("q_bg_multiplier"), "AP metadata q_bg_multiplier", positive=True
    )
    training_q_ap = _finite_number(
        ap_training.get("q_ap_multiplier"), "AP training q_ap_multiplier", positive=True
    )
    training_q_bg = _finite_number(
        ap_training.get("q_bg_multiplier"), "AP training q_bg_multiplier", positive=True
    )
    score = ap_meta.get("score", {})
    score_q_ap = _finite_number(
        score.get("q_ap_multiplier"), "AP score q_ap_multiplier", positive=True
    )
    score_q_bg = _finite_number(
        score.get("q_bg_multiplier"), "AP score q_bg_multiplier", positive=True
    )

    if (
        _nonnegative_int(ap_meta.get("compression_seed"), "AP compression seed")
        != _nonnegative_int(compression_seed, "frozen AP compression seed")
        or meta_q_ap != training_q_ap
        or meta_q_bg != training_q_bg
        or score.get("score_artifact_sha256")
        != ap_training.get("score_sha256")
        or score_q_ap != training_q_ap
        or score_q_bg != training_q_bg
    ):
        raise ValueError("AP seed/quantizer/score closure differs from frozen training")


def _validate_runtime_provenance_shape(value: Any, label: str) -> Dict[str, Any]:
    required = {
        "schema",
        "manifest_sha256",
        "official_commit",
        "patch_sha256",
        "normalized_code_tree",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{label} fields are incomplete or unexpected")
    tree = value["normalized_code_tree"]
    tree_required = {
        "schema",
        "normalization",
        "roots",
        "root_files",
        "suffixes",
        "special_names",
        "file_count",
        "sha256",
    }
    if (
        value["schema"] != "h007.ap_gifstream.runtime_provenance.v1"
        or value["official_commit"] != OFFICIAL_COMMIT
        or not isinstance(value["patch_sha256"], list)
        or len(value["patch_sha256"]) != 9
        or not isinstance(tree, dict)
        or set(tree) != tree_required
        or tree["schema"] != "h007.normalized_code_tree.v1"
    ):
        raise ValueError(f"{label} identity/tree closure is invalid")
    _require_sha256(value["manifest_sha256"], f"{label} manifest SHA-256")
    _require_sha256(tree["sha256"], f"{label} tree SHA-256")
    for index, digest in enumerate(value["patch_sha256"]):
        _require_sha256(digest, f"{label} patch {index} SHA-256")
    return dict(value)


def _validate_meta_parameter(name: str, row: Any, method: str) -> None:
    ordinary = {
        "anchors": {"shape", "dtype", "mins", "maxs", "voxel_size"},
        "scales": {"shape", "dtype", "scaling"},
        "quats": {"shape", "dtype"},
        "opacities": {"shape", "dtype"},
        "offsets": {"shape", "dtype", "scaling"},
        "anchor_features": {"shape", "dtype", "scaling", "length", "channel"},
        "factors": {"shape", "dtype", "scaling"},
    }
    quantized_ap = method in QUANTIZED_AP_VARIANTS
    if name == "anchor_features" and quantized_ap:
        expected = {
            "shape",
            "dtype",
            "ap_two_class_anchor_features",
            "base_scaling",
            "families",
        }
    elif name == "factors" and quantized_ap:
        expected = {
            "shape",
            "dtype",
            "ap_two_class_factors",
            "base_scaling",
            "factor0_activation_value",
            "reconstruction_rule",
            "families",
        }
    elif name == "time_features":
        expected = (
            {
                "shape",
                "dtype",
                "ap_two_class",
                "precision_mask_contract",
                "base_scaling",
                "length",
                "channel",
                "families",
            }
            if quantized_ap
            else {"shape", "dtype", "scaling", "length", "channel"}
        )
    else:
        expected = ordinary[name]
    if not isinstance(row, dict) or set(row) != expected:
        raise ValueError(f"GIFStream metadata fields are unexpected: {name}")
    shape = row["shape"]
    if (
        not isinstance(shape, list)
        or not shape
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape)
        or type(row["dtype"]) is not str
        or not row["dtype"]
    ):
        raise ValueError(f"GIFStream tensor shape/dtype is malformed: {name}")
    if name == "anchors":
        for bound_name in ("mins", "maxs"):
            bounds = row[bound_name]
            if not isinstance(bounds, list) or len(bounds) != 3:
                raise ValueError(f"GIFStream anchor {bound_name} are malformed")
            for index, value in enumerate(bounds):
                _finite_number(value, f"GIFStream anchor {bound_name}[{index}]")
        _finite_number(row["voxel_size"], "GIFStream anchor voxel size", positive=True)
    elif "scaling" in row:
        _finite_number(row["scaling"], f"GIFStream {name} scaling", positive=True)
    if "length" in row:
        _positive_int(row["length"], f"GIFStream {name} stream length")
    if "channel" in row:
        _positive_int(row["channel"], f"GIFStream {name} channel count")
    if name in {"anchor_features", "time_features"} and quantized_ap:
        flag = (
            row.get("ap_two_class_anchor_features")
            if name == "anchor_features"
            else row.get("ap_two_class")
        )
        if flag is not True:
            raise ValueError(f"AP {name} metadata lacks two-class coding")
        if name == "time_features" and row.get("precision_mask_contract") != "path_input_mask":
            raise ValueError("AP temporal metadata lacks the path-input precision contract")
        families = row["families"]
        if not isinstance(families, dict) or set(families) != {"path", "bg"}:
            raise ValueError(f"AP {name} families are incomplete")
        _finite_number(row["base_scaling"], f"AP {name} base scaling", positive=True)
        for family in ("path", "bg"):
            family_row = families[family]
            if not isinstance(family_row, dict) or set(family_row) != {
                "rows",
                "scaling",
                "multiplier",
                "length",
                "channel",
            }:
                raise ValueError(f"AP {name} family fields are unexpected: {family}")
            _nonnegative_int(family_row["rows"], f"AP {family} {name} row count")
            _finite_number(family_row["scaling"], f"AP {family} {name} scaling", positive=True)
            _finite_number(family_row["multiplier"], f"AP {family} {name} multiplier", positive=True)
            _positive_int(family_row["length"], f"AP {family} {name} stream length")
            _positive_int(family_row["channel"], f"AP {family} {name} channel count")
    if name == "factors" and quantized_ap:
        if (
            row["ap_two_class_factors"] is not True
            or row["reconstruction_rule"]
            != "adaptive-symbols+counted-factor0/factor3-semantics"
        ):
            raise ValueError("AP factor metadata lacks the v6 reconstruction contract")
        _finite_number(row["base_scaling"], "AP factor base scaling", positive=True)
        _finite_number(
            row["factor0_activation_value"],
            "AP factor0 activation value",
            positive=True,
        )
        families = row["families"]
        if not isinstance(families, dict) or set(families) != {"path", "bg"}:
            raise ValueError("AP factor families are incomplete")
        for family in ("path", "bg"):
            family_row = families[family]
            if not isinstance(family_row, dict) or set(family_row) != {
                "rows",
                "scaling",
                "multiplier",
                "path",
                "bytes",
                "sha256",
            }:
                raise ValueError(f"AP factor family fields are unexpected: {family}")
            _nonnegative_int(family_row["rows"], f"AP {family} factor row count")
            _finite_number(family_row["scaling"], f"AP {family} factor scaling", positive=True)
            _finite_number(family_row["multiplier"], f"AP {family} factor multiplier", positive=True)
            _nonnegative_int(family_row["bytes"], f"AP {family} factor bytes")
            _require_sha256(family_row["sha256"], f"AP {family} factor SHA-256")
            expected_path = f"factors_{family}.bin"
            if family_row["path"] != expected_path:
                raise ValueError(f"AP {family} factor stream path is noncanonical")
def _required_gifstream_streams(meta: Mapping[str, Any], method: str) -> Dict[str, str]:
    """Derive the exact codec payload consumed by GIFStream's decoder.

    This is intentionally independent of the byte census and payload manifest:
    neither a generic ``payload.bin`` nor a self-declared file list can satisfy
    the closure without the parameter metadata and streams expected by the real
    GIFStream decoder.
    """

    core = {
        "anchors",
        "scales",
        "quats",
        "opacities",
        "offsets",
        "anchor_features",
        "factors",
        "time_features",
    }
    expected_keys = core | ({"__ap__"} if method != "official" else set())
    if not isinstance(meta, dict) or set(meta) != expected_keys:
        raise ValueError("GIFStream meta parameter set is incomplete or unexpected")
    for name in core:
        _validate_meta_parameter(name, meta[name], method)
    anchor_shape = meta["anchors"]["shape"]
    if len(anchor_shape) != 2:
        raise ValueError("GIFStream anchor tensor shape is malformed")
    anchor_count = _positive_int(anchor_shape[0], "GIFStream anchor row count")
    anchor_width = _positive_int(anchor_shape[1], "GIFStream anchor width")
    if (
        anchor_width != 3
        or meta["anchors"]["dtype"] != "float32"
        or meta["quats"]["shape"] != [anchor_count, 4]
        or meta["quats"]["dtype"] != "float32"
        or meta["opacities"]["shape"] != [anchor_count, 1]
        or meta["opacities"]["dtype"] != "float32"
    ):
        raise ValueError("GIFStream anchor/quaternion/opacity tensor roles are malformed")

    streams: Dict[str, str] = {
        "meta.json": "codec_metadata",
        "nets.pt": "counted_decoder_models",
        "anchors_l.png": "anchor_low_bytes",
        "anchors_u.png": "anchor_high_bytes",
        "quats.npz": "quaternion_payload",
        "opacities.npz": "opacity_payload",
        "scales.bin": "scale_entropy_stream",
        "offsets.bin": "offset_entropy_stream",
    }
    anchor_meta = meta["anchor_features"]
    if method in QUANTIZED_AP_VARIANTS:
        for family in ("path", "bg"):
            family_meta = anchor_meta["families"][family]
            if _nonnegative_int(
                family_meta["rows"], f"{family} anchor-feature rows"
            ):
                for index in range(
                    _positive_int(
                        family_meta["length"],
                        f"{family} anchor-feature stream length",
                    )
                ):
                    streams[
                        f"anchor_features_{family}_{index:05d}.bin"
                    ] = f"{family}_anchor_feature_entropy_stream"
        factor_meta = meta["factors"]
        for family in ("path", "bg"):
            family_meta = factor_meta["families"][family]
            if _nonnegative_int(family_meta["rows"], f"{family} factor rows"):
                streams[str(family_meta["path"])] = (
                    f"{family}_factor_entropy_stream"
                )
    else:
        streams["factors.bin"] = "factor_entropy_stream"
        anchor_length = _positive_int(
            anchor_meta.get("length"), "anchor-feature stream length"
        )
        for index in range(anchor_length):
            streams[
                f"anchor_features_{index:05d}.bin"
            ] = "anchor_feature_entropy_stream"

    time_meta = meta["time_features"]
    if method == "official":
        if time_meta.get("ap_two_class") is not None or "families" in time_meta:
            raise ValueError("official GIFStream metadata declares AP temporal families")
        time_length = _positive_int(time_meta.get("length"), "temporal stream length")
        for index in range(time_length):
            streams[f"time_features_{index:05d}.bin"] = "temporal_entropy_stream"
    else:
        if time_meta.get("ap_two_class") is True:
            families = time_meta.get("families")
            if not isinstance(families, dict) or set(families) != {"path", "bg"}:
                raise ValueError("AP GIFStream temporal families are incomplete")
            for family in ("path", "bg"):
                family_meta = families[family]
                if not isinstance(family_meta, dict):
                    raise ValueError(f"AP temporal family metadata is malformed: {family}")
                rows = _nonnegative_int(
                    family_meta.get("rows"), f"{family} temporal family row count"
                )
                length = _positive_int(
                    family_meta.get("length"), f"{family} temporal stream length"
                )
                if rows < 0:
                    raise ValueError(f"AP temporal family row count is invalid: {family}")
                if rows:
                    for index in range(length):
                        streams[f"time_features_{family}_{index:05d}.bin"] = (
                            f"{family}_temporal_entropy_stream"
                        )
        else:
            if "families" in time_meta:
                raise ValueError("AP temporal metadata has families without two-class coding")
            time_length = _positive_int(
                time_meta.get("length"), "AP temporal stream length"
            )
            for index in range(time_length):
                streams[f"time_features_{index:05d}.bin"] = "temporal_entropy_stream"
        ap_meta = meta["__ap__"]
        ap_required = {
            "schema",
            "variant",
            "score",
            "allocation",
            "runtime_provenance",
            "compression_seed",
            "q_ap_multiplier",
            "q_bg_multiplier",
            "mask",
            "path_input_mask",
            "path_contract",
            "real_row_mask",
            "padding_row_mask",
            "active_row_mask",
            "identity_corrections",
        }
        if (
            not isinstance(ap_meta, dict)
            or set(ap_meta) != ap_required
            or ap_meta.get("schema") != "h007.ap_gifstream.codec.v6"
            or ap_meta.get("variant", {}).get("name") != method
        ):
            raise ValueError("AP codec metadata variant differs from decoder config")
        if ap_meta["variant"] != AP_VARIANT_METADATA.get(method):
            raise ValueError("AP variant metadata differs from the frozen producer")
        path_contract = ap_meta["path_contract"]
        if (
            not isinstance(path_contract, dict)
            or set(path_contract)
            != {
                "schema",
                "knn_count",
                "knn_rule",
                "dependency_rule",
                "retained_knn_graph_sha256",
                "canonical_anchor_reconstruction",
                "factor_protected_multiplier",
                "factor_background_multiplier",
                "anchor_feature_protected_multiplier",
                "anchor_feature_background_multiplier",
            }
            or path_contract["schema"]
            != "h007.ap_gifstream.path_contract.v1"
            or path_contract["dependency_rule"]
            != "protected-plus-one-hop-retained-knn"
            or path_contract["knn_rule"]
            != "retained-canonical-radius-complete-distance+lexicographic-id"
            or path_contract["canonical_anchor_reconstruction"] is not True
            or path_contract["factor_protected_multiplier"] != 1.0 / 256.0
            or path_contract["factor_background_multiplier"] != 1.0 / 64.0
            or path_contract["anchor_feature_protected_multiplier"] != 0.25
            or path_contract["anchor_feature_background_multiplier"] != 1.0
        ):
            raise ValueError("AP-v6 path contract is incomplete")
        _positive_int(path_contract["knn_count"], "AP path-contract KNN count")
        _require_sha256(
            path_contract["retained_knn_graph_sha256"],
            "AP retained-KNN graph SHA-256",
        )
        for name in (
            "factor_protected_multiplier",
            "factor_background_multiplier",
            "anchor_feature_protected_multiplier",
            "anchor_feature_background_multiplier",
        ):
            _finite_number(
                path_contract[name], f"AP path-contract {name}", positive=True
            )
        score_required = {
            "schema",
            "score_artifact",
            "score_artifact_sha256",
            "ranking",
            "variant",
            "protected_fraction",
            "q_ap_multiplier",
            "q_bg_multiplier",
            "random_seed",
            "estimator_version",
            "time_entropy_model_sha256",
            "time_feature_scaling",
            "time_entropy_model_frozen_after_freeze",
            "runtime_manifest_sha256",
            "normalized_code_tree_sha256",
            "patch_chain_sha256",
            "anchor_count",
            "eligible_count",
        }
        allocation_required = {
            "budget_source",
            "official_retain_count",
            "ap_retain_count",
            "current_vs_frozen_whole_xor",
            "current_vs_frozen_temporal_xor",
            "encoded_row_count",
            "real_row_count",
            "padding_row_count",
            "active_row_count",
            "ap_class_real_count",
            "whole_promoted_count",
            "whole_demoted_count",
            "official_estimated_time_bytes",
            "ap_estimated_time_bytes",
            "plas_permutation_complete",
            "id_order_definition",
        }
        if not isinstance(ap_meta["score"], dict) or set(ap_meta["score"]) != score_required:
            raise ValueError("AP score-audit fields are incomplete or unexpected")
        if not isinstance(ap_meta["allocation"], dict) or set(
            ap_meta["allocation"]
        ) != allocation_required:
            raise ValueError("AP allocation-audit fields are incomplete or unexpected")
        allocation = ap_meta["allocation"]
        score = ap_meta["score"]
        for name in (
            "protected_fraction",
            "q_ap_multiplier",
            "q_bg_multiplier",
            "time_feature_scaling",
        ):
            _finite_number(score[name], f"AP score {name}", positive=True)
        random_seed = score["random_seed"]
        if random_seed is not None:
            _nonnegative_int(random_seed, "AP score random seed")
        score_anchor_count = _positive_int(
            score["anchor_count"], "AP score anchor count"
        )
        score_eligible_count = _nonnegative_int(
            score["eligible_count"], "AP score eligible count"
        )
        # ``score_anchor_count`` is the frozen pre-allocation producer count.
        # ``anchor_count`` is the encoded row count after whole-anchor
        # allocation and square padding.  They are deliberately different for
        # a real AP stream, so only validate the score-internal bounds here;
        # the retained/encoded relationships are checked below.
        if (
            score_eligible_count > score_anchor_count
            or type(score["time_entropy_model_frozen_after_freeze"]) is not bool
        ):
            raise ValueError("AP score count/freeze metadata are invalid")
        encoded_count = _positive_int(
            allocation["encoded_row_count"], "AP encoded row count"
        )
        real_count = _positive_int(
            allocation["real_row_count"], "AP real row count"
        )
        padding_count = _nonnegative_int(
            allocation["padding_row_count"], "AP padding row count"
        )
        active_count = _nonnegative_int(
            allocation["active_row_count"], "AP active row count"
        )
        class_count = _nonnegative_int(
            allocation["ap_class_real_count"], "AP class row count"
        )
        official_retain_count = _positive_int(
            allocation["official_retain_count"], "AP official retain count"
        )
        ap_retain_count = _positive_int(
            allocation["ap_retain_count"], "AP retain count"
        )
        current_whole_xor = _nonnegative_int(
            allocation["current_vs_frozen_whole_xor"],
            "AP current/frozen whole-mask XOR count",
        )
        current_temporal_xor = _nonnegative_int(
            allocation["current_vs_frozen_temporal_xor"],
            "AP current/frozen temporal-mask XOR count",
        )
        for name in (
            "whole_promoted_count",
            "whole_demoted_count",
            "official_estimated_time_bytes",
            "ap_estimated_time_bytes",
        ):
            _nonnegative_int(allocation[name], f"AP allocation {name}")
        if (
            allocation["budget_source"] != "frozen_score_artifact"
            or official_retain_count != ap_retain_count
            or ap_retain_count != real_count
            or real_count > score_anchor_count
            or current_whole_xor > score_anchor_count
            or current_temporal_xor > score_anchor_count
            or allocation["whole_promoted_count"]
            != allocation["whole_demoted_count"]
            or allocation["official_estimated_time_bytes"]
            != allocation["ap_estimated_time_bytes"]
            or allocation["plas_permutation_complete"] is not True
            or allocation["id_order_definition"]
            != "counted_corrections_restore_prequantization_ids_in_encoded_real_row_order"
        ):
            raise ValueError("AP allocation categorical metadata are invalid")
        if (
            encoded_count != anchor_count
            or real_count + padding_count != encoded_count
            or active_count > real_count
            or class_count > real_count
        ):
            raise ValueError("AP allocation row counts differ from the encoded tensor roles")
        _validate_runtime_provenance_shape(
            ap_meta["runtime_provenance"], "AP metadata runtime provenance"
        )
        fixed_ap = {
            "mask": ("ap_class_mask.bin", "ap_class_mask"),
            "path_input_mask": (
                "ap_path_input_mask.bin",
                "ap_path_input_mask",
            ),
            "real_row_mask": ("ap_real_row_mask.bin", "ap_real_row_mask"),
            "padding_row_mask": ("ap_padding_row_mask.bin", "ap_padding_row_mask"),
            "active_row_mask": ("ap_active_row_mask.bin", "ap_active_row_mask"),
            "identity_corrections": (
                "ap_identity_corrections.bin",
                "ap_identity_corrections",
            ),
        }
        for key, (filename, role) in fixed_ap.items():
            record = ap_meta.get(key)
            if not isinstance(record, dict) or record.get("path") != filename:
                raise ValueError(f"AP metadata has an unsafe/noncanonical sidecar path: {key}")
            if key in {
                "mask",
                "path_input_mask",
                "real_row_mask",
                "padding_row_mask",
                "active_row_mask",
            }:
                required = {"path", "count", "true_count", "bytes", "sha256", "bitorder"}
                expected_true_count = {
                    "mask": class_count,
                    "path_input_mask": record.get("true_count"),
                    "real_row_mask": real_count,
                    "padding_row_mask": padding_count,
                    "active_row_mask": active_count,
                }[key]
                if (
                    set(record) != required
                    or record.get("bitorder") != "little"
                    or record.get("count") != encoded_count
                    or record.get("true_count") != expected_true_count
                    or (
                        key == "path_input_mask"
                        and (
                            type(record.get("true_count")) is not int
                            or record.get("true_count") < class_count
                            or record.get("true_count") > real_count
                        )
                    )
                ):
                    raise ValueError(f"AP mask metadata is malformed: {key}")
            else:
                required = {
                    "schema",
                    "path",
                    "row_count",
                    "mismatch_count",
                    "mask_bytes",
                    "bytes",
                    "sha256",
                    "bitorder",
                    "base",
                    "code",
                }
                if (
                    set(record) != required
                    or record.get("schema") != "h007.ap_identity_corrections.v1"
                    or record.get("row_count") != real_count
                    or type(record.get("mismatch_count")) is not int
                    or record.get("mismatch_count") < 0
                    or record.get("mismatch_count") > real_count
                    or record.get("mask_bytes") != (real_count + 7) // 8
                    or record.get("bytes")
                    != record.get("mask_bytes") + record.get("mismatch_count")
                    or record.get("bitorder") != "little"
                    or record.get("base")
                    != "round-decoded-anchor-div-voxel-size"
                    or record.get("code") != "uint8-base3-dx-dy-dz-plus1"
                ):
                    raise ValueError(f"AP identity-correction metadata is malformed: {key}")
            _nonnegative_int(record.get("bytes"), f"AP sidecar {key} byte count")
            _require_sha256(record.get("sha256"), f"AP sidecar {key} SHA-256")
            streams[filename] = role
    return streams


def build_gifstream_payload_manifest(
    root: Path, *, scene: str, variant: str, start_frame: int, gop_size: int
) -> Dict[str, Any]:
    """Build the decoder-input inventory from actual GIFStream metadata."""

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("GIFStream payload root is unavailable or a symlink")
    root = root.resolve()
    meta_path = root / "meta.json"
    nets_path = root / "nets.pt"
    if meta_path.is_symlink() or nets_path.is_symlink():
        raise ValueError("GIFStream core payload symlink is forbidden")
    meta_payload = meta_path.read_bytes()
    meta = json.loads(meta_payload.decode("utf-8"))
    streams = _required_gifstream_streams(meta, variant)
    rows = []
    for name, role in sorted(streams.items()):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"GIFStream decoder input is unavailable: {name}")
        payload = path.read_bytes()
        rows.append(
            {
                "path": name,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "role": role,
            }
        )
    return {
        "schema": GIFSTREAM_PAYLOAD_SCHEMA,
        "codec_family": "GIFStream",
        "scene": scene,
        "variant": variant,
        "start_frame": int(start_frame),
        "GOP_size": int(gop_size),
        "meta_sha256": sha256_bytes(meta_payload),
        "nets_sha256": sha256_file(nets_path),
        "decoder_inputs": rows,
        "outcome_fields_read": [],
    }


def _validate_gifstream_decoder_closure(
    handle: zipfile.ZipFile,
    rows: Mapping[str, zipfile.ZipInfo],
    config: Mapping[str, Any],
    scene: str,
    method: str,
    gop_id: int,
) -> Dict[str, Any]:
    if not isinstance(config, dict) or set(config) != DECODER_CONFIG_FIELDS:
        raise ValueError("GOP decoder config fields are incomplete or unexpected")
    required_members = {
        "meta.json",
        "nets.pt",
        "gifstream_payload_manifest.json",
        "producer_receipt.json",
        "training_receipt.json",
        "runtime.json",
        "runtime_provenance.json",
        "preregistered_patch_chain_manifest.json",
        "clean_decode_request.json",
        "camera_metadata/camera_keys.npy",
        "camera_metadata/intrinsics.npy",
        "camera_metadata/image_sizes.npy",
        "camera_metadata/camtoworlds.npy",
        "camera_metadata/camera_ids.npy",
        "camera_metadata/camera_names.npy",
        "camera_metadata/transform.npy",
        "camera_metadata/bounds.npy",
    }
    missing = sorted(required_members - set(rows))
    if missing:
        raise ValueError(f"GIFStream decoder closure lacks counted members: {missing}")
    if (
        config.get("schema") != DECODER_CONFIG_SCHEMA
        or config.get("codec_family") != "GIFStream"
        or config.get("official_commit") != OFFICIAL_COMMIT
        or config.get("clean_decode_entrypoint")
        != "examples/h007_clean_decode_gifstream.py"
    ):
        raise ValueError("GOP decoder config is not the frozen GIFStream decoder contract")
    _validate_decoder_config_discrete_types(config)
    patch_chain = config.get("patch_chain_sha256")
    if not isinstance(patch_chain, list) or len(patch_chain) != 9:
        raise ValueError("GIFStream GOP is not bound to the registered nine-stage runtime")
    for index, digest in enumerate(patch_chain):
        _require_sha256(digest, f"patch-chain entry {index} SHA-256")
    manifest_sha = _require_sha256(
        config.get("runtime_manifest_sha256"), "runtime manifest SHA-256"
    )
    tree_sha = _require_sha256(
        config.get("normalized_code_tree_sha256"), "normalized code-tree SHA-256"
    )

    runtime_provenance = _validate_runtime_provenance_shape(
        _read_zip_json(handle, rows, "runtime_provenance.json"),
        "counted GIFStream runtime provenance",
    )
    if (
        runtime_provenance.get("schema")
        != "h007.ap_gifstream.runtime_provenance.v1"
        or runtime_provenance.get("official_commit") != OFFICIAL_COMMIT
        or runtime_provenance.get("patch_sha256") != patch_chain
        or runtime_provenance.get("manifest_sha256") != manifest_sha
        or runtime_provenance.get("normalized_code_tree", {}).get("sha256") != tree_sha
    ):
        raise ValueError("counted GIFStream runtime provenance differs from decoder config")
    preregistered = handle.read("preregistered_patch_chain_manifest.json")
    preregistered_value = _strict_canonical_json(
        preregistered, "counted preregistration manifest"
    )
    if not isinstance(preregistered_value, dict) or set(preregistered_value) != {
        "schema",
        "official_commit",
        "patches",
        "normalized_code_tree",
    }:
        raise ValueError("counted preregistration manifest fields are unexpected")
    prereg_patches = preregistered_value["patches"]
    if (
        preregistered_value["schema"]
        != "h007.ap_gifstream.patch_chain_manifest.v1"
        or preregistered_value["official_commit"] != OFFICIAL_COMMIT
        or not isinstance(prereg_patches, list)
        or [row.get("stage") for row in prereg_patches]
        != [
            "patch1",
            "patch2",
            "patch2b",
            "patch3",
            "patch4",
            "patch5",
            "patch6",
            "patch7",
            "patch8",
        ]
        or any(
            not isinstance(row, dict)
            or set(row) != {"stage", "path", "sha256"}
            or row["sha256"] != patch_chain[index]
            or type(row["path"]) is not str
            or not row["path"]
            or len(row["path"]) > 4096
            for index, row in enumerate(prereg_patches)
        )
        or preregistered_value["normalized_code_tree"]
        != runtime_provenance["normalized_code_tree"]
    ):
        raise ValueError("counted preregistration manifest closure is invalid")
    if sha256_bytes(preregistered) != manifest_sha:
        raise ValueError("counted preregistration manifest differs from decoder config")

    meta_payload = handle.read("meta.json")
    meta = _strict_canonical_json(meta_payload, "counted GIFStream metadata")
    expected_streams = _required_gifstream_streams(meta, method)
    expected_anchor_png_geometry = _anchor_png_geometry(meta)
    if method != "official":
        ap_meta = meta["__ap__"]
        if ap_meta.get("runtime_provenance") != runtime_provenance:
            raise ValueError("AP codec metadata runtime provenance differs from the GOP")
        for key in (
            "mask",
            "path_input_mask",
            "real_row_mask",
            "padding_row_mask",
            "active_row_mask",
            "identity_corrections",
        ):
            record = ap_meta[key]
            sidecar_payload = handle.read(record["path"])
            if (
                len(sidecar_payload) != record["bytes"]
                or sha256_bytes(sidecar_payload) != record["sha256"]
            ):
                raise ValueError(f"AP metadata/sidecar byte binding mismatch: {key}")
            if key in {
                "mask",
                "path_input_mask",
                "real_row_mask",
                "padding_row_mask",
                "active_row_mask",
            }:
                _packed_bool_mask_contract(
                    sidecar_payload,
                    record["count"],
                    record["true_count"],
                    f"AP {key}",
                )
            else:
                _identity_correction_payload_contract(
                    sidecar_payload, record, "AP identity corrections"
                )
    payload_manifest_payload = handle.read("gifstream_payload_manifest.json")
    payload_manifest = _strict_canonical_json(
        payload_manifest_payload, "counted GIFStream payload manifest"
    )
    payload_required = {
        "schema",
        "codec_family",
        "scene",
        "variant",
        "start_frame",
        "GOP_size",
        "meta_sha256",
        "nets_sha256",
        "decoder_inputs",
        "outcome_fields_read",
    }
    if not isinstance(payload_manifest, dict) or set(payload_manifest) != payload_required:
        raise ValueError("GIFStream payload manifest fields are incomplete or unexpected")
    if (
        payload_manifest["schema"] != GIFSTREAM_PAYLOAD_SCHEMA
        or payload_manifest["codec_family"] != "GIFStream"
        or payload_manifest["scene"] != scene
        or payload_manifest["variant"] != method
        or _nonnegative_int(
            payload_manifest["start_frame"], "payload-manifest start frame"
        )
        != GOP_STARTS[gop_id]
        or _positive_int(payload_manifest["GOP_size"], "payload-manifest GOP size")
        != 60
        or payload_manifest["meta_sha256"] != sha256_bytes(meta_payload)
        or payload_manifest["nets_sha256"] != sha256_bytes(handle.read("nets.pt"))
        or payload_manifest["outcome_fields_read"] != []
    ):
        raise ValueError("GIFStream payload manifest identity/provenance mismatch")
    if sha256_bytes(payload_manifest_payload) != _require_sha256(
        config.get("payload_manifest_sha256"), "payload manifest SHA-256"
    ):
        raise ValueError("decoder config/payload manifest SHA-256 mismatch")
    declared_inputs: Dict[str, Mapping[str, Any]] = {}
    for row in payload_manifest["decoder_inputs"]:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256", "role"}:
            raise ValueError("GIFStream payload-manifest row fields are unexpected")
        name = _nonempty_string(row["path"], "GIFStream payload path")
        if name in declared_inputs:
            raise ValueError("GIFStream payload manifest duplicates a decoder input")
        _nonnegative_int(row["bytes"], f"GIFStream payload {name} byte count")
        declared_inputs[name] = row
    if set(declared_inputs) != set(expected_streams):
        raise ValueError("GIFStream decoder inputs differ from metadata-derived streams")
    for name, role in expected_streams.items():
        if name not in rows:
            raise ValueError(f"GIFStream decoder stream is missing: {name}")
        payload = handle.read(name)
        row = declared_inputs[name]
        if (
            row["role"] != role
            or row["bytes"] != len(payload)
            or row["sha256"] != sha256_bytes(payload)
        ):
            raise ValueError(f"GIFStream payload-manifest binding mismatch: {name}")
    if method in QUANTIZED_AP_VARIANTS:
        for family in ("path", "bg"):
            factor_row = meta["factors"]["families"][family]
            if int(factor_row["rows"]) == 0:
                if int(factor_row["bytes"]) != 0:
                    raise ValueError("empty AP factor family has nonzero bytes")
                continue
            bound = declared_inputs[str(factor_row["path"])]
            if (
                int(factor_row["bytes"]) != int(bound["bytes"])
                or factor_row["sha256"] != bound["sha256"]
            ):
                raise ValueError(
                    f"AP {family} factor metadata differs from counted payload"
                )
    optional_ap = {"ap_training_receipt.json"} if method != "official" else set()
    edit_group = {
        "ap_edit_hook.json",
        "ap_edit_ids.npz",
        "ap_edit_source_score.npz",
        "ap_edit_reference_manifest.json",
    }
    present_edit = edit_group & set(rows)
    if present_edit and present_edit != edit_group:
        raise ValueError("GIFStream archive has a partial AP edit closure")
    expected_members = (
        required_members
        | {"decoder_config.json", "byte_census.json"}
        | set(expected_streams)
        | optional_ap
        | present_edit
    )
    if set(rows) != expected_members:
        raise ValueError("GIFStream archive members differ from the exact decoder contract")

    for name, role in expected_streams.items():
        payload = handle.read(name)
        if name.endswith(".png"):
            _png_payload_contract(payload, name, expected_anchor_png_geometry)
        elif name.endswith(".npz"):
            if name in {"quats.npz", "opacities.npz"}:
                parameter = name[:-4]
                _codec_npz_payload_contract(payload, name, meta[parameter])
            else:
                raise ValueError(f"unrouted counted NumPy ZIP role: {name}")
        elif name.endswith(".npy"):
            _npy_payload_contract(payload, name)
        elif name == "ap_identity_corrections.bin":
            _identity_correction_payload_contract(
                payload, meta["__ap__"]["identity_corrections"], name
            )
        elif name.endswith(".bin") and "mask" not in role:
            _entropy_stream_contract(payload, name)
    nets_audit = _torch_save_zip_contract(
        handle.read("nets.pt"), "nets.pt", expected_app_opt=bool(config["app_opt"])
    )
    _camera_metadata_contract(handle, frozen_camera_names(scene))
    if present_edit:
        _npz_payload_contract(
            handle.read("ap_edit_ids.npz"),
            "ap_edit_ids.npz",
            AP_EDIT_IDS_NPZ_MEMBERS,
            expected_compression=zipfile.ZIP_STORED,
            expected_order=AP_EDIT_IDS_NPZ_ORDER,
        )
        _ap_score_npz_contract(
            handle.read("ap_edit_source_score.npz"),
            "ap_edit_source_score.npz",
            anchor_count=(
                _positive_int(
                    meta["__ap__"]["score"]["anchor_count"],
                    "AP score anchor count",
                )
                if method != "official"
                else None
            ),
            scene=scene,
            variant=_edit_source_variant(method),
        )

    producer_payload = handle.read("producer_receipt.json")
    producer = _strict_canonical_json(producer_payload, "counted producer receipt")
    producer_required = {
        "schema",
        "official_commit",
        "scene",
        "variant",
        "start_frame",
        "GOP_size",
        "training_step",
        "state_position",
        "training_config",
        "training_config_sha256",
        "source_checkpoints",
        "model_state_sha256",
        "ap_training_receipt_sha256",
        "runtime_provenance",
        "training_receipt_sha256",
        "outcome_fields_read",
    }
    if not isinstance(producer, dict) or set(producer) != producer_required:
        raise ValueError("GIFStream producer receipt fields are incomplete or unexpected")
    training_config_payload = canonical_json_bytes(producer["training_config"])
    training_config_sha = _require_sha256(
        producer["training_config_sha256"], "producer training-config SHA-256"
    )
    training_config = producer["training_config"]
    training_types = _validate_producer_training_config_types(
        training_config, "producer training config"
    )
    max_steps = training_types["max_steps"]
    rate_index = training_types["rate_index"]
    rd_lambda = training_types["rd_lambda"]
    compression_seed = training_types["compression_seed"]
    config_bindings = {
        "data_factor": "data_factor",
        "GOP_size": "GOP_size",
        "rate": "rate",
        "voxel_size": "voxel_size",
        "anchor_feature_dim": "anchor_feature_dim",
        "c_perframe": "c_perframe",
        "entropy_channel": "entropy_channel",
        "n_offsets": "n_offsets",
        "n_knn": "n_knn",
        "knn": "knn",
        "time_dim": "time_dim",
        "view_adaptive": "view_adaptive",
        "app_opt": "app_opt",
        "compression_seed": "compression_seed",
    }
    for config_name, training_name in config_bindings.items():
        left = config[config_name]
        right = training_config[training_name]
        if config_name == "rate":
            if left != rate_index:
                raise ValueError("decoder config differs from producer training config: rate")
        elif left != right:
            raise ValueError(
                f"decoder config differs from producer training config: {config_name}"
            )
    scene_camera_count = frozen_scene_camera_count(scene)
    for name in ("test_set", "remove_set"):
        values = config[name]
        if (
            not isinstance(values, list)
            or len(values) > scene_camera_count
            or len(values) != len(set(values))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value not in range(scene_camera_count)
                for value in values
            )
        ):
            raise ValueError(f"decoder config {name} is outside the frozen camera grid")
    if config["appearance_embedding_count"] != scene_camera_count:
        raise ValueError("decoder config camera count differs from the frozen camera grid")
    if (
        config["camera_model"] not in {"pinhole", "ortho", "fisheye"}
        or any(
            not isinstance(config[name], bool)
            for name in (
                "view_adaptive",
                "add_opacity_dist",
                "add_cov_dist",
                "add_color_dist",
                "app_opt",
                "packed",
                "antialiased",
                "knn",
            )
        )
    ):
        raise ValueError("decoder config categorical controls are invalid")
    if (
        producer["schema"] != PRODUCER_RECEIPT_SCHEMA
        or producer["official_commit"] != OFFICIAL_COMMIT
        or producer["scene"] != scene
        or producer["variant"] != method
        or _nonnegative_int(producer["start_frame"], "producer start frame")
        != GOP_STARTS[gop_id]
        or _positive_int(producer["GOP_size"], "producer GOP size") != 60
        or _nonnegative_int(producer["training_step"], "producer training step")
        != max_steps - 1
        or producer["state_position"]
        != "after_optimizer_entropy_and_strategy_post_backward"
        or config["compression_seed"] != compression_seed
        or config["rate"] != rate_index
        or sha256_bytes(training_config_payload) != training_config_sha
        or producer["runtime_provenance"] != runtime_provenance
        or producer["outcome_fields_read"] != []
    ):
        raise ValueError("GIFStream producer receipt identity/provenance mismatch")
    if sha256_bytes(producer_payload) != _require_sha256(
        config.get("producer_receipt_sha256"), "producer receipt SHA-256"
    ):
        raise ValueError("decoder config/producer receipt SHA-256 mismatch")
    training_payload = handle.read("training_receipt.json")
    training_sha = _require_sha256(
        producer["training_receipt_sha256"], "frozen training receipt SHA-256"
    )
    if (
        sha256_bytes(training_payload) != training_sha
        or config.get("training_receipt_sha256") != training_sha
    ):
        raise ValueError("counted frozen training receipt hash binding mismatch")
    training = _strict_canonical_json(training_payload, "counted frozen training receipt")
    training_required = {
        "schema",
        "official_commit",
        "scene",
        "variant",
        "training_step",
        "state_position",
        "training_config",
        "training_config_sha256",
        "source_checkpoints",
        "model_state_sha256",
        "ap_training_receipt_sha256",
        "runtime_provenance",
        "outcome_fields_read",
    }
    if not isinstance(training, dict) or set(training) != training_required:
        raise ValueError("counted frozen training receipt fields are incomplete or unexpected")
    validate_frozen_training_receipt_contract(
        training,
        expected_scene=scene,
        expected_variant=method,
        expected_training_config=producer["training_config"],
        expected_runtime_provenance=runtime_provenance,
        expected_source_checkpoints=producer["source_checkpoints"],
    )
    for name in (
        "official_commit",
        "scene",
        "variant",
        "training_step",
        "state_position",
        "training_config",
        "training_config_sha256",
        "source_checkpoints",
        "model_state_sha256",
        "ap_training_receipt_sha256",
        "runtime_provenance",
        "outcome_fields_read",
    ):
        if training.get(name) != producer.get(name):
            raise ValueError(f"producer/frozen training receipt mismatch: {name}")
    if training.get("schema") != "h007.gifstream_frozen_training_receipt.v1":
        raise ValueError("counted frozen training receipt schema is unsupported")
    state_hashes = producer["model_state_sha256"]
    if not isinstance(state_hashes, dict) or set(state_hashes) != {
        "splats",
        "decoders",
        "entropy_models",
        "codec_scaling",
        "appearance_module",
    }:
        raise ValueError("producer receipt model-state closure is incomplete")
    _require_sha256(state_hashes["splats"], "producer splat-state SHA-256")
    _require_sha256(state_hashes["decoders"], "producer decoder-state SHA-256")
    _require_sha256(state_hashes["codec_scaling"], "producer codec-scaling SHA-256")
    appearance_hash = state_hashes["appearance_module"]
    if appearance_hash is not None:
        _require_sha256(appearance_hash, "producer appearance-module SHA-256")
    entropy_hashes = state_hashes["entropy_models"]
    if not isinstance(entropy_hashes, dict) or set(entropy_hashes) != {
        "scales",
        "offsets",
        "anchor_features",
        "factors",
        "time_features",
    }:
        raise ValueError("producer receipt entropy-model closure is incomplete")
    for name, digest in entropy_hashes.items():
        if not name:
            raise ValueError("producer receipt has an empty entropy-model name")
        _require_sha256(digest, f"producer entropy model {name} SHA-256")
    _validate_nets_audit(nets_audit, config, state_hashes)
    checkpoints = producer["source_checkpoints"]
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("producer source-checkpoint closure is malformed")
    for row in checkpoints:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("producer source-checkpoint row fields are unexpected")
        _require_sha256(row["sha256"], "source checkpoint SHA-256")
        checkpoint_path = _nonempty_string(
            row["path"], "producer source-checkpoint path"
        )
        if _positive_int(row["bytes"], "producer source-checkpoint bytes") <= 0:
            raise ValueError("producer source-checkpoint row is invalid")
        if not Path(checkpoint_path).is_absolute():
            raise ValueError("producer source-checkpoint path is invalid")
    ap_training_sha = validate_ap_training_receipt_binding(
        method=method,
        producer_sha256=producer["ap_training_receipt_sha256"],
        frozen_sha256=training["ap_training_receipt_sha256"],
        counted_payload=(
            handle.read("ap_training_receipt.json")
            if "ap_training_receipt.json" in rows
            else None
        ),
    )
    ap_training = None
    if ap_training_sha is not None:
        ap_training = _validate_ap_training_receipt_contract(
            handle.read("ap_training_receipt.json"),
            method=method,
            expected_sha256=ap_training_sha,
            runtime_provenance=runtime_provenance,
        )
        validate_ap_seed_quantizer_closure(
            meta["__ap__"], ap_training, compression_seed=compression_seed
        )

    runtime = _read_zip_json(handle, rows, "runtime.json")
    if set(runtime) != RUNTIME_RECEIPT_FIELDS or runtime.get("schema") != "h007.gifstream_runtime.v1":
        raise ValueError("counted GIFStream runtime receipt schema is unsupported")
    encode_seconds = _finite_number(
        runtime.get("encode_seconds"), "counted encode seconds", positive=True
    )
    decode_seconds = _finite_number(
        runtime.get("model_load_plus_entropy_decode_seconds"),
        "counted decode seconds",
        positive=True,
    )
    peak_decode = runtime.get("peak_decode_cuda_bytes")
    if peak_decode is not None:
        _nonnegative_int(peak_decode, "counted peak decode CUDA bytes")
    if (
        runtime.get("warm_render")
        != {
            "status": "REQUIRED_IN_CLEAN_PROCESS",
            "camera_metadata_source": "counted_archive_only",
        }
        or runtime.get("warm_render_fps") is not None
        or runtime.get("outcome_fields_read") != []
    ):
        raise ValueError("counted GIFStream deferred-render runtime contract is invalid")
    if method == "official":
        if runtime.get("ap_score_seconds") is not None:
            raise ValueError("official GIFStream runtime declares AP score timing")
    else:
        ap_score_seconds = _finite_number(
            runtime.get("ap_score_seconds"), "counted AP score seconds", positive=True
        )
        if ap_training is None or ap_score_seconds != _finite_number(
            ap_training["ap_score_seconds"], "AP training score seconds", positive=True
        ):
            raise ValueError("counted AP score timing differs from frozen training")
    clean_request = _read_zip_json(handle, rows, "clean_decode_request.json")
    if (
        set(clean_request) != CLEAN_DECODE_REQUEST_FIELDS
        or clean_request.get("schema") != "h007.clean_decode_request.v2"
        or clean_request.get("archive_only") is not True
        or clean_request.get("entrypoint")
        != "examples/h007_clean_decode_gifstream.py"
        or clean_request.get("external_shared_runtime", {}).get(
            "provenance_manifest_required"
        )
        is not True
        or clean_request.get("external_shared_runtime", {}).get(
            "provenance_manifest_sha256"
        )
        != manifest_sha
    ):
        raise ValueError("counted clean-decode request is not fail-closed")
    if set(clean_request["external_shared_runtime"]) != {
        "provenance_manifest_required",
        "provenance_manifest_sha256",
    }:
        raise ValueError("counted clean-decode external-runtime fields are unexpected")

    if present_edit:
        edit_hook = _read_zip_json(handle, rows, "ap_edit_hook.json")
        edit_required = {
            "schema",
            "scene",
            "artifact",
            "artifact_sha256",
            "edited_anchor_count",
            "edit_strength",
            "target_rgb",
            "alignment",
            "source_score",
            "source_score_sha256",
            "selection",
            "reference_manifest",
            "reference_manifest_sha256",
            "selected_canonical_ids_sha256",
            "counted_artifact",
            "counted_artifact_sha256",
            "counted_source_score",
            "counted_source_score_sha256",
            "counted_reference_manifest",
            "counted_reference_manifest_sha256",
        }
        if not isinstance(edit_hook, dict) or set(edit_hook) != edit_required:
            raise ValueError("counted AP edit hook fields are incomplete or unexpected")
        if (
            edit_hook["schema"] != "h007.ap_edit_hook.v1"
            or edit_hook["scene"] != scene
            or edit_hook["alignment"] != "exact_canonical_voxel_id"
            or edit_hook["selection"] != AP_EDIT_SELECTION
            or edit_hook["counted_artifact"] != "ap_edit_ids.npz"
            or edit_hook["counted_source_score"] != "ap_edit_source_score.npz"
            or edit_hook["counted_reference_manifest"]
            != "ap_edit_reference_manifest.json"
        ):
            raise ValueError("counted AP edit hook identity/roles are invalid")
        edit_count = _positive_int(
            edit_hook["edited_anchor_count"], "counted AP edit anchor count"
        )
        edit_strength = _finite_number(
            edit_hook["edit_strength"], "counted AP edit strength", positive=True
        )
        target_rgb = edit_hook["target_rgb"]
        if (
            edit_strength > 1.0
            or not isinstance(target_rgb, list)
            or len(target_rgb) != 3
            or any(
                not 0.0
                <= _finite_number(value, f"counted AP target RGB[{index}]")
                <= 1.0
                for index, value in enumerate(target_rgb)
            )
        ):
            raise ValueError("counted AP edit strength/target RGB are invalid")
        _ap_edit_ids_npz_contract(
            handle.read("ap_edit_ids.npz"),
            "ap_edit_ids.npz",
            edit_count=edit_count,
            scene=scene,
        )
        edit_reference_payload = handle.read("ap_edit_reference_manifest.json")
        edit_reference = _strict_canonical_json(
            edit_reference_payload, "counted AP edit reference manifest"
        )
        if set(edit_reference) != {
            "schema",
            "scene",
            "source_score_sha256",
            "selection",
            "selection_count",
            "selected_canonical_ids_sha256",
        }:
            raise ValueError("counted AP edit reference fields are unexpected")
        if (
            edit_reference["schema"] != "h007.ap_edit_reference_manifest.v1"
            or edit_reference["scene"] != scene
            or edit_reference["selection"] != AP_EDIT_SELECTION
            or edit_reference["source_score_sha256"]
            != edit_hook["source_score_sha256"]
            or edit_reference["selected_canonical_ids_sha256"]
            != edit_hook["selected_canonical_ids_sha256"]
            or _positive_int(
                edit_reference["selection_count"], "counted AP edit selection count"
            )
            != edit_count
        ):
            raise ValueError("counted AP edit reference count differs from its artifact")
        bindings = (
            ("ap_edit_ids.npz", "counted_artifact_sha256"),
            ("ap_edit_source_score.npz", "counted_source_score_sha256"),
            (
                "ap_edit_reference_manifest.json",
                "counted_reference_manifest_sha256",
            ),
        )
        for name, key in bindings:
            if sha256_bytes(handle.read(name)) != edit_hook[key]:
                raise ValueError(f"counted AP edit artifact binding mismatch: {name}")
        if (
            edit_hook["artifact_sha256"] != edit_hook["counted_artifact_sha256"]
            or edit_hook["source_score_sha256"]
            != edit_hook["counted_source_score_sha256"]
            or edit_hook["reference_manifest_sha256"]
            != edit_hook["counted_reference_manifest_sha256"]
            or edit_hook["reference_manifest_sha256"]
            != sha256_bytes(edit_reference_payload)
        ):
            raise ValueError("counted AP edit source/reference hash closure is invalid")
    return {
        "training_config_sha256": training_config_sha,
        "training_step": int(producer["training_step"]),
        "producer_rate": rate_index,
        "producer_rd_lambda": rd_lambda,
        "compression_seed": compression_seed,
        "producer_receipt_sha256": sha256_bytes(producer_payload),
        "payload_manifest_sha256": sha256_bytes(payload_manifest_payload),
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "codec_stream_count": len(expected_streams),
    }


def validate_gop_archive(
    archive: Path, scene: str, method: str, gop_id: int
) -> Dict[str, Any]:
    """Verify one complete counted 60-frame GOP archive without extraction."""

    if scene not in ALL_SCENES or gop_id not in range(5):
        raise ValueError("unsupported scene/GOP identity")
    handle, rows = _safe_zip_rows(archive)
    try:
        config = _read_zip_json(handle, rows, "decoder_config.json")
        if set(config) != DECODER_CONFIG_FIELDS or config.get("schema") != DECODER_CONFIG_SCHEMA:
            raise ValueError("unsupported GOP decoder config schema")
        if config.get("scene") != scene:
            raise ValueError("GOP decoder config scene mismatch")
        if config.get("variant") != method:
            raise ValueError("GOP decoder config method/variant mismatch")
        if _positive_int(config.get("GOP_size"), "GOP size") != 60 or _nonnegative_int(
            config.get("start_frame"), "GOP start frame"
        ) != GOP_STARTS[gop_id]:
            raise ValueError("GOP decoder config frame range mismatch")
        census = _read_zip_json(handle, rows, "byte_census.json")
        if set(census) != {
            "schema",
            "root",
            "file_count",
            "raw_bytes",
            "files",
            "self_exclusion",
        } or census.get("schema") != "h007.container_byte_census.v1":
            raise ValueError("unsupported inner GOP byte census")
        census_rows = census.get("files")
        if not isinstance(census_rows, list):
            raise ValueError("inner GOP byte census lacks file rows")
        file_count = _nonnegative_int(census.get("file_count"), "inner GOP file count")
        raw_bytes = _nonnegative_int(census.get("raw_bytes"), "inner GOP raw bytes")
        declared = {}
        for row in census_rows:
            if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
                raise ValueError("inner GOP census row fields are unexpected")
            name = _nonempty_string(row["path"], "inner GOP census path")
            if (
                not name
                or name in declared
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                or "\\" in name
            ):
                raise ValueError("inner GOP census path is unsafe or duplicated")
            _nonnegative_int(row["bytes"], f"inner GOP member {name} bytes")
            declared[name] = row
        actual = set(rows) - {"byte_census.json"}
        if set(declared) != actual:
            raise ValueError("inner GOP members differ from counted byte census")
        if file_count != len(declared) or raw_bytes != sum(
            row["bytes"] for row in declared.values()
        ):
            raise ValueError("inner GOP census totals mismatch")
        expected_rows = []
        for name in sorted(declared):
            row = declared[name]
            payload = handle.read(name)
            if row["bytes"] != len(payload) or _require_sha256(
                row["sha256"], "inner GOP census SHA-256"
            ) != sha256_bytes(payload):
                raise ValueError(f"inner GOP census mismatch: {name}")
            expected_rows.append(
                {"path": name, "bytes": len(payload), "sha256": sha256_bytes(payload)}
            )
        expected_census = {
            "schema": "h007.container_byte_census.v1",
            "root": "rank0",
            "file_count": len(expected_rows),
            "raw_bytes": sum(row["bytes"] for row in expected_rows),
            "files": expected_rows,
            "self_exclusion": "byte_census.json is counted by the archive but cannot hash itself",
        }
        if census != expected_census:
            raise ValueError("inner GOP byte census is not the exact recomputed object")
        decoder_closure = _validate_gifstream_decoder_closure(
            handle, rows, config, scene, method, gop_id
        )
        archive_payload = archive.read_bytes()
        member_payloads = {name: handle.read(name) for name in rows}
        if _canonical_zip_bytes(member_payloads) != archive_payload:
            raise ValueError("inner GOP ZIP bytes/metadata are not canonical")
        return {
            "gop_id": int(gop_id),
            "start_frame": int(GOP_STARTS[gop_id]),
            "frame_count": 60,
            "path": f"gops/gop_{gop_id}.zip",
            "bytes": len(archive_payload),
            "sha256": sha256_bytes(archive_payload),
            "decoder_config_sha256": sha256_bytes(handle.read("decoder_config.json")),
            "byte_census_sha256": sha256_bytes(handle.read("byte_census.json")),
            "runtime_manifest_sha256": config.get("runtime_manifest_sha256"),
            "normalized_code_tree_sha256": config.get("normalized_code_tree_sha256"),
            "patch_chain_sha256": list(config.get("patch_chain_sha256") or []),
            "training_config_sha256": decoder_closure[
                "training_config_sha256"
            ],
            "training_step": decoder_closure["training_step"],
            "producer_rate": decoder_closure["producer_rate"],
            "producer_rd_lambda": decoder_closure["producer_rd_lambda"],
            "compression_seed": decoder_closure["compression_seed"],
            "producer_receipt_sha256": decoder_closure[
                "producer_receipt_sha256"
            ],
            "payload_manifest_sha256": decoder_closure[
                "payload_manifest_sha256"
            ],
            "encode_seconds": decoder_closure["encode_seconds"],
            "decode_seconds": decoder_closure["decode_seconds"],
            "codec_stream_count": decoder_closure["codec_stream_count"],
        }
    finally:
        handle.close()


def _directory_census(root: Path) -> Dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"sequence-container symlink is forbidden: {path}")
        if not path.is_file() or path.name == "byte_census.json":
            continue
        payload = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    return {
        "schema": SEQUENCE_CENSUS_SCHEMA,
        "files": rows,
        "payload_bytes": sum(int(row["bytes"]) for row in rows),
    }


def deterministic_zip(root: Path, output: Path) -> Dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"sequence-container symlink is forbidden: {path}")
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            handle.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    payload = output.read_bytes()
    return {"bytes": len(payload), "sha256": sha256_bytes(payload)}


def build_sequence_container(
    *,
    scene: str,
    method: str,
    gop_archives: Sequence[Path],
    output: Path,
    training_config_sha256: str,
    seed: int,
) -> Dict[str, Any]:
    if scene not in ALL_SCENES:
        raise ValueError("sequence scene is not in the frozen six-scene universe")
    if len(gop_archives) != 5:
        raise ValueError("a 300-frame sequence requires exactly five GOP archives")
    training_config_sha256 = _require_sha256(
        training_config_sha256, "training config SHA-256"
    )
    audits = [
        validate_gop_archive(Path(path), scene, method, gop_id)
        for gop_id, path in enumerate(gop_archives)
    ]
    producer_training_configs = {
        row["training_config_sha256"] for row in audits
    }
    if producer_training_configs != {training_config_sha256}:
        raise ValueError(
            "five GOP producer receipts do not bind the requested training config"
        )
    producer_seeds = {row["compression_seed"] for row in audits}
    if producer_seeds != {int(seed)}:
        raise ValueError(
            "sequence seed differs from its five producer/decoder compression seeds"
        )
    if len({row["training_step"] for row in audits}) != 1:
        raise ValueError("five GOP archives do not share one terminal training step")
    if len({row["producer_rate"] for row in audits}) != 1 or len(
        {row["producer_rd_lambda"] for row in audits}
    ) != 1:
        raise ValueError("five GOP archives do not share one frozen rate configuration")
    provenance = {
        (
            tuple(row["patch_chain_sha256"]),
            row["runtime_manifest_sha256"],
            row["normalized_code_tree_sha256"],
        )
        for row in audits
    }
    if len(provenance) != 1:
        raise ValueError("five GOP archives do not share one runtime provenance")
    with tempfile.TemporaryDirectory(prefix="h007_sequence_") as temporary:
        root = Path(temporary) / "sequence"
        (root / "gops").mkdir(parents=True)
        for gop_id, source in enumerate(gop_archives):
            shutil.copyfile(source, root / f"gops/gop_{gop_id}.zip")
        manifest = {
            "schema": SEQUENCE_SCHEMA,
            "scene": scene,
            "method": method,
            "frame_count": 300,
            "gop_size": 60,
            "gop_count": 5,
            "gop_starts": list(GOP_STARTS),
            "training_config_sha256": training_config_sha256,
            "seed": int(seed),
            "gops": audits,
            "outcome_fields_read": [],
        }
        (root / "sequence_manifest.json").write_bytes(canonical_json_bytes(manifest))
        census = _directory_census(root)
        (root / "byte_census.json").write_bytes(canonical_json_bytes(census))
        outer = deterministic_zip(root, output)
    receipt = {
        "schema": "h007.gifstream_sequence_build_receipt.v1",
        "scene": scene,
        "method": method,
        "archive": str(output.resolve()),
        "archive_bytes": int(outer["bytes"]),
        "archive_sha256": outer["sha256"],
        "sequence_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "byte_census_sha256": sha256_bytes(canonical_json_bytes(census)),
        "gops": audits,
        "outcome_fields_read": [],
    }
    output.with_suffix(output.suffix + ".receipt.json").write_bytes(
        canonical_json_bytes(receipt)
    )
    validate_sequence_container(output, expected_scene=scene, expected_method=method)
    return receipt


def validate_sequence_container(
    archive: Path,
    *,
    expected_scene: Optional[str] = None,
    expected_method: Optional[str] = None,
) -> Dict[str, Any]:
    handle, rows = _safe_zip_rows(archive)
    try:
        manifest = _read_zip_json(handle, rows, "sequence_manifest.json")
        census = _read_zip_json(handle, rows, "byte_census.json")
        if manifest.get("schema") != SEQUENCE_SCHEMA or census.get(
            "schema"
        ) != SEQUENCE_CENSUS_SCHEMA:
            raise ValueError("unsupported sequence manifest/census schema")
        if set(manifest) != {
            "schema",
            "scene",
            "method",
            "frame_count",
            "gop_size",
            "gop_count",
            "gop_starts",
            "training_config_sha256",
            "seed",
            "gops",
            "outcome_fields_read",
        }:
            raise ValueError("sequence manifest fields are incomplete or unexpected")
        if set(census) != {"schema", "files", "payload_bytes"}:
            raise ValueError("sequence census fields are incomplete or unexpected")
        expected_outer_members = {
            "sequence_manifest.json",
            "byte_census.json",
            *(f"gops/gop_{index}.zip" for index in range(5)),
        }
        if set(rows) != expected_outer_members:
            raise ValueError("outer sequence members differ from the exact contract")
        _require_sha256(manifest["training_config_sha256"], "sequence training config SHA-256")
        if manifest["outcome_fields_read"] != []:
            raise ValueError("sequence construction read outcome fields")
        if expected_scene is not None and manifest.get("scene") != expected_scene:
            raise ValueError("sequence scene mismatch")
        if expected_method is not None and manifest.get("method") != expected_method:
            raise ValueError("sequence method mismatch")
        if (
            _positive_int(manifest.get("frame_count"), "sequence frame count") != 300
            or _positive_int(manifest.get("gop_count"), "sequence GOP count") != 5
            or _positive_int(manifest.get("gop_size"), "sequence GOP size") != 60
            or not isinstance(manifest.get("gop_starts"), list)
            or any(type(value) is not int for value in manifest["gop_starts"])
            or manifest["gop_starts"] != list(GOP_STARTS)
        ):
            raise ValueError("sequence frame/GOP contract mismatch")
        census_rows = census.get("files")
        if not isinstance(census_rows, list):
            raise ValueError("outer sequence census lacks file rows")
        declared = {}
        for row in census_rows:
            if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
                raise ValueError("outer sequence census row fields are unexpected")
            name = _nonempty_string(row["path"], "outer sequence census path")
            if (
                not name
                or name in declared
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                or "\\" in name
            ):
                raise ValueError("outer sequence census path is unsafe or duplicated")
            _nonnegative_int(row["bytes"], f"outer sequence member {name} bytes")
            declared[name] = row
        actual = set(rows) - {"byte_census.json"}
        if set(declared) != actual:
            raise ValueError("outer sequence members differ from byte census")
        for name, row in declared.items():
            payload = handle.read(name)
            if len(payload) != row["bytes"] or sha256_bytes(payload) != row[
                "sha256"
            ]:
                raise ValueError(f"outer sequence census mismatch: {name}")
        if _nonnegative_int(census["payload_bytes"], "outer payload bytes") != sum(
            row["bytes"] for row in declared.values()
        ):
            raise ValueError("outer sequence payload-byte total mismatch")
        expected_census = {
            "schema": SEQUENCE_CENSUS_SCHEMA,
            "files": [
                {
                    "path": name,
                    "bytes": len(handle.read(name)),
                    "sha256": sha256_bytes(handle.read(name)),
                }
                for name in sorted(actual)
            ],
            "payload_bytes": sum(len(handle.read(name)) for name in actual),
        }
        if census != expected_census:
            raise ValueError("outer sequence byte census is not the exact recomputed object")
        gops = manifest.get("gops")
        if not isinstance(gops, list) or len(gops) != 5:
            raise ValueError("sequence manifest lacks five GOP receipts")
        if {
            row.get("training_config_sha256") for row in gops
        } != {manifest["training_config_sha256"]}:
            raise ValueError(
                "sequence training config differs from counted producer receipts"
            )
        sequence_seed = _nonnegative_int(manifest["seed"], "sequence seed")
        if {
            _nonnegative_int(row.get("compression_seed"), "GOP compression seed")
            for row in gops
        } != {sequence_seed}:
            raise ValueError(
                "sequence seed differs from counted producer compression seeds"
            )
        if len(
            {
                _nonnegative_int(row.get("training_step"), "GOP training step")
                for row in gops
            }
        ) != 1:
            raise ValueError("sequence GOPs do not share one terminal training step")
        if len(
            {
                _nonnegative_int(row.get("producer_rate"), "GOP producer rate")
                for row in gops
            }
        ) != 1 or len(
            {
                _finite_number(
                    row.get("producer_rd_lambda"), "GOP producer RD lambda", positive=True
                )
                for row in gops
            }
        ) != 1:
            raise ValueError("sequence GOPs do not share one frozen rate configuration")
        for gop_id, row in enumerate(gops):
            name = f"gops/gop_{gop_id}.zip"
            payload = handle.read(name)
            if (
                row.get("path") != name
                or _nonnegative_int(row.get("gop_id"), "sequence GOP ID") != gop_id
                or _nonnegative_int(row.get("start_frame"), "sequence GOP start")
                != GOP_STARTS[gop_id]
                or _positive_int(row.get("bytes"), "sequence GOP bytes") != len(payload)
                or row.get("sha256") != sha256_bytes(payload)
            ):
                raise ValueError(f"sequence GOP receipt mismatch: {gop_id}")
            with tempfile.TemporaryDirectory(prefix="h007_embedded_gop_") as temporary:
                inner = Path(temporary) / f"gop_{gop_id}.zip"
                inner.write_bytes(payload)
                actual_inner = validate_gop_archive(
                    inner,
                    _nonempty_string(manifest["scene"], "sequence scene"),
                    _nonempty_string(manifest["method"], "sequence method"),
                    gop_id,
                )
            expected_inner = dict(row)
            actual_inner["path"] = expected_inner["path"]
            if actual_inner != expected_inner:
                raise ValueError(f"embedded GOP audit differs from sequence manifest: {gop_id}")
        payload = archive.read_bytes()
        with tempfile.TemporaryDirectory(prefix="h007_outer_canonical_") as temporary:
            canonical_root = Path(temporary) / "sequence"
            for name in sorted(rows):
                destination = canonical_root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(handle.read(name))
            canonical_archive = Path(temporary) / "canonical.zip"
            deterministic_zip(canonical_root, canonical_archive)
            if canonical_archive.read_bytes() != payload:
                raise ValueError("outer sequence ZIP is not the deterministic canonical archive")
        return {
            "schema": "h007.gifstream_sequence_validation.v1",
            "scene": manifest["scene"],
            "method": manifest["method"],
            "archive_bytes": len(payload),
            "archive_sha256": sha256_bytes(payload),
            "sequence_manifest_sha256": sha256_bytes(
                handle.read("sequence_manifest.json")
            ),
            "training_config_sha256": manifest["training_config_sha256"],
            "seed": _nonnegative_int(manifest["seed"], "sequence seed"),
            "producer_rate": _nonnegative_int(
                gops[0]["producer_rate"], "sequence producer rate"
            ),
            "producer_rd_lambda": _finite_number(
                gops[0]["producer_rd_lambda"],
                "sequence producer RD lambda",
                positive=True,
            ),
            "training_step": _nonnegative_int(
                gops[0]["training_step"], "sequence training step"
            ),
            "gop_count": 5,
            "frame_count": 300,
            "gops": gops,
            "outcome_fields_read": [],
        }
    finally:
        handle.close()


def _load_registry(path: Path) -> List[Dict[str, Any]]:
    allowed = {
        "scene",
        "method",
        "point_id",
        "archive",
        "eligibility_receipt",
        "training_config_sha256",
        "seed",
    }
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("candidate registry must end with one canonical newline")
    rows = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), 1):
        if line == b"\n" or not line.endswith(b"\n"):
            raise ValueError("candidate registry contains a blank or unterminated row")
        row = _strict_canonical_json(
            line[:-1], f"candidate registry row {line_number}"
        )
        if set(row) != allowed:
            raise ValueError(f"candidate registry row {line_number} has unexpected fields")
        rows.append(row)
    if not rows:
        raise ValueError("candidate registry is empty")
    return rows


def _bound_relative(base: Path, declared: Any, label: str) -> Path:
    declared_text = _nonempty_string(declared, f"{label} path")
    relative = Path(declared_text)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in declared_text
    ):
        raise ValueError(f"{label} path is not a safe relative path")
    raw = base / relative
    if raw.is_symlink():
        raise ValueError(f"{label} symlink is forbidden")
    resolved = raw.resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes its evidence root") from error
    return raw


def _eligible_registry_row(row: Mapping[str, Any], base: Path) -> Dict[str, Any]:
    for name in ("scene", "method", "point_id"):
        _nonempty_string(row[name], f"candidate registry {name}")
    raw_archive = _bound_relative(base, row["archive"], "candidate sequence archive")
    raw_eligibility = _bound_relative(
        base, row["eligibility_receipt"], "candidate eligibility receipt"
    )
    if raw_archive.is_symlink():
        raise ValueError("candidate sequence archive symlink is forbidden")
    if raw_eligibility.is_symlink() or not raw_eligibility.is_file():
        raise ValueError("candidate eligibility receipt is unavailable or a symlink")
    archive = raw_archive.resolve()
    eligibility_path = raw_eligibility.resolve()
    eligibility_payload = eligibility_path.read_bytes()
    eligibility = _strict_canonical_json(
        eligibility_payload, "H-SOTA eligibility receipt"
    )
    required = {
        "schema",
        "scene",
        "method",
        "point_id",
        "source",
        "source_sha256",
    }
    if not isinstance(eligibility, dict) or set(eligibility) != required:
        raise ValueError("H-SOTA eligibility receipt fields are incomplete or unexpected")
    if eligibility["schema"] != ELIGIBILITY_SCHEMA:
        raise ValueError("candidate eligibility request schema is unsupported")
    for key in ("scene", "method", "point_id"):
        if eligibility[key] != row[key]:
            raise ValueError(f"candidate eligibility identity mismatch: {key}")
    source_sha = _require_sha256(
        eligibility["source_sha256"], "eligibility source SHA-256"
    )
    raw_source = _bound_relative(
        eligibility_path.parent, eligibility["source"], "eligibility source evidence"
    )
    if raw_source.is_symlink() or not raw_source.is_file():
        raise ValueError("eligibility source evidence is unavailable or a symlink")
    source_path = raw_source.resolve()
    source_payload = source_path.read_bytes()
    if sha256_bytes(source_payload) != source_sha:
        raise ValueError("eligibility source evidence SHA-256 mismatch")
    validation = validate_sequence_container(
        archive,
        expected_scene=row["scene"],
        expected_method=row["method"],
    )
    training_sha = _require_sha256(
        row["training_config_sha256"], "registry training config SHA-256"
    )
    if (
        validation["training_config_sha256"] != training_sha
        or _nonnegative_int(validation["seed"], "validated sequence seed")
        != _nonnegative_int(row["seed"], "registry sequence seed")
    ):
        raise ValueError("candidate registry differs from sequence training config/seed")
    evidence_audit = _validate_ordinary_rate_quality_evidence(
        source_path=source_path,
        payload=source_payload,
        selected_row=row,
        selected_validation=validation,
    )
    return {
        **dict(row),
        "archive": str(archive),
        "registry_base": str(base.resolve()),
        "archive_registry_relative": _nonempty_string(
            row["archive"], "candidate archive path"
        ),
        "eligibility_receipt_registry_relative": _nonempty_string(
            row["eligibility_receipt"], "candidate eligibility path"
        ),
        "eligibility_receipt_path": str(eligibility_path),
        "eligibility_source_path": str(source_path),
        "archive_bytes": _positive_int(
            validation["archive_bytes"], "validated archive bytes"
        ),
        "archive_sha256": validation["archive_sha256"],
        "eligibility_receipt_sha256": sha256_bytes(eligibility_payload),
        "eligibility_source_sha256": source_sha,
        "eligibility_recomputed": evidence_audit,
    }


def revalidate_selected_eligibility(selected: Mapping[str, Any]) -> Dict[str, Any]:
    """Reopen a frozen selected row from its original registry trust root."""

    required = {
        "scene",
        "method",
        "point_id",
        "registry_base",
        "archive_registry_relative",
        "eligibility_receipt_registry_relative",
        "training_config_sha256",
        "seed",
    }
    if not required.issubset(selected):
        raise ValueError("selected row lacks its eligibility source bindings")
    base = Path(
        _nonempty_string(selected["registry_base"], "selected registry base")
    )
    row = {
        "scene": selected["scene"],
        "method": selected["method"],
        "point_id": selected["point_id"],
        "archive": selected["archive_registry_relative"],
        "eligibility_receipt": selected[
            "eligibility_receipt_registry_relative"
        ],
        "training_config_sha256": selected["training_config_sha256"],
        "seed": selected["seed"],
    }
    return _eligible_registry_row(row, base)


def _load_runtime_provenance_module():
    path = Path(__file__).with_name("h007_runtime_provenance.py")
    spec = importlib.util.spec_from_file_location(
        "h007_runtime_provenance_for_sequence", path
    )
    if spec is None or spec.loader is None:
        raise ValueError("runtime-provenance verifier cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ORDINARY_EVALUATOR_MODULES: Dict[str, Any] = {}


def _load_ordinary_evaluator_module(path: Path):
    key = str(path.resolve())
    cached = _ORDINARY_EVALUATOR_MODULES.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        "h007_ordinary_rate_quality_replay", path
    )
    if spec is None or spec.loader is None:
        raise ValueError("ordinary metric evaluator cannot be loaded for replay")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "recompute_receipt_metrics", None)):
        raise ValueError("ordinary metric evaluator lacks its replay entrypoint")
    _ORDINARY_EVALUATOR_MODULES[key] = module
    return module


def _validate_source_data_manifest(path: Path, expected_scene: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("source-data manifest is unavailable or a symlink")
    payload = path.read_bytes()
    value = _strict_canonical_json(payload, "source-data manifest")
    required = {
        "schema",
        "scene",
        "frame_count",
        "cameras",
        "files",
        "outcome_fields_read",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("source-data manifest fields are incomplete or unexpected")
    camera_count = frozen_scene_camera_count(expected_scene)
    cameras = list(frozen_camera_names(expected_scene, ""))
    if (
        value["schema"] != SOURCE_DATA_SCHEMA
        or value["scene"] != expected_scene
        or _positive_int(value["frame_count"], "source-data frame count") != 300
        or value["cameras"] != cameras
        or value["outcome_fields_read"] != []
    ):
        raise ValueError("source-data manifest identity/protocol mismatch")
    files = value["files"]
    if not isinstance(files, list) or len(files) != camera_count * 300:
        raise ValueError("source-data manifest must census the exact scene-specific camera x 300 RGB grid")
    expected_names = [
        f"png/cam{camera:02d}/{frame:05d}.png"
        for camera in range(camera_count)
        for frame in range(1, 301)
    ]
    seen = set()
    observed_names = []
    member_sha256: Dict[str, str] = {}
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("source-data census row fields are unexpected")
        name = _nonempty_string(row["path"], "source-data member path")
        candidate = Path(name)
        if (
            not name
            or candidate.is_absolute()
            or ".." in candidate.parts
            or "\\" in name
            or name in seen
        ):
            raise ValueError("source-data census path is unsafe or duplicated")
        seen.add(name)
        observed_names.append(name)
        raw = _bound_relative(path.parent, name, "source-data member")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(str(raw), flags)
        try:
            opened = os.fstat(fd)
            expected_bytes = _positive_int(row["bytes"], "source-data member bytes")
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise ValueError(f"source-data member must be regular and single-link: {name}")
            if opened.st_size != expected_bytes:
                raise ValueError(f"source-data member census mismatch: {name}")
            chunks = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            member = b"".join(chunks)
            if (
                len(member) != expected_bytes
                or sha256_bytes(member) != _require_sha256(row["sha256"], "data member SHA-256")
            ):
                raise ValueError(f"source-data member census mismatch: {name}")
        finally:
            os.close(fd)
        member_sha256[name] = _require_sha256(
            row["sha256"], "source-data member SHA-256"
        )
    if observed_names != expected_names:
        raise ValueError("source-data census is not the exact scene-specific camera x frame00001..00300 grid")
    return {
        "manifest_sha256": sha256_bytes(payload),
        "manifest_path": str(path.resolve()),
        "file_count": len(files),
        "scene": expected_scene,
        "member_sha256": member_sha256,
    }


def _finite_number(
    value: Any,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if type(value) is not float:
        raise ValueError(f"{label} is not an exact JSON floating measurement")
    number = value
    if (
        not math.isfinite(number)
        or (positive and number <= 0)
        or (nonnegative and number < 0)
    ):
        raise ValueError(f"{label} is nonfinite or outside its allowed range")
    return number


def validate_eligibility_recomputation_contract(
    value: Mapping[str, Any],
    *,
    expected_scene: str,
    expected_point_id: str,
    expected_source_evidence_sha256: str,
    expected_archive_bytes: int,
    expected_training_config_sha256: str,
    expected_seed: int,
) -> Dict[str, Any]:
    """Shared v4 selector/Stage-02 contract; no embedded PASS shortcut."""

    required = {
        "schema",
        "eligible",
        "ordinary_rate_quality_only",
        "required_point_count",
        "distinct_archive_count",
        "distinct_rate_count",
        "distinct_training_config_count",
        "frozen_rate_lambda_grid",
        "minimum_adjacent_actual_byte_fraction",
        "actual_bytes_by_rate",
        "metrics_recomputed_from_evaluator_receipts",
        "selected_point_id",
        "selected_rate",
        "selected_archive_bytes",
        "selected_training_config_sha256",
        "selected_seed",
        "source_evidence_sha256",
        "source_data",
        "runtime_provenance",
        "evaluator",
        "selected_evaluator_receipt_sha256",
        "selected_evaluator_receipt_path",
        "selected_metrics",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("H-SOTA v4 recomputation fields are incomplete or unexpected")
    if (
        value["schema"] != ELIGIBILITY_RECOMPUTATION_SCHEMA
        or value["eligible"] is not True
        or value["ordinary_rate_quality_only"] is not True
        or value["metrics_recomputed_from_evaluator_receipts"] is not True
        or any(
            _positive_int(value[name], f"H-SOTA v4 {name}") != 4
            for name in (
                "required_point_count",
                "distinct_archive_count",
                "distinct_rate_count",
                "distinct_training_config_count",
            )
        )
    ):
        raise ValueError("H-SOTA v4 recomputation gate/count closure is invalid")
    expected_grid = {
        str(index): FROZEN_ORDINARY_RATE_LAMBDAS[index]
        for index in sorted(FROZEN_ORDINARY_RATE_LAMBDAS)
    }
    if value["frozen_rate_lambda_grid"] != expected_grid or _finite_number(
        value["minimum_adjacent_actual_byte_fraction"],
        "minimum adjacent actual-byte fraction",
        positive=True,
    ) != MIN_ADJACENT_RATE_BYTE_FRACTION:
        raise ValueError("H-SOTA v4 frozen rate grid differs from the contract")
    actual_bytes = value["actual_bytes_by_rate"]
    if not isinstance(actual_bytes, dict) or set(actual_bytes) != set(expected_grid):
        raise ValueError("H-SOTA v4 actual-byte grid is incomplete")
    ordered_bytes = []
    for index in sorted(FROZEN_ORDINARY_RATE_LAMBDAS):
        item = actual_bytes[str(index)]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError("H-SOTA v4 actual-byte grid contains an invalid count")
        ordered_bytes.append(item)
    for left, right in zip(ordered_bytes, ordered_bytes[1:]):
        if right >= left or (left - right) / float(right) < MIN_ADJACENT_RATE_BYTE_FRACTION:
            raise ValueError("H-SOTA v4 actual-byte grid lacks frozen separation")
    selected_rate = value["selected_rate"]
    selected_archive_bytes = _positive_int(
        value["selected_archive_bytes"], "selected archive bytes"
    )
    selected_seed = _nonnegative_int(value["selected_seed"], "selected seed")
    if (
        isinstance(selected_rate, bool)
        or not isinstance(selected_rate, int)
        or selected_rate not in FROZEN_ORDINARY_RATE_LAMBDAS
        or value["selected_point_id"] != expected_point_id
        or selected_archive_bytes != expected_archive_bytes
        or actual_bytes[str(selected_rate)] != expected_archive_bytes
        or value["selected_training_config_sha256"]
        != _require_sha256(
            expected_training_config_sha256, "selected training config SHA-256"
        )
        or selected_seed != expected_seed
        or value["source_evidence_sha256"]
        != _require_sha256(
            expected_source_evidence_sha256, "selected eligibility source SHA-256"
        )
    ):
        raise ValueError("H-SOTA v4 selected point is not closed to its registry row")

    source_data = value["source_data"]
    if not isinstance(source_data, dict) or set(source_data) != {
        "manifest_sha256",
        "manifest_path",
        "file_count",
        "scene",
    }:
        raise ValueError("H-SOTA v4 source-data closure is incomplete")
    source_path = Path(
        _nonempty_string(source_data["manifest_path"], "source-data manifest path")
    )
    if (
        source_data["scene"] != expected_scene
        or _positive_int(source_data["file_count"], "source-data file count")
        != frozen_scene_camera_count(expected_scene) * 300
        or not source_path.is_absolute()
        or ".." in source_path.parts
    ):
        raise ValueError("H-SOTA v4 source-data identity is invalid")
    _require_sha256(source_data["manifest_sha256"], "source-data manifest SHA-256")

    runtime = value["runtime_provenance"]
    if not isinstance(runtime, dict) or set(runtime) != {
        "schema",
        "manifest_sha256",
        "official_commit",
        "patch_sha256",
        "normalized_code_tree",
    }:
        raise ValueError("H-SOTA v4 runtime closure is incomplete")
    patches = runtime["patch_sha256"]
    tree = runtime["normalized_code_tree"]
    if (
        runtime["schema"] != "h007.ap_gifstream.runtime_provenance.v1"
        or runtime["official_commit"] != OFFICIAL_COMMIT
        or not isinstance(patches, list)
        or len(patches) != 9
        or not isinstance(tree, dict)
        or set(tree)
        != {
            "schema",
            "normalization",
            "roots",
            "root_files",
            "suffixes",
            "special_names",
            "file_count",
            "sha256",
        }
        or tree["schema"] != "h007.normalized_code_tree.v1"
        or tree["normalization"]
        != "sorted-posix-path+lf-bytes+uint64le-lengths"
        or tree["roots"] != ["examples", "gsplat", "third_party"]
        or tree["root_files"] != ["setup.py"]
        or tree["special_names"] != ["CMakeLists.txt"]
        or isinstance(tree["file_count"], bool)
        or not isinstance(tree["file_count"], int)
        or tree["file_count"] <= 0
    ):
        raise ValueError("H-SOTA v4 runtime/tree identity is invalid")
    _require_sha256(runtime["manifest_sha256"], "runtime manifest SHA-256")
    _require_sha256(tree["sha256"], "normalized tree SHA-256")
    for index, digest in enumerate(patches):
        _require_sha256(digest, f"runtime patch {index} SHA-256")

    evaluator = value["evaluator"]
    if (
        not isinstance(evaluator, dict)
        or set(evaluator) != {"relative_path", "sha256"}
        or evaluator["relative_path"] != ORDINARY_EVALUATOR_RELATIVE_PATH
    ):
        raise ValueError("H-SOTA v4 evaluator closure is incomplete")
    _require_sha256(evaluator["sha256"], "ordinary evaluator SHA-256")
    _require_sha256(
        value["selected_evaluator_receipt_sha256"],
        "selected evaluator receipt SHA-256",
    )
    receipt_path = Path(
        _nonempty_string(
            value["selected_evaluator_receipt_path"],
            "selected evaluator receipt path",
        )
    )
    if not receipt_path.is_absolute() or ".." in receipt_path.parts:
        raise ValueError("H-SOTA v4 selected evaluator receipt path is invalid")
    metrics = value["selected_metrics"]
    if not isinstance(metrics, dict) or set(metrics) != {
        "psnr",
        "ssim",
        "lpips",
        "encode_seconds",
        "decode_seconds",
        "render_fps",
    }:
        raise ValueError("H-SOTA v4 selected metrics are incomplete")
    psnr = _finite_number(metrics["psnr"], "selected PSNR", positive=True)
    ssim = _finite_number(metrics["ssim"], "selected SSIM")
    lpips = _finite_number(metrics["lpips"], "selected LPIPS")
    for name in ("encode_seconds", "decode_seconds", "render_fps"):
        _finite_number(metrics[name], f"selected {name}", positive=True)
    if psnr <= 0 or not 0 <= ssim <= 1 or not 0 <= lpips <= 1:
        raise ValueError("H-SOTA v4 selected image metrics are outside range")
    return dict(value)


def _validate_rate_quality_evaluator_receipt(
    *,
    receipt_path: Path,
    receipt_sha256: str,
    point: Mapping[str, Any],
    evidence: Mapping[str, Any],
    validation: Mapping[str, Any],
    data_audit: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
    evaluator_path: Path,
) -> Dict[str, Any]:
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("rate-quality evaluator receipt is unavailable or a symlink")
    payload = receipt_path.read_bytes()
    if sha256_bytes(payload) != _require_sha256(
        receipt_sha256, "rate-quality evaluator receipt SHA-256"
    ):
        raise ValueError("rate-quality evaluator receipt SHA-256 mismatch")
    receipt = _strict_canonical_json(payload, "rate-quality evaluator receipt")
    required = {
        "schema",
        "scene",
        "method",
        "point_id",
        "sequence_archive",
        "archive_sha256",
        "archive_bytes",
        "training_config_sha256",
        "seed",
        "source_data_manifest",
        "source_data_manifest_sha256",
        "runtime_provenance_manifest",
        "runtime_provenance_manifest_sha256",
        "clean_decoder_relative_path",
        "clean_decoder_sha256",
        "generated_predictions_root",
        "evaluator_relative_path",
        "evaluator_sha256",
        "metric_device",
        "metric_protocol",
        "frame_metrics",
        "timing_trials",
        "outcome_fields_read",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise ValueError("rate-quality evaluator receipt fields are incomplete or unexpected")
    bound_sequence = _bound_relative(
        receipt_path.parent,
        receipt["sequence_archive"],
        "rate-quality receipt sequence archive",
    )
    bound_source = _bound_relative(
        receipt_path.parent,
        receipt["source_data_manifest"],
        "rate-quality receipt source-data manifest",
    )
    bound_runtime = _bound_relative(
        receipt_path.parent,
        receipt["runtime_provenance_manifest"],
        "rate-quality receipt runtime manifest",
    )
    generated_root = _bound_relative(
        receipt_path.parent,
        receipt["generated_predictions_root"],
        "generated prediction root",
    )
    clean_decoder_relative = _nonempty_string(
        receipt["clean_decoder_relative_path"], "clean decoder relative path"
    )
    if clean_decoder_relative != "examples/h007_clean_decode_gifstream.py":
        raise ValueError("rate-quality receipt names an untrusted clean decoder")
    clean_decoder = evaluator_path.parents[1] / clean_decoder_relative
    if (
        bound_sequence.is_symlink()
        or not bound_sequence.is_file()
        or sha256_file(bound_sequence) != validation["archive_sha256"]
        or bound_source.resolve()
        != Path(
            _nonempty_string(data_audit["manifest_path"], "source-data manifest path")
        ).resolve()
        or bound_source.is_symlink()
        or not bound_source.is_file()
        or sha256_file(bound_source) != data_audit["manifest_sha256"]
        or bound_runtime.is_symlink()
        or not bound_runtime.is_file()
        or sha256_file(bound_runtime) != runtime_receipt["manifest_sha256"]
        or generated_root.is_symlink()
        or not generated_root.is_dir()
        or clean_decoder.is_symlink()
        or not clean_decoder.is_file()
        or sha256_file(clean_decoder)
        != _require_sha256(receipt["clean_decoder_sha256"], "clean decoder SHA-256")
    ):
        raise ValueError("rate-quality evaluator archive/source/runtime/render binding mismatch")
    if (
        receipt["schema"] != EVALUATOR_RECEIPT_SCHEMA
        or receipt["scene"] != evidence["scene"]
        or receipt["method"] != evidence["method"]
        or receipt["point_id"] != point["point_id"]
        or receipt["archive_sha256"] != validation["archive_sha256"]
        or _positive_int(receipt["archive_bytes"], "evaluator archive bytes")
        != _positive_int(validation["archive_bytes"], "validated archive bytes")
        or receipt["training_config_sha256"]
        != validation["training_config_sha256"]
        or _nonnegative_int(receipt["seed"], "evaluator seed")
        != _nonnegative_int(validation["seed"], "validated evaluator seed")
        or receipt["source_data_manifest_sha256"]
        != data_audit["manifest_sha256"]
        or receipt["runtime_provenance_manifest_sha256"]
        != runtime_receipt["manifest_sha256"]
        or receipt["evaluator_relative_path"]
        != evidence["evaluator_relative_path"]
        or receipt["evaluator_sha256"] != evidence["evaluator_sha256"]
        or not _nonempty_string(receipt["metric_device"], "metric device")
        or receipt["metric_protocol"] != ORDINARY_METRIC_PROTOCOL
        or receipt["outcome_fields_read"]
        != ["ordinary_unedited_fidelity", "real_container_accounting"]
    ):
        raise ValueError("rate-quality evaluator receipt identity/provenance mismatch")

    frame_rows = receipt["frame_metrics"]
    if not isinstance(frame_rows, list) or len(frame_rows) != 300:
        raise ValueError("evaluator receipt must contain exactly 300 frame metric rows")
    frame_required = {
        "frame",
        "prediction",
        "reference",
        "prediction_bytes",
        "prediction_sha256",
        "reference_sha256",
        "psnr",
        "ssim",
        "lpips",
    }
    by_frame: Dict[int, Tuple[float, float, float]] = {}
    prediction_sha_by_frame: Dict[int, str] = {}
    for row in frame_rows:
        if not isinstance(row, dict) or set(row) != frame_required:
            raise ValueError("evaluator frame-metric row fields are unexpected")
        frame = _nonnegative_int(row["frame"], "evaluator frame ID")
        if frame not in range(300) or frame in by_frame:
            raise ValueError("evaluator frame IDs are duplicated or outside 0..299")
        prediction = _bound_relative(
            receipt_path.parent,
            row["prediction"],
            f"evaluator prediction frame {frame}",
        )
        expected_prediction = (
            generated_root / "predictions" / f"frame_{frame:05d}.png"
        )
        try:
            prediction.resolve().relative_to(generated_root.resolve())
        except ValueError as error:
            raise ValueError(
                f"evaluator prediction is outside archive-derived output: frame {frame}"
            ) from error
        if (
            prediction.resolve() != expected_prediction.resolve()
            or any(
                parent.is_symlink()
                for parent in prediction.parents
                if parent != generated_root.parent
                and generated_root.parent in parent.parents
            )
            or prediction.is_symlink()
            or not prediction.is_file()
        ):
            raise ValueError(f"evaluator prediction is unavailable: frame {frame}")
        prediction_payload = prediction.read_bytes()
        if (
            len(prediction_payload)
            != _positive_int(row["prediction_bytes"], "evaluator prediction bytes")
            or sha256_bytes(prediction_payload)
            != _require_sha256(row["prediction_sha256"], "prediction SHA-256")
        ):
            raise ValueError(f"evaluator prediction binding mismatch: frame {frame}")
        expected_reference_sha = data_audit["member_sha256"][
            f"cam00/{frame + 1:05d}.png"
        ]
        reference = _bound_relative(
            receipt_path.parent,
            row["reference"],
            f"evaluator reference frame {frame}",
        )
        expected_reference = (
            Path(
                _nonempty_string(
                    data_audit["manifest_path"], "source-data manifest path"
                )
            ).parent
            / f"cam00/{frame + 1:05d}.png"
        ).resolve()
        if (
            reference.resolve() != expected_reference
            or reference.is_symlink()
            or not reference.is_file()
            or row["reference_sha256"] != expected_reference_sha
            or sha256_file(reference) != expected_reference_sha
        ):
            raise ValueError(f"evaluator reference binding mismatch: frame {frame}")
        psnr = _finite_number(row["psnr"], "per-frame PSNR", positive=True)
        ssim = _finite_number(row["ssim"], "per-frame SSIM")
        lpips = _finite_number(row["lpips"], "per-frame LPIPS")
        if not 0 <= ssim <= 1 or not 0 <= lpips <= 1:
            raise ValueError(f"per-frame image metric is outside range: frame {frame}")
        by_frame[frame] = (psnr, ssim, lpips)
        prediction_sha_by_frame[frame] = _require_sha256(
            row["prediction_sha256"], "prediction SHA-256"
        )
    if set(by_frame) != set(range(300)):
        raise ValueError("evaluator receipt frame grid is not exactly 0..299")
    replay_audit = _load_ordinary_evaluator_module(evaluator_path).recompute_receipt_metrics(
        receipt_path
    )
    if (
        not isinstance(replay_audit, dict)
        or replay_audit.get("archive_sha256") != validation["archive_sha256"]
        or replay_audit.get("clean_decoder_sha256")
        != receipt["clean_decoder_sha256"]
        or set(replay_audit.get("prediction_sha256", {})) != set(range(300))
        or set(replay_audit.get("decoded_tensor_manifest_sha256", {}))
        != set(range(5))
        or set(replay_audit.get("prediction_camera_binding", {})) != set(range(5))
        or replay_audit.get("prediction_sha256") != prediction_sha_by_frame
    ):
        raise ValueError("frozen evaluator did not replay the exact archive/render chain")
    replayed = replay_audit.get("metrics", {})
    if set(replayed) != set(range(300)):
        raise ValueError("frozen evaluator replay did not return the exact 300-frame grid")
    for frame, declared in by_frame.items():
        replay = replayed[frame]
        for index, name in enumerate(("psnr", "ssim", "lpips")):
            if abs(
                declared[index]
                - _finite_number(
                    replay[name],
                    f"replayed {name}",
                    positive=name == "psnr",
                )
            ) > 1e-9:
                raise ValueError(
                    f"evaluator receipt metric differs from frozen-code replay: frame {frame}/{name}"
                )
    psnr = sum(row[0] for row in by_frame.values()) / 300.0
    ssim = sum(row[1] for row in by_frame.values()) / 300.0
    lpips = sum(row[2] for row in by_frame.values()) / 300.0

    timing_rows = receipt["timing_trials"]
    if not isinstance(timing_rows, list) or len(timing_rows) != 5:
        raise ValueError("evaluator receipt must contain exactly five GOP timing trials")
    timing_required = {
        "gop_id",
        "inner_gop_sha256",
        "encode_seconds",
        "decode_seconds",
        "clean_decode_receipt",
        "clean_decode_receipt_sha256",
        "decoded_splats_sha256",
        "decoded_tensor_manifest_sha256",
        "prediction_camera_binding",
        "rendered_frames",
        "render_elapsed_seconds",
        "render_fps",
    }
    timing_by_gop: Dict[int, Tuple[float, float, int, float]] = {}
    for row in timing_rows:
        if not isinstance(row, dict) or set(row) != timing_required:
            raise ValueError("evaluator timing-trial row fields are unexpected")
        gop_id = _nonnegative_int(row["gop_id"], "timing GOP ID")
        if gop_id not in range(5) or gop_id in timing_by_gop:
            raise ValueError("timing GOP IDs are duplicated or outside 0..4")
        gop_audit = validation["gops"][gop_id]
        encode = _finite_number(row["encode_seconds"], "GOP encode seconds", positive=True)
        decode = _finite_number(row["decode_seconds"], "GOP decode seconds", positive=True)
        if (
            row["inner_gop_sha256"] != gop_audit["sha256"]
            or abs(
                encode
                - _finite_number(
                    gop_audit["encode_seconds"], "counted GOP encode seconds", positive=True
                )
            )
            > 1e-12
            or abs(
                decode
                - _finite_number(
                    gop_audit["decode_seconds"], "counted GOP decode seconds", positive=True
                )
            )
            > 1e-12
        ):
            raise ValueError(f"timing trial differs from counted GOP runtime: {gop_id}")
        clean_path = _bound_relative(
            receipt_path.parent,
            row["clean_decode_receipt"],
            f"clean-decode timing receipt {gop_id}",
        )
        expected_clean_path = (
            generated_root
            / f"gop_{gop_id}"
            / "clean_bundle"
            / "clean_decode_manifest.json"
        )
        if (
            clean_path.resolve() != expected_clean_path.resolve()
            or clean_path.is_symlink()
            or not clean_path.is_file()
        ):
            raise ValueError(f"clean-decode timing receipt is unavailable: {gop_id}")
        clean_payload = clean_path.read_bytes()
        if sha256_bytes(clean_payload) != _require_sha256(
            row["clean_decode_receipt_sha256"], "clean-decode receipt SHA-256"
        ):
            raise ValueError(f"clean-decode timing receipt hash mismatch: {gop_id}")
        clean = _strict_canonical_json(
            clean_payload, f"clean-decode timing receipt {gop_id}"
        )
        decoded_splats_sha = _require_sha256(
            row["decoded_splats_sha256"], "decoded splats SHA-256"
        )
        tensor_manifest_sha = _require_sha256(
            row["decoded_tensor_manifest_sha256"],
            "decoded tensor manifest SHA-256",
        )
        camera_binding = row["prediction_camera_binding"]
        if (
            not isinstance(camera_binding, dict)
            or set(camera_binding)
            != {
                "source_camera",
                "dataset_camera_index",
                "pose_index",
                "camera_key",
                "counted_camera_name",
                "frame_size",
                "local_frames",
            }
            or camera_binding["source_camera"] != "cam00"
            or _nonnegative_int(
                camera_binding["dataset_camera_index"], "dataset camera index"
            )
            != 0
            or _nonnegative_int(camera_binding["pose_index"], "camera pose index")
            != 0
            or not _nonempty_string(
                camera_binding["counted_camera_name"], "counted camera name"
            )
            or len(camera_binding["frame_size"]) != 2
            or any(
                _positive_int(value, "camera frame-size dimension") <= 0
                for value in camera_binding["frame_size"]
            )
            or camera_binding["local_frames"] != list(range(60))
            or replay_audit["prediction_camera_binding"][gop_id]
            != camera_binding
        ):
            raise ValueError(f"ordinary render camera/frame binding mismatch: {gop_id}")
        rendered_frames = _positive_int(row["rendered_frames"], "rendered frame count")
        elapsed = _finite_number(
            row["render_elapsed_seconds"], "render elapsed seconds", positive=True
        )
        fps = _finite_number(row["render_fps"], "render FPS", positive=True)
        clean_render = clean.get("counted_camera_render", {})
        if (
            clean.get("schema") != "h007.clean_decode_result.v2"
            or clean.get("source_sequence_archive_sha256")
            != validation["archive_sha256"]
            or _nonnegative_int(clean.get("source_gop_id"), "clean source GOP ID")
            != gop_id
            or clean.get("source_inner_gop_sha256") != gop_audit["sha256"]
            or clean.get("runtime_provenance") != runtime_receipt
            or clean.get("decoded_splats_sha256") != decoded_splats_sha
            or sha256_bytes(canonical_json_bytes(clean.get("tensors", {})))
            != tensor_manifest_sha
            or replay_audit["decoded_tensor_manifest_sha256"][gop_id]
            != tensor_manifest_sha
            or _positive_int(
                clean_render.get("timed_renders"), "clean timed-render count"
            )
            != rendered_frames
            or abs(
                _finite_number(
                    clean_render.get("seconds"), "clean-render seconds", positive=True
                )
                - elapsed
            )
            > 1e-12
            or abs(
                _finite_number(clean_render.get("fps"), "clean-render FPS", positive=True)
                - fps
            )
            > 1e-12
            or abs(fps - rendered_frames / elapsed) > 1e-12
        ):
            raise ValueError(f"clean-decode timing receipt binding mismatch: {gop_id}")
        timing_by_gop[gop_id] = (encode, decode, rendered_frames, elapsed)
    if set(timing_by_gop) != set(range(5)):
        raise ValueError("evaluator timing grid is not exactly GOP 0..4")
    encode_seconds = sum(row[0] for row in timing_by_gop.values())
    decode_seconds = sum(row[1] for row in timing_by_gop.values())
    rendered_frames = sum(row[2] for row in timing_by_gop.values())
    render_elapsed = sum(row[3] for row in timing_by_gop.values())
    render_fps = rendered_frames / render_elapsed
    return {
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": sha256_bytes(payload),
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "render_fps": render_fps,
        "frame_count": 300,
        "timing_trial_count": 5,
    }


def _validate_ordinary_rate_quality_evidence(
    *,
    source_path: Path,
    payload: bytes,
    selected_row: Mapping[str, Any],
    selected_validation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Recompute eligibility from real data, code and nested-ZIP evidence.

    The request JSON contains no trusted PASS flag.  This function opens every
    referenced archive and every source RGB member, validates the active
    registered nine-stage runtime, and recomputes the ordinary rate/quality accounting.
    """

    evidence = _strict_canonical_json(payload, "ordinary rate-quality evidence")
    required = {
        "schema",
        "scene",
        "method",
        "frame_count",
        "metric_protocol",
        "ordinary_rate_quality_only",
        "outcome_fields_read",
        "source_data_manifest",
        "source_data_manifest_sha256",
        "runtime_provenance_manifest",
        "runtime_provenance_manifest_sha256",
        "runtime_repo_root",
        "evaluator_relative_path",
        "evaluator_sha256",
        "points",
    }
    if not isinstance(evidence, dict) or set(evidence) != required:
        raise ValueError("ordinary rate-quality evidence fields are incomplete or unexpected")
    if (
        evidence["schema"] != ELIGIBILITY_EVIDENCE_SCHEMA
        or evidence["scene"] != selected_row["scene"]
        or evidence["method"] != selected_row["method"]
        or _positive_int(evidence["frame_count"], "ordinary evidence frame count")
        != 300
        or evidence["metric_protocol"] != ORDINARY_METRIC_PROTOCOL
        or evidence["ordinary_rate_quality_only"] is not True
        or evidence["outcome_fields_read"]
        != ["ordinary_unedited_fidelity", "real_container_accounting"]
    ):
        raise ValueError("ordinary rate-quality evidence protocol/identity mismatch")

    raw_data = _bound_relative(
        source_path.parent, evidence["source_data_manifest"], "source-data manifest"
    )
    data_sha = _require_sha256(
        evidence["source_data_manifest_sha256"], "source-data manifest SHA-256"
    )
    if raw_data.is_symlink() or not raw_data.is_file() or sha256_file(raw_data) != data_sha:
        raise ValueError("source-data manifest binding mismatch")
    data_audit = _validate_source_data_manifest(
        raw_data.resolve(),
        _nonempty_string(selected_row["scene"], "selected scene"),
    )

    raw_runtime = _bound_relative(
        source_path.parent,
        evidence["runtime_provenance_manifest"],
        "runtime-provenance manifest",
    )
    runtime_sha = _require_sha256(
        evidence["runtime_provenance_manifest_sha256"],
        "runtime-provenance manifest SHA-256",
    )
    if raw_runtime.is_symlink() or not raw_runtime.is_file():
        raise ValueError("runtime-provenance manifest is unavailable or a symlink")
    runtime_module = _load_runtime_provenance_module()
    runtime_repo_root = Path(
        _nonempty_string(evidence["runtime_repo_root"], "runtime repository root")
    )
    runtime_receipt = runtime_module.verify_runtime_provenance(
        raw_runtime,
        runtime_repo_root,
        runtime_sha,
    )
    if len(runtime_receipt.get("patch_sha256", [])) != 9:
        raise ValueError("ordinary evidence does not bind the registered nine-stage runtime")
    evaluator_relative = _nonempty_string(
        evidence["evaluator_relative_path"], "ordinary evaluator relative path"
    )
    if evaluator_relative != ORDINARY_EVALUATOR_RELATIVE_PATH:
        raise ValueError("ordinary evidence names an untrusted evaluator entrypoint")
    evaluator_path = _bound_relative(
        runtime_repo_root,
        evaluator_relative,
        "ordinary metric evaluator",
    )
    evaluator_sha = _require_sha256(
        evidence["evaluator_sha256"], "ordinary metric evaluator SHA-256"
    )
    if (
        evaluator_path.is_symlink()
        or not evaluator_path.is_file()
        or sha256_file(evaluator_path) != evaluator_sha
    ):
        raise ValueError("ordinary metric evaluator is unavailable or changed")

    points = evidence["points"]
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError("ordinary evidence requires the exact frozen four-point curve")
    point_required = {
        "point_id",
        "archive",
        "archive_bytes",
        "archive_sha256",
        "mb_per_frame",
        "psnr",
        "ssim",
        "lpips",
        "encode_seconds",
        "decode_seconds",
        "render_fps",
        "training_config_sha256",
        "seed",
        "evaluator_receipt",
        "evaluator_receipt_sha256",
    }
    audits: Dict[str, Dict[str, Any]] = {}
    archive_hashes = set()
    archive_rates = set()
    training_config_hashes = set()
    producer_rates: Dict[int, Tuple[int, float, str]] = {}
    for point in points:
        if not isinstance(point, dict) or set(point) != point_required:
            raise ValueError("rate-quality point fields are incomplete or unexpected")
        point_id = _nonempty_string(point["point_id"], "rate-quality point ID")
        if not point_id or point_id in audits:
            raise ValueError("rate-quality point IDs are empty or duplicated")
        raw_archive = _bound_relative(
            source_path.parent, point["archive"], f"rate-quality archive {point_id}"
        )
        if raw_archive.is_symlink() or not raw_archive.is_file():
            raise ValueError(f"rate-quality archive is unavailable or a symlink: {point_id}")
        validation = validate_sequence_container(
            raw_archive.resolve(),
            expected_scene=_nonempty_string(evidence["scene"], "evidence scene"),
            expected_method=_nonempty_string(evidence["method"], "evidence method"),
        )
        archive_sha = _require_sha256(point["archive_sha256"], "point archive SHA-256")
        archive_bytes = _positive_int(point["archive_bytes"], "rate-quality archive bytes")
        point_seed = _nonnegative_int(point["seed"], "rate-quality point seed")
        if (
            validation["archive_sha256"] != archive_sha
            or _positive_int(validation["archive_bytes"], "validated archive bytes")
            != archive_bytes
            or validation["training_config_sha256"]
            != _require_sha256(point["training_config_sha256"], "point training SHA-256")
            or _nonnegative_int(validation["seed"], "validated sequence seed")
            != point_seed
        ):
            raise ValueError(f"rate-quality point differs from its real nested archive: {point_id}")
        if archive_sha in archive_hashes:
            raise ValueError("distinct rate-quality point IDs reuse one sequence archive")
        if archive_bytes in archive_rates:
            raise ValueError("distinct rate-quality point IDs do not define distinct rates")
        archive_hashes.add(archive_sha)
        archive_rates.add(archive_bytes)
        training_sha = _require_sha256(
            point["training_config_sha256"], "point training SHA-256"
        )
        if training_sha in training_config_hashes:
            raise ValueError(
                "distinct rate-quality points reuse one producer training configuration"
            )
        training_config_hashes.add(training_sha)
        producer_rate = _nonnegative_int(
            validation.get("producer_rate"), "validated producer rate"
        )
        producer_lambda = _finite_number(
            validation.get("producer_rd_lambda"),
            "validated producer RD lambda",
            positive=True,
        )
        if (
            producer_rate not in FROZEN_ORDINARY_RATE_LAMBDAS
            or not math.isfinite(producer_lambda)
            or abs(
                producer_lambda
                - FROZEN_ORDINARY_RATE_LAMBDAS[producer_rate]
            )
            > 1e-15
            or producer_rate in producer_rates
        ):
            raise ValueError(
                "rate-quality points do not bind the frozen rate/RD-lambda grid"
            )
        producer_rates[producer_rate] = (
            archive_bytes,
            producer_lambda,
            training_sha,
        )
        mb_per_frame = _finite_number(point["mb_per_frame"], "MB/frame", positive=True)
        expected_mb_per_frame = archive_bytes / 1_000_000.0 / 300.0
        if abs(mb_per_frame - expected_mb_per_frame) > 1e-15:
            raise ValueError(f"rate-quality point MB/frame is not recomputed from bytes: {point_id}")
        receipt_path = _bound_relative(
            source_path.parent,
            point["evaluator_receipt"],
            f"rate-quality evaluator receipt {point_id}",
        )
        evaluator_audit = _validate_rate_quality_evaluator_receipt(
            receipt_path=receipt_path.resolve(),
            receipt_sha256=_require_sha256(
                point["evaluator_receipt_sha256"],
                "rate-quality evaluator receipt SHA-256",
            ),
            point=point,
            evidence=evidence,
            validation=validation,
            data_audit=data_audit,
            runtime_receipt=runtime_receipt,
            evaluator_path=evaluator_path,
        )
        for name in (
            "psnr",
            "ssim",
            "lpips",
            "encode_seconds",
            "decode_seconds",
            "render_fps",
        ):
            declared = _finite_number(
                point[name], name.replace("_", " "), positive=name not in {"ssim", "lpips"}
            )
            if abs(
                declared
                - _finite_number(
                    evaluator_audit[name],
                    f"recomputed {name.replace('_', ' ')}",
                    positive=name not in {"ssim", "lpips"},
                )
            ) > 1e-12:
                raise ValueError(
                    f"rate-quality {name} is not recomputed from its evaluator receipt: {point_id}"
                )
        if not 0 <= evaluator_audit["ssim"] <= 1 or not 0 <= evaluator_audit["lpips"] <= 1:
            raise ValueError(f"ordinary image metric is outside range: {point_id}")
        audits[point_id] = {
            "validation": validation,
            "evaluator": evaluator_audit,
        }

    if (
        len(archive_hashes) != 4
        or len(archive_rates) != 4
        or len(training_config_hashes) != 4
        or set(producer_rates) != set(FROZEN_ORDINARY_RATE_LAMBDAS)
    ):
        raise ValueError("ordinary evidence lacks four distinct archives at four distinct rates")
    ordered_bytes = [producer_rates[index][0] for index in sorted(producer_rates)]
    for lower_rate, left, right in zip(range(3), ordered_bytes, ordered_bytes[1:]):
        if right >= left:
            raise ValueError(
                "actual sequence bytes are not strictly decreasing across the frozen rate grid"
            )
        relative_gap = (left - right) / float(right)
        if relative_gap < MIN_ADJACENT_RATE_BYTE_FRACTION:
            raise ValueError(
                "adjacent frozen rate points are separated by less than 5% actual bytes"
            )

    selected_id = _nonempty_string(selected_row["point_id"], "selected point ID")
    if selected_id not in audits:
        raise ValueError("selected operating point is absent from the required real curve")
    selected = audits[selected_id]["validation"]
    if (
        selected["archive_sha256"] != selected_validation["archive_sha256"]
        or _positive_int(selected["archive_bytes"], "selected evidence archive bytes")
        != _positive_int(
            selected_validation["archive_bytes"], "selected registry archive bytes"
        )
        or selected["training_config_sha256"]
        != _require_sha256(selected_row["training_config_sha256"], "registry training SHA-256")
        or _nonnegative_int(selected["seed"], "selected evidence seed")
        != _nonnegative_int(selected_row["seed"], "selected registry seed")
    ):
        raise ValueError("selected registry archive is not the selected evidence point")
    result = {
        "schema": ELIGIBILITY_RECOMPUTATION_SCHEMA,
        "eligible": True,
        "ordinary_rate_quality_only": True,
        "required_point_count": len(points),
        "distinct_archive_count": len(archive_hashes),
        "distinct_rate_count": len(archive_rates),
        "distinct_training_config_count": len(training_config_hashes),
        "frozen_rate_lambda_grid": {
            str(index): FROZEN_ORDINARY_RATE_LAMBDAS[index]
            for index in sorted(FROZEN_ORDINARY_RATE_LAMBDAS)
        },
        "minimum_adjacent_actual_byte_fraction": MIN_ADJACENT_RATE_BYTE_FRACTION,
        "actual_bytes_by_rate": {
            str(index): producer_rates[index][0] for index in sorted(producer_rates)
        },
        "metrics_recomputed_from_evaluator_receipts": True,
        "selected_point_id": selected_id,
        "selected_rate": _nonnegative_int(
            selected["producer_rate"], "selected producer rate"
        ),
        "selected_archive_bytes": _positive_int(
            selected["archive_bytes"], "selected archive bytes"
        ),
        "selected_training_config_sha256": selected["training_config_sha256"],
        "selected_seed": _nonnegative_int(selected["seed"], "selected seed"),
        "source_evidence_sha256": sha256_bytes(payload),
        "source_data": {
            key: value for key, value in data_audit.items() if key != "member_sha256"
        },
        "runtime_provenance": runtime_receipt,
        "evaluator": {
            "relative_path": evaluator_relative,
            "sha256": evaluator_sha,
        },
        "selected_evaluator_receipt_sha256": audits[selected_id]["evaluator"][
            "receipt_sha256"
        ],
        "selected_evaluator_receipt_path": audits[selected_id]["evaluator"][
            "receipt_path"
        ],
        "selected_metrics": {
            name: audits[selected_id]["evaluator"][name]
            for name in (
                "psnr",
                "ssim",
                "lpips",
                "encode_seconds",
                "decode_seconds",
                "render_fps",
            )
        },
    }
    validate_eligibility_recomputation_contract(
        result,
        expected_scene=_nonempty_string(selected_row["scene"], "selected scene"),
        expected_point_id=selected_id,
        expected_source_evidence_sha256=sha256_bytes(payload),
        expected_archive_bytes=_positive_int(
            selected_validation["archive_bytes"], "selected registry archive bytes"
        ),
        expected_training_config_sha256=_require_sha256(
            selected_row["training_config_sha256"], "registry training SHA-256"
        ),
        expected_seed=_nonnegative_int(selected_row["seed"], "selected registry seed"),
    )
    return result


def select_real_zip_operating_points(
    registry: Path,
    output: Path,
    methods: Sequence[str],
) -> Dict[str, Any]:
    """Freeze actual-byte official/AP points without reading H-DOWN outcomes."""

    if not methods or "official" in methods or len(set(methods)) != len(methods):
        raise ValueError("selection methods must be unique non-official variants")
    rows = [
        _eligible_registry_row(row, registry.parent)
        for row in _load_registry(registry)
    ]
    selected = []
    for scene in CONFIRMATORY_SCENES:
        official = [
            row for row in rows if row["scene"] == scene and row["method"] == "official"
        ]
        if not official:
            raise ValueError(f"scene lacks an eligible official point: {scene}")
        off = min(
            official,
            key=lambda row: (
                abs(row["archive_bytes"] - TARGET_BYTES),
                _nonempty_string(row["point_id"], "official point ID"),
            ),
        )
        target_error = abs(off["archive_bytes"] - TARGET_BYTES) / TARGET_BYTES
        if target_error > 0.025:
            raise ValueError(f"official point is outside 2.5% target window: {scene}")
        selected.append(
            {
                **off,
                "role": "official",
                "target_bytes": TARGET_BYTES,
                "target_relative_error": target_error,
            }
        )
        for method in methods:
            candidates = [
                row for row in rows if row["scene"] == scene and row["method"] == method
            ]
            if not candidates:
                raise ValueError(f"scene/method lacks an eligible point: {scene}/{method}")
            chosen = min(
                candidates,
                key=lambda row: (
                    abs(row["archive_bytes"] - off["archive_bytes"]),
                    _nonempty_string(row["point_id"], "candidate point ID"),
                ),
            )
            relative = abs(chosen["archive_bytes"] - off["archive_bytes"]) / off[
                "archive_bytes"
            ]
            if relative > 0.0125:
                raise ValueError(f"AP point is outside 1.25% official-byte window: {scene}/{method}")
            selected.append(
                {
                    **chosen,
                    "role": "candidate",
                    "matched_official_archive_sha256": off["archive_sha256"],
                    "relative_byte_error_vs_official": relative,
                }
            )
    result = {
        "schema": SELECTION_SCHEMA,
        "target_bytes": TARGET_BYTES,
        "official_target_tolerance": 0.025,
        "candidate_official_tolerance": 0.0125,
        "scenes": list(CONFIRMATORY_SCENES),
        "methods": list(methods),
        "selected": selected,
        "selection_fields": ["actual_sequence_archive_bytes"],
        "hdown_outcome_fields_read": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(result))
    return result
