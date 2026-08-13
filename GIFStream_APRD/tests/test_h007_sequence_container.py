import ast
import binascii
import hashlib
import importlib.util
import io
import json
import os
import pickle
import struct
import sys
import tempfile
import types
import unittest
import zipfile
import zlib
from collections import OrderedDict
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "gsplat/compression/h007_sequence_container.py"
SPEC = importlib.util.spec_from_file_location("h007_sequence_container_test", MODULE_PATH)
sequence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sequence)

TRAINING_CONFIG = {
    "scene": "flame_salmon_1",
    "variant": "official",
    "data_factor": 2,
    "GOP_size": 60,
    "rate": "0",
    "rd_lambda": 0.0005,
    "max_steps": 30_000,
    "random_seed": 42,
    "compression_seed": 20260715,
    "voxel_size": 0.01,
    "anchor_feature_dim": 32,
    "c_perframe": 4,
    "entropy_channel": 4,
    "n_offsets": 10,
    "n_knn": 3,
    "knn": True,
    "time_dim": 4,
    "view_adaptive": True,
    "app_opt": False,
    "compression_sim": True,
    "entropy_model_opt": True,
}
TRAINING_CONFIG_SHA = hashlib.sha256(
    sequence.canonical_json_bytes(TRAINING_CONFIG)
).hexdigest()

FIXTURE_DECODER_CONFIG = {
    "anchor_feature_dim": 32,
    "c_perframe": 4,
    "entropy_channel": 4,
    "n_offsets": 10,
    "time_dim": 4,
    "view_adaptive": True,
    "add_opacity_dist": False,
    "add_cov_dist": False,
    "add_color_dist": False,
    "app_opt": False,
    "app_embed_dim": 16,
    "appearance_embedding_count": 19,
    "rate": 0,
}


def fixture_ap_meta(method="ap-gifstream-full"):
    anchor_count = 4
    real_count = 3
    active_count = 2
    class_count = 1
    core = {
        "anchors": {
            "shape": [anchor_count, 3],
            "dtype": "float32",
            "mins": [0.0, 0.0, 0.0],
            "maxs": [1.0, 1.0, 1.0],
            "voxel_size": 0.01,
        },
        "scales": {"shape": [anchor_count, 6], "dtype": "float32", "scaling": 0.1},
        "quats": {"shape": [anchor_count, 4], "dtype": "float32"},
        "opacities": {"shape": [anchor_count, 1], "dtype": "float32"},
        "offsets": {"shape": [anchor_count, 30], "dtype": "float32", "scaling": 0.1},
        "anchor_features": {
            "shape": [anchor_count, 32],
            "dtype": "float32",
            "scaling": 0.1,
            "length": 1,
            "channel": 4,
        },
        "factors": {"shape": [anchor_count, 4], "dtype": "float32", "scaling": 0.1},
    }
    if method in sequence.QUANTIZED_AP_VARIANTS:
        core["anchor_features"] = {
            "shape": [anchor_count, 32],
            "dtype": "float32",
            "ap_two_class_anchor_features": True,
            "base_scaling": 0.1,
            "families": {
                "path": {
                    "rows": 2,
                    "scaling": 0.025,
                    "multiplier": 0.25,
                    "length": 8,
                    "channel": 4,
                },
                "bg": {
                    "rows": 2,
                    "scaling": 0.1,
                    "multiplier": 1.0,
                    "length": 8,
                    "channel": 4,
                },
            },
        }
        core["factors"] = {
            "shape": [anchor_count, 4],
            "dtype": "float32",
            "ap_two_class_factors": True,
            "base_scaling": 0.1,
            "factor0_activation_value": 0.125,
            "reconstruction_rule": (
                "adaptive-symbols+counted-factor0/factor3-semantics"
            ),
            "families": {
                "path": {
                    "rows": 2,
                    "scaling": 0.1 / 256.0,
                    "multiplier": 1.0 / 256.0,
                    "path": "factors_path.bin",
                    "bytes": 8,
                    "sha256": "b" * 64,
                },
                "bg": {
                    "rows": 2,
                    "scaling": 0.1 / 64.0,
                    "multiplier": 1.0 / 64.0,
                    "path": "factors_bg.bin",
                    "bytes": 8,
                    "sha256": "c" * 64,
                },
            },
        }
        core["time_features"] = {
            "shape": [active_count, 60, 4],
            "dtype": "float32",
            "ap_two_class": True,
            "precision_mask_contract": "path_input_mask",
            "base_scaling": 0.1,
            "length": 1,
            "channel": 4,
            "families": {
                "path": {
                    "rows": 1,
                    "scaling": 0.05,
                    "multiplier": 0.5,
                    "length": 1,
                    "channel": 4,
                },
                "bg": {
                    "rows": 1,
                    "scaling": 0.125,
                    "multiplier": 1.25,
                    "length": 1,
                    "channel": 4,
                },
            },
        }
    else:
        core["time_features"] = {
            "shape": [active_count, 60, 4],
            "dtype": "float32",
            "scaling": 0.1,
            "length": 1,
            "channel": 4,
        }
    digest = "a" * 64
    score = {
        "schema": "h007.ap_scores.v3",
        "score_artifact": "score.npz",
        "score_artifact_sha256": digest,
        "ranking": sequence.AP_VARIANT_METADATA[method]["ranking"],
        "variant": method,
        "protected_fraction": 0.05,
        "q_ap_multiplier": 0.5,
        "q_bg_multiplier": 1.25,
        "random_seed": 11,
        "estimator_version": sequence.AP_ESTIMATOR_VERSION,
        "time_entropy_model_sha256": digest,
        "time_feature_scaling": 0.1,
        "time_entropy_model_frozen_after_freeze": True,
        "runtime_manifest_sha256": digest,
        "normalized_code_tree_sha256": digest,
        "patch_chain_sha256": [digest] * 9,
        "anchor_count": anchor_count,
        "eligible_count": 3,
    }
    allocation = {
        "budget_source": "frozen_score_artifact",
        "official_retain_count": real_count,
        "ap_retain_count": real_count,
        "current_vs_frozen_whole_xor": 0,
        "current_vs_frozen_temporal_xor": 0,
        "encoded_row_count": anchor_count,
        "real_row_count": real_count,
        "padding_row_count": anchor_count - real_count,
        "active_row_count": active_count,
        "ap_class_real_count": class_count,
        "whole_promoted_count": 1,
        "whole_demoted_count": 1,
        "official_estimated_time_bytes": 100,
        "ap_estimated_time_bytes": 100,
        "plas_permutation_complete": True,
        "id_order_definition": "counted_corrections_restore_prequantization_ids_in_encoded_real_row_order",
    }
    runtime = {
        "schema": "h007.ap_gifstream.runtime_provenance.v1",
        "manifest_sha256": digest,
        "official_commit": sequence.OFFICIAL_COMMIT,
        "patch_sha256": [digest] * 9,
        "normalized_code_tree": {
            "schema": "h007.normalized_code_tree.v1",
            "normalization": "fixture",
            "roots": [],
            "root_files": [],
            "suffixes": [],
            "special_names": [],
            "file_count": 0,
            "sha256": digest,
        },
    }
    mask = {
        "path": "",
        "count": anchor_count,
        "true_count": 0,
        "bytes": 1,
        "sha256": digest,
        "bitorder": "little",
    }
    corrections = {
        "schema": "h007.ap_identity_corrections.v1",
        "path": "ap_identity_corrections.bin",
        "row_count": real_count,
        "mismatch_count": 0,
        "mask_bytes": (real_count + 7) // 8,
        "bytes": (real_count + 7) // 8,
        "sha256": digest,
        "bitorder": "little",
        "base": "round-decoded-anchor-div-voxel-size",
        "code": "uint8-base3-dx-dy-dz-plus1",
    }
    core["__ap__"] = {
        "schema": "h007.ap_gifstream.codec.v6",
        "variant": dict(sequence.AP_VARIANT_METADATA[method]),
        "score": score,
        "allocation": allocation,
        "runtime_provenance": runtime,
        "compression_seed": 20260715,
        "q_ap_multiplier": 0.5,
        "q_bg_multiplier": 1.25,
        "mask": {**mask, "path": "ap_class_mask.bin", "true_count": class_count},
        "path_input_mask": {
            **mask,
            "path": "ap_path_input_mask.bin",
            "true_count": 2,
        },
        "path_contract": {
            "schema": "h007.ap_gifstream.path_contract.v1",
            "knn_count": 3,
            "knn_rule": (
                "retained-canonical-radius-complete-distance+lexicographic-id"
            ),
            "dependency_rule": "protected-plus-one-hop-retained-knn",
            "retained_knn_graph_sha256": "b" * 64,
            "canonical_anchor_reconstruction": True,
            "factor_protected_multiplier": 1.0 / 256.0,
            "factor_background_multiplier": 1.0 / 64.0,
            "anchor_feature_protected_multiplier": 0.25,
            "anchor_feature_background_multiplier": 1.0,
        },
        "real_row_mask": {**mask, "path": "ap_real_row_mask.bin", "true_count": real_count},
        "padding_row_mask": {
            **mask,
            "path": "ap_padding_row_mask.bin",
            "true_count": anchor_count - real_count,
        },
        "active_row_mask": {**mask, "path": "ap_active_row_mask.bin", "true_count": active_count},
        "identity_corrections": corrections,
    }
    return core


def fixture_npy(shape=(1,), descr="<f4", fill=b"\0", data=None):
    descriptor = {"descr": descr, "fortran_order": False, "shape": tuple(shape)}
    base = (
        "{'descr': "
        + repr(descriptor["descr"])
        + ", 'fortran_order': False, 'shape': "
        + repr(descriptor["shape"])
        + ", }"
    ).encode("latin1")
    header_length = ((10 + len(base) + 1 + 63) // 64) * 64 - 10
    header = base + b" " * (header_length - len(base) - 1) + b"\n"
    width = int(descr[2:]) * (4 if descr[1] == "U" else 1)
    count = 1
    for value in shape:
        count *= value
    data_bytes = count * width
    if data is None:
        data = (fill * ((data_bytes + len(fill) - 1) // len(fill)))[:data_bytes]
    elif len(data) != data_bytes:
        raise ValueError("fixture NPY data extent mismatch")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", header_length) + header + data


def fixture_camera_names(width=None):
    names = tuple(f"cam{index:02d}.png" for index in range(19))
    required_width = max(len(name) for name in names)
    width = required_width if width is None else int(width)
    if width < required_width:
        raise ValueError("fixture camera-name width is too small")
    data = b"".join(
        struct.pack(
            f"<{width}I",
            *(ord(character) for character in name),
            *(0 for _ in range(width - len(name))),
        )
        for name in names
    )
    return fixture_npy((len(names),), descr=f"<U{width}", data=data)


def fixture_npz(members, *, compressed=True):
    output = io.BytesIO()
    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    with zipfile.ZipFile(output, "w", compression=compression, allowZip64=True) as handle:
        for name, payload in members.items():
            with handle.open(name, "w", force_zip64=True) as destination:
                destination.write(payload)
    return output.getvalue()


def fixture_png(
    *, split_idat=False, filter_type=0, compression_level=9, bit_depth=8, color_type=2
):
    def chunk(name, payload):
        return (
            struct.pack(">I", len(payload))
            + name
            + payload
            + struct.pack(">I", binascii.crc32(name + payload) & 0xFFFFFFFF)
        )

    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    ihdr = struct.pack(">IIBBBBB", 1, 1, bit_depth, color_type, 0, 0, 0)
    stream = zlib.compress(
        bytes([filter_type]) + b"\0" * (channels * bit_depth // 8),
        compression_level,
    )
    idat = (
        b"".join(chunk(b"IDAT", bytes([value])) for value in stream)
        if split_idat
        else chunk(b"IDAT", stream)
    )
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + idat + chunk(b"IEND", b"")


def fixture_entropy(payload=b"x"):
    return struct.pack(">I", len(payload)) + payload


_FixtureFloatStorage = type("FloatStorage", (), {})
_FixtureFloatStorage.__module__ = "torch"


class _FixtureStorage:
    def __init__(self, key, numel, payload):
        self.key = int(key)
        self.numel = int(numel)
        self.payload = bytes(payload)


def _fixture_rebuild_tensor_v2(
    storage,
    storage_offset,
    shape,
    stride,
    requires_grad,
    backward_hooks,
):
    raise AssertionError("fixture tensor rebuild is pickle-only")


_fixture_rebuild_tensor_v2.__module__ = "torch._utils"
_fixture_rebuild_tensor_v2.__name__ = "_rebuild_tensor_v2"
_fixture_rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"


class _FixtureTensor:
    def __init__(self, storage, shape, *, stride=None, storage_offset=0):
        self.storage = storage
        self.shape = tuple(shape)
        self.stride = (
            sequence._contiguous_stride(self.shape)
            if stride is None
            else tuple(stride)
        )
        self.storage_offset = int(storage_offset)

    def __reduce_ex__(self, protocol):
        return (
            _fixture_rebuild_tensor_v2,
            (
                self.storage,
                self.storage_offset,
                self.shape,
                self.stride,
                False,
                OrderedDict(),
            ),
        )


class _FixtureStoragePickler(pickle.Pickler):
    def persistent_id(self, value):
        if isinstance(value, _FixtureStorage):
            return (
                "storage",
                _FixtureFloatStorage,
                str(value.key),
                "cuda:0",
                value.numel,
            )
        return None


def fixture_torch_save(
    *,
    pickle_tail=b"",
    pickle_noops=0,
    nonminimal_numel=False,
    unreferenced_storage=False,
    storage_padding=b"",
    root="nets",
    extra_fixed_records=False,
    extra_root_padding=0,
    serialization_id=None,
    tensor_shape=None,
    tensor_stride=None,
    tensor_storage_offset=0,
    config=None,
    return_audit=False,
):
    config = dict(FIXTURE_DECODER_CONFIG if config is None else config)
    schema = sequence._expected_nets_tensor_schema(config)
    storages = []
    model = {}
    first_tensor = True
    for role, rows in schema.items():
        state = OrderedDict()
        for name, expected_shape in rows:
            shape = tuple(tensor_shape) if first_tensor and tensor_shape else expected_shape
            stride = tensor_stride if first_tensor and tensor_stride is not None else None
            offset = tensor_storage_offset if first_tensor else 0
            numel = 1
            for value in shape:
                numel *= value
            storage = _FixtureStorage(len(storages), numel, b"\0" * (4 * numel))
            storages.append(storage)
            state[name] = _FixtureTensor(
                storage,
                shape,
                stride=stride,
                storage_offset=offset,
            )
            first_tensor = False
        state._metadata = sequence._expected_state_metadata(role)["_metadata"]
        model[role] = state
    model["scaling"] = sequence._expected_codec_scaling(config["rate"])
    if "app_module" in model:
        app_module = model.pop("app_module")
        model["app_module"] = app_module
    if extra_root_padding:
        model["padding"] = "x" * int(extra_root_padding)

    pickle_output = io.BytesIO()
    fake_torch = types.ModuleType("torch")
    fake_torch_utils = types.ModuleType("torch._utils")
    fake_torch.FloatStorage = _FixtureFloatStorage
    fake_torch._utils = fake_torch_utils
    fake_torch_utils._rebuild_tensor_v2 = _fixture_rebuild_tensor_v2
    previous_torch = sys.modules.get("torch")
    previous_torch_utils = sys.modules.get("torch._utils")
    sys.modules["torch"] = fake_torch
    sys.modules["torch._utils"] = fake_torch_utils
    try:
        _FixtureStoragePickler(pickle_output, protocol=2).dump(model)
    finally:
        if previous_torch is None:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = previous_torch
        if previous_torch_utils is None:
            del sys.modules["torch._utils"]
        else:
            sys.modules["torch._utils"] = previous_torch_utils
    pickle_payload = pickle_output.getvalue()
    if nonminimal_numel:
        target = storages[0].numel
        operations = list(sequence.pickletools.genops(pickle_payload))
        candidates = [
            (index, position)
            for index, (operation, argument, position) in enumerate(operations)
            if operation.name in {"BININT1", "BININT2", "BININT", "LONG1", "LONG4"}
            and argument == target
        ]
        if not candidates:
            raise AssertionError("fixture lacks its storage numel opcode")
        index, start = candidates[0]
        end = operations[index + 1][2]
        encoded = pickle.encode_long(target)
        replacement = pickle.LONG1 + bytes([250]) + encoded + b"\0" * (250 - len(encoded))
        pickle_payload = pickle_payload[:start] + replacement + pickle_payload[end:]
    if serialization_id is None:
        serialization_id = b"0" * 40
    records = {
        f"{root}/data.pkl": (
            pickle_payload[:2] + b"N0" * pickle_noops + pickle_payload[2:] + pickle_tail
        ),
        f"{root}/byteorder": b"little",
        f"{root}/version": b"3\n",
        f"{root}/.data/serialization_id": serialization_id,
    }
    for storage in storages:
        payload = storage.payload
        if storage.key == 0:
            payload += storage_padding
        records[f"{root}/data/{storage.key}"] = payload
    if extra_fixed_records:
        records[f"{root}/.format_version"] = b"1"
        records[f"{root}/.storage_alignment"] = b"64"
    if unreferenced_storage:
        records[f"{root}/data/{len(storages)}"] = b"unreferenced-padding"
    payload = sequence._canonical_torch_zip_bytes(root, records)
    structurally_valid = (
        not pickle_tail
        and not pickle_noops
        and not nonminimal_numel
        and not unreferenced_storage
        and not storage_padding
        and root == "nets"
        and not extra_fixed_records
        and not extra_root_padding
        and serialization_id == b"0" * 40
        and tensor_stride is None
        and tensor_storage_offset == 0
    )
    audit = None
    if structurally_valid:
        payload, audit = sequence._canonical_torch_save_payload(payload, "fixture nets.pt")
    if return_audit:
        if audit is None:
            raise ValueError("audit is only available for a structurally valid Torch fixture")
        return payload, audit
    return payload


class H007SequenceContainerTest(unittest.TestCase):
    def test_frozen_npz_member_schemas_match_the_actual_producers(self):
        repo = Path(__file__).resolve().parents[1]

        def savez_keyword_members(path, minimum_keywords):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            matches = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr != "savez" or len(node.keywords) < minimum_keywords:
                    continue
                matches.append({f"{item.arg}.npy" for item in node.keywords})
            self.assertEqual(len(matches), 1, path.name)
            return matches[0]

        self.assertEqual(
            sequence.AP_SCORE_NPZ_MEMBERS,
            savez_keyword_members(repo / "examples/simple_trainer_GIFStream.py", 20),
        )
        self.assertEqual(
            sequence.AP_EDIT_IDS_NPZ_MEMBERS,
            savez_keyword_members(repo / "examples/h007_make_ap_edit_ids.py", 8),
        )

    @staticmethod
    def _frozen_training_contract(variant: str, app_opt: bool):
        config = {**TRAINING_CONFIG, "variant": variant, "app_opt": app_opt}
        runtime = {
            "schema": "h007.ap_gifstream.runtime_provenance.v1",
            "manifest_sha256": "a" * 64,
        }
        checkpoints = [
            {"path": "/frozen/ckpt.pt", "bytes": 7, "sha256": "b" * 64}
        ]
        receipt = {
            "schema": sequence.FROZEN_TRAINING_RECEIPT_SCHEMA,
            "official_commit": sequence.OFFICIAL_COMMIT,
            "scene": config["scene"],
            "variant": variant,
            "training_step": 29999,
            "state_position": "after_optimizer_entropy_and_strategy_post_backward",
            "training_config": config,
            "training_config_sha256": hashlib.sha256(
                sequence.canonical_json_bytes(config)
            ).hexdigest(),
            "source_checkpoints": checkpoints,
            "model_state_sha256": {
                "splats": "c" * 64,
                "decoders": "d" * 64,
                "entropy_models": {
                    name: "e" * 64
                    for name in (
                        "scales",
                        "offsets",
                        "anchor_features",
                        "factors",
                        "time_features",
                    )
                },
                "codec_scaling": "f" * 64,
                "appearance_module": "1" * 64 if app_opt else None,
            },
            "ap_training_receipt_sha256": (
                None if variant == "official" else "2" * 64
            ),
            "runtime_provenance": runtime,
            "outcome_fields_read": [],
        }
        return receipt, config, runtime, checkpoints

    def _validate_frozen_training_contract(self, receipt, config, runtime, checkpoints):
        return sequence.validate_frozen_training_receipt_contract(
            receipt,
            expected_scene=config["scene"],
            expected_variant=config["variant"],
            expected_training_config=config,
            expected_runtime_provenance=runtime,
            expected_source_checkpoints=checkpoints,
        )

    def test_official_codec_entry_accepts_five_state_training_receipt(self):
        values = self._frozen_training_contract("official", False)
        validated = self._validate_frozen_training_contract(*values)
        self.assertIsNone(validated["model_state_sha256"]["appearance_module"])

    def test_ap_codec_entry_accepts_five_state_training_receipt(self):
        values = self._frozen_training_contract("ap-gifstream-full", True)
        validated = self._validate_frozen_training_contract(*values)
        self.assertEqual(
            validated["model_state_sha256"]["appearance_module"], "1" * 64
        )

    def test_codec_entry_rejects_frozen_training_receipt_tamper(self):
        receipt, config, runtime, checkpoints = self._frozen_training_contract(
            "ap-gifstream-full", True
        )
        missing_scaling = json.loads(json.dumps(receipt))
        missing_scaling["model_state_sha256"].pop("codec_scaling")
        with self.assertRaisesRegex(ValueError, "model-state closure"):
            self._validate_frozen_training_contract(
                missing_scaling, config, runtime, checkpoints
            )
        missing_appearance = json.loads(json.dumps(receipt))
        missing_appearance["model_state_sha256"]["appearance_module"] = None
        with self.assertRaisesRegex(ValueError, "appearance-module"):
            self._validate_frozen_training_contract(
                missing_appearance, config, runtime, checkpoints
            )
        changed_checkpoint = [dict(checkpoints[0], sha256="3" * 64)]
        with self.assertRaisesRegex(ValueError, "identity/config/runtime"):
            self._validate_frozen_training_contract(
                receipt, config, runtime, changed_checkpoint
            )

    def test_training_receipt_requires_terminal_max_steps_minus_one(self):
        receipt, config, runtime, checkpoints = self._frozen_training_contract(
            "official", False
        )
        receipt["training_step"] = 29_998
        with self.assertRaisesRegex(ValueError, "identity/config/runtime"):
            self._validate_frozen_training_contract(
                receipt, config, runtime, checkpoints
            )

    def test_ap_training_receipt_hash_is_exact_across_all_three_bindings(self):
        payload = b"counted-ap-training-receipt"
        digest = hashlib.sha256(payload).hexdigest()
        self.assertEqual(
            sequence.validate_ap_training_receipt_binding(
                method="ap-gifstream-full",
                producer_sha256=digest,
                frozen_sha256=digest,
                counted_payload=payload,
            ),
            digest,
        )
        with self.assertRaisesRegex(ValueError, "producer/frozen"):
            sequence.validate_ap_training_receipt_binding(
                method="ap-gifstream-full",
                producer_sha256=digest,
                frozen_sha256="f" * 64,
                counted_payload=payload,
            )
        with self.assertRaisesRegex(ValueError, "counted AP"):
            sequence.validate_ap_training_receipt_binding(
                method="ap-gifstream-full",
                producer_sha256=digest,
                frozen_sha256=digest,
                counted_payload=b"tampered",
            )

    def test_compression_validation_keeps_ap_training_receipt_immutable(self):
        trainer = MODULE_PATH.parents[2] / "examples/simple_trainer_GIFStream.py"
        tree = ast.parse(trainer.read_text(encoding="utf-8"))
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_validate_ap_compression_receipt"
        )
        receipt_field_writes = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
            and node.value.id == "receipt"
        ]
        self.assertEqual(receipt_field_writes, [])

    def test_sequence_shared_producer_config_excludes_gop_start(self):
        trainer = MODULE_PATH.parents[2] / "examples/simple_trainer_GIFStream.py"
        tree = ast.parse(trainer.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "producer_training_config"
        )
        returned = next(
            node.value for node in ast.walk(function) if isinstance(node, ast.Return)
        )
        keys = {
            key.value
            for key in returned.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        self.assertNotIn("start_frame", keys)
        self.assertIn("app_opt", keys)
        self.assertIn("rd_lambda", keys)

    def test_decoder_camera_lists_are_canonicalized_by_the_producer(self):
        trainer = MODULE_PATH.parents[2] / "examples/simple_trainer_GIFStream.py"
        source = trainer.read_text(encoding="utf-8")
        self.assertIn('"test_set": list(cfg.test_set or [])', source)
        self.assertIn('"remove_set": list(cfg.remove_set or [])', source)

    def test_registered_f02_rate_controls_match_producer_and_validator(self):
        trainer = MODULE_PATH.parents[2] / "examples/simple_trainer_GIFStream.py"
        tree = ast.parse(trainer.read_text(encoding="utf-8"))
        config = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Config"
        )
        assignment = next(
            node
            for node in config.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "compression_scaling"
                for target in node.targets
            )
        )
        producer_grid = eval(
            compile(ast.Expression(assignment.value), str(trainer), "eval"),
            {"__builtins__": {}},
        )
        expected = {
            "anchors": None,
            "scales": 0.038,
            "quats": None,
            "opacities": None,
            "anchor_features": 1,
            "offsets": 0.038,
            "factors": 0.0625,
            "time_features": 1,
        }
        self.assertEqual(len(producer_grid), 6)
        self.assertEqual(producer_grid[5], expected)
        self.assertEqual(sequence.FROZEN_RATE_LAMBDAS[5], 0.00095)
        self.assertEqual(sequence._expected_codec_scaling(5), expected)

    def test_ordinary_evaluator_has_no_external_prediction_input(self):
        evaluator = MODULE_PATH.parents[2] / "examples/h007_ordinary_rate_quality.py"
        source = evaluator.read_text(encoding="utf-8")
        self.assertNotIn("--predictions-root", source)
        self.assertNotIn("--clean-decode-receipts", source)
        tree = ast.parse(source)
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        for name in ("generate", "recompute_receipt_metrics"):
            calls = {
                node.func.id
                for node in ast.walk(functions[name])
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            self.assertIn("_fresh_sequence_predictions", calls)

    def test_ordinary_evaluator_matches_archive_render_resolution(self):
        evaluator = MODULE_PATH.parents[2] / "examples/h007_ordinary_rate_quality.py"
        source = evaluator.read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        image_calls = [
            node
            for node in ast.walk(functions["_image"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cv2"
            and node.func.attr == "resize"
        ]
        self.assertEqual(len(image_calls), 1)
        resize_keywords = {keyword.arg: keyword.value for keyword in image_calls[0].keywords}
        interpolation = resize_keywords["interpolation"]
        self.assertIsInstance(interpolation, ast.Attribute)
        self.assertEqual(interpolation.attr, "INTER_LINEAR")
        for name in ("generate", "recompute_receipt_metrics"):
            self.assertIn("int(replay['data_factor'])", ast.unparse(functions[name]))
        self.assertIn("data_factor = int(config.get", source)
        self.assertIn("five-GOP archive replay disagrees on the data factor", source)

    def _recompute_curve(
        self,
        root: Path,
        archive_hashes,
        archive_bytes,
        evaluator_overrides=None,
        training_hashes=None,
        producer_lambdas=None,
    ):
        source_data = root / "source_data.json"
        source_data.write_bytes(b"{}")
        runtime_manifest = root / "runtime_manifest.json"
        runtime_manifest.write_bytes(b"runtime")
        evaluator = root / sequence.ORDINARY_EVALUATOR_RELATIVE_PATH
        evaluator.parent.mkdir(parents=True)
        evaluator.write_bytes(b"frozen evaluator")
        validations = []
        points = []
        training_hashes = training_hashes or [f"{index + 5:x}" * 64 for index in range(4)]
        producer_lambdas = producer_lambdas or [
            sequence.FROZEN_RATE_LAMBDAS[index] for index in range(4)
        ]
        for index, (digest, size) in enumerate(zip(archive_hashes, archive_bytes)):
            training_sha = training_hashes[index]
            archive = root / f"point_{index}.zip"
            archive.write_bytes(f"archive-{index}".encode())
            receipt = root / f"receipt_{index}.json"
            receipt.write_bytes(b"{}")
            validations.append(
                {
                    "archive_sha256": digest,
                    "archive_bytes": size,
                    "training_config_sha256": training_sha,
                    "seed": 17 + index,
                    "producer_rate": index,
                    "producer_rd_lambda": producer_lambdas[index],
                    "gops": [],
                }
            )
            points.append(
                {
                    "point_id": f"p{index}",
                    "archive": archive.name,
                    "archive_bytes": size,
                    "archive_sha256": digest,
                    "mb_per_frame": size / 1_000_000.0 / 300.0,
                    "psnr": 30.0,
                    "ssim": 0.9,
                    "lpips": 0.1,
                    "encode_seconds": 5.0,
                    "decode_seconds": 6.0,
                    "render_fps": 7.0,
                    "training_config_sha256": training_sha,
                    "seed": 17 + index,
                    "evaluator_receipt": receipt.name,
                    "evaluator_receipt_sha256": hashlib.sha256(
                        receipt.read_bytes()
                    ).hexdigest(),
                }
            )
        runtime_receipt = {
            "schema": "h007.ap_gifstream.runtime_provenance.v1",
            "manifest_sha256": hashlib.sha256(runtime_manifest.read_bytes()).hexdigest(),
            "official_commit": sequence.OFFICIAL_COMMIT,
            "patch_sha256": ["a" * 64] * 9,
            "normalized_code_tree": {
                "schema": "h007.normalized_code_tree.v1",
                "normalization": "sorted-posix-path+lf-bytes+uint64le-lengths",
                "roots": ["examples", "gsplat", "third_party"],
                "root_files": ["setup.py"],
                "suffixes": [".py"],
                "special_names": ["CMakeLists.txt"],
                "file_count": 1,
                "sha256": "b" * 64,
            },
        }
        evidence = {
            "schema": sequence.ELIGIBILITY_EVIDENCE_SCHEMA,
            "scene": "flame_salmon_1",
            "method": "official",
            "frame_count": 300,
            "metric_protocol": sequence.ORDINARY_METRIC_PROTOCOL,
            "ordinary_rate_quality_only": True,
            "outcome_fields_read": [
                "ordinary_unedited_fidelity",
                "real_container_accounting",
            ],
            "source_data_manifest": source_data.name,
            "source_data_manifest_sha256": hashlib.sha256(
                source_data.read_bytes()
            ).hexdigest(),
            "runtime_provenance_manifest": runtime_manifest.name,
            "runtime_provenance_manifest_sha256": runtime_receipt[
                "manifest_sha256"
            ],
            "runtime_repo_root": str(root),
            "evaluator_relative_path": sequence.ORDINARY_EVALUATOR_RELATIVE_PATH,
            "evaluator_sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest(),
            "points": points,
        }
        selected_row = {
            "scene": "flame_salmon_1",
            "method": "official",
            "point_id": "p0",
            "training_config_sha256": "5" * 64,
            "seed": 17,
        }
        evaluator_audit = {
            "receipt_path": str((root / "receipt_0.json").resolve()),
            "receipt_sha256": hashlib.sha256(b"{}").hexdigest(),
            "psnr": 30.0,
            "ssim": 0.9,
            "lpips": 0.1,
            "encode_seconds": 5.0,
            "decode_seconds": 6.0,
            "render_fps": 7.0,
        }
        evaluator_audit.update(evaluator_overrides or {})
        runtime_module = types.SimpleNamespace(
            verify_runtime_provenance=lambda *args, **kwargs: runtime_receipt
        )
        source_audit = {
            "manifest_sha256": evidence["source_data_manifest_sha256"],
            "manifest_path": str(source_data.resolve()),
            "file_count": 19 * 300,
            "scene": "flame_salmon_1",
            "member_sha256": {
                f"cam00/{frame + 1:05d}.png": "b" * 64 for frame in range(300)
            },
        }
        with mock.patch.object(
            sequence, "validate_sequence_container", side_effect=validations
        ), mock.patch.object(
            sequence, "_validate_source_data_manifest", return_value=source_audit
        ), mock.patch.object(
            sequence, "_load_runtime_provenance_module", return_value=runtime_module
        ), mock.patch.object(
            sequence,
            "_validate_rate_quality_evaluator_receipt",
            return_value=evaluator_audit,
        ):
            return sequence._validate_ordinary_rate_quality_evidence(
                source_path=root / "evidence.json",
                payload=sequence.canonical_json_bytes(evidence),
                selected_row=selected_row,
                selected_validation=validations[0],
            )

    @staticmethod
    def _repack(path: Path, mutate) -> Path:
        with zipfile.ZipFile(path, "r") as handle:
            members = {
                info.filename: handle.read(info)
                for info in handle.infolist()
                if not info.is_dir() and info.filename != "byte_census.json"
            }
        mutate(members)
        census = {
            "schema": "h007.container_byte_census.v1",
            "root": "rank0",
            "file_count": len(members),
            "raw_bytes": sum(len(payload) for payload in members.values()),
            "files": [
                {
                    "path": name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                for name, payload in sorted(members.items())
            ],
            "self_exclusion": "byte_census.json is counted by the archive but cannot hash itself",
        }
        members["byte_census.json"] = sequence.canonical_json_bytes(census)
        output = path.with_name(path.stem + "_repacked.zip")
        output.write_bytes(sequence._canonical_zip_bytes(members))
        return output

    @staticmethod
    def _gop(root: Path, gop_id: int) -> Path:
        payload_root = root / f"payload_{gop_id}"
        payload_root.mkdir()
        normalized_tree = {
            "schema": "h007.normalized_code_tree.v1",
            "normalization": "sorted-posix-path+lf-bytes+uint64le-lengths",
            "roots": ["examples", "gsplat", "third_party"],
            "root_files": ["setup.py"],
            "suffixes": [".py"],
            "special_names": ["CMakeLists.txt"],
            "file_count": 1,
            "sha256": "a" * 64,
        }
        patch_hashes = [f"{index:x}" * 64 for index in range(1, 10)]
        runtime_manifest_payload = sequence.canonical_json_bytes(
            {
                "schema": "h007.ap_gifstream.patch_chain_manifest.v1",
                "official_commit": sequence.OFFICIAL_COMMIT,
                "patches": [
                    {
                        "stage": stage,
                        "path": f"../{stage}.patch",
                        "sha256": digest,
                    }
                    for stage, digest in zip(
                        (
                            "patch1",
                            "patch2",
                            "patch2b",
                            "patch3",
                            "patch4",
                            "patch5",
                            "patch6",
                            "patch7",
                            "patch8",
                        ),
                        patch_hashes,
                    )
                ],
                "normalized_code_tree": normalized_tree,
            }
        )
        runtime_manifest_sha = hashlib.sha256(runtime_manifest_payload).hexdigest()
        runtime_provenance = {
            "schema": "h007.ap_gifstream.runtime_provenance.v1",
            "manifest_sha256": runtime_manifest_sha,
            "official_commit": sequence.OFFICIAL_COMMIT,
            "patch_sha256": patch_hashes,
            "normalized_code_tree": normalized_tree,
        }
        meta = {
            "anchors": {
                "shape": [1, 3],
                "dtype": "float32",
                "mins": [0.0, 0.0, 0.0],
                "maxs": [1.0, 1.0, 1.0],
                "voxel_size": 0.01,
            },
            "scales": {"shape": [1, 6], "dtype": "float32", "scaling": 1.0},
            "quats": {"shape": [1, 4], "dtype": "float32"},
            "opacities": {"shape": [1, 1], "dtype": "float32"},
            "offsets": {
                "shape": [1, 10, 3],
                "dtype": "float32",
                "scaling": 1.0,
            },
            "anchor_features": {
                "shape": [1, 32],
                "dtype": "float32",
                "scaling": 1.0,
                "length": 1,
                "channel": 32,
            },
            "factors": {"shape": [1, 4], "dtype": "float32", "scaling": 1.0},
            "time_features": {
                "shape": [1, 60, 4],
                "dtype": "float32",
                "scaling": 1.0,
                "length": 1,
                "channel": 240,
            },
        }
        (payload_root / "meta.json").write_bytes(sequence.canonical_json_bytes(meta))
        nets_payload, nets_audit = fixture_torch_save(return_audit=True)
        codec_payloads = {
            "nets.pt": nets_payload,
            "anchors_l.png": fixture_png(),
            "anchors_u.png": fixture_png(),
            "quats.npz": fixture_npz({"arr.npy": fixture_npy((1, 4))}),
            "opacities.npz": fixture_npz({"arr.npy": fixture_npy((1, 1))}),
            "scales.bin": fixture_entropy(b"scales00"),
            "offsets.bin": fixture_entropy(b"offsets0"),
            "factors.bin": fixture_entropy(b"factors0"),
            "anchor_features_00000.bin": fixture_entropy(b"features"),
            "time_features_00000.bin": fixture_entropy(b"time0000"),
        }
        for name, payload in codec_payloads.items():
            (payload_root / name).write_bytes(payload)
        payload_manifest = sequence.build_gifstream_payload_manifest(
            payload_root,
            scene="flame_salmon_1",
            variant="official",
            start_frame=60 * gop_id,
            gop_size=60,
        )
        training_receipt = {
            "schema": "h007.gifstream_frozen_training_receipt.v1",
            "official_commit": sequence.OFFICIAL_COMMIT,
            "scene": "flame_salmon_1",
            "variant": "official",
            "training_step": 29999,
            "state_position": "after_optimizer_entropy_and_strategy_post_backward",
            "training_config": TRAINING_CONFIG,
            "training_config_sha256": TRAINING_CONFIG_SHA,
            "source_checkpoints": [
                {"path": "/frozen/ckpt.pt", "bytes": 1, "sha256": "b" * 64}
            ],
            "model_state_sha256": {
                "splats": "c" * 64,
                "decoders": nets_audit["state_sha256"]["decoders"],
                "entropy_models": {
                    name: nets_audit["state_sha256"][f"{name}_entropy_model"]
                    for name in (
                        "scales",
                        "offsets",
                        "anchor_features",
                        "factors",
                        "time_features",
                    )
                },
                "codec_scaling": nets_audit["scaling_sha256"],
                "appearance_module": None,
            },
            "ap_training_receipt_sha256": None,
            "runtime_provenance": runtime_provenance,
            "outcome_fields_read": [],
        }
        training_payload = sequence.canonical_json_bytes(training_receipt)
        producer_receipt = {
            "schema": sequence.PRODUCER_RECEIPT_SCHEMA,
            "official_commit": sequence.OFFICIAL_COMMIT,
            "scene": "flame_salmon_1",
            "variant": "official",
            "start_frame": 60 * gop_id,
            "GOP_size": 60,
            "training_step": 29999,
            "state_position": "after_optimizer_entropy_and_strategy_post_backward",
            "training_config": TRAINING_CONFIG,
            "training_config_sha256": TRAINING_CONFIG_SHA,
            "source_checkpoints": training_receipt["source_checkpoints"],
            "model_state_sha256": training_receipt["model_state_sha256"],
            "runtime_provenance": runtime_provenance,
            "training_receipt_sha256": hashlib.sha256(training_payload).hexdigest(),
            "ap_training_receipt_sha256": None,
            "outcome_fields_read": [],
        }
        producer_payload = sequence.canonical_json_bytes(producer_receipt)
        payload_manifest_payload = sequence.canonical_json_bytes(payload_manifest)
        config = {
            "schema": sequence.DECODER_CONFIG_SCHEMA,
            "codec_family": "GIFStream",
            "official_commit": sequence.OFFICIAL_COMMIT,
            "patch_chain_sha256": runtime_provenance["patch_sha256"],
            "runtime_manifest_sha256": runtime_manifest_sha,
            "normalized_code_tree_sha256": "a" * 64,
            "producer_receipt_sha256": hashlib.sha256(producer_payload).hexdigest(),
            "training_receipt_sha256": hashlib.sha256(training_payload).hexdigest(),
            "payload_manifest_sha256": hashlib.sha256(
                payload_manifest_payload
            ).hexdigest(),
            "variant": "official",
            "scene": "flame_salmon_1",
            "data_factor": 2,
            "start_frame": 60 * gop_id,
            "GOP_size": 60,
            "rate": 0,
            "voxel_size": 0.01,
            "anchor_feature_dim": 32,
            "c_perframe": 4,
            "entropy_channel": 4,
            "n_offsets": 10,
            "n_knn": 3,
            "knn": True,
            "time_dim": 4,
            "view_adaptive": True,
            "add_opacity_dist": False,
            "add_cov_dist": False,
            "add_color_dist": False,
            "app_opt": False,
            "app_embed_dim": 16,
            "appearance_embedding_count": 19,
            "packed": False,
            "antialiased": False,
            "camera_model": "pinhole",
            "phi": 1.0,
            "test_set": [0],
            "remove_set": [],
            "compression_seed": 20260715,
            "warm_camera_pose_index": 0,
            "warm_frame_index": 0,
            "warmup_renders": 5,
            "timed_renders": 20,
            "clean_decode_entrypoint": "examples/h007_clean_decode_gifstream.py",
        }
        members = {
            name: path.read_bytes()
            for name, path in (
                (path.name, path) for path in payload_root.iterdir() if path.is_file()
            )
        }
        members.update(
            {
                "decoder_config.json": sequence.canonical_json_bytes(config),
                "gifstream_payload_manifest.json": payload_manifest_payload,
                "producer_receipt.json": producer_payload,
                "training_receipt.json": training_payload,
                "runtime_provenance.json": sequence.canonical_json_bytes(
                    runtime_provenance
                ),
                "preregistered_patch_chain_manifest.json": runtime_manifest_payload,
                "runtime.json": sequence.canonical_json_bytes(
                    {
                        "schema": "h007.gifstream_runtime.v1",
                        "encode_seconds": 1.0 + gop_id,
                        "model_load_plus_entropy_decode_seconds": 2.0 + gop_id,
                        "peak_decode_cuda_bytes": None,
                        "warm_render": {
                            "status": "REQUIRED_IN_CLEAN_PROCESS",
                            "camera_metadata_source": "counted_archive_only",
                        },
                        "warm_render_fps": None,
                        "ap_score_seconds": None,
                        "outcome_fields_read": [],
                    }
                ),
                "clean_decode_request.json": sequence.canonical_json_bytes(
                    {
                        "schema": "h007.clean_decode_request.v2",
                        "archive_only": True,
                        "entrypoint": "examples/h007_clean_decode_gifstream.py",
                        "expected_output": "decoded_splats.pt",
                        "expected_runtime_output": "counted_camera_render",
                        "external_shared_runtime": {
                            "provenance_manifest_required": True,
                            "provenance_manifest_sha256": runtime_manifest_sha,
                        },
                    }
                ),
            }
        )
        camera_payloads = {
            "camera_keys": fixture_npy((19,), descr="<i8"),
            "intrinsics": fixture_npy((19, 3, 3), descr="<f8"),
            "image_sizes": fixture_npy((19, 2), descr="<i8"),
            "camtoworlds": fixture_npy((19, 4, 4), descr="<f8"),
            "camera_ids": fixture_npy((19,), descr="<i8"),
            "camera_names": fixture_camera_names(),
            "transform": fixture_npy((4, 4), descr="<f8"),
            "bounds": fixture_npy((19, 2), descr="<f8"),
        }
        for name, payload in camera_payloads.items():
            members[f"camera_metadata/{name}.npy"] = payload
        census = {
            "schema": "h007.container_byte_census.v1",
            "root": "rank0",
            "file_count": len(members),
            "raw_bytes": sum(len(payload) for payload in members.values()),
            "files": [
                {
                    "path": name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                for name, payload in sorted(members.items())
            ],
            "self_exclusion": "byte_census.json is counted by the archive but cannot hash itself",
        }
        members["byte_census.json"] = sequence.canonical_json_bytes(census)
        path = root / f"gop_{gop_id}.zip"
        path.write_bytes(sequence._canonical_zip_bytes(members))
        return path

    def test_five_gop_build_is_deterministic_and_nested_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gops = [self._gop(root, index) for index in range(5)]
            first = root / "first.zip"
            second = root / "second.zip"
            kwargs = {
                "scene": "flame_salmon_1",
                "method": "official",
                "gop_archives": gops,
                "training_config_sha256": TRAINING_CONFIG_SHA,
                "seed": 20260715,
            }
            left = sequence.build_sequence_container(output=first, **kwargs)
            right = sequence.build_sequence_container(output=second, **kwargs)
            self.assertEqual(left["archive_sha256"], right["archive_sha256"])
            audit = sequence.validate_sequence_container(
                first,
                expected_scene="flame_salmon_1",
                expected_method="official",
            )
            self.assertEqual(audit["frame_count"], 300)
            self.assertEqual(audit["gop_count"], 5)

            with zipfile.ZipFile(first, "a") as handle:
                handle.writestr("unexpected.bin", b"fault")
            with self.assertRaisesRegex(ValueError, "census|exact contract"):
                sequence.validate_sequence_container(first)

    def test_sequence_seed_must_equal_all_counted_compression_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gops = [self._gop(root, index) for index in range(5)]
            with self.assertRaisesRegex(ValueError, "compression seeds"):
                sequence.build_sequence_container(
                    scene="flame_salmon_1",
                    method="official",
                    gop_archives=gops,
                    output=root / "wrong_seed.zip",
                    training_config_sha256=TRAINING_CONFIG_SHA,
                    seed=20260716,
                )

    def test_outer_census_duplicate_key_padding_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gops = [self._gop(root, index) for index in range(5)]
            archive = root / "sequence.zip"
            sequence.build_sequence_container(
                scene="flame_salmon_1",
                method="official",
                gop_archives=gops,
                output=archive,
                training_config_sha256=TRAINING_CONFIG_SHA,
                seed=20260715,
            )
            with zipfile.ZipFile(archive, "r") as handle:
                members = {info.filename: handle.read(info) for info in handle.infolist()}
            original = members["byte_census.json"]
            members["byte_census.json"] = (
                b'{"schema":"stale","padding":"'
                + b"x" * 200_000
                + b'",'
                + original[1:]
            )
            forged = root / "outer_duplicate.zip"
            forged.write_bytes(sequence._canonical_zip_bytes(members))
            with self.assertRaisesRegex(ValueError, "duplicate JSON|canonical JSON|strict JSON"):
                sequence.validate_sequence_container(forged)

    def test_32_byte_non_zip_sequence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            forged = Path(temporary) / "forged.zip"
            forged.write_bytes(b"0" * 32)
            with self.assertRaises((ValueError, zipfile.BadZipFile)):
                sequence.validate_sequence_container(forged)

    def test_generic_counted_payload_is_not_a_gifstream_gop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._gop(root, 0)

            def generic(members):
                meta = json.loads(members["meta.json"])
                for name in sequence._required_gifstream_streams(meta, "official"):
                    members.pop(name, None)
                members["payload.bin"] = b"generic-but-counted"

            forged = self._repack(valid, generic)
            with self.assertRaisesRegex(ValueError, "GIFStream|meta|decoder closure"):
                sequence.validate_gop_archive(
                    forged, "flame_salmon_1", "official", 0
                )

    def test_missing_metadata_required_stream_is_rejected_after_recensus(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._gop(root, 0)
            forged = self._repack(valid, lambda members: members.pop("scales.bin"))
            with self.assertRaisesRegex(ValueError, "stream|decoder inputs|closure"):
                sequence.validate_gop_archive(
                    forged, "flame_salmon_1", "official", 0
                )

    def test_extra_generic_payload_is_rejected_after_recensus(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._gop(root, 0)
            forged = self._repack(
                valid, lambda members: members.__setitem__("payload.bin", b"extra")
            )
            with self.assertRaisesRegex(ValueError, "unmanaged|decoder inputs|exact decoder contract"):
                sequence.validate_gop_archive(
                    forged, "flame_salmon_1", "official", 0
                )

    def test_padding_json_member_is_rejected_after_recensus(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._gop(root, 0)
            forged = self._repack(
                valid, lambda members: members.__setitem__("padding.json", b"{}")
            )
            with self.assertRaisesRegex(ValueError, "exact decoder contract"):
                sequence.validate_gop_archive(
                    forged, "flame_salmon_1", "official", 0
                )

    def test_structural_json_extra_fields_are_rejected(self):
        for member in (
            "decoder_config.json",
            "runtime.json",
            "clean_decode_request.json",
        ):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                valid = self._gop(root, 0)

                def mutate(members):
                    value = json.loads(members[member])
                    value["padding"] = "x" * 200_000
                    members[member] = sequence.canonical_json_bytes(value)

                forged = self._repack(valid, mutate)
                with self.assertRaises(ValueError):
                    sequence.validate_gop_archive(
                        forged, "flame_salmon_1", "official", 0
                    )

    def test_inner_census_self_padding_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._gop(root, 0)
            with zipfile.ZipFile(valid, "r") as handle:
                members = {info.filename: handle.read(info) for info in handle.infolist()}
            census = json.loads(members["byte_census.json"])
            census["padding"] = "x" * 200_000
            members["byte_census.json"] = sequence.canonical_json_bytes(census)
            forged = root / "census_padding.zip"
            forged.write_bytes(sequence._canonical_zip_bytes(members))
            with self.assertRaisesRegex(ValueError, "census|canonical"):
                sequence.validate_gop_archive(
                    forged, "flame_salmon_1", "official", 0
                )

    def test_inner_census_integral_float_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._gop(root, 0)
            with zipfile.ZipFile(valid, "r") as handle:
                members = {info.filename: handle.read(info) for info in handle.infolist()}
            census = json.loads(members["byte_census.json"])
            census["file_count"] = float(census["file_count"])
            census["raw_bytes"] = float(census["raw_bytes"])
            for row in census["files"]:
                row["bytes"] = float(row["bytes"])
            members["byte_census.json"] = sequence.canonical_json_bytes(census)
            forged = root / "census_float_counts.zip"
            forged.write_bytes(sequence._canonical_zip_bytes(members))
            with self.assertRaisesRegex(ValueError, "integer|bytes"):
                sequence.validate_gop_archive(
                    forged, "flame_salmon_1", "official", 0
                )

    def test_inner_census_root_must_be_exact_rank0(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._gop(root, 0)
            with zipfile.ZipFile(valid, "r") as handle:
                members = {info.filename: handle.read(info) for info in handle.infolist()}
            census = json.loads(members["byte_census.json"])
            census["root"] = "payload"
            members["byte_census.json"] = sequence.canonical_json_bytes(census)
            forged = root / "census_wrong_root.zip"
            forged.write_bytes(sequence._canonical_zip_bytes(members))
            with self.assertRaisesRegex(ValueError, "exact recomputed object"):
                sequence.validate_gop_archive(
                    forged, "flame_salmon_1", "official", 0
                )

    def test_payload_manifest_integral_float_frame_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._gop(root, 0)

            def float_frame_fields(members):
                payload = json.loads(members["gifstream_payload_manifest.json"])
                payload["start_frame"] = float(payload["start_frame"])
                payload["GOP_size"] = float(payload["GOP_size"])
                payload_bytes = sequence.canonical_json_bytes(payload)
                members["gifstream_payload_manifest.json"] = payload_bytes
                config = json.loads(members["decoder_config.json"])
                config["payload_manifest_sha256"] = hashlib.sha256(
                    payload_bytes
                ).hexdigest()
                members["decoder_config.json"] = sequence.canonical_json_bytes(config)

            forged = self._repack(valid, float_frame_fields)
            with self.assertRaisesRegex(ValueError, "integer"):
                sequence.validate_gop_archive(
                    forged, "flame_salmon_1", "official", 0
                )

    def test_decoder_config_integral_float_discrete_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for field in (
                sequence.DECODER_POSITIVE_INT_FIELDS
                + sequence.DECODER_NONNEGATIVE_INT_FIELDS
            ):
                with self.subTest(field=field):
                    case_root = root / field
                    case_root.mkdir()
                    valid = self._gop(case_root, 0)

                    def float_field(members, field=field):
                        config = json.loads(members["decoder_config.json"])
                        config[field] = float(config[field])
                        members["decoder_config.json"] = sequence.canonical_json_bytes(
                            config
                        )

                    forged = self._repack(valid, float_field)
                    with self.assertRaisesRegex(ValueError, "integer"):
                        sequence.validate_gop_archive(
                            forged, "flame_salmon_1", "official", 0
                        )

    def test_producer_receipt_integral_float_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for field in ("start_frame", "GOP_size", "training_step"):
                with self.subTest(field=field):
                    case_root = root / field
                    case_root.mkdir()
                    valid = self._gop(case_root, 0)

                    def float_field(members, field=field):
                        producer = json.loads(members["producer_receipt.json"])
                        producer[field] = float(producer[field])
                        producer_payload = sequence.canonical_json_bytes(producer)
                        members["producer_receipt.json"] = producer_payload
                        config = json.loads(members["decoder_config.json"])
                        config["producer_receipt_sha256"] = hashlib.sha256(
                            producer_payload
                        ).hexdigest()
                        members["decoder_config.json"] = sequence.canonical_json_bytes(
                            config
                        )

                    forged = self._repack(valid, float_field)
                    with self.assertRaisesRegex(ValueError, "integer"):
                        sequence.validate_gop_archive(
                            forged, "flame_salmon_1", "official", 0
                        )

    def test_producer_training_config_noncanonical_discrete_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = [
                (field, lambda value: float(value))
                for field in (
                    sequence.PRODUCER_TRAINING_POSITIVE_INT_FIELDS
                    + sequence.PRODUCER_TRAINING_NONNEGATIVE_INT_FIELDS
                )
            ] + [("rate", lambda value: int(value))]
            for field, replacement in cases:
                with self.subTest(field=field):
                    case_root = root / field
                    case_root.mkdir()
                    valid = self._gop(case_root, 0)

                    def noncanonical_field(
                        members, field=field, replacement=replacement
                    ):
                        producer = json.loads(members["producer_receipt.json"])
                        training = producer["training_config"]
                        training[field] = replacement(training[field])
                        producer["training_config_sha256"] = hashlib.sha256(
                            sequence.canonical_json_bytes(training)
                        ).hexdigest()
                        producer_payload = sequence.canonical_json_bytes(producer)
                        members["producer_receipt.json"] = producer_payload
                        config = json.loads(members["decoder_config.json"])
                        config["producer_receipt_sha256"] = hashlib.sha256(
                            producer_payload
                        ).hexdigest()
                        members["decoder_config.json"] = sequence.canonical_json_bytes(
                            config
                        )

                    forged = self._repack(valid, noncanonical_field)
                    with self.assertRaisesRegex(
                        ValueError, "integer|canonical frozen rate string"
                    ):
                        sequence.validate_gop_archive(
                            forged, "flame_salmon_1", "official", 0
                        )

    def test_training_rate_and_boolean_types_are_role_exact(self):
        audit = sequence._validate_producer_training_config_types(
            dict(TRAINING_CONFIG), "training"
        )
        self.assertEqual(audit["rate_index"], 0)
        for value in (0, 0.0, "00", True):
            with self.subTest(rate=value):
                config = dict(TRAINING_CONFIG)
                config["rate"] = value
                with self.assertRaisesRegex(ValueError, "canonical frozen rate string"):
                    sequence._validate_producer_training_config_types(config, "training")
        for field in sequence.PRODUCER_TRAINING_BOOL_FIELDS:
            with self.subTest(field=field):
                config = dict(TRAINING_CONFIG)
                config[field] = int(config[field])
                with self.assertRaisesRegex(ValueError, "boolean"):
                    sequence._validate_producer_training_config_types(config, "training")

    def test_linked_training_floating_strings_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for field, value in (
                ("voxel_size", "0.010000000000000000000000000000000000000000"),
                ("rd_lambda", "0.000500000000000000000000000000000000000000"),
            ):
                with self.subTest(field=field):
                    case_root = root / field
                    case_root.mkdir()
                    valid = self._gop(case_root, 0)

                    def string_float(members, field=field, value=value):
                        training_receipt = json.loads(members["training_receipt.json"])
                        producer = json.loads(members["producer_receipt.json"])
                        config = json.loads(members["decoder_config.json"])
                        training = dict(producer["training_config"])
                        training[field] = value
                        training_sha = hashlib.sha256(
                            sequence.canonical_json_bytes(training)
                        ).hexdigest()
                        training_receipt["training_config"] = training
                        training_receipt["training_config_sha256"] = training_sha
                        training_payload = sequence.canonical_json_bytes(
                            training_receipt
                        )
                        producer["training_config"] = training
                        producer["training_config_sha256"] = training_sha
                        producer["training_receipt_sha256"] = hashlib.sha256(
                            training_payload
                        ).hexdigest()
                        producer_payload = sequence.canonical_json_bytes(producer)
                        config["training_receipt_sha256"] = hashlib.sha256(
                            training_payload
                        ).hexdigest()
                        config["producer_receipt_sha256"] = hashlib.sha256(
                            producer_payload
                        ).hexdigest()
                        if field in config:
                            config[field] = value
                        members["training_receipt.json"] = training_payload
                        members["producer_receipt.json"] = producer_payload
                        members["decoder_config.json"] = sequence.canonical_json_bytes(
                            config
                        )

                    forged = self._repack(valid, string_float)
                    with self.assertRaisesRegex(ValueError, "exact JSON floating"):
                        sequence.validate_gop_archive(
                            forged, "flame_salmon_1", "official", 0
                        )

    def test_counted_runtime_floating_strings_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for field in (
                "encode_seconds",
                "model_load_plus_entropy_decode_seconds",
            ):
                with self.subTest(field=field):
                    case_root = root / field
                    case_root.mkdir()
                    valid = self._gop(case_root, 0)

                    def string_float(members, field=field):
                        runtime = json.loads(members["runtime.json"])
                        runtime[field] = f"{runtime[field]:.40f}"
                        members["runtime.json"] = sequence.canonical_json_bytes(runtime)

                    forged = self._repack(valid, string_float)
                    with self.assertRaisesRegex(ValueError, "exact JSON floating"):
                        sequence.validate_gop_archive(
                            forged, "flame_salmon_1", "official", 0
                        )

    def test_counted_metadata_floats_cannot_use_integer_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for field in ("anchor_min", "scale"):
                with self.subTest(field=field):
                    case_root = root / field
                    case_root.mkdir()
                    valid = self._gop(case_root, 0)

                    def integer_alias(members, field=field):
                        meta = json.loads(members["meta.json"])
                        if field == "anchor_min":
                            meta["anchors"]["mins"][0] = 0
                        else:
                            meta["scales"]["scaling"] = 1
                        meta_payload = sequence.canonical_json_bytes(meta)
                        members["meta.json"] = meta_payload
                        payload = json.loads(
                            members["gifstream_payload_manifest.json"]
                        )
                        payload["meta_sha256"] = hashlib.sha256(
                            meta_payload
                        ).hexdigest()
                        meta_row = next(
                            row
                            for row in payload["decoder_inputs"]
                            if row["path"] == "meta.json"
                        )
                        meta_row["bytes"] = len(meta_payload)
                        meta_row["sha256"] = hashlib.sha256(meta_payload).hexdigest()
                        payload_bytes = sequence.canonical_json_bytes(payload)
                        members["gifstream_payload_manifest.json"] = payload_bytes
                        config = json.loads(members["decoder_config.json"])
                        config["payload_manifest_sha256"] = hashlib.sha256(
                            payload_bytes
                        ).hexdigest()
                        members["decoder_config.json"] = sequence.canonical_json_bytes(
                            config
                        )

                    forged = self._repack(valid, integer_alias)
                    with self.assertRaisesRegex(ValueError, "exact JSON floating"):
                        sequence.validate_gop_archive(
                            forged, "flame_salmon_1", "official", 0
                        )

    def test_inner_zip_comment_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            forged = self._gop(root, 0)
            with zipfile.ZipFile(forged, "a") as handle:
                handle.comment = b"x" * 65535
            with self.assertRaisesRegex(ValueError, "ZIP bytes/metadata"):
                sequence.validate_gop_archive(
                    forged, "flame_salmon_1", "official", 0
                )

    def test_counted_binary_ignored_tails_are_rejected(self):
        for member in (
            "camera_metadata/intrinsics.npy",
            "quats.npz",
            "anchors_l.png",
            "scales.bin",
        ):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                valid = self._gop(root, 0)
                forged = self._repack(
                    valid,
                    lambda members, name=member: members.__setitem__(
                        name, members[name] + b"x" * 200_000
                    ),
                )
                with self.assertRaises(ValueError):
                    sequence.validate_gop_archive(
                        forged, "flame_salmon_1", "official", 0
                    )

    def test_camera_npy_roles_reject_unconsumed_extent_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._gop(root, 0)
            padded_name = fixture_camera_names(width=100)
            forged = self._repack(
                valid,
                lambda members: members.__setitem__(
                    "camera_metadata/camera_names.npy", padded_name
                ),
            )
            with self.assertRaisesRegex(ValueError, "camera-name.*width"):
                sequence.validate_gop_archive(
                    forged, "flame_salmon_1", "official", 0
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._gop(root, 0)
            forged = self._repack(
                valid,
                lambda members: members.__setitem__(
                    "camera_metadata/bounds.npy",
                    fixture_npy((2,), descr="<f8"),
                ),
            )
            with self.assertRaisesRegex(ValueError, "bounds dtype/shape"):
                sequence.validate_gop_archive(
                    forged, "flame_salmon_1", "official", 0
                )

        coordinated = {
            "camera_metadata/camera_keys.npy": fixture_npy((1000,), descr="<i8"),
            "camera_metadata/intrinsics.npy": fixture_npy((1000, 3, 3), descr="<f8"),
            "camera_metadata/image_sizes.npy": fixture_npy((1000, 2), descr="<i8"),
            "camera_metadata/camtoworlds.npy": fixture_npy((1000, 4, 4), descr="<f8"),
            "camera_metadata/camera_ids.npy": fixture_npy((1000,), descr="<i8"),
            "camera_metadata/camera_names.npy": fixture_npy(
                (1000,), descr="<U1", fill=b"a\0\0\0"
            ),
            "camera_metadata/transform.npy": fixture_npy((4, 4), descr="<f8"),
            "camera_metadata/bounds.npy": fixture_npy((1000, 2), descr="<f8"),
        }
        with zipfile.ZipFile(
            io.BytesIO(sequence._canonical_zip_bytes(coordinated)), "r"
        ) as handle, self.assertRaisesRegex(ValueError, "frozen camera grid"):
            sequence._camera_metadata_contract(
                handle, sequence.frozen_camera_names("flame_salmon_1")
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._gop(root, 0)
            forged = self._repack(
                valid,
                lambda members: members.__setitem__(
                    "camera_metadata/transform.npy",
                    fixture_npy((1000, 1000), descr="<f8"),
                ),
            )
            with self.assertRaisesRegex(ValueError, "camera metadata role"):
                sequence.validate_gop_archive(
                    forged, "flame_salmon_1", "official", 0
                )

    def test_png_canonical_pixel_representation_rejects_rate_padding(self):
        canonical = fixture_png()
        expected_geometry = {
            "width": 1,
            "height": 1,
            "bit_depth": 8,
            "color_type": 2,
        }
        sequence._png_payload_contract(canonical, "canonical", expected_geometry)
        for forged in (
            fixture_png(split_idat=True),
            fixture_png(filter_type=1),
            fixture_png(compression_level=0),
        ):
            with self.assertRaisesRegex(ValueError, "canonical pixel representation"):
                sequence._png_payload_contract(forged, "forged", expected_geometry)
        with self.assertRaisesRegex(ValueError, "geometry/type"):
            sequence._png_payload_contract(
                fixture_png(bit_depth=16), "forged", expected_geometry
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "meta.json").write_bytes(
                sequence.canonical_json_bytes({"anchors": {"shape": [1, 3]}})
            )
            paths = [root / "anchors_l.png", root / "anchors_u.png"]
            for path in paths:
                path.write_bytes(fixture_png(split_idat=True))
            audit = sequence.canonicalize_gifstream_png_payloads(root)
            self.assertEqual(len(audit), 2)
            for path in paths:
                self.assertEqual(path.read_bytes(), canonical)
                sequence._png_payload_contract(
                    path.read_bytes(), path.name, expected_geometry
                )

    def test_npz_compression_must_match_the_frozen_producer(self):
        members = {"arr.npy": fixture_npy((1, 3))}
        compressed = fixture_npz(members, compressed=True)
        stored = fixture_npz(members, compressed=False)
        sequence._npz_payload_contract(
            compressed,
            "codec.npz",
            {"arr.npy"},
            expected_compression=zipfile.ZIP_DEFLATED,
        )
        with self.assertRaisesRegex(ValueError, "differs from its producer"):
            sequence._npz_payload_contract(
                stored,
                "codec.npz",
                {"arr.npy"},
                expected_compression=zipfile.ZIP_DEFLATED,
            )
        sequence._npz_payload_contract(
            stored,
            "score.npz",
            {"arr.npy"},
            expected_compression=zipfile.ZIP_STORED,
        )
        with self.assertRaisesRegex(ValueError, "differs from its producer"):
            sequence._npz_payload_contract(
                compressed,
                "score.npz",
                {"arr.npy"},
                expected_compression=zipfile.ZIP_STORED,
            )

    def test_npz_member_order_must_match_the_frozen_producer(self):
        first = fixture_npy((1,), descr="<i8", fill=b"\1")
        second = fixture_npy((1,), descr="<i8", fill=b"\2")
        expected_order = ("first.npy", "second.npy")
        canonical = fixture_npz(
            {"first.npy": first, "second.npy": second}, compressed=False
        )
        reordered = fixture_npz(
            {"second.npy": second, "first.npy": first}, compressed=False
        )
        sequence._npz_payload_contract(
            canonical,
            "ordered.npz",
            set(expected_order),
            expected_compression=zipfile.ZIP_STORED,
            expected_order=expected_order,
        )
        with self.assertRaisesRegex(ValueError, "member order"):
            sequence._npz_payload_contract(
                reordered,
                "reordered.npz",
                set(expected_order),
                expected_compression=zipfile.ZIP_STORED,
                expected_order=expected_order,
            )

    def test_codec_npz_dtype_and_shape_bind_counted_metadata(self):
        metadata = {"shape": [1024, 4], "dtype": "float32"}
        sequence._codec_npz_payload_contract(
            fixture_npz({"arr.npy": fixture_npy((1024, 4), descr="<f4")}),
            "quats.npz",
            metadata,
        )
        for payload in (
            fixture_npz({"arr.npy": fixture_npy((1024, 4), descr="<f8")}),
            fixture_npz({"arr.npy": fixture_npy((2048, 2), descr="<f4")}),
        ):
            with self.assertRaisesRegex(ValueError, "dtype/shape"):
                sequence._codec_npz_payload_contract(
                    payload, "quats.npz", metadata
                )

    def test_npz_member_schema_binds_each_counted_dtype_and_shape(self):
        schema = {
            "score.npy": ("<f8", (3,)),
            "mask.npy": ("|b1", (3,)),
        }
        order = ("score.npy", "mask.npy")
        honest = fixture_npz(
            {
                "score.npy": fixture_npy((3,), descr="<f8"),
                "mask.npy": fixture_npy((3,), descr="|b1"),
            },
            compressed=False,
        )
        sequence._npz_member_schema_contract(
            honest, "score.npz", schema, order
        )
        wider = fixture_npz(
            {
                "score.npy": fixture_npy((3,), descr="<f16"),
                "mask.npy": fixture_npy((3,), descr="|b1"),
            },
            compressed=False,
        )
        with self.assertRaisesRegex(ValueError, "dtype/shape"):
            sequence._npz_member_schema_contract(
                wider, "score.npz", schema, order
            )

    def test_ap_identity_corrections_bind_exact_mask_and_codes(self):
        payload = bytes([0b00000101, 14, 12])
        record = {
            "schema": "h007.ap_identity_corrections.v1",
            "path": "ap_identity_corrections.bin",
            "row_count": 3,
            "mismatch_count": 2,
            "mask_bytes": 1,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bitorder": "little",
            "base": "round-decoded-anchor-div-voxel-size",
            "code": "uint8-base3-dx-dy-dz-plus1",
        }
        sequence._identity_correction_payload_contract(payload, record, "fixture")
        for forged in (bytes([0b00000101, 27, 12]), bytes([0b00000101, 13, 12])):
            changed = dict(record)
            changed["sha256"] = hashlib.sha256(forged).hexdigest()
            with self.assertRaisesRegex(ValueError, "invalid base-3"):
                sequence._identity_correction_payload_contract(
                    forged, changed, "fixture"
                )

    def test_json_integer_helpers_reject_integral_floats(self):
        self.assertEqual(sequence._positive_int(3, "positive"), 3)
        self.assertEqual(sequence._nonnegative_int(0, "nonnegative"), 0)
        for helper, value in (
            (sequence._positive_int, 3.0),
            (sequence._nonnegative_int, 0.0),
        ):
            with self.assertRaisesRegex(ValueError, "integer"):
                helper(value, "count")

    def test_reference_nets_have_a_separate_exact_model_key_contract(self):
        compressed = sequence.expected_gifstream_nets_keys(False)
        reference = sequence.expected_gifstream_reference_nets_keys(False)
        self.assertEqual(reference, {"decoders", "scaling"})
        self.assertEqual(
            sequence.expected_gifstream_reference_nets_keys(True),
            {"decoders", "scaling", "app_module"},
        )
        self.assertEqual(
            compressed - reference,
            {
                "scales_entropy_model",
                "anchor_features_entropy_model",
                "offsets_entropy_model",
                "factors_entropy_model",
                "time_features_entropy_model",
            },
        )

    def test_entropy_and_mask_payload_extents_are_canonical(self):
        sequence._entropy_stream_contract(fixture_entropy(b"12345678"), "entropy")
        with self.assertRaisesRegex(ValueError, "ignored or truncated"):
            sequence._entropy_stream_contract(fixture_entropy(b"123456789"), "entropy")
        sequence._packed_bool_mask_contract(b"\x05", 3, 2, "mask")
        for payload, count, true_count in (
            (b"\x05\0", 3, 2),
            (b"\x85", 3, 3),
            (b"\x05", 3, 1),
        ):
            with self.assertRaises(ValueError):
                sequence._packed_bool_mask_contract(payload, count, true_count, "mask")

    def test_torch_save_container_rejects_ignored_tail_and_comment(self):
        payload = fixture_torch_save()
        sequence._torch_save_zip_contract(payload, "nets.pt")
        with self.assertRaisesRegex(ValueError, "comment or ignored trailing"):
            sequence._torch_save_zip_contract(payload + b"padding", "nets.pt")
        commented = io.BytesIO()
        with zipfile.ZipFile(commented, "w", compression=zipfile.ZIP_STORED) as handle:
            handle.writestr("nets/data.pkl", b"pickle")
            handle.writestr("nets/data/0", b"storage")
            handle.writestr("nets/version", b"3\n")
            handle.comment = b"padding"
        with self.assertRaisesRegex(ValueError, "comment or ignored trailing"):
            sequence._torch_save_zip_contract(commented.getvalue(), "nets.pt")

    def test_torch_fixture_matches_frozen_roles_shapes_scaling_and_receipt_hashes(self):
        payload, audit = fixture_torch_save(return_audit=True)
        self.assertEqual(sequence._torch_save_zip_contract(payload, "nets.pt"), audit)
        expected_schema = sequence._expected_nets_tensor_schema(FIXTURE_DECODER_CONFIG)
        self.assertEqual(set(audit["state_schema"]), set(expected_schema))
        self.assertEqual(
            audit["storage_count"],
            sum(len(rows) for rows in expected_schema.values()),
        )
        self.assertEqual(
            audit["top_level_keys"],
            [
                "decoders",
                "scales_entropy_model",
                "anchor_features_entropy_model",
                "offsets_entropy_model",
                "factors_entropy_model",
                "time_features_entropy_model",
                "scaling",
            ],
        )
        self.assertEqual(
            audit["scaling"], sequence._expected_codec_scaling(FIXTURE_DECODER_CONFIG["rate"])
        )
        producer_state = {
            "decoders": audit["state_sha256"]["decoders"],
            "entropy_models": {
                name: audit["state_sha256"][f"{name}_entropy_model"]
                for name in sequence.GIFSTREAM_ENTROPY_MODEL_KEYS
            },
            "codec_scaling": audit["scaling_sha256"],
            "appearance_module": None,
        }
        sequence._validate_nets_audit(
            audit, FIXTURE_DECODER_CONFIG, producer_state
        )

    def test_torch_save_container_rejects_stale_serialization_id(self):
        payload, audit = fixture_torch_save(return_audit=True)
        with zipfile.ZipFile(io.BytesIO(payload), "r") as handle:
            members = {info.filename: handle.read(info) for info in handle.infolist()}
        stale = b"9" * 40
        if stale.decode("ascii") == audit["serialization_id"]:
            stale = b"8" * 40
        members["nets/.data/serialization_id"] = stale
        forged = sequence._canonical_torch_zip_bytes("nets", members)
        with self.assertRaisesRegex(ValueError, "bytes are not canonical"):
            sequence._torch_save_zip_contract(forged, "stale serialization fixture")

    def test_torch_save_container_rejects_shape_stride_and_offset_drift(self):
        _, honest_audit = fixture_torch_save(return_audit=True)
        producer_state = {
            "decoders": honest_audit["state_sha256"]["decoders"],
            "entropy_models": {
                name: honest_audit["state_sha256"][f"{name}_entropy_model"]
                for name in sequence.GIFSTREAM_ENTROPY_MODEL_KEYS
            },
            "codec_scaling": honest_audit["scaling_sha256"],
            "appearance_module": None,
        }
        _, wrong_shape_audit = fixture_torch_save(
            tensor_shape=(32, 40), return_audit=True
        )
        with self.assertRaisesRegex(ValueError, "tensor schema differs"):
            sequence._validate_nets_audit(
                wrong_shape_audit, FIXTURE_DECODER_CONFIG, producer_state
            )
        with self.assertRaisesRegex(ValueError, "storage closure is malformed"):
            sequence._torch_save_zip_contract(
                fixture_torch_save(tensor_stride=(1, 32)), "stride fixture"
            )
        with self.assertRaisesRegex(ValueError, "storage closure is malformed"):
            sequence._torch_save_zip_contract(
                fixture_torch_save(tensor_storage_offset=1), "offset fixture"
            )

    def test_torch_save_container_rejects_inner_padding_channels(self):
        with self.assertRaisesRegex(ValueError, "pickle has ignored trailing"):
            sequence._torch_save_zip_contract(
                fixture_torch_save(pickle_tail=b"padding"), "nets.pt"
            )
        with self.assertRaisesRegex(ValueError, "unreferenced or missing storage"):
            sequence._torch_save_zip_contract(
                fixture_torch_save(unreferenced_storage=True), "nets.pt"
            )
        with self.assertRaisesRegex(ValueError, "storage byte extent"):
            sequence._torch_save_zip_contract(
                fixture_torch_save(storage_padding=b"padding"), "nets.pt"
            )
        with self.assertRaisesRegex(ValueError, "semantic no-op"):
            sequence._torch_save_zip_contract(
                fixture_torch_save(pickle_noops=100_000), "nets.pt"
            )
        with self.assertRaisesRegex(ValueError, "integer encoding is nonminimal"):
            sequence._torch_save_zip_contract(
                fixture_torch_save(nonminimal_numel=True), "nets.pt"
            )
        with self.assertRaisesRegex(ValueError, "root differs"):
            sequence._torch_save_zip_contract(
                fixture_torch_save(root="n" * 20_000), "nets.pt"
            )
        with self.assertRaisesRegex(ValueError, "unmanaged record"):
            sequence._torch_save_zip_contract(
                fixture_torch_save(extra_fixed_records=True), "nets.pt"
            )
        with self.assertRaisesRegex(ValueError, "serialization ID width"):
            sequence._torch_save_zip_contract(
                fixture_torch_save(serialization_id=b"0"), "nets.pt"
            )
        malformed_extra = bytearray(fixture_torch_save())
        first_extra = 30 + len("nets/data.pkl")
        malformed_extra[first_extra] = ord("X")
        with self.assertRaisesRegex(ValueError, "alignment extra"):
            sequence._torch_save_zip_contract(bytes(malformed_extra), "nets.pt")

    def test_torch_save_container_rejects_duplicate_key_overwrite_padding(self):
        payload = fixture_torch_save()
        with zipfile.ZipFile(io.BytesIO(payload), "r") as handle:
            members = {info.filename: handle.read(info) for info in handle.infolist()}
        pickle_payload = members["nets/data.pkl"]
        self.assertEqual(pickle_payload[:3], b"\x80\x02}")
        key = b"decoders"
        padding = b"x" * 100_000
        prior_write = (
            pickle.BINUNICODE
            + struct.pack("<I", len(key))
            + key
            + pickle.BINUNICODE
            + struct.pack("<I", len(padding))
            + padding
            + pickle.SETITEM
        )
        members["nets/data.pkl"] = pickle_payload[:3] + prior_write + pickle_payload[3:]
        forged = sequence._canonical_torch_zip_bytes("nets", members)
        with self.assertRaisesRegex(ValueError, "storage closure is malformed"):
            sequence._torch_save_zip_contract(forged, "duplicate-setitem fixture")

    def test_torch_save_container_rejects_unconsumed_top_level_key(self):
        with self.assertRaisesRegex(ValueError, "dictionary keys are not exact"):
            sequence._torch_save_zip_contract(
                fixture_torch_save(extra_root_padding=200_000), "extra-root fixture"
            )

    def test_torch_save_container_rejects_ignored_lower_stack_padding(self):
        payload = fixture_torch_save()
        with zipfile.ZipFile(io.BytesIO(payload), "r") as handle:
            members = {info.filename: handle.read(info) for info in handle.infolist()}
        pickle_payload = members["nets/data.pkl"]
        self.assertEqual(pickle_payload[:2], b"\x80\x02")
        padding = b"x" * 100_000
        hidden_value = (
            pickle.BINUNICODE + struct.pack("<I", len(padding)) + padding
        )
        members["nets/data.pkl"] = pickle_payload[:2] + hidden_value + pickle_payload[2:]
        forged = sequence._canonical_torch_zip_bytes("nets", members)
        with self.assertRaisesRegex(ValueError, "storage closure is malformed"):
            sequence._torch_save_zip_contract(forged, "lower-stack fixture")

    def test_ap_metadata_cannot_redirect_decoder_outside_archive(self):
        meta = {
            name: {"shape": [1], "dtype": "float32"}
            for name in (
                "anchors",
                "scales",
                "quats",
                "opacities",
                "offsets",
                "factors",
            )
        }
        meta["anchor_features"] = {
            "shape": [1, 1],
            "dtype": "float32",
            "length": 1,
        }
        meta["time_features"] = {
            "shape": [1, 60, 1],
            "dtype": "float32",
            "length": 1,
        }
        mask = {
            "path": "ap_class_mask.bin",
            "count": 1,
            "true_count": 1,
            "bytes": 1,
            "sha256": "a" * 64,
            "bitorder": "little",
        }
        identity_corrections = {
            "schema": "h007.ap_identity_corrections.v1",
            "path": "ap_identity_corrections.bin",
            "row_count": 1,
            "mismatch_count": 0,
            "mask_bytes": 1,
            "bytes": 1,
            "sha256": "a" * 64,
            "bitorder": "little",
            "base": "round-decoded-anchor-div-voxel-size",
            "code": "uint8-base3-dx-dy-dz-plus1",
        }
        meta["__ap__"] = {
            "schema": "h007.ap_gifstream.codec.v6",
            "variant": {"name": "ap-gifstream-full"},
            "score": {"score_artifact_sha256": "a" * 64},
            "allocation": {},
            "runtime_provenance": {},
            "compression_seed": 1,
            "q_ap_multiplier": 0.5,
            "q_bg_multiplier": 1.25,
            "mask": {**mask, "path": "../outside.bin"},
            "real_row_mask": {**mask, "path": "ap_real_row_mask.bin"},
            "padding_row_mask": {**mask, "path": "ap_padding_row_mask.bin"},
            "active_row_mask": {**mask, "path": "ap_active_row_mask.bin"},
            "identity_corrections": identity_corrections,
        }
        with self.assertRaisesRegex(ValueError, "unsafe/noncanonical|fields are unexpected"):
            sequence._required_gifstream_streams(meta, "ap-gifstream-full")

    def test_real_ap_producer_metadata_is_accepted_for_quantized_and_swap_only_variants(self):
        quantized = fixture_ap_meta("ap-gifstream-full")
        quantized_streams = sequence._required_gifstream_streams(
            quantized, "ap-gifstream-full"
        )
        self.assertIn("time_features_path_00000.bin", quantized_streams)
        self.assertIn("time_features_bg_00000.bin", quantized_streams)
        self.assertIn("anchor_features_path_00000.bin", quantized_streams)
        self.assertIn("factors_path.bin", quantized_streams)

        swap_only = fixture_ap_meta("path-swap")
        swap_streams = sequence._required_gifstream_streams(swap_only, "path-swap")
        self.assertIn("time_features_00000.bin", swap_streams)
        self.assertNotIn("time_features_path_00000.bin", swap_streams)

    def test_ap_score_source_count_may_differ_from_encoded_padded_count(self):
        meta = fixture_ap_meta("ap-gifstream-full")
        meta["__ap__"]["score"]["anchor_count"] = 6
        meta["__ap__"]["score"]["eligible_count"] = 5
        streams = sequence._required_gifstream_streams(
            meta, "ap-gifstream-full"
        )
        self.assertIn("ap_identity_corrections.bin", streams)

        meta["__ap__"]["score"]["anchor_count"] = 2
        meta["__ap__"]["score"]["eligible_count"] = 2
        with self.assertRaisesRegex(ValueError, "allocation categorical"):
            sequence._required_gifstream_streams(meta, "ap-gifstream-full")

    def test_real_ap_allocation_and_variant_contract_rejects_coordinated_aliases(self):
        for path, value in (
            (("allocation", "budget_source"), "runtime_recomputed"),
            (("allocation", "official_retain_count"), 4),
            (("allocation", "current_vs_frozen_whole_xor"), 5),
            (("allocation", "official_estimated_time_bytes"), 101),
            (("variant", "quant"), False),
        ):
            meta = fixture_ap_meta("ap-gifstream-full")
            target = meta["__ap__"]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaisesRegex(
                ValueError, "allocation categorical|variant metadata"
            ):
                sequence._required_gifstream_streams(meta, "ap-gifstream-full")

    def test_swap_only_rejects_two_class_temporal_representation(self):
        meta = fixture_ap_meta("path-swap")
        meta["time_features"] = fixture_ap_meta("ap-gifstream-full")["time_features"]
        with self.assertRaisesRegex(ValueError, "metadata fields are unexpected"):
            sequence._required_gifstream_streams(meta, "path-swap")

    def test_edit_source_variant_matches_honest_official_and_ablation_producers(self):
        self.assertEqual(
            sequence._edit_source_variant("official"), "ap-gifstream-full"
        )
        for method in sequence.AP_VARIANT_METADATA:
            self.assertEqual(sequence._edit_source_variant(method), method)
        with self.assertRaisesRegex(ValueError, "outside the frozen AP variants"):
            sequence._edit_source_variant("unknown")

    def test_ap_seed_and_quantizers_bind_frozen_training_controls(self):
        score_sha = "a" * 64
        ap_meta = {
            "compression_seed": 20260715,
            "q_ap_multiplier": 0.5,
            "q_bg_multiplier": 1.25,
            "score": {
                "score_artifact_sha256": score_sha,
                "q_ap_multiplier": 0.5,
                "q_bg_multiplier": 1.25,
            },
        }
        training = {
            "score_sha256": score_sha,
            "q_ap_multiplier": 0.5,
            "q_bg_multiplier": 1.25,
        }
        sequence.validate_ap_seed_quantizer_closure(
            ap_meta, training, compression_seed=20260715
        )
        for path, value in (
            (("compression_seed",), 20260716),
            (("q_ap_multiplier",), 0.75),
            (("score", "q_bg_multiplier"), 1.5),
        ):
            tampered = json.loads(json.dumps(ap_meta))
            target = tampered
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "closure"):
                sequence.validate_ap_seed_quantizer_closure(
                    tampered, training, compression_seed=20260715
                )

    def test_forged_frame_metrics_fail_frozen_evaluator_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated_root = root / "generated"
            prediction_dir = generated_root / "predictions"
            reference_dir = root / "cam00"
            prediction_dir.mkdir(parents=True)
            reference_dir.mkdir()
            member_hashes = {}
            frame_rows = []
            for frame in range(300):
                prediction = prediction_dir / f"frame_{frame:05d}.png"
                reference = reference_dir / f"{frame + 1:05d}.png"
                prediction.write_bytes(f"prediction-{frame}".encode())
                reference.write_bytes(f"reference-{frame}".encode())
                reference_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
                member_hashes[f"cam00/{frame + 1:05d}.png"] = reference_sha
                frame_rows.append(
                    {
                        "frame": frame,
                        "prediction": f"generated/predictions/frame_{frame:05d}.png",
                        "reference": f"cam00/{frame + 1:05d}.png",
                        "prediction_bytes": prediction.stat().st_size,
                        "prediction_sha256": hashlib.sha256(
                            prediction.read_bytes()
                        ).hexdigest(),
                        "reference_sha256": reference_sha,
                        "psnr": 30.0,
                        "ssim": 0.9,
                        "lpips": 0.1,
                    }
                )
            sequence_archive = root / "sequence.zip"
            sequence_archive.write_bytes(b"counted-sequence")
            source_manifest = root / "source_data.json"
            source_manifest.write_bytes(b"source-data-manifest")
            runtime_manifest = root / "runtime.json"
            runtime_manifest.write_bytes(b"runtime-manifest")
            evaluator_path = root / sequence.ORDINARY_EVALUATOR_RELATIVE_PATH
            evaluator_path.parent.mkdir(exist_ok=True)
            evaluator_path.write_bytes(b"frozen-evaluator")
            clean_decoder_path = evaluator_path.with_name(
                "h007_clean_decode_gifstream.py"
            )
            clean_decoder_path.write_bytes(b"frozen-clean-decoder")
            runtime = {
                "manifest_sha256": hashlib.sha256(
                    runtime_manifest.read_bytes()
                ).hexdigest()
            }
            validation = {
                "archive_sha256": hashlib.sha256(
                    sequence_archive.read_bytes()
                ).hexdigest(),
                "archive_bytes": sequence_archive.stat().st_size,
                "training_config_sha256": "c" * 64,
                "seed": 17,
                "gops": [
                    {
                        "sha256": f"{index + 1:x}" * 64,
                        "encode_seconds": 1.0,
                        "decode_seconds": 2.0,
                    }
                    for index in range(5)
                ],
            }
            timing_rows = []
            for gop_id in range(5):
                clean = {
                    "schema": "h007.clean_decode_result.v2",
                    "source_sequence_archive_sha256": validation["archive_sha256"],
                    "source_gop_id": gop_id,
                    "source_inner_gop_sha256": validation["gops"][gop_id]["sha256"],
                    "runtime_provenance": runtime,
                    "decoded_splats_sha256": f"{gop_id + 5:x}" * 64,
                    "tensors": {"anchors": {"sha256": f"{gop_id + 10:x}" * 64}},
                    "counted_camera_render": {
                        "timed_renders": 30,
                        "seconds": 3.0,
                        "fps": 10.0,
                    },
                }
                clean_path = (
                    generated_root
                    / f"gop_{gop_id}"
                    / "clean_bundle"
                    / "clean_decode_manifest.json"
                )
                clean_path.parent.mkdir(parents=True)
                clean_path.write_bytes(sequence.canonical_json_bytes(clean))
                timing_rows.append(
                    {
                        "gop_id": gop_id,
                        "inner_gop_sha256": validation["gops"][gop_id]["sha256"],
                        "encode_seconds": 1.0,
                        "decode_seconds": 2.0,
                        "clean_decode_receipt": clean_path.relative_to(root).as_posix(),
                        "clean_decode_receipt_sha256": hashlib.sha256(
                            clean_path.read_bytes()
                        ).hexdigest(),
                        "decoded_splats_sha256": clean[
                            "decoded_splats_sha256"
                        ],
                        "decoded_tensor_manifest_sha256": hashlib.sha256(
                            sequence.canonical_json_bytes(clean["tensors"])
                        ).hexdigest(),
                        "prediction_camera_binding": {
                            "source_camera": "cam00",
                            "dataset_camera_index": 0,
                            "pose_index": 0,
                            "camera_key": 1,
                            "counted_camera_name": "camera_0",
                            "frame_size": [64, 64],
                            "local_frames": list(range(60)),
                        },
                        "rendered_frames": 30,
                        "render_elapsed_seconds": 3.0,
                        "render_fps": 10.0,
                    }
                )
            receipt = {
                "schema": sequence.EVALUATOR_RECEIPT_SCHEMA,
                "scene": "flame_salmon_1",
                "method": "official",
                "point_id": "p0",
                "sequence_archive": sequence_archive.name,
                "archive_sha256": validation["archive_sha256"],
                "archive_bytes": validation["archive_bytes"],
                "training_config_sha256": validation["training_config_sha256"],
                "seed": 17,
                "source_data_manifest": source_manifest.name,
                "source_data_manifest_sha256": hashlib.sha256(
                    source_manifest.read_bytes()
                ).hexdigest(),
                "runtime_provenance_manifest": runtime_manifest.name,
                "runtime_provenance_manifest_sha256": runtime["manifest_sha256"],
                "clean_decoder_relative_path": "examples/h007_clean_decode_gifstream.py",
                "clean_decoder_sha256": hashlib.sha256(
                    clean_decoder_path.read_bytes()
                ).hexdigest(),
                "generated_predictions_root": generated_root.name,
                "evaluator_relative_path": sequence.ORDINARY_EVALUATOR_RELATIVE_PATH,
                "evaluator_sha256": hashlib.sha256(
                    evaluator_path.read_bytes()
                ).hexdigest(),
                "metric_device": "cpu",
                "metric_protocol": sequence.ORDINARY_METRIC_PROTOCOL,
                "frame_metrics": frame_rows,
                "timing_trials": timing_rows,
                "outcome_fields_read": [
                    "ordinary_unedited_fidelity",
                    "real_container_accounting",
                ],
            }
            receipt_path = root / "receipt.json"
            receipt_path.write_bytes(sequence.canonical_json_bytes(receipt))
            replay = {
                "archive_sha256": validation["archive_sha256"],
                "clean_decoder_sha256": receipt["clean_decoder_sha256"],
                "prediction_sha256": {
                    int(row["frame"]): row["prediction_sha256"]
                    for row in frame_rows
                },
                "decoded_splats_sha256": {
                    int(row["gop_id"]): row["decoded_splats_sha256"]
                    for row in timing_rows
                },
                "decoded_tensor_manifest_sha256": {
                    int(row["gop_id"]): row[
                        "decoded_tensor_manifest_sha256"
                    ]
                    for row in timing_rows
                },
                "prediction_camera_binding": {
                    int(row["gop_id"]): row["prediction_camera_binding"]
                    for row in timing_rows
                },
                "metrics": {
                    frame: {"psnr": 31.0, "ssim": 0.9, "lpips": 0.1}
                    for frame in range(300)
                },
            }
            arbitrary_png_replay = {
                **replay,
                "prediction_sha256": dict(replay["prediction_sha256"]),
            }
            arbitrary_png_replay["prediction_sha256"][0] = "0" * 64
            arbitrary_evaluator = types.SimpleNamespace(
                recompute_receipt_metrics=lambda path: arbitrary_png_replay
            )
            evaluator = types.SimpleNamespace(
                recompute_receipt_metrics=lambda path: replay
            )
            evidence = {
                "scene": "flame_salmon_1",
                "method": "official",
                "evaluator_relative_path": sequence.ORDINARY_EVALUATOR_RELATIVE_PATH,
                "evaluator_sha256": receipt["evaluator_sha256"],
            }
            point = {"point_id": "p0"}
            data_audit = {
                "manifest_sha256": receipt["source_data_manifest_sha256"],
                "manifest_path": str(source_manifest),
                "member_sha256": member_hashes,
            }
            with mock.patch.object(
                sequence,
                "_load_ordinary_evaluator_module",
                return_value=arbitrary_evaluator,
            ):
                with self.assertRaisesRegex(ValueError, "archive/render chain"):
                    sequence._validate_rate_quality_evaluator_receipt(
                        receipt_path=receipt_path,
                        receipt_sha256=hashlib.sha256(
                            receipt_path.read_bytes()
                        ).hexdigest(),
                        point=point,
                        evidence=evidence,
                        validation=validation,
                        data_audit=data_audit,
                        runtime_receipt=runtime,
                        evaluator_path=evaluator_path,
                    )
            with mock.patch.object(
                sequence, "_load_ordinary_evaluator_module", return_value=evaluator
            ):
                with self.assertRaisesRegex(ValueError, "frozen-code replay"):
                    sequence._validate_rate_quality_evaluator_receipt(
                        receipt_path=receipt_path,
                        receipt_sha256=hashlib.sha256(
                            receipt_path.read_bytes()
                        ).hexdigest(),
                        point=point,
                        evidence=evidence,
                        validation=validation,
                        data_audit=data_audit,
                        runtime_receipt=runtime,
                        evaluator_path=evaluator_path,
                    )

    def test_counted_training_receipt_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._gop(root, 0)
            forged = self._repack(
                valid,
                lambda members: members.__setitem__(
                    "training_receipt.json", members["training_receipt.json"] + b"\n"
                ),
            )
            with self.assertRaisesRegex(ValueError, "training receipt"):
                sequence.validate_gop_archive(
                    forged, "flame_salmon_1", "official", 0
                )

    def test_official_gop_requires_a_counted_training_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._gop(root, 0)
            forged = self._repack(
                valid, lambda members: members.pop("training_receipt.json")
            )
            with self.assertRaisesRegex(ValueError, "counted members"):
                sequence.validate_gop_archive(
                    forged, "flame_salmon_1", "official", 0
                )

    def test_official_gop_rejects_ap_training_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._gop(root, 0)
            forged = self._repack(
                valid,
                lambda members: members.__setitem__(
                    "ap_training_receipt.json", b"forged-ap-state"
                ),
            )
            with self.assertRaisesRegex(ValueError, "unexpectedly declares AP|exact decoder contract"):
                sequence.validate_gop_archive(
                    forged, "flame_salmon_1", "official", 0
                )

    def test_empty_source_checkpoint_closure_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._gop(root, 0)

            def empty_checkpoints(members):
                training = json.loads(members["training_receipt.json"])
                training["source_checkpoints"] = []
                training_payload = sequence.canonical_json_bytes(training)
                producer = json.loads(members["producer_receipt.json"])
                producer["source_checkpoints"] = []
                producer["training_receipt_sha256"] = hashlib.sha256(
                    training_payload
                ).hexdigest()
                producer_payload = sequence.canonical_json_bytes(producer)
                config = json.loads(members["decoder_config.json"])
                config["training_receipt_sha256"] = hashlib.sha256(
                    training_payload
                ).hexdigest()
                config["producer_receipt_sha256"] = hashlib.sha256(
                    producer_payload
                ).hexdigest()
                members["training_receipt.json"] = training_payload
                members["producer_receipt.json"] = producer_payload
                members["decoder_config.json"] = sequence.canonical_json_bytes(config)

            forged = self._repack(valid, empty_checkpoints)
            with self.assertRaisesRegex(ValueError, "checkpoint grid|source-checkpoint closure"):
                sequence.validate_gop_archive(
                    forged, "flame_salmon_1", "official", 0
                )

    def test_four_point_ids_reusing_one_archive_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "reuse one sequence archive"):
                self._recompute_curve(
                    Path(temporary), ["1" * 64] * 4, [1300, 1200, 1100, 1000]
                )

    def test_four_archives_at_one_rate_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "distinct rates"):
                self._recompute_curve(
                    Path(temporary), [f"{index + 1:x}" * 64 for index in range(4)], [1000] * 4
                )

    def test_four_distinct_archives_and_rates_are_recomputed(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self._recompute_curve(
                Path(temporary),
                [f"{index + 1:x}" * 64 for index in range(4)],
                [1300, 1200, 1100, 1000],
            )
            self.assertEqual(result["distinct_archive_count"], 4)
            self.assertEqual(result["distinct_rate_count"], 4)
            self.assertEqual(result["distinct_training_config_count"], 4)
            self.assertEqual(
                result["minimum_adjacent_actual_byte_fraction"], 0.05
            )
            self.assertTrue(result["metrics_recomputed_from_evaluator_receipts"])

    def test_eligibility_recomputation_integral_float_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self._recompute_curve(
                Path(temporary),
                [f"{index + 1:x}" * 64 for index in range(4)],
                [1300, 1200, 1100, 1000],
            )
            cases = (
                ("required_point_count",),
                ("distinct_archive_count",),
                ("distinct_rate_count",),
                ("distinct_training_config_count",),
                ("selected_archive_bytes",),
                ("selected_seed",),
                ("source_data", "file_count"),
            )
            for path in cases:
                with self.subTest(path=path):
                    forged = json.loads(json.dumps(result))
                    target = forged
                    for name in path[:-1]:
                        target = target[name]
                    target[path[-1]] = float(target[path[-1]])
                    with self.assertRaisesRegex(ValueError, "integer"):
                        sequence.validate_eligibility_recomputation_contract(
                            forged,
                            expected_scene=result["source_data"]["scene"],
                            expected_point_id=result["selected_point_id"],
                            expected_source_evidence_sha256=result[
                                "source_evidence_sha256"
                            ],
                            expected_archive_bytes=result["selected_archive_bytes"],
                            expected_training_config_sha256=result[
                                "selected_training_config_sha256"
                            ],
                            expected_seed=result["selected_seed"],
                        )

    def test_four_repackages_of_one_producer_config_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "reuse one producer training"):
                self._recompute_curve(
                    Path(temporary),
                    [f"{index + 1:x}" * 64 for index in range(4)],
                    [1300, 1200, 1100, 1000],
                    training_hashes=["5" * 64] * 4,
                )

    def test_adjacent_rate_points_below_five_percent_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "less than 5%"):
                self._recompute_curve(
                    Path(temporary),
                    [f"{index + 1:x}" * 64 for index in range(4)],
                    [1300, 1200, 1150, 1000],
                )

    def test_rate_index_lambda_mapping_is_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "rate/RD-lambda grid"):
                self._recompute_curve(
                    Path(temporary),
                    [f"{index + 1:x}" * 64 for index in range(4)],
                    [1300, 1200, 1100, 1000],
                    producer_lambdas=[0.0005, 0.001, 0.003, 0.004],
                )

    def test_self_reported_metric_aggregate_differing_from_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "not recomputed"):
                self._recompute_curve(
                    Path(temporary),
                    [f"{index + 1:x}" * 64 for index in range(4)],
                    [1300, 1200, 1100, 1000],
                    {"psnr": 31.0},
                )

    def test_self_reported_timing_differing_from_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "not recomputed"):
                self._recompute_curve(
                    Path(temporary),
                    [f"{index + 1:x}" * 64 for index in range(4)],
                    [1300, 1200, 1100, 1000],
                    {"encode_seconds": 5.5},
                )

    def test_self_reported_eligibility_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gops = [self._gop(root, index) for index in range(5)]
            archive = root / "sequence.zip"
            built = sequence.build_sequence_container(
                scene="flame_salmon_1",
                method="official",
                gop_archives=gops,
                output=archive,
                training_config_sha256=TRAINING_CONFIG_SHA,
                seed=20260715,
            )
            receipt = root / "forged.json"
            receipt.write_bytes(
                sequence.canonical_json_bytes(
                    {
                        "schema": "h007.h_sota_operating_point_eligibility.v1",
                        "scene": "flame_salmon_1",
                        "method": "official",
                        "point_id": "p0",
                        "eligible": True,
                        "ordinary_rate_quality_only": True,
                        "source_sha256": "f" * 64,
                    }
                )
            )
            row = {
                "scene": "flame_salmon_1",
                "method": "official",
                "point_id": "p0",
                "archive": archive.name,
                "eligibility_receipt": receipt.name,
                "training_config_sha256": TRAINING_CONFIG_SHA,
                "seed": 20260715,
            }
            with self.assertRaisesRegex(ValueError, "fields|schema"):
                sequence._eligible_registry_row(row, root)

    def test_scene_specific_camera_census_is_exact(self):
        self.assertEqual(
            sequence.FROZEN_SCENE_CAMERA_COUNTS,
            {
                "flame_salmon_1": 19,
                "coffee_martini": 18,
                "cook_spinach": 21,
                "cut_roasted_beef": 20,
                "flame_steak": 21,
                "sear_steak": 21,
            },
        )
        self.assertEqual(
            sequence.frozen_camera_names("coffee_martini", ""),
            tuple(f"cam{index:02d}" for index in range(18)),
        )
        self.assertEqual(
            sequence.frozen_camera_names("cook_spinach"),
            tuple(f"cam{index:02d}.png" for index in range(21)),
        )
        with self.assertRaisesRegex(ValueError, "outside the frozen Neu3D"):
            sequence.frozen_scene_camera_count("unknown_scene")

    def test_source_data_manifest_uses_scene_specific_camera_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            member = root / "member.png"
            member.write_bytes(b"x")
            member_sha = hashlib.sha256(b"x").hexdigest()
            files = [
                {
                    "path": f"png/cam{camera:02d}/{frame:05d}.png",
                    "bytes": 1,
                    "sha256": member_sha,
                }
                for camera in range(18)
                for frame in range(1, 301)
            ]
            manifest = root / "source_data.json"
            manifest.write_bytes(
                sequence.canonical_json_bytes(
                    {
                        "schema": sequence.SOURCE_DATA_SCHEMA,
                        "scene": "coffee_martini",
                        "frame_count": 300,
                        "cameras": [f"cam{index:02d}" for index in range(18)],
                        "files": files,
                        "outcome_fields_read": [],
                    }
                )
            )
            with mock.patch.object(
                sequence, "_bound_relative", return_value=member
            ):
                result = sequence._validate_source_data_manifest(
                    manifest, "coffee_martini"
                )
            self.assertEqual(result["file_count"], 18 * 300)
            wrong = json.loads(manifest.read_text(encoding="ascii"))
            wrong["cameras"].append("cam18")
            manifest.write_bytes(sequence.canonical_json_bytes(wrong))
            with self.assertRaisesRegex(ValueError, "identity/protocol"):
                sequence._validate_source_data_manifest(manifest, "coffee_martini")

            wrong = json.loads(manifest.read_text(encoding="ascii"))
            wrong["cameras"] = [f"cam{index:02d}" for index in range(18)]
            wrong["files"][0]["path"] = "cam00/00001.png"
            manifest.write_bytes(sequence.canonical_json_bytes(wrong))
            with mock.patch.object(
                sequence, "_bound_relative", return_value=member
            ):
                with self.assertRaisesRegex(ValueError, "exact scene-specific"):
                    sequence._validate_source_data_manifest(
                        manifest, "coffee_martini"
                    )

            reordered = {
                "schema": sequence.SOURCE_DATA_SCHEMA,
                "scene": "coffee_martini",
                "frame_count": 300,
                "cameras": [f"cam{index:02d}" for index in range(18)],
                "files": list(files),
                "outcome_fields_read": [],
            }
            reordered["files"][0], reordered["files"][1] = (
                reordered["files"][1], reordered["files"][0]
            )
            manifest.write_bytes(sequence.canonical_json_bytes(reordered))
            with mock.patch.object(
                sequence, "_bound_relative", return_value=member
            ):
                with self.assertRaisesRegex(ValueError, "exact scene-specific"):
                    sequence._validate_source_data_manifest(
                        manifest, "coffee_martini"
                    )

            linked = root / "linked.png"
            os.link(str(member), str(linked))
            honest = dict(reordered)
            honest["files"] = files
            manifest.write_bytes(sequence.canonical_json_bytes(honest))
            with mock.patch.object(
                sequence, "_bound_relative", return_value=linked
            ):
                with self.assertRaisesRegex(ValueError, "regular and single-link"):
                    sequence._validate_source_data_manifest(
                        manifest, "coffee_martini"
                    )


if __name__ == "__main__":
    unittest.main()
