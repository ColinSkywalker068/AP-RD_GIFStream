#!/usr/bin/env python3
"""Archive-only clean-process decoder for the H007 GIFStream container.

The decoder reads no source images, checkpoints, score files, or downstream
outcomes outside the counted ZIP.  It reconstructs the entropy models from the
counted configuration and ``nets.pt``, restores exact canonical identities from
the counted anchor-relative correction stream, validates camera sidecars, and
writes a tensor-hash manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from gsplat.compression import GIFStreamEnd2endCompression
from gsplat.compression.ap_gifstream import (
    canonical_json_bytes,
    tensor_mapping_sha256,
    variant_spec,
)
from gsplat.compression.gifstream_end2end_compression import (
    _validate_counted_retained_identity,
)
from gsplat.compression.h007_clean_runtime import (
    counted_camera_render_benchmark,
    instantiate_counted_models,
)
from gsplat.compression.h007_runtime_provenance import verify_runtime_provenance
from gsplat.compression.h007_sequence_container import (
    expected_gifstream_nets_keys,
    frozen_camera_names,
    validate_sequence_container,
)
from gsplat.compression_simulation.simulation import GIFStreamCompressionSimulation


OFFICIAL_COMMIT = "c98486632e7dafd830740b1a1692bd08c48b96e3"
PATCH1_SHA256 = "65ecd8f1fe30fac5d5bce3aed295cfe04f0b0582f106efaf37489b005d2e431b"
ENTROPY_KEYS = (
    "anchors",
    "quats",
    "scales",
    "opacities",
    "anchor_features",
    "offsets",
    "factors",
    "time_features",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("clean-decode output directory must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive, "r") as handle:
        names = []
        for info in handle.infolist():
            name = info.filename
            if name in names:
                raise ValueError(f"duplicate ZIP member: {name}")
            names.append(name)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"symlink ZIP member forbidden: {name}")
            target = (destination / name).resolve()
            if os.path.commonpath([str(root), str(target)]) != str(root):
                raise ValueError(f"unsafe ZIP member: {name}")
        handle.extractall(destination)


def _validate_census(root: Path) -> None:
    census_path = root / "byte_census.json"
    census = json.loads(census_path.read_text(encoding="utf-8"))
    if census.get("schema") != "h007.container_byte_census.v1":
        raise ValueError("unsupported byte-census schema")
    declared = {str(row["path"]): row for row in census["files"]}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != census_path
    }
    if set(declared) != actual:
        raise ValueError("container members differ from counted byte census")
    for name, row in declared.items():
        path = root / name
        payload = path.read_bytes()
        if len(payload) != int(row["bytes"]):
            raise ValueError(f"byte-census size mismatch: {name}")
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError(f"byte-census hash mismatch: {name}")


def _validate_camera_metadata(
    root: Path,
    expected_camera_names: Tuple[str, ...],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    camera_dir = root / "camera_metadata"
    arrays = {
        name: np.load(camera_dir / f"{name}.npy", allow_pickle=False)
        for name in (
            "camera_keys",
            "intrinsics",
            "image_sizes",
            "camtoworlds",
            "camera_ids",
            "camera_names",
            "transform",
            "bounds",
        )
    }
    n_cameras = int(arrays["camera_keys"].shape[0])
    n_poses = int(arrays["camtoworlds"].shape[0])
    expected_camera_count = len(expected_camera_names)
    if n_cameras != expected_camera_count or n_poses != expected_camera_count:
        raise ValueError("counted camera count differs from the frozen camera grid")
    if arrays["camera_keys"].dtype.str != "<i8" or arrays["camera_keys"].shape != (
        n_cameras,
    ):
        raise ValueError("counted camera keys are malformed")
    if arrays["intrinsics"].dtype.str != "<f8" or arrays["intrinsics"].shape != (
        n_cameras,
        3,
        3,
    ):
        raise ValueError("counted camera intrinsics are malformed")
    if arrays["image_sizes"].dtype.str != "<i8" or arrays["image_sizes"].shape != (
        n_cameras,
        2,
    ):
        raise ValueError("counted camera image sizes are malformed")
    if arrays["camtoworlds"].dtype.str != "<f8" or arrays["camtoworlds"].shape != (
        n_poses,
        4,
        4,
    ):
        raise ValueError("counted camera poses are malformed")
    if arrays["camera_ids"].dtype.str != "<i8" or arrays["camera_ids"].shape != (
        n_poses,
    ):
        raise ValueError("counted camera IDs/poses disagree")
    if arrays["camera_names"].dtype.kind != "U" or arrays["camera_names"].shape != (
        n_poses,
    ):
        raise ValueError("counted camera names/poses disagree")
    name_width = arrays["camera_names"].dtype.itemsize // 4
    actual_name_width = max((len(str(value)) for value in arrays["camera_names"]), default=0)
    if name_width <= 0 or name_width > 4096 or name_width != actual_name_width:
        raise ValueError("counted camera-name width is noncanonical")
    if tuple(str(value) for value in arrays["camera_names"].tolist()) != expected_camera_names:
        raise ValueError("counted camera names differ from the frozen camera grid")
    if arrays["transform"].dtype.str != "<f8" or arrays["transform"].shape != (4, 4):
        raise ValueError("counted camera transform is malformed")
    if arrays["bounds"].dtype.str != "<f8" or arrays["bounds"].shape != (
        n_poses,
        2,
    ):
        raise ValueError("counted camera bounds are malformed")
    if np.unique(arrays["camera_keys"]).shape[0] != n_cameras:
        raise ValueError("counted camera keys are duplicated")
    if not set(int(value) for value in arrays["camera_ids"].tolist()).issubset(
        set(int(value) for value in arrays["camera_keys"].tolist())
    ):
        raise ValueError("counted pose camera IDs are absent from intrinsic keys")
    if not all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind in "f"):
        raise ValueError("nonfinite counted camera metadata")
    return arrays, {
        "camera_key_count": n_cameras,
        "pose_count": n_poses,
        "camera_name_width": name_width,
        "source": "counted_archive_only",
    }


def _tensor_manifest(splats: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for name, value in sorted(splats.items()):
        tensor = value.detach().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"decoded tensor is nonfinite: {name}")
        payload = tensor.numpy().tobytes(order="C")
        output[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return output


def clean_decode(
    archive: Path,
    sequence_archive: Path,
    gop_id: int,
    output_dir: Path,
    device: str,
    provenance_manifest: Optional[Path],
    provenance_manifest_sha256: Optional[str],
    warmup_renders: Optional[int] = None,
    timed_renders: Optional[int] = None,
) -> Dict[str, Any]:
    archive = archive.resolve()
    if not archive.is_file():
        raise ValueError(f"archive does not exist: {archive}")
    if gop_id not in range(5):
        raise ValueError("clean-decode GOP ID must be in 0..4")
    sequence_archive = sequence_archive.resolve()
    sequence_validation = validate_sequence_container(sequence_archive)
    gop_audit = sequence_validation["gops"][gop_id]
    if (
        _sha256(archive) != gop_audit["sha256"]
        or archive.stat().st_size != int(gop_audit["bytes"])
    ):
        raise ValueError("clean-decode inner GOP is not the selected nested sequence member")
    container = output_dir / "container"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("clean-decode output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    _safe_extract(archive, container)
    _validate_census(container)

    config = json.loads((container / "decoder_config.json").read_text(encoding="utf-8"))
    if config.get("schema") != "h007.gifstream_decoder_config.v3":
        raise ValueError("unsupported decoder-config schema")
    if config.get("codec_family") != "GIFStream":
        raise ValueError("counted decoder config is not GIFStream")
    if config.get("official_commit") != OFFICIAL_COMMIT:
        raise ValueError("container official commit mismatch")
    if provenance_manifest is None or provenance_manifest_sha256 is None:
        raise ValueError("all official/AP clean decodes require registered runtime provenance")
    repo_root = Path(__file__).resolve().parents[1]
    runtime_provenance = verify_runtime_provenance(
        provenance_manifest,
        repo_root,
        provenance_manifest_sha256,
    )
    if len(runtime_provenance.get("patch_sha256", [])) != 9:
        raise ValueError("clean decoder is not bound to the registered nine-stage patch chain")
    counted_manifest = container / "preregistered_patch_chain_manifest.json"
    external_payload = provenance_manifest.resolve().read_bytes()
    if counted_manifest.read_bytes() != external_payload:
        raise ValueError("counted/external preregistration manifests differ")
    counted_receipt = json.loads(
        (container / "runtime_provenance.json").read_text(encoding="utf-8")
    )
    if counted_receipt != runtime_provenance:
        raise ValueError("counted runtime-provenance receipt mismatch")
    if config.get("patch_chain_sha256") != runtime_provenance["patch_sha256"]:
        raise ValueError("decoder config patch chain differs from preregistration")
    if config.get("runtime_manifest_sha256") != runtime_provenance[
        "manifest_sha256"
    ] or config.get("normalized_code_tree_sha256") != runtime_provenance[
        "normalized_code_tree"
    ]["sha256"]:
        raise ValueError("decoder config runtime/code-tree provenance mismatch")
    if (
        config.get("scene") != sequence_validation["scene"]
        or str(config.get("variant")) != sequence_validation["method"]
        or int(config.get("start_frame", -1)) != int(gop_audit["start_frame"])
    ):
        raise ValueError("clean-decode GOP config differs from the selected sequence binding")
    if config.get("clean_decode_entrypoint") != "examples/h007_clean_decode_gifstream.py":
        raise ValueError("container clean-decode entrypoint mismatch")
    if int(config["GOP_size"]) != 60:
        raise ValueError("H007 official/AP container must contain a 60-frame GOP")
    producer_payload = (container / "producer_receipt.json").read_bytes()
    producer = json.loads(producer_payload.decode("utf-8"))
    if (
        producer.get("schema") != "h007.gifstream_producer_receipt.v1"
        or producer.get("scene") != config.get("scene")
        or producer.get("variant") != config.get("variant")
        or producer.get("runtime_provenance") != runtime_provenance
        or hashlib.sha256(producer_payload).hexdigest()
        != config.get("producer_receipt_sha256")
    ):
        raise ValueError("counted producer receipt differs from decoder/runtime provenance")
    training_payload = (container / "training_receipt.json").read_bytes()
    if (
        producer.get("training_receipt_sha256")
        != hashlib.sha256(training_payload).hexdigest()
        or config.get("training_receipt_sha256")
        != hashlib.sha256(training_payload).hexdigest()
    ):
        raise ValueError("counted frozen training receipt binding mismatch")
    camera_arrays, camera_audit = _validate_camera_metadata(
        container, frozen_camera_names(str(config["scene"]))
    )
    edit_audit_path = container / "ap_edit_hook.json"
    if edit_audit_path.is_file():
        edit_audit = json.loads(edit_audit_path.read_text(encoding="utf-8"))
        edit_artifact = container / str(edit_audit["counted_artifact"])
        if _sha256(edit_artifact) != edit_audit["counted_artifact_sha256"]:
            raise ValueError("counted edit artifact differs from edit-hook receipt")
        edit_score = container / str(edit_audit["counted_source_score"])
        edit_reference = container / str(edit_audit["counted_reference_manifest"])
        if _sha256(edit_score) != edit_audit["counted_source_score_sha256"] or _sha256(
            edit_score
        ) != edit_audit["source_score_sha256"]:
            raise ValueError("counted edit source score differs from edit-hook receipt")
        if _sha256(edit_reference) != edit_audit[
            "counted_reference_manifest_sha256"
        ] or _sha256(edit_reference) != edit_audit["reference_manifest_sha256"]:
            raise ValueError("counted edit reference manifest differs from edit-hook receipt")
        reference = json.loads(edit_reference.read_text(encoding="utf-8"))
        if reference.get("source_score_sha256") != edit_audit[
            "source_score_sha256"
        ] or reference.get("selection") != edit_audit["selection"]:
            raise ValueError("edit reference manifest score/selection mismatch")
        with np.load(edit_artifact, allow_pickle=False) as edit:
            if str(np.asarray(edit["source_score_sha256"]).item()) != edit_audit[
                "source_score_sha256"
            ] or str(np.asarray(edit["selection"]).item()) != edit_audit["selection"]:
                raise ValueError("counted edit artifact score/selection mismatch")
            if str(np.asarray(edit["reference_manifest_sha256"]).item()) != edit_audit[
                "reference_manifest_sha256"
            ]:
                raise ValueError("counted edit artifact reference-manifest mismatch")
    receipt = None
    if config.get("variant") != "official":
        receipt_path = container / "ap_training_receipt.json"
        if not receipt_path.is_file():
            raise ValueError("AP container lacks its counted training receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema") != "h007.ap_training_receipt.v2" or receipt.get(
            "variant"
        ) != config["variant"]:
            raise ValueError("counted AP training receipt is incompatible")
        if receipt.get("scene") != config.get("scene"):
            raise ValueError("counted AP training receipt scene mismatch")
        spec = variant_spec(str(config["variant"]))
        if bool(receipt.get("path_loss_required")) != bool(spec.action_loss):
            raise ValueError("counted AP receipt action-loss flag mismatch")
        if not bool(receipt.get("time_entropy_model_frozen_after_freeze")):
            raise ValueError("counted AP receipt lacks entropy-estimator freeze evidence")
        if receipt.get("factor_membership_columns_frozen") != [0, 3]:
            raise ValueError("counted AP receipt lacks factor-membership freeze evidence")
        if (
            receipt.get("path_contract_schema")
            != "h007.ap_gifstream.path_contract.v1"
            or receipt.get("path_dependency_rule")
            != "protected-plus-one-hop-retained-knn"
        ):
            raise ValueError("counted AP receipt lacks the Patch8 path contract")
        if receipt.get("runtime_provenance") != runtime_provenance:
            raise ValueError("counted AP receipt runtime provenance mismatch")
        if spec.action_loss and (
            int(receipt.get("path_loss_applications", 0)) <= 0
            or not receipt.get("path_loss_steps")
        ):
            raise ValueError("full AP container lacks action-loss evidence")
        codec_meta = json.loads((container / "meta.json").read_text(encoding="utf-8"))
        if (
            receipt.get("score_sha256")
            != codec_meta["__ap__"]["score"]["score_artifact_sha256"]
            or receipt.get("path_contract_schema")
            != codec_meta["__ap__"]["path_contract"]["schema"]
            or int(receipt.get("path_knn_count", -1))
            != int(codec_meta["__ap__"]["path_contract"]["knn_count"])
            or receipt.get("path_knn_graph_sha256")
            != codec_meta["__ap__"]["path_contract"][
                "retained_knn_graph_sha256"
            ]
        ):
            raise ValueError("AP receipt and counted score artifact disagree")

    target_device = torch.device(device)
    nets = torch.load(container / "nets.pt", map_location=target_device, weights_only=True)
    if not isinstance(nets, dict) or set(nets) != expected_gifstream_nets_keys(
        bool(config.get("app_opt"))
    ):
        raise ValueError("counted nets.pt top-level model dictionary is not exact")
    producer_state = producer["model_state_sha256"]
    counted_entropy_hashes = {
        key[: -len("_entropy_model")]: tensor_mapping_sha256(value)
        for key, value in nets.items()
        if key.endswith("_entropy_model") and isinstance(value, dict)
    }
    counted_appearance_hash = (
        tensor_mapping_sha256(nets["app_module"])
        if "app_module" in nets
        else None
    )
    if (
        tensor_mapping_sha256(nets["decoders"]) != producer_state["decoders"]
        or counted_entropy_hashes != producer_state["entropy_models"]
        or hashlib.sha256(canonical_json_bytes(nets["scaling"])).hexdigest()
        != producer_state["codec_scaling"]
        or counted_appearance_hash != producer_state["appearance_module"]
        or ("app_module" in nets) != bool(config.get("app_opt"))
    ):
        raise ValueError("counted decoder/scaling/appearance state differs from producer receipt")
    decoders, app_module, model_audit = instantiate_counted_models(
        nets, config, target_device
    )
    entropy_steps = {name: 1 for name in ENTROPY_KEYS}
    simulation = GIFStreamCompressionSimulation(
        entropy_model_enable=True,
        entropy_model_type="conditional_gaussian_model",
        entropy_steps=entropy_steps,
        device=target_device,
        feature_dim=int(config["anchor_feature_dim"]),
        n_offsets=int(config["n_offsets"]),
        c_channel=int(config["entropy_channel"]),
        p_channel=int(config["c_perframe"]),
        scaling=nets["scaling"],
    )
    for name, model in simulation.entropy_models.items():
        if model is None:
            continue
        key = f"{name}_entropy_model"
        if key not in nets:
            raise ValueError(f"counted nets.pt lacks {key}")
        if not isinstance(nets[key], dict) or any(
            not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
            for value in nets[key].values()
        ):
            raise ValueError(f"counted entropy state is malformed/nonfinite: {key}")
        model.load_state_dict(nets[key], strict=True)
        model.eval()
    if receipt is not None:
        current_time_hash = tensor_mapping_sha256(
            simulation.entropy_models["time_features"].state_dict()
        )
        score = codec_meta["__ap__"]["score"]
        frozen_time_hash = str(score["time_entropy_model_sha256"])
        frozen_scaling = float(score["time_feature_scaling"])
        estimator_version = str(score["estimator_version"])
        score_runtime_manifest = str(score["runtime_manifest_sha256"])
        score_code_tree = str(score["normalized_code_tree_sha256"])
        score_patch_chain = [str(value) for value in score["patch_chain_sha256"]]
        if current_time_hash != frozen_time_hash or receipt.get(
            "time_entropy_model_sha256"
        ) != frozen_time_hash:
            raise ValueError("clean decoder entropy model differs from byte estimator")
        if frozen_scaling != float(nets["scaling"]["time_features"]):
            raise ValueError("clean decoder temporal scaling differs from byte estimator")
        if estimator_version != "h007.conditional_gaussian_per_row_bits.v1":
            raise ValueError("clean decoder byte estimator version mismatch")
        if (
            score_runtime_manifest != runtime_provenance["manifest_sha256"]
            or score_code_tree != runtime_provenance["normalized_code_tree"]["sha256"]
            or score_patch_chain != runtime_provenance["patch_sha256"]
        ):
            raise ValueError("clean decoder score/runtime provenance mismatch")

    codec = GIFStreamEnd2endCompression(
        ap_config={
            "provenance_manifest_path": str(provenance_manifest.resolve())
            if provenance_manifest is not None
            else None,
            "provenance_manifest_sha256": provenance_manifest_sha256,
            "repo_root": str(repo_root),
        }
    )
    splats = codec.decompress(container.as_posix(), simulation.entropy_models, target_device)
    expected = {
        "anchors",
        "scales",
        "quats",
        "opacities",
        "offsets",
        "anchor_features",
        "time_features",
        "factors",
    }
    if set(splats) != expected:
        raise ValueError("decoded splat fields are incomplete or unexpected")
    render_device = torch.empty(0, device=target_device).device
    if any(not isinstance(value, torch.Tensor) for value in splats.values()):
        raise ValueError("decoded splat fields must all be tensors")
    splats = {
        name: value.to(device=render_device) for name, value in splats.items()
    }
    if any(value.device != render_device for value in splats.values()):
        raise ValueError("decoded splat fields are not co-located on the render device")
    if splats["time_features"].shape[:2] != (
        splats["anchors"].shape[0],
        int(config["GOP_size"]),
    ):
        raise ValueError("decoded temporal tensor shape disagrees with configuration")
    decoded_ids_path = None
    decoded_ids_sha256 = None
    if receipt is not None:
        if codec.decoded_canonical_ids is None:
            raise ValueError("AP decoder did not restore counted canonical identities")
        decoded_ids = codec.decoded_canonical_ids.detach().cpu()
        _validate_counted_retained_identity(
            decoded_ids, splats["anchors"].shape[0]
        )
        decoded_ids_path = output_dir / "decoded_canonical_ids.npy"
        np.save(
            decoded_ids_path,
            decoded_ids.numpy().astype(np.int64, copy=False),
            allow_pickle=False,
        )
        decoded_ids_sha256 = _sha256(decoded_ids_path)

    clean_render = counted_camera_render_benchmark(
        splats,
        decoders,
        app_module,
        config,
        camera_arrays,
        int(config["warmup_renders"] if warmup_renders is None else warmup_renders),
        int(config["timed_renders"] if timed_renders is None else timed_renders),
    )

    decoded_path = output_dir / "decoded_splats.pt"
    torch.save({name: value.detach().cpu() for name, value in splats.items()}, decoded_path)
    manifest = {
        "schema": "h007.clean_decode_result.v2",
        "archive": archive.name,
        "archive_sha256": _sha256(archive),
        "source_sequence_archive": str(sequence_archive),
        "source_sequence_archive_sha256": sequence_validation["archive_sha256"],
        "source_sequence_manifest_sha256": sequence_validation[
            "sequence_manifest_sha256"
        ],
        "source_gop_id": int(gop_id),
        "source_inner_gop_sha256": gop_audit["sha256"],
        "source_inner_gop_decoder_config_sha256": gop_audit[
            "decoder_config_sha256"
        ],
        "official_commit": OFFICIAL_COMMIT,
        "variant": config["variant"],
        "device": str(target_device),
        "source_images_read": 0,
        "external_checkpoint_files_read": 0,
        "outcome_fields_read": [],
        "decoded_splats": decoded_path.name,
        "decoded_splats_sha256": _sha256(decoded_path),
        "decoded_canonical_ids": (
            decoded_ids_path.name if decoded_ids_path is not None else None
        ),
        "decoded_canonical_ids_sha256": decoded_ids_sha256,
        "camera_audit": camera_audit,
        "counted_model_audit": model_audit,
        "counted_camera_render": clean_render,
        "warm_render_fps": clean_render["fps"],
        "runtime_provenance_validated": True,
        "runtime_provenance": runtime_provenance,
        "producer_receipt_validated": True,
        "producer_receipt_sha256": hashlib.sha256(producer_payload).hexdigest(),
        "training_receipt_sha256": hashlib.sha256(training_payload).hexdigest(),
        "payload_manifest_sha256": config["payload_manifest_sha256"],
        "ap_training_receipt_validated": receipt is not None,
        "edit_hook_validated": edit_audit_path.is_file(),
        "tensors": _tensor_manifest(splats),
    }
    (output_dir / "clean_decode_manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sequence-archive", type=Path, required=True)
    parser.add_argument("--gop-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--provenance-manifest", type=Path)
    parser.add_argument("--provenance-manifest-sha256")
    parser.add_argument("--warmup-renders", type=int)
    parser.add_argument("--timed-renders", type=int)
    args = parser.parse_args()
    manifest = clean_decode(
        args.archive,
        args.sequence_archive,
        args.gop_id,
        args.output_dir,
        args.device,
        args.provenance_manifest,
        args.provenance_manifest_sha256,
        args.warmup_renders,
        args.timed_renders,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
