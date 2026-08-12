#!/usr/bin/env python3
"""Executable final-codec H-DOWN reference construction and evaluation.

``reference`` reads only raw cam00 RGB plus an uncompressed reference bundle,
runs the frozen local CoTracker3-offline checkpoint, lifts the selected click
through the real gsplat alpha compositor, freezes the 90% parent prefix and
``w_i A_i``, and renders the reference edit masks.  ``evaluate`` reads only
that frozen artifact plus one clean-decoded GOP bundle and produces penalized
edit-mask loss with explicit missing-identity accounting.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from gsplat.compression.h007_clean_runtime import (
    alpha_parent_contributions,
    counted_knn_indices,
    decode_anchor_paths,
    instantiate_counted_models,
    render_hdown_frame,
)
from gsplat.compression.h007_certification import (
    CONFIRMATORY_SCENES as FROZEN_CONFIRMATORY_SCENES,
    FREEZE_NAME,
    canonical_json_bytes as certification_json_bytes,
    exclusive_write,
    freeze_stage02,
    read_regular_bytes,
    validate_case_static_closure,
    validate_stage02_freeze,
)
from gsplat.compression.h007_sequence_container import validate_sequence_container


REFERENCE_SCHEMA = "h007.hdown_final_reference_case.v3"
EVALUATION_SCHEMA = "h007.hdown_final_candidate_case.v3"
AGGREGATE_SCHEMA = "h007.hdown_final_verdict.v2"
GRID_COORDS = (0.25, 0.375, 0.50, 0.625, 0.75)
MASK_SCALES = (0.05, 0.10, 0.15)
CONFIRMATORY_SCENES = set(FROZEN_CONFIRMATORY_SCENES)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    buffer_out = io.BytesIO()
    with zipfile.ZipFile(buffer_out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for name in sorted(arrays):
            if not name or "/" in name or "\\" in name:
                raise ValueError("invalid deterministic NPZ member name")
            value = np.asarray(arrays[name])
            if value.dtype == np.dtype("O"):
                raise ValueError("object arrays are forbidden in H-DOWN artifacts")
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, value, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            handle.writestr(
                info,
                buffer.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    exclusive_write(Path(os.path.abspath(os.fspath(path))), buffer_out.getvalue())


def _reject_duplicate_object_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def strict_canonical_json_bytes(payload: bytes, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON token: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    try:
        canonical = canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} cannot be canonically serialized") from error
    if payload != canonical:
        raise ValueError(f"{label} bytes are not canonical JSON")
    return value


def read_strict_canonical_json(path: Path, label: str) -> Dict[str, Any]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    return strict_canonical_json_bytes(read_regular_bytes(absolute), label)


def exact_string(value: Any, label: str) -> str:
    if type(value) is not str or not value or len(value) > 4096:
        raise ValueError(f"{label} is not an exact bounded JSON string")
    return value


def exact_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} is not an exact nonnegative JSON integer")
    return value


def exact_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} is not an exact positive JSON integer")
    return value


def exact_finite_float(value: Any, label: str, *, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{label} is not an exact finite JSON float")
    return value


def require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        ch not in "0123456789abcdef" for ch in value
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def validate_directory_census(root: Path) -> None:
    if root.is_symlink():
        raise ValueError("bundle root symlink is forbidden")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"bundle symlink is forbidden: {path}")
    census_path = root / "byte_census.json"
    census = read_strict_canonical_json(census_path, "bundle byte census")
    if census.get("schema") != "h007.container_byte_census.v1":
        raise ValueError("bundle byte-census schema is unsupported")
    rows = census.get("files")
    if not isinstance(rows, list):
        raise ValueError("bundle byte census lacks file rows")
    declared = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("bundle byte-census row fields are unexpected")
        name = exact_string(row["path"], "bundle byte-census path")
        relative = Path(name)
        if (
            name in declared
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
        ):
            raise ValueError("bundle byte-census path is unsafe or duplicated")
        exact_nonnegative_int(row["bytes"], f"bundle member {name} bytes")
        require_sha256(row["sha256"], f"bundle member {name} SHA-256")
        declared[name] = row
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != census_path
    }
    if set(declared) != actual:
        raise ValueError("bundle members differ from byte census")
    if "file_count" in census and exact_nonnegative_int(
        census["file_count"], "bundle byte-census file count"
    ) != len(declared):
        raise ValueError("bundle byte-census file count mismatch")
    if "raw_bytes" in census and exact_nonnegative_int(
        census["raw_bytes"], "bundle byte-census raw bytes"
    ) != sum(row["bytes"] for row in declared.values()):
        raise ValueError("bundle byte-census raw-byte total mismatch")
    for name, row in declared.items():
        payload = read_regular_bytes(Path(os.path.abspath(os.fspath(root / name))))
        if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError(f"bundle byte census mismatch: {name}")


def validate_sha256_manifest(path: Path, expected_sha256: str) -> Dict[str, str]:
    """Compatibility wrapper returning the full validated Stage-02 inventory."""

    return validate_stage02_freeze(path, expected_sha256)["files"]


def directory_tree_hash(root: Path) -> Dict[str, Any]:
    if root.is_symlink():
        raise ValueError("CoTracker repository symlink is forbidden")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("CoTracker repository is unavailable")
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet"], check=False
    ).returncode != 0 or subprocess.run(
        ["git", "-C", str(root), "diff", "--cached", "--quiet"], check=False
    ).returncode != 0:
        raise ValueError("CoTracker repository has modified tracked files")
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    rows = []
    for encoded in sorted(value for value in tracked if value):
        relative = encoded.decode("utf-8")
        path = root / relative
        if path.is_symlink():
            raise ValueError(f"CoTracker tracked symlink is forbidden: {path}")
        if not path.is_file():
            raise ValueError(f"CoTracker tracked file is unavailable: {relative}")
        rows.append((relative, path.read_bytes()))
    if not rows:
        raise ValueError("CoTracker repository has no tracked files")
    digest = hashlib.sha256()
    for name, payload in rows:
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return {
        "git_commit": head,
        "normalization": "git-ls-files+sorted-posix-path+raw-bytes+uint64le-lengths",
        "file_count": len(rows),
        "sha256": digest.hexdigest(),
    }


def _load_camera(root: Path, config: Mapping[str, Any], device: torch.device):
    camera_dir = root / "camera_metadata"
    keys = np.load(camera_dir / "camera_keys.npy", allow_pickle=False)
    intrinsics = np.load(camera_dir / "intrinsics.npy", allow_pickle=False)
    sizes = np.load(camera_dir / "image_sizes.npy", allow_pickle=False)
    poses = np.load(camera_dir / "camtoworlds.npy", allow_pickle=False)
    camera_ids = np.load(camera_dir / "camera_ids.npy", allow_pickle=False)
    pose_index = exact_nonnegative_int(
        config.get("warm_camera_pose_index"), "decoder warm camera pose index"
    )
    if pose_index < 0 or pose_index >= poses.shape[0]:
        raise ValueError("cam00 pose index is outside counted camera metadata")
    camera_key = int(camera_ids[pose_index])
    locations = np.flatnonzero(keys == camera_key)
    if locations.size != 1:
        raise ValueError("cam00 camera key is absent or duplicated")
    meta_index = int(locations[0])
    width, height = [int(value) for value in sizes[meta_index]]
    if width <= 0 or height <= 0:
        raise ValueError("cam00 render dimensions are invalid")
    pose = torch.from_numpy(np.asarray(poses[pose_index], dtype=np.float32)).to(device)
    if pose.shape == (3, 4):
        pose = torch.cat([pose, pose.new_tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
    intrinsic = torch.from_numpy(
        np.asarray(intrinsics[meta_index], dtype=np.float32)
    ).to(device)
    if pose.shape != (4, 4) or intrinsic.shape != (3, 3):
        raise ValueError("cam00 pose/intrinsic shapes are invalid")
    if not torch.isfinite(pose).all() or not torch.isfinite(intrinsic).all():
        raise ValueError("cam00 pose/intrinsics are nonfinite")
    return pose[None], intrinsic[None], width, height, camera_key


def load_bundle(bundle: Path, device: torch.device, reference: bool):
    bundle = bundle.resolve()
    clean_manifest = None
    if reference:
        root = bundle
        splat_path = root / "reference_splats.pt"
    else:
        root = bundle / "container"
        splat_path = bundle / "decoded_splats.pt"
    validate_directory_census(root)
    if reference:
        reference_manifest = read_strict_canonical_json(
            root / "reference_bundle_manifest.json", "reference bundle manifest"
        )
        if reference_manifest.get("schema") != "h007.hdown_reference_bundle.v1":
            raise ValueError("reference bundle manifest schema is unsupported")
    else:
        clean_manifest_path = bundle / "clean_decode_manifest.json"
        clean_manifest = read_strict_canonical_json(
            clean_manifest_path, "clean-decode manifest"
        )
        if clean_manifest.get("schema") != "h007.clean_decode_result.v2":
            raise ValueError("candidate bundle lacks a clean-decode receipt")
        if require_sha256(
            clean_manifest.get("decoded_splats_sha256"),
            "clean-decoded splats SHA-256",
        ) != sha256_file(splat_path):
            raise ValueError("candidate decoded splats differ from clean-decode receipt")
        if exact_nonnegative_int(
            clean_manifest.get("source_images_read"), "clean-decode source-image count"
        ) != 0:
            raise ValueError("candidate clean decode read source images")
        clean_manifest["manifest_path"] = str(clean_manifest_path)
    config = read_strict_canonical_json(
        root / "decoder_config.json", "H-DOWN decoder config"
    )
    nets = torch.load(root / "nets.pt", map_location=device, weights_only=True)
    decoders, app_module, model_audit = instantiate_counted_models(
        nets, config, device, reference_nets=reference
    )
    splats = torch.load(splat_path, map_location=device, weights_only=True)
    if not isinstance(splats, dict):
        raise ValueError("H-DOWN bundle splats are not a tensor mapping")
    splats = {name: value.to(device) for name, value in splats.items()}
    camera = _load_camera(root, config, device)
    return root, splats, decoders, app_module, config, camera, model_audit, clean_manifest


def load_raw_gop(raw_dir: Path, start_frame: int) -> Tuple[np.ndarray, List[Path]]:
    paths = [raw_dir / f"{frame + 1:05d}.png" for frame in range(start_frame, start_frame + 60)]
    if not all(path.is_file() and not path.is_symlink() for path in paths):
        missing = next(path for path in paths if not path.is_file() or path.is_symlink())
        raise ValueError(f"raw cam00 GOP frame is unavailable: {missing}")
    frames = []
    shape = None
    for path in paths:
        value = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        if shape is None:
            shape = value.shape
        if value.shape != shape:
            raise ValueError("raw cam00 GOP frame shapes differ")
        frames.append(value)
    return np.stack(frames), paths


def load_cotracker(
    repository: Path, checkpoint: Path, device: torch.device
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    if checkpoint.is_symlink():
        raise ValueError("frozen CoTracker checkpoint symlink is forbidden")
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise ValueError("frozen CoTracker checkpoint is unavailable")
    tree = directory_tree_hash(repository)
    model = torch.hub.load(
        str(repository.resolve()),
        "cotracker3_offline",
        source="local",
        pretrained=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("frozen CoTracker checkpoint state is malformed")
    model.model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return model, {
        "repository": str(repository.resolve()),
        "repository_tree": tree,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "entrypoint": "cotracker3_offline",
        "pretrained_network_access": False,
        "strict_checkpoint_load": True,
    }


def track_fixed_grid(
    model: torch.nn.Module, frames: np.ndarray, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], int, int]:
    height, width = frames.shape[1:3]
    points = np.asarray(
        [[x * (width - 1), y * (height - 1)] for y in GRID_COORDS for x in GRID_COORDS],
        dtype=np.float32,
    )
    queries = np.concatenate(
        [np.full((25, 1), 30, dtype=np.float32), points], axis=1
    )
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)
    query_tensor = torch.from_numpy(queries)[None].to(device)
    with torch.no_grad():
        tracks, visibility = model(
            video, queries=query_tensor, backward_tracking=True
        )
    tracks_np = tracks[0].detach().cpu().numpy().astype(np.float64)
    visibility_np = visibility[0].detach().cpu().numpy().astype(np.bool_)
    if visibility_np.ndim == 3 and visibility_np.shape[-1] == 1:
        visibility_np = visibility_np[..., 0]
    if tracks_np.shape != (60, 25, 2) or visibility_np.shape != (60, 25):
        raise ValueError("CoTracker output shape differs from frozen 60x25 contract")
    diagonal = math.hypot(width, height)
    rows = []
    eligible_indices = []
    for index in range(25):
        visible = visibility_np[:, index]
        visible_count = int(visible.sum())
        inside = (
            (tracks_np[:, index, 0] >= 12)
            & (tracks_np[:, index, 0] <= width - 1 - 12)
            & (tracks_np[:, index, 1] >= 12)
            & (tracks_np[:, index, 1] <= height - 1 - 12)
        )
        eligible = visible_count >= 48 and bool(np.all(inside[visible]))
        joint = visible[:-1] & visible[1:]
        displacement = float(
            np.linalg.norm(
                tracks_np[1:, index] - tracks_np[:-1, index], axis=-1
            )[joint].sum()
            / diagonal
        )
        rows.append(
            {
                "grid_index": index,
                "grid_row": index // 5,
                "grid_col": index % 5,
                "visible_frames": visible_count,
                "inside_12px_when_visible": bool(np.all(inside[visible])),
                "eligible": eligible,
                "normalized_path_action": displacement,
            }
        )
        if eligible:
            eligible_indices.append(index)
    if not eligible_indices:
        raise ValueError("REFERENCE_INELIGIBLE:no eligible fixed-grid CoTracker track")
    primary = min(eligible_indices, key=lambda index: (-rows[index]["normalized_path_action"], index))
    static = min(eligible_indices, key=lambda index: (rows[index]["normalized_path_action"], index))
    return tracks_np, visibility_np, rows, primary, static


def canonical_ids(anchors: torch.Tensor, voxel_size: float) -> torch.Tensor:
    ids = torch.round(anchors / float(voxel_size)).to(torch.int64)
    if ids.ndim != 2 or ids.shape[1] != 3:
        raise ValueError("canonical ID tensor has an invalid shape")
    if torch.unique(ids, dim=0).shape[0] != ids.shape[0]:
        raise ValueError("duplicate canonical IDs in H-DOWN bundle")
    return ids


def _camera_click(
    raw_click: Sequence[float], raw_width: int, raw_height: int, width: int, height: int
) -> Tuple[float, float]:
    return (
        float(raw_click[0]) * float(width - 1) / float(raw_width - 1),
        float(raw_click[1]) * float(height - 1) / float(raw_height - 1),
    )


def _select_parent_prefix(
    ids: torch.Tensor, mass: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    ids_np = ids.detach().cpu().numpy().astype(np.int64, copy=False)
    mass_np = mass.detach().cpu().numpy().astype(np.float64, copy=False)
    order = np.lexsort((ids_np[:, 2], ids_np[:, 1], ids_np[:, 0], -mass_np))
    total = float(mass_np.sum())
    count = int(np.searchsorted(np.cumsum(mass_np[order]), 0.90 * total, side="left") + 1)
    selected_ids = ids_np[order[:count]]
    selected_mass = mass_np[order[:count]]
    return selected_ids, selected_mass / total, {
        "parent_prefix_count": count,
        "positive_parent_count": int(ids_np.shape[0]),
        "prefix_fraction_of_positive_patch_mass": float(selected_mass.sum() / total),
        "prefix_threshold": 0.90,
        "tie_break": "canonical_id_lexicographic",
    }


def _render_reference_masks(
    splats,
    decoders,
    app_module,
    config,
    camera,
    ids: torch.Tensor,
    knn_indices: Optional[torch.Tensor],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    pose, intrinsic, width, height, _ = camera
    masks = {f"{scale:.2f}": [] for scale in MASK_SCALES}
    render_rows = []
    for frame in range(60):
        with torch.no_grad():
            before, _, before_audit = render_hdown_frame(
                splats,
                decoders,
                app_module,
                config,
                pose,
                intrinsic,
                width,
                height,
                frame,
                knn_indices=knn_indices,
            )
            after, _, after_audit = render_hdown_frame(
                splats,
                decoders,
                app_module,
                config,
                pose,
                intrinsic,
                width,
                height,
                frame,
                edit_parent_ids=ids,
                knn_indices=knn_indices,
            )
        difference = torch.mean(torch.abs(after - before), dim=-1)[0]
        if not torch.isfinite(difference).all():
            raise ValueError("reference edited-render difference is nonfinite")
        for scale in MASK_SCALES:
            masks[f"{scale:.2f}"].append(
                torch.clamp(difference / scale, 0.0, 1.0).cpu().numpy().astype(np.float32)
            )
        render_rows.append(
            {
                "frame": frame,
                "before_children": before_audit["rendered_child_count"],
                "after_children": after_audit["rendered_child_count"],
                "edited_children": after_audit["edited_child_count"],
            }
        )
    stacked = {key: np.stack(value) for key, value in masks.items()}
    eligible = stacked["0.10"].sum(axis=(1, 2)) >= 25.0
    return stacked, {
        "eligible_frame_count": int(eligible.sum()),
        "eligibility_rule": "reference_soft_mask_0.10_mass_ge_25_pixels",
        "render_rows": render_rows,
        "eligible_frames": eligible,
    }


def build_reference(args: argparse.Namespace) -> Dict[str, Any]:
    if args.gop_id not in range(5):
        raise ValueError("gop-id must be in 0..4")
    if args.scene not in CONFIRMATORY_SCENES and args.scene != "flame_salmon_1":
        raise ValueError("scene is outside the frozen H-DOWN universe")
    device = torch.device(args.device)
    torch.manual_seed(20260715)
    np.random.seed(20260715)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    root, splats, decoders, app_module, config, camera, model_audit, _ = load_bundle(
        args.reference_bundle, device, reference=True
    )
    if (
        exact_string(config.get("scene"), "reference decoder scene") != args.scene
        or exact_nonnegative_int(
            config.get("start_frame"), "reference decoder start frame"
        )
        != 60 * args.gop_id
    ):
        raise ValueError("reference bundle scene/GOP mismatch")
    frames, frame_paths = load_raw_gop(args.raw_cam00_dir, 60 * args.gop_id)
    tracker, tracker_audit = load_cotracker(args.cotracker_repo, args.cotracker_checkpoint, device)
    tracks, visibility, grid_rows, primary_index, static_index = track_fixed_grid(
        tracker, frames, device
    )
    pose, intrinsic, width, height, camera_key = camera
    raw_height, raw_width = frames.shape[1:3]
    knn_indices = counted_knn_indices(splats, config)
    paths = decode_anchor_paths(splats, decoders, config, knn_indices=knn_indices)
    all_ids = canonical_ids(
        splats["anchors"],
        exact_finite_float(
            config.get("voxel_size"), "reference decoder voxel size", positive=True
        ),
    )
    id_map = {
        tuple(int(value) for value in row): index
        for index, row in enumerate(all_ids.detach().cpu().tolist())
    }
    arrays: Dict[str, np.ndarray] = {}
    case_rows = []
    for label, grid_index in (("primary", primary_index), ("static", static_index)):
        raw_click = tracks[30, grid_index]
        click = _camera_click(raw_click, raw_width, raw_height, width, height)
        parent_ids, parent_mass, lift_audit = alpha_parent_contributions(
            splats,
            decoders,
            app_module,
            config,
            pose,
            intrinsic,
            width,
            height,
            30,
            click,
            knn_indices=knn_indices,
        )
        selected_ids, weights, prefix_audit = _select_parent_prefix(parent_ids, parent_mass)
        if selected_ids.shape[0] < 4:
            raise ValueError(f"REFERENCE_INELIGIBLE:{label} alpha lift has fewer than four parents")
        rows = [id_map.get(tuple(int(value) for value in row)) for row in selected_ids.tolist()]
        if any(row is None for row in rows):
            raise ValueError("alpha-lift parent ID is absent from reference anchors")
        selected_paths = paths[torch.tensor(rows, dtype=torch.long, device=device)]
        action = torch.linalg.vector_norm(
            selected_paths[:, 1:] - selected_paths[:, :-1], dim=-1
        ).sum(1).detach().cpu().numpy().astype(np.float64)
        action_mass = weights * action
        if label == "primary" and float(action_mass.sum()) <= 0:
            raise ValueError("REFERENCE_INELIGIBLE:primary alpha lift has zero path-action mass")
        selected_tensor = torch.from_numpy(selected_ids.copy()).to(device=device, dtype=torch.int64)
        masks, render_audit = _render_reference_masks(
            splats,
            decoders,
            app_module,
            config,
            camera,
            selected_tensor,
            knn_indices,
        )
        if int(render_audit["eligible_frame_count"]) < 30:
            raise ValueError(f"REFERENCE_INELIGIBLE:{label} has fewer than 30 eligible frames")
        arrays[f"{label}_canonical_ids"] = selected_ids.astype(np.int64, copy=False)
        arrays[f"{label}_alpha_weights"] = weights.astype(np.float64, copy=False)
        arrays[f"{label}_path_action"] = action
        arrays[f"{label}_action_mass"] = action_mass
        arrays[f"{label}_track_xy"] = tracks[:, grid_index].astype(np.float64, copy=False)
        arrays[f"{label}_track_visibility"] = visibility[:, grid_index].astype(np.bool_, copy=False)
        arrays[f"{label}_eligible_frames"] = render_audit.pop("eligible_frames")
        for scale in MASK_SCALES:
            arrays[f"{label}_reference_mask_{int(round(scale * 100)):03d}"] = masks[f"{scale:.2f}"]
        case_rows.append(
            {
                "label": label,
                "grid_index": int(grid_index),
                "raw_click_xy": [float(value) for value in raw_click],
                "render_click_xy": [float(value) for value in click],
                "selected_parent_count": int(selected_ids.shape[0]),
                "selected_alpha_mass": float(weights.sum()),
                "selected_path_action": float(action.sum()),
                "selected_weighted_action_mass": float(action_mass.sum()),
                "lift": {**lift_audit, **prefix_audit},
                "render": render_audit,
            }
        )
    write_deterministic_npz(args.output_npz, arrays)
    artifact_sha = sha256_file(args.output_npz)
    reference_census_path = root / "byte_census.json"
    reference_census_sha = sha256_file(reference_census_path)
    manifest = {
        "schema": REFERENCE_SCHEMA,
        "status": "ELIGIBLE",
        "scene": args.scene,
        "gop_id": int(args.gop_id),
        "gop_start_frame": int(60 * args.gop_id),
        "source_frame_global": int(30 + 60 * args.gop_id),
        "source_frame_local": 30,
        "camera": "cam00",
        "camera_key": int(camera_key),
        "raw_frame_size": [int(raw_width), int(raw_height)],
        "render_frame_size": [int(width), int(height)],
        "raw_to_render_rule": "align_corners_coordinate_scale",
        "raw_frame_sha256": [sha256_file(path) for path in frame_paths],
        "raw_cam00_dir": str(args.raw_cam00_dir.resolve()),
        "fixed_grid_coordinates": list(GRID_COORDS),
        "track_eligibility": "visible_ge_48_of_60_and_ge_12_raw_pixels_from_boundary_when_visible",
        "grid_tracks": grid_rows,
        "cases": case_rows,
        "artifact": str(args.output_npz.resolve()),
        "artifact_bytes": args.output_npz.stat().st_size,
        "artifact_sha256": artifact_sha,
        "reference_bundle": str(args.reference_bundle.resolve()),
        "reference_bundle_byte_census_sha256": reference_census_sha,
        "reference_model_audit": model_audit,
        "tracker": tracker_audit,
        "rebuild_seed": 20260715,
        "rebuild_device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "deterministic_algorithms": True,
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "candidate_inputs_read": [],
        "outcome_fields_read": [],
    }
    exclusive_write(
        Path(os.path.abspath(os.fspath(args.output_manifest))),
        canonical_json_bytes(manifest),
    )
    return manifest


def verify_reference_rebuild(
    manifest_path: Path, artifact_path: Path
) -> Dict[str, Any]:
    """Replay eligible NPZ bytes or an ineligible status from frozen primitives."""

    manifest_payload = read_regular_bytes(
        Path(os.path.abspath(os.fspath(manifest_path)))
    )
    manifest = strict_canonical_json_bytes(manifest_payload, "H-DOWN reference manifest")
    if manifest.get("schema") != REFERENCE_SCHEMA or manifest.get("status") not in {
        "ELIGIBLE",
        "REFERENCE_INELIGIBLE",
    }:
        raise ValueError("reference rebuild status/schema is unsupported")
    scene = exact_string(manifest.get("scene"), "reference scene")
    gop_id = exact_nonnegative_int(manifest.get("gop_id"), "reference GOP ID")
    if gop_id not in range(5):
        raise ValueError("reference GOP ID is outside 0..4")
    if exact_nonnegative_int(
        manifest.get("gop_start_frame"), "reference GOP start frame"
    ) != 60 * gop_id:
        raise ValueError("reference GOP start differs from its GOP ID")
    raw_dir = Path(exact_string(manifest.get("raw_cam00_dir"), "raw cam00 path"))
    reference_bundle = Path(
        exact_string(manifest.get("reference_bundle"), "reference bundle path")
    )
    tracker = manifest.get("tracker", {})
    if not isinstance(tracker, dict):
        raise ValueError("reference tracker audit is not an object")
    cotracker_repo = Path(
        exact_string(tracker.get("repository"), "CoTracker repository path")
    )
    cotracker_checkpoint = Path(
        exact_string(tracker.get("checkpoint"), "CoTracker checkpoint path")
    )
    raw_hashes_declared = manifest.get("raw_frame_sha256")
    if not isinstance(raw_hashes_declared, list) or len(raw_hashes_declared) != 60:
        raise ValueError("reference raw-frame hash list is not exact")
    for index, digest in enumerate(raw_hashes_declared):
        require_sha256(digest, f"reference raw frame {index} SHA-256")
    census_sha = require_sha256(
        manifest.get("reference_bundle_byte_census_sha256"),
        "reference-bundle census SHA-256",
    )
    rebuild_device = exact_string(
        manifest.get("rebuild_device"), "reference rebuild device"
    )
    if exact_nonnegative_int(
        manifest.get("rebuild_seed"), "reference rebuild seed"
    ) != 20260715:
        raise ValueError("reference rebuild seed differs from the frozen evaluator")

    def open_directory_no_follow(path: Path, label: str) -> None:
        absolute = Path(os.path.abspath(os.fspath(path)))
        parts = [part for part in absolute.parts if part not in ("", "/")]
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in parts:
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=fd,
                )
                os.close(fd)
                fd = next_fd
        except Exception as error:
            raise ValueError(
                f"reference rebuild {label} has a missing/symlink parent"
            ) from error
        finally:
            os.close(fd)

    open_directory_no_follow(raw_dir, "raw cam00 directory")
    open_directory_no_follow(reference_bundle, "reference bundle")
    open_directory_no_follow(cotracker_repo, "CoTracker repository")
    checkpoint_payload = read_regular_bytes(
        Path(os.path.abspath(os.fspath(cotracker_checkpoint)))
    )
    validate_directory_census(reference_bundle)
    census_path = reference_bundle / "byte_census.json"
    if hashlib.sha256(read_regular_bytes(census_path)).hexdigest() != census_sha:
        raise ValueError("reference bundle census changed before rebuild")
    start_frame = gop_id * 60
    _, raw_paths = load_raw_gop(raw_dir, start_frame)
    raw_hashes = [
        hashlib.sha256(
            read_regular_bytes(Path(os.path.abspath(os.fspath(path))))
        ).hexdigest()
        for path in raw_paths
    ]
    if raw_hashes != raw_hashes_declared:
        raise ValueError("raw cam00 primitive bytes changed before reference rebuild")
    if (
        exact_positive_int(
            tracker.get("checkpoint_bytes"), "CoTracker checkpoint bytes"
        )
        != len(checkpoint_payload)
        or require_sha256(
            tracker.get("checkpoint_sha256"), "CoTracker checkpoint SHA-256"
        )
        != hashlib.sha256(checkpoint_payload).hexdigest()
        or tracker.get("repository_tree") != directory_tree_hash(cotracker_repo)
        or exact_string(tracker.get("entrypoint"), "CoTracker entrypoint")
        != "cotracker3_offline"
        or tracker.get("pretrained_network_access") is not False
        or tracker.get("strict_checkpoint_load") is not True
    ):
        raise ValueError("CoTracker primitive closure changed before reference rebuild")
    if (
        exact_string(manifest.get("torch_version"), "reference torch version")
        != torch.__version__
        or exact_string(manifest.get("numpy_version"), "reference NumPy version")
        != np.__version__
        or exact_string(manifest.get("python_version"), "reference Python version")
        != platform.python_version()
        or manifest.get("deterministic_algorithms") is not True
    ):
        raise ValueError("reference rebuild runtime differs from the frozen primitive closure")
    build_args = dict(
        scene=scene,
        gop_id=gop_id,
        raw_cam00_dir=raw_dir,
        reference_bundle=reference_bundle,
        cotracker_repo=cotracker_repo,
        cotracker_checkpoint=cotracker_checkpoint,
        device=rebuild_device,
    )
    with tempfile.TemporaryDirectory(prefix="h007_reference_rebuild_") as temporary:
        output_npz = Path(temporary) / "reference.npz"
        output_manifest = Path(temporary) / "reference.json"
        try:
            rebuilt = build_reference(
                argparse.Namespace(
                    **build_args,
                    output_npz=output_npz,
                    output_manifest=output_manifest,
                )
            )
        except ValueError as error:
            if manifest["status"] != "REFERENCE_INELIGIBLE" or not str(error).startswith(
                "REFERENCE_INELIGIBLE:"
            ):
                raise
            reason = str(error).split(":", 1)[1]
            if reason != manifest.get("reason"):
                raise ValueError("reference ineligible reason is not reproducible")
            return {
                "schema": "h007.reference_rebuild_audit.v1",
                "scene": scene,
                "gop_id": gop_id,
                "status": "REFERENCE_INELIGIBLE",
                "reason": reason,
                "artifact_sha256": None,
                "source_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "reference_bundle_byte_census_sha256": manifest[
                    "reference_bundle_byte_census_sha256"
                ],
                "raw_frame_count": len(manifest["raw_frame_sha256"]),
                "source_inputs_revalidated": True,
                "byte_reproducible": False,
                "status_reproducible": True,
            }
        if manifest["status"] != "ELIGIBLE":
            raise ValueError("reference ineligible status replayed as eligible")
        artifact_payload = read_regular_bytes(
            Path(os.path.abspath(os.fspath(artifact_path)))
        )
        if (
            require_sha256(
                manifest.get("artifact_sha256"), "reference artifact SHA-256"
            )
            != hashlib.sha256(artifact_payload).hexdigest()
            or exact_positive_int(
                manifest.get("artifact_bytes"), "reference artifact bytes"
            )
            != len(artifact_payload)
        ):
            raise ValueError("reference rebuild input artifact differs from its manifest")
        rebuilt_payload = output_npz.read_bytes()
        if rebuilt_payload != artifact_payload:
            raise ValueError("reference NPZ is not byte-reproducible from frozen primitives")

    def normalized(value: Mapping[str, Any]) -> Dict[str, Any]:
        result = dict(value)
        result["artifact"] = "<FROZEN_REFERENCE_ARTIFACT>"
        return result

    if canonical_json_bytes(normalized(rebuilt)) != canonical_json_bytes(
        normalized(manifest)
    ):
        raise ValueError("reference manifest is not reproducible from frozen primitives")
    return {
        "schema": "h007.reference_rebuild_audit.v1",
        "scene": scene,
        "gop_id": gop_id,
        "status": "ELIGIBLE",
        "artifact_sha256": hashlib.sha256(artifact_payload).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "reference_bundle_byte_census_sha256": manifest[
            "reference_bundle_byte_census_sha256"
        ],
        "raw_frame_count": len(manifest["raw_frame_sha256"]),
        "source_inputs_revalidated": True,
        "byte_reproducible": True,
        "status_reproducible": True,
    }


def _soft_iou(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.maximum(candidate, reference).sum())
    if denominator <= 0:
        return float("nan")
    return float(np.minimum(candidate, reference).sum() / denominator)


def _selected_sequence_binding(
    stage: Mapping[str, Any], scene: str, method: str, gop_id: int
) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    matches = [
        row
        for row in stage["selection"]["selected"]
        if row.get("scene") == scene and row.get("method") == method
    ]
    if len(matches) != 1:
        raise ValueError(f"Stage-02 selection lacks one selected sequence: {scene}/{method}")
    selected = matches[0]
    archive = Path(exact_string(selected.get("archive"), "selected archive path"))
    validation = validate_sequence_container(
        archive, expected_scene=scene, expected_method=method
    )
    if (
        validation["archive_sha256"]
        != require_sha256(selected.get("archive_sha256"), "selected archive SHA-256")
        or validation["archive_bytes"]
        != exact_positive_int(selected.get("archive_bytes"), "selected archive bytes")
        or validation["training_config_sha256"]
        != require_sha256(
            selected.get("training_config_sha256"),
            "selected training config SHA-256",
        )
        or validation["seed"]
        != exact_nonnegative_int(selected.get("seed"), "selected seed")
    ):
        raise ValueError("selected nested sequence differs from frozen selection")
    if gop_id not in range(5):
        raise ValueError("selected GOP ID is outside 0..4")
    return selected, validation["gops"][gop_id], archive


def _load_frozen_clean_decoder(stage: Mapping[str, Any]) -> Any:
    """Load only the clean-decoder bytes frozen into the Stage-02 contract."""

    repo_root = Path(
        exact_string(stage["contract"].get("repo_root"), "Stage-02 repository root")
    )
    decoder_path = repo_root / exact_string(
        stage["contract"].get("clean_decoder_relative_path"),
        "clean decoder relative path",
    )
    decoder_payload = read_regular_bytes(decoder_path)
    if hashlib.sha256(decoder_payload).hexdigest() != require_sha256(
        stage["contract"].get("clean_decoder_sha256"),
        "clean decoder SHA-256",
    ):
        raise ValueError("frozen clean-decoder bytes changed before aggregation")
    spec = importlib.util.spec_from_file_location(
        "h007_frozen_clean_decoder_for_aggregate", decoder_path
    )
    if spec is None or spec.loader is None:
        raise ValueError("frozen clean decoder cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "clean_decode", None)):
        raise ValueError("frozen clean decoder lacks the clean_decode entrypoint")
    return module


def _fresh_decode_selected_gop(
    *,
    stage: Mapping[str, Any],
    selected_sequence: Path,
    scene: str,
    method: str,
    gop_id: int,
    output_dir: Path,
    device: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract the selected inner GOP and invoke the frozen clean decoder."""

    gop_id = exact_nonnegative_int(gop_id, "selected GOP ID")
    if gop_id not in range(5):
        raise ValueError("selected GOP ID is outside 0..4")
    validation = validate_sequence_container(
        selected_sequence, expected_scene=scene, expected_method=method
    )
    gop_audit = validation["gops"][gop_id]
    member = f"gops/gop_{gop_id}.zip"
    with zipfile.ZipFile(selected_sequence, "r") as handle:
        infos = [info for info in handle.infolist() if info.filename == member]
        if len(infos) != 1 or infos[0].is_dir():
            raise ValueError("selected sequence lacks one exact inner GOP member")
        if int(infos[0].file_size) != int(gop_audit["bytes"]):
            raise ValueError("selected inner GOP size differs before extraction")
        payload = handle.read(infos[0])
    if (
        len(payload) != int(gop_audit["bytes"])
        or hashlib.sha256(payload).hexdigest() != gop_audit["sha256"]
    ):
        raise ValueError("selected inner GOP payload differs before clean decode")
    with zipfile.ZipFile(io.BytesIO(payload), "r") as inner:
        decoder_rows = [
            info for info in inner.infolist() if info.filename == "decoder_config.json"
        ]
        if len(decoder_rows) != 1:
            raise ValueError("selected inner GOP lacks one decoder config")
        if hashlib.sha256(inner.read(decoder_rows[0])).hexdigest() != gop_audit[
            "decoder_config_sha256"
        ]:
            raise ValueError("selected inner GOP decoder config differs before decode")

    module = _load_frozen_clean_decoder(stage)
    inner_path = output_dir.parent / f"selected_gop_{gop_id}.zip"
    inner_path.write_bytes(payload)
    clean_manifest = module.clean_decode(
        inner_path,
        selected_sequence,
        gop_id,
        output_dir,
        device,
        Path(
            exact_string(
                stage["contract"].get("provenance_manifest"),
                "Stage-02 provenance manifest path",
            )
        ),
        require_sha256(
            stage["contract"].get("provenance_manifest_sha256"),
            "Stage-02 provenance manifest SHA-256",
        ),
    )
    if (
        clean_manifest.get("schema") != "h007.clean_decode_result.v2"
        or clean_manifest.get("source_sequence_archive_sha256")
        != validation["archive_sha256"]
        or clean_manifest.get("source_inner_gop_sha256") != gop_audit["sha256"]
        or clean_manifest.get("runtime_provenance") != stage["runtime_provenance"]
        or clean_manifest.get("producer_receipt_validated") is not True
    ):
        raise ValueError("fresh clean-decode result differs from the selected inner GOP")
    audit = {
        "scene": scene,
        "method": method,
        "gop_id": gop_id,
        "selected_sequence_sha256": validation["archive_sha256"],
        "selected_inner_gop_sha256": gop_audit["sha256"],
        "clean_decoder_sha256": stage["contract"]["clean_decoder_sha256"],
        "clean_decode_manifest_sha256": sha256_file(
            output_dir / "clean_decode_manifest.json"
        ),
        "decoded_splats_sha256": clean_manifest["decoded_splats_sha256"],
        "producer_receipt_sha256": clean_manifest["producer_receipt_sha256"],
        "fresh_inner_gop_decode": True,
    }
    return clean_manifest, audit


def _candidate_provenance_fields(
    *,
    stage: Mapping[str, Any],
    clean_manifest: Mapping[str, Any],
    sequence_validation: Mapping[str, Any],
    gop_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    required = {
        "source_sequence_archive_sha256": sequence_validation["archive_sha256"],
        "source_sequence_manifest_sha256": sequence_validation[
            "sequence_manifest_sha256"
        ],
        "source_inner_gop_sha256": gop_audit["sha256"],
        "source_inner_gop_decoder_config_sha256": gop_audit[
            "decoder_config_sha256"
        ],
        "runtime_provenance": stage["runtime_provenance"],
    }
    for key, expected in required.items():
        if clean_manifest.get(key) != expected:
            raise ValueError(f"clean-decode/selected-sequence binding mismatch: {key}")
    for key in (
        "source_sequence_archive_sha256",
        "source_sequence_manifest_sha256",
        "source_inner_gop_sha256",
        "source_inner_gop_decoder_config_sha256",
        "decoded_splats_sha256",
    ):
        require_sha256(clean_manifest.get(key), f"clean-decode {key}")
    return {
        "candidate_clean_decode_manifest_sha256": sha256_file(
            Path(exact_string(clean_manifest.get("manifest_path"), "clean manifest path"))
        ),
        "candidate_decoded_splats_sha256": clean_manifest[
            "decoded_splats_sha256"
        ],
        "selected_sequence_archive_sha256": sequence_validation[
            "archive_sha256"
        ],
        "selected_sequence_manifest_sha256": sequence_validation[
            "sequence_manifest_sha256"
        ],
        "selected_inner_gop_sha256": gop_audit["sha256"],
        "selected_inner_gop_decoder_config_sha256": gop_audit[
            "decoder_config_sha256"
        ],
        "evaluator_sha256": stage["freeze"]["evaluator_sha256"],
        "runtime_provenance": stage["runtime_provenance"],
    }


def _write_total_candidate_failure(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    reference_sha: str,
    manifest_sha: str,
    freeze_sha: str,
    provenance_fields: Mapping[str, Any],
    reference_payload: bytes,
    error: Exception,
) -> Dict[str, Any]:
    cases = []
    with np.load(io.BytesIO(reference_payload), allow_pickle=False) as reference:
        for label in ("primary", "static"):
            ids = np.asarray(reference[f"{label}_canonical_ids"], dtype=np.int64)
            eligible = np.asarray(reference[f"{label}_eligible_frames"], dtype=np.bool_)
            if ids.ndim != 2 or ids.shape[1] != 3 or eligible.shape != (60,):
                raise ValueError("frozen reference identity/eligibility arrays are malformed")
            per_frame = [
                (
                    {
                        "frame": frame,
                        "status": "OPERATIONAL_FAILURE",
                        "soft_iou": {"005": 0.0, "010": 0.0, "015": 0.0},
                        "penalized_loss": {"005": 1.0, "010": 1.0, "015": 1.0},
                        "render_audit": {
                            "error": f"{type(error).__name__}:{error}"
                        },
                    }
                    if bool(eligible[frame])
                    else {"frame": frame, "status": "REFERENCE_INELIGIBLE_FRAME"}
                )
                for frame in range(60)
            ]
            eligible_count = int(eligible.sum())
            if eligible_count < 30:
                raise ValueError("frozen reference case has fewer than 30 eligible frames")
            cases.append(
                {
                    "label": label,
                    "selected_identity_count": int(ids.shape[0]),
                    "present_identity_count": 0,
                    "missing_identity_count": int(ids.shape[0]),
                    "missing_ids": ids.tolist(),
                    "missing_id_fraction": 1.0,
                    "missing_alpha_mass": 1.0,
                    "missing_path_action_mass": 1.0,
                    "missing_weighted_action_mass": 1.0,
                    "penalty_mass": 1.0,
                    "operational_failure": True,
                    "operational_failure_rate": 1.0,
                    "eligible_frame_count": eligible_count,
                    "mean_soft_iou": {"005": 0.0, "010": 0.0, "015": 0.0},
                    "mean_penalized_loss": {"005": 1.0, "010": 1.0, "015": 1.0},
                    "per_frame": per_frame,
                }
            )
    result = {
        "schema": EVALUATION_SCHEMA,
        "scene": manifest["scene"],
        "gop_id": int(manifest["gop_id"]),
        "gop_start_frame": int(manifest["gop_start_frame"]),
        "camera": "cam00",
        "camera_key": None,
        "method": args.method,
        "reference_artifact_sha256": reference_sha,
        "reference_manifest_sha256": manifest_sha,
        "freeze_manifest_sha256": freeze_sha,
        **dict(provenance_fields),
        "candidate_bundle": str(args.bundle.resolve()),
        "candidate_model_audit": {
            "operational_failure": f"{type(error).__name__}:{error}"
        },
        "candidate_source_images_read": 0,
        "identity_alignment": "exact_canonical_voxel_id",
        "edit": "c_edit=clip(0.25*c+0.75*(1,0,1),0,1)",
        "cases": cases,
    }
    exclusive_write(
        Path(os.path.abspath(os.fspath(args.output))), canonical_json_bytes(result)
    )
    return result


def _recompute_or_total_failure(recompute, total_failure):
    """Convert any fresh-decode/re-evaluation exception into a retained case."""

    try:
        return recompute()
    except Exception as error:
        return total_failure(error)


def evaluate_candidate(
    args: argparse.Namespace, *, validated_stage: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    reference_sha = require_sha256(args.reference_artifact_sha256, "reference artifact SHA-256")
    manifest_sha = require_sha256(args.reference_manifest_sha256, "reference manifest SHA-256")
    reference_payload = read_regular_bytes(
        Path(os.path.abspath(os.fspath(args.reference_artifact)))
    )
    manifest_payload = read_regular_bytes(
        Path(os.path.abspath(os.fspath(args.reference_manifest)))
    )
    if hashlib.sha256(reference_payload).hexdigest() != reference_sha:
        raise ValueError("frozen reference artifact SHA-256 mismatch")
    if hashlib.sha256(manifest_payload).hexdigest() != manifest_sha:
        raise ValueError("frozen reference manifest SHA-256 mismatch")
    freeze_sha = require_sha256(args.freeze_manifest_sha256, "freeze manifest SHA-256")
    stage = (
        validated_stage
        if validated_stage is not None
        else validate_stage02_freeze(args.freeze_manifest, freeze_sha)
    )
    if stage["freeze"]["closure_sha256"] is None:
        raise ValueError("validated Stage-02 closure is malformed")
    frozen_files = stage["files"]
    for frozen_path, digest in (
        (args.reference_artifact.resolve(), reference_sha),
        (args.reference_manifest.resolve(), manifest_sha),
    ):
        try:
            name = frozen_path.relative_to(args.freeze_manifest.parent.resolve()).as_posix()
        except ValueError as error:
            raise ValueError("reference case is outside the frozen Stage-02 tree") from error
        if frozen_files.get(name) != digest:
            raise ValueError("reference case is absent from the frozen Stage-02 manifest")
    manifest = strict_canonical_json_bytes(manifest_payload, "frozen reference manifest")
    if manifest.get("schema") != REFERENCE_SCHEMA or manifest.get("status") != "ELIGIBLE":
        raise ValueError("frozen reference case is not eligible")
    scene = exact_string(manifest.get("scene"), "frozen reference scene")
    gop_id = exact_nonnegative_int(manifest.get("gop_id"), "frozen reference GOP ID")
    if gop_id not in range(5):
        raise ValueError("frozen reference GOP ID is outside 0..4")
    gop_start = exact_nonnegative_int(
        manifest.get("gop_start_frame"), "frozen reference GOP start"
    )
    if gop_start != 60 * gop_id:
        raise ValueError("frozen reference GOP start differs from its GOP ID")
    if require_sha256(
        manifest.get("artifact_sha256"), "frozen reference artifact SHA-256"
    ) != reference_sha:
        raise ValueError("reference manifest/artifact hash mismatch")
    if require_sha256(
        manifest.get("evaluator_sha256"), "frozen reference evaluator SHA-256"
    ) != stage["freeze"]["evaluator_sha256"]:
        raise ValueError("reference case was not produced by the frozen evaluator")
    expected_reference = stage["references"].get((scene, gop_id))
    if expected_reference != manifest:
        raise ValueError("candidate reference differs from the exact Stage-02 case")
    method = exact_string(args.method, "candidate method")
    selected, gop_audit, selected_sequence = _selected_sequence_binding(
        stage,
        scene,
        method,
        gop_id,
    )
    if Path(os.path.abspath(os.fspath(args.selected_sequence))) != Path(
        os.path.abspath(os.fspath(selected_sequence))
    ):
        raise ValueError("candidate invocation does not name the frozen selected sequence")
    sequence_validation = validate_sequence_container(
        selected_sequence,
        expected_scene=scene,
        expected_method=method,
    )
    device = torch.device(args.device)
    try:
        root, splats, decoders, app_module, config, camera, model_audit, clean_manifest = load_bundle(
            args.bundle, device, reference=False
        )
        if clean_manifest is None:
            raise ValueError("candidate bundle lacks a clean-decode manifest")
        provenance_fields = _candidate_provenance_fields(
            stage=stage,
            clean_manifest=clean_manifest,
            sequence_validation=sequence_validation,
            gop_audit=gop_audit,
        )
        current_ids = canonical_ids(
            splats["anchors"],
            exact_finite_float(
                config.get("voxel_size"),
                "candidate decoder voxel size",
                positive=True,
            ),
        )
    except Exception as error:
        raise ValueError(
            "candidate clean-decode closure is unavailable; a manual/fake case is forbidden"
        ) from error
    if (
        exact_string(config.get("scene"), "candidate decoder scene") != scene
        or exact_nonnegative_int(
            config.get("start_frame"), "candidate decoder start frame"
        )
        != gop_start
        or exact_string(config.get("variant"), "candidate decoder variant")
        != method
    ):
        raise ValueError("candidate clean-decoded bundle identity differs from frozen case")
    try:
        knn_indices = counted_knn_indices(splats, config)
    except Exception as error:
        return _write_total_candidate_failure(
            args,
            manifest,
            reference_sha,
            manifest_sha,
            freeze_sha,
            provenance_fields,
            reference_payload,
            error,
        )
    current_map = {
        tuple(int(value) for value in row): index
        for index, row in enumerate(current_ids.detach().cpu().tolist())
    }
    pose, intrinsic, width, height, camera_key = camera
    render_frame_size = manifest.get("render_frame_size")
    if (
        not isinstance(render_frame_size, list)
        or len(render_frame_size) != 2
        or [
            exact_positive_int(value, "frozen reference render dimension")
            for value in render_frame_size
        ]
        != [int(width), int(height)]
    ):
        raise ValueError("candidate render dimensions differ from frozen reference")
    case_outputs = []
    with np.load(io.BytesIO(reference_payload), allow_pickle=False) as reference:
        required = {
            f"{label}_{suffix}"
            for label in ("primary", "static")
            for suffix in (
                "canonical_ids",
                "alpha_weights",
                "path_action",
                "action_mass",
                "eligible_frames",
                "reference_mask_005",
                "reference_mask_010",
                "reference_mask_015",
            )
        }
        missing_members = sorted(required - set(reference.files))
        if missing_members:
            raise ValueError(f"reference artifact lacks members: {missing_members}")
        for label in ("primary", "static"):
            ids = np.asarray(reference[f"{label}_canonical_ids"], dtype=np.int64)
            weights = np.asarray(reference[f"{label}_alpha_weights"], dtype=np.float64)
            action = np.asarray(reference[f"{label}_path_action"], dtype=np.float64)
            action_mass = np.asarray(reference[f"{label}_action_mass"], dtype=np.float64)
            eligible = np.asarray(reference[f"{label}_eligible_frames"], dtype=np.bool_)
            if (
                ids.ndim != 2
                or ids.shape[1] != 3
                or weights.shape != (ids.shape[0],)
                or action.shape != weights.shape
                or action_mass.shape != weights.shape
                or not np.allclose(action_mass, weights * action, rtol=0, atol=1e-12)
                or eligible.shape != (60,)
            ):
                raise ValueError("reference identity/mass arrays are malformed")
            for scale in MASK_SCALES:
                key = f"{int(round(scale * 100)):03d}"
                mask = np.asarray(reference[f"{label}_reference_mask_{key}"])
                if mask.shape != (60, height, width) or not np.isfinite(mask).all():
                    raise ValueError("frozen reference edit-mask tensor is malformed")
            missing = np.asarray(
                [tuple(int(value) for value in row) not in current_map for row in ids],
                dtype=np.bool_,
            )
            present_ids = ids[~missing]
            missing_id_fraction = float(missing.mean())
            missing_alpha_mass = float(weights[missing].sum() / weights.sum())
            missing_path_action_mass = (
                float(action[missing].sum() / action.sum()) if float(action.sum()) > 0 else missing_alpha_mass
            )
            missing_weighted_action_mass = (
                float(action_mass[missing].sum() / action_mass.sum())
                if float(action_mass.sum()) > 0
                else missing_alpha_mass
            )
            penalty_mass = (
                missing_weighted_action_mass if label == "primary" else missing_alpha_mass
            )
            operational_failure = present_ids.shape[0] == 0
            present_tensor = torch.from_numpy(present_ids.copy()).to(
                device=device, dtype=torch.int64
            )
            per_frame = []
            for frame in range(60):
                if not bool(eligible[frame]):
                    per_frame.append(
                        {"frame": frame, "status": "REFERENCE_INELIGIBLE_FRAME"}
                    )
                    continue
                soft_ious: Dict[str, float] = {}
                frame_failure = operational_failure
                render_audit = None
                if not frame_failure:
                    try:
                        with torch.no_grad():
                            before, _, _ = render_hdown_frame(
                                splats,
                                decoders,
                                app_module,
                                config,
                                pose,
                                intrinsic,
                                width,
                                height,
                                frame,
                                knn_indices=knn_indices,
                            )
                            after, _, render_audit = render_hdown_frame(
                                splats,
                                decoders,
                                app_module,
                                config,
                                pose,
                                intrinsic,
                                width,
                                height,
                                frame,
                                edit_parent_ids=present_tensor,
                                knn_indices=knn_indices,
                            )
                        difference = torch.mean(torch.abs(after - before), dim=-1)[0]
                        if not torch.isfinite(difference).all():
                            raise ValueError("candidate edit difference is nonfinite")
                        for scale in MASK_SCALES:
                            key = f"{int(round(scale * 100)):03d}"
                            candidate_mask = (
                                torch.clamp(difference / scale, 0.0, 1.0)
                                .cpu()
                                .numpy()
                            )
                            reference_mask = np.asarray(
                                reference[f"{label}_reference_mask_{key}"][frame],
                                dtype=np.float32,
                            )
                            soft_ious[key] = _soft_iou(candidate_mask, reference_mask)
                        if not all(math.isfinite(value) for value in soft_ious.values()):
                            raise ValueError("candidate soft-IoU is nonfinite")
                    except Exception as error:
                        frame_failure = True
                        render_audit = {"error": f"{type(error).__name__}:{error}"}
                losses = {
                    key: 1.0 if frame_failure else 1.0 - (1.0 - penalty_mass) * value
                    for key, value in soft_ious.items()
                }
                if frame_failure:
                    losses = {"005": 1.0, "010": 1.0, "015": 1.0}
                    soft_ious = {"005": 0.0, "010": 0.0, "015": 0.0}
                per_frame.append(
                    {
                        "frame": frame,
                        "status": "OPERATIONAL_FAILURE" if frame_failure else "OK",
                        "soft_iou": soft_ious,
                        "penalized_loss": losses,
                        "render_audit": render_audit,
                    }
                )
            eligible_rows = [row for row in per_frame if row["status"] != "REFERENCE_INELIGIBLE_FRAME"]
            if len(eligible_rows) < 30:
                raise ValueError("frozen reference case has fewer than 30 eligible frames")
            case_outputs.append(
                {
                    "label": label,
                    "selected_identity_count": int(ids.shape[0]),
                    "present_identity_count": int((~missing).sum()),
                    "missing_identity_count": int(missing.sum()),
                    "missing_ids": ids[missing].tolist(),
                    "missing_id_fraction": missing_id_fraction,
                    "missing_alpha_mass": missing_alpha_mass,
                    "missing_path_action_mass": missing_path_action_mass,
                    "missing_weighted_action_mass": missing_weighted_action_mass,
                    "penalty_mass": penalty_mass,
                    "operational_failure": operational_failure,
                    "operational_failure_rate": float(
                        np.mean([row["status"] == "OPERATIONAL_FAILURE" for row in eligible_rows])
                    ),
                    "eligible_frame_count": len(eligible_rows),
                    "mean_soft_iou": {
                        key: float(np.mean([row["soft_iou"][key] for row in eligible_rows]))
                        for key in ("005", "010", "015")
                    },
                    "mean_penalized_loss": {
                        key: float(np.mean([row["penalized_loss"][key] for row in eligible_rows]))
                        for key in ("005", "010", "015")
                    },
                    "per_frame": per_frame,
                }
            )
    result = {
        "schema": EVALUATION_SCHEMA,
        "scene": manifest["scene"],
        "gop_id": gop_id,
        "gop_start_frame": gop_start,
        "camera": "cam00",
        "camera_key": int(camera_key),
        "method": method,
        "reference_artifact_sha256": reference_sha,
        "reference_manifest_sha256": manifest_sha,
        "freeze_manifest_sha256": freeze_sha,
        **provenance_fields,
        "candidate_bundle": str(args.bundle.resolve()),
        "candidate_model_audit": model_audit,
        "candidate_source_images_read": 0,
        "identity_alignment": "exact_canonical_voxel_id",
        "edit": "c_edit=clip(0.25*c+0.75*(1,0,1),0,1)",
        "cases": case_outputs,
    }
    exclusive_write(
        Path(os.path.abspath(os.fspath(args.output))), canonical_json_bytes(result)
    )
    return result


def _case(result: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    matches = [row for row in result["cases"] if row["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"candidate result lacks one {label} case")
    return matches[0]


def aggregate_results(args: argparse.Namespace) -> Dict[str, Any]:
    """Apply the frozen five-scene H-DOWN gate without dropping failures."""

    freeze_sha = require_sha256(args.freeze_manifest_sha256, "Stage-02 freeze SHA-256")
    stage = validate_stage02_freeze(args.freeze_manifest, freeze_sha)
    selection = read_strict_canonical_json(
        args.selection_manifest, "operating-point selection manifest"
    )
    if (
        canonical_json_bytes(selection)
        != canonical_json_bytes(stage["selection"])
        or sha256_file(args.selection_manifest) != stage["freeze"]["selection_sha256"]
    ):
        raise ValueError("aggregator selection is not the frozen Stage-02 selection")
    if selection.get("schema") != "h007.real_zip_operating_point_selection.v2":
        raise ValueError("operating-point selection manifest schema is unsupported")
    selected = selection.get("selected", [])
    if not isinstance(selected, list):
        raise ValueError("operating-point selection lacks selected rows")
    selected_map = {}
    for row in selected:
        if not isinstance(row, dict):
            raise ValueError("operating-point selection row is not an object")
        identity = (
            exact_string(row.get("scene"), "selected scene"),
            exact_string(row.get("method"), "selected method"),
        )
        if identity in selected_map:
            raise ValueError("operating-point selection identity is duplicated")
        selected_map[identity] = row
    preconditions = read_strict_canonical_json(
        args.preconditions, "image/rate preconditions"
    )
    if (
        canonical_json_bytes(preconditions)
        != canonical_json_bytes(stage["preconditions"])
        or sha256_file(args.preconditions) != stage["freeze"]["preconditions_sha256"]
    ):
        raise ValueError("aggregator preconditions are not the frozen Stage-02 preconditions")
    if preconditions.get("schema") != "h007.hdown_image_rate_preconditions.v2":
        raise ValueError("image/rate precondition schema is unsupported")
    precondition_audits = list(stage["source_revalidation"]["preconditions"])
    if [row["scene"] for row in precondition_audits] != list(
        sorted(CONFIRMATORY_SCENES)
    ):
        raise ValueError("Stage-02 precondition replay audit is incomplete")
    for scene in sorted(CONFIRMATORY_SCENES):
        if (scene, "official") not in selected_map or (
            scene,
            "ap-gifstream-full",
        ) not in selected_map:
            raise ValueError(f"selection manifest lacks primary pair: {scene}")
        official = selected_map[(scene, "official")]
        ap = selected_map[(scene, "ap-gifstream-full")]
        for row in (official, ap):
            method = exact_string(row.get("method"), "selected method")
            archive = Path(exact_string(row.get("archive"), "selected archive path"))
            validation = validate_sequence_container(
                archive,
                expected_scene=scene,
                expected_method=method,
            )
            if (
                validation["archive_bytes"]
                != exact_positive_int(row.get("archive_bytes"), "selected archive bytes")
                or validation["archive_sha256"]
                != require_sha256(
                    row.get("archive_sha256"), "selected archive SHA-256"
                )
                or len(validation["gops"]) != 5
            ):
                raise ValueError(f"selected sequence archive changed: {scene}/{method}")

    reference_status: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for path in args.reference_manifests:
        row = read_strict_canonical_json(path, "reference manifest ledger row")
        if row.get("schema") != REFERENCE_SCHEMA:
            raise ValueError(f"reference manifest schema is unsupported: {path}")
        key = (
            exact_string(row.get("scene"), "reference ledger scene"),
            exact_nonnegative_int(row.get("gop_id"), "reference ledger GOP ID"),
        )
        if key[1] not in range(5):
            raise ValueError("reference ledger GOP ID is outside 0..4")
        if key in reference_status:
            raise ValueError(f"duplicate reference manifest: {key}")
        expected_row = stage["references"].get(key)
        expected_path = (
            stage["root"]
            / "reference_cases"
            / key[0]
            / f"gop_{key[1]}"
            / "reference.json"
        )
        if (
            row != expected_row
            or Path(os.path.abspath(os.fspath(path)))
            != Path(os.path.abspath(os.fspath(expected_path)))
        ):
            raise ValueError(f"reference manifest is not the frozen evaluator output: {key}")
        reference_status[key] = row
    expected_reference = {
        (scene, gop) for scene in CONFIRMATORY_SCENES for gop in range(5)
    }
    if set(reference_status) != expected_reference:
        raise ValueError("reference manifest ledger is not the exact five-scene x five-GOP grid")

    results: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for path in args.case_results:
        row = read_strict_canonical_json(path, "candidate case ledger row")
        if row.get("schema") != EVALUATION_SCHEMA:
            raise ValueError(f"candidate case schema is unsupported: {path}")
        key = (
            exact_string(row.get("scene"), "candidate case scene"),
            exact_string(row.get("method"), "candidate case method"),
            exact_nonnegative_int(row.get("gop_id"), "candidate case GOP ID"),
        )
        if key[2] not in range(5):
            raise ValueError("candidate case GOP ID is outside 0..4")
        if key in results:
            raise ValueError(f"duplicate candidate case result: {key}")
        results[key] = row
    expected_results = {
        (scene, method, gop)
        for scene in CONFIRMATORY_SCENES
        for method in ("official", "ap-gifstream-full")
        for gop in range(5)
        if reference_status[(scene, gop)].get("status") == "ELIGIBLE"
    }
    if set(results) != expected_results:
        raise ValueError("candidate case ledger is not the exact eligible frozen grid")
    freeze_hashes = {
        require_sha256(row.get("freeze_manifest_sha256"), "case freeze SHA-256")
        for row in results.values()
    }
    if freeze_hashes != {freeze_sha}:
        raise ValueError("candidate cases do not bind one aggregator Stage-02 freeze")

    # Submitted JSON is only a complete case ledger.  Every metric used below is
    # regenerated after extracting and clean-decoding the exact selected inner
    # GOP; the submitted candidate_bundle and its tensor are never opened.
    fresh_decode_audits = []
    for key in sorted(expected_results):
        scene, method, gop_id = key
        original = results[key]
        reference_manifest = (
            stage["root"]
            / "reference_cases"
            / scene
            / f"gop_{gop_id}"
            / "reference.json"
        )
        reference_artifact = reference_manifest.with_name("reference.npz")
        selected_sequence = Path(selected_map[(scene, method)]["archive"])
        selected_validation = validate_sequence_container(
            selected_sequence, expected_scene=scene, expected_method=method
        )
        gop_audit = selected_validation["gops"][gop_id]
        submitted_binding = {
            "selected_sequence_archive_sha256": selected_validation["archive_sha256"],
            "selected_sequence_manifest_sha256": selected_validation[
                "sequence_manifest_sha256"
            ],
            "selected_inner_gop_sha256": gop_audit["sha256"],
            "selected_inner_gop_decoder_config_sha256": gop_audit[
                "decoder_config_sha256"
            ],
            "evaluator_sha256": stage["freeze"]["evaluator_sha256"],
            "runtime_provenance": stage["runtime_provenance"],
        }
        for name, expected in submitted_binding.items():
            if original.get(name) != expected:
                raise ValueError(
                    f"submitted case ledger differs from selected evidence: {key}/{name}"
                )
        with tempfile.TemporaryDirectory(prefix="h007_case_recompute_") as temporary:
            temporary_root = Path(temporary)
            fresh_bundle = temporary_root / "fresh_clean_bundle"
            recompute_output = Path(temporary) / "case.json"
            recompute_args = argparse.Namespace(
                bundle=fresh_bundle,
                method=method,
                selected_sequence=selected_sequence,
                reference_artifact=reference_artifact,
                reference_artifact_sha256=sha256_file(reference_artifact),
                reference_manifest=reference_manifest,
                reference_manifest_sha256=sha256_file(reference_manifest),
                freeze_manifest=args.freeze_manifest,
                freeze_manifest_sha256=freeze_sha,
                output=recompute_output,
                device=args.device,
            )
            def recompute_case():
                _, decode_audit = _fresh_decode_selected_gop(
                    stage=stage,
                    selected_sequence=selected_sequence,
                    scene=scene,
                    method=method,
                    gop_id=gop_id,
                    output_dir=fresh_bundle,
                    device=args.device,
                )
                recomputed = evaluate_candidate(
                    recompute_args,
                    validated_stage=stage,
                )
                return recomputed, decode_audit

            def total_failure(error):
                # Operational decode/load failures are scientific outcomes, not
                # permission to drop the case or abort the aggregate.  Preserve
                # the exact selected-archive binding and assign total penalty.
                manifest = read_strict_canonical_json(
                    reference_manifest, "reference manifest for total failure"
                )
                failure_provenance = {
                    **submitted_binding,
                    "candidate_clean_decode_manifest_sha256": None,
                    "candidate_decoded_splats_sha256": None,
                }
                recomputed = _write_total_candidate_failure(
                    recompute_args,
                    manifest,
                    recompute_args.reference_artifact_sha256,
                    recompute_args.reference_manifest_sha256,
                    freeze_sha,
                    failure_provenance,
                    reference_artifact.read_bytes(),
                    error,
                )
                decode_audit = {
                    "scene": scene,
                    "method": method,
                    "gop_id": int(gop_id),
                    "selected_sequence_sha256": selected_validation[
                        "archive_sha256"
                    ],
                    "selected_inner_gop_sha256": gop_audit["sha256"],
                    "clean_decoder_sha256": stage["contract"][
                        "clean_decoder_sha256"
                    ],
                    "fresh_inner_gop_decode": False,
                    "operational_failure": f"{type(error).__name__}:{error}",
                    "total_penalty_assigned": True,
                }
                return recomputed, decode_audit

            recomputed, decode_audit = _recompute_or_total_failure(
                recompute_case, total_failure
            )
            decode_audit["submitted_case_sha256"] = hashlib.sha256(
                canonical_json_bytes(original)
            ).hexdigest()
            decode_audit["recomputed_case_sha256"] = hashlib.sha256(
                canonical_json_bytes(recomputed)
            ).hexdigest()
            fresh_decode_audits.append(decode_audit)
            results[key] = recomputed

    scene_rows = []
    missing_rows = []
    for scene in sorted(CONFIRMATORY_SCENES):
        eligible_gops = [
            gop
            for gop in range(5)
            if reference_status[(scene, gop)].get("status") == "ELIGIBLE"
        ]
        reference_pass = len(eligible_gops) >= 3
        method_rows = {}
        for method in ("official", "ap-gifstream-full"):
            expected = {(scene, method, gop) for gop in eligible_gops}
            missing_results = sorted(expected - set(results))
            if missing_results:
                raise ValueError(f"missing candidate case results: {missing_results}")
            rows = [results[(scene, method, gop)] for gop in eligible_gops]
            metrics: Dict[str, Any] = {
                "method": method,
                "eligible_gops": eligible_gops,
                "primary_mean_loss": {},
                "primary_mean_soft_iou": {},
            }
            for key in ("005", "010", "015"):
                metrics["primary_mean_loss"][key] = (
                    float(
                        np.mean(
                            [
                                _case(row, "primary")["mean_penalized_loss"][key]
                                for row in rows
                            ]
                        )
                    )
                    if rows
                    else None
                )
                metrics["primary_mean_soft_iou"][key] = (
                    float(
                        np.mean(
                            [
                                _case(row, "primary")["mean_soft_iou"][key]
                                for row in rows
                            ]
                        )
                    )
                    if rows
                    else None
                )
            metrics["static_mean_soft_iou_010"] = (
                float(
                    np.mean(
                        [
                            _case(row, "static")["mean_soft_iou"]["010"]
                            for row in rows
                        ]
                    )
                )
                if rows
                else None
            )
            method_rows[method] = metrics
            for row in rows:
                for label in ("primary", "static"):
                    case = _case(row, label)
                    missing_rows.append(
                        {
                            "scene": scene,
                            "gop_id": int(row["gop_id"]),
                            "method": method,
                            "label": label,
                            "missing_id_fraction": case["missing_id_fraction"],
                            "missing_alpha_mass": case["missing_alpha_mass"],
                            "missing_path_action_mass": case["missing_path_action_mass"],
                            "missing_weighted_action_mass": case[
                                "missing_weighted_action_mass"
                            ],
                            "operational_failure_rate": case[
                                "operational_failure_rate"
                            ],
                        }
                    )
        official = method_rows["official"]
        ap = method_rows["ap-gifstream-full"]
        comparable = bool(eligible_gops)
        scene_rows.append(
            {
                "scene": scene,
                "reference_eligible_gops": eligible_gops,
                "reference_ineligible_gops": [
                    {
                        "gop_id": gop,
                        "reason": reference_status[(scene, gop)].get("reason"),
                    }
                    for gop in range(5)
                    if gop not in eligible_gops
                ],
                "reference_minimum_eligibility_pass": reference_pass,
                "official": official,
                "ap_gifstream_full": ap,
                "loss_difference_official_minus_ap": {
                    key: (
                        official["primary_mean_loss"][key]
                        - ap["primary_mean_loss"][key]
                        if comparable
                        else None
                    )
                    for key in ("005", "010", "015")
                },
                "soft_iou_difference_ap_minus_official": {
                    key: (
                        ap["primary_mean_soft_iou"][key]
                        - official["primary_mean_soft_iou"][key]
                        if comparable
                        else None
                    )
                    for key in ("005", "010", "015")
                },
                "static_soft_iou_difference_ap_minus_official": (
                    ap["static_mean_soft_iou_010"]
                    - official["static_mean_soft_iou_010"]
                    if comparable
                    else None
                ),
            }
        )

    statistics_evaluable = all(
        row["reference_minimum_eligibility_pass"]
        and row["loss_difference_official_minus_ap"]["010"] is not None
        for row in scene_rows
    )
    if statistics_evaluable:
        differences = np.asarray(
            [row["loss_difference_official_minus_ap"]["010"] for row in scene_rows],
            dtype=np.float64,
        )
        rng = np.random.default_rng(20260715)
        bootstrap = differences[rng.integers(0, 5, size=(100_000, 5))].mean(axis=1)
        bootstrap_lower = float(np.percentile(bootstrap, 2.5))
        all_positive = bool(np.all(differences > 0))
        sign_p = 1.0 / 32.0 if all_positive else float(
            sum(math.comb(5, k) for k in range(int((differences > 0).sum()), 6))
            / 32.0
        )
        soft_iou_improvement = float(
            np.mean(
                [
                    row["soft_iou_difference_ap_minus_official"]["010"]
                    for row in scene_rows
                ]
            )
        )
        official_loss = float(
            np.mean(
                [row["official"]["primary_mean_loss"]["010"] for row in scene_rows]
            )
        )
        ap_loss = float(
            np.mean(
                [
                    row["ap_gifstream_full"]["primary_mean_loss"]["010"]
                    for row in scene_rows
                ]
            )
        )
        relative_reduction = (
            (official_loss - ap_loss) / official_loss if official_loss > 0 else None
        )
        sensitivity = all(
            all(
                row["loss_difference_official_minus_ap"][key] > 0
                for row in scene_rows
            )
            for key in ("005", "010", "015")
        )
        static_difference = abs(
            float(
                np.mean(
                    [
                        row["static_soft_iou_difference_ap_minus_official"]
                        for row in scene_rows
                    ]
                )
            )
        )
    else:
        differences = np.asarray([], dtype=np.float64)
        bootstrap_lower = sign_p = soft_iou_improvement = relative_reduction = None
        all_positive = sensitivity = False
        static_difference = None
    gates = {
        "all_scene_preconditions": all(row["pass"] for row in precondition_audits),
        "all_scene_reference_eligibility": all(
            row["reference_minimum_eligibility_pass"] for row in scene_rows
        ),
        "all_five_primary_differences_positive": all_positive,
        "exact_sign_test_p_le_0_03125": sign_p is not None and sign_p <= 0.03125,
        "bootstrap_95pct_lower_above_zero": bootstrap_lower is not None
        and bootstrap_lower > 0,
        "minimum_effect": soft_iou_improvement is not None
        and (
            soft_iou_improvement >= 0.10
            or (relative_reduction is not None and relative_reduction >= 0.25)
        ),
        "sensitivity_sign_unchanged": sensitivity,
        "static_control_abs_difference_le_0_02": static_difference is not None
        and static_difference <= 0.02,
    }
    result = {
        "schema": AGGREGATE_SCHEMA,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "scene_rows": scene_rows,
        "preconditions": precondition_audits,
        "statistics": {
            "differences": differences.tolist(),
            "evaluable": statistics_evaluable,
            "exact_sign_test_p": sign_p,
            "bootstrap_resamples": 100_000,
            "bootstrap_seed": 20260715,
            "bootstrap_95pct_lower": bootstrap_lower,
            "mean_absolute_soft_iou_improvement": soft_iou_improvement,
            "mean_relative_penalized_loss_reduction": relative_reduction,
            "static_abs_mean_soft_iou_difference": static_difference,
        },
        "missing_identity_metrics": missing_rows,
        "development_scene_excluded": "flame_salmon_1",
        "freeze_manifest_sha256": next(iter(freeze_hashes)),
        "negative_and_mixed_results_retained": True,
        "fresh_inner_gop_decode_audits": fresh_decode_audits,
        "submitted_case_metrics_used": False,
    }
    exclusive_write(
        Path(os.path.abspath(os.fspath(args.output))), canonical_json_bytes(result)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    reference = sub.add_parser("reference")
    reference.add_argument("--scene", required=True)
    reference.add_argument("--gop-id", type=int, required=True)
    reference.add_argument("--raw-cam00-dir", type=Path, required=True)
    reference.add_argument("--reference-bundle", type=Path, required=True)
    reference.add_argument("--cotracker-repo", type=Path, required=True)
    reference.add_argument("--cotracker-checkpoint", type=Path, required=True)
    reference.add_argument("--output-npz", type=Path, required=True)
    reference.add_argument("--output-manifest", type=Path, required=True)
    reference.add_argument("--device", default="cuda:0")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument("--method", required=True)
    evaluate.add_argument("--selected-sequence", type=Path, required=True)
    evaluate.add_argument("--reference-artifact", type=Path, required=True)
    evaluate.add_argument("--reference-artifact-sha256", required=True)
    evaluate.add_argument("--reference-manifest", type=Path, required=True)
    evaluate.add_argument("--reference-manifest-sha256", required=True)
    evaluate.add_argument("--freeze-manifest", type=Path, required=True)
    evaluate.add_argument("--freeze-manifest-sha256", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda:0")

    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--selection-manifest", type=Path, required=True)
    aggregate.add_argument("--preconditions", type=Path, required=True)
    aggregate.add_argument("--reference-manifests", type=Path, nargs=25, required=True)
    aggregate.add_argument("--case-results", type=Path, nargs="+", required=True)
    aggregate.add_argument("--freeze-manifest", type=Path, required=True)
    aggregate.add_argument("--freeze-manifest-sha256", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--device", default="cuda:0")

    freeze = sub.add_parser("freeze-stage02")
    freeze.add_argument("--root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--repo-root", type=Path, required=True)
    freeze.add_argument("--provenance-manifest", type=Path, required=True)
    freeze.add_argument("--provenance-manifest-sha256", required=True)

    args = parser.parse_args()
    if args.command == "reference":
        try:
            result = build_reference(args)
        except ValueError as error:
            if not str(error).startswith("REFERENCE_INELIGIBLE:"):
                raise
            _, frame_paths = load_raw_gop(
                args.raw_cam00_dir, 60 * int(args.gop_id)
            )
            reference_bundle = args.reference_bundle.resolve()
            validate_directory_census(reference_bundle)
            checkpoint = args.cotracker_checkpoint.resolve()
            tracker_audit = {
                "repository": str(args.cotracker_repo.resolve()),
                "repository_tree": directory_tree_hash(args.cotracker_repo),
                "checkpoint": str(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": sha256_file(checkpoint),
                "entrypoint": "cotracker3_offline",
                "pretrained_network_access": False,
                "strict_checkpoint_load": True,
            }
            result = {
                "schema": REFERENCE_SCHEMA,
                "status": "REFERENCE_INELIGIBLE",
                "scene": args.scene,
                "gop_id": int(args.gop_id),
                "gop_start_frame": int(60 * args.gop_id),
                "source_frame_global": int(30 + 60 * args.gop_id),
                "camera": "cam00",
                "reason": str(error).split(":", 1)[1],
                "raw_frame_sha256": [sha256_file(path) for path in frame_paths],
                "raw_cam00_dir": str(args.raw_cam00_dir.resolve()),
                "reference_bundle": str(reference_bundle),
                "reference_bundle_byte_census_sha256": sha256_file(
                    reference_bundle / "byte_census.json"
                ),
                "tracker": tracker_audit,
                "rebuild_seed": 20260715,
                "rebuild_device": str(torch.device(args.device)),
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "python_version": platform.python_version(),
                "deterministic_algorithms": True,
                "evaluator_sha256": sha256_file(Path(__file__).resolve()),
                "candidate_inputs_read": [],
                "outcome_fields_read": [],
            }
            exclusive_write(
                Path(os.path.abspath(os.fspath(args.output_manifest))),
                canonical_json_bytes(result),
            )
    elif args.command == "evaluate":
        result = evaluate_candidate(args)
    elif args.command == "aggregate":
        result = aggregate_results(args)
    else:
        result = freeze_stage02(
            root=args.root,
            output=args.output,
            repo_root=args.repo_root,
            provenance_manifest=args.provenance_manifest,
            provenance_manifest_sha256=args.provenance_manifest_sha256,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
