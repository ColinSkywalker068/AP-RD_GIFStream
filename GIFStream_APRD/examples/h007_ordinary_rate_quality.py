#!/usr/bin/env python3
"""Generate archive-replayed 300-frame ordinary rate-quality receipts.

Every prediction is rendered in this process from the exact five nested GOPs:
outer sequence ZIP -> frozen clean decoder -> counted model/camera state -> real
GIFStream renderer.  The selector repeats that chain in a fresh temporary tree
before accepting any metric, so arbitrary external PNGs cannot become evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Mapping

import cv2
import numpy as np
import torch
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from gsplat.compression.h007_clean_runtime import (
    counted_knn_indices,
    instantiate_counted_models,
    render_hdown_frame,
)
from gsplat.compression.h007_sequence_container import (
    EVALUATOR_RECEIPT_SCHEMA,
    ORDINARY_EVALUATOR_RELATIVE_PATH,
    ORDINARY_METRIC_PROTOCOL,
    canonical_json_bytes,
    validate_sequence_container,
)


CLEAN_DECODER_RELATIVE_PATH = "examples/h007_clean_decode_gifstream.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image(
    path: Path,
    device: torch.device,
    *,
    target_size: tuple[int, int] | None = None,
    source_factor: int | None = None,
) -> torch.Tensor:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"metric image is unavailable or a symlink: {path}")
    value = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if target_size is not None and (value.shape[1], value.shape[0]) != target_size:
        if source_factor is None or source_factor <= 1:
            raise ValueError("prediction/reference image shape mismatch")
        target_width, target_height = target_size
        if (
            value.shape[1] // source_factor != target_width
            or value.shape[0] // source_factor != target_height
        ):
            raise ValueError("prediction/reference image shape mismatch")
        value = cv2.resize(
            value,
            dsize=target_size,
            interpolation=cv2.INTER_LINEAR,
        )
    value = value.astype(np.float32) / 255.0
    return torch.from_numpy(value).permute(2, 0, 1)[None].to(device)


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("evaluator artifact must live below the evidence root") from error


def _bound_relative(root: Path, declared: Any, label: str) -> Path:
    relative = Path(str(declared))
    if (
        not str(declared)
        or relative.is_absolute()
        or ".." in relative.parts
        or "\\" in str(declared)
    ):
        raise ValueError(f"{label} is not a safe relative path")
    raw = root / relative
    if raw.is_symlink():
        raise ValueError(f"{label} symlink is forbidden")
    try:
        raw.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the evidence root") from error
    return raw


def _metric_suite(device: torch.device):
    return (
        PeakSignalNoiseRatio(data_range=1.0).to(device),
        StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
        LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device),
    )


def _frame_metrics(
    prediction_path: Path,
    reference_path: Path,
    device: torch.device,
    suite,
    reference_factor: int,
) -> Dict[str, float]:
    prediction = _image(prediction_path, device)
    reference = _image(
        reference_path,
        device,
        target_size=(int(prediction.shape[-1]), int(prediction.shape[-2])),
        source_factor=reference_factor,
    )
    if prediction.shape != reference.shape:
        raise ValueError("prediction/reference image shape mismatch")
    psnr_metric, ssim_metric, lpips_metric = suite
    psnr = float(psnr_metric(prediction, reference).item())
    psnr_metric.reset()
    ssim = float(ssim_metric(prediction, reference).item())
    ssim_metric.reset()
    lpips = float(
        lpips_metric(prediction * 2.0 - 1.0, reference * 2.0 - 1.0).item()
    )
    lpips_metric.reset()
    return {"psnr": psnr, "ssim": ssim, "lpips": lpips}


def _load_clean_decoder(path: Path, expected_sha256: str):
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError("frozen clean decoder is unavailable or changed")
    spec = importlib.util.spec_from_file_location(
        "h007_clean_decoder_for_ordinary_replay", path
    )
    if spec is None or spec.loader is None:
        raise ValueError("frozen clean decoder cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "clean_decode", None)):
        raise ValueError("frozen clean decoder lacks clean_decode")
    return module


def _load_counted_bundle(bundle: Path, device: torch.device):
    root = bundle / "container"
    clean_path = bundle / "clean_decode_manifest.json"
    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    decoded = bundle / "decoded_splats.pt"
    if (
        clean.get("schema") != "h007.clean_decode_result.v2"
        or clean.get("decoded_splats_sha256") != _sha256(decoded)
        or int(clean.get("source_images_read", -1)) != 0
    ):
        raise ValueError("fresh ordinary replay lacks a valid clean-decode closure")
    config = json.loads((root / "decoder_config.json").read_text(encoding="utf-8"))
    nets = torch.load(root / "nets.pt", map_location=device, weights_only=True)
    decoders, app_module, _ = instantiate_counted_models(nets, config, device)
    splats = torch.load(decoded, map_location=device, weights_only=True)
    if not isinstance(splats, dict):
        raise ValueError("fresh decoded splats are not a tensor mapping")
    splats = {name: value.to(device) for name, value in splats.items()}

    camera_dir = root / "camera_metadata"
    keys = np.load(camera_dir / "camera_keys.npy", allow_pickle=False)
    intrinsics = np.load(camera_dir / "intrinsics.npy", allow_pickle=False)
    sizes = np.load(camera_dir / "image_sizes.npy", allow_pickle=False)
    poses = np.load(camera_dir / "camtoworlds.npy", allow_pickle=False)
    camera_ids = np.load(camera_dir / "camera_ids.npy", allow_pickle=False)
    camera_names = np.load(camera_dir / "camera_names.npy", allow_pickle=False)
    pose_index = int(config["warm_camera_pose_index"])
    if (
        config.get("test_set") != [0]
        or pose_index != 0
        or int(config.get("warm_frame_index", -1)) != 0
        or pose_index >= poses.shape[0]
        or camera_names.shape != (poses.shape[0],)
    ):
        raise ValueError("counted ordinary replay is not fixed to cam00/pose 0/frame 0")
    camera_key = int(camera_ids[pose_index])
    locations = np.flatnonzero(keys == camera_key)
    if locations.size != 1:
        raise ValueError("counted cam00 key is absent or duplicated")
    meta_index = int(locations[0])
    width, height = [int(value) for value in sizes[meta_index]]
    pose = torch.from_numpy(np.asarray(poses[pose_index], dtype=np.float32)).to(device)
    if pose.shape == (3, 4):
        pose = torch.cat([pose, pose.new_tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
    intrinsic = torch.from_numpy(
        np.asarray(intrinsics[meta_index], dtype=np.float32)
    ).to(device)
    if pose.shape != (4, 4) or intrinsic.shape != (3, 3):
        raise ValueError("counted cam00 pose/intrinsic shapes are invalid")
    camera_binding = {
        "source_camera": "cam00",
        "dataset_camera_index": 0,
        "pose_index": 0,
        "camera_key": camera_key,
        "counted_camera_name": str(camera_names[pose_index]),
        "frame_size": [width, height],
        "local_frames": list(range(60)),
    }
    return (
        splats,
        decoders,
        app_module,
        config,
        pose[None],
        intrinsic[None],
        width,
        height,
        clean,
        camera_binding,
    )


def _fresh_sequence_predictions(
    *,
    sequence_archive: Path,
    output_root: Path,
    device_name: str,
    provenance_manifest: Path,
    provenance_manifest_sha256: str,
    clean_decoder_path: Path,
    clean_decoder_sha256: str,
) -> Dict[str, Any]:
    """Freshly decode and render all 300 predictions from counted archive bytes."""

    sequence_archive = sequence_archive.resolve()
    provenance_manifest = provenance_manifest.resolve()
    validation = validate_sequence_container(sequence_archive)
    if output_root.exists() or output_root.parent.is_symlink():
        raise ValueError("generated prediction root must be a new path")
    output_root.mkdir(parents=True)
    prediction_root = output_root / "predictions"
    prediction_root.mkdir()
    decoder = _load_clean_decoder(clean_decoder_path, clean_decoder_sha256)
    device = torch.device(device_name)
    prediction_rows = []
    timing_rows = []
    data_factors = set()

    with zipfile.ZipFile(sequence_archive, "r") as outer, torch.no_grad():
        for gop_id in range(5):
            gop_audit = validation["gops"][gop_id]
            member = f"gops/gop_{gop_id}.zip"
            infos = [info for info in outer.infolist() if info.filename == member]
            if len(infos) != 1 or infos[0].is_dir():
                raise ValueError("selected sequence lacks one exact nested GOP")
            payload = outer.read(infos[0])
            if (
                len(payload) != int(gop_audit["bytes"])
                or hashlib.sha256(payload).hexdigest() != gop_audit["sha256"]
            ):
                raise ValueError("selected nested GOP differs before clean decode")
            gop_root = output_root / f"gop_{gop_id}"
            gop_root.mkdir()
            inner = gop_root / f"gop_{gop_id}.zip"
            inner.write_bytes(payload)
            bundle = gop_root / "clean_bundle"
            clean = decoder.clean_decode(
                inner,
                sequence_archive,
                gop_id,
                bundle,
                device_name,
                provenance_manifest,
                provenance_manifest_sha256,
            )
            loaded = _load_counted_bundle(bundle, device)
            (
                splats,
                decoders,
                app_module,
                config,
                pose,
                intrinsic,
                width,
                height,
                loaded_clean,
                camera_binding,
            ) = loaded
            if loaded_clean != clean:
                raise ValueError("returned/written clean-decode manifests differ")
            data_factor = int(config.get("data_factor", -1))
            if data_factor <= 0:
                raise ValueError("counted decoder config has an invalid data factor")
            data_factors.add(data_factor)
            knn_indices = counted_knn_indices(splats, config)
            for local_frame in range(60):
                global_frame = gop_id * 60 + local_frame
                rendered, _, _ = render_hdown_frame(
                    splats,
                    decoders,
                    app_module,
                    config,
                    pose,
                    intrinsic,
                    width,
                    height,
                    local_frame,
                    knn_indices=knn_indices,
                )
                rgb = (
                    torch.clamp(rendered[..., :3], 0.0, 1.0)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                rgb_u8 = (rgb * 255.0).astype(np.uint8)
                prediction = prediction_root / f"frame_{global_frame:05d}.png"
                Image.fromarray(rgb_u8, mode="RGB").save(
                    prediction, format="PNG", optimize=False, compress_level=9
                )
                prediction_rows.append(
                    {
                        "frame": global_frame,
                        "path": prediction,
                        "bytes": prediction.stat().st_size,
                        "sha256": _sha256(prediction),
                    }
                )
            clean_manifest_path = bundle / "clean_decode_manifest.json"
            render = clean["counted_camera_render"]
            timing_rows.append(
                {
                    "gop_id": gop_id,
                    "inner_gop_sha256": gop_audit["sha256"],
                    "encode_seconds": gop_audit["encode_seconds"],
                    "decode_seconds": gop_audit["decode_seconds"],
                    "clean_decode_receipt_path": clean_manifest_path,
                    "clean_decode_receipt_sha256": _sha256(clean_manifest_path),
                    "decoded_splats_sha256": clean["decoded_splats_sha256"],
                    "decoded_tensor_manifest_sha256": hashlib.sha256(
                        canonical_json_bytes(clean["tensors"])
                    ).hexdigest(),
                    "prediction_camera_binding": camera_binding,
                    "rendered_frames": int(render["timed_renders"]),
                    "render_elapsed_seconds": float(render["seconds"]),
                    "render_fps": float(render["fps"]),
                }
            )
    if [row["frame"] for row in prediction_rows] != list(range(300)):
        raise ValueError("archive replay did not produce the exact 300-frame grid")
    if len(data_factors) != 1:
        raise ValueError("five-GOP archive replay disagrees on the data factor")
    return {
        "validation": validation,
        "predictions": prediction_rows,
        "timing_trials": timing_rows,
        "clean_decoder_sha256": clean_decoder_sha256,
        "data_factor": data_factors.pop(),
    }


def recompute_receipt_metrics(receipt_path: Path) -> Dict[str, Any]:
    """Repeat archive->decode->render->metrics and return a chain audit."""

    receipt_path = Path(os.path.abspath(os.fspath(receipt_path)))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    root = receipt_path.parent
    sequence_archive = _bound_relative(
        root, receipt.get("sequence_archive"), "receipt sequence archive"
    )
    provenance_manifest = _bound_relative(
        root,
        receipt.get("runtime_provenance_manifest"),
        "receipt runtime-provenance manifest",
    )
    repo_root = Path(__file__).resolve().parents[1]
    clean_decoder = repo_root / str(receipt.get("clean_decoder_relative_path", ""))
    with tempfile.TemporaryDirectory(prefix="h007_ordinary_replay_") as temporary:
        replay = _fresh_sequence_predictions(
            sequence_archive=sequence_archive,
            output_root=Path(temporary) / "generated",
            device_name=str(receipt.get("metric_device", "")),
            provenance_manifest=provenance_manifest,
            provenance_manifest_sha256=str(
                receipt.get("runtime_provenance_manifest_sha256", "")
            ),
            clean_decoder_path=clean_decoder,
            clean_decoder_sha256=str(receipt.get("clean_decoder_sha256", "")),
        )
        declared_frames = {
            int(row["frame"]): row for row in receipt.get("frame_metrics", [])
        }
        fresh_frames = {int(row["frame"]): row for row in replay["predictions"]}
        if set(declared_frames) != set(range(300)) or set(fresh_frames) != set(range(300)):
            raise ValueError("metric replay receipt is not the exact 300-frame grid")
        metrics = {}
        prediction_sha256 = {}
        device = torch.device(str(receipt.get("metric_device", "")))
        suite = _metric_suite(device)
        for frame in range(300):
            declared = declared_frames[frame]
            fresh = fresh_frames[frame]
            if (
                int(declared["prediction_bytes"]) != int(fresh["bytes"])
                or declared["prediction_sha256"] != fresh["sha256"]
            ):
                raise ValueError(
                    f"stored prediction differs from archive-derived replay: frame {frame}"
                )
            reference = _bound_relative(
                root, declared["reference"], f"receipt reference frame {frame}"
            )
            metrics[frame] = _frame_metrics(
                fresh["path"],
                reference,
                device,
                suite,
                int(replay["data_factor"]),
            )
            prediction_sha256[frame] = fresh["sha256"]

        declared_timing = {
            int(row["gop_id"]): row for row in receipt.get("timing_trials", [])
        }
        fresh_timing = {
            int(row["gop_id"]): row for row in replay["timing_trials"]
        }
        if set(declared_timing) != set(range(5)) or set(fresh_timing) != set(range(5)):
            raise ValueError("metric replay timing grid is not the exact five GOPs")
        decoded_tensor_manifest_sha256 = {}
        prediction_camera_binding = {}
        for gop_id in range(5):
            declared = declared_timing[gop_id]
            fresh = fresh_timing[gop_id]
            if (
                declared["decoded_tensor_manifest_sha256"]
                != fresh["decoded_tensor_manifest_sha256"]
                or declared["prediction_camera_binding"]
                != fresh["prediction_camera_binding"]
            ):
                raise ValueError(
                    f"stored clean decode differs from archive-derived replay: GOP {gop_id}"
                )
            decoded_tensor_manifest_sha256[gop_id] = fresh[
                "decoded_tensor_manifest_sha256"
            ]
            prediction_camera_binding[gop_id] = fresh[
                "prediction_camera_binding"
            ]
        return {
            "archive_sha256": replay["validation"]["archive_sha256"],
            "clean_decoder_sha256": replay["clean_decoder_sha256"],
            "prediction_sha256": prediction_sha256,
            "decoded_tensor_manifest_sha256": decoded_tensor_manifest_sha256,
            "prediction_camera_binding": prediction_camera_binding,
            "metrics": metrics,
        }


def generate(args: argparse.Namespace) -> Dict[str, Any]:
    output = Path(os.path.abspath(os.fspath(args.output)))
    if output.exists() or output.parent.is_symlink():
        raise ValueError("evaluator receipt output must be a new regular path")
    output_root = Path(os.path.abspath(os.fspath(args.generated_output_root)))
    source_payload = args.source_data_manifest.read_bytes()
    source = json.loads(source_payload.decode("utf-8"))
    source_rows = {str(row["path"]): row for row in source.get("files", [])}
    expected = {f"cam00/{frame + 1:05d}.png" for frame in range(300)}
    if not expected.issubset(source_rows):
        raise ValueError("source-data manifest lacks the exact cam00 300-frame grid")
    clean_decoder_path = Path(__file__).resolve().with_name(
        "h007_clean_decode_gifstream.py"
    )
    replay = _fresh_sequence_predictions(
        sequence_archive=args.sequence_archive,
        output_root=output_root,
        device_name=args.device,
        provenance_manifest=args.provenance_manifest,
        provenance_manifest_sha256=args.provenance_manifest_sha256,
        clean_decoder_path=clean_decoder_path,
        clean_decoder_sha256=_sha256(clean_decoder_path),
    )
    validation = replay["validation"]
    if validation["scene"] != args.scene or validation["method"] != args.method:
        raise ValueError("archive replay scene/method differs from evaluator request")

    device = torch.device(args.device)
    suite = _metric_suite(device)
    frame_rows = []
    with torch.no_grad():
        for fresh in replay["predictions"]:
            frame = int(fresh["frame"])
            reference_name = f"cam00/{frame + 1:05d}.png"
            reference_path = args.source_data_manifest.parent / reference_name
            metrics = _frame_metrics(
                fresh["path"],
                reference_path,
                device,
                suite,
                int(replay["data_factor"]),
            )
            frame_rows.append(
                {
                    "frame": frame,
                    "prediction": _relative(output.parent, fresh["path"]),
                    "reference": _relative(output.parent, reference_path),
                    "prediction_bytes": fresh["bytes"],
                    "prediction_sha256": fresh["sha256"],
                    "reference_sha256": str(source_rows[reference_name]["sha256"]),
                    **metrics,
                }
            )

    timing_rows = []
    for row in replay["timing_trials"]:
        timing_rows.append(
            {
                "gop_id": row["gop_id"],
                "inner_gop_sha256": row["inner_gop_sha256"],
                "encode_seconds": row["encode_seconds"],
                "decode_seconds": row["decode_seconds"],
                "clean_decode_receipt": _relative(
                    output.parent, row["clean_decode_receipt_path"]
                ),
                "clean_decode_receipt_sha256": row[
                    "clean_decode_receipt_sha256"
                ],
                "decoded_splats_sha256": row["decoded_splats_sha256"],
                "decoded_tensor_manifest_sha256": row[
                    "decoded_tensor_manifest_sha256"
                ],
                "prediction_camera_binding": row["prediction_camera_binding"],
                "rendered_frames": row["rendered_frames"],
                "render_elapsed_seconds": row["render_elapsed_seconds"],
                "render_fps": row["render_fps"],
            }
        )

    evaluator_path = Path(__file__).resolve()
    receipt = {
        "schema": EVALUATOR_RECEIPT_SCHEMA,
        "scene": args.scene,
        "method": args.method,
        "point_id": args.point_id,
        "sequence_archive": _relative(output.parent, args.sequence_archive),
        "archive_sha256": validation["archive_sha256"],
        "archive_bytes": validation["archive_bytes"],
        "training_config_sha256": validation["training_config_sha256"],
        "seed": validation["seed"],
        "source_data_manifest": _relative(output.parent, args.source_data_manifest),
        "source_data_manifest_sha256": hashlib.sha256(source_payload).hexdigest(),
        "runtime_provenance_manifest": _relative(
            output.parent, args.provenance_manifest
        ),
        "runtime_provenance_manifest_sha256": args.provenance_manifest_sha256,
        "clean_decoder_relative_path": CLEAN_DECODER_RELATIVE_PATH,
        "clean_decoder_sha256": _sha256(clean_decoder_path),
        "generated_predictions_root": _relative(output.parent, output_root),
        "evaluator_relative_path": ORDINARY_EVALUATOR_RELATIVE_PATH,
        "evaluator_sha256": _sha256(evaluator_path),
        "metric_device": str(device),
        "metric_protocol": ORDINARY_METRIC_PROTOCOL,
        "frame_metrics": frame_rows,
        "timing_trials": timing_rows,
        "outcome_fields_read": [
            "ordinary_unedited_fidelity",
            "real_container_accounting",
        ],
    }
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        payload = canonical_json_bytes(receipt)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short evaluator-receipt write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--point-id", required=True)
    parser.add_argument("--sequence-archive", type=Path, required=True)
    parser.add_argument("--source-data-manifest", type=Path, required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--provenance-manifest-sha256", required=True)
    parser.add_argument("--generated-output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(generate(args), sort_keys=True))


if __name__ == "__main__":
    main()
