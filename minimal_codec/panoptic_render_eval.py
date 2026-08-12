#!/usr/bin/env python
"""Render PanopticSports D3DG trajectory variants and compute image metrics.

Candidates may contain only `means3D`; all other attributes are inherited from
the reference `params.npz`. This matches the same-identity trajectory variants
used by `panoptic_track_eval.py`. Candidates can also be q16 payload `.npz`
files or APRDZ entropy bitstreams.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from panoptic_entropy_codec import HEADER_STRUCT, MAGIC, decode_bitstream


def parse_variant(raw: str) -> tuple[str, str]:
    if "::" not in raw:
        raise argparse.ArgumentTypeError(f"variant must be label::path, got {raw!r}")
    label, path = raw.split("::", 1)
    if not label:
        raise argparse.ArgumentTypeError(f"empty label in {raw!r}")
    return label, path


def parse_int_list(raw: str) -> list[int]:
    return [int(value) for value in raw.replace(",", " ").split() if value.strip()]


def temporal_sampling_groups(candidate: str, num_frames: int, num_gaussians: int) -> list[dict] | None:
    """Return temporal sampling groups without decoding candidate payloads.

    The result is used only to label evaluated frames as coded or interpolated;
    it never changes which frames are evaluated.
    """
    if candidate == "reference":
        return [{"stride": 1, "gaussians": int(num_gaussians)}]
    path = Path(candidate)
    if path.suffix == ".aprdz":
        with path.open("rb") as f:
            if f.read(len(MAGIC)) != MAGIC:
                raise ValueError(f"{path} is not an APRDZ bitstream")
            raw = f.read(HEADER_STRUCT.size)
            if len(raw) != HEADER_STRUCT.size:
                raise ValueError(f"{path} has a truncated APRDZ header")
            (header_len,) = HEADER_STRUCT.unpack(raw)
            header = json.loads(f.read(header_len).decode("utf-8"))
        return [
            {"stride": int(group["stride"]), "gaussians": int(group["gaussians"])}
            for group in header.get("groups", [])
        ]
    if path.suffix == ".npz":
        payload = np.load(path, allow_pickle=True)
        if "stride_values" in payload.files:
            return [
                {
                    "stride": int(stride),
                    "gaussians": int(len(payload[f"indices_s{int(stride)}"])),
                }
                for stride in payload["stride_values"].astype(int).tolist()
            ]
    return None


def coded_frame_status(groups: list[dict] | None, frame: int, num_frames: int) -> dict:
    if not groups:
        return {
            "coded_keyframe_fraction": None,
            "is_coded_keyframe_any": None,
            "is_coded_keyframe_all": None,
        }
    total = sum(int(group["gaussians"]) for group in groups)
    coded = sum(
        int(group["gaussians"])
        for group in groups
        if frame == num_frames - 1 or frame % int(group["stride"]) == 0
    )
    return {
        "coded_keyframe_fraction": float(coded / max(total, 1)),
        "is_coded_keyframe_any": bool(coded > 0),
        "is_coded_keyframe_all": bool(coded == total),
    }


def psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    err = (pred - target) ** 2
    if mask is not None:
        weight = mask[None].expand_as(err)
        denom = torch.clamp(weight.sum(), min=1.0)
        mse = (err * weight).sum() / denom
    else:
        mse = err.mean()
    return float((-10.0 * torch.log10(torch.clamp(mse, min=1e-12))).detach().cpu())


def load_image(path: Path) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).cuda().permute(2, 0, 1).contiguous()


def load_mask(path: Path) -> torch.Tensor:
    arr = np.array(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.max() > 1.0:
        arr = arr / 255.0
    return torch.from_numpy(arr > 0.5).cuda()


def maybe_load_mask(data_dir: Path, fn: str) -> torch.Tensor | None:
    candidates = [
        data_dir / "seg" / fn.replace(".jpg", ".png"),
        data_dir / "seg" / Path(fn).name.replace(".jpg", ".png"),
    ]
    for path in candidates:
        if path.exists():
            return load_mask(path)
    return None


def make_rendervar(
    ref: np.lib.npyio.NpzFile,
    means_t: np.ndarray,
    t: int,
    static: dict[str, torch.Tensor],
    colors_override: np.ndarray | None = None,
) -> dict:
    means = torch.from_numpy(np.asarray(means_t, dtype=np.float32)).cuda()
    colors_np = ref["rgb_colors"][t] if colors_override is None else colors_override
    colors = torch.from_numpy(np.asarray(colors_np, dtype=np.float32)).cuda()
    rotations = torch.from_numpy(np.asarray(ref["unnorm_rotations"][t], dtype=np.float32)).cuda()
    return {
        "means3D": means,
        "colors_precomp": colors,
        "rotations": torch.nn.functional.normalize(rotations),
        "opacities": static["opacities"],
        "scales": static["scales"],
        "means2D": torch.zeros_like(means, device="cuda"),
    }


def render_image(cam, rendervar: dict, cam_m: torch.Tensor, cam_c: torch.Tensor, cam_id: int) -> torch.Tensor:
    im, _, _ = Renderer(raster_settings=cam)(**rendervar)
    im = torch.exp(cam_m[cam_id])[:, None, None] * im + cam_c[cam_id][:, None, None]
    return im.clamp(0.0, 1.0)


def render_raw(cam, rendervar: dict) -> torch.Tensor:
    im, _, _ = Renderer(raster_settings=cam)(**rendervar)
    return im.clamp(0.0, 1.0)


def decode_payload_means(payload: np.lib.npyio.NpzFile) -> np.ndarray:
    num_frames = int(payload["num_frames"][0])
    num_gaussians = int(payload["num_gaussians"][0])
    axis_min = payload["axis_min"].astype(np.float32)
    axis_scale = payload["axis_scale"].astype(np.float32)
    out = np.empty((num_frames, num_gaussians, 3), dtype=np.float32)
    for stride in payload["stride_values"].astype(int).tolist():
        idx = payload[f"indices_s{stride}"].astype(np.int64)
        keys = payload[f"keys_s{stride}"].astype(np.int64)
        q = payload[f"means_q16_s{stride}"].astype(np.float32)
        vals = q * axis_scale[None, None, :] + axis_min[None, None, :]
        for ka, kb, va, vb in zip(keys[:-1], keys[1:], vals[:-1], vals[1:]):
            denom = max(1, int(kb - ka))
            for t in range(int(ka), int(kb) + 1):
                alpha = (t - ka) / denom
                out[t, idx, :] = (1.0 - alpha) * va + alpha * vb
    return out


def lpips_distance(model, pred: torch.Tensor, target: torch.Tensor) -> float | None:
    if model is None:
        return None
    pred_b = pred.mul(2.0).sub(1.0).unsqueeze(0)
    target_b = target.mul(2.0).sub(1.0).unsqueeze(0)
    return float(model(pred_b, target_b).detach().cpu().reshape(-1)[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d3dg-root", required=True)
    parser.add_argument("--seq", default="basketball")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--per-frame-json", default="")
    parser.add_argument("--camera-indices", default="", help="optional comma-separated metadata camera indices")
    parser.add_argument("--mask-source", choices=["none", "file", "rendered_reference"], default="none")
    parser.add_argument("--mask-root", default="", help="sequence root containing seg/; defaults to D3DG data/<seq>")
    parser.add_argument("--lpips", action="store_true", help="compute LPIPS if the package is importable")
    parser.add_argument("--variants", nargs="+", type=parse_variant, required=True)
    args = parser.parse_args()

    root = Path(args.d3dg_root)
    sys.path.insert(0, str(root))
    from helpers import setup_camera  # noqa: PLC0415
    from external import calc_ssim  # noqa: PLC0415

    torch.set_grad_enabled(False)
    lpips_model = None
    if args.lpips:
        try:
            import lpips  # noqa: PLC0415

            lpips_model = lpips.LPIPS(net="alex").cuda().eval()
            print("LPIPS_ENABLED", flush=True)
        except Exception as exc:  # pragma: no cover - depends on HPC env
            print(f"LPIPS_UNAVAILABLE {type(exc).__name__}: {exc}", flush=True)
    ref = np.load(args.reference, allow_pickle=True)
    metadata = json.loads((root / "data" / args.seq / f"{args.split}_meta.json").read_text())
    data_dir = root / "data" / args.seq
    mask_data_dir = Path(args.mask_root) if args.mask_root else data_dir
    frame_offset = min(max(args.frame_offset, 0), max(len(metadata["fn"]) - 1, 0))
    frame_ids = list(range(frame_offset, len(metadata["fn"]), max(args.frame_stride, 1)))
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    requested_cameras = parse_int_list(args.camera_indices) if args.camera_indices else None

    static = {
        "opacities": torch.sigmoid(torch.from_numpy(np.asarray(ref["logit_opacities"], dtype=np.float32)).cuda()),
        "scales": torch.exp(torch.from_numpy(np.asarray(ref["log_scales"], dtype=np.float32)).cuda()),
    }
    cam_m = torch.from_numpy(np.asarray(ref["cam_m"], dtype=np.float32)).cuda()
    cam_c = torch.from_numpy(np.asarray(ref["cam_c"], dtype=np.float32)).cuda()
    seg_colors = np.asarray(ref["seg_colors"], dtype=np.float32)
    mask_cache: dict[tuple[int, int], torch.Tensor | None] = {}

    def get_mask(t: int, cam_id: int, fn: str, cam) -> torch.Tensor | None:
        key = (t, cam_id)
        if key in mask_cache:
            return mask_cache[key]
        if args.mask_source == "file":
            mask_cache[key] = maybe_load_mask(mask_data_dir, fn)
        elif args.mask_source == "rendered_reference":
            seg_rendervar = make_rendervar(ref, ref["means3D"][t], t, static, colors_override=seg_colors)
            seg_render = render_raw(cam, seg_rendervar)
            mask_cache[key] = seg_render[0] > seg_render[2]
            del seg_rendervar, seg_render
        else:
            mask_cache[key] = None
        return mask_cache[key]

    rows = []
    per_frame_rows = []
    for label, cand_path in args.variants:
        if cand_path == "reference":
            means = ref["means3D"]
            candidate_bytes = int(Path(args.reference).stat().st_size)
        else:
            cand_path_obj = Path(cand_path)
            if cand_path_obj.suffix == ".aprdz":
                means = decode_bitstream(cand_path_obj)
            else:
                cand = np.load(cand_path_obj, allow_pickle=True)
                if "means3D" in cand.files:
                    means = cand["means3D"]
                elif "stride_values" in cand.files:
                    means = decode_payload_means(cand)
                else:
                    raise SystemExit(f"{cand_path} has neither means3D, q16 payload keys, nor APRDZ suffix")
            candidate_bytes = int(cand_path_obj.stat().st_size)

        sampling_groups = temporal_sampling_groups(
            cand_path, len(metadata["fn"]), int(np.asarray(means).shape[1])
        )

        metrics = []
        print(f"===== render/eval {label} =====", flush=True)
        for t in frame_ids:
            rendervar = make_rendervar(ref, means[t], t, static)
            camera_ids = range(len(metadata["fn"][t])) if requested_cameras is None else requested_cameras
            for cam_id in camera_ids:
                if cam_id < 0 or cam_id >= len(metadata["fn"][t]):
                    raise SystemExit(f"camera index {cam_id} invalid for frame {t}")
                fn = metadata["fn"][t][cam_id]
                cam = setup_camera(
                    metadata["w"],
                    metadata["h"],
                    metadata["k"][t][cam_id],
                    metadata["w2c"][t][cam_id],
                    near=1.0,
                    far=100.0,
                )
                pred = render_image(cam, rendervar, cam_m, cam_c, cam_id)
                target = load_image(data_dir / "ims" / fn)
                fg = get_mask(t, cam_id, fn, cam)
                item = {
                    "psnr": psnr(pred, target),
                    "ssim": float(calc_ssim(pred, target).detach().cpu()),
                }
                lp = lpips_distance(lpips_model, pred, target)
                if lp is not None:
                    item["lpips"] = lp
                if fg is not None:
                    item["fg_psnr"] = psnr(pred, target, fg)
                    item["bg_psnr"] = psnr(pred, target, ~fg)
                metrics.append(item)
                per_frame_rows.append(
                    {
                        "label": label,
                        "candidate": cand_path,
                        "candidate_bytes": candidate_bytes,
                        "rate_scope": "trajectory_payload" if cand_path != "reference" else "reference_asset",
                        "frame": int(t),
                        "camera_index": int(cam_id),
                        "image_file": str(fn),
                        **coded_frame_status(sampling_groups, t, len(metadata["fn"])),
                        **item,
                    }
                )
            del rendervar
            torch.cuda.empty_cache()

        row = {
            "label": label,
            "candidate": cand_path,
            "candidate_bytes": candidate_bytes,
            "split": args.split,
            "frame_stride": args.frame_stride,
            "frame_offset": frame_offset,
            "frames": len(frame_ids),
            "images": len(metrics),
        }
        for key in ("psnr", "ssim", "lpips", "fg_psnr", "bg_psnr"):
            vals = np.array([m[key] for m in metrics if key in m], dtype=np.float64)
            row[key] = None if len(vals) == 0 else float(vals.mean())
            row[f"{key}_std"] = None if len(vals) == 0 else float(vals.std())
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2))
    if args.per_frame_json:
        per_frame_path = Path(args.per_frame_json)
        per_frame_path.parent.mkdir(parents=True, exist_ok=True)
        per_frame_path.write_text(json.dumps(per_frame_rows, indent=2))

    lines = [
        "# Panoptic Render Image Metrics",
        "",
        f"seq: `{args.seq}`",
        f"split: `{args.split}`, frame_stride: `{args.frame_stride}`, frame_offset: `{frame_offset}`, frames: `{len(frame_ids)}`",
        f"mask_source: `{args.mask_source}`, lpips: `{args.lpips and lpips_model is not None}`",
        "",
        "| variant | images | PSNR | SSIM | LPIPS | fg PSNR | bg PSNR | artifact MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(key: str, digits: int = 3) -> str:
            val = row.get(key)
            return "nan" if val is None else f"{val:.{digits}f}"

        lines.append(
            f"| {row['label']} | {row['images']} | {fmt('psnr')} | {fmt('ssim', 4)} | "
            f"{fmt('lpips', 4)} | "
            f"{fmt('fg_psnr')} | {fmt('bg_psnr')} | {row['candidate_bytes'] / 1e6:.2f} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(out_md.read_text(), flush=True)
    print(f"WROTE {out_json}", flush=True)
    print(f"WROTE {out_md}", flush=True)
    if args.per_frame_json:
        print(f"WROTE {args.per_frame_json}", flush=True)
    print("PANOPTIC_RENDER_EVAL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
