"""Counted-model construction and camera-metadata render smoke for H007."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from gsplat.compression.h007_path_contract import deterministic_knn_indices
from gsplat.compression.h007_sequence_container import (
    expected_gifstream_nets_keys,
    expected_gifstream_reference_nets_keys,
)
from gsplat.compression_simulation.ops import fake_quantize_factors
from gsplat.rendering import rasterization, view_to_visible_anchors


DECODER_NAMES = ("mlp_opacity", "mlp_cov", "mlp_color", "mlp_motion")


class CountedCameraEmbedding(torch.nn.Module):
    def __init__(self, count: int, embed_dim: int) -> None:
        super().__init__()
        if count <= 0 or embed_dim <= 0:
            raise ValueError("appearance embedding count/dimension must be positive")
        self.embed_dim = int(embed_dim)
        self.embeds = torch.nn.Embedding(int(count), int(embed_dim))

    def forward(self, ids: Tensor) -> Tensor:
        if ids.ndim != 1:
            raise ValueError("appearance camera IDs must be one-dimensional")
        if torch.any(ids < 0):
            if not torch.all(ids < 0):
                raise ValueError("mixed negative/nonnegative appearance IDs are forbidden")
            return torch.zeros((ids.shape[0], self.embed_dim), device=ids.device)
        return self.embeds(ids)


def build_decoder_modules(config: Mapping[str, Any], device: torch.device) -> torch.nn.ModuleDict:
    feature_dim = int(config["anchor_feature_dim"])
    c_perframe = int(config["c_perframe"])
    n_offsets = int(config["n_offsets"])
    time_dim = int(config["time_dim"])
    view_dim = 3 if bool(config["view_adaptive"]) else 0
    app_dim = int(config["app_embed_dim"]) if bool(config["app_opt"]) else 0
    opacity_dist_dim = 1 if bool(config["add_opacity_dist"]) else 0
    cov_dist_dim = 1 if bool(config["add_cov_dist"]) else 0
    color_dist_dim = 1 if bool(config["add_color_dist"]) else 0
    if feature_dim <= 0 or c_perframe <= 0 or n_offsets <= 0 or time_dim <= 0:
        raise ValueError("decoder dimensions must be positive")
    if time_dim % 2:
        raise ValueError("time positional-embedding dimension must be even")

    modules = torch.nn.ModuleDict(
        {
            "mlp_opacity": torch.nn.Sequential(
                torch.nn.Linear(
                    feature_dim + view_dim + opacity_dist_dim + c_perframe,
                    feature_dim,
                ),
                torch.nn.ReLU(True),
                torch.nn.Linear(feature_dim, n_offsets),
                torch.nn.Tanh(),
            ),
            "mlp_cov": torch.nn.Sequential(
                torch.nn.Linear(
                    feature_dim + view_dim + cov_dist_dim + c_perframe,
                    feature_dim,
                ),
                torch.nn.ReLU(True),
                torch.nn.Linear(feature_dim, 7 * n_offsets),
            ),
            "mlp_color": torch.nn.Sequential(
                torch.nn.Linear(
                    feature_dim + view_dim + color_dist_dim + app_dim + c_perframe,
                    feature_dim,
                ),
                torch.nn.ReLU(True),
                torch.nn.Linear(feature_dim, 3 * n_offsets),
                torch.nn.Sigmoid(),
            ),
            "mlp_motion": torch.nn.Sequential(
                torch.nn.Linear(feature_dim + time_dim + c_perframe, feature_dim),
                torch.nn.ReLU(True),
                torch.nn.Linear(feature_dim, 7),
            ),
        }
    ).to(device)
    return modules


def _validate_finite_state(state: Mapping[str, Any], label: str) -> None:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{label} state is empty or malformed")
    for name, value in state.items():
        if not isinstance(value, Tensor):
            raise ValueError(f"{label} state contains non-tensor member: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{label} state contains nonfinite tensor: {name}")


def instantiate_counted_models(
    nets: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
    *,
    reference_nets: bool = False,
) -> Tuple[torch.nn.ModuleDict, Optional[CountedCameraEmbedding], Dict[str, Any]]:
    expected_nets = (
        expected_gifstream_reference_nets_keys(bool(config.get("app_opt")))
        if reference_nets
        else expected_gifstream_nets_keys(bool(config.get("app_opt")))
    )
    if set(nets) != expected_nets:
        raise ValueError("counted nets.pt top-level model dictionary is not exact")
    scaling = nets["scaling"]
    expected_scaling = {
        "anchors",
        "quats",
        "scales",
        "opacities",
        "anchor_features",
        "offsets",
        "factors",
        "time_features",
    }
    if not isinstance(scaling, Mapping) or set(scaling) != expected_scaling:
        raise ValueError("counted scaling keys are incomplete or unexpected")
    for name, value in scaling.items():
        if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
            raise ValueError(f"counted scaling is invalid: {name}")
    _validate_finite_state(nets["decoders"], "decoder")
    decoders = build_decoder_modules(config, device)
    decoders.load_state_dict(nets["decoders"], strict=True)
    decoders.eval()

    app_module: Optional[CountedCameraEmbedding] = None
    if bool(config["app_opt"]):
        if "app_module" not in nets:
            raise ValueError("app_opt container is missing counted appearance state")
        _validate_finite_state(nets["app_module"], "appearance")
        app_module = CountedCameraEmbedding(
            int(config["appearance_embedding_count"]), int(config["app_embed_dim"])
        ).to(device)
        app_module.load_state_dict(nets["app_module"], strict=True)
        app_module.eval()
    elif "app_module" in nets:
        raise ValueError("non-app_opt container unexpectedly contains appearance state")

    for name, module in decoders.items():
        for parameter in module.parameters():
            if not torch.isfinite(parameter).all():
                raise ValueError(f"strict-loaded decoder is nonfinite: {name}")
    return decoders, app_module, {
        "decoder_names": list(decoders.keys()),
        "decoder_parameter_count": int(
            sum(parameter.numel() for parameter in decoders.parameters())
        ),
        "app_module_loaded": app_module is not None,
        "app_parameter_count": int(
            sum(parameter.numel() for parameter in app_module.parameters())
        )
        if app_module is not None
        else 0,
        "strict_load": True,
    }


def _knn_indices(anchors: Tensor, count: int) -> Tensor:
    if count <= 0 or anchors.shape[0] <= count:
        raise ValueError("counted KNN request exceeds decoded anchor population")
    # Rendering-only KNN over decoded anchors: the official codec's lossy
    # anchor round-trip can collapse two anchors onto one position, so
    # duplicates are tolerated here (deterministic candidate-index tie-break;
    # a no-op for unique rows).  Identity-bearing contract paths stay strict.
    return deterministic_knn_indices(
        anchors, int(count), allow_duplicate_rows=True
    )


def counted_knn_indices(
    splats: Mapping[str, Tensor], config: Mapping[str, Any]
) -> Optional[Tensor]:
    """Build the deterministic counted KNN set once for a render/evaluation run."""

    if not bool(config["knn"]):
        return None
    return _knn_indices(splats["anchors"], int(config["n_knn"]))


def _quaternion_to_rotation_matrix(quaternion: Tensor) -> Tensor:
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).view(-1, 3, 3)


def _validate_render_splats(
    splats: Mapping[str, Tensor], config: Mapping[str, Any]
) -> int:
    required = {
        "anchors",
        "quats",
        "scales",
        "opacities",
        "anchor_features",
        "offsets",
        "factors",
        "time_features",
    }
    if set(splats) != required:
        raise ValueError("decoded clean-render splat fields are incomplete or unexpected")
    count = int(splats["anchors"].shape[0])
    expected = {
        "anchors": (count, 3),
        "quats": (count, 4),
        "scales": (count, 6),
        "opacities": (count, 1),
        "anchor_features": (count, int(config["anchor_feature_dim"])),
        "offsets": (count, int(config["n_offsets"]), 3),
        "factors": (count, 4),
        "time_features": (
            count,
            int(config["GOP_size"]),
            int(config["c_perframe"]),
        ),
    }
    if count <= 0 or int(config["GOP_size"]) <= 1:
        raise ValueError("decoded clean-render population/GOP is invalid")
    for name, shape in expected.items():
        value = splats[name]
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"decoded clean-render tensor shape is invalid: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"decoded clean-render tensor is nonfinite: {name}")
    return count


def _render_once(
    splats: Mapping[str, Tensor],
    decoders: torch.nn.ModuleDict,
    app_module: Optional[CountedCameraEmbedding],
    config: Mapping[str, Any],
    camtoworlds: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    frame_index: int,
    knn_indices: Optional[Tensor],
    edit_parent_ids: Optional[Tensor] = None,
    voxel_size: Optional[float] = None,
    return_children: bool = False,
) -> Union[
    Tuple[Tensor, Tensor, Dict[str, int]],
    Tuple[Tensor, Tensor, Dict[str, int], Dict[str, Tensor]],
]:
    packed = bool(config["packed"])
    rasterize_mode = "antialiased" if bool(config["antialiased"]) else "classic"
    visible = view_to_visible_anchors(
        means=splats["anchors"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"][:, :3]),
        viewmats=torch.linalg.inv(camtoworlds),
        Ks=intrinsics,
        width=int(width),
        height=int(height),
        packed=packed,
        rasterize_mode=rasterize_mode,
        camera_model=str(config["camera_model"]),
    )
    if visible.shape != (splats["anchors"].shape[0],) or not visible.any():
        raise ValueError("counted camera sees no decoded anchors")

    feature = splats["anchor_features"]
    time_feature = splats["time_features"][:, int(frame_index)]
    factors = fake_quantize_factors(splats["factors"], q_aware=False)
    selected_feature = feature[visible]
    selected_anchor = splats["anchors"][visible]
    selected_offset = splats["offsets"][visible]
    selected_scale = torch.exp(splats["scales"][visible])
    selected_time = time_feature[visible]
    selected_factor = factors[visible]

    cam_pos = camtoworlds[:, :3, 3]
    view = selected_anchor - cam_pos
    view = view / view.norm(dim=1, keepdim=True).clamp_min(1e-12)
    view_feature = (
        torch.cat([selected_feature, view], dim=1)
        if bool(config["view_adaptive"])
        else selected_feature
    )
    time_input = torch.cat(
        [view_feature, selected_time * selected_factor[:, 0, None]], dim=1
    )

    if bool(config["knn"]):
        if knn_indices is None:
            raise ValueError("counted KNN indices were not constructed")
        selected_neighbors = knn_indices[visible].reshape(-1)
        k = int(config["n_knn"])
        knn_feature = feature[selected_neighbors].reshape(-1, k, feature.shape[-1]).mean(1)
        knn_time = (
            time_feature * factors[:, 0, None]
        )[selected_neighbors].reshape(-1, k, time_feature.shape[-1]).mean(1)
        adaptive = selected_factor[:, 2, None] * torch.cat(
            [selected_feature, selected_time * selected_factor[:, 0, None]], dim=1
        ) + (1.0 - selected_factor[:, 2, None]) * torch.cat(
            [knn_feature, knn_time], dim=1
        )
    else:
        adaptive = torch.cat(
            [selected_feature, selected_time * selected_factor[:, 0, None]], dim=1
        )
    time_value = float(frame_index) / float(int(config["GOP_size"]) - 1)
    unit = torch.ones((1,), dtype=torch.float32, device=selected_anchor.device)
    embedding = torch.cat(
        [
            torch.sin(float(config["phi"]) ** n * torch.pi * unit * time_value)
            for n in range(int(config["time_dim"]) // 2)
        ]
        + [
            torch.cos(float(config["phi"]) ** n * torch.pi * unit * time_value)
            for n in range(int(config["time_dim"]) // 2)
        ]
    )
    motion_input = torch.cat(
        [adaptive, embedding[None].expand(adaptive.shape[0], -1)], dim=1
    )
    pruning_factor = selected_factor[:, 3, None]
    selected_scale = torch.cat(
        [selected_scale[:, :3], selected_scale[:, 3:] * pruning_factor], dim=1
    )
    opacity = decoders["mlp_opacity"](time_input).reshape(-1, 1)
    opacity = opacity * pruning_factor.expand(-1, int(config["n_offsets"])).reshape(-1, 1)
    color_input = time_input
    if app_module is not None:
        app_ids = torch.full((1,), -1, dtype=torch.long, device=selected_anchor.device)
        app = app_module(app_ids).expand(time_input.shape[0], -1)
        color_input = torch.cat([time_input, app], dim=1)
    colors = decoders["mlp_color"](color_input).reshape(-1, 3)
    scale_rot = decoders["mlp_cov"](time_input).reshape(-1, 7)
    motion = decoders["mlp_motion"](motion_input) * selected_factor[:, 1, None]
    tensors = {"opacity": opacity, "colors": colors, "scale_rot": scale_rot, "motion": motion}
    for name, value in tensors.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"clean render decoder output is nonfinite: {name}")

    selected_anchor = selected_anchor + motion[:, -7:-4]
    anchor_quat = torch.nn.functional.normalize(
        0.1 * motion[:, -4:] + motion.new_tensor([[1.0, 0.0, 0.0, 0.0]]), dim=-1
    )
    rotation = _quaternion_to_rotation_matrix(anchor_quat)
    offset = torch.bmm(
        selected_offset * selected_scale[:, None, :3], rotation.transpose(1, 2)
    ).reshape(-1, 3)
    repeated_scale = selected_scale[:, None, :].expand(
        -1, int(config["n_offsets"]), -1
    ).reshape(-1, 6)
    repeated_anchor = selected_anchor[:, None, :].expand(
        -1, int(config["n_offsets"]), -1
    ).reshape(-1, 3)
    keep = opacity.reshape(-1) > 0
    if not keep.any():
        raise ValueError("counted camera render produced no positive-opacity children")
    means = repeated_anchor[keep] + offset[keep]
    scales = repeated_scale[keep, 3:] * torch.sigmoid(scale_rot[keep, :3])
    quats = torch.nn.functional.normalize(scale_rot[keep, 3:7], dim=-1)
    colors = colors[keep]
    opacities = opacity[keep, 0]
    visible_rows = torch.nonzero(visible, as_tuple=False).reshape(-1)
    parent_rows = (
        visible_rows[:, None]
        .expand(-1, int(config["n_offsets"]))
        .reshape(-1)[keep]
    )
    parent_ids = None
    if edit_parent_ids is not None or return_children:
        if voxel_size is None or not math.isfinite(float(voxel_size)) or float(voxel_size) <= 0:
            raise ValueError("H-DOWN child decoding requires a positive voxel size")
        parent_ids = torch.round(splats["anchors"][parent_rows] / float(voxel_size)).to(
            torch.int64
        )
        if torch.unique(parent_ids, dim=0).shape[0] > splats["anchors"].shape[0]:
            raise AssertionError("child parent-ID population exceeds decoded anchors")
    edited_child_count = 0
    if edit_parent_ids is not None:
        edit_parent_ids = edit_parent_ids.to(device=parent_ids.device, dtype=torch.int64)
        if edit_parent_ids.ndim != 2 or edit_parent_ids.shape[1] != 3:
            raise ValueError("H-DOWN edit parent IDs must have shape [J,3]")
        if torch.unique(edit_parent_ids, dim=0).shape[0] != edit_parent_ids.shape[0]:
            raise ValueError("H-DOWN edit parent IDs are duplicated")
        edited = torch.all(
            parent_ids[:, None, :] == edit_parent_ids[None, :, :], dim=-1
        ).any(dim=1)
        edited_child_count = int(edited.sum().item())
        target = colors.new_tensor([1.0, 0.0, 1.0])
        edited_colors = (0.25 * colors + 0.75 * target).clamp(0.0, 1.0)
        colors = torch.where(edited[:, None], edited_colors, colors)
    render, alpha, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.linalg.inv(camtoworlds),
        Ks=intrinsics,
        width=int(width),
        height=int(height),
        packed=packed,
        absgrad=False,
        sparse_grad=False,
        rasterize_mode=rasterize_mode,
        distributed=False,
        camera_model=str(config["camera_model"]),
    )
    if render.shape != (1, int(height), int(width), 3):
        raise ValueError("counted camera render has an unexpected shape")
    if alpha.shape[:3] != (1, int(height), int(width)):
        raise ValueError("counted camera alpha has an unexpected shape")
    if not torch.isfinite(render).all() or not torch.isfinite(alpha).all():
        raise ValueError("counted camera render/alpha is nonfinite")
    audit = {
        "visible_anchor_count": int(visible.sum().item()),
        "rendered_child_count": int(keep.sum().item()),
        "edited_child_count": edited_child_count,
    }
    if return_children:
        return render, alpha, audit, {
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": opacities,
            "colors": colors,
            "parent_ids": parent_ids,
        }
    return render, alpha, audit


@torch.no_grad()
def decode_anchor_paths(
    splats: Mapping[str, Tensor],
    decoders: torch.nn.ModuleDict,
    config: Mapping[str, Any],
    knn_indices: Optional[Tensor] = None,
) -> Tensor:
    """Decode the exact view-independent parent-anchor path used by AP scoring."""

    _validate_render_splats(splats, config)
    anchors = splats["anchors"]
    features = splats["anchor_features"]
    time_features = splats["time_features"]
    factors = fake_quantize_factors(splats["factors"], q_aware=False)
    if bool(config["knn"]):
        if knn_indices is None:
            knn_indices = counted_knn_indices(splats, config)
        flat = knn_indices.reshape(-1)
        knn_features = features[flat].reshape(
            -1, int(config["n_knn"]), features.shape[-1]
        ).mean(1)
    paths = []
    unit = torch.ones((1,), dtype=torch.float32, device=anchors.device)
    for frame in range(int(config["GOP_size"])):
        frame_features = time_features[:, frame]
        if bool(config["knn"]):
            knn_time = (
                time_features[:, frame] * factors[:, 0, None]
            )[flat].reshape(
                -1, int(config["n_knn"]), int(config["c_perframe"])
            ).mean(1)
            adaptive = factors[:, 2, None] * torch.cat(
                [features, frame_features * factors[:, 0, None]], dim=-1
            ) + (1.0 - factors[:, 2, None]) * torch.cat(
                [knn_features, knn_time], dim=-1
            )
        else:
            adaptive = torch.cat(
                [features, frame_features * factors[:, 0, None]], dim=-1
            )
        time_value = float(frame) / float(int(config["GOP_size"]) - 1)
        embedding = torch.cat(
            [
                torch.sin(float(config["phi"]) ** n * torch.pi * unit * time_value)
                for n in range(int(config["time_dim"]) // 2)
            ]
            + [
                torch.cos(float(config["phi"]) ** n * torch.pi * unit * time_value)
                for n in range(int(config["time_dim"]) // 2)
            ]
        )
        motion = decoders["mlp_motion"](
            torch.cat([adaptive, embedding[None].expand(adaptive.shape[0], -1)], dim=1)
        ) * factors[:, 1, None]
        paths.append(anchors + motion[:, :3])
    output = torch.stack(paths, dim=1)
    if output.shape != (anchors.shape[0], int(config["GOP_size"]), 3):
        raise ValueError("H-DOWN anchor path decoder returned an invalid shape")
    if not torch.isfinite(output).all():
        raise ValueError("H-DOWN anchor paths are nonfinite")
    return output


def render_hdown_frame(
    splats: Mapping[str, Tensor],
    decoders: torch.nn.ModuleDict,
    app_module: Optional[CountedCameraEmbedding],
    config: Mapping[str, Any],
    camtoworlds: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    frame_index: int,
    edit_parent_ids: Optional[Tensor] = None,
    knn_indices: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Dict[str, int]]:
    """Render one actual decoded frame, optionally applying the frozen edit."""

    _validate_render_splats(splats, config)
    neighbors = knn_indices if knn_indices is not None else counted_knn_indices(splats, config)
    return _render_once(
        splats,
        decoders,
        app_module,
        config,
        camtoworlds,
        intrinsics,
        width,
        height,
        frame_index,
        neighbors,
        edit_parent_ids=edit_parent_ids,
        voxel_size=float(config["voxel_size"]),
    )


def alpha_parent_contributions(
    splats: Mapping[str, Tensor],
    decoders: torch.nn.ModuleDict,
    app_module: Optional[CountedCameraEmbedding],
    config: Mapping[str, Any],
    camtoworlds: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    frame_index: int,
    click_xy: Tuple[float, float],
    knn_indices: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
    """Return exact 7x7 alpha-compositing mass grouped by canonical parent ID.

    The derivative of the patch color sum with respect to a per-child probe
    color is exactly that child's front-to-back compositing weight ``T*alpha``.
    This uses the real gsplat rasterizer and its actual depth ordering.
    """

    _validate_render_splats(splats, config)
    neighbors = knn_indices if knn_indices is not None else counted_knn_indices(splats, config)
    with torch.no_grad():
        _, _, child_audit, children = _render_once(
            splats,
            decoders,
            app_module,
            config,
            camtoworlds,
            intrinsics,
            width,
            height,
            frame_index,
            neighbors,
            voxel_size=float(config["voxel_size"]),
            return_children=True,
        )
    center_x = int(math.floor(float(click_xy[0]) + 0.5))
    center_y = int(math.floor(float(click_xy[1]) + 0.5))
    if center_x - 3 < 0 or center_x + 3 >= int(width) or center_y - 3 < 0 or center_y + 3 >= int(height):
        raise ValueError("H-DOWN click does not admit a full 7x7 patch")
    probe = torch.zeros(
        (children["means"].shape[0], 1),
        dtype=children["means"].dtype,
        device=children["means"].device,
        requires_grad=True,
    )
    with torch.enable_grad():
        rendered, _, _ = rasterization(
            means=children["means"].detach(),
            quats=children["quats"].detach(),
            scales=children["scales"].detach(),
            opacities=children["opacities"].detach(),
            colors=probe,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=intrinsics,
            width=int(width),
            height=int(height),
            packed=bool(config["packed"]),
            absgrad=False,
            sparse_grad=False,
            rasterize_mode=("antialiased" if bool(config["antialiased"]) else "classic"),
            distributed=False,
            camera_model=str(config["camera_model"]),
        )
        patch_sum = rendered[
            0, center_y - 3 : center_y + 4, center_x - 3 : center_x + 4, 0
        ].sum()
        child_mass = torch.autograd.grad(patch_sum, probe, create_graph=False)[0][
            :, 0
        ]
    if not torch.isfinite(child_mass).all():
        raise ValueError("H-DOWN child alpha contributions are nonfinite")
    child_mass = child_mass.clamp_min(0)
    parent_ids, inverse = torch.unique(
        children["parent_ids"], dim=0, sorted=True, return_inverse=True
    )
    parent_mass = torch.zeros(
        parent_ids.shape[0], dtype=child_mass.dtype, device=child_mass.device
    )
    parent_mass.scatter_add_(0, inverse, child_mass)
    positive = parent_mass > 0
    parent_ids = parent_ids[positive]
    parent_mass = parent_mass[positive]
    if parent_ids.shape[0] == 0 or float(parent_mass.sum().item()) <= 0:
        raise ValueError("H-DOWN 7x7 patch has no positive alpha contribution")
    return parent_ids, parent_mass, {
        "schema": "h007.hdown_alpha_lift.v1",
        "patch_size": [7, 7],
        "patch_center_xy": [center_x, center_y],
        "patch_center_rounding": "floor_coordinate_plus_0.5",
        "positive_parent_count": int(parent_ids.shape[0]),
        "positive_child_count": int((child_mass > 0).sum().item()),
        "positive_alpha_mass": float(parent_mass.sum().item()),
        "contribution_definition": "autograd_d_patch_color_sum_d_child_probe_color_equals_T_alpha",
        **child_audit,
    }


@torch.no_grad()
def counted_camera_render_benchmark(
    splats: Mapping[str, Tensor],
    decoders: torch.nn.ModuleDict,
    app_module: Optional[CountedCameraEmbedding],
    config: Mapping[str, Any],
    camera_arrays: Mapping[str, np.ndarray],
    warmup_renders: int,
    timed_renders: int,
) -> Dict[str, Any]:
    if warmup_renders <= 0 or timed_renders <= 0:
        raise ValueError("clean-process warm/timed render counts must be positive")
    anchor_count = _validate_render_splats(splats, config)
    if bool(config["knn"]) and anchor_count <= int(config["n_knn"]):
        raise ValueError("decoded anchor population is too small for counted KNN")
    pose_index = int(config["warm_camera_pose_index"])
    frame_index = int(config["warm_frame_index"])
    poses = camera_arrays["camtoworlds"]
    camera_ids = camera_arrays["camera_ids"]
    keys = camera_arrays["camera_keys"]
    if pose_index < 0 or pose_index >= poses.shape[0]:
        raise ValueError("counted warm camera pose index is out of range")
    if frame_index < 0 or frame_index >= int(config["GOP_size"]):
        raise ValueError("counted warm frame index is out of range")
    key = int(camera_ids[pose_index])
    locations = np.flatnonzero(keys == key)
    if locations.size != 1:
        raise ValueError("counted warm camera key is absent or duplicated")
    meta_index = int(locations[0])
    width, height = [int(v) for v in camera_arrays["image_sizes"][meta_index]]
    device = splats["anchors"].device
    pose = torch.from_numpy(np.asarray(poses[pose_index], dtype=np.float32)).to(device)
    if pose.shape == (3, 4):
        pose = torch.cat([pose, pose.new_tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
    pose = pose[None]
    intrinsic = torch.from_numpy(
        np.asarray(camera_arrays["intrinsics"][meta_index], dtype=np.float32)
    ).to(device)[None]
    knn_indices = (
        _knn_indices(splats["anchors"], int(config["n_knn"]))
        if bool(config["knn"])
        else None
    )

    last_render = last_alpha = None
    shape_audit: Dict[str, int] = {}
    for _ in range(int(warmup_renders)):
        last_render, last_alpha, shape_audit = _render_once(
            splats,
            decoders,
            app_module,
            config,
            pose,
            intrinsic,
            width,
            height,
            frame_index,
            knn_indices,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(int(timed_renders)):
        last_render, last_alpha, shape_audit = _render_once(
            splats,
            decoders,
            app_module,
            config,
            pose,
            intrinsic,
            width,
            height,
            frame_index,
            knn_indices,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if not math.isfinite(seconds) or seconds <= 0 or last_render is None or last_alpha is None:
        raise ValueError("clean-process render benchmark timing failed")
    return {
        "schema": "h007.clean_counted_camera_render.v1",
        "warmup_renders": int(warmup_renders),
        "timed_renders": int(timed_renders),
        "seconds": float(seconds),
        "fps": float(timed_renders / seconds),
        "camera_pose_index": pose_index,
        "camera_key": key,
        "frame_index": frame_index,
        "width": width,
        "height": height,
        "source_pixels_read": 0,
        "camera_metadata_source": "counted_archive_only",
        "render_shape": list(last_render.shape),
        "alpha_shape": list(last_alpha.shape),
        **shape_audit,
    }
