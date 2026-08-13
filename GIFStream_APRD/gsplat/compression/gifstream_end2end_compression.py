import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from gsplat.compression.stream_helper import encode_x, decode_x, filesize

from gsplat.compression.outlier_filter import filter_splats
from gsplat.compression.sort import sort_splats, sort_anchors
from gsplat.compression.ap_gifstream import (
    AP_META_SCHEMA,
    canonical_voxel_ids,
    load_aligned_score_artifact,
    read_ap_mask,
    read_identity_corrections,
    recompute_sorted_codec_invariants,
    tensor_mapping_sha256,
    variant_metadata,
    variant_spec,
    write_ap_mask,
    write_identity_corrections,
)
from gsplat.compression.h007_runtime_provenance import verify_runtime_provenance
from gsplat.compression.h007_path_contract import (
    ANCHOR_FEATURE_BACKGROUND_MULTIPLIER,
    ANCHOR_FEATURE_PROTECTED_MULTIPLIER,
    FACTOR_BACKGROUND_MULTIPLIER,
    FACTOR_PROTECTED_MULTIPLIER,
    PATH_CONTRACT_SCHEMA,
    build_path_input_precision_mask,
    normalize_factor_semantics,
    reconstruct_ap_factors,
    retained_knn_graph_sha256,
    deterministic_knn_indices,
    ste_row_quantize,
)
from gsplat.utils import inverse_log_transform, log_transform
import math


def _apply_factor_mask(
    factors: Tensor, column: int, target: Tensor, activation_value: float
) -> int:
    if factors.ndim != 2 or column < 0 or column >= factors.shape[1]:
        raise ValueError("factor tensor/column is malformed")
    target = target.to(device=factors.device, dtype=torch.bool)
    if target.shape != (factors.shape[0],):
        raise ValueError(f"factor column {column} target mask has an invalid shape")
    current = factors[:, column] > 0
    promoted = target & ~current
    demoted = ~target & current
    if not math.isfinite(float(activation_value)) or float(activation_value) <= 0:
        raise ValueError(f"invalid frozen activation value for factor column {column}")
    if promoted.any():
        factors[promoted, column] = float(activation_value)
    factors[demoted, column] = 0.0
    if not torch.equal(factors[:, column] > 0, target):
        raise AssertionError(f"factor column {column} does not match AP mask")
    return int(promoted.sum().item() + demoted.sum().item())


def _tensor_equal_device_agnostic(left: Tensor, right: Tensor) -> bool:
    """Compare counted tensors without requiring their storage devices to match."""

    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.device != right.device:
        left = left.to(device=right.device)
    return torch.equal(left, right)


def _validate_counted_retained_identity(
    retained_ids: Tensor, decoded_row_count: int
) -> None:
    """Validate the counted identity authority without re-voxelizing lossy anchors.

    The retained-ID sidecar is written before anchor quantization and is part of
    the counted container.  Decoded anchor coordinates are therefore attributes
    of those identities, not a second identity authority.
    """

    if retained_ids.dtype != torch.int64 or retained_ids.ndim != 2 or retained_ids.shape[1] != 3:
        raise ValueError("counted retained canonical IDs are malformed")
    if int(retained_ids.shape[0]) != int(decoded_row_count):
        raise ValueError("counted retained canonical ID count differs from decoded rows")
    if int(torch.unique(retained_ids, dim=0).shape[0]) != int(retained_ids.shape[0]):
        raise ValueError("counted retained canonical IDs are not unique")


@dataclass
class GIFStreamEnd2endCompression:
    """Uses quantization and sorting to compress splats into PNG files and uses
    K-means clustering to compress the spherical harmonic coefficents.

    .. warning::
        This class requires the `imageio <https://pypi.org/project/imageio/>`_,
        `plas <https://github.com/fraunhoferhhi/PLAS.git>`_
        and `torchpq <https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install>`_ packages to be installed.

    .. warning::
        This class might throw away a few lowest opacities splats if the number of
        splats is not a square number.

    .. note::
        The splats parameters are expected to be pre-activation values. It expects
        the following fields in the splats dictionary: "means", "scales", "quats",
        "opacities", "sh0", "shN". More fields can be added to the dictionary, but
        they will only be compressed using NPZ compression.

    References:
        - `Compact 3D Scene Representation via Self-Organizing Gaussian Grids <https://arxiv.org/abs/2312.13299>`_
        - `Making Gaussian Splats more smaller <https://aras-p.info/blog/2023/09/27/Making-Gaussian-Splats-more-smaller/>`_

    Args:
        use_sort (bool, optional): Whether to sort splats before compression. Defaults to True.
        verbose (bool, optional): Whether to print verbose information. Default to True.
    """

    use_sort: bool = True
    verbose: bool = True
    ap_config: Optional[Dict[str, Any]] = None
    decoded_canonical_ids: Optional[Tensor] = None

    def _get_compress_fn(self, param_name: str) -> Callable:
        compress_fn_map = {
            "anchors": _compress_png_16bit,
            "scales": _compress_end2end,
            "offsets": _compress_end2end,
            "anchor_features": _compress_end2end_ar,
            "factors": _compress_end2end,
            "time_features": _compress_end2end_ar,
        }
        if param_name in compress_fn_map:
            return compress_fn_map[param_name]
        else:
            return _compress_npz

    def _get_decompress_fn(self, param_name: str) -> Callable:
        decompress_fn_map = {
            "anchors": _decompress_png_16bit,
            "scales": _decompress_end2end,
            "offsets": _decompress_end2end,
            "anchor_features": _decompress_end2end_ar,
            "factors": _decompress_end2end,
            "time_features": _decompress_end2end_ar,
        }
        if param_name in decompress_fn_map:
            return decompress_fn_map[param_name]
        else:
            return _decompress_npz

    def compress(
        self,
        compress_dir: str,
        splats: Dict[str, Tensor],
        entropy_models: Dict[str, Module] = None,
        c_channel: int = 0,
        p_channel: int = 0,
        scaling = None,
        voxel_size = 0.01,
        raw_time_features: Optional[Tensor] = None,
        raw_anchor_features: Optional[Tensor] = None,
        raw_factors: Optional[Tensor] = None,
    ) -> None:
        """Run compression

        Args:
            compress_dir (str): directory to save compressed files
            splats (Dict[str, Tensor]): Gaussian splats to compress
        """
        
        # quantization scaling
        if scaling is None:
            scaling = {
                "anchors": None,
                "scales": 0.01,
                "quats": None,
                "opacities": None,
                "anchor_features": 1,
                "offsets": 0.01,
                "factors": 1/16,
                "time_features": 1,
            }

        os.makedirs(compress_dir, exist_ok=True)
        # The official implementation mutates the caller's dictionary.  Clone
        # every tensor so an allocation candidate cannot alter the reference
        # state used by a later candidate or by the round-trip audit.
        splats = {name: value.detach().clone() for name, value in splats.items()}

        ap_cfg = dict(self.ap_config or {})
        ap_variant = str(ap_cfg.get("variant", "official"))
        ap_spec = variant_spec(ap_variant)
        ap_enabled = ap_variant != "official"
        runtime_provenance = None
        if ap_enabled:
            provenance_required = {
                "provenance_manifest_path",
                "provenance_manifest_sha256",
                "repo_root",
                "runtime_provenance",
                "scene",
                "n_knn",
            }
            missing = sorted(provenance_required - set(ap_cfg))
            if missing:
                raise ValueError(f"AP codec lacks runtime provenance fields: {missing}")
            runtime_provenance = verify_runtime_provenance(
                Path(str(ap_cfg["provenance_manifest_path"])),
                Path(str(ap_cfg["repo_root"])),
                str(ap_cfg["provenance_manifest_sha256"]),
                expected_container_receipt=ap_cfg["runtime_provenance"],
            )
        if raw_time_features is not None:
            if raw_time_features.shape != splats["time_features"].shape:
                raise ValueError("raw_time_features shape mismatch")
            # Only a two-class quantized AP stream may recover precision finer
            # than the official base-Q simulation.  Other variants continue to
            # consume the official already-simulated tensor byte-for-byte.
            if ap_enabled and ap_spec.quant:
                splats["time_features"] = raw_time_features.detach().clone()
        raw_factor_logits = None
        if ap_enabled and ap_spec.quant:
            if raw_anchor_features is None or raw_factors is None:
                raise ValueError(
                    "AP-v6 compression requires raw anchor features and raw factor logits"
                )
            if raw_anchor_features.shape != splats["anchor_features"].shape:
                raise ValueError("raw_anchor_features shape mismatch")
            if raw_factors.shape != splats["factors"].shape:
                raise ValueError("raw_factors shape mismatch")
            splats["anchor_features"] = raw_anchor_features.detach().clone()
            raw_factor_logits = raw_factors.detach().clone()
        compression_seed = ap_cfg.get("compression_seed")
        if ap_enabled and compression_seed is None:
            raise ValueError("AP-GIFStream requires a frozen compression_seed")

        score = eligible = estimated_bytes = frozen_allocation = canonical_ids = None
        score_audit = allocation_audit = None
        if ap_enabled:
            score_path = ap_cfg.get("score_path")
            if not score_path:
                raise ValueError("AP-GIFStream requires a frozen score_path")
            score, eligible, estimated_bytes, frozen_allocation, score_audit = load_aligned_score_artifact(
                str(score_path),
                splats["anchors"],
                voxel_size,
                ap_spec.ranking,
                int(ap_cfg.get("random_seed", 11)),
                runtime_provenance,
                str(ap_cfg["scene"]),
                expected_variant=ap_variant,
            )
            for config_name, frozen_name, default in (
                ("protected_fraction", "protected_fraction", 0.05),
                ("q_ap_multiplier", "q_ap_multiplier", 0.5),
                ("q_bg_multiplier", "q_bg_multiplier", 1.25),
            ):
                if float(ap_cfg.get(config_name, default)) != float(score_audit[frozen_name]):
                    raise ValueError(f"codec {config_name} differs from frozen score artifact")
            canonical_ids = canonical_voxel_ids(splats["anchors"], voxel_size)
            if tensor_mapping_sha256(
                entropy_models["time_features"].state_dict()
            ) != score_audit["time_entropy_model_sha256"]:
                raise ValueError("codec temporal entropy model differs from frozen byte estimator")
            if float(scaling["time_features"]) != float(
                score_audit["time_feature_scaling"]
            ):
                raise ValueError("codec temporal scaling differs from frozen byte estimator")

        source_estimated_bytes = estimated_bytes

        # Param-specific preprocessing
        # splats["anchors"] = log_transform(splats["anchors"])
        splats["quats"] = F.normalize(splats["quats"], dim=-1)
        pruning_mask = splats["factors"][:,-1] > 0
        ap_active = ap_class = ap_path_input = None
        if ap_enabled:
            current_retain = pruning_mask.clone()
            current_active = current_retain & (splats["factors"][:, 0] > 0)
            ap_retain = frozen_allocation["ap_retain_mask"]
            ap_active = frozen_allocation["ap_active_mask"]
            ap_class = frozen_allocation["ap_class_mask"]
            if torch.any(ap_active & ~ap_retain):
                raise ValueError("frozen AP active mask escapes whole-retain mask")
            _apply_factor_mask(
                splats["factors"],
                3,
                ap_retain,
                float(frozen_allocation["factor3_activation_value"]),
            )
            pruning_mask = ap_retain
            _apply_factor_mask(
                splats["factors"],
                0,
                ap_active,
                float(frozen_allocation["factor0_activation_value"]),
            )
            if torch.any(ap_class & ~ap_active):
                raise ValueError("frozen protected class contains an inactive row")
            ap_path_input, retained_knn = build_path_input_precision_mask(
                splats["anchors"],
                ap_class,
                ap_retain,
                int(ap_cfg["n_knn"]),
                canonical_ids=canonical_ids,
            )
            retained_rows = torch.nonzero(
                ap_retain, as_tuple=False
            ).flatten()
            path_knn_graph_sha256 = retained_knn_graph_sha256(
                canonical_ids[retained_rows], retained_knn
            )
            allocation_audit = {
                "budget_source": "frozen_score_artifact",
                "official_retain_count": int(
                    frozen_allocation["official_retain_mask"].sum().item()
                ),
                "ap_retain_count": int(ap_retain.sum().item()),
                "official_estimated_time_bytes": int(
                    estimated_bytes[frozen_allocation["official_active_mask"]].sum().item()
                ),
                "ap_estimated_time_bytes": int(estimated_bytes[ap_active].sum().item()),
                "current_vs_frozen_whole_xor": int(
                    torch.logical_xor(
                        current_retain, frozen_allocation["official_retain_mask"]
                    ).sum().item()
                ),
                "current_vs_frozen_temporal_xor": int(
                    torch.logical_xor(
                        current_active, frozen_allocation["official_active_mask"]
                    ).sum().item()
                ),
            }
        for k,v in splats.items():
            splats[k] = v[pruning_mask]
        if raw_factor_logits is not None:
            raw_factor_logits = raw_factor_logits[pruning_mask]
        if ap_enabled:
            ap_active = ap_active[pruning_mask]
            ap_class = ap_class[pruning_mask]
            ap_path_input = ap_path_input[pruning_mask]
            score = score[pruning_mask]
            eligible = eligible[pruning_mask]
            estimated_bytes = estimated_bytes[pruning_mask]
            canonical_ids = canonical_ids[pruning_mask]
            pre_sort_retained_ids = canonical_ids.clone()
            pre_sort_estimated_bytes = estimated_bytes.clone()
            real_mask = torch.ones(
                canonical_ids.shape[0], dtype=torch.bool, device=canonical_ids.device
            )
            padding_mask = ~real_mask

        n_gs = len(splats["anchors"])
        n_sidelen = math.ceil(n_gs**0.5)
        n_crop = n_gs - n_sidelen**2

        if n_crop != 0:
            splats = _crop_n_splats(splats, n_crop)
            if ap_enabled:
                if n_crop > 0:
                    raise AssertionError("ceil-square preprocessing unexpectedly requested cropping")
                pad = -n_crop
                ap_active = torch.cat(
                    [ap_active, torch.zeros(pad, dtype=torch.bool, device=ap_active.device)]
                )
                ap_class = torch.cat(
                    [ap_class, torch.zeros(pad, dtype=torch.bool, device=ap_class.device)]
                )
                ap_path_input = torch.cat(
                    [
                        ap_path_input,
                        torch.zeros(
                            pad, dtype=torch.bool, device=ap_path_input.device
                        ),
                    ]
                )
                if raw_factor_logits is not None:
                    raw_factor_logits = torch.cat(
                        [
                            raw_factor_logits,
                            torch.zeros(
                                (pad, raw_factor_logits.shape[1]),
                                dtype=raw_factor_logits.dtype,
                                device=raw_factor_logits.device,
                            ),
                        ],
                        dim=0,
                    )
                score = torch.cat(
                    [score, torch.full((pad,), -torch.inf, dtype=score.dtype, device=score.device)]
                )
                eligible = torch.cat(
                    [eligible, torch.zeros(pad, dtype=torch.bool, device=eligible.device)]
                )
                estimated_bytes = torch.cat(
                    [estimated_bytes, torch.zeros(pad, dtype=torch.int64, device=estimated_bytes.device)]
                )
                sentinel = torch.zeros((pad, 3), dtype=torch.int64, device=canonical_ids.device)
                sentinel[:, 0] = torch.arange(pad, device=canonical_ids.device, dtype=torch.int64)
                sentinel[:, 0] += torch.iinfo(torch.int64).min
                canonical_ids = torch.cat([canonical_ids, sentinel], dim=0)
                real_mask = torch.cat(
                    [real_mask, torch.zeros(pad, dtype=torch.bool, device=real_mask.device)]
                )
                padding_mask = ~real_mask
            print(
                f"Warning: Number of Gaussians was not square. Removed {n_crop} Gaussians."
            )

        if self.use_sort:
            if ap_enabled:
                splats, sort_indices = sort_anchors(
                    splats,
                    return_indices=True,
                    seed=int(compression_seed),
                )
                ap_active = ap_active[sort_indices]
                ap_class = ap_class[sort_indices]
                ap_path_input = ap_path_input[sort_indices]
                if raw_factor_logits is not None:
                    raw_factor_logits = raw_factor_logits[sort_indices]
                score = score[sort_indices]
                eligible = eligible[sort_indices]
                estimated_bytes = estimated_bytes[sort_indices]
                canonical_ids = canonical_ids[sort_indices]
                real_mask = real_mask[sort_indices]
                padding_mask = padding_mask[sort_indices]
            else:
                splats = sort_anchors(splats)
                sort_indices = None
        else:
            sort_indices = torch.arange(len(splats["anchors"]), device=splats["anchors"].device)

        if ap_enabled:
            if not torch.equal(splats["factors"][:, 0] > 0, ap_active):
                raise AssertionError("sorted factor activation disagrees with frozen AP allocation")
            if torch.any(ap_class & ~ap_path_input):
                raise AssertionError("sorted path-input mask lost a protected row")
            if torch.any(ap_path_input & ~real_mask):
                raise AssertionError("sorted path-input mask includes padding")
            allocation_audit.update(
                recompute_sorted_codec_invariants(
                    official_retain_mask=frozen_allocation["official_retain_mask"],
                    official_active_mask=frozen_allocation["official_active_mask"],
                    ap_retain_mask=frozen_allocation["ap_retain_mask"],
                    expected_ap_active_mask=frozen_allocation["ap_active_mask"],
                    expected_ap_class_mask=frozen_allocation["ap_class_mask"],
                    source_estimated_bytes=source_estimated_bytes,
                    pre_sort_retained_ids=pre_sort_retained_ids,
                    pre_sort_estimated_bytes=pre_sort_estimated_bytes,
                    sort_indices=sort_indices,
                    encoded_ids=canonical_ids,
                    encoded_estimated_bytes=estimated_bytes,
                    encoded_factors=splats["factors"],
                    real_mask=real_mask,
                    padding_mask=padding_mask,
                    ap_class_mask=ap_class,
                )
            )

        choose_idx = splats["factors"][:,0] > 0
        splats["time_features"] = splats["time_features"][choose_idx]
        encoded_anchor_features = torch.round(
            splats["anchor_features"] / scaling["anchor_features"]
        ) * scaling["anchor_features"]
        encoded_factor_reconstruction = None
        if ap_enabled and ap_spec.quant:
            encoded_anchor_features = ste_row_quantize(
                splats["anchor_features"],
                ap_path_input,
                float(scaling["anchor_features"]),
                ANCHOR_FEATURE_PROTECTED_MULTIPLIER,
                ANCHOR_FEATURE_BACKGROUND_MULTIPLIER,
            )
            encoded_factor_reconstruction, _ = reconstruct_ap_factors(
                raw_factor_logits,
                encoded_anchor_features,
                entropy_models["factors"],
                float(scaling["factors"]),
                ap_path_input,
                FACTOR_PROTECTED_MULTIPLIER,
                FACTOR_BACKGROUND_MULTIPLIER,
                real_mask & ap_active,
                real_mask,
                float(frozen_allocation["factor0_activation_value"]),
            )

        meta = {}
        print(entropy_models.keys())
        for param_name in splats.keys():
            compress_fn = self._get_compress_fn(param_name)
            kwargs = {
                "n_sidelen": n_sidelen,
                "verbose": self.verbose,
                "anchor_features": encoded_anchor_features,
                "c_channel": c_channel,
                "p_channel": p_channel,
                "scaling": scaling[param_name],
                "entropy_model": entropy_models[param_name],
                "voxel_size": voxel_size
            }

            if (
                param_name == "anchor_features"
                and ap_enabled
                and ap_spec.quant
            ):
                meta[param_name] = _compress_ap_anchor_features_ar(
                    compress_dir,
                    param_name,
                    splats[param_name],
                    precision_mask=ap_path_input,
                    protected_multiplier=ANCHOR_FEATURE_PROTECTED_MULTIPLIER,
                    background_multiplier=ANCHOR_FEATURE_BACKGROUND_MULTIPLIER,
                    expected_reconstruction=encoded_anchor_features,
                    **kwargs,
                )
            elif param_name == "factors" and ap_enabled and ap_spec.quant:
                meta[param_name] = _compress_ap_factors(
                    compress_dir,
                    param_name,
                    raw_factor_logits,
                    precision_mask=ap_path_input,
                    active_mask=real_mask & ap_active,
                    real_mask=real_mask,
                    factor0_activation_value=float(
                        frozen_allocation["factor0_activation_value"]
                    ),
                    protected_multiplier=FACTOR_PROTECTED_MULTIPLIER,
                    background_multiplier=FACTOR_BACKGROUND_MULTIPLIER,
                    expected_reconstruction=encoded_factor_reconstruction,
                    **kwargs,
                )
            elif param_name == "time_features" and ap_enabled and ap_spec.quant:
                meta[param_name] = _compress_ap_time_features_ar(
                    compress_dir,
                    param_name,
                    splats[param_name],
                    class_mask=ap_path_input[choose_idx],
                    q_ap_multiplier=float(ap_cfg.get("q_ap_multiplier", 0.5)),
                    q_bg_multiplier=float(ap_cfg.get("q_bg_multiplier", 1.25)),
                    **kwargs,
                )
            else:
                meta[param_name] = compress_fn(
                    compress_dir, param_name, splats[param_name], **kwargs
                )

        if ap_enabled:
            mask_meta = write_ap_mask(os.path.join(compress_dir, "ap_class_mask.bin"), ap_class)
            path_input_mask_meta = write_ap_mask(
                os.path.join(compress_dir, "ap_path_input_mask.bin"),
                ap_path_input,
            )
            real_mask_meta = write_ap_mask(
                os.path.join(compress_dir, "ap_real_row_mask.bin"), real_mask
            )
            padding_mask_meta = write_ap_mask(
                os.path.join(compress_dir, "ap_padding_row_mask.bin"), padding_mask
            )
            active_mask_meta = write_ap_mask(
                os.path.join(compress_dir, "ap_active_row_mask.bin"),
                real_mask & (splats["factors"][:, 0] > 0),
            )
            decoded_anchor_rows = _decompress_png_16bit(
                compress_dir, "anchors", meta["anchors"]
            )
            decoded_ids = (
                torch.round(
                    decoded_anchor_rows[real_mask.detach().cpu()] / float(voxel_size)
                )
                .to(torch.int64)
                .numpy()
                .astype(np.int64, copy=False)
            )
            retained_ids = (
                canonical_ids[real_mask]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            )
            identity_path = os.path.join(compress_dir, "ap_identity_corrections.bin")
            identity_meta = write_identity_corrections(
                identity_path, decoded_ids, retained_ids
            )
            if not np.array_equal(
                read_identity_corrections(compress_dir, identity_meta, decoded_ids),
                retained_ids,
            ):
                raise AssertionError("counted identity corrections failed exact encoder replay")
            meta["__ap__"] = {
                "schema": AP_META_SCHEMA,
                "variant": variant_metadata(ap_variant),
                "score": score_audit,
                "allocation": allocation_audit,
                "runtime_provenance": runtime_provenance,
                "compression_seed": int(compression_seed),
                "q_ap_multiplier": float(ap_cfg.get("q_ap_multiplier", 0.5)),
                "q_bg_multiplier": float(ap_cfg.get("q_bg_multiplier", 1.25)),
                "mask": mask_meta,
                "path_input_mask": path_input_mask_meta,
                "path_contract": {
                    "schema": PATH_CONTRACT_SCHEMA,
                    "knn_count": int(ap_cfg["n_knn"]),
                    "knn_rule": (
                        "retained-canonical-radius-complete-distance+lexicographic-id"
                    ),
                    "dependency_rule": "protected-plus-one-hop-retained-knn",
                    "retained_knn_graph_sha256": path_knn_graph_sha256,
                    "canonical_anchor_reconstruction": True,
                    "factor_protected_multiplier": FACTOR_PROTECTED_MULTIPLIER,
                    "factor_background_multiplier": FACTOR_BACKGROUND_MULTIPLIER,
                    "anchor_feature_protected_multiplier": (
                        ANCHOR_FEATURE_PROTECTED_MULTIPLIER
                    ),
                    "anchor_feature_background_multiplier": (
                        ANCHOR_FEATURE_BACKGROUND_MULTIPLIER
                    ),
                },
                "real_row_mask": real_mask_meta,
                "padding_row_mask": padding_mask_meta,
                "active_row_mask": active_mask_meta,
                "identity_corrections": identity_meta,
            }

        with open(os.path.join(compress_dir, "meta.json"), "w") as f:
            json.dump(meta, f, sort_keys=True, separators=(",", ":"))

    def decompress(self, compress_dir: str, entropy_models, device) -> Dict[str, Tensor]:
        """Run decompression

        Args:
            compress_dir (str): directory that contains compressed files

        Returns:
            Dict[str, Tensor]: decompressed Gaussian splats
        """
        def inverse_sigmoid(x):
            return -torch.log(1/(x.clamp(1e-7,1-1e-7)) - 1)
        
        self.decoded_canonical_ids = None
        with open(os.path.join(compress_dir, "meta.json"), "r") as f:
            meta = json.load(f)

        ap_meta = meta.get("__ap__")
        if ap_meta is not None and ap_meta.get("schema") != AP_META_SCHEMA:
            raise ValueError(
                "unsupported AP-GIFStream metadata schema; v5 remains bound "
                "to the frozen Patch7 runtime"
            )
        ap_masks = None
        if ap_meta is not None:
            ap_cfg = dict(self.ap_config or {})
            provenance_required = {
                "provenance_manifest_path",
                "provenance_manifest_sha256",
                "repo_root",
            }
            missing = sorted(provenance_required - set(ap_cfg))
            if missing:
                raise ValueError(f"AP decoder lacks runtime provenance fields: {missing}")
            verify_runtime_provenance(
                Path(str(ap_cfg["provenance_manifest_path"])),
                Path(str(ap_cfg["repo_root"])),
                str(ap_cfg["provenance_manifest_sha256"]),
                expected_container_receipt=ap_meta["runtime_provenance"],
            )
            score_audit = ap_meta["score"]
            if (
                score_audit.get("runtime_manifest_sha256")
                != ap_meta["runtime_provenance"]["manifest_sha256"]
                or score_audit.get("normalized_code_tree_sha256")
                != ap_meta["runtime_provenance"]["normalized_code_tree"]["sha256"]
                or score_audit.get("patch_chain_sha256")
                != ap_meta["runtime_provenance"]["patch_sha256"]
            ):
                raise ValueError("counted AP score receipt/runtime provenance mismatch")
            ap_masks = {
                name: read_ap_mask(compress_dir, ap_meta[name], device=device)
                for name in (
                    "mask",
                    "path_input_mask",
                    "real_row_mask",
                    "padding_row_mask",
                    "active_row_mask",
                )
            }
            path_contract = ap_meta.get("path_contract")
            if (
                not isinstance(path_contract, dict)
                or path_contract.get("schema") != PATH_CONTRACT_SCHEMA
                or path_contract.get("dependency_rule")
                != "protected-plus-one-hop-retained-knn"
                or not isinstance(
                    path_contract.get("retained_knn_graph_sha256"), str
                )
                or len(path_contract["retained_knn_graph_sha256"]) != 64
                or any(
                    char not in "0123456789abcdef"
                    for char in path_contract[
                        "retained_knn_graph_sha256"
                    ].lower()
                )
                or path_contract.get("canonical_anchor_reconstruction") is not True
                or float(path_contract.get("factor_protected_multiplier", -1))
                != FACTOR_PROTECTED_MULTIPLIER
                or float(path_contract.get("factor_background_multiplier", -1))
                != FACTOR_BACKGROUND_MULTIPLIER
            ):
                raise ValueError("counted AP-v6 path contract is incomplete")

        splats = {}
        ap_quantized = bool(
            ap_meta is not None
            and ap_meta.get("variant", {}).get("quant", False)
        )
        kwargs = {
            "entropy_model": entropy_models["anchor_features"],
            "device": device,
        }
        if meta["anchor_features"].get("ap_two_class_anchor_features", False):
            if ap_meta is None:
                raise ValueError("protected anchor-feature stream lacks AP metadata")
            splats["anchor_features"] = _decompress_ap_anchor_features_ar(
                compress_dir,
                "anchor_features",
                meta["anchor_features"],
                precision_mask=ap_masks["path_input_mask"],
                **kwargs,
            )
        else:
            if ap_quantized:
                raise ValueError(
                    "quantized AP-v6 container lacks protected anchor features"
                )
            splats["anchor_features"] = _decompress_end2end_ar(
                compress_dir,
                "anchor_features",
                meta["anchor_features"],
                **kwargs,
            )
        if "factors" in meta:
            if meta["factors"].get("ap_two_class_factors", False):
                if ap_meta is None:
                    raise ValueError("protected factor stream lacks AP metadata")
                splats["factors"] = _decompress_ap_factors(
                    compress_dir,
                    "factors",
                    meta["factors"],
                    anchor_features=splats["anchor_features"],
                    precision_mask=ap_masks["path_input_mask"],
                    active_mask=ap_masks["active_row_mask"],
                    real_mask=ap_masks["real_row_mask"],
                    entropy_model=entropy_models["factors"],
                    device=device,
                )
            else:
                splats["factors"] = _decompress_end2end(
                    compress_dir,
                    "factors",
                    meta["factors"],
                    anchor_features=splats["anchor_features"],
                    entropy_model=entropy_models["factors"],
                    device=device,
                )
        if ap_meta is not None:
            encoded_count = int(splats["factors"].shape[0])
            if any(mask.shape != (encoded_count,) for mask in ap_masks.values()):
                raise ValueError("counted AP masks do not match encoded factor rows")
            if not torch.equal(ap_masks["real_row_mask"], ~ap_masks["padding_row_mask"]):
                raise ValueError("counted real/padding row masks are not complements")
            if not torch.equal(
                splats["factors"][:, 3] > 0, ap_masks["real_row_mask"]
            ):
                raise ValueError("decoded factor-3 whole mask differs from counted real rows")
            if torch.any(ap_masks["active_row_mask"] & ~ap_masks["real_row_mask"]):
                raise ValueError("counted active rows escape the explicit real-row mask")
            if ap_quantized and not meta["factors"].get(
                "ap_two_class_factors", False
            ):
                raise ValueError("AP-v6 container lacks the protected factor stream")
            if not ap_quantized and meta["factors"].get(
                "ap_two_class_factors", False
            ):
                raise ValueError(
                    "non-quantized AP ablation unexpectedly uses protected factors"
                )
            if not ap_quantized:
                _apply_factor_mask(
                    splats["factors"],
                    0,
                    ap_masks["active_row_mask"],
                    float(meta["factors"]["scaling"]),
                )
            if torch.any(ap_masks["mask"] & ~ap_masks["real_row_mask"]):
                raise ValueError("decoded AP class includes padding rows")
            if torch.any(ap_masks["mask"] & ~ap_masks["active_row_mask"]):
                raise ValueError("decoded AP class contains an inactive row")
            if torch.any(ap_masks["mask"] & ~ap_masks["path_input_mask"]):
                raise ValueError("decoded AP path-input closure lost a protected row")
            if torch.any(
                ap_masks["path_input_mask"] & ~ap_masks["real_row_mask"]
            ):
                raise ValueError("decoded AP path-input closure includes padding")
        for param_name, param_meta in meta.items():
            if param_name in {"anchor_features", "factors", "__ap__"}: continue
            decompress_fn = self._get_decompress_fn(param_name)
            kwargs = {
                "anchor_features": splats["anchor_features"],
                "entropy_model": entropy_models[param_name],
                "device": device
            }
            if param_name == "time_features" and param_meta.get("ap_two_class", False):
                if ap_meta is None:
                    raise ValueError("two-class temporal stream is missing AP metadata")
                active_class_mask = ap_masks["path_input_mask"][
                    ap_masks["active_row_mask"]
                ]
                splats[param_name] = _decompress_ap_time_features_ar(
                    compress_dir,
                    param_name,
                    param_meta,
                    class_mask=active_class_mask,
                    **kwargs,
                )
            else:
                splats[param_name] = decompress_fn(compress_dir, param_name, param_meta, **kwargs)

        # Param-specific postprocessing
        # splats["anchors"] = inverse_log_transform(splats["anchors"])
        #* re-voxelize
        mask = (splats["quats"].any(dim=1) != 0)
        if ap_meta is not None and not _tensor_equal_device_agnostic(
            mask, ap_masks["real_row_mask"]
        ):
            raise ValueError("decoded quaternion real rows differ from explicit real/padding masks")
        for k,v in splats.items():
            if k != "time_features":
                splats[k] = v[mask]
        voxel_size = meta["anchors"]["voxel_size"]
        splats["anchors"] = torch.round(splats["anchors"]/voxel_size)*voxel_size

        if ap_meta is not None:
            decoded_ids = (
                torch.round(splats["anchors"].detach().cpu() / float(voxel_size))
                .to(torch.int64)
                .numpy()
                .astype(np.int64, copy=False)
            )
            retained_ids = torch.from_numpy(
                read_identity_corrections(
                    compress_dir, ap_meta["identity_corrections"], decoded_ids
                )
            ).to(device=device, dtype=torch.int64)
            _validate_counted_retained_identity(retained_ids, len(splats["anchors"]))
            self.decoded_canonical_ids = retained_ids.detach().clone()
            # AP-v6 makes the counted identity authority the coordinate
            # authority as well.  This removes lossy anchor-PNG drift and gives
            # training, clean decode, and path evaluation the same KNN geometry.
            splats["anchors"] = retained_ids.to(
                dtype=splats["anchors"].dtype, device=splats["anchors"].device
            ) * float(voxel_size)
            retained_knn = deterministic_knn_indices(
                splats["anchors"],
                int(path_contract["knn_count"]),
                canonical_ids=retained_ids,
            )
            decoded_graph_sha256 = retained_knn_graph_sha256(
                retained_ids, retained_knn
            )
            if (
                decoded_graph_sha256
                != path_contract["retained_knn_graph_sha256"]
            ):
                raise ValueError("decoded AP retained-KNN graph hash mismatch")
            decoded_class_mask = ap_masks["mask"][
                ap_masks["real_row_mask"]
            ]
            decoded_path_mask = ap_masks["path_input_mask"][
                ap_masks["real_row_mask"]
            ]
            expected_path_mask, expected_knn = build_path_input_precision_mask(
                splats["anchors"],
                decoded_class_mask,
                torch.ones_like(decoded_class_mask),
                int(path_contract["knn_count"]),
                canonical_ids=retained_ids,
            )
            if not _tensor_equal_device_agnostic(
                expected_knn, retained_knn
            ) or not _tensor_equal_device_agnostic(
                expected_path_mask, decoded_path_mask
            ):
                raise ValueError(
                    "decoded AP path-input mask is not the exact retained-KNN closure"
                )

        #* recover time features
        choose_idx = splats["factors"][:,0] > 0
        if ap_meta is not None and int(choose_idx.sum().item()) != int(
            ap_masks["active_row_mask"].sum().item()
        ):
            raise ValueError("decoded active-row count changed after padding removal")
        time_features = torch.zeros((len(splats["anchors"]),meta["time_features"]["shape"][1], meta["time_features"]["shape"][2]),device=device)
        time_features[choose_idx] = splats["time_features"]
        splats["time_features"] = time_features

        splats["factors"] = inverse_sigmoid(splats["factors"])
        return splats


def _crop_n_splats(splats: Dict[str, Tensor], n_crop: int) -> Dict[str, Tensor]:
    if n_crop > 0:
        opacities = splats["opacities"].view((-1))
        keep_indices = torch.argsort(opacities, descending=True)[:-n_crop]
        for k, v in splats.items():
            splats[k] = v[keep_indices]
        return splats
    else:
        for k, v in splats.items():
            splats[k] = torch.cat([v,torch.zeros([-n_crop]+list(v.shape[1:]), device = v.device)],dim=0)
        return splats


def _compress_png(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with 8-bit quantization and lossless PNG compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length

    Returns:
        Dict[str, Any]: metadata
    """
    import imageio.v2 as imageio

    if torch.numel == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta

    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()

    img = (img_norm * (2**8 - 1)).round().astype(np.uint8)
    img = img.squeeze()
    imageio.imwrite(os.path.join(compress_dir, f"{param_name}.png"), img)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    return meta


def _decompress_png(compress_dir: str, param_name: str, meta: Dict[str, Any], **kwargs) -> Tensor:
    """Decompress parameters from PNG file.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    import imageio.v2 as imageio

    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return meta

    img = imageio.imread(os.path.join(compress_dir, f"{param_name}.png"))
    img_norm = img / (2**8 - 1)

    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params

def _compress_png_kbit(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, quantization: int = 8, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with k-bit quantization and lossless PNG compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length

    Returns:
        Dict[str, Any]: metadata
    """
    import imageio.v2 as imageio

    if torch.numel == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta

    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()

    img = (img_norm * (2**quantization - 1)).round().astype(np.uint8)
    img = img << (8 - quantization)
    img = img.squeeze()
    if grid.shape[-1] > 4:
        for ind in range(grid.shape[-1]//3):
            imageio.imwrite(os.path.join(compress_dir, f"{param_name}_{ind}.png"), img[:,:,3*ind:3*ind+3])
    else:
        imageio.imwrite(os.path.join(compress_dir, f"{param_name}.png"), img)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "quantization": quantization, 
    }
    return meta


def _decompress_png_kbit(compress_dir: str, param_name: str, meta: Dict[str, Any], **kwargs) -> Tensor:
    """Decompress parameters from PNG file.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    import imageio.v2 as imageio

    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return meta

    if np.prod(meta["shape"][1:]) > 4:
        for ind in range(np.prod(meta["shape"][1:])//3):
            tmp_img = imageio.imread(os.path.join(compress_dir, f"{param_name}_{ind}.png")) 
            img = tmp_img if ind == 0 else np.concatenate([img, tmp_img], axis=-1)
    else:
        img = imageio.imread(os.path.join(compress_dir, f"{param_name}.png"))
    img = img >> (8 - meta["quantization"])
    img_norm = img / (2**meta["quantization"] - 1)

    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_png_16bit(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with 16-bit quantization and PNG compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length

    Returns:
        Dict[str, Any]: metadata
    """
    import imageio.v2 as imageio

    if torch.numel == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta

    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * (2**16 - 1)).round().astype(np.uint16)

    img_l = img & 0xFF
    img_u = (img >> 8) & 0xFF
    imageio.imwrite(
        os.path.join(compress_dir, f"{param_name}_l.png"), img_l.astype(np.uint8)
    )
    imageio.imwrite(
        os.path.join(compress_dir, f"{param_name}_u.png"), img_u.astype(np.uint8)
    )

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "voxel_size": float(kwargs["voxel_size"])
    }
    return meta


def _decompress_png_16bit(
    compress_dir: str, param_name: str, meta: Dict[str, Any], **kwargs
) -> Tensor:
    """Decompress parameters from PNG files.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    import imageio.v2 as imageio

    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return meta

    img_l = imageio.imread(os.path.join(compress_dir, f"{param_name}_l.png"))
    img_u = imageio.imread(os.path.join(compress_dir, f"{param_name}_u.png"))
    img_u = img_u.astype(np.uint16)
    img = (img_u << 8) + img_l

    img_norm = img / (2**16 - 1)
    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_npz(
    compress_dir: str, param_name: str, params: Tensor, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with numpy's NPZ compression."""
    npz_dict = {"arr": params.detach().cpu().numpy()}
    save_fp = os.path.join(compress_dir, f"{param_name}.npz")
    os.makedirs(os.path.dirname(save_fp), exist_ok=True)
    np.savez_compressed(save_fp, **npz_dict)
    meta = {
        "shape": params.shape,
        "dtype": str(params.dtype).split(".")[1],
    }
    return meta


def _decompress_npz(compress_dir: str, param_name: str, meta: Dict[str, Any], **kwargs) -> Tensor:
    """Decompress parameters with numpy's NPZ compression."""
    arr = np.load(os.path.join(compress_dir, f"{param_name}.npz"))["arr"]
    params = torch.tensor(arr)
    params = params.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_end2end(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with 16-bit quantization and PNG compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length

    Returns:
        Dict[str, Any]: metadata
    """
    import imageio.v2 as imageio

    if torch.numel == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta

    params = params/kwargs["scaling"]
    anchor_features = kwargs["anchor_features"]
    entropy_model = kwargs["entropy_model"]
    output_path = os.path.join(compress_dir,f"{param_name}.bin")
    entropy_model.compress(params.flatten(1),anchor_features,output_path, adaptive=True)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "scaling": float(kwargs["scaling"])
    }
    return meta


def _decompress_end2end(
    compress_dir: str, param_name: str, meta: Dict[str, Any], **kwargs
) -> Tensor:
    """Decompress parameters from PNG files.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    import imageio.v2 as imageio

    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return meta

    
    anchor_features = kwargs["anchor_features"]
    entropy_model = kwargs["entropy_model"]
    output_path = os.path.join(compress_dir,f"{param_name}.bin")
    params = entropy_model.decompress(anchor_features, output_path, adaptive=True) * meta["scaling"]

    params = params.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params

def _compress_end2end_ar(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with 16-bit quantization and PNG compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length

    Returns:
        Dict[str, Any]: metadata
    """
    import imageio.v2 as imageio

    if torch.numel == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta

    params = params/kwargs["scaling"]
    channel = kwargs["c_channel"] if param_name == "anchor_features" else kwargs["p_channel"]
    N, f_channel = params.flatten(1).shape
    condition = torch.cat([torch.zeros((N,3*channel), device=params.device), params.flatten(1)],dim=-1)
    condition = torch.cat([condition.view((N,-1,channel))[:,x:-3+x] for x in range(3)],dim=-1)
    entropy_model = kwargs["entropy_model"]
    for ind in range(f_channel//channel):
        output_path = os.path.join(compress_dir,f"{param_name}_{ind:05d}.bin")
        entropy_model.compress(params.flatten(1)[:,ind*channel:ind*channel+channel],condition[:,ind],output_path)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "scaling": float(kwargs["scaling"]),
        "length": f_channel // channel,
        "channel": channel
    }
    return meta


def _decompress_end2end_ar(
    compress_dir: str, param_name: str, meta: Dict[str, Any], **kwargs
) -> Tensor:
    """Decompress parameters from PNG files.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    import imageio.v2 as imageio

    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return meta

    entropy_model = kwargs["entropy_model"]
    condition = torch.zeros((meta["shape"][0],meta["channel"] * 3), device=kwargs["device"])
    for ind in range(meta["length"]):
        output_path = os.path.join(compress_dir,f"{param_name}_{ind:05d}.bin")
        tmp = entropy_model.decompress(condition, output_path)
        condition = torch.cat([condition[:,meta["channel"]:], tmp],dim=-1)
        params = tmp if ind == 0 else torch.cat([params,tmp],dim=-1)

    params = params.reshape(meta["shape"]) * meta["scaling"]
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_ap_anchor_features_ar(
    compress_dir: str,
    param_name: str,
    params: Tensor,
    n_sidelen: int,
    precision_mask: Tensor,
    protected_multiplier: float,
    background_multiplier: float,
    expected_reconstruction: Tensor,
    **kwargs,
) -> Dict[str, Any]:
    if param_name != "anchor_features" or params.ndim != 2:
        raise ValueError("protected anchor features must be a [N,C] tensor")
    precision_mask = precision_mask.to(device=params.device, dtype=torch.bool)
    if precision_mask.shape != (params.shape[0],):
        raise ValueError("anchor-feature precision mask row mismatch")
    base_scaling = float(kwargs["scaling"])
    channel = int(kwargs["c_channel"])
    if channel <= 0 or params.shape[1] % channel:
        raise ValueError("invalid protected anchor-feature AR partition")
    reconstruction = ste_row_quantize(
        params,
        precision_mask,
        base_scaling,
        protected_multiplier,
        background_multiplier,
    )
    if not torch.equal(reconstruction.detach(), expected_reconstruction.detach()):
        raise AssertionError("anchor-feature simulator/encoder reconstruction differs")
    entropy_model = kwargs["entropy_model"]
    families: Dict[str, Any] = {}
    for family, family_mask, multiplier in (
        ("path", precision_mask, float(protected_multiplier)),
        ("bg", ~precision_mask, float(background_multiplier)),
    ):
        family_params = params[family_mask]
        q_scaling = base_scaling * multiplier
        scaled = torch.round(family_params / q_scaling)
        row_count, feature_channels = scaled.shape
        length = feature_channels // channel
        if row_count:
            condition = torch.cat(
                [
                    torch.zeros(
                        (row_count, 3 * channel), device=params.device
                    ),
                    scaled,
                ],
                dim=-1,
            )
            condition = torch.cat(
                [
                    condition.view((row_count, -1, channel))[:, x : -3 + x]
                    for x in range(3)
                ],
                dim=-1,
            )
            for index in range(length):
                output_path = os.path.join(
                    compress_dir, f"{param_name}_{family}_{index:05d}.bin"
                )
                entropy_model.compress(
                    scaled[:, index * channel : (index + 1) * channel],
                    condition[:, index],
                    output_path,
                )
        families[family] = {
            "rows": int(row_count),
            "scaling": float(q_scaling),
            "multiplier": float(multiplier),
            "length": int(length),
            "channel": int(channel),
        }
    return {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "ap_two_class_anchor_features": True,
        "base_scaling": base_scaling,
        "families": families,
    }


def _decompress_ap_anchor_features_ar(
    compress_dir: str,
    param_name: str,
    meta: Dict[str, Any],
    precision_mask: Tensor,
    **kwargs,
) -> Tensor:
    if not meta.get("ap_two_class_anchor_features", False):
        raise ValueError("not a protected anchor-feature stream")
    shape = tuple(int(value) for value in meta["shape"])
    if len(shape) != 2:
        raise ValueError("protected anchor-feature shape is malformed")
    precision_mask = precision_mask.to(
        device=kwargs["device"], dtype=torch.bool
    )
    if precision_mask.shape != (shape[0],):
        raise ValueError("decoded anchor-feature precision mask mismatch")
    output = torch.empty(shape, device=kwargs["device"])
    entropy_model = kwargs["entropy_model"]
    for family, family_mask in (("path", precision_mask), ("bg", ~precision_mask)):
        family_meta = meta["families"][family]
        row_count = int(family_meta["rows"])
        if int(family_mask.sum().item()) != row_count:
            raise ValueError(f"{family} anchor-feature row count mismatch")
        channel = int(family_meta["channel"])
        length = int(family_meta["length"])
        if length * channel != shape[1]:
            raise ValueError(f"{family} anchor-feature stream length mismatch")
        if not row_count:
            continue
        condition = torch.zeros(
            (row_count, 3 * channel), device=kwargs["device"]
        )
        chunks = []
        for index in range(length):
            output_path = os.path.join(
                compress_dir, f"{param_name}_{family}_{index:05d}.bin"
            )
            if not os.path.isfile(output_path):
                raise ValueError(f"missing protected anchor stream: {output_path}")
            chunk = entropy_model.decompress(condition, output_path)
            if chunk.shape != (row_count, channel):
                raise ValueError("decoded protected anchor chunk shape mismatch")
            condition = torch.cat([condition[:, channel:], chunk], dim=-1)
            chunks.append(chunk)
        output[family_mask] = torch.cat(chunks, dim=-1) * float(
            family_meta["scaling"]
        )
    output = output.to(dtype=getattr(torch, meta["dtype"]))
    if not torch.isfinite(output).all():
        raise ValueError("nonfinite protected anchor-feature decode")
    return output


def _compress_ap_factors(
    compress_dir: str,
    param_name: str,
    raw_factor_logits: Tensor,
    n_sidelen: int,
    precision_mask: Tensor,
    active_mask: Tensor,
    real_mask: Tensor,
    factor0_activation_value: float,
    protected_multiplier: float,
    background_multiplier: float,
    expected_reconstruction: Tensor,
    **kwargs,
) -> Dict[str, Any]:
    if param_name != "factors" or raw_factor_logits.shape[1:] != (4,):
        raise ValueError("protected factors must be [N,4]")
    precision_mask = precision_mask.to(
        device=raw_factor_logits.device, dtype=torch.bool
    )
    active_mask = active_mask.to(device=raw_factor_logits.device, dtype=torch.bool)
    real_mask = real_mask.to(device=raw_factor_logits.device, dtype=torch.bool)
    if any(
        mask.shape != (raw_factor_logits.shape[0],)
        for mask in (precision_mask, active_mask, real_mask)
    ):
        raise ValueError("protected factor masks do not align")
    base_scaling = float(kwargs["scaling"])
    reconstruction, _ = reconstruct_ap_factors(
        raw_factor_logits,
        kwargs["anchor_features"],
        kwargs["entropy_model"],
        base_scaling,
        precision_mask,
        protected_multiplier,
        background_multiplier,
        active_mask,
        real_mask,
        factor0_activation_value,
    )
    if not torch.allclose(
        reconstruction.detach(),
        expected_reconstruction.detach(),
        rtol=0.0,
        atol=1e-7,
    ):
        raise AssertionError("factor simulator/encoder reconstruction differs")
    activated = normalize_factor_semantics(
        torch.sigmoid(raw_factor_logits),
        active_mask,
        real_mask,
        factor0_activation_value,
    )
    entropy_model = kwargs["entropy_model"]
    families: Dict[str, Any] = {}
    for family, family_mask, multiplier in (
        ("path", precision_mask, float(protected_multiplier)),
        ("bg", ~precision_mask, float(background_multiplier)),
    ):
        q_scaling = base_scaling * multiplier
        row_count = int(family_mask.sum().item())
        output_path = os.path.join(compress_dir, f"{param_name}_{family}.bin")
        if row_count:
            entropy_model.compress(
                activated[family_mask] / q_scaling,
                kwargs["anchor_features"][family_mask],
                output_path,
                adaptive=True,
            )
            payload = Path(output_path).read_bytes()
            stream_bytes = len(payload)
            stream_sha256 = hashlib.sha256(payload).hexdigest()
        else:
            stream_bytes = 0
            stream_sha256 = hashlib.sha256(b"").hexdigest()
        families[family] = {
            "rows": row_count,
            "scaling": float(q_scaling),
            "multiplier": float(multiplier),
            "path": os.path.basename(output_path),
            "bytes": int(stream_bytes),
            "sha256": stream_sha256,
        }
    return {
        "shape": list(raw_factor_logits.shape),
        "dtype": str(raw_factor_logits.dtype).split(".")[1],
        "ap_two_class_factors": True,
        "base_scaling": base_scaling,
        "factor0_activation_value": float(factor0_activation_value),
        "reconstruction_rule": "adaptive-symbols+counted-factor0/factor3-semantics",
        "families": families,
    }


def _decompress_ap_factors(
    compress_dir: str,
    param_name: str,
    meta: Dict[str, Any],
    anchor_features: Tensor,
    precision_mask: Tensor,
    active_mask: Tensor,
    real_mask: Tensor,
    **kwargs,
) -> Tensor:
    if not meta.get("ap_two_class_factors", False):
        raise ValueError("not a protected AP factor stream")
    shape = tuple(int(value) for value in meta["shape"])
    if len(shape) != 2 or shape[1] != 4:
        raise ValueError("protected factor shape is malformed")
    precision_mask = precision_mask.to(device=kwargs["device"], dtype=torch.bool)
    active_mask = active_mask.to(device=kwargs["device"], dtype=torch.bool)
    real_mask = real_mask.to(device=kwargs["device"], dtype=torch.bool)
    if any(mask.shape != (shape[0],) for mask in (precision_mask, active_mask, real_mask)):
        raise ValueError("decoded protected factor masks do not align")
    output = torch.empty(shape, device=kwargs["device"])
    entropy_model = kwargs["entropy_model"]
    for family, family_mask in (("path", precision_mask), ("bg", ~precision_mask)):
        family_meta = meta["families"][family]
        row_count = int(family_meta["rows"])
        if int(family_mask.sum().item()) != row_count:
            raise ValueError(f"{family} factor row count mismatch")
        if not row_count:
            if int(family_meta["bytes"]) != 0:
                raise ValueError("empty factor family has nonzero counted bytes")
            continue
        output_path = os.path.join(compress_dir, str(family_meta["path"]))
        if not os.path.isfile(output_path):
            raise ValueError(f"missing protected factor stream: {output_path}")
        payload = Path(output_path).read_bytes()
        if (
            len(payload) != int(family_meta["bytes"])
            or hashlib.sha256(payload).hexdigest() != family_meta["sha256"]
        ):
            raise ValueError(f"{family} protected factor stream binding mismatch")
        decoded = entropy_model.decompress(
            anchor_features[family_mask], output_path, adaptive=True
        )
        if decoded.shape != (row_count, 4):
            raise ValueError(f"{family} protected factor decode shape mismatch")
        output[family_mask] = decoded * float(family_meta["scaling"])
    output = normalize_factor_semantics(
        output,
        active_mask,
        real_mask,
        float(meta["factor0_activation_value"]),
    )
    output = output.to(dtype=getattr(torch, meta["dtype"]))
    if not torch.isfinite(output).all():
        raise ValueError("nonfinite protected factor decode")
    return output


def _compress_ap_time_features_ar(
    compress_dir: str,
    param_name: str,
    params: Tensor,
    n_sidelen: int,
    class_mask: Tensor,
    q_ap_multiplier: float,
    q_bg_multiplier: float,
    **kwargs,
) -> Dict[str, Any]:
    """Encode active temporal rows as separately quantized AP/background AR families."""

    if param_name != "time_features":
        raise ValueError("two-class AR coding is defined only for time_features")
    if params.ndim != 3:
        raise ValueError(f"time_features must be [N,T,C], got {tuple(params.shape)}")
    class_mask = class_mask.to(device=params.device, dtype=torch.bool)
    if class_mask.shape != (params.shape[0],):
        raise ValueError("AP class mask does not match active temporal rows")
    if not (0.0 < q_ap_multiplier <= 1.0):
        raise ValueError("q_ap_multiplier must be in (0,1]")
    if q_bg_multiplier < 1.0:
        raise ValueError("q_bg_multiplier must be >= 1")
    base_scaling = float(kwargs["scaling"])
    channel = int(kwargs["p_channel"])
    if channel <= 0 or params.flatten(1).shape[1] % channel:
        raise ValueError("invalid temporal AR channel partition")
    entropy_model = kwargs["entropy_model"]

    family_meta: Dict[str, Any] = {}
    for family, family_mask, multiplier in (
        ("path", class_mask, float(q_ap_multiplier)),
        ("bg", ~class_mask, float(q_bg_multiplier)),
    ):
        family_params = params[family_mask]
        q_scaling = base_scaling * multiplier
        # Encode explicit integer reconstruction symbols.  Conditions are then
        # identical to the decoder's previous reconstructed chunks, including
        # when q_ap is finer than the official base Q.
        scaled = torch.round(family_params / q_scaling)
        flat = scaled.flatten(1)
        row_count, feature_channels = flat.shape
        length = feature_channels // channel
        if row_count:
            condition = torch.cat(
                [torch.zeros((row_count, 3 * channel), device=params.device), flat], dim=-1
            )
            condition = torch.cat(
                [condition.view((row_count, -1, channel))[:, x : -3 + x] for x in range(3)],
                dim=-1,
            )
            for index in range(length):
                output_path = os.path.join(
                    compress_dir, f"{param_name}_{family}_{index:05d}.bin"
                )
                entropy_model.compress(
                    flat[:, index * channel : (index + 1) * channel],
                    condition[:, index],
                    output_path,
                )
        family_meta[family] = {
            "rows": int(row_count),
            "scaling": q_scaling,
            "multiplier": multiplier,
            "length": int(length),
            "channel": channel,
        }

    if family_meta["path"]["rows"] + family_meta["bg"]["rows"] != params.shape[0]:
        raise AssertionError("two-class temporal encoder lost rows")
    return {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "ap_two_class": True,
        "precision_mask_contract": "path_input_mask",
        "base_scaling": base_scaling,
        "length": int(params.flatten(1).shape[1] // channel),
        "channel": channel,
        "families": family_meta,
    }


def _decompress_ap_time_features_ar(
    compress_dir: str,
    param_name: str,
    meta: Dict[str, Any],
    class_mask: Tensor,
    **kwargs,
) -> Tensor:
    if not meta.get("ap_two_class", False):
        raise ValueError("not an AP two-class temporal stream")
    shape = tuple(int(v) for v in meta["shape"])
    if len(shape) != 3 or class_mask.shape != (shape[0],):
        raise ValueError("decoded AP class mask/temporal shape mismatch")
    class_mask = class_mask.to(device=kwargs["device"], dtype=torch.bool)
    entropy_model = kwargs["entropy_model"]
    flat_channels = int(np.prod(shape[1:]))
    flat = torch.empty((shape[0], flat_channels), device=kwargs["device"])

    if meta.get("precision_mask_contract") != "path_input_mask":
        raise ValueError("temporal precision stream lacks the path-input contract")
    for family, family_mask in (("path", class_mask), ("bg", ~class_mask)):
        family_meta = meta["families"][family]
        row_count = int(family_meta["rows"])
        if int(family_mask.sum().item()) != row_count:
            raise ValueError(f"{family} temporal row-count mismatch")
        channel = int(family_meta["channel"])
        length = int(family_meta["length"])
        if length * channel != flat_channels:
            raise ValueError(f"{family} temporal stream length mismatch")
        if not row_count:
            continue
        condition = torch.zeros((row_count, channel * 3), device=kwargs["device"])
        decoded_chunks = []
        for index in range(length):
            output_path = os.path.join(
                compress_dir, f"{param_name}_{family}_{index:05d}.bin"
            )
            if not os.path.isfile(output_path):
                raise ValueError(f"missing AP temporal stream: {output_path}")
            chunk = entropy_model.decompress(condition, output_path)
            if chunk.shape != (row_count, channel):
                raise ValueError(f"decoded {family} temporal chunk shape mismatch")
            condition = torch.cat([condition[:, channel:], chunk], dim=-1)
            decoded_chunks.append(chunk)
        flat[family_mask] = torch.cat(decoded_chunks, dim=-1) * float(family_meta["scaling"])

    params = flat.reshape(shape).to(dtype=getattr(torch, meta["dtype"]))
    if not torch.isfinite(params).all():
        raise ValueError("nonfinite AP temporal decode")
    return params

def save_ply(splats: torch.nn.ParameterDict, path: str):
    from plyfile import PlyData, PlyElement

    means = splats["means"].detach().cpu().numpy()
    normals = np.zeros_like(means)
    sh0 = splats["sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    shN = splats["shN"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = splats["opacities"].detach().unsqueeze(1).cpu().numpy()
    scales = splats["scales"].detach().cpu().numpy()
    quats = splats["quats"].detach().cpu().numpy()

    def construct_list_of_attributes(splats):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

        for i in range(splats["sh0"].shape[1]*splats["sh0"].shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(splats["shN"].shape[1]*splats["shN"].shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(splats["scales"].shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(splats["quats"].shape[1]):
            l.append('rot_{}'.format(i))
        return l

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(splats)]

    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = np.concatenate((means, normals, sh0, shN, opacities, scales, quats), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def save_params_into_ply_file(
    splats, path
):
    """Save parameters of Gaussian Splats into .ply file"""
    ply_dir = f"{path}/ply"
    os.makedirs(ply_dir, exist_ok=True)
    ply_file = ply_dir + "/pruned_splats.ply"
    save_ply(splats, ply_file)
    print(f"Saved parameters of splats into file: {ply_file}.")
