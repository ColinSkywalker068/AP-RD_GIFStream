"""Auditable allocation and container helpers for H007 AP-GIFStream.

This module deliberately contains no rendering- or downstream-dependent code.  An
AP score artifact is aligned to anchors only through integer canonical voxel IDs.
The final codec is therefore unable to inspect a candidate render or task result
while choosing its allocation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


AP_SCORE_SCHEMA = "h007.ap_scores.v3"
AP_META_SCHEMA_V5 = "h007.ap_gifstream.codec.v5"
AP_META_SCHEMA = "h007.ap_gifstream.codec.v6"
AP_IDENTITY_CORRECTION_SCHEMA = "h007.ap_identity_corrections.v1"
DETERMINISTIC_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class APVariantSpec:
    ranking: str
    swap: bool
    quant: bool
    action_loss: bool


AP_VARIANTS: Dict[str, APVariantSpec] = {
    "official": APVariantSpec("official", False, False, False),
    "random-full": APVariantSpec("random", True, True, True),
    "motion-full": APVariantSpec("motion", True, True, True),
    "path-swap": APVariantSpec("path", True, False, False),
    "path-quant": APVariantSpec("path", False, True, False),
    "path-swap-quant": APVariantSpec("path", True, True, False),
    "ap-gifstream-full": APVariantSpec("path", True, True, True),
}


def variant_spec(name: str) -> APVariantSpec:
    if name not in AP_VARIANTS:
        raise ValueError(f"Unknown AP-GIFStream variant: {name!r}")
    return AP_VARIANTS[name]


def canonical_voxel_ids(anchors: Tensor, voxel_size: float) -> Tensor:
    if anchors.ndim != 2 or anchors.shape[1] != 3:
        raise ValueError(f"anchors must be [N,3], got {tuple(anchors.shape)}")
    if not math.isfinite(float(voxel_size)) or float(voxel_size) <= 0:
        raise ValueError("voxel_size must be finite and positive")
    ids = torch.round(anchors.detach() / float(voxel_size)).to(torch.int64)
    if torch.unique(ids, dim=0).shape[0] != ids.shape[0]:
        raise ValueError("duplicate canonical voxel IDs")
    return ids


def tensor_mapping_sha256(values: Mapping[str, Tensor]) -> str:
    """Hash a tensor mapping independent of ``torch.save`` container metadata."""
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name]
        if not isinstance(value, Tensor):
            raise ValueError(f"state mapping contains a non-tensor value: {name}")
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


def _lexicographic_rank(
    ids: np.ndarray,
    scores: np.ndarray,
    descending: bool,
    secondary: Optional[np.ndarray] = None,
) -> np.ndarray:
    if ids.ndim != 2 or ids.shape[1] != 3:
        raise ValueError("canonical IDs must be [N,3]")
    if scores.shape != (ids.shape[0],):
        raise ValueError("score length does not match canonical IDs")
    primary = -scores if descending else scores
    if secondary is None:
        return np.lexsort((ids[:, 2], ids[:, 1], ids[:, 0], primary))
    if secondary.shape != (ids.shape[0],):
        raise ValueError("secondary key length does not match canonical IDs")
    tiebreak = -secondary if descending else secondary
    return np.lexsort((ids[:, 2], ids[:, 1], ids[:, 0], tiebreak, primary))


def frozen_backbone_importance(
    opacity_accum: Tensor,
    anchor_demon: Tensor,
    peak_ratio: float = 0.1,
) -> Tensor:
    """The backbone's own pruning statistic, frozen as the donor tie-break key.

    Mirrors the GIFStream densification prune criterion (peak/mean-blended
    accumulated rendered opacity, normalized by accumulated visit count) up to
    a constant GOP factor that cannot change any ordering.  The inputs are
    backbone training statistics recorded before the AP freeze, so the
    estimate never sees a compression outcome.  Anchors the backbone never
    rendered score exactly zero.
    """
    if opacity_accum.ndim != 2 or anchor_demon.shape != opacity_accum.shape:
        raise ValueError("opacity_accum/anchor_demon must share the same [N,GOP] shape")
    if not (0.0 <= float(peak_ratio) <= 1.0):
        raise ValueError("peak_ratio must be in [0,1]")
    accum = opacity_accum.detach().to(dtype=torch.float64, device="cpu")
    demon = anchor_demon.detach().to(dtype=torch.float64, device="cpu")
    if not torch.isfinite(accum).all() or not torch.isfinite(demon).all():
        raise ValueError("backbone importance statistics must be finite")
    if torch.any(accum < 0) or torch.any(demon < 0):
        raise ValueError("backbone importance statistics must be nonnegative")
    blended = float(peak_ratio) * accum.max(dim=-1).values + (
        1.0 - float(peak_ratio)
    ) * accum.mean(dim=-1)
    visits = demon.sum(dim=-1)
    safe_visits = torch.where(visits > 0, visits, torch.ones_like(visits))
    return torch.where(visits > 0, blended / safe_visits, torch.zeros_like(blended))


def _validate_donor_secondary(
    importance: Optional[Tensor], count: int
) -> Optional[np.ndarray]:
    if importance is None:
        return None
    if importance.ndim != 1 or importance.numel() != count:
        raise ValueError("donor importance length does not match the anchor dimension")
    importance_np = importance.detach().to(torch.float64).cpu().numpy()
    if not np.isfinite(importance_np).all() or np.any(importance_np < 0):
        raise ValueError("donor importance must be finite and nonnegative")
    return importance_np


def _rows_to_index(ids: np.ndarray) -> Dict[Tuple[int, int, int], int]:
    out: Dict[Tuple[int, int, int], int] = {}
    for index, row in enumerate(ids.tolist()):
        key = tuple(int(v) for v in row)
        if key in out:
            raise ValueError(f"duplicate canonical ID in score artifact: {key}")
        out[key] = index
    return out


def load_aligned_score_artifact(
    path: str,
    anchors: Tensor,
    voxel_size: float,
    ranking: str,
    random_seed: int,
    expected_runtime_provenance: Mapping[str, Any],
    expected_scene: str,
    expected_variant: Optional[str] = None,
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any], Dict[str, Any]]:
    """Load a frozen reference-only score artifact and align by canonical ID.

    Required NPZ members are ``schema``, ``canonical_ids``, ``eligible``,
    ``path_score``, ``motion_score``, ``allocation_score``,
    ``importance_score`` and ``estimated_time_bytes``. Random rankings consume
    the score vector frozen by training; they never regenerate a second RNG
    stream in the codec.
    """

    current_ids = canonical_voxel_ids(anchors, voxel_size).cpu().numpy()
    with np.load(path, allow_pickle=False) as artifact:
        members = set(artifact.files)
        required = {
            "schema",
            "scene",
            "voxel_size",
            "frame_count",
            "variant",
            "protected_fraction",
            "q_ap_multiplier",
            "q_bg_multiplier",
            "random_seed",
            "canonical_ids",
            "eligible",
            "path_score",
            "motion_score",
            "allocation_score",
            "importance_score",
            "estimated_time_bytes",
            "official_retain_mask",
            "official_factor0_mask",
            "official_active_mask",
            "ap_retain_mask",
            "ap_active_mask",
            "ap_class_mask",
            "factor0_activation_value",
            "factor3_activation_value",
            "estimator_version",
            "time_entropy_model_sha256",
            "time_feature_scaling",
            "time_entropy_model_frozen_after_freeze",
            "runtime_manifest_sha256",
            "normalized_code_tree_sha256",
            "patch_chain_sha256",
        }
        missing = sorted(required - members)
        if missing:
            raise ValueError(f"AP score artifact missing members: {missing}")
        schema_value = artifact["schema"]
        schema = str(schema_value.item() if schema_value.shape == () else schema_value.tolist())
        if schema != AP_SCORE_SCHEMA:
            raise ValueError(f"unexpected AP score schema: {schema!r}")
        scene_value = artifact["scene"]
        scene = str(scene_value.item() if scene_value.shape == () else scene_value.tolist())
        if scene != expected_scene:
            raise ValueError(
                f"score artifact scene {scene!r} != requested scene {expected_scene!r}"
            )
        artifact_voxel_size = float(np.asarray(artifact["voxel_size"]).item())
        if artifact_voxel_size != float(voxel_size):
            raise ValueError(
                f"score artifact voxel_size {artifact_voxel_size} != codec {voxel_size}"
            )
        if int(np.asarray(artifact["frame_count"]).item()) != 60:
            raise ValueError("U3 score artifact must contain exactly 60 GOP frames")
        artifact_variant = str(np.asarray(artifact["variant"]).item())
        if expected_variant is not None and artifact_variant != expected_variant:
            raise ValueError(
                f"score artifact variant {artifact_variant!r} != requested {expected_variant!r}"
            )
        protected_fraction = float(np.asarray(artifact["protected_fraction"]).item())
        q_ap_multiplier = float(np.asarray(artifact["q_ap_multiplier"]).item())
        q_bg_multiplier = float(np.asarray(artifact["q_bg_multiplier"]).item())
        artifact_random_seed = int(np.asarray(artifact["random_seed"]).item())
        if not (0.0 < protected_fraction <= 1.0):
            raise ValueError("invalid frozen protected_fraction")
        if not (0.0 < q_ap_multiplier <= 1.0) or q_bg_multiplier < 1.0:
            raise ValueError("invalid frozen AP/background quantizer multipliers")
        if ranking == "random" and artifact_random_seed != int(random_seed):
            raise ValueError("random ranking seed differs from frozen score artifact")
        artifact_ids = np.asarray(artifact["canonical_ids"], dtype=np.int64)
        eligible_src = np.asarray(artifact["eligible"], dtype=np.bool_)
        allocation_score_src = np.asarray(artifact["allocation_score"], dtype=np.float64)
        if ranking == "path":
            score_src = np.asarray(artifact["path_score"], dtype=np.float64)
        elif ranking == "motion":
            score_src = np.asarray(artifact["motion_score"], dtype=np.float64)
        elif ranking == "random":
            score_src = allocation_score_src
        else:
            raise ValueError(f"ranking {ranking!r} does not consume an AP score artifact")
        importance_src = np.asarray(artifact["importance_score"], dtype=np.float64)
        byte_src = np.asarray(artifact["estimated_time_bytes"], dtype=np.int64)
        frozen_mask_src = {
            name: np.asarray(artifact[name], dtype=np.bool_)
            for name in (
                "official_retain_mask",
                "official_factor0_mask",
                "official_active_mask",
                "ap_retain_mask",
                "ap_active_mask",
                "ap_class_mask",
            )
        }
        factor0_activation_value = float(np.asarray(artifact["factor0_activation_value"]).item())
        factor3_activation_value = float(np.asarray(artifact["factor3_activation_value"]).item())
        estimator_version = str(np.asarray(artifact["estimator_version"]).item())
        time_entropy_model_sha256 = str(
            np.asarray(artifact["time_entropy_model_sha256"]).item()
        )
        time_feature_scaling = float(np.asarray(artifact["time_feature_scaling"]).item())
        time_entropy_model_frozen = bool(
            np.asarray(artifact["time_entropy_model_frozen_after_freeze"]).item()
        )
        runtime_manifest_sha256 = str(np.asarray(artifact["runtime_manifest_sha256"]).item())
        normalized_code_tree_sha256 = str(
            np.asarray(artifact["normalized_code_tree_sha256"]).item()
        )
        patch_chain_sha256 = [str(value) for value in np.asarray(artifact["patch_chain_sha256"]).tolist()]

    if (
        eligible_src.shape != (artifact_ids.shape[0],)
        or score_src.shape != (artifact_ids.shape[0],)
        or allocation_score_src.shape != (artifact_ids.shape[0],)
        or importance_src.shape != (artifact_ids.shape[0],)
        or byte_src.shape != (artifact_ids.shape[0],)
    ):
        raise ValueError("malformed AP score artifact arrays")
    if not np.isfinite(importance_src).all() or np.any(importance_src < 0):
        raise ValueError("frozen backbone importance must be finite and nonnegative")
    for name, value in frozen_mask_src.items():
        if value.shape != (artifact_ids.shape[0],):
            raise ValueError(f"malformed frozen allocation mask: {name}")
    if np.any(byte_src < 0):
        raise ValueError("estimated_time_bytes must be nonnegative")
    if factor0_activation_value <= 0 or factor3_activation_value <= 0:
        raise ValueError("frozen factor activation values must be positive")
    if estimator_version != "h007.conditional_gaussian_per_row_bits.v1":
        raise ValueError("unsupported estimated-byte estimator version")
    if (
        len(time_entropy_model_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in time_entropy_model_sha256.lower())
        or time_feature_scaling <= 0
    ):
        raise ValueError("malformed frozen estimated-byte estimator provenance")
    if not time_entropy_model_frozen:
        raise ValueError("estimated-byte entropy model was not frozen after allocation")
    expected_runtime = dict(expected_runtime_provenance)
    if expected_runtime.get("schema") != "h007.ap_gifstream.runtime_provenance.v1":
        raise ValueError("score loader lacks verified runtime provenance")
    if runtime_manifest_sha256 != expected_runtime.get("manifest_sha256"):
        raise ValueError("score artifact runtime-manifest provenance mismatch")
    if normalized_code_tree_sha256 != expected_runtime.get(
        "normalized_code_tree", {}
    ).get("sha256"):
        raise ValueError("score artifact normalized code-tree provenance mismatch")
    if patch_chain_sha256 != list(expected_runtime.get("patch_sha256", [])):
        raise ValueError("score artifact patch-chain provenance mismatch")
    source_index = _rows_to_index(artifact_ids)
    aligned_indices = []
    for row in current_ids.tolist():
        key = tuple(int(v) for v in row)
        if key not in source_index:
            raise ValueError(f"current anchor absent from frozen score artifact: {key}")
        aligned_indices.append(source_index[key])
    if len(aligned_indices) != len(source_index):
        raise ValueError("score artifact/current anchor sets differ; silent subset alignment is forbidden")

    eligible = eligible_src[np.asarray(aligned_indices, dtype=np.int64)]
    score = score_src[np.asarray(aligned_indices, dtype=np.int64)]
    importance = importance_src[np.asarray(aligned_indices, dtype=np.int64)]
    estimated_bytes = byte_src[np.asarray(aligned_indices, dtype=np.int64)]
    aligned_masks = {
        name: value[np.asarray(aligned_indices, dtype=np.int64)]
        for name, value in frozen_mask_src.items()
    }
    if ranking in {"path", "motion"} and not np.array_equal(
        score_src[eligible_src], allocation_score_src[eligible_src]
    ):
        raise ValueError("frozen allocation score disagrees with declared ranking")
    if not np.all(np.isfinite(score[eligible])):
        raise ValueError("eligible AP scores must be finite")
    score[~eligible] = -np.inf

    raw = Path(path).read_bytes()
    audit = {
        "schema": schema,
        "score_artifact": os.path.abspath(path),
        "score_artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "ranking": ranking,
        "variant": artifact_variant,
        "protected_fraction": protected_fraction,
        "q_ap_multiplier": q_ap_multiplier,
        "q_bg_multiplier": q_bg_multiplier,
        "random_seed": int(random_seed) if ranking == "random" else None,
        "estimator_version": estimator_version,
        "time_entropy_model_sha256": time_entropy_model_sha256,
        "time_feature_scaling": time_feature_scaling,
        "time_entropy_model_frozen_after_freeze": time_entropy_model_frozen,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "normalized_code_tree_sha256": normalized_code_tree_sha256,
        "patch_chain_sha256": patch_chain_sha256,
        "anchor_count": int(current_ids.shape[0]),
        "eligible_count": int(eligible.sum()),
    }
    frozen_allocation = {
        name: torch.from_numpy(value).to(device=anchors.device)
        for name, value in aligned_masks.items()
    }
    frozen_allocation.update(
        {
            "factor0_activation_value": factor0_activation_value,
            "factor3_activation_value": factor3_activation_value,
        }
    )
    if int(frozen_allocation["official_retain_mask"].sum()) != int(
        frozen_allocation["ap_retain_mask"].sum()
    ):
        raise ValueError("frozen whole-anchor allocation changed retained count")
    if torch.any(
        frozen_allocation["official_active_mask"]
        & ~frozen_allocation["official_retain_mask"]
    ):
        raise ValueError("frozen official active mask escapes official retain mask")
    if torch.any(
        frozen_allocation["ap_active_mask"]
        & ~frozen_allocation["ap_retain_mask"]
    ):
        raise ValueError("frozen AP active mask escapes AP retain mask")
    if torch.any(
        frozen_allocation["ap_class_mask"]
        & ~frozen_allocation["ap_active_mask"]
    ):
        raise ValueError("frozen AP class contains a temporally inactive row")
    if not torch.equal(
        frozen_allocation["official_active_mask"],
        frozen_allocation["official_retain_mask"]
        & frozen_allocation["official_factor0_mask"],
    ):
        raise ValueError("frozen official active mask disagrees with factor0/retain masks")
    official_mass = int(
        estimated_bytes[frozen_allocation["official_active_mask"].cpu().numpy()].sum()
    )
    ap_mass = int(estimated_bytes[frozen_allocation["ap_active_mask"].cpu().numpy()].sum())
    if official_mass != ap_mass:
        raise ValueError("frozen temporal allocation changed estimated-byte mass")
    spec = variant_spec(artifact_variant)
    expected_retain, expected_class, _ = build_count_preserving_anchor_allocation(
        frozen_allocation["official_retain_mask"],
        torch.from_numpy(score).to(device=anchors.device, dtype=torch.float64),
        torch.from_numpy(eligible).to(device=anchors.device),
        torch.from_numpy(current_ids).to(device=anchors.device, dtype=torch.int64),
        protected_fraction,
        spec.swap,
        importance=torch.from_numpy(importance).to(
            device=anchors.device, dtype=torch.float64
        ),
    )
    if not torch.equal(expected_retain, frozen_allocation["ap_retain_mask"]):
        raise ValueError("frozen whole-anchor allocation is not reproducible")
    expected_active, expected_temporal_class, _ = build_equal_estimated_byte_allocation(
        frozen_allocation["official_active_mask"],
        torch.from_numpy(score).to(device=anchors.device, dtype=torch.float64),
        torch.from_numpy(eligible).to(device=anchors.device),
        torch.from_numpy(current_ids).to(device=anchors.device, dtype=torch.int64),
        torch.from_numpy(estimated_bytes).to(device=anchors.device, dtype=torch.int64),
        protected_fraction,
        spec.swap,
        starting_active=(
            frozen_allocation["official_factor0_mask"] & expected_retain
        ),
        retain_mask=expected_retain,
        importance=torch.from_numpy(importance).to(
            device=anchors.device, dtype=torch.float64
        ),
    )
    if not torch.equal(expected_class, expected_temporal_class) or not torch.equal(
        expected_class, frozen_allocation["ap_class_mask"]
    ):
        raise ValueError("frozen AP class mask is not reproducible")
    if not torch.equal(expected_active, frozen_allocation["ap_active_mask"]):
        raise ValueError("frozen temporal allocation is not reproducible")
    return (
        torch.from_numpy(score).to(device=anchors.device, dtype=torch.float64),
        torch.from_numpy(eligible).to(device=anchors.device),
        torch.from_numpy(estimated_bytes).to(device=anchors.device, dtype=torch.int64),
        frozen_allocation,
        audit,
    )


def build_equal_estimated_byte_allocation(
    official_active: Tensor,
    scores: Tensor,
    eligible: Tensor,
    canonical_ids: Tensor,
    estimated_bytes: Tensor,
    protected_fraction: float,
    enable_swap: bool,
    starting_active: Optional[Tensor] = None,
    retain_mask: Optional[Tensor] = None,
    max_dp_states: int = 2_000_000,
    importance: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
    """Close final temporal membership to official estimated-byte mass.

    ``starting_active`` may already differ because whole-anchor membership was
    swapped. Protected retained anchors are forced active, then a deterministic
    exact integer subset-sum promotes or demotes eligible unprotected rows until
    the official frozen byte mass is restored.  The final mask must be a subset
    of ``retain_mask``.  No count-based fallback is allowed.

    ``importance`` (the frozen backbone importance estimate) only refines the
    demotion-side candidate order: lowest score first, then lowest importance,
    then canonical ID.  The protected-class selection never sees it.
    """

    n = official_active.numel()
    if (
        scores.shape != (n,)
        or eligible.shape != (n,)
        or estimated_bytes.shape != (n,)
        or canonical_ids.shape != (n, 3)
    ):
        raise ValueError("estimated-byte allocation arrays disagree")
    if torch.any(estimated_bytes < 0):
        raise ValueError("estimated temporal bytes must be nonnegative")
    if not (0.0 < float(protected_fraction) <= 1.0):
        raise ValueError("protected_fraction must be in (0,1]")

    active_np = official_active.detach().to(torch.bool).cpu().numpy()
    starting_np = (
        active_np.copy()
        if starting_active is None
        else starting_active.detach().to(torch.bool).cpu().numpy()
    )
    retain_np = (
        np.ones((n,), dtype=np.bool_)
        if retain_mask is None
        else retain_mask.detach().to(torch.bool).cpu().numpy()
    )
    if starting_np.shape != (n,) or retain_np.shape != (n,):
        raise ValueError("starting_active/retain_mask shape mismatch")
    if np.any(starting_np & ~retain_np):
        raise ValueError("starting temporal allocation is not a subset of retained anchors")
    eligible_np = eligible.detach().to(torch.bool).cpu().numpy()
    scores_np = scores.detach().to(torch.float64).cpu().numpy()
    ids_np = canonical_ids.detach().to(torch.int64).cpu().numpy()
    costs_np = estimated_bytes.detach().to(torch.int64).cpu().numpy()
    importance_np = _validate_donor_secondary(importance, n)
    eligible_indices = np.flatnonzero(eligible_np)
    if eligible_indices.size == 0:
        raise ValueError("AP eligible universe is empty")
    k = max(1, int(math.ceil(float(protected_fraction) * eligible_indices.size)))
    ranked_eligible = eligible_indices[
        _lexicographic_rank(ids_np[eligible_indices], scores_np[eligible_indices], descending=True)
    ]
    ap_class_np = np.zeros(n, dtype=np.bool_)
    ap_class_np[ranked_eligible[:k]] = True

    output_np = starting_np.copy()
    if enable_swap:
        protected_retain = ap_class_np & retain_np
        if int(protected_retain.sum()) != int(ap_class_np.sum()):
            raise ValueError("whole-anchor allocation failed to retain every AP-class anchor")
        output_np[protected_retain] = True

    target_total = int(costs_np[active_np].sum())
    current_total = int(costs_np[output_np].sum())

    def exact_subset(candidates: np.ndarray, target: int) -> np.ndarray:
        if target == 0:
            return np.empty(0, dtype=np.int64)
        parents: Dict[int, Tuple[int, int]] = {0: (-1, -1)}
        for candidate in candidates.tolist():
            cost = int(costs_np[candidate])
            additions: Dict[int, Tuple[int, int]] = {}
            for subtotal in sorted(parents.keys(), reverse=True):
                new_total = subtotal + cost
                if new_total > target or new_total in parents or new_total in additions:
                    continue
                additions[new_total] = (subtotal, candidate)
            parents.update(additions)
            if len(parents) > int(max_dp_states):
                raise ValueError("estimated-byte subset DP exceeded frozen state cap")
            if target in parents:
                break
        if target not in parents:
            raise ValueError(
                f"no exact estimated-byte adjustment subset for {target} bytes"
            )
        selected = []
        cursor = target
        while cursor:
            previous, candidate = parents[cursor]
            selected.append(candidate)
            cursor = previous
        return np.asarray(selected, dtype=np.int64)

    if not enable_swap and current_total != target_total:
        raise ValueError("non-swap temporal allocation changed estimated-byte mass")
    if enable_swap and current_total > target_total:
        candidates = np.flatnonzero(
            output_np & eligible_np & ~ap_class_np & (costs_np > 0)
        )
        candidates = candidates[
            _lexicographic_rank(
                ids_np[candidates],
                scores_np[candidates],
                descending=False,
                secondary=None
                if importance_np is None
                else importance_np[candidates],
            )
        ]
        output_np[exact_subset(candidates, current_total - target_total)] = False
    elif enable_swap and current_total < target_total:
        candidates = np.flatnonzero(
            ~output_np & retain_np & eligible_np & ~ap_class_np & (costs_np > 0)
        )
        candidates = candidates[
            _lexicographic_rank(
                ids_np[candidates], scores_np[candidates], descending=True
            )
        ]
        output_np[exact_subset(candidates, target_total - current_total)] = True

    if np.any(output_np & ~retain_np):
        raise AssertionError("final temporal allocation escaped whole-retain mask")
    final_total = int(costs_np[output_np].sum())
    if final_total != target_total:
        raise AssertionError("temporal swap changed frozen estimated-byte mass")
    promoted = np.flatnonzero(output_np & ~active_np)
    demoted = np.flatnonzero(active_np & ~output_np)
    promoted_mass = int(costs_np[promoted].sum())
    demoted_mass = int(costs_np[demoted].sum())
    audit = {
        "official_active_count": int(active_np.sum()),
        "ap_active_count": int(output_np.sum()),
        "eligible_count": int(eligible_np.sum()),
        "protected_count": int(ap_class_np.sum()),
        "promoted_count": int(promoted.size),
        "demoted_count": int(demoted.size),
        "promoted_estimated_bytes": promoted_mass,
        "demoted_estimated_bytes": demoted_mass,
        "official_estimated_bytes": target_total,
        "starting_estimated_bytes": current_total,
        "ap_estimated_bytes": final_total,
        "estimated_byte_delta": final_total - target_total,
        "protected_fraction": float(protected_fraction),
        "budget_definition": "exact_frozen_integer_estimated_time_bytes",
        "donor_ranking_keys": (
            ["score_asc", "backbone_importance_asc", "canonical_id"]
            if importance_np is not None
            else ["score_asc", "canonical_id"]
        ),
        "real_zip_match_still_required": True,
        "promoted_canonical_ids": ids_np[promoted].tolist(),
        "demoted_canonical_ids": ids_np[demoted].tolist(),
    }
    return (
        torch.from_numpy(output_np).to(device=official_active.device),
        torch.from_numpy(ap_class_np).to(device=official_active.device),
        audit,
    )


def build_equal_stream_budget_allocation(
    official_active: Tensor,
    scores: Tensor,
    eligible: Tensor,
    canonical_ids: Tensor,
    protected_fraction: float,
    enable_swap: bool,
    importance: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
    """Build a deterministic top-score class and count-exact promote/demote swap.

    Every temporal-feature row has the same GOP x channel shape.  The operator
    therefore preserves the exact number of active stream rows.  Entropy-coded
    byte equality is *not* inferred from this property; the actual deterministic
    ZIP feedback loop remains mandatory.

    ``importance`` (the frozen backbone importance estimate) only refines the
    donor order: lowest score demoted first, ties broken by lowest importance,
    then canonical ID.  The protected-class selection never sees it.
    """

    for name, value in {
        "official_active": official_active,
        "scores": scores,
        "eligible": eligible,
    }.items():
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
    n = official_active.numel()
    if scores.numel() != n or eligible.numel() != n or canonical_ids.shape != (n, 3):
        raise ValueError("allocation arrays do not share the same anchor dimension")
    if not (0.0 < float(protected_fraction) <= 1.0):
        raise ValueError("protected_fraction must be in (0,1]")

    active_np = official_active.detach().to(torch.bool).cpu().numpy()
    eligible_np = eligible.detach().to(torch.bool).cpu().numpy()
    scores_np = scores.detach().to(torch.float64).cpu().numpy()
    ids_np = canonical_ids.detach().to(torch.int64).cpu().numpy()
    importance_np = _validate_donor_secondary(importance, n)
    eligible_indices = np.flatnonzero(eligible_np)
    if eligible_indices.size == 0:
        raise ValueError("AP eligible universe is empty")
    k = max(1, int(math.ceil(float(protected_fraction) * eligible_indices.size)))
    ranked_eligible = eligible_indices[
        _lexicographic_rank(ids_np[eligible_indices], scores_np[eligible_indices], descending=True)
    ]
    ap_class_np = np.zeros(n, dtype=np.bool_)
    ap_class_np[ranked_eligible[:k]] = True

    output_np = active_np.copy()
    promoted = np.flatnonzero(ap_class_np & ~active_np) if enable_swap else np.empty(0, dtype=np.int64)
    demoted = np.empty(0, dtype=np.int64)
    if promoted.size:
        candidates = np.flatnonzero(active_np & eligible_np & ~ap_class_np)
        if candidates.size < promoted.size:
            raise ValueError(
                f"cannot preserve stream-row budget: {promoted.size} promotions but only "
                f"{candidates.size} eligible demotion candidates"
            )
        # Lowest score is demoted first; ties fall to the lowest frozen backbone
        # importance, and the canonical ID is the final deterministic tie break.
        ranked_candidates = candidates[
            _lexicographic_rank(
                ids_np[candidates],
                scores_np[candidates],
                descending=False,
                secondary=None
                if importance_np is None
                else importance_np[candidates],
            )
        ]
        demoted = ranked_candidates[: promoted.size]
        output_np[promoted] = True
        output_np[demoted] = False

    if int(output_np.sum()) != int(active_np.sum()):
        raise AssertionError("AP allocation changed the active temporal-stream row budget")
    audit = {
        "official_active_count": int(active_np.sum()),
        "ap_active_count": int(output_np.sum()),
        "eligible_count": int(eligible_np.sum()),
        "protected_count": int(ap_class_np.sum()),
        "promoted_count": int(promoted.size),
        "demoted_count": int(demoted.size),
        "protected_fraction": float(protected_fraction),
        "budget_definition": "exact_active_temporal_stream_rows",
        "donor_ranking_keys": (
            ["score_asc", "backbone_importance_asc", "canonical_id"]
            if importance_np is not None
            else ["score_asc", "canonical_id"]
        ),
        "entropy_byte_match_deferred_to_deterministic_zip": True,
        "promoted_canonical_ids": ids_np[promoted].tolist(),
        "demoted_canonical_ids": ids_np[demoted].tolist(),
    }
    return (
        torch.from_numpy(output_np).to(device=official_active.device),
        torch.from_numpy(ap_class_np).to(device=official_active.device),
        audit,
    )


def build_count_preserving_anchor_allocation(
    official_retain: Tensor,
    scores: Tensor,
    eligible: Tensor,
    canonical_ids: Tensor,
    protected_fraction: float,
    enable_swap: bool,
    importance: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
    """Protect/swap whole anchors while preserving the exact retained count."""
    retain, ap_class, audit = build_equal_stream_budget_allocation(
        official_retain,
        scores,
        eligible,
        canonical_ids,
        protected_fraction,
        enable_swap,
        importance=importance,
    )
    audit = dict(audit)
    audit["official_retain_count"] = audit.pop("official_active_count")
    audit["ap_retain_count"] = audit.pop("ap_active_count")
    audit["budget_definition"] = "exact_whole_anchor_retain_count"
    audit.pop("entropy_byte_match_deferred_to_deterministic_zip", None)
    return retain, ap_class, audit


def pack_bool_mask(mask: Tensor) -> bytes:
    arr = mask.detach().to(torch.bool).cpu().numpy().astype(np.uint8, copy=False)
    return np.packbits(arr, bitorder="little").tobytes()


def unpack_bool_mask(payload: bytes, count: int, device: Optional[torch.device] = None) -> Tensor:
    if count < 0:
        raise ValueError("mask count must be nonnegative")
    if len(payload) != (count + 7) // 8:
        raise ValueError("packed AP mask byte extent is noncanonical")
    if count % 8 and payload and payload[-1] & ~((1 << (count % 8)) - 1):
        raise ValueError("packed AP mask tail bits are nonzero")
    unpacked = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")[:count]
    if unpacked.shape[0] != count:
        raise ValueError("packed AP mask is shorter than declared")
    return torch.from_numpy(unpacked.astype(np.bool_, copy=True)).to(device=device)


def write_ap_mask(path: str, mask: Tensor) -> Dict[str, Any]:
    payload = pack_bool_mask(mask)
    Path(path).write_bytes(payload)
    return {
        "path": os.path.basename(path),
        "count": int(mask.numel()),
        "true_count": int(mask.to(torch.bool).sum().item()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bitorder": "little",
    }


def read_ap_mask(compress_dir: str, meta: Mapping[str, Any], device: torch.device) -> Tensor:
    path = os.path.join(compress_dir, str(meta["path"]))
    payload = Path(path).read_bytes()
    if len(payload) != int(meta["bytes"]):
        raise ValueError("AP mask byte count mismatch")
    if hashlib.sha256(payload).hexdigest() != str(meta["sha256"]):
        raise ValueError("AP mask SHA-256 mismatch")
    mask = unpack_bool_mask(payload, int(meta["count"]), device=device)
    if int(mask.sum().item()) != int(meta["true_count"]):
        raise ValueError("AP mask true-count mismatch")
    return mask


def write_numpy_sidecar(path: str, array: np.ndarray) -> Dict[str, Any]:
    """Write a non-pickle NumPy sidecar with strict shape/dtype provenance."""

    value = np.asarray(array)
    if value.dtype == np.dtype("O"):
        raise ValueError("object arrays are forbidden in counted sidecars")
    np.save(path, value, allow_pickle=False)
    payload = Path(path).read_bytes()
    return {
        "path": os.path.basename(path),
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def read_numpy_sidecar(compress_dir: str, meta: Mapping[str, Any]) -> np.ndarray:
    path = os.path.join(compress_dir, str(meta["path"]))
    payload = Path(path).read_bytes()
    if len(payload) != int(meta["bytes"]):
        raise ValueError(f"counted NumPy sidecar byte mismatch: {meta['path']}")
    if hashlib.sha256(payload).hexdigest() != str(meta["sha256"]):
        raise ValueError(f"counted NumPy sidecar hash mismatch: {meta['path']}")
    array = np.load(path, allow_pickle=False)
    if list(array.shape) != list(meta["shape"]) or array.dtype.str != str(meta["dtype"]):
        raise ValueError(f"counted NumPy sidecar shape/dtype mismatch: {meta['path']}")
    return array


def write_identity_corrections(
    path: str, decoded_ids: np.ndarray, retained_ids: np.ndarray
) -> Dict[str, Any]:
    """Encode exact pre-quantization IDs as corrections to decoded anchors.

    The anchor stream supplies the base ID for every real encoded row.  This
    sidecar counts only a packed mismatch mask and one base-3 byte per changed
    row.  Each axis correction is restricted to ``{-1, 0, 1}``; larger anchor
    quantization drift is rejected instead of silently changing identity.
    """

    base = np.asarray(decoded_ids)
    target = np.asarray(retained_ids)
    if (
        base.dtype != np.int64
        or target.dtype != np.int64
        or base.ndim != 2
        or base.shape[1:] != (3,)
        or target.shape != base.shape
    ):
        raise ValueError("identity-correction inputs must be matching int64 [N,3] arrays")
    delta = target - base
    mismatch = np.any(delta != 0, axis=1)
    changed = delta[mismatch]
    if changed.size and np.any((changed < -1) | (changed > 1)):
        raise ValueError("decoded anchor identity drift exceeds one canonical voxel")
    mask_payload = np.packbits(
        mismatch.astype(np.uint8, copy=False), bitorder="little"
    ).tobytes()
    codes = (
        (changed[:, 0] + 1) * 9
        + (changed[:, 1] + 1) * 3
        + (changed[:, 2] + 1)
    ).astype(np.uint8, copy=False)
    payload = mask_payload + codes.tobytes()
    Path(path).write_bytes(payload)
    return {
        "schema": AP_IDENTITY_CORRECTION_SCHEMA,
        "path": os.path.basename(path),
        "row_count": int(base.shape[0]),
        "mismatch_count": int(mismatch.sum()),
        "mask_bytes": len(mask_payload),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bitorder": "little",
        "base": "round-decoded-anchor-div-voxel-size",
        "code": "uint8-base3-dx-dy-dz-plus1",
    }


def read_identity_corrections(
    compress_dir: str, meta: Mapping[str, Any], decoded_ids: np.ndarray
) -> np.ndarray:
    """Restore and validate exact retained identities from counted corrections."""

    base = np.asarray(decoded_ids)
    if base.dtype != np.int64 or base.ndim != 2 or base.shape[1:] != (3,):
        raise ValueError("decoded identity base must be int64 [N,3]")
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
    if set(meta) != required or meta.get("schema") != AP_IDENTITY_CORRECTION_SCHEMA:
        raise ValueError("identity-correction metadata are malformed")
    if (
        meta.get("path") != "ap_identity_corrections.bin"
        or meta.get("bitorder") != "little"
        or meta.get("base") != "round-decoded-anchor-div-voxel-size"
        or meta.get("code") != "uint8-base3-dx-dy-dz-plus1"
        or type(meta.get("row_count")) is not int
        or type(meta.get("mismatch_count")) is not int
        or type(meta.get("mask_bytes")) is not int
        or int(meta["row_count"]) != int(base.shape[0])
        or int(meta["mismatch_count"]) < 0
        or int(meta["mismatch_count"]) > int(base.shape[0])
        or int(meta["mask_bytes"]) != (int(base.shape[0]) + 7) // 8
    ):
        raise ValueError("identity-correction dimensions/codec are invalid")
    path = Path(compress_dir) / str(meta["path"])
    payload = path.read_bytes()
    if len(payload) != int(meta["bytes"]) or hashlib.sha256(payload).hexdigest() != str(
        meta["sha256"]
    ):
        raise ValueError("identity-correction byte binding mismatch")
    mask_extent = int(meta["mask_bytes"])
    mask_tensor = unpack_bool_mask(payload[:mask_extent], int(base.shape[0]))
    mismatch = mask_tensor.cpu().numpy().astype(np.bool_, copy=False)
    if int(mismatch.sum()) != int(meta["mismatch_count"]):
        raise ValueError("identity-correction mismatch count differs from packed mask")
    codes = np.frombuffer(payload[mask_extent:], dtype=np.uint8)
    if (
        codes.shape != (int(meta["mismatch_count"]),)
        or np.any(codes > 26)
        or np.any(codes == 13)
    ):
        raise ValueError("identity-correction code extent/value is invalid")
    restored = base.copy()
    code64 = codes.astype(np.int64, copy=False)
    restored[mismatch, 0] += code64 // 9 - 1
    restored[mismatch, 1] += (code64 // 3) % 3 - 1
    restored[mismatch, 2] += code64 % 3 - 1
    if np.unique(restored, axis=0).shape[0] != restored.shape[0]:
        raise ValueError("restored retained canonical IDs are not unique")
    return restored


def recompute_sorted_codec_invariants(
    *,
    official_retain_mask: Tensor,
    official_active_mask: Tensor,
    ap_retain_mask: Tensor,
    expected_ap_active_mask: Tensor,
    expected_ap_class_mask: Tensor,
    source_estimated_bytes: Tensor,
    pre_sort_retained_ids: Tensor,
    pre_sort_estimated_bytes: Tensor,
    sort_indices: Tensor,
    encoded_ids: Tensor,
    encoded_estimated_bytes: Tensor,
    encoded_factors: Tensor,
    real_mask: Tensor,
    padding_mask: Tensor,
    ap_class_mask: Tensor,
) -> Dict[str, Any]:
    """Recompute codec allocation after retention, padding and PLAS ordering."""

    encoded_count = int(encoded_ids.shape[0])
    one_dimensional = {
        "encoded_estimated_bytes": encoded_estimated_bytes,
        "real_mask": real_mask,
        "padding_mask": padding_mask,
        "ap_class_mask": ap_class_mask,
        "sort_indices": sort_indices,
    }
    for name, value in one_dimensional.items():
        if value.shape != (encoded_count,):
            raise ValueError(f"post-PLAS {name} has an invalid shape")
    if encoded_ids.shape != (encoded_count, 3) or encoded_factors.shape != (
        encoded_count,
        4,
    ):
        raise ValueError("post-PLAS ID/factor shapes are malformed")
    if not torch.equal(real_mask, ~padding_mask):
        raise ValueError("explicit real/padding masks are not complements")
    if int(real_mask.sum().item()) != int(ap_retain_mask.sum().item()):
        raise ValueError("real-row count differs from frozen retained-anchor count")
    expected_permutation = torch.arange(encoded_count, device=sort_indices.device)
    if not torch.equal(torch.sort(sort_indices.to(torch.long)).values, expected_permutation):
        raise ValueError("PLAS permutation is not a complete encoded-row permutation")

    pad = encoded_count - int(pre_sort_retained_ids.shape[0])
    if pad < 0 or pre_sort_estimated_bytes.shape != (pre_sort_retained_ids.shape[0],):
        raise ValueError("pre-sort retained ID/estimated-byte shapes disagree")
    sentinel = torch.zeros((pad, 3), dtype=torch.int64, device=encoded_ids.device)
    if pad:
        sentinel[:, 0] = torch.arange(pad, device=encoded_ids.device, dtype=torch.int64)
        sentinel[:, 0] += torch.iinfo(torch.int64).min
    expected_pre_sort_ids = torch.cat([pre_sort_retained_ids, sentinel], dim=0)
    expected_pre_sort_bytes = torch.cat(
        [
            pre_sort_estimated_bytes,
            torch.zeros(pad, dtype=torch.int64, device=encoded_estimated_bytes.device),
        ],
        dim=0,
    )
    if not torch.equal(encoded_ids, expected_pre_sort_ids[sort_indices]):
        raise ValueError("canonical IDs were not transformed by the counted PLAS permutation")
    if not torch.equal(
        encoded_estimated_bytes, expected_pre_sort_bytes[sort_indices]
    ):
        raise ValueError("estimated-byte rows were not transformed by the counted PLAS permutation")
    expected_real = torch.arange(encoded_count, device=real_mask.device) < int(
        pre_sort_retained_ids.shape[0]
    )
    if not torch.equal(real_mask, expected_real[sort_indices]):
        raise ValueError("explicit real/padding mask was not transformed by PLAS")
    factor_whole = encoded_factors[:, 3] > 0
    factor_active = real_mask & (encoded_factors[:, 0] > 0)
    if not torch.equal(factor_whole, real_mask):
        raise ValueError("post-PLAS factor-3 whole mask differs from explicit real mask")
    if torch.any(ap_class_mask & ~real_mask):
        raise ValueError("post-PLAS AP class marks padding rows")

    whole_promoted = ap_retain_mask & ~official_retain_mask
    whole_demoted = official_retain_mask & ~ap_retain_mask
    if int(whole_promoted.sum().item()) != int(whole_demoted.sum().item()):
        raise ValueError("whole-anchor count swap is not balanced")
    source_shape = official_retain_mask.shape
    if any(
        value.shape != source_shape
        for value in (
            official_active_mask,
            ap_retain_mask,
            expected_ap_active_mask,
            expected_ap_class_mask,
            source_estimated_bytes,
        )
    ):
        raise ValueError("score-derived whole/active masks disagree in shape")
    if torch.any(official_active_mask & ~official_retain_mask):
        raise ValueError("score-derived official active mask escapes retention")
    expected_pre_sort_active = expected_ap_active_mask[ap_retain_mask]
    expected_pre_sort_class = expected_ap_class_mask[ap_retain_mask]
    expected_encoded_active = torch.cat(
        [
            expected_pre_sort_active,
            torch.zeros(pad, dtype=torch.bool, device=real_mask.device),
        ],
        dim=0,
    )[sort_indices]
    expected_encoded_class = torch.cat(
        [
            expected_pre_sort_class,
            torch.zeros(pad, dtype=torch.bool, device=real_mask.device),
        ],
        dim=0,
    )[sort_indices]
    if not torch.equal(factor_active, expected_encoded_active):
        raise ValueError("factor-derived active mask differs from score/PLAS/padding recomputation")
    if not torch.equal(ap_class_mask, expected_encoded_class):
        raise ValueError("AP class mask differs from score/PLAS/padding recomputation")
    official_mass = int(source_estimated_bytes[official_active_mask].sum().item())
    ap_mass = int(encoded_estimated_bytes[factor_active].sum().item())
    if official_mass != ap_mass:
        raise ValueError("post-PLAS AP estimated mass differs from score-derived official mass")
    return {
        "encoded_row_count": encoded_count,
        "real_row_count": int(real_mask.sum().item()),
        "padding_row_count": int(padding_mask.sum().item()),
        "active_row_count": int(factor_active.sum().item()),
        "ap_class_real_count": int((ap_class_mask & real_mask).sum().item()),
        "whole_promoted_count": int(whole_promoted.sum().item()),
        "whole_demoted_count": int(whole_demoted.sum().item()),
        "official_estimated_time_bytes": official_mass,
        "ap_estimated_time_bytes": ap_mass,
        "plas_permutation_complete": True,
        "id_order_definition": "counted_corrections_restore_prequantization_ids_in_encoded_real_row_order",
    }


def file_byte_census(root: str) -> Dict[str, Any]:
    root_path = Path(root)
    files = []
    for path in sorted(p for p in root_path.rglob("*") if p.is_file()):
        if path.is_symlink():
            raise ValueError(f"symlink forbidden in counted container: {path}")
        payload = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(root_path).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "schema": "h007.container_byte_census.v1",
        "root": root_path.name,
        "file_count": len(files),
        "raw_bytes": int(sum(row["bytes"] for row in files)),
        "files": files,
    }


def deterministic_zip_directory(root: str, output_zip: str) -> Dict[str, Any]:
    """ZIP a container with sorted names and frozen metadata, then report real bytes."""

    root_path = Path(root)
    output_path = Path(output_zip)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in root_path.rglob("*") if p.is_file())
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            if path.is_symlink():
                raise ValueError(f"symlink forbidden in counted container: {path}")
            info = zipfile.ZipInfo(path.relative_to(root_path).as_posix(), DETERMINISTIC_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    payload = output_path.read_bytes()
    return {
        "schema": "h007.deterministic_zip.v1",
        "archive": output_path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "file_count": len(paths),
        "timestamp": list(DETERMINISTIC_ZIP_EPOCH),
        "compression": "ZIP_DEFLATED level 9",
    }


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def variant_metadata(name: str) -> Dict[str, Any]:
    return {"name": name, **asdict(variant_spec(name))}
