"""Shared path-state reconstruction contract for AP-GIFStream.

The training simulator, real encoder, and clean decoder must agree on three
things that affect a persistent path:

* the retained-row KNN graph,
* the rows whose path inputs receive the protected precision family, and
* the reconstructed values of the four factor channels.

This module contains the deterministic, side-effect-free parts of that
contract.  Stream I/O remains in ``gifstream_end2end_compression.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch import Tensor


PATH_CONTRACT_SCHEMA = "h007.ap_gifstream.path_contract.v1"

# Patch-bound development operating point.  These are separate from the
# temporal AP/background multipliers frozen in the score artifact: factors are
# substantially more path-sensitive, while anchor features only need the
# one-hop dependency family.
FACTOR_PROTECTED_MULTIPLIER = 1.0 / 256.0
FACTOR_BACKGROUND_MULTIPLIER = 1.0 / 64.0
ANCHOR_FEATURE_PROTECTED_MULTIPLIER = 0.25
ANCHOR_FEATURE_BACKGROUND_MULTIPLIER = 1.0


def _validate_row_mask(mask: Tensor, row_count: int, name: str, device: torch.device) -> Tensor:
    mask = mask.to(device=device, dtype=torch.bool)
    if mask.shape != (row_count,):
        raise ValueError(f"{name} must have shape [{row_count}]")
    return mask


def deterministic_knn_indices(
    anchors: Tensor,
    count: int,
    *,
    canonical_ids: Optional[Tensor] = None,
    allow_duplicate_rows: bool = False,
) -> Tensor:
    """Return row-order-invariant KNN indices with an exact tie rule.

    KDTree query order alone is not a sufficient contract on a voxel lattice:
    more than ``count`` candidates may lie at the same boundary distance.  We
    therefore query the full boundary-radius set and rank it by
    ``(squared-distance, canonical-id-x, canonical-id-y, canonical-id-z)``.
    The result is finally mapped back to the caller's row order.
    """

    if anchors.ndim != 2 or anchors.shape[1] != 3:
        raise ValueError("anchors must be [N,3]")
    row_count = int(anchors.shape[0])
    if type(count) is not int or count <= 0 or row_count <= count:
        raise ValueError("KNN count must be positive and smaller than the row count")
    if not torch.isfinite(anchors).all():
        raise ValueError("anchors contain nonfinite values")

    if canonical_ids is None:
        # The lexicographic tie authority need only be stable under row
        # permutation.  Float64 coordinate bytes provide that authority when
        # an explicit voxel identity is unavailable during ordinary training.
        points64 = anchors.detach().to(torch.float64).cpu().numpy()
        unique = np.unique(points64, axis=0)
        if unique.shape[0] != row_count and not allow_duplicate_rows:
            # Rendering-only callers may opt in to duplicates (the official
            # codec's lossy anchor round-trip can collapse two anchors onto
            # one voxel); the identity-bearing contract paths stay strict.
            raise ValueError("deterministic KNN requires unique anchor rows")
        identity = points64
    else:
        if canonical_ids.dtype != torch.int64 or canonical_ids.shape != (row_count, 3):
            raise ValueError("canonical_ids must be int64 [N,3]")
        if int(torch.unique(canonical_ids, dim=0).shape[0]) != row_count:
            raise ValueError("canonical KNN identities are not unique")
        identity = canonical_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        points64 = anchors.detach().to(torch.float64).cpu().numpy()

    order = np.lexsort((identity[:, 2], identity[:, 1], identity[:, 0]))
    sorted_points = points64[order]
    sorted_identity = identity[order]
    tree = KDTree(sorted_points)

    # First find the boundary distance.  Then include every exact-distance tie
    # at that boundary before applying the canonical lexicographic rule.
    distances, _ = tree.query(sorted_points, k=count + 1)
    boundary = np.nextafter(distances[:, count], np.inf)
    radius_rows = tree.query_radius(sorted_points, r=boundary)
    result_sorted = np.empty((row_count, count), dtype=np.int64)
    for row, candidates in enumerate(radius_rows):
        candidates = np.asarray(candidates, dtype=np.int64)
        candidates = candidates[candidates != row]
        if candidates.size < count:
            raise AssertionError("KNN boundary query returned too few non-self rows")
        delta = sorted_points[candidates] - sorted_points[row]
        squared_distance = np.einsum("ij,ij->i", delta, delta)
        candidate_ids = sorted_identity[candidates]
        # The trailing candidate-index key only ever decides between rows the
        # identity keys cannot separate (exact duplicates); with unique rows
        # it is unreachable and the ranking is unchanged.
        ranked = np.lexsort(
            (
                candidates,
                candidate_ids[:, 2],
                candidate_ids[:, 1],
                candidate_ids[:, 0],
                squared_distance,
            )
        )
        result_sorted[row] = candidates[ranked[:count]]

    # ``order[sorted_row]`` is the original row.  Populate both dimensions in
    # the original caller order.
    result = np.empty_like(result_sorted)
    result[order] = order[result_sorted]
    output = torch.from_numpy(result).to(device=anchors.device, dtype=torch.int64)
    if output.shape != (row_count, count):
        raise AssertionError("deterministic KNN returned an invalid shape")
    self_rows = torch.arange(row_count, device=anchors.device).unsqueeze(1)
    if torch.any(output == self_rows):
        raise AssertionError("deterministic KNN retained a self edge")
    return output


def build_path_input_precision_mask(
    anchors: Tensor,
    protected_mask: Tensor,
    retain_mask: Tensor,
    count: int,
    *,
    canonical_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Expand protected rows to the one-hop retained KNN dependency closure."""

    row_count = int(anchors.shape[0])
    protected = _validate_row_mask(
        protected_mask, row_count, "protected_mask", anchors.device
    )
    retained = _validate_row_mask(retain_mask, row_count, "retain_mask", anchors.device)
    if torch.any(protected & ~retained):
        raise ValueError("protected rows escape the retained universe")
    retained_rows = torch.nonzero(retained, as_tuple=False).flatten()
    if int(retained_rows.numel()) <= count:
        raise ValueError("retained universe is too small for the frozen KNN count")
    retained_anchors = anchors[retained_rows]
    retained_ids = canonical_ids[retained_rows] if canonical_ids is not None else None
    retained_knn = deterministic_knn_indices(
        retained_anchors, count, canonical_ids=retained_ids
    )
    retained_protected = protected[retained_rows]
    precision_retained = retained_protected.clone()
    if torch.any(retained_protected):
        neighbor_rows = retained_knn[retained_protected].reshape(-1)
        precision_retained[neighbor_rows] = True
    precision = torch.zeros(row_count, dtype=torch.bool, device=anchors.device)
    precision[retained_rows] = precision_retained
    if torch.any(protected & ~precision):
        raise AssertionError("path-input closure lost a protected row")
    if torch.any(precision & ~retained):
        raise AssertionError("path-input closure escaped retained rows")
    return precision, retained_knn


def build_codec_knn_indices(
    anchors: Tensor,
    retain_mask: Tensor,
    count: int,
    *,
    canonical_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Build the graph seen by retained rows in the real decoded container.

    Training keeps zero-retention rows in tensor storage until the real codec
    runs.  Those rows must not remain eligible as KNN context for retained
    rows, because the decoder never receives them.  Non-retained query rows are
    left on the full graph only to keep training tensor shapes unchanged; their
    rendered contribution is zero by the factor-3 contract.
    """

    row_count = int(anchors.shape[0])
    retained = _validate_row_mask(
        retain_mask, row_count, "retain_mask", anchors.device
    )
    retained_rows = torch.nonzero(retained, as_tuple=False).flatten()
    if int(retained_rows.numel()) <= count:
        raise ValueError("retained universe is too small for the frozen KNN count")
    full_knn = deterministic_knn_indices(
        anchors, count, canonical_ids=canonical_ids
    )
    retained_ids = (
        canonical_ids[retained_rows] if canonical_ids is not None else None
    )
    retained_knn = deterministic_knn_indices(
        anchors[retained_rows], count, canonical_ids=retained_ids
    )
    codec_graph = full_knn.clone()
    codec_graph[retained_rows] = retained_rows[retained_knn]
    if torch.any(~retained[codec_graph[retained_rows]]):
        raise AssertionError("codec KNN graph depends on an unretained row")
    return codec_graph, retained_rows, retained_knn


def retained_knn_graph_sha256(
    retained_canonical_ids: Tensor, retained_knn: Tensor
) -> str:
    """Hash a retained KNN graph independently of its tensor row order."""

    if (
        retained_canonical_ids.dtype != torch.int64
        or retained_canonical_ids.ndim != 2
        or retained_canonical_ids.shape[1] != 3
    ):
        raise ValueError("retained canonical IDs must be int64 [N,3]")
    row_count = int(retained_canonical_ids.shape[0])
    if (
        retained_knn.dtype != torch.int64
        or retained_knn.ndim != 2
        or retained_knn.shape[0] != row_count
        or retained_knn.shape[1] <= 0
    ):
        raise ValueError("retained KNN graph shape/dtype is malformed")
    if torch.any(retained_knn < 0) or torch.any(retained_knn >= row_count):
        raise ValueError("retained KNN graph contains an out-of-range row")
    ids = (
        retained_canonical_ids.detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    graph = retained_knn.detach().cpu().numpy().astype(np.int64, copy=False)
    order = np.lexsort((ids[:, 2], ids[:, 1], ids[:, 0]))
    canonical_rows = []
    for source in order:
        canonical_rows.append(
            np.concatenate(
                [
                    ids[source],
                    ids[graph[source]].reshape(-1),
                ]
            )
        )
    payload = (
        np.stack(canonical_rows, axis=0)
        .astype("<i8", copy=False)
        .tobytes(order="C")
    )
    header = json.dumps(
        {
            "schema": PATH_CONTRACT_SCHEMA,
            "row_count": row_count,
            "knn_count": int(retained_knn.shape[1]),
            "row": "source-canonical-id+ordered-neighbor-canonical-ids",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(payload)
    return digest.hexdigest()


def row_quantization_scaling(
    base_scaling: float,
    precision_mask: Tensor,
    protected_multiplier: float,
    background_multiplier: float,
    *,
    rank: int,
    dtype: torch.dtype,
) -> Tensor:
    if not math.isfinite(float(base_scaling)) or float(base_scaling) <= 0:
        raise ValueError("base scaling must be finite and positive")
    if not (0.0 < float(protected_multiplier) <= 1.0):
        raise ValueError("protected multiplier must be in (0,1]")
    if not math.isfinite(float(background_multiplier)) or float(background_multiplier) <= 0:
        raise ValueError("background multiplier must be finite and positive")
    if rank < 1:
        raise ValueError("rank must be positive")
    shape = (precision_mask.shape[0],) + (1,) * (rank - 1)
    protected = torch.full(
        shape,
        float(base_scaling) * float(protected_multiplier),
        dtype=dtype,
        device=precision_mask.device,
    )
    background = torch.full(
        shape,
        float(base_scaling) * float(background_multiplier),
        dtype=dtype,
        device=precision_mask.device,
    )
    return torch.where(precision_mask.view(shape), protected, background)


def ste_row_quantize(
    values: Tensor,
    precision_mask: Tensor,
    base_scaling: float,
    protected_multiplier: float,
    background_multiplier: float,
) -> Tensor:
    precision = _validate_row_mask(
        precision_mask, int(values.shape[0]), "precision_mask", values.device
    )
    scaling = row_quantization_scaling(
        base_scaling,
        precision,
        protected_multiplier,
        background_multiplier,
        rank=values.ndim,
        dtype=values.dtype,
    )
    rounded = torch.round(values / scaling) * scaling
    return (rounded - values).detach() + values


def factor_distribution(
    entropy_model: torch.nn.Module, condition: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    distribution = entropy_model.model(condition)
    if distribution.ndim != 2 or distribution.shape[1] % 3:
        raise ValueError("factor entropy distribution is malformed")
    width = distribution.shape[1] // 3
    means, scalings, qs = distribution.split([width, width, width], dim=-1)
    qs = 1.0 + 0.8 * torch.tanh(qs)
    return means, scalings, qs


def normalize_factor_semantics(
    activated: Tensor,
    active_mask: Tensor,
    real_mask: Tensor,
    factor0_activation_value: float,
) -> Tensor:
    """Apply the shared continuous/categorical factor semantics."""

    if activated.ndim != 2 or activated.shape[1] != 4:
        raise ValueError("factors must be [N,4]")
    row_count = int(activated.shape[0])
    active = _validate_row_mask(active_mask, row_count, "active_mask", activated.device)
    real = _validate_row_mask(real_mask, row_count, "real_mask", activated.device)
    if torch.any(active & ~real):
        raise ValueError("active factor rows escape real rows")
    activation = float(factor0_activation_value)
    if not math.isfinite(activation) or activation < 0.1 or activation > 1.0:
        raise ValueError("factor0 activation value must be in [0.1,1]")

    # Built column-wise without in-place writes: the in-place variant bumps the
    # base tensor's version after torch.maximum saves its inputs, which kills
    # the backward pass of the path-alignment loss.
    thresholded = torch.where(activated[:, :3] < 0.1, 0.0, activated[:, :3])
    factor0 = torch.where(
        active,
        torch.maximum(
            thresholded[:, 0],
            torch.full_like(thresholded[:, 0], activation),
        ),
        torch.zeros_like(thresholded[:, 0]),
    )
    factor3 = real.to(dtype=activated.dtype)
    output = torch.cat(
        [factor0.unsqueeze(1), thresholded[:, 1:3], factor3.unsqueeze(1)], dim=1
    )
    if not torch.equal(output[:, 0] > 0, active):
        raise AssertionError("factor0 reconstruction disagrees with active mask")
    if not torch.equal(output[:, 3] > 0, real):
        raise AssertionError("factor3 reconstruction disagrees with real mask")
    return output


def reconstruct_ap_factors(
    raw_factor_logits: Tensor,
    reconstructed_anchor_features: Tensor,
    entropy_model: torch.nn.Module,
    base_scaling: float,
    precision_mask: Tensor,
    protected_multiplier: float,
    background_multiplier: float,
    active_mask: Tensor,
    real_mask: Tensor,
    factor0_activation_value: float,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Reconstruct the exact values represented by the AP factor symbols."""

    if raw_factor_logits.ndim != 2 or raw_factor_logits.shape[1] != 4:
        raise ValueError("raw factor logits must be [N,4]")
    if reconstructed_anchor_features.shape[0] != raw_factor_logits.shape[0]:
        raise ValueError("factor condition row count mismatch")
    precision = _validate_row_mask(
        precision_mask,
        int(raw_factor_logits.shape[0]),
        "factor_precision_mask",
        raw_factor_logits.device,
    )
    _, _, qs = factor_distribution(entropy_model, reconstructed_anchor_features)
    if qs.shape != raw_factor_logits.shape:
        raise ValueError("adaptive factor Q shape mismatch")
    activated = torch.sigmoid(raw_factor_logits)
    activated = normalize_factor_semantics(
        activated, active_mask, real_mask, factor0_activation_value
    )
    scaling = row_quantization_scaling(
        base_scaling,
        precision,
        protected_multiplier,
        background_multiplier,
        rank=2,
        dtype=activated.dtype,
    )
    symbols = torch.round((activated / scaling) * qs)
    quantized = symbols / qs * scaling
    reconstruction = (quantized - activated).detach() + activated
    reconstruction = normalize_factor_semantics(
        reconstruction, active_mask, real_mask, factor0_activation_value
    )
    if not torch.isfinite(reconstruction).all():
        raise ValueError("nonfinite AP factor reconstruction")
    return reconstruction, {
        "symbols": symbols.detach(),
        "qs": qs.detach(),
        "scaling": scaling.detach(),
    }


def reconstruct_decoded_ap_factors(
    decoded_scaled_factors: Tensor,
    q_scaling: float,
    active_mask: Tensor,
    real_mask: Tensor,
    factor0_activation_value: float,
) -> Tensor:
    if not math.isfinite(float(q_scaling)) or float(q_scaling) <= 0:
        raise ValueError("decoded factor scaling must be finite and positive")
    reconstructed = decoded_scaled_factors * float(q_scaling)
    return normalize_factor_semantics(
        reconstructed, active_mask, real_mask, factor0_activation_value
    )
