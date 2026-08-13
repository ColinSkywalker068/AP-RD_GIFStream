import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch

from gsplat.compression.ap_gifstream import (
    AP_VARIANTS,
    AP_SCORE_SCHEMA,
    build_equal_estimated_byte_allocation,
    build_equal_stream_budget_allocation,
    canonical_voxel_ids,
    deterministic_zip_directory,
    frozen_backbone_importance,
    load_aligned_score_artifact,
    pack_bool_mask,
    read_identity_corrections,
    recompute_sorted_codec_invariants,
    tensor_mapping_sha256,
    unpack_bool_mask,
    write_identity_corrections,
)
from gsplat.compression.h007_clean_runtime import (
    build_decoder_modules,
    instantiate_counted_models,
)
from gsplat.compression import h007_runtime_provenance as provenance
from gsplat.compression.gifstream_end2end_compression import (
    _apply_factor_mask,
    _compress_ap_anchor_features_ar,
    _compress_ap_factors,
    _compress_ap_time_features_ar,
    _decompress_ap_anchor_features_ar,
    _decompress_ap_factors,
    _decompress_ap_time_features_ar,
    _tensor_equal_device_agnostic,
    _validate_counted_retained_identity,
)
from gsplat.compression.h007_path_contract import (
    build_codec_knn_indices,
    build_path_input_precision_mask,
    deterministic_knn_indices,
    reconstruct_ap_factors,
    retained_knn_graph_sha256,
    ste_row_quantize,
)


class _RoundEntropy:
    """CPU-only test double for the official conditional entropy API."""

    def compress(self, x, condition, output_path, adaptive=False):
        torch.save(torch.round(x).cpu(), output_path)

    def decompress(self, condition, output_path, adaptive=False):
        return torch.load(output_path, map_location=condition.device, weights_only=True)


class _UnitAdaptiveRoundEntropy(_RoundEntropy):
    """Adaptive entropy double with a unit reconstruction multiplier."""

    def model(self, condition):
        rows = condition.shape[0]
        zeros = torch.zeros((rows, 4), dtype=condition.dtype, device=condition.device)
        ones = torch.ones_like(zeros)
        return torch.cat([zeros, ones, zeros], dim=-1)


class APGIFStreamCoreTest(unittest.TestCase):
    @staticmethod
    def _runtime_receipt():
        return {
            "schema": "h007.ap_gifstream.runtime_provenance.v1",
            "manifest_sha256": "1" * 64,
            "official_commit": provenance.OFFICIAL_COMMIT,
            "patch_sha256": [
                "2" * 64,
                "3" * 64,
                "4" * 64,
                "6" * 64,
                "7" * 64,
                "8" * 64,
                "9" * 64,
                "a" * 64,
                "b" * 64,
                "c" * 64,
                "d" * 64,
            ],
            "normalized_code_tree": {"sha256": "5" * 64},
        }

    def test_variant_matrix_is_frozen(self):
        self.assertEqual(
            set(AP_VARIANTS),
            {
                "official",
                "random-full",
                "motion-full",
                "path-swap",
                "path-quant",
                "path-swap-quant",
                "ap-gifstream-full",
            },
        )
        self.assertTrue(AP_VARIANTS["path-swap"].swap)
        self.assertFalse(AP_VARIANTS["path-swap"].quant)
        self.assertTrue(AP_VARIANTS["path-quant"].quant)
        self.assertFalse(AP_VARIANTS["path-quant"].swap)

    def test_lossy_factor_decode_restores_counted_categorical_mask(self):
        factors = torch.tensor(
            [[0.0, 1.0], [0.125, 1.0], [0.25, 1.0]], dtype=torch.float32
        )
        target = torch.tensor([True, False, True])
        repaired = _apply_factor_mask(factors, 0, target, 0.0625)
        self.assertEqual(repaired, 2)
        torch.testing.assert_close(
            factors[:, 0], torch.tensor([0.0625, 0.0, 0.25])
        )
        self.assertTrue(torch.equal(factors[:, 0] > 0, target))

    def test_factor_mask_restore_rejects_wrong_row_count(self):
        with self.assertRaisesRegex(ValueError, "invalid shape"):
            _apply_factor_mask(
                torch.zeros((3, 2)), 0, torch.tensor([True, False]), 0.0625
            )

    def test_deterministic_knn_is_row_permutation_invariant_with_ties(self):
        anchors = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        ids = anchors.to(torch.int64)
        first = deterministic_knn_indices(anchors, 2, canonical_ids=ids)
        permutation = torch.tensor([3, 0, 4, 2, 1], dtype=torch.int64)
        second = deterministic_knn_indices(
            anchors[permutation], 2, canonical_ids=ids[permutation]
        )

        def canonical_edges(row_ids, graph):
            return {
                tuple(row_ids[row].tolist()): tuple(
                    tuple(value)
                    for value in row_ids[graph[row]].tolist()
                )
                for row in range(row_ids.shape[0])
            }

        self.assertEqual(
            canonical_edges(ids, first),
            canonical_edges(ids[permutation], second),
        )

    def test_path_input_closure_and_codec_graph_exclude_dropped_rows(self):
        anchors = torch.tensor(
            [[float(index), 0.0, 0.0] for index in range(7)],
            dtype=torch.float32,
        )
        ids = anchors.to(torch.int64)
        retain = torch.tensor([True, False, True, True, True, True, False])
        protected = torch.tensor([False, False, False, True, False, False, False])
        precision, retained_knn = build_path_input_precision_mask(
            anchors, protected, retain, 2, canonical_ids=ids
        )
        codec_graph, retained_rows, codec_retained_knn = build_codec_knn_indices(
            anchors, retain, 2, canonical_ids=ids
        )
        self.assertTrue(torch.equal(retained_knn, codec_retained_knn))
        self.assertTrue(torch.equal(precision & ~retain, torch.zeros_like(retain)))
        self.assertTrue(precision[protected].all())
        self.assertTrue(retain[codec_graph[retained_rows]].all())

    def test_retained_knn_graph_hash_is_row_order_independent(self):
        anchors = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        ids = anchors.to(torch.int64)
        first = deterministic_knn_indices(anchors, 2, canonical_ids=ids)
        first_hash = retained_knn_graph_sha256(ids, first)
        permutation = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
        second = deterministic_knn_indices(
            anchors[permutation], 2, canonical_ids=ids[permutation]
        )
        second_hash = retained_knn_graph_sha256(ids[permutation], second)
        self.assertEqual(first_hash, second_hash)

    def test_ap_factor_symbols_round_trip_all_four_semantics(self):
        activated = torch.tensor(
            [
                [0.2, 0.3, 0.7, 0.8],
                [0.4, 0.6, 0.2, 0.9],
                [0.7, 0.8, 0.4, 0.6],
                [0.9, 0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        logits = torch.logit(activated)
        anchor_features = torch.zeros((4, 3), dtype=torch.float32)
        precision = torch.tensor([True, False, True, False])
        active = torch.tensor([True, False, True, False])
        real = torch.tensor([True, True, True, False])
        entropy = _UnitAdaptiveRoundEntropy()
        expected, _ = reconstruct_ap_factors(
            logits,
            anchor_features,
            entropy,
            0.25,
            precision,
            0.25,
            0.5,
            active,
            real,
            0.125,
        )
        with tempfile.TemporaryDirectory() as tmp:
            meta = _compress_ap_factors(
                tmp,
                "factors",
                logits,
                n_sidelen=2,
                precision_mask=precision,
                active_mask=active,
                real_mask=real,
                factor0_activation_value=0.125,
                protected_multiplier=0.25,
                background_multiplier=0.5,
                expected_reconstruction=expected,
                scaling=0.25,
                anchor_features=anchor_features,
                entropy_model=entropy,
            )
            decoded = _decompress_ap_factors(
                tmp,
                "factors",
                meta,
                anchor_features=anchor_features,
                precision_mask=precision,
                active_mask=active,
                real_mask=real,
                entropy_model=entropy,
                device=torch.device("cpu"),
            )
            torch.testing.assert_close(decoded, expected, rtol=0.0, atol=1e-7)
            self.assertTrue(torch.equal(decoded[:, 0] > 0, active))
            self.assertTrue(torch.equal(decoded[:, 3] > 0, real))
            path = Path(tmp) / "factors_path.bin"
            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "stream binding mismatch"):
                _decompress_ap_factors(
                    tmp,
                    "factors",
                    meta,
                    anchor_features=anchor_features,
                    precision_mask=precision,
                    active_mask=active,
                    real_mask=real,
                    entropy_model=entropy,
                    device=torch.device("cpu"),
                )

    def test_ap_factor_reconstruction_preserves_continuous_gradients(self):
        logits = torch.zeros((3, 4), dtype=torch.float32, requires_grad=True)
        anchor_features = torch.zeros((3, 2), dtype=torch.float32)
        reconstruction, _ = reconstruct_ap_factors(
            logits,
            anchor_features,
            _UnitAdaptiveRoundEntropy(),
            0.25,
            torch.tensor([True, False, True]),
            0.25,
            0.5,
            torch.tensor([True, False, True]),
            torch.tensor([True, True, True]),
            0.125,
        )
        reconstruction[:, 1:3].sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.all(logits.grad[:, 1:3] != 0))

    def test_ap_anchor_feature_two_family_round_trip_matches_simulator(self):
        values = torch.tensor(
            [
                [0.12, 0.62, 1.12, 1.62],
                [0.37, 0.87, 1.37, 1.87],
            ],
            dtype=torch.float32,
        )
        precision = torch.tensor([True, False])
        expected = ste_row_quantize(values, precision, 0.5, 0.25, 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            meta = _compress_ap_anchor_features_ar(
                tmp,
                "anchor_features",
                values,
                n_sidelen=2,
                precision_mask=precision,
                protected_multiplier=0.25,
                background_multiplier=1.0,
                expected_reconstruction=expected,
                scaling=0.5,
                c_channel=2,
                entropy_model=_RoundEntropy(),
            )
            decoded = _decompress_ap_anchor_features_ar(
                tmp,
                "anchor_features",
                meta,
                precision_mask=precision,
                entropy_model=_RoundEntropy(),
                device=torch.device("cpu"),
            )
        torch.testing.assert_close(decoded, expected, rtol=0.0, atol=1e-7)

    def test_tensor_equality_helper_preserves_dtype_and_value(self):
        left = torch.tensor([1, 2], dtype=torch.int64)
        self.assertTrue(_tensor_equal_device_agnostic(left, left.clone()))
        self.assertFalse(
            _tensor_equal_device_agnostic(left, left.to(dtype=torch.int32))
        )
        self.assertFalse(
            _tensor_equal_device_agnostic(left, torch.tensor([1, 3], dtype=torch.int64))
        )

    def test_counted_sidecar_remains_identity_authority_after_lossy_decode(self):
        retained_ids = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int64)
        _validate_counted_retained_identity(retained_ids, decoded_row_count=2)
        with self.assertRaisesRegex(ValueError, "not unique"):
            _validate_counted_retained_identity(
                torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.int64),
                decoded_row_count=2,
            )
        with self.assertRaisesRegex(ValueError, "decoded rows"):
            _validate_counted_retained_identity(retained_ids, decoded_row_count=1)

    def test_score_alignment_uses_ids_not_rows(self):
        anchors = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "scores.npz"
            np.savez(
                artifact,
                schema=np.asarray(AP_SCORE_SCHEMA),
                scene=np.asarray("flame_salmon_1"),
                voxel_size=np.asarray(0.1),
                frame_count=np.asarray(60),
                variant=np.asarray("path-swap"),
                protected_fraction=np.asarray(1 / 3),
                q_ap_multiplier=np.asarray(0.5),
                q_bg_multiplier=np.asarray(1.25),
                random_seed=np.asarray(11),
                canonical_ids=np.asarray([[2, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.int64),
                eligible=np.asarray([True, True, False]),
                path_score=np.asarray([20.0, 0.0, 10.0]),
                motion_score=np.asarray([2.0, 0.0, 1.0]),
                allocation_score=np.asarray([20.0, 0.0, 10.0]),
                importance_score=np.asarray([0.0, 0.0, 0.0]),
                estimated_time_bytes=np.asarray([7, 5, 6], dtype=np.int64),
                official_retain_mask=np.asarray([True, True, True]),
                official_factor0_mask=np.asarray([True, False, True]),
                official_active_mask=np.asarray([True, False, True]),
                ap_retain_mask=np.asarray([True, True, True]),
                ap_active_mask=np.asarray([True, False, True]),
                ap_class_mask=np.asarray([True, False, False]),
                factor0_activation_value=np.asarray(0.125),
                factor3_activation_value=np.asarray(1.0),
                estimator_version=np.asarray("h007.conditional_gaussian_per_row_bits.v1"),
                time_entropy_model_sha256=np.asarray("0" * 64),
                time_feature_scaling=np.asarray(1.0),
                time_entropy_model_frozen_after_freeze=np.asarray(True),
                runtime_manifest_sha256=np.asarray("1" * 64),
                normalized_code_tree_sha256=np.asarray("5" * 64),
                patch_chain_sha256=np.asarray(
                    [
                        "2" * 64,
                        "3" * 64,
                        "4" * 64,
                        "6" * 64,
                        "7" * 64,
                        "8" * 64,
                        "9" * 64,
                        "a" * 64,
                        "b" * 64,
                        "c" * 64,
                        "d" * 64,
                    ]
                ),
            )
            score, eligible, estimated_bytes, allocation, audit = load_aligned_score_artifact(
                str(artifact),
                anchors,
                0.1,
                "path",
                11,
                self._runtime_receipt(),
                "flame_salmon_1",
                expected_variant="path-swap",
            )
        torch.testing.assert_close(score, torch.tensor([0.0, -torch.inf, 20.0], dtype=torch.float64))
        self.assertEqual(eligible.tolist(), [True, False, True])
        self.assertEqual(estimated_bytes.tolist(), [5, 6, 7])
        self.assertEqual(allocation["ap_active_mask"].tolist(), [False, True, True])
        self.assertEqual(audit["eligible_count"], 2)

    def test_duplicate_canonical_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "duplicate canonical"):
            canonical_voxel_ids(torch.tensor([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]]), 0.1)

    def test_swap_preserves_exact_active_stream_rows(self):
        ids = torch.tensor([[i, 0, 0] for i in range(8)], dtype=torch.int64)
        official = torch.tensor([True, True, True, True, False, False, False, False])
        scores = torch.tensor([0.0, 1.0, 2.0, 3.0, 100.0, 90.0, 5.0, 4.0])
        eligible = torch.ones(8, dtype=torch.bool)
        active, ap_class, audit = build_equal_stream_budget_allocation(
            official, scores, eligible, ids, protected_fraction=0.25, enable_swap=True
        )
        self.assertEqual(int(active.sum()), int(official.sum()))
        self.assertEqual(torch.nonzero(ap_class).reshape(-1).tolist(), [4, 5])
        self.assertEqual(torch.nonzero(active & ~official).reshape(-1).tolist(), [4, 5])
        self.assertEqual(torch.nonzero(official & ~active).reshape(-1).tolist(), [0, 1])
        self.assertEqual(audit["promoted_count"], audit["demoted_count"])
        self.assertEqual(audit["donor_ranking_keys"], ["score_asc", "canonical_id"])

    def test_donor_dual_key_demotes_least_important_on_score_tie(self):
        ids = torch.tensor([[i, 0, 0] for i in range(8)], dtype=torch.int64)
        official = torch.tensor([True, True, True, True, False, False, False, False])
        scores = torch.tensor([0.0, 0.0, 0.0, 3.0, 100.0, 90.0, 5.0, 4.0])
        eligible = torch.ones(8, dtype=torch.bool)
        importance = torch.tensor([5.0, 0.5, 1.0, 9.0, 0.0, 0.0, 0.0, 0.0])
        active, ap_class, audit = build_equal_stream_budget_allocation(
            official,
            scores,
            eligible,
            ids,
            protected_fraction=0.25,
            enable_swap=True,
            importance=importance,
        )
        self.assertEqual(int(active.sum()), int(official.sum()))
        self.assertEqual(torch.nonzero(ap_class).reshape(-1).tolist(), [4, 5])
        # Anchors 0, 1, 2 tie on motion; the two least important donors pay.
        self.assertEqual(torch.nonzero(official & ~active).reshape(-1).tolist(), [1, 2])
        self.assertEqual(audit["demoted_canonical_ids"], [[1, 0, 0], [2, 0, 0]])
        self.assertEqual(
            audit["donor_ranking_keys"],
            ["score_asc", "backbone_importance_asc", "canonical_id"],
        )

    def test_donor_dual_key_tie_falls_back_to_canonical_ids(self):
        ids = torch.tensor([[i, 0, 0] for i in range(8)], dtype=torch.int64)
        official = torch.tensor([True, True, True, True, False, False, False, False])
        scores = torch.tensor([0.0, 0.0, 0.0, 3.0, 100.0, 90.0, 5.0, 4.0])
        eligible = torch.ones(8, dtype=torch.bool)
        importance = torch.tensor([2.0, 2.0, 2.0, 9.0, 0.0, 0.0, 0.0, 0.0])
        active, _, audit = build_equal_stream_budget_allocation(
            official,
            scores,
            eligible,
            ids,
            protected_fraction=0.25,
            enable_swap=True,
            importance=importance,
        )
        # Both keys tie for anchors 0, 1, 2: the canonical ID adjudicates.
        self.assertEqual(torch.nonzero(official & ~active).reshape(-1).tolist(), [0, 1])
        self.assertEqual(audit["demoted_canonical_ids"], [[0, 0, 0], [1, 0, 0]])

    def test_protected_class_selection_never_sees_importance(self):
        ids = torch.tensor([[i, 0, 0] for i in range(8)], dtype=torch.int64)
        official = torch.tensor([True, True, True, True, False, False, False, False])
        scores = torch.tensor([10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        eligible = torch.ones(8, dtype=torch.bool)
        importance = torch.tensor([0.1, 0.2, 99.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        _, with_importance, _ = build_equal_stream_budget_allocation(
            official,
            scores,
            eligible,
            ids,
            protected_fraction=0.25,
            enable_swap=True,
            importance=importance,
        )
        _, without_importance, _ = build_equal_stream_budget_allocation(
            official, scores, eligible, ids, protected_fraction=0.25, enable_swap=True
        )
        # The top-score class boundary tie stays adjudicated by canonical ID
        # alone; the outcome-blind selection rule never consumes importance.
        self.assertTrue(torch.equal(with_importance, without_importance))
        self.assertEqual(torch.nonzero(with_importance).reshape(-1).tolist(), [0, 1])

    def test_equal_byte_swap_demotes_least_important_on_score_tie(self):
        ids = torch.tensor([[i, 0, 0] for i in range(4)], dtype=torch.int64)
        official_active = torch.tensor([True, True, True, False])
        scores = torch.tensor([1.0, 1.0, 1.0, 100.0], dtype=torch.float64)
        eligible = torch.ones(4, dtype=torch.bool)
        estimated_bytes = torch.tensor([5, 5, 5, 5], dtype=torch.int64)
        importance = torch.tensor([3.0, 0.5, 1.0, 0.0])
        active, ap_class, audit = build_equal_estimated_byte_allocation(
            official_active,
            scores,
            eligible,
            ids,
            estimated_bytes,
            protected_fraction=0.25,
            enable_swap=True,
            importance=importance,
        )
        self.assertEqual(torch.nonzero(ap_class).reshape(-1).tolist(), [3])
        # Anchors 0, 1, 2 tie on motion and cost; the least important one pays
        # the promoted anchor's exact byte mass.
        self.assertEqual(torch.nonzero(official_active & ~active).reshape(-1).tolist(), [1])
        self.assertEqual(audit["demoted_canonical_ids"], [[1, 0, 0]])
        self.assertEqual(audit["estimated_byte_delta"], 0)
        self.assertEqual(
            audit["donor_ranking_keys"],
            ["score_asc", "backbone_importance_asc", "canonical_id"],
        )

    def test_frozen_backbone_importance_matches_prune_statistic(self):
        opacity_accum = torch.tensor([[1.0, 3.0], [0.0, 0.0], [2.0, 2.0]])
        anchor_demon = torch.tensor([[1.0, 1.0], [0.0, 0.0], [2.0, 2.0]])
        importance = frozen_backbone_importance(
            opacity_accum, anchor_demon, peak_ratio=0.1
        )
        # blended = 0.1 * max + 0.9 * mean over the GOP; normalized by visits;
        # never-rendered anchors score exactly zero.
        torch.testing.assert_close(
            importance,
            torch.tensor([1.05, 0.0, 0.5], dtype=torch.float64),
        )
        with self.assertRaisesRegex(ValueError, "same \\[N,GOP\\] shape"):
            frozen_backbone_importance(opacity_accum, anchor_demon[:2])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            frozen_backbone_importance(-opacity_accum, anchor_demon)
        with self.assertRaisesRegex(ValueError, "peak_ratio"):
            frozen_backbone_importance(opacity_accum, anchor_demon, peak_ratio=1.5)

    def test_donor_importance_must_be_finite_and_nonnegative(self):
        ids = torch.tensor([[i, 0, 0] for i in range(4)], dtype=torch.int64)
        official = torch.tensor([True, True, False, False])
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
        eligible = torch.ones(4, dtype=torch.bool)
        for bad in (
            torch.tensor([1.0, -1.0, 0.0, 0.0]),
            torch.tensor([1.0, float("nan"), 0.0, 0.0]),
            torch.tensor([1.0, float("inf"), 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0]),
        ):
            with self.assertRaisesRegex(ValueError, "donor importance"):
                build_equal_stream_budget_allocation(
                    official,
                    scores,
                    eligible,
                    ids,
                    protected_fraction=0.25,
                    enable_swap=True,
                    importance=bad,
                )

    def test_mask_pack_round_trip(self):
        mask = torch.tensor([True, False, True, True, False, False, False, True, True])
        payload = pack_bool_mask(mask)
        restored = unpack_bool_mask(payload, mask.numel())
        self.assertTrue(torch.equal(mask, restored))

    def test_identity_corrections_restore_unique_prequantization_ids(self):
        decoded = np.asarray(
            [[0, 0, 0], [0, 0, 0], [2, 1, 0], [4, 0, 0]], dtype=np.int64
        )
        retained = np.asarray(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [4, 0, 1]], dtype=np.int64
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ap_identity_corrections.bin"
            meta = write_identity_corrections(str(path), decoded, retained)
            restored = read_identity_corrections(tmp, meta, decoded)
            np.testing.assert_array_equal(restored, retained)
            self.assertEqual(meta["row_count"], 4)
            self.assertEqual(meta["mismatch_count"], 3)
            self.assertEqual(meta["bytes"], 4)
            tampered = bytearray(path.read_bytes())
            tampered[-1] = 27
            path.write_bytes(tampered)
            with self.assertRaisesRegex(ValueError, "binding|code"):
                read_identity_corrections(tmp, meta, decoded)

    def test_identity_corrections_reject_more_than_one_voxel_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "exceeds one"):
                write_identity_corrections(
                    str(Path(tmp) / "ap_identity_corrections.bin"),
                    np.asarray([[0, 0, 0]], dtype=np.int64),
                    np.asarray([[2, 0, 0]], dtype=np.int64),
                )

    def test_tensor_mapping_hash_is_order_invariant(self):
        left = {
            "b": torch.tensor([2.0]),
            "a": torch.tensor([[1.0, 3.0]]),
        }
        right = {"a": left["a"].clone(), "b": left["b"].clone()}
        self.assertEqual(tensor_mapping_sha256(left), tensor_mapping_sha256(right))

    def test_exact_estimated_byte_swap(self):
        ids = torch.tensor([[i, 0, 0] for i in range(7)], dtype=torch.int64)
        official = torch.tensor([True, True, True, True, False, False, False])
        scores = torch.tensor([0.0, 1.0, 2.0, 3.0, 100.0, 90.0, 80.0])
        costs = torch.tensor([3, 5, 7, 11, 8, 10, 100], dtype=torch.int64)
        active, ap_class, audit = build_equal_estimated_byte_allocation(
            official,
            scores,
            torch.ones(7, dtype=torch.bool),
            ids,
            costs,
            protected_fraction=2 / 7,
            enable_swap=True,
        )
        self.assertEqual(torch.nonzero(ap_class).reshape(-1).tolist(), [4, 5])
        self.assertEqual(torch.nonzero(active & ~official).reshape(-1).tolist(), [4, 5])
        self.assertEqual(audit["promoted_estimated_bytes"], 18)
        self.assertEqual(audit["demoted_estimated_bytes"], 18)

    def test_temporal_mass_closes_after_whole_membership_swap(self):
        ids = torch.tensor([[i, 0, 0] for i in range(4)], dtype=torch.int64)
        official_active = torch.tensor([True, False, False, False])
        ap_retain = torch.tensor([True, False, True, False])
        starting_active = torch.tensor([True, False, False, False])
        scores = torch.tensor([0.0, 1.0, 100.0, 2.0])
        costs = torch.tensor([5, 7, 5, 11], dtype=torch.int64)
        active, ap_class, audit = build_equal_estimated_byte_allocation(
            official_active,
            scores,
            torch.ones(4, dtype=torch.bool),
            ids,
            costs,
            protected_fraction=0.25,
            enable_swap=True,
            starting_active=starting_active,
            retain_mask=ap_retain,
        )
        self.assertEqual(torch.nonzero(ap_class).reshape(-1).tolist(), [2])
        self.assertEqual(active.tolist(), [False, False, True, False])
        self.assertFalse(bool(torch.any(active & ~ap_retain)))
        self.assertEqual(audit["official_estimated_bytes"], 5)
        self.assertEqual(audit["ap_estimated_bytes"], 5)

    def test_estimated_byte_allocator_fails_without_exact_subset(self):
        ids = torch.tensor([[i, 0, 0] for i in range(3)], dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "no exact estimated-byte"):
            build_equal_estimated_byte_allocation(
                torch.tensor([True, False, False]),
                torch.tensor([0.0, 100.0, 1.0]),
                torch.ones(3, dtype=torch.bool),
                ids,
                torch.tensor([3, 5, 7], dtype=torch.int64),
                protected_fraction=1 / 3,
                enable_swap=True,
            )

    def test_two_class_quantized_ar_round_trip(self):
        params = torch.tensor(
            [
                [[0.2, 0.6], [1.2, 1.6]],
                [[0.2, 0.6], [1.2, 1.6]],
                [[0.2, 0.6], [1.2, 1.6]],
            ],
            dtype=torch.float32,
        )
        class_mask = torch.tensor([True, False, True])
        with tempfile.TemporaryDirectory() as tmp:
            meta = _compress_ap_time_features_ar(
                tmp,
                "time_features",
                params,
                n_sidelen=2,
                class_mask=class_mask,
                q_ap_multiplier=0.5,
                q_bg_multiplier=2.0,
                scaling=1.0,
                p_channel=2,
                entropy_model=_RoundEntropy(),
            )
            decoded = _decompress_ap_time_features_ar(
                tmp,
                "time_features",
                meta,
                class_mask=class_mask,
                entropy_model=_RoundEntropy(),
                device=torch.device("cpu"),
            )
        expected = params.clone()
        expected[class_mask] = torch.round(expected[class_mask] / 0.5) * 0.5
        expected[~class_mask] = torch.round(expected[~class_mask] / 2.0) * 2.0
        torch.testing.assert_close(decoded, expected)

    def test_deterministic_zip_real_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "container"
            root.mkdir()
            (root / "b.bin").write_bytes(b"bbb")
            (root / "a.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
            first = deterministic_zip_directory(str(root), str(Path(tmp) / "first.zip"))
            second = deterministic_zip_directory(str(root), str(Path(tmp) / "second.zip"))
            self.assertEqual(first["bytes"], second["bytes"])
            self.assertEqual(first["sha256"], second["sha256"])

    def test_post_plas_recomputation_and_fault_injection(self):
        ids = torch.tensor([[0, 0, 0], [2, 0, 0]], dtype=torch.int64)
        source_bytes = torch.tensor([5, 7, 5, 9], dtype=torch.int64)
        retained_bytes = torch.tensor([5, 5], dtype=torch.int64)
        permutation = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
        sentinel = torch.tensor(
            [[torch.iinfo(torch.int64).min, 0, 0], [torch.iinfo(torch.int64).min + 1, 0, 0]],
            dtype=torch.int64,
        )
        pre_ids = torch.cat([ids, sentinel], dim=0)
        pre_bytes = torch.tensor([5, 5, 0, 0], dtype=torch.int64)
        pre_real = torch.tensor([True, True, False, False])
        pre_active = torch.tensor([False, True, False, False])
        pre_class = torch.tensor([False, True, False, False])
        factors = torch.zeros((4, 4))
        factors[pre_real, 3] = 1.0
        factors[pre_active, 0] = 1.0
        kwargs = {
            "official_retain_mask": torch.tensor([True, True, False, False]),
            "official_active_mask": torch.tensor([True, False, False, False]),
            "ap_retain_mask": torch.tensor([True, False, True, False]),
            "expected_ap_active_mask": torch.tensor([False, False, True, False]),
            "expected_ap_class_mask": torch.tensor([False, False, True, False]),
            "source_estimated_bytes": source_bytes,
            "pre_sort_retained_ids": ids,
            "pre_sort_estimated_bytes": retained_bytes,
            "sort_indices": permutation,
            "encoded_ids": pre_ids[permutation],
            "encoded_estimated_bytes": pre_bytes[permutation],
            "encoded_factors": factors[permutation],
            "real_mask": pre_real[permutation],
            "padding_mask": ~pre_real[permutation],
            "ap_class_mask": pre_class[permutation],
        }
        audit = recompute_sorted_codec_invariants(**kwargs)
        self.assertEqual(audit["whole_promoted_count"], audit["whole_demoted_count"])
        self.assertEqual(audit["official_estimated_time_bytes"], 5)
        self.assertEqual(audit["ap_estimated_time_bytes"], 5)
        for field, bad, message in (
            ("encoded_ids", pre_ids, "canonical IDs"),
            ("encoded_estimated_bytes", pre_bytes, "estimated-byte rows"),
            ("real_mask", pre_real, "real/padding mask"),
        ):
            corrupted = dict(kwargs)
            corrupted[field] = bad
            with self.assertRaisesRegex(ValueError, message):
                recompute_sorted_codec_invariants(**corrupted)

    def test_runtime_provenance_rejects_patch_and_tree_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            manifest_dir = base / "registration"
            for root in provenance.TREE_ROOTS:
                (repo / root).mkdir(parents=True)
                (repo / root / "source.py").write_text(f"ROOT={root!r}\n", encoding="utf-8")
            (repo / "setup.py").write_text("NAME='fixture'\n", encoding="utf-8")
            manifest_dir.mkdir()
            hashes = []
            rows = []
            for stage in provenance.PATCH_STAGES:
                path = manifest_dir / f"{stage}.patch"
                path.write_text(f"{stage}\n", encoding="utf-8")
                digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
                hashes.append(digest)
                rows.append({"stage": stage, "path": path.name, "sha256": digest})
            manifest = {
                "schema": provenance.MANIFEST_SCHEMA,
                "official_commit": provenance.OFFICIAL_COMMIT,
                "patches": rows,
                "normalized_code_tree": provenance.normalized_code_tree(repo),
            }
            path = manifest_dir / "manifest.json"
            path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            with mock.patch.object(provenance, "PATCH1_SHA256", hashes[0]), mock.patch.object(
                provenance, "PATCH2_SHA256", hashes[1]
            ), mock.patch.object(
                provenance, "PATCH2B_SHA256", hashes[2]
            ), mock.patch.object(provenance, "_git_head", return_value=provenance.OFFICIAL_COMMIT):
                receipt = provenance.verify_runtime_provenance(path, repo, digest)
                self.assertEqual(receipt["patch_sha256"], hashes)
                rows[2]["sha256"] = "f" * 64
                path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
                bad_digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
                with self.assertRaisesRegex(ValueError, "payload SHA-256 mismatch"):
                    provenance.verify_runtime_provenance(path, repo, bad_digest)
                rows[2]["sha256"] = hashes[2]
                path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
                (repo / "examples" / "source.py").write_text("TAMPERED=True\n", encoding="utf-8")
                bad_digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
                with self.assertRaisesRegex(ValueError, "code tree"):
                    provenance.verify_runtime_provenance(path, repo, bad_digest)

    def test_clean_models_strict_load_and_nonfinite_faults(self):
        config = {
            "anchor_feature_dim": 8,
            "c_perframe": 2,
            "n_offsets": 3,
            "time_dim": 4,
            "view_adaptive": False,
            "app_opt": True,
            "app_embed_dim": 2,
            "appearance_embedding_count": 4,
            "add_opacity_dist": False,
            "add_cov_dist": False,
            "add_color_dist": False,
        }
        decoders = build_decoder_modules(config, torch.device("cpu"))
        app = torch.nn.Module()
        app.embeds = torch.nn.Embedding(4, 2)
        nets = {
            "decoders": decoders.state_dict(),
            "app_module": app.state_dict(),
            "scaling": {
                "anchors": None,
                "quats": None,
                "scales": 0.1,
                "opacities": None,
                "anchor_features": 1.0,
                "offsets": 0.1,
                "factors": 0.0625,
                "time_features": 1.0,
            },
        }
        for name in ("scales", "anchor_features", "offsets", "factors", "time_features"):
            nets[f"{name}_entropy_model"] = {}
        _, loaded_app, audit = instantiate_counted_models(nets, config, torch.device("cpu"))
        self.assertIsNotNone(loaded_app)
        self.assertTrue(audit["strict_load"])
        missing = dict(nets)
        missing["decoders"] = dict(nets["decoders"])
        missing["decoders"].pop(next(iter(missing["decoders"])))
        with self.assertRaises(RuntimeError):
            instantiate_counted_models(missing, config, torch.device("cpu"))
        nonfinite = dict(nets)
        nonfinite["decoders"] = {name: value.clone() for name, value in nets["decoders"].items()}
        first = next(iter(nonfinite["decoders"]))
        nonfinite["decoders"][first].reshape(-1)[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            instantiate_counted_models(nonfinite, config, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
