import hashlib
import importlib.util
import io
import json
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples/h007_hdown_final.py"
SPEC = importlib.util.spec_from_file_location("h007_hdown_final_test", MODULE_PATH)
hdown = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hdown)


class H007HDownContractTest(unittest.TestCase):
    @staticmethod
    def _selected_sequence(root: Path):
        decoder_payload = b'{"codec_family":"GIFStream"}'
        inner_buffer = io.BytesIO()
        with zipfile.ZipFile(inner_buffer, "w", compression=zipfile.ZIP_STORED) as inner:
            inner.writestr("decoder_config.json", decoder_payload)
        inner_payload = inner_buffer.getvalue()
        sequence = root / "selected_sequence.zip"
        with zipfile.ZipFile(sequence, "w", compression=zipfile.ZIP_STORED) as outer:
            outer.writestr("gops/gop_0.zip", inner_payload)
        validation = {
            "archive_sha256": "a" * 64,
            "gops": [
                {
                    "bytes": len(inner_payload),
                    "sha256": hashlib.sha256(inner_payload).hexdigest(),
                    "decoder_config_sha256": hashlib.sha256(decoder_payload).hexdigest(),
                }
            ],
        }
        return sequence, inner_payload, validation

    @staticmethod
    def _stage(root: Path):
        return {
            "contract": {
                "repo_root": str(root),
                "clean_decoder_relative_path": "examples/h007_clean_decode_gifstream.py",
                "clean_decoder_sha256": "c" * 64,
                "provenance_manifest": str(root / "manifest.json"),
                "provenance_manifest_sha256": "d" * 64,
            },
            "runtime_provenance": {"manifest_sha256": "d" * 64},
        }

    def test_deterministic_npz_and_no_pickle(self):
        arrays = {
            "ids": np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
            "mass": np.asarray([0.75, 0.25], dtype=np.float64),
        }
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.npz"
            second = Path(temporary) / "second.npz"
            hdown.write_deterministic_npz(first, arrays)
            hdown.write_deterministic_npz(second, arrays)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with np.load(first, allow_pickle=False) as restored:
                np.testing.assert_array_equal(restored["ids"], arrays["ids"])
                np.testing.assert_array_equal(restored["mass"], arrays["mass"])

    def test_strict_json_rejects_noncanonical_and_duplicate_keys(self):
        canonical = hdown.canonical_json_bytes({"count": 1, "name": "case"})
        self.assertEqual(
            hdown.strict_canonical_json_bytes(canonical, "fixture"),
            {"count": 1, "name": "case"},
        )
        for payload in (
            canonical + b"\n",
            b'{"count":1,"count":1,"name":"case"}',
            b'{"name":"case", "count":1}',
        ):
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, "strict JSON|canonical JSON"):
                    hdown.strict_canonical_json_bytes(payload, "fixture")

    def test_exact_json_role_helpers_reject_coercive_aliases(self):
        for value in (0.0, False, "0", None):
            with self.subTest(role="nonnegative-int", value=value):
                with self.assertRaises(ValueError):
                    hdown.exact_nonnegative_int(value, "fixture integer")
        for value in (1, True, "1.0", None):
            with self.subTest(role="float", value=value):
                with self.assertRaises(ValueError):
                    hdown.exact_finite_float(value, "fixture float", positive=True)
        for value in (1, ["a"], None):
            with self.subTest(role="string", value=value):
                with self.assertRaises(ValueError):
                    hdown.exact_string(value, "fixture string")
        with self.assertRaises(ValueError):
            hdown.require_sha256(1, "fixture SHA-256")

    def test_directory_census_rejects_numeric_and_duplicate_path_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b"counted payload"
            (root / "payload.bin").write_bytes(payload)
            honest = {
                "schema": "h007.container_byte_census.v1",
                "files": [
                    {
                        "path": "payload.bin",
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
                "file_count": 1,
                "raw_bytes": len(payload),
            }
            census = root / "byte_census.json"
            census.write_bytes(hdown.canonical_json_bytes(honest))
            hdown.validate_directory_census(root)

            for mutate in ("float-bytes", "duplicate-path", "float-total"):
                forged = json.loads(json.dumps(honest))
                if mutate == "float-bytes":
                    forged["files"][0]["bytes"] = float(len(payload))
                elif mutate == "duplicate-path":
                    forged["files"].append(dict(forged["files"][0]))
                    forged["file_count"] = 2
                    forged["raw_bytes"] = 2 * len(payload)
                else:
                    forged["file_count"] = 1.0
                census.write_bytes(hdown.canonical_json_bytes(forged))
                with self.subTest(mutate=mutate):
                    with self.assertRaises(ValueError):
                        hdown.validate_directory_census(root)

    def test_clean_decode_source_image_count_float_alias_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            container = bundle / "container"
            container.mkdir()
            (container / "byte_census.json").write_bytes(
                hdown.canonical_json_bytes(
                    {"schema": "h007.container_byte_census.v1", "files": []}
                )
            )
            splats = bundle / "decoded_splats.pt"
            splats.write_bytes(b"not reached")
            (bundle / "clean_decode_manifest.json").write_bytes(
                hdown.canonical_json_bytes(
                    {
                        "schema": "h007.clean_decode_result.v2",
                        "decoded_splats_sha256": hashlib.sha256(
                            splats.read_bytes()
                        ).hexdigest(),
                        "source_images_read": 0.0,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "source-image count"):
                hdown.load_bundle(bundle, torch.device("cpu"), reference=False)

    def test_reference_manifest_integral_float_gop_is_rejected_before_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "schema": hdown.REFERENCE_SCHEMA,
                "status": "ELIGIBLE",
                "scene": "coffee_martini",
                "gop_id": 0.0,
                "gop_start_frame": 0,
                "raw_cam00_dir": str(root),
                "reference_bundle": str(root),
                "tracker": {},
                "raw_frame_sha256": ["a" * 64] * 60,
                "reference_bundle_byte_census_sha256": "b" * 64,
                "rebuild_device": "cpu",
                "rebuild_seed": 20260715,
            }
            manifest_path = root / "reference.json"
            manifest_path.write_bytes(hdown.canonical_json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "GOP ID"):
                hdown.verify_reference_rebuild(
                    manifest_path, root / "reference.npz"
                )

    def test_parent_prefix_is_smallest_90_percent_with_id_tie_break(self):
        ids = torch.tensor([[2, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=torch.int64)
        mass = torch.tensor([0.05, 0.45, 0.50], dtype=torch.float64)
        selected, weights, audit = hdown._select_parent_prefix(ids, mass)
        self.assertEqual(selected.tolist(), [[1, 0, 0], [0, 0, 0]])
        self.assertEqual(audit["parent_prefix_count"], 2)
        self.assertAlmostEqual(float(weights.sum()), 0.95)

    def test_soft_iou(self):
        left = np.asarray([[1.0, 0.5], [0.0, 0.0]], dtype=np.float32)
        right = np.asarray([[0.5, 1.0], [0.0, 0.0]], dtype=np.float32)
        self.assertAlmostEqual(hdown._soft_iou(left, right), 0.5)

    def test_fresh_decode_uses_exact_extracted_inner_gop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence, inner_payload, validation = self._selected_sequence(root)
            stage = self._stage(root)
            observed = {}

            def clean_decode(
                inner_path,
                selected_sequence,
                gop_id,
                output_dir,
                device,
                provenance_manifest,
                provenance_sha,
            ):
                observed["inner_payload"] = inner_path.read_bytes()
                observed["selected_sequence"] = selected_sequence
                output_dir.mkdir(parents=True)
                result = {
                    "schema": "h007.clean_decode_result.v2",
                    "source_sequence_archive_sha256": validation["archive_sha256"],
                    "source_inner_gop_sha256": validation["gops"][0]["sha256"],
                    "runtime_provenance": stage["runtime_provenance"],
                    "producer_receipt_validated": True,
                    "decoded_splats_sha256": "e" * 64,
                    "producer_receipt_sha256": "f" * 64,
                }
                (output_dir / "clean_decode_manifest.json").write_bytes(
                    hdown.canonical_json_bytes(result)
                )
                return result

            with mock.patch.object(
                hdown, "validate_sequence_container", return_value=validation
            ), mock.patch.object(
                hdown,
                "_load_frozen_clean_decoder",
                return_value=types.SimpleNamespace(clean_decode=clean_decode),
            ):
                result, audit = hdown._fresh_decode_selected_gop(
                    stage=stage,
                    selected_sequence=sequence,
                    scene="coffee_martini",
                    method="official",
                    gop_id=0,
                    output_dir=root / "fresh_bundle",
                    device="cpu",
                )
            self.assertEqual(observed["inner_payload"], inner_payload)
            self.assertEqual(observed["selected_sequence"], sequence)
            self.assertEqual(result["source_inner_gop_sha256"], validation["gops"][0]["sha256"])
            self.assertTrue(audit["fresh_inner_gop_decode"])

    def test_inner_gop_hash_mismatch_rejects_before_decoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence, _, validation = self._selected_sequence(root)
            validation["gops"][0]["sha256"] = "b" * 64
            loader = mock.Mock()
            with mock.patch.object(
                hdown, "validate_sequence_container", return_value=validation
            ), mock.patch.object(hdown, "_load_frozen_clean_decoder", loader):
                with self.assertRaisesRegex(ValueError, "payload differs"):
                    hdown._fresh_decode_selected_gop(
                        stage=self._stage(root),
                        selected_sequence=sequence,
                        scene="coffee_martini",
                        method="official",
                        gop_id=0,
                        output_dir=root / "fresh_bundle",
                        device="cpu",
                    )
            loader.assert_not_called()

    def test_clean_decoder_failure_never_falls_back_to_stale_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence, _, validation = self._selected_sequence(root)
            output_dir = root / "fresh_bundle"
            output_dir.mkdir()
            stale = output_dir / "clean_decode_manifest.json"
            stale.write_text(json.dumps({"schema": "forged.stale"}), encoding="utf-8")

            def fail(*args, **kwargs):
                raise RuntimeError("decoder failed")

            with mock.patch.object(
                hdown, "validate_sequence_container", return_value=validation
            ), mock.patch.object(
                hdown,
                "_load_frozen_clean_decoder",
                return_value=types.SimpleNamespace(clean_decode=fail),
            ):
                with self.assertRaisesRegex(RuntimeError, "decoder failed"):
                    hdown._fresh_decode_selected_gop(
                        stage=self._stage(root),
                        selected_sequence=sequence,
                        scene="coffee_martini",
                        method="official",
                        gop_id=0,
                        output_dir=output_dir,
                        device="cpu",
                    )
            self.assertEqual(json.loads(stale.read_text())["schema"], "forged.stale")

    def test_aggregate_converts_fresh_decode_exception_to_total_failure(self):
        def fail_decode():
            raise RuntimeError("fresh decode failed")

        def total_failure(error):
            return (
                {
                    "status": "FAIL",
                    "operational_failure_rate": 1.0,
                    "penalized_loss": 1.0,
                },
                {
                    "fresh_inner_gop_decode": False,
                    "total_penalty_assigned": True,
                    "error": f"{type(error).__name__}:{error}",
                },
            )

        result, audit = hdown._recompute_or_total_failure(
            fail_decode, total_failure
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["operational_failure_rate"], 1.0)
        self.assertEqual(result["penalized_loss"], 1.0)
        self.assertFalse(audit["fresh_inner_gop_decode"])
        self.assertTrue(audit["total_penalty_assigned"])
        self.assertIn("fresh decode failed", audit["error"])

    def test_frozen_clean_decoder_hash_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decoder = root / "examples/h007_clean_decode_gifstream.py"
            decoder.parent.mkdir()
            decoder.write_text("def clean_decode(*args):\n    return {}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                hdown._load_frozen_clean_decoder(self._stage(root))


if __name__ == "__main__":
    unittest.main()
