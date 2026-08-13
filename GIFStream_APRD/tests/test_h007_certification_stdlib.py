import ast
import importlib.util
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "gsplat/compression/h007_certification.py"
)
SPEC = importlib.util.spec_from_file_location("h007_certification_test", MODULE_PATH)
certification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(certification)


class H007CertificationStdlibTest(unittest.TestCase):
    SCENE_CAMERA_COUNTS = {
        "coffee_martini": 18,
        "cook_spinach": 21,
        "cut_roasted_beef": 20,
        "flame_steak": 21,
        "sear_steak": 21,
    }

    def _selection_grid(self, root: Path):
        rows = []
        for scene_index, scene in enumerate(certification.CONFIRMATORY_SCENES):
            for method in ("official", "ap-gifstream-full"):
                rows.append(
                    self._selected_row(
                        root,
                        method,
                        f"{scene_index + (0 if method == 'official' else 8):x}" * 64,
                        1000 + scene_index * 20 + (1 if method != "official" else 0),
                        scene,
                    )
                )
        return {"schema": certification.SELECTION_SCHEMA, "selected": rows}

    def test_hdown_fresh_decode_exception_is_retained_as_total_failure(self):
        hdown_path = MODULE_PATH.parents[2] / "examples/h007_hdown_final.py"
        tree = ast.parse(hdown_path.read_text(encoding="utf-8"))
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_recompute_or_total_failure"
        )
        namespace = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(hdown_path), "exec"), namespace)

        def decode_failure():
            raise RuntimeError("fresh decode failed")

        def total_failure(error):
            return {
                "status": "FAIL",
                "penalized_loss": 1.0,
                "operational_failure_rate": 1.0,
                "error": f"{type(error).__name__}:{error}",
            }

        result = namespace["_recompute_or_total_failure"](
            decode_failure, total_failure
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["penalized_loss"], 1.0)
        self.assertEqual(result["operational_failure_rate"], 1.0)
        self.assertIn("fresh decode failed", result["error"])

    @staticmethod
    def _selected_row(
        root: Path, method: str, digest: str, size: int, scene: str = "coffee_martini"
    ):
        receipt = root / f"{scene}_{method}_evaluator.json"
        if not receipt.exists():
            receipt.write_bytes(method.encode())
        receipt_sha = certification.sha256_file(receipt)
        return {
            "scene": scene,
            "method": method,
            "point_id": f"{method}-p0",
            "archive": str(root / f"{method}.zip"),
            "registry_base": str(root),
            "archive_registry_relative": f"{method}.zip",
            "eligibility_receipt_registry_relative": f"{method}_request.json",
            "eligibility_receipt_path": str(root / f"{method}_request.json"),
            "eligibility_source_path": str(root / f"{method}_source.json"),
            "training_config_sha256": "c" * 64,
            "seed": 17,
            "archive_bytes": size,
            "archive_sha256": digest,
            "eligibility_receipt_sha256": "d" * 64,
            "eligibility_source_sha256": "e" * 64,
            "eligibility_recomputed": {
                "schema": "h007.h_sota_eligibility_recomputation.v4",
                "eligible": True,
                "ordinary_rate_quality_only": True,
                "required_point_count": 4,
                "distinct_archive_count": 4,
                "distinct_rate_count": 4,
                "distinct_training_config_count": 4,
                "frozen_rate_lambda_grid": {
                    "0": 0.0005,
                    "1": 0.001,
                    "2": 0.002,
                    "3": 0.004,
                },
                "minimum_adjacent_actual_byte_fraction": 0.05,
                "actual_bytes_by_rate": {
                    "0": size,
                    "1": int(size * 0.9),
                    "2": int(size * 0.8),
                    "3": int(size * 0.7),
                },
                "metrics_recomputed_from_evaluator_receipts": True,
                "selected_point_id": f"{method}-p0",
                "selected_rate": 0,
                "selected_archive_bytes": size,
                "selected_training_config_sha256": "c" * 64,
                "selected_seed": 17,
                "source_evidence_sha256": "e" * 64,
                "source_data": {
                    "manifest_sha256": "f" * 64,
                    "manifest_path": str((root / "source_data.json").resolve()),
                    "file_count": H007CertificationStdlibTest.SCENE_CAMERA_COUNTS[scene] * 300,
                    "scene": scene,
                },
                "runtime_provenance": {
                    "schema": "h007.ap_gifstream.runtime_provenance.v1",
                    "manifest_sha256": "1" * 64,
                    "official_commit": "c98486632e7dafd830740b1a1692bd08c48b96e3",
                    "patch_sha256": [
                        "2" * 64,
                        "3" * 64,
                        "4" * 64,
                        "5" * 64,
                        "6" * 64,
                        "7" * 64,
                        "8" * 64,
                        "9" * 64,
                        "a" * 64,
                        "b" * 64,
                        "c" * 64,
                    ],
                    "normalized_code_tree": {
                        "schema": "h007.normalized_code_tree.v1",
                        "normalization": "sorted-posix-path+lf-bytes+uint64le-lengths",
                        "roots": ["examples", "gsplat", "third_party"],
                        "root_files": ["setup.py"],
                        "suffixes": [".py"],
                        "special_names": ["CMakeLists.txt"],
                        "file_count": 1,
                        "sha256": "6" * 64,
                    },
                },
                "evaluator": {
                    "relative_path": "examples/h007_ordinary_rate_quality.py",
                    "sha256": "7" * 64,
                },
                "selected_evaluator_receipt_path": str(receipt.resolve()),
                "selected_evaluator_receipt_sha256": receipt_sha,
                "selected_metrics": {
                    "psnr": 30.0 if method == "official" else 30.1,
                    "ssim": 0.90 if method == "official" else 0.901,
                    "lpips": 0.10 if method == "official" else 0.099,
                    "encode_seconds": 5.0,
                    "decode_seconds": 6.0,
                    "render_fps": 7.0,
                },
            },
        }

    def test_exclusive_append_only_write_rejects_second_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stage02_freeze.json"
            first = certification.exclusive_write(path.resolve(), b"first")
            self.assertEqual(first["creation"], "O_CREAT|O_EXCL|O_NOFOLLOW")
            self.assertEqual(path.read_bytes(), b"first")
            with self.assertRaises(FileExistsError):
                certification.exclusive_write(path.resolve(), b"second")
            self.assertEqual(path.read_bytes(), b"first")

    def test_external_json_requires_exact_canonical_duplicate_free_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = (Path(temporary) / "selection.json").resolve()
            canonical = certification.canonical_json_bytes(
                {"schema": "fixture", "selected": []}
            )
            path.write_bytes(canonical)
            self.assertEqual(certification.read_json(path)["schema"], "fixture")

            for malformed in (
                canonical + b"\n",
                b'{"schema":"fixture","schema":"fixture","selected":[]}',
                b'{ "schema": "fixture", "selected": [] }',
            ):
                path.write_bytes(malformed)
                with self.assertRaisesRegex(ValueError, "canonical|duplicate"):
                    certification.read_json(path)

    def test_v4_selection_grid_passes_and_stale_or_tampered_closure_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            selection = self._selection_grid(root)
            certification._validate_selection_shape(selection)

            integral_float = json.loads(json.dumps(selection))
            integral_float["selected"][0]["archive_bytes"] = float(
                integral_float["selected"][0]["archive_bytes"]
            )
            with self.assertRaisesRegex(ValueError, "complete v4"):
                certification._validate_selection_shape(integral_float)

            numeric_sha = json.loads(json.dumps(selection))
            numeric_sha["selected"][0]["eligibility_source_sha256"] = int(
                "1" * 64
            )
            with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
                certification._validate_selection_shape(numeric_sha)

            stale = json.loads(json.dumps(selection))
            stale["selected"][0]["eligibility_recomputed"]["schema"] = (
                "h007.h_sota_eligibility_recomputation.v3"
            )
            with self.assertRaisesRegex(ValueError, "complete v4"):
                certification._validate_selection_shape(stale)

            tampered = json.loads(json.dumps(selection))
            tampered["selected"][0]["eligibility_recomputed"][
                "actual_bytes_by_rate"
            ]["0"] += 1
            with self.assertRaisesRegex(ValueError, "complete v4"):
                certification._validate_selection_shape(tampered)

    def test_real_v4_selection_freezes_and_validates_through_stage02(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            repo = MODULE_PATH.parents[2]
            scene = "coffee_martini"
            with mock.patch.object(certification, "CONFIRMATORY_SCENES", (scene,)):
                selection = self._selection_grid(root)
                (root / certification.SELECTION_NAME).write_bytes(
                    certification.canonical_json_bytes(selection)
                )
                preconditions = {
                    "schema": certification.PRECONDITION_SCHEMA,
                    "rows": [{"scene": scene}],
                    "outcome_fields_read": [
                        "ordinary_unedited_fidelity",
                        "real_container_accounting",
                    ],
                }
                (root / certification.PRECONDITIONS_NAME).write_bytes(
                    certification.canonical_json_bytes(preconditions)
                )
                source_revalidation = {
                    "schema": "h007.stage02_source_revalidation.v1",
                    "eligibility": [],
                    "preconditions": [{"scene": scene, "pass": True}],
                    "outcome_fields_read": [
                        "ordinary_unedited_fidelity",
                        "real_container_accounting",
                    ],
                    "revalidation_sha256": "1" * 64,
                }
                cases = {
                    (scene, index): {"status": "REFERENCE_INELIGIBLE"}
                    for index in range(25)
                }
                rebuilds = {
                    (scene, index): {
                        "status": "REFERENCE_INELIGIBLE",
                        "status_reproducible": True,
                    }
                    for index in range(25)
                }
                runtime = {"schema": "fixture.runtime.v1", "sha256": "2" * 64}
                with mock.patch.object(
                    certification,
                    "_revalidate_stage02_sources",
                    return_value=source_revalidation,
                ), mock.patch.object(
                    certification,
                    "_validate_reference_grid",
                    return_value=(cases, {}, rebuilds),
                ), mock.patch.object(
                    certification, "_verify_runtime", return_value=runtime
                ):
                    freeze = certification.freeze_stage02(
                        root=root,
                        output=root / certification.FREEZE_NAME,
                        repo_root=repo,
                        provenance_manifest=root / "manifest.json",
                        provenance_manifest_sha256="3" * 64,
                    )
                    freeze_sha = certification.sha256_file(
                        root / certification.FREEZE_NAME
                    )
                    validated = certification.validate_stage02_freeze(
                        root / certification.FREEZE_NAME, freeze_sha
                    )
                self.assertEqual(freeze["state"], "FROZEN_APPEND_ONLY")
                self.assertEqual(
                    validated["selection"]["selected"][0]["eligibility_recomputed"][
                        "schema"
                    ],
                    "h007.h_sota_eligibility_recomputation.v4",
                )

    def test_terminal_checkpoint_call_occurs_after_all_state_mutations(self):
        trainer = MODULE_PATH.parents[2] / "examples/simple_trainer_GIFStream.py"
        source = trainer.read_text(encoding="utf-8")
        optimize = source.index("for optimizer in self.optimizers.values()")
        entropy = source.index(
            "for name, optimizer in self.compression_sim_method.entropy_model_optimizers.items()"
        )
        post_backward = source.index("self.cfg.strategy.step_post_backward(")
        save_call = source.index("self._save_training_checkpoint_after_update(step, global_tic)")
        self.assertLess(optimize, entropy)
        self.assertLess(entropy, post_backward)
        self.assertLess(post_backward, save_call)

    def test_exclusive_write_does_not_follow_dangling_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            target = root / "outside.json"
            link = root / "stage02_freeze.json"
            os.symlink(target, link)
            with self.assertRaises(FileExistsError):
                certification.exclusive_write(root / link.name, b"forged")
            self.assertFalse(target.exists())

    def test_missing_stage02_freeze_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "stage02_freeze.json"
            with self.assertRaises(FileNotFoundError):
                certification.validate_stage02_freeze(path, "0" * 64)

    def test_failing_precondition_grid_cannot_freeze(self):
        rows = [
            {"scene": scene, "pass": scene != certification.CONFIRMATORY_SCENES[0]}
            for scene in sorted(certification.CONFIRMATORY_SCENES)
        ]
        with self.assertRaisesRegex(ValueError, "precondition gate did not pass"):
            certification._require_preconditions_pass({"preconditions": rows})

    def test_manual_case_without_generation_closure_is_rejected(self):
        stage = {
            "freeze": {"evaluator_sha256": "1" * 64},
            "runtime_provenance": {"patch_sha256": ["a" * 64] * 11},
        }
        sequence = {
            "archive_sha256": "2" * 64,
            "sequence_manifest_sha256": "3" * 64,
        }
        gop = {"sha256": "4" * 64, "decoder_config_sha256": "5" * 64}
        forged = {
            "schema": "h007.hdown_final_candidate_case.v2",
            "status": "PASS",
            "cases": [{"mean_soft_iou": {"010": 1.0}}],
        }
        with self.assertRaisesRegex(ValueError, "manual candidate case"):
            certification.validate_case_static_closure(
                forged, stage=stage, sequence_validation=sequence, gop_audit=gop
            )

    def test_stage02_freeze_requires_all_25_reference_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            selected = []
            for scene_index, scene in enumerate(certification.CONFIRMATORY_SCENES):
                for method in ("official", "ap-gifstream-full"):
                    selected.append(
                        self._selected_row(
                            root,
                            method,
                            f"{scene_index + (0 if method == 'official' else 8):x}" * 64,
                            1000 + scene_index * 20 + (1 if method != "official" else 0),
                            scene,
                        )
                    )
            (root / certification.SELECTION_NAME).write_bytes(
                certification.canonical_json_bytes(
                    {
                        "schema": certification.SELECTION_SCHEMA,
                        "selected": selected,
                    }
                )
            )
            (root / certification.PRECONDITIONS_NAME).write_bytes(
                certification.canonical_json_bytes(
                    {
                        "schema": certification.PRECONDITION_SCHEMA,
                        "rows": [
                            {"scene": scene}
                            for scene in certification.CONFIRMATORY_SCENES
                        ],
                        "outcome_fields_read": [
                            "ordinary_unedited_fidelity",
                            "real_container_accounting",
                        ],
                    }
                )
            )
            repo = MODULE_PATH.parents[2]
            source_revalidation = {
                "schema": "h007.stage02_source_revalidation.v1",
                "eligibility": [],
                "preconditions": [
                    {"scene": scene, "pass": True}
                    for scene in sorted(certification.CONFIRMATORY_SCENES)
                ],
                "outcome_fields_read": [
                    "ordinary_unedited_fidelity",
                    "real_container_accounting",
                ],
                "revalidation_sha256": "1" * 64,
            }
            with mock.patch.object(
                certification,
                "_revalidate_stage02_sources",
                return_value=source_revalidation,
            ):
                with self.assertRaisesRegex(FileNotFoundError, "reference_cases"):
                    certification.freeze_stage02(
                        root=root,
                        output=root / certification.FREEZE_NAME,
                        repo_root=repo,
                        provenance_manifest=root / "forged_manifest.json",
                        provenance_manifest_sha256="0" * 64,
                    )

    def test_embedded_eligibility_true_without_source_paths_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            selected = []
            for scene_index, scene in enumerate(certification.CONFIRMATORY_SCENES):
                for method in ("official", "ap-gifstream-full"):
                    row = self._selected_row(
                        root,
                        method,
                        f"{scene_index + (0 if method == 'official' else 8):x}" * 64,
                        1000 + scene_index * 20 + (1 if method != "official" else 0),
                        scene,
                    )
                    for name in (
                        "registry_base",
                        "archive_registry_relative",
                        "eligibility_receipt_registry_relative",
                        "eligibility_receipt_path",
                        "eligibility_source_path",
                    ):
                        row.pop(name)
                    selected.append(row)
            with self.assertRaisesRegex(ValueError, "source paths"):
                certification._validate_selection_shape(
                    {"schema": certification.SELECTION_SCHEMA, "selected": selected}
                )

    def test_stage02_reopens_eligibility_and_rejects_replay_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            selected = self._selected_row(root, "official", "a" * 64, 1000)
            replay = dict(selected)
            replay["archive_sha256"] = "b" * 64
            module = types.SimpleNamespace(
                revalidate_selected_eligibility=lambda row: replay
            )
            with mock.patch.object(
                certification, "_load_sequence_module", return_value=module
            ):
                with self.assertRaisesRegex(ValueError, "source replay differs"):
                    certification._revalidate_stage02_sources(
                        root=root,
                        repo_root=root,
                        selection={"selected": [selected]},
                        preconditions={"rows": []},
                    )

    def test_precondition_summary_cannot_override_300frame_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            official = self._selected_row(root, "official", "a" * 64, 1000)
            ap = self._selected_row(root, "ap-gifstream-full", "b" * 64, 1001)
            module = types.SimpleNamespace(
                revalidate_selected_eligibility=lambda row: dict(row)
            )
            row = {
                "scene": "coffee_martini",
                "official_archive_sha256": official["archive_sha256"],
                "ap_archive_sha256": ap["archive_sha256"],
                "official_bytes": official["archive_bytes"],
                "ap_bytes": ap["archive_bytes"],
                "frame_count": 300,
                "psnr_official": 99.0,
                "psnr_ap": 30.1,
                "ssim_official": 0.90,
                "ssim_ap": 0.901,
                "lpips_official": 0.10,
                "lpips_ap": 0.099,
                "official_evaluator_receipt": official["eligibility_recomputed"][
                    "selected_evaluator_receipt_path"
                ],
                "official_evaluator_receipt_sha256": official[
                    "eligibility_recomputed"
                ]["selected_evaluator_receipt_sha256"],
                "ap_evaluator_receipt": ap["eligibility_recomputed"][
                    "selected_evaluator_receipt_path"
                ],
                "ap_evaluator_receipt_sha256": ap["eligibility_recomputed"][
                    "selected_evaluator_receipt_sha256"
                ],
            }
            with mock.patch.object(
                certification, "_load_sequence_module", return_value=module
            ):
                with self.assertRaisesRegex(ValueError, "not recomputed from 300-frame"):
                    certification._revalidate_stage02_sources(
                        root=root,
                        repo_root=root,
                        selection={"selected": [official, ap]},
                        preconditions={"rows": [row]},
                    )

    def test_static_reference_hash_cannot_replace_rebuild(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory = root / "reference_cases/coffee_martini/gop_0"
            directory.mkdir(parents=True)
            artifact = directory / "reference.npz"
            artifact.write_bytes(b"self-consistent-static-artifact")
            evaluator_sha = "a" * 64
            manifest = {
                "schema": certification.REFERENCE_SCHEMA,
                "status": "ELIGIBLE",
                "scene": "coffee_martini",
                "gop_id": 0,
                "gop_start_frame": 0,
                "evaluator_sha256": evaluator_sha,
                "candidate_inputs_read": [],
                "outcome_fields_read": [],
                "artifact": str(artifact),
                "artifact_bytes": artifact.stat().st_size,
                "artifact_sha256": certification.sha256_file(artifact),
                "cases": [{"label": "primary"}, {"label": "static"}],
            }
            (directory / "reference.json").write_bytes(
                certification.canonical_json_bytes(manifest)
            )
            evaluator = types.SimpleNamespace(
                verify_reference_rebuild=lambda *args: {
                    "schema": "h007.reference_rebuild_audit.v1",
                    "scene": "coffee_martini",
                    "gop_id": 0,
                    "status": "ELIGIBLE",
                    "artifact_sha256": "f" * 64,
                    "source_inputs_revalidated": True,
                    "byte_reproducible": True,
                    "status_reproducible": True,
                }
            )
            with mock.patch.object(
                certification, "CONFIRMATORY_SCENES", ("coffee_martini",)
            ), mock.patch.object(
                certification, "GOP_STARTS", (0,)
            ), mock.patch.object(
                certification, "_load_evaluator_module", return_value=evaluator
            ):
                with self.assertRaisesRegex(ValueError, "reference rebuild failed"):
                    certification._validate_reference_grid(root, evaluator_sha, root)

    def test_ineligible_reference_status_must_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory = root / "reference_cases/coffee_martini/gop_0"
            directory.mkdir(parents=True)
            manifest = {
                "schema": certification.REFERENCE_SCHEMA,
                "status": "REFERENCE_INELIGIBLE",
                "scene": "coffee_martini",
                "gop_id": 0,
                "gop_start_frame": 0,
                "evaluator_sha256": "a" * 64,
                "candidate_inputs_read": [],
                "outcome_fields_read": [],
                "reason": "frozen reason",
            }
            (directory / "reference.json").write_bytes(
                certification.canonical_json_bytes(manifest)
            )
            evaluator = types.SimpleNamespace(
                verify_reference_rebuild=lambda *args: {
                    "schema": "h007.reference_rebuild_audit.v1",
                    "scene": "coffee_martini",
                    "gop_id": 0,
                    "status": "ELIGIBLE",
                    "artifact_sha256": None,
                    "source_inputs_revalidated": True,
                    "status_reproducible": True,
                }
            )
            with mock.patch.object(
                certification, "CONFIRMATORY_SCENES", ("coffee_martini",)
            ), mock.patch.object(
                certification, "GOP_STARTS", (0,)
            ), mock.patch.object(
                certification, "_load_evaluator_module", return_value=evaluator
            ):
                with self.assertRaisesRegex(ValueError, "ineligible.*replay failed"):
                    certification._validate_reference_grid(root, "a" * 64, root)


if __name__ == "__main__":
    unittest.main()
