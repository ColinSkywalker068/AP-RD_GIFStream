"""Stdlib-only tamper tests for the nine-stage H007 provenance contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "gsplat"
    / "compression"
    / "h007_runtime_provenance.py"
)
SPEC = importlib.util.spec_from_file_location("h007_runtime_provenance_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
provenance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provenance)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class H007RuntimeProvenanceStdlibTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        for relative in provenance.TREE_ROOTS:
            (self.repo / relative).mkdir(parents=True)
        (self.repo / "setup.py").write_text("name = 'fixture'\n", encoding="utf-8")
        (self.repo / "examples" / "fixture.py").write_text("VALUE = 1\n", encoding="utf-8")
        patch_dir = self.root / "patches"
        patch_dir.mkdir()
        self.patch_payloads = [f"patch-{index}\n".encode() for index in range(1, 10)]
        self.patch_hashes = []
        rows = []
        for stage, payload in zip(provenance.PATCH_STAGES, self.patch_payloads):
            path = patch_dir / f"{stage}.patch"
            path.write_bytes(payload)
            digest = _sha(payload)
            self.patch_hashes.append(digest)
            rows.append(
                {"stage": stage, "path": f"patches/{path.name}", "sha256": digest}
            )
        self.manifest = self.root / "manifest.json"
        self.manifest.write_bytes(
            json.dumps(
                {
                    "schema": provenance.MANIFEST_SCHEMA,
                    "official_commit": provenance.OFFICIAL_COMMIT,
                    "patches": rows,
                    "normalized_code_tree": provenance.normalized_code_tree(self.repo),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        self.manifest_sha = _sha(self.manifest.read_bytes())

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _verify(self):
        with mock.patch.object(provenance, "_git_head", return_value=provenance.OFFICIAL_COMMIT), mock.patch.object(
            provenance, "PATCH1_SHA256", self.patch_hashes[0]
        ), mock.patch.object(provenance, "PATCH2_SHA256", self.patch_hashes[1]), mock.patch.object(
            provenance, "PATCH2B_SHA256", self.patch_hashes[2]
        ):
            return provenance.verify_runtime_provenance(
                self.manifest, self.repo, self.manifest_sha
            )

    def test_clean_nine_stage_chain_is_accepted(self) -> None:
        receipt = self._verify()
        self.assertEqual(receipt["patch_sha256"], self.patch_hashes)
        self.assertEqual(receipt["normalized_code_tree"]["file_count"], 2)

    def test_patch_payload_tamper_is_rejected(self) -> None:
        (self.root / "patches" / "patch3.patch").write_bytes(b"tampered\n")
        with self.assertRaisesRegex(ValueError, "Patch3|patch3"):
            self._verify()

    def test_active_tree_tamper_is_rejected(self) -> None:
        (self.repo / "examples" / "fixture.py").write_text("VALUE = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "normalized post-apply code tree"):
            self._verify()

    def test_manifest_payload_tamper_is_rejected_first(self) -> None:
        self.manifest.write_bytes(self.manifest.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
            self._verify()


if __name__ == "__main__":
    unittest.main()
