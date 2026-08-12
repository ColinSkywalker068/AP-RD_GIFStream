"""Fail-closed runtime provenance for the H007 AP-GIFStream experiment.

The preregistration manifest deliberately lives outside the patched GIFStream
tree.  It binds the registered patch payloads and a normalized post-apply source-tree
digest without making a source file contain its own hash.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


MANIFEST_SCHEMA = "h007.ap_gifstream.patch_chain_manifest.v1"
TREE_SCHEMA = "h007.normalized_code_tree.v1"
OFFICIAL_COMMIT = "c98486632e7dafd830740b1a1692bd08c48b96e3"
PATCH1_SHA256 = "65ecd8f1fe30fac5d5bce3aed295cfe04f0b0582f106efaf37489b005d2e431b"
PATCH2_SHA256 = "347e3fbb59cdab49160a0c5291ecbd375e2de37d6bf3dea3e927d07f0ed8a253"
PATCH2B_SHA256 = "537b7a6c1607204000f6d4ce198d60c0b69917e46cdd89e9d6fcc431d37af40b"
PATCH_STAGES = (
    "patch1",
    "patch2",
    "patch2b",
    "patch3",
    "patch4",
    "patch5",
    "patch6",
    "patch7",
    "patch8",
)
TREE_ROOTS = ("examples", "gsplat", "third_party")
TREE_ROOT_FILES: Tuple[str, ...] = ("setup.py",)
TREE_SUFFIXES = (".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py")
TREE_SPECIAL_NAMES = ("CMakeLists.txt",)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_hex_sha256(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{name} is not a lowercase SHA-256")
    return text


def _iter_code_files(repo_root: Path) -> Iterable[Path]:
    for relative in TREE_ROOT_FILES:
        path = repo_root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required normalized-tree source is unavailable: {relative}")
        yield path
    for relative in TREE_ROOTS:
        root = repo_root / relative
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"required normalized-tree root is unavailable: {relative}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"symlink forbidden in normalized code tree: {path}")
            if not path.is_file():
                continue
            if path.suffix in TREE_SUFFIXES or path.name in TREE_SPECIAL_NAMES:
                yield path


def normalized_code_tree(repo_root: Path) -> Dict[str, Any]:
    """Hash normalized source paths and LF-normalized bytes in stable order."""

    repo_root = repo_root.resolve()
    rows = []
    for path in sorted(set(_iter_code_files(repo_root))):
        relative = path.relative_to(repo_root).as_posix()
        payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        rows.append((relative, payload))
    digest = hashlib.sha256()
    for relative, payload in rows:
        name = relative.encode("utf-8")
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return {
        "schema": TREE_SCHEMA,
        "normalization": "sorted-posix-path+lf-bytes+uint64le-lengths",
        "roots": list(TREE_ROOTS),
        "root_files": list(TREE_ROOT_FILES),
        "suffixes": list(TREE_SUFFIXES),
        "special_names": list(TREE_SPECIAL_NAMES),
        "file_count": len(rows),
        "sha256": digest.hexdigest(),
    }


def _resolve_patch_path(manifest_path: Path, declared: str) -> Path:
    candidate = Path(declared)
    if candidate.is_absolute():
        raise ValueError("patch-chain manifest paths must be relative")
    unresolved = manifest_path.parent / candidate
    if unresolved.is_symlink():
        raise ValueError(f"registered patch payload symlink is forbidden: {declared}")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise ValueError(f"registered patch payload is unavailable: {declared}")
    return resolved


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_runtime_provenance(
    manifest_path: Path,
    repo_root: Path,
    expected_manifest_sha256: str,
    expected_container_receipt: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Verify preregistration, patch payloads, git base and active source tree.

    ``expected_manifest_sha256`` is mandatory and comes from the frozen run
    configuration (or clean-decoder invocation), never from the container.
    """

    if manifest_path.is_symlink():
        raise ValueError("preregistered patch-chain manifest symlink is forbidden")
    manifest_path = manifest_path.resolve()
    repo_root = repo_root.resolve()
    expected_manifest_sha256 = _require_hex_sha256(
        expected_manifest_sha256, "expected manifest SHA-256"
    )
    if not manifest_path.is_file():
        raise ValueError("preregistered patch-chain manifest is unavailable")
    raw = manifest_path.read_bytes()
    manifest_sha256 = _sha256_bytes(raw)
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError("preregistered patch-chain manifest SHA-256 mismatch")
    manifest = json.loads(raw.decode("utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema",
        "official_commit",
        "patches",
        "normalized_code_tree",
    }:
        raise ValueError("patch-chain manifest fields are incomplete or unexpected")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported patch-chain manifest schema")
    if manifest.get("official_commit") != OFFICIAL_COMMIT:
        raise ValueError("patch-chain manifest official commit mismatch")
    if _git_head(repo_root) != OFFICIAL_COMMIT:
        raise ValueError("active GIFStream checkout is not at the registered official commit")

    patches = manifest.get("patches")
    if not isinstance(patches, list) or [row.get("stage") for row in patches] != list(
        PATCH_STAGES
    ):
        raise ValueError("patch-chain manifest stages/order mismatch")
    patch_hashes = []
    for index, row in enumerate(patches):
        if set(row) != {"stage", "path", "sha256"}:
            raise ValueError("patch-chain manifest patch row has unexpected fields")
        declared_sha = _require_hex_sha256(row["sha256"], f"{row['stage']} SHA-256")
        patch_path = _resolve_patch_path(manifest_path, str(row["path"]))
        actual_sha = _sha256_bytes(patch_path.read_bytes())
        if actual_sha != declared_sha:
            raise ValueError(f"registered {row['stage']} payload SHA-256 mismatch")
        patch_hashes.append(actual_sha)
        if index == 0 and actual_sha != PATCH1_SHA256:
            raise ValueError("registered Patch1 is not the frozen Patch1 payload")
        if index == 1 and actual_sha != PATCH2_SHA256:
            raise ValueError("registered Patch2 is not the frozen Patch2 payload")
        if index == 2 and actual_sha != PATCH2B_SHA256:
            raise ValueError("registered Patch2b is not the frozen Patch2b payload")

    declared_tree = manifest.get("normalized_code_tree")
    actual_tree = normalized_code_tree(repo_root)
    if declared_tree != actual_tree:
        raise ValueError("active normalized post-apply code tree is not preregistered")
    receipt = {
        "schema": "h007.ap_gifstream.runtime_provenance.v1",
        "manifest_sha256": manifest_sha256,
        "official_commit": OFFICIAL_COMMIT,
        "patch_sha256": patch_hashes,
        "normalized_code_tree": actual_tree,
    }
    if expected_container_receipt is not None and dict(expected_container_receipt) != receipt:
        raise ValueError("container/runtime provenance receipt mismatch")
    return receipt


def provenance_manifest_template(
    patch_paths_and_hashes: Iterable[Tuple[str, str]], repo_root: Path
) -> Dict[str, Any]:
    """Build an external manifest payload after the final patch is archived."""

    patch_paths_and_hashes = list(patch_paths_and_hashes)
    if len(patch_paths_and_hashes) != len(PATCH_STAGES):
        raise ValueError("manifest template requires the registered patch count")
    rows = []
    for stage, (path, digest) in zip(PATCH_STAGES, patch_paths_and_hashes):
        rows.append(
            {
                "stage": stage,
                "path": str(path),
                "sha256": _require_hex_sha256(digest, f"{stage} SHA-256"),
            }
        )
    return {
        "schema": MANIFEST_SCHEMA,
        "official_commit": OFFICIAL_COMMIT,
        "patches": rows,
        "normalized_code_tree": normalized_code_tree(repo_root),
    }
