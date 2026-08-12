"""Stdlib-only, fail-closed certification closure for H007 Stage 02.

The Stage-02 marker is not a generic directory checksum.  It is an exclusive,
no-follow freeze of an exact five-scene x five-GOP reference grid, the selected
real nested sequence archives, ordinary image/rate preconditions, and the
active nine-stage evaluator provenance.  Validation reopens and recomputes the
complete closure; a caller-supplied PASS object is never trusted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


STAGE02_SCHEMA = "h007.stage02_freeze.v3"
STAGE02_CONTRACT_SCHEMA = "h007.stage02_contract.v3"
REFERENCE_SCHEMA = "h007.hdown_final_reference_case.v3"
SELECTION_SCHEMA = "h007.real_zip_operating_point_selection.v2"
PRECONDITION_SCHEMA = "h007.hdown_image_rate_preconditions.v2"
CONFIRMATORY_SCENES = (
    "coffee_martini",
    "cook_spinach",
    "cut_roasted_beef",
    "flame_steak",
    "sear_steak",
)
GOP_STARTS = (0, 60, 120, 180, 240)
EVALUATOR_RELATIVE_PATH = "examples/h007_hdown_final.py"
CLEAN_DECODER_RELATIVE_PATH = "examples/h007_clean_decode_gifstream.py"
SELECTION_NAME = "operating_point_selection.json"
PRECONDITIONS_NAME = "image_rate_preconditions.json"
CONTRACT_NAME = "stage02_contract.json"
RUNTIME_RECEIPT_NAME = "runtime_provenance.json"
SOURCE_REVALIDATION_NAME = "source_revalidation.json"
REFERENCE_REBUILD_NAME = "reference_rebuilds.json"
FREEZE_NAME = "stage02_freeze.json"


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _reject_duplicate_object_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate certification JSON key: {key}")
        value[key] = item
    return value


def _strict_canonical_json(payload: bytes, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_object_pairs
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    if canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} JSON bytes are not canonical")
    return value


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def require_sha256(value: Any, label: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} is not a lowercase SHA-256")
    text = value
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return text


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} is not a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} is not a nonnegative integer")
    return value


def _finite_number(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} is not an exact finite JSON float")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if type(value) is not str or not value or len(value) > 4096:
        raise ValueError(f"{label} is not an exact bounded JSON string")
    return value


def _absolute_parts(path: Path) -> Tuple[str, ...]:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("certification paths must be literal absolute paths without '..'")
    parts = tuple(part for part in path.parts if part not in ("", "/"))
    if not parts:
        raise ValueError("filesystem root is not a certification file path")
    return parts


def _open_parent_no_follow(path: Path) -> Tuple[int, str]:
    parts = _absolute_parts(path)
    parent_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in parts[:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            os.close(parent_fd)
            parent_fd = next_fd
        return parent_fd, parts[-1]
    except Exception:
        os.close(parent_fd)
        raise


def read_regular_bytes(path: Path) -> bytes:
    parent_fd, name = _open_parent_no_follow(path)
    fd = None
    try:
        fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"certification input is not one regular single-link file: {path}")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        if (info.st_dev, info.st_ino, info.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise ValueError(f"certification input changed while being read: {path}")
        return b"".join(chunks)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent_fd)


def read_json(path: Path) -> Dict[str, Any]:
    return _strict_canonical_json(
        read_regular_bytes(path), f"certification input {path.name}"
    )


def exclusive_write(path: Path, payload: bytes) -> Dict[str, Any]:
    """Create once through a held directory FD; never follow a leaf symlink."""

    parent_fd, name = _open_parent_no_follow(path)
    fd = None
    try:
        fd = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("exclusive certification write made no progress")
            view = view[written:]
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != len(payload):
            raise ValueError("exclusive certification output inode is invalid")
        linked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (linked.st_dev, linked.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError("exclusive certification output path changed after creation")
        os.fsync(parent_fd)
        return {
            "path": str(path),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "device": int(info.st_dev),
            "inode": int(info.st_ino),
            "mode": "0444",
            "creation": "O_CREAT|O_EXCL|O_NOFOLLOW",
        }
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent_fd)


def _load_runtime_module(repo_root: Path):
    path = repo_root / "gsplat/compression/h007_runtime_provenance.py"
    spec = importlib.util.spec_from_file_location("h007_stage02_runtime", path)
    if spec is None or spec.loader is None:
        raise ValueError("Stage-02 runtime-provenance verifier cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sequence_module(repo_root: Path):
    path = repo_root / "gsplat/compression/h007_sequence_container.py"
    spec = importlib.util.spec_from_file_location("h007_stage02_sequence", path)
    if spec is None or spec.loader is None:
        raise ValueError("Stage-02 sequence verifier cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_evaluator_module(repo_root: Path):
    path = repo_root / EVALUATOR_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location("h007_stage02_evaluator", path)
    if spec is None or spec.loader is None:
        raise ValueError("Stage-02 H-DOWN evaluator cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bound_source(root: Path, declared: Any, label: str) -> Path:
    declared_text = _nonempty_string(declared, f"{label} path")
    relative = Path(declared_text)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in declared_text
    ):
        raise ValueError(f"{label} path is not a safe relative path")
    path = root / relative
    # read_regular_bytes walks every parent component through O_NOFOLLOW.
    read_regular_bytes(Path(os.path.abspath(os.fspath(path))))
    return path


def _verify_runtime(
    repo_root: Path, provenance_manifest: Path, provenance_manifest_sha256: str
) -> Dict[str, Any]:
    module = _load_runtime_module(repo_root)
    receipt = module.verify_runtime_provenance(
        provenance_manifest,
        repo_root,
        require_sha256(provenance_manifest_sha256, "patch-chain manifest SHA-256"),
    )
    if len(receipt.get("patch_sha256", [])) != 9:
        raise ValueError("Stage 02 requires the exact nine-stage runtime provenance")
    return receipt


def _fixed_reference_paths(root: Path, scene: str, gop_id: int) -> Tuple[Path, Path]:
    directory = root / "reference_cases" / scene / f"gop_{gop_id}"
    return directory / "reference.json", directory / "reference.npz"


def _validate_reference_grid(
    root: Path, evaluator_sha256: str, repo_root: Path
) -> Tuple[
    Dict[Tuple[str, int], Dict[str, Any]],
    Dict[str, str],
    Dict[Tuple[str, int], Dict[str, Any]],
]:
    cases: Dict[Tuple[str, int], Dict[str, Any]] = {}
    files: Dict[str, str] = {}
    rebuilds: Dict[Tuple[str, int], Dict[str, Any]] = {}
    evaluator_module = None
    for scene in CONFIRMATORY_SCENES:
        for gop_id, start in enumerate(GOP_STARTS):
            manifest_path, artifact_path = _fixed_reference_paths(root, scene, gop_id)
            manifest_payload = read_regular_bytes(manifest_path)
            manifest = _strict_canonical_json(
                manifest_payload, "Stage-02 reference manifest"
            )
            if (
                manifest.get("schema") != REFERENCE_SCHEMA
                or manifest.get("scene") != scene
                or _nonnegative_int(
                    manifest.get("gop_id"), "Stage-02 reference GOP ID"
                )
                != gop_id
                or _nonnegative_int(
                    manifest.get("gop_start_frame"),
                    "Stage-02 reference GOP start",
                )
                != start
                or manifest.get("evaluator_sha256") != evaluator_sha256
                or manifest.get("candidate_inputs_read") != []
                or manifest.get("outcome_fields_read") != []
                or manifest.get("status") not in {"ELIGIBLE", "REFERENCE_INELIGIBLE"}
            ):
                raise ValueError(f"Stage-02 reference identity/provenance mismatch: {scene}/{gop_id}")
            relative_manifest = manifest_path.relative_to(root).as_posix()
            files[relative_manifest] = sha256_bytes(manifest_payload)
            if manifest["status"] == "ELIGIBLE":
                artifact_payload = read_regular_bytes(artifact_path)
                if (
                    _nonempty_string(
                        manifest.get("artifact"), "Stage-02 reference artifact path"
                    )
                    != str(artifact_path)
                    or _positive_int(
                        manifest.get("artifact_bytes"),
                        "Stage-02 reference artifact bytes",
                    )
                    != len(artifact_payload)
                    or manifest.get("artifact_sha256") != sha256_bytes(artifact_payload)
                    or not isinstance(manifest.get("cases"), list)
                    or [row.get("label") for row in manifest["cases"]]
                    != ["primary", "static"]
                ):
                    raise ValueError(f"eligible Stage-02 reference artifact mismatch: {scene}/{gop_id}")
                files[artifact_path.relative_to(root).as_posix()] = sha256_bytes(
                    artifact_payload
                )
                if evaluator_module is None:
                    evaluator_module = _load_evaluator_module(repo_root)
                rebuild = evaluator_module.verify_reference_rebuild(
                    manifest_path, artifact_path
                )
                if (
                    rebuild.get("schema") != "h007.reference_rebuild_audit.v1"
                    or rebuild.get("scene") != scene
                    or _nonnegative_int(
                        rebuild.get("gop_id"), "Stage-02 rebuilt reference GOP ID"
                    )
                    != gop_id
                    or rebuild.get("status") != "ELIGIBLE"
                    or rebuild.get("artifact_sha256") != sha256_bytes(artifact_payload)
                    or rebuild.get("source_inputs_revalidated") is not True
                    or rebuild.get("byte_reproducible") is not True
                    or rebuild.get("status_reproducible") is not True
                ):
                    raise ValueError(
                        f"eligible Stage-02 reference rebuild failed: {scene}/{gop_id}"
                    )
                rebuilds[(scene, gop_id)] = rebuild
            else:
                try:
                    read_regular_bytes(artifact_path)
                except FileNotFoundError:
                    pass
                else:
                    raise ValueError("ineligible Stage-02 reference unexpectedly has an artifact")
                if "artifact_sha256" in manifest or not _nonempty_string(
                    manifest.get("reason"), "ineligible reference reason"
                ):
                    raise ValueError("ineligible Stage-02 reference record is malformed")
                if evaluator_module is None:
                    evaluator_module = _load_evaluator_module(repo_root)
                rebuild = evaluator_module.verify_reference_rebuild(
                    manifest_path, artifact_path
                )
                if (
                    rebuild.get("schema") != "h007.reference_rebuild_audit.v1"
                    or rebuild.get("scene") != scene
                    or _nonnegative_int(
                        rebuild.get("gop_id"), "Stage-02 rebuilt reference GOP ID"
                    )
                    != gop_id
                    or rebuild.get("status") != "REFERENCE_INELIGIBLE"
                    or rebuild.get("reason") != manifest.get("reason")
                    or rebuild.get("artifact_sha256") is not None
                    or rebuild.get("source_inputs_revalidated") is not True
                    or rebuild.get("status_reproducible") is not True
                ):
                    raise ValueError(
                        f"ineligible Stage-02 reference replay failed: {scene}/{gop_id}"
                    )
                rebuilds[(scene, gop_id)] = rebuild
            cases[(scene, gop_id)] = manifest
    return cases, files, rebuilds


def _validate_selection_shape(selection: Mapping[str, Any]) -> None:
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError("Stage-02 operating-point selection schema is unsupported")
    selected = selection.get("selected")
    if not isinstance(selected, list):
        raise ValueError("Stage-02 selection lacks selected rows")
    identities = [(row.get("scene"), row.get("method")) for row in selected]
    expected = [
        (scene, method)
        for scene in CONFIRMATORY_SCENES
        for method in ("official", "ap-gifstream-full")
    ]
    if sorted(identities) != sorted(expected) or len(set(identities)) != len(expected):
        raise ValueError("Stage-02 selection is not the exact five-scene official/AP grid")
    sequence_contract = _load_sequence_module(Path(__file__).resolve().parents[2])
    for row in selected:
        require_sha256(row.get("archive_sha256"), "selected sequence SHA-256")
        require_sha256(
            row.get("eligibility_source_sha256"), "selected eligibility evidence SHA-256"
        )
        recomputed = row.get("eligibility_recomputed")
        try:
            sequence_contract.validate_eligibility_recomputation_contract(
                recomputed,
                expected_scene=_nonempty_string(
                    row.get("scene"), "selected scene"
                ),
                expected_point_id=_nonempty_string(
                    row.get("point_id"), "selected point ID"
                ),
                expected_source_evidence_sha256=require_sha256(
                    row.get("eligibility_source_sha256"),
                    "selected eligibility evidence SHA-256",
                ),
                expected_archive_bytes=_positive_int(
                    row.get("archive_bytes"), "selected sequence archive bytes"
                ),
                expected_training_config_sha256=require_sha256(
                    row.get("training_config_sha256"),
                    "selected training config SHA-256",
                ),
                expected_seed=_nonnegative_int(
                    row.get("seed"), "selected sequence seed"
                ),
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                "Stage-02 selection lacks the complete v4 H-SOTA recomputation closure"
            ) from error
        for name in (
            "registry_base",
            "archive_registry_relative",
            "eligibility_receipt_registry_relative",
            "eligibility_receipt_path",
            "eligibility_source_path",
        ):
            try:
                _nonempty_string(row.get(name), f"selected {name}")
            except ValueError as error:
                raise ValueError(
                    "Stage-02 selection lacks eligibility source paths"
                ) from error
            if not row[name]:
                raise ValueError("Stage-02 selection lacks eligibility source paths")


def _validate_preconditions_shape(preconditions: Mapping[str, Any]) -> None:
    if (
        preconditions.get("schema") != PRECONDITION_SCHEMA
        or set(preconditions) != {"schema", "rows", "outcome_fields_read"}
        or preconditions.get("outcome_fields_read")
        != ["ordinary_unedited_fidelity", "real_container_accounting"]
    ):
        raise ValueError("Stage-02 image/rate precondition schema is unsupported")
    rows = preconditions.get("rows")
    if not isinstance(rows, list) or [row.get("scene") for row in rows] != list(
        CONFIRMATORY_SCENES
    ):
        raise ValueError("Stage-02 preconditions are not the ordered five-scene grid")


def _revalidate_stage02_sources(
    *,
    root: Path,
    repo_root: Path,
    selection: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> Dict[str, Any]:
    """Reopen eligibility and ordinary precondition sources at freeze/validate."""

    sequence_module = _load_sequence_module(repo_root)
    eligibility_audits = []
    selected = selection["selected"]
    selected_map = {(row["scene"], row["method"]): row for row in selected}
    comparison_keys = {
        "scene",
        "method",
        "point_id",
        "archive",
        "registry_base",
        "archive_registry_relative",
        "eligibility_receipt_registry_relative",
        "eligibility_receipt_path",
        "eligibility_source_path",
        "training_config_sha256",
        "seed",
        "archive_bytes",
        "archive_sha256",
        "eligibility_receipt_sha256",
        "eligibility_source_sha256",
        "eligibility_recomputed",
    }
    for row in sorted(selected, key=lambda value: (value["scene"], value["method"])):
        recomputed = sequence_module.revalidate_selected_eligibility(row)
        for key in comparison_keys:
            if recomputed.get(key) != row.get(key):
                raise ValueError(
                    f"Stage-02 eligibility source replay differs: {row['scene']}/{row['method']}/{key}"
                )
        eligibility_audits.append(
            {
                "scene": row["scene"],
                "method": row["method"],
                "point_id": row["point_id"],
                "archive_sha256": row["archive_sha256"],
                "eligibility_receipt_sha256": row[
                    "eligibility_receipt_sha256"
                ],
                "eligibility_source_sha256": row["eligibility_source_sha256"],
                "selected_evaluator_receipt_sha256": row[
                    "eligibility_recomputed"
                ]["selected_evaluator_receipt_sha256"],
                "source_recomputed": True,
            }
        )

    precondition_audits = []
    required = {
        "scene",
        "official_archive_sha256",
        "ap_archive_sha256",
        "official_bytes",
        "ap_bytes",
        "frame_count",
        "psnr_official",
        "psnr_ap",
        "ssim_official",
        "ssim_ap",
        "lpips_official",
        "lpips_ap",
        "official_evaluator_receipt",
        "official_evaluator_receipt_sha256",
        "ap_evaluator_receipt",
        "ap_evaluator_receipt_sha256",
    }
    for row in preconditions["rows"]:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("precondition fields are incomplete or unexpected")
        scene = _nonempty_string(row.get("scene"), "precondition scene")
        official = selected_map.get((scene, "official"))
        ap = selected_map.get((scene, "ap-gifstream-full"))
        if official is None or ap is None:
            raise ValueError(f"precondition selection pair is unavailable: {scene}")
        if (
            row["official_archive_sha256"] != official["archive_sha256"]
            or row["ap_archive_sha256"] != ap["archive_sha256"]
            or _positive_int(row["official_bytes"], "official precondition bytes")
            != _positive_int(
                official["archive_bytes"], "selected official archive bytes"
            )
            or _positive_int(row["ap_bytes"], "AP precondition bytes")
            != _positive_int(ap["archive_bytes"], "selected AP archive bytes")
            or _positive_int(row["frame_count"], "precondition frame count") != 300
        ):
            raise ValueError(f"precondition/selected-container identity mismatch: {scene}")
        official_recomputed = official["eligibility_recomputed"]
        ap_recomputed = ap["eligibility_recomputed"]
        evaluator_bindings = (
            (
                "official",
                official_recomputed,
                row["official_evaluator_receipt"],
                row["official_evaluator_receipt_sha256"],
            ),
            (
                "ap",
                ap_recomputed,
                row["ap_evaluator_receipt"],
                row["ap_evaluator_receipt_sha256"],
            ),
        )
        for label, recomputed, declared_path, declared_sha in evaluator_bindings:
            path = Path(
                _nonempty_string(
                    declared_path, f"{label} evaluator receipt path"
                )
            )
            payload = read_regular_bytes(
                Path(os.path.abspath(os.fspath(path)))
            )
            digest = require_sha256(
                declared_sha, f"{label} evaluator receipt SHA-256"
            )
            if (
                str(path) != recomputed["selected_evaluator_receipt_path"]
                or digest
                != recomputed["selected_evaluator_receipt_sha256"]
                or sha256_bytes(payload) != digest
            ):
                raise ValueError(
                    f"precondition evaluator receipt differs from eligibility replay: {scene}/{label}"
                )
        metrics = {
            "psnr_official": official_recomputed["selected_metrics"]["psnr"],
            "psnr_ap": ap_recomputed["selected_metrics"]["psnr"],
            "ssim_official": official_recomputed["selected_metrics"]["ssim"],
            "ssim_ap": ap_recomputed["selected_metrics"]["ssim"],
            "lpips_official": official_recomputed["selected_metrics"]["lpips"],
            "lpips_ap": ap_recomputed["selected_metrics"]["lpips"],
        }
        for name, expected in metrics.items():
            if abs(
                _finite_number(row[name], f"precondition metric {name}")
                - _finite_number(expected, f"recomputed metric {name}")
            ) > 1e-12:
                raise ValueError(
                    f"precondition metric is not recomputed from 300-frame evaluator rows: {scene}/{name}"
                )
        official_bytes = _positive_int(
            row["official_bytes"], "official precondition bytes"
        )
        ap_bytes = _positive_int(row["ap_bytes"], "AP precondition bytes")
        rate_error = abs(ap_bytes - official_bytes) / official_bytes
        psnr_delta = _finite_number(
            row["psnr_ap"], "AP precondition PSNR"
        ) - _finite_number(row["psnr_official"], "official precondition PSNR")
        ssim_delta = _finite_number(
            row["ssim_ap"], "AP precondition SSIM"
        ) - _finite_number(row["ssim_official"], "official precondition SSIM")
        lpips_delta = _finite_number(
            row["lpips_ap"], "AP precondition LPIPS"
        ) - _finite_number(row["lpips_official"], "official precondition LPIPS")
        if not all(
            math.isfinite(value)
            for value in (rate_error, psnr_delta, ssim_delta, lpips_delta)
        ):
            raise ValueError(f"nonfinite image/rate precondition metric: {scene}")
        precondition_audits.append(
            {
                "scene": scene,
                "official_evaluator_receipt_sha256": row[
                    "official_evaluator_receipt_sha256"
                ],
                "ap_evaluator_receipt_sha256": row[
                    "ap_evaluator_receipt_sha256"
                ],
                "pass": bool(
                    rate_error <= 0.0125
                    and psnr_delta >= -0.10
                    and ssim_delta >= -0.002
                    and lpips_delta <= 0.005
                ),
                "rate_error": rate_error,
                "psnr_delta_db": psnr_delta,
                "ssim_delta": ssim_delta,
                "lpips_delta": lpips_delta,
                "source_recomputed": True,
            }
        )
    result = {
        "schema": "h007.stage02_source_revalidation.v1",
        "eligibility": eligibility_audits,
        "preconditions": precondition_audits,
        "outcome_fields_read": [
            "ordinary_unedited_fidelity",
            "real_container_accounting",
        ],
    }
    result["revalidation_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def _require_preconditions_pass(source_revalidation: Mapping[str, Any]) -> None:
    rows = source_revalidation.get("preconditions")
    if (
        not isinstance(rows, list)
        or [row.get("scene") for row in rows] != sorted(CONFIRMATORY_SCENES)
        or any(row.get("pass") is not True for row in rows)
    ):
        raise ValueError("Stage-02 ordinary image/rate precondition gate did not pass")


def freeze_stage02(
    *,
    root: Path,
    output: Path,
    repo_root: Path,
    provenance_manifest: Path,
    provenance_manifest_sha256: str,
) -> Dict[str, Any]:
    """Exclusively freeze the exact Stage-02 closure.

    A second invocation is intentionally rejected because ``stage02_contract``,
    ``runtime_provenance`` and the final marker are all create-once files.
    """

    root = Path(os.path.abspath(os.fspath(root)))
    output = Path(os.path.abspath(os.fspath(output)))
    repo_root = Path(os.path.abspath(os.fspath(repo_root)))
    provenance_manifest = Path(os.path.abspath(os.fspath(provenance_manifest)))
    if output != root / FREEZE_NAME:
        raise ValueError(f"Stage-02 freeze marker must be {FREEZE_NAME} at the root")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Stage-02 root is unavailable or a symlink")
    selection_path = root / SELECTION_NAME
    preconditions_path = root / PRECONDITIONS_NAME
    selection_payload = read_regular_bytes(selection_path)
    preconditions_payload = read_regular_bytes(preconditions_path)
    selection = _strict_canonical_json(
        selection_payload, "Stage-02 operating-point selection"
    )
    preconditions = _strict_canonical_json(
        preconditions_payload, "Stage-02 image/rate preconditions"
    )
    _validate_selection_shape(selection)
    _validate_preconditions_shape(preconditions)
    source_revalidation = _revalidate_stage02_sources(
        root=root,
        repo_root=repo_root,
        selection=selection,
        preconditions=preconditions,
    )
    _require_preconditions_pass(source_revalidation)
    evaluator_path = repo_root / EVALUATOR_RELATIVE_PATH
    evaluator_payload = read_regular_bytes(evaluator_path)
    evaluator_sha = sha256_bytes(evaluator_payload)
    clean_decoder_sha = sha256_bytes(
        read_regular_bytes(repo_root / CLEAN_DECODER_RELATIVE_PATH)
    )
    reference_cases, reference_files, reference_rebuilds = _validate_reference_grid(
        root, evaluator_sha, repo_root
    )
    reference_rebuild_record = {
        "schema": "h007.stage02_reference_rebuilds.v1",
        "rows": [
            reference_rebuilds[key] for key in sorted(reference_rebuilds)
        ],
        "eligible_rebuild_count": sum(
            row["status"] == "ELIGIBLE" for row in reference_rebuilds.values()
        ),
        "eligible_byte_reproducible": all(
            row["byte_reproducible"]
            for row in reference_rebuilds.values()
            if row["status"] == "ELIGIBLE"
        ),
        "all_status_reproducible": all(
            row["status_reproducible"] for row in reference_rebuilds.values()
        ),
    }
    runtime = _verify_runtime(
        repo_root, provenance_manifest, provenance_manifest_sha256
    )

    contract = {
        "schema": STAGE02_CONTRACT_SCHEMA,
        "repo_root": str(repo_root),
        "evaluator_relative_path": EVALUATOR_RELATIVE_PATH,
        "evaluator_sha256": evaluator_sha,
        "clean_decoder_relative_path": CLEAN_DECODER_RELATIVE_PATH,
        "clean_decoder_sha256": clean_decoder_sha,
        "confirmatory_scenes": list(CONFIRMATORY_SCENES),
        "gop_starts": list(GOP_STARTS),
        "reference_case_count": 25,
        "selection_sha256": sha256_bytes(selection_payload),
        "preconditions_sha256": sha256_bytes(preconditions_payload),
        "source_revalidation_sha256": source_revalidation[
            "revalidation_sha256"
        ],
        "reference_rebuilds_sha256": sha256_bytes(
            canonical_json_bytes(reference_rebuild_record)
        ),
        "provenance_manifest": str(provenance_manifest),
        "provenance_manifest_sha256": require_sha256(
            provenance_manifest_sha256, "patch-chain manifest SHA-256"
        ),
        "outcome_fields_read": [],
    }
    runtime_payload = canonical_json_bytes(runtime)
    contract_payload = canonical_json_bytes(contract)
    source_revalidation_payload = canonical_json_bytes(source_revalidation)
    reference_rebuild_payload = canonical_json_bytes(reference_rebuild_record)
    exclusive_write(root / RUNTIME_RECEIPT_NAME, runtime_payload)
    exclusive_write(root / SOURCE_REVALIDATION_NAME, source_revalidation_payload)
    exclusive_write(root / REFERENCE_REBUILD_NAME, reference_rebuild_payload)
    exclusive_write(root / CONTRACT_NAME, contract_payload)
    files = {
        SELECTION_NAME: sha256_bytes(selection_payload),
        PRECONDITIONS_NAME: sha256_bytes(preconditions_payload),
        RUNTIME_RECEIPT_NAME: sha256_bytes(runtime_payload),
        SOURCE_REVALIDATION_NAME: sha256_bytes(source_revalidation_payload),
        REFERENCE_REBUILD_NAME: sha256_bytes(reference_rebuild_payload),
        CONTRACT_NAME: sha256_bytes(contract_payload),
        **reference_files,
    }
    rows = [{"path": name, "sha256": files[name]} for name in sorted(files)]
    closure_digest = hashlib.sha256()
    for row in rows:
        encoded = row["path"].encode("utf-8")
        closure_digest.update(len(encoded).to_bytes(8, "little"))
        closure_digest.update(encoded)
        closure_digest.update(bytes.fromhex(row["sha256"]))
    freeze = {
        "schema": STAGE02_SCHEMA,
        "state": "FROZEN_APPEND_ONLY",
        "root": str(root),
        "reference_case_count": len(reference_cases),
        "eligible_reference_case_count": sum(
            row["status"] == "ELIGIBLE" for row in reference_cases.values()
        ),
        "selection_sha256": sha256_bytes(selection_payload),
        "preconditions_sha256": sha256_bytes(preconditions_payload),
        "runtime_provenance_sha256": sha256_bytes(runtime_payload),
        "evaluator_sha256": evaluator_sha,
        "clean_decoder_sha256": clean_decoder_sha,
        "source_revalidation_sha256": source_revalidation[
            "revalidation_sha256"
        ],
        "reference_rebuilds_sha256": sha256_bytes(reference_rebuild_payload),
        "file_count": len(rows),
        "files": rows,
        "closure_sha256": closure_digest.hexdigest(),
        "creation": "O_CREAT|O_EXCL|O_NOFOLLOW;0444;second_freeze_rejected",
        "candidate_inputs_read": [],
        "outcome_fields_read": [],
    }
    exclusive_write(output, canonical_json_bytes(freeze))
    return freeze


def validate_stage02_freeze(
    freeze_path: Path, expected_sha256: str
) -> Dict[str, Any]:
    freeze_path = Path(os.path.abspath(os.fspath(freeze_path)))
    if freeze_path.name != FREEZE_NAME:
        raise ValueError("Stage-02 freeze marker has a noncanonical basename")
    payload = read_regular_bytes(freeze_path)
    if sha256_bytes(payload) != require_sha256(expected_sha256, "Stage-02 freeze SHA-256"):
        raise ValueError("Stage-02 freeze SHA-256 mismatch")
    freeze = _strict_canonical_json(payload, "Stage-02 freeze marker")
    required = {
        "schema",
        "state",
        "root",
        "reference_case_count",
        "eligible_reference_case_count",
        "selection_sha256",
        "preconditions_sha256",
        "runtime_provenance_sha256",
        "evaluator_sha256",
        "clean_decoder_sha256",
        "source_revalidation_sha256",
        "reference_rebuilds_sha256",
        "file_count",
        "files",
        "closure_sha256",
        "creation",
        "candidate_inputs_read",
        "outcome_fields_read",
    }
    if not isinstance(freeze, dict) or set(freeze) != required:
        raise ValueError("Stage-02 freeze fields are incomplete or unexpected")
    root = freeze_path.parent
    if (
        freeze["schema"] != STAGE02_SCHEMA
        or freeze["state"] != "FROZEN_APPEND_ONLY"
        or freeze["root"] != str(root)
        or _positive_int(
            freeze["reference_case_count"], "Stage-02 reference case count"
        )
        != 25
        or freeze["candidate_inputs_read"] != []
        or freeze["outcome_fields_read"] != []
    ):
        raise ValueError("Stage-02 freeze identity/state mismatch")
    rows = freeze["files"]
    if not isinstance(rows, list) or _positive_int(
        freeze["file_count"], "Stage-02 frozen file count"
    ) != len(rows):
        raise ValueError("Stage-02 freeze inventory count mismatch")
    declared = {}
    digest = hashlib.sha256()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("Stage-02 freeze inventory row fields are unexpected")
        name = _nonempty_string(row["path"], "Stage-02 inventory path")
        path = Path(name)
        if not name or path.is_absolute() or ".." in path.parts or name in declared:
            raise ValueError("Stage-02 freeze inventory path is unsafe or duplicated")
        member_payload = read_regular_bytes(root / path)
        member_sha = require_sha256(row["sha256"], "Stage-02 member SHA-256")
        if sha256_bytes(member_payload) != member_sha:
            raise ValueError(f"Stage-02 frozen member changed: {name}")
        declared[name] = member_sha
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(bytes.fromhex(member_sha))
    if digest.hexdigest() != require_sha256(freeze["closure_sha256"], "closure SHA-256"):
        raise ValueError("Stage-02 closure digest mismatch")

    contract = read_json(root / CONTRACT_NAME)
    contract_required = {
        "schema",
        "repo_root",
        "evaluator_relative_path",
        "evaluator_sha256",
        "clean_decoder_relative_path",
        "clean_decoder_sha256",
        "confirmatory_scenes",
        "gop_starts",
        "reference_case_count",
        "selection_sha256",
        "preconditions_sha256",
        "source_revalidation_sha256",
        "reference_rebuilds_sha256",
        "provenance_manifest",
        "provenance_manifest_sha256",
        "outcome_fields_read",
    }
    if not isinstance(contract, dict) or set(contract) != contract_required:
        raise ValueError("Stage-02 contract fields are incomplete or unexpected")
    if (
        contract["schema"] != STAGE02_CONTRACT_SCHEMA
        or contract["evaluator_relative_path"] != EVALUATOR_RELATIVE_PATH
        or contract["clean_decoder_relative_path"] != CLEAN_DECODER_RELATIVE_PATH
        or contract["confirmatory_scenes"] != list(CONFIRMATORY_SCENES)
        or contract["gop_starts"] != list(GOP_STARTS)
        or _positive_int(
            contract["reference_case_count"], "Stage-02 contract reference count"
        )
        != 25
        or contract["outcome_fields_read"] != []
    ):
        raise ValueError("Stage-02 contract constants differ from preregistration")
    repo_root = Path(
        _nonempty_string(contract["repo_root"], "Stage-02 repository root")
    )
    evaluator_sha = sha256_bytes(read_regular_bytes(repo_root / EVALUATOR_RELATIVE_PATH))
    if evaluator_sha != contract["evaluator_sha256"] or evaluator_sha != freeze["evaluator_sha256"]:
        raise ValueError("frozen evaluator bytes changed")
    clean_decoder_sha = sha256_bytes(
        read_regular_bytes(repo_root / CLEAN_DECODER_RELATIVE_PATH)
    )
    if (
        clean_decoder_sha != contract["clean_decoder_sha256"]
        or clean_decoder_sha != freeze["clean_decoder_sha256"]
    ):
        raise ValueError("frozen clean decoder bytes changed")
    selection = read_json(root / SELECTION_NAME)
    preconditions = read_json(root / PRECONDITIONS_NAME)
    _validate_selection_shape(selection)
    _validate_preconditions_shape(preconditions)
    source_revalidation = _revalidate_stage02_sources(
        root=root,
        repo_root=repo_root,
        selection=selection,
        preconditions=preconditions,
    )
    _require_preconditions_pass(source_revalidation)
    source_revalidation_payload = canonical_json_bytes(source_revalidation)
    if (
        read_regular_bytes(root / SOURCE_REVALIDATION_NAME)
        != source_revalidation_payload
        or source_revalidation["revalidation_sha256"]
        != contract["source_revalidation_sha256"]
        or source_revalidation["revalidation_sha256"]
        != freeze["source_revalidation_sha256"]
    ):
        raise ValueError("Stage-02 eligibility/precondition source replay changed")
    if (
        sha256_bytes(read_regular_bytes(root / SELECTION_NAME)) != contract["selection_sha256"]
        or contract["selection_sha256"] != freeze["selection_sha256"]
        or sha256_bytes(read_regular_bytes(root / PRECONDITIONS_NAME))
        != contract["preconditions_sha256"]
        or contract["preconditions_sha256"] != freeze["preconditions_sha256"]
    ):
        raise ValueError("Stage-02 selection/precondition binding mismatch")
    runtime = _verify_runtime(
        repo_root,
        Path(
            _nonempty_string(
                contract["provenance_manifest"],
                "Stage-02 provenance manifest path",
            )
        ),
        require_sha256(
            contract["provenance_manifest_sha256"],
            "Stage-02 provenance manifest SHA-256",
        ),
    )
    runtime_payload = canonical_json_bytes(runtime)
    if (
        read_regular_bytes(root / RUNTIME_RECEIPT_NAME) != runtime_payload
        or sha256_bytes(runtime_payload) != freeze["runtime_provenance_sha256"]
    ):
        raise ValueError("Stage-02 nine-stage runtime receipt mismatch")
    reference_cases, reference_files, reference_rebuilds = _validate_reference_grid(
        root, evaluator_sha, repo_root
    )
    reference_rebuild_record = {
        "schema": "h007.stage02_reference_rebuilds.v1",
        "rows": [reference_rebuilds[key] for key in sorted(reference_rebuilds)],
        "eligible_rebuild_count": sum(
            row["status"] == "ELIGIBLE" for row in reference_rebuilds.values()
        ),
        "eligible_byte_reproducible": all(
            row["byte_reproducible"]
            for row in reference_rebuilds.values()
            if row["status"] == "ELIGIBLE"
        ),
        "all_status_reproducible": all(
            row["status_reproducible"] for row in reference_rebuilds.values()
        ),
    }
    reference_rebuild_payload = canonical_json_bytes(reference_rebuild_record)
    if (
        read_regular_bytes(root / REFERENCE_REBUILD_NAME)
        != reference_rebuild_payload
        or sha256_bytes(reference_rebuild_payload)
        != contract["reference_rebuilds_sha256"]
        or sha256_bytes(reference_rebuild_payload)
        != freeze["reference_rebuilds_sha256"]
    ):
        raise ValueError("Stage-02 reference rebuild evidence changed")
    expected_files = {
        SELECTION_NAME: freeze["selection_sha256"],
        PRECONDITIONS_NAME: freeze["preconditions_sha256"],
        RUNTIME_RECEIPT_NAME: freeze["runtime_provenance_sha256"],
        SOURCE_REVALIDATION_NAME: sha256_bytes(source_revalidation_payload),
        REFERENCE_REBUILD_NAME: sha256_bytes(reference_rebuild_payload),
        CONTRACT_NAME: sha256_bytes(read_regular_bytes(root / CONTRACT_NAME)),
        **reference_files,
    }
    if declared != dict(sorted(expected_files.items())):
        raise ValueError("Stage-02 freeze is not the exact required closure")
    if _nonnegative_int(
        freeze["eligible_reference_case_count"],
        "Stage-02 eligible reference count",
    ) != sum(
        row["status"] == "ELIGIBLE" for row in reference_cases.values()
    ):
        raise ValueError("Stage-02 eligible reference count mismatch")
    return {
        "freeze": freeze,
        "root": root,
        "contract": contract,
        "runtime_provenance": runtime,
        "selection": selection,
        "preconditions": preconditions,
        "source_revalidation": source_revalidation,
        "reference_rebuilds": reference_rebuilds,
        "references": reference_cases,
        "files": declared,
    }


def validate_case_static_closure(
    case: Mapping[str, Any],
    *,
    stage: Mapping[str, Any],
    sequence_validation: Mapping[str, Any],
    gop_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    """Reject unbound/manual case JSON before any metric is aggregated."""

    required = {
        "candidate_bundle",
        "candidate_clean_decode_manifest_sha256",
        "candidate_decoded_splats_sha256",
        "selected_sequence_archive_sha256",
        "selected_sequence_manifest_sha256",
        "selected_inner_gop_sha256",
        "selected_inner_gop_decoder_config_sha256",
        "evaluator_sha256",
        "runtime_provenance",
        "freeze_manifest_sha256",
    }
    if not required.issubset(case):
        raise ValueError("manual candidate case lacks the complete generation closure")
    if (
        case["selected_sequence_archive_sha256"]
        != sequence_validation["archive_sha256"]
        or case["selected_sequence_manifest_sha256"]
        != sequence_validation["sequence_manifest_sha256"]
        or case["selected_inner_gop_sha256"] != gop_audit["sha256"]
        or case["selected_inner_gop_decoder_config_sha256"]
        != gop_audit["decoder_config_sha256"]
        or case["evaluator_sha256"] != stage["freeze"]["evaluator_sha256"]
        or case["runtime_provenance"] != stage["runtime_provenance"]
    ):
        raise ValueError("candidate case generation closure differs from frozen evidence")
    bundle = Path(
        _nonempty_string(case["candidate_bundle"], "candidate bundle path")
    )
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("candidate clean-decode bundle is unavailable or a symlink")
    clean_path = bundle / "clean_decode_manifest.json"
    clean_payload = read_regular_bytes(Path(os.path.abspath(os.fspath(clean_path))))
    if sha256_bytes(clean_payload) != require_sha256(
        case["candidate_clean_decode_manifest_sha256"],
        "candidate clean-decode manifest SHA-256",
    ):
        raise ValueError("candidate clean-decode manifest changed")
    clean = _strict_canonical_json(
        clean_payload, "candidate clean-decode manifest"
    )
    decoded_path = bundle / _nonempty_string(
        clean.get("decoded_splats"), "decoded splats relative path"
    )
    decoded_payload = read_regular_bytes(Path(os.path.abspath(os.fspath(decoded_path))))
    decoded_sha = require_sha256(
        case["candidate_decoded_splats_sha256"], "candidate decoded tensor SHA-256"
    )
    if sha256_bytes(decoded_payload) != decoded_sha or clean.get(
        "decoded_splats_sha256"
    ) != decoded_sha:
        raise ValueError("candidate decoded tensor changed")
    expected_clean = {
        "source_sequence_archive_sha256": sequence_validation["archive_sha256"],
        "source_sequence_manifest_sha256": sequence_validation[
            "sequence_manifest_sha256"
        ],
        "source_inner_gop_sha256": gop_audit["sha256"],
        "source_inner_gop_decoder_config_sha256": gop_audit[
            "decoder_config_sha256"
        ],
        "runtime_provenance": stage["runtime_provenance"],
    }
    for key, expected in expected_clean.items():
        if clean.get(key) != expected:
            raise ValueError(f"clean-decode receipt is not bound to the selected GOP: {key}")
    return {
        "clean_decode_manifest_sha256": sha256_bytes(clean_payload),
        "decoded_splats_sha256": decoded_sha,
        "selected_sequence_sha256": sequence_validation["archive_sha256"],
        "selected_inner_gop_sha256": gop_audit["sha256"],
        "evaluator_sha256": stage["freeze"]["evaluator_sha256"],
    }
