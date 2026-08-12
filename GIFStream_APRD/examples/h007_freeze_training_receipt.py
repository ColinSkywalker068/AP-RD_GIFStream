#!/usr/bin/env python3
"""Freeze checkpoint bytes and decoded training state before codec production."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from gsplat.compression.ap_gifstream import canonical_json_bytes, tensor_mapping_sha256
from gsplat.compression.h007_runtime_provenance import verify_runtime_provenance
from gsplat.compression.h007_sequence_container import (
    _validate_producer_training_config_types,
)


SCHEMA = "h007.gifstream_frozen_training_receipt.v1"
OFFICIAL_COMMIT = "c98486632e7dafd830740b1a1692bd08c48b96e3"


def freeze(args: argparse.Namespace):
    output = Path(os.path.abspath(os.fspath(args.output)))
    if output.exists():
        raise ValueError("frozen training receipt output already exists")
    config_payload = args.training_config.read_bytes()
    config = json.loads(config_payload.decode("utf-8"))
    if not isinstance(config, dict):
        raise ValueError("producer training config is not a JSON object")
    checkpoints = []
    checkpoint_rows = []
    for raw in args.checkpoints:
        if raw.is_symlink() or not raw.is_file():
            raise ValueError("source checkpoint is unavailable or a symlink")
        payload = raw.read_bytes()
        checkpoint_rows.append(
            {
                "path": str(raw.resolve()),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        checkpoints.append(torch.load(raw, map_location="cpu", weights_only=True))
    if not checkpoints or any(not isinstance(row, dict) for row in checkpoints):
        raise ValueError("source checkpoint grid is empty or malformed")
    raw_steps = [row.get("step") for row in checkpoints]
    if any(type(step) is not int or step < 0 for step in raw_steps):
        raise ValueError("source checkpoints do not have exact integer training steps")
    steps = set(raw_steps)
    if len(steps) != 1:
        raise ValueError("source checkpoints do not share one training step")
    training_types = _validate_producer_training_config_types(
        config, "freeze producer training config"
    )
    max_steps = training_types["max_steps"]
    if next(iter(steps)) != max_steps - 1:
        raise ValueError("source checkpoints are not the terminal max_steps-1 state")
    state_positions = {row.get("state_position") for row in checkpoints}
    if state_positions != {"after_optimizer_entropy_and_strategy_post_backward"}:
        raise ValueError("source checkpoints are not post-update terminal states")
    splat_keys = set(checkpoints[0].get("splats", {}))
    if not splat_keys or any(set(row.get("splats", {})) != splat_keys for row in checkpoints):
        raise ValueError("source checkpoint splat mappings differ")
    splats = {
        name: torch.cat([row["splats"][name] for row in checkpoints], dim=0)
        for name in sorted(splat_keys)
    }
    decoders = checkpoints[0].get("decoders")
    if not isinstance(decoders, dict) or not decoders:
        raise ValueError("source checkpoint decoder state is unavailable")
    entropy = {
        key[: -len("_entropy_model")]: value
        for key, value in checkpoints[0].items()
        if key.endswith("_entropy_model") and isinstance(value, dict)
    }
    if not entropy:
        raise ValueError("source checkpoint entropy state is unavailable")
    scaling = checkpoints[0].get("scaling")
    if not isinstance(scaling, dict) or not scaling:
        raise ValueError("source checkpoint codec scaling is unavailable")
    app_opt = config["app_opt"]
    appearance = checkpoints[0].get("app_module")
    if app_opt and (not isinstance(appearance, dict) or not appearance):
        raise ValueError("source checkpoint appearance-module state is unavailable")
    if not app_opt and appearance is not None:
        raise ValueError("appearance-disabled training config contains appearance state")
    variant = str(config.get("variant", ""))
    ap_receipt_sha = None
    if variant == "official":
        if any(
            "ap_state" in checkpoint or "ap_training_receipt" in checkpoint
            for checkpoint in checkpoints
        ):
            raise ValueError("official checkpoint grid contains AP training state")
    else:
        ap_receipt = checkpoints[0].get("ap_training_receipt")
        if not isinstance(ap_receipt, dict):
            raise ValueError("AP checkpoint lacks its training receipt")
        ap_receipt_sha = hashlib.sha256(canonical_json_bytes(ap_receipt)).hexdigest()
    runtime = verify_runtime_provenance(
        args.provenance_manifest,
        args.repo_root,
        args.provenance_manifest_sha256,
    )
    receipt = {
        "schema": SCHEMA,
        "official_commit": OFFICIAL_COMMIT,
        "scene": str(config.get("scene", "")),
        "variant": variant,
        "training_step": next(iter(steps)),
        "state_position": "after_optimizer_entropy_and_strategy_post_backward",
        "training_config": config,
        "training_config_sha256": hashlib.sha256(
            canonical_json_bytes(config)
        ).hexdigest(),
        "source_checkpoints": checkpoint_rows,
        "model_state_sha256": {
            "splats": tensor_mapping_sha256(splats),
            "decoders": tensor_mapping_sha256(decoders),
            "entropy_models": {
                name: tensor_mapping_sha256(value)
                for name, value in sorted(entropy.items())
            },
            "codec_scaling": hashlib.sha256(
                canonical_json_bytes(scaling)
            ).hexdigest(),
            "appearance_module": (
                tensor_mapping_sha256(appearance) if app_opt else None
            ),
        },
        "ap_training_receipt_sha256": ap_receipt_sha,
        "runtime_provenance": runtime,
        "outcome_fields_read": [],
    }
    payload = canonical_json_bytes(receipt)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("frozen training receipt write made no progress")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--provenance-manifest-sha256", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args), sort_keys=True))


if __name__ == "__main__":
    main()
