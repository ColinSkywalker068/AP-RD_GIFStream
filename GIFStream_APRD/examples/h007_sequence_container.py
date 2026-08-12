#!/usr/bin/env python3
"""Build, validate, or select counted five-GOP GIFStream sequence ZIPs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gsplat.compression.h007_sequence_container import (
    build_sequence_container,
    select_real_zip_operating_points,
    validate_sequence_container,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--scene", required=True)
    build.add_argument("--method", required=True)
    build.add_argument("--gop-archives", type=Path, nargs=5, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--training-config-sha256", required=True)
    build.add_argument("--seed", type=int, required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--archive", type=Path, required=True)
    validate.add_argument("--scene")
    validate.add_argument("--method")

    select = sub.add_parser("select")
    select.add_argument("--registry", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--methods", nargs="+", default=["ap-gifstream-full"])

    args = parser.parse_args()
    if args.command == "build":
        result = build_sequence_container(
            scene=args.scene,
            method=args.method,
            gop_archives=args.gop_archives,
            output=args.output,
            training_config_sha256=args.training_config_sha256,
            seed=args.seed,
        )
    elif args.command == "validate":
        result = validate_sequence_container(
            args.archive, expected_scene=args.scene, expected_method=args.method
        )
    else:
        result = select_real_zip_operating_points(
            args.registry, args.output, args.methods
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
