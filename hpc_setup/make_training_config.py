#!/usr/bin/env python3
"""Emit the producer training-config JSON for a given trainer CLI invocation.

Parses the SAME argv the trainer sees (preset + overrides) through the same
tyro machinery and the same Config dataclass, applies the same steps_scaler
adjustment, and dumps ``producer_training_config(cfg)``.  The preset literals
are copied from simple_trainer_GIFStream.py's __main__ block (not importable);
a source-text guard below fails loudly if they drift upstream.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / "GIFStream_APRD"
sys.path.insert(0, str(REPO / "examples"))

import tyro  # noqa: E402
from simple_trainer_GIFStream import Config, producer_training_config  # noqa: E402
from gsplat.strategy import GIFStreamStrategy  # noqa: E402

# Preset copies -- guarded against upstream drift below.
CONFIGS = {
    "neur3d_0": (
        "neur3d dataset",
        Config(
            strategy=GIFStreamStrategy(verbose=True, densify_grad_threshold=0.0005, deformation_gate=0.03),
            test_set=[0],
            normalize_world_space=False,
            anchor_feature_dim=24,
            c_perframe=4,
            app_opt=True,
            app_embed_dim=6,
        ),
    ),
    "neur3d_1": (
        "neur3d dataset",
        Config(
            strategy=GIFStreamStrategy(verbose=True, densify_grad_threshold=0.0006, deformation_gate=0.03),
            test_set=[0],
            normalize_world_space=False,
            anchor_feature_dim=48,
            c_perframe=4,
            app_opt=False,
        ),
    ),
    "neur3d_2": (
        "neur3d dataset",
        Config(
            strategy=GIFStreamStrategy(verbose=True, densify_grad_threshold=0.0006, deformation_gate=0.03),
            test_set=[0],
            remove_set=[12],
            normalize_world_space=False,
            anchor_feature_dim=48,
            c_perframe=4,
            app_opt=False,
        ),
    ),
}

_TRAINER_SOURCE = (REPO / "examples" / "simple_trainer_GIFStream.py").read_text()
for token in (
    "densify_grad_threshold=0.0005",
    "densify_grad_threshold=0.0006",
    "anchor_feature_dim=24",
    "anchor_feature_dim=48",
    "remove_set=[12]",
    "app_embed_dim=6",
):
    if token not in _TRAINER_SOURCE:
        raise SystemExit(f"preset drift: {token!r} no longer in trainer source; update this helper")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    trainer_argv = [a for a in args.trainer_args if a != "--"]
    cfg = tyro.extras.overridable_config_cli(CONFIGS, args=trainer_argv)
    cfg.adjust_steps(cfg.steps_scaler)
    config = producer_training_config(cfg)
    args.output.write_text(json.dumps(config, indent=1, sort_keys=True))
    print(json.dumps(config, sort_keys=True))


if __name__ == "__main__":
    main()
