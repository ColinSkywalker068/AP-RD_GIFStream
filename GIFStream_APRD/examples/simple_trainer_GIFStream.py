import hashlib
import json
import math
import os
import time
import shutil
from contextlib import nullcontext
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from typing import Any, ContextManager, Dict, List, Mapping, Optional, Tuple, Union

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.GIFStream_new import Dataset, Parser
from datasets.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from gsplat.compression_simulation.ops import fake_quantize_factors
from utils import CameraEmbedding, knn, set_random_seed, find_k_neighbors
import random

from gsplat.compression import GIFStreamEnd2endCompression, GIFStream2dcodecCompression
from gsplat.compression.ap_gifstream import (
    AP_SCORE_SCHEMA,
    canonical_json_bytes,
    canonical_voxel_ids,
    build_count_preserving_anchor_allocation,
    build_equal_estimated_byte_allocation,
    deterministic_zip_directory,
    frozen_backbone_importance,
    file_byte_census,
    tensor_mapping_sha256,
    variant_spec,
)
from gsplat.compression.h007_runtime_provenance import verify_runtime_provenance
from gsplat.compression.h007_path_contract import (
    ANCHOR_FEATURE_BACKGROUND_MULTIPLIER,
    ANCHOR_FEATURE_PROTECTED_MULTIPLIER,
    FACTOR_BACKGROUND_MULTIPLIER,
    FACTOR_PROTECTED_MULTIPLIER,
    PATH_CONTRACT_SCHEMA,
    build_codec_knn_indices,
    build_path_input_precision_mask,
    deterministic_knn_indices,
    retained_knn_graph_sha256,
)
from gsplat.compression.h007_sequence_container import (
    DECODER_CONFIG_SCHEMA,
    FROZEN_RATE_LAMBDAS,
    PRODUCER_RECEIPT_SCHEMA,
    build_gifstream_payload_manifest,
    canonicalize_gifstream_png_payloads,
    canonicalize_gifstream_torch_payload,
    validate_frozen_training_receipt_contract,
)
from gsplat.distributed import cli
from gsplat.rendering import rasterization, view_to_visible_anchors
from gsplat.strategy import GIFStreamStrategy

from gsplat.compression_simulation.simulation import GIFStreamCompressionSimulation

H007_GIFSTREAM_SCENES = {
    "flame_salmon_1",
    "coffee_martini",
    "cook_spinach",
    "cut_roasted_beef",
    "flame_steak",
    "sear_steak",
}

class ProfilerConfig:
    def __init__(self):
        self.enabled = False
        self.activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        
        self.wait = 1 
        self.warmup = 2 
        self.active = 30_000 
        
        self.schedule = self._create_schedule()
        
        self.on_trace_ready = torch.profiler.tensorboard_trace_handler('./log/profiler')
        self.record_shapes = True
        self.profile_memory = True
        self.with_stack = True
    
    def _create_schedule(self):
        return torch.profiler.schedule(
            wait=self.wait,
            warmup=self.warmup,
            active=self.active,
        )
    
    def update_schedule(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.schedule = self._create_schedule()

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["end2end", "2dcodec"]] = None
    # Quantization parameters when set to hevc
    qp: Optional[int] = None

    # Enable profiler
    profiler_enabled: bool = False

    # Enable compression simulation
    compression_sim: bool = False

    # Enable entropy model
    entropy_model_opt: bool = False
    # Define the type of entropy model
    entropy_model_type: Literal["conditional_gaussian_model"] = "conditional_gaussian_model"
    # Bit-rate distortion trade-off parameter
    rd_lambda: float = 5e-4 # default: 1e-2
    # Steps to enable entropy model into training pipeline
    # conditional gaussian model:
    entropy_steps: Dict[str, int] = field(default_factory=lambda: {"anchors": -1, 
                                                                   "quats": 10_000, 
                                                                   "scales": 10_000, 
                                                                   "opacities": 10_000, 
                                                                   "anchor_features": 10_000, 
                                                                   "offsets": 10_000,
                                                                   "factors": 10_000,
                                                                   "time_features": 10_000,
                                                                   })

    
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera camera_modelmodel
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: GIFStreamStrategy = field(
        default_factory=GIFStreamStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Scale regularization
    scale_reg: float = 0.01

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 12
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # Dimensionality of anchor features
    anchor_feature_dim: int = 24
    # Dimensionality of entropy model predicting the distribution of anchor features
    entropy_channel: int = 8
    # Number offsets
    n_offsets: int = 5
    # voxel size for Scaffold-GS
    voxel_size = 0.01
    # whether add dist for neural gaussaian decoding mlps
    add_opacity_dist = False
    add_cov_dist = False
    add_cov_dist = False
    add_color_dist = False

    # GIFStream setup
    # Dimensionality of time-dependent feature per frame
    c_perframe: int = 8
    # GOP size for training
    GOP_size: int = 50
    # number of anchors for feature aggregation
    knn: bool = False
    n_knn: int = 6
    # time positional embedding dim, must be even
    time_dim: int = 16
    # time positional embedding base
    phi: float = 2
    # whether add view to neural gaussian decoding mlps
    view_adaptive: bool = False
    # test view cameras
    test_set: Optional[List[int]] = None
    # filter out cameras
    remove_set: Optional[List[int]] = None
    # regularization lambda
    factor_reg: float = 0.005
    smooth_reg: float = 0.005
    # GOP start frame
    start_frame: int = 0
    # whether continue from an existing ckpt
    continue_training: bool = False
    # rate number
    rate: int = 0
    # H007 AP-GIFStream switches.  flame_salmon_1 is development-only; the
    # other five names are the frozen confirmatory universe.
    ap_variant: Literal[
        "official",
        "random-full",
        "motion-full",
        "path-swap",
        "path-quant",
        "path-swap-quant",
        "ap-gifstream-full",
    ] = "official"
    ap_score_path: Optional[str] = None
    # External preregistration.  The manifest is intentionally not patched into
    # this repository, avoiding a code-tree/hash self-reference.
    ap_provenance_manifest: Optional[str] = None
    ap_provenance_manifest_sha256: Optional[str] = None
    # Codec production is a second, checkpoint-only phase.  This external
    # receipt is frozen after training and copied byte-for-byte into every GOP.
    h007_training_receipt: Optional[str] = None
    h007_training_receipt_sha256: Optional[str] = None
    ap_protected_fraction: float = 0.05
    ap_random_seed: int = 11
    ap_q_ap_multiplier: float = 0.5
    ap_q_bg_multiplier: float = 1.25
    ap_compression_seed: int = 20260715
    ap_freeze_step: int = 20_000
    ap_path_loss_lambda: float = 0.01
    ap_path_loss_every: int = 50
    # Optional deterministic parent-anchor recolor witness.  The same frozen
    # canonical-ID artifact is consumed by official and AP variants.
    ap_edit_ids_path: Optional[str] = None
    ap_edit_reference_manifest_path: Optional[str] = None
    ap_edit_strength: float = 0.75
    ap_warmup_renders: int = 5
    ap_timed_renders: int = 30
    # Optional reference-only export consumed by the isolated H-DOWN builder.
    # Only official, world-size-one runs may write this bundle.
    hdown_reference_export_dir: Optional[str] = None
    # quantization scalings
    compression_scaling = [
        {
            "anchors": None,
            "scales": 0.02,
            "quats": None,
            "opacities": None,
            "anchor_features": 1,
            "offsets": 0.02,
            "factors": 1/16,
            "time_features": 1,
        },
        {
            "anchors": None,
            "scales": 0.04,
            "quats": None,
            "opacities": None,
            "anchor_features": 1,
            "offsets": 0.04,
            "factors": 1/16,
            "time_features": 1,
        },
        {
            "anchors": None,
            "scales": 0.06,
            "quats": None,
            "opacities": None,
            "anchor_features": 1.5,
            "offsets": 0.06,
            "factors": 1/16,
            "time_features": 1.5,
        },
        {
            "anchors": None,
            "scales": 0.08,
            "quats": None,
            "opacities": None,
            "anchor_features": 2,
            "offsets": 0.08,
            "factors": 1/16,
            "time_features": 2,
        },
        {
            "anchors": None,
            "scales": 0.036,
            "quats": None,
            "opacities": None,
            "anchor_features": 1,
            "offsets": 0.036,
            "factors": 1/16,
            "time_features": 1,
        },
        {
            "anchors": None,
            "scales": 0.038,
            "quats": None,
            "opacities": None,
            "anchor_features": 1,
            "offsets": 0.038,
            "factors": 1/16,
            "time_features": 1,
        },
    ]

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)

        strategy = self.strategy
        if isinstance(strategy, GIFStreamStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    app_embed_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    voxel_size: int = 0.001,
    anchor_feature_dim: int = 48,
    n_offsets: int = 5,
    use_feat_bank: bool = False,
    add_opacity_dist: bool = False,
    add_cov_dist: bool = False,
    add_color_dist: bool = False,
    c_perframe: int = 4,
    GOP_size: int = 50,
    n_knn: int = 8,
    time_dim: int = 16,
    view_adaptive: bool = False
) -> Tuple[torch.nn.ParameterDict, torch.nn.ModuleDict, Dict[str, torch.optim.Optimizer], Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = parser.points
        np.random.shuffle(points)
        points = torch.from_numpy(np.unique(np.round(points/voxel_size), axis=0)*voxel_size).float()
    else:
        raise ValueError("Now only Support SFM Initialization")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 6)  # [N, 6]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.zeros((N, 4))  # [N, 4]
    quats[:,0] = 1
    opacities = torch.logit(torch.full((N,1), init_opacity))  # [N,]
    anchor_features = torch.zeros((N, anchor_feature_dim))
    offsets = torch.zeros((N, n_offsets, 3))
    time_features = torch.zeros((N, GOP_size, c_perframe))
    factors = torch.zeros((N, 4)) # [time_feature factor, motion_factor, knn_factor, pruning_factor]
    
    params = [
        # name, value, lr
        ("anchors", torch.nn.Parameter(points), 0),
        ("scales", torch.nn.Parameter(scales.requires_grad_(True)), 7e-3),
        ("quats", torch.nn.Parameter(quats), 0),
        ("opacities", torch.nn.Parameter(opacities), 0),
        ("offsets", torch.nn.Parameter(offsets.requires_grad_(True)), 1e-2),
        ("anchor_features", torch.nn.Parameter(anchor_features.requires_grad_(True)), 0.0075),
        ("time_features", torch.nn.Parameter(time_features.requires_grad_(True)), 0.0075),
        ("factors", torch.nn.Parameter(factors.requires_grad_(True)), 1e-3),
    ]

    view_dim = 3 if view_adaptive else 0
    opacity_dist_dim = 1 if add_opacity_dist else 0
    mlp_opacity = torch.nn.Sequential(
        torch.nn.Linear(anchor_feature_dim+view_dim+opacity_dist_dim+c_perframe, anchor_feature_dim),
        torch.nn.ReLU(True),
        torch.nn.Linear(anchor_feature_dim, n_offsets),
        torch.nn.Tanh()
    ).to(device)

    cov_dist_dim = 1 if add_cov_dist else 0
    mlp_cov = torch.nn.Sequential(
        torch.nn.Linear(anchor_feature_dim+view_dim+cov_dist_dim+c_perframe, anchor_feature_dim),
        torch.nn.ReLU(True),
        torch.nn.Linear(anchor_feature_dim, 7*n_offsets),
    ).to(device)


    color_dist_dim = 1 if add_color_dist else 0
    mlp_color = torch.nn.Sequential(
        torch.nn.Linear(anchor_feature_dim+view_dim+color_dist_dim+app_embed_dim+c_perframe, anchor_feature_dim),
        torch.nn.ReLU(True),
        torch.nn.Linear(anchor_feature_dim, 3*n_offsets),
        torch.nn.Sigmoid()
    ).to(device)

    # deformation net
    mlp_motion = torch.nn.Sequential(
        torch.nn.Linear(anchor_feature_dim+time_dim+c_perframe, anchor_feature_dim),
        torch.nn.ReLU(True),
        torch.nn.Linear(anchor_feature_dim, 3+4),
    ).to(device)
    torch.nn.init.constant_(mlp_motion[-1].weight,0)
    torch.nn.init.constant_(mlp_motion[-1].bias,0)

    net_params = [
        # name, value, lr
        ("mlp_opacity", mlp_opacity, 2e-3),
        ("mlp_cov", mlp_cov, 4e-3),
        ("mlp_color", mlp_color, 8e-3),
        ("mlp_motion", mlp_motion, 8e-3),
    ]

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    decoders = torch.nn.ModuleDict({n: v for n, v, _ in net_params}).to(device)

    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = torch.optim.Adam

    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    decoder_optimizers = {
        name: optimizer_class(
            [{"params": decoders[name].parameters(), "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in net_params
    }
    return splats, decoders, optimizers, decoder_optimizers


def producer_training_config(cfg: Config) -> Dict[str, Union[str, int, float, bool]]:
    # Sequence-shared training identity.  ``start_frame`` is deliberately bound
    # by each producer receipt instead: the five canonical GOPs must share one
    # training-config hash while retaining distinct frame-range identities.
    return {
        "scene": os.path.basename(os.path.normpath(cfg.data_dir)),
        "variant": cfg.ap_variant,
        "data_factor": int(cfg.data_factor),
        "GOP_size": int(cfg.GOP_size),
        "rate": str(cfg.rate),
        "rd_lambda": float(cfg.rd_lambda),
        "max_steps": int(cfg.max_steps),
        "random_seed": 42,
        "compression_seed": int(cfg.ap_compression_seed),
        "voxel_size": float(cfg.voxel_size),
        "anchor_feature_dim": int(cfg.anchor_feature_dim),
        "c_perframe": int(cfg.c_perframe),
        "entropy_channel": int(cfg.entropy_channel),
        "n_offsets": int(cfg.n_offsets),
        "n_knn": int(cfg.n_knn),
        "knn": bool(cfg.knn),
        "time_dim": int(cfg.time_dim),
        "view_adaptive": bool(cfg.view_adaptive),
        "app_opt": bool(cfg.app_opt),
        "compression_sim": bool(cfg.compression_sim),
        "entropy_model_opt": bool(cfg.entropy_model_opt),
    }


def verify_frozen_training_receipt(
    cfg: Config, runtime_provenance: Mapping[str, Any]
) -> Tuple[Dict, bytes]:
    if not cfg.ckpt:
        raise ValueError(
            "H007 codec production is checkpoint-only and requires frozen checkpoint inputs"
        )
    if not cfg.h007_training_receipt or not cfg.h007_training_receipt_sha256:
        raise ValueError("H007 codec production requires an external frozen training receipt")
    raw_receipt = Path(cfg.h007_training_receipt)
    if raw_receipt.is_symlink() or not raw_receipt.is_file():
        raise ValueError("frozen training receipt is unavailable or a symlink")
    payload = raw_receipt.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != cfg.h007_training_receipt_sha256:
        raise ValueError("frozen training receipt SHA-256 mismatch")
    receipt = json.loads(payload.decode("utf-8"))
    training_config = producer_training_config(cfg)
    checkpoint_rows = []
    for declared in cfg.ckpt:
        raw_checkpoint = Path(str(declared))
        if raw_checkpoint.is_symlink() or not raw_checkpoint.is_file():
            raise ValueError("frozen source checkpoint is unavailable or a symlink")
        checkpoint_payload = raw_checkpoint.read_bytes()
        checkpoint_rows.append(
            {
                "path": str(raw_checkpoint.resolve()),
                "bytes": len(checkpoint_payload),
                "sha256": hashlib.sha256(checkpoint_payload).hexdigest(),
            }
        )
    validated = validate_frozen_training_receipt_contract(
        receipt,
        expected_scene=str(training_config["scene"]),
        expected_variant=cfg.ap_variant,
        expected_training_config=training_config,
        expected_runtime_provenance=runtime_provenance,
        expected_source_checkpoints=checkpoint_rows,
    )
    return validated, payload


def validate_ap_entry_config(cfg: Config, world_size: int) -> Optional[Dict]:
    """Fail before dataset/result access for every H007 official/AP entry."""

    if cfg.hdown_reference_export_dir is not None and (
        cfg.ap_variant != "official" or int(world_size) != 1
    ):
        raise ValueError("H-DOWN reference export requires official variant and world_size=1")
    if cfg.ap_variant == "official" and cfg.ap_edit_ids_path is not None:
        if not cfg.ap_score_path or not cfg.ap_edit_reference_manifest_path:
            raise ValueError(
                "official edit evaluation requires the frozen score and reference manifest"
            )
    h007_codec_entry = cfg.compression == "end2end" or cfg.hdown_reference_export_dir is not None
    if cfg.ap_variant == "official" and not h007_codec_entry:
        return None
    scene = os.path.basename(os.path.normpath(cfg.data_dir))
    if scene not in H007_GIFSTREAM_SCENES:
        raise ValueError("AP-GIFStream scene is outside the frozen dev/confirmatory universe")
    if int(cfg.GOP_size) != 60:
        raise ValueError("U3 AP-GIFStream requires an exact 60-frame GOP")
    if int(world_size) != 1:
        raise ValueError("U3 AP-GIFStream frozen-ID allocation requires world_size=1")
    if int(cfg.rate) not in FROZEN_RATE_LAMBDAS or abs(
        float(cfg.rd_lambda) - FROZEN_RATE_LAMBDAS[int(cfg.rate)]
    ) > 1e-15:
        raise ValueError("H007 rate index/RD-lambda differs from the registered rate grid")
    if cfg.ap_variant != "official" and cfg.continue_training:
        raise ValueError("AP continuation is unsupported and rejected before Runner construction")
    if not cfg.compression_sim or not cfg.entropy_model_opt:
        raise ValueError("every H007 official/AP codec entry requires compression_sim and entropy_model_opt")
    if cfg.compression not in (None, "end2end"):
        raise ValueError("H007 official/AP entries support only the end2end codec")
    if not cfg.ap_provenance_manifest or not cfg.ap_provenance_manifest_sha256:
        raise ValueError("H007 official/AP entry requires an external preregistered provenance manifest and hash")
    repo_root = Path(__file__).resolve().parents[1]
    receipt = verify_runtime_provenance(
        Path(cfg.ap_provenance_manifest),
        repo_root,
        cfg.ap_provenance_manifest_sha256,
    )
    if h007_codec_entry:
        verify_frozen_training_receipt(cfg, receipt)
    if cfg.ap_variant != "official" and cfg.ap_score_path is not None and not Path(cfg.ap_score_path).is_file():
        raise ValueError("explicit AP score path does not exist")
    if cfg.ap_variant != "official" and cfg.ckpt is None and cfg.ap_score_path is not None:
        raise ValueError(
            "fresh AP training must generate its score path at the frozen step"
        )
    if cfg.ap_variant != "official" and cfg.ap_edit_ids_path is not None:
        if not cfg.ap_score_path or not cfg.ap_edit_reference_manifest_path:
            raise ValueError("AP edit evaluation requires score and reference-manifest paths")
        if not Path(cfg.ap_edit_ids_path).is_file() or not Path(
            cfg.ap_edit_reference_manifest_path
        ).is_file():
            raise ValueError("AP edit artifact/reference manifest is unavailable")
    return receipt


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        self.ap_runtime_provenance = validate_ap_entry_config(cfg, world_size)
        if cfg.compression == "end2end" or cfg.hdown_reference_export_dir is not None:
            (
                self.h007_training_receipt,
                self.h007_training_receipt_payload,
            ) = verify_frozen_training_receipt(cfg, self.ap_runtime_provenance)
        else:
            self.h007_training_receipt = None
            self.h007_training_receipt_payload = None
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)
        print("results will be saved to ", cfg.result_dir)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            first_frame=cfg.start_frame,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=False,
            test_set=cfg.test_set,
            remove_set=cfg.remove_set,
            GOP_size=cfg.GOP_size,
            start_frame=cfg.start_frame,
        )
        self.valset = Dataset(self.parser, split="val", test_set=cfg.test_set, remove_set=cfg.remove_set, GOP_size=cfg.GOP_size, start_frame=cfg.start_frame)
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        app_embed_dim = cfg.app_embed_dim if cfg.app_opt else 0
        self.splats,self.decoders, self.optimizers, self.net_optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=None,
            sparse_grad=False,
            visible_adam=False,
            batch_size=cfg.batch_size,
            app_embed_dim=app_embed_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            voxel_size=cfg.voxel_size,
            anchor_feature_dim=cfg.anchor_feature_dim,
            n_offsets=cfg.n_offsets,
            use_feat_bank=False,
            add_opacity_dist=cfg.add_opacity_dist,
            add_cov_dist=cfg.add_cov_dist,
            add_color_dist=cfg.add_color_dist,
            c_perframe=cfg.c_perframe,
            GOP_size=cfg.GOP_size,
            n_knn=cfg.n_knn,
            time_dim=cfg.time_dim,
            view_adaptive=cfg.view_adaptive
        )
        print("Model initialized. Number of Anchor:", len(self.splats["anchors"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, GIFStreamStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale,
                n_offsets=cfg.n_offsets,
                voxel_size=cfg.voxel_size,
                anchor_feature_dim=cfg.anchor_feature_dim
            )
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "end2end":
                self.compression_method = GIFStreamEnd2endCompression(
                    ap_config={
                        "variant": cfg.ap_variant,
                        "score_path": cfg.ap_score_path,
                        "protected_fraction": cfg.ap_protected_fraction,
                        "random_seed": cfg.ap_random_seed,
                        "q_ap_multiplier": cfg.ap_q_ap_multiplier,
                        "q_bg_multiplier": cfg.ap_q_bg_multiplier,
                        "n_knn": int(cfg.n_knn),
                        "compression_seed": cfg.ap_compression_seed,
                        "provenance_manifest_path": cfg.ap_provenance_manifest,
                        "provenance_manifest_sha256": cfg.ap_provenance_manifest_sha256,
                        "repo_root": str(Path(__file__).resolve().parents[1]),
                        "runtime_provenance": self.ap_runtime_provenance,
                        "scene": os.path.basename(os.path.normpath(cfg.data_dir)),
                    }
                )
            elif cfg.compression == "2dcodec":
                self.compression_method = GIFStream2dcodecCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")
        
        if cfg.compression_sim:
            cap_max = cfg.strategy.cap_max if cfg.strategy.cap_max is not None else None
            self.compression_sim_method = GIFStreamCompressionSimulation(cfg.entropy_model_opt,
                                                    cfg.entropy_model_type,
                                                    cfg.entropy_steps,
                                                    self.device,
                                                    False,
                                                    None,
                                                    None,
                                                    cap_max=cap_max,
                                                    feature_dim=cfg.anchor_feature_dim,
                                                    n_offsets=cfg.n_offsets,
                                                    c_channel=self.cfg.entropy_channel,
                                                    p_channel=self.cfg.c_perframe,
                                                    scaling=self.cfg.compression_scaling[self.cfg.rate],
                                                    max_steps=self.cfg.max_steps)

            if cfg.entropy_model_opt:
                selected_key = min((k for k, v in cfg.entropy_steps.items() if v > 0), key=lambda k: cfg.entropy_steps[k])
                self.entropy_min_step = cfg.entropy_steps[selected_key]
        if cfg.ap_variant != "official":
            if not cfg.knn:
                raise ValueError("Patch8 AP path contract requires KNN decoding")
            last_topology_boundary = max(
                int(cfg.strategy.refine_stop_iter), int(2 * cfg.max_steps / 3)
            )
            if cfg.ap_freeze_step < last_topology_boundary or cfg.ap_freeze_step >= cfg.max_steps:
                raise ValueError(
                    "ap_freeze_step must follow all topology/factor pruning and precede training end"
                )
            if cfg.ap_path_loss_every <= 0:
                raise ValueError("ap_path_loss_every must be positive")
            if variant_spec(cfg.ap_variant).action_loss and cfg.ap_path_loss_lambda <= 0:
                raise ValueError("full AP variants require a positive path-loss weight")
        self.ap_state = None
        self.ap_training_receipt = None
        self.ap_loss_applications = 0
        self.ap_loss_steps: List[int] = []
        self.ap_codec_knn_indices = None
        self._ap_factor_gradient_hook = None
        self._ap_edit_cache_signature = None
        self._ap_edit_anchor_mask = None
        self.ap_edit_audit = None
        if cfg.ap_edit_ids_path is not None and not (0.0 < cfg.ap_edit_strength <= 1.0):
            raise ValueError("ap_edit_strength must be in (0,1]")
        if cfg.ap_warmup_renders < 1 or cfg.ap_timed_renders < 1:
            raise ValueError("warm-render benchmark counts must be positive")
        
        # Profiler
        self.profiler: Optional[torch.profiler.profile] = None
        self.profiler_config = ProfilerConfig()
        if cfg.profiler_enabled:
            self.profiler_config.enabled = True

        self.app_optimizers = []
        if cfg.app_opt:
            assert cfg.app_embed_dim > 0
            self.app_module = CameraEmbedding(
                self.trainset.cameras_length, cfg.app_embed_dim
            ).to(self.device)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=False
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )
        if self.cfg.knn:
            self.indices = None
        self.istraining = False

    def init_dynamic(self) -> None:
        grads = self.strategy_state["offset_grad2d"] / self.strategy_state["offset_demon"]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1).view((-1,self.cfg.n_offsets)).mean(dim=-1)
        mini = grads_norm.min()
        maxi = grads_norm.max()
        grads_norm = ((grads_norm - mini) / (maxi - mini + 1e-6)).clamp(0.15, 1)
        grads_norm = - torch.log(1/grads_norm - 1)
        self.splats["factors"].data[:,1] = grads_norm

    def decode_persistent_anchor_paths(
        self,
        splats: Dict[str, Tensor],
        factors_are_logits: bool,
        knn_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """Decode view-independent persistent anchor centers for all GOP frames."""
        anchors = splats["anchors"]
        features = splats["anchor_features"]
        time_features = splats["time_features"]
        factors = (
            fake_quantize_factors(splats["factors"], q_aware=False)
            if factors_are_logits
            else splats["factors"]
        )
        if anchors.ndim != 2 or time_features.shape[:2] != (
            anchors.shape[0],
            self.cfg.GOP_size,
        ):
            raise ValueError("persistent path decoder received malformed splats")
        if self.cfg.knn:
            path_indices = (
                deterministic_knn_indices(anchors, int(self.cfg.n_knn))
                if knn_indices is None
                else knn_indices
            )
            if path_indices.shape != (anchors.shape[0], int(self.cfg.n_knn)):
                raise ValueError("persistent path KNN contract has an invalid shape")
            flat_indices = path_indices.reshape(-1)
            knn_features = features[flat_indices].reshape(
                -1, self.cfg.n_knn, features.shape[-1]
            ).mean(dim=1)
        paths = []
        base_i = torch.ones((1,), dtype=torch.float32, device=anchors.device)
        for frame in range(self.cfg.GOP_size):
            time_value = float(frame) / float(self.cfg.GOP_size - 1)
            frame_features = time_features[:, frame]
            if self.cfg.knn:
                knn_time = (
                    time_features[:, frame] * factors[:, 0].unsqueeze(-1)
                )[flat_indices].reshape(
                    -1, self.cfg.n_knn, self.cfg.c_perframe
                ).mean(dim=1)
                adaptive = factors[:, 2].unsqueeze(-1) * torch.cat(
                    [features, frame_features * factors[:, 0].unsqueeze(-1)], dim=-1
                ) + (1.0 - factors[:, 2].unsqueeze(-1)) * torch.cat(
                    [knn_features, knn_time], dim=-1
                )
            else:
                adaptive = torch.cat(
                    [features, frame_features * factors[:, 0].unsqueeze(-1)], dim=-1
                )
            embedding = torch.cat(
                [
                    torch.sin(self.cfg.phi**n * torch.pi * base_i * time_value)
                    for n in range(self.cfg.time_dim // 2)
                ]
                + [
                    torch.cos(self.cfg.phi**n * torch.pi * base_i * time_value)
                    for n in range(self.cfg.time_dim // 2)
                ]
            )
            adaptive = torch.cat(
                [adaptive, embedding.unsqueeze(0).expand((adaptive.shape[0], -1))], dim=1
            )
            motion = self.decoders["mlp_motion"](adaptive) * factors[:, 1].unsqueeze(-1)
            paths.append(anchors + motion[:, :3])
        output = torch.stack(paths, dim=1)
        if output.shape != (anchors.shape[0], self.cfg.GOP_size, 3):
            raise AssertionError("persistent path decoder returned unexpected shape")
        return output

    @torch.no_grad()
    def freeze_ap_reference_state(self, step: int) -> None:
        if self.cfg.ap_variant == "official":
            return
        if step != self.cfg.ap_freeze_step or self.ap_state is not None:
            raise ValueError("AP reference state must be frozen exactly once at ap_freeze_step")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        freeze_start = time.perf_counter()
        ids = canonical_voxel_ids(self.splats["anchors"], self.cfg.voxel_size)
        reference_paths = self.decode_persistent_anchor_paths(
            self.splats, factors_are_logits=True
        ).detach()
        path_score = torch.linalg.vector_norm(
            reference_paths[:, 1:] - reference_paths[:, :-1], dim=-1
        ).sum(dim=1)
        motion_score = torch.linalg.vector_norm(
            reference_paths - self.splats["anchors"].detach()[:, None, :], dim=-1
        ).mean(dim=1)
        eligible = torch.isfinite(reference_paths).all(dim=(1, 2)) & torch.isfinite(path_score)
        opacity_accum = self.strategy_state.get("opacity_accum")
        anchor_demon = self.strategy_state.get("anchor_demon")
        if opacity_accum is None or anchor_demon is None:
            raise ValueError(
                "AP freeze requires the backbone opacity statistics accumulated "
                "during densification"
            )
        if opacity_accum.shape[0] != ids.shape[0]:
            raise ValueError(
                "backbone opacity statistics disagree with the frozen anchor count"
            )
        importance_score = frozen_backbone_importance(opacity_accum, anchor_demon)
        estimated_bytes = self.compression_sim_method.estimate_time_stream_bytes(
            self.splats["time_features"].detach()
        )
        time_entropy_model = self.compression_sim_method.entropy_models["time_features"]
        time_entropy_model_sha256 = tensor_mapping_sha256(
            time_entropy_model.state_dict()
        )
        for parameter in time_entropy_model.parameters():
            parameter.requires_grad_(False)
        time_feature_scaling = float(
            self.compression_sim_method.scaling["time_features"]
        )
        ranking = variant_spec(self.cfg.ap_variant).ranking
        if ranking == "path":
            scores = path_score.clone()
        elif ranking == "motion":
            scores = motion_score.clone()
        elif ranking == "random":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(self.cfg.ap_random_seed))
            scores = torch.rand(ids.shape[0], generator=generator, dtype=torch.float64).to(
                ids.device
            )
        else:
            raise ValueError(f"unexpected AP ranking during freeze: {ranking}")
        scores = scores.to(torch.float64)
        scores[~eligible] = -torch.inf
        if not eligible.any():
            raise ValueError("AP reference score freeze produced no eligible anchors")

        factor_kwargs = {
            "anchor_features": self.splats["anchor_features"],
            "factors": self.splats["factors"],
            "frame_idx": 0,
            "c_channel": self.cfg.entropy_channel,
        }
        quantized_factors, _ = self.compression_sim_method.simulate_compression_factors(
            self.splats["factors"], step, None, None, **factor_kwargs
        )
        quantized_factors = quantized_factors.detach()
        official_retain = quantized_factors[:, 3] > 0
        official_factor0 = quantized_factors[:, 0] > 0
        ap_retain, whole_class, whole_audit = build_count_preserving_anchor_allocation(
            official_retain,
            scores,
            eligible,
            ids,
            self.cfg.ap_protected_fraction,
            enable_swap=variant_spec(self.cfg.ap_variant).swap,
            importance=importance_score,
        )
        official_active = official_factor0 & official_retain
        ap_starting_active = official_factor0 & ap_retain
        ap_active, ap_class, temporal_audit = build_equal_estimated_byte_allocation(
            official_active,
            scores,
            eligible,
            ids,
            estimated_bytes,
            self.cfg.ap_protected_fraction,
            enable_swap=variant_spec(self.cfg.ap_variant).swap,
            starting_active=ap_starting_active,
            retain_mask=ap_retain,
            importance=importance_score,
        )
        if not torch.equal(whole_class, ap_class):
            raise AssertionError("frozen whole and temporal AP classes disagree")
        if torch.any(ap_active & ~ap_retain):
            raise AssertionError("frozen temporal allocation escaped whole-retain mask")
        path_input_mask, retained_knn = build_path_input_precision_mask(
            self.splats["anchors"],
            ap_class,
            ap_retain,
            int(self.cfg.n_knn),
            canonical_ids=ids,
        )
        (
            self.ap_codec_knn_indices,
            retained_rows,
            codec_retained_knn,
        ) = build_codec_knn_indices(
            self.splats["anchors"],
            ap_retain,
            int(self.cfg.n_knn),
            canonical_ids=ids,
        )
        if not torch.equal(retained_knn, codec_retained_knn):
            raise AssertionError("path closure and codec KNN graph disagree")
        path_knn_graph_sha256 = retained_knn_graph_sha256(
            ids[retained_rows], retained_knn
        )
        if torch.any(ap_class & ~ap_active):
            raise AssertionError("protected AP class contains a temporally inactive row")
        positive0 = quantized_factors[quantized_factors[:, 0] > 0, 0]
        positive3 = quantized_factors[quantized_factors[:, 3] > 0, 3]
        if positive0.numel() == 0 or positive3.numel() == 0:
            raise ValueError("official frozen factor masks have no positive activation value")
        factor0_value = float(positive0.min().item())
        factor3_value = float(positive3.min().item())

        self.ap_state = {
            "schema": "h007.ap_training_state.v3",
            "scene": os.path.basename(os.path.normpath(self.cfg.data_dir)),
            "variant": self.cfg.ap_variant,
            "voxel_size": float(self.cfg.voxel_size),
            "canonical_ids": ids.detach(),
            "eligible": eligible.detach(),
            "scores": scores.detach(),
            "path_score": path_score.detach(),
            "motion_score": motion_score.detach(),
            "importance_score": importance_score.detach(),
            "estimated_time_bytes": estimated_bytes.detach(),
            "reference_paths": reference_paths.detach(),
            "official_retain_mask": official_retain.detach(),
            "official_factor0_mask": official_factor0.detach(),
            "official_active_mask": official_active.detach(),
            "ap_retain_mask": ap_retain.detach(),
            "ap_active_mask": ap_active.detach(),
            "ap_class_mask": ap_class.detach(),
            "ap_path_input_mask": path_input_mask.detach(),
            "path_contract_schema": PATH_CONTRACT_SCHEMA,
            "path_knn_graph_sha256": path_knn_graph_sha256,
            "factor0_activation_value": factor0_value,
            "factor3_activation_value": factor3_value,
            "whole_allocation_audit": whole_audit,
            "temporal_allocation_audit": temporal_audit,
            "protected_fraction": float(self.cfg.ap_protected_fraction),
            "q_ap_multiplier": float(self.cfg.ap_q_ap_multiplier),
            "q_bg_multiplier": float(self.cfg.ap_q_bg_multiplier),
            "random_seed": int(self.cfg.ap_random_seed),
            "freeze_step": int(step),
            "estimator_version": "h007.conditional_gaussian_per_row_bits.v1",
            "time_entropy_model_sha256": time_entropy_model_sha256,
            "time_feature_scaling": time_feature_scaling,
            "time_entropy_model_frozen_after_freeze": True,
            "factor_membership_columns_frozen": [0, 3],
            "runtime_provenance": dict(self.ap_runtime_provenance),
        }
        factor_param = self.splats["factors"]
        if self._ap_factor_gradient_hook is not None:
            raise ValueError("AP factor-membership gradient hook was already installed")
        gradient_mask = torch.ones_like(factor_param)
        gradient_mask[:, 0] = 0
        gradient_mask[:, 3] = 0
        self._ap_factor_gradient_hook = factor_param.register_hook(
            lambda gradient: gradient * gradient_mask
        )
        factor_optimizer_state = self.optimizers["factors"].state.get(factor_param, {})
        for value in factor_optimizer_state.values():
            if isinstance(value, Tensor) and value.shape == factor_param.shape:
                value[:, 0] = 0
                value[:, 3] = 0
        freeze_dir = os.path.join(self.cfg.result_dir, "ap_freeze")
        os.makedirs(freeze_dir, exist_ok=True)
        score_path = os.path.join(freeze_dir, f"reference_scores_step{step}.npz")
        np.savez(
            score_path,
            schema=np.asarray(AP_SCORE_SCHEMA),
            scene=np.asarray(os.path.basename(os.path.normpath(self.cfg.data_dir))),
            voxel_size=np.asarray(self.cfg.voxel_size, dtype=np.float64),
            frame_count=np.asarray(self.cfg.GOP_size, dtype=np.int64),
            variant=np.asarray(self.cfg.ap_variant),
            protected_fraction=np.asarray(self.cfg.ap_protected_fraction, dtype=np.float64),
            q_ap_multiplier=np.asarray(self.cfg.ap_q_ap_multiplier, dtype=np.float64),
            q_bg_multiplier=np.asarray(self.cfg.ap_q_bg_multiplier, dtype=np.float64),
            random_seed=np.asarray(self.cfg.ap_random_seed, dtype=np.int64),
            canonical_ids=ids.cpu().numpy().astype(np.int64, copy=False),
            eligible=eligible.cpu().numpy().astype(np.bool_, copy=False),
            path_score=path_score.cpu().numpy().astype(np.float64, copy=False),
            motion_score=motion_score.cpu().numpy().astype(np.float64, copy=False),
            allocation_score=scores.cpu().numpy().astype(np.float64, copy=False),
            importance_score=importance_score.cpu().numpy().astype(np.float64, copy=False),
            estimated_time_bytes=estimated_bytes.cpu().numpy().astype(np.int64, copy=False),
            official_retain_mask=official_retain.cpu().numpy().astype(np.bool_, copy=False),
            official_factor0_mask=official_factor0.cpu().numpy().astype(np.bool_, copy=False),
            official_active_mask=official_active.cpu().numpy().astype(np.bool_, copy=False),
            ap_retain_mask=ap_retain.cpu().numpy().astype(np.bool_, copy=False),
            ap_active_mask=ap_active.cpu().numpy().astype(np.bool_, copy=False),
            ap_class_mask=ap_class.cpu().numpy().astype(np.bool_, copy=False),
            factor0_activation_value=np.asarray(factor0_value, dtype=np.float64),
            factor3_activation_value=np.asarray(factor3_value, dtype=np.float64),
            estimator_version=np.asarray("h007.conditional_gaussian_per_row_bits.v1"),
            time_entropy_model_sha256=np.asarray(time_entropy_model_sha256),
            time_feature_scaling=np.asarray(time_feature_scaling, dtype=np.float64),
            time_entropy_model_frozen_after_freeze=np.asarray(True),
            runtime_manifest_sha256=np.asarray(
                self.ap_runtime_provenance["manifest_sha256"]
            ),
            normalized_code_tree_sha256=np.asarray(
                self.ap_runtime_provenance["normalized_code_tree"]["sha256"]
            ),
            patch_chain_sha256=np.asarray(
                self.ap_runtime_provenance["patch_sha256"], dtype=np.str_
            ),
            path_definition=np.asarray("sum_consecutive_euclidean_displacement"),
            motion_definition=np.asarray("mean_distance_from_canonical_anchor"),
            importance_definition=np.asarray(
                "backbone_blended_opacity_per_visit_prune_statistic"
            ),
            estimated_byte_definition=np.asarray(
                "ceil_deterministic_conditional_gaussian_bits_over_8"
            ),
        )
        path_file = os.path.join(freeze_dir, f"reference_paths_step{step}.npy")
        np.save(path_file, reference_paths.cpu().numpy().astype(np.float32, copy=False))
        score_payload = Path(score_path).read_bytes()
        path_payload = Path(path_file).read_bytes()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ap_score_seconds = time.perf_counter() - freeze_start
        freeze_audit = {
            "schema": "h007.ap_score_freeze.v1",
            "scene": os.path.basename(os.path.normpath(self.cfg.data_dir)),
            "variant": self.cfg.ap_variant,
            "step": step,
            "score_path": os.path.basename(score_path),
            "score_sha256": hashlib.sha256(score_payload).hexdigest(),
            "path_file": os.path.basename(path_file),
            "path_sha256": hashlib.sha256(path_payload).hexdigest(),
            "anchor_count": int(ids.shape[0]),
            "eligible_count": int(eligible.sum().item()),
            "ap_score_seconds": float(ap_score_seconds),
            "runtime_provenance": dict(self.ap_runtime_provenance),
            "outcome_fields_read": [],
        }
        with open(os.path.join(freeze_dir, "freeze_audit.json"), "wb") as f:
            f.write(canonical_json_bytes(freeze_audit))
        self.ap_state["score_path"] = score_path
        self.ap_state["score_sha256"] = freeze_audit["score_sha256"]
        self.ap_state["ap_score_seconds"] = float(ap_score_seconds)
        # Every training entry creates exactly one frozen score artifact and
        # immediately wires that exact path into the eventual real codec.
        self.cfg.ap_score_path = score_path
        if isinstance(self.compression_method, GIFStreamEnd2endCompression):
            self.compression_method.ap_config["score_path"] = score_path

    def compute_ap_path_loss(self) -> Tensor:
        if self.ap_state is None:
            raise ValueError("AP path loss requested before score freeze")
        if self.ap_codec_knn_indices is None:
            raise ValueError("AP codec KNN graph is unavailable")
        reference_knn = deterministic_knn_indices(
            self.splats["anchors"], int(self.cfg.n_knn)
        )
        decoded_paths = self.decode_persistent_anchor_paths(
            self.comp_sim_splats,
            factors_are_logits=False,
            knn_indices=self.ap_codec_knn_indices,
        )
        # Match the real codec: retained rows consume only retained KNN
        # context, while the target remains the current uncompressed full-graph
        # path on those same retained identities.
        with torch.no_grad():
            reference_paths = self.decode_persistent_anchor_paths(
                self.splats,
                factors_are_logits=True,
                knn_indices=reference_knn,
            ).detach()
        weights = self.ap_state["path_score"].to(decoded_paths.device).clamp_min(0)
        weights = weights * self.ap_state["eligible"].to(decoded_paths.device)
        weights = weights * self.ap_state["ap_retain_mask"].to(decoded_paths.device)
        if float(weights.sum().detach().item()) <= 0:
            raise ValueError("AP path loss has zero reference action mass")
        weights = weights / weights.sum()
        per_anchor = torch.linalg.vector_norm(decoded_paths - reference_paths, dim=-1).mean(dim=1)
        loss = torch.sum(weights * per_anchor)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite AP path loss")
        return loss

    def restore_ap_training_state(self, checkpoint: Dict) -> None:
        if self.cfg.ap_variant == "official":
            return
        if "ap_state" not in checkpoint or "ap_training_receipt" not in checkpoint:
            raise ValueError("AP checkpoint is missing frozen state or training receipt")
        state = checkpoint["ap_state"]
        receipt = checkpoint["ap_training_receipt"]
        if state.get("schema") != "h007.ap_training_state.v3":
            raise ValueError("unsupported AP training-state schema")
        if receipt.get("schema") != "h007.ap_training_receipt.v2":
            raise ValueError("unsupported AP training-receipt schema")
        if state["variant"] != self.cfg.ap_variant or receipt["variant"] != self.cfg.ap_variant:
            raise ValueError("checkpoint AP variant differs from requested variant")
        expected_scene = os.path.basename(os.path.normpath(self.cfg.data_dir))
        if state.get("scene") != expected_scene or receipt.get("scene") != expected_scene:
            raise ValueError("checkpoint AP scene differs from requested scene")
        if state.get("runtime_provenance") != self.ap_runtime_provenance or receipt.get(
            "runtime_provenance"
        ) != self.ap_runtime_provenance:
            raise ValueError("checkpoint AP runtime provenance differs from active preregistration")
        restored = {}
        for key, value in state.items():
            restored[key] = value.to(self.device) if isinstance(value, Tensor) else value
        ids = canonical_voxel_ids(self.splats["anchors"], self.cfg.voxel_size)
        if not torch.equal(ids, restored["canonical_ids"]):
            raise ValueError("restored AP state no longer matches checkpoint anchors")
        if restored.get("path_contract_schema") != PATH_CONTRACT_SCHEMA:
            raise ValueError("restored AP state lacks the Patch8 path contract")
        expected_path_mask, retained_knn = build_path_input_precision_mask(
            self.splats["anchors"],
            restored["ap_class_mask"],
            restored["ap_retain_mask"],
            int(self.cfg.n_knn),
            canonical_ids=ids,
        )
        if not torch.equal(expected_path_mask, restored["ap_path_input_mask"]):
            raise ValueError("restored AP path-input closure is not reproducible")
        (
            self.ap_codec_knn_indices,
            retained_rows,
            codec_retained_knn,
        ) = build_codec_knn_indices(
            self.splats["anchors"],
            restored["ap_retain_mask"],
            int(self.cfg.n_knn),
            canonical_ids=ids,
        )
        if not torch.equal(retained_knn, codec_retained_knn):
            raise ValueError("restored path closure and codec KNN graph disagree")
        graph_sha256 = retained_knn_graph_sha256(ids[retained_rows], retained_knn)
        if restored.get("path_knn_graph_sha256") != graph_sha256:
            raise ValueError("restored AP retained-KNN graph hash mismatch")
        self.ap_state = restored
        self.ap_training_receipt = receipt
        self.ap_loss_applications = int(receipt["path_loss_applications"])
        self.ap_loss_steps = [int(value) for value in receipt.get("path_loss_steps", [])]

        score_path = self.cfg.ap_score_path or restored.get("score_path")
        if not score_path or not os.path.isfile(score_path):
            raise ValueError("frozen AP score artifact is unavailable")
        payload = open(score_path, "rb").read()
        if hashlib.sha256(payload).hexdigest() != restored["score_sha256"]:
            raise ValueError("frozen AP score artifact hash mismatch")
        self.cfg.ap_score_path = score_path
        if isinstance(self.compression_method, GIFStreamEnd2endCompression):
            self.compression_method.ap_config["score_path"] = score_path

    def _validate_ap_compression_receipt(self, step: int) -> Optional[Dict]:
        if self.cfg.ap_variant == "official":
            return None
        if self.ap_state is None or self.ap_training_receipt is None:
            raise ValueError("AP compression requires frozen state and a training receipt")
        receipt = dict(self.ap_training_receipt)
        state = self.ap_state
        spec = variant_spec(self.cfg.ap_variant)
        if receipt.get("schema") != "h007.ap_training_receipt.v2":
            raise ValueError("unsupported AP training receipt")
        if receipt.get("variant") != self.cfg.ap_variant or state.get(
            "variant"
        ) != self.cfg.ap_variant:
            raise ValueError("AP training receipt variant mismatch")
        expected_scene = os.path.basename(os.path.normpath(self.cfg.data_dir))
        if receipt.get("scene") != expected_scene or state.get("scene") != expected_scene:
            raise ValueError("AP state/receipt scene mismatch")
        if bool(receipt.get("path_loss_required")) != bool(spec.action_loss):
            raise ValueError("AP receipt action-loss flag disagrees with the variant")
        for name, expected in (
            ("freeze_step", int(self.cfg.ap_freeze_step)),
            ("random_seed", int(self.cfg.ap_random_seed)),
        ):
            if int(receipt.get(name, -1)) != expected:
                raise ValueError(f"AP receipt {name} mismatch")
        for name, expected in (
            ("protected_fraction", float(self.cfg.ap_protected_fraction)),
            ("q_ap_multiplier", float(self.cfg.ap_q_ap_multiplier)),
            ("q_bg_multiplier", float(self.cfg.ap_q_bg_multiplier)),
        ):
            if float(receipt.get(name, float("nan"))) != expected:
                raise ValueError(f"AP receipt {name} mismatch")
        if receipt.get("score_sha256") != state.get("score_sha256"):
            raise ValueError("AP state/receipt score hashes disagree")
        if receipt.get("runtime_provenance") != self.ap_runtime_provenance or state.get(
            "runtime_provenance"
        ) != self.ap_runtime_provenance:
            raise ValueError("AP runtime provenance changed before compression")
        ids = canonical_voxel_ids(self.splats["anchors"], self.cfg.voxel_size)
        (
            self.ap_codec_knn_indices,
            retained_rows,
            retained_knn,
        ) = build_codec_knn_indices(
            self.splats["anchors"],
            state["ap_retain_mask"],
            int(self.cfg.n_knn),
            canonical_ids=ids,
        )
        graph_sha256 = retained_knn_graph_sha256(
            ids[retained_rows], retained_knn
        )
        if state.get("path_knn_graph_sha256") != graph_sha256:
            raise ValueError("AP retained-KNN graph changed before compression")
        expected_path_fields = {
            "path_contract_schema": PATH_CONTRACT_SCHEMA,
            "path_dependency_rule": "protected-plus-one-hop-retained-knn",
            "path_knn_count": int(self.cfg.n_knn),
            "path_knn_graph_sha256": graph_sha256,
            "factor_protected_multiplier": FACTOR_PROTECTED_MULTIPLIER,
            "factor_background_multiplier": FACTOR_BACKGROUND_MULTIPLIER,
            "anchor_feature_protected_multiplier": (
                ANCHOR_FEATURE_PROTECTED_MULTIPLIER
            ),
            "anchor_feature_background_multiplier": (
                ANCHOR_FEATURE_BACKGROUND_MULTIPLIER
            ),
        }
        for name, expected in expected_path_fields.items():
            if receipt.get(name) != expected or state.get(
                "path_contract_schema"
            ) != PATH_CONTRACT_SCHEMA:
                raise ValueError(f"AP path-contract receipt mismatch: {name}")
        for name in (
            "estimator_version",
            "time_entropy_model_sha256",
            "time_feature_scaling",
            "time_entropy_model_frozen_after_freeze",
        ):
            if receipt.get(name) != state.get(name):
                raise ValueError(f"AP state/receipt estimator provenance mismatch: {name}")
        if receipt.get("factor_membership_columns_frozen") != [0, 3] or state.get(
            "factor_membership_columns_frozen"
        ) != [0, 3]:
            raise ValueError("AP factor membership was not frozen after allocation")
        entropy_models = getattr(
            self, "entropy_models", self.compression_sim_method.entropy_models
        )
        current_entropy_hash = tensor_mapping_sha256(
            entropy_models["time_features"].state_dict()
        )
        if current_entropy_hash != state["time_entropy_model_sha256"]:
            raise ValueError("temporal entropy model changed after estimated-byte freeze")
        score_path = Path(str(self.cfg.ap_score_path or state.get("score_path", "")))
        if not score_path.is_file():
            raise ValueError("AP score artifact is missing during compression")
        if hashlib.sha256(score_path.read_bytes()).hexdigest() != state["score_sha256"]:
            raise ValueError("AP score artifact changed after training")
        if not math.isfinite(float(receipt.get("ap_score_seconds", float("nan")))) or float(
            receipt["ap_score_seconds"]
        ) <= 0:
            raise ValueError("AP score timing is missing or invalid")

        cadence = int(receipt.get("path_loss_every", 0))
        applications = int(receipt.get("path_loss_applications", -1))
        application_steps = [int(value) for value in receipt.get("path_loss_steps", [])]
        if len(application_steps) != applications:
            raise ValueError("AP receipt path-loss count/list disagree")
        if spec.action_loss:
            if cadence != int(self.cfg.ap_path_loss_every) or float(
                receipt.get("path_loss_lambda", float("nan"))
            ) != float(self.cfg.ap_path_loss_lambda):
                raise ValueError("full AP variant path-loss hyperparameters changed")
            if receipt.get("path_loss_reference") != (
                "current_raw_full_graph_vs_simulated_retained_graph_on_retained_rows"
            ):
                raise ValueError("full AP variant path-loss reference rule is unsupported")
            first = (
                (int(self.cfg.ap_freeze_step) + cadence - 1) // cadence
            ) * cadence
            expected_steps = (
                [] if step < first else list(range(first, int(step) + 1, cadence))
            )
            if sorted(set(application_steps)) != expected_steps or applications <= 0:
                raise ValueError(
                    "full AP variant lacks action-loss coverage at every cadence step"
                )
            if any(application_steps.count(value) > int(self.cfg.batch_size) for value in expected_steps):
                raise ValueError("full AP variant applied path loss too often within a step")
        elif applications != 0 or application_steps:
            raise ValueError("non-full AP ablation unexpectedly applied path loss")

        n = int(self.splats["anchors"].shape[0])
        for key in (
            "canonical_ids",
            "official_retain_mask",
            "official_factor0_mask",
            "official_active_mask",
            "ap_retain_mask",
            "ap_active_mask",
            "ap_class_mask",
            "ap_path_input_mask",
            "estimated_time_bytes",
        ):
            if int(state[key].shape[0]) != n:
                raise ValueError(f"AP frozen state {key} no longer matches anchor count")
        if int(state["official_retain_mask"].sum()) != int(
            state["ap_retain_mask"].sum()
        ):
            raise ValueError("AP frozen whole-anchor allocation changed count")
        official_mass = int(
            state["estimated_time_bytes"][state["official_active_mask"]].sum().item()
        )
        ap_mass = int(
            state["estimated_time_bytes"][state["ap_active_mask"]].sum().item()
        )
        if official_mass != ap_mass:
            raise ValueError("AP frozen temporal allocation changed estimated-byte mass")
        return receipt

    def _get_ap_edit_anchor_mask(self) -> Optional[Tensor]:
        """Align an evaluation-only edit artifact by exact canonical voxel ID."""
        edit_path = self.cfg.ap_edit_ids_path
        if edit_path is None:
            return None
        if self.istraining:
            raise ValueError("the AP edit witness is evaluation-only")
        anchors = self.splats["anchors"]
        signature = (int(anchors.data_ptr()), int(anchors.shape[0]), str(edit_path))
        if self._ap_edit_cache_signature == signature:
            return self._ap_edit_anchor_mask

        path = Path(edit_path)
        if not path.is_file():
            raise ValueError(f"AP edit-ID artifact does not exist: {path}")
        with np.load(path, allow_pickle=False) as artifact:
            required = {
                "schema",
                "scene",
                "voxel_size",
                "canonical_ids",
                "source_score_sha256",
                "selection",
                "reference_manifest_sha256",
                "selected_canonical_ids_sha256",
            }
            missing = sorted(required - set(artifact.files))
            if missing:
                raise ValueError(f"AP edit-ID artifact missing members: {missing}")
            if str(np.asarray(artifact["schema"]).item()) != "h007.ap_edit_ids.v1":
                raise ValueError("unsupported AP edit-ID schema")
            if str(np.asarray(artifact["scene"]).item()) != "flame_salmon_1":
                raise ValueError("U3 edit witness is development-locked to flame_salmon_1")
            if float(np.asarray(artifact["voxel_size"]).item()) != float(
                self.cfg.voxel_size
            ):
                raise ValueError("AP edit-ID voxel size differs from decoder")
            edit_ids = np.asarray(artifact["canonical_ids"], dtype=np.int64)
            source_score_sha256 = str(np.asarray(artifact["source_score_sha256"]).item())
            selection = str(np.asarray(artifact["selection"]).item())
            reference_manifest_sha256 = str(
                np.asarray(artifact["reference_manifest_sha256"]).item()
            )
            selected_ids_sha256 = str(
                np.asarray(artifact["selected_canonical_ids_sha256"]).item()
            )
        if edit_ids.ndim != 2 or edit_ids.shape[1] != 3 or edit_ids.shape[0] == 0:
            raise ValueError("AP edit-ID artifact must contain a nonempty [J,3] array")
        if np.unique(edit_ids, axis=0).shape[0] != edit_ids.shape[0]:
            raise ValueError("AP edit-ID artifact contains duplicates")
        if selection != "top_path_score_intersection_official_and_ap_retained":
            raise ValueError("AP edit-ID artifact selection rule is not preregistered")
        actual_ids_sha256 = hashlib.sha256(
            np.asarray(edit_ids, dtype="<i8").tobytes(order="C")
        ).hexdigest()
        if actual_ids_sha256 != selected_ids_sha256:
            raise ValueError("AP edit-ID artifact selected-ID hash mismatch")
        score_path = Path(str(self.cfg.ap_score_path or ""))
        reference_path = Path(str(self.cfg.ap_edit_reference_manifest_path or ""))
        if not score_path.is_file() or not reference_path.is_file():
            raise ValueError("AP edit consumer lacks source score/reference manifest")
        score_payload = score_path.read_bytes()
        reference_payload = reference_path.read_bytes()
        if hashlib.sha256(score_payload).hexdigest() != source_score_sha256:
            raise ValueError("AP edit artifact source-score SHA-256 mismatch")
        if hashlib.sha256(reference_payload).hexdigest() != reference_manifest_sha256:
            raise ValueError("AP edit artifact reference-manifest SHA-256 mismatch")
        reference = json.loads(reference_payload.decode("utf-8"))
        if reference != {
            "schema": "h007.ap_edit_reference_manifest.v1",
            "scene": "flame_salmon_1",
            "source_score_sha256": source_score_sha256,
            "selection": selection,
            "selection_count": int(edit_ids.shape[0]),
            "selected_canonical_ids_sha256": selected_ids_sha256,
        }:
            raise ValueError("AP edit reference manifest content mismatch")
        with np.load(score_path, allow_pickle=False) as score_artifact:
            score_ids = np.asarray(score_artifact["canonical_ids"], dtype=np.int64)
            path_score = np.asarray(score_artifact["path_score"], dtype=np.float64)
            eligible = np.asarray(score_artifact["eligible"], dtype=np.bool_)
            official_retain = np.asarray(
                score_artifact["official_retain_mask"], dtype=np.bool_
            )
            ap_retain = np.asarray(score_artifact["ap_retain_mask"], dtype=np.bool_)
        universe = eligible & official_retain & ap_retain & np.isfinite(path_score)
        candidates = np.flatnonzero(universe)
        order = np.lexsort(
            (
                score_ids[candidates, 2],
                score_ids[candidates, 1],
                score_ids[candidates, 0],
                -path_score[candidates],
            )
        )
        expected_edit_ids = score_ids[candidates[order[: edit_ids.shape[0]]]]
        if not np.array_equal(edit_ids, expected_edit_ids):
            raise ValueError("AP edit IDs differ from source-score deterministic selection")

        current_ids = canonical_voxel_ids(anchors, self.cfg.voxel_size)
        current_map = {
            tuple(int(v) for v in row): index
            for index, row in enumerate(current_ids.detach().cpu().tolist())
        }
        missing_ids = [
            tuple(int(v) for v in row)
            for row in edit_ids.tolist()
            if tuple(int(v) for v in row) not in current_map
        ]
        if missing_ids:
            raise ValueError(
                f"{len(missing_ids)} frozen edit IDs are absent from the decoded anchors"
            )
        mask = torch.zeros(
            (anchors.shape[0],), dtype=torch.bool, device=anchors.device
        )
        indices = [current_map[tuple(int(v) for v in row)] for row in edit_ids.tolist()]
        mask[torch.tensor(indices, dtype=torch.long, device=anchors.device)] = True
        payload = path.read_bytes()
        self._ap_edit_cache_signature = signature
        self._ap_edit_anchor_mask = mask
        self.ap_edit_audit = {
            "schema": "h007.ap_edit_hook.v1",
            "scene": "flame_salmon_1",
            "artifact": str(path.resolve()),
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "edited_anchor_count": int(mask.sum().item()),
            "edit_strength": float(self.cfg.ap_edit_strength),
            "target_rgb": [1.0, 0.0, 1.0],
            "alignment": "exact_canonical_voxel_id",
            "source_score": str(score_path.resolve()),
            "source_score_sha256": source_score_sha256,
            "selection": selection,
            "reference_manifest": str(reference_path.resolve()),
            "reference_manifest_sha256": reference_manifest_sha256,
            "selected_canonical_ids_sha256": selected_ids_sha256,
        }
        return mask

    def get_profiler(self, tb_writer) -> ContextManager:
        if self.profiler_config.enabled:
            return torch.profiler.profile(
                activities=self.profiler_config.activities,
                schedule=self.profiler_config.schedule,
                # on_trace_ready=self.profiler_config.on_trace_ready, 
                on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_writer.log_dir),
                record_shapes=self.profiler_config.record_shapes,
                profile_memory=self.profiler_config.profile_memory,
                with_stack=self.profiler_config.with_stack
            )
        return nullcontext()

    def step_profiler(self):
        """step profiler"""
        if self.profiler is not None:
            self.profiler.step()

    def decoding_features(self,
        camtoworlds: Tensor,
        time: float,
        visible_anchor_mask: Tensor,
        canonical: bool = False,
        step: int = -1,
        camera_ids: Tensor = None,
    )-> Dict:
        feat_start = int(time * (self.cfg.GOP_size-1))
        # coarse to fine training for time-dependent features
        if step > 0 and self.istraining:
            gap = int((self.cfg.GOP_size // 5) * (1 - min(1, 5 * step / (self.cfg.max_steps - 1))))
            pre = max(0, feat_start - gap)
            aft = min(self.cfg.GOP_size, feat_start+gap+1)
        else:
            pre = feat_start
            aft = feat_start + 1

        # consider dynamic gaussians which may be unselected (not visible in canonical space)
        # if step == -1 or step > self.cfg.max_steps // 6:
        #     visible_anchor_mask = torch.logical_or(visible_anchor_mask, torch.sigmoid(self.splats["factors"][:,1]) > 0.2 )

        if not self.cfg.compression_sim:
            selected_features = self.splats["anchor_features"][visible_anchor_mask]  # [M, c]
            selected_anchors = self.splats["anchors"][visible_anchor_mask]  # [M, 3]
            selected_scales = torch.exp(self.splats["scales"][visible_anchor_mask])  # [M, 6]
            selected_time_features = self.splats["time_features"][visible_anchor_mask][:,pre:aft].mean(dim=1) if aft - pre >1 else self.splats["time_features"][visible_anchor_mask][:,feat_start]# [M,T,C]
            factors = fake_quantize_factors(self.splats["factors"], q_aware=False)
            selected_factors = factors[visible_anchor_mask]
            if self.cfg.knn:
                if self.indices is None or self.indices.shape[0] != self.splats["anchors"].shape[0]:
                    _, self.indices = find_k_neighbors(self.splats["anchors"], self.cfg.n_knn)
                selected_indices = self.indices[visible_anchor_mask].reshape(-1)
                knn_features = self.splats["anchor_features"][selected_indices].reshape(-1,self.cfg.n_knn,self.cfg.anchor_feature_dim).mean(dim=1)
                knn_time_features = (
                    self.splats["time_features"][:,feat_start] * 
                    (factors[:,0].unsqueeze(-1) if not canonical else 0)
                )[selected_indices].reshape(-1,self.cfg.n_knn,self.cfg.c_perframe).mean(dim=1)
        else:
            selected_features = self.comp_sim_splats["anchor_features"][visible_anchor_mask]  # [M, c]
            selected_anchors = self.comp_sim_splats["anchors"][visible_anchor_mask]  # [M, 3]
            selected_scales = torch.exp(self.comp_sim_splats["scales"][visible_anchor_mask])  # [M, 6]
            selected_factors = self.comp_sim_splats["factors"][visible_anchor_mask] # [M,4]
            selected_time_features = self.comp_sim_splats["time_features"][visible_anchor_mask][:,pre:aft].mean(dim=1) if aft - pre > 1 else self.comp_sim_splats["time_features"][visible_anchor_mask][:,feat_start] 

            if self.cfg.knn:
                if self.indices is None:
                    _, self.indices = find_k_neighbors(self.splats["anchors"], self.cfg.n_knn)
                if self.ap_state is not None:
                    if self.ap_codec_knn_indices is None:
                        raise ValueError(
                            "AP compression simulation lacks the retained codec KNN graph"
                        )
                    knn_indices = self.ap_codec_knn_indices
                else:
                    knn_indices = self.indices
                selected_indices = knn_indices[visible_anchor_mask].reshape(-1)
                knn_features = self.comp_sim_splats["anchor_features"][selected_indices].reshape(-1,self.cfg.n_knn,selected_features.shape[-1]).mean(dim=1)
                knn_time_features = (
                    self.comp_sim_splats["time_features"][:,feat_start] * (
                    self.comp_sim_splats["factors"][:,0].unsqueeze(-1) if not canonical else 0)
                )[selected_indices].reshape(-1, self.cfg.n_knn, self.cfg.c_perframe).mean(dim=1)

        cam_pos = camtoworlds[:, :3, 3]
        view_dir = selected_anchors - cam_pos  
        length = view_dir.norm(dim=1, keepdim=True)
        view_dir_normalized = view_dir / length  

        if self.cfg.view_adaptive:
            feature_view_dir = torch.cat([selected_features, view_dir_normalized], dim=1)
        else:
            feature_view_dir = selected_features
        
        if self.cfg.knn:
            knn_feature_view_dir = knn_features

        i = torch.ones((1),dtype=torch.float32)
        time_embedding = torch.cat(
            [torch.sin(self.cfg.phi**n * torch.pi * i * time) for n in range(self.cfg.time_dim // 2)] + 
            [torch.cos(self.cfg.phi**n * torch.pi * i * time) for n in range(self.cfg.time_dim // 2)]
        ).to(self.splats["anchors"].device)

        time_feature_factor = selected_factors[:,0].unsqueeze(-1)
        motion_factor = selected_factors[:,1].unsqueeze(-1)
        knn_factor = selected_factors[:,2].unsqueeze(-1)
        pruning_factor = selected_factors[:,3].unsqueeze(-1)

        selected_scales = torch.cat([selected_scales[:,:3], selected_scales[:,3:] * pruning_factor],dim=-1)
        if canonical:
            time_feature_factor = 0
            motion_factor = 0
            knn_factor = 0.5
        if self.cfg.knn:
            time_adaptive_features = torch.cat([
                feature_view_dir, 
                selected_time_features * time_feature_factor
            ],dim=-1)
            time_adaptive_features_ = knn_factor * torch.cat([
                selected_features, 
                selected_time_features * time_feature_factor
            ],dim=-1) + (1 - knn_factor) * torch.cat([knn_feature_view_dir, knn_time_features],dim=-1)
        else:
            time_adaptive_features = torch.cat([
                feature_view_dir, 
                selected_time_features * time_feature_factor
            ],dim=-1)
            time_adaptive_features_ = torch.cat([
                selected_features, 
                selected_time_features * time_feature_factor
            ],dim=-1)
        time_adaptive_features_ = torch.cat([time_adaptive_features_, time_embedding.unsqueeze(0).expand((time_adaptive_features.shape[0],-1))],dim=1)


        k = self.cfg.n_offsets  # Number of offsets per anchor

        # Apply MLPs
        neural_opacity = self.decoders["mlp_opacity"](
            time_adaptive_features
        )
        neural_opacity = neural_opacity.view(-1, 1) * pruning_factor.view(-1,1).expand((-1,k)).reshape((-1,1)) 

        # Get color
        neural_colors = self.decoders["mlp_color"](
            torch.cat([time_adaptive_features, self.app_module(camera_ids).to(self.device).view((1,-1)).expand(time_adaptive_features.shape[0],-1)],dim=-1) if self.cfg.app_opt else time_adaptive_features
        )
        neural_colors = neural_colors.view(-1, 3)  # [M*k, 3]

        # Get scale and rotation
        neural_scale_rot = self.decoders["mlp_cov"](
            time_adaptive_features
        )
        neural_scale_rot = neural_scale_rot.view(-1, 7)  # [M*k, 7]

        # Get anchor motion
        motion = self.decoders["mlp_motion"](
            time_adaptive_features_
        )  
        motion = motion * motion_factor

        return {
            "neural_opacity":neural_opacity,
            "neural_colors":neural_colors,
            "neural_scale_rot":neural_scale_rot,
            "motion":motion,
            "selected_factors":selected_factors,
            "selected_scales":selected_scales,
        }

    def get_neural_gaussians(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        packed: bool,
        rasterize_mode: str,
        time: float,
        canonical: bool = False,
        regular: bool = False,
        step: int = -1,
        camera_ids: Tensor = None,
    ) -> Dict:
        """
        Compute the neural Gaussian parameters for the current view and time.

        Args:
            camtoworlds (Tensor): Camera-to-world transformation matrices, shape [C, 4, 4].
            Ks (Tensor): Camera intrinsic matrices, shape [C, 3, 3].
            width (int): Image width.
            height (int): Image height.
            packed (bool): Whether to use packed mode for rasterization.
            rasterize_mode (str): Rasterization mode (e.g., 'classic', 'antialiased').
            time (float): Normalized time in [0, 1] for dynamic scenes.
            canonical (bool, optional): Whether to use canonical (static) mode. Defaults to False.
            regular (bool, optional): Whether to compute regularization loss. Defaults to False.
            step (int, optional): Current training step. Defaults to -1.
            camera_ids (Tensor, optional): Camera IDs for appearance embedding. Defaults to None.

        Returns:
            Dict: A dictionary containing the parameters of visible neural Gaussians, including means, colors, opacities, scales, rotations, and auxiliary losses.
        """
        # Compute which anchors (Gaussians) are visible in the current view
        visible_anchor_mask = view_to_visible_anchors(
            means=self.splats["anchors"],
            quats=self.splats["quats"],
            scales=torch.exp(self.splats["scales"][:, :3]),
            viewmats=torch.linalg.inv(camtoworlds), 
            Ks=Ks,
            width=width,
            height=height,
            packed=packed,
            rasterize_mode=rasterize_mode,
        )

        # Select anchors and offsets for visible Gaussians
        if not self.cfg.compression_sim:
            selected_anchors = self.splats["anchors"][visible_anchor_mask]  # [M, 3]
            selected_offsets = self.splats["offsets"][visible_anchor_mask]  # [M, k, 3]
        else:
            selected_anchors = self.comp_sim_splats["anchors"][visible_anchor_mask]  # [M, 3]
            selected_offsets = self.comp_sim_splats["offsets"][visible_anchor_mask]  # [M, k, 3]

        # Decode neural features (opacity, color, scale/rotation, motion, etc.)
        results = self.decoding_features(
            camtoworlds,
            time,
            visible_anchor_mask,
            canonical,
            step,
            camera_ids,
        )
        # Compute smoothness loss by comparing with a nearby time step (every x steps)
        if not canonical and step % 4 == 0 and self.istraining:
            with torch.no_grad():
                idx_dif = random.choice([-2,-1,1,2])
                results_ = self.decoding_features(
                    camtoworlds,
                    torch.tensor(time + idx_dif / (self.cfg.GOP_size - 1)).clamp(0,1).item(),
                    visible_anchor_mask,
                    canonical,
                    step,
                    camera_ids,
                )
            item_list = ["neural_opacity", "neural_colors", "neural_scale_rot", "motion"]
            smooth_loss = sum([torch.abs(results[k] - results_[k]).mean() for k in results.keys() if (k in item_list)])
        else:
            smooth_loss = 0

        # Unpack decoded features
        neural_opacity = results["neural_opacity"]
        neural_colors = results["neural_colors"]
        neural_scale_rot = results["neural_scale_rot"]
        motion = results["motion"]
        selected_scales = results["selected_scales"]
        selected_factors = results["selected_factors"]
        
        # Mask out Gaussians with non-positive opacity (they do not contribute to rendering)
        neural_selection_mask = (neural_opacity > 0.0).view(-1)  # [M*k]
        # Apply motion offset to anchor positions
        anchor_offset = motion[:,-7:-4]
        selected_anchors += anchor_offset
        # Compute anchor rotation from motion output (as quaternion)
        anchor_rot = torch.nn.functional.normalize(
            0.1 * motion[:, -4:]
            + motion.new_tensor([[1.0, 0.0, 0.0, 0.0]])
        )
        anchor_rotation = quaternion_to_rotation_matrix(anchor_rot)
        # Transform offsets by scale and rotation
        selected_offsets = torch.bmm(selected_offsets.view(-1,self.cfg.n_offsets,3) * selected_scales.unsqueeze(1)[:,:,:3] ,anchor_rotation.reshape((-1,3,3)).transpose(1, 2)).reshape((-1,3))
        # Repeat scales and anchors for each offset
        scales_repeated = (selected_scales.unsqueeze(1).repeat(1, self.cfg.n_offsets, 1).view(-1, 6))  # [M*k, 6]
        anchors_repeated = (selected_anchors.unsqueeze(1).repeat(1, self.cfg.n_offsets, 1).view(-1, 3))  # [M*k, 3]
        # Combine neural and anchor rotations
        # neural_scale_rot = torch.cat([neural_scale_rot[:,:3],quaternion_multiply(anchor_rot.unsqueeze(1).expand([-1,self.cfg.n_offsets,-1]).flatten(0,1), neural_scale_rot[:, 3:7])],dim=-1)
        
        # Apply mask to select valid Gaussians
        selected_opacity = neural_opacity[neural_selection_mask].squeeze(-1)  # [M]
        selected_colors = neural_colors[neural_selection_mask]  # [M, 3]
        selected_scale_rot = neural_scale_rot[neural_selection_mask]  # [M, 7]
        selected_offsets = selected_offsets[neural_selection_mask]  # [M, 3]
        scales_repeated = scales_repeated[neural_selection_mask]  # [M, 6]
        anchors_repeated = anchors_repeated[neural_selection_mask]  # [M, 3]

        # Optional downstream behavioral witness: recolor every selected child
        # of the exact frozen parent-anchor IDs.  It is intentionally generic
        # across official/AP variants and never uses nearest-neighbor matching.
        visible_rows = torch.nonzero(visible_anchor_mask, as_tuple=False).reshape(-1)
        parent_rows = (
            visible_rows[:, None]
            .expand(-1, self.cfg.n_offsets)
            .reshape(-1)[neural_selection_mask]
        )
        edit_anchor_mask = self._get_ap_edit_anchor_mask()
        edited_gaussian_mask = torch.zeros_like(parent_rows, dtype=torch.bool)
        if edit_anchor_mask is not None:
            edited_gaussian_mask = edit_anchor_mask[parent_rows]
            target = selected_colors.new_tensor([1.0, 0.0, 1.0])
            blended = (
                (1.0 - float(self.cfg.ap_edit_strength)) * selected_colors
                + float(self.cfg.ap_edit_strength) * target
            ).clamp(0.0, 1.0)
            selected_colors = torch.where(
                edited_gaussian_mask[:, None], blended, selected_colors
            )

        # Compute final scales and rotations
        scales = scales_repeated[:, 3:] * torch.sigmoid(selected_scale_rot[:, :3])
        rotation = torch.nn.functional.normalize(selected_scale_rot[:, 3:7])

        # Compute final means (positions) of Gaussians
        offsets = selected_offsets  # [M, 3]
        means = anchors_repeated + offsets  # [M, 3]

        info = {
            "means": means,  # Final positions of Gaussians
            "colors": selected_colors,  # RGB colors
            "opacities": selected_opacity,  # Opacity values
            "scales": scales,  # Scale parameters
            "quats": rotation,  # Rotation as quaternion
            "neural_opacity": neural_opacity,  # All predicted opacities
            "neural_selection_mask": neural_selection_mask,  # Mask for valid Gaussians
            "anchor_visible_mask": visible_anchor_mask,  # Mask for visible anchors
            "reg_loss": selected_factors[:,:-1].mean() + 0.1 * selected_factors[:,-1].mean() if regular else 0,  # Regularization loss
            "smooth_loss": smooth_loss,# Smoothness loss
            "motion": anchor_offset,  # Anchor offset
            "parent_anchor_rows": parent_rows,
            "edited_gaussian_mask": edited_gaussian_mask,
        }
        return info

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        time: float = 0.,
        canonical: bool = False,
        regular: bool = False,
        step: int = -1,
        camera_ids: Tensor = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        neural_gaussians = self.get_neural_gaussians(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            packed=self.cfg.packed,
            rasterize_mode="antialiased" if self.cfg.antialiased else "classic",
            time=time,
            canonical=canonical,
            regular=regular,
            step=step,
            camera_ids=camera_ids,
        )
        
        means = neural_gaussians["means"]  # [N, 3]
        quats = neural_gaussians["quats"]  # [N, 4]
        scales = neural_gaussians["scales"]  # [N, 3]
        opacities = neural_gaussians["opacities"]  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        
        colors = neural_gaussians["colors"]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, GIFStreamStrategy)
                else False
            ),
            sparse_grad=False,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        info["anchor_visible_mask"] = neural_gaussians["anchor_visible_mask"]
        info["neural_selection_mask"] = neural_gaussians["neural_selection_mask"]
        info["update_filter"] = info["radii"] > 0
        info["scales"] = neural_gaussians["scales"]
        info["neural_opacity"] = neural_gaussians["neural_opacity"]
        info["reg_loss"] = neural_gaussians["reg_loss"]
        info["smooth_loss"] = neural_gaussians["smooth_loss"]
        info["gop"] = self.cfg.GOP_size
        info["time"] = int(time * (self.cfg.GOP_size - 1))
        info["motion"] = neural_gaussians["motion"]
        info["parent_anchor_rows"] = neural_gaussians["parent_anchor_rows"]
        info["edited_gaussian_mask"] = neural_gaussians["edited_gaussian_mask"]
        return render_colors, render_alphas, info

    def _save_training_checkpoint_after_update(
        self, step: int, global_tic: float
    ) -> None:
        """Persist the state after all mutations belonging to ``step``."""

        cfg = self.cfg
        mem = torch.cuda.max_memory_allocated() / 1024**3
        stats = {
            "mem": mem,
            "ellipse_time": time.time() - global_tic,
            "num_GS": len(self.splats["anchors"]),
            "state_position": "after_optimizer_entropy_and_strategy_post_backward",
        }
        print("Step: ", step, stats)
        with open(
            f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
            "w",
        ) as f:
            json.dump(stats, f)

        data = {
            "step": step,
            "state_position": "after_optimizer_entropy_and_strategy_post_backward",
            "splats": self.splats.state_dict(),
            "decoders": self.decoders.state_dict(),
        }
        if cfg.app_opt:
            data["app_module"] = (
                self.app_module.module.state_dict()
                if self.world_size > 1
                else self.app_module.state_dict()
            )
        if self.ap_state is not None:
            serializable_ap_state = {
                key: value.detach().cpu() if isinstance(value, Tensor) else value
                for key, value in self.ap_state.items()
            }
            data["ap_state"] = serializable_ap_state
            self.ap_training_receipt = {
                "schema": "h007.ap_training_receipt.v2",
                "scene": os.path.basename(os.path.normpath(cfg.data_dir)),
                "variant": cfg.ap_variant,
                "freeze_step": int(cfg.ap_freeze_step),
                "score_sha256": self.ap_state["score_sha256"],
                "ap_score_seconds": float(self.ap_state["ap_score_seconds"]),
                "protected_fraction": float(cfg.ap_protected_fraction),
                "q_ap_multiplier": float(cfg.ap_q_ap_multiplier),
                "q_bg_multiplier": float(cfg.ap_q_bg_multiplier),
                "random_seed": int(cfg.ap_random_seed),
                "estimator_version": self.ap_state["estimator_version"],
                "time_entropy_model_sha256": self.ap_state[
                    "time_entropy_model_sha256"
                ],
                "time_feature_scaling": float(
                    self.ap_state["time_feature_scaling"]
                ),
                "time_entropy_model_frozen_after_freeze": True,
                "factor_membership_columns_frozen": [0, 3],
                "path_contract_schema": PATH_CONTRACT_SCHEMA,
                "path_dependency_rule": "protected-plus-one-hop-retained-knn",
                "path_knn_count": int(cfg.n_knn),
                "path_knn_graph_sha256": self.ap_state[
                    "path_knn_graph_sha256"
                ],
                "factor_protected_multiplier": FACTOR_PROTECTED_MULTIPLIER,
                "factor_background_multiplier": FACTOR_BACKGROUND_MULTIPLIER,
                "anchor_feature_protected_multiplier": (
                    ANCHOR_FEATURE_PROTECTED_MULTIPLIER
                ),
                "anchor_feature_background_multiplier": (
                    ANCHOR_FEATURE_BACKGROUND_MULTIPLIER
                ),
                "path_loss_required": variant_spec(cfg.ap_variant).action_loss,
                "path_loss_lambda": float(cfg.ap_path_loss_lambda),
                "path_loss_every": int(cfg.ap_path_loss_every),
                "path_loss_applications": int(self.ap_loss_applications),
                "path_loss_steps": list(self.ap_loss_steps),
                "simulation_mask_rule": "quantized_factor3_gt0_and_factor0_gt0",
                "estimated_byte_rule": "exact_frozen_integer_subset_sum",
                "path_loss_reference": (
                    "current_raw_full_graph_vs_simulated_retained_graph_on_retained_rows"
                ),
                "runtime_provenance": dict(self.ap_runtime_provenance),
            }
            data["ap_training_receipt"] = self.ap_training_receipt

        if cfg.compression_sim and cfg.entropy_model_opt:
            self.entropy_models = self.compression_sim_method.entropy_models
            for name, entropy_model in self.compression_sim_method.entropy_models.items():
                if entropy_model is not None:
                    data[name + "_entropy_model"] = entropy_model.state_dict()
        data["compression_sim"] = cfg.compression_sim
        data["scaling"] = self.compression_sim_method.scaling
        torch.save(data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt")

    def train(self, init_step: int=0):
        self.istraining = True
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = init_step

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["offsets"], gamma=0.03 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.net_optimizers["mlp_opacity"], gamma=0.01 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.net_optimizers["mlp_color"], gamma=0.01 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.net_optimizers["mlp_motion"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=16,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        self.decoders["mlp_opacity"].train()
        self.decoders["mlp_cov"].train()
        self.decoders["mlp_color"].train()
        self.decoders["mlp_motion"].train()

        with self.get_profiler(self.writer) as prof:
            self.profiler = prof if self.profiler_config.enabled else None

            # Training loop.
            global_tic = time.time()
            pbar = tqdm.tqdm(range(init_step, max_steps))
            for step in pbar:
                if not cfg.disable_viewer:
                    while self.viewer.state.status == "paused":
                        time.sleep(0.01)
                    self.viewer.lock.acquire()
                    tic = time.time()

                try:
                    batch_data = next(trainloader_iter)
                except StopIteration:
                    trainloader_iter = iter(trainloader)
                    batch_data = next(trainloader_iter)
                if step == int(max_steps * self.cfg.strategy.deformation_gate):
                    self.init_dynamic()
                if (
                    self.cfg.ap_variant != "official"
                    and step == self.cfg.ap_freeze_step
                    and self.ap_state is None
                ):
                    self.freeze_ap_reference_state(step)
                
                #* batch forward
                info_list = []
                for batch_ind in range(self.cfg.batch_size):
                    if batch_ind >= batch_data["camtoworld"].shape[0]:
                        info_list.append(None)
                        continue
                    else:
                        data = {}
                        for k,v in batch_data.items():
                            data[k] = v[batch_ind].unsqueeze(0)
                    camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
                    Ks = data["K"].to(device)  # [1, 3, 3]
                    pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
                    num_train_rays_per_step = (
                        pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
                    )
                    image_ids = data["image_id"].to(device)
                    masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
                    camera_ids = data["camera_id"].to(device)

                    height, width = pixels.shape[1:3]


                    # sh schedule
                    sh_degree_to_use = None

                    # compression simulation
                    if cfg.compression_sim and cfg.entropy_model_opt and cfg.entropy_model_type == "gaussian_model": # if hash-based gaussian model, need to estiblish bbox
                        if step == self.entropy_min_step:
                            self.compression_sim_method._estiblish_bbox(self.splats["means"])

                    if cfg.compression_sim:
                        active_ap_state = (
                            self.ap_state
                            if self.cfg.ap_variant != "official"
                            and step >= self.cfg.ap_freeze_step
                            else None
                        )
                        self.comp_sim_splats, self.esti_bits_dict = self.compression_sim_method.simulate_compression(
                            self.splats,
                            step,
                            int(float(data["time"]) * (self.cfg.GOP_size - 1)),
                            self.cfg.entropy_channel,
                            ap_state=active_ap_state,
                        )

                    # forward
                    renders, alphas, info = self.rasterize_splats(
                        camtoworlds=camtoworlds,
                        Ks=Ks,
                        width=width,
                        height=height,
                        sh_degree=sh_degree_to_use,
                        near_plane=cfg.near_plane,
                        far_plane=cfg.far_plane,
                        image_ids=image_ids,
                        render_mode="RGB",
                        masks=masks,
                        time=float(data["time"]),
                        canonical= (step <= int(max_steps * self.cfg.strategy.deformation_gate)),
                        regular= (step > int(max_steps * (self.cfg.strategy.deformation_gate + 0.1))),
                        step=step,
                        camera_ids=camera_ids,
                    )
                    if renders.shape[-1] == 4:
                        colors, depths = renders[..., 0:3], renders[..., 3:4]
                    else:
                        colors, depths = renders, None


                    if cfg.random_bkgd:
                        bkgd = torch.rand(1, 3, device=device)
                        colors = colors + bkgd * (1.0 - alphas)

                    self.cfg.strategy.step_pre_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                    )

                    # loss
                    l1loss = F.l1_loss(colors, pixels)
                    ssimloss = 1.0 - fused_ssim(
                        colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
                    )
                    loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda + info["scales"].prod(dim=1).mean() * cfg.scale_reg + info["reg_loss"] * cfg.factor_reg + info["smooth_loss"] * cfg.smooth_reg
                    scale_loss = info["scales"].prod(dim=1).mean()
                    reg_loss = info["reg_loss"]
                    smooth_loss = info["smooth_loss"]
                    ap_path_loss = None
                    if (
                        variant_spec(cfg.ap_variant).action_loss
                        and self.ap_state is not None
                        and step % cfg.ap_path_loss_every == 0
                    ):
                        ap_path_loss = self.compute_ap_path_loss()
                        loss = loss + (
                            cfg.ap_path_loss_lambda
                            * cfg.ap_path_loss_every
                            * ap_path_loss
                        )
                        self.ap_loss_applications += 1
                        self.ap_loss_steps.append(int(step))
                    # entropy constraint
                    if cfg.entropy_model_opt and step>self.entropy_min_step:
                        total_esti_bits = 0
                        for n, n_step in cfg.entropy_steps.items():
                            if step > n_step and self.esti_bits_dict[n] is not None:
                                # maybe give different params with different weights
                                total_esti_bits += torch.sum(self.esti_bits_dict[n]) / self.esti_bits_dict[n].numel() # bpp
                            else:
                                total_esti_bits += 0

                        loss = (
                            loss
                            + cfg.rd_lambda * total_esti_bits
                        )
                    
                    # tmp workaround
                    loss_show = loss.detach().cpu()
                    loss.backward()
                    info_list.append(info)
                
                desc = f"loss={loss_show.item():.3f}| " f"sh degree={sh_degree_to_use}| "
                pbar.set_description(desc)

                # tensorboard monitor
                if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    self.writer.add_scalar("train/loss", loss.item(), step)
                    self.writer.add_scalar("train/scale_loss", scale_loss.item(), step)
                    self.writer.add_scalar("train/reg_loss", reg_loss.item() if reg_loss>0 else reg_loss, step)
                    self.writer.add_scalar("train/smooth_loss", smooth_loss.item() if smooth_loss>0 else smooth_loss, step)
                    self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                    self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                    self.writer.add_scalar("train/num_anchor", len(self.splats["anchors"]), step)
                    self.writer.add_scalar("train/mem", mem, step)
                    if self.cfg.compression_sim:
                        self.writer.add_scalar("train/dynamic", (self.comp_sim_splats["factors"][:,0] > 0).to(torch.float32).mean(), step)
                        self.writer.add_scalar("train/dynamic_", torch.logical_or((self.comp_sim_splats["factors"][:,0] > 0),(self.comp_sim_splats["factors"][:,1] > 0)).to(torch.float32).mean(), step)
                        self.writer.add_scalar("train/pruning", (self.comp_sim_splats["factors"][:,-1] > 0).to(torch.float32).mean(), step)
                    if cfg.tb_save_image:
                        canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                        canvas = canvas.reshape(-1, *canvas.shape[2:])
                        self.writer.add_image("train/render", canvas, step)
                    if cfg.compression_sim:
                        if cfg.entropy_model_opt and step>self.entropy_min_step:
                            self.writer.add_histogram("train_hist/quats", self.splats["quats"], step)
                            self.writer.add_histogram("train_hist/scales", self.splats["scales"], step)
                            self.writer.add_histogram("train_hist/anchor_features", self.splats["anchor_features"], step)
                            self.writer.add_histogram("train_hist/offsets", self.splats["offsets"], step)
                            self.writer.add_histogram("train_hist/factors", self.splats["factors"], step)
                            if total_esti_bits > 0:
                                self.writer.add_scalar("train/bpp_loss", total_esti_bits.item(), step)
                            if ap_path_loss is not None:
                                self.writer.add_scalar(
                                    "train/ap_path_loss", ap_path_loss.detach().item(), step
                                )
                        
                    self.writer.add_histogram("train_hist/means", self.splats["anchors"], step)
                    self.writer.flush()

                # Checkpoints are emitted only after every optimizer, entropy
                # optimizer and strategy post-backward mutation for this step.
                save_after_update = (
                    step in [i - 1 for i in cfg.save_steps]
                    or step == max_steps - 1
                )

                # optimize
                for optimizer in self.optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.net_optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.app_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.bil_grid_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for scheduler in schedulers:
                    scheduler.step()
                # (optional) entropy model params. optimize
                if cfg.compression_sim:
                    if cfg.entropy_model_opt:
                        for name, optimizer in self.compression_sim_method.entropy_model_optimizers.items():
                            if optimizer is not None:
                                optimizer.step()
                                optimizer.zero_grad(set_to_none=True)
                        for name, scheduler in self.compression_sim_method.entropy_model_schedulers.items():
                            if scheduler is not None and step > cfg.entropy_steps[name]:
                                scheduler.step()

                # Run post-backward steps after backward and optimizer
                if isinstance(self.cfg.strategy, GIFStreamStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info_list,
                        packed=cfg.packed,
                        mask=(self.comp_sim_splats["factors"][:,-1] == 0) if self.cfg.compression_sim else None,
                        max_steps=self.cfg.max_steps
                    )
                else:
                    assert_never(self.cfg.strategy)

                if (
                    step > self.cfg.strategy.refine_start_iter
                    and step % self.cfg.strategy.refine_every == 0
                    and self.cfg.knn
                ):
                    _, self.indices = find_k_neighbors(self.splats["anchors"], self.cfg.n_knn)

                if save_after_update:
                    self._save_training_checkpoint_after_update(step, global_tic)

                self.step_profiler()

                # eval the full set
                if step in [i - 1 for i in cfg.eval_steps]:
                    self.eval(step)
                    self.render_traj(step)

                # run compression
                if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                    self.run_compression(step=step)

                if not cfg.disable_viewer:
                    self.viewer.lock.release()
                    num_train_steps_per_sec = 1.0 / (time.time() - tic)
                    num_train_rays_per_sec = (
                        num_train_rays_per_step * num_train_steps_per_sec
                    )
                    # Update the viewer state.
                    self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                    # Update the scene.
                    self.viewer.update(step, num_train_rays_per_step)
        self.istraining = False
        

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        training_state = self.istraining
        self.istraining = False
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            camera_ids = data["camera_id"].to(device)
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=None,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                time=float(data["time"]),
                camera_ids=camera_ids,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images 
                # canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy() # side by side
                canvas = canvas_list[1].squeeze(0).cpu().numpy() # signle image
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["anchors"]),
                }
            )
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()
        self.istraining = training_state

    @torch.no_grad()
    def render_traj(self, step: int, stage: str = "val"):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        training_state = self.istraining
        self.istraining = False
        cfg = self.cfg
        device = self.device

        num_imgs = len(self.parser.camtoworlds)

        camtoworlds_all = self.parser.camtoworlds[: num_imgs//2]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 6 #1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/{stage}_traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=None,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                time=(i%self.cfg.GOP_size)/(self.cfg.GOP_size - 1)
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            # canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = canvas_list[0].squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/{stage}_traj_{step}.mp4")
        self.istraining = training_state

    @torch.no_grad()
    def benchmark_warm_render_fps(self) -> Dict[str, Union[int, float]]:
        raise RuntimeError(
            "warm FPS is archive-only and must run in h007_clean_decode_gifstream.py"
        )

    def _producer_training_config(self) -> Dict[str, Union[str, int, float, bool]]:
        return producer_training_config(self.cfg)

    def _build_producer_receipt(
        self, step: int, ap_training_receipt_payload: Optional[bytes]
    ) -> Dict:
        if self.ap_runtime_provenance is None:
            raise ValueError("official/AP producer lacks verified registered provenance")
        if self.h007_training_receipt is None or self.h007_training_receipt_payload is None:
            raise ValueError("official/AP producer lacks its external frozen training receipt")
        training_config = self._producer_training_config()
        entropy_hashes = {
            name: tensor_mapping_sha256(model.state_dict())
            for name, model in sorted(self.entropy_models.items())
            if model is not None
        }
        if not entropy_hashes:
            raise ValueError("producer receipt has no entropy-model states")
        model_state = {
            "splats": tensor_mapping_sha256(self.splats),
            "decoders": tensor_mapping_sha256(self.decoders.state_dict()),
            "entropy_models": entropy_hashes,
            "codec_scaling": hashlib.sha256(
                canonical_json_bytes(self.compression_sim_method.scaling)
            ).hexdigest(),
            "appearance_module": (
                tensor_mapping_sha256(
                    (
                        self.app_module.module
                        if isinstance(self.app_module, DDP)
                        else self.app_module
                    ).state_dict()
                )
                if self.cfg.app_opt
                else None
            ),
        }
        if (
            int(self.h007_training_receipt["training_step"]) != int(step)
            or self.h007_training_receipt.get("state_position")
            != "after_optimizer_entropy_and_strategy_post_backward"
            or self.h007_training_receipt["training_config"] != training_config
            or self.h007_training_receipt["model_state_sha256"] != model_state
        ):
            raise ValueError("active producer state differs from the frozen training receipt")
        if self.cfg.ap_variant != "official":
            checkpoint_ap_payload = canonical_json_bytes(self.ap_training_receipt or {})
            if hashlib.sha256(checkpoint_ap_payload).hexdigest() != self.h007_training_receipt[
                "ap_training_receipt_sha256"
            ]:
                raise ValueError("active AP checkpoint receipt differs from frozen training receipt")
        return {
            "schema": PRODUCER_RECEIPT_SCHEMA,
            "official_commit": "c98486632e7dafd830740b1a1692bd08c48b96e3",
            "scene": training_config["scene"],
            "variant": self.cfg.ap_variant,
            "start_frame": int(self.cfg.start_frame),
            "GOP_size": int(self.cfg.GOP_size),
            "training_step": int(step),
            "state_position": "after_optimizer_entropy_and_strategy_post_backward",
            "training_config": training_config,
            "training_config_sha256": hashlib.sha256(
                canonical_json_bytes(training_config)
            ).hexdigest(),
            "source_checkpoints": self.h007_training_receipt[
                "source_checkpoints"
            ],
            "model_state_sha256": model_state,
            "runtime_provenance": dict(self.ap_runtime_provenance),
            "training_receipt_sha256": hashlib.sha256(
                self.h007_training_receipt_payload
            ).hexdigest(),
            "ap_training_receipt_sha256": (
                hashlib.sha256(ap_training_receipt_payload).hexdigest()
                if ap_training_receipt_payload is not None
                else None
            ),
            "outcome_fields_read": [],
        }

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        cfg = self.cfg
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"

        if os.path.exists(compress_dir):
            shutil.rmtree(compress_dir)
        os.makedirs(compress_dir)

        validated_receipt = self._validate_ap_compression_receipt(step)
        ap_training_receipt_payload = None
        if validated_receipt is not None:
            ap_training_receipt_payload = canonical_json_bytes(validated_receipt)
            with open(os.path.join(compress_dir, "ap_training_receipt.json"), "wb") as f:
                f.write(ap_training_receipt_payload)

        if self.h007_training_receipt_payload is None:
            raise ValueError("codec production lacks a frozen training receipt payload")
        with open(os.path.join(compress_dir, "training_receipt.json"), "wb") as f:
            f.write(self.h007_training_receipt_payload)

        producer_receipt = self._build_producer_receipt(
            step, ap_training_receipt_payload
        )
        producer_receipt_payload = canonical_json_bytes(producer_receipt)
        with open(os.path.join(compress_dir, "producer_receipt.json"), "wb") as f:
            f.write(producer_receipt_payload)

        self.run_param_distribution_vis(self.splats, save_dir=f"{cfg.result_dir}/visualization/raw")

        if self.ap_runtime_provenance is None:
            raise ValueError("official/AP compression lacks runtime provenance")
        patch_chain = list(self.ap_runtime_provenance["patch_sha256"])
        if len(patch_chain) != 9:
            raise ValueError("official/AP compression lacks the registered nine-stage runtime chain")
        decoder_config = {
            "schema": DECODER_CONFIG_SCHEMA,
            "codec_family": "GIFStream",
            "official_commit": "c98486632e7dafd830740b1a1692bd08c48b96e3",
            "patch_chain_sha256": patch_chain,
            "runtime_manifest_sha256": self.ap_runtime_provenance["manifest_sha256"],
            "normalized_code_tree_sha256": self.ap_runtime_provenance[
                "normalized_code_tree"
            ]["sha256"],
            "producer_receipt_sha256": hashlib.sha256(
                producer_receipt_payload
            ).hexdigest(),
            "training_receipt_sha256": hashlib.sha256(
                self.h007_training_receipt_payload
            ).hexdigest(),
            "payload_manifest_sha256": None,
            "variant": cfg.ap_variant,
            "scene": os.path.basename(os.path.normpath(cfg.data_dir)),
            "data_factor": cfg.data_factor,
            "start_frame": cfg.start_frame,
            "GOP_size": cfg.GOP_size,
            "rate": cfg.rate,
            "voxel_size": float(cfg.voxel_size),
            "anchor_feature_dim": cfg.anchor_feature_dim,
            "c_perframe": cfg.c_perframe,
            "entropy_channel": cfg.entropy_channel,
            "n_offsets": cfg.n_offsets,
            "n_knn": cfg.n_knn,
            "knn": cfg.knn,
            "time_dim": cfg.time_dim,
            "view_adaptive": cfg.view_adaptive,
            "add_opacity_dist": cfg.add_opacity_dist,
            "add_cov_dist": cfg.add_cov_dist,
            "add_color_dist": cfg.add_color_dist,
            "app_opt": cfg.app_opt,
            "app_embed_dim": cfg.app_embed_dim,
            "appearance_embedding_count": int(self.trainset.cameras_length),
            "packed": bool(cfg.packed),
            "antialiased": bool(cfg.antialiased),
            "camera_model": str(cfg.camera_model),
            "phi": float(cfg.phi),
            "test_set": list(cfg.test_set or []),
            "remove_set": list(cfg.remove_set or []),
            "compression_seed": cfg.ap_compression_seed,
            "warm_camera_pose_index": int(cfg.test_set[0]) if cfg.test_set else 0,
            "warm_frame_index": 0,
            "warmup_renders": int(cfg.ap_warmup_renders),
            "timed_renders": int(cfg.ap_timed_renders),
            "clean_decode_entrypoint": "examples/h007_clean_decode_gifstream.py",
        }
        with open(os.path.join(compress_dir, "decoder_config.json"), "wb") as f:
            f.write(canonical_json_bytes(decoder_config))
        with open(os.path.join(compress_dir, "runtime_provenance.json"), "wb") as f:
            f.write(canonical_json_bytes(self.ap_runtime_provenance))
        manifest_payload = Path(str(cfg.ap_provenance_manifest)).read_bytes()
        if hashlib.sha256(manifest_payload).hexdigest() != self.ap_runtime_provenance[
            "manifest_sha256"
        ]:
            raise ValueError("external preregistration manifest changed before container build")
        (Path(compress_dir) / "preregistered_patch_chain_manifest.json").write_bytes(
            manifest_payload
        )

        camera_dir = os.path.join(compress_dir, "camera_metadata")
        os.makedirs(camera_dir)
        camera_keys = sorted(self.parser.Ks_dict)
        np.save(os.path.join(camera_dir, "camera_keys.npy"), np.asarray(camera_keys, dtype=np.int64))
        np.save(
            os.path.join(camera_dir, "intrinsics.npy"),
            np.stack([self.parser.Ks_dict[key] for key in camera_keys]).astype(np.float64),
        )
        np.save(
            os.path.join(camera_dir, "image_sizes.npy"),
            np.asarray([self.parser.imsize_dict[key] for key in camera_keys], dtype=np.int64),
        )
        np.save(
            os.path.join(camera_dir, "camtoworlds.npy"),
            np.asarray(self.parser.camtoworlds, dtype=np.float64),
        )
        np.save(
            os.path.join(camera_dir, "camera_ids.npy"),
            np.asarray(self.parser.camera_ids, dtype=np.int64),
        )
        np.save(
            os.path.join(camera_dir, "camera_names.npy"),
            np.asarray(self.parser.camera_names, dtype=np.str_),
        )
        np.save(os.path.join(camera_dir, "transform.npy"), np.asarray(self.parser.transform))
        np.save(os.path.join(camera_dir, "bounds.npy"), np.asarray(self.parser.bounds))

        reference_dir = None
        if cfg.hdown_reference_export_dir is not None:
            scene = os.path.basename(os.path.normpath(cfg.data_dir))
            if cfg.start_frame not in (0, 60, 120, 180, 240):
                raise ValueError("H-DOWN reference export requires a canonical five-GOP start")
            reference_dir = (
                Path(cfg.hdown_reference_export_dir)
                / scene
                / f"gop_{cfg.start_frame // 60}"
            )
            if reference_dir.exists() and any(reference_dir.iterdir()):
                raise ValueError("H-DOWN reference bundle directory is not empty")
            reference_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                os.path.join(compress_dir, "decoder_config.json"),
                reference_dir / "decoder_config.json",
            )
            shutil.copytree(camera_dir, reference_dir / "camera_metadata")
            reference_nets = {
                "decoders": self.decoders.state_dict(),
                "scaling": self.compression_sim_method.scaling,
            }
            if cfg.app_opt:
                reference_nets["app_module"] = (
                    self.app_module.module.state_dict()
                    if isinstance(self.app_module, DDP)
                    else self.app_module.state_dict()
                )
            torch.save(reference_nets, reference_dir / "nets.pt")
            torch.save(
                {
                    name: value.detach().cpu()
                    for name, value in self.splats.items()
                },
                reference_dir / "reference_splats.pt",
            )
            reference_manifest = {
                "schema": "h007.hdown_reference_bundle.v1",
                "scene": scene,
                "gop_id": int(cfg.start_frame // 60),
                "start_frame": int(cfg.start_frame),
                "frame_count": 60,
                "camera": "cam00",
                "variant": "official_uncompressed_reference",
                "source_checkpoint": [str(path) for path in (cfg.ckpt or [])],
                "candidate_inputs_read": [],
                "outcome_fields_read": [],
            }
            (reference_dir / "reference_bundle_manifest.json").write_bytes(
                canonical_json_bytes(reference_manifest)
            )
            reference_census = file_byte_census(str(reference_dir))
            (reference_dir / "byte_census.json").write_bytes(
                canonical_json_bytes(reference_census)
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        encode_start = time.perf_counter()
        
        if isinstance(self.compression_method, GIFStreamEnd2endCompression):
            self.compression_method.compress(
                compress_dir,
                self.comp_sim_splats,
                self.entropy_models,
                self.cfg.entropy_channel,
                self.cfg.c_perframe,
                self.scaling,
                self.cfg.voxel_size,
                raw_time_features=self.splats["time_features"],
                raw_anchor_features=self.splats["anchor_features"],
                raw_factors=self.splats["factors"],
            )
            nets = {}
            nets["decoders"] = self.decoders.state_dict()
            for name, entropy_model in self.entropy_models.items():
                if entropy_model is not None:
                    nets[name+"_entropy_model"] = entropy_model.state_dict()
            nets["scaling"] = self.compression_sim_method.scaling
            if self.cfg.app_opt:
                nets["app_module"] = (
                    self.app_module.module.state_dict()
                    if isinstance(self.app_module, DDP)
                    else self.app_module.state_dict()
                )
            torch.save(nets, os.path.join(compress_dir, "nets.pt"))
        else:
            raise NotImplementedError(f"The compression method is not implemented yet.")
        canonicalize_gifstream_png_payloads(Path(compress_dir))
        canonicalize_gifstream_torch_payload(
            Path(compress_dir) / "nets.pt", app_opt=bool(self.cfg.app_opt)
        )
        payload_manifest = build_gifstream_payload_manifest(
            Path(compress_dir),
            scene=os.path.basename(os.path.normpath(cfg.data_dir)),
            variant=cfg.ap_variant,
            start_frame=int(cfg.start_frame),
            gop_size=int(cfg.GOP_size),
        )
        payload_manifest_payload = canonical_json_bytes(payload_manifest)
        (Path(compress_dir) / "gifstream_payload_manifest.json").write_bytes(
            payload_manifest_payload
        )
        decoder_config["payload_manifest_sha256"] = hashlib.sha256(
            payload_manifest_payload
        ).hexdigest()
        (Path(compress_dir) / "decoder_config.json").write_bytes(
            canonical_json_bytes(decoder_config)
        )
        if reference_dir is not None:
            shutil.copyfile(
                os.path.join(compress_dir, "decoder_config.json"),
                reference_dir / "decoder_config.json",
            )
            (reference_dir / "byte_census.json").unlink()
            reference_census = file_byte_census(str(reference_dir))
            (reference_dir / "byte_census.json").write_bytes(
                canonical_json_bytes(reference_census)
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        encode_seconds = time.perf_counter() - encode_start

        # evaluate compression
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        decode_start = time.perf_counter()
        if isinstance(self.compression_method, GIFStreamEnd2endCompression):
            self.load_models_from_compressed_dir(compress_dir, self.cfg.entropy_model_type)
        splats_c = self.compression_method.decompress(compress_dir, self.entropy_models, self.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - decode_start
        self.run_param_distribution_vis(splats_c, save_dir=f"{cfg.result_dir}/visualization/quant")
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        if self.cfg.knn:
            _, self.indices = find_k_neighbors(self.splats["anchors"], self.cfg.n_knn)
        # Render the actual archive reconstruction. Re-running base-Q
        # compression simulation here would erase q_ap < 1 precision and no
        # longer evaluate the decoded container.
        self.cfg.compression_sim = False
        runtime = {
            "schema": "h007.gifstream_runtime.v1",
            "encode_seconds": encode_seconds,
            "model_load_plus_entropy_decode_seconds": decode_seconds,
            "peak_decode_cuda_bytes": int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else None,
            "warm_render": {
                "status": "REQUIRED_IN_CLEAN_PROCESS",
                "camera_metadata_source": "counted_archive_only",
            },
            "warm_render_fps": None,
            "ap_score_seconds": (
                float(self.ap_state["ap_score_seconds"])
                if self.ap_state is not None
                else None
            ),
            "outcome_fields_read": [],
        }
        with open(os.path.join(compress_dir, "runtime.json"), "wb") as f:
            f.write(canonical_json_bytes(runtime))
        if self.ap_edit_audit is not None:
            edit_payload = Path(str(self.cfg.ap_edit_ids_path)).read_bytes()
            edit_score_payload = Path(str(self.cfg.ap_score_path)).read_bytes()
            edit_reference_payload = Path(
                str(self.cfg.ap_edit_reference_manifest_path)
            ).read_bytes()
            counted_edit_path = Path(compress_dir) / "ap_edit_ids.npz"
            counted_score_path = Path(compress_dir) / "ap_edit_source_score.npz"
            counted_reference_path = (
                Path(compress_dir) / "ap_edit_reference_manifest.json"
            )
            counted_edit_path.write_bytes(edit_payload)
            counted_score_path.write_bytes(edit_score_payload)
            counted_reference_path.write_bytes(edit_reference_payload)
            self.ap_edit_audit["counted_artifact"] = counted_edit_path.name
            self.ap_edit_audit["counted_artifact_sha256"] = hashlib.sha256(
                edit_payload
            ).hexdigest()
            self.ap_edit_audit["counted_source_score"] = counted_score_path.name
            self.ap_edit_audit["counted_source_score_sha256"] = hashlib.sha256(
                edit_score_payload
            ).hexdigest()
            self.ap_edit_audit[
                "counted_reference_manifest"
            ] = counted_reference_path.name
            self.ap_edit_audit[
                "counted_reference_manifest_sha256"
            ] = hashlib.sha256(edit_reference_payload).hexdigest()
            with open(os.path.join(compress_dir, "ap_edit_hook.json"), "wb") as f:
                f.write(canonical_json_bytes(self.ap_edit_audit))
        clean_decode_request = {
            "schema": "h007.clean_decode_request.v2",
            "archive_only": True,
            "entrypoint": "examples/h007_clean_decode_gifstream.py",
            "expected_output": "decoded_splats.pt",
            "expected_runtime_output": "counted_camera_render",
            "external_shared_runtime": {
                "provenance_manifest_required": True,
                "provenance_manifest_sha256": self.ap_runtime_provenance[
                    "manifest_sha256"
                ],
            },
        }
        with open(os.path.join(compress_dir, "clean_decode_request.json"), "wb") as f:
            f.write(canonical_json_bytes(clean_decode_request))

        census = file_byte_census(compress_dir)
        census["self_exclusion"] = "byte_census.json is counted by the archive but cannot hash itself"
        with open(os.path.join(compress_dir, "byte_census.json"), "wb") as f:
            f.write(canonical_json_bytes(census))
        zip_path = os.path.join(cfg.result_dir, f"compression_rank{world_rank}.zip")
        zip_audit = deterministic_zip_directory(compress_dir, zip_path)
        with open(os.path.join(cfg.result_dir, f"compression_rank{world_rank}_zip_audit.json"), "wb") as f:
            f.write(canonical_json_bytes(zip_audit))
        feedback_record = {
            "schema": "h007.gifstream_byte_feedback_record.v1",
            "scene": os.path.basename(os.path.normpath(cfg.data_dir)),
            "variant": cfg.ap_variant,
            "q_ap_multiplier": float(cfg.ap_q_ap_multiplier),
            "q_bg_multiplier": float(cfg.ap_q_bg_multiplier),
            "bytes": int(zip_audit["bytes"]),
            "sha256": zip_audit["sha256"],
            "archive": zip_audit["archive"],
            "outcome_fields_read": [],
        }
        with open(
            os.path.join(
                cfg.result_dir,
                f"compression_rank{world_rank}_byte_feedback_record.json",
            ),
            "wb",
        ) as f:
            f.write(canonical_json_bytes(feedback_record))

        self.eval(step=step, stage="compress")
        self.render_traj(step=step, stage="compress")

    @torch.no_grad()
    def run_param_distribution_vis(self, param_dict: Dict[str, Tensor], save_dir: str):
        import matplotlib.pyplot as plt

        os.makedirs(save_dir, exist_ok=True)
        for param_name, value in param_dict.items():
            
            tensor_np = value.flatten().detach().cpu().numpy()
            min_val, max_val = tensor_np.min(), tensor_np.max()
            plt.figure(figsize=(6, 4))
            n, bins, patches = plt.hist(tensor_np, bins=50, density=False, alpha=0.7, color='b')

            for count, bin_edge in zip(n, bins):
                plt.text(bin_edge, count, f'{int(count)}', fontsize=8, va='bottom', ha='center')

            plt.annotate(f'Min: {min_val:.2f}', xy=(min_val, 0), xytext=(min_val, max(n) * 0.1),
                        arrowprops=dict(facecolor='green', shrink=0.05), fontsize=10, color='green')

            plt.annotate(f'Max: {max_val:.2f}', xy=(max_val, 0), xytext=(max_val, max(n) * 0.1),
                        arrowprops=dict(facecolor='red', shrink=0.05), fontsize=10, color='red')

            plt.title(f'{param_name} Distribution')
            plt.xlabel('Value')
            plt.ylabel('Density')

            plt.savefig(os.path.join(save_dir, f'{param_name}.png'))

            plt.close()
        
        print(f"Histograms saved in '{save_dir}' directory.")
    
    def load_entropy_model_from_ckpt(self, ckpt: Dict, entropy_model_type: str):
        self.entropy_models = {}
        if entropy_model_type == "conditional_gaussian_model":
                self.compression_sim_method = GIFStreamCompressionSimulation(self.cfg.compression_sim,
                                                    self.cfg.entropy_model_type,
                                                    self.cfg.entropy_steps,
                                                    self.device,
                                                    False,
                                                    None,
                                                    None,
                                                    feature_dim=self.cfg.anchor_feature_dim,
                                                    n_offsets=self.cfg.n_offsets,
                                                    c_channel=self.cfg.entropy_channel,
                                                    p_channel=self.cfg.c_perframe)
        self.entropy_models = self.compression_sim_method.entropy_models
        for name, value in ckpt.items():
            if "_entropy_model" in name:
                attr_name = name[:(len(name) - len("_entropy_model"))]
                num_ch = ckpt["splats"][attr_name].shape[-1]
                if value is not None:
                    self.entropy_models[attr_name].load_state_dict(value)
        self.compression_sim_method.scaling = ckpt["scaling"]
        self.scaling = ckpt["scaling"]
        self.comp_sim_splats, self.esti_bits_dict = self.compression_sim_method.simulate_compression(self.splats, self.cfg.max_steps, 0, self.cfg.entropy_channel)

    def load_models_from_compressed_dir(self, compress_dir, entropy_model_type: str):
        self.entropy_models = {}
        if entropy_model_type == "conditional_gaussian_model":
            if hasattr(self, 'compression_sim_method'):
                simulation = self.compression_sim_method
            else:
                self.compression_sim_method = GIFStreamCompressionSimulation(self.cfg.compression_sim,
                                                    self.cfg.entropy_model_type,
                                                    self.cfg.entropy_steps,
                                                    self.device,
                                                    False,
                                                    None,
                                                    None,
                                                    feature_dim=self.cfg.anchor_feature_dim,
                                                    n_offsets=self.cfg.n_offsets,
                                                    c_channel=self.cfg.entropy_channel,
                                                    p_channel=self.cfg.c_perframe)
        self.entropy_models = self.compression_sim_method.entropy_models
        
        ckpt = torch.load(os.path.join(compress_dir, "nets.pt"), map_location=self.device, weights_only=False)
        for name, value in ckpt.items():
            if "_entropy_model" in name:
                attr_name = name[:(len(name) - len("_entropy_model"))]
                if value is not None:
                    self.entropy_models[attr_name].load_state_dict(value)
        self.compression_sim_method.scaling = ckpt["scaling"]
        self.scaling = ckpt["scaling"]
        self.decoders.load_state_dict(ckpt["decoders"])
        if self.cfg.app_opt:
            if "app_module" not in ckpt:
                raise ValueError("counted GIFStream container is missing appearance state")
            target = self.app_module.module if isinstance(self.app_module, DDP) else self.app_module
            target.load_state_dict(ckpt["app_module"])

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=None,
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    # Duplicate the check in Runner deliberately: CLI entry and Runner are both
    # fail-closed, so alternate callers cannot bypass preregistration.
    validate_ap_entry_config(cfg, world_size)
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        if not cfg.continue_training:
            # run eval only
            ckpts = [
                torch.load(file, map_location=runner.device, weights_only=False)
                for file in cfg.ckpt
            ]
            for k in runner.splats.keys():
                runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
            runner.decoders.load_state_dict(ckpts[0]["decoders"])
            if runner.cfg.app_opt:
                if "app_module" not in ckpts[0]:
                    raise ValueError("evaluation checkpoint is missing appearance state")
                target = (
                    runner.app_module.module
                    if isinstance(runner.app_module, DDP)
                    else runner.app_module
                )
                target.load_state_dict(ckpts[0]["app_module"])
            if runner.cfg.ap_variant != "official":
                runner.restore_ap_training_state(ckpts[0])
            step = ckpts[0]["step"]
            runner.cfg.compression_sim = ckpts[0]["compression_sim"]
            if runner.cfg.compression_sim:
                runner.load_entropy_model_from_ckpt(ckpts[0], cfg.entropy_model_type)
            if cfg.knn:
                _, runner.indices = find_k_neighbors(runner.splats["anchors"], cfg.n_knn)
            runner.eval(step=step)
            runner.render_traj(step=step)
            if cfg.compression is not None:
                if cfg.compression == "end2end":
                    assert ckpts[0]["compression_sim"]
                    runner.run_compression(step=step)
                else:
                    print(f"Do not support {cfg.compression} now !")
        else:
            ckpts = [
                torch.load(file, map_location=runner.device, weights_only=False)
                for file in cfg.ckpt
            ]
            for k in runner.splats.keys():
                runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
            runner.decoders.load_state_dict(ckpts[0]["decoders"])
            if runner.cfg.app_opt:
                runner.app_module.load_state_dict(ckpts[0]["app_module"])
            if runner.cfg.ap_variant != "official" and "ap_state" in ckpts[0]:
                runner.restore_ap_training_state(ckpts[0])
            if runner.cfg.ap_variant != "official":
                raise ValueError(
                    "AP continuation is fail-closed because the official checkpoint omits "
                    "optimizer/scheduler state; run the frozen variant uninterrupted"
                )
            runner.train(init_step=int(ckpts[0]["step"]) + 1)
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)

def quaternion_to_rotation_matrix(quaternion):
    if quaternion.dim() == 1:
        quaternion = quaternion.unsqueeze(0)
    
    w, x, y, z = quaternion.unbind(dim=-1)
    
    B = quaternion.size(0)
    
    rotation_matrix = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)
    ], dim=-1).view(B, 3, 3)
    
    return rotation_matrix

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    w_new = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x_new = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y_new = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z_new = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    q_new = torch.stack([w_new, x_new, y_new, z_new], dim=-1)
    return q_new

if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer_scaffold.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "neur3d_0": (
            "neur3d dataset",
            Config(
                strategy=GIFStreamStrategy(verbose=True,densify_grad_threshold=0.0005,deformation_gate=0.03),
                test_set=[0],
                normalize_world_space=False,
                anchor_feature_dim=24,
                c_perframe = 4,
                app_opt=True,
                app_embed_dim=6,
            ),
        ),
        "neur3d_1": (
            "neur3d dataset",
            Config(
                strategy=GIFStreamStrategy(verbose=True,densify_grad_threshold=0.0006,deformation_gate=0.03),
                test_set=[0],
                normalize_world_space=False,
                anchor_feature_dim=48,
                c_perframe = 4,
                app_opt=False,
            ),
        ),
        "neur3d_2": (
            "neur3d dataset",
            Config(
                strategy=GIFStreamStrategy(verbose=True,densify_grad_threshold=0.0006,deformation_gate=0.03),
                test_set=[0],
                remove_set=[12],
                normalize_world_space=False,
                anchor_feature_dim=48,
                c_perframe = 4,
                app_opt=False,
            ),
        ),
        "GSC": (
            "mpeg dataset",
            Config(
                strategy=GIFStreamStrategy(verbose=True,densify_grad_threshold=0.0002,deformation_gate=0.03),
                test_set=[8,10,12],
                normalize_world_space=False,
                anchor_feature_dim=24,
                c_perframe = 8,
                app_opt=False,
                batch_size=4,
            ),
        ),
        "default": (
            "GIFStream with compression.",
            Config(
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    cli(main, cfg, verbose=True)
