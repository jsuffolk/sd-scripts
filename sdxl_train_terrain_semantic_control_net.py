import argparse
import copy
import csv
import json
import os
import random
import re
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import toml
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers.utils.torch_utils import is_compiled_module
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from PIL import Image, ImageDraw

from library import sdxl_model_util, sdxl_train_util, strategy_sdxl
from library import train_util
from library.device_utils import init_ipex, clean_memory_on_device
from library.sdxl_original_control_net import (
    SdxlControlNet,
    SdxlControlledUNet,
    build_seam_local_adapter_maps_from_conditioning,
)
from library.terrain_semantic_eval import (
    build_progression_boards,
    resolve_eval_samples,
    run_eval_step,
    summarize_attempt,
    resolve_swap_pairs,
    run_semantic_binding_eval,
)
from library.terrain_semantic_manifest_dataset import (
    SemanticChannelSpec,
    TerrainSemanticManifestDataset,
    build_seam_region_maps as shared_build_seam_region_maps,
    build_seam_supervision_mask as shared_build_seam_supervision_mask,
    center_embed_spatial_tensor as shared_center_embed_spatial_tensor,
    resolve_fixed_defined_edge_index,
    summarize_seam_edge_qualification,
    terrain_mask_to_occupancy as shared_terrain_mask_to_occupancy,
)
from library.utils import setup_logging
import networks.lora as lora_network


init_ipex()
setup_logging()
import logging

logger = logging.getLogger(__name__)


STEP_STATE_RE = re.compile(r"step(\d+)-state$")

COMPACT_SEAM_LOSS_TRACE_FIELDS = (
    "step",
    "total_loss",
    "diffusion_loss",
    "rgb_total_loss",
    "continuation_weighted_rgb_loss",
    "continuation_rgb_weight_mode",
    "continuation_gradient_loss",
    "continuation_gradient_weight",
    "halo_inner_rgb_loss",
    "halo_outer_rgb_loss",
    "interior_core_rgb_loss",
    "continuation_falloff_power",
    "continuation_peak_weight",
    "continuation_weight_sum",
    "continuation_effective_weight_mean",
    "continuation_effective_weight_max",
    "continuation_weight_fraction_first_8px",
    "continuation_weight_fraction_first_16px",
    "continuation_weight_fraction_first_24px",
    "seam_valid_edge_ratio",
    "valid_edges_for_loss",
    "continuation_px",
    "halo_inner_px",
    "halo_outer_px",
    "halo_without_continuation_count_after_gate",
)

SEAM_ADAPTER_DIAG_FIELDS = (
    "step",
    "block_id",
    "edge",
    "enabled",
    "scale",
    "band_px",
    "input_energy",
    "sobel_input_energy",
    "output_energy",
    "adapter_to_activation_ratio",
    "active_px",
    "edge_valid",
    "edge_valid_count",
    "undefined_edge_input_energy",
    "undefined_edge_output_energy",
    "combined_output_energy",
    "combined_adapter_to_activation_ratio",
    "combined_adapter_to_activation_ratio_block0",
    "combined_adapter_to_activation_ratio_block1",
    "total_multi_inject_ratio",
    "residual_energy_block0",
    "residual_energy_block1",
    "num_active_edges",
    "corner_active_px",
)

SEAM_ADAPTER_EDGE_NAMES = ("north", "south", "east", "west")


@dataclass(frozen=True)
class TrainingPromptSpec:
    name: str
    prompt: str
    prompt2: str
    weight: float
    mode: str


class TrainingPromptSampler:
    def __init__(self, prompts: List[TrainingPromptSpec], seed: int) -> None:
        if not prompts:
            raise ValueError("training prompt sampler requires at least one prompt")
        self._prompts = prompts
        self._rng = random.Random(seed)
        self._total_weight = sum(prompt.weight for prompt in prompts)
        if self._total_weight <= 0.0:
            raise ValueError("training prompt weights must sum to > 0")

    def sample(self) -> TrainingPromptSpec:
        draw = self._rng.uniform(0.0, self._total_weight)
        cumulative = 0.0
        for prompt in self._prompts:
            cumulative += prompt.weight
            if draw <= cumulative:
                return prompt
        return self._prompts[-1]


def load_training_prompt_pool(config: Dict[str, object]) -> List[TrainingPromptSpec]:
    training = config.get("training", {}) or {}
    prompt_pool_path = training.get("prompt_pool_path")
    if not prompt_pool_path:
        return []

    if not os.path.isabs(prompt_pool_path):
        prompt_pool_path = os.path.normpath(os.path.join(config["__config_dir"], str(prompt_pool_path)))

    with open(prompt_pool_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    raw_prompts = payload.get("prompts", []) if isinstance(payload, dict) else []
    prompt_specs: List[TrainingPromptSpec] = []
    for index, row in enumerate(raw_prompts):
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        prompt2 = str(row.get("prompt2", prompt)).strip() or prompt
        weight = float(row.get("weight", 1.0))
        if weight <= 0.0:
            continue
        name = str(row.get("name", f"prompt_{index:02d}")).strip() or f"prompt_{index:02d}"
        mode = str(row.get("mode", row.get("prompt_mode", "matching"))).strip().lower() or "matching"
        prompt_specs.append(TrainingPromptSpec(name=name, prompt=prompt, prompt2=prompt2, weight=weight, mode=mode))

    if not prompt_specs:
        raise ValueError(f"training.prompt_pool_path did not yield any usable prompts: {prompt_pool_path}")

    return prompt_specs


def summarize_training_prompt_pool(prompts: List[TrainingPromptSpec]) -> str:
    total_weight = sum(prompt.weight for prompt in prompts)
    parts = []
    for prompt in prompts:
        normalized = (prompt.weight / total_weight) if total_weight > 0.0 else 0.0
        parts.append(f"{prompt.name}:{normalized:.3f}")
    return ", ".join(parts)


def get_or_prepare_text_conditioning(
    text_cache: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    prompt: str,
    prompt2: str,
    tokenize_strategy: strategy_sdxl.SdxlTokenizeStrategy,
    text_encoding_strategy: strategy_sdxl.SdxlTextEncodingStrategy,
    text_encoders: List[torch.nn.Module],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_key = (prompt, prompt2)
    cached = text_cache.get(cache_key)
    if cached is not None:
        return cached

    cached = prepare_text_conditioning(
        prompt,
        prompt2,
        tokenize_strategy,
        text_encoding_strategy,
        text_encoders,
        device,
        dtype,
    )
    text_cache[cache_key] = cached
    return cached


def parse_resume_step(resume_path: Optional[str]) -> int:
    if not resume_path:
        return 0
    match = STEP_STATE_RE.search(os.path.basename(os.path.normpath(resume_path)))
    if match:
        return int(match.group(1))
    return 0


def ema_state_path(state_dir: str) -> str:
    return os.path.join(state_dir, "ema_state.safetensors")


def metadata_state_path(state_dir: str) -> str:
    return os.path.join(state_dir, "trainer_state.json")


def save_extended_training_state(
    args: argparse.Namespace,
    accelerator: Accelerator,
    global_step: int,
    ema_state: Optional[Dict[str, torch.Tensor]],
    on_train_end: bool = False,
) -> None:
    if on_train_end:
      state_dir = os.path.join(args.output_dir, f"{args.output_name}-state")
    else:
      state_dir = os.path.join(args.output_dir, f"{args.output_name}-step{global_step:08d}-state")

    os.makedirs(args.output_dir, exist_ok=True)
    accelerator.save_state(state_dir)

    if ema_state is not None:
        save_file(ema_state, ema_state_path(state_dir))

    with open(metadata_state_path(state_dir), "w", encoding="utf-8") as handle:
        json.dump({"global_step": global_step}, handle, indent=2)

    if not on_train_end and getattr(args, "save_last_n_steps_state", None):
        last_n_steps = int(args.save_last_n_steps_state)
        remove_step_no = global_step - last_n_steps - 1
        remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)
        if remove_step_no > 0:
            old_state_dir = os.path.join(args.output_dir, f"{args.output_name}-step{remove_step_no:08d}-state")
            if os.path.exists(old_state_dir):
                import shutil

                shutil.rmtree(old_state_dir)


def load_extended_training_state(
    args: argparse.Namespace,
    accelerator: Accelerator,
    ema_decay_enabled: bool,
    control_net_model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[int, Optional[Dict[str, torch.Tensor]]]:
    fresh_optimizer_state = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None

    def _restore_fresh_optimizer_state() -> None:
        if optimizer is None or fresh_optimizer_state is None:
            return
        optimizer.load_state_dict(fresh_optimizer_state)

    def _load_checkpoint_state_dict(path: str) -> Dict[str, torch.Tensor]:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".safetensors":
            return load_safetensors(path)
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if isinstance(payload, dict):
            return payload
        raise ValueError(f"unsupported checkpoint payload type at {path}: {type(payload)}")

    def _resolve_model_state_path(state_dir: str) -> Optional[str]:
        candidates = [
            os.path.join(state_dir, "model.safetensors"),
            os.path.join(state_dir, "pytorch_model.bin"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return None

    def _build_compatible_state(
        target_state: Dict[str, torch.Tensor],
        source_state: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
        merged: Dict[str, torch.Tensor] = {}
        adapted_keys: List[str] = []
        skipped_keys: List[str] = []

        for key, value in source_state.items():
            if key not in target_state:
                continue

            target_value = target_state[key]
            if not hasattr(value, "shape") or not hasattr(target_value, "shape"):
                continue

            if tuple(value.shape) == tuple(target_value.shape):
                merged[key] = value
                continue

            # Seam migration: widen input channels while preserving the original learned channels.
            if (
                value.ndim == 4
                and target_value.ndim == 4
                and value.shape[0] == target_value.shape[0]
                and value.shape[2] == target_value.shape[2]
                and value.shape[3] == target_value.shape[3]
                and value.shape[1] < target_value.shape[1]
            ):
                adapted = target_value.detach().to(device="cpu", dtype=value.dtype).clone()
                adapted[:, : value.shape[1], :, :] = value
                merged[key] = adapted
                adapted_keys.append(key)
                continue

            skipped_keys.append(key)

        return merged, adapted_keys, skipped_keys

    def _fallback_resume_with_model_only() -> tuple[int, Optional[Dict[str, torch.Tensor]]]:
        if control_net_model is None:
            raise RuntimeError("resume fallback requires control_net_model")

        _restore_fresh_optimizer_state()

        model_state_path = _resolve_model_state_path(args.resume)
        if model_state_path is None:
            raise FileNotFoundError(
                f"resume fallback could not find model state in {args.resume}; expected model.safetensors or pytorch_model.bin"
            )

        logger.warning(
            "[resume] accelerator state restore failed; falling back to model/EMA compatible restore "
            f"from {model_state_path}"
        )

        raw_model = control_net_model
        target_state = {key: value.detach().to(device="cpu") for key, value in raw_model.state_dict().items()}
        source_state = _load_checkpoint_state_dict(model_state_path)
        compatible_state, adapted_keys, skipped_keys = _build_compatible_state(target_state, source_state)

        info = raw_model.load_state_dict(compatible_state, strict=False)
        if adapted_keys:
            logger.warning(
                "[resume] adapted checkpoint tensors for widened channels: "
                + ", ".join(adapted_keys[:8])
                + (" ..." if len(adapted_keys) > 8 else "")
            )
        if skipped_keys:
            logger.warning(
                "[resume] skipped incompatible checkpoint tensors: "
                + ", ".join(skipped_keys[:8])
                + (" ..." if len(skipped_keys) > 8 else "")
            )
        if info.missing_keys:
            logger.warning(f"[resume] model fallback missing_keys={len(info.missing_keys)}")
        if info.unexpected_keys:
            logger.warning(f"[resume] model fallback unexpected_keys={len(info.unexpected_keys)}")

        resumed_step = parse_resume_step(args.resume)
        metadata_path = metadata_state_path(args.resume)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            resumed_step = int(metadata.get("global_step", resumed_step))

        resumed_ema_state: Optional[Dict[str, torch.Tensor]] = None
        ema_path = ema_state_path(args.resume)
        if ema_decay_enabled and os.path.exists(ema_path):
            source_ema = _load_checkpoint_state_dict(ema_path)
            target_ema = {
                key: value.detach().to(device="cpu", dtype=torch.float32).clone()
                for key, value in raw_model.state_dict().items()
            }
            compatible_ema, adapted_ema_keys, skipped_ema_keys = _build_compatible_state(target_ema, source_ema)
            for key, value in compatible_ema.items():
                target_ema[key] = value.to(device="cpu", dtype=torch.float32)
            resumed_ema_state = target_ema
            if adapted_ema_keys:
                logger.warning(
                    "[resume] adapted EMA tensors for widened channels: "
                    + ", ".join(adapted_ema_keys[:8])
                    + (" ..." if len(adapted_ema_keys) > 8 else "")
                )
            if skipped_ema_keys:
                logger.warning(
                    "[resume] skipped incompatible EMA tensors: "
                    + ", ".join(skipped_ema_keys[:8])
                    + (" ..." if len(skipped_ema_keys) > 8 else "")
                )
            logger.info(f"[resume] loaded compatible EMA state from {ema_path}")

        logger.info(f"[resume] restored_global_step={resumed_step} (fallback=model+ema-only)")
        return resumed_step, resumed_ema_state

    if not getattr(args, "resume", None):
        return 0, None

    logger.info(f"[resume] loading training state from {args.resume}")
    try:
        accelerator.load_state(args.resume)
        if optimizer is not None and control_net_model is not None:
            verify_optimizer_parameter_membership(optimizer, control_net_model)
    except (RuntimeError, ValueError) as resume_error:
        logger.warning(f"[resume] accelerator.load_state failed: {resume_error}")
        return _fallback_resume_with_model_only()

    resumed_step = parse_resume_step(args.resume)
    metadata_path = metadata_state_path(args.resume)
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        resumed_step = int(metadata.get("global_step", resumed_step))

    resumed_ema_state: Optional[Dict[str, torch.Tensor]] = None
    ema_path = ema_state_path(args.resume)
    if ema_decay_enabled and os.path.exists(ema_path):
        resumed_ema_state = load_safetensors(ema_path)
        resumed_ema_state = {key: value.to(device="cpu", dtype=torch.float32) for key, value in resumed_ema_state.items()}
        logger.info(f"[resume] loaded EMA state from {ema_path}")

    logger.info(f"[resume] restored_global_step={resumed_step}")
    return resumed_step, resumed_ema_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic_config", type=str, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default="terrain-semantic-controlnet")
    parser.add_argument("--material_lora_weights", type=str, required=True)
    parser.add_argument("--material_lora_multiplier", type=float, default=1.0)
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None)
    parser.add_argument("--controlnet_multiplier", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bind_pair_seed", type=int, default=None,
                        help="Override bind_pair_seed from config (seeds bind negative mode/ROI/shift/retry RNG). "
                             "Set per-replicate to vary the bind sampling trajectory independently of --seed.")
    parser.add_argument("--resolution", type=int, nargs=2, default=None)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--train_only_seam_adapter",
        action="store_true",
        help="Freeze all ControlNet parameters except controlnet_seam_adapter.* tensors.",
    )
    parser.add_argument(
        "--seam_adapter_lr_multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to --learning_rate for controlnet_seam_adapter parameter groups.",
    )
    parser.add_argument("--max_train_steps", type=int, default=8000)
    parser.add_argument("--save_every_n_steps", type=int, default=500)
    parser.add_argument("--save_warmup_every_n_steps", type=int, default=0)
    parser.add_argument("--save_warmup_steps", type=int, default=0)
    parser.add_argument("--save_every_n_steps_after_warmup", type=int, default=0)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--xformers", action="store_true")
    parser.add_argument("--sdpa", action="store_true")
    parser.add_argument("--cache_latents", action="store_true")
    parser.add_argument("--latent_cache_dir", type=str, default=None)
    parser.add_argument("--latent_cache_version", type=str, default="v1")
    parser.add_argument("--latent_cache_vae_key", type=str, default=None)
    parser.add_argument("--max_token_length", type=int, default=None)
    parser.add_argument("--tokenizer_cache_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=4)
    parser.add_argument("--dataloader_pin_memory", dest="dataloader_pin_memory", action="store_true")
    parser.add_argument("--no_dataloader_pin_memory", dest="dataloader_pin_memory", action="store_false")
    parser.add_argument("--dataloader_persistent_workers", dest="dataloader_persistent_workers", action="store_true")
    parser.add_argument("--no_dataloader_persistent_workers", dest="dataloader_persistent_workers", action="store_false")
    parser.set_defaults(dataloader_pin_memory=True, dataloader_persistent_workers=True)
    parser.add_argument("--enable_cudnn_benchmark", dest="enable_cudnn_benchmark", action="store_true")
    parser.add_argument("--disable_cudnn_benchmark", dest="enable_cudnn_benchmark", action="store_false")
    parser.add_argument("--enable_tf32", dest="enable_tf32", action="store_true")
    parser.add_argument("--disable_tf32", dest="enable_tf32", action="store_false")
    parser.set_defaults(enable_cudnn_benchmark=True, enable_tf32=True)
    parser.add_argument("--vae", type=str, default=None)
    parser.add_argument("--lowram", action="store_true")
    parser.add_argument("--disable_mmap_load_safetensors", action="store_true")
    parser.add_argument("--full_fp16", action="store_true")
    parser.add_argument("--full_bf16", action="store_true")
    parser.add_argument("--no_half_vae", action="store_true")
    parser.add_argument("--save_dtype", type=str, default="fp16", choices=["fp16", "bf16", "float"])
    parser.add_argument("--save_state", action="store_true")
    parser.add_argument("--save_state_on_train_end", action="store_true")
    parser.add_argument(
        "--skip_step_checkpoint_weights",
        action="store_true",
        help="Do not write top-level step-XXXXXX.safetensors files during intermediate checkpoints; state folders still save normally.",
    )
    parser.add_argument("--save_last_n_steps_state", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--sanity_samples", type=int, default=32)
    parser.add_argument("--debug_dump_samples", type=int, default=0)
    parser.add_argument("--skip_lora_sanity_check", action="store_true")
    parser.add_argument("--lora_sanity_steps", type=int, default=6)
    parser.add_argument(
        "--lora_sanity_prompt",
        type=str,
        default="rock surface, natural stone, neutral lighting",
    )
    parser.add_argument("--lora_sanity_prompt2", type=str, default=None)
    parser.add_argument("--loss_trace_every", type=int, default=5)
    parser.add_argument("--ema_decay", type=float, default=0.0)
    parser.add_argument("--ema_eval_at_anchors", action="store_true")
    parser.add_argument("--eval_steps_csv", type=str, default="")
    parser.add_argument("--skip_step0_eval", action="store_true", help="Skip step 0 evaluation to reduce memory pressure")
    return parser.parse_args()


def _parse_steps_csv(steps_csv: str) -> List[int]:
    if not steps_csv:
        return []
    steps: List[int] = []
    for raw in str(steps_csv).split(","):
        token = raw.strip()
        if not token:
            continue
        steps.append(int(token))
    return sorted(set(steps))


def _normalize_loss_trace_mode(mode: object) -> str:
    normalized = str(mode or "full").strip().lower() or "full"
    if normalized not in {"full", "compact_seam"}:
        raise ValueError(f"unsupported loss_trace_mode='{mode}'")
    return normalized


def _compact_edge_metric_sum(row: Dict[str, object], names: Tuple[str, ...], fallback_key: str) -> float:
    values: List[float] = []
    for name in names:
        if name in row:
            values.append(float(row.get(name, 0.0) or 0.0))
    if values:
        return float(sum(values))
    return float(row.get(fallback_key, 0.0) or 0.0)


def _format_loss_trace_row(row: Dict[str, object], mode: str) -> Dict[str, object]:
    normalized_mode = _normalize_loss_trace_mode(mode)
    if normalized_mode == "full":
        return row

    continuation_px = _compact_edge_metric_sum(
        row,
        (
            "seam_edge_north_continuation_px_after_gate",
            "seam_edge_south_continuation_px_after_gate",
            "seam_edge_east_continuation_px_after_gate",
            "seam_edge_west_continuation_px_after_gate",
        ),
        "seam_continuation_distance_band_px_count",
    )
    halo_inner_px = _compact_edge_metric_sum(
        row,
        (
            "seam_edge_north_halo_inner_px_after_gate",
            "seam_edge_south_halo_inner_px_after_gate",
            "seam_edge_east_halo_inner_px_after_gate",
            "seam_edge_west_halo_inner_px_after_gate",
        ),
        "seam_halo_supervised_px",
    )
    halo_outer_px = _compact_edge_metric_sum(
        row,
        (
            "seam_edge_north_halo_outer_px_after_gate",
            "seam_edge_south_halo_outer_px_after_gate",
            "seam_edge_east_halo_outer_px_after_gate",
            "seam_edge_west_halo_outer_px_after_gate",
        ),
        "seam_halo_supervised_px",
    )

    compact_row = {
        "step": float(row.get("step", 0.0) or 0.0),
        "total_loss": float(row.get("loss", 0.0) or 0.0),
        "diffusion_loss": float(row.get("diffusion_loss", 0.0) or 0.0),
        "rgb_total_loss": float(row.get("seam_rgb_total_loss", 0.0) or 0.0),
        "continuation_weighted_rgb_loss": float(row.get("seam_rgb_continuation_weighted_loss", 0.0) or 0.0),
        "continuation_rgb_weight_mode": str(row.get("seam_continuation_rgb_weight_mode", "falloff") or "falloff"),
        "continuation_gradient_loss": float(row.get("seam_continuation_gradient_loss", 0.0) or 0.0),
        "continuation_gradient_weight": float(row.get("seam_continuation_gradient_loss_weight", 0.0) or 0.0),
        "halo_inner_rgb_loss": float(row.get("seam_rgb_halo_inner_loss", 0.0) or 0.0),
        "halo_outer_rgb_loss": float(row.get("seam_rgb_halo_outer_loss", 0.0) or 0.0),
        "interior_core_rgb_loss": float(row.get("seam_rgb_interior_core_loss", 0.0) or 0.0),
        "continuation_falloff_power": float(row.get("seam_continuation_falloff_power", 0.0) or 0.0),
        "continuation_peak_weight": float(row.get("seam_continuation_peak_weight", 0.0) or 0.0),
        "continuation_weight_sum": float(row.get("seam_continuation_weight_sum", 0.0) or 0.0),
        "continuation_effective_weight_mean": float(row.get("seam_continuation_effective_weight_mean", 0.0) or 0.0),
        "continuation_effective_weight_max": float(row.get("seam_continuation_effective_weight_max", 0.0) or 0.0),
        "continuation_weight_fraction_first_8px": float(row.get("seam_continuation_weight_fraction_first_8px", 0.0) or 0.0),
        "continuation_weight_fraction_first_16px": float(row.get("seam_continuation_weight_fraction_first_16px", 0.0) or 0.0),
        "continuation_weight_fraction_first_24px": float(row.get("seam_continuation_weight_fraction_first_24px", 0.0) or 0.0),
        "seam_valid_edge_ratio": float(row.get("seam_valid_edge_ratio", 0.0) or 0.0),
        "valid_edges_for_loss": float(row.get("seam_valid_edges_for_loss_count", 0.0) or 0.0),
        "continuation_px": float(continuation_px),
        "halo_inner_px": float(halo_inner_px),
        "halo_outer_px": float(halo_outer_px),
        "halo_without_continuation_count_after_gate": float(row.get("halo_without_continuation_count_after_gate", 0.0) or 0.0),
    }
    return {field: compact_row.get(field, 0.0) for field in COMPACT_SEAM_LOSS_TRACE_FIELDS}


def _format_seam_adapter_diag_row(row: Dict[str, object]) -> Dict[str, object]:
    formatted = {}
    for field in SEAM_ADAPTER_DIAG_FIELDS:
        default_value = "" if field in {"block_id", "edge"} else 0.0
        formatted[field] = row.get(field, default_value)
    return formatted


def _tensor_edge_values(value: object) -> List[float]:
    if value is None:
        return [0.0, 0.0, 0.0, 0.0]
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float().cpu()
        if tensor.ndim == 2:
            tensor = tensor.mean(dim=0)
        return [float(v) for v in tensor.flatten().tolist()[: len(SEAM_ADAPTER_EDGE_NAMES)]] + [0.0] * max(0, len(SEAM_ADAPTER_EDGE_NAMES) - int(tensor.numel()))
    if isinstance(value, (list, tuple)):
        return [float(v) for v in list(value)[: len(SEAM_ADAPTER_EDGE_NAMES)]] + [0.0] * max(0, len(SEAM_ADAPTER_EDGE_NAMES) - len(value))
    return [float(value)] + [0.0] * (len(SEAM_ADAPTER_EDGE_NAMES) - 1)


def _build_seam_adapter_diag_rows(step: int, diagnostics: Dict[str, object]) -> List[Dict[str, object]]:
    if not diagnostics:
        return []

    enabled = 1.0 if diagnostics.get("seam_adapter_enabled", False) else 0.0
    scale = float(diagnostics.get("seam_adapter_scale", 0.0) or 0.0)
    band_px = float(diagnostics.get("seam_adapter_band_px", 0.0) or 0.0)
    edge_valid = _tensor_edge_values(diagnostics.get("seam_adapter_edge_valid_flags"))
    rows = []
    block_diagnostics = diagnostics.get("seam_adapter_block_diagnostics") or {}
    block_order = diagnostics.get("seam_adapter_injection_block_ids") or list(block_diagnostics.keys())

    if not block_order:
        block_order = ["block0"]
        block_diagnostics = {
            "block0": {
                "scale": scale,
                "input_energy": float(diagnostics.get("seam_adapter_input_energy", 0.0) or 0.0),
                "sobel_input_energy": float(diagnostics.get("seam_adapter_sobel_input_energy", 0.0) or 0.0),
                "output_energy": float(diagnostics.get("seam_adapter_output_energy", 0.0) or 0.0),
                "adapter_to_activation_ratio": float(diagnostics.get("seam_adapter_to_activation_ratio", 0.0) or 0.0),
                "active_px": float(diagnostics.get("seam_adapter_combined_active_px", diagnostics.get("seam_adapter_active_px", 0.0)) or 0.0),
                "undefined_edge_input_energy": float(diagnostics.get("seam_adapter_undefined_edge_input_energy", 0.0) or 0.0),
                "undefined_edge_output_energy": float(diagnostics.get("seam_adapter_undefined_edge_output_energy", 0.0) or 0.0),
                "per_edge_input_energy": diagnostics.get("seam_adapter_per_edge_input_energy"),
                "per_edge_sobel_input_energy": diagnostics.get("seam_adapter_per_edge_sobel_input_energy"),
                "per_edge_output_energy": diagnostics.get("seam_adapter_per_edge_output_energy"),
                "per_edge_ratio": diagnostics.get("seam_adapter_per_edge_ratio"),
                "per_edge_active_px": diagnostics.get("seam_adapter_per_edge_active_px"),
                "per_edge_invalid_input_energy": diagnostics.get("seam_adapter_per_edge_invalid_input_energy"),
                "per_edge_invalid_output_energy": diagnostics.get("seam_adapter_per_edge_invalid_output_energy"),
            }
        }

    for block_id in block_order:
        block_diag = block_diagnostics.get(block_id, {})
        block_scale = float(block_diag.get("scale", scale) or 0.0)
        input_energy = _tensor_edge_values(block_diag.get("per_edge_input_energy", diagnostics.get("seam_adapter_per_edge_input_energy")))
        sobel_input_energy = _tensor_edge_values(
            block_diag.get("per_edge_sobel_input_energy", diagnostics.get("seam_adapter_per_edge_sobel_input_energy"))
        )
        output_energy = _tensor_edge_values(block_diag.get("per_edge_output_energy"))
        ratio = _tensor_edge_values(block_diag.get("per_edge_ratio"))
        active_px = _tensor_edge_values(block_diag.get("per_edge_active_px"))
        invalid_input_energy = _tensor_edge_values(
            block_diag.get("per_edge_invalid_input_energy", diagnostics.get("seam_adapter_per_edge_invalid_input_energy"))
        )
        invalid_output_energy = _tensor_edge_values(block_diag.get("per_edge_invalid_output_energy"))

        for edge_index, edge_name in enumerate(SEAM_ADAPTER_EDGE_NAMES):
            rows.append(
                _format_seam_adapter_diag_row(
                    {
                        "step": float(step),
                        "block_id": block_id,
                        "edge": edge_name,
                        "enabled": enabled,
                        "scale": block_scale,
                        "band_px": band_px,
                        "input_energy": input_energy[edge_index],
                        "sobel_input_energy": sobel_input_energy[edge_index],
                        "output_energy": output_energy[edge_index],
                        "adapter_to_activation_ratio": ratio[edge_index],
                        "active_px": active_px[edge_index],
                        "edge_valid": edge_valid[edge_index],
                        "edge_valid_count": edge_valid[edge_index],
                        "undefined_edge_input_energy": invalid_input_energy[edge_index],
                        "undefined_edge_output_energy": invalid_output_energy[edge_index],
                    }
                )
            )

        rows.append(
            _format_seam_adapter_diag_row(
                {
                    "step": float(step),
                    "block_id": block_id,
                    "edge": "combined",
                    "enabled": enabled,
                    "scale": block_scale,
                    "band_px": band_px,
                    "input_energy": float(block_diag.get("input_energy", diagnostics.get("seam_adapter_input_energy", 0.0)) or 0.0),
                    "sobel_input_energy": float(block_diag.get("sobel_input_energy", diagnostics.get("seam_adapter_sobel_input_energy", 0.0)) or 0.0),
                    "output_energy": float(block_diag.get("output_energy", 0.0) or 0.0),
                    "adapter_to_activation_ratio": float(block_diag.get("adapter_to_activation_ratio", 0.0) or 0.0),
                    "active_px": float(block_diag.get("active_px", 0.0) or 0.0),
                    "edge_valid": float(diagnostics.get("seam_adapter_edge_valid_count", 0.0) or 0.0),
                    "edge_valid_count": float(diagnostics.get("seam_adapter_edge_valid_count", 0.0) or 0.0),
                    "undefined_edge_input_energy": float(
                        block_diag.get("undefined_edge_input_energy", diagnostics.get("seam_adapter_undefined_edge_input_energy", 0.0)) or 0.0
                    ),
                    "undefined_edge_output_energy": float(block_diag.get("undefined_edge_output_energy", 0.0) or 0.0),
                    "combined_output_energy": float(block_diag.get("output_energy", 0.0) or 0.0),
                    "combined_adapter_to_activation_ratio": float(block_diag.get("adapter_to_activation_ratio", 0.0) or 0.0),
                    "num_active_edges": float(sum(1.0 for value in edge_valid if value > 0.5)),
                    "corner_active_px": float(diagnostics.get("seam_adapter_corner_active_px", 0.0) or 0.0),
                }
            )
        )

    rows.append(
        _format_seam_adapter_diag_row(
            {
                "step": float(step),
                "block_id": "overall",
                "edge": "combined",
                "enabled": enabled,
                "scale": scale,
                "band_px": band_px,
                "input_energy": float(diagnostics.get("seam_adapter_input_energy", 0.0) or 0.0),
                "sobel_input_energy": float(diagnostics.get("seam_adapter_sobel_input_energy", 0.0) or 0.0),
                "output_energy": float(sum(float(block_diagnostics.get(block_id, {}).get("output_energy", 0.0) or 0.0) for block_id in block_order)),
                "adapter_to_activation_ratio": float(diagnostics.get("total_multi_inject_ratio", diagnostics.get("seam_adapter_to_activation_ratio", 0.0)) or 0.0),
                "active_px": float(diagnostics.get("seam_adapter_combined_active_px", diagnostics.get("seam_adapter_active_px", 0.0)) or 0.0),
                "edge_valid": float(diagnostics.get("seam_adapter_edge_valid_count", 0.0) or 0.0),
                "edge_valid_count": float(diagnostics.get("seam_adapter_edge_valid_count", 0.0) or 0.0),
                "undefined_edge_input_energy": float(diagnostics.get("seam_adapter_undefined_edge_input_energy", 0.0) or 0.0),
                "undefined_edge_output_energy": float(diagnostics.get("seam_adapter_undefined_edge_output_energy", 0.0) or 0.0),
                "combined_output_energy": float(sum(float(block_diagnostics.get(block_id, {}).get("output_energy", 0.0) or 0.0) for block_id in block_order)),
                "combined_adapter_to_activation_ratio": float(diagnostics.get("total_multi_inject_ratio", diagnostics.get("seam_adapter_to_activation_ratio", 0.0)) or 0.0),
                "combined_adapter_to_activation_ratio_block0": float(diagnostics.get("combined_adapter_to_activation_ratio_block0", 0.0) or 0.0),
                "combined_adapter_to_activation_ratio_block1": float(diagnostics.get("combined_adapter_to_activation_ratio_block1", 0.0) or 0.0),
                "total_multi_inject_ratio": float(diagnostics.get("total_multi_inject_ratio", diagnostics.get("seam_adapter_to_activation_ratio", 0.0)) or 0.0),
                "residual_energy_block0": float(diagnostics.get("residual_energy_block0", diagnostics.get("seam_adapter_output_energy", 0.0)) or 0.0),
                "residual_energy_block1": float(diagnostics.get("residual_energy_block1", 0.0) or 0.0),
                "num_active_edges": float(sum(1.0 for value in edge_valid if value > 0.5)),
                "corner_active_px": float(diagnostics.get("seam_adapter_corner_active_px", 0.0) or 0.0),
            }
        )
    )
    return rows


def _build_seam_local_kwargs_from_payload(payload: Dict[str, torch.Tensor], zero_input: bool = False) -> Dict[str, torch.Tensor]:
    seam_local_maps = payload["adapter_input"]
    if zero_input:
        seam_local_maps = torch.zeros_like(seam_local_maps)
    kwargs = {
        "seam_local_maps": seam_local_maps,
        "seam_local_mask": payload["active_mask"],
        "seam_local_invalid_mask": payload["invalid_active_mask"],
        "seam_local_edge_valid_count": payload["edge_valid_count"],
    }
    if payload.get("edge_valid_flags") is not None:
        kwargs["seam_local_edge_valid_flags"] = payload["edge_valid_flags"]
    if payload.get("combined_active_mask") is not None:
        kwargs["seam_local_combined_active_mask"] = payload["combined_active_mask"]
    return kwargs


def should_save_checkpoint(args: argparse.Namespace, global_step: int, resumed_step: int) -> bool:
    warmup_every = int(getattr(args, "save_warmup_every_n_steps", 0) or 0)
    warmup_steps = int(getattr(args, "save_warmup_steps", 0) or 0)
    after_every = int(getattr(args, "save_every_n_steps_after_warmup", 0) or 0)

    if warmup_every > 0 and warmup_steps > 0 and after_every > 0 and global_step > resumed_step:
        steps_since_resume = global_step - resumed_step
        if steps_since_resume <= warmup_steps:
            return (steps_since_resume % warmup_every) == 0
        return ((steps_since_resume - warmup_steps) % after_every) == 0

    return (global_step % max(1, int(args.save_every_n_steps))) == 0


def unwrap_model(accelerator: Accelerator, model):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def init_ema_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for key, value in model.state_dict().items():
            state[key] = value.detach().to(device="cpu", dtype=torch.float32, copy=True)
    return state


def update_ema_state(model: torch.nn.Module, ema_state: Dict[str, torch.Tensor], decay: float) -> None:
    with torch.no_grad():
        for key, value in model.state_dict().items():
            ema_state[key].mul_(decay).add_(value.detach().to(device="cpu", dtype=torch.float32), alpha=1.0 - decay)


def swap_model_state(model: torch.nn.Module, target_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    backup: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        current_state = model.state_dict()
        for key, value in current_state.items():
            backup[key] = value.detach().to(device="cpu", copy=True)
            value.copy_(target_state[key].to(device=value.device, dtype=value.dtype))
    return backup


def restore_model_state(model: torch.nn.Module, backup_state: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        current_state = model.state_dict()
        for key, value in current_state.items():
            value.copy_(backup_state[key].to(device=value.device, dtype=value.dtype))


def resolve_save_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def resolve_weight_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.mixed_precision == "fp16":
        return torch.float16
    if args.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def load_semantic_config(path: str) -> Dict[str, object]:
    config = toml.load(path)
    config_dir = os.path.dirname(path)

    def resolve_config_path(value: str) -> str:
        if os.path.isabs(value):
            return value
        return os.path.normpath(os.path.join(config_dir, value))

    config["dataset_root"] = resolve_config_path(config["dataset_root"])
    config["manifest_path"] = resolve_config_path(config["manifest_path"])
    config["__config_dir"] = config_dir
    return config


def parse_alpha_config(config: Dict[str, object]) -> Dict[str, object]:
    alpha = config.get("alpha", {})
    head_scales = alpha.get("head_scales", [0, 3, 6, 8])
    bind_ratio = float(alpha.get("bind_paired_batch_ratio", 0.33))
    bind_ratio = max(0.0, min(1.0, bind_ratio))
    raw_mode_weights = alpha.get(
        "bind_negative_mode_weights",
        {
            "wall_to_flat_suppression": 0.40,
            "flat_to_wall_injection": 0.35,
            "local_spatial_misalignment": 0.25,
            "channel_consistent_scramble": 0.0,
            "hybrid_local_warp": 0.0,
        },
    )
    if not isinstance(raw_mode_weights, dict):
        raw_mode_weights = {}
    bind_negative_mode_weights = {
        "wall_to_flat_suppression": float(raw_mode_weights.get("wall_to_flat_suppression", 0.40)),
        "flat_to_wall_injection": float(raw_mode_weights.get("flat_to_wall_injection", 0.35)),
        "local_spatial_misalignment": float(raw_mode_weights.get("local_spatial_misalignment", 0.25)),
        "channel_consistent_scramble": float(raw_mode_weights.get("channel_consistent_scramble", 0.0)),
        "hybrid_local_warp": float(raw_mode_weights.get("hybrid_local_warp", 0.0)),
    }
    
    # Parse step-gated C activation parameters
    bind_negative_mode_c_activation_step = int(alpha.get("bind_negative_mode_c_activation_step", -1))
    raw_mode_weights_until_c = alpha.get("bind_negative_mode_weights_until_c_activation", None)
    if not isinstance(raw_mode_weights_until_c, dict):
        raw_mode_weights_until_c = None
    bind_negative_mode_weights_until_c = None
    if raw_mode_weights_until_c is not None:
        bind_negative_mode_weights_until_c = {
            "wall_to_flat_suppression": float(raw_mode_weights_until_c.get("wall_to_flat_suppression", 0.40)),
            "flat_to_wall_injection": float(raw_mode_weights_until_c.get("flat_to_wall_injection", 0.35)),
            "local_spatial_misalignment": float(raw_mode_weights_until_c.get("local_spatial_misalignment", 0.25)),
            "channel_consistent_scramble": float(raw_mode_weights_until_c.get("channel_consistent_scramble", 0.0)),
            "hybrid_local_warp": float(raw_mode_weights_until_c.get("hybrid_local_warp", 0.0)),
        }

    bind_negative_mode_schedule = str(alpha.get("bind_negative_mode_schedule", "none")).strip().lower()
    bind_negative_mode_c_ramp_start_step = int(alpha.get("bind_negative_mode_c_ramp_start_step", 0))
    bind_negative_mode_c_ramp_end_step = int(alpha.get("bind_negative_mode_c_ramp_end_step", 50))
    bind_negative_mode_c_ramp_start_weight = float(alpha.get("bind_negative_mode_c_ramp_start_weight", 0.05))
    bind_negative_mode_c_ramp_end_weight = float(alpha.get("bind_negative_mode_c_ramp_end_weight", 0.25))
    raw_probe_steps = alpha.get("bind_negative_mode_schedule_probe_steps", [0, 10, 20, 30, 40, 50, 75, 100])
    if not isinstance(raw_probe_steps, list):
        raw_probe_steps = [0, 10, 20, 30, 40, 50, 75, 100]
    bind_negative_mode_schedule_probe_steps = sorted({max(0, int(step)) for step in raw_probe_steps})
    raw_c_targeted_classes = alpha.get("bind_c_targeted_classes", "mixed,ceiling")
    if isinstance(raw_c_targeted_classes, str):
        bind_c_targeted_classes = [v.strip().lower() for v in raw_c_targeted_classes.split(",") if v.strip()]
    elif isinstance(raw_c_targeted_classes, list):
        bind_c_targeted_classes = [str(v).strip().lower() for v in raw_c_targeted_classes if str(v).strip()]
    else:
        bind_c_targeted_classes = ["mixed", "ceiling"]
    
    return {
        "enabled": bool(alpha.get("enabled", False)),
        "strict_alpha": bool(alpha.get("strict_alpha", False)),
        "head_scales": [int(index) for index in head_scales],
        "loss_weight": float(alpha.get("loss_weight", 0.1)),
        "edge_loss_scale": float(alpha.get("edge_loss_scale", 0.5)),
        "binary_threshold": float(alpha.get("binary_threshold", 0.5)),
        "warmup_steps": int(alpha.get("warmup_steps", 50)),
        "prior_start_weight": float(alpha.get("prior_start_weight", 0.75)),
        "prior_end_weight": float(alpha.get("prior_end_weight", 0.0)),
        "logit_temperature_start": float(alpha.get("logit_temperature_start", 2.0)),
        "logit_temperature_end": float(alpha.get("logit_temperature_end", 1.0)),
        "edge_band_weight": float(alpha.get("edge_band_weight", 2.0)),
        "edge_dilate_px": int(alpha.get("edge_dilate_px", 3)),
        "supervision_expand_px": int(alpha.get("supervision_expand_px", 6)),
        "terrain_coupling_weight": float(alpha.get("terrain_coupling_weight", 0.0)),
        "terrain_iou_weight": float(alpha.get("terrain_iou_weight", 0.0)),
        "terrain_curriculum_start": float(alpha.get("terrain_curriculum_start", 1.0)),
        "terrain_curriculum_end": float(alpha.get("terrain_curriculum_end", 0.25)),
        "terrain_curriculum_steps": int(alpha.get("terrain_curriculum_steps", 200)),
        "terrain_presence_floor": float(alpha.get("terrain_presence_floor", 0.02)),
        "terrain_presence_boost": float(alpha.get("terrain_presence_boost", 1.5)),
        "synthesized_alpha_warn_fraction": float(alpha.get("synthesized_alpha_warn_fraction", 0.25)),
        "dominance_warn_ratio": float(alpha.get("dominance_warn_ratio", 0.5)),
        "head_mode": str(alpha.get("head_mode", "single_high")),
        "single_scale_index": int(alpha.get("single_scale_index", -1)),
        "output_source": str(alpha.get("output_source", "main")),
        "baseline_mode": str(alpha.get("baseline_mode", "none")),
        "pre_gate_target_terrain_iou_min": float(alpha.get("pre_gate_target_terrain_iou_min", 0.60)),
        "terrain_mask_black_is_terrain": bool(alpha.get("terrain_mask_black_is_terrain", True)),
        "bind_preference_weight": float(alpha.get("bind_preference_weight", 0.0)),
        "bind_preference_margin": float(alpha.get("bind_preference_margin", 0.0)),
        "bind_paired_batch_ratio": bind_ratio,
        "bind_pair_seed": int(alpha.get("bind_pair_seed", 1337)),
        "bind_negative_mode": str(alpha.get("bind_negative_mode", "mixed")),
        "bind_negative_mode_weights": bind_negative_mode_weights,
        "bind_negative_roi_min_frac": float(alpha.get("bind_negative_roi_min_frac", 0.08)),
        "bind_negative_roi_max_frac": float(alpha.get("bind_negative_roi_max_frac", 0.28)),
        "bind_log_margin_variant": bool(alpha.get("bind_log_margin_variant", True)),
        "bind_negative_mode_c_activation_step": bind_negative_mode_c_activation_step,
        "bind_negative_mode_weights_until_c": bind_negative_mode_weights_until_c,
        "bind_negative_mode_schedule": bind_negative_mode_schedule,
        "bind_negative_mode_c_ramp_start_step": bind_negative_mode_c_ramp_start_step,
        "bind_negative_mode_c_ramp_end_step": bind_negative_mode_c_ramp_end_step,
        "bind_negative_mode_c_ramp_start_weight": bind_negative_mode_c_ramp_start_weight,
        "bind_negative_mode_c_ramp_end_weight": bind_negative_mode_c_ramp_end_weight,
        "bind_negative_mode_schedule_probe_steps": bind_negative_mode_schedule_probe_steps,
        "bind_c_directional_rejection_strength": float(alpha.get("bind_c_directional_rejection_strength", 0.14)),
        "bind_c_signed_polarity_strength": float(alpha.get("bind_c_signed_polarity_strength", 0.18)),
        "bind_c_interior_haze_suppression": float(alpha.get("bind_c_interior_haze_suppression", 0.12)),
        "bind_c_transition_softness": float(alpha.get("bind_c_transition_softness", 0.12)),
        "bind_c_directional_gate_threshold": float(alpha.get("bind_c_directional_gate_threshold", 0.08)),
        "bind_c_targeted_classes": bind_c_targeted_classes,
    }


def _resolve_effective_mode_weights(alpha_config: Dict[str, object], current_step: int = -1) -> Dict[str, float]:
    schedule = str(alpha_config.get("bind_negative_mode_schedule", "none")).strip().lower()
    full_weights = dict(alpha_config.get("bind_negative_mode_weights", {}) or {})

    if schedule == "linear_c_ramp" and current_step >= 0:
        ramp_start = int(alpha_config.get("bind_negative_mode_c_ramp_start_step", 0))
        ramp_end = int(alpha_config.get("bind_negative_mode_c_ramp_end_step", 50))
        c_start = float(alpha_config.get("bind_negative_mode_c_ramp_start_weight", 0.05))
        c_end = float(alpha_config.get("bind_negative_mode_c_ramp_end_weight", 0.25))

        if ramp_end <= ramp_start:
            ramp_end = ramp_start + 1

        if current_step <= ramp_start:
            t = 0.0
        elif current_step >= ramp_end:
            t = 1.0
        else:
            t = float(current_step - ramp_start) / float(ramp_end - ramp_start)

        c_weight = c_start + (c_end - c_start) * t
        c_weight = max(0.0, min(1.0, c_weight))
        remaining = max(0.0, 1.0 - c_weight)

        a_base = max(0.0, float(full_weights.get("wall_to_flat_suppression", 0.40)))
        b_base = max(0.0, float(full_weights.get("flat_to_wall_injection", 0.35)))
        ab_total = a_base + b_base
        if ab_total <= 0.0:
            a_base, b_base = 0.40, 0.35
            ab_total = 0.75

        return {
            "wall_to_flat_suppression": remaining * (a_base / ab_total),
            "flat_to_wall_injection": remaining * (b_base / ab_total),
            "local_spatial_misalignment": c_weight,
        }

    # Fallback to existing hard-delay behavior.
    c_activation_step = int(alpha_config.get("bind_negative_mode_c_activation_step", -1))
    weights_until_c = alpha_config.get("bind_negative_mode_weights_until_c", None)
    if c_activation_step >= 0 and weights_until_c is not None and current_step >= 0 and current_step < c_activation_step:
        weights = dict(weights_until_c or {})
    else:
        weights = full_weights

    return {
        "wall_to_flat_suppression": max(0.0, float(weights.get("wall_to_flat_suppression", 0.40))),
        "flat_to_wall_injection": max(0.0, float(weights.get("flat_to_wall_injection", 0.35))),
        "local_spatial_misalignment": max(0.0, float(weights.get("local_spatial_misalignment", 0.25))),
    }


def _sample_negative_mode(alpha_config: Dict[str, object], rng: random.Random, current_step: int = -1) -> str:
    configured_mode = str(alpha_config.get("bind_negative_mode", "mixed")).strip().lower()
    supported = (
        "wall_to_flat_suppression",
        "flat_to_wall_injection",
        "local_spatial_misalignment",
    )

    if configured_mode != "mixed":
        if configured_mode in supported:
            return configured_mode
        logger.warning(f"unsupported bind_negative_mode='{configured_mode}', falling back to wall_to_flat_suppression")
        return "wall_to_flat_suppression"

    mode_weights = _resolve_effective_mode_weights(alpha_config, current_step=current_step)
    total = sum(mode_weights.values())
    if total <= 0.0:
        logger.warning("bind_negative_mode_weights sum to <= 0; falling back to wall_to_flat_suppression")
        return "wall_to_flat_suppression"

    draw = rng.uniform(0.0, total)
    cumulative = 0.0
    for mode, weight in mode_weights.items():
        cumulative += weight
        if draw <= cumulative:
            return mode
    return "wall_to_flat_suppression"


def _mode_c_precondition(
    dataset: TerrainSemanticManifestDataset,
    conditioning: torch.Tensor,
    terrain_mask_black_is_terrain: bool = True,
) -> Tuple[bool, str]:
    _, _, groups = _build_extreme_contrast_conditioning(dataset, conditioning)
    terrain_indices = groups.get("terrain_mask", [])
    edge_indices = groups.get("edge", [])
    openness_indices = groups.get("openness", [])
    non_mask_indices = edge_indices + openness_indices
    if not terrain_indices:
        return False, "missing_terrain_mask_channel"
    if not non_mask_indices:
        return False, "missing_nonmask_geometry_channels"

    terrain_idx = terrain_indices[0]
    terrain = conditioning[:, terrain_idx : terrain_idx + 1, :, :]
    if terrain_mask_black_is_terrain:
        terrain_presence = (terrain < 0.5).float().mean().item()
    else:
        terrain_presence = (terrain > 0.5).float().mean().item()
    if terrain_presence < 0.01:
        return False, "terrain_presence_too_low"
    if terrain_presence > 0.99:
        return False, "terrain_presence_too_high"
    return True, "none"


def parse_verification_config(config: Dict[str, object]) -> Dict[str, object]:
    verification = config.get("verification", {})
    raw_sweep = verification.get("multiplier_sweep", [2.0, 3.0, 5.0])
    if not isinstance(raw_sweep, list) or not raw_sweep:
        raise ValueError("verification.multiplier_sweep must be a non-empty list")
    multiplier_sweep = [float(value) for value in raw_sweep]
    return {
        "enabled": bool(verification.get("enabled", True)),
        "log_every": int(verification.get("log_every", 25)),
        "gradient_warn_threshold": float(verification.get("gradient_warn_threshold", 1e-10)),
        "param_delta_warn_threshold": float(verification.get("param_delta_warn_threshold", 1e-10)),
        "run_controlnet_sanity": bool(verification.get("run_controlnet_sanity", True)),
        "controlnet_sanity_steps": int(verification.get("controlnet_sanity_steps", 6)),
        "controlnet_min_mse": float(verification.get("controlnet_min_mse", 1e-6)),
        "save_sanity_previews": bool(verification.get("save_sanity_previews", True)),
        "run_multiplier_sweep_sanity": bool(verification.get("run_multiplier_sweep_sanity", True)),
        "run_extreme_contrast_test": bool(verification.get("run_extreme_contrast_test", True)),
        "multiplier_sweep": multiplier_sweep,
        "always_log_during_tiny_overfit": bool(verification.get("always_log_during_tiny_overfit", True)),
        "tiny_overfit_max_steps": int(verification.get("tiny_overfit_max_steps", 400)),
        "enable_seam_diagnostic_guards": bool(verification.get("enable_seam_diagnostic_guards", False)),
        "seam_halo_target_energy_min": float(verification.get("seam_halo_target_energy_min", 1e-3)),
        "seam_loss_contribution_ratio_min": float(verification.get("seam_loss_contribution_ratio_min", 0.05)),
        "train_expanded_supervision_enabled": bool(verification.get("train_expanded_supervision_enabled", False)),
        "train_expanded_halo_px": int(verification.get("train_expanded_halo_px", 0)),
        "save_seam_visual_debug": bool(verification.get("save_seam_visual_debug", False)),
        "seam_visual_debug_max_steps": int(verification.get("seam_visual_debug_max_steps", 2)),
        "loss_trace_mode": _normalize_loss_trace_mode(verification.get("loss_trace_mode", "full")),
    }


def parse_conditioning_config(config: Dict[str, object]) -> Dict[str, object]:
    conditioning = config.get("conditioning", {})
    return {
        "cond_embedding_lr_multiplier": float(conditioning.get("cond_embedding_lr_multiplier", 10.0)),
    }


def _parse_step_value_schedule(raw_schedule: object, field_name: str) -> List[Tuple[int, float]]:
    if raw_schedule is None:
        return []
    if not isinstance(raw_schedule, list):
        raise ValueError(f"{field_name} must be a list of {{step, value}} entries")

    schedule: List[Tuple[int, float]] = []
    for index, entry in enumerate(raw_schedule):
        if not isinstance(entry, dict):
            raise ValueError(f"{field_name}[{index}] must be a table with step and value")
        if entry.get("step") is None or entry.get("value") is None:
            raise ValueError(f"{field_name}[{index}] must include step and value")
        schedule.append((max(0, int(entry["step"])), float(entry["value"])))

    schedule.sort(key=lambda item: item[0])
    return schedule


def _resolve_step_value_schedule(default_value: float, schedule: object, current_step: int) -> float:
    resolved_value = float(default_value)
    if current_step < 0 or not isinstance(schedule, list):
        return resolved_value

    for start_step, step_value in schedule:
        if current_step < int(start_step):
            break
        resolved_value = float(step_value)

    return resolved_value


def parse_seam_config(config: Dict[str, object]) -> Dict[str, object]:
    seam = config.get("seam", {})

    raw_adapter_inputs = seam.get(
        "seam_adapter_inputs",
        ["projected_halo_rgb", "projected_halo_alpha", "edge_valid_map", "distance_to_seam"],
    )
    if isinstance(raw_adapter_inputs, str):
        seam_adapter_inputs = [value.strip() for value in raw_adapter_inputs.split(",") if value.strip()]
    elif isinstance(raw_adapter_inputs, list):
        seam_adapter_inputs = [str(value).strip() for value in raw_adapter_inputs if str(value).strip()]
    else:
        seam_adapter_inputs = ["projected_halo_rgb", "projected_halo_alpha", "edge_valid_map", "distance_to_seam"]

    def _get_float(keys: List[str], default: float) -> float:
        for key in keys:
            if seam.get(key) is not None:
                return float(seam.get(key))
        return float(default)

    def _get_int(keys: List[str], default: int) -> int:
        for key in keys:
            if seam.get(key) is not None:
                return int(seam.get(key))
        return int(default)

    halo_inner_rgb_weight = _get_float(["halo_inner_rgb_weight", "halo_inner_weight", "margin_inner_weight"], 2.0)
    halo_outer_rgb_weight = _get_float(["halo_outer_rgb_weight", "halo_outer_weight", "margin_outer_weight"], 1.5)
    continuation_width_px = _get_int(["continuation_width_px", "continuation_band_px", "interior_band_inner_px"], 48)
    continuation_peak_rgb_weight = _get_float(
        ["continuation_peak_rgb_weight", "interior_continuation_weight", "interior_band_inner_weight"],
        2.5,
    )
    interior_core_rgb_weight = _get_float(["interior_core_rgb_weight", "interior_core_weight", "interior_band_outer_weight"], 0.10)
    continuation_gradient_loss_weight = _get_float(["continuation_gradient_loss_weight"], 0.25)
    continuation_gradient_loss_weight_schedule = _parse_step_value_schedule(
        seam.get("continuation_gradient_loss_weight_schedule"),
        "seam.continuation_gradient_loss_weight_schedule",
    )
    continuation_falloff_power = _get_float(["continuation_falloff_power"], 2.0)
    gradient_loss_sobel_radius_px = _get_int(["gradient_loss_sobel_radius_px"], 1)
    seam_adapter_per_edge = bool(seam.get("seam_adapter_per_edge", False))
    seam_adapter_extrusion_mode = str(seam.get("seam_adapter_extrusion_mode", "decay") or "decay").strip().lower()
    seam_adapter_target = str(seam.get("seam_adapter_target", "first_high_res") or "first_high_res").strip().lower()
    seam_adapter_multi_inject = bool(seam.get("seam_adapter_multi_inject", False))
    raw_injection_blocks = seam.get("seam_adapter_injection_blocks", ["first_high_res"])
    if isinstance(raw_injection_blocks, str):
        seam_adapter_injection_blocks = [raw_injection_blocks.strip().lower()] if raw_injection_blocks.strip() else []
    elif isinstance(raw_injection_blocks, (list, tuple)):
        seam_adapter_injection_blocks = [str(block).strip().lower() for block in raw_injection_blocks if str(block).strip()]
    else:
        raise ValueError("seam_adapter_injection_blocks must be a string or list of strings")

    seam_adapter_multi_inject_mode = str(
        seam.get("seam_adapter_multi_inject_mode", "shared_trunk_per_block_heads") or "shared_trunk_per_block_heads"
    ).strip().lower()
    if seam_adapter_multi_inject_mode not in {"shared_residual", "shared_trunk_per_block_heads"}:
        raise ValueError(
            f"unsupported seam_adapter_multi_inject_mode='{seam_adapter_multi_inject_mode}'"
        )
    if seam_adapter_multi_inject:
        allowed_multi_blocks = {"first_high_res", "second_high_res"}
        if not seam_adapter_injection_blocks:
            raise ValueError("seam_adapter_injection_blocks must not be empty when seam_adapter_multi_inject=true")
        if len(set(seam_adapter_injection_blocks)) != len(seam_adapter_injection_blocks):
            raise ValueError("seam_adapter_injection_blocks contains duplicates")
        invalid_blocks = [block for block in seam_adapter_injection_blocks if block not in allowed_multi_blocks]
        if invalid_blocks:
            raise ValueError(
                "this seam adapter escalation only supports first_high_res and second_high_res blocks; got "
                + ", ".join(invalid_blocks)
            )
        if len(seam_adapter_injection_blocks) > 2:
            raise ValueError("this seam adapter escalation supports at most two injection blocks")
    else:
        seam_adapter_injection_blocks = [seam_adapter_target]
    continuation_rgb_weight_mode_raw = str(seam.get("continuation_rgb_weight_mode", "") or "").strip().lower()
    if continuation_rgb_weight_mode_raw:
        continuation_rgb_weight_mode = continuation_rgb_weight_mode_raw
    elif seam_adapter_per_edge and seam_adapter_extrusion_mode == "full_strength":
        continuation_rgb_weight_mode = "uniform"
    else:
        continuation_rgb_weight_mode = "falloff"
    if continuation_rgb_weight_mode not in {"falloff", "uniform", "linear"}:
        raise ValueError(
            "seam.continuation_rgb_weight_mode must be one of 'falloff', 'uniform', or 'linear': "
            + f"got {continuation_rgb_weight_mode!r}"
        )

    return {
        "enabled": bool(seam.get("enabled", False)),
        "strip_width_px": int(seam.get("strip_width_px", 64)),
        "state_all_defined_weight": float(seam.get("state_all_defined_weight", 0.25)),
        "state_partial_defined_weight": float(seam.get("state_partial_defined_weight", 0.50)),
        "state_none_defined_weight": float(seam.get("state_none_defined_weight", 0.25)),
        "partial_one_edge_ratio": float(seam.get("partial_one_edge_ratio", 0.45)),
        "undefined_zero_prob": float(seam.get("undefined_zero_prob", 0.40)),
        "undefined_noise_prob": float(seam.get("undefined_noise_prob", 0.40)),
        "alpha_local_loss_weight": float(seam.get("alpha_local_loss_weight", 0.0)),
        "loss_narrow_min_decay": float(seam.get("loss_narrow_min_decay", 0.5)),
        "margin_inner_px": int(seam.get("margin_inner_px", 32)),
        "margin_inner_weight": halo_inner_rgb_weight,
        "margin_outer_weight": halo_outer_rgb_weight,
        "interior_band_inner_px": int(seam.get("interior_band_inner_px", continuation_width_px)),
        "interior_band_outer_px": int(seam.get("interior_band_outer_px", max(continuation_width_px, 32))),
        "interior_band_inner_weight": continuation_peak_rgb_weight,
        "interior_band_outer_weight": interior_core_rgb_weight,
        "halo_inner_rgb_weight": halo_inner_rgb_weight,
        "halo_outer_rgb_weight": halo_outer_rgb_weight,
        "continuation_width_px": continuation_width_px,
        "continuation_peak_rgb_weight": continuation_peak_rgb_weight,
        "interior_core_rgb_weight": interior_core_rgb_weight,
        "continuation_gradient_loss_weight": continuation_gradient_loss_weight,
        "continuation_gradient_loss_weight_schedule": continuation_gradient_loss_weight_schedule,
        "continuation_falloff_power": continuation_falloff_power,
        "continuation_rgb_weight_mode": continuation_rgb_weight_mode,
        "gradient_loss_sobel_radius_px": gradient_loss_sobel_radius_px,
        "rgb_recon_loss_weight": float(seam.get("rgb_recon_loss_weight", 0.0)),
        "normalize_region_losses": bool(seam.get("normalize_region_losses", True)),
        "require_defined_for_margin_and_band": bool(seam.get("require_defined_for_margin_and_band", True)),
        "force_defined_strip_supervision": bool(seam.get("force_defined_strip_supervision", True)),
        "seam_qualified_sampling_enabled": bool(seam.get("seam_qualified_sampling_enabled", False)),
        "seam_qualified_min_continuation_px": int(seam.get("seam_qualified_min_continuation_px", 0)),
        "seam_qualified_min_halo_px": int(seam.get("seam_qualified_min_halo_px", 0)),
        "seam_supervision_expand_px": int(seam.get("seam_supervision_expand_px", 0)),
        "boundary_chunk_stride_px": int(seam.get("boundary_chunk_stride_px", 16)),
        "boundary_grid_offset_x_px": int(seam.get("boundary_grid_offset_x_px", 0)),
        "boundary_grid_offset_y_px": int(seam.get("boundary_grid_offset_y_px", 0)),
        "boundary_alignment_error_max_px": float(seam.get("boundary_alignment_error_max_px", 0.5)),
        "boundary_consistency_error_max_px": float(seam.get("boundary_consistency_error_max_px", 0.5)),
        "seed": int(seam.get("seed", 1337)),
        "fixed_defined_edge": str(seam.get("fixed_defined_edge", "") or "").strip().lower(),
        "seam_adapter_enabled": bool(seam.get("seam_adapter_enabled", False)),
        "seam_adapter_band_px": int(seam.get("seam_adapter_band_px", continuation_width_px)),
        "seam_adapter_scale": float(seam.get("seam_adapter_scale", 1.0)),
        "seam_adapter_zero_init": bool(seam.get("seam_adapter_zero_init", True)),
        "seam_adapter_target": seam_adapter_target,
        "seam_adapter_multi_inject": bool(seam.get("seam_adapter_multi_inject", False)),
        "seam_adapter_injection_blocks": seam_adapter_injection_blocks,
        "seam_adapter_scale_block0": float(seam.get("seam_adapter_scale_block0", 1.0)),
        "seam_adapter_scale_block1": float(seam.get("seam_adapter_scale_block1", 0.5)),
        "seam_adapter_multi_inject_mode": seam_adapter_multi_inject_mode,
        "seam_adapter_per_edge": seam_adapter_per_edge,
        "seam_adapter_extrusion_mode": seam_adapter_extrusion_mode,
        "seam_adapter_inputs": seam_adapter_inputs,
        "seam_adapter_conditioning_offset": -1,
    }


def parse_evaluation_config(
    config: Dict[str, object],
    alpha_config: Dict[str, object],
    output_name: str,
    max_train_steps: int,
) -> Dict[str, object]:
    evaluation = config.get("evaluation", {})
    training = config.get("training", {})

    enabled = bool(evaluation.get("enabled", True))
    if enabled and not alpha_config["enabled"]:
        raise RuntimeError("alpha-aware pipeline requires [alpha].enabled=true when evaluation is enabled")

    eval_steps = [int(step) for step in evaluation.get("eval_steps", [60, 120, 200])]
    eval_steps = sorted(set([step for step in eval_steps if step > 0 and step <= int(max_train_steps)]))
    seeds = [int(seed) for seed in evaluation.get("seeds", [1234])]
    if not seeds:
        raise ValueError("evaluation.seeds must contain at least one seed")

    manifest_path = evaluation.get("eval_manifest_path")
    if not manifest_path:
        raise ValueError("evaluation.eval_manifest_path is required")
    if not os.path.isabs(manifest_path):
        manifest_path = os.path.normpath(os.path.join(config["__config_dir"], manifest_path))

    return {
        "enabled": enabled,
        "include_step0": bool(evaluation.get("include_step0", True)),
        "eval_steps": eval_steps,
        "prompt": str(evaluation.get("prompt", training.get("prompt", ""))),
        "prompt2": str(evaluation.get("prompt2", training.get("prompt2", training.get("prompt", "")))),
        "seeds": seeds,
        "inference_steps": int(evaluation.get("inference_steps", 8)),
        "guidance_scale": float(evaluation.get("guidance_scale", 1.0)),
        "eval_manifest_path": manifest_path,
        "max_samples": int(evaluation.get("max_samples", 12)),
        "board_tile_min_size": int(evaluation.get("board_tile_min_size", 256)),
        "full_scene_tile_min_size": int(evaluation.get("full_scene_tile_min_size", 512)),
        "collapse_mse_threshold": float(evaluation.get("collapse_mse_threshold", 1e-4)),
        "write_latent_debug": bool(evaluation.get("write_latent_debug", False)),
        "alpha_preview_mode": str(evaluation.get("alpha_preview_mode", "mask")),
        "binary_threshold": float(alpha_config.get("binary_threshold", 0.5)),
        "supervision_expand_px": int(alpha_config.get("supervision_expand_px", 0)),
        "alpha_output_source": str(alpha_config.get("output_source", "main")),
        "terrain_mask_black_is_terrain": bool(alpha_config.get("terrain_mask_black_is_terrain", True)),
        "expanded_prediction_enabled": bool(evaluation.get("expanded_prediction_enabled", False)),
        "expanded_halo_px": int(evaluation.get("expanded_halo_px", 0)),
        "alpha_iou_min": float(evaluation.get("alpha_iou_min", 0.35)),
        "alpha_bce_max": float(evaluation.get("alpha_bce_max", 0.62)),
        "alpha_corr_min": float(evaluation.get("alpha_corr_min", 0.30)),
        "alpha_occ_min": float(evaluation.get("alpha_occ_min", 0.08)),
        "alpha_occ_max": float(evaluation.get("alpha_occ_max", 0.92)),
        "alpha_speckle_max": float(evaluation.get("alpha_speckle_max", 0.45)),
        "alpha_iou_delta_min": float(evaluation.get("alpha_iou_delta_min", 0.08)),
        "alpha_bce_delta_min": float(evaluation.get("alpha_bce_delta_min", 0.05)),
        "diffusion_tail_slope_max": float(evaluation.get("diffusion_tail_slope_max", 0.002)),
        "run_name": output_name,
    }


def parse_binding_eval_config(
    config: Dict[str, object],
    alpha_config: Dict[str, object],
) -> Dict[str, object]:
    """Parse [binding_eval] section from the semantic config.

    Returns a dict consumed by resolve_swap_pairs() and run_semantic_binding_eval().
    All fields have safe defaults so the section is optional.
    """
    binding = config.get("binding_eval", {})
    enabled = bool(binding.get("enabled", False))
    if not enabled:
        return {"enabled": False}

    swap_manifest_path = binding.get("swap_manifest_path")
    if not swap_manifest_path:
        return {"enabled": False}
    if not os.path.isabs(swap_manifest_path):
        swap_manifest_path = os.path.normpath(
            os.path.join(config["__config_dir"], str(swap_manifest_path))
        )

    return {
        "enabled": True,
        "swap_manifest_path": swap_manifest_path,
        "seeds_panel": [int(s) for s in binding.get("seeds_panel", [1234, 5678, 9012])],
        "inference_steps": int(binding.get("inference_steps", 8)),
        "blur_radius": float(binding.get("blur_radius", 1.5)),
        "ablation_shuffle_seed": int(binding.get("ablation_shuffle_seed", 9999)),
        "terrain_mask_channel_index": int(binding.get("terrain_mask_channel_index", 3)),
        "null_space_channel_groups": list(binding.get("null_space_channel_groups", [[4, 5, 6, 7], [8, 9, 10, 11], [0, 1, 2]])),
        "null_space_group_names": list(binding.get("null_space_group_names", ["edge_channels", "openness_channels", "base_semantic"])),
        "alpha_output_source": str(alpha_config.get("output_source", "main")),
    }


def parse_channel_specs(config: Dict[str, object]) -> List[SemanticChannelSpec]:
    channel_items = []
    for key, value in config.items():
        if not key.startswith("channel_"):
            continue
        try:
            index = int(key.split("_", 1)[1])
        except ValueError as exc:
            raise ValueError(f"invalid channel key: {key}") from exc
        channel_items.append((index, value))

    if not channel_items:
        raise ValueError("semantic config must define channel_0 ... channel_n entries")

    channel_items.sort(key=lambda item: item[0])
    specs: List[SemanticChannelSpec] = []
    for index, entry in channel_items:
        semantic_range = tuple(float(v) for v in entry["range"])
        clamp_range = entry.get("clamp")
        disk_range = entry.get("disk_range")
        specs.append(
            SemanticChannelSpec(
                name=entry.get("name", f"channel_{index}"),
                source=entry["source"],
                semantic_range=(semantic_range[0], semantic_range[1]),
                clamp_range=None if clamp_range is None else (float(clamp_range[0]), float(clamp_range[1])),
                disk_range=None if disk_range is None else (float(disk_range[0]), float(disk_range[1])),
            )
        )
    return specs


def build_dataset(
    args: argparse.Namespace,
    semantic_config: Dict[str, object],
    alpha_config: Dict[str, object],
    seam_config: Optional[Dict[str, object]] = None,
) -> TerrainSemanticManifestDataset:
    training = semantic_config["training"]
    verification = semantic_config.get("verification", {})
    resolution = tuple(args.resolution) if args.resolution is not None else tuple(training["resolution"])
    channel_specs = parse_channel_specs(semantic_config)
    vae_cache_key = args.latent_cache_vae_key
    if not vae_cache_key:
        vae_source = args.vae if args.vae else args.pretrained_model_name_or_path
        vae_cache_key = f"{vae_source}|mp={args.mixed_precision}|no_half_vae={int(args.no_half_vae)}"

    return TerrainSemanticManifestDataset(
        root_dir=semantic_config["dataset_root"],
        manifest_path=semantic_config["manifest_path"],
        channel_specs=channel_specs,
        train_size=(int(resolution[0]), int(resolution[1])),
        prompt=training["prompt"],
        prompt2=training.get("prompt2"),
        min_trusted_mask_ratio=float(training.get("min_trusted_mask_ratio", 0.05)),
        image_resize_mode=training.get("image_resize_mode", "bicubic"),
        semantic_resize_mode=training.get("semantic_resize_mode", "bilinear"),
        latent_cache_dir=args.latent_cache_dir,
        latent_cache_version=args.latent_cache_version,
        latent_cache_vae_key=vae_cache_key,
        enable_alpha_supervision=alpha_config["enabled"],
        strict_alpha=alpha_config["strict_alpha"],
        seam_enabled=bool((seam_config or {}).get("enabled", False)),
        seam_strip_width_px=int((seam_config or {}).get("strip_width_px", 64)),
        seam_state_all_defined_weight=float((seam_config or {}).get("state_all_defined_weight", 0.25)),
        seam_state_partial_defined_weight=float((seam_config or {}).get("state_partial_defined_weight", 0.50)),
        seam_state_none_defined_weight=float((seam_config or {}).get("state_none_defined_weight", 0.25)),
        seam_partial_one_edge_ratio=float((seam_config or {}).get("partial_one_edge_ratio", 0.45)),
        seam_undefined_zero_prob=float((seam_config or {}).get("undefined_zero_prob", 0.40)),
        seam_undefined_noise_prob=float((seam_config or {}).get("undefined_noise_prob", 0.40)),
        seam_fixed_defined_edge_index=resolve_fixed_defined_edge_index((seam_config or {}).get("fixed_defined_edge", "")),
        seam_runtime_config=dict(seam_config or {}),
        expanded_target_halo_px=(
            int(verification.get("train_expanded_halo_px", 0))
            if bool(verification.get("train_expanded_supervision_enabled", False))
            else 0
        ),
        terrain_mask_black_is_terrain=bool(alpha_config.get("terrain_mask_black_is_terrain", True)),
        alpha_binary_threshold=float(alpha_config.get("binary_threshold", 0.5)),
        boundary_chunk_stride_px=int((seam_config or {}).get("boundary_chunk_stride_px", 16)),
        boundary_grid_offset_x_px=int((seam_config or {}).get("boundary_grid_offset_x_px", 0)),
        boundary_grid_offset_y_px=int((seam_config or {}).get("boundary_grid_offset_y_px", 0)),
        boundary_alignment_error_max_px=float((seam_config or {}).get("boundary_alignment_error_max_px", 0.5)),
        boundary_consistency_error_max_px=float((seam_config or {}).get("boundary_consistency_error_max_px", 0.5)),
        seam_seed=int((seam_config or {}).get("seed", 1337)),
    )


def semantic_collate(samples: List[Dict[str, object]]) -> Dict[str, object]:
    batch: Dict[str, object] = {}
    tensor_keys = {
        "images",
        "conditioning_images",
        "trusted_mask",
        "alpha_has_native",
        "original_sizes_hw",
        "crop_top_lefts",
        "target_sizes_hw",
        "crop_box",
        "trusted_box",
        "trusted_ratio",
        "sampling_weight",
        "seam_enabled",
    }
    optional_tensor_keys = {
        "latents",
        "alpha_target",
        "expanded_images",
        "expanded_alpha_target",
        "expanded_trusted_mask",
        "expanded_target_sizes_hw",
        "expanded_zero_mask",
        "expanded_crop_box",
        "seam_strip_tensor",
        "edge_defined_flags",
        "edge_flag_maps",
        "edge_band_masks",
        "seam_decay_maps",
        "expanded_edge_band_masks",
        "expanded_seam_decay_maps",
        "seam_state_label",
        "seam_undefined_mode",
        "seam_strip_width_px",
        "seam_qualified_edge_mask",
        "seam_qualified_continuation_px",
        "seam_qualified_halo_inner_px",
        "seam_qualified_halo_outer_px",
        "seam_qualified_continuation_weight_sum",
        "seam_qualified_valid_edges_count",
        "boundary_alignment_error",
        "boundary_consistency_error",
    }

    for key in tensor_keys:
        batch[key] = torch.stack([sample[key] for sample in samples], dim=0)

    for key in optional_tensor_keys:
        values = [sample[key] for sample in samples]
        if any(value is None for value in values):
            batch[key] = None
        else:
            batch[key] = torch.stack(values, dim=0)

    batch["image_name"] = [sample["image_name"] for sample in samples]
    batch["special_structure_tags"] = [sample["special_structure_tags"] for sample in samples]
    batch["assigned_crop_class"] = [sample.get("assigned_crop_class", "") for sample in samples]
    batch["crop_size_class"] = [sample["crop_size_class"] for sample in samples]
    batch["generation_strategy"] = [sample["generation_strategy"] for sample in samples]
    batch["prompt"] = samples[0]["prompt"]
    batch["prompt2"] = samples[0]["prompt2"]
    batch["channel_names"] = samples[0]["channel_names"]
    if "full_conditioning_channel_names" in samples[0]:
        batch["full_conditioning_channel_names"] = samples[0]["full_conditioning_channel_names"]
    return batch


def build_model_visible_conditioning(
    batch: Dict[str, object],
    dataset: TerrainSemanticManifestDataset,
    seam_config: Dict[str, object],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, List[str], Dict[str, float]]:
    conditioning = batch["conditioning_images"].to(device, dtype=dtype)
    seam_enabled = bool(seam_config.get("enabled", False))
    diagnostics = {
        "seam_enabled": 1.0 if seam_enabled else 0.0,
        "seam_defined_ratio": 0.0,
        "seam_pre_gate_l2": 0.0,
        "seam_post_gate_l2": 0.0,
        "seam_visible_conditioning_l2": 0.0,
        "seam_undefined_edges_count": 0.0,
        "seam_undefined_pre_gate_l2": 0.0,
        "seam_undefined_post_gate_l2": 0.0,
    }
    edge_names = ("north", "south", "east", "west")
    for edge_name in edge_names:
        diagnostics[f"seam_edge_{edge_name}_defined"] = 0.0
        diagnostics[f"seam_edge_{edge_name}_pre_gate_l2"] = 0.0
        diagnostics[f"seam_edge_{edge_name}_post_gate_l2"] = 0.0
        diagnostics[f"seam_edge_{edge_name}_visible_l2"] = 0.0
    if not seam_enabled or batch.get("seam_strip_tensor") is None or batch.get("edge_defined_flags") is None or batch.get("edge_flag_maps") is None:
        return conditioning, dataset.channel_names, diagnostics

    seam_strip = batch["seam_strip_tensor"].to(device, dtype=dtype)
    edge_defined_flags = batch["edge_defined_flags"].to(device, dtype=dtype)
    edge_flag_maps = batch["edge_flag_maps"].to(device, dtype=dtype)

    # Final post-augmentation gating right before ControlNet input assembly.
    seam_gate = edge_defined_flags.repeat_interleave(4, dim=1).unsqueeze(-1).unsqueeze(-1)
    seam_visible = seam_strip * seam_gate
    full_conditioning = torch.cat([conditioning, seam_visible, edge_flag_maps], dim=1)

    undefined_edge_flags = (1.0 - edge_defined_flags).clamp(0.0, 1.0)
    undefined_gate = undefined_edge_flags.repeat_interleave(4, dim=1).unsqueeze(-1).unsqueeze(-1)
    seam_undefined = seam_strip * undefined_gate
    seam_undefined_visible = seam_visible * undefined_gate

    diagnostics["seam_defined_ratio"] = float(edge_defined_flags.mean().detach().item())
    diagnostics["seam_pre_gate_l2"] = float(seam_strip.detach().float().pow(2.0).mean().sqrt().item())
    diagnostics["seam_post_gate_l2"] = float(seam_visible.detach().float().pow(2.0).mean().sqrt().item())
    diagnostics["seam_visible_conditioning_l2"] = float(full_conditioning.detach().float().pow(2.0).mean().sqrt().item())
    diagnostics["seam_undefined_edges_count"] = float(undefined_edge_flags.sum().detach().item())
    diagnostics["seam_undefined_pre_gate_l2"] = float(seam_undefined.detach().float().pow(2.0).mean().sqrt().item())
    diagnostics["seam_undefined_post_gate_l2"] = float(seam_undefined_visible.detach().float().pow(2.0).mean().sqrt().item())
    for edge_idx, edge_name in enumerate(edge_names):
        edge_slice = slice(edge_idx * 4, edge_idx * 4 + 4)
        edge_pre = seam_strip[:, edge_slice]
        edge_post = seam_visible[:, edge_slice]
        diagnostics[f"seam_edge_{edge_name}_defined"] = float(edge_defined_flags[:, edge_idx].mean().detach().item())
        diagnostics[f"seam_edge_{edge_name}_pre_gate_l2"] = float(edge_pre.detach().float().pow(2.0).mean().sqrt().item())
        diagnostics[f"seam_edge_{edge_name}_post_gate_l2"] = float(edge_post.detach().float().pow(2.0).mean().sqrt().item())
        diagnostics[f"seam_edge_{edge_name}_visible_l2"] = float(edge_post.detach().float().pow(2.0).mean().sqrt().item())
    return full_conditioning, dataset.full_conditioning_channel_names, diagnostics


def apply_runtime_material_lora(
    text_encoders: List[torch.nn.Module],
    unet: torch.nn.Module,
    control_net: torch.nn.Module,
    material_lora_path: str,
    multiplier: float,
):
    unet_network, weights_sd = lora_network.create_network_from_weights(
        multiplier,
        material_lora_path,
        None,
        text_encoders,
        unet,
        None,
        True,
    )
    # Keep runtime style attachment in spatial pathways only for this phase.
    unet_network.apply_to(text_encoders, unet, apply_text_encoder=False, apply_unet=True)
    unet_network.load_state_dict(weights_sd, strict=False)
    unet_network.requires_grad_(False)
    unet_network.eval()

    control_network, _ = lora_network.create_network_from_weights(
        multiplier,
        material_lora_path,
        None,
        text_encoders,
        control_net,
        weights_sd,
        True,
    )
    control_network.apply_to(text_encoders, control_net, apply_text_encoder=False, apply_unet=True)
    control_network.load_state_dict(weights_sd, strict=False)
    control_network.requires_grad_(False)
    control_network.eval()

    return unet_network, control_network


def prepare_text_conditioning(
    prompt: str,
    prompt2: str,
    tokenize_strategy: strategy_sdxl.SdxlTokenizeStrategy,
    text_encoding_strategy: strategy_sdxl.SdxlTextEncodingStrategy,
    text_encoders: List[torch.nn.Module],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_batch = [prompt] if isinstance(prompt, str) else list(prompt)
    input_ids1, input_ids2 = tokenize_strategy.tokenize(prompt_batch)
    encoder_hidden_states1, encoder_hidden_states2, pool2 = text_encoding_strategy.encode_tokens(
        tokenize_strategy,
        [text_encoders[0], text_encoders[1], text_encoders[1]],
        [input_ids1.to(device), input_ids2.to(device)],
    )
    del prompt2
    return (
        encoder_hidden_states1.to(device=device, dtype=dtype),
        encoder_hidden_states2.to(device=device, dtype=dtype),
        pool2.to(device=device, dtype=dtype),
    )


def build_size_embeddings(batch: Dict[str, object], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return sdxl_train_util.get_size_embeddings(
        batch["original_sizes_hw"],
        batch["crop_top_lefts"],
        batch["target_sizes_hw"],
        device,
    ).to(dtype)


def _pad_spatial_tensor(tensor: torch.Tensor, pad_px: int, mode: str = "constant", value: float = 0.0) -> torch.Tensor:
    pad = int(max(0, pad_px))
    if pad <= 0:
        return tensor
    if mode == "constant":
        return F.pad(tensor, (pad, pad, pad, pad), mode="constant", value=float(value))
    return F.pad(tensor, (pad, pad, pad, pad), mode=mode)


def _center_embed_spatial_tensor(tensor: torch.Tensor, halo_px: int, fill_value: float = 0.0) -> torch.Tensor:
    return shared_center_embed_spatial_tensor(tensor, halo_px, fill_value=fill_value)


def _linear_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 0:
        return end
    progress = min(max(step, 0), total_steps) / float(total_steps)
    return start + (end - start) * progress


def _find_channel_index(channel_names: List[str], target_name: str) -> int:
    if target_name not in channel_names:
        raise ValueError(f"required channel '{target_name}' not found in {channel_names}")
    return channel_names.index(target_name)


def _fill_channel_with_value_and_clamp(
    tensor: torch.Tensor,
    channel_index: int,
    value: float,
    channel_specs: List[SemanticChannelSpec],
) -> None:
    tensor[:, channel_index, :, :] = value
    if channel_index < len(channel_specs):
        vmin, vmax = channel_specs[channel_index].semantic_range
        tensor[:, channel_index, :, :] = tensor[:, channel_index, :, :].clamp(float(vmin), float(vmax))


def _build_extreme_contrast_conditioning(
    dataset: TerrainSemanticManifestDataset,
    conditioning: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List[int]]]:
    """Construct a geometry-only flat-vs-wall pair.

    Only terrain_mask / edge_* / openness_* channels are changed.
    All other channels remain bit-identical for strict isolation.
    """
    channel_specs = dataset.channel_specs
    channel_names = dataset.channel_names
    flat = conditioning.clone()
    wall = conditioning.clone()

    terrain_idx = _find_channel_index(channel_names, "terrain_mask")
    edge_indices = [idx for idx, spec in enumerate(channel_specs) if spec.atlas_name == "edge"]

    openness_indices = [
        idx
        for idx, spec in enumerate(channel_specs)
        if "open" in spec.name.lower() or "open" in spec.channel_name.lower()
    ]
    if not openness_indices:
        openness_indices = [
            idx
            for idx, spec in enumerate(channel_specs)
            if spec.atlas_name == "interior" and spec.channel_name in {"G", "B", "A"}
        ]

    _, _, _, width = conditioning.shape
    split_col = width // 2
    boundary_start = max(0, split_col - 1)
    boundary_end = min(width, split_col + 1)

    # terrain_mask is geometry-critical: flat is constant, wall has sharp left/right split.
    _fill_channel_with_value_and_clamp(flat, terrain_idx, 0.5, channel_specs)
    wall[:, terrain_idx, :, :split_col] = 0.0
    wall[:, terrain_idx, :, split_col:] = 1.0
    if terrain_idx < len(channel_specs):
        vmin, vmax = channel_specs[terrain_idx].semantic_range
        wall[:, terrain_idx, :, :] = wall[:, terrain_idx, :, :].clamp(float(vmin), float(vmax))

    # edge channels are zero in flat and only active on the wall boundary.
    for idx in edge_indices:
        _fill_channel_with_value_and_clamp(flat, idx, 0.0, channel_specs)
        wall[:, idx, :, :] = 0.0
        signed = idx < len(channel_specs) and channel_specs[idx].semantic_range[0] < 0.0
        if signed:
            wall[:, idx, :, :split_col] = -1.0
            wall[:, idx, :, split_col:] = 1.0
        else:
            wall[:, idx, :, boundary_start:boundary_end] = 1.0
        if idx < len(channel_specs):
            vmin, vmax = channel_specs[idx].semantic_range
            wall[:, idx, :, :] = wall[:, idx, :, :].clamp(float(vmin), float(vmax))

    # openness channels are constant in flat and directional in wall.
    for idx in openness_indices:
        _fill_channel_with_value_and_clamp(flat, idx, 0.5, channel_specs)
        signed = idx < len(channel_specs) and channel_specs[idx].semantic_range[0] < 0.0
        if signed:
            wall[:, idx, :, :split_col] = 1.0
            wall[:, idx, :, split_col:] = -1.0
        else:
            wall[:, idx, :, :split_col] = 1.0
            wall[:, idx, :, split_col:] = 0.0
        if idx < len(channel_specs):
            vmin, vmax = channel_specs[idx].semantic_range
            wall[:, idx, :, :] = wall[:, idx, :, :].clamp(float(vmin), float(vmax))

    return flat, wall, {
        "terrain_mask": [terrain_idx],
        "edge": edge_indices,
        "openness": openness_indices,
    }


def _is_targeted_c_sample(
    assigned_crop_class: str,
    special_structure_tags: str,
    targeted_classes: List[str],
) -> bool:
    cls = (assigned_crop_class or "").strip().lower()
    tags = (special_structure_tags or "").strip().lower()
    if cls and cls in targeted_classes:
        return True
    if "ceiling" in tags and "ceiling" in targeted_classes:
        return True
    return False


def _gaussian_blur5x5(x: torch.Tensor) -> torch.Tensor:
    """Lightweight gaussian low-pass used for haze suppression."""
    if x.ndim != 4:
        return x
    dtype = x.dtype
    device = x.device
    k1 = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=dtype)
    k2 = torch.outer(k1, k1)
    k2 = (k2 / k2.sum()).view(1, 1, 5, 5)
    weight = k2.repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, weight, stride=1, padding=2, groups=x.shape[1])


def _build_corrupted_geometry_conditioning(
    dataset: TerrainSemanticManifestDataset,
    conditioning: torch.Tensor,
    mode: str,
    rng: random.Random,
    alpha_config: Optional[Dict[str, object]] = None,
    assigned_crop_class: Optional[List[str]] = None,
    special_structure_tags: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, str, float]:
    """Build plausible-but-wrong geometry conditioning while preserving non-geometry channels.

    Returns a corrupted conditioning tensor and a mode string.
    """
    flat, wall, groups = _build_extreme_contrast_conditioning(dataset, conditioning)
    terrain_indices = groups.get("terrain_mask", [])
    edge_indices = groups.get("edge", [])
    openness_indices = groups.get("openness", [])
    non_mask_indices = edge_indices + openness_indices
    geom_indices = terrain_indices + non_mask_indices
    if not geom_indices:
        return conditioning.clone(), "none", 0.0

    base_geom = conditioning[:, geom_indices]
    flat_geom = flat[:, geom_indices]
    wall_geom = wall[:, geom_indices]
    dist_to_flat = torch.mean((base_geom - flat_geom) ** 2).item()
    dist_to_wall = torch.mean((base_geom - wall_geom) ** 2).item()
    opposite = wall if dist_to_flat <= dist_to_wall else flat

    corrupted = conditioning.clone()
    if mode == "flat_to_wall_injection" and non_mask_indices:
        indices_to_replace = non_mask_indices
        realized_mode = "flat_to_wall_injection"
    elif mode == "local_spatial_misalignment" and non_mask_indices:
        indices_to_replace = non_mask_indices
        realized_mode = "local_spatial_misalignment"
    else:
        indices_to_replace = geom_indices
        realized_mode = "wall_to_flat_suppression"

    if realized_mode == "local_spatial_misalignment":
        shift_max = max(2, int(min(conditioning.shape[-2], conditioning.shape[-1]) * 0.06))
        shift_x = rng.randint(1, shift_max) * (-1 if rng.random() < 0.5 else 1)
        shift_y = rng.randint(1, shift_max) * (-1 if rng.random() < 0.5 else 1)
        base_selected = conditioning[:, indices_to_replace, :, :]
        shifted = torch.roll(base_selected, shifts=(shift_y, shift_x), dims=(2, 3))

        cfg = alpha_config or {}
        directional_strength = float(cfg.get("bind_c_directional_rejection_strength", 0.14))
        polarity_strength = float(cfg.get("bind_c_signed_polarity_strength", 0.18))
        haze_suppression = float(cfg.get("bind_c_interior_haze_suppression", 0.12))
        transition_softness = float(cfg.get("bind_c_transition_softness", 0.12))
        gate_threshold = max(1e-6, float(cfg.get("bind_c_directional_gate_threshold", 0.08)))
        targeted_classes = [str(v).strip().lower() for v in cfg.get("bind_c_targeted_classes", ["mixed", "ceiling"]) or []]
        if not targeted_classes:
            targeted_classes = ["mixed", "ceiling"]

        bsz = conditioning.shape[0]
        if assigned_crop_class is None:
            assigned_crop_class = [""] * bsz
        if special_structure_tags is None:
            special_structure_tags = [""] * bsz
        target_mask = torch.zeros((bsz, 1, 1, 1), device=conditioning.device, dtype=conditioning.dtype)
        for b in range(bsz):
            if _is_targeted_c_sample(
                assigned_crop_class[b] if b < len(assigned_crop_class) else "",
                special_structure_tags[b] if b < len(special_structure_tags) else "",
                targeted_classes,
            ):
                target_mask[b, 0, 0, 0] = 1.0

        # Directional residual: normalize and gate low-signal regions to avoid unstable directions.
        residual = shifted - base_selected
        eps = 1e-6
        magnitude = torch.abs(residual)
        r_dir_raw = residual / (magnitude + eps)
        gate_w = (magnitude / gate_threshold).clamp(0.0, 1.0)
        r_dir = gate_w * r_dir_raw

        anti = shifted - (directional_strength * r_dir)

        # Signed channels only: optional polarity blend toward meaningful opposite references.
        if polarity_strength > 0.0:
            opposite_selected = opposite[:, indices_to_replace, :, :]
            for local_idx, channel_idx in enumerate(indices_to_replace):
                if channel_idx < len(dataset.channel_specs) and dataset.channel_specs[channel_idx].semantic_range[0] < 0.0:
                    anti[:, local_idx : local_idx + 1, :, :] = (
                        (1.0 - polarity_strength) * anti[:, local_idx : local_idx + 1, :, :]
                        + polarity_strength * opposite_selected[:, local_idx : local_idx + 1, :, :]
                    )

        # Remove only low-frequency interior attenuation to reduce haze in openness channels.
        if haze_suppression > 0.0 and openness_indices:
            for openness_idx in openness_indices:
                if openness_idx not in indices_to_replace:
                    continue
                local_idx = indices_to_replace.index(openness_idx)
                channel = anti[:, local_idx : local_idx + 1, :, :]
                low_freq = _gaussian_blur5x5(channel)
                anti[:, local_idx : local_idx + 1, :, :] = channel - (haze_suppression * low_freq)

        # Keep non-target classes near baseline shift behavior.
        shifted = ((1.0 - target_mask) * shifted) + (target_mask * anti)

        terrain_idx = terrain_indices[0] if terrain_indices else 0
        terrain = conditioning[:, terrain_idx : terrain_idx + 1, :, :]
        roi = (terrain > 0.5).float()
        if float(roi.mean().item()) <= 0.0:
            return conditioning.clone(), "none", 0.0
        if transition_softness > 0.0:
            roi_blur = _gaussian_blur5x5(roi)
            roi = ((1.0 - transition_softness) * roi) + (transition_softness * roi_blur)
            roi = roi.clamp(0.0, 1.0)
        replacement = (
            (1.0 - roi) * base_selected
            + roi * shifted
        )
        corrupted[:, indices_to_replace, :, :] = replacement.to(corrupted.dtype)
    else:
        corrupted[:, indices_to_replace, :, :] = opposite[:, indices_to_replace, :, :].to(corrupted.dtype)

    if terrain_indices and non_mask_indices:
        terrain_idx = terrain_indices[0]
        terrain = conditioning[:, terrain_idx : terrain_idx + 1, :, :]
        wall_zone = terrain > 0.6
        flat_zone = terrain < 0.4
        for idx in non_mask_indices:
            channel = corrupted[:, idx : idx + 1, :, :]
            channel = torch.where(wall_zone, torch.zeros_like(channel), channel)
            channel = torch.where(flat_zone, opposite[:, idx : idx + 1, :, :], channel)
            if idx < len(dataset.channel_specs):
                vmin, vmax = dataset.channel_specs[idx].semantic_range
                channel = channel.clamp(float(vmin), float(vmax))
            corrupted[:, idx : idx + 1, :, :] = channel

    delta = torch.mean(torch.abs(corrupted[:, indices_to_replace, :, :] - conditioning[:, indices_to_replace, :, :])).item()
    return corrupted, realized_mode, float(delta)


def _expand_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel_size = (radius * 2) + 1
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=radius)


def _build_edge_band(mask: torch.Tensor, dilation_radius: int) -> torch.Tensor:
    grad_x = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1])
    grad_y = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :])
    grad_x = F.pad(grad_x, (0, 1, 0, 0))
    grad_y = F.pad(grad_y, (0, 0, 0, 1))
    edge = torch.maximum(grad_x, grad_y).clamp(0.0, 1.0)
    return _expand_mask(edge, dilation_radius).clamp(0.0, 1.0)


def _masked_mean_per_sample(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_sum = (values * mask).sum(dim=(1, 2, 3))
    masked_area = mask.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    return masked_sum / masked_area


def _summarize_weighted_seam_regions(
    error_map: torch.Tensor,
    seam_maps: Dict[str, torch.Tensor],
    region_weights: Dict[str, float],
    normalize_region_losses: bool,
    continuation_weighted_mask: Optional[torch.Tensor] = None,
    seam_config: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    if continuation_weighted_mask is not None:
        zero = error_map.new_tensor(0.0)

        def _mask_or_zeros(*keys: str) -> torch.Tensor:
            for key in keys:
                value = seam_maps.get(key)
                if isinstance(value, torch.Tensor):
                    return value.float()
            return torch.zeros_like(error_map, dtype=error_map.dtype)

        def _masked_mean(mask: torch.Tensor) -> torch.Tensor:
            area = mask.sum()
            if float(area.detach().item()) <= 0.0:
                return zero
            return (error_map * mask).sum() / area.clamp_min(1e-6)

        def _weighted_mean(mask: torch.Tensor) -> torch.Tensor:
            weight_sum = mask.sum()
            if float(weight_sum.detach().item()) <= 0.0:
                return zero
            return (error_map * mask).sum() / weight_sum.clamp_min(1e-6)

        def _resolve_continuation_rgb_weight_mask(continuation_binary_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, str]:
            cfg = seam_config or {}
            mode = str(cfg.get("continuation_rgb_weight_mode", "falloff") or "falloff").strip().lower()
            distance_map = seam_maps.get("continuation_distance_px")
            if not isinstance(distance_map, torch.Tensor):
                distance_map = torch.zeros_like(continuation_binary_mask, dtype=error_map.dtype)
            else:
                distance_map = distance_map.float()

            if mode == "uniform":
                return continuation_binary_mask.float(), distance_map, mode
            if mode == "linear":
                band_px = max(1.0, float(cfg.get("continuation_width_px", 1.0)))
                linear_mask = (1.0 - (distance_map / band_px)).clamp(0.0, 1.0) * continuation_binary_mask.float()
                return linear_mask, distance_map, mode
            return continuation_weighted_mask.float(), distance_map, "falloff"

        def _weight_fraction(mask: torch.Tensor, distance_map: torch.Tensor, upper_px: float) -> torch.Tensor:
            total_weight = mask.sum()
            if float(total_weight.detach().item()) <= 0.0:
                return zero
            selected = mask * (distance_map < upper_px).to(dtype=mask.dtype)
            return selected.sum() / total_weight.clamp_min(1e-6)

        def _weight_sum_below(mask: torch.Tensor, distance_map: torch.Tensor, upper_px: float) -> torch.Tensor:
            return (mask * (distance_map < upper_px).to(dtype=mask.dtype)).sum()

        def _bin_rgb_loss(continuation_binary_mask: torch.Tensor, distance_map: torch.Tensor, start_px: float, end_px: float) -> torch.Tensor:
            bin_mask = continuation_binary_mask * ((distance_map >= start_px) & (distance_map < end_px)).to(dtype=continuation_binary_mask.dtype)
            return _masked_mean(bin_mask)

        def _bin_rgb_raw_sum(continuation_binary_mask: torch.Tensor, distance_map: torch.Tensor, start_px: float, end_px: float) -> torch.Tensor:
            bin_mask = continuation_binary_mask * ((distance_map >= start_px) & (distance_map < end_px)).to(dtype=continuation_binary_mask.dtype)
            return (error_map * bin_mask).sum()

        def _bin_rgb_area(continuation_binary_mask: torch.Tensor, distance_map: torch.Tensor, start_px: float, end_px: float) -> torch.Tensor:
            bin_mask = continuation_binary_mask * ((distance_map >= start_px) & (distance_map < end_px)).to(dtype=continuation_binary_mask.dtype)
            return bin_mask.sum()

        halo_inner_mask = _mask_or_zeros("halo_inner", "margin_inner")
        halo_outer_mask = _mask_or_zeros("halo_outer", "margin_outer")
        continuation_binary_mask = _mask_or_zeros("interior_continuation", "interior_inner")
        interior_core_mask = _mask_or_zeros("interior_core")
        interior_union_mask = torch.maximum(interior_core_mask, continuation_binary_mask)
        continuation_rgb_weight_mask, continuation_distance_px_map, continuation_rgb_weight_mode = _resolve_continuation_rgb_weight_mask(continuation_binary_mask)

        halo_inner_loss = _masked_mean(halo_inner_mask)
        halo_outer_loss = _masked_mean(halo_outer_mask)
        continuation_raw_loss = _masked_mean(continuation_binary_mask)
        continuation_weighted_loss = _weighted_mean(continuation_rgb_weight_mask.float()) * float(
            region_weights.get("interior_continuation", region_weights.get("continuation_distance_weighted", 1.0))
        )
        interior_core_loss = _masked_mean(interior_core_mask)

        continuation_px = continuation_binary_mask.sum()
        continuation_weight_sum = continuation_rgb_weight_mask.float().sum()
        if float(continuation_px.detach().item()) > 0.0:
            continuation_effective_weight_mean = continuation_weight_sum / continuation_px.clamp_min(1e-6)
        else:
            continuation_effective_weight_mean = zero
        continuation_effective_weight_max = continuation_rgb_weight_mask.float().max() if continuation_rgb_weight_mask.numel() > 0 else zero
        continuation_weight_fraction_first_8px = _weight_fraction(continuation_rgb_weight_mask.float(), continuation_distance_px_map, 8.0)
        continuation_weight_fraction_first_16px = _weight_fraction(continuation_rgb_weight_mask.float(), continuation_distance_px_map, 16.0)
        continuation_weight_fraction_first_24px = _weight_fraction(continuation_rgb_weight_mask.float(), continuation_distance_px_map, 24.0)
        continuation_weight_fraction_full_band = _weight_fraction(
            continuation_rgb_weight_mask.float(),
            continuation_distance_px_map,
            max(1.0, float((seam_config or {}).get("continuation_width_px", 1.0))) + 1e-6,
        )
        continuation_rgb_loss_bin_0_8 = _bin_rgb_loss(continuation_binary_mask, continuation_distance_px_map, 0.0, 8.0)
        continuation_rgb_loss_bin_8_16 = _bin_rgb_loss(continuation_binary_mask, continuation_distance_px_map, 8.0, 16.0)
        continuation_rgb_loss_bin_16_24 = _bin_rgb_loss(continuation_binary_mask, continuation_distance_px_map, 16.0, 24.0)
        continuation_rgb_loss_bin_24_32 = _bin_rgb_loss(continuation_binary_mask, continuation_distance_px_map, 24.0, 32.0)
        continuation_rgb_loss_bin_32_40 = _bin_rgb_loss(continuation_binary_mask, continuation_distance_px_map, 32.0, 40.0)
        continuation_rgb_loss_bin_40_48 = _bin_rgb_loss(continuation_binary_mask, continuation_distance_px_map, 40.0, 48.0)
        continuation_weighted_raw_sum = (error_map * continuation_rgb_weight_mask.float()).sum()
        continuation_weight_sum_first_8px = _weight_sum_below(continuation_rgb_weight_mask.float(), continuation_distance_px_map, 8.0)
        continuation_weight_sum_first_16px = _weight_sum_below(continuation_rgb_weight_mask.float(), continuation_distance_px_map, 16.0)
        continuation_weight_sum_first_24px = _weight_sum_below(continuation_rgb_weight_mask.float(), continuation_distance_px_map, 24.0)
        continuation_rgb_loss_bin_raw_sum_0_8 = _bin_rgb_raw_sum(continuation_binary_mask, continuation_distance_px_map, 0.0, 8.0)
        continuation_rgb_loss_bin_raw_sum_8_16 = _bin_rgb_raw_sum(continuation_binary_mask, continuation_distance_px_map, 8.0, 16.0)
        continuation_rgb_loss_bin_raw_sum_16_24 = _bin_rgb_raw_sum(continuation_binary_mask, continuation_distance_px_map, 16.0, 24.0)
        continuation_rgb_loss_bin_raw_sum_24_32 = _bin_rgb_raw_sum(continuation_binary_mask, continuation_distance_px_map, 24.0, 32.0)
        continuation_rgb_loss_bin_raw_sum_32_40 = _bin_rgb_raw_sum(continuation_binary_mask, continuation_distance_px_map, 32.0, 40.0)
        continuation_rgb_loss_bin_raw_sum_40_48 = _bin_rgb_raw_sum(continuation_binary_mask, continuation_distance_px_map, 40.0, 48.0)
        continuation_rgb_bin_area_0_8 = _bin_rgb_area(continuation_binary_mask, continuation_distance_px_map, 0.0, 8.0)
        continuation_rgb_bin_area_8_16 = _bin_rgb_area(continuation_binary_mask, continuation_distance_px_map, 8.0, 16.0)
        continuation_rgb_bin_area_16_24 = _bin_rgb_area(continuation_binary_mask, continuation_distance_px_map, 16.0, 24.0)
        continuation_rgb_bin_area_24_32 = _bin_rgb_area(continuation_binary_mask, continuation_distance_px_map, 24.0, 32.0)
        continuation_rgb_bin_area_32_40 = _bin_rgb_area(continuation_binary_mask, continuation_distance_px_map, 32.0, 40.0)
        continuation_rgb_bin_area_40_48 = _bin_rgb_area(continuation_binary_mask, continuation_distance_px_map, 40.0, 48.0)

        halo_inner_contribution = halo_inner_loss * float(region_weights.get("halo_inner", 1.0))
        halo_outer_contribution = halo_outer_loss * float(region_weights.get("halo_outer", 1.0))
        interior_core_contribution = interior_core_loss * float(region_weights.get("interior_core", 1.0))

        region_losses = {
            "halo_inner": halo_inner_loss,
            "halo_outer": halo_outer_loss,
            "interior_continuation": continuation_raw_loss,
            "interior_core": interior_core_loss,
            "margin_inner": halo_inner_loss,
            "margin_outer": halo_outer_loss,
            "interior_inner": continuation_raw_loss,
            "interior_outer": interior_core_loss,
        }
        region_weighted_contributions = {
            "halo_inner": halo_inner_contribution,
            "halo_outer": halo_outer_contribution,
            "interior_continuation": continuation_weighted_loss,
            "continuation_distance_weighted": continuation_weighted_loss,
            "interior_core": interior_core_contribution,
            "margin_inner": halo_inner_contribution,
            "margin_outer": halo_outer_contribution,
            "interior_inner": continuation_weighted_loss,
            "interior_outer": interior_core_contribution,
        }
        halo_raw_sum = (error_map * halo_inner_mask).sum() + (error_map * halo_outer_mask).sum()
        interior_raw_sum = (error_map * interior_union_mask).sum()
        halo_supervised_px = halo_inner_mask.sum() + halo_outer_mask.sum()
        interior_supervised_px = interior_union_mask.sum()

        return {
            "region_losses": region_losses,
            "halo_loss_raw": halo_raw_sum,
            "interior_loss_raw": interior_raw_sum,
            "halo_supervised_px": halo_supervised_px,
            "interior_supervised_px": interior_supervised_px,
            "halo_loss_weighted": halo_inner_contribution + halo_outer_contribution,
            "interior_loss_weighted": continuation_weighted_loss + interior_core_contribution,
            "total_loss": halo_inner_contribution + halo_outer_contribution + continuation_weighted_loss + interior_core_contribution,
            "region_weighted_contributions": region_weighted_contributions,
            "continuation_loss_raw": continuation_raw_loss,
            "continuation_loss_weighted": continuation_weighted_loss,
            "continuation_supervised_px": continuation_px,
            "continuation_weight_sum": continuation_weight_sum,
            "continuation_effective_weight_mean": continuation_effective_weight_mean,
            "continuation_effective_weight_max": continuation_effective_weight_max,
            "continuation_weight_fraction_first_8px": continuation_weight_fraction_first_8px,
            "continuation_weight_fraction_first_16px": continuation_weight_fraction_first_16px,
            "continuation_weight_fraction_first_24px": continuation_weight_fraction_first_24px,
            "continuation_weight_fraction_full_band": continuation_weight_fraction_full_band,
            "continuation_rgb_weight_mode": continuation_rgb_weight_mode,
            "continuation_rgb_loss_bin_0_8": continuation_rgb_loss_bin_0_8,
            "continuation_rgb_loss_bin_8_16": continuation_rgb_loss_bin_8_16,
            "continuation_rgb_loss_bin_16_24": continuation_rgb_loss_bin_16_24,
            "continuation_rgb_loss_bin_24_32": continuation_rgb_loss_bin_24_32,
            "continuation_rgb_loss_bin_32_40": continuation_rgb_loss_bin_32_40,
            "continuation_rgb_loss_bin_40_48": continuation_rgb_loss_bin_40_48,
            "interior_core_supervised_px": interior_core_mask.sum(),
            "region_raw_sums": {
                "halo_inner": (error_map * halo_inner_mask).sum(),
                "halo_outer": (error_map * halo_outer_mask).sum(),
                "interior_continuation": (error_map * continuation_binary_mask).sum(),
                "interior_core": (error_map * interior_core_mask).sum(),
                "margin_inner": (error_map * halo_inner_mask).sum(),
                "margin_outer": (error_map * halo_outer_mask).sum(),
                "interior_inner": (error_map * continuation_binary_mask).sum(),
                "interior_outer": (error_map * interior_core_mask).sum(),
            },
            "region_areas": {
                "halo_inner": halo_inner_mask.sum(),
                "halo_outer": halo_outer_mask.sum(),
                "interior_continuation": continuation_binary_mask.sum(),
                "interior_core": interior_core_mask.sum(),
                "margin_inner": halo_inner_mask.sum(),
                "margin_outer": halo_outer_mask.sum(),
                "interior_inner": continuation_binary_mask.sum(),
                "interior_outer": interior_core_mask.sum(),
            },
            "continuation_weighted_raw_sum": continuation_weighted_raw_sum,
            "continuation_weight_sum_first_8px": continuation_weight_sum_first_8px,
            "continuation_weight_sum_first_16px": continuation_weight_sum_first_16px,
            "continuation_weight_sum_first_24px": continuation_weight_sum_first_24px,
            "continuation_rgb_loss_bin_raw_sum_0_8": continuation_rgb_loss_bin_raw_sum_0_8,
            "continuation_rgb_loss_bin_raw_sum_8_16": continuation_rgb_loss_bin_raw_sum_8_16,
            "continuation_rgb_loss_bin_raw_sum_16_24": continuation_rgb_loss_bin_raw_sum_16_24,
            "continuation_rgb_loss_bin_raw_sum_24_32": continuation_rgb_loss_bin_raw_sum_24_32,
            "continuation_rgb_loss_bin_raw_sum_32_40": continuation_rgb_loss_bin_raw_sum_32_40,
            "continuation_rgb_loss_bin_raw_sum_40_48": continuation_rgb_loss_bin_raw_sum_40_48,
            "continuation_rgb_bin_area_0_8": continuation_rgb_bin_area_0_8,
            "continuation_rgb_bin_area_8_16": continuation_rgb_bin_area_8_16,
            "continuation_rgb_bin_area_16_24": continuation_rgb_bin_area_16_24,
            "continuation_rgb_bin_area_24_32": continuation_rgb_bin_area_24_32,
            "continuation_rgb_bin_area_32_40": continuation_rgb_bin_area_32_40,
            "continuation_rgb_bin_area_40_48": continuation_rgb_bin_area_40_48,
        }

    halo_regions = ("margin_inner", "margin_outer")
    interior_regions = ("interior_inner", "interior_outer")
    zero = error_map.new_tensor(0.0)

    region_losses: Dict[str, torch.Tensor] = {}
    region_raw_sums: Dict[str, torch.Tensor] = {}
    region_areas: Dict[str, torch.Tensor] = {}
    for region_name, region_mask in seam_maps.items():
        area = region_mask.sum()
        if float(area.detach().item()) <= 0.0:
            continue
        raw_sum = (error_map * region_mask).sum()
        region_raw_sums[region_name] = raw_sum
        region_areas[region_name] = area
        region_losses[region_name] = raw_sum / area.clamp_min(1e-6)

    def _sum_named(source: Dict[str, torch.Tensor], names: Tuple[str, ...]) -> torch.Tensor:
        values = [source[name] for name in names if name in source]
        if not values:
            return zero
        return torch.stack(values).sum()

    def _sum_named_weights(names: Tuple[str, ...]) -> torch.Tensor:
        weights = [zero.new_tensor(float(region_weights[name])) for name in names if name in region_losses and name in region_weights]
        if not weights:
            return zero
        return torch.stack(weights).sum()

    weighted_region_terms_raw = {
        name: region_losses[name] * float(region_weights.get(name, 1.0))
        for name in region_losses
    }
    region_weighted_contributions: Dict[str, torch.Tensor] = {}
    halo_weight_den = _sum_named_weights(halo_regions).clamp_min(1e-6)
    interior_weight_den = _sum_named_weights(interior_regions).clamp_min(1e-6)
    for region_name, raw_term in weighted_region_terms_raw.items():
        if normalize_region_losses:
            if region_name in halo_regions:
                region_weighted_contributions[region_name] = raw_term / halo_weight_den
            elif region_name in interior_regions:
                region_weighted_contributions[region_name] = raw_term / interior_weight_den
            else:
                region_weighted_contributions[region_name] = raw_term
        else:
            region_weighted_contributions[region_name] = raw_term

    halo_weighted = _sum_named(region_weighted_contributions, halo_regions)
    interior_weighted = _sum_named(region_weighted_contributions, interior_regions)

    return {
        "region_losses": region_losses,
        "halo_loss_raw": _sum_named(region_raw_sums, halo_regions),
        "interior_loss_raw": _sum_named(region_raw_sums, interior_regions),
        "halo_supervised_px": _sum_named(region_areas, halo_regions),
        "interior_supervised_px": _sum_named(region_areas, interior_regions),
        "halo_loss_weighted": halo_weighted,
        "interior_loss_weighted": interior_weighted,
        "total_loss": halo_weighted + interior_weighted,
        "region_weighted_contributions": region_weighted_contributions,
    }


_SOBEL_X_KERNEL = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
).view(1, 1, 3, 3)
_SOBEL_Y_KERNEL = torch.tensor(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
).view(1, 1, 3, 3)


def _compute_masked_rgb_gradient_loss(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    weight_mask: torch.Tensor,
    sobel_radius_px: int = 1,
) -> Dict[str, torch.Tensor]:
    if pred_rgb.ndim != 4 or target_rgb.ndim != 4 or pred_rgb.shape[1] != 3 or target_rgb.shape[1] != 3:
        raise ValueError(
            "masked RGB gradient loss expects RGB tensors with shape [batch, 3, height, width]: "
            + f"pred={tuple(pred_rgb.shape)} target={tuple(target_rgb.shape)}"
        )
    if int(sobel_radius_px) != 1:
        logger.warning("[seam/gradient] only 3x3 Sobel is implemented; requested radius=%d, using 1", int(sobel_radius_px))

    sobel_x = _SOBEL_X_KERNEL.to(device=pred_rgb.device, dtype=pred_rgb.dtype).expand(3, 1, 3, 3)
    sobel_y = _SOBEL_Y_KERNEL.to(device=pred_rgb.device, dtype=pred_rgb.dtype).expand(3, 1, 3, 3)

    pred_grad_x = F.conv2d(pred_rgb, sobel_x, padding=1, groups=3)
    pred_grad_y = F.conv2d(pred_rgb, sobel_y, padding=1, groups=3)
    target_grad_x = F.conv2d(target_rgb, sobel_x, padding=1, groups=3)
    target_grad_y = F.conv2d(target_rgb, sobel_y, padding=1, groups=3)

    pred_grad_mag = torch.sqrt(pred_grad_x.square() + pred_grad_y.square() + 1e-6)
    target_grad_mag = torch.sqrt(target_grad_x.square() + target_grad_y.square() + 1e-6)
    grad_diff = (pred_grad_mag - target_grad_mag).abs().mean(dim=1, keepdim=True)

    active_mask = (weight_mask > 0.0).to(dtype=grad_diff.dtype)
    active_px = active_mask.sum()
    weight_sum = weight_mask.sum()
    raw_loss = grad_diff.new_tensor(0.0)
    weighted_loss = grad_diff.new_tensor(0.0)
    raw_sum = grad_diff.new_tensor(0.0)
    weighted_raw_sum = grad_diff.new_tensor(0.0)
    if float(active_px.detach().item()) > 0.0:
        raw_sum = (grad_diff * active_mask).sum()
        raw_loss = raw_sum / active_px.clamp_min(1e-6)
    if float(weight_sum.detach().item()) > 0.0:
        weighted_raw_sum = (grad_diff * weight_mask).sum()
        weighted_loss = weighted_raw_sum / weight_sum.clamp_min(1e-6)

    return {
        "raw_loss": raw_loss,
        "weighted_loss": weighted_loss,
        "active_px": active_px,
        "weight_sum": weight_sum,
        "raw_sum": raw_sum,
        "weighted_raw_sum": weighted_raw_sum,
    }


def _compute_continuation_gradient_loss(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    continuation_distance_weighted_mask: torch.Tensor,
    sobel_radius_px: int = 1,
) -> Dict[str, torch.Tensor]:
    return _compute_masked_rgb_gradient_loss(
        pred_rgb=pred_rgb,
        target_rgb=target_rgb,
        weight_mask=continuation_distance_weighted_mask,
        sobel_radius_px=sobel_radius_px,
    )


def _edge_halo_distance_map(
    edge_name: str,
    *,
    height: int,
    width: int,
    crop_x0: int,
    crop_y0: int,
    full_height: int,
    full_width: int,
    expanded_halo_px: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    yy = torch.arange(height, device=device, dtype=dtype).view(1, 1, height, 1) + float(crop_y0)
    xx = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width) + float(crop_x0)
    halo = float(max(0, int(expanded_halo_px)))
    if edge_name == "north":
        return (halo - yy).clamp(min=0.0).expand(1, 1, height, width)
    if edge_name == "south":
        boundary = float(full_height - 1) - halo
        return (yy - boundary).clamp(min=0.0).expand(1, 1, height, width)
    if edge_name == "east":
        boundary = float(full_width - 1) - halo
        return (xx - boundary).clamp(min=0.0).expand(1, 1, height, width)
    if edge_name == "west":
        return (halo - xx).clamp(min=0.0).expand(1, 1, height, width)
    raise ValueError(f"unsupported seam edge for halo distance map: {edge_name}")


def _masked_map_stats(value_map: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    zero = value_map.new_tensor(0.0)
    active_mask = (mask > 0.0).to(dtype=value_map.dtype)
    area = active_mask.sum()
    raw_sum = (value_map * active_mask).sum()
    mean = zero
    max_value = zero
    if float(area.detach().item()) > 0.0:
        mean = raw_sum / area.clamp_min(1e-6)
        max_value = value_map.masked_fill(active_mask <= 0.0, 0.0).max()
    return {
        "raw_sum": raw_sum,
        "area": area,
        "mean": mean,
        "max": max_value,
    }


def _compute_edge_halo_copy_metrics(
    error_map: torch.Tensor,
    *,
    edge_name: str,
    edge_seam_maps: Dict[str, object],
    crop_x0: int,
    crop_y0: int,
    full_height: int,
    full_width: int,
    expanded_halo_px: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    zero = error_map.new_tensor(0.0)
    if int(expanded_halo_px) <= 0:
        return {
            "halo_copy": {"raw_sum": zero, "area": zero, "mean": zero, "max": zero},
            "halo_inner_1px": {"raw_sum": zero, "area": zero, "mean": zero, "max": zero},
            "halo_inner_4px": {"raw_sum": zero, "area": zero, "mean": zero, "max": zero},
            "halo_inner_8px": {"raw_sum": zero, "area": zero, "mean": zero, "max": zero},
            "halo_inner_16px": {"raw_sum": zero, "area": zero, "mean": zero, "max": zero},
        }

    halo_inner = edge_seam_maps.get("margin_inner")
    halo_outer = edge_seam_maps.get("margin_outer")
    if not isinstance(halo_inner, torch.Tensor) or not isinstance(halo_outer, torch.Tensor):
        return {
            "halo_copy": {"raw_sum": zero, "area": zero, "mean": zero, "max": zero},
            "halo_inner_1px": {"raw_sum": zero, "area": zero, "mean": zero, "max": zero},
            "halo_inner_4px": {"raw_sum": zero, "area": zero, "mean": zero, "max": zero},
            "halo_inner_8px": {"raw_sum": zero, "area": zero, "mean": zero, "max": zero},
            "halo_inner_16px": {"raw_sum": zero, "area": zero, "mean": zero, "max": zero},
        }

    distance_map = _edge_halo_distance_map(
        edge_name,
        height=int(error_map.shape[-2]),
        width=int(error_map.shape[-1]),
        crop_x0=crop_x0,
        crop_y0=crop_y0,
        full_height=full_height,
        full_width=full_width,
        expanded_halo_px=expanded_halo_px,
        device=error_map.device,
        dtype=error_map.dtype,
    )
    halo_mask = (halo_inner + halo_outer).clamp(0.0, 1.0)

    def _ring_stats(limit_px: float) -> Dict[str, torch.Tensor]:
        ring_mask = halo_inner * ((distance_map > 0.0) & (distance_map <= limit_px)).to(dtype=error_map.dtype)
        return _masked_map_stats(error_map, ring_mask)

    return {
        "halo_copy": _masked_map_stats(error_map, halo_mask),
        "halo_inner_1px": _ring_stats(1.0),
        "halo_inner_4px": _ring_stats(4.0),
        "halo_inner_8px": _ring_stats(8.0),
        "halo_inner_16px": _ring_stats(16.0),
    }


def _build_seam_region_maps(
    edge_band_masks: torch.Tensor,
    seam_decay_maps: torch.Tensor,
    edge_defined_flags: torch.Tensor,
    seam_strip_width_px: torch.Tensor,
    supervision_mask: torch.Tensor,
    seam_config: Dict[str, object],
    expanded_halo_px: int = 0,
    continuation_valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    return shared_build_seam_region_maps(
        edge_band_masks=edge_band_masks,
        seam_decay_maps=seam_decay_maps,
        edge_defined_flags=edge_defined_flags,
        seam_strip_width_px=seam_strip_width_px,
        supervision_mask=supervision_mask,
        seam_config=seam_config,
        expanded_halo_px=expanded_halo_px,
        continuation_valid_mask=continuation_valid_mask,
    )


def _build_seam_supervision_mask(
    trusted_mask: torch.Tensor,
    edge_band_masks: torch.Tensor,
    edge_defined_flags: torch.Tensor,
    seam_config: Dict[str, object],
) -> torch.Tensor:
    return shared_build_seam_supervision_mask(trusted_mask, edge_band_masks, edge_defined_flags, seam_config)


def _mask_bounds(mask: torch.Tensor) -> Tuple[float, float, float, float]:
    if mask.ndim != 4:
        raise ValueError(f"mask bounds expect [batch, channels, height, width], got shape={tuple(mask.shape)}")
    active = (mask.detach().float() > 0.0).any(dim=1)
    batch_size, height, width = active.shape
    y_coords = torch.arange(height, device=mask.device, dtype=torch.float32).view(1, height, 1).expand(batch_size, height, width)
    x_coords = torch.arange(width, device=mask.device, dtype=torch.float32).view(1, 1, width).expand(batch_size, height, width)
    has_active = active.view(batch_size, -1).any(dim=1)
    if not bool(has_active.any().item()):
        return -1.0, -1.0, -1.0, -1.0

    inf = torch.full((batch_size, height, width), float("inf"), device=mask.device, dtype=torch.float32)
    neg_inf = torch.full((batch_size, height, width), float("-inf"), device=mask.device, dtype=torch.float32)
    min_x = torch.where(active, x_coords, inf).amin(dim=(1, 2))
    max_x = torch.where(active, x_coords, neg_inf).amax(dim=(1, 2))
    min_y = torch.where(active, y_coords, inf).amin(dim=(1, 2))
    max_y = torch.where(active, y_coords, neg_inf).amax(dim=(1, 2))

    valid = has_active.float().sum().clamp_min(1.0)
    return (
        float((min_x * has_active.float()).sum().item() / valid.item()),
        float((max_x * has_active.float()).sum().item() / valid.item()),
        float((min_y * has_active.float()).sum().item() / valid.item()),
        float((max_y * has_active.float()).sum().item() / valid.item()),
    )

def _compute_seam_decode_crop(
    support_mask: torch.Tensor,
    *,
    full_height: int,
    full_width: int,
    latent_height: int,
    latent_width: int,
    context_px: int = 64,
) -> Dict[str, int]:
    min_x, max_x, min_y, max_y = _mask_bounds(support_mask)
    if max_x < 0.0 or max_y < 0.0:
        return {
            "pixel_x0": 0,
            "pixel_x1": full_width,
            "pixel_y0": 0,
            "pixel_y1": full_height,
            "latent_x0": 0,
            "latent_x1": latent_width,
            "latent_y0": 0,
            "latent_y1": latent_height,
        }

    latent_scale_x = max(1, full_width // max(1, latent_width))
    latent_scale_y = max(1, full_height // max(1, latent_height))
    pad_x = max(0, int(context_px))
    pad_y = max(0, int(context_px))

    pixel_x0 = max(0, int(math.floor(min_x)) - pad_x)
    pixel_x1 = min(full_width, int(math.ceil(max_x + 1.0)) + pad_x)
    pixel_y0 = max(0, int(math.floor(min_y)) - pad_y)
    pixel_y1 = min(full_height, int(math.ceil(max_y + 1.0)) + pad_y)

    latent_x0 = max(0, pixel_x0 // latent_scale_x)
    latent_x1 = min(latent_width, int(math.ceil(pixel_x1 / float(latent_scale_x))))
    latent_y0 = max(0, pixel_y0 // latent_scale_y)
    latent_y1 = min(latent_height, int(math.ceil(pixel_y1 / float(latent_scale_y))))

    pixel_x0 = latent_x0 * latent_scale_x
    pixel_x1 = min(full_width, latent_x1 * latent_scale_x)
    pixel_y0 = latent_y0 * latent_scale_y
    pixel_y1 = min(full_height, latent_y1 * latent_scale_y)

    return {
        "pixel_x0": pixel_x0,
        "pixel_x1": pixel_x1,
        "pixel_y0": pixel_y0,
        "pixel_y1": pixel_y1,
        "latent_x0": latent_x0,
        "latent_x1": latent_x1,
        "latent_y0": latent_y0,
        "latent_y1": latent_y1,
    }

def _crop_spatial_tensor(tensor: torch.Tensor, *, x0: int, x1: int, y0: int, y1: int) -> torch.Tensor:
    return tensor[..., y0:y1, x0:x1]

def _crop_seam_maps(
    seam_maps: Dict[str, object],
    *,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    full_height: int,
    full_width: int,
) -> Dict[str, object]:
    cropped: Dict[str, object] = {}
    for key, value in seam_maps.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 4 and tuple(value.shape[-2:]) == (full_height, full_width):
            cropped[key] = value[..., y0:y1, x0:x1]
        else:
            cropped[key] = value
    return cropped


def _select_seam_maps_for_edge(seam_maps: Dict[str, object], edge_index: int) -> Dict[str, object]:
    selected = dict(seam_maps)
    per_edge_key_map = {
        "margin_inner": "margin_inner_per_edge",
        "margin_outer": "margin_outer_per_edge",
        "interior_inner": "interior_inner_per_edge",
        "interior_outer": "interior_outer_per_edge",
        "interior_continuation": "interior_continuation_per_edge",
        "interior_core": "interior_core_per_edge",
        "continuation_distance_weighted": "continuation_distance_weighted_per_edge",
        "continuation_linear_weight": "continuation_linear_weight_per_edge",
        "continuation_distance_px": "continuation_distance_px_per_edge",
    }
    for base_key, per_edge_key in per_edge_key_map.items():
        value = seam_maps.get(per_edge_key)
        if isinstance(value, torch.Tensor) and value.ndim >= 4 and value.shape[1] > edge_index:
            selected[base_key] = value[:, edge_index : edge_index + 1]
    return selected


def _aggregate_weighted_seam_summaries(
    edge_summaries: List[Dict[str, object]],
    *,
    reference_tensor: torch.Tensor,
    region_weights: Dict[str, float],
    continuation_width_px: float,
) -> Dict[str, object]:
    zero = reference_tensor.new_tensor(0.0)
    if not edge_summaries:
        return {
            "region_losses": {},
            "halo_loss_raw": zero,
            "interior_loss_raw": zero,
            "halo_supervised_px": zero,
            "interior_supervised_px": zero,
            "halo_loss_weighted": zero,
            "interior_loss_weighted": zero,
            "total_loss": zero,
            "region_weighted_contributions": {},
            "continuation_loss_raw": zero,
            "continuation_loss_weighted": zero,
            "continuation_supervised_px": zero,
            "continuation_weight_sum": zero,
            "continuation_effective_weight_mean": zero,
            "continuation_effective_weight_max": zero,
            "continuation_weight_fraction_first_8px": zero,
            "continuation_weight_fraction_first_16px": zero,
            "continuation_weight_fraction_first_24px": zero,
            "continuation_weight_fraction_full_band": zero,
            "continuation_rgb_weight_mode": "falloff",
            "continuation_rgb_loss_bin_0_8": zero,
            "continuation_rgb_loss_bin_8_16": zero,
            "continuation_rgb_loss_bin_16_24": zero,
            "continuation_rgb_loss_bin_24_32": zero,
            "continuation_rgb_loss_bin_32_40": zero,
            "continuation_rgb_loss_bin_40_48": zero,
            "interior_core_supervised_px": zero,
        }

    def _sum_summary_tensor(field_name: str) -> torch.Tensor:
        values = [value for value in (summary.get(field_name) for summary in edge_summaries) if isinstance(value, torch.Tensor)]
        if not values:
            return zero
        return torch.stack(values).sum()

    def _sum_summary_region(field_name: str, region_name: str) -> torch.Tensor:
        values = []
        for summary in edge_summaries:
            field = summary.get(field_name)
            if isinstance(field, dict):
                value = field.get(region_name)
                if isinstance(value, torch.Tensor):
                    values.append(value)
        if not values:
            return zero
        return torch.stack(values).sum()

    def _max_summary_tensor(field_name: str) -> torch.Tensor:
        values = [value for value in (summary.get(field_name) for summary in edge_summaries) if isinstance(value, torch.Tensor)]
        if not values:
            return zero
        return torch.stack(values).max()

    def _safe_mean(raw_sum: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
        if float(denom.detach().item()) <= 0.0:
            return zero
        return raw_sum / denom.clamp_min(1e-6)

    def _fraction(numer: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
        if float(denom.detach().item()) <= 0.0:
            return zero
        return numer / denom.clamp_min(1e-6)

    region_raw_sums = {
        "halo_inner": _sum_summary_region("region_raw_sums", "halo_inner"),
        "halo_outer": _sum_summary_region("region_raw_sums", "halo_outer"),
        "interior_continuation": _sum_summary_region("region_raw_sums", "interior_continuation"),
        "interior_core": _sum_summary_region("region_raw_sums", "interior_core"),
    }
    region_areas = {
        "halo_inner": _sum_summary_region("region_areas", "halo_inner"),
        "halo_outer": _sum_summary_region("region_areas", "halo_outer"),
        "interior_continuation": _sum_summary_region("region_areas", "interior_continuation"),
        "interior_core": _sum_summary_region("region_areas", "interior_core"),
    }

    halo_inner_loss = _safe_mean(region_raw_sums["halo_inner"], region_areas["halo_inner"])
    halo_outer_loss = _safe_mean(region_raw_sums["halo_outer"], region_areas["halo_outer"])
    continuation_raw_loss = _safe_mean(region_raw_sums["interior_continuation"], region_areas["interior_continuation"])
    interior_core_loss = _safe_mean(region_raw_sums["interior_core"], region_areas["interior_core"])

    continuation_weighted_raw_sum = _sum_summary_tensor("continuation_weighted_raw_sum")
    continuation_weight_sum = _sum_summary_tensor("continuation_weight_sum")
    continuation_supervised_px = _sum_summary_tensor("continuation_supervised_px")
    continuation_weight_sum_first_8px = _sum_summary_tensor("continuation_weight_sum_first_8px")
    continuation_weight_sum_first_16px = _sum_summary_tensor("continuation_weight_sum_first_16px")
    continuation_weight_sum_first_24px = _sum_summary_tensor("continuation_weight_sum_first_24px")
    continuation_effective_weight_max = _max_summary_tensor("continuation_effective_weight_max")
    continuation_weighted_mean = _safe_mean(continuation_weighted_raw_sum, continuation_weight_sum)
    continuation_weighted_loss = continuation_weighted_mean * float(
        region_weights.get("interior_continuation", region_weights.get("continuation_distance_weighted", 1.0))
    )

    continuation_rgb_bin_raw_sum_0_8 = _sum_summary_tensor("continuation_rgb_loss_bin_raw_sum_0_8")
    continuation_rgb_bin_raw_sum_8_16 = _sum_summary_tensor("continuation_rgb_loss_bin_raw_sum_8_16")
    continuation_rgb_bin_raw_sum_16_24 = _sum_summary_tensor("continuation_rgb_loss_bin_raw_sum_16_24")
    continuation_rgb_bin_raw_sum_24_32 = _sum_summary_tensor("continuation_rgb_loss_bin_raw_sum_24_32")
    continuation_rgb_bin_raw_sum_32_40 = _sum_summary_tensor("continuation_rgb_loss_bin_raw_sum_32_40")
    continuation_rgb_bin_raw_sum_40_48 = _sum_summary_tensor("continuation_rgb_loss_bin_raw_sum_40_48")
    continuation_rgb_bin_area_0_8 = _sum_summary_tensor("continuation_rgb_bin_area_0_8")
    continuation_rgb_bin_area_8_16 = _sum_summary_tensor("continuation_rgb_bin_area_8_16")
    continuation_rgb_bin_area_16_24 = _sum_summary_tensor("continuation_rgb_bin_area_16_24")
    continuation_rgb_bin_area_24_32 = _sum_summary_tensor("continuation_rgb_bin_area_24_32")
    continuation_rgb_bin_area_32_40 = _sum_summary_tensor("continuation_rgb_bin_area_32_40")
    continuation_rgb_bin_area_40_48 = _sum_summary_tensor("continuation_rgb_bin_area_40_48")

    halo_inner_contribution = halo_inner_loss * float(region_weights.get("halo_inner", 1.0))
    halo_outer_contribution = halo_outer_loss * float(region_weights.get("halo_outer", 1.0))
    interior_core_contribution = interior_core_loss * float(region_weights.get("interior_core", 1.0))

    region_losses = {
        "halo_inner": halo_inner_loss,
        "halo_outer": halo_outer_loss,
        "interior_continuation": continuation_raw_loss,
        "interior_core": interior_core_loss,
        "margin_inner": halo_inner_loss,
        "margin_outer": halo_outer_loss,
        "interior_inner": continuation_raw_loss,
        "interior_outer": interior_core_loss,
    }
    region_weighted_contributions = {
        "halo_inner": halo_inner_contribution,
        "halo_outer": halo_outer_contribution,
        "interior_continuation": continuation_weighted_loss,
        "continuation_distance_weighted": continuation_weighted_loss,
        "interior_core": interior_core_contribution,
        "margin_inner": halo_inner_contribution,
        "margin_outer": halo_outer_contribution,
        "interior_inner": continuation_weighted_loss,
        "interior_outer": interior_core_contribution,
    }

    return {
        "region_losses": region_losses,
        "halo_loss_raw": region_raw_sums["halo_inner"] + region_raw_sums["halo_outer"],
        "interior_loss_raw": region_raw_sums["interior_continuation"] + region_raw_sums["interior_core"],
        "halo_supervised_px": region_areas["halo_inner"] + region_areas["halo_outer"],
        "interior_supervised_px": region_areas["interior_continuation"] + region_areas["interior_core"],
        "halo_loss_weighted": halo_inner_contribution + halo_outer_contribution,
        "interior_loss_weighted": continuation_weighted_loss + interior_core_contribution,
        "total_loss": halo_inner_contribution + halo_outer_contribution + continuation_weighted_loss + interior_core_contribution,
        "region_weighted_contributions": region_weighted_contributions,
        "region_raw_sums": region_raw_sums,
        "region_areas": region_areas,
        "continuation_loss_raw": continuation_raw_loss,
        "continuation_loss_weighted": continuation_weighted_loss,
        "continuation_supervised_px": continuation_supervised_px,
        "continuation_weight_sum": continuation_weight_sum,
        "continuation_effective_weight_mean": _safe_mean(continuation_weight_sum, continuation_supervised_px),
        "continuation_effective_weight_max": continuation_effective_weight_max,
        "continuation_weight_fraction_first_8px": _fraction(continuation_weight_sum_first_8px, continuation_weight_sum),
        "continuation_weight_fraction_first_16px": _fraction(continuation_weight_sum_first_16px, continuation_weight_sum),
        "continuation_weight_fraction_first_24px": _fraction(continuation_weight_sum_first_24px, continuation_weight_sum),
        "continuation_weight_fraction_full_band": _fraction(continuation_weight_sum, continuation_weight_sum),
        "continuation_rgb_weight_mode": str(edge_summaries[0].get("continuation_rgb_weight_mode", "falloff")),
        "continuation_rgb_loss_bin_0_8": _safe_mean(continuation_rgb_bin_raw_sum_0_8, continuation_rgb_bin_area_0_8),
        "continuation_rgb_loss_bin_8_16": _safe_mean(continuation_rgb_bin_raw_sum_8_16, continuation_rgb_bin_area_8_16),
        "continuation_rgb_loss_bin_16_24": _safe_mean(continuation_rgb_bin_raw_sum_16_24, continuation_rgb_bin_area_16_24),
        "continuation_rgb_loss_bin_24_32": _safe_mean(continuation_rgb_bin_raw_sum_24_32, continuation_rgb_bin_area_24_32),
        "continuation_rgb_loss_bin_32_40": _safe_mean(continuation_rgb_bin_raw_sum_32_40, continuation_rgb_bin_area_32_40),
        "continuation_rgb_loss_bin_40_48": _safe_mean(continuation_rgb_bin_raw_sum_40_48, continuation_rgb_bin_area_40_48),
        "interior_core_supervised_px": region_areas["interior_core"],
    }


def _save_seam_visual_debug(
    output_dir: str,
    step: int,
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    supervision_mask: torch.Tensor,
    seam_maps: Dict[str, torch.Tensor],
) -> None:
    seam_dir = os.path.join(output_dir, "sanity", "seam_debug")
    os.makedirs(seam_dir, exist_ok=True)

    pred0 = pred_rgb[0].detach().float().cpu()
    target0 = target_rgb[0].detach().float().cpu()
    supervision0 = supervision_mask[0, 0].detach().float().cpu().clamp(0.0, 1.0)
    halo_mask0 = (seam_maps["margin_inner"][0, 0] + seam_maps["margin_outer"][0, 0]).detach().float().cpu().clamp(0.0, 1.0)
    interior_mask0 = (seam_maps["interior_inner"][0, 0] + seam_maps["interior_outer"][0, 0]).detach().float().cpu().clamp(0.0, 1.0)

    diff_map = (pred0 - target0).abs().mean(dim=0)
    halo_diff = (diff_map * halo_mask0).clamp(0.0, 1.0)
    interior_diff = (diff_map * interior_mask0).clamp(0.0, 1.0)

    halo_norm = halo_diff / max(float(halo_diff.max().item()), 1e-6)
    interior_norm = interior_diff / max(float(interior_diff.max().item()), 1e-6)

    _tensor_to_image(pred0).save(os.path.join(seam_dir, f"step_{step:06d}_expanded_prediction.png"))
    _tensor_to_image(target0).save(os.path.join(seam_dir, f"step_{step:06d}_expanded_target.png"))
    _mask_to_image(supervision0).save(os.path.join(seam_dir, f"step_{step:06d}_supervision_mask.png"))
    _mask_to_image(halo_norm).save(os.path.join(seam_dir, f"step_{step:06d}_halo_only_diff_heatmap.png"))
    _mask_to_image(interior_norm).save(os.path.join(seam_dir, f"step_{step:06d}_interior_band_diff.png"))

    region_overlay = torch.zeros((3, target0.shape[-2], target0.shape[-1]), dtype=torch.float32)
    margin_inner = seam_maps["margin_inner"][0, 0].detach().float().cpu().clamp(0.0, 1.0)
    margin_outer = seam_maps["margin_outer"][0, 0].detach().float().cpu().clamp(0.0, 1.0)
    interior_inner = seam_maps["interior_inner"][0, 0].detach().float().cpu().clamp(0.0, 1.0)
    interior_outer = seam_maps["interior_outer"][0, 0].detach().float().cpu().clamp(0.0, 1.0)

    # Color key: halo_inner=red, halo_outer=orange, interior_continuation=cyan, interior_core=blue.
    region_overlay[0] = torch.maximum(region_overlay[0], margin_inner)
    region_overlay[0] = torch.maximum(region_overlay[0], margin_outer * 0.95)
    region_overlay[1] = torch.maximum(region_overlay[1], margin_outer * 0.55)
    region_overlay[1] = torch.maximum(region_overlay[1], interior_inner * 0.9)
    region_overlay[2] = torch.maximum(region_overlay[2], interior_inner * 0.95)
    region_overlay[2] = torch.maximum(region_overlay[2], interior_outer)

    base01 = ((target0.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
    blended = (0.45 * base01) + (0.55 * region_overlay.clamp(0.0, 1.0))
    _tensor_to_image((blended * 2.0) - 1.0).save(os.path.join(seam_dir, f"step_{step:06d}_region_overlay.png"))


def _terrain_mask_to_occupancy(mask: torch.Tensor, black_is_terrain: bool) -> torch.Tensor:
    return shared_terrain_mask_to_occupancy(mask, black_is_terrain)


def _resolve_alpha_head_indices(head_scales: List[int], head_mode: str, single_scale_index: int) -> List[int]:
    unique_scales = sorted(set([int(v) for v in head_scales]))
    if not unique_scales:
        return []

    mode = str(head_mode).strip().lower()
    if mode == "fused":
        return unique_scales
    if mode == "single_high":
        return [min(unique_scales)]
    if mode == "single_mid":
        return [unique_scales[len(unique_scales) // 2]]
    if mode == "single_scale":
        if single_scale_index < 0:
            raise ValueError("alpha.single_scale_index must be set when alpha.head_mode='single_scale'")
        if single_scale_index not in unique_scales:
            raise ValueError(
                f"alpha.single_scale_index={single_scale_index} is not in head_scales={unique_scales}"
            )
        return [int(single_scale_index)]
    raise ValueError(f"unsupported alpha.head_mode='{head_mode}'")


def _select_alpha_logits(alpha_outputs: Optional[Dict[str, object]], output_source: str) -> torch.Tensor:
    if alpha_outputs is None:
        raise RuntimeError("alpha outputs are missing while selecting logits")

    source = str(output_source).strip().lower()
    if source in {"main", "fused", "fused_logits"}:
        logits = alpha_outputs.get("fused_logits")
    elif source in {"terrain_mask", "terrain_baseline", "terrain_mask_baseline"}:
        logits = (alpha_outputs.get("baseline_logits") or {}).get("terrain_mask")
    elif source in {"pre_stem", "prestem", "pre_stem_baseline"}:
        logits = (alpha_outputs.get("baseline_logits") or {}).get("pre_stem")
    else:
        raise ValueError(f"unsupported alpha.output_source='{output_source}'")

    if logits is None:
        available_baselines = sorted(list((alpha_outputs.get("baseline_logits") or {}).keys()))
        raise RuntimeError(
            f"requested alpha logits source '{output_source}' was not produced; "
            f"available_baselines={available_baselines}"
        )
    return logits


def _histogram_counts(values: torch.Tensor, bin_edges: List[float]) -> List[int]:
    flat = values.detach().float().flatten().cpu().numpy()
    counts, _ = np.histogram(flat, bins=np.asarray(bin_edges, dtype=np.float32))
    return [int(v) for v in counts.tolist()]


def _run_pre_gate_target_terrain_iou_sanity(
    dataset: TerrainSemanticManifestDataset,
    alpha_config: Dict[str, object],
    resolved_eval_samples,
) -> None:
    if not alpha_config["enabled"]:
        return

    if resolved_eval_samples:
        sanity_index = int(resolved_eval_samples[0].dataset_index)
    else:
        sanity_index = 0
    sample = dataset[sanity_index]

    alpha_target = sample["alpha_target"]
    if alpha_target is None:
        logger.warning(
            "[sanity/pre_gate_target] alpha_target missing for selected sanity sample; skipping target-terrain IoU check"
        )
        return

    terrain_mask_index = _find_channel_index(dataset.channel_names, "terrain_mask")
    terrain_mask_raw = sample["conditioning_images"][terrain_mask_index].detach().float().clamp(0.0, 1.0)
    terrain_mask_occ = _terrain_mask_to_occupancy(
        terrain_mask_raw,
        bool(alpha_config["terrain_mask_black_is_terrain"]),
    )
    threshold = float(alpha_config["binary_threshold"])
    alpha_bin = (alpha_target.detach().float() >= threshold).float()
    terrain_bin_raw = (terrain_mask_raw >= threshold).float()
    terrain_bin_raw_inv = 1.0 - terrain_bin_raw
    terrain_bin = (terrain_mask_occ >= threshold).float()

    inter = float((alpha_bin * terrain_bin).sum().item())
    union = float((alpha_bin + terrain_bin - alpha_bin * terrain_bin).sum().item())
    iou = inter / max(union, 1e-6)
    inter_raw = float((alpha_bin * terrain_bin_raw).sum().item())
    union_raw = float((alpha_bin + terrain_bin_raw - alpha_bin * terrain_bin_raw).sum().item())
    iou_raw = inter_raw / max(union_raw, 1e-6)
    inter_raw_inv = float((alpha_bin * terrain_bin_raw_inv).sum().item())
    union_raw_inv = float((alpha_bin + terrain_bin_raw_inv - alpha_bin * terrain_bin_raw_inv).sum().item())
    iou_raw_inv = inter_raw_inv / max(union_raw_inv, 1e-6)
    alpha_occ = float(alpha_bin.mean().item())
    terrain_occ = float(terrain_bin.mean().item())
    logger.info(
        "[sanity/pre_gate_target] "
        + f"dataset_index={sanity_index} iou_target_vs_terrain={iou:.6f} "
        + f"iou_target_vs_terrain_rawpol={iou_raw:.6f} iou_target_vs_terrain_inverted_rawpol={iou_raw_inv:.6f} "
        + f"target_occ={alpha_occ:.4f} terrain_occ={terrain_occ:.4f} threshold={threshold:.3f} "
        + f"terrain_mask_black_is_terrain={bool(alpha_config['terrain_mask_black_is_terrain'])}"
    )
    min_iou = float(alpha_config["pre_gate_target_terrain_iou_min"])
    if iou < min_iou:
        raise RuntimeError(
            "pre-gate target sanity failed: target-vs-terrain IoU is below configured minimum. "
            + f"iou={iou:.6f} min_required={min_iou:.6f}"
        )


def _log_alpha_loss_breakdown(
    global_step: int,
    diffusion_loss_value: float,
    alpha_bce_value: float,
    alpha_edge_value: float,
    alpha_total_value: float,
    alpha_terrain_bce_value: float,
    alpha_terrain_iou_value: float,
    terrain_curriculum_factor: float,
    warn_ratio: float,
) -> None:
    if diffusion_loss_value <= 0.0:
        return
    ratio = alpha_total_value / diffusion_loss_value
    logger.info(
        "[alpha/loss] "
        + f"step={global_step} diffusion={diffusion_loss_value:.6f} "
        + f"alpha_bce={alpha_bce_value:.6f} alpha_edge={alpha_edge_value:.6f} "
        + f"terrain_bce={alpha_terrain_bce_value:.6f} terrain_iou={alpha_terrain_iou_value:.6f} "
        + f"terrain_curriculum={terrain_curriculum_factor:.4f} "
        + f"alpha_total={alpha_total_value:.6f} alpha_over_diffusion={ratio:.4f}"
    )
    if ratio > warn_ratio:
        logger.warning(
            f"[alpha/loss] alpha supervision is large relative to diffusion loss at step {global_step}: ratio={ratio:.4f}"
        )


def save_controlnet_checkpoint(
    accelerator: Accelerator,
    control_net: torch.nn.Module,
    output_dir: str,
    output_name: str,
    step: int,
    save_dtype: torch.dtype,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{output_name}-step-{step:06d}.safetensors")
    state_dict = unwrap_model(accelerator, control_net).state_dict()
    converted = {key: value.detach().to("cpu", dtype=save_dtype) for key, value in state_dict.items()}
    save_file(converted, filename)
    logger.info(f"saved checkpoint: {filename}")


def _collect_module_param_counts(named_parameters) -> Tuple[int, int, Dict[str, int]]:
    total = 0
    trainable = 0
    top_level_counts: Dict[str, int] = {}
    for name, parameter in named_parameters:
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            top = name.split(".", 1)[0]
            top_level_counts[top] = top_level_counts.get(top, 0) + count
    return total, trainable, top_level_counts


def _collect_trainable_module_stats(control_net: torch.nn.Module) -> Dict[str, Dict[str, float]]:
    stats = {
        "semantic_pre_stem": {"l2": 0.0, "abs_sum": 0.0, "count": 0},
        "seam_adapter": {"l2": 0.0, "abs_sum": 0.0, "count": 0},
        "control_residual_blocks": {"l2": 0.0, "abs_sum": 0.0, "count": 0},
        "control_mid_block": {"l2": 0.0, "abs_sum": 0.0, "count": 0},
        "alpha_heads": {"l2": 0.0, "abs_sum": 0.0, "count": 0},
    }

    for name, parameter in control_net.named_parameters():
        if not parameter.requires_grad:
            continue
        tensor = parameter.detach().float()
        if "controlnet_down_blocks" in name:
            bucket = "control_residual_blocks"
        elif "controlnet_mid_block" in name:
            bucket = "control_mid_block"
        elif "controlnet_alpha_heads" in name or "controlnet_alpha_baseline_heads" in name:
            bucket = "alpha_heads"
        elif "controlnet_seam_adapter" in name:
            bucket = "seam_adapter"
        elif "controlnet_cond_embedding" in name:
            bucket = "semantic_pre_stem"
        else:
            continue

        stats[bucket]["l2"] += float(torch.sum(tensor * tensor).item())
        stats[bucket]["abs_sum"] += float(torch.sum(torch.abs(tensor)).item())
        stats[bucket]["count"] += int(tensor.numel())

    for values in stats.values():
        values["l2"] = values["l2"] ** 0.5
    return stats


def _collect_trainable_grad_stats(control_net: torch.nn.Module) -> Dict[str, Dict[str, float]]:
    stats = {
        "semantic_pre_stem": {"l2": 0.0, "abs_sum": 0.0, "count": 0},
        "seam_adapter": {"l2": 0.0, "abs_sum": 0.0, "count": 0},
        "control_residual_blocks": {"l2": 0.0, "abs_sum": 0.0, "count": 0},
        "control_mid_block": {"l2": 0.0, "abs_sum": 0.0, "count": 0},
        "alpha_heads": {"l2": 0.0, "abs_sum": 0.0, "count": 0},
    }

    for name, parameter in control_net.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        if "controlnet_down_blocks" in name:
            bucket = "control_residual_blocks"
        elif "controlnet_mid_block" in name:
            bucket = "control_mid_block"
        elif "controlnet_alpha_heads" in name or "controlnet_alpha_baseline_heads" in name:
            bucket = "alpha_heads"
        elif "controlnet_seam_adapter" in name:
            bucket = "seam_adapter"
        elif "controlnet_cond_embedding" in name:
            bucket = "semantic_pre_stem"
        else:
            continue

        stats[bucket]["l2"] += float(torch.sum(grad * grad).item())
        stats[bucket]["abs_sum"] += float(torch.sum(torch.abs(grad)).item())
        stats[bucket]["count"] += int(grad.numel())

    for values in stats.values():
        values["l2"] = values["l2"] ** 0.5
    return stats


def _compute_module_update_deltas(
    before_stats: Dict[str, Dict[str, float]],
    after_stats: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for key in before_stats.keys():
        deltas[key] = abs(after_stats[key]["abs_sum"] - before_stats[key]["abs_sum"])
    return deltas


def verify_optimizer_parameter_membership(
    optimizer: torch.optim.Optimizer,
    control_net: torch.nn.Module,
) -> None:
    expected = {
        id(parameter): name
        for name, parameter in control_net.named_parameters()
        if parameter.requires_grad
    }
    actual = {}
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter in group["params"]:
            actual[id(parameter)] = group_index

    missing = [name for param_id, name in expected.items() if param_id not in actual]
    extra = [param_id for param_id in actual.keys() if param_id not in expected]

    logger.info(
        f"[verify/optimizer] expected_trainable_tensors={len(expected)} optimizer_tensors={len(actual)} param_groups={len(optimizer.param_groups)}"
    )
    if missing:
        raise RuntimeError(f"optimizer is missing trainable tensors: {missing[:8]}")
    if extra:
        raise RuntimeError(f"optimizer has unexpected tensors not marked trainable: count={len(extra)}")


def _log_alpha_health(
    global_step: int,
    binary_alpha_target: torch.Tensor,
    supervision_mask: torch.Tensor,
    scaled_alpha_logits: torch.Tensor,
    edge_band: torch.Tensor,
) -> None:
    with torch.no_grad():
        probabilities = torch.sigmoid(scaled_alpha_logits.float())
        target_occ = float(binary_alpha_target.float().mean().item())
        pred_occ = float(probabilities.mean().item())
        pred_std = float(probabilities.std().item())
        edge_coverage = float(edge_band.float().mean().item())
        supervision_coverage = float(supervision_mask.float().mean().item())
        opaque_frac = float((probabilities > 0.95).float().mean().item())
        transparent_frac = float((probabilities < 0.05).float().mean().item())
        near_zero_prob_frac = float((probabilities <= 0.01).float().mean().item())
        near_one_prob_frac = float((probabilities >= 0.99).float().mean().item())
        logit_hist_bins = [-12.0, -8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 12.0]
        prob_hist_bins = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
        logit_hist_counts = _histogram_counts(scaled_alpha_logits.float(), logit_hist_bins)
        prob_hist_counts = _histogram_counts(probabilities, prob_hist_bins)
        logger.info(
            "[verify/alpha] "
            + f"step={global_step} target_occ={target_occ:.4f} pred_occ={pred_occ:.4f} pred_std={pred_std:.4f} "
            + f"supervision_cov={supervision_coverage:.4f} edge_cov={edge_coverage:.4f} "
            + f"pred_opaque95={opaque_frac:.4f} pred_transparent05={transparent_frac:.4f} "
            + f"pred_near0_01={near_zero_prob_frac:.4f} pred_near1_99={near_one_prob_frac:.4f}"
        )
        logger.info(
            "[verify/alpha_hist] "
            + f"step={global_step} logits_bins={logit_hist_bins} logits_counts={logit_hist_counts} "
            + f"prob_bins={prob_hist_bins} prob_counts={prob_hist_counts}"
        )


def _log_controlnet_diagnostics(global_step: int, diagnostics: Optional[Dict[str, object]]) -> None:
    if diagnostics is None:
        return
    residuals = diagnostics.get("down_block_residual_norms") or []
    activations = diagnostics.get("down_block_activation_norms") or []
    ratios = diagnostics.get("down_block_residual_to_activation_ratios") or []
    residual_str = ",".join([f"{value:.4f}" for value in residuals])
    activation_str = ",".join([f"{value:.4f}" for value in activations])
    ratio_str = ",".join([f"{value:.4f}" for value in ratios])
    logger.info(
        "[verify/controlnet] "
        + f"step={global_step} multiplier={float(diagnostics.get('multiplier', 1.0)):.4f} "
        + f"cond_embedding_norm={float(diagnostics.get('cond_embedding_norm', 0.0) or 0.0):.4f} "
        + f"mid_norm={float(diagnostics.get('mid_block_residual_norm', 0.0)):.4f} "
        + f"down_norms=[{residual_str}] "
        + f"down_activation_norms=[{activation_str}] "
        + f"down_ratio_residual_to_activation=[{ratio_str}]"
    )


def run_controlnet_influence_sanity_check(
    args: argparse.Namespace,
    verification_config: Dict[str, object],
    dataset: TerrainSemanticManifestDataset,
    unet: torch.nn.Module,
    control_net: torch.nn.Module,
    material_lora_unet: torch.nn.Module,
    material_lora_control: torch.nn.Module,
    vae,
    noise_scheduler: DDPMScheduler,
    cached_text: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    seam_config: Dict[str, object],
):
    sanity_dir = os.path.join(args.output_dir, "sanity")
    os.makedirs(sanity_dir, exist_ok=True)

    unet_dtype = next(unet.parameters()).dtype
    control_dtype = next(control_net.parameters()).dtype

    sample = dataset[0]
    sample_batch: Dict[str, object] = {
        "conditioning_images": sample["conditioning_images"].unsqueeze(0),
        "seam_strip_tensor": sample.get("seam_strip_tensor").unsqueeze(0) if sample.get("seam_strip_tensor") is not None else None,
        "edge_defined_flags": sample.get("edge_defined_flags").unsqueeze(0) if sample.get("edge_defined_flags") is not None else None,
        "edge_flag_maps": sample.get("edge_flag_maps").unsqueeze(0) if sample.get("edge_flag_maps") is not None else None,
    }
    conditioning, _, _ = build_model_visible_conditioning(
        sample_batch,
        dataset,
        seam_config,
        device,
        torch.float32,
    )
    contrast_flat, contrast_wall, contrast_channel_groups = _build_extreme_contrast_conditioning(dataset, conditioning)
    target_size = sample["target_sizes_hw"].tolist()
    latent_h = target_size[0] // 8
    latent_w = target_size[1] // 8

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 29)
    latents = torch.randn((1, 4, latent_h, latent_w), generator=generator, device=device, dtype=torch.float32)

    encoder_hidden_states1, encoder_hidden_states2, pool2 = cached_text
    text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(dtype=torch.float32)
    size_embeddings = sdxl_train_util.get_size_embeddings(
        sample["original_sizes_hw"].unsqueeze(0),
        sample["crop_top_lefts"].unsqueeze(0),
        sample["target_sizes_hw"].unsqueeze(0),
        device,
    ).to(torch.float32)
    vector_embedding = torch.cat([pool2, size_embeddings], dim=1).to(torch.float32)

    original_multiplier = getattr(control_net, "multiplier", None)

    vae_original_device = next(vae.parameters()).device
    vae_original_dtype = next(vae.parameters()).dtype
    vae.to(device=device, dtype=dtype)

    original_model_modes = {
        "unet": unet.training,
        "control_net": control_net.training,
        "material_lora_unet": material_lora_unet.training,
        "material_lora_control": material_lora_control.training,
    }
    unet.eval()
    control_net.eval()
    material_lora_unet.eval()
    material_lora_control.eval()

    def predict_x0(
        control_multiplier: float,
        cond_tensor: torch.Tensor,
        collect_alpha_logits: bool = False,
        seam_local_zero_input: bool = False,
        disable_seam_adapter: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        material_lora_unet.set_multiplier(args.material_lora_multiplier)
        material_lora_control.set_multiplier(args.material_lora_multiplier)
        control_net.multiplier = control_multiplier
        last_alpha_logits: Optional[torch.Tensor] = None
        original_seam_adapter_scale = getattr(control_net, "seam_adapter_scale", None)
        seam_local_kwargs: Dict[str, torch.Tensor] = {}
        if seam_local_zero_input:
            seam_payload = build_seam_local_adapter_maps_from_conditioning(
                cond_tensor,
                seam_conditioning_offset=int(seam_config.get("seam_adapter_conditioning_offset", -1)),
                band_px=int(seam_config.get("seam_adapter_band_px", 0)),
                seam_adapter_per_edge=bool(seam_config.get("seam_adapter_per_edge", False)),
                seam_adapter_extrusion_mode=str(seam_config.get("seam_adapter_extrusion_mode", "decay")),
            )
            seam_local_kwargs = _build_seam_local_kwargs_from_payload(seam_payload, zero_input=True)
        if disable_seam_adapter and original_seam_adapter_scale is not None:
            control_net.seam_adapter_scale = 0.0
        with torch.no_grad():
            inference_steps = max(2, int(verification_config["controlnet_sanity_steps"]))
            noise_scheduler.set_timesteps(inference_steps, device=device)
            noisy = latents.clone()
            for timestep in noise_scheduler.timesteps:
                t = timestep.expand(1).to(device=device, dtype=torch.long)
                noisy_control = noisy.to(dtype=control_dtype)
                cond_control = cond_tensor.to(dtype=control_dtype)
                text_control = text_embedding.to(dtype=control_dtype)
                vector_control = vector_embedding.to(dtype=control_dtype)
                if collect_alpha_logits:
                    input_resi_add, mid_add, alpha_outputs = control_net(
                        noisy_control,
                        t,
                        text_control,
                        vector_control,
                        cond_control,
                        return_alpha=True,
                        alpha_target_size=tuple(sample["target_sizes_hw"].tolist()),
                        **seam_local_kwargs,
                    )
                    if alpha_outputs is not None:
                        candidate_logits = alpha_outputs.get("fused_logits")
                        if candidate_logits is not None:
                            last_alpha_logits = candidate_logits.detach().float()
                else:
                    input_resi_add, mid_add = control_net(
                        noisy_control,
                        t,
                        text_control,
                        vector_control,
                        cond_control,
                        **seam_local_kwargs,
                    )

                noisy_unet = noisy.to(dtype=unet_dtype)
                text_unet = text_embedding.to(dtype=unet_dtype)
                vector_unet = vector_embedding.to(dtype=unet_dtype)
                input_resi_add = [x.to(dtype=unet_dtype) for x in input_resi_add]
                mid_add = mid_add.to(dtype=unet_dtype)
                eps = unet(noisy_unet, t, text_unet, vector_unet, input_resi_add, mid_add)
                noisy = noise_scheduler.step(eps, timestep, noisy).prev_sample

            alpha_t = noise_scheduler.alphas_cumprod[noise_scheduler.timesteps[-1]].to(device=device, dtype=torch.float32)
            alpha_t = alpha_t.view(1, 1, 1, 1)
            pred_x0 = noisy / alpha_t.sqrt()
        if disable_seam_adapter and original_seam_adapter_scale is not None:
            control_net.seam_adapter_scale = original_seam_adapter_scale
        return pred_x0, last_alpha_logits

    def measure_controlnet_zero_input_equivalence(cond_tensor: torch.Tensor) -> tuple[float, float]:
        original_seam_adapter_scale = getattr(control_net, "seam_adapter_scale", None)
        seam_payload = build_seam_local_adapter_maps_from_conditioning(
            cond_tensor,
            seam_conditioning_offset=int(seam_config.get("seam_adapter_conditioning_offset", -1)),
            band_px=int(seam_config.get("seam_adapter_band_px", 0)),
            seam_adapter_per_edge=bool(seam_config.get("seam_adapter_per_edge", False)),
            seam_adapter_extrusion_mode=str(seam_config.get("seam_adapter_extrusion_mode", "decay")),
        )
        seam_local_kwargs = _build_seam_local_kwargs_from_payload(seam_payload, zero_input=True)
        direct_timestep = torch.full((1,), noise_scheduler.config.num_train_timesteps // 2, device=device, dtype=torch.long)
        noisy_control = latents.to(dtype=control_dtype)
        cond_control = cond_tensor.to(dtype=control_dtype)
        text_control = text_embedding.to(dtype=control_dtype)
        vector_control = vector_embedding.to(dtype=control_dtype)
        with torch.no_grad():
            if original_seam_adapter_scale is not None:
                control_net.seam_adapter_scale = 0.0
            disabled_input_resi_add, disabled_mid_add = control_net(
                noisy_control,
                direct_timestep,
                text_control,
                vector_control,
                cond_control,
            )
            if original_seam_adapter_scale is not None:
                control_net.seam_adapter_scale = original_seam_adapter_scale
            zero_input_resi_add, zero_input_mid_add = control_net(
                noisy_control,
                direct_timestep,
                text_control,
                vector_control,
                cond_control,
                **seam_local_kwargs,
            )
        if original_seam_adapter_scale is not None:
            control_net.seam_adapter_scale = original_seam_adapter_scale

        residual_mse_terms = [
            F.mse_loss(disabled.float(), zero.float())
            for disabled, zero in zip(disabled_input_resi_add, zero_input_resi_add)
        ]
        residual_mse_terms.append(F.mse_loss(disabled_mid_add.float(), zero_input_mid_add.float()))
        mean_residual_mse = float(torch.stack(residual_mse_terms).mean().item())

        max_abs_terms = [
            (disabled.float() - zero.float()).abs().max()
            for disabled, zero in zip(disabled_input_resi_add, zero_input_resi_add)
        ]
        max_abs_terms.append((disabled_mid_add.float() - zero_input_mid_add.float()).abs().max())
        max_abs_diff = float(torch.stack(max_abs_terms).max().item())
        return mean_residual_mse, max_abs_diff

    def _decode_preview(pred_latents: torch.Tensor) -> Image.Image:
        vae_dtype = next(vae.parameters()).dtype
        latents_for_decode = (pred_latents / sdxl_model_util.VAE_SCALE_FACTOR).to(dtype=vae_dtype)
        with torch.no_grad():
            image = vae.decode(latents_for_decode).sample[0]
        return _tensor_to_image(image)

    def _alpha_logits_to_preview(alpha_logits: Optional[torch.Tensor]) -> Optional[Image.Image]:
        if alpha_logits is None:
            return None
        alpha_prob = torch.sigmoid(alpha_logits[0, 0].detach().float()).clamp(0.0, 1.0)
        alpha_img = (alpha_prob * 255.0).round().to(torch.uint8).cpu().numpy()
        return Image.fromarray(alpha_img, mode="L")

    pred_on, _ = predict_x0(1.0, conditioning)
    pred_off, _ = predict_x0(0.0, conditioning)
    control_net.multiplier = original_multiplier

    mse_diff = F.mse_loss(pred_on.float(), pred_off.float()).item()
    logger.info(f"[verify/controlnet] on_vs_off_pred_x0_mse={mse_diff:.8f}")
    if mse_diff < float(verification_config["controlnet_min_mse"]):
        raise RuntimeError(
            "ControlNet influence sanity check failed: on/off outputs are effectively identical. "
            f"mse={mse_diff:.8f}"
        )

    if verification_config["save_sanity_previews"]:
        _decode_preview(pred_on).save(os.path.join(sanity_dir, "controlnet_on.png"))
        _decode_preview(pred_off).save(os.path.join(sanity_dir, "controlnet_off.png"))
        logger.info(f"[verify/controlnet] saved previews: {sanity_dir}")

    if verification_config.get("run_multiplier_sweep_sanity", True):
        sweep_values = [float(v) for v in verification_config.get("multiplier_sweep", [2.0, 3.0, 5.0])]
        logger.info(
            "[verify/multiplier_sweep] "
            + f"values={sweep_values} run_extreme_contrast={bool(verification_config.get('run_extreme_contrast_test', True))} "
            + f"geometry_channels={contrast_channel_groups}"
        )
        for sweep_multiplier in sweep_values:
            pred_full, _ = predict_x0(sweep_multiplier, conditioning)
            pred_zero, _ = predict_x0(0.0, conditioning)
            mse_full_vs_zero = float(F.mse_loss(pred_full.float(), pred_zero.float()).item())
            logger.info(
                "[verify/multiplier_sweep] "
                + f"multiplier={sweep_multiplier:.4f} full_vs_zero_pred_x0_mse={mse_full_vs_zero:.8f}"
            )

            if verification_config.get("run_extreme_contrast_test", True):
                collect_alpha = bool(verification_config.get("save_sanity_previews", False))
                pred_flat, alpha_flat_logits = predict_x0(
                    sweep_multiplier,
                    contrast_flat,
                    collect_alpha_logits=collect_alpha,
                )
                pred_wall, alpha_wall_logits = predict_x0(
                    sweep_multiplier,
                    contrast_wall,
                    collect_alpha_logits=collect_alpha,
                )
                mse_flat_vs_wall = float(F.mse_loss(pred_flat.float(), pred_wall.float()).item())
                logger.info(
                    "[verify/extreme_contrast] "
                    + f"multiplier={sweep_multiplier:.4f} flat_vs_wall_pred_x0_mse={mse_flat_vs_wall:.8f} "
                    + "constraint=only_geometry_channels_changed"
                )

                if verification_config["save_sanity_previews"]:
                    suffix = str(sweep_multiplier).replace(".", "p")
                    _decode_preview(pred_flat).save(os.path.join(sanity_dir, f"controlnet_flat_m{suffix}.png"))
                    _decode_preview(pred_wall).save(os.path.join(sanity_dir, f"controlnet_wall_m{suffix}.png"))
                    alpha_flat_img = _alpha_logits_to_preview(alpha_flat_logits)
                    alpha_wall_img = _alpha_logits_to_preview(alpha_wall_logits)
                    if alpha_flat_img is not None:
                        alpha_flat_img.save(os.path.join(sanity_dir, f"controlnet_alpha_flat_m{suffix}.png"))
                    if alpha_wall_img is not None:
                        alpha_wall_img.save(os.path.join(sanity_dir, f"controlnet_alpha_wall_m{suffix}.png"))

    if seam_config.get("enabled") and seam_config.get("seam_adapter_enabled"):
        zero_input_control_mse, zero_input_control_max_abs = measure_controlnet_zero_input_equivalence(conditioning)
        logger.info(
            "[verify/seam_adapter] zero_input_vs_disabled_control_residual_mse=%.10f max_abs=%.10f",
            zero_input_control_mse,
            zero_input_control_max_abs,
        )
        if zero_input_control_max_abs > 1e-7:
            raise RuntimeError(
                "Seam adapter zero-input ablation failed at the ControlNet residual level: zero seam-local inputs should "
                f"match disabled adapter. mse={zero_input_control_mse:.10f} max_abs={zero_input_control_max_abs:.10f}"
            )
        pred_adapter_disabled, _ = predict_x0(1.0, conditioning, disable_seam_adapter=True)
        pred_adapter_zero_input, _ = predict_x0(1.0, conditioning, seam_local_zero_input=True)
        seam_zero_input_mse = float(F.mse_loss(pred_adapter_disabled.float(), pred_adapter_zero_input.float()).item())
        logger.info("[verify/seam_adapter] zero_input_vs_disabled_pred_x0_mse=%.8f", seam_zero_input_mse)
        if seam_zero_input_mse > 1e-7:
            logger.warning(
                "[verify/seam_adapter] pred_x0 zero-input vs disabled mismatch did not fail startup because "
                "ControlNet residual-level equivalence passed. pred_x0_mse=%.8f",
                seam_zero_input_mse,
            )

    control_net.multiplier = original_multiplier

    vae.to(device=vae_original_device, dtype=vae_original_dtype)
    unet.train(original_model_modes["unet"])
    control_net.train(original_model_modes["control_net"])
    material_lora_unet.train(original_model_modes["material_lora_unet"])
    material_lora_control.train(original_model_modes["material_lora_control"])



def run_channel_perturbation_sanity_check(
    args: argparse.Namespace,
    dataset: TerrainSemanticManifestDataset,
    unet: torch.nn.Module,
    control_net: torch.nn.Module,
    noise_scheduler: DDPMScheduler,
    cached_text: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    seam_config: Dict[str, object],
    sanity_steps: int = 6,
    min_nonzero_channels: int = 4,
) -> None:
    """For each of the 12 conditioning channels, zero it and measure pred_x0 MSE vs full conditioning.
    Logs per-channel sensitivity and warns if too many channels are below 1e-5 MSE (ignored).
    """
    sanity_dir = os.path.join(args.output_dir, "sanity")
    os.makedirs(sanity_dir, exist_ok=True)

    control_dtype = next(control_net.parameters()).dtype
    unet_dtype = next(unet.parameters()).dtype

    sample = dataset[0]
    sample_batch: Dict[str, object] = {
        "conditioning_images": sample["conditioning_images"].unsqueeze(0),
        "seam_strip_tensor": sample.get("seam_strip_tensor").unsqueeze(0) if sample.get("seam_strip_tensor") is not None else None,
        "edge_defined_flags": sample.get("edge_defined_flags").unsqueeze(0) if sample.get("edge_defined_flags") is not None else None,
        "edge_flag_maps": sample.get("edge_flag_maps").unsqueeze(0) if sample.get("edge_flag_maps") is not None else None,
    }
    full_conditioning, full_channel_names, _ = build_model_visible_conditioning(
        sample_batch,
        dataset,
        seam_config,
        device,
        torch.float32,
    )
    target_size = sample["target_sizes_hw"].tolist()
    latent_h = target_size[0] // 8
    latent_w = target_size[1] // 8

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 53)
    latents = torch.randn((1, 4, latent_h, latent_w), generator=generator, device=device, dtype=torch.float32)

    encoder_hidden_states1, encoder_hidden_states2, pool2 = cached_text
    text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(dtype=torch.float32)
    size_embeddings = sdxl_train_util.get_size_embeddings(
        sample["original_sizes_hw"].unsqueeze(0),
        sample["crop_top_lefts"].unsqueeze(0),
        sample["target_sizes_hw"].unsqueeze(0),
        device,
    ).to(torch.float32)
    vector_embedding = torch.cat([pool2, size_embeddings], dim=1).to(torch.float32)

    def predict_x0(conditioning: torch.Tensor) -> torch.Tensor:
        steps = max(2, sanity_steps)
        noise_scheduler.set_timesteps(steps, device=device)
        noisy = latents.clone()
        with torch.no_grad():
            for timestep in noise_scheduler.timesteps:
                t = timestep.expand(1).to(device=device, dtype=torch.long)
                input_resi_add, mid_add = control_net(
                    noisy.to(dtype=control_dtype), t,
                    text_embedding.to(dtype=control_dtype),
                    vector_embedding.to(dtype=control_dtype),
                    conditioning.to(dtype=control_dtype),
                )
                eps = unet(
                    noisy.to(dtype=unet_dtype), t,
                    text_embedding.to(dtype=unet_dtype),
                    vector_embedding.to(dtype=unet_dtype),
                    [x.to(dtype=unet_dtype) for x in input_resi_add],
                    mid_add.to(dtype=unet_dtype),
                )
                noisy = noise_scheduler.step(eps, timestep, noisy).prev_sample
        alpha_t = noise_scheduler.alphas_cumprod[noise_scheduler.timesteps[-1]].view(1, 1, 1, 1).to(device, dtype=torch.float32)
        return noisy.float() / alpha_t.sqrt()

    pred_full = predict_x0(full_conditioning)

    channel_names = full_channel_names
    sensitivity_scores: List[Tuple[str, float]] = []
    dead_channels: List[str] = []

    for ch_idx in range(full_conditioning.shape[1]):
        perturbed = full_conditioning.clone()
        perturbed[:, ch_idx, :, :] = 0.0
        pred_perturbed = predict_x0(perturbed)
        mse = float(F.mse_loss(pred_full, pred_perturbed).item())
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"ch{ch_idx}"
        sensitivity_scores.append((ch_name, mse))
        logger.info(f"[verify/channel_sensitivity] ch={ch_idx} name={ch_name} zero_ablation_mse={mse:.8f}")
        if mse < 1e-5:
            dead_channels.append(ch_name)
            logger.warning(f"[verify/channel_sensitivity] WARN ch={ch_idx} {ch_name} is effectively ignored (mse={mse:.2e})")

    # Channel shuffle ablation: permute all channels
    shuffled = full_conditioning[:, torch.randperm(full_conditioning.shape[1]), :, :]
    pred_shuffled = predict_x0(shuffled)
    shuffle_mse = float(F.mse_loss(pred_full, pred_shuffled).item())
    logger.info(f"[verify/channel_sensitivity] channel_shuffle_mse={shuffle_mse:.8f}")
    if shuffle_mse < 1e-4:
        logger.warning("[verify/channel_sensitivity] WARN channel shuffle has near-zero effect, conditioning embedding may be channel-order invariant or collapsed")

    alive = len(sensitivity_scores) - len(dead_channels)
    logger.info(f"[verify/channel_sensitivity] alive={alive}/{len(sensitivity_scores)} dead_channels={dead_channels or 'none'}")
    if alive < min_nonzero_channels:
        logger.warning(
            f"[verify/channel_sensitivity] WARN only {alive}/{len(sensitivity_scores)} channels have detectable effect on output — "
            "conditioning embedding may not be learning meaningful per-channel semantics"
        )

    # Save sensitivity scores
    sens_path = os.path.join(sanity_dir, "channel_sensitivity.csv")
    with open(sens_path, "w") as f:
        f.write("channel_index,channel_name,zero_ablation_mse\n")
        for idx, (name, mse) in enumerate(sensitivity_scores):
            f.write(f"{idx},{name},{mse:.10f}\n")
    logger.info(f"[verify/channel_sensitivity] saved: {sens_path}")


def log_trainable_parameter_summary(
    unet: torch.nn.Module,
    text_encoder1: torch.nn.Module,
    text_encoder2: torch.nn.Module,
    vae: torch.nn.Module,
    control_net: torch.nn.Module,
    material_lora_unet: torch.nn.Module,
    material_lora_control: torch.nn.Module,
) -> None:
    modules = {
        "unet": unet,
        "text_encoder1": text_encoder1,
        "text_encoder2": text_encoder2,
        "vae": vae,
        "control_net": control_net,
        "material_lora_unet": material_lora_unet,
        "material_lora_control": material_lora_control,
    }

    grand_total = 0
    grand_trainable = 0
    trainable_roots = set()
    for module_name, module in modules.items():
        total, trainable, top_level = _collect_module_param_counts(module.named_parameters())
        grand_total += total
        grand_trainable += trainable
        logger.info(f"[sanity/params] {module_name}: total={total:,} trainable={trainable:,}")
        for top_name, count in sorted(top_level.items()):
            trainable_roots.add(f"{module_name}.{top_name}")
            logger.info(f"[sanity/params] trainable-root {module_name}.{top_name}: {count:,}")

    logger.info(f"[sanity/params] total_params={grand_total:,}")
    logger.info(f"[sanity/params] trainable_params={grand_trainable:,}")
    logger.info(
        "[sanity/params] trainable_top_level_modules="
        + (", ".join(sorted(trainable_roots)) if trainable_roots else "<none>")
    )

def assert_trainable_policy(
    unet: torch.nn.Module,
    text_encoder1: torch.nn.Module,
    text_encoder2: torch.nn.Module,
    vae: torch.nn.Module,
    control_net: torch.nn.Module,
    material_lora_unet: torch.nn.Module,
    material_lora_control: torch.nn.Module,
    train_only_seam_adapter: bool = False,
) -> None:
    assert not any(parameter.requires_grad for parameter in unet.parameters()), "UNet must be frozen"
    assert not any(parameter.requires_grad for parameter in text_encoder1.parameters()), "Text encoder 1 must be frozen"
    assert not any(parameter.requires_grad for parameter in text_encoder2.parameters()), "Text encoder 2 must be frozen"
    assert not any(parameter.requires_grad for parameter in vae.parameters()), "VAE must be frozen"
    assert not any(parameter.requires_grad for parameter in material_lora_unet.parameters()), "Material LoRA UNet path must be frozen"
    assert not any(parameter.requires_grad for parameter in material_lora_control.parameters()), "Material LoRA ControlNet path must be frozen"

    trainable_names = [name for name, parameter in control_net.named_parameters() if parameter.requires_grad]
    assert trainable_names, "No trainable ControlNet parameters found"
    bad_trainable = [name for name in trainable_names if not name.startswith("controlnet_")]
    assert not bad_trainable, f"Unexpected non-controlnet trainable params: {bad_trainable[:5]}"

    if train_only_seam_adapter:
        seam_only_bad = [name for name in trainable_names if not name.startswith("controlnet_seam_adapter")]
        assert not seam_only_bad, f"Unexpected non-seam-adapter trainable params: {seam_only_bad[:5]}"
        seam_adapter_trainable_count = sum(
            parameter.numel()
            for name, parameter in control_net.named_parameters()
            if parameter.requires_grad and name.startswith("controlnet_seam_adapter")
        )
        assert seam_adapter_trainable_count > 0, "No trainable seam adapter parameters found"
        logger.info(f"[sanity/params] seam_adapter_trainable_params={seam_adapter_trainable_count:,}")
        return

    stem_trainable_count = sum(
        parameter.numel()
        for name, parameter in control_net.named_parameters()
        if parameter.requires_grad and "controlnet_cond_embedding.pre_blocks" in name
    )
    assert stem_trainable_count > 0, "Semantic pre-stem parameters are not trainable"
    logger.info(f"[sanity/params] semantic_pre_stem_trainable_params={stem_trainable_count:,}")


def _tensor_to_image(array: torch.Tensor) -> Image.Image:
    array = array.detach().float().clamp(-1.0, 1.0)
    array = (array + 1.0) * 0.5
    array = (array * 255.0).round().to(torch.uint8)
    array = array.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def _mask_overlay(image: Image.Image, mask: torch.Tensor, color=(255, 64, 64), alpha=0.45) -> Image.Image:
    image_arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask_arr = mask.detach().float().cpu().numpy()
    if mask_arr.ndim != 2:
        raise ValueError(f"expected 2D mask, got {mask_arr.shape}")
    mask_arr = np.clip(mask_arr, 0.0, 1.0)[..., None]
    color_arr = np.array(color, dtype=np.float32)[None, None, :]
    blended = image_arr * (1.0 - alpha * mask_arr) + color_arr * (alpha * mask_arr)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def _semantic_grid(
    semantic: torch.Tensor,
    channel_names: List[str],
    channel_ranges: Optional[List[Tuple[float, float]]] = None,
    tile_size: int = 192,
) -> Image.Image:
    channels = semantic.detach().float().cpu()
    rows, cols = 3, 4
    canvas = Image.new("RGB", (cols * tile_size, rows * tile_size), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    for idx in range(min(channels.shape[0], rows * cols)):
        channel = channels[idx]
        if channel_ranges is not None and idx < len(channel_ranges):
            cmin, cmax = channel_ranges[idx]
        else:
            cmin = float(channel.min().item())
            cmax = float(channel.max().item())

        if abs(cmax - cmin) < 1e-12:
            norm = torch.zeros_like(channel)
        else:
            norm = (channel - cmin) / (cmax - cmin)
        tile = (norm.clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
        tile_image = Image.fromarray(tile, mode="L").convert("RGB").resize((tile_size, tile_size), Image.Resampling.NEAREST)

        r = idx // cols
        c = idx % cols
        x = c * tile_size
        y = r * tile_size
        canvas.paste(tile_image, (x, y))
        label = channel_names[idx] if idx < len(channel_names) else f"ch{idx}"
        draw.rectangle((x, y, x + tile_size - 1, y + 22), fill=(0, 0, 0))
        draw.text((x + 4, y + 4), f"{idx}: {label}", fill=(255, 255, 255))

    return canvas


def _mask_to_image(mask: torch.Tensor) -> Image.Image:
    array = mask.detach().float().cpu().numpy()
    if array.ndim != 2:
        raise ValueError(f"expected 2D mask, got {array.shape}")
    array = np.clip(array, 0.0, 1.0)
    return Image.fromarray((array * 255.0).astype(np.uint8), mode="L")


def _save_seam_adapter_debug_board(
    output_dir: str,
    cond_image: torch.Tensor,
    seam_config: Dict[str, object],
    controlnet_diagnostics: Optional[Dict[str, object]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    payload = build_seam_local_adapter_maps_from_conditioning(
        cond_image.unsqueeze(0),
        seam_conditioning_offset=int(seam_config.get("seam_adapter_conditioning_offset", -1)),
        band_px=int(seam_config.get("seam_adapter_band_px", 0)),
        seam_adapter_per_edge=bool(seam_config.get("seam_adapter_per_edge", False)),
        seam_adapter_extrusion_mode=str(seam_config.get("seam_adapter_extrusion_mode", "decay")),
    )
    adapter_input = payload["adapter_input"][0].detach().float()
    if adapter_input.ndim == 4:
        active_mask = payload["active_mask"][0, 0].detach().float()
        invalid_mask = payload["invalid_active_mask"][0, 0].detach().float()
        projected_rgb = _tensor_to_image(adapter_input[:3])
        projected_alpha = _mask_to_image(adapter_input[3])
        distance_map = _mask_to_image(adapter_input[5])
        active_overlay = _mask_overlay(projected_rgb.copy(), active_mask, color=(96, 224, 255), alpha=0.40)
        invalid_overlay = _mask_overlay(projected_rgb.copy(), invalid_mask, color=(255, 96, 96), alpha=0.45)

        tiles = [
            ("projected_rgb", projected_rgb.convert("RGB")),
            ("projected_alpha", projected_alpha.convert("RGB")),
            ("distance_to_seam", distance_map.convert("RGB")),
            ("active_band", active_overlay.convert("RGB")),
            ("undefined_edges", invalid_overlay.convert("RGB")),
        ]
        tile_size = 256
        cols = len(tiles)
        canvas = Image.new("RGB", (cols * tile_size, tile_size), (18, 18, 18))
        draw = ImageDraw.Draw(canvas)
        for index, (label, tile) in enumerate(tiles):
            x = index * tile_size
            tile = tile.resize((tile_size, tile_size), Image.Resampling.NEAREST)
            canvas.paste(tile, (x, 0))
            draw.rectangle((x, 0, x + tile_size - 1, 24), fill=(0, 0, 0))
            draw.text((x + 6, 5), label, fill=(255, 255, 255))
        canvas.save(os.path.join(output_dir, "seam_adapter_input_legacy.png"))
        return

    per_edge_mask = payload["active_mask"][0].detach().float()
    invalid_mask = payload["invalid_active_mask"][0].detach().float()
    combined_active_mask = payload["combined_active_mask"][0, 0].detach().float()
    residual_energy_map = None
    if controlnet_diagnostics is not None and controlnet_diagnostics.get("seam_adapter_residual_energy_map") is not None:
        residual_energy_map = controlnet_diagnostics["seam_adapter_residual_energy_map"][0, 0].detach().float()

    tile_size = 256
    for edge_index, edge_name in enumerate(SEAM_ADAPTER_EDGE_NAMES):
        edge_input = adapter_input[edge_index]
        projected_rgb = _tensor_to_image(edge_input[:3])
        projected_alpha = _mask_to_image(edge_input[3])
        sobel_x_map = _mask_to_image(((edge_input[4] + 1.0) * 0.5).clamp(0.0, 1.0))
        sobel_y_map = _mask_to_image(((edge_input[5] + 1.0) * 0.5).clamp(0.0, 1.0))
        distance_map = _mask_to_image(edge_input[6])
        edge_valid = _mask_to_image(edge_input[7])
        active_band = _mask_to_image(edge_input[8])
        undefined_overlay = _mask_overlay(projected_rgb.copy(), invalid_mask[edge_index, 0], color=(255, 96, 96), alpha=0.45)

        tiles = [
            ("projected_rgb", projected_rgb.convert("RGB")),
            ("projected_alpha", projected_alpha.convert("RGB")),
            ("projected_sobel_x", sobel_x_map.convert("RGB")),
            ("projected_sobel_y", sobel_y_map.convert("RGB")),
            ("distance_to_seam", distance_map.convert("RGB")),
            ("active_band_mask", active_band.convert("RGB")),
            ("edge_valid", edge_valid.convert("RGB")),
            ("undefined_edge_mask", undefined_overlay.convert("RGB")),
        ]
        canvas = Image.new("RGB", (len(tiles) * tile_size, tile_size), (18, 18, 18))
        draw = ImageDraw.Draw(canvas)
        for tile_index, (label, tile) in enumerate(tiles):
            x = tile_index * tile_size
            tile = tile.resize((tile_size, tile_size), Image.Resampling.NEAREST)
            canvas.paste(tile, (x, 0))
            draw.rectangle((x, 0, x + tile_size - 1, 24), fill=(0, 0, 0))
            draw.text((x + 6, 5), label, fill=(255, 255, 255))
        canvas.save(os.path.join(output_dir, f"seam_adapter_input_{edge_name}.png"))

    summary_tiles = [
        ("north_mask", _mask_to_image(per_edge_mask[0, 0]).convert("RGB")),
        ("south_mask", _mask_to_image(per_edge_mask[1, 0]).convert("RGB")),
        ("east_mask", _mask_to_image(per_edge_mask[2, 0]).convert("RGB")),
        ("west_mask", _mask_to_image(per_edge_mask[3, 0]).convert("RGB")),
        ("combined_active_mask", _mask_to_image(combined_active_mask).convert("RGB")),
        (
            "residual_energy_map",
            _mask_to_image(
                (residual_energy_map / max(float(residual_energy_map.max().item()), 1e-6)).clamp(0.0, 1.0)
                if residual_energy_map is not None
                else torch.zeros_like(combined_active_mask)
            ).convert("RGB"),
        ),
    ]
    summary = Image.new("RGB", (len(summary_tiles) * tile_size, tile_size), (18, 18, 18))
    draw = ImageDraw.Draw(summary)
    for tile_index, (label, tile) in enumerate(summary_tiles):
        x = tile_index * tile_size
        tile = tile.resize((tile_size, tile_size), Image.Resampling.NEAREST)
        summary.paste(tile, (x, 0))
        draw.rectangle((x, 0, x + tile_size - 1, 24), fill=(0, 0, 0))
        draw.text((x + 6, 5), label, fill=(255, 255, 255))
    summary.save(os.path.join(output_dir, "seam_adapter_residual_summary.png"))

    block_diagnostics = (controlnet_diagnostics or {}).get("seam_adapter_block_diagnostics") or {}
    if block_diagnostics:
        def _extract_debug_map(value: Optional[torch.Tensor], default: torch.Tensor) -> torch.Tensor:
            if value is None:
                return torch.zeros_like(default)
            tensor = value.detach().float()
            while tensor.ndim > 2:
                tensor = tensor[0]
            if tensor.ndim != 2:
                return torch.zeros_like(default)
            return tensor

        def _normalized_debug_map(value: Optional[torch.Tensor], default: torch.Tensor) -> torch.Tensor:
            tensor = _extract_debug_map(value, default)
            peak = max(float(tensor.max().item()), 1e-6)
            return (tensor / peak).clamp(0.0, 1.0)

        block0_diag = block_diagnostics.get("block0", {})
        block1_diag = block_diagnostics.get("block1", {})
        block0_mask = _extract_debug_map(block0_diag.get("combined_active_mask"), combined_active_mask)
        block1_mask = _extract_debug_map(block1_diag.get("combined_active_mask"), combined_active_mask)
        block0_invalid = _extract_debug_map(block0_diag.get("combined_invalid_mask"), combined_active_mask)
        block1_invalid = _extract_debug_map(block1_diag.get("combined_invalid_mask"), combined_active_mask)

        multi_tiles = [
            ("block0_residual", _mask_to_image(_normalized_debug_map(block0_diag.get("residual_energy_map"), combined_active_mask)).convert("RGB")),
            ("block1_residual", _mask_to_image(_normalized_debug_map(block1_diag.get("residual_energy_map"), combined_active_mask)).convert("RGB")),
            ("block0_mask", _mask_to_image(block0_mask).convert("RGB")),
            ("block1_mask", _mask_to_image(block1_mask).convert("RGB")),
            ("block0_undefined", _mask_to_image(block0_invalid).convert("RGB")),
            ("block1_undefined", _mask_to_image(block1_invalid).convert("RGB")),
        ]
        multi_summary = Image.new("RGB", (len(multi_tiles) * tile_size, tile_size), (18, 18, 18))
        draw = ImageDraw.Draw(multi_summary)
        for tile_index, (label, tile) in enumerate(multi_tiles):
            x = tile_index * tile_size
            tile = tile.resize((tile_size, tile_size), Image.Resampling.NEAREST)
            multi_summary.paste(tile, (x, 0))
            draw.rectangle((x, 0, x + tile_size - 1, 24), fill=(0, 0, 0))
            draw.text((x + 6, 5), label, fill=(255, 255, 255))
        multi_summary.save(os.path.join(output_dir, "seam_adapter_multi_inject_summary.png"))


def _select_debug_sample_indices(sample_count: int, dataset: TerrainSemanticManifestDataset) -> List[int]:
    indices = list(range(min(sample_count, len(dataset))))
    if sample_count < 3 or len(dataset) <= sample_count:
        return indices

    edge_channel_indices = [i for i, spec in enumerate(dataset.channel_specs) if spec.atlas_name == "edge"]
    interior_channel_indices = [i for i, spec in enumerate(dataset.channel_specs) if spec.atlas_name == "interior"]
    atlas_c_indices = [
        i for i, spec in enumerate(dataset.channel_specs) if spec.atlas_name == "interior" and spec.channel_name == "A"
    ]
    if not interior_channel_indices:
        return indices

    terrain_mask_indices = [i for i, spec in enumerate(dataset.channel_specs) if spec.name == "terrain_mask"]
    terrain_mask_index = terrain_mask_indices[0] if terrain_mask_indices else 3

    search_limit = min(len(dataset), max(256, sample_count * 64))
    strict_search_limit = len(dataset)
    reserved = set(indices[:-1])

    full_interior_index = None
    full_interior_stats = (0.0, 0.0, 0.0)

    # Prefer a strict 100% interior sample where terrain_mask is zero everywhere.
    for index in range(strict_search_limit):
        if index in reserved:
            continue

        sample = dataset[index]
        semantic = sample["conditioning_images"].detach().float()
        terrain_mask = semantic[terrain_mask_index]
        if torch.count_nonzero(terrain_mask).item() == 0:
            edge_strength = (
                semantic[edge_channel_indices].abs().mean().item() if edge_channel_indices else 0.0
            )
            interior_strength = semantic[interior_channel_indices].abs().mean().item()
            atlas_c_strength = (
                semantic[atlas_c_indices].mean().item() if atlas_c_indices else interior_strength
            )
            full_interior_index = index
            full_interior_stats = (edge_strength, interior_strength, atlas_c_strength)
            break

    if full_interior_index is not None:
        indices[-1] = full_interior_index
        logger.info(
            "[sanity/debug_dump] full_interior_index="
            f"{full_interior_index} edge_mean={full_interior_stats[0]:.6f} "
            f"interior_mean={full_interior_stats[1]:.6f} atlas_c_mean={full_interior_stats[2]:.6f}"
        )
        return indices

    logger.warning(
        "[sanity/debug_dump] no strict full-interior sample found (terrain_mask all zeros) "
        f"after scanning {strict_search_limit} samples; using interior-dominant fallback"
    )

    best_index = None
    best_score = -float("inf")
    best_stats = (0.0, 0.0, 0.0)

    for index in range(search_limit):
        if index in reserved:
            continue

        sample = dataset[index]
        semantic = sample["conditioning_images"].detach().float()

        edge_strength = (
            semantic[edge_channel_indices].abs().mean().item() if edge_channel_indices else 0.0
        )
        interior_strength = semantic[interior_channel_indices].abs().mean().item()
        atlas_c_strength = (
            semantic[atlas_c_indices].mean().item() if atlas_c_indices else interior_strength
        )
        score = (2.0 * atlas_c_strength) + interior_strength - edge_strength

        if score > best_score:
            best_score = score
            best_index = index
            best_stats = (edge_strength, interior_strength, atlas_c_strength)

    if best_index is not None:
        indices[-1] = best_index
        logger.info(
            "[sanity/debug_dump] interior_dominant_index="
            f"{best_index} edge_mean={best_stats[0]:.6f} interior_mean={best_stats[1]:.6f} atlas_c_mean={best_stats[2]:.6f}"
        )

    return indices


def write_debug_alignment_dump(args: argparse.Namespace, dataset: TerrainSemanticManifestDataset) -> None:
    sample_count = min(args.debug_dump_samples, len(dataset))
    if sample_count <= 0:
        return

    out_dir = os.path.join(args.output_dir, "sanity", "debug_dump")
    os.makedirs(out_dir, exist_ok=True)

    sample_indices = _select_debug_sample_indices(sample_count, dataset)
    logger.info(f"[sanity/debug_dump] selected_indices={sample_indices}")

    for out_idx, sample_index in enumerate(sample_indices):
        sample = dataset[sample_index]
        image = _tensor_to_image(sample["images"])
        trusted_mask = sample["trusted_mask"]
        trusted_overlay = _mask_overlay(image, trusted_mask)

        latent_mask = F.interpolate(
            trusted_mask.unsqueeze(0).unsqueeze(0),
            size=(trusted_mask.shape[0] // 8, trusted_mask.shape[1] // 8),
            mode="area",
        )
        latent_mask_small = latent_mask.squeeze(0).squeeze(0)
        latent_mask_up = F.interpolate(latent_mask, size=trusted_mask.shape[-2:], mode="nearest").squeeze(0).squeeze(0)
        latent_overlay = _mask_overlay(image, latent_mask_up, color=(80, 160, 255), alpha=0.45)

        channel_ranges = [tuple(spec.semantic_range) for spec in dataset.channel_specs]
        semantic_grid = _semantic_grid(sample["conditioning_images"], sample["channel_names"], channel_ranges)
        latent_mask_small_image = _mask_to_image(latent_mask_small).resize(
            (trusted_mask.shape[1], trusted_mask.shape[0]), Image.Resampling.NEAREST
        )
        latent_mask_up_image = _mask_to_image(latent_mask_up)

        prefix = f"sample_{out_idx:02d}_{sample['image_name']}_idx{sample_index:04d}"
        image.save(os.path.join(out_dir, f"{prefix}_target.png"))
        trusted_overlay.save(os.path.join(out_dir, f"{prefix}_trusted_overlay.png"))
        latent_overlay.save(os.path.join(out_dir, f"{prefix}_latent_overlay.png"))
        latent_overlay.save(os.path.join(out_dir, f"{prefix}_latent_mask_after_vae_downsample_overlay.png"))
        latent_mask_small_image.save(os.path.join(out_dir, f"{prefix}_latent_mask_after_vae_downsample_grid.png"))
        latent_mask_up_image.save(os.path.join(out_dir, f"{prefix}_latent_mask_after_vae_downsample_upsampled.png"))
        semantic_grid.save(os.path.join(out_dir, f"{prefix}_semantic_grid.png"))

    logger.info(f"[sanity/debug_dump] wrote {sample_count} samples to {out_dir}")


def run_material_lora_sanity_check(
    args: argparse.Namespace,
    dataset: TerrainSemanticManifestDataset,
    unet: torch.nn.Module,
    control_net: torch.nn.Module,
    material_lora_unet: torch.nn.Module,
    material_lora_control: torch.nn.Module,
    vae,
    noise_scheduler: DDPMScheduler,
    cached_text: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    seam_config: Dict[str, object],
):
    sanity_dir = os.path.join(args.output_dir, "sanity")
    os.makedirs(sanity_dir, exist_ok=True)

    unet_dtype = next(unet.parameters()).dtype
    control_dtype = next(control_net.parameters()).dtype

    sample = dataset[0]
    sample_batch: Dict[str, object] = {
        "conditioning_images": sample["conditioning_images"].unsqueeze(0),
        "seam_strip_tensor": sample.get("seam_strip_tensor").unsqueeze(0) if sample.get("seam_strip_tensor") is not None else None,
        "edge_defined_flags": sample.get("edge_defined_flags").unsqueeze(0) if sample.get("edge_defined_flags") is not None else None,
        "edge_flag_maps": sample.get("edge_flag_maps").unsqueeze(0) if sample.get("edge_flag_maps") is not None else None,
    }
    conditioning, _, _ = build_model_visible_conditioning(
        sample_batch,
        dataset,
        seam_config,
        device,
        torch.float32,
    )
    target_size = sample["target_sizes_hw"].tolist()
    latent_h = target_size[0] // 8
    latent_w = target_size[1] // 8

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 17)
    latents = torch.randn((1, 4, latent_h, latent_w), generator=generator, device=device, dtype=torch.float32)

    encoder_hidden_states1, encoder_hidden_states2, pool2 = cached_text
    text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(dtype=torch.float32)
    size_embeddings = sdxl_train_util.get_size_embeddings(
        sample["original_sizes_hw"].unsqueeze(0),
        sample["crop_top_lefts"].unsqueeze(0),
        sample["target_sizes_hw"].unsqueeze(0),
        device,
    ).to(torch.float32)
    vector_embedding = torch.cat([pool2, size_embeddings], dim=1).to(torch.float32)

    vae_original_device = next(vae.parameters()).device
    vae_original_dtype = next(vae.parameters()).dtype

    vae.to(device=device, dtype=dtype)

    def predict_eps(scale: float) -> torch.Tensor:
        material_lora_unet.set_multiplier(scale)
        material_lora_control.set_multiplier(scale)
        with torch.no_grad():
            inference_steps = max(2, args.lora_sanity_steps)
            noise_scheduler.set_timesteps(inference_steps, device=device)
            noisy = latents.clone()
            for timestep in noise_scheduler.timesteps:
                t = timestep.expand(1).to(device=device, dtype=torch.long)
                noisy_control = noisy.to(dtype=control_dtype)
                cond_control = conditioning.to(dtype=control_dtype)
                text_control = text_embedding.to(dtype=control_dtype)
                vector_control = vector_embedding.to(dtype=control_dtype)
                input_resi_add, mid_add = control_net(noisy_control, t, text_control, vector_control, cond_control)

                noisy_unet = noisy.to(dtype=unet_dtype)
                text_unet = text_embedding.to(dtype=unet_dtype)
                vector_unet = vector_embedding.to(dtype=unet_dtype)
                input_resi_add = [x.to(dtype=unet_dtype) for x in input_resi_add]
                mid_add = mid_add.to(dtype=unet_dtype)
                eps = unet(noisy_unet, t, text_unet, vector_unet, input_resi_add, mid_add)
                noisy = noise_scheduler.step(eps, timestep, noisy).prev_sample

            alpha_t = noise_scheduler.alphas_cumprod[noise_scheduler.timesteps[-1]].to(device=device, dtype=torch.float32)
            alpha_t = alpha_t.view(1, 1, 1, 1)
            pred_x0 = noisy / alpha_t.sqrt()
        return pred_x0

    pred_base = predict_eps(0.0)
    pred_lora = predict_eps(args.material_lora_multiplier)
    mse_diff = F.mse_loss(pred_base.float(), pred_lora.float()).item()
    logger.info(f"[sanity/lora] base_vs_lora_pred_x0_mse={mse_diff:.8f}")

    with torch.no_grad():
        vae_dtype = next(vae.parameters()).dtype
        base_latents = (pred_base / sdxl_model_util.VAE_SCALE_FACTOR).to(dtype=vae_dtype)
        lora_latents = (pred_lora / sdxl_model_util.VAE_SCALE_FACTOR).to(dtype=vae_dtype)
        base_image = vae.decode(base_latents).sample[0]
        lora_image = vae.decode(lora_latents).sample[0]

    _tensor_to_image(base_image).save(os.path.join(sanity_dir, "material_lora_base_off.png"))
    _tensor_to_image(lora_image).save(os.path.join(sanity_dir, "material_lora_on.png"))
    logger.info(f"[sanity/lora] saved previews: {sanity_dir}")
    if mse_diff < 1e-6:
        raise RuntimeError("Runtime LoRA sanity check failed: base/off and lora/on outputs are effectively identical")

    vae.to(device=vae_original_device, dtype=vae_original_dtype)


def run_startup_sanity_report(
    args: argparse.Namespace,
    dataset: TerrainSemanticManifestDataset,
    alpha_config: Dict[str, object],
) -> None:
    audit = dataset.manifest_audit
    logger.info("[sanity/data] manifest_audit=" + ", ".join([f"{key}={value}" for key, value in audit.items()]))

    if alpha_config["enabled"]:
        alpha_summary = dataset.alpha_source_summary
        logger.info(
            "[sanity/alpha] coverage="
            + f"native={int(alpha_summary['native_alpha_count'])} "
            + f"synthesized={int(alpha_summary['synthesized_opaque_count'])} "
            + f"native_fraction={alpha_summary['native_alpha_fraction']:.4f} "
            + f"synthesized_fraction={alpha_summary['synthesized_opaque_fraction']:.4f}"
        )
        if alpha_summary["synthesized_opaque_fraction"] > alpha_config["synthesized_alpha_warn_fraction"]:
            logger.warning(
                "[sanity/alpha] synthesized opaque alpha covers a large fraction of the dataset; "
                "occupancy supervision may be weak"
            )

    class_counts = Counter([record["crop_size_class"] for record in dataset.records])
    full_scene_count = sum(1 for record in dataset.records if record["generation_strategy"] == "full_scene")
    weights = dataset.sampling_weights
    logger.info(
        "[sanity/data] sample_weight_stats="
        + f"min={weights.min().item():.4f} mean={weights.mean().item():.4f} max={weights.max().item():.4f}"
    )
    logger.info(
        "[sanity/data] crop_size_class_counts="
        + ", ".join([f"{key}:{value}" for key, value in sorted(class_counts.items())])
    )
    logger.info(f"[sanity/data] full_scene_count={full_scene_count}")
    logger.info(
        f"[sanity/data] interpolation image={dataset.image_resize_mode} semantic={dataset.semantic_resize_mode} trusted_mask=area"
    )

    max_samples = min(args.sanity_samples, len(dataset))
    decode_issue_count = 0
    skipped_under_threshold = 0
    channel_stats = {
        index: {"min": float("inf"), "max": float("-inf"), "sum": 0.0, "sum_sq": 0.0, "count": 0}
        for index in range(len(dataset.channel_names))
    }
    trusted_coverages: List[float] = []
    atlas_mode_counts = Counter()
    atlas_dtype_counts = Counter()
    atlas_channel_counts = Counter()

    for index in range(max_samples):
        record = dataset.records[index]
        for atlas_key in ["base_atlas_path", "edge_atlas_path", "interior_atlas_path"]:
            path = record[atlas_key]
            with Image.open(path) as atlas_image:
                atlas_array = torch.from_numpy(np.array(atlas_image))
                atlas_mode_counts[f"{atlas_key}:{atlas_image.mode}"] += 1
                atlas_dtype_counts[f"{atlas_key}:{atlas_array.numpy().dtype}"] += 1
                channels = 1 if atlas_array.ndim == 2 else atlas_array.shape[2]
                atlas_channel_counts[f"{atlas_key}:{channels}"] += 1

        try:
            sample = dataset[index]
        except Exception as exc:
            decode_issue_count += 1
            logger.warning(f"[sanity/data] decode_issue index={index}: {exc}")
            continue

        coverage = float(sample["trusted_mask"].mean().item())
        trusted_coverages.append(coverage)
        if coverage < dataset.min_trusted_mask_ratio:
            skipped_under_threshold += 1

        conditioning = sample["conditioning_images"]
        for channel_index in range(conditioning.shape[0]):
            channel = conditioning[channel_index]
            stats = channel_stats[channel_index]
            stats["min"] = min(stats["min"], float(channel.min().item()))
            stats["max"] = max(stats["max"], float(channel.max().item()))
            stats["sum"] += float(channel.sum().item())
            stats["sum_sq"] += float((channel * channel).sum().item())
            stats["count"] += int(channel.numel())

    logger.info(f"[sanity/data] decode_issue_count={decode_issue_count}")
    logger.info(f"[sanity/data] under_threshold_after_resize_count={skipped_under_threshold}")
    if trusted_coverages:
        logger.info(
            "[sanity/data] trusted_coverage_stats="
            + f"min={min(trusted_coverages):.6f} mean={sum(trusted_coverages)/len(trusted_coverages):.6f} max={max(trusted_coverages):.6f}"
        )

    for channel_index, channel_name in enumerate(dataset.channel_names):
        stats = channel_stats[channel_index]
        if stats["count"] == 0:
            logger.info(f"[sanity/channels] {channel_index}:{channel_name} no_samples")
            continue
        mean = stats["sum"] / stats["count"]
        var = max(stats["sum_sq"] / stats["count"] - mean * mean, 0.0)
        std = var ** 0.5
        logger.info(
            "[sanity/channels] "
            + f"{channel_index}:{channel_name} min={stats['min']:.6f} max={stats['max']:.6f} mean={mean:.6f} std={std:.6f}"
        )

    logger.info("[sanity/atlas] mode_counts=" + ", ".join([f"{key}={value}" for key, value in sorted(atlas_mode_counts.items())]))
    logger.info("[sanity/atlas] dtype_counts=" + ", ".join([f"{key}={value}" for key, value in sorted(atlas_dtype_counts.items())]))
    logger.info(
        "[sanity/atlas] channel_counts=" + ", ".join([f"{key}={value}" for key, value in sorted(atlas_channel_counts.items())])
    )


def train(args: argparse.Namespace) -> None:
    semantic_config = load_semantic_config(args.semantic_config)
    alpha_config = parse_alpha_config(semantic_config)
    verification_config = parse_verification_config(semantic_config)
    conditioning_config = parse_conditioning_config(semantic_config)
    seam_config = parse_seam_config(semantic_config)
    evaluation_config = parse_evaluation_config(semantic_config, alpha_config, args.output_name, args.max_train_steps)
    if args.eval_steps_csv:
        override_steps = [step for step in _parse_steps_csv(args.eval_steps_csv) if step > 0 and step <= int(args.max_train_steps)]
        evaluation_config["eval_steps"] = override_steps
        logger.info(f"[eval/config] override eval_steps via --eval_steps_csv: {override_steps}")
    binding_config = parse_binding_eval_config(semantic_config, alpha_config)
    training_expanded_supervision_enabled = bool(verification_config.get("train_expanded_supervision_enabled", False))
    training_expanded_halo_px = int(verification_config.get("train_expanded_halo_px", 0))
    if (
        not training_expanded_supervision_enabled
        and bool(evaluation_config.get("expanded_prediction_enabled", False))
        and int(evaluation_config.get("expanded_halo_px", 0)) > 0
    ):
        # Keep training and eval geometry consistent unless explicitly disabled.
        training_expanded_supervision_enabled = True
        if training_expanded_halo_px <= 0:
            training_expanded_halo_px = int(evaluation_config.get("expanded_halo_px", 0))

    if args.seed is None:
        args.seed = random.randint(0, 2**32)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        if bool(args.enable_tf32):
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
        if bool(args.enable_cudnn_benchmark):
            torch.backends.cudnn.benchmark = True

    if args.bind_pair_seed is not None:
        alpha_config["bind_pair_seed"] = args.bind_pair_seed
    bind_pair_rng = random.Random(alpha_config["bind_pair_seed"])
    logger.info(f"[seed] train_seed={args.seed} bind_pair_seed={alpha_config['bind_pair_seed']}")
    bind_pair_batch_count = 0

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    weight_dtype = resolve_weight_dtype(args)
    save_dtype = resolve_save_dtype(args.save_dtype)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    dataset = build_dataset(args, semantic_config, alpha_config, seam_config=seam_config)
    if bool(evaluation_config.get("expanded_prediction_enabled", False)) and int(evaluation_config.get("expanded_halo_px", 0)) > 0:
        logger.warning(
            "[seam/geometry] eval uses expanded prediction; training expanded supervision is "
            + ("enabled" if training_expanded_supervision_enabled else "disabled")
        )
    logger.info(
        "[seam/geometry] "
        + f"training_expanded_supervision_enabled={'yes' if training_expanded_supervision_enabled else 'no'} "
        + f"training_expanded_halo_px={int(training_expanded_halo_px)} "
        + f"eval_expanded_prediction_enabled={'yes' if evaluation_config.get('expanded_prediction_enabled', False) else 'no'} "
        + f"eval_expanded_halo_px={int(evaluation_config.get('expanded_halo_px', 0))}"
    )
    training_prompt_pool = load_training_prompt_pool(semantic_config)
    training_prompt_sampler = None
    if training_prompt_pool:
        prompt_seed = args.seed if args.seed is not None else int(alpha_config.get("bind_pair_seed", 1337))
        training_prompt_sampler = TrainingPromptSampler(training_prompt_pool, prompt_seed)
        logger.info(
            "[sanity/prompt_pool] enabled count=%d seed=%d weights=%s",
            len(training_prompt_pool),
            prompt_seed,
            summarize_training_prompt_pool(training_prompt_pool),
        )
        for prompt_spec in training_prompt_pool:
            logger.info(
                "[sanity/prompt_pool] prompt name=%s mode=%s weight=%.4f prompt='%s' prompt2='%s'",
                prompt_spec.name,
                prompt_spec.mode,
                prompt_spec.weight,
                prompt_spec.prompt,
                prompt_spec.prompt2,
            )
    resolved_eval_samples = []
    eval_output_dir = os.path.join(args.output_dir, "eval")
    use_ema = args.ema_decay > 0.0
    eval_output_dir_raw = os.path.join(eval_output_dir, "raw") if use_ema and args.ema_eval_at_anchors else eval_output_dir
    eval_output_dir_ema = os.path.join(eval_output_dir, "ema") if use_ema and args.ema_eval_at_anchors else None
    eval_step_summaries: Dict[str, Dict[str, float]] = {}
    if evaluation_config["enabled"]:
        resolved_eval_samples = resolve_eval_samples(
            dataset,
            evaluation_config["eval_manifest_path"],
            evaluation_config["max_samples"],
        )
        logger.info(
            "[eval/config] "
            + f"samples={len(resolved_eval_samples)} steps={evaluation_config['eval_steps']} seeds={evaluation_config['seeds']} "
            + f"manifest={evaluation_config['eval_manifest_path']}"
        )
        for sample_info in resolved_eval_samples:
            logger.info(
                "[eval/sample] "
                + f"eval_id={sample_info.eval_id} category={sample_info.category} sample_key={sample_info.sample_key} "
                + f"dataset_index={sample_info.dataset_index} image_name={sample_info.image_name}"
            )
        resolved_swap_pairs = []
        if binding_config["enabled"]:
            resolved_swap_pairs = resolve_swap_pairs(dataset, binding_config["swap_manifest_path"])
            logger.info(
                "[binding_eval/config] "
                + f"pairs={len(resolved_swap_pairs)} manifest={binding_config['swap_manifest_path']}"
            )
            for sp in resolved_swap_pairs:
                logger.info(
                    "[binding_eval/pair] "
                    + f"pair_id={sp.pair_id} base={sp.base_image} swap={sp.swap_image} type={sp.edit_type}"
                )
    _run_pre_gate_target_terrain_iou_sanity(dataset, alpha_config, resolved_eval_samples)
    run_startup_sanity_report(args, dataset, alpha_config)
    write_debug_alignment_dump(args, dataset)
    sampler = WeightedRandomSampler(dataset.sampling_weights, num_samples=len(dataset), replacement=True)
    dataloader_kwargs: Dict[str, object] = {
        "batch_size": args.train_batch_size,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "collate_fn": semantic_collate,
        "drop_last": True,
        "pin_memory": bool(args.dataloader_pin_memory),
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = bool(args.dataloader_persistent_workers)
        if int(args.dataloader_prefetch_factor) > 0:
            dataloader_kwargs["prefetch_factor"] = int(args.dataloader_prefetch_factor)
    dataloader = DataLoader(
        dataset,
        **dataloader_kwargs,
    )

    tokenize_strategy = strategy_sdxl.SdxlTokenizeStrategy(args.max_token_length, args.tokenizer_cache_dir)
    text_encoding_strategy = strategy_sdxl.SdxlTextEncodingStrategy()

    (
        _,
        text_encoder1,
        text_encoder2,
        vae,
        unet,
        _,
        _,
    ) = sdxl_train_util.load_target_model(args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype)

    unet_sd = unet.state_dict()
    unet = SdxlControlledUNet()
    unet.load_state_dict(unet_sd, strict=True)
    del unet_sd

    selected_alpha_head_indices = (
        _resolve_alpha_head_indices(
            alpha_config["head_scales"],
            alpha_config["head_mode"],
            int(alpha_config["single_scale_index"]),
        )
        if alpha_config["enabled"]
        else []
    )
    terrain_mask_index_for_baseline = _find_channel_index(dataset.channel_names, "terrain_mask") if alpha_config["enabled"] else 3
    seam_config["seam_adapter_conditioning_offset"] = len(dataset.channel_names) if seam_config["enabled"] else -1
    control_net = SdxlControlNet(
        conditioning_channels=dataset.full_conditioning_channels if seam_config["enabled"] else len(dataset.channel_names),
        conditioning_pre_embed_channels=[32, 32],
        alpha_head_indices=selected_alpha_head_indices,
        alpha_baseline_mode=alpha_config["baseline_mode"] if alpha_config["enabled"] else "none",
        alpha_baseline_terrain_channel_index=terrain_mask_index_for_baseline,
        seam_adapter_enabled=seam_config["enabled"] and bool(seam_config.get("seam_adapter_enabled", False)),
        seam_adapter_band_px=int(seam_config.get("seam_adapter_band_px", 64)),
        seam_adapter_scale=float(seam_config.get("seam_adapter_scale", 1.0)),
        seam_adapter_zero_init=bool(seam_config.get("seam_adapter_zero_init", True)),
        seam_adapter_target=str(seam_config.get("seam_adapter_target", "first_high_res")),
        seam_adapter_multi_inject=bool(seam_config.get("seam_adapter_multi_inject", False)),
        seam_adapter_injection_blocks=list(seam_config.get("seam_adapter_injection_blocks", ["first_high_res"])),
        seam_adapter_scale_block0=float(seam_config.get("seam_adapter_scale_block0", 1.0)),
        seam_adapter_scale_block1=float(seam_config.get("seam_adapter_scale_block1", 0.5)),
        seam_adapter_multi_inject_mode=str(seam_config.get("seam_adapter_multi_inject_mode", "shared_trunk_per_block_heads")),
        seam_adapter_conditioning_offset=int(seam_config.get("seam_adapter_conditioning_offset", -1)),
        seam_adapter_per_edge=bool(seam_config.get("seam_adapter_per_edge", False)),
        seam_adapter_extrusion_mode=str(seam_config.get("seam_adapter_extrusion_mode", "decay")),
    )
    if args.controlnet_multiplier is not None:
        control_net.multiplier = float(args.controlnet_multiplier)
        logger.info(f"[config] controlnet_multiplier_override={control_net.multiplier:.4f}")
    if args.controlnet_model_name_or_path:
        ckpt_path = args.controlnet_model_name_or_path
        if os.path.isdir(ckpt_path):
            # Allow passing an Accelerator state directory directly.
            candidates = [
                os.path.join(ckpt_path, "ema_state.safetensors"),
                os.path.join(ckpt_path, "controlnet.safetensors"),
                os.path.join(ckpt_path, "model.safetensors"),
                os.path.join(ckpt_path, "pytorch_model.bin"),
            ]
            resolved = next((p for p in candidates if os.path.isfile(p)), None)
            if resolved is None:
                raise FileNotFoundError(
                    f"No supported checkpoint artifact found in directory: {ckpt_path}. "
                    "Expected one of ema_state.safetensors, controlnet.safetensors, model.safetensors, pytorch_model.bin"
                )
            ckpt_path = resolved
            logger.info(f"[config] resolved controlnet checkpoint file: {ckpt_path}")

        if os.path.splitext(ckpt_path)[1] == ".safetensors":
            state_dict = load_safetensors(ckpt_path)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")

        # Seam migration support: skip keys with incompatible tensor shapes.
        model_sd = control_net.state_dict()
        filtered_state_dict = {}
        skipped_shape_keys: List[str] = []
        for key, value in state_dict.items():
            if key in model_sd and hasattr(value, "shape") and hasattr(model_sd[key], "shape"):
                if tuple(value.shape) != tuple(model_sd[key].shape):
                    skipped_shape_keys.append(key)
                    continue
            filtered_state_dict[key] = value

        if skipped_shape_keys:
            logger.warning(
                "[config] skipped incompatible checkpoint tensors for ControlNet init: "
                + ", ".join(skipped_shape_keys[:8])
                + (" ..." if len(skipped_shape_keys) > 8 else "")
            )

        strict_load = not alpha_config["enabled"] and not (
            seam_config["enabled"]
            and bool(seam_config.get("seam_adapter_enabled", False))
            and bool(seam_config.get("seam_adapter_per_edge", False))
        )
        info = control_net.load_state_dict(filtered_state_dict, strict=strict_load)
        if alpha_config["enabled"] or not strict_load:
            allowed_prefixes: Tuple[str, ...] = tuple()
            if alpha_config["enabled"]:
                allowed_prefixes = allowed_prefixes + ("controlnet_alpha_heads", "controlnet_alpha_baseline_heads")
            if seam_config["enabled"] and alpha_config["enabled"]:
                allowed_prefixes = allowed_prefixes + ("controlnet_cond_embedding",)
            if seam_config["enabled"] and seam_config.get("seam_adapter_enabled", False):
                allowed_prefixes = allowed_prefixes + ("controlnet_seam_adapter",)
            unexpected = [key for key in info.unexpected_keys if not key.startswith(allowed_prefixes)]
            missing = [key for key in info.missing_keys if not key.startswith(allowed_prefixes)]
            if unexpected or missing:
                raise RuntimeError(
                    f"unexpected checkpoint mismatch with ControlNet init: missing={missing[:5]} unexpected={unexpected[:5]}"
                )
    else:
        control_net.init_from_unet(unet)

    text_encoders = [text_encoder1, text_encoder2]
    material_lora_unet, material_lora_control = apply_runtime_material_lora(
        text_encoders,
        unet,
        control_net,
        args.material_lora_weights,
        args.material_lora_multiplier,
    )

    unet.requires_grad_(False)
    control_net.requires_grad_(False)
    text_encoder1.requires_grad_(False)
    text_encoder2.requires_grad_(False)
    vae.requires_grad_(False)
    material_lora_unet.requires_grad_(False)
    material_lora_control.requires_grad_(False)

    trainable_params = []
    trainable_prefix = "controlnet_seam_adapter" if args.train_only_seam_adapter else "controlnet_"
    for name, parameter in control_net.named_parameters():
        if name.startswith(trainable_prefix):
            parameter.requires_grad = True
            trainable_params.append(parameter)

    cond_embedding_params = [
        parameter for name, parameter in control_net.named_parameters()
        if parameter.requires_grad and "controlnet_cond_embedding" in name
    ]
    seam_adapter_params = [
        parameter for name, parameter in control_net.named_parameters()
        if parameter.requires_grad and "controlnet_seam_adapter" in name
    ]
    other_trainable_params = [
        parameter for name, parameter in control_net.named_parameters()
        if parameter.requires_grad
        and "controlnet_cond_embedding" not in name
        and "controlnet_seam_adapter" not in name
    ]

    assert_trainable_policy(
        unet,
        text_encoder1,
        text_encoder2,
        vae,
        control_net,
        material_lora_unet,
        material_lora_control,
        train_only_seam_adapter=args.train_only_seam_adapter,
    )
    log_trainable_parameter_summary(
        unet,
        text_encoder1,
        text_encoder2,
        vae,
        control_net,
        material_lora_unet,
        material_lora_control,
    )

    cond_lr = args.learning_rate * conditioning_config["cond_embedding_lr_multiplier"]
    seam_adapter_lr = args.learning_rate * args.seam_adapter_lr_multiplier
    optimizer_param_groups = []
    if cond_embedding_params:
        optimizer_param_groups.append({"params": cond_embedding_params, "lr": cond_lr})
    if seam_adapter_params:
        optimizer_param_groups.append({"params": seam_adapter_params, "lr": seam_adapter_lr})
    if other_trainable_params:
        optimizer_param_groups.append({"params": other_trainable_params, "lr": args.learning_rate})

    optimizer = torch.optim.AdamW(optimizer_param_groups, betas=(0.9, 0.999), weight_decay=1e-2)
    logger.info(
        f"[optimizer] cond_embedding_lr={cond_lr:.2e} (×{conditioning_config['cond_embedding_lr_multiplier']}) "
        f"seam_adapter_lr={seam_adapter_lr:.2e} (×{args.seam_adapter_lr_multiplier}) "
        f"other_lr={args.learning_rate:.2e} cond_params={len(cond_embedding_params)} "
        f"seam_adapter_params={len(seam_adapter_params)} other_params={len(other_trainable_params)} "
        f"train_only_seam_adapter={args.train_only_seam_adapter}"
    )
    verify_optimizer_parameter_membership(optimizer, control_net)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        control_net.enable_gradient_checkpointing()
    if args.xformers:
        unet.set_use_memory_efficient_attention(True, False)
        control_net.set_use_memory_efficient_attention(True, False)
    elif args.sdpa:
        unet.set_use_sdpa(True)
        control_net.set_use_sdpa(True)

    unet.to(accelerator.device, dtype=weight_dtype)
    # Keep trainable ControlNet params in fp32; amp autocast handles compute casting.
    control_net.to(accelerator.device, dtype=torch.float32)
    text_encoder1.to(accelerator.device)
    text_encoder2.to(accelerator.device)
    material_lora_unet.to(accelerator.device, dtype=weight_dtype)
    material_lora_control.to(accelerator.device, dtype=torch.float32)

    unet.eval()
    text_encoder1.eval()
    text_encoder2.eval()
    material_lora_unet.eval()
    material_lora_control.eval()
    control_net.train()

    needs_vae_decode_during_train = (
        seam_config["enabled"] and float(seam_config.get("rgb_recon_loss_weight", 0.0)) > 0.0
    )
    if needs_vae_decode_during_train:
        if hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        logger.info("[sanity/latents] enabled VAE slicing+tiling for seam RGB reconstruction decode")

    if args.cache_latents:
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        dataset.cache_latents(vae, accelerator.device, vae_dtype, batch_size=args.train_batch_size)
        cache_report = dataset.latent_cache_report
        logger.info(
            "[sanity/latents] cache_summary="
            + f"total={cache_report['total']} "
            + f"in_memory_hits={cache_report['in_memory_hits']} "
            + f"disk_hits={cache_report['disk_hits']} "
            + f"encoded={cache_report['encoded']} "
            + f"disk_misses={cache_report['disk_misses']} "
            + f"cache_dir={cache_report['cache_dir'] or '<none>'}"
        )
        if needs_vae_decode_during_train:
            vae.to(accelerator.device, dtype=vae_dtype)
            vae.requires_grad_(False)
            vae.eval()
            logger.info("[sanity/latents] retaining VAE on accelerator for seam RGB reconstruction decode")
        else:
            vae.to("cpu")
            clean_memory_on_device(accelerator.device)
    else:
        if args.latent_cache_dir:
            logger.info(
                f"[sanity/latents] startup_precompute=disabled cache_dir={args.latent_cache_dir} mode=lazy_read"
            )
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()

    text_conditioning_cache: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    logged_prompt_names = set()
    cached_text = get_or_prepare_text_conditioning(
        text_conditioning_cache,
        dataset.prompt,
        dataset.prompt2,
        tokenize_strategy,
        text_encoding_strategy,
        text_encoders,
        accelerator.device,
        weight_dtype,
    )
    logger.info(f"[sanity/prompt] prompt='{dataset.prompt}'")
    logger.info(f"[sanity/prompt] prompt2='{dataset.prompt2}'")
    logger.info(f"[sanity/prompt] cached_text_shapes={[tuple(t.shape) for t in cached_text]}")
    logger.info(
        "[sanity/prompt] cached_text_norms="
        + f"te1={cached_text[0].float().norm().item():.6f} "
        + f"te2={cached_text[1].float().norm().item():.6f} "
        + f"pool={cached_text[2].float().norm().item():.6f}"
    )
    if alpha_config["enabled"]:
        logger.info(
            "[alpha/config] "
            + f"head_scales={alpha_config['head_scales']} selected_head_scales={selected_alpha_head_indices} "
            + f"head_mode={alpha_config['head_mode']} output_source={alpha_config['output_source']} baseline_mode={alpha_config['baseline_mode']} "
            + f"loss_weight={alpha_config['loss_weight']:.4f} "
            + f"edge_loss_scale={alpha_config['edge_loss_scale']:.4f} warmup_steps={alpha_config['warmup_steps']} "
            + f"temperature_start={alpha_config['logit_temperature_start']:.3f} temperature_end={alpha_config['logit_temperature_end']:.3f}"
        )

    if not args.skip_lora_sanity_check:
        lora_sanity_prompt = args.lora_sanity_prompt or dataset.prompt
        lora_sanity_prompt2 = args.lora_sanity_prompt2 or lora_sanity_prompt
        lora_sanity_text = prepare_text_conditioning(
            lora_sanity_prompt,
            lora_sanity_prompt2,
            tokenize_strategy,
            text_encoding_strategy,
            text_encoders,
            accelerator.device,
            weight_dtype,
        )
        logger.info(f"[sanity/lora] prompt='{lora_sanity_prompt}'")
        logger.info(f"[sanity/lora] prompt2='{lora_sanity_prompt2}'")
        run_material_lora_sanity_check(
            args,
            dataset,
            unet,
            control_net,
            material_lora_unet,
            material_lora_control,
            vae,
            DDPMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
                clip_sample=False,
            ),
            lora_sanity_text,
            accelerator.device,
            weight_dtype,
            seam_config,
        )

    if verification_config["enabled"] and verification_config["run_controlnet_sanity"]:
        run_controlnet_influence_sanity_check(
            args,
            verification_config,
            dataset,
            unet,
            control_net,
            material_lora_unet,
            material_lora_control,
            vae,
            DDPMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
                clip_sample=False,
            ),
            cached_text,
            accelerator.device,
            weight_dtype,
            seam_config,
        )


    if verification_config["enabled"] and verification_config["run_controlnet_sanity"]:
        run_channel_perturbation_sanity_check(
            args,
            dataset,
            unet,
            control_net,
            DDPMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
                clip_sample=False,
            ),
            cached_text,
            accelerator.device,
            weight_dtype,
            seam_config,
            sanity_steps=int(verification_config["controlnet_sanity_steps"]),
        )

    control_net, optimizer, dataloader = accelerator.prepare(control_net, optimizer, dataloader)

    ema_state: Optional[Dict[str, torch.Tensor]] = None
    if use_ema:
        ema_decay = float(args.ema_decay)
        if ema_decay <= 0.0 or ema_decay >= 1.0:
            raise ValueError("--ema_decay must be in (0,1) when EMA is enabled")
        ema_state = init_ema_state(unwrap_model(accelerator, control_net))
        logger.info(f"[ema] enabled decay={ema_decay:.6f} eval_at_anchors={args.ema_eval_at_anchors}")

    resumed_step, resumed_ema_state = load_extended_training_state(
        args,
        accelerator,
        use_ema,
        control_net_model=unwrap_model(accelerator, control_net),
        optimizer=optimizer,
    )
    if resumed_ema_state is not None:
        ema_state = resumed_ema_state

    # Resume fallback can load tensors from CPU artifacts; reassert runtime device placement.
    control_net.to(accelerator.device, dtype=torch.float32)
    material_lora_control.to(accelerator.device, dtype=torch.float32)

    scheduler_config = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "num_train_timesteps": 1000,
        "clip_sample": False,
    }

    if resumed_step == 0 and evaluation_config["enabled"] and evaluation_config["include_step0"] and not args.skip_step0_eval:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(eval_output_dir_raw, exist_ok=True)
            raw_model = unwrap_model(accelerator, control_net)
            step0_summary = run_eval_step(
                step_label="step_0000_pretrain",
                output_dir=eval_output_dir_raw,
                run_name=args.output_name,
                pretrain=True,
                optimizer_steps_completed=0,
                dataset=dataset,
                resolved_samples=resolved_eval_samples,
                unet=unet,
                control_net=raw_model,
                vae=vae,
                cached_text=cached_text,
                eval_config=evaluation_config,
                scheduler_config=scheduler_config,
                device=accelerator.device,
                weight_dtype=weight_dtype,
                control_dtype=torch.float32,
                vae_dtype=vae_dtype,
            )
            eval_step_summaries["step_0000_pretrain"] = step0_summary
            if binding_config["enabled"] and resolved_swap_pairs:
                run_semantic_binding_eval(
                    step_label="step_0000_pretrain",
                    output_dir=eval_output_dir_raw,
                    run_name=args.output_name,
                    dataset=dataset,
                    swap_pairs=resolved_swap_pairs,
                    unet=unet,
                    control_net=raw_model,
                    vae=vae,
                    cached_text=cached_text,
                    binding_config=binding_config,
                    scheduler_config=scheduler_config,
                    device=accelerator.device,
                    weight_dtype=weight_dtype,
                    control_dtype=torch.float32,
                    vae_dtype=vae_dtype,
                )
            clean_memory_on_device(accelerator.device)
            if ema_state is not None and args.ema_eval_at_anchors and eval_output_dir_ema is not None:
                os.makedirs(eval_output_dir_ema, exist_ok=True)
                backup_state = swap_model_state(raw_model, ema_state)
                try:
                    run_eval_step(
                        step_label="step_0000_pretrain",
                        output_dir=eval_output_dir_ema,
                        run_name=args.output_name,
                        pretrain=True,
                        optimizer_steps_completed=0,
                        dataset=dataset,
                        resolved_samples=resolved_eval_samples,
                        unet=unet,
                        control_net=raw_model,
                        vae=vae,
                        cached_text=cached_text,
                        eval_config=evaluation_config,
                        scheduler_config=scheduler_config,
                        device=accelerator.device,
                        weight_dtype=weight_dtype,
                        control_dtype=torch.float32,
                        vae_dtype=vae_dtype,
                    )
                    if binding_config["enabled"] and resolved_swap_pairs:
                        run_semantic_binding_eval(
                            step_label="step_0000_pretrain",
                            output_dir=eval_output_dir_ema,
                            run_name=args.output_name,
                            dataset=dataset,
                            swap_pairs=resolved_swap_pairs,
                            unet=unet,
                            control_net=raw_model,
                            vae=vae,
                            cached_text=cached_text,
                            binding_config=binding_config,
                            scheduler_config=scheduler_config,
                            device=accelerator.device,
                            weight_dtype=weight_dtype,
                            control_dtype=torch.float32,
                            vae_dtype=vae_dtype,
                        )
                finally:
                    restore_model_state(raw_model, backup_state)
                    clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()
        control_net.train()

    noise_scheduler = DDPMScheduler(**scheduler_config)

    terrain_mask_index = _find_channel_index(dataset.channel_names, "terrain_mask") if alpha_config["enabled"] else -1
    global_step = resumed_step
    if (
        int(getattr(args, "save_warmup_every_n_steps", 0) or 0) > 0
        and int(getattr(args, "save_warmup_steps", 0) or 0) > 0
        and int(getattr(args, "save_every_n_steps_after_warmup", 0) or 0) > 0
    ):
        logger.info(
            "[checkpoint/schedule] mode=warmup_then_regular resumed_step=%d warmup_every=%d warmup_steps=%d after_every=%d",
            resumed_step,
            int(args.save_warmup_every_n_steps),
            int(args.save_warmup_steps),
            int(args.save_every_n_steps_after_warmup),
        )
    else:
        logger.info(
            "[checkpoint/schedule] mode=fixed every=%d",
            int(args.save_every_n_steps),
        )
    loss_trace: List[Dict[str, float]] = []
    loss_trace_mode = _normalize_loss_trace_mode(verification_config.get("loss_trace_mode", "full"))
    loss_trace_path = os.path.join(args.output_dir, "sanity", "loss_trace.csv")
    seam_adapter_diag_trace: List[Dict[str, float]] = []
    seam_adapter_diag_path = os.path.join(args.output_dir, "sanity", "seam_adapter_diag.csv")
    if resumed_step > 0 and os.path.exists(loss_trace_path):
        with open(loss_trace_path, "r", newline="", encoding="utf-8") as handle:
            loss_trace = list(csv.DictReader(handle))
    if resumed_step > 0 and os.path.exists(seam_adapter_diag_path):
        with open(seam_adapter_diag_path, "r", newline="", encoding="utf-8") as handle:
            seam_adapter_diag_trace = list(csv.DictReader(handle))

    def _append_loss_trace_row_live(row: Dict[str, object]) -> None:
        if not accelerator.is_main_process:
            return
        try:
            sanity_dir = os.path.join(args.output_dir, "sanity")
            os.makedirs(sanity_dir, exist_ok=True)
            write_header = (not os.path.exists(loss_trace_path)) or os.path.getsize(loss_trace_path) == 0
            with open(loss_trace_path, "a", newline="", encoding="utf-8") as _fh:
                _writer = csv.DictWriter(_fh, fieldnames=list(row.keys()), extrasaction="ignore")
                if write_header:
                    _writer.writeheader()
                _writer.writerow(row)
        except Exception as _e:
            logger.warning("[sanity/loss] live append failed: %s", _e)

    def _append_seam_adapter_diag_row_live(row: Dict[str, object]) -> None:
        if not accelerator.is_main_process:
            return
        try:
            sanity_dir = os.path.join(args.output_dir, "sanity")
            os.makedirs(sanity_dir, exist_ok=True)
            write_header = (not os.path.exists(seam_adapter_diag_path)) or os.path.getsize(seam_adapter_diag_path) == 0
            with open(seam_adapter_diag_path, "a", newline="", encoding="utf-8") as _fh:
                _writer = csv.DictWriter(_fh, fieldnames=list(SEAM_ADAPTER_DIAG_FIELDS), extrasaction="ignore")
                if write_header:
                    _writer.writeheader()
                _writer.writerow(_format_seam_adapter_diag_row(row))
        except Exception as _e:
            logger.warning("[sanity/seam_adapter] live append failed: %s", _e)

    tiny_overfit_run = bool(
        verification_config["always_log_during_tiny_overfit"]
        and args.max_train_steps <= int(verification_config["tiny_overfit_max_steps"])
    )
    warned_expanded_latent_cache_bypass = False
    seam_adapter_debug_saved = False
    progress_bar = tqdm(total=args.max_train_steps, initial=global_step, disable=not accelerator.is_local_main_process, desc="steps")
    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(control_net):
                batch_images = batch["images"].to(accelerator.device, dtype=vae_dtype)
                trusted_mask = batch["trusted_mask"].to(accelerator.device, dtype=weight_dtype).unsqueeze(1)
                alpha_target = batch["alpha_target"]
                if alpha_target is not None:
                    alpha_target = alpha_target.to(accelerator.device, dtype=weight_dtype).unsqueeze(1)

                conditioning_images = batch["conditioning_images"].to(accelerator.device, dtype=weight_dtype)
                model_conditioning_images, model_channel_names, seam_diag = build_model_visible_conditioning(
                    batch,
                    dataset,
                    seam_config,
                    accelerator.device,
                    weight_dtype,
                )

                if training_expanded_supervision_enabled and int(training_expanded_halo_px) > 0:
                    halo_px = int(training_expanded_halo_px)
                    if batch.get("expanded_images") is None or batch.get("expanded_trusted_mask") is None or batch.get("expanded_target_sizes_hw") is None:
                        raise ValueError(
                            "expanded supervision requires dataset-provided expanded target tensors; padding fallback is disabled"
                        )
                    batch_images = batch["expanded_images"].to(accelerator.device, dtype=vae_dtype)
                    trusted_mask = batch["expanded_trusted_mask"].to(accelerator.device, dtype=weight_dtype).unsqueeze(1)
                    if alpha_target is not None:
                        if batch.get("expanded_alpha_target") is None:
                            raise ValueError("expanded supervision requires expanded alpha targets when alpha supervision is enabled")
                        alpha_target = batch["expanded_alpha_target"].to(accelerator.device, dtype=weight_dtype).unsqueeze(1)
                    conditioning_images = batch["conditioning_images"].to(accelerator.device, dtype=weight_dtype)
                    model_conditioning_images = _center_embed_spatial_tensor(model_conditioning_images, halo_px, fill_value=0.0)

                    target_sizes_hw = batch["expanded_target_sizes_hw"].to(accelerator.device).clone()
                    size_embeddings = sdxl_train_util.get_size_embeddings(
                        target_sizes_hw,
                        torch.zeros_like(target_sizes_hw),
                        target_sizes_hw,
                        accelerator.device,
                    ).to(weight_dtype)
                else:
                    size_embeddings = build_size_embeddings(batch, accelerator.device, weight_dtype)

                if batch["latents"] is not None and not training_expanded_supervision_enabled:
                    latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
                else:
                    if batch["latents"] is not None and training_expanded_supervision_enabled and not warned_expanded_latent_cache_bypass:
                        logger.warning(
                            "[seam/geometry] training expanded supervision bypasses cached latents and re-encodes expanded images"
                        )
                        warned_expanded_latent_cache_bypass = True
                    with torch.no_grad():
                        latents = vae.encode(batch_images).latent_dist.sample()
                        latents = latents.to(dtype=weight_dtype)
                latents = latents * sdxl_model_util.VAE_SCALE_FACTOR

                batch_size = latents.shape[0]
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                train_prompt = batch["prompt"]
                train_prompt2 = batch["prompt2"]
                train_prompt_name = "default"
                train_prompt_mode = "default"
                if training_prompt_sampler is not None:
                    selected_prompt = training_prompt_sampler.sample()
                    train_prompt = selected_prompt.prompt
                    train_prompt2 = selected_prompt.prompt2
                    train_prompt_name = selected_prompt.name
                    train_prompt_mode = selected_prompt.mode
                if train_prompt_name not in logged_prompt_names:
                    logger.info(
                        "[train/prompt] activated name=%s prompt='%s' prompt2='%s'",
                        train_prompt_name,
                        train_prompt,
                        train_prompt2,
                    )
                    logged_prompt_names.add(train_prompt_name)
                encoder_hidden_states1, encoder_hidden_states2, pool2 = get_or_prepare_text_conditioning(
                    text_conditioning_cache,
                    train_prompt,
                    train_prompt2,
                    tokenize_strategy,
                    text_encoding_strategy,
                    text_encoders,
                    accelerator.device,
                    weight_dtype,
                )
                text_embedding = torch.cat(
                    [encoder_hidden_states1.expand(batch_size, -1, -1), encoder_hidden_states2.expand(batch_size, -1, -1)],
                    dim=2,
                ).to(dtype=weight_dtype)
                vector_embedding = torch.cat([pool2.expand(batch_size, -1), size_embeddings], dim=1).to(dtype=weight_dtype)

                diffusion_loss_value = 0.0
                alpha_bce_value = 0.0
                alpha_edge_value = 0.0
                alpha_total_value = 0.0
                seam_alpha_local_value = 0.0
                seam_margin_inner_loss_value = 0.0
                seam_margin_outer_loss_value = 0.0
                seam_interior_inner_loss_value = 0.0
                seam_interior_outer_loss_value = 0.0
                seam_continuation_loss_value = 0.0
                seam_continuation_gradient_loss_value = 0.0
                seam_continuation_gradient_loss_raw_value = 0.0
                seam_continuation_rgb_loss_raw_value = 0.0
                seam_continuation_distance_band_px_count_value = 0.0
                seam_continuation_width_px_value = float(seam_config.get("continuation_width_px", 0.0))
                seam_continuation_peak_weight_value = float(seam_config.get("continuation_peak_rgb_weight", 0.0))
                seam_continuation_falloff_power_value = float(seam_config.get("continuation_falloff_power", 2.0))
                seam_continuation_rgb_weight_mode_value = str(seam_config.get("continuation_rgb_weight_mode", "falloff"))
                seam_continuation_gradient_loss_weight_value = 0.0
                seam_continuation_weight_sum_value = 0.0
                seam_continuation_effective_weight_mean_value = 0.0
                seam_continuation_effective_weight_max_value = 0.0
                seam_continuation_weight_fraction_first_8px_value = 0.0
                seam_continuation_weight_fraction_first_16px_value = 0.0
                seam_continuation_weight_fraction_first_24px_value = 0.0
                seam_continuation_weight_fraction_full_band_value = 0.0
                seam_continuation_rgb_loss_bin_0_8_value = 0.0
                seam_continuation_rgb_loss_bin_8_16_value = 0.0
                seam_continuation_rgb_loss_bin_16_24_value = 0.0
                seam_continuation_rgb_loss_bin_24_32_value = 0.0
                seam_continuation_rgb_loss_bin_32_40_value = 0.0
                seam_continuation_rgb_loss_bin_40_48_value = 0.0
                seam_rgb_margin_inner_loss_value = 0.0
                seam_rgb_margin_outer_loss_value = 0.0
                seam_rgb_interior_inner_loss_value = 0.0
                seam_rgb_interior_outer_loss_value = 0.0
                seam_rgb_halo_inner_loss_value = 0.0
                seam_rgb_halo_outer_loss_value = 0.0
                seam_rgb_interior_core_loss_value = 0.0
                seam_rgb_continuation_weighted_loss_value = 0.0
                seam_rgb_total_loss_value = 0.0
                seam_rgb_halo_loss_raw_value = 0.0
                seam_rgb_interior_loss_raw_value = 0.0
                seam_rgb_halo_loss_weighted_value = 0.0
                seam_rgb_interior_loss_weighted_value = 0.0
                seam_rgb_halo_inner_loss_raw_value = 0.0
                seam_rgb_halo_outer_loss_raw_value = 0.0
                seam_rgb_halo_inner_loss_weighted_value = 0.0
                seam_rgb_halo_outer_loss_weighted_value = 0.0
                seam_halo_supervised_px_value = 0.0
                seam_interior_supervised_px_value = 0.0
                seam_halo_to_interior_px_ratio_value = 0.0
                halo_inner_edge_1px_rgb_loss_value = 0.0
                halo_inner_edge_4px_rgb_loss_value = 0.0
                halo_inner_8px_rgb_loss_value = 0.0
                halo_inner_16px_rgb_loss_value = 0.0
                expanded_halo_copy_diff_mean_value = 0.0
                expanded_halo_copy_diff_max_value = 0.0
                seam_halo_gradient_loss_value = 0.0
                seam_margin_inner_coverage_ratio = 0.0
                seam_margin_inner_coverage_px = 0.0
                seam_margin_inner_raw_px = 0.0
                seam_active_edges_count = 0.0
                seam_edge_recon_score = 0.0
                seam_alpha_halo_loss_raw_value = 0.0
                seam_alpha_interior_loss_raw_value = 0.0
                seam_alpha_halo_loss_weighted_value = 0.0
                seam_alpha_interior_loss_weighted_value = 0.0
                seam_halo_target_energy_value = 0.0
                seam_loss_contribution_ratio_value = 0.0
                halo_inner_weighted_contribution = 0.0
                halo_outer_weighted_contribution = 0.0
                interior_continuation_weighted_contribution = 0.0
                interior_core_weighted_contribution = 0.0
                continuation_gradient_weighted_contribution = 0.0
                # Per-edge gating diagnostics (computed after seam_maps_supervised)
                seam_total_edges_defined = 0.0
                seam_valid_edges_for_loss_count = 0.0
                seam_valid_edge_ratio = 0.0
                seam_dataset_qualified_edges_count = 0.0
                seam_trainer_qualified_edges_count = 0.0
                halo_without_continuation_count_after_gate = 0.0
                seam_halo_without_continuation_count = 0.0
                seam_edge_north_valid_for_seam_supervision = 0.0
                seam_edge_south_valid_for_seam_supervision = 0.0
                seam_edge_east_valid_for_seam_supervision = 0.0
                seam_edge_west_valid_for_seam_supervision = 0.0
                seam_edge_north_halo_inner_px_after_gate = 0.0
                seam_edge_south_halo_inner_px_after_gate = 0.0
                seam_edge_east_halo_inner_px_after_gate = 0.0
                seam_edge_west_halo_inner_px_after_gate = 0.0
                seam_edge_north_halo_outer_px_after_gate = 0.0
                seam_edge_south_halo_outer_px_after_gate = 0.0
                seam_edge_east_halo_outer_px_after_gate = 0.0
                seam_edge_west_halo_outer_px_after_gate = 0.0
                seam_edge_north_continuation_px_after_gate = 0.0
                seam_edge_south_continuation_px_after_gate = 0.0
                seam_edge_east_continuation_px_after_gate = 0.0
                seam_edge_west_continuation_px_after_gate = 0.0
                prompt_conflict_score = 0.0
                prompt_conflict_score_norm = 0.0
                alpha_terrain_bce_value = 0.0
                alpha_terrain_iou_value = 0.0
                terrain_curriculum_factor = 0.0
                alpha_temperature = 1.0
                alpha_prior_weight = 0.0
                train_prediction_h = float(batch_images.shape[-2])
                train_prediction_w = float(batch_images.shape[-1])
                train_alpha_target_h = 0.0
                train_alpha_target_w = 0.0
                train_supervision_mask_h = 0.0
                train_supervision_mask_w = 0.0
                train_target_min_x = 0.0
                train_target_max_x = 0.0
                train_target_min_y = 0.0
                train_target_max_y = 0.0
                train_halo_min_x = -1.0
                train_halo_max_x = -1.0
                train_halo_min_y = -1.0
                train_halo_max_y = -1.0
                train_geometry_mode = "interior_only"
                controlnet_diagnostics = None
                grad_l2_alpha = 0.0
                grad_l2_control_residual = 0.0
                grad_l2_control_mid = 0.0
                param_delta_alpha = 0.0
                param_delta_control_residual = 0.0
                param_delta_control_mid = 0.0
                bind_loss_value = 0.0
                bind_margin_loss_value = 0.0
                bind_pos_loss_value = 0.0
                bind_neg_loss_value = 0.0
                bind_ranking_gap_value = 0.0
                bind_win_rate_value = 0.0
                bind_active_batch = 0.0
                bind_negative_mode = "none"
                seam_rgb_summary: Dict[str, object] = {}
                seam_alpha_summary: Dict[str, object] = {}
                step_for_logging = global_step + 1
                seam_resume_relative_step = max(0, global_step - resumed_step)
                seam_continuation_gradient_loss_weight_value = _resolve_step_value_schedule(
                    default_value=float(seam_config.get("continuation_gradient_loss_weight", 0.25)),
                    schedule=seam_config.get("continuation_gradient_loss_weight_schedule", []),
                    current_step=seam_resume_relative_step,
                )
                seam_adapter_log_now = bool(
                    seam_config["enabled"]
                    and seam_config.get("seam_adapter_enabled", False)
                    and (
                        step_for_logging == 1
                        or step_for_logging % max(1, args.loss_trace_every) == 0
                        or step_for_logging == args.max_train_steps
                        or tiny_overfit_run
                    )
                )
                verification_log_now = verification_config["enabled"] and (
                    step_for_logging == 1
                    or step_for_logging % max(1, int(verification_config["log_every"])) == 0
                    or tiny_overfit_run
                )
                request_controlnet_diagnostics = verification_log_now or seam_adapter_log_now
                if verification_log_now:
                    logger.info(
                        "[train/prompt] step=%d prompt_name=%s prompt='%s' prompt2='%s'",
                        step_for_logging,
                        train_prompt_name,
                        train_prompt,
                        train_prompt2,
                    )
                pre_step_param_stats = None

                if verification_log_now:
                    _cond_fp32 = model_conditioning_images.float()
                    _cond_dtype = model_conditioning_images
                    _ch_names = model_channel_names
                    for _ci in range(_cond_fp32.shape[1]):
                        _ch_fp32 = _cond_fp32[:, _ci]
                        _ch_cast = _cond_dtype[:, _ci].detach().float()
                        _name = _ch_names[_ci] if _ci < len(_ch_names) else f"ch{_ci}"
                        logger.info(
                            f"[verify/channels] step={step_for_logging} ch={_ci} {_name} "
                            f"fp32_range=[{_ch_fp32.min():.4f},{_ch_fp32.max():.4f}] "
                            f"fp32_mean={_ch_fp32.mean():.4f} fp32_std={_ch_fp32.std():.4f} "
                            f"cast_range=[{_ch_cast.min():.4f},{_ch_cast.max():.4f}] "
                            f"cast_mean={_ch_cast.mean():.4f}"
                        )
                    if seam_config["enabled"]:
                        logger.info(
                            "[verify/seam] step=%d seam_enabled=1 defined_ratio=%.4f pre_gate_l2=%.6f post_gate_l2=%.6f",
                            step_for_logging,
                            seam_diag["seam_defined_ratio"],
                            seam_diag["seam_pre_gate_l2"],
                            seam_diag["seam_post_gate_l2"],
                        )
                        for _edge_name in ("north", "south", "east", "west"):
                            logger.info(
                                "[verify/seam_edge] step=%d edge=%s defined=%.4f pre_gate_l2=%.6f post_gate_l2=%.6f visible_l2=%.6f",
                                step_for_logging,
                                _edge_name,
                                seam_diag.get(f"seam_edge_{_edge_name}_defined", 0.0),
                                seam_diag.get(f"seam_edge_{_edge_name}_pre_gate_l2", 0.0),
                                seam_diag.get(f"seam_edge_{_edge_name}_post_gate_l2", 0.0),
                                seam_diag.get(f"seam_edge_{_edge_name}_visible_l2", 0.0),
                            )

                with accelerator.autocast():
                    if alpha_config["enabled"]:
                        input_resi_add, mid_add, alpha_outputs = control_net(
                            noisy_latents,
                            timesteps,
                            text_embedding,
                            vector_embedding,
                            model_conditioning_images,
                            return_alpha=True,
                            return_diagnostics=request_controlnet_diagnostics,
                            alpha_target_size=tuple(batch_images.shape[-2:]),
                        )
                        controlnet_diagnostics = alpha_outputs.get("diagnostics") if alpha_outputs is not None else None
                    else:
                        if request_controlnet_diagnostics:
                            input_resi_add, mid_add, controlnet_diagnostics = control_net(
                                noisy_latents,
                                timesteps,
                                text_embedding,
                                vector_embedding,
                                model_conditioning_images,
                                return_diagnostics=True,
                            )
                        else:
                            input_resi_add, mid_add = control_net(
                                noisy_latents,
                                timesteps,
                                text_embedding,
                                vector_embedding,
                                model_conditioning_images,
                            )
                        alpha_outputs = None
                    if seam_adapter_log_now and not seam_adapter_debug_saved and accelerator.is_main_process:
                        try:
                            sanity_dir = os.path.join(args.output_dir, "sanity")
                            os.makedirs(sanity_dir, exist_ok=True)
                            _save_seam_adapter_debug_board(
                                os.path.join(sanity_dir, "seam_adapter_debug"),
                                model_conditioning_images[0].detach().float(),
                                seam_config,
                                controlnet_diagnostics,
                            )
                            seam_adapter_debug_saved = True
                        except Exception as _e:
                            logger.warning("[sanity/seam_adapter] debug board generation failed: %s", _e)
                    noise_pred = unet(
                        noisy_latents,
                        timesteps,
                        text_embedding,
                        vector_embedding,
                        input_resi_add,
                        mid_add,
                    )

                    loss_map = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
                    pixel_loss = loss_map.mean(dim=1, keepdim=True)
                    latent_mask = F.interpolate(trusted_mask, size=pixel_loss.shape[-2:], mode="area")
                    assert latent_mask.shape[-2:] == pixel_loss.shape[-2:], (
                        f"latent mask shape {latent_mask.shape[-2:]} does not match loss shape {pixel_loss.shape[-2:]}"
                    )
                    masked_sum = (pixel_loss * latent_mask).sum(dim=(1, 2, 3))
                    masked_area = latent_mask.sum(dim=(1, 2, 3)).clamp_min(1e-6)
                    pos_loss_per_sample = masked_sum / masked_area
                    diffusion_loss = pos_loss_per_sample.mean()
                    loss = diffusion_loss

                    seam_decay_maps = None
                    edge_band_masks = None
                    edge_defined = None
                    seam_strip_width_px = None
                    seam_supervision_mask = trusted_mask.float()
                    seam_maps_supervised: Optional[Dict[str, torch.Tensor]] = None
                    if (
                        seam_config["enabled"]
                        and batch.get("seam_decay_maps") is not None
                        and batch.get("edge_defined_flags") is not None
                        and batch.get("edge_band_masks") is not None
                        and batch.get("seam_strip_width_px") is not None
                    ):
                        seam_decay_maps = batch["seam_decay_maps"].to(accelerator.device, dtype=weight_dtype)
                        edge_band_masks = batch["edge_band_masks"].to(accelerator.device, dtype=weight_dtype)
                        edge_defined = batch["edge_defined_flags"].to(accelerator.device, dtype=weight_dtype)
                        # Enforce fixed_defined_edge: override batch flags to allow only the configured edge.
                        _fixed_edge_name = str(seam_config.get("fixed_defined_edge", "") or "").strip().lower()
                        _edge_name_to_idx = {"north": 0, "south": 1, "east": 2, "west": 3}
                        if _fixed_edge_name in _edge_name_to_idx:
                            _forced_idx = _edge_name_to_idx[_fixed_edge_name]
                            _override = torch.zeros_like(edge_defined)
                            _override[:, _forced_idx] = 1.0
                            edge_defined = _override
                        seam_strip_width_px = batch["seam_strip_width_px"].to(accelerator.device, dtype=weight_dtype)
                        if training_expanded_supervision_enabled and int(training_expanded_halo_px) > 0:
                            if batch.get("expanded_seam_decay_maps") is None or batch.get("expanded_edge_band_masks") is None:
                                raise ValueError(
                                    "expanded supervision requires dataset-provided expanded seam geometry tensors; padding fallback is disabled"
                                )
                            seam_decay_maps = batch["expanded_seam_decay_maps"].to(accelerator.device, dtype=weight_dtype)
                            edge_band_masks = batch["expanded_edge_band_masks"].to(accelerator.device, dtype=weight_dtype)
                        if terrain_mask_index >= 0:
                            continuation_valid_mask = _terrain_mask_to_occupancy(
                                conditioning_images[:, terrain_mask_index : terrain_mask_index + 1],
                                bool(alpha_config.get("terrain_mask_black_is_terrain", True)),
                            )
                            continuation_valid_mask = (continuation_valid_mask >= float(alpha_config.get("binary_threshold", 0.5))).to(dtype=weight_dtype)
                        elif alpha_target is not None:
                            continuation_valid_mask = (alpha_target >= float(alpha_config.get("binary_threshold", 0.5))).to(dtype=weight_dtype)
                        else:
                            continuation_valid_mask = torch.ones_like(trusted_mask.float())
                        if training_expanded_supervision_enabled and int(training_expanded_halo_px) > 0:
                            continuation_valid_mask = _center_embed_spatial_tensor(
                                continuation_valid_mask,
                                int(training_expanded_halo_px),
                                fill_value=0.0,
                            )
                        seam_supervision_mask_pre = _build_seam_supervision_mask(
                            trusted_mask=trusted_mask.float(),
                            edge_band_masks=edge_band_masks,
                            edge_defined_flags=edge_defined,
                            seam_config=seam_config,
                        )
                        seam_maps_candidate = _build_seam_region_maps(
                            edge_band_masks=edge_band_masks,
                            seam_decay_maps=seam_decay_maps,
                            edge_defined_flags=edge_defined,
                            seam_strip_width_px=seam_strip_width_px,
                            supervision_mask=seam_supervision_mask_pre,
                            seam_config=seam_config,
                            expanded_halo_px=(int(training_expanded_halo_px) if training_expanded_supervision_enabled else 0),
                            continuation_valid_mask=continuation_valid_mask,
                        )
                        seam_edge_qualification = summarize_seam_edge_qualification(
                            seam_maps=seam_maps_candidate,
                            edge_defined_flags=edge_defined,
                            seam_config=seam_config,
                        )
                        trainer_qualified_edge_mask = seam_edge_qualification["qualified_edge_mask"].to(
                            accelerator.device,
                            dtype=weight_dtype,
                        )
                        seam_trainer_qualified_edges_count = float(
                            seam_edge_qualification["valid_edges_for_loss_count"].mean().detach().item()
                        )
                        dataset_qualified_edge_mask = None
                        if batch.get("seam_qualified_edge_mask") is not None:
                            dataset_qualified_edge_mask = batch["seam_qualified_edge_mask"].to(accelerator.device, dtype=weight_dtype)
                            seam_dataset_qualified_edges_count = float(
                                batch["seam_qualified_valid_edges_count"].to(accelerator.device, dtype=weight_dtype).mean().detach().item()
                            )

                        if bool(seam_config.get("seam_qualified_sampling_enabled", False)):
                            if dataset_qualified_edge_mask is None or batch.get("seam_qualified_valid_edges_count") is None:
                                raise RuntimeError("seam-qualified sampling enabled but dataset qualification tensors are missing from the batch")
                            if not torch.equal(dataset_qualified_edge_mask >= 0.5, trainer_qualified_edge_mask >= 0.5):
                                raise RuntimeError(
                                    "dataset/trainer seam qualification mismatch: "
                                    + f"dataset={dataset_qualified_edge_mask.detach().cpu().tolist()} "
                                    + f"trainer={trainer_qualified_edge_mask.detach().cpu().tolist()}"
                                )
                            edge_defined = edge_defined * trainer_qualified_edge_mask

                        seam_supervision_mask = _build_seam_supervision_mask(
                            trusted_mask=trusted_mask.float(),
                            edge_band_masks=edge_band_masks,
                            edge_defined_flags=edge_defined,
                            seam_config=seam_config,
                        )
                        seam_maps_supervised = _build_seam_region_maps(
                            edge_band_masks=edge_band_masks,
                            seam_decay_maps=seam_decay_maps,
                            edge_defined_flags=edge_defined,
                            seam_strip_width_px=seam_strip_width_px,
                            supervision_mask=seam_supervision_mask,
                            seam_config=seam_config,
                            expanded_halo_px=(int(training_expanded_halo_px) if training_expanded_supervision_enabled else 0),
                            continuation_valid_mask=continuation_valid_mask,
                        )
                        seam_maps_raw = _build_seam_region_maps(
                            edge_band_masks=edge_band_masks,
                            seam_decay_maps=seam_decay_maps,
                            edge_defined_flags=edge_defined,
                            seam_strip_width_px=seam_strip_width_px,
                            supervision_mask=torch.ones_like(trusted_mask.float()),
                            seam_config=seam_config,
                            expanded_halo_px=(int(training_expanded_halo_px) if training_expanded_supervision_enabled else 0),
                            continuation_valid_mask=continuation_valid_mask,
                        )
                        seam_margin_inner_coverage_px = float(seam_maps_supervised["margin_inner"].sum().detach().item())
                        seam_margin_inner_raw_px = float(seam_maps_raw["margin_inner"].sum().detach().item())
                        seam_margin_inner_coverage_ratio = seam_margin_inner_coverage_px / max(seam_margin_inner_raw_px, 1e-6)
                        seam_active_edges_count = float((edge_defined >= 0.5).float().sum(dim=1).mean().detach().item())
                        if seam_active_edges_count > 0.0 and seam_margin_inner_coverage_ratio < 0.2:
                            logger.warning(
                                "[seam/coverage] step=%d low margin_inner coverage ratio=%.4f active_edges=%.2f",
                                step_for_logging,
                                seam_margin_inner_coverage_ratio,
                                seam_active_edges_count,
                            )
                        # Per-edge gating diagnostics
                        _mi_pe = seam_maps_supervised["margin_inner_per_edge"]   # [B,4,H,W]
                        _mo_pe = seam_maps_supervised["margin_outer_per_edge"]   # [B,4,H,W]
                        _ii_pe = seam_maps_supervised["interior_inner_per_edge"] # [B,4,H,W]
                        _def_pe = (edge_defined >= 0.5)  # [B,4]
                        _total_def = float(_def_pe.float().sum(dim=1).mean().detach().item())
                        seam_total_edges_defined = _total_def
                        _vcnt = 0.0
                        _hno = 0.0
                        for _ei, _en in enumerate(["north", "south", "east", "west"]):
                            _isdef = float(_def_pe[:, _ei].float().mean().detach().item())
                            _hip = float(_mi_pe[:, _ei].sum(dim=(-2, -1)).mean().detach().item())
                            _hop = float(_mo_pe[:, _ei].sum(dim=(-2, -1)).mean().detach().item())
                            _cp = float(_ii_pe[:, _ei].sum(dim=(-2, -1)).mean().detach().item())
                            _vld = 1.0 if (_isdef >= 0.5 and (_hip > 0 or _hop > 0) and _cp > 0) else 0.0
                            if _en == "north":
                                seam_edge_north_valid_for_seam_supervision = _vld
                                seam_edge_north_halo_inner_px_after_gate = _hip
                                seam_edge_north_halo_outer_px_after_gate = _hop
                                seam_edge_north_continuation_px_after_gate = _cp
                            elif _en == "south":
                                seam_edge_south_valid_for_seam_supervision = _vld
                                seam_edge_south_halo_inner_px_after_gate = _hip
                                seam_edge_south_halo_outer_px_after_gate = _hop
                                seam_edge_south_continuation_px_after_gate = _cp
                            elif _en == "east":
                                seam_edge_east_valid_for_seam_supervision = _vld
                                seam_edge_east_halo_inner_px_after_gate = _hip
                                seam_edge_east_halo_outer_px_after_gate = _hop
                                seam_edge_east_continuation_px_after_gate = _cp
                            else:
                                seam_edge_west_valid_for_seam_supervision = _vld
                                seam_edge_west_halo_inner_px_after_gate = _hip
                                seam_edge_west_halo_outer_px_after_gate = _hop
                                seam_edge_west_continuation_px_after_gate = _cp
                            _vcnt += _vld
                            if _isdef >= 0.5 and (_hip > 0 or _hop > 0) and _cp <= 0:
                                _hno += 1.0
                        seam_valid_edges_for_loss_count = _vcnt
                        seam_valid_edge_ratio = _vcnt / max(_total_def, 1.0)
                        halo_without_continuation_count_after_gate = _hno
                        seam_halo_without_continuation_count = _hno
                        if _hno > 0.0:
                            logger.warning(
                                "[seam/per_edge_gate] step=%d halo_without_continuation=%.0f valid=%.0f defined=%.0f",
                                step_for_logging, _hno, _vcnt, _total_def,
                            )

                    # Explicit seam-supervised RGB reconstruction on known seam margins and seam-adjacent interior.
                    if (
                        seam_config["enabled"]
                        and float(seam_config.get("rgb_recon_loss_weight", 0.0)) > 0.0
                        and seam_maps_supervised is not None
                    ):
                        alpha_t = noise_scheduler.alphas_cumprod[timesteps].to(
                            device=noisy_latents.device,
                            dtype=noisy_latents.dtype,
                        ).view(-1, 1, 1, 1)
                        sqrt_alpha_t = alpha_t.sqrt().clamp_min(1e-6)
                        sqrt_one_minus_alpha_t = (1.0 - alpha_t).clamp_min(0.0).sqrt()
                        pred_x0_latents = (noisy_latents - (sqrt_one_minus_alpha_t * noise_pred)) / sqrt_alpha_t
                        pred_x0_latents_for_decode = (pred_x0_latents / sdxl_model_util.VAE_SCALE_FACTOR).to(dtype=vae_dtype)

                        region_weights = {
                            "halo_inner": float(seam_config.get("halo_inner_rgb_weight", 2.0)),
                            "halo_outer": float(seam_config.get("halo_outer_rgb_weight", 1.5)),
                            "interior_continuation": float(seam_config.get("continuation_peak_rgb_weight", 2.5)),
                            "interior_core": float(seam_config.get("interior_core_rgb_weight", 0.10)),
                        }
                        def _decode_seam_rgb(latents: torch.Tensor) -> torch.Tensor:
                            return vae.decode(latents).sample

                        seam_rgb_edge_summaries: List[Dict[str, object]] = []
                        seam_gradient_raw_sum = noisy_latents.new_tensor(0.0)
                        seam_gradient_active_px = noisy_latents.new_tensor(0.0)
                        seam_gradient_weighted_raw_sum = noisy_latents.new_tensor(0.0)
                        seam_gradient_weight_sum = noisy_latents.new_tensor(0.0)
                        halo_copy_raw_sum = noisy_latents.new_tensor(0.0)
                        halo_copy_area = noisy_latents.new_tensor(0.0)
                        halo_copy_max = noisy_latents.new_tensor(0.0)
                        halo_inner_1px_raw_sum = noisy_latents.new_tensor(0.0)
                        halo_inner_1px_area = noisy_latents.new_tensor(0.0)
                        halo_inner_4px_raw_sum = noisy_latents.new_tensor(0.0)
                        halo_inner_4px_area = noisy_latents.new_tensor(0.0)
                        halo_inner_8px_raw_sum = noisy_latents.new_tensor(0.0)
                        halo_inner_8px_area = noisy_latents.new_tensor(0.0)
                        halo_inner_16px_raw_sum = noisy_latents.new_tensor(0.0)
                        halo_inner_16px_area = noisy_latents.new_tensor(0.0)
                        seam_halo_gradient_raw_sum = noisy_latents.new_tensor(0.0)
                        seam_halo_gradient_active_px = noisy_latents.new_tensor(0.0)
                        pred_rgb = None
                        target_rgb = None
                        seam_supervision_mask_rgb = None
                        seam_maps_rgb = None
                        for edge_index, edge_name in enumerate(SEAM_ADAPTER_EDGE_NAMES):
                            edge_seam_maps = _select_seam_maps_for_edge(seam_maps_supervised, edge_index)
                            seam_loss_support_terms = []
                            if region_weights["halo_inner"] > 0.0:
                                seam_loss_support_terms.append(edge_seam_maps["margin_inner"])
                            if region_weights["halo_outer"] > 0.0:
                                seam_loss_support_terms.append(edge_seam_maps["margin_outer"])
                            if region_weights["interior_continuation"] > 0.0:
                                seam_loss_support_terms.append(edge_seam_maps["interior_inner"])
                            if region_weights["interior_core"] > 0.0:
                                seam_loss_support_terms.append(edge_seam_maps["interior_core"])
                            if seam_loss_support_terms:
                                seam_loss_support_mask = torch.stack(seam_loss_support_terms, dim=0).sum(dim=0).clamp(0.0, 1.0)
                            else:
                                seam_loss_support_mask = torch.zeros_like(edge_seam_maps["margin_inner"])
                            if float(seam_loss_support_mask.sum().detach().item()) <= 0.0:
                                seam_loss_support_mask = (
                                    edge_seam_maps["margin_inner"]
                                    + edge_seam_maps["margin_outer"]
                                    + edge_seam_maps["interior_inner"]
                                ).clamp(0.0, 1.0)
                            if float(seam_loss_support_mask.sum().detach().item()) <= 0.0:
                                continue

                            decode_crop = _compute_seam_decode_crop(
                                seam_loss_support_mask,
                                full_height=int(batch_images.shape[-2]),
                                full_width=int(batch_images.shape[-1]),
                                latent_height=int(pred_x0_latents_for_decode.shape[-2]),
                                latent_width=int(pred_x0_latents_for_decode.shape[-1]),
                                context_px=int(seam_config.get("rgb_decode_context_px", 64)),
                            )
                            edge_pred_x0_latents_for_decode = _crop_spatial_tensor(
                                pred_x0_latents_for_decode,
                                x0=decode_crop["pixel_x0"] // max(1, int(batch_images.shape[-1] // pred_x0_latents_for_decode.shape[-1])),
                                x1=decode_crop["pixel_x1"] // max(1, int(batch_images.shape[-1] // pred_x0_latents_for_decode.shape[-1])),
                                y0=decode_crop["pixel_y0"] // max(1, int(batch_images.shape[-2] // pred_x0_latents_for_decode.shape[-2])),
                                y1=decode_crop["pixel_y1"] // max(1, int(batch_images.shape[-2] // pred_x0_latents_for_decode.shape[-2])),
                            )
                            edge_target_rgb = _crop_spatial_tensor(
                                batch_images,
                                x0=decode_crop["pixel_x0"],
                                x1=decode_crop["pixel_x1"],
                                y0=decode_crop["pixel_y0"],
                                y1=decode_crop["pixel_y1"],
                            ).to(dtype=weight_dtype)
                            edge_seam_supervision_mask = _crop_spatial_tensor(
                                seam_supervision_mask,
                                x0=decode_crop["pixel_x0"],
                                x1=decode_crop["pixel_x1"],
                                y0=decode_crop["pixel_y0"],
                                y1=decode_crop["pixel_y1"],
                            )
                            edge_seam_maps = _crop_seam_maps(
                                edge_seam_maps,
                                x0=decode_crop["pixel_x0"],
                                x1=decode_crop["pixel_x1"],
                                y0=decode_crop["pixel_y0"],
                                y1=decode_crop["pixel_y1"],
                                full_height=int(batch_images.shape[-2]),
                                full_width=int(batch_images.shape[-1]),
                            )

                            edge_pred_rgb = activation_checkpoint(
                                _decode_seam_rgb,
                                edge_pred_x0_latents_for_decode,
                                use_reentrant=False,
                            ).to(dtype=weight_dtype)
                            expanded_rgb_mask = edge_seam_supervision_mask.expand(-1, edge_pred_rgb.shape[1], -1, -1)
                            assert edge_pred_rgb.shape == edge_target_rgb.shape == expanded_rgb_mask.shape, (
                                f"seam RGB geometry mismatch edge={edge_name} pred={tuple(edge_pred_rgb.shape)} target={tuple(edge_target_rgb.shape)} mask={tuple(expanded_rgb_mask.shape)}"
                            )
                            if pred_rgb is None:
                                pred_rgb = edge_pred_rgb
                                target_rgb = edge_target_rgb
                                seam_supervision_mask_rgb = edge_seam_supervision_mask
                                seam_maps_rgb = edge_seam_maps
                                train_prediction_h = float(edge_pred_rgb.shape[-2])
                                train_prediction_w = float(edge_pred_rgb.shape[-1])
                                train_alpha_target_h = float(edge_target_rgb.shape[-2])
                                train_alpha_target_w = float(edge_target_rgb.shape[-1])
                                train_supervision_mask_h = float(expanded_rgb_mask.shape[-2])
                                train_supervision_mask_w = float(expanded_rgb_mask.shape[-1])

                            edge_rgb_l1_map = (edge_pred_rgb.float() - edge_target_rgb.float()).abs().mean(dim=1, keepdim=True)
                            edge_summary = _summarize_weighted_seam_regions(
                                error_map=edge_rgb_l1_map,
                                seam_maps=edge_seam_maps,
                                region_weights=region_weights,
                                normalize_region_losses=False,
                                continuation_weighted_mask=edge_seam_maps["continuation_distance_weighted"],
                                seam_config=seam_config,
                            )
                            if edge_summary["region_losses"]:
                                seam_rgb_edge_summaries.append(edge_summary)
                                edge_halo_metrics = _compute_edge_halo_copy_metrics(
                                    edge_rgb_l1_map,
                                    edge_name=edge_name,
                                    edge_seam_maps=edge_seam_maps,
                                    crop_x0=int(decode_crop["pixel_x0"]),
                                    crop_y0=int(decode_crop["pixel_y0"]),
                                    full_height=int(batch_images.shape[-2]),
                                    full_width=int(batch_images.shape[-1]),
                                    expanded_halo_px=(int(training_expanded_halo_px) if training_expanded_supervision_enabled else 0),
                                )
                                halo_copy_raw_sum = halo_copy_raw_sum + edge_halo_metrics["halo_copy"]["raw_sum"]
                                halo_copy_area = halo_copy_area + edge_halo_metrics["halo_copy"]["area"]
                                halo_copy_max = torch.maximum(halo_copy_max, edge_halo_metrics["halo_copy"]["max"])
                                halo_inner_1px_raw_sum = halo_inner_1px_raw_sum + edge_halo_metrics["halo_inner_1px"]["raw_sum"]
                                halo_inner_1px_area = halo_inner_1px_area + edge_halo_metrics["halo_inner_1px"]["area"]
                                halo_inner_4px_raw_sum = halo_inner_4px_raw_sum + edge_halo_metrics["halo_inner_4px"]["raw_sum"]
                                halo_inner_4px_area = halo_inner_4px_area + edge_halo_metrics["halo_inner_4px"]["area"]
                                halo_inner_8px_raw_sum = halo_inner_8px_raw_sum + edge_halo_metrics["halo_inner_8px"]["raw_sum"]
                                halo_inner_8px_area = halo_inner_8px_area + edge_halo_metrics["halo_inner_8px"]["area"]
                                halo_inner_16px_raw_sum = halo_inner_16px_raw_sum + edge_halo_metrics["halo_inner_16px"]["raw_sum"]
                                halo_inner_16px_area = halo_inner_16px_area + edge_halo_metrics["halo_inner_16px"]["area"]
                                edge_gradient_summary = _compute_continuation_gradient_loss(
                                    pred_rgb=edge_pred_rgb.float(),
                                    target_rgb=edge_target_rgb.float(),
                                    continuation_distance_weighted_mask=edge_seam_maps["continuation_distance_weighted"].float(),
                                    sobel_radius_px=int(seam_config.get("gradient_loss_sobel_radius_px", 1)),
                                )
                                seam_gradient_raw_sum = seam_gradient_raw_sum + edge_gradient_summary["raw_sum"]
                                seam_gradient_active_px = seam_gradient_active_px + edge_gradient_summary["active_px"]
                                seam_gradient_weighted_raw_sum = seam_gradient_weighted_raw_sum + edge_gradient_summary["weighted_raw_sum"]
                                seam_gradient_weight_sum = seam_gradient_weight_sum + edge_gradient_summary["weight_sum"]
                                edge_halo_gradient_summary = _compute_masked_rgb_gradient_loss(
                                    pred_rgb=edge_pred_rgb.float(),
                                    target_rgb=edge_target_rgb.float(),
                                    weight_mask=(edge_seam_maps["margin_inner"] + edge_seam_maps["margin_outer"]).clamp(0.0, 1.0).float(),
                                    sobel_radius_px=int(seam_config.get("gradient_loss_sobel_radius_px", 1)),
                                )
                                seam_halo_gradient_raw_sum = seam_halo_gradient_raw_sum + edge_halo_gradient_summary["raw_sum"]
                                seam_halo_gradient_active_px = seam_halo_gradient_active_px + edge_halo_gradient_summary["active_px"]

                        halo_mask = (seam_maps_supervised["margin_inner"] + seam_maps_supervised["margin_outer"]).clamp(0.0, 1.0)
                        train_geometry_mode = "expanded" if training_expanded_supervision_enabled else "interior"
                        seam_rgb_summary = _aggregate_weighted_seam_summaries(
                            seam_rgb_edge_summaries,
                            reference_tensor=noisy_latents,
                            region_weights=region_weights,
                            continuation_width_px=float(seam_config.get("continuation_width_px", 1.0)),
                        )
                        seam_rgb_region_losses = seam_rgb_summary["region_losses"]
                        seam_continuation_distance_band_px_count_value = float(seam_rgb_summary["continuation_supervised_px"].detach().item())
                        seam_gradient_summary = {
                            "raw_loss": seam_gradient_raw_sum / seam_gradient_active_px.clamp_min(1e-6)
                            if float(seam_gradient_active_px.detach().item()) > 0.0
                            else noisy_latents.new_tensor(0.0),
                            "weighted_loss": seam_gradient_weighted_raw_sum / seam_gradient_weight_sum.clamp_min(1e-6)
                            if float(seam_gradient_weight_sum.detach().item()) > 0.0
                            else noisy_latents.new_tensor(0.0),
                            "active_px": seam_gradient_active_px,
                            "weight_sum": seam_gradient_weight_sum,
                        }
                        seam_continuation_gradient_loss_value = float(seam_gradient_summary["weighted_loss"].detach().item())
                        seam_continuation_gradient_loss_raw_value = float(seam_gradient_summary["raw_loss"].detach().item())

                        if seam_rgb_region_losses:
                            seam_rgb_total_loss = seam_rgb_summary["total_loss"] + (
                                seam_continuation_gradient_loss_weight_value
                                * seam_gradient_summary["weighted_loss"]
                            )
                            loss = loss + (float(seam_config.get("rgb_recon_loss_weight", 0.0)) * seam_rgb_total_loss)
                            seam_rgb_total_loss_value = float(seam_rgb_total_loss.detach().item())
                            seam_rgb_halo_loss_raw_value = float(seam_rgb_summary["halo_loss_raw"].detach().item())
                            seam_rgb_interior_loss_raw_value = float(seam_rgb_summary["interior_loss_raw"].detach().item())
                            seam_rgb_halo_loss_weighted_value = float(seam_rgb_summary["halo_loss_weighted"].detach().item())
                            seam_rgb_interior_loss_weighted_value = float(seam_rgb_summary["interior_loss_weighted"].detach().item())
                            seam_rgb_halo_inner_loss_raw_value = float(
                                seam_rgb_summary["region_raw_sums"]["halo_inner"].detach().item()
                            )
                            seam_rgb_halo_outer_loss_raw_value = float(
                                seam_rgb_summary["region_raw_sums"]["halo_outer"].detach().item()
                            )
                            seam_rgb_halo_inner_loss_weighted_value = float(
                                seam_rgb_summary["region_weighted_contributions"]["halo_inner"].detach().item()
                            )
                            seam_rgb_halo_outer_loss_weighted_value = float(
                                seam_rgb_summary["region_weighted_contributions"]["halo_outer"].detach().item()
                            )
                            seam_halo_supervised_px_value = float(seam_rgb_summary["halo_supervised_px"].detach().item())
                            seam_interior_supervised_px_value = float(seam_rgb_summary["interior_supervised_px"].detach().item())
                            seam_rgb_continuation_weighted_loss_value = float(seam_rgb_summary["continuation_loss_weighted"].detach().item())
                            seam_continuation_rgb_loss_raw_value = float(seam_rgb_summary["continuation_loss_raw"].detach().item())
                            seam_continuation_weight_sum_value = float(seam_rgb_summary["continuation_weight_sum"].detach().item())
                            seam_continuation_rgb_weight_mode_value = str(seam_rgb_summary.get("continuation_rgb_weight_mode", seam_continuation_rgb_weight_mode_value))
                            seam_continuation_effective_weight_mean_value = float(seam_rgb_summary["continuation_effective_weight_mean"].detach().item())
                            seam_continuation_effective_weight_max_value = float(seam_rgb_summary["continuation_effective_weight_max"].detach().item())
                            seam_continuation_weight_fraction_first_8px_value = float(seam_rgb_summary["continuation_weight_fraction_first_8px"].detach().item())
                            seam_continuation_weight_fraction_first_16px_value = float(seam_rgb_summary["continuation_weight_fraction_first_16px"].detach().item())
                            seam_continuation_weight_fraction_first_24px_value = float(seam_rgb_summary["continuation_weight_fraction_first_24px"].detach().item())
                            seam_continuation_weight_fraction_full_band_value = float(seam_rgb_summary["continuation_weight_fraction_full_band"].detach().item())
                            seam_continuation_rgb_loss_bin_0_8_value = float(seam_rgb_summary["continuation_rgb_loss_bin_0_8"].detach().item())
                            seam_continuation_rgb_loss_bin_8_16_value = float(seam_rgb_summary["continuation_rgb_loss_bin_8_16"].detach().item())
                            seam_continuation_rgb_loss_bin_16_24_value = float(seam_rgb_summary["continuation_rgb_loss_bin_16_24"].detach().item())
                            seam_continuation_rgb_loss_bin_24_32_value = float(seam_rgb_summary["continuation_rgb_loss_bin_24_32"].detach().item())
                            seam_continuation_rgb_loss_bin_32_40_value = float(seam_rgb_summary["continuation_rgb_loss_bin_32_40"].detach().item())
                            seam_continuation_rgb_loss_bin_40_48_value = float(seam_rgb_summary["continuation_rgb_loss_bin_40_48"].detach().item())
                            expanded_halo_copy_diff_mean_value = float(
                                (halo_copy_raw_sum / halo_copy_area.clamp_min(1e-6)).detach().item()
                            ) if float(halo_copy_area.detach().item()) > 0.0 else 0.0
                            expanded_halo_copy_diff_max_value = float(halo_copy_max.detach().item())
                            halo_inner_edge_1px_rgb_loss_value = float(
                                (halo_inner_1px_raw_sum / halo_inner_1px_area.clamp_min(1e-6)).detach().item()
                            ) if float(halo_inner_1px_area.detach().item()) > 0.0 else 0.0
                            halo_inner_edge_4px_rgb_loss_value = float(
                                (halo_inner_4px_raw_sum / halo_inner_4px_area.clamp_min(1e-6)).detach().item()
                            ) if float(halo_inner_4px_area.detach().item()) > 0.0 else 0.0
                            halo_inner_8px_rgb_loss_value = float(
                                (halo_inner_8px_raw_sum / halo_inner_8px_area.clamp_min(1e-6)).detach().item()
                            ) if float(halo_inner_8px_area.detach().item()) > 0.0 else 0.0
                            halo_inner_16px_rgb_loss_value = float(
                                (halo_inner_16px_raw_sum / halo_inner_16px_area.clamp_min(1e-6)).detach().item()
                            ) if float(halo_inner_16px_area.detach().item()) > 0.0 else 0.0
                            seam_halo_gradient_loss_value = float(
                                (seam_halo_gradient_raw_sum / seam_halo_gradient_active_px.clamp_min(1e-6)).detach().item()
                            ) if float(seam_halo_gradient_active_px.detach().item()) > 0.0 else 0.0
                            seam_halo_to_interior_px_ratio_value = seam_halo_supervised_px_value / max(seam_interior_supervised_px_value, 1e-6)
                            halo_area = halo_mask.sum(dim=(1, 2, 3)).clamp_min(1e-6)
                            halo_target_energy = ((batch_images.float().abs().mean(dim=1, keepdim=True) * halo_mask).sum(dim=(1, 2, 3)) / halo_area).mean()
                            seam_halo_target_energy_value = float(halo_target_energy.detach().item())
                            if verification_config.get("enable_seam_diagnostic_guards", False):
                                min_halo_energy = float(verification_config.get("seam_halo_target_energy_min", 1e-3))
                                if seam_halo_supervised_px_value > 0.0 and seam_halo_target_energy_value <= min_halo_energy:
                                    raise RuntimeError(
                                        "seam halo target is structurally empty or constant despite supervised halo pixels: "
                                        + f"halo_target_energy={seam_halo_target_energy_value:.6f} "
                                        + f"halo_supervised_px={seam_halo_supervised_px_value:.1f}"
                                    )

                            if (
                                verification_config.get("save_seam_visual_debug", False)
                                and (step_for_logging - int(resumed_step)) <= int(verification_config.get("seam_visual_debug_max_steps", 2))
                            ):
                                _save_seam_visual_debug(
                                    output_dir=args.output_dir,
                                    step=step_for_logging,
                                    pred_rgb=pred_rgb,
                                    target_rgb=target_rgb,
                                    supervision_mask=seam_supervision_mask_rgb,
                                    seam_maps=seam_maps_rgb,
                                )

                        seam_rgb_margin_inner_loss_value = float(
                            seam_rgb_region_losses.get("margin_inner", torch.tensor(0.0, device=accelerator.device)).detach().item()
                        )
                        seam_rgb_margin_outer_loss_value = float(
                            seam_rgb_region_losses.get("margin_outer", torch.tensor(0.0, device=accelerator.device)).detach().item()
                        )
                        seam_rgb_interior_inner_loss_value = float(
                            seam_rgb_region_losses.get("interior_inner", torch.tensor(0.0, device=accelerator.device)).detach().item()
                        )
                        seam_rgb_interior_outer_loss_value = float(
                            seam_rgb_region_losses.get("interior_outer", torch.tensor(0.0, device=accelerator.device)).detach().item()
                        )
                        seam_rgb_halo_inner_loss_value = float(
                            seam_rgb_region_losses.get("halo_inner", torch.tensor(0.0, device=accelerator.device)).detach().item()
                        )
                        seam_rgb_halo_outer_loss_value = float(
                            seam_rgb_region_losses.get("halo_outer", torch.tensor(0.0, device=accelerator.device)).detach().item()
                        )
                        seam_rgb_interior_core_loss_value = float(
                            seam_rgb_region_losses.get("interior_core", torch.tensor(0.0, device=accelerator.device)).detach().item()
                        )
                        seam_continuation_loss_value = seam_rgb_continuation_weighted_loss_value
                        seam_edge_recon_score = 0.5 * (seam_rgb_margin_inner_loss_value + seam_rgb_margin_outer_loss_value)
                        prompt_conflict_score = seam_rgb_total_loss_value / max(seam_margin_inner_coverage_ratio, 1e-3)

                    bind_weight = float(alpha_config.get("bind_preference_weight", 0.0))
                    bind_ratio = float(alpha_config.get("bind_paired_batch_ratio", 0.0))
                    bind_enabled = bind_weight > 0.0 and bind_ratio > 0.0
                    if seam_config["enabled"] and bind_enabled:
                        bind_enabled = False
                    bind_active_batch = 0.0
                    bind_active_count = 0.0
                    precondition_c_true_count = 0.0
                    candidate_c_count = 0.0
                    eligible_c_count = 0.0
                    requested_c_count = 0.0
                    realized_c_count = 0.0
                    fallback_from_c_count = 0.0
                    rejected_c_count_total = 0.0
                    rejected_c_count_precondition = 0.0
                    retry_count_for_c = 0.0
                    delta_c_mean = 0.0
                    delta_c_min = 0.0
                    delta_c_max = 0.0
                    drop_stage_for_c = "none"
                    bind_negative_mode_requested = "none"
                    bind_negative_mode_realized = "none"
                    bind_negative_delta_retry_count = 0.0
                    bind_negative_fallback_triggered = 0.0
                    bind_negative_fallback_reason = "none"
                    bind_negative_delta_mean = 0.0
                    bind_schedule_name = str(alpha_config.get("bind_negative_mode_schedule", "none"))
                    bind_effective_weight_a = 0.0
                    bind_effective_weight_b = 0.0
                    bind_effective_weight_c = 0.0

                    bind_negative_mode = "none"
                    effective_mode_weights = _resolve_effective_mode_weights(alpha_config, current_step=global_step + 1)
                    bind_effective_weight_a = float(effective_mode_weights.get("wall_to_flat_suppression", 0.0))
                    bind_effective_weight_b = float(effective_mode_weights.get("flat_to_wall_injection", 0.0))
                    bind_effective_weight_c = float(effective_mode_weights.get("local_spatial_misalignment", 0.0))
                    if (global_step + 1) in set(alpha_config.get("bind_negative_mode_schedule_probe_steps", [])):
                        logger.info(
                            "[bind/schedule] "
                            + f"step={global_step + 1} schedule={bind_schedule_name} "
                            + f"wA={bind_effective_weight_a:.6f} "
                            + f"wB={bind_effective_weight_b:.6f} "
                            + f"wC={bind_effective_weight_c:.6f} "
                            + f"sum={bind_effective_weight_a + bind_effective_weight_b + bind_effective_weight_c:.6f}"
                        )
                    if bind_enabled and bind_pair_rng.random() < bind_ratio:
                        bind_pair_batch_count += 1
                        bind_active_count = 1.0
                        bind_negative_mode_requested = _sample_negative_mode(alpha_config, bind_pair_rng, current_step=global_step + 1)
                        bind_negative_mode_realized = bind_negative_mode_requested

                        precondition_c_ok, precondition_c_reason = _mode_c_precondition(
                            dataset,
                            conditioning_images,
                            terrain_mask_black_is_terrain=bool(alpha_config.get("terrain_mask_black_is_terrain", True)),
                        )
                        if precondition_c_ok:
                            precondition_c_true_count = 1.0

                        if bind_negative_mode_requested == "local_spatial_misalignment":
                            candidate_c_count = 1.0
                            if precondition_c_ok:
                                eligible_c_count = 1.0
                                requested_c_count = 1.0
                            else:
                                rejected_c_count_total = 1.0
                                rejected_c_count_precondition = 1.0
                                drop_stage_for_c = "eligibility"
                                bind_negative_fallback_triggered = 1.0
                                fallback_from_c_count = 1.0
                                bind_negative_fallback_reason = precondition_c_reason
                                bind_negative_mode_realized = "wall_to_flat_suppression"

                        corrupted_conditioning, bind_negative_mode_realized, bind_negative_delta_mean = _build_corrupted_geometry_conditioning(
                            dataset,
                            conditioning_images,
                            mode=bind_negative_mode_realized,
                            rng=bind_pair_rng,
                            alpha_config=alpha_config,
                            assigned_crop_class=batch.get("assigned_crop_class", None),
                            special_structure_tags=batch.get("special_structure_tags", None),
                        )

                        if bind_negative_mode_requested == "local_spatial_misalignment" and bind_negative_mode_realized == "local_spatial_misalignment":
                            realized_c_count = 1.0
                            drop_stage_for_c = "none"
                            delta_c_mean = bind_negative_delta_mean
                            delta_c_min = bind_negative_delta_mean
                            delta_c_max = bind_negative_delta_mean
                        elif bind_negative_mode_requested == "local_spatial_misalignment" and bind_negative_mode_realized != "local_spatial_misalignment":
                            if drop_stage_for_c == "none":
                                drop_stage_for_c = "fallback"
                            bind_negative_fallback_triggered = 1.0
                            fallback_from_c_count = 1.0
                            if bind_negative_fallback_reason == "none":
                                bind_negative_fallback_reason = "mode_realization_mismatch"

                        bind_negative_mode = bind_negative_mode_realized
                        if (global_step + 1) in set(alpha_config.get("bind_negative_mode_schedule_probe_steps", [])):
                            logger.info(
                                "[bind/realized] "
                                + f"step={global_step + 1} requested={bind_negative_mode_requested} "
                                + f"realized={bind_negative_mode_realized} c_ok={realized_c_count} "
                                + f"reason={bind_negative_fallback_reason if bind_negative_fallback_reason != 'none' else 'success'}"
                            )
                        input_resi_add_neg, mid_add_neg = control_net(
                            noisy_latents,
                            timesteps,
                            text_embedding,
                            vector_embedding,
                            corrupted_conditioning,
                        )
                        noise_pred_neg = unet(
                            noisy_latents,
                            timesteps,
                            text_embedding,
                            vector_embedding,
                            input_resi_add_neg,
                            mid_add_neg,
                        )
                        neg_loss_map = F.mse_loss(noise_pred_neg.float(), noise.float(), reduction="none")
                        neg_pixel_loss = neg_loss_map.mean(dim=1, keepdim=True)
                        neg_masked_sum = (neg_pixel_loss * latent_mask).sum(dim=(1, 2, 3))
                        neg_loss_per_sample = neg_masked_sum / masked_area
                        diffusion_loss_neg = neg_loss_per_sample.mean()
                        ranking_gap = diffusion_loss_neg - diffusion_loss
                        bind_loss = F.relu(-ranking_gap)
                        loss = loss + (bind_weight * bind_loss)
                        bind_active_batch = 1.0

                        bind_pos_loss_value = float(diffusion_loss.detach().item())
                        bind_neg_loss_value = float(diffusion_loss_neg.detach().item())
                        bind_ranking_gap_value = float(ranking_gap.detach().item())
                        bind_loss_value = float(bind_loss.detach().item())
                        bind_win_rate_value = float((neg_loss_per_sample > pos_loss_per_sample).float().mean().detach().item())
                        bind_negative_delta_retry_count = retry_count_for_c
                        if alpha_config.get("bind_log_margin_variant", True):
                            bind_margin = float(alpha_config.get("bind_preference_margin", 0.0))
                            bind_margin_loss = F.relu(
                                torch.tensor(bind_margin, device=ranking_gap.device, dtype=ranking_gap.dtype) - ranking_gap
                            )
                            bind_margin_loss_value = float(bind_margin_loss.detach().item())

                    if alpha_config["enabled"]:
                        if alpha_outputs is None:
                            raise RuntimeError("alpha mode enabled but alpha outputs were not produced")
                        if alpha_target is None:
                            raise RuntimeError("alpha mode enabled but alpha targets are missing from the batch")

                        alpha_prior_weight = _linear_schedule(
                            alpha_config["prior_start_weight"],
                            alpha_config["prior_end_weight"],
                            global_step,
                            alpha_config["warmup_steps"],
                        )
                        alpha_temperature = _linear_schedule(
                            alpha_config["logit_temperature_start"],
                            alpha_config["logit_temperature_end"],
                            global_step,
                            alpha_config["warmup_steps"],
                        )

                        binary_alpha_target = (alpha_target >= alpha_config["binary_threshold"]).to(dtype=weight_dtype)
                        terrain_mask_prior_raw = conditioning_images[:, terrain_mask_index : terrain_mask_index + 1].clamp(0.0, 1.0)
                        if terrain_mask_prior_raw.shape[-2:] != alpha_target.shape[-2:]:
                            terrain_mask_prior_raw = F.interpolate(
                                terrain_mask_prior_raw,
                                size=alpha_target.shape[-2:],
                                mode="area",
                            )
                        terrain_mask_prior = _terrain_mask_to_occupancy(
                            terrain_mask_prior_raw,
                            bool(alpha_config["terrain_mask_black_is_terrain"]),
                        )
                        blended_alpha_target = (
                            (1.0 - alpha_prior_weight) * binary_alpha_target
                            + alpha_prior_weight * terrain_mask_prior
                        )

                        # Alpha loss is scored only over the interior crop region (no expanded seam halo).
                        supervision_mask = torch.ones_like(alpha_target, dtype=weight_dtype)
                        if training_expanded_supervision_enabled and int(training_expanded_halo_px) > 0:
                            _halo = int(training_expanded_halo_px)
                            _h = int(alpha_target.shape[-2])
                            _w = int(alpha_target.shape[-1])
                            if _h <= (2 * _halo) or _w <= (2 * _halo):
                                raise ValueError(
                                    f"invalid expanded interior geometry for alpha supervision: hw=({_h},{_w}) halo={_halo}"
                                )
                            supervision_mask = torch.zeros_like(alpha_target, dtype=weight_dtype)
                            supervision_mask[:, :, _halo:-_halo, _halo:-_halo] = 1.0
                        selected_alpha_logits = _select_alpha_logits(alpha_outputs, alpha_config["output_source"])
                        assert selected_alpha_logits.shape == alpha_target.shape == supervision_mask.shape, (
                            f"seam alpha geometry mismatch pred={tuple(selected_alpha_logits.shape)} target={tuple(alpha_target.shape)} mask={tuple(supervision_mask.shape)}"
                        )
                        train_prediction_h = float(selected_alpha_logits.shape[-2])
                        train_prediction_w = float(selected_alpha_logits.shape[-1])
                        train_alpha_target_h = float(alpha_target.shape[-2])
                        train_alpha_target_w = float(alpha_target.shape[-1])
                        train_supervision_mask_h = float(supervision_mask.shape[-2])
                        train_supervision_mask_w = float(supervision_mask.shape[-1])
                        scaled_alpha_logits = selected_alpha_logits / max(alpha_temperature, 1e-6)
                        alpha_bce_map = F.binary_cross_entropy_with_logits(
                            scaled_alpha_logits.float(),
                            blended_alpha_target.float(),
                            reduction="none",
                        )
                        alpha_bce = (alpha_bce_map * supervision_mask.float()).sum() / supervision_mask.float().sum().clamp_min(1e-6)

                        edge_band = _build_edge_band(binary_alpha_target.float(), alpha_config["edge_dilate_px"])
                        edge_weight_map = 1.0 + (alpha_config["edge_band_weight"] * edge_band)
                        weighted_supervision = supervision_mask.float() * edge_weight_map
                        alpha_edge = (alpha_bce_map * weighted_supervision).sum() / weighted_supervision.sum().clamp_min(1e-6)

                        pred_alpha_prob = torch.sigmoid(scaled_alpha_logits.float())
                        terrain_mask_target = terrain_mask_prior.float()
                        terrain_bce_map = F.binary_cross_entropy_with_logits(
                            scaled_alpha_logits.float(),
                            terrain_mask_target,
                            reduction="none",
                        )
                        terrain_bce_per_sample = _masked_mean_per_sample(terrain_bce_map, supervision_mask.float())
                        terrain_presence = terrain_mask_target.mean(dim=(1, 2, 3))
                        terrain_presence_gate = (
                            (terrain_presence - float(alpha_config["terrain_presence_floor"]))
                            / max(1.0 - float(alpha_config["terrain_presence_floor"]), 1e-6)
                        ).clamp(0.0, 1.0)
                        terrain_sample_weight = 1.0 + (float(alpha_config["terrain_presence_boost"]) * terrain_presence_gate)
                        terrain_bce = (terrain_bce_per_sample * terrain_sample_weight).sum() / terrain_sample_weight.sum().clamp_min(1e-6)

                        terrain_intersection = _masked_mean_per_sample(
                            pred_alpha_prob * terrain_mask_target,
                            supervision_mask.float(),
                        )
                        terrain_union = _masked_mean_per_sample(
                            pred_alpha_prob + terrain_mask_target - (pred_alpha_prob * terrain_mask_target),
                            supervision_mask.float(),
                        ).clamp_min(1e-6)
                        terrain_iou_loss = (1.0 - (terrain_intersection / terrain_union)).mean()

                        terrain_curriculum_factor = _linear_schedule(
                            float(alpha_config["terrain_curriculum_start"]),
                            float(alpha_config["terrain_curriculum_end"]),
                            global_step,
                            int(alpha_config["terrain_curriculum_steps"]),
                        )
                        terrain_coupling_loss = terrain_curriculum_factor * (
                            float(alpha_config["terrain_coupling_weight"]) * terrain_bce
                            + float(alpha_config["terrain_iou_weight"]) * terrain_iou_loss
                        )

                        alpha_total = alpha_config["loss_weight"] * (
                            alpha_bce + (alpha_config["edge_loss_scale"] * alpha_edge)
                        )

                        seam_alpha_local_loss = None
                        if (
                            False
                            and seam_config["enabled"]
                            and float(seam_config.get("alpha_local_loss_weight", 0.0)) > 0.0
                            and seam_maps_supervised is not None
                        ):
                            seam_maps = seam_maps_supervised

                            seam_region_losses: Dict[str, torch.Tensor] = {}
                            for region_name, region_mask in seam_maps.items():
                                area = region_mask.sum()
                                if float(area.detach().item()) <= 0.0:
                                    continue
                                seam_region_losses[region_name] = (alpha_bce_map * region_mask).sum() / area.clamp_min(1e-6)

                            region_weights = {
                                "margin_inner": float(seam_config.get("margin_inner_weight", 1.0)),
                                "margin_outer": float(seam_config.get("margin_outer_weight", 0.7)),
                                "interior_inner": float(seam_config.get("interior_band_inner_weight", 0.8)),
                                "interior_outer": float(seam_config.get("interior_band_outer_weight", 0.5)),
                            }
                            seam_alpha_summary = _summarize_weighted_seam_regions(
                                error_map=alpha_bce_map,
                                seam_maps=seam_maps,
                                region_weights=region_weights,
                                normalize_region_losses=bool(seam_config.get("normalize_region_losses", True)),
                            )
                            seam_region_losses = seam_alpha_summary["region_losses"]

                            if seam_region_losses:
                                seam_alpha_local_loss = seam_alpha_summary["total_loss"]
                                seam_alpha_halo_loss_raw_value = float(seam_alpha_summary["halo_loss_raw"].detach().item())
                                seam_alpha_interior_loss_raw_value = float(seam_alpha_summary["interior_loss_raw"].detach().item())
                                seam_alpha_halo_loss_weighted_value = float(seam_alpha_summary["halo_loss_weighted"].detach().item())
                                seam_alpha_interior_loss_weighted_value = float(seam_alpha_summary["interior_loss_weighted"].detach().item())

                            seam_margin_inner_loss_value = float(
                                seam_region_losses.get("margin_inner", torch.tensor(0.0, device=accelerator.device)).detach().item()
                            )
                            seam_margin_outer_loss_value = float(
                                seam_region_losses.get("margin_outer", torch.tensor(0.0, device=accelerator.device)).detach().item()
                            )
                            seam_interior_inner_loss_value = float(
                                seam_region_losses.get("interior_inner", torch.tensor(0.0, device=accelerator.device)).detach().item()
                            )
                            seam_interior_outer_loss_value = float(
                                seam_region_losses.get("interior_outer", torch.tensor(0.0, device=accelerator.device)).detach().item()
                            )
                            seam_continuation_loss_value = (
                                seam_interior_inner_loss_value + seam_interior_outer_loss_value
                            )

                        if seam_alpha_local_loss is not None:
                            alpha_total = alpha_total + float(seam_config["alpha_local_loss_weight"]) * seam_alpha_local_loss

                        alpha_total = alpha_total + terrain_coupling_loss
                        loss = loss + alpha_total

                        diffusion_loss_value = float(diffusion_loss.detach().item())
                        alpha_bce_value = float(alpha_bce.detach().item())
                        alpha_edge_value = float(alpha_edge.detach().item())
                        alpha_terrain_bce_value = float(terrain_bce.detach().item())
                        alpha_terrain_iou_value = float(terrain_iou_loss.detach().item())
                        seam_alpha_local_value = (
                            0.0 if seam_alpha_local_loss is None else float(seam_alpha_local_loss.detach().item())
                        )
                        alpha_total_value = float(alpha_total.detach().item())
                    else:
                        diffusion_loss_value = float(diffusion_loss.detach().item())

                    rgb_region_contrib = seam_rgb_summary.get("region_weighted_contributions", {}) if isinstance(seam_rgb_summary, dict) else {}
                    alpha_region_contrib = seam_alpha_summary.get("region_weighted_contributions", {}) if isinstance(seam_alpha_summary, dict) else {}
                    rgb_scale = float(seam_config.get("rgb_recon_loss_weight", 0.0))
                    alpha_scale = float(seam_config.get("alpha_local_loss_weight", 0.0))

                    def _region_term_value(source: Dict[str, torch.Tensor], key: str) -> float:
                        if key not in source:
                            return 0.0
                        return float(source[key].detach().item())

                    halo_inner_weighted_contribution = (
                        (rgb_scale * _region_term_value(rgb_region_contrib, "halo_inner"))
                        + (alpha_scale * _region_term_value(alpha_region_contrib, "margin_inner"))
                    )
                    halo_outer_weighted_contribution = (
                        (rgb_scale * _region_term_value(rgb_region_contrib, "halo_outer"))
                        + (alpha_scale * _region_term_value(alpha_region_contrib, "margin_outer"))
                    )
                    interior_continuation_weighted_contribution = (
                        (rgb_scale * _region_term_value(rgb_region_contrib, "continuation_distance_weighted"))
                        + (alpha_scale * _region_term_value(alpha_region_contrib, "interior_inner"))
                    )
                    interior_core_weighted_contribution = (
                        (rgb_scale * _region_term_value(rgb_region_contrib, "interior_core"))
                        + (alpha_scale * _region_term_value(alpha_region_contrib, "interior_outer"))
                    )
                    continuation_gradient_weighted_contribution = (
                        rgb_scale
                        * seam_continuation_gradient_loss_weight_value
                        * seam_continuation_gradient_loss_value
                    )

                    seam_total_contribution = (
                        halo_inner_weighted_contribution
                        + halo_outer_weighted_contribution
                        + interior_continuation_weighted_contribution
                        + interior_core_weighted_contribution
                        + continuation_gradient_weighted_contribution
                    )
                    seam_loss_contribution_ratio_value = seam_total_contribution / max(float(loss.detach().item()), 1e-6)
                    if verification_config.get("enable_seam_diagnostic_guards", False) and step_for_logging == 1:
                        min_ratio = float(verification_config.get("seam_loss_contribution_ratio_min", 0.05))
                        if seam_loss_contribution_ratio_value <= min_ratio:
                            raise RuntimeError(
                                "seam loss contribution is too small on the first batch: "
                                + f"ratio={seam_loss_contribution_ratio_value:.6f} threshold={min_ratio:.6f}"
                            )

                    if verification_log_now:
                        logger.info(
                            "[verify/seam_geometry] step=%d geometry_mode=%s pred_hw=%sx%s target_hw=%sx%s supervision_hw=%sx%s target_x=[%.1f,%.1f] target_y=[%.1f,%.1f] halo_bounds=[x:%.1f-%.1f,y:%.1f-%.1f] halo_target_energy=%.6f halo_loss_ratio=%.6f",
                            step_for_logging,
                            train_geometry_mode,
                            int(train_prediction_h),
                            int(train_prediction_w),
                            int(train_alpha_target_h),
                            int(train_alpha_target_w),
                            int(train_supervision_mask_h),
                            int(train_supervision_mask_w),
                            train_target_min_x,
                            train_target_max_x,
                            train_target_min_y,
                            train_target_max_y,
                            train_halo_min_x,
                            train_halo_max_x,
                            train_halo_min_y,
                            train_halo_max_y,
                            seam_halo_target_energy_value,
                            seam_loss_contribution_ratio_value,
                        )
                        logger.info(
                            "[verify/seam_region_contrib] step=%d halo_inner_weighted_contribution=%.6f halo_outer_weighted_contribution=%.6f interior_continuation_weighted_contribution=%.6f interior_core_weighted_contribution=%.6f continuation_gradient_weighted_contribution=%.6f",
                            step_for_logging,
                            halo_inner_weighted_contribution,
                            halo_outer_weighted_contribution,
                            interior_continuation_weighted_contribution,
                            interior_core_weighted_contribution,
                            continuation_gradient_weighted_contribution,
                        )

                    if diffusion_loss_value > 0.0:
                        prompt_conflict_score_norm = prompt_conflict_score / max(diffusion_loss_value, 1e-6)

                    if verification_log_now and accelerator.sync_gradients:
                        pre_step_param_stats = _collect_trainable_module_stats(control_net)

                accelerator.backward(loss)
                grad_stats = None
                if verification_log_now and accelerator.sync_gradients:
                    grad_stats = _collect_trainable_grad_stats(control_net)
                if accelerator.sync_gradients:
                    if args.max_grad_norm > 0.0:
                        accelerator.clip_grad_norm_(control_net.parameters(), args.max_grad_norm)
                optimizer.step()
                if ema_state is not None:
                    update_ema_state(unwrap_model(accelerator, control_net), ema_state, float(args.ema_decay))
                post_step_param_stats = None
                if verification_log_now and accelerator.sync_gradients and pre_step_param_stats is not None:
                    post_step_param_stats = _collect_trainable_module_stats(control_net)
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                loss_value = float(loss.detach().item())
                progress_bar.update(1)
                if alpha_config["enabled"]:
                    progress_bar.set_postfix(loss=loss_value, diffusion=diffusion_loss_value, alpha=alpha_total_value)
                else:
                    progress_bar.set_postfix(loss=loss_value)

                if grad_stats is not None:
                    grad_l2_alpha = float(grad_stats.get("alpha_heads", {}).get("l2", 0.0))
                    grad_l2_control_residual = float(grad_stats.get("control_residual_blocks", {}).get("l2", 0.0))
                    grad_l2_control_mid = float(grad_stats.get("control_mid_block", {}).get("l2", 0.0))
                if pre_step_param_stats is not None and post_step_param_stats is not None:
                    _deltas_for_trace = _compute_module_update_deltas(pre_step_param_stats, post_step_param_stats)
                    param_delta_alpha = float(_deltas_for_trace.get("alpha_heads", 0.0))
                    param_delta_control_residual = float(_deltas_for_trace.get("control_residual_blocks", 0.0))
                    param_delta_control_mid = float(_deltas_for_trace.get("control_mid_block", 0.0))

                if global_step == 1 or global_step % max(1, args.loss_trace_every) == 0 or global_step == args.max_train_steps:
                    if alpha_config["enabled"]:
                        _log_alpha_loss_breakdown(
                            global_step,
                            diffusion_loss_value,
                            alpha_bce_value,
                            alpha_edge_value,
                            alpha_total_value,
                            alpha_terrain_bce_value,
                            alpha_terrain_iou_value,
                            terrain_curriculum_factor,
                            alpha_config["dominance_warn_ratio"],
                        )
                    loss_trace_row = {
                            "step": float(global_step),
                            "loss": loss_value,
                            "diffusion_loss": diffusion_loss_value,
                            "alpha_bce_loss": alpha_bce_value,
                            "alpha_edge_loss": alpha_edge_value,
                            "alpha_terrain_bce_loss": alpha_terrain_bce_value,
                            "alpha_terrain_iou_loss": alpha_terrain_iou_value,
                            "seam_alpha_local_loss": seam_alpha_local_value,
                            "seam_margin_inner_loss": seam_margin_inner_loss_value,
                            "seam_margin_outer_loss": seam_margin_outer_loss_value,
                            "seam_interior_inner_loss": seam_interior_inner_loss_value,
                            "seam_interior_outer_loss": seam_interior_outer_loss_value,
                            "seam_continuation_loss": seam_continuation_loss_value,
                            "seam_rgb_margin_inner_loss": seam_rgb_margin_inner_loss_value,
                            "seam_rgb_margin_outer_loss": seam_rgb_margin_outer_loss_value,
                            "seam_rgb_interior_inner_loss": seam_rgb_interior_inner_loss_value,
                            "seam_rgb_interior_outer_loss": seam_rgb_interior_outer_loss_value,
                            "seam_rgb_halo_inner_loss": seam_rgb_halo_inner_loss_value,
                            "seam_rgb_halo_outer_loss": seam_rgb_halo_outer_loss_value,
                            "seam_rgb_halo_inner_loss_raw": seam_rgb_halo_inner_loss_raw_value,
                            "seam_rgb_halo_outer_loss_raw": seam_rgb_halo_outer_loss_raw_value,
                            "seam_rgb_halo_inner_loss_weighted": seam_rgb_halo_inner_loss_weighted_value,
                            "seam_rgb_halo_outer_loss_weighted": seam_rgb_halo_outer_loss_weighted_value,
                            "seam_rgb_interior_core_loss": seam_rgb_interior_core_loss_value,
                            "seam_rgb_continuation_weighted_loss": seam_rgb_continuation_weighted_loss_value,
                            "seam_halo_gradient_loss": seam_halo_gradient_loss_value,
                            "expanded_halo_copy_diff_mean": expanded_halo_copy_diff_mean_value,
                            "expanded_halo_copy_diff_max": expanded_halo_copy_diff_max_value,
                            "halo_inner_edge_1px_rgb_loss": halo_inner_edge_1px_rgb_loss_value,
                            "halo_inner_edge_4px_rgb_loss": halo_inner_edge_4px_rgb_loss_value,
                            "halo_inner_8px_rgb_loss": halo_inner_8px_rgb_loss_value,
                            "halo_inner_16px_rgb_loss": halo_inner_16px_rgb_loss_value,
                            "seam_continuation_rgb_loss_raw": seam_continuation_rgb_loss_raw_value,
                            "seam_continuation_gradient_loss": seam_continuation_gradient_loss_value,
                            "seam_continuation_gradient_loss_raw": seam_continuation_gradient_loss_raw_value,
                            "seam_continuation_width_px": seam_continuation_width_px_value,
                            "seam_continuation_peak_weight": seam_continuation_peak_weight_value,
                            "seam_continuation_falloff_power": seam_continuation_falloff_power_value,
                            "seam_continuation_rgb_weight_mode": seam_continuation_rgb_weight_mode_value,
                            "seam_continuation_gradient_loss_weight": seam_continuation_gradient_loss_weight_value,
                            "seam_continuation_weight_sum": seam_continuation_weight_sum_value,
                            "seam_continuation_effective_weight_mean": seam_continuation_effective_weight_mean_value,
                            "seam_continuation_effective_weight_max": seam_continuation_effective_weight_max_value,
                            "seam_continuation_weight_fraction_first_8px": seam_continuation_weight_fraction_first_8px_value,
                            "seam_continuation_weight_fraction_first_16px": seam_continuation_weight_fraction_first_16px_value,
                            "seam_continuation_weight_fraction_first_24px": seam_continuation_weight_fraction_first_24px_value,
                            "seam_continuation_weight_fraction_full_band": seam_continuation_weight_fraction_full_band_value,
                            "seam_continuation_rgb_loss_bin_0_8": seam_continuation_rgb_loss_bin_0_8_value,
                            "seam_continuation_rgb_loss_bin_8_16": seam_continuation_rgb_loss_bin_8_16_value,
                            "seam_continuation_rgb_loss_bin_16_24": seam_continuation_rgb_loss_bin_16_24_value,
                            "seam_continuation_rgb_loss_bin_24_32": seam_continuation_rgb_loss_bin_24_32_value,
                            "seam_continuation_rgb_loss_bin_32_40": seam_continuation_rgb_loss_bin_32_40_value,
                            "seam_continuation_rgb_loss_bin_40_48": seam_continuation_rgb_loss_bin_40_48_value,
                            "seam_continuation_distance_band_px_count": seam_continuation_distance_band_px_count_value,
                            "seam_rgb_total_loss": seam_rgb_total_loss_value,
                            "seam_edge_recon_score": seam_edge_recon_score,
                            "seam_margin_inner_coverage_ratio": seam_margin_inner_coverage_ratio,
                            "seam_margin_inner_coverage_px": seam_margin_inner_coverage_px,
                            "seam_margin_inner_raw_px": seam_margin_inner_raw_px,
                            "seam_active_edges_count": seam_active_edges_count,
                            "prompt_conflict_score": prompt_conflict_score,
                            "prompt_conflict_score_norm": prompt_conflict_score_norm,
                            "alpha_total_loss": alpha_total_value,
                            "seam_defined_ratio": seam_diag.get("seam_defined_ratio", 0.0),
                            "seam_pre_gate_l2": seam_diag.get("seam_pre_gate_l2", 0.0),
                            "seam_post_gate_l2": seam_diag.get("seam_post_gate_l2", 0.0),
                            "seam_undefined_edges_count": seam_diag.get("seam_undefined_edges_count", 0.0),
                            "seam_undefined_pre_gate_l2": seam_diag.get("seam_undefined_pre_gate_l2", 0.0),
                            "seam_undefined_post_gate_l2": seam_diag.get("seam_undefined_post_gate_l2", 0.0),
                            "seam_halo_supervised_px": seam_halo_supervised_px_value,
                            "seam_interior_supervised_px": seam_interior_supervised_px_value,
                            "seam_halo_to_interior_px_ratio": seam_halo_to_interior_px_ratio_value,
                            "seam_rgb_halo_loss_raw": seam_rgb_halo_loss_raw_value,
                            "seam_rgb_interior_loss_raw": seam_rgb_interior_loss_raw_value,
                            "seam_rgb_halo_loss_weighted": seam_rgb_halo_loss_weighted_value,
                            "seam_rgb_interior_loss_weighted": seam_rgb_interior_loss_weighted_value,
                            "seam_alpha_halo_loss_raw": seam_alpha_halo_loss_raw_value,
                            "seam_alpha_interior_loss_raw": seam_alpha_interior_loss_raw_value,
                            "seam_alpha_halo_loss_weighted": seam_alpha_halo_loss_weighted_value,
                            "seam_alpha_interior_loss_weighted": seam_alpha_interior_loss_weighted_value,
                            "seam_halo_target_energy": seam_halo_target_energy_value,
                            "seam_loss_contribution_ratio": seam_loss_contribution_ratio_value,
                            "halo_inner_weighted_contribution": halo_inner_weighted_contribution,
                            "halo_outer_weighted_contribution": halo_outer_weighted_contribution,
                            "interior_continuation_weighted_contribution": interior_continuation_weighted_contribution,
                            "interior_core_weighted_contribution": interior_core_weighted_contribution,
                            "continuation_gradient_weighted_contribution": continuation_gradient_weighted_contribution,
                            "train_prediction_h": train_prediction_h,
                            "train_prediction_w": train_prediction_w,
                            "train_alpha_target_h": train_alpha_target_h,
                            "train_alpha_target_w": train_alpha_target_w,
                            "train_supervision_mask_h": train_supervision_mask_h,
                            "train_supervision_mask_w": train_supervision_mask_w,
                            "train_target_min_x": train_target_min_x,
                            "train_target_max_x": train_target_max_x,
                            "train_target_min_y": train_target_min_y,
                            "train_target_max_y": train_target_max_y,
                            "train_halo_min_x": train_halo_min_x,
                            "train_halo_max_x": train_halo_max_x,
                            "train_halo_min_y": train_halo_min_y,
                            "train_halo_max_y": train_halo_max_y,
                            "training_expanded_supervision_enabled": 1.0 if training_expanded_supervision_enabled else 0.0,
                            "eval_expanded_prediction_enabled": 1.0 if evaluation_config.get("expanded_prediction_enabled", False) else 0.0,
                            "eval_expanded_halo_px": float(int(evaluation_config.get("expanded_halo_px", 0))),
                            "train_geometry_mode": train_geometry_mode,
                            "seam_edge_north_defined": seam_diag.get("seam_edge_north_defined", 0.0),
                            "seam_edge_south_defined": seam_diag.get("seam_edge_south_defined", 0.0),
                            "seam_edge_east_defined": seam_diag.get("seam_edge_east_defined", 0.0),
                            "seam_edge_west_defined": seam_diag.get("seam_edge_west_defined", 0.0),
                            "seam_edge_north_pre_gate_l2": seam_diag.get("seam_edge_north_pre_gate_l2", 0.0),
                            "seam_edge_south_pre_gate_l2": seam_diag.get("seam_edge_south_pre_gate_l2", 0.0),
                            "seam_edge_east_pre_gate_l2": seam_diag.get("seam_edge_east_pre_gate_l2", 0.0),
                            "seam_edge_west_pre_gate_l2": seam_diag.get("seam_edge_west_pre_gate_l2", 0.0),
                            "seam_edge_north_post_gate_l2": seam_diag.get("seam_edge_north_post_gate_l2", 0.0),
                            "seam_edge_south_post_gate_l2": seam_diag.get("seam_edge_south_post_gate_l2", 0.0),
                            "seam_edge_east_post_gate_l2": seam_diag.get("seam_edge_east_post_gate_l2", 0.0),
                            "seam_edge_west_post_gate_l2": seam_diag.get("seam_edge_west_post_gate_l2", 0.0),
                            "seam_edge_north_visible_l2": seam_diag.get("seam_edge_north_visible_l2", 0.0),
                            "seam_edge_south_visible_l2": seam_diag.get("seam_edge_south_visible_l2", 0.0),
                            "seam_edge_east_visible_l2": seam_diag.get("seam_edge_east_visible_l2", 0.0),
                            "seam_edge_west_visible_l2": seam_diag.get("seam_edge_west_visible_l2", 0.0),
                            "seam_total_edges_defined": seam_total_edges_defined,
                            "seam_valid_edges_for_loss_count": seam_valid_edges_for_loss_count,
                            "seam_valid_edge_ratio": seam_valid_edge_ratio,
                            "seam_dataset_qualified_edges_count": seam_dataset_qualified_edges_count,
                            "seam_trainer_qualified_edges_count": seam_trainer_qualified_edges_count,
                            "seam_halo_without_continuation_count": seam_halo_without_continuation_count,
                            "halo_without_continuation_count_after_gate": halo_without_continuation_count_after_gate,
                            "seam_edge_north_valid_for_seam_supervision": seam_edge_north_valid_for_seam_supervision,
                            "seam_edge_south_valid_for_seam_supervision": seam_edge_south_valid_for_seam_supervision,
                            "seam_edge_east_valid_for_seam_supervision": seam_edge_east_valid_for_seam_supervision,
                            "seam_edge_west_valid_for_seam_supervision": seam_edge_west_valid_for_seam_supervision,
                            "seam_edge_north_halo_inner_px_after_gate": seam_edge_north_halo_inner_px_after_gate,
                            "seam_edge_south_halo_inner_px_after_gate": seam_edge_south_halo_inner_px_after_gate,
                            "seam_edge_east_halo_inner_px_after_gate": seam_edge_east_halo_inner_px_after_gate,
                            "seam_edge_west_halo_inner_px_after_gate": seam_edge_west_halo_inner_px_after_gate,
                            "seam_edge_north_halo_outer_px_after_gate": seam_edge_north_halo_outer_px_after_gate,
                            "seam_edge_south_halo_outer_px_after_gate": seam_edge_south_halo_outer_px_after_gate,
                            "seam_edge_east_halo_outer_px_after_gate": seam_edge_east_halo_outer_px_after_gate,
                            "seam_edge_west_halo_outer_px_after_gate": seam_edge_west_halo_outer_px_after_gate,
                            "seam_edge_north_continuation_px_after_gate": seam_edge_north_continuation_px_after_gate,
                            "seam_edge_south_continuation_px_after_gate": seam_edge_south_continuation_px_after_gate,
                            "seam_edge_east_continuation_px_after_gate": seam_edge_east_continuation_px_after_gate,
                            "seam_edge_west_continuation_px_after_gate": seam_edge_west_continuation_px_after_gate,
                            "terrain_curriculum_factor": terrain_curriculum_factor,
                            "alpha_temperature": alpha_temperature,
                            "alpha_prior_weight": alpha_prior_weight,
                            "grad_l2_alpha_heads": grad_l2_alpha,
                            "grad_l2_control_residual_blocks": grad_l2_control_residual,
                            "grad_l2_control_mid_block": grad_l2_control_mid,
                            "param_delta_alpha_heads": param_delta_alpha,
                            "param_delta_control_residual_blocks": param_delta_control_residual,
                            "param_delta_control_mid_block": param_delta_control_mid,
                            "bind_active_count": bind_active_count,
                            "precondition_c_true_count": precondition_c_true_count,
                            "candidate_c_count": candidate_c_count,
                            "eligible_c_count": eligible_c_count,
                            "requested_c_count": requested_c_count,
                            "realized_c_count": realized_c_count,
                            "fallback_from_c_count": fallback_from_c_count,
                            "rejected_c_count_total": rejected_c_count_total,
                            "rejected_c_count_precondition": rejected_c_count_precondition,
                            "retry_count_for_c": retry_count_for_c,
                            "delta_c_mean": delta_c_mean,
                            "delta_c_min": delta_c_min,
                            "delta_c_max": delta_c_max,
                            "drop_stage_for_c": drop_stage_for_c,
                            "bind_negative_mode_requested": bind_negative_mode_requested,
                            "bind_negative_mode_realized": bind_negative_mode_realized,
                            "bind_negative_delta_retry_count": bind_negative_delta_retry_count,
                            "bind_negative_fallback_triggered": bind_negative_fallback_triggered,
                            "bind_negative_fallback_reason": bind_negative_fallback_reason,
                            "bind_negative_delta_mean": bind_negative_delta_mean,
                            "bind_active_batch": bind_active_batch,
                            "bind_negative_mode": bind_negative_mode,
                            "bind_schedule_name": bind_schedule_name,
                            "bind_effective_weight_a": bind_effective_weight_a,
                            "bind_effective_weight_b": bind_effective_weight_b,
                            "bind_effective_weight_c": bind_effective_weight_c,
                            "bind_pos_denoise_loss": bind_pos_loss_value,
                            "bind_neg_denoise_loss": bind_neg_loss_value,
                            "bind_ranking_gap": bind_ranking_gap_value,
                            "bind_loss": bind_loss_value,
                            "bind_margin_loss": bind_margin_loss_value,
                            "bind_win_rate": bind_win_rate_value,
                            "train_prompt_name": train_prompt_name,
                            "train_prompt_mode": train_prompt_mode,
                            "train_prompt": train_prompt,
                            "train_prompt2": train_prompt2,
                        }
                    formatted_loss_trace_row = _format_loss_trace_row(loss_trace_row, loss_trace_mode)
                    loss_trace.append(formatted_loss_trace_row)
                    if loss_trace:
                        _append_loss_trace_row_live(loss_trace[-1])
                    if seam_adapter_log_now and controlnet_diagnostics is not None:
                        seam_adapter_diag_rows = _build_seam_adapter_diag_rows(global_step, controlnet_diagnostics)
                        seam_adapter_diag_trace.extend(seam_adapter_diag_rows)
                        for seam_adapter_diag_row in seam_adapter_diag_rows:
                            _append_seam_adapter_diag_row_live(seam_adapter_diag_row)

                if verification_log_now:
                    _log_controlnet_diagnostics(global_step, controlnet_diagnostics)
                    if grad_stats is not None:
                        grad_l2_alpha = float(grad_stats.get("alpha_heads", {}).get("l2", 0.0))
                        grad_l2_control_residual = float(grad_stats.get("control_residual_blocks", {}).get("l2", 0.0))
                        grad_l2_control_mid = float(grad_stats.get("control_mid_block", {}).get("l2", 0.0))
                        for module_name, module_stats in grad_stats.items():
                            logger.info(
                                "[verify/grad] "
                                + f"step={global_step} module={module_name} grad_l2={module_stats['l2']:.8e} "
                                + f"grad_abs_sum={module_stats['abs_sum']:.8e} tensors={module_stats['count']}"
                            )
                            if module_stats["count"] > 0 and module_stats["l2"] <= float(
                                verification_config["gradient_warn_threshold"]
                            ):
                                logger.warning(
                                    f"[verify/grad] near-zero gradient detected for module={module_name} at step={global_step}"
                                )

                    if pre_step_param_stats is not None and post_step_param_stats is not None:
                        deltas = _compute_module_update_deltas(pre_step_param_stats, post_step_param_stats)
                        param_delta_alpha = float(deltas.get("alpha_heads", 0.0))
                        param_delta_control_residual = float(deltas.get("control_residual_blocks", 0.0))
                        param_delta_control_mid = float(deltas.get("control_mid_block", 0.0))
                        for module_name, delta_abs_sum in deltas.items():
                            logger.info(
                                "[verify/param_update] "
                                + f"step={global_step} module={module_name} delta_abs_sum={delta_abs_sum:.8e} "
                                + f"pre_l2={pre_step_param_stats[module_name]['l2']:.8e} "
                                + f"post_l2={post_step_param_stats[module_name]['l2']:.8e}"
                            )
                            if post_step_param_stats[module_name]["count"] > 0 and delta_abs_sum <= float(
                                verification_config["param_delta_warn_threshold"]
                            ):
                                logger.warning(
                                    f"[verify/param_update] near-zero parameter update for module={module_name} at step={global_step}"
                                )

                    if alpha_config["enabled"]:
                        _log_alpha_health(
                            global_step,
                            binary_alpha_target,
                            supervision_mask,
                            scaled_alpha_logits,
                            edge_band,
                        )

                if should_save_checkpoint(args, global_step, resumed_step):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        if not args.skip_step_checkpoint_weights:
                            save_controlnet_checkpoint(
                                accelerator,
                                control_net,
                                args.output_dir,
                                args.output_name,
                                global_step,
                                save_dtype,
                            )
                        if args.save_state:
                            save_extended_training_state(args, accelerator, global_step, ema_state, on_train_end=False)
                        if loss_trace:
                            _ckpt_trace_path = os.path.join(args.output_dir, "sanity", "loss_trace.csv")
                            try:
                                os.makedirs(os.path.join(args.output_dir, "sanity"), exist_ok=True)
                                _fieldnames = list(loss_trace[0].keys())
                                with open(_ckpt_trace_path, "w", newline="", encoding="utf-8") as _fh:
                                    _w = csv.DictWriter(_fh, fieldnames=_fieldnames, extrasaction="ignore")
                                    _w.writeheader()
                                    _w.writerows(loss_trace)
                                logger.info("[sanity/loss] checkpoint trace written to %s (%d rows)", _ckpt_trace_path, len(loss_trace))
                            except Exception as _e:
                                logger.warning("[sanity/loss] failed to write checkpoint trace: %s", _e)
                        if seam_adapter_diag_trace:
                            _ckpt_adapter_diag_path = os.path.join(args.output_dir, "sanity", "seam_adapter_diag.csv")
                            try:
                                os.makedirs(os.path.join(args.output_dir, "sanity"), exist_ok=True)
                                with open(_ckpt_adapter_diag_path, "w", newline="", encoding="utf-8") as _fh:
                                    _w = csv.DictWriter(_fh, fieldnames=list(SEAM_ADAPTER_DIAG_FIELDS), extrasaction="ignore")
                                    _w.writeheader()
                                    _w.writerows(seam_adapter_diag_trace)
                                logger.info(
                                    "[sanity/seam_adapter] checkpoint trace written to %s (%d rows)",
                                    _ckpt_adapter_diag_path,
                                    len(seam_adapter_diag_trace),
                                )
                            except Exception as _e:
                                logger.warning("[sanity/seam_adapter] failed to write checkpoint trace: %s", _e)

                if evaluation_config["enabled"] and global_step in evaluation_config["eval_steps"]:
                    step_label = f"step_{global_step:04d}"
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        os.makedirs(eval_output_dir_raw, exist_ok=True)
                        raw_model = unwrap_model(accelerator, control_net)
                        summary = run_eval_step(
                            step_label=step_label,
                            output_dir=eval_output_dir_raw,
                            run_name=args.output_name,
                            pretrain=False,
                            optimizer_steps_completed=global_step,
                            dataset=dataset,
                            resolved_samples=resolved_eval_samples,
                            unet=unet,
                            control_net=raw_model,
                            vae=vae,
                            cached_text=cached_text,
                            eval_config=evaluation_config,
                            scheduler_config=scheduler_config,
                            device=accelerator.device,
                            weight_dtype=weight_dtype,
                            control_dtype=torch.float32,
                            vae_dtype=vae_dtype,
                        )
                        eval_step_summaries[step_label] = summary
                        progression_steps = []
                        if "step_0000_pretrain" in eval_step_summaries:
                            progression_steps.append("step_0000_pretrain")
                        progression_steps.extend(
                            [f"step_{s:04d}" for s in evaluation_config["eval_steps"] if f"step_{s:04d}" in eval_step_summaries]
                        )
                        build_progression_boards(
                            output_dir=eval_output_dir_raw,
                            run_name=args.output_name,
                            resolved_samples=resolved_eval_samples,
                            step_labels=progression_steps,
                            primary_seed=evaluation_config["seeds"][0],
                        )
                        if binding_config["enabled"] and resolved_swap_pairs:
                            run_semantic_binding_eval(
                                step_label=step_label,
                                output_dir=eval_output_dir_raw,
                                run_name=args.output_name,
                                dataset=dataset,
                                swap_pairs=resolved_swap_pairs,
                                unet=unet,
                                control_net=raw_model,
                                vae=vae,
                                cached_text=cached_text,
                                binding_config=binding_config,
                                scheduler_config=scheduler_config,
                                device=accelerator.device,
                                weight_dtype=weight_dtype,
                                control_dtype=torch.float32,
                                vae_dtype=vae_dtype,
                            )
                        clean_memory_on_device(accelerator.device)
                        if ema_state is not None and args.ema_eval_at_anchors and eval_output_dir_ema is not None:
                            os.makedirs(eval_output_dir_ema, exist_ok=True)
                            backup_state = swap_model_state(raw_model, ema_state)
                            try:
                                run_eval_step(
                                    step_label=step_label,
                                    output_dir=eval_output_dir_ema,
                                    run_name=args.output_name,
                                    pretrain=False,
                                    optimizer_steps_completed=global_step,
                                    dataset=dataset,
                                    resolved_samples=resolved_eval_samples,
                                    unet=unet,
                                    control_net=raw_model,
                                    vae=vae,
                                    cached_text=cached_text,
                                    eval_config=evaluation_config,
                                    scheduler_config=scheduler_config,
                                    device=accelerator.device,
                                    weight_dtype=weight_dtype,
                                    control_dtype=torch.float32,
                                    vae_dtype=vae_dtype,
                                )
                                build_progression_boards(
                                    output_dir=eval_output_dir_ema,
                                    run_name=args.output_name,
                                    resolved_samples=resolved_eval_samples,
                                    step_labels=progression_steps,
                                    primary_seed=evaluation_config["seeds"][0],
                                )
                                if binding_config["enabled"] and resolved_swap_pairs:
                                    run_semantic_binding_eval(
                                        step_label=step_label,
                                        output_dir=eval_output_dir_ema,
                                        run_name=args.output_name,
                                        dataset=dataset,
                                        swap_pairs=resolved_swap_pairs,
                                        unet=unet,
                                        control_net=raw_model,
                                        vae=vae,
                                        cached_text=cached_text,
                                        binding_config=binding_config,
                                        scheduler_config=scheduler_config,
                                        device=accelerator.device,
                                        weight_dtype=weight_dtype,
                                        control_dtype=torch.float32,
                                        vae_dtype=vae_dtype,
                                    )
                            finally:
                                restore_model_state(raw_model, backup_state)
                                clean_memory_on_device(accelerator.device)
                    accelerator.wait_for_everyone()
                    control_net.train()

                if global_step >= args.max_train_steps:
                    break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if loss_trace:
            sanity_dir = os.path.join(args.output_dir, "sanity")
            os.makedirs(sanity_dir, exist_ok=True)
            loss_trace_path = os.path.join(sanity_dir, "loss_trace.csv")
            with open(loss_trace_path, "w", newline="", encoding="utf-8") as handle:
                fieldnames = list(loss_trace[0].keys())
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(loss_trace)
            logger.info(f"[sanity/loss] wrote trace to {loss_trace_path} ({len(loss_trace)} rows)")
        if seam_adapter_diag_trace:
            sanity_dir = os.path.join(args.output_dir, "sanity")
            os.makedirs(sanity_dir, exist_ok=True)
            seam_adapter_diag_path = os.path.join(sanity_dir, "seam_adapter_diag.csv")
            with open(seam_adapter_diag_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(SEAM_ADAPTER_DIAG_FIELDS), extrasaction="ignore")
                writer.writeheader()
                writer.writerows(seam_adapter_diag_trace)
            logger.info(
                f"[sanity/seam_adapter] wrote trace to {seam_adapter_diag_path} ({len(seam_adapter_diag_trace)} rows)"
            )

        save_controlnet_checkpoint(
            accelerator,
            control_net,
            args.output_dir,
            args.output_name,
            global_step,
            save_dtype,
        )
        if args.save_state_on_train_end:
            save_extended_training_state(args, accelerator, global_step, ema_state, on_train_end=True)

        if evaluation_config["enabled"]:
            attempt_summary = summarize_attempt(
                output_dir=eval_output_dir_raw,
                eval_step_summaries=eval_step_summaries,
                loss_trace=loss_trace,
                eval_config=evaluation_config,
            )
            logger.info(
                "[eval/attempt] "
                + f"decision={attempt_summary.get('decision')} failed_thresholds={attempt_summary.get('failed_threshold_keys', [])}"
            )


if __name__ == "__main__":
    train(parse_args())