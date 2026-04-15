import argparse
import csv
import os
import random
from collections import Counter
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from PIL import Image, ImageDraw

from library import sdxl_model_util, sdxl_train_util, strategy_sdxl
from library.device_utils import init_ipex, clean_memory_on_device
from library.sdxl_original_control_net import SdxlControlNet, SdxlControlledUNet
from library.terrain_semantic_eval import (
    build_progression_boards,
    resolve_eval_samples,
    run_eval_step,
    summarize_attempt,
)
from library.terrain_semantic_manifest_dataset import SemanticChannelSpec, TerrainSemanticManifestDataset
from library.utils import setup_logging
import networks.lora as lora_network


init_ipex()
setup_logging()
import logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic_config", type=str, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default="terrain-semantic-controlnet")
    parser.add_argument("--material_lora_weights", type=str, required=True)
    parser.add_argument("--material_lora_multiplier", type=float, default=1.0)
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, nargs=2, default=None)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=8000)
    parser.add_argument("--save_every_n_steps", type=int, default=500)
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
    parser.add_argument("--vae", type=str, default=None)
    parser.add_argument("--lowram", action="store_true")
    parser.add_argument("--disable_mmap_load_safetensors", action="store_true")
    parser.add_argument("--full_fp16", action="store_true")
    parser.add_argument("--full_bf16", action="store_true")
    parser.add_argument("--no_half_vae", action="store_true")
    parser.add_argument("--save_dtype", type=str, default="fp16", choices=["fp16", "bf16", "float"])
    parser.add_argument("--sanity_samples", type=int, default=32)
    parser.add_argument("--debug_dump_samples", type=int, default=3)
    parser.add_argument("--skip_lora_sanity_check", action="store_true")
    parser.add_argument("--lora_sanity_steps", type=int, default=6)
    parser.add_argument(
        "--lora_sanity_prompt",
        type=str,
        default="rock surface, natural stone, neutral lighting",
    )
    parser.add_argument("--lora_sanity_prompt2", type=str, default=None)
    parser.add_argument("--loss_trace_every", type=int, default=5)
    return parser.parse_args()


def unwrap_model(accelerator: Accelerator, model):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


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
    }


def parse_verification_config(config: Dict[str, object]) -> Dict[str, object]:
    verification = config.get("verification", {})
    return {
        "enabled": bool(verification.get("enabled", True)),
        "log_every": int(verification.get("log_every", 25)),
        "gradient_warn_threshold": float(verification.get("gradient_warn_threshold", 1e-10)),
        "param_delta_warn_threshold": float(verification.get("param_delta_warn_threshold", 1e-10)),
        "run_controlnet_sanity": bool(verification.get("run_controlnet_sanity", True)),
        "controlnet_sanity_steps": int(verification.get("controlnet_sanity_steps", 6)),
        "controlnet_min_mse": float(verification.get("controlnet_min_mse", 1e-6)),
        "save_sanity_previews": bool(verification.get("save_sanity_previews", True)),
        "always_log_during_tiny_overfit": bool(verification.get("always_log_during_tiny_overfit", True)),
        "tiny_overfit_max_steps": int(verification.get("tiny_overfit_max_steps", 400)),
    }


def parse_conditioning_config(config: Dict[str, object]) -> Dict[str, object]:
    conditioning = config.get("conditioning", {})
    return {
        "cond_embedding_lr_multiplier": float(conditioning.get("cond_embedding_lr_multiplier", 10.0)),
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
) -> TerrainSemanticManifestDataset:
    training = semantic_config["training"]
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
    }
    optional_tensor_keys = {"latents", "alpha_target"}

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
    batch["crop_size_class"] = [sample["crop_size_class"] for sample in samples]
    batch["generation_strategy"] = [sample["generation_strategy"] for sample in samples]
    batch["prompt"] = samples[0]["prompt"]
    batch["prompt2"] = samples[0]["prompt2"]
    batch["channel_names"] = samples[0]["channel_names"]
    return batch


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
    input_ids1, input_ids2 = tokenize_strategy.tokenize([prompt])
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


def _linear_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 0:
        return end
    progress = min(max(step, 0), total_steps) / float(total_steps)
    return start + (end - start) * progress


def _find_channel_index(channel_names: List[str], target_name: str) -> int:
    if target_name not in channel_names:
        raise ValueError(f"required channel '{target_name}' not found in {channel_names}")
    return channel_names.index(target_name)


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


def _terrain_mask_to_occupancy(mask: torch.Tensor, black_is_terrain: bool) -> torch.Tensor:
    mask = mask.detach().float() if not mask.is_floating_point() else mask.float()
    mask = mask.clamp(0.0, 1.0)
    return (1.0 - mask) if black_is_terrain else mask


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
    residual_str = ",".join([f"{value:.4f}" for value in residuals])
    logger.info(
        "[verify/controlnet] "
        + f"step={global_step} multiplier={float(diagnostics.get('multiplier', 1.0)):.4f} "
        + f"cond_embedding_norm={float(diagnostics.get('cond_embedding_norm', 0.0) or 0.0):.4f} "
        + f"mid_norm={float(diagnostics.get('mid_block_residual_norm', 0.0)):.4f} "
        + f"down_norms=[{residual_str}]"
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
):
    sanity_dir = os.path.join(args.output_dir, "sanity")
    os.makedirs(sanity_dir, exist_ok=True)

    unet_dtype = next(unet.parameters()).dtype
    control_dtype = next(control_net.parameters()).dtype

    sample = dataset[0]
    conditioning = sample["conditioning_images"].unsqueeze(0).to(device=device, dtype=torch.float32)
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

    def predict_x0(control_multiplier: float) -> torch.Tensor:
        material_lora_unet.set_multiplier(args.material_lora_multiplier)
        material_lora_control.set_multiplier(args.material_lora_multiplier)
        control_net.multiplier = control_multiplier
        with torch.no_grad():
            inference_steps = max(2, int(verification_config["controlnet_sanity_steps"]))
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

    pred_on = predict_x0(1.0)
    pred_off = predict_x0(0.0)
    control_net.multiplier = original_multiplier

    mse_diff = F.mse_loss(pred_on.float(), pred_off.float()).item()
    logger.info(f"[verify/controlnet] on_vs_off_pred_x0_mse={mse_diff:.8f}")
    if mse_diff < float(verification_config["controlnet_min_mse"]):
        raise RuntimeError(
            "ControlNet influence sanity check failed: on/off outputs are effectively identical. "
            f"mse={mse_diff:.8f}"
        )

    if verification_config["save_sanity_previews"]:
        with torch.no_grad():
            vae_dtype = next(vae.parameters()).dtype
            on_latents = (pred_on / sdxl_model_util.VAE_SCALE_FACTOR).to(dtype=vae_dtype)
            off_latents = (pred_off / sdxl_model_util.VAE_SCALE_FACTOR).to(dtype=vae_dtype)
            on_image = vae.decode(on_latents).sample[0]
            off_image = vae.decode(off_latents).sample[0]
        _tensor_to_image(on_image).save(os.path.join(sanity_dir, "controlnet_on.png"))
        _tensor_to_image(off_image).save(os.path.join(sanity_dir, "controlnet_off.png"))
        logger.info(f"[verify/controlnet] saved previews: {sanity_dir}")

    vae.to(device=vae_original_device, dtype=vae_original_dtype)



def run_channel_perturbation_sanity_check(
    args: argparse.Namespace,
    dataset: TerrainSemanticManifestDataset,
    unet: torch.nn.Module,
    control_net: torch.nn.Module,
    noise_scheduler: DDPMScheduler,
    cached_text: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
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
    full_conditioning = sample["conditioning_images"].unsqueeze(0).to(device=device, dtype=torch.float32)
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

    channel_names = dataset.channel_names
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
):
    sanity_dir = os.path.join(args.output_dir, "sanity")
    os.makedirs(sanity_dir, exist_ok=True)

    unet_dtype = next(unet.parameters()).dtype
    control_dtype = next(control_net.parameters()).dtype

    sample = dataset[0]
    conditioning = sample["conditioning_images"].unsqueeze(0).to(device=device, dtype=torch.float32)
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
    evaluation_config = parse_evaluation_config(semantic_config, alpha_config, args.output_name, args.max_train_steps)
    if args.seed is None:
        args.seed = random.randint(0, 2**32)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    weight_dtype = resolve_weight_dtype(args)
    save_dtype = resolve_save_dtype(args.save_dtype)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    dataset = build_dataset(args, semantic_config, alpha_config)
    resolved_eval_samples = []
    eval_output_dir = os.path.join(args.output_dir, "eval")
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
    _run_pre_gate_target_terrain_iou_sanity(dataset, alpha_config, resolved_eval_samples)
    run_startup_sanity_report(args, dataset, alpha_config)
    write_debug_alignment_dump(args, dataset)
    sampler = WeightedRandomSampler(dataset.sampling_weights, num_samples=len(dataset), replacement=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=semantic_collate,
        drop_last=True,
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
    control_net = SdxlControlNet(
        conditioning_channels=len(dataset.channel_names),
        conditioning_pre_embed_channels=[32, 32],
        alpha_head_indices=selected_alpha_head_indices,
        alpha_baseline_mode=alpha_config["baseline_mode"] if alpha_config["enabled"] else "none",
        alpha_baseline_terrain_channel_index=terrain_mask_index_for_baseline,
    )
    if args.controlnet_model_name_or_path:
        if os.path.splitext(args.controlnet_model_name_or_path)[1] == ".safetensors":
            state_dict = load_safetensors(args.controlnet_model_name_or_path)
        else:
            state_dict = torch.load(args.controlnet_model_name_or_path, map_location="cpu")
        strict_load = not alpha_config["enabled"]
        info = control_net.load_state_dict(state_dict, strict=strict_load)
        if alpha_config["enabled"]:
            allowed_prefixes = ("controlnet_alpha_heads", "controlnet_alpha_baseline_heads")
            unexpected = [key for key in info.unexpected_keys if not key.startswith(allowed_prefixes)]
            missing = [key for key in info.missing_keys if not key.startswith(allowed_prefixes)]
            if unexpected or missing:
                raise RuntimeError(
                    f"unexpected checkpoint mismatch with alpha-enabled ControlNet: missing={missing[:5]} unexpected={unexpected[:5]}"
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
    for name, parameter in control_net.named_parameters():
        if name.startswith("controlnet_"):
            parameter.requires_grad = True
            trainable_params.append(parameter)
        cond_embedding_params = [
            parameter for name, parameter in control_net.named_parameters()
            if parameter.requires_grad and "controlnet_cond_embedding" in name
        ]
        other_trainable_params = [
            parameter for name, parameter in control_net.named_parameters()
            if parameter.requires_grad and "controlnet_cond_embedding" not in name
        ]

    assert_trainable_policy(
        unet,
        text_encoder1,
        text_encoder2,
        vae,
        control_net,
        material_lora_unet,
        material_lora_control,
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
    optimizer = torch.optim.AdamW(
        [
            {"params": cond_embedding_params, "lr": cond_lr},
            {"params": other_trainable_params, "lr": args.learning_rate},
        ],
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )
    logger.info(
        f"[optimizer] cond_embedding_lr={cond_lr:.2e} (×{conditioning_config['cond_embedding_lr_multiplier']}) "
        f"other_lr={args.learning_rate:.2e} cond_params={len(cond_embedding_params)} other_params={len(other_trainable_params)}"
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

    cached_text = prepare_text_conditioning(
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
            sanity_steps=int(verification_config["controlnet_sanity_steps"]),
        )

    control_net, optimizer, dataloader = accelerator.prepare(control_net, optimizer, dataloader)

    scheduler_config = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "num_train_timesteps": 1000,
        "clip_sample": False,
    }

    if evaluation_config["enabled"] and evaluation_config["include_step0"]:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(eval_output_dir, exist_ok=True)
            step0_summary = run_eval_step(
                step_label="step_0000_pretrain",
                output_dir=eval_output_dir,
                run_name=args.output_name,
                pretrain=True,
                optimizer_steps_completed=0,
                dataset=dataset,
                resolved_samples=resolved_eval_samples,
                unet=unet,
                control_net=unwrap_model(accelerator, control_net),
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
        accelerator.wait_for_everyone()
        control_net.train()

    noise_scheduler = DDPMScheduler(**scheduler_config)

    terrain_mask_index = _find_channel_index(dataset.channel_names, "terrain_mask") if alpha_config["enabled"] else -1
    global_step = 0
    loss_trace: List[Dict[str, float]] = []
    tiny_overfit_run = bool(
        verification_config["always_log_during_tiny_overfit"]
        and args.max_train_steps <= int(verification_config["tiny_overfit_max_steps"])
    )
    progress_bar = tqdm(total=args.max_train_steps, disable=not accelerator.is_local_main_process, desc="steps")
    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(control_net):
                if batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
                else:
                    with torch.no_grad():
                        latents = vae.encode(batch["images"].to(accelerator.device, dtype=vae_dtype)).latent_dist.sample()
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

                encoder_hidden_states1, encoder_hidden_states2, pool2 = cached_text
                text_embedding = torch.cat(
                    [encoder_hidden_states1.expand(batch_size, -1, -1), encoder_hidden_states2.expand(batch_size, -1, -1)],
                    dim=2,
                ).to(dtype=weight_dtype)
                size_embeddings = build_size_embeddings(batch, accelerator.device, weight_dtype)
                vector_embedding = torch.cat([pool2.expand(batch_size, -1), size_embeddings], dim=1).to(dtype=weight_dtype)

                conditioning_images = batch["conditioning_images"].to(accelerator.device, dtype=weight_dtype)
                trusted_mask = batch["trusted_mask"].to(accelerator.device, dtype=weight_dtype).unsqueeze(1)
                alpha_target = batch["alpha_target"]
                if alpha_target is not None:
                    alpha_target = alpha_target.to(accelerator.device, dtype=weight_dtype).unsqueeze(1)

                diffusion_loss_value = 0.0
                alpha_bce_value = 0.0
                alpha_edge_value = 0.0
                alpha_total_value = 0.0
                alpha_terrain_bce_value = 0.0
                alpha_terrain_iou_value = 0.0
                terrain_curriculum_factor = 0.0
                alpha_temperature = 1.0
                alpha_prior_weight = 0.0
                controlnet_diagnostics = None
                grad_l2_alpha = 0.0
                grad_l2_control_residual = 0.0
                grad_l2_control_mid = 0.0
                param_delta_alpha = 0.0
                param_delta_control_residual = 0.0
                param_delta_control_mid = 0.0
                step_for_logging = global_step + 1
                verification_log_now = verification_config["enabled"] and (
                    step_for_logging == 1
                    or step_for_logging % max(1, int(verification_config["log_every"])) == 0
                    or tiny_overfit_run
                )
                pre_step_param_stats = None

                if verification_log_now:
                    _cond_fp32 = batch["conditioning_images"].float()
                    _cond_dtype = conditioning_images
                    _ch_names = batch.get("channel_names") or dataset.channel_names
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

                with accelerator.autocast():
                    if alpha_config["enabled"]:
                        input_resi_add, mid_add, alpha_outputs = control_net(
                            noisy_latents,
                            timesteps,
                            text_embedding,
                            vector_embedding,
                            conditioning_images,
                            return_alpha=True,
                            return_diagnostics=verification_log_now,
                            alpha_target_size=tuple(batch["images"].shape[-2:]),
                        )
                        controlnet_diagnostics = alpha_outputs.get("diagnostics") if alpha_outputs is not None else None
                    else:
                        if verification_log_now:
                            input_resi_add, mid_add, controlnet_diagnostics = control_net(
                                noisy_latents,
                                timesteps,
                                text_embedding,
                                vector_embedding,
                                conditioning_images,
                                return_diagnostics=True,
                            )
                        else:
                            input_resi_add, mid_add = control_net(
                                noisy_latents,
                                timesteps,
                                text_embedding,
                                vector_embedding,
                                conditioning_images,
                            )
                        alpha_outputs = None
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
                    diffusion_loss = (masked_sum / masked_area).mean()
                    loss = diffusion_loss

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
                        terrain_mask_prior = _terrain_mask_to_occupancy(
                            terrain_mask_prior_raw,
                            bool(alpha_config["terrain_mask_black_is_terrain"]),
                        )
                        blended_alpha_target = (
                            (1.0 - alpha_prior_weight) * binary_alpha_target
                            + alpha_prior_weight * terrain_mask_prior
                        )

                        supervision_mask = _expand_mask(trusted_mask, alpha_config["supervision_expand_px"]).clamp(0.0, 1.0)
                        selected_alpha_logits = _select_alpha_logits(alpha_outputs, alpha_config["output_source"])
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
                        alpha_total = alpha_total + terrain_coupling_loss
                        loss = loss + alpha_total

                        diffusion_loss_value = float(diffusion_loss.detach().item())
                        alpha_bce_value = float(alpha_bce.detach().item())
                        alpha_edge_value = float(alpha_edge.detach().item())
                        alpha_terrain_bce_value = float(terrain_bce.detach().item())
                        alpha_terrain_iou_value = float(terrain_iou_loss.detach().item())
                        alpha_total_value = float(alpha_total.detach().item())
                    else:
                        diffusion_loss_value = float(diffusion_loss.detach().item())

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
                    loss_trace.append(
                        {
                            "step": float(global_step),
                            "loss": loss_value,
                            "diffusion_loss": diffusion_loss_value,
                            "alpha_bce_loss": alpha_bce_value,
                            "alpha_edge_loss": alpha_edge_value,
                            "alpha_terrain_bce_loss": alpha_terrain_bce_value,
                            "alpha_terrain_iou_loss": alpha_terrain_iou_value,
                            "alpha_total_loss": alpha_total_value,
                            "terrain_curriculum_factor": terrain_curriculum_factor,
                            "alpha_temperature": alpha_temperature,
                            "alpha_prior_weight": alpha_prior_weight,
                            "grad_l2_alpha_heads": grad_l2_alpha,
                            "grad_l2_control_residual_blocks": grad_l2_control_residual,
                            "grad_l2_control_mid_block": grad_l2_control_mid,
                            "param_delta_alpha_heads": param_delta_alpha,
                            "param_delta_control_residual_blocks": param_delta_control_residual,
                            "param_delta_control_mid_block": param_delta_control_mid,
                        }
                    )

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

                if global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_controlnet_checkpoint(
                            accelerator,
                            control_net,
                            args.output_dir,
                            args.output_name,
                            global_step,
                            save_dtype,
                        )

                if evaluation_config["enabled"] and global_step in evaluation_config["eval_steps"]:
                    step_label = f"step_{global_step:04d}"
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        os.makedirs(eval_output_dir, exist_ok=True)
                        summary = run_eval_step(
                            step_label=step_label,
                            output_dir=eval_output_dir,
                            run_name=args.output_name,
                            pretrain=False,
                            optimizer_steps_completed=global_step,
                            dataset=dataset,
                            resolved_samples=resolved_eval_samples,
                            unet=unet,
                            control_net=unwrap_model(accelerator, control_net),
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
                            output_dir=eval_output_dir,
                            run_name=args.output_name,
                            resolved_samples=resolved_eval_samples,
                            step_labels=progression_steps,
                            primary_seed=evaluation_config["seeds"][0],
                        )
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
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "step",
                        "loss",
                        "diffusion_loss",
                        "alpha_bce_loss",
                        "alpha_edge_loss",
                        "alpha_terrain_bce_loss",
                        "alpha_terrain_iou_loss",
                        "alpha_total_loss",
                        "terrain_curriculum_factor",
                        "alpha_temperature",
                        "alpha_prior_weight",
                        "grad_l2_alpha_heads",
                        "grad_l2_control_residual_blocks",
                        "grad_l2_control_mid_block",
                        "param_delta_alpha_heads",
                        "param_delta_control_residual_blocks",
                        "param_delta_control_mid_block",
                    ]
                )
                for row in loss_trace:
                    writer.writerow(
                        [
                            int(row["step"]),
                            row["loss"],
                            row["diffusion_loss"],
                            row["alpha_bce_loss"],
                            row["alpha_edge_loss"],
                            row["alpha_terrain_bce_loss"],
                            row["alpha_terrain_iou_loss"],
                            row["alpha_total_loss"],
                            row["terrain_curriculum_factor"],
                            row["alpha_temperature"],
                            row["alpha_prior_weight"],
                            row.get("grad_l2_alpha_heads", 0.0),
                            row.get("grad_l2_control_residual_blocks", 0.0),
                            row.get("grad_l2_control_mid_block", 0.0),
                            row.get("param_delta_alpha_heads", 0.0),
                            row.get("param_delta_control_residual_blocks", 0.0),
                            row.get("param_delta_control_mid_block", 0.0),
                        ]
                    )
            logger.info(f"[sanity/loss] wrote trace to {loss_trace_path} ({len(loss_trace)} rows)")

        save_controlnet_checkpoint(
            accelerator,
            control_net,
            args.output_dir,
            args.output_name,
            global_step,
            save_dtype,
        )

        if evaluation_config["enabled"]:
            attempt_summary = summarize_attempt(
                output_dir=eval_output_dir,
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