"""Regional multi-target diffusion loss.

Implements the Phase 1 spec from the regional multi-target plan:
- Per-candidate selection score `score = -(R - mean) / (std + eps) + beta * Q`
    computed over active candidates with optional std-normalization.
- Finite scores are blurred with a depthwise Gaussian (sigma tied to pool kernel),
    then structurally inactive candidates are masked before softmax / tau.
- Gate is detached and used to weight per-candidate eps-MSE losses
  pooled at the same K_lat / S_lat.
- RGB auxiliary loss on hard-argmax-selected winner via pooled-argmax
  -> nearest-upsample -> torch.gather, confidence-weighted by max(gate).

All public functions are torch-friendly (autograd-safe where intended).
This module is pure: no I/O, no logger.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RegionalLossConfig:
    enabled: bool = False
    regional_diff_loss_weight: float = 1.0

    # Pooling on the latent grid.
    kernel_train_px: int = 32   # informational only; training pixel scale
    stride_train_px: int = 12   # informational only
    kernel_lat: int = 8         # actual latent kernel
    stride_lat: int = 2         # actual latent stride

    # Gate.
    tau_start: float = 0.35
    tau_end: float = 0.20
    tau_anneal_steps: int = 200
    gamma: float = 1.0
    beta: float = 0.20
    score_normalize: bool = True
    score_norm_eps: float = 1e-4
    score_clamp_abs: float = 5.0
    gate_blur_sigma_pooled: float = 2.0  # = K_lat / 4 with K_lat=8

    # Weak one-sided Q-regret binding. This is auxiliary to L_region: it only
    # pulls locally high-Q candidates toward competitiveness against detached
    # lower-Q competitors, and never pushes lower-Q candidates worse.
    q_regret_loss_weight: float = 0.03
    q_regret_q_tol: float = 0.05
    q_regret_alpha: float = 0.50
    q_regret_power: float = 1.0
    q_regret_beta_q: float = 0.0
    q_regret_huber_delta: float = 1.0
    q_mix_weight: float = 0.0
    lambda_q_mix_start: float = 0.0
    lambda_q_mix_end: float = 0.0
    q_mix_hold_steps: int = 0
    q_mix_decay_steps: int = 0

    # Q-routing curriculum: route by a conservative sharpened smooth Q, then
    # hand off to the learned gate by reducing rho_q_route.
    gamma_route: float = 3.0
    gamma_boot: float = 3.0
    q_confidence_threshold: float = 0.20
    q_confidence_mask_q_regret: bool = True
    q_confidence_mask_bind: bool = True
    q_confidence_mask_phase_diagnostics: bool = True
    q_route_hold_steps: int = 1000
    q_route_decay_steps: int = 3000
    rho_q_floor: float = 0.05
    q_high_threshold: float = 0.35
    q_low_threshold: float = 0.15
    q_regret_phase3_scale: float = 0.60
    bind_preference_weight: float = 0.015
    bind_phase3_scale: float = 0.60
    hard_band_phase1_scale: float = 0.50
    hard_band_phase3_scale: float = 1.00

    # Terrain-aware active content masking. The active diffusion mask remains
    # responsible for excluding halo/conditioning-only pixels; this mask further
    # removes invisible background from candidate competition.
    terrain_loss_mask_enabled: bool = True
    terrain_alpha_threshold: float = 0.05
    terrain_dilate_px: int = 6
    terrain_blur_sigma: float = 3.0
    terrain_boundary_radius_px: int = 6
    seam_band_weight: float = 1.0
    min_active_content_fraction: float = 0.05
    min_terrain_soft_sum: float = 1.0
    target_eps_sigma_floor: float = 0.05
    robust_diffusion_loss_enabled: bool = True
    safe_pixel_loss_cap: float = 4.0
    regional_loss_sigma_ref: float = 0.20
    regional_loss_sigma_weight_power: float = 4.0
    regional_loss_sigma_weight_min: float = 0.02
    regional_competition_skip_sigma_threshold: float = 0.0
    regional_competition_skip_snr_threshold: float = 0.0
    candidate_competition_sigma_ref: float = 0.20
    candidate_competition_power: float = 2.0
    candidate_competition_min_weight: float = 0.10

    # Regional handoff curriculum. The training loop uses the base ownership
    # ramp to blend standard diffusion into the regional objective; this module
    # uses the competitive surplus ramp to neutralize competitive scaling until
    # candidate advantages have had time to become meaningful.
    regional_base_ownership_initial_weight: float = 1.0
    regional_base_ownership_target_weight: float = 1.0
    regional_base_ownership_ramp_steps: int = 0
    regional_competitive_surplus_initial_weight: float = 1.0
    regional_competitive_surplus_target_weight: float = 1.0
    regional_competitive_surplus_ramp_steps: int = 0
    regional_routing_protected_warmup_enabled: bool = False
    regional_routing_protected_warmup_steps: int = 0
    regional_freeze_gate_routing_head_during_warmup: bool = False
    regional_detach_routing_from_regional_loss_during_warmup: bool = False
    teacher_preservation_enabled: bool = False
    teacher_preservation_ramp_down_steps: int = 0
    q_blend_diffusion_enabled: bool = False
    q_blend_diffusion_weight: float = 0.0
    individual_regional_weight_initial: float = 1.0
    individual_regional_weight_target: float = 1.0
    individual_regional_ramp_steps: int = 0
    competitive_regret_weight: float = 0.0
    regional_use_fixed_q_route_for_ownership: bool = False
    regional_freeze_gate_routing_head: bool = False
    regional_detach_routing_from_regional_loss: bool = False
    seam_edge_support_min_fraction: float = 0.03
    seam_halo_support_min_fraction: float = 0.05
    seam_validity_use_expanded_region_terrain_mask: bool = True
    seam_validity_max_directional_seams: int = 2
    seam_validity_over_max_policy: str = "fail_fast"

    # Soft in-region competitive weighting. This never removes a candidate's
    # assigned regional loss entirely; it only scales the surplus above a base
    # floor using EMA-smoothed in-region advantage.
    competitive_gate_enabled: bool = True
    competitive_gate_beta: float = 0.90
    competitive_gate_tolerance: float = 0.005
    competitive_gate_scale: float = 0.03
    competitive_gate_max_delta_per_step: float = 0.05
    competitive_gate_min_region_support: float = 1.0
    competitive_gate_base_weight_floor: float = 0.70
    competitive_gate_competitor_mode: str = "mean"
    competitive_catchup_enabled: bool = True
    competitive_catchup_weight: float = 0.02
    competitive_catchup_margin: float = 0.0

    # Deprecated: q-floor hard pruning is intentionally inactive for regional runs.
    q_floor: float = 0.0

    # RGB aux.
    rgb_aux_loss_weight: float = 0.01
    rgb_aux_confidence_weighted: bool = True

    # Substrate selection (consumed by dataset, kept here for parity).
    regenerated_image_run_tag: str = ""

    @classmethod
    def from_dict(cls, src: Optional[Dict[str, object]]) -> "RegionalLossConfig":
        d = dict(src) if src else {}
        out = cls()
        for k, v in d.items():
            if hasattr(out, k):
                setattr(out, k, type(getattr(out, k))(v) if getattr(out, k) is not None and v is not None else v)
        # If kernel_lat / stride_lat not explicitly provided, derive from train px.
        if "kernel_lat" not in d and "kernel_train_px" in d:
            out.kernel_lat = max(1, int(round(float(d["kernel_train_px"]) / 8.0)))
        if "stride_lat" not in d and "stride_train_px" in d:
            out.stride_lat = max(1, int(round(float(d["stride_train_px"]) / 8.0)))
        # Gate blur sigma defaults to K/4 for selector-stability diagnostics.
        if "gate_blur_sigma_pooled" not in d:
            out.gate_blur_sigma_pooled = float(out.kernel_lat) / 4.0
        if "q_regret_beta_q" not in d:
            out.q_regret_beta_q = float(out.beta)
        if "gamma_route" not in d and "gamma_boot" in d:
            out.gamma_route = float(out.gamma_boot)
        elif "gamma_boot" not in d:
            out.gamma_boot = float(out.gamma_route)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def current_tau(cfg: RegionalLossConfig, step: int) -> float:
    if cfg.tau_anneal_steps <= 0:
        return cfg.tau_end
    frac = max(0.0, min(1.0, float(step) / float(cfg.tau_anneal_steps)))
    return cfg.tau_start + (cfg.tau_end - cfg.tau_start) * frac


def current_lambda_q_mix(cfg: RegionalLossConfig, step: int) -> float:
    start = max(0.0, float(cfg.lambda_q_mix_start))
    end = max(0.0, float(cfg.lambda_q_mix_end))
    hold_steps = max(0, int(cfg.q_mix_hold_steps))
    decay_steps = max(0, int(cfg.q_mix_decay_steps))
    t = max(0, int(step))
    if t < hold_steps:
        return start
    if decay_steps <= 0:
        return end
    u = max(0.0, min(1.0, float(t - hold_steps) / float(decay_steps)))
    return end + (0.5 * (start - end) * (1.0 + math.cos(math.pi * u)))


def _smoothstep(u: float) -> float:
    u = max(0.0, min(1.0, float(u)))
    return u * u * (3.0 - 2.0 * u)


def _linear_ramp(initial: float, target: float, ramp_steps: int, step: int) -> float:
    steps = max(0, int(ramp_steps))
    if steps <= 0:
        return float(target)
    u = max(0.0, min(1.0, float(max(0, int(step))) / float(steps)))
    return float(initial) + ((float(target) - float(initial)) * u)


def current_q_route_state(cfg: RegionalLossConfig, step: int) -> Tuple[float, float, float]:
    hold_steps = max(0, int(cfg.q_route_hold_steps))
    decay_steps = max(0, int(cfg.q_route_decay_steps))
    rho_floor = max(0.0, min(1.0, float(cfg.rho_q_floor)))
    t = max(0, int(step))
    if t < hold_steps:
        return 1.0, 0.0, 0.0
    if decay_steps <= 0:
        return rho_floor, 1.0, 1.0
    u = max(0.0, min(1.0, float(t - hold_steps) / float(decay_steps)))
    rho = rho_floor + 0.5 * (1.0 - rho_floor) * (1.0 + math.cos(math.pi * u))
    return rho, _smoothstep(u), u


def current_aux_curriculum_scales(cfg: RegionalLossConfig, step: int) -> Dict[str, float]:
    hold_steps = max(0, int(cfg.q_route_hold_steps))
    decay_steps = max(0, int(cfg.q_route_decay_steps))
    t = max(0, int(step))
    rho_q_route, aux_ramp, q_route_u = current_q_route_state(cfg, t)
    if t < hold_steps:
        aux_scale = 0.0
        hard_band_scale = float(cfg.hard_band_phase1_scale)
    elif decay_steps > 0 and t < (hold_steps + decay_steps):
        aux_scale = float(aux_ramp)
        hard_band_scale = float(cfg.hard_band_phase1_scale) + (float(cfg.hard_band_phase3_scale) - float(cfg.hard_band_phase1_scale)) * float(aux_ramp)
    else:
        aux_scale = 1.0
        hard_band_scale = float(cfg.hard_band_phase3_scale)
    return {
        "rho_q_route": float(rho_q_route),
        "aux_ramp": float(aux_ramp),
        "q_route_u": float(q_route_u),
        "q_regret_scale": 0.0 if aux_scale <= 0.0 else (float(cfg.q_regret_phase3_scale) if aux_scale >= 1.0 else float(aux_scale)),
        "bind_scale": 0.0 if aux_scale <= 0.0 else (float(cfg.bind_phase3_scale) if aux_scale >= 1.0 else float(aux_scale)),
        "hard_band_scale": float(hard_band_scale),
    }


def _gaussian_kernel_1d(sigma: float, device, dtype) -> torch.Tensor:
    if sigma <= 0:
        return torch.ones((1,), device=device, dtype=dtype)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum().clamp_min(1e-12)
    return k


def _depthwise_gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """x: [B, T, H, W]. Blur each channel independently."""
    if sigma <= 0:
        return x
    B, T, H, W = x.shape
    k1d = _gaussian_kernel_1d(sigma, x.device, x.dtype)
    pad = (k1d.numel() - 1) // 2
    kx = k1d.view(1, 1, 1, -1).repeat(T, 1, 1, 1)
    ky = k1d.view(1, 1, -1, 1).repeat(T, 1, 1, 1)
    x = F.conv2d(x, kx, padding=(0, pad), groups=T)
    x = F.conv2d(x, ky, padding=(pad, 0), groups=T)
    return x


def _avg_pool(x: torch.Tensor, k: int, s: int) -> torch.Tensor:
    """Average-pool with reflective behaviour via count_include_pad=False."""
    if k <= 1 and s <= 1:
        return x
    pad = k // 2
    return F.avg_pool2d(x, kernel_size=k, stride=s, padding=pad, count_include_pad=False)


def _coerce_b1hw(mask: torch.Tensor, *, name: str) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() == 5 and mask.shape[2] == 1:
        mask = mask.squeeze(2)
    if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError(f"{name} must be [B,1,H,W], got {tuple(mask.shape)}")
    return mask


def _dilate_mask(mask: torch.Tensor, radius_px: int) -> torch.Tensor:
    radius = max(0, int(radius_px))
    if radius <= 0:
        return mask
    kernel = (2 * radius) + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)


def _erode_mask(mask: torch.Tensor, radius_px: int) -> torch.Tensor:
    radius = max(0, int(radius_px))
    if radius <= 0:
        return mask
    return 1.0 - _dilate_mask(1.0 - mask, radius)


def build_terrain_soft_masks(
    terrain_alpha: torch.Tensor,
    *,
    threshold: float = 0.05,
    dilate_px: int = 6,
    blur_sigma: float = 3.0,
    boundary_radius_px: int = 6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build detached soft terrain and alpha-boundary masks in source/image space."""
    alpha = _coerce_b1hw(terrain_alpha.detach().float(), name="terrain_alpha").clamp(0.0, 1.0)
    terrain_core = (alpha > float(threshold)).to(dtype=alpha.dtype)
    terrain_dilated = _dilate_mask(terrain_core, int(dilate_px))
    terrain_soft = _depthwise_gaussian_blur(terrain_dilated, float(blur_sigma)).clamp(0.0, 1.0)

    boundary_radius = max(0, int(boundary_radius_px))
    if boundary_radius > 0:
        boundary = (_dilate_mask(terrain_core, boundary_radius) - _erode_mask(terrain_core, boundary_radius)).clamp(0.0, 1.0)
        boundary = _depthwise_gaussian_blur(boundary, max(0.0, float(blur_sigma) * 0.5)).clamp(0.0, 1.0)
    else:
        grad_y = F.pad((terrain_soft[:, :, 1:, :] - terrain_soft[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        grad_x = F.pad((terrain_soft[:, :, :, 1:] - terrain_soft[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        boundary = (grad_x + grad_y).clamp(0.0, 1.0)
    return terrain_soft.detach(), boundary.detach()


def _masked_avg_pool_spatial(x: torch.Tensor, mask: torch.Tensor, k: int, s: int, eps: float) -> torch.Tensor:
    mask = mask.to(device=x.device, dtype=x.dtype)
    pooled_mask = _avg_pool(mask, k, s)
    pooled_x = _avg_pool(x * mask, k, s) / pooled_mask.clamp_min(eps)
    fallback_x = _avg_pool(x, k, s)
    return torch.where(pooled_mask > eps, pooled_x, fallback_x)


def masked_spatial_mean(
    loss_map: torch.Tensor,
    mask: torch.Tensor,
    *,
    candidate_dim: Optional[int] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reduce a masked loss map without accidentally averaging candidate slots.

    The last two dimensions are spatial. If ``candidate_dim`` is provided, that
    axis is preserved and only non-candidate feature axes (for example latent
    channels) expand the denominator.
    """
    if loss_map.dim() < 3:
        raise ValueError(f"loss_map must have at least [B,H,W], got {tuple(loss_map.shape)}")
    if mask.dim() == 3:
        mask_b1hw = mask.unsqueeze(1)
    else:
        mask_b1hw = _coerce_b1hw(mask, name="mask")
    if loss_map.shape[0] != mask_b1hw.shape[0] or tuple(loss_map.shape[-2:]) != tuple(mask_b1hw.shape[-2:]):
        raise ValueError(
            "loss_map and mask batch/spatial dimensions must match: "
            + f"loss_map={tuple(loss_map.shape)} mask={tuple(mask_b1hw.shape)}"
        )

    candidate_axis: Optional[int]
    if candidate_dim is None:
        candidate_axis = None
    else:
        candidate_axis = int(candidate_dim)
        if candidate_axis < 0:
            candidate_axis += loss_map.dim()
        if candidate_axis <= 0 or candidate_axis >= loss_map.dim() - 2:
            raise ValueError(f"candidate_dim must be a non-spatial, non-batch axis, got {candidate_dim}")

    view_shape = [loss_map.shape[0]] + [1] * (loss_map.dim() - 1)
    view_shape[-2:] = list(mask_b1hw.shape[-2:])
    mask_view = mask_b1hw.to(device=loss_map.device, dtype=loss_map.dtype).view(*view_shape)
    numerator = loss_map * mask_view

    reduce_dims = []
    feature_factor = 1
    for dim, size in enumerate(loss_map.shape):
        if dim == 0 or dim == candidate_axis:
            continue
        reduce_dims.append(dim)
        if dim < loss_map.dim() - 2:
            feature_factor *= int(size)

    reduced = numerator.sum(dim=tuple(reduce_dims))
    spatial_den = mask_b1hw.to(device=loss_map.device, dtype=loss_map.dtype).sum(dim=(1, 2, 3)).clamp_min(float(eps))
    denominator = spatial_den * float(max(1, feature_factor))
    if candidate_axis is not None:
        denominator = denominator.view(loss_map.shape[0], *([1] * (reduced.dim() - 1)))
    return reduced / denominator


def _safe_log_capped_loss(raw_loss: torch.Tensor, loss_cap: float) -> torch.Tensor:
    cap = max(1e-6, float(loss_cap))
    safe = cap * torch.log1p(raw_loss.clamp_min(0.0) / cap)
    return torch.nan_to_num(safe, nan=cap, posinf=cap, neginf=0.0)


def _per_sample_sum(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).sum(dim=1)


def _per_sample_max(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).amax(dim=1)


def _per_sample_mean(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).mean(dim=1)


# ---------------------------------------------------------------------------
# VAE-encode candidate stack
# ---------------------------------------------------------------------------


def encode_candidate_targets(
    vae,
    candidate_targets_rgb: torch.Tensor,   # [B, T, 3, H, W] in [-1, 1]
    candidate_active_mask: torch.Tensor,   # [B, T] in {0,1}
    vae_scale_factor: float,
    encode_groups: int = 1,
    enable_vae_tiling: bool = False,
    tile_sample_min_size: int = 512,
    tile_overlap_factor: float = 0.0,
) -> torch.Tensor:
    """Returns latents [B, T, 4, H/8, W/8] with inactive slots zeroed.

    ``encode_groups`` is the maximum number of candidate images encoded in one
    VAE call. The regional smoke runs at SDXL resolution with up to 5 candidates,
    so the safe default is 1 candidate per encode call.
    """
    if candidate_targets_rgb.dim() != 5:
        raise ValueError(f"expected [B,T,3,H,W], got {tuple(candidate_targets_rgb.shape)}")
    B, T, C, H, W = candidate_targets_rgb.shape
    flat = candidate_targets_rgb.reshape(B * T, C, H, W)
    max_per_encode = max(1, int(encode_groups))
    out_chunks = []
    restore_state = {
        "use_tiling": getattr(vae, "use_tiling", None),
        "use_slicing": getattr(vae, "use_slicing", None),
        "tile_sample_min_size": getattr(vae, "tile_sample_min_size", None),
        "tile_latent_min_size": getattr(vae, "tile_latent_min_size", None),
        "tile_overlap_factor": getattr(vae, "tile_overlap_factor", None),
    }
    try:
        if enable_vae_tiling:
            if hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
            if hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            if hasattr(vae, "tile_sample_min_size"):
                vae.tile_sample_min_size = max(64, int(tile_sample_min_size))
            if hasattr(vae, "tile_latent_min_size"):
                vae.tile_latent_min_size = max(8, int(getattr(vae, "tile_sample_min_size", max(64, int(tile_sample_min_size)))) // 8)
            if hasattr(vae, "tile_overlap_factor"):
                vae.tile_overlap_factor = float(max(0.0, min(0.5, tile_overlap_factor)))
            if flat.is_cuda:
                torch.cuda.empty_cache()
        with torch.no_grad():
            if flat.shape[0] <= max_per_encode:
                lat = vae.encode(flat).latent_dist.sample() * vae_scale_factor
            else:
                for start in range(0, flat.shape[0], max_per_encode):
                    sub = flat[start : start + max_per_encode]
                    out_chunks.append(vae.encode(sub).latent_dist.sample() * vae_scale_factor)
                lat = torch.cat(out_chunks, dim=0)
    finally:
        if restore_state["use_slicing"] is not None:
            if restore_state["use_slicing"] and hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
            elif not restore_state["use_slicing"] and hasattr(vae, "disable_slicing"):
                vae.disable_slicing()
        if restore_state["use_tiling"] is not None:
            if restore_state["use_tiling"] and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            elif not restore_state["use_tiling"] and hasattr(vae, "disable_tiling"):
                vae.disable_tiling()
        for attr_name in ("tile_sample_min_size", "tile_latent_min_size", "tile_overlap_factor"):
            attr_value = restore_state[attr_name]
            if attr_value is not None and hasattr(vae, attr_name):
                setattr(vae, attr_name, attr_value)
        if flat.is_cuda:
            torch.cuda.empty_cache()
    Hl, Wl = lat.shape[-2:]
    lat = lat.view(B, T, lat.shape[1], Hl, Wl)
    mask = candidate_active_mask.view(B, T, 1, 1, 1).to(dtype=lat.dtype, device=lat.device)
    lat = lat * mask
    return lat


# ---------------------------------------------------------------------------
# Core regional loss
# ---------------------------------------------------------------------------


def compute_regional_loss(
    pred_x0_latents: torch.Tensor,           # [B, 4, Hl, Wl]
    targets_x0_latents: torch.Tensor,        # [B, T, 4, Hl, Wl]
    candidate_active_mask: torch.Tensor,     # [B, T] in {0,1}
    candidate_q_field_latent: torch.Tensor,  # [B, T, 1, Hl, Wl]
    noise: torch.Tensor,                      # [B, 4, Hl, Wl]
    noise_pred: torch.Tensor,                 # [B, 4, Hl, Wl]
    noisy_latents: torch.Tensor,              # [B, 4, Hl, Wl]
    sqrt_alpha_t: torch.Tensor,               # [B, 1, 1, 1]
    sqrt_one_minus_alpha_t: torch.Tensor,     # [B, 1, 1, 1]
    trusted_mask_latent: torch.Tensor,        # [B, 1, Hl, Wl]
    cfg: RegionalLossConfig,
    current_step: int,
    terrain_soft_latent: Optional[torch.Tensor] = None,
    seam_boundary_mask_latent: Optional[torch.Tensor] = None,
    alpha_boundary_mask_latent: Optional[torch.Tensor] = None,
    candidate_valid: Optional[torch.Tensor] = None,
    candidate_valid_diagnostics: Optional[Dict[str, torch.Tensor]] = None,
    q_mix_step: Optional[int] = None,
    rho_q_route_override: Optional[float] = None,
    lambda_q_regret_override: Optional[float] = None,
    lambda_bind_override: Optional[float] = None,
    hard_band_scale_override: Optional[float] = None,
    phase_override: Optional[float] = None,
    advantage_ema_prev: Optional[torch.Tensor] = None,
    competitive_gate_prev: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Returns dict with keys:
        loss, gate, gate_pooled, winner_idx_pooled, winner_idx_latent,
        conf_pooled, conf_latent, R_select_pooled, score_pooled, tau,
        diagnostics: gate_entropy_mean, gate_entropy_normalized,
                     gate_score_gap_mean, gate_smoothness, q_winner_alignment,
                     winner_share[T]
    """
    B, T, Cl, Hl, Wl = targets_x0_latents.shape
    assert pred_x0_latents.shape == (B, Cl, Hl, Wl)
    K = max(1, int(cfg.kernel_lat))
    S = max(1, int(cfg.stride_lat))

    structural_active_mask = candidate_active_mask.to(dtype=pred_x0_latents.dtype, device=pred_x0_latents.device).clamp(0.0, 1.0)  # [B, T]
    if candidate_valid is None:
        candidate_valid_mask = torch.ones_like(structural_active_mask)
    else:
        candidate_valid_mask = candidate_valid.detach().to(dtype=pred_x0_latents.dtype, device=pred_x0_latents.device).clamp(0.0, 1.0)
        if candidate_valid_mask.shape != (B, T):
            raise ValueError(f"candidate_valid shape {tuple(candidate_valid_mask.shape)} does not match {(B, T)}")
    candidate_valid_mask[:, 0] = 1.0
    assert not candidate_valid_mask.requires_grad, "candidate_valid must be detached"
    active_mask = (structural_active_mask * candidate_valid_mask).clamp(0.0, 1.0)  # [B, T]
    active_mask[:, 0] = structural_active_mask[:, 0]
    invalid_candidate_mask = (structural_active_mask * (1.0 - candidate_valid_mask)).clamp(0.0, 1.0)
    active_diffusion_mask_latent = _coerce_b1hw(
        trusted_mask_latent.detach().to(device=pred_x0_latents.device, dtype=pred_x0_latents.dtype),
        name="trusted_mask_latent",
    ).clamp(0.0, 1.0)

    if bool(cfg.terrain_loss_mask_enabled) and terrain_soft_latent is not None:
        terrain_soft = _coerce_b1hw(
            terrain_soft_latent.detach().to(device=pred_x0_latents.device, dtype=pred_x0_latents.dtype),
            name="terrain_soft_latent",
        ).clamp(0.0, 1.0)
        if terrain_soft.shape[-2:] != (Hl, Wl):
            terrain_soft = F.interpolate(terrain_soft, size=(Hl, Wl), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
    else:
        terrain_soft = torch.ones_like(active_diffusion_mask_latent)

    terrain_sum_per_sample = _per_sample_sum(terrain_soft)
    terrain_empty_per_sample = terrain_sum_per_sample < max(0.0, float(cfg.min_terrain_soft_sum))
    active_content_mask_latent = active_diffusion_mask_latent * terrain_soft
    if seam_boundary_mask_latent is not None and float(cfg.seam_band_weight) > 0.0:
        seam_boundary = _coerce_b1hw(
            seam_boundary_mask_latent.detach().to(device=pred_x0_latents.device, dtype=pred_x0_latents.dtype),
            name="seam_boundary_mask_latent",
        ).clamp(0.0, 1.0)
        if seam_boundary.shape[-2:] != (Hl, Wl):
            seam_boundary = F.interpolate(seam_boundary, size=(Hl, Wl), mode="area").clamp(0.0, 1.0)
        seam_boundary_content_mask_latent = (active_diffusion_mask_latent * seam_boundary * float(cfg.seam_band_weight)).clamp(0.0, 1.0)
    else:
        seam_boundary_content_mask_latent = torch.zeros_like(active_diffusion_mask_latent)

    if alpha_boundary_mask_latent is not None:
        alpha_boundary = _coerce_b1hw(
            alpha_boundary_mask_latent.detach().to(device=pred_x0_latents.device, dtype=pred_x0_latents.dtype),
            name="alpha_boundary_mask_latent",
        ).clamp(0.0, 1.0)
        if alpha_boundary.shape[-2:] != (Hl, Wl):
            alpha_boundary = F.interpolate(alpha_boundary, size=(Hl, Wl), mode="area").clamp(0.0, 1.0)
    else:
        alpha_boundary = torch.zeros_like(active_diffusion_mask_latent)
    hard_band_content_mask_latent = active_diffusion_mask_latent * torch.maximum(terrain_soft, alpha_boundary)

    active_diffusion_sum_per_sample = _per_sample_sum(active_diffusion_mask_latent).clamp_min(1e-6)
    active_content_sum_pre_fallback = _per_sample_sum(active_content_mask_latent)
    active_content_fraction_per_sample_raw = active_content_sum_pre_fallback / active_diffusion_sum_per_sample
    active_content_fallback_per_sample = (active_content_fraction_per_sample_raw < float(cfg.min_active_content_fraction)) & (~terrain_empty_per_sample)
    fallback_view = active_content_fallback_per_sample.view(B, 1, 1, 1)
    terrain_empty_view = terrain_empty_per_sample.view(B, 1, 1, 1)
    active_content_mask_latent = torch.where(fallback_view, active_diffusion_mask_latent, active_content_mask_latent)
    active_content_mask_latent = torch.where(terrain_empty_view, torch.zeros_like(active_content_mask_latent), active_content_mask_latent)
    hard_band_content_mask_latent = torch.where(fallback_view, active_diffusion_mask_latent, hard_band_content_mask_latent)
    active_content_sum_per_sample = _per_sample_sum(active_content_mask_latent)
    active_content_fraction_per_sample = active_content_sum_per_sample / active_diffusion_sum_per_sample
    active_content_fraction = active_content_fraction_per_sample.mean()
    assert not active_diffusion_mask_latent.requires_grad, "active diffusion mask must be detached"
    assert not terrain_soft.requires_grad, "terrain soft mask must be detached"
    assert not active_content_mask_latent.requires_grad, "active content mask must be detached"
    assert not hard_band_content_mask_latent.requires_grad, "hard-band content mask must be detached"
    assert active_content_mask_latent.shape == (B, 1, Hl, Wl), (
        f"active content mask shape {tuple(active_content_mask_latent.shape)} does not match {(B, 1, Hl, Wl)}"
    )
    assert hard_band_content_mask_latent.shape == (B, 1, Hl, Wl), (
        f"hard-band content mask shape {tuple(hard_band_content_mask_latent.shape)} does not match {(B, 1, Hl, Wl)}"
    )
    eps = 1e-6
    sigma_competition = sqrt_one_minus_alpha_t.view(B, 1, 1, 1).to(
        device=pred_x0_latents.device,
        dtype=pred_x0_latents.dtype,
    ).clamp_min(eps)
    sigma_ref = max(eps, float(cfg.candidate_competition_sigma_ref))
    competition_power = max(0.0, float(cfg.candidate_competition_power))
    competition_min_weight = max(0.0, min(1.0, float(cfg.candidate_competition_min_weight)))
    candidate_competition_weight_t = (sigma_competition / sigma_ref).pow(competition_power).clamp(competition_min_weight, 1.0)
    regional_sigma_ref = max(eps, float(cfg.regional_loss_sigma_ref))
    regional_sigma_power = max(0.0, float(cfg.regional_loss_sigma_weight_power))
    regional_sigma_min = max(0.0, min(1.0, float(cfg.regional_loss_sigma_weight_min)))
    snr_competition = sqrt_alpha_t.view(B, 1, 1, 1).pow(2) / sigma_competition.square().clamp_min(eps)
    competition_skip_sigma_threshold = max(0.0, float(cfg.regional_competition_skip_sigma_threshold))
    competition_skip_snr_threshold = max(0.0, float(cfg.regional_competition_skip_snr_threshold))
    competition_skipped_low_sigma = torch.zeros_like(sigma_competition, dtype=torch.bool)
    if competition_skip_sigma_threshold > 0.0:
        competition_skipped_low_sigma = competition_skipped_low_sigma | (sigma_competition <= competition_skip_sigma_threshold)
    if competition_skip_snr_threshold > 0.0:
        competition_skipped_low_sigma = competition_skipped_low_sigma | (snr_competition >= competition_skip_snr_threshold)
    if regional_sigma_power > 0.0:
        regional_loss_sigma_weight_t = (sigma_competition / regional_sigma_ref).pow(regional_sigma_power).clamp(regional_sigma_min, 1.0)
    else:
        regional_loss_sigma_weight_t = torch.ones_like(sigma_competition)
    candidate_competition_weight_t = torch.where(
        competition_skipped_low_sigma,
        torch.zeros_like(candidate_competition_weight_t),
        candidate_competition_weight_t,
    )

    # -------------------------------------------------------------------
    # Per-candidate selection map (L1 over latent C), pooled.
    # -------------------------------------------------------------------
    diff_full = pred_x0_latents.unsqueeze(1) - targets_x0_latents
    diff = diff_full.abs()  # [B, T, 4, Hl, Wl]
    R_pixel = diff.mean(dim=2)                                         # [B, T, Hl, Wl]
    # Detach R for selection (gate is detached anyway, but no need to
    # backprop through this path).
    max_r_for_selection = math.sqrt(max(1e-6, float(cfg.safe_pixel_loss_cap)))
    R_pixel_det = torch.nan_to_num(R_pixel.detach(), nan=max_r_for_selection, posinf=max_r_for_selection, neginf=0.0).clamp_max(max_r_for_selection)
    R_pooled = _masked_avg_pool_spatial(R_pixel_det, active_content_mask_latent, K, S, 1e-6)  # [B, T, Hl', Wl']

    active_candidate_hw = active_mask.view(B, T, 1, 1)
    inactive_high = torch.full_like(R_pixel_det, max_r_for_selection)
    R_pixel_active_candidates = torch.where(active_candidate_hw > 0.0, R_pixel_det, inactive_high)
    comp_count_pixel = active_candidate_hw.sum(dim=1, keepdim=True).sub(active_candidate_hw).clamp_min(1.0)
    R_comp_pixel = ((R_pixel_det * active_candidate_hw).sum(dim=1, keepdim=True) - (R_pixel_det * active_candidate_hw)) / comp_count_pixel
    R_advantage_pixel = (R_comp_pixel - R_pixel_det) * active_candidate_hw

    def _candidate_pixel_contrast(mask: torch.Tensor, prefix: str) -> Dict[str, torch.Tensor]:
        mask = mask.to(device=R_pixel_det.device, dtype=R_pixel_det.dtype).clamp(0.0, 1.0)
        mask_sum = mask.sum().clamp_min(eps)
        candidate_count = active_candidate_hw.sum(dim=1, keepdim=True).clamp_min(1.0)
        r_mean_pixel = (R_pixel_det * active_candidate_hw).sum(dim=1, keepdim=True) / candidate_count
        r_var_pixel = ((R_pixel_det - r_mean_pixel).square() * active_candidate_hw).sum(dim=1, keepdim=True) / candidate_count
        r_std_pixel = r_var_pixel.clamp_min(0.0).sqrt()
        contrast: Dict[str, torch.Tensor] = {
            f"{prefix}_candidate_std_mean": ((r_std_pixel * mask).sum() / mask_sum).detach(),
            f"{prefix}_candidate_std_max": r_std_pixel.detach().amax(),
        }
        if T > 1:
            pair_active = (active_mask[:, :, None] * active_mask[:, None, :]).to(dtype=R_pixel_det.dtype)
            pair_active = pair_active * (~torch.eye(T, dtype=torch.bool, device=R_pixel_det.device)).view(1, T, T).to(dtype=R_pixel_det.dtype)
            pair_mask = pair_active.view(B, T, T, 1, 1) * mask.unsqueeze(1)
            pair_den = pair_mask.sum().clamp_min(eps)
            pair_abs = (R_pixel_det[:, :, None] - R_pixel_det[:, None, :]).abs()
            contrast[f"{prefix}_pairwise_absdiff_mean"] = ((pair_abs * pair_mask).sum() / pair_den).detach()
            contrast[f"{prefix}_pairwise_absdiff_max"] = torch.where(pair_mask > 0.0, pair_abs, torch.zeros_like(pair_abs)).detach().amax()
        else:
            contrast[f"{prefix}_pairwise_absdiff_mean"] = R_pixel_det.new_tensor(0.0).detach()
            contrast[f"{prefix}_pairwise_absdiff_max"] = R_pixel_det.new_tensor(0.0).detach()
        lowest_idx = R_pixel_active_candidates.argmin(dim=1, keepdim=True)
        q_lat_for_corr = q_diagnostics_lat
        if q_lat_for_corr.shape[-2:] != (Hl, Wl):
            q_lat_for_corr = F.interpolate(q_lat_for_corr.reshape(B * T, 1, *q_lat_for_corr.shape[-2:]), size=(Hl, Wl), mode="area").view(B, T, Hl, Wl)
        corr_values = []
        for ti in range(T):
            slot_active = active_mask[:, ti].view(B, 1, 1, 1).to(dtype=R_pixel_det.dtype)
            slot_mask = mask * slot_active
            slot_den = slot_mask.sum().clamp_min(eps)
            contrast[f"{prefix}_lowest_frac_slot{ti}"] = (((lowest_idx == ti).to(dtype=R_pixel_det.dtype) * slot_mask).sum() / slot_den).detach()
            x = q_lat_for_corr[:, ti : ti + 1].to(dtype=R_pixel_det.dtype)
            y = -R_pixel_det[:, ti : ti + 1]
            x_mean = (x * slot_mask).sum() / slot_den
            y_mean = (y * slot_mask).sum() / slot_den
            x_centered = (x - x_mean) * slot_mask
            y_centered = (y - y_mean) * slot_mask
            corr_den = (x_centered.square().sum() * y_centered.square().sum()).clamp_min(eps).sqrt()
            corr = (x_centered * y_centered).sum() / corr_den
            corr = corr.detach()
            contrast[f"{prefix}_q_neg_r_corr_slot{ti}"] = corr
            if float(active_mask[:, ti].sum().detach().item()) > 0.0:
                corr_values.append(corr)
        if corr_values:
            corr_stack = torch.stack(corr_values)
            contrast[f"{prefix}_q_neg_r_corr_mean"] = corr_stack.mean().detach()
            contrast[f"{prefix}_q_neg_r_corr_min"] = corr_stack.min().detach()
        else:
            contrast[f"{prefix}_q_neg_r_corr_mean"] = R_pixel_det.new_tensor(0.0).detach()
            contrast[f"{prefix}_q_neg_r_corr_min"] = R_pixel_det.new_tensor(0.0).detach()
        return contrast

    # Q field pooled.
    q_lat = candidate_q_field_latent
    if q_lat.dim() == 5:
        q_lat = q_lat.squeeze(2)                                       # [B, T, Hl, Wl]
    if q_lat.dim() != 4 or q_lat.shape != (B, T, Hl, Wl):
        raise ValueError(f"candidate_q_field_latent must be [B,T,H,W] or [B,T,1,H,W], got {tuple(candidate_q_field_latent.shape)}")
    Q_pooled = _avg_pool(q_lat, K, S)                                  # [B, T, Hl', Wl']

    Hp, Wp = R_pooled.shape[-2:]

    # -------------------------------------------------------------------
    # Score: centered + std-normalised across active candidates.
    # -------------------------------------------------------------------
    score_eps = max(1e-8, float(cfg.score_norm_eps))
    am = active_mask.view(B, T, 1, 1)
    am_sum = am.sum(dim=1, keepdim=True).clamp_min(1.0)                # [B, 1, 1, 1]
    R_mean = (R_pooled * am).sum(dim=1, keepdim=True) / am_sum          # [B, 1, Hp, Wp]
    R_centered = R_pooled - R_mean
    if cfg.score_normalize:
        R_var = ((R_centered * R_centered) * am).sum(dim=1, keepdim=True) / am_sum
        R_std = R_var.clamp_min(0.0).sqrt().clamp_min(score_eps)
        Rz_select = R_centered / (R_std + score_eps)
        score = -(candidate_competition_weight_t * Rz_select) + cfg.beta * Q_pooled
    else:
        score = -(candidate_competition_weight_t * R_centered) + cfg.beta * Q_pooled
    R_competition_metric = R_mean + (candidate_competition_weight_t * R_centered)
    score_clamp_abs = float(cfg.score_clamp_abs)
    if score_clamp_abs > 0.0:
        score = score.clamp(-score_clamp_abs, score_clamp_abs)

    # -------------------------------------------------------------------
    # Blur scores depthwise on the pooled grid (sigma in pooled-pixel units).
    # Scores are finite here; only structurally inactive candidates are masked
    # after blur so -inf / sentinel values never bleed spatially.
    # -------------------------------------------------------------------
    sigma = float(cfg.gate_blur_sigma_pooled)
    if sigma > 0.0:
        score = _depthwise_gaussian_blur(score, sigma)

    # Mask inactive candidates with -inf so softmax ignores them. Active slots
    # are never q-pruned; beta * Q and temperature handle preference.
    inactive = (am == 0)
    score = score.masked_fill(inactive.expand_as(score), float("-inf"))

    tau = current_tau(cfg, current_step)
    tau = max(tau, 1e-3)
    gate = F.softmax(score / tau, dim=1)                              # [B, T, Hp, Wp]
    if cfg.gamma != 1.0:
        gate = gate.pow(cfg.gamma)
        gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(eps)

    gate_detached = gate.detach()

    # -------------------------------------------------------------------
    # Eps targets and per-candidate train loss.
    # target_eps_i = (noisy_latents - sqrt_alpha_t * targets_i) / sqrt_one_minus_alpha_t
    # -------------------------------------------------------------------
    n = noisy_latents.unsqueeze(1)                                    # [B, 1, 4, Hl, Wl]
    sa = sqrt_alpha_t.view(B, 1, 1, 1, 1)
    soma_raw = sqrt_one_minus_alpha_t.view(B, 1, 1, 1, 1).clamp_min(eps)
    sigma_floor = max(eps, float(cfg.target_eps_sigma_floor))
    soma_safe = soma_raw.clamp_min(sigma_floor)
    target_eps_raw = (n - sa * targets_x0_latents) / soma_raw          # [B, T, 4, Hl, Wl]
    target_eps_safe = (n - sa * targets_x0_latents) / soma_safe        # [B, T, 4, Hl, Wl]
    noise_pred_expanded = noise_pred.unsqueeze(1).float()
    L_train_pixel_raw = (noise_pred_expanded - target_eps_raw.float()).pow(2).mean(dim=2)  # [B, T, Hl, Wl]
    L_train_pixel_pre_safe = (noise_pred_expanded - target_eps_safe.float()).pow(2).mean(dim=2)
    if bool(cfg.robust_diffusion_loss_enabled):
        L_train_pixel = _safe_log_capped_loss(L_train_pixel_pre_safe, float(cfg.safe_pixel_loss_cap))
    else:
        L_train_pixel = torch.nan_to_num(
            L_train_pixel_pre_safe,
            nan=float(cfg.safe_pixel_loss_cap),
            posinf=float(cfg.safe_pixel_loss_cap),
            neginf=0.0,
        )
    L_train_pixel = L_train_pixel * regional_loss_sigma_weight_t
    L_train_pooled_raw = _masked_avg_pool_spatial(L_train_pixel_raw, active_content_mask_latent, K, S, 1e-6)  # [B, T, Hp, Wp]
    L_train_pooled = _masked_avg_pool_spatial(L_train_pixel, active_content_mask_latent, K, S, 1e-6)  # [B, T, Hp, Wp]
    L_train_pooled_unmasked = _avg_pool(L_train_pixel, K, S)           # [B, T, Hp, Wp]

    # Pool trusted mask on the same grid.
    trusted_pooled = _avg_pool(active_diffusion_mask_latent.float(), K, S)       # [B, 1, Hp, Wp]
    active_content_pooled = _avg_pool(active_content_mask_latent.float(), K, S)  # [B, 1, Hp, Wp]
    terrain_soft_pooled = (
        _avg_pool(terrain_soft_latent.float(), K, S)
        if terrain_soft_latent is not None
        else torch.zeros_like(active_content_pooled)
    )

    q_route_schedule_step = current_step if q_mix_step is None else int(q_mix_step)
    curriculum = current_aux_curriculum_scales(cfg, q_route_schedule_step)
    rho_q_route = float(curriculum["rho_q_route"] if rho_q_route_override is None else rho_q_route_override)
    aux_ramp = float(curriculum["aux_ramp"])
    q_route_u = float(curriculum["q_route_u"])
    am_structural = structural_active_mask.view(B, T, 1, 1)
    am_structural_sum = am_structural.sum(dim=1, keepdim=True).clamp_min(1.0)
    uniform_structural = am_structural / am_structural_sum
    q_active_raw = Q_pooled * am_structural
    q_active_raw_sum = q_active_raw.sum(dim=1, keepdim=True)
    q_routing_raw = torch.where(
        q_active_raw_sum > eps,
        q_active_raw / q_active_raw_sum.clamp_min(eps),
        uniform_structural.expand_as(q_active_raw),
    )
    q_active = q_routing_raw * am
    q_active_sum = q_active.sum(dim=1, keepdim=True)
    uniform_active = am / am_sum
    q_routing = torch.where(
        q_active_sum > eps,
        q_active / q_active_sum.clamp_min(eps),
        uniform_active.expand_as(q_active),
    )
    gamma_route = max(1.0, float(getattr(cfg, "gamma_route", cfg.gamma_boot)))
    q_route_raw_unnorm = (Q_pooled.clamp_min(0.0) + eps).pow(gamma_route) * am_structural
    q_route_raw_sum = q_route_raw_unnorm.sum(dim=1, keepdim=True)
    q_route_raw = torch.where(
        q_route_raw_sum > eps,
        q_route_raw_unnorm / q_route_raw_sum.clamp_min(eps),
        uniform_structural.expand_as(q_route_raw_unnorm),
    )
    q_route_unnorm = q_route_raw * am
    q_route_sum = q_route_unnorm.sum(dim=1, keepdim=True)
    q_route = torch.where(
        q_route_sum > eps,
        q_route_unnorm / q_route_sum.clamp_min(eps),
        uniform_active.expand_as(q_route_unnorm),
    )
    active_candidate_raw_hw = structural_active_mask.view(B, T, 1, 1).to(device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    q_lat_active_raw = q_lat.to(device=pred_x0_latents.device, dtype=pred_x0_latents.dtype).clamp_min(0.0) * active_candidate_raw_hw
    q_route_lat_raw_unnorm = (q_lat_active_raw + eps).pow(gamma_route) * active_candidate_raw_hw
    q_route_lat_raw_sum = q_route_lat_raw_unnorm.sum(dim=1, keepdim=True)
    uniform_active_raw_lat = active_candidate_raw_hw / active_candidate_raw_hw.sum(dim=1, keepdim=True).clamp_min(1.0)
    q_route_lat_raw = torch.where(
        q_route_lat_raw_sum > eps,
        q_route_lat_raw_unnorm / q_route_lat_raw_sum.clamp_min(eps),
        uniform_active_raw_lat.expand_as(q_route_lat_raw_unnorm),
    ).detach()
    q_route_lat_unnorm = q_route_lat_raw * active_candidate_hw
    q_route_lat_sum = q_route_lat_unnorm.sum(dim=1, keepdim=True)
    uniform_active_lat = active_candidate_hw / active_candidate_hw.sum(dim=1, keepdim=True).clamp_min(1.0)
    q_route_lat = torch.where(
        q_route_lat_sum > eps,
        q_route_lat_unnorm / q_route_lat_sum.clamp_min(eps),
        uniform_active_lat.expand_as(q_route_lat_unnorm),
    ).detach()
    assert not q_route_lat.requires_grad, "q-blend ownership masks must be detached"
    q_diagnostics_lat = q_route_lat.detach()
    q_boot = q_route
    phase_value = 1.0 if phase_override is None else float(phase_override)
    q_boot_usage = phase_value < 2.0
    q_route_target = q_route
    if T >= 2:
        q_route_top2 = torch.topk(q_route, k=2, dim=1).values
        q_confidence = (q_route_top2[:, 0:1] - q_route_top2[:, 1:2]).clamp_min(0.0)
    else:
        q_confidence = torch.ones((B, 1, Hp, Wp), device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    q_confidence_threshold = max(0.0, min(1.0, float(cfg.q_confidence_threshold)))
    q_confidence_mask = (q_confidence > q_confidence_threshold).to(dtype=pred_x0_latents.dtype)
    confident_content_pooled = active_content_pooled * q_confidence_mask
    confident_content_sum = confident_content_pooled.sum()
    if float(confident_content_sum.detach().item()) <= eps:
        confident_content_pooled = active_content_pooled
        confident_content_sum = active_content_pooled.sum().clamp_min(eps)
    confident_content_sum = confident_content_sum.clamp_min(eps)
    rho_tensor = torch.tensor(float(rho_q_route), device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    use_fixed_q_route_for_ownership = bool(cfg.regional_use_fixed_q_route_for_ownership)
    if use_fixed_q_route_for_ownership:
        routing_pre = q_route_target.detach()
    else:
        routing_pre = ((1.0 - rho_tensor) * gate_detached) + (rho_tensor * q_route_target)
    routing = routing_pre / routing_pre.sum(dim=1, keepdim=True).clamp_min(eps)
    routing_detached = routing.detach()
    tensor_finite_flags = {
        "pred_x0": torch.isfinite(pred_x0_latents).all(),
        "targets_x0": torch.isfinite(targets_x0_latents).all(),
        "R_pooled": torch.isfinite(R_pooled).all(),
        "L_train_raw": torch.isfinite(L_train_pixel_raw).all(),
        "L_train_safe": torch.isfinite(L_train_pixel).all(),
        "routing": torch.isfinite(routing_detached).all(),
    }
    q_lat_mass = q_route_lat.sum(dim=1, keepdim=True)
    q_lat_mass_abs_error = ((q_lat_mass - 1.0).abs() * active_content_mask_latent).sum() / active_content_mask_latent.sum().clamp_min(eps)
    invalid_candidate_hw = invalid_candidate_mask.view(B, T, 1, 1).to(device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    q_blend_invalid_candidate_leak = (
        (q_route_lat * invalid_candidate_hw).sum(dim=1, keepdim=True) * active_content_mask_latent
    ).sum() / active_content_mask_latent.sum().clamp_min(eps)
    target_eps_blend_raw = (q_route_lat.unsqueeze(2) * target_eps_raw.float()).sum(dim=1)
    target_eps_blend_safe = (q_route_lat.unsqueeze(2) * target_eps_safe.float()).sum(dim=1)
    pred_eps_blend = (q_route_lat.unsqueeze(2) * noise_pred_expanded).sum(dim=1)
    q_blend_pixel_raw = (pred_eps_blend - target_eps_blend_raw).pow(2).mean(dim=1, keepdim=True)
    q_blend_pixel_pre_safe = (pred_eps_blend - target_eps_blend_safe).pow(2).mean(dim=1, keepdim=True)
    if bool(cfg.robust_diffusion_loss_enabled):
        q_blend_pixel = _safe_log_capped_loss(q_blend_pixel_pre_safe, float(cfg.safe_pixel_loss_cap))
    else:
        q_blend_pixel = torch.nan_to_num(
            q_blend_pixel_pre_safe,
            nan=float(cfg.safe_pixel_loss_cap),
            posinf=float(cfg.safe_pixel_loss_cap),
            neginf=0.0,
        )
    q_blend_pixel = q_blend_pixel * regional_loss_sigma_weight_t
    q_blend_denominator = active_content_mask_latent.sum().clamp_min(eps)
    L_q_blend_raw = (q_blend_pixel_raw * active_content_mask_latent).sum() / q_blend_denominator
    L_q_blend = (q_blend_pixel * active_content_mask_latent).sum() / q_blend_denominator
    individual_regional_pixel_raw = L_train_pixel_raw * q_route_lat
    individual_regional_pixel = L_train_pixel * q_route_lat
    regional_loss_invalid_candidate_leak = (
        individual_regional_pixel * invalid_candidate_hw * active_content_mask_latent
    ).sum() / q_blend_denominator
    if float(q_blend_invalid_candidate_leak.detach().item()) > 1e-6:
        raise RuntimeError(f"q_blend_invalid_candidate_leak={float(q_blend_invalid_candidate_leak.detach().item()):.8g}")
    if float(regional_loss_invalid_candidate_leak.detach().item()) > 1e-6:
        raise RuntimeError(f"regional_loss_invalid_candidate_leak={float(regional_loss_invalid_candidate_leak.detach().item()):.8g}")
    L_individual_regional_raw = (individual_regional_pixel_raw * active_content_mask_latent).sum() / q_blend_denominator
    L_individual_regional = (individual_regional_pixel * active_content_mask_latent).sum() / q_blend_denominator

    q_blend_weight = float(cfg.q_blend_diffusion_weight) if bool(cfg.q_blend_diffusion_enabled) else 0.0
    individual_regional_weight = _linear_ramp(
        float(cfg.individual_regional_weight_initial),
        float(cfg.individual_regional_weight_target),
        int(cfg.individual_regional_ramp_steps),
        q_route_schedule_step,
    )
    if not bool(cfg.q_blend_diffusion_enabled):
        individual_regional_weight = 1.0

    weighted = (routing_detached * L_train_pooled).sum(dim=1, keepdim=True)  # [B, 1, Hp, Wp]
    weighted_raw = (routing_detached * L_train_pooled_raw).sum(dim=1, keepdim=True)
    denominator_per_sample = _per_sample_sum(active_content_pooled).clamp_min(eps)
    regional_loss_per_sample_safe = _per_sample_sum(weighted * active_content_pooled) / denominator_per_sample
    regional_loss_per_sample_raw = _per_sample_sum(weighted_raw * active_content_pooled) / denominator_per_sample
    competitive_surplus_initial = float(cfg.regional_competitive_surplus_initial_weight)
    competitive_surplus_target = float(cfg.regional_competitive_surplus_target_weight)
    competitive_surplus_ramp_steps = int(cfg.regional_competitive_surplus_ramp_steps)
    protected_surplus_hold_steps = (
        max(0, int(cfg.regional_routing_protected_warmup_steps))
        if bool(cfg.regional_routing_protected_warmup_enabled)
        else 0
    )
    if q_route_schedule_step <= protected_surplus_hold_steps:
        competitive_surplus_weight = competitive_surplus_initial
    else:
        competitive_surplus_weight = _linear_ramp(
            competitive_surplus_initial,
            competitive_surplus_target,
            competitive_surplus_ramp_steps,
            q_route_schedule_step - protected_surplus_hold_steps,
        )
    competitive_surplus_weight = max(0.0, min(1.0, competitive_surplus_weight))
    configured_base_weight_floor = max(0.0, min(1.0, float(cfg.competitive_gate_base_weight_floor)))
    competitive_gate_enabled = bool(cfg.competitive_gate_enabled)
    if competitive_gate_enabled:
        base_weight_floor = 1.0 - ((1.0 - configured_base_weight_floor) * competitive_surplus_weight)
    else:
        base_weight_floor = 1.0
    competitive_gate_neutralized = (not competitive_gate_enabled) or competitive_surplus_weight <= eps or base_weight_floor >= (1.0 - eps)
    gate_beta = max(0.0, min(0.9999, float(cfg.competitive_gate_beta)))
    gate_tolerance = float(cfg.competitive_gate_tolerance)
    gate_scale = max(eps, float(cfg.competitive_gate_scale))
    gate_max_delta = max(0.0, float(cfg.competitive_gate_max_delta_per_step))
    min_region_support = max(0.0, float(cfg.competitive_gate_min_region_support))
    competitor_mode = str(getattr(cfg, "competitive_gate_competitor_mode", "mean") or "mean").strip().lower()
    use_best_competitor = competitor_mode == "best"

    if advantage_ema_prev is not None:
        advantage_ema_prev = advantage_ema_prev.detach().to(device=pred_x0_latents.device, dtype=pred_x0_latents.dtype).reshape(-1)
    if competitive_gate_prev is not None:
        competitive_gate_prev = competitive_gate_prev.detach().to(device=pred_x0_latents.device, dtype=pred_x0_latents.dtype).reshape(-1)

    masked_den = active_content_pooled.sum().clamp_min(eps)

    per_slot_base_losses = []
    per_slot_competitive_losses = []
    per_slot_final_losses = []
    per_slot_catchup_losses = []
    per_slot_advantage_ema = []
    per_slot_gate_raw = []
    per_slot_gate = []
    per_slot_gate_delta = []
    per_slot_region_support = []
    per_slot_r_self = []
    per_slot_r_comp_mean = []
    per_slot_r_comp_best = []
    per_slot_advantage = []
    per_slot_hard_gate = []
    catchup_loss = L_train_pooled.new_tensor(0.0)
    R_metric = R_competition_metric
    R_metric_detached = R_metric.detach()
    for ti in range(T):
        q_route_slot = q_route[:, ti : ti + 1].detach()
        slot_valid_pooled = active_mask[:, ti].view(B, 1, 1, 1).to(dtype=L_train_pooled.dtype, device=L_train_pooled.device)
        region_weight = (q_route_slot * active_content_pooled * slot_valid_pooled).clamp_min(0.0)
        region_support = region_weight.sum().detach()
        region_den = region_support.clamp_min(eps)
        r_self_slot_live = ((R_metric[:, ti : ti + 1] * region_weight).sum() / region_den)
        r_self_slot = r_self_slot_live.detach()

        comp_values = []
        for tj in range(T):
            if tj == ti:
                continue
            comp_valid_pooled = active_mask[:, tj].view(B, 1, 1, 1).to(dtype=L_train_pooled.dtype, device=L_train_pooled.device)
            comp_region_weight = (region_weight * comp_valid_pooled).clamp_min(0.0)
            comp_region_den = comp_region_weight.sum().detach()
            if float(comp_region_den.item()) <= eps:
                continue
            comp_values.append(((R_metric_detached[:, tj : tj + 1] * comp_region_weight).sum() / comp_region_den.clamp_min(eps)).detach())
        if comp_values:
            comp_stack = torch.stack(comp_values)
            r_comp_mean_slot = comp_stack.mean().detach()
            r_comp_best_slot = comp_stack.amin().detach()
        else:
            r_comp_mean_slot = r_self_slot.detach()
            r_comp_best_slot = r_self_slot.detach()
        r_comp_slot = r_comp_best_slot if use_best_competitor else r_comp_mean_slot
        advantage_slot = (r_comp_slot - r_self_slot).detach()

        if competitive_gate_neutralized or region_support < min_region_support:
            advantage_ema_slot = torch.ones_like(advantage_slot)
            gate_raw_slot = torch.ones_like(advantage_slot)
            gate_slot = torch.ones_like(advantage_slot)
            gate_delta_slot = torch.zeros_like(advantage_slot)
        else:
            prev_adv = advantage_slot if advantage_ema_prev is None or ti >= advantage_ema_prev.numel() else advantage_ema_prev[ti]
            advantage_ema_slot = (gate_beta * prev_adv) + ((1.0 - gate_beta) * advantage_slot)
            gate_raw_slot = ((advantage_ema_slot + gate_tolerance) / gate_scale).clamp(0.0, 1.0)
            prev_gate = torch.ones_like(gate_raw_slot) if competitive_gate_prev is None or ti >= competitive_gate_prev.numel() else competitive_gate_prev[ti].clamp(0.0, 1.0)
            gate_delta_unclamped = gate_raw_slot - prev_gate
            gate_delta_slot = gate_delta_unclamped.clamp(-gate_max_delta, gate_max_delta)
            gate_slot = (prev_gate + gate_delta_slot).clamp(0.0, 1.0)

        weighted_slot_loss = ((routing_detached[:, ti : ti + 1] * L_train_pooled[:, ti : ti + 1] * active_content_pooled).sum() / masked_den)
        base_region_loss_slot = base_weight_floor * weighted_slot_loss
        competitive_region_loss_slot = ((1.0 - base_weight_floor) * gate_slot) * weighted_slot_loss
        final_region_loss_slot = base_region_loss_slot + competitive_region_loss_slot

        catchup_slot = L_train_pooled.new_tensor(0.0)
        if bool(cfg.competitive_catchup_enabled) and region_support >= min_region_support:
            catchup_slot = F.relu(r_self_slot_live - r_comp_slot.detach() + float(cfg.competitive_catchup_margin))
            catchup_loss = catchup_loss + (float(cfg.competitive_catchup_weight) * catchup_slot)

        per_slot_base_losses.append(base_region_loss_slot)
        per_slot_competitive_losses.append(competitive_region_loss_slot)
        per_slot_final_losses.append(final_region_loss_slot)
        per_slot_catchup_losses.append(catchup_slot.detach())
        per_slot_advantage_ema.append(advantage_ema_slot.detach())
        per_slot_gate_raw.append(gate_raw_slot.detach())
        per_slot_gate.append(gate_slot.detach())
        per_slot_gate_delta.append(gate_delta_slot.detach().abs())
        per_slot_region_support.append(region_support)
        per_slot_r_self.append(r_self_slot.detach())
        per_slot_r_comp_mean.append(r_comp_mean_slot.detach())
        per_slot_r_comp_best.append(r_comp_best_slot.detach())
        per_slot_advantage.append(advantage_slot.detach())
        per_slot_hard_gate.append((advantage_ema_slot.detach() <= 0.0).to(dtype=pred_x0_latents.dtype))

    L_region_hard = torch.stack(per_slot_final_losses, dim=0).sum() if per_slot_final_losses else L_train_pooled.new_tensor(0.0)
    L_region_raw_hard = (weighted_raw * active_content_pooled).sum() / masked_den
    if bool(cfg.q_blend_diffusion_enabled):
        L_region = (q_blend_weight * L_q_blend) + (individual_regional_weight * L_individual_regional)
        L_region_raw = (q_blend_weight * L_q_blend_raw) + (individual_regional_weight * L_individual_regional_raw)
    else:
        L_region = L_region_hard
        L_region_raw = L_region_raw_hard
    L_region_confident = (weighted * confident_content_pooled).sum() / confident_content_sum
    weighted_unmasked = (routing_detached * L_train_pooled_unmasked).sum(dim=1, keepdim=True)
    L_region_unmasked = (weighted_unmasked * trusted_pooled).sum() / trusted_pooled.sum().clamp_min(eps)

    q_regret_loss = L_region.new_tensor(0.0)
    q_regret_active_pair_fraction = L_region.new_tensor(0.0)
    q_regret_violation_mean = L_region.new_tensor(0.0)
    q_regret_violation_max = L_region.new_tensor(0.0)
    lambda_q_regret_current = float(
        (float(cfg.q_regret_loss_weight) * float(curriculum["q_regret_scale"]))
        if lambda_q_regret_override is None
        else lambda_q_regret_override
    )
    if bool(cfg.q_blend_diffusion_enabled):
        lambda_q_regret_current = float(cfg.competitive_regret_weight)
    lambda_bind_current = float(
        (float(cfg.bind_preference_weight) * float(curriculum["bind_scale"]))
        if lambda_bind_override is None
        else lambda_bind_override
    )
    if lambda_q_regret_current > 0.0 and T > 1:
        q_safe = q_route.clamp_min(eps)
        active_pair = (am[:, :, None] * am[:, None, :]).to(dtype=L_train_pooled.dtype)  # [B,T,T,1,1]
        self_mask = (~torch.eye(T, dtype=torch.bool, device=L_train_pooled.device)).view(1, T, T, 1, 1).to(dtype=L_train_pooled.dtype)
        active_content_pair = active_content_pooled.to(dtype=L_train_pooled.dtype).unsqueeze(1)
        assert active_content_pair.shape == (B, 1, 1, Hp, Wp), (
            f"q-regret content pair mask shape {tuple(active_content_pair.shape)} does not match {(B, 1, 1, Hp, Wp)}"
        )
        if bool(cfg.q_confidence_mask_q_regret):
            active_content_pair = active_content_pair * q_confidence_mask.to(dtype=L_train_pooled.dtype).unsqueeze(1)

        dq = q_safe[:, :, None] - q_safe[:, None, :]
        dq_excess = (dq - float(cfg.q_regret_q_tol)).clamp_min(0.0)
        constrained_pair = (dq_excess > 0.0).to(dtype=L_train_pooled.dtype)

        R_for_regret = L_train_pooled
        R_mean_regret = (R_for_regret * am).sum(dim=1, keepdim=True) / am_sum
        R_centered_regret = R_for_regret - R_mean_regret
        R_var_regret = ((R_centered_regret * R_centered_regret) * am).sum(dim=1, keepdim=True) / am_sum
        R_std_regret = R_var_regret.clamp_min(0.0).sqrt().clamp_min(score_eps)
        Rz = candidate_competition_weight_t * ((R_for_regret - R_mean_regret.detach()) / (R_std_regret.detach() + score_eps))

        prior = float(cfg.q_regret_beta_q) * torch.log(q_safe)
        s = -Rz + prior
        s_i = s[:, :, None]
        s_j_ref = s[:, None, :].detach()

        denom_scale = max(1e-6, 1.0 - float(cfg.q_regret_q_tol) + eps)
        pair_weight = (dq_excess / denom_scale).clamp_min(0.0).pow(max(0.0, float(cfg.q_regret_power))).detach()
        margin = float(cfg.q_regret_alpha) * dq_excess
        violation = (s_j_ref + margin - s_i).clamp_min(0.0)
        if float(cfg.q_regret_huber_delta) > 0.0:
            pair_loss = F.smooth_l1_loss(
                violation,
                torch.zeros_like(violation),
                reduction="none",
                beta=float(cfg.q_regret_huber_delta),
            )
        else:
            pair_loss = violation.square()
        pair_loss = pair_loss * candidate_competition_weight_t.view(B, 1, 1, 1, 1)

        finite_pair = torch.isfinite(pair_loss).to(dtype=L_train_pooled.dtype)
        valid_pair = active_pair * self_mask * active_content_pair * constrained_pair * finite_pair
        weighted_pair = pair_weight * valid_pair
        weighted_pair_den = weighted_pair.sum().clamp_min(eps)
        q_regret_loss = (weighted_pair * pair_loss).sum() / weighted_pair_den

        eligible_pair = (active_pair * self_mask * active_content_pair).sum().clamp_min(eps)
        constrained_pair_count = valid_pair.sum()
        q_regret_active_pair_fraction = constrained_pair_count / eligible_pair
        violation_den = valid_pair.sum().clamp_min(1.0)
        q_regret_violation_mean = (violation.detach() * valid_pair).sum() / violation_den
        q_regret_violation_max = torch.where(
            valid_pair > 0.0,
            violation.detach(),
            torch.zeros_like(violation.detach()),
        ).amax()

    # -------------------------------------------------------------------
    # Hard winner indices (pooled) + confidence.
    # -------------------------------------------------------------------
    # Mask inactive in gate to avoid argmax to dead slot when softmax over -inf
    # (already handled by softmax, but pin to active for safety).
    gate_for_argmax = gate_detached
    winner_idx_pooled = gate_for_argmax.argmax(dim=1, keepdim=True)   # [B, 1, Hp, Wp]
    conf_pooled = gate_for_argmax.max(dim=1, keepdim=True).values      # [B, 1, Hp, Wp]

    # -------------------------------------------------------------------
    # Diagnostics on the pooled gate.
    # -------------------------------------------------------------------
    diags: Dict[str, torch.Tensor] = {}
    valid_mask = active_content_pooled.float()
    valid_sum = valid_mask.sum().clamp_min(eps)

    diags["regional_diff_loss_raw"] = L_region_raw.detach()
    diags["regional_diff_loss_safe"] = L_region.detach()
    diags["hard_routed_regional_diff_loss_raw"] = L_region_raw_hard.detach()
    diags["hard_routed_regional_diff_loss_safe"] = L_region_hard.detach()
    diags["q_blend_diffusion_enabled"] = L_region.new_tensor(1.0 if bool(cfg.q_blend_diffusion_enabled) else 0.0).detach()
    diags["q_blend_diffusion_space"] = L_region.new_tensor(1.0).detach()
    diags["q_blend_weight"] = L_region.new_tensor(float(q_blend_weight)).detach()
    diags["individual_regional_weight"] = L_region.new_tensor(float(individual_regional_weight)).detach()
    diags["competitive_regret_weight"] = L_region.new_tensor(float(cfg.competitive_regret_weight)).detach()
    diags["q_blend_loss_raw"] = L_q_blend_raw.detach()
    diags["q_blend_loss_safe"] = L_q_blend.detach()
    diags["q_blend_loss_weighted"] = (L_q_blend.detach() * float(q_blend_weight)).detach()
    diags["individual_regional_loss_raw"] = L_individual_regional_raw.detach()
    diags["individual_regional_loss_safe"] = L_individual_regional.detach()
    diags["individual_regional_loss_weighted"] = (L_individual_regional.detach() * float(individual_regional_weight)).detach()
    diags["q_blend_q_mass_abs_error"] = q_lat_mass_abs_error.detach()
    diags["q_valid_sum_deviation_from_1"] = q_lat_mass_abs_error.detach()
    diags["q_blend_invalid_candidate_leak"] = q_blend_invalid_candidate_leak.detach()
    diags["regional_loss_invalid_candidate_leak"] = regional_loss_invalid_candidate_leak.detach()
    diags["invalid_candidate_loss_leak"] = regional_loss_invalid_candidate_leak.detach()
    diags["q_blend_active_content_mask_sum"] = q_blend_denominator.detach()
    directional_valid = (active_mask[:, 1:] > 0.0).to(dtype=pred_x0_latents.dtype)
    directional_invalid = (invalid_candidate_mask[:, 1:] > 0.0).to(dtype=pred_x0_latents.dtype)
    num_valid_directional = directional_valid.sum(dim=1)
    over_two_valid = (num_valid_directional > float(cfg.seam_validity_max_directional_seams)).to(dtype=pred_x0_latents.dtype)
    diags["num_valid_directional_seams"] = num_valid_directional.mean().detach()
    diags["valid_directional_seams_hist_0"] = (num_valid_directional == 0).to(dtype=pred_x0_latents.dtype).sum().detach()
    diags["valid_directional_seams_hist_1"] = (num_valid_directional == 1).to(dtype=pred_x0_latents.dtype).sum().detach()
    diags["valid_directional_seams_hist_2"] = (num_valid_directional == 2).to(dtype=pred_x0_latents.dtype).sum().detach()
    diags["valid_directional_seams_hist_gt2"] = over_two_valid.sum().detach()
    diags["no_valid_directional_seam_count"] = diags["valid_directional_seams_hist_0"]
    diags["one_valid_directional_seam_count"] = diags["valid_directional_seams_hist_1"]
    diags["two_valid_directional_seam_count"] = diags["valid_directional_seams_hist_2"]
    diags["over_two_valid_directional_seam_count"] = diags["valid_directional_seams_hist_gt2"]
    diags["individual_loss_skipped_invalid_count"] = directional_invalid.sum().detach()
    diags["competition_skipped_invalid_count"] = directional_invalid.sum().detach()
    invalid_pair_mask = (
        structural_active_mask[:, :, None] * structural_active_mask[:, None, :]
        * (1.0 - active_mask[:, :, None] * active_mask[:, None, :])
    )
    diags["q_regret_invalid_pair_count"] = invalid_pair_mask.sum().detach()
    for _slot_idx, _slot_name in enumerate(("interior", "north", "south", "east", "west")):
        if _slot_idx < T:
            _candidate_valid_mean = candidate_valid_mask[:, _slot_idx].mean().detach()
        else:
            _candidate_valid_mean = L_region.new_tensor(0.0).detach()
        diags[f"candidate_valid_{_slot_name}_mean"] = _candidate_valid_mean
        diags[f"candidate_valid_slot{_slot_idx}_mean"] = _candidate_valid_mean
    diags["valid_north_rate"] = diags["candidate_valid_north_mean"]
    diags["valid_south_rate"] = diags["candidate_valid_south_mean"]
    diags["valid_east_rate"] = diags["candidate_valid_east_mean"]
    diags["valid_west_rate"] = diags["candidate_valid_west_mean"]
    if candidate_valid_diagnostics:
        for _diag_name, _diag_value in candidate_valid_diagnostics.items():
            if isinstance(_diag_value, torch.Tensor):
                _tensor_value = _diag_value.detach().to(device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
                diags[_diag_name] = _tensor_value.mean().detach() if _tensor_value.numel() > 1 else _tensor_value.reshape(()).detach()
    diags["regional_use_fixed_q_route_for_ownership"] = L_region.new_tensor(1.0 if use_fixed_q_route_for_ownership else 0.0).detach()
    diags["regional_freeze_gate_routing_head"] = L_region.new_tensor(1.0 if bool(cfg.regional_freeze_gate_routing_head) else 0.0).detach()
    diags["regional_detach_routing_from_regional_loss"] = L_region.new_tensor(1.0 if bool(cfg.regional_detach_routing_from_regional_loss) else 0.0).detach()
    diags["competitive_gate_enabled"] = L_region.new_tensor(1.0 if competitive_gate_enabled else 0.0).detach()
    diags["competitive_gate_neutralized_for_equivalence"] = L_region.new_tensor(1.0 if competitive_gate_neutralized else 0.0).detach()
    diags["base_weight_floor"] = L_region.new_tensor(float(base_weight_floor)).detach()
    diags["competitive_surplus_weight"] = L_region.new_tensor(float(competitive_surplus_weight)).detach()
    diags["competitive_gate_effective_weight"] = L_region.new_tensor(float(1.0 - base_weight_floor)).detach()
    diags["catchup_weight"] = L_region.new_tensor(float(cfg.competitive_catchup_weight if cfg.competitive_catchup_enabled else 0.0)).detach()
    diags["regional_diff_loss_q_confident"] = L_region_confident.detach()
    diags["regional_loss_raw_safe_ratio"] = (L_region_raw.detach() / L_region.detach().clamp_min(eps)).detach()
    diags["regional_loss_per_sample_raw"] = regional_loss_per_sample_raw.detach()
    diags["regional_loss_per_sample_safe"] = regional_loss_per_sample_safe.detach()
    diags["denominator_per_sample"] = denominator_per_sample.detach()
    diags["regional_loss_denominator"] = masked_den.detach()
    diags["regional_loss_min_denominator"] = denominator_per_sample.min().detach()
    diags["active_diffusion_mask_sum_per_sample"] = active_diffusion_sum_per_sample.detach()
    diags["active_content_mask_sum_per_sample"] = active_content_sum_per_sample.detach()
    diags["active_content_fraction_per_sample"] = active_content_fraction_per_sample.detach()
    diags["active_content_fraction_per_sample_raw"] = active_content_fraction_per_sample_raw.detach()
    diags["fraction_of_active_diffusion_pixels_kept"] = active_content_fraction.detach()
    diags["terrain_soft_sum"] = terrain_sum_per_sample.sum().detach()
    diags["terrain_soft_sum_per_sample"] = terrain_sum_per_sample.detach()
    diags["terrain_empty_sample_count"] = terrain_empty_per_sample.to(dtype=pred_x0_latents.dtype).sum().detach()
    diags["terrain_empty_count"] = diags["terrain_empty_sample_count"]
    diags["seam_boundary_content_mask_sum"] = seam_boundary_content_mask_latent.sum().detach()
    diags["fallback_active_mask_used_per_sample"] = active_content_fallback_per_sample.to(dtype=pred_x0_latents.dtype).detach()
    diags["active_content_fallback_count"] = active_content_fallback_per_sample.to(dtype=pred_x0_latents.dtype).sum().detach()
    diags["terrain_soft_mean_per_sample"] = _per_sample_mean(terrain_soft).detach()
    diags["terrain_soft_min"] = terrain_soft.amin().detach()
    diags["terrain_soft_max"] = terrain_soft.amax().detach()
    diags["max_pixel_loss_per_sample"] = _per_sample_max(L_train_pixel_raw).detach()
    diags["mean_pixel_loss_per_sample"] = _per_sample_mean(L_train_pixel_raw).detach()
    candidate_loss_mean = (L_train_pixel_raw * active_content_mask_latent).sum(dim=(-2, -1)) / active_content_sum_per_sample.view(B, 1).clamp_min(eps)
    max_candidate_loss_per_sample, candidate_with_max_loss = candidate_loss_mean.max(dim=1)
    diags["max_candidate_loss_per_sample"] = max_candidate_loss_per_sample.detach()
    diags["candidate_with_max_loss"] = candidate_with_max_loss.to(dtype=pred_x0_latents.dtype).detach()
    flat_pixel_idx = L_train_pixel_raw.reshape(B, -1).argmax(dim=1)
    pixel_area = Hl * Wl
    pixel_candidate = torch.div(flat_pixel_idx, pixel_area, rounding_mode="floor")
    pixel_offset = flat_pixel_idx % pixel_area
    pixel_y = torch.div(pixel_offset, Wl, rounding_mode="floor")
    pixel_x = pixel_offset % Wl
    diags["pixel_max_candidate"] = pixel_candidate.to(dtype=pred_x0_latents.dtype).detach()
    diags["pixel_max_y"] = pixel_y.to(dtype=pred_x0_latents.dtype).detach()
    diags["pixel_max_x"] = pixel_x.to(dtype=pred_x0_latents.dtype).detach()
    diags["isfinite_pred_x0"] = tensor_finite_flags["pred_x0"].to(dtype=pred_x0_latents.dtype).detach()
    diags["isfinite_target_i"] = tensor_finite_flags["targets_x0"].to(dtype=pred_x0_latents.dtype).detach()
    diags["isfinite_R_i"] = tensor_finite_flags["R_pooled"].to(dtype=pred_x0_latents.dtype).detach()
    diags["isfinite_L_train_i"] = tensor_finite_flags["L_train_raw"].to(dtype=pred_x0_latents.dtype).detach()
    diags["isfinite_L_train_i_safe"] = tensor_finite_flags["L_train_safe"].to(dtype=pred_x0_latents.dtype).detach()
    diags["isfinite_routing"] = tensor_finite_flags["routing"].to(dtype=pred_x0_latents.dtype).detach()
    diags["regional_nonfinite_detected"] = torch.tensor(
        0.0 if all(bool(flag.detach().item()) for flag in tensor_finite_flags.values()) else 1.0,
        device=pred_x0_latents.device,
        dtype=pred_x0_latents.dtype,
    )
    diags["max_abs_pred_x0"] = pred_x0_latents.detach().abs().amax()
    diags["max_abs_target_i"] = targets_x0_latents.detach().abs().amax()
    diags["max_abs_pred_target_delta"] = diff_full.detach().abs().amax()
    diags["max_abs_R_i"] = R_pooled.detach().abs().amax()
    diags["max_abs_L_train_i"] = L_train_pixel_raw.detach().abs().amax()
    diags["max_abs_L_train_i_safe"] = L_train_pixel.detach().abs().amax()
    diags["diffusion_loss_weight_t_raw"] = torch.ones((B,), device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    diags["diffusion_loss_weight_t_safe"] = torch.ones((B,), device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    diags["diffusion_loss_weight_t_max"] = torch.tensor(1.0, device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    diags["candidate_competition_weight_t"] = candidate_competition_weight_t.detach().reshape(B)
    diags["candidate_competition_weight_mean"] = candidate_competition_weight_t.detach().mean()
    diags["candidate_competition_weight_min"] = candidate_competition_weight_t.detach().amin()
    diags["candidate_competition_weight_max"] = candidate_competition_weight_t.detach().amax()
    diags["sigma"] = sigma_competition.detach().mean()
    diags["sigma_min"] = sigma_competition.detach().amin()
    diags["sigma_max"] = sigma_competition.detach().amax()
    diags["snr"] = snr_competition.detach().mean()
    diags["snr_min"] = snr_competition.detach().amin()
    diags["snr_max"] = snr_competition.detach().amax()
    diags["regional_loss_sigma_weight_t"] = regional_loss_sigma_weight_t.detach().reshape(B)
    diags["regional_loss_sigma_weight_mean"] = regional_loss_sigma_weight_t.detach().mean()
    diags["regional_loss_sigma_weight_min"] = regional_loss_sigma_weight_t.detach().amin()
    diags["regional_loss_sigma_weight_max"] = regional_loss_sigma_weight_t.detach().amax()
    diags["regional_low_sigma_weight"] = diags["regional_loss_sigma_weight_mean"]
    diags["regional_competition_skipped_low_sigma"] = torch.tensor(
        1.0 if bool(competition_skipped_low_sigma.any().detach().item()) else 0.0,
        device=pred_x0_latents.device,
        dtype=pred_x0_latents.dtype,
    )
    diags["regional_competition_skipped_low_sigma_count"] = competition_skipped_low_sigma.to(dtype=pred_x0_latents.dtype).sum().detach()
    diags["target_eps_sigma_min_raw"] = soma_raw.detach().reshape(B, -1).amin(dim=1)
    diags["target_eps_sigma_min_safe"] = soma_safe.detach().reshape(B, -1).amin(dim=1)
    diags["active_content_mask_mean"] = active_content_mask_latent.mean().detach()
    diags["active_content_mask_sum"] = active_content_mask_latent.sum().detach()
    diags["active_content_mask_sum_latent"] = active_content_mask_latent.sum().detach()
    diags["active_content_latent_mask_sum"] = active_content_mask_latent.sum().detach()
    diags["active_content_mask_sum_pooled"] = active_content_pooled.sum().detach()
    diags["active_content_pooled_mask_sum"] = active_content_pooled.sum().detach()
    diags["terrain_soft_mean"] = terrain_soft.mean().detach()
    diags["active_content_fraction"] = active_content_fraction.detach()
    diags["active_content_mask_fallback"] = torch.tensor(
        1.0 if bool(active_content_fallback_per_sample.any().detach().item()) else 0.0,
        device=pred_x0_latents.device,
        dtype=pred_x0_latents.dtype,
    )
    diags["hard_band_valid_mask_sum"] = hard_band_content_mask_latent.sum().detach()
    diags["regional_diff_loss_unmasked"] = L_region_unmasked.detach()
    diags.update(_candidate_pixel_contrast(active_content_mask_latent, "ri_active"))
    diags.update(_candidate_pixel_contrast(active_diffusion_mask_latent, "ri_unmasked"))
    diags["q_confidence_mean"] = ((q_confidence * valid_mask).sum() / valid_sum).detach()
    diags["q_confidence_max"] = q_confidence.detach().amax()
    diags["q_confidence_threshold"] = torch.tensor(float(q_confidence_threshold), device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    diags["q_confidence_mask_fraction"] = ((q_confidence_mask * valid_mask).sum() / valid_sum).detach()
    phase_diag_mask = valid_mask
    if bool(cfg.q_confidence_mask_phase_diagnostics):
        phase_diag_mask = valid_mask * q_confidence_mask
        if float(phase_diag_mask.sum().detach().item()) <= eps:
            phase_diag_mask = valid_mask
    phase_diag_sum = phase_diag_mask.sum().clamp_min(eps)

    log_gate = torch.log(gate_for_argmax.clamp_min(1e-12))
    entropy_per_cell = -(gate_for_argmax * log_gate).sum(dim=1, keepdim=True)  # [B, 1, Hp, Wp]
    gate_entropy_mean = (entropy_per_cell * valid_mask).sum() / valid_sum
    T_active_mean = active_mask.sum(dim=1).mean().clamp_min(1.0)
    log_T = torch.log(T_active_mean)
    gate_entropy_normalized = gate_entropy_mean / log_T.clamp_min(1e-6)
    diags["gate_entropy_mean"] = gate_entropy_mean
    diags["gate_entropy_normalized"] = gate_entropy_normalized

    # Top1-top2 gap on gate.
    if T >= 2:
        top2 = torch.topk(gate_for_argmax, k=2, dim=1).values  # [B, 2, Hp, Wp]
        gap_per_cell = (top2[:, 0:1] - top2[:, 1:2])
    else:
        gap_per_cell = torch.zeros_like(conf_pooled)
    diags["gate_score_gap_mean"] = (gap_per_cell * valid_mask).sum() / valid_sum

    # Gate smoothness: L2 of laplacian of argmax map (treat as float).
    arg_f = winner_idx_pooled.float()
    lap_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                              device=arg_f.device, dtype=arg_f.dtype).view(1, 1, 3, 3)
    lap = F.conv2d(arg_f, lap_kernel, padding=1)
    diags["gate_smoothness"] = (lap.pow(2) * valid_mask).sum().sqrt() / valid_sum.sqrt()

    # q_winner_alignment: fraction of valid cells where argmax(gate) matches argmax(Q).
    q_argmax = q_route.argmax(dim=1, keepdim=True)
    q_align = (winner_idx_pooled == q_argmax).float()
    diags["q_winner_alignment_all"] = (q_align * valid_mask).sum() / valid_sum
    diags["q_winner_alignment"] = (q_align * phase_diag_mask).sum() / phase_diag_sum

    diags["q_regret_loss"] = q_regret_loss.detach()
    diags["q_regret_loss_weighted"] = q_regret_loss.detach() * float(lambda_q_regret_current)
    diags["q_regret_active_pair_fraction"] = q_regret_active_pair_fraction.detach()
    diags["q_regret_violation_mean"] = q_regret_violation_mean.detach()
    diags["q_regret_violation_max"] = q_regret_violation_max.detach()
    routing_mass = routing_detached.sum(dim=1, keepdim=True)
    routing_entropy_per_cell = -(routing_detached * torch.log(routing_detached.clamp_min(1e-12))).sum(dim=1, keepdim=True)
    q_entropy_per_cell = -(q_routing * torch.log(q_routing.clamp_min(1e-12))).sum(dim=1, keepdim=True)
    q_raw_entropy_per_cell = -(q_routing_raw * torch.log(q_routing_raw.clamp_min(1e-12))).sum(dim=1, keepdim=True)
    q_boot_entropy_per_cell = -(q_boot * torch.log(q_boot.clamp_min(1e-12))).sum(dim=1, keepdim=True)
    routing_gate_l1 = (routing_detached - gate_detached).abs().sum(dim=1, keepdim=True)
    routing_q_l1 = (routing_detached - q_route_target).abs().sum(dim=1, keepdim=True)
    diags["lambda_q_mix"] = torch.tensor(0.0, device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    diags["q_mix_weight"] = diags["lambda_q_mix"]
    diags["q_mix_schedule_step"] = torch.tensor(float(max(0, int(q_route_schedule_step))), device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    diags["q_route_schedule_step"] = diags["q_mix_schedule_step"]
    diags["rho_q_route"] = torch.tensor(float(rho_q_route), device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    diags["q_route_u"] = torch.tensor(float(q_route_u), device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    diags["gamma_boot"] = torch.tensor(float(gamma_route), device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    diags["gamma_route"] = torch.tensor(float(gamma_route), device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    diags["aux_ramp"] = torch.tensor(float(aux_ramp), device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    diags["lambda_q_regret_current"] = torch.tensor(float(lambda_q_regret_current), device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    diags["lambda_bind_current"] = torch.tensor(float(lambda_bind_current), device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    diags["hard_band_current_scale"] = torch.tensor(
        float(curriculum["hard_band_scale"] if hard_band_scale_override is None else hard_band_scale_override),
        device=pred_x0_latents.device,
        dtype=pred_x0_latents.dtype,
    )
    diags["q_entropy_mean"] = ((q_entropy_per_cell * valid_mask).sum() / valid_sum).detach()
    diags["q_entropy_normalized"] = (diags["q_entropy_mean"] / log_T.clamp_min(1e-6)).detach()
    T_structural_mean = structural_active_mask.sum(dim=1).mean().clamp_min(1.0)
    log_T_structural = torch.log(T_structural_mean).clamp_min(1e-6)
    diags["q_raw_entropy_mean"] = ((q_raw_entropy_per_cell * valid_mask).sum() / valid_sum).detach()
    diags["q_raw_entropy_normalized"] = (diags["q_raw_entropy_mean"] / log_T_structural).detach()
    q_route_entropy_per_cell = -(q_route * torch.log(q_route.clamp_min(1e-12))).sum(dim=1, keepdim=True)
    diags["q_route_entropy_mean"] = ((q_route_entropy_per_cell * valid_mask).sum() / valid_sum).detach()
    diags["q_route_entropy_normalized"] = (diags["q_route_entropy_mean"] / log_T.clamp_min(1e-6)).detach()
    diags["q_boot_entropy_mean"] = ((q_boot_entropy_per_cell * valid_mask).sum() / valid_sum).detach()
    diags["q_boot_entropy_normalized"] = (diags["q_boot_entropy_mean"] / log_T.clamp_min(1e-6)).detach()
    diags["q_boot_usage"] = torch.tensor(
        1.0 if q_boot_usage else 0.0,
        device=pred_x0_latents.device,
        dtype=pred_x0_latents.dtype,
    )
    diags["routing_entropy_mean"] = ((routing_entropy_per_cell * valid_mask).sum() / valid_sum).detach()
    diags["routing_entropy_normalized"] = (diags["routing_entropy_mean"] / log_T.clamp_min(1e-6)).detach()
    diags["routing_gate_l1_mean"] = ((routing_gate_l1 * valid_mask).sum() / valid_sum).detach()
    diags["routing_q_l1_mean"] = ((routing_q_l1 * valid_mask).sum() / valid_sum).detach()
    diags["routing_mass_mean"] = ((routing_mass * valid_mask).sum() / valid_sum).detach()
    diags["routing_mass_abs_error"] = (((routing_mass - 1.0).abs() * valid_mask).sum() / valid_sum).detach()
    diags["q_mix_added_mass"] = diags["routing_mass_abs_error"]

    in_region_advantages = []
    in_region_advantage_emas = []
    valid_directional_advantages = []
    ri_qboot_advantages = []
    q_mass_sum = torch.tensor(0.0, device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    q_raw_mass_sum = torch.tensor(0.0, device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    q_diagnostics_lat = q_route_lat.detach()
    # winner_share per slot index 0..T-1.
    for ti in range(T):
        share_mask = (winner_idx_pooled == ti).float() * valid_mask
        diags[f"winner_share_slot{ti}"] = share_mask.sum() / valid_sum
        q_raw_slot = Q_pooled[:, ti : ti + 1]
        q_raw_route_slot = q_route_raw[:, ti : ti + 1]
        q_route_slot = q_route[:, ti : ti + 1]
        q_slot = q_route_slot
        gate_slot = gate_detached[:, ti : ti + 1]
        routing_slot = routing_detached[:, ti : ti + 1]
        invalid_slot_mask = invalid_candidate_mask[:, ti].view(B, 1, 1, 1).to(dtype=valid_mask.dtype, device=valid_mask.device)
        q_mass = (q_route_slot * valid_mask).sum() / valid_sum
        q_raw_mass = (q_raw_route_slot * valid_mask).sum() / valid_sum
        invalid_q_mass = (q_raw_route_slot * valid_mask * invalid_slot_mask).sum() / valid_sum
        gate_share = (gate_slot * valid_mask).sum() / valid_sum
        routing_share = (routing_slot * valid_mask).sum() / valid_sum
        invalid_gate_mass = (gate_slot * valid_mask * invalid_slot_mask).sum() / valid_sum
        invalid_routing_mass = (routing_slot * valid_mask * invalid_slot_mask).sum() / valid_sum
        invalid_winner_share = (((winner_idx_pooled == ti).float() * valid_mask * invalid_slot_mask).sum() / valid_sum).detach()
        q_route_weight = (q_route_slot * valid_mask).clamp_min(0.0)
        q_route_weight_den = q_route_weight.sum().clamp_min(eps)
        q_route_terrain_support = ((terrain_soft_pooled * q_route_weight).sum() / q_route_weight_den).detach()
        regional_loss_contribution = per_slot_final_losses[ti].detach()
        diags[f"q_mass_slot{ti}"] = q_mass.detach()
        diags[f"q_valid_mass_slot{ti}"] = q_mass.detach()
        diags[f"q_raw_mass_slot{ti}"] = q_raw_mass.detach()
        diags[f"invalid_q_mass_slot{ti}"] = invalid_q_mass.detach()
        diags[f"invalid_raw_q_mass_slot{ti}"] = invalid_q_mass.detach()
        diags[f"invalid_routing_mass_slot{ti}"] = invalid_routing_mass.detach()
        diags[f"invalid_gate_mass_slot{ti}"] = invalid_gate_mass.detach()
        diags[f"invalid_winner_share_slot{ti}"] = invalid_winner_share.detach()
        q_mass_sum = q_mass_sum + q_mass.detach()
        q_raw_mass_sum = q_raw_mass_sum + q_raw_mass.detach()
        diags[f"gate_share_slot{ti}"] = gate_share.detach()
        diags[f"routing_share_slot{ti}"] = routing_share.detach()
        diags[f"q_route_terrain_support_slot{ti}"] = q_route_terrain_support
        diags[f"regional_loss_contribution_slot{ti}"] = regional_loss_contribution
        diags[f"gate_q_ratio_slot{ti}"] = (gate_share / q_mass.clamp_min(eps)).detach()

        q_band = ((q_slot >= float(cfg.q_high_threshold)).float() * phase_diag_mask)
        q_band_den = q_band.sum().clamp_min(1.0)
        diags[f"q_band_gate_slot{ti}"] = ((gate_slot * q_band).sum() / q_band_den).detach()

        q_phase_mass = (q_slot * phase_diag_mask).sum() / phase_diag_sum
        gate_phase_share = (gate_slot * phase_diag_mask).sum() / phase_diag_sum
        routing_phase_share = (routing_slot * phase_diag_mask).sum() / phase_diag_sum
        q_centered = (q_slot - q_phase_mass) * phase_diag_mask
        g_centered = (gate_slot - gate_phase_share) * phase_diag_mask
        corr_num = (q_centered * g_centered).sum()
        corr_den = ((q_centered.square().sum() * g_centered.square().sum()).clamp_min(eps)).sqrt()
        diags[f"q_gate_corr_slot{ti}"] = (corr_num / corr_den).detach()

        r_centered = (routing_slot - routing_phase_share) * phase_diag_mask
        routing_corr_num = (q_centered * r_centered).sum()
        routing_corr_den = ((q_centered.square().sum() * r_centered.square().sum()).clamp_min(eps)).sqrt()
        diags[f"q_routing_corr_slot{ti}"] = (routing_corr_num / routing_corr_den).detach()

        r_slot = R_competition_metric[:, ti : ti + 1]
        diags[f"r_mean_slot{ti}"] = ((r_slot * valid_mask).sum() / valid_sum).detach()
        high_q_mask = ((q_slot >= float(cfg.q_high_threshold)).float() * phase_diag_mask)
        low_q_mask = ((q_slot <= float(cfg.q_low_threshold)).float() * phase_diag_mask)
        high_q_sum = high_q_mask.sum()
        low_q_sum = low_q_mask.sum()
        if float(high_q_sum.detach().item()) > 1.0 and float(low_q_sum.detach().item()) > 1.0:
            high_q_den = high_q_sum.clamp_min(1.0)
            low_q_den = low_q_sum.clamp_min(1.0)
            r_high_q = ((r_slot * high_q_mask).sum() / high_q_den).detach()
            r_low_q = ((r_slot * low_q_mask).sum() / low_q_den).detach()
        else:
            q_weight = (q_slot * phase_diag_mask).clamp_min(0.0)
            inv_q_weight = ((1.0 - q_slot) * phase_diag_mask).clamp_min(0.0)
            r_high_q = ((r_slot * q_weight).sum() / q_weight.sum().clamp_min(1.0)).detach()
            r_low_q = ((r_slot * inv_q_weight).sum() / inv_q_weight.sum().clamp_min(1.0)).detach()
        diags[f"r_high_q_slot{ti}"] = r_high_q
        diags[f"r_low_q_slot{ti}"] = r_low_q
        diags[f"specialization_advantage_slot{ti}"] = (r_low_q - r_high_q).detach()

        comp_mask = am.clone()
        comp_mask[:, ti : ti + 1] = 0.0
        comp_count = comp_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        r_self = per_slot_r_self[ti]
        r_comp = per_slot_r_comp_mean[ti] if not use_best_competitor else per_slot_r_comp_best[ti]
        in_region_advantage = per_slot_advantage[ti]
        qboot_weight = (q_boot[:, ti : ti + 1].detach() * valid_mask).clamp_min(0.0)
        qboot_den = qboot_weight.sum().clamp_min(eps)
        r_pooled_slot = R_pooled[:, ti : ti + 1]
        r_pooled_comp = (R_pooled * comp_mask).sum(dim=1, keepdim=True) / comp_count
        ri_qboot_self = ((r_pooled_slot * qboot_weight).sum() / qboot_den).detach()
        ri_qboot_comp = ((r_pooled_comp * qboot_weight).sum() / qboot_den).detach()
        in_region_advantage_ema = per_slot_advantage_ema[ti]
        diags[f"R_self_slot{ti}"] = r_self
        diags[f"R_comp_slot{ti}"] = r_comp
        diags[f"R_comp_mean_slot{ti}"] = per_slot_r_comp_mean[ti]
        diags[f"R_comp_best_slot{ti}"] = per_slot_r_comp_best[ti]
        diags[f"in_region_advantage_slot{ti}"] = in_region_advantage
        diags[f"in_region_advantage_ema_slot{ti}"] = in_region_advantage_ema
        diags[f"region_weight_sum_slot{ti}"] = per_slot_region_support[ti]
        diags[f"competitive_gate_raw_slot{ti}"] = per_slot_gate_raw[ti]
        diags[f"competitive_gate_slot{ti}"] = per_slot_gate[ti]
        diags[f"competitive_gate_delta_slot{ti}"] = per_slot_gate_delta[ti]
        diags[f"base_region_loss_slot{ti}"] = per_slot_base_losses[ti]
        diags[f"competitive_region_loss_slot{ti}"] = per_slot_competitive_losses[ti]
        diags[f"final_region_loss_slot{ti}"] = per_slot_final_losses[ti]
        diags[f"catchup_loss_slot{ti}"] = per_slot_catchup_losses[ti]
        diags[f"catchup_active_slot{ti}"] = torch.tensor(1.0 if float(per_slot_catchup_losses[ti].item()) > 0.0 else 0.0, device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
        diags[f"would_hard_gate_suppress_slot{ti}"] = per_slot_hard_gate[ti]
        diags[f"ri_qboot_R_self_slot{ti}"] = ri_qboot_self
        diags[f"ri_qboot_R_comp_slot{ti}"] = ri_qboot_comp
        diags[f"ri_qboot_in_region_advantage_slot{ti}"] = (ri_qboot_comp - ri_qboot_self).detach()
        if float(active_mask[:, ti].sum().detach().item()) > 0.0:
            in_region_advantages.append(in_region_advantage)
            in_region_advantage_emas.append(in_region_advantage_ema)
            ri_qboot_advantages.append((ri_qboot_comp - ri_qboot_self).detach())
            if ti > 0:
                valid_directional_advantages.append(in_region_advantage)

    if in_region_advantages:
        in_region_stack = torch.stack(in_region_advantages)
        diags["min_in_region_advantage"] = in_region_stack.min().detach()
        diags["mean_in_region_advantage"] = in_region_stack.mean().detach()
        worst_region_index = int(in_region_stack.argmin().detach().item())
        diags["worst_region_index"] = torch.tensor(float(worst_region_index), device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
        diags["worst_region_slot"] = diags["worst_region_index"]
        diags["worst_region_advantage"] = in_region_stack[worst_region_index].detach()
        diags["worst_region_is_active"] = torch.tensor(1.0, device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
        diags["worst_region_R_self"] = diags[f"R_self_slot{worst_region_index}"].detach()
        diags["worst_region_R_comp"] = diags[f"R_comp_slot{worst_region_index}"].detach()
        diags["worst_region_q_route_mass"] = diags[f"q_mass_slot{worst_region_index}"].detach()
        diags["worst_region_q_raw_mass"] = diags[f"q_raw_mass_slot{worst_region_index}"].detach()
        diags["worst_region_gate_share"] = diags[f"gate_share_slot{worst_region_index}"].detach()
        diags["worst_region_routing_share"] = diags[f"routing_share_slot{worst_region_index}"].detach()
        diags["worst_region_winner_share"] = diags[f"winner_share_slot{worst_region_index}"].detach()
    else:
        zero_diag = L_region.new_tensor(0.0).detach()
        diags["min_in_region_advantage"] = zero_diag
        diags["mean_in_region_advantage"] = zero_diag
        diags["worst_region_index"] = zero_diag
        diags["worst_region_slot"] = zero_diag
        diags["worst_region_advantage"] = zero_diag
        diags["worst_region_is_active"] = zero_diag
        diags["worst_region_R_self"] = zero_diag
        diags["worst_region_R_comp"] = zero_diag
        diags["worst_region_q_route_mass"] = zero_diag
        diags["worst_region_q_raw_mass"] = zero_diag
        diags["worst_region_gate_share"] = zero_diag
        diags["worst_region_routing_share"] = zero_diag
        diags["worst_region_winner_share"] = zero_diag
    if valid_directional_advantages:
        valid_directional_stack = torch.stack(valid_directional_advantages)
        diags["valid_min_in_region_advantage"] = valid_directional_stack.min().detach()
        diags["valid_mean_in_region_advantage"] = valid_directional_stack.mean().detach()
    else:
        zero_diag = L_region.new_tensor(0.0).detach()
        diags["valid_min_in_region_advantage"] = zero_diag
        diags["valid_mean_in_region_advantage"] = zero_diag
    diags["invalid_excluded_advantage_count"] = directional_invalid.sum().detach()
    diags["q_route_mass_sum"] = q_mass_sum.detach()
    diags["q_raw_mass_sum"] = q_raw_mass_sum.detach()
    diags["competitive_catchup_loss"] = catchup_loss.detach()
    diags["advantage_ema_state"] = torch.stack(per_slot_advantage_ema).detach() if per_slot_advantage_ema else torch.zeros((0,), device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    diags["competitive_gate_state"] = torch.stack(per_slot_gate).detach() if per_slot_gate else torch.zeros((0,), device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    if ri_qboot_advantages:
        ri_qboot_stack = torch.stack(ri_qboot_advantages)
        diags["ri_qboot_min_in_region_advantage"] = ri_qboot_stack.min().detach()
        diags["ri_qboot_mean_in_region_advantage"] = ri_qboot_stack.mean().detach()
    else:
        zero_diag = L_region.new_tensor(0.0).detach()
        diags["ri_qboot_min_in_region_advantage"] = zero_diag
        diags["ri_qboot_mean_in_region_advantage"] = zero_diag

    diags["tau_current"] = torch.tensor(float(tau), device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    diags["gamma_current"] = torch.tensor(float(cfg.gamma), device=pred_x0_latents.device, dtype=pred_x0_latents.dtype)
    diags["regional_diff_loss_unweighted"] = L_region.detach()

    # -------------------------------------------------------------------
    # Upsample winner indices and confidence back to latent grid.
    # -------------------------------------------------------------------
    winner_idx_latent = F.interpolate(
        winner_idx_pooled.float(), size=(Hl, Wl), mode="nearest"
    ).long()  # [B, 1, Hl, Wl]
    conf_latent = F.interpolate(conf_pooled, size=(Hl, Wl), mode="nearest")  # [B, 1, Hl, Wl]

    return {
        "loss": L_region,
        "loss_q_confident": L_region_confident,
        "q_regret_loss": q_regret_loss,
        "catchup_loss": catchup_loss,
        "gate": gate_detached,
        "winner_idx_pooled": winner_idx_pooled,
        "winner_idx_latent": winner_idx_latent,
        "conf_pooled": conf_pooled,
        "conf_latent": conf_latent,
        "active_content_mask_latent": active_content_mask_latent.detach(),
        "terrain_soft_latent": terrain_soft.detach(),
        "hard_band_content_mask_latent": hard_band_content_mask_latent.detach(),
        "R_pixel": R_pixel_det.detach(),
        "R_pixel_active": (R_pixel_det * active_content_mask_latent).detach(),
        "R_pixel_unmasked_active": (R_pixel_det * active_diffusion_mask_latent).detach(),
        "R_advantage_pixel": R_advantage_pixel.detach(),
        "L_train_pixel_raw": L_train_pixel_raw.detach(),
        "L_train_pixel_safe": L_train_pixel.detach(),
        "L_train_pooled_raw": L_train_pooled_raw.detach(),
        "L_train_pooled_safe": L_train_pooled.detach(),
        "weighted_loss_map_raw": weighted_raw.detach(),
        "weighted_loss_map_safe": weighted.detach(),
        "q_blend_loss_map_raw": q_blend_pixel_raw.detach(),
        "q_blend_loss_map_safe": q_blend_pixel.detach(),
        "individual_regional_loss_map_safe": individual_regional_pixel.detach(),
        "target_eps_blend_raw": target_eps_blend_raw.detach(),
        "target_eps_blend_safe": target_eps_blend_safe.detach(),
        "pred_eps_blend": pred_eps_blend.detach(),
        "q_blend_target_x0_latents": (q_route_lat.unsqueeze(2) * targets_x0_latents.float()).sum(dim=1).detach(),
        "q_blend_q_latent": q_route_lat.detach(),
        "active_content_pooled": active_content_pooled.detach(),
        "trusted_pooled": trusted_pooled.detach(),
        "q_raw_pooled": q_routing_raw.detach(),
        "q_route_pooled": q_route.detach(),
        "routing_pooled": routing_detached,
        "q_boot_pooled": q_boot.detach(),
        "R_pooled": R_pooled,
        "score_pooled": score,
        "tau": tau,
        "diagnostics": diags,
    }


# ---------------------------------------------------------------------------
# RGB auxiliary loss
# ---------------------------------------------------------------------------


def gather_winner_latents(
    targets_x0_latents: torch.Tensor,   # [B, T, 4, Hl, Wl]
    winner_idx_latent: torch.Tensor,    # [B, 1, Hl, Wl] long
) -> torch.Tensor:
    B, T, Cl, Hl, Wl = targets_x0_latents.shape
    idx = winner_idx_latent.unsqueeze(2).expand(B, 1, Cl, Hl, Wl)  # [B, 1, 4, Hl, Wl]
    winner = torch.gather(targets_x0_latents, dim=1, index=idx).squeeze(1)  # [B, 4, Hl, Wl]
    return winner


def compute_rgb_aux_loss(
    pred_rgb: torch.Tensor,        # [B, 3, H, W] from vae.decode(pred_x0)
    winner_rgb: torch.Tensor,       # [B, 3, H, W] from vae.decode(winner_latents) under no_grad
    trusted_mask_full: torch.Tensor,  # [B, 1, H, W] in [0,1]
    conf_full: torch.Tensor,        # [B, 1, H, W] in [0,1]
    confidence_weighted: bool,
) -> torch.Tensor:
    eps = 1e-6
    diff = (pred_rgb.float() - winner_rgb.float()).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
    if confidence_weighted:
        w = trusted_mask_full * conf_full
    else:
        w = trusted_mask_full
    num = (diff * w).sum()
    den = w.sum().clamp_min(eps)
    return num / den
