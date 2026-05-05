from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from library.terrain_semantic_manifest_dataset import (
    build_seam_region_maps,
    build_style_support_valid_mask,
    terrain_mask_to_occupancy,
)


@dataclass(frozen=True)
class ModelVisibleConditioningSpec:
    seam_enabled: bool
    channel_names: Tuple[str, ...]
    full_conditioning_channel_names: Tuple[str, ...]
    style_conditioning_channel_names: Tuple[str, ...]
    seam_config: Dict[str, object]
    style_ratio_config: Dict[str, object]
    terrain_mask_channel_index: int
    terrain_mask_black_is_terrain: bool
    alpha_binary_threshold: float


def build_model_visible_conditioning_spec(
    *,
    seam_enabled: bool,
    channel_names: Sequence[str],
    full_conditioning_channel_names: Sequence[str],
    style_conditioning_channel_names: Sequence[str],
    seam_config: Dict[str, object],
    style_ratio_config: Optional[Dict[str, object]],
    terrain_mask_channel_index: int,
    terrain_mask_black_is_terrain: bool,
    alpha_binary_threshold: float,
) -> ModelVisibleConditioningSpec:
    return ModelVisibleConditioningSpec(
        seam_enabled=bool(seam_enabled),
        channel_names=tuple(str(name) for name in channel_names),
        full_conditioning_channel_names=tuple(str(name) for name in full_conditioning_channel_names),
        style_conditioning_channel_names=tuple(str(name) for name in style_conditioning_channel_names),
        seam_config=dict(seam_config or {}),
        style_ratio_config=dict(style_ratio_config or {}),
        terrain_mask_channel_index=int(terrain_mask_channel_index),
        terrain_mask_black_is_terrain=bool(terrain_mask_black_is_terrain),
        alpha_binary_threshold=float(alpha_binary_threshold),
    )


def _ensure_spatial_batch(tensor: Optional[torch.Tensor], name: str) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}")
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f"{name} must have shape [C,H,W] or [B,C,H,W], got {tuple(tensor.shape)}")


def _ensure_flag_batch(tensor: Optional[torch.Tensor], name: str) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}")
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == 2:
        return tensor
    raise ValueError(f"{name} must have shape [edges] or [B,edges], got {tuple(tensor.shape)}")


def _ensure_width_batch(tensor: Optional[torch.Tensor], batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"seam_strip_width_px must be a torch.Tensor, got {type(tensor)!r}")
    value = tensor.to(device=device, dtype=dtype).reshape(-1)
    if value.numel() == 1 and batch_size > 1:
        value = value.expand(batch_size)
    if value.numel() != batch_size:
        raise ValueError(f"seam_strip_width_px batch mismatch: got {tuple(value.shape)} expected ({batch_size},)")
    return value.contiguous()


def _ensure_hw_batch(tensor: Optional[torch.Tensor], batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"original_sizes_hw must be a torch.Tensor, got {type(tensor)!r}")
    value = tensor.to(device=device, dtype=dtype)
    if value.ndim == 1:
        if value.numel() != 2:
            raise ValueError(f"original_sizes_hw must have 2 elements when 1D, got {tuple(value.shape)}")
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"original_sizes_hw must have shape [2] or [B,2], got {tuple(value.shape)}")
    if value.shape[0] == 1 and batch_size > 1:
        value = value.expand(batch_size, -1)
    if value.shape[0] != batch_size:
        raise ValueError(f"original_sizes_hw batch mismatch: got {tuple(value.shape)} expected ({batch_size}, 2)")
    return value.contiguous()


def _ensure_mask_batch(
    tensor: Optional[torch.Tensor],
    name: str,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    height: int,
    width: int,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
        value = tensor.unsqueeze(0).unsqueeze(0)
    else:
        value = _ensure_spatial_batch(tensor, name)
    if value.shape[1] != 1:
        raise ValueError(f"{name} must have a single channel, got {tuple(value.shape)}")
    value = value.to(device=device, dtype=dtype)
    if value.shape[0] == 1 and batch_size > 1:
        value = value.expand(batch_size, -1, -1, -1)
    if value.shape[0] != batch_size or tuple(value.shape[-2:]) != (height, width):
        raise ValueError(
            f"{name} shape mismatch: got {tuple(value.shape)} expected ({batch_size}, 1, {height}, {width})"
        )
    return value.contiguous()


def _build_edge_flag_maps(
    edge_defined_flags: Optional[torch.Tensor],
    *,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    edge_flags_batch = _ensure_flag_batch(edge_defined_flags, "edge_defined_flags")
    if edge_flags_batch is None:
        return None
    edge_flags_batch = edge_flags_batch.to(device=device, dtype=dtype)
    if edge_flags_batch.shape[0] == 1 and batch_size > 1:
        edge_flags_batch = edge_flags_batch.expand(batch_size, -1)
    if edge_flags_batch.shape[0] != batch_size:
        raise ValueError(
            f"edge_defined_flags batch mismatch: got {tuple(edge_flags_batch.shape)} expected ({batch_size}, 4)"
        )
    return edge_flags_batch.unsqueeze(-1).unsqueeze(-1).expand(batch_size, edge_flags_batch.shape[1], height, width).contiguous()


def _masked_l2(tensor: torch.Tensor, mask: torch.Tensor) -> float:
    denom = float(mask.sum().item()) * float(tensor.shape[1])
    if denom <= 0.0:
        return 0.0
    value = (tensor.detach().float().pow(2.0) * mask).sum().div(denom).sqrt()
    return float(value.item())


def _masked_nonzero_fraction(tensor: torch.Tensor, mask: torch.Tensor, epsilon: float) -> float:
    denom = float(mask.sum().item()) * float(tensor.shape[1])
    if denom <= 0.0:
        return 0.0
    value = ((tensor.detach().float().abs() > float(epsilon)).float() * mask).sum().div(denom)
    return float(value.item())


def _build_interior_mask(
    batch_size: int,
    height: int,
    width: int,
    interior_sizes_hw: Optional[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.ones((batch_size, 1, height, width), device=device, dtype=dtype)
    sizes = _ensure_hw_batch(interior_sizes_hw, batch_size, device=device, dtype=dtype)
    if sizes is None:
        return mask
    built = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)
    for batch_index in range(batch_size):
        target_h = max(0, min(height, int(round(float(sizes[batch_index, 0].item())))))
        target_w = max(0, min(width, int(round(float(sizes[batch_index, 1].item())))))
        top = max(0, (height - target_h) // 2)
        left = max(0, (width - target_w) // 2)
        built[batch_index, :, top : top + target_h, left : left + target_w] = 1.0
    return built.contiguous()


def _compute_grouped_conditioning_diagnostics(
    *,
    full_conditioning: torch.Tensor,
    spec: ModelVisibleConditioningSpec,
    seam_channel_count: int,
    seam_flag_count: int,
    style_channel_count: int,
    edge_band_masks: Optional[torch.Tensor],
    valid_region_mask: Optional[torch.Tensor],
    interior_sizes_hw: Optional[torch.Tensor],
    epsilon: float = 1e-6,
) -> Dict[str, object]:
    conditioning = _ensure_spatial_batch(full_conditioning, "full_conditioning")
    batch_size = int(conditioning.shape[0])
    device = conditioning.device
    dtype = conditioning.dtype
    height = int(conditioning.shape[-2])
    width = int(conditioning.shape[-1])

    valid_mask = _ensure_mask_batch(
        valid_region_mask,
        "valid_region_mask",
        batch_size,
        device=device,
        dtype=dtype,
        height=height,
        width=width,
    )
    if valid_mask is None:
        valid_mask = torch.ones((batch_size, 1, height, width), device=device, dtype=dtype)

    interior_mask = _build_interior_mask(
        batch_size,
        height,
        width,
        interior_sizes_hw,
        device=device,
        dtype=dtype,
    )
    halo_mask = (1.0 - interior_mask).clamp(0.0, 1.0)
    valid_halo_mask = (halo_mask * valid_mask).clamp(0.0, 1.0)
    invalid_padding_mask = (halo_mask * (1.0 - valid_mask)).clamp(0.0, 1.0)

    hard_band_mask = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)
    edge_band_batch = _ensure_spatial_batch(edge_band_masks, "edge_band_masks") if edge_band_masks is not None else None
    if edge_band_batch is not None:
        edge_band_batch = edge_band_batch.to(device=device, dtype=dtype)
        if edge_band_batch.shape[0] == 1 and batch_size > 1:
            edge_band_batch = edge_band_batch.expand(batch_size, -1, -1, -1)
        if edge_band_batch.shape[0] != batch_size or tuple(edge_band_batch.shape[-2:]) != (height, width):
            raise ValueError(
                "edge_band_masks shape mismatch for grouped diagnostics: "
                + f"got={tuple(edge_band_batch.shape)} expected=({batch_size}, C, {height}, {width})"
            )
        hard_band_mask = edge_band_batch.amax(dim=1, keepdim=True).clamp(0.0, 1.0)

    region_masks = {
        "interior": interior_mask,
        "halo": halo_mask,
        "valid_halo": valid_halo_mask,
        "invalid_padding": invalid_padding_mask,
        "hard_band": hard_band_mask,
        "valid_expanded_source": valid_mask,
    }

    base_count = min(len(spec.channel_names), int(conditioning.shape[1]))
    cursor = base_count
    seam_end = min(cursor + max(0, int(seam_channel_count)), int(conditioning.shape[1]))
    flag_end = min(seam_end + max(0, int(seam_flag_count)), int(conditioning.shape[1]))
    style_end = min(flag_end + max(0, int(style_channel_count)), int(conditioning.shape[1]))
    groups = {
        "semantic_base": (0, base_count),
        "seam_strip": (cursor, seam_end),
        "seam_flag": (seam_end, flag_end),
        "style": (flag_end, style_end),
    }

    group_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for group_name, (start, end) in groups.items():
        channels = max(0, end - start)
        if channels <= 0:
            group_metrics[group_name] = {
                "all": {"channels": 0.0, "l2": 0.0, "nonzero_fraction": 0.0},
                "interior": {"l2": 0.0, "nonzero_fraction": 0.0},
                "valid_halo": {"l2": 0.0, "nonzero_fraction": 0.0},
                "invalid_padding": {"l2": 0.0, "nonzero_fraction": 0.0},
                "hard_band": {"l2": 0.0, "nonzero_fraction": 0.0},
            }
            continue
        group_tensor = conditioning[:, start:end]
        metrics = {
            "all": {
                "channels": float(channels),
                "l2": _masked_l2(group_tensor, torch.ones((batch_size, 1, height, width), device=device, dtype=dtype)),
                "nonzero_fraction": _masked_nonzero_fraction(
                    group_tensor,
                    torch.ones((batch_size, 1, height, width), device=device, dtype=dtype),
                    epsilon,
                ),
            }
        }
        for region_name in ("interior", "valid_halo", "invalid_padding", "hard_band"):
            metrics[region_name] = {
                "l2": _masked_l2(group_tensor, region_masks[region_name]),
                "nonzero_fraction": _masked_nonzero_fraction(group_tensor, region_masks[region_name], epsilon),
            }
        group_metrics[group_name] = metrics

    region_metrics = {
        region_name: {
            "l2": _masked_l2(conditioning, region_mask),
            "nonzero_fraction": _masked_nonzero_fraction(conditioning, region_mask, epsilon),
            "pixels": float(region_mask.sum().item()),
        }
        for region_name, region_mask in region_masks.items()
    }

    return {
        "groups": group_metrics,
        "regions": region_metrics,
        "source_contract": "real_expanded" if region_metrics["valid_halo"]["l2"] > float(epsilon) else "interior_padded",
        "has_invalid_padding_leak": bool(region_metrics["invalid_padding"]["l2"] > float(epsilon)),
    }


def flatten_grouped_conditioning_diagnostics(report: Dict[str, object]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    if not report:
        return flat
    flat["conditioning_source_contract_real_expanded"] = 1.0 if report.get("source_contract") == "real_expanded" else 0.0
    flat["conditioning_invalid_padding_leak"] = 1.0 if bool(report.get("has_invalid_padding_leak", False)) else 0.0
    for region_name, region_values in (report.get("regions") or {}).items():
        for metric_name, metric_value in (region_values or {}).items():
            flat[f"conditioning_region_{region_name}_{metric_name}"] = float(metric_value)
    for group_name, group_values in (report.get("groups") or {}).items():
        for region_name, region_metrics in (group_values or {}).items():
            for metric_name, metric_value in (region_metrics or {}).items():
                flat[f"conditioning_group_{group_name}_{region_name}_{metric_name}"] = float(metric_value)
    return flat


def compose_sample_aware_model_visible_conditioning(
    *,
    sample: Dict[str, object],
    spec: ModelVisibleConditioningSpec,
    base_conditioning: Optional[torch.Tensor] = None,
    expanded_conditioning_mode: str = "real_expanded",
) -> Tuple[torch.Tensor, Dict[str, object]]:
    mode = str(expanded_conditioning_mode or "real_expanded").strip().lower()
    if mode not in {"real_expanded", "legacy_zero_padded"}:
        raise ValueError(f"unsupported expanded_conditioning_mode: {expanded_conditioning_mode}")

    sample_base = sample.get("conditioning_images")
    expanded_base = sample.get("expanded_conditioning_images")
    requested = base_conditioning if isinstance(base_conditioning, torch.Tensor) else None
    sample_base_shape = tuple(int(v) for v in sample_base.shape[-2:]) if isinstance(sample_base, torch.Tensor) else None
    expanded_base_shape = tuple(int(v) for v in expanded_base.shape[-2:]) if isinstance(expanded_base, torch.Tensor) else None
    requested_shape = tuple(int(v) for v in requested.shape[-2:]) if isinstance(requested, torch.Tensor) else None

    use_expanded = False
    if mode == "real_expanded":
        if expanded_base_shape is not None:
            use_expanded = True
        elif requested_shape is not None and sample_base_shape is not None and requested_shape != sample_base_shape:
            use_expanded = True

    if use_expanded:
        if requested is not None and requested_shape is not None and (sample_base_shape is None or requested_shape != sample_base_shape):
            selected_base = requested
        elif isinstance(expanded_base, torch.Tensor):
            selected_base = expanded_base
        elif requested is not None:
            selected_base = requested
        else:
            raise ValueError("real_expanded conditioning requested but expanded base conditioning is unavailable")
        seam_strip = sample.get("expanded_seam_strip_tensor")
        edge_band_masks = sample.get("expanded_edge_band_masks")
        seam_decay_maps = sample.get("expanded_seam_decay_maps")
        original_sizes_hw = sample.get("expanded_target_sizes_hw")
        alpha_target = sample.get("expanded_alpha_target")
        valid_region_mask = sample.get("valid_expanded_source_mask")
    else:
        if requested is not None:
            selected_base = requested
        elif isinstance(sample_base, torch.Tensor):
            selected_base = sample_base
        else:
            raise ValueError("conditioning_images are unavailable for sample-aware composition")
        seam_strip = sample.get("seam_strip_tensor")
        edge_band_masks = sample.get("edge_band_masks")
        seam_decay_maps = sample.get("seam_decay_maps")
        original_sizes_hw = sample.get("original_sizes_hw")
        alpha_target = sample.get("alpha_target")
        valid_region_mask = None

    selected_base_batch = _ensure_spatial_batch(selected_base, "selected_base_conditioning")
    batch_size = int(selected_base_batch.shape[0])
    device = selected_base_batch.device
    dtype = selected_base_batch.dtype
    height = int(selected_base_batch.shape[-2])
    width = int(selected_base_batch.shape[-1])

    edge_flag_maps = sample.get("edge_flag_maps")
    edge_flag_maps_batch = _ensure_spatial_batch(edge_flag_maps, "edge_flag_maps") if isinstance(edge_flag_maps, torch.Tensor) else None
    if edge_flag_maps_batch is None or tuple(edge_flag_maps_batch.shape[-2:]) != (height, width):
        edge_flag_maps = _build_edge_flag_maps(
            sample.get("edge_defined_flags"),
            batch_size=batch_size,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )

    full_conditioning, details = compose_model_visible_conditioning(
        base_conditioning=selected_base,
        spec=spec,
        seam_strip=seam_strip,
        edge_defined_flags=sample.get("edge_defined_flags"),
        edge_flag_maps=edge_flag_maps,
        edge_band_masks=edge_band_masks,
        seam_decay_maps=seam_decay_maps,
        seam_strip_width_px=sample.get("seam_strip_width_px"),
        original_sizes_hw=original_sizes_hw,
        alpha_target=alpha_target,
        valid_region_mask=valid_region_mask,
    )

    seam_channel_count = int(details.get("seam_visible").shape[1]) if isinstance(details.get("seam_visible"), torch.Tensor) else 0
    seam_flag_count = int(edge_flag_maps.shape[1]) if isinstance(edge_flag_maps, torch.Tensor) and edge_flag_maps.ndim == 4 else (int(edge_flag_maps.shape[0]) if isinstance(edge_flag_maps, torch.Tensor) and edge_flag_maps.ndim == 3 else 0)
    style_channel_count = int(details.get("style_conditioning").shape[1]) if isinstance(details.get("style_conditioning"), torch.Tensor) else 0
    grouped_diagnostics = _compute_grouped_conditioning_diagnostics(
        full_conditioning=full_conditioning,
        spec=spec,
        seam_channel_count=seam_channel_count,
        seam_flag_count=seam_flag_count,
        style_channel_count=style_channel_count,
        edge_band_masks=edge_band_masks,
        valid_region_mask=valid_region_mask,
        interior_sizes_hw=sample.get("target_sizes_hw"),
    )
    details["grouped_diagnostics"] = grouped_diagnostics
    details["source_contract"] = grouped_diagnostics["source_contract"]
    details["has_invalid_padding_leak"] = grouped_diagnostics["has_invalid_padding_leak"]
    details["used_expanded_context"] = use_expanded
    details["channel_names"] = (
        tuple(spec.full_conditioning_channel_names)
        if int(full_conditioning.shape[1] if full_conditioning.ndim == 4 else full_conditioning.shape[0]) == len(spec.full_conditioning_channel_names)
        else tuple(spec.channel_names)
    )
    return full_conditioning, details


def compose_model_visible_conditioning(
    *,
    base_conditioning: torch.Tensor,
    spec: ModelVisibleConditioningSpec,
    seam_strip: Optional[torch.Tensor] = None,
    edge_defined_flags: Optional[torch.Tensor] = None,
    edge_flag_maps: Optional[torch.Tensor] = None,
    edge_band_masks: Optional[torch.Tensor] = None,
    seam_decay_maps: Optional[torch.Tensor] = None,
    seam_strip_width_px: Optional[torch.Tensor] = None,
    original_sizes_hw: Optional[torch.Tensor] = None,
    alpha_target: Optional[torch.Tensor] = None,
    valid_region_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    conditioning = _ensure_spatial_batch(base_conditioning, "base_conditioning")
    squeeze_output = base_conditioning.ndim == 3
    batch_size = int(conditioning.shape[0])
    device = conditioning.device
    dtype = conditioning.dtype
    valid_region_mask_batch = _ensure_mask_batch(
        valid_region_mask,
        "valid_region_mask",
        batch_size,
        device=device,
        dtype=dtype,
        height=int(conditioning.shape[-2]),
        width=int(conditioning.shape[-1]),
    )

    style_conditioning = conditioning.new_zeros(
        batch_size,
        len(spec.style_conditioning_channel_names),
        conditioning.shape[-2],
        conditioning.shape[-1],
    )
    details: Dict[str, torch.Tensor] = {
        "style_conditioning": style_conditioning,
    }
    if valid_region_mask_batch is not None:
        details["valid_region_mask"] = valid_region_mask_batch

    seam_strip_batch = _ensure_spatial_batch(seam_strip, "seam_strip") if seam_strip is not None else None
    edge_flags_batch = _ensure_flag_batch(edge_defined_flags, "edge_defined_flags") if edge_defined_flags is not None else None
    edge_flag_maps_batch = _ensure_spatial_batch(edge_flag_maps, "edge_flag_maps") if edge_flag_maps is not None else None

    if (
        not spec.seam_enabled
        or seam_strip_batch is None
        or edge_flags_batch is None
        or edge_flag_maps_batch is None
    ):
        output = torch.cat([conditioning, style_conditioning], dim=1) if style_conditioning.shape[1] > 0 else conditioning
        if squeeze_output:
            output = output.squeeze(0)
        return output, details

    seam_strip_batch = seam_strip_batch.to(device=device, dtype=dtype)
    edge_flags_batch = edge_flags_batch.to(device=device, dtype=dtype)
    edge_flag_maps_batch = edge_flag_maps_batch.to(device=device, dtype=dtype)
    if seam_strip_batch.shape[0] != batch_size or edge_flags_batch.shape[0] != batch_size or edge_flag_maps_batch.shape[0] != batch_size:
        raise ValueError(
            "conditioning batch mismatch: "
            + f"conditioning={tuple(conditioning.shape)} seam_strip={tuple(seam_strip_batch.shape)} "
            + f"edge_defined_flags={tuple(edge_flags_batch.shape)} edge_flag_maps={tuple(edge_flag_maps_batch.shape)}"
        )

    seam_gate = edge_flags_batch.repeat_interleave(4, dim=1).unsqueeze(-1).unsqueeze(-1)
    seam_visible = seam_strip_batch * seam_gate
    full_conditioning_parts = [conditioning, seam_visible, edge_flag_maps_batch]
    details["seam_strip"] = seam_strip_batch
    details["seam_visible"] = seam_visible
    details["edge_defined_flags"] = edge_flags_batch

    edge_band_masks_batch = _ensure_spatial_batch(edge_band_masks, "edge_band_masks") if edge_band_masks is not None else None
    seam_decay_maps_batch = _ensure_spatial_batch(seam_decay_maps, "seam_decay_maps") if seam_decay_maps is not None else None
    seam_strip_width_batch = _ensure_width_batch(seam_strip_width_px, batch_size, device=device, dtype=dtype)
    original_sizes_batch = _ensure_hw_batch(original_sizes_hw, batch_size, device=device, dtype=dtype)

    if (
        style_conditioning.shape[1] > 0
        and edge_band_masks_batch is not None
        and seam_decay_maps_batch is not None
        and seam_strip_width_batch is not None
    ):
        continuation_valid_mask = torch.ones_like(conditioning[:, :1])
        if spec.terrain_mask_channel_index >= 0:
            continuation_valid_mask = terrain_mask_to_occupancy(
                conditioning[:, spec.terrain_mask_channel_index : spec.terrain_mask_channel_index + 1],
                spec.terrain_mask_black_is_terrain,
            )
            continuation_valid_mask = (
                continuation_valid_mask >= float(spec.alpha_binary_threshold)
            ).to(dtype=dtype)
        if valid_region_mask_batch is not None:
            continuation_valid_mask = continuation_valid_mask * valid_region_mask_batch

        style_support_valid_mask = build_style_support_valid_mask(
            conditioning_images=conditioning,
            alpha_target=alpha_target,
            halo_px=0,
            alpha_binary_threshold=float(spec.alpha_binary_threshold),
            terrain_mask_channel_index=int(spec.terrain_mask_channel_index),
            terrain_mask_black_is_terrain=bool(spec.terrain_mask_black_is_terrain),
            style_ratio_config=spec.style_ratio_config,
        ).to(device=device, dtype=dtype)
        if valid_region_mask_batch is not None:
            style_support_valid_mask = style_support_valid_mask * valid_region_mask_batch

        seam_maps = build_seam_region_maps(
            edge_band_masks=edge_band_masks_batch.to(device=device, dtype=dtype),
            seam_decay_maps=seam_decay_maps_batch.to(device=device, dtype=dtype),
            edge_defined_flags=edge_flags_batch,
            seam_strip_width_px=seam_strip_width_batch,
            supervision_mask=(valid_region_mask_batch if valid_region_mask_batch is not None else torch.ones_like(conditioning[:, :1])),
            seam_config=spec.seam_config,
            source_sizes_hw=original_sizes_batch,
            continuation_valid_mask=continuation_valid_mask,
            style_support_valid_mask=style_support_valid_mask,
            style_ratio_config=spec.style_ratio_config,
        )
        ramp_mask = seam_maps["controlnet_style_effect_mask"].to(dtype=dtype)
        q_per_edge = seam_maps["soft_field_q_per_edge"].to(dtype=dtype) * ramp_mask
        q_interior = seam_maps["soft_field_q_interior"].to(dtype=dtype) * ramp_mask
        influence_c = seam_maps["soft_field_influence_c"].to(dtype=dtype) * ramp_mask
        near_band_mask = seam_maps["near_band_mask"].to(dtype=dtype)
        overlap_band_mask = seam_maps["overlap_band_mask"].to(dtype=dtype)

        edge_rgb_means = []
        edge_count = int(seam_strip_batch.shape[1] // 4)
        for edge_index in range(edge_count):
            edge_slice = seam_strip_batch[:, edge_index * 4 : edge_index * 4 + 4]
            edge_rgb = edge_slice[:, :3]
            edge_alpha = edge_slice[:, 3:4].clamp(0.0, 1.0)
            support = torch.maximum(edge_alpha, edge_rgb.detach().abs().amax(dim=1, keepdim=True))
            support_sum = support.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            edge_rgb_means.append((edge_rgb * support).sum(dim=(-2, -1), keepdim=True) / support_sum)
        edge_rgb_mean_tensor = torch.stack(edge_rgb_means, dim=1)

        blended_lowfreq = conditioning.new_zeros(batch_size, 3, conditioning.shape[-2], conditioning.shape[-1])
        for edge_index in range(edge_rgb_mean_tensor.shape[1]):
            blended_lowfreq = blended_lowfreq + (
                q_per_edge[:, edge_index : edge_index + 1] * edge_rgb_mean_tensor[:, edge_index]
            )

        style_conditioning = torch.cat(
            [
                q_per_edge,
                q_interior,
                influence_c,
                near_band_mask,
                overlap_band_mask,
                ramp_mask,
                blended_lowfreq * ramp_mask,
            ],
            dim=1,
        )
        full_conditioning_parts.append(style_conditioning)
        details["style_conditioning"] = style_conditioning
        details["seam_maps"] = seam_maps

    if valid_region_mask_batch is not None:
        full_conditioning_parts = [part * valid_region_mask_batch for part in full_conditioning_parts]
        if "seam_visible" in details:
            details["seam_visible"] = details["seam_visible"] * valid_region_mask_batch
        if "style_conditioning" in details:
            details["style_conditioning"] = details["style_conditioning"] * valid_region_mask_batch

    output = torch.cat(full_conditioning_parts, dim=1)
    if squeeze_output:
        output = output.squeeze(0)
    return output, details