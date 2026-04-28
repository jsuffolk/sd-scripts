# some parts are modified from Diffusers library (Apache License 2.0)

import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library import sdxl_original_unet
from library.sdxl_model_util import convert_sdxl_unet_state_dict_to_diffusers, convert_diffusers_unet_state_dict_to_sdxl


SEAM_EDGE_SLICES = {
    "north": (0, 4),
    "south": (4, 8),
    "east": (8, 12),
    "west": (12, 16),
}
SEAM_EDGE_ORDER = tuple(SEAM_EDGE_SLICES.keys())
SEAM_ADAPTER_LEGACY_INPUT_CHANNELS = 6
SEAM_ADAPTER_PER_EDGE_INPUT_CHANNELS = 9

_SEAM_SOBEL_X_KERNEL = torch.tensor(
    [[-0.25, 0.0, 0.25], [-0.5, 0.0, 0.5], [-0.25, 0.0, 0.25]], dtype=torch.float32
).view(1, 1, 3, 3)
_SEAM_SOBEL_Y_KERNEL = torch.tensor(
    [[-0.25, -0.5, -0.25], [0.0, 0.0, 0.0], [0.25, 0.5, 0.25]], dtype=torch.float32
).view(1, 1, 3, 3)


def _zero_module(module: nn.Module) -> None:
    for name, parameter in module.named_parameters(recurse=False):
        if parameter is not None and (name.endswith("weight") or name.endswith("bias")):
            nn.init.zeros_(parameter)


def _edge_distance_fraction(edge_name: str, height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    def normalized_positions(size: int, reverse: bool = False) -> torch.Tensor:
        if size <= 0:
            raise ValueError("edge distance size must be positive")
        base = torch.arange(size, device=device, dtype=dtype) / float(size)
        if reverse:
            base = torch.flip(base, dims=[0])
        return base

    if edge_name == "north":
        base = normalized_positions(height).view(1, 1, height, 1)
        return base.expand(1, 1, height, width)
    if edge_name == "south":
        base = normalized_positions(height, reverse=True).view(1, 1, height, 1)
        return base.expand(1, 1, height, width)
    if edge_name == "east":
        base = normalized_positions(width, reverse=True).view(1, 1, 1, width)
        return base.expand(1, 1, height, width)
    if edge_name == "west":
        base = normalized_positions(width).view(1, 1, 1, width)
        return base.expand(1, 1, height, width)
    raise ValueError(f"unsupported seam edge: {edge_name}")


def _expand_edge_strip_across_band(
    edge_name: str,
    edge_tensor: torch.Tensor,
    bounds: tuple[int, int, int, int],
    *,
    target_height: int,
    target_width: int,
) -> Optional[torch.Tensor]:
    row_start, row_end, col_start, col_end = bounds
    crop = edge_tensor[:, :, row_start:row_end, col_start:col_end]
    if crop.shape[-2] == 0 or crop.shape[-1] == 0:
        return None

    eps = torch.finfo(crop.dtype).eps
    if edge_name in {"north", "south"}:
        tangential = F.interpolate(crop, size=(crop.shape[-2], target_width), mode="bilinear", align_corners=False)
        support = torch.maximum(tangential[:, 3:4].abs(), tangential.detach().abs().amax(dim=1, keepdim=True))
        support_sum = support.sum(dim=2, keepdim=True).clamp_min(eps)
        rgb_line = (tangential[:, :3] * support).sum(dim=2, keepdim=True) / support_sum
        alpha_line = tangential[:, 3:4].amax(dim=2, keepdim=True)
        edge_profile = torch.cat([rgb_line, alpha_line], dim=1)
        return edge_profile.expand(-1, -1, target_height, -1)

    tangential = F.interpolate(crop, size=(target_height, crop.shape[-1]), mode="bilinear", align_corners=False)
    support = torch.maximum(tangential[:, 3:4].abs(), tangential.detach().abs().amax(dim=1, keepdim=True))
    support_sum = support.sum(dim=3, keepdim=True).clamp_min(eps)
    rgb_line = (tangential[:, :3] * support).sum(dim=3, keepdim=True) / support_sum
    alpha_line = tangential[:, 3:4].amax(dim=3, keepdim=True)
    edge_profile = torch.cat([rgb_line, alpha_line], dim=1)
    return edge_profile.expand(-1, -1, -1, target_width)


def _edge_support_bounds(edge_tensor: torch.Tensor, eps: float = 1e-6) -> Optional[tuple[int, int, int, int]]:
    activity = edge_tensor.detach().abs().amax(dim=(0, 1))
    row_idx = torch.nonzero(activity.amax(dim=1) > eps, as_tuple=False).flatten()
    col_idx = torch.nonzero(activity.amax(dim=0) > eps, as_tuple=False).flatten()
    if row_idx.numel() == 0 or col_idx.numel() == 0:
        return None
    return int(row_idx[0].item()), int(row_idx[-1].item()) + 1, int(col_idx[0].item()), int(col_idx[-1].item()) + 1


def _compute_projected_edge_sobel(edge_rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if edge_rgb.ndim != 4 or edge_rgb.shape[1] != 3:
        raise ValueError(f"expected RGB edge tensor with shape [batch, 3, height, width], got {tuple(edge_rgb.shape)}")
    luma = (
        (0.299 * edge_rgb[:, 0:1])
        + (0.587 * edge_rgb[:, 1:2])
        + (0.114 * edge_rgb[:, 2:3])
    )
    sobel_x = _SEAM_SOBEL_X_KERNEL.to(device=edge_rgb.device, dtype=edge_rgb.dtype)
    sobel_y = _SEAM_SOBEL_Y_KERNEL.to(device=edge_rgb.device, dtype=edge_rgb.dtype)
    return F.conv2d(luma, sobel_x, padding=1), F.conv2d(luma, sobel_y, padding=1)


def _build_legacy_seam_local_adapter_maps_from_conditioning(
    cond_image: torch.Tensor,
    seam_conditioning_offset: int,
    band_px: int,
) -> Dict[str, torch.Tensor]:
    batch_size, _, height, width = cond_image.shape
    device = cond_image.device
    dtype = cond_image.dtype

    def zeros(channels: int) -> torch.Tensor:
        return torch.zeros((batch_size, channels, height, width), device=device, dtype=dtype)

    adapter_input = zeros(6)
    active_mask = zeros(1)
    invalid_active_mask = zeros(1)

    if seam_conditioning_offset < 0 or band_px <= 0 or cond_image.shape[1] < seam_conditioning_offset + 20:
        return {
            "adapter_input": adapter_input,
            "active_mask": active_mask,
            "invalid_active_mask": invalid_active_mask,
            "edge_valid_count": torch.zeros((batch_size,), device=device, dtype=dtype),
        }

    seam_visible = cond_image[:, seam_conditioning_offset : seam_conditioning_offset + 16]
    edge_flag_maps = cond_image[:, seam_conditioning_offset + 16 : seam_conditioning_offset + 20]
    edge_valid = (edge_flag_maps.mean(dim=(2, 3)) > 0.5).to(dtype=dtype)

    edge_bounds = {
        edge_name: _edge_support_bounds(seam_visible[:, start_channel:end_channel])
        for edge_name, (start_channel, end_channel) in SEAM_EDGE_SLICES.items()
    }
    row_margin_candidates = []
    col_margin_candidates = []
    for edge_name, bounds in edge_bounds.items():
        if bounds is None:
            continue
        row_start, row_end, col_start, col_end = bounds
        if edge_name == "north":
            row_margin_candidates.append(row_start)
            col_margin_candidates.extend([col_start, width - col_end])
        elif edge_name == "south":
            row_margin_candidates.append(height - row_end)
            col_margin_candidates.extend([col_start, width - col_end])
        elif edge_name == "east":
            row_margin_candidates.extend([row_start, height - row_end])
            col_margin_candidates.append(width - col_end)
        else:
            row_margin_candidates.extend([row_start, height - row_end])
            col_margin_candidates.append(col_start)

    if not row_margin_candidates or not col_margin_candidates:
        return {
            "adapter_input": adapter_input,
            "active_mask": active_mask,
            "invalid_active_mask": invalid_active_mask,
            "edge_valid_count": edge_valid.sum(dim=1),
        }

    row_margin = max(0, min(int(candidate) for candidate in row_margin_candidates))
    col_margin = max(0, min(int(candidate) for candidate in col_margin_candidates))
    interior_row_start = row_margin
    interior_row_end = max(interior_row_start, height - row_margin)
    interior_col_start = col_margin
    interior_col_end = max(interior_col_start, width - col_margin)

    rgb_accum = zeros(3)
    alpha_accum = zeros(1)
    valid_accum = zeros(1)
    distance_accum = zeros(1)
    valid_count = zeros(1)
    invalid_count = zeros(1)

    for edge_index, edge_name in enumerate(SEAM_EDGE_ORDER):
        start_channel, end_channel = SEAM_EDGE_SLICES[edge_name]
        edge_tensor = seam_visible[:, start_channel:end_channel]
        if edge_name == "north":
            target_row_start = interior_row_start
            target_row_end = min(interior_row_end, interior_row_start + band_px)
            target_col_start = interior_col_start
            target_col_end = interior_col_end
        elif edge_name == "south":
            target_row_end = interior_row_end
            target_row_start = max(interior_row_start, interior_row_end - band_px)
            target_col_start = interior_col_start
            target_col_end = interior_col_end
        elif edge_name == "east":
            target_row_start = interior_row_start
            target_row_end = interior_row_end
            target_col_end = interior_col_end
            target_col_start = max(interior_col_start, interior_col_end - band_px)
        else:
            target_row_start = interior_row_start
            target_row_end = interior_row_end
            target_col_start = interior_col_start
            target_col_end = min(interior_col_end, interior_col_start + band_px)

        target_height = max(0, target_row_end - target_row_start)
        target_width = max(0, target_col_end - target_col_start)
        if target_height == 0 or target_width == 0:
            continue

        distance_fraction = _edge_distance_fraction(edge_name, target_height, target_width, device=device, dtype=dtype)
        projection_weight = (1.0 - distance_fraction).pow(2.0)

        valid_gate = edge_valid[:, edge_index].view(batch_size, 1, 1, 1)
        invalid_gate = 1.0 - valid_gate
        active = torch.ones((batch_size, 1, target_height, target_width), device=device, dtype=dtype)

        row_slice = slice(target_row_start, target_row_end)
        col_slice = slice(target_col_start, target_col_end)
        valid_count[:, :, row_slice, col_slice] += active * valid_gate
        invalid_count[:, :, row_slice, col_slice] += active * invalid_gate

        bounds = edge_bounds[edge_name]
        if bounds is None:
            continue

        row_start, row_end, col_start, col_end = bounds
        projected = _expand_edge_strip_across_band(
            edge_name,
            edge_tensor,
            (row_start, row_end, col_start, col_end),
            target_height=target_height,
            target_width=target_width,
        )
        if projected is None:
            continue
        rgb_accum[:, :, row_slice, col_slice] += projected[:, :3] * projection_weight * valid_gate
        alpha_accum[:, :, row_slice, col_slice] += projected[:, 3:4] * projection_weight * valid_gate
        valid_accum[:, :, row_slice, col_slice] += active * valid_gate
        distance_accum[:, :, row_slice, col_slice] += distance_fraction * valid_gate

    overlap_mask = (valid_count + invalid_count) > 1.0
    if overlap_mask.any():
        rgb_accum = rgb_accum.masked_fill(overlap_mask.expand_as(rgb_accum), 0.0)
        alpha_accum = alpha_accum.masked_fill(overlap_mask, 0.0)
        valid_accum = valid_accum.masked_fill(overlap_mask, 0.0)
        distance_accum = distance_accum.masked_fill(overlap_mask, 0.0)
        valid_count = valid_count.masked_fill(overlap_mask, 0.0)
        invalid_count = invalid_count.masked_fill(overlap_mask, 0.0)

    adapter_input = torch.cat([rgb_accum, alpha_accum, valid_accum.clamp_(0.0, 1.0), distance_accum.clamp_(0.0, 1.0)], dim=1)
    active_mask = (valid_count > 0).to(dtype=dtype)
    invalid_active_mask = (invalid_count > 0).to(dtype=dtype)
    return {
        "adapter_input": adapter_input,
        "active_mask": active_mask,
        "invalid_active_mask": invalid_active_mask,
        "edge_valid_count": edge_valid.sum(dim=1),
    }


def _infer_default_halo_width(edge_bounds: Dict[str, Optional[Tuple[int, int, int, int]]]) -> int:
    candidates: List[int] = []
    for edge_name, bounds in edge_bounds.items():
        if bounds is None:
            continue
        row_start, row_end, col_start, col_end = bounds
        if edge_name in {"north", "south"}:
            candidates.append(max(0, row_end - row_start))
        else:
            candidates.append(max(0, col_end - col_start))
    return max(0, min(candidates)) if candidates else 0


def _resolve_edge_projection_geometry(
    edge_name: str,
    bounds: Optional[Tuple[int, int, int, int]],
    *,
    height: int,
    width: int,
    default_halo_width: int,
    band_px: int,
) -> Optional[Dict[str, object]]:
    halo_width = int(max(0, default_halo_width))
    if bounds is not None:
        row_start, row_end, col_start, col_end = bounds
        if edge_name in {"north", "south"}:
            halo_width = max(0, row_end - row_start)
        else:
            halo_width = max(0, col_end - col_start)
    else:
        if halo_width <= 0:
            return None
        if edge_name == "north":
            row_start, row_end, col_start, col_end = 0, min(height, halo_width), 0, width
        elif edge_name == "south":
            row_start, row_end, col_start, col_end = max(0, height - halo_width), height, 0, width
        elif edge_name == "east":
            row_start, row_end, col_start, col_end = 0, height, max(0, width - halo_width), width
        else:
            row_start, row_end, col_start, col_end = 0, height, 0, min(width, halo_width)

    if halo_width <= 0:
        return None

    if edge_name == "north":
        extrusion_row_start = row_end
        extrusion_row_end = min(height, row_end + max(0, band_px))
        return {
            "halo_row_slice": slice(row_start, row_end),
            "halo_col_slice": slice(0, width),
            "extrusion_row_slice": slice(extrusion_row_start, extrusion_row_end),
            "extrusion_col_slice": slice(0, width),
            "boundary_index": row_end - 1,
            "support_start": col_start,
            "support_end": col_end,
            "halo_width": halo_width,
            "orientation": "horizontal",
        }
    if edge_name == "south":
        extrusion_row_start = max(0, row_start - max(0, band_px))
        extrusion_row_end = row_start
        return {
            "halo_row_slice": slice(row_start, row_end),
            "halo_col_slice": slice(0, width),
            "extrusion_row_slice": slice(extrusion_row_start, extrusion_row_end),
            "extrusion_col_slice": slice(0, width),
            "boundary_index": row_start,
            "support_start": col_start,
            "support_end": col_end,
            "halo_width": halo_width,
            "orientation": "horizontal",
        }
    if edge_name == "east":
        extrusion_col_start = max(0, col_start - max(0, band_px))
        extrusion_col_end = col_start
        return {
            "halo_row_slice": slice(0, height),
            "halo_col_slice": slice(col_start, col_end),
            "extrusion_row_slice": slice(0, height),
            "extrusion_col_slice": slice(extrusion_col_start, extrusion_col_end),
            "boundary_index": col_start,
            "support_start": row_start,
            "support_end": row_end,
            "halo_width": halo_width,
            "orientation": "vertical",
        }
    extrusion_col_start = col_end
    extrusion_col_end = min(width, col_end + max(0, band_px))
    return {
        "halo_row_slice": slice(0, height),
        "halo_col_slice": slice(col_start, col_end),
        "extrusion_row_slice": slice(0, height),
        "extrusion_col_slice": slice(extrusion_col_start, extrusion_col_end),
        "boundary_index": col_end - 1,
        "support_start": row_start,
        "support_end": row_end,
        "halo_width": halo_width,
        "orientation": "vertical",
    }


def _make_support_mask(
    batch_size: int,
    *,
    length: int,
    support_start: int,
    support_end: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.zeros((batch_size, 1, length), device=device, dtype=dtype)
    if support_end > support_start:
        mask[:, :, support_start:support_end] = 1.0
    return mask


def _fill_per_edge_projection(
    edge_name: str,
    *,
    edge_tensor: torch.Tensor,
    geometry: Dict[str, object],
    projected_rgb: torch.Tensor,
    projected_alpha: torch.Tensor,
    projected_sobel_x: torch.Tensor,
    projected_sobel_y: torch.Tensor,
    distance_to_seam: torch.Tensor,
    edge_valid_map: torch.Tensor,
    active_band_mask: torch.Tensor,
    valid_gate: torch.Tensor,
    band_px: int,
) -> torch.Tensor:
    batch_size, _, height, width = edge_tensor.shape
    device = edge_tensor.device
    dtype = edge_tensor.dtype

    halo_row_slice = geometry["halo_row_slice"]
    halo_col_slice = geometry["halo_col_slice"]
    extrusion_row_slice = geometry["extrusion_row_slice"]
    extrusion_col_slice = geometry["extrusion_col_slice"]
    support_start = int(geometry["support_start"])
    support_end = int(geometry["support_end"])
    orientation = str(geometry["orientation"])
    edge_sobel_x, edge_sobel_y = _compute_projected_edge_sobel(edge_tensor[:, :3])

    invalid_mask = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)
    if orientation == "horizontal":
        support_line = _make_support_mask(
            batch_size,
            length=width,
            support_start=support_start,
            support_end=support_end,
            device=device,
            dtype=dtype,
        )
        halo_height = max(0, halo_row_slice.stop - halo_row_slice.start)
        extrusion_height = max(0, extrusion_row_slice.stop - extrusion_row_slice.start)
        halo_mask = support_line.unsqueeze(2).expand(batch_size, 1, halo_height, width)
        active_band_mask[:, :, halo_row_slice, halo_col_slice] = halo_mask * valid_gate
        edge_valid_map[:, :, halo_row_slice, halo_col_slice] = halo_mask * valid_gate
        invalid_mask[:, :, halo_row_slice, halo_col_slice] = halo_mask * (1.0 - valid_gate)
        if halo_height > 0:
            projected_rgb[:, :, halo_row_slice, halo_col_slice] = edge_tensor[:, :3, halo_row_slice, halo_col_slice] * valid_gate
            projected_alpha[:, :, halo_row_slice, halo_col_slice] = edge_tensor[:, 3:4, halo_row_slice, halo_col_slice] * valid_gate
            projected_sobel_x[:, :, halo_row_slice, halo_col_slice] = edge_sobel_x[:, :, halo_row_slice, halo_col_slice] * valid_gate
            projected_sobel_y[:, :, halo_row_slice, halo_col_slice] = edge_sobel_y[:, :, halo_row_slice, halo_col_slice] * valid_gate
        if extrusion_height > 0:
            boundary = edge_tensor[:, :, int(geometry["boundary_index"]): int(geometry["boundary_index"]) + 1, :] * valid_gate
            boundary_sobel_x = edge_sobel_x[:, :, int(geometry["boundary_index"]): int(geometry["boundary_index"]) + 1, :] * valid_gate
            boundary_sobel_y = edge_sobel_y[:, :, int(geometry["boundary_index"]): int(geometry["boundary_index"]) + 1, :] * valid_gate
            extrusion_mask = support_line.unsqueeze(2).expand(batch_size, 1, extrusion_height, width)
            projected_rgb[:, :, extrusion_row_slice, extrusion_col_slice] = boundary[:, :3].expand(-1, -1, extrusion_height, -1)
            projected_alpha[:, :, extrusion_row_slice, extrusion_col_slice] = boundary[:, 3:4].expand(-1, -1, extrusion_height, -1)
            projected_sobel_x[:, :, extrusion_row_slice, extrusion_col_slice] = boundary_sobel_x.expand(-1, -1, extrusion_height, -1)
            projected_sobel_y[:, :, extrusion_row_slice, extrusion_col_slice] = boundary_sobel_y.expand(-1, -1, extrusion_height, -1)
            active_band_mask[:, :, extrusion_row_slice, extrusion_col_slice] = extrusion_mask * valid_gate
            edge_valid_map[:, :, extrusion_row_slice, extrusion_col_slice] = extrusion_mask * valid_gate
            invalid_mask[:, :, extrusion_row_slice, extrusion_col_slice] = extrusion_mask * (1.0 - valid_gate)
            ramp = (torch.arange(extrusion_height, device=device, dtype=dtype) / float(max(1, band_px))).view(1, 1, extrusion_height, 1)
            if edge_name == "south":
                ramp = torch.flip(ramp, dims=[2])
            distance_to_seam[:, :, extrusion_row_slice, extrusion_col_slice] = ramp.expand(batch_size, 1, extrusion_height, width) * extrusion_mask * valid_gate
        return invalid_mask

    support_line = _make_support_mask(
        batch_size,
        length=height,
        support_start=support_start,
        support_end=support_end,
        device=device,
        dtype=dtype,
    )
    halo_width = max(0, halo_col_slice.stop - halo_col_slice.start)
    extrusion_width = max(0, extrusion_col_slice.stop - extrusion_col_slice.start)
    halo_mask = support_line.unsqueeze(3).expand(batch_size, 1, height, halo_width)
    active_band_mask[:, :, halo_row_slice, halo_col_slice] = halo_mask * valid_gate
    edge_valid_map[:, :, halo_row_slice, halo_col_slice] = halo_mask * valid_gate
    invalid_mask[:, :, halo_row_slice, halo_col_slice] = halo_mask * (1.0 - valid_gate)
    if halo_width > 0:
        projected_rgb[:, :, halo_row_slice, halo_col_slice] = edge_tensor[:, :3, halo_row_slice, halo_col_slice] * valid_gate
        projected_alpha[:, :, halo_row_slice, halo_col_slice] = edge_tensor[:, 3:4, halo_row_slice, halo_col_slice] * valid_gate
        projected_sobel_x[:, :, halo_row_slice, halo_col_slice] = edge_sobel_x[:, :, halo_row_slice, halo_col_slice] * valid_gate
        projected_sobel_y[:, :, halo_row_slice, halo_col_slice] = edge_sobel_y[:, :, halo_row_slice, halo_col_slice] * valid_gate
    if extrusion_width > 0:
        boundary = edge_tensor[:, :, :, int(geometry["boundary_index"]): int(geometry["boundary_index"]) + 1] * valid_gate
        boundary_sobel_x = edge_sobel_x[:, :, :, int(geometry["boundary_index"]): int(geometry["boundary_index"]) + 1] * valid_gate
        boundary_sobel_y = edge_sobel_y[:, :, :, int(geometry["boundary_index"]): int(geometry["boundary_index"]) + 1] * valid_gate
        extrusion_mask = support_line.unsqueeze(3).expand(batch_size, 1, height, extrusion_width)
        projected_rgb[:, :, extrusion_row_slice, extrusion_col_slice] = boundary[:, :3].expand(-1, -1, -1, extrusion_width)
        projected_alpha[:, :, extrusion_row_slice, extrusion_col_slice] = boundary[:, 3:4].expand(-1, -1, -1, extrusion_width)
        projected_sobel_x[:, :, extrusion_row_slice, extrusion_col_slice] = boundary_sobel_x.expand(-1, -1, -1, extrusion_width)
        projected_sobel_y[:, :, extrusion_row_slice, extrusion_col_slice] = boundary_sobel_y.expand(-1, -1, -1, extrusion_width)
        active_band_mask[:, :, extrusion_row_slice, extrusion_col_slice] = extrusion_mask * valid_gate
        edge_valid_map[:, :, extrusion_row_slice, extrusion_col_slice] = extrusion_mask * valid_gate
        invalid_mask[:, :, extrusion_row_slice, extrusion_col_slice] = extrusion_mask * (1.0 - valid_gate)
        ramp = (torch.arange(extrusion_width, device=device, dtype=dtype) / float(max(1, band_px))).view(1, 1, 1, extrusion_width)
        if edge_name == "east":
            ramp = torch.flip(ramp, dims=[3])
        distance_to_seam[:, :, extrusion_row_slice, extrusion_col_slice] = ramp.expand(batch_size, 1, height, extrusion_width) * extrusion_mask * valid_gate
    return invalid_mask


def _build_per_edge_seam_local_adapter_maps_from_conditioning(
    cond_image: torch.Tensor,
    seam_conditioning_offset: int,
    band_px: int,
) -> Dict[str, torch.Tensor]:
    batch_size, _, height, width = cond_image.shape
    device = cond_image.device
    dtype = cond_image.dtype

    def zeros(channels: int) -> torch.Tensor:
        return torch.zeros((batch_size, channels, height, width), device=device, dtype=dtype)

    def zeros_per_edge(channels: int) -> torch.Tensor:
        return torch.zeros((batch_size, len(SEAM_EDGE_ORDER), channels, height, width), device=device, dtype=dtype)

    per_edge_input = zeros_per_edge(SEAM_ADAPTER_PER_EDGE_INPUT_CHANNELS)
    per_edge_active_mask = zeros_per_edge(1)
    per_edge_invalid_mask = zeros_per_edge(1)
    combined_active_mask = zeros(1)

    if seam_conditioning_offset < 0 or band_px <= 0 or cond_image.shape[1] < seam_conditioning_offset + 20:
        return {
            "adapter_input": per_edge_input,
            "active_mask": per_edge_active_mask,
            "invalid_active_mask": per_edge_invalid_mask,
            "combined_active_mask": combined_active_mask,
            "edge_valid_count": torch.zeros((batch_size,), device=device, dtype=dtype),
            "edge_valid_flags": torch.zeros((batch_size, len(SEAM_EDGE_ORDER)), device=device, dtype=dtype),
            "adapter_mode": "per_edge",
        }

    seam_visible = cond_image[:, seam_conditioning_offset : seam_conditioning_offset + 16]
    edge_flag_maps = cond_image[:, seam_conditioning_offset + 16 : seam_conditioning_offset + 20]
    edge_valid = (edge_flag_maps.mean(dim=(2, 3)) > 0.5).to(dtype=dtype)
    edge_bounds = {
        edge_name: _edge_support_bounds(seam_visible[:, start_channel:end_channel])
        for edge_name, (start_channel, end_channel) in SEAM_EDGE_SLICES.items()
    }
    default_halo_width = _infer_default_halo_width(edge_bounds)

    for edge_index, edge_name in enumerate(SEAM_EDGE_ORDER):
        geometry = _resolve_edge_projection_geometry(
            edge_name,
            edge_bounds.get(edge_name),
            height=height,
            width=width,
            default_halo_width=default_halo_width,
            band_px=band_px,
        )
        if geometry is None:
            continue
        start_channel, end_channel = SEAM_EDGE_SLICES[edge_name]
        edge_tensor = seam_visible[:, start_channel:end_channel]
        valid_gate = edge_valid[:, edge_index].view(batch_size, 1, 1, 1)
        projected_rgb = zeros(3)
        projected_alpha = zeros(1)
        projected_sobel_x = zeros(1)
        projected_sobel_y = zeros(1)
        distance_to_seam = zeros(1)
        edge_valid_map = zeros(1)
        active_band_mask = zeros(1)
        invalid_mask = _fill_per_edge_projection(
            edge_name,
            edge_tensor=edge_tensor,
            geometry=geometry,
            projected_rgb=projected_rgb,
            projected_alpha=projected_alpha,
            projected_sobel_x=projected_sobel_x,
            projected_sobel_y=projected_sobel_y,
            distance_to_seam=distance_to_seam,
            edge_valid_map=edge_valid_map,
            active_band_mask=active_band_mask,
            valid_gate=valid_gate,
            band_px=band_px,
        )
        per_edge_input[:, edge_index, 0:3] = projected_rgb
        per_edge_input[:, edge_index, 3:4] = projected_alpha
        per_edge_input[:, edge_index, 4:5] = projected_sobel_x.clamp_(-1.0, 1.0)
        per_edge_input[:, edge_index, 5:6] = projected_sobel_y.clamp_(-1.0, 1.0)
        per_edge_input[:, edge_index, 6:7] = distance_to_seam.clamp_(0.0, 1.0)
        per_edge_input[:, edge_index, 7:8] = edge_valid_map.clamp_(0.0, 1.0)
        per_edge_input[:, edge_index, 8:9] = active_band_mask.clamp_(0.0, 1.0)
        per_edge_active_mask[:, edge_index, 0:1] = active_band_mask
        per_edge_invalid_mask[:, edge_index, 0:1] = invalid_mask

    combined_active_mask = per_edge_active_mask.sum(dim=1).clamp_(0.0, 1.0)
    return {
        "adapter_input": per_edge_input,
        "active_mask": per_edge_active_mask,
        "invalid_active_mask": per_edge_invalid_mask,
        "combined_active_mask": combined_active_mask,
        "edge_valid_count": edge_valid.sum(dim=1),
        "edge_valid_flags": edge_valid,
        "adapter_mode": "per_edge",
    }


def build_seam_local_adapter_maps_from_conditioning(
    cond_image: torch.Tensor,
    seam_conditioning_offset: int,
    band_px: int,
    seam_adapter_per_edge: bool = False,
    seam_adapter_extrusion_mode: str = "decay",
) -> Dict[str, torch.Tensor]:
    if bool(seam_adapter_per_edge):
        mode = str(seam_adapter_extrusion_mode or "decay").strip().lower()
        if mode != "full_strength":
            raise ValueError(f"unsupported seam_adapter_extrusion_mode='{seam_adapter_extrusion_mode}' for per-edge adapter")
        return _build_per_edge_seam_local_adapter_maps_from_conditioning(
            cond_image,
            seam_conditioning_offset=seam_conditioning_offset,
            band_px=band_px,
        )
    payload = _build_legacy_seam_local_adapter_maps_from_conditioning(
        cond_image,
        seam_conditioning_offset=seam_conditioning_offset,
        band_px=band_px,
    )
    payload["combined_active_mask"] = payload["active_mask"]
    payload["edge_valid_flags"] = None
    payload["adapter_mode"] = "legacy"
    return payload


class SeamLocalHiResAdapter(nn.Module):
    def __init__(self, in_channels: int = 6, hidden_channels: int = 128, out_channels: int = 320, zero_init: bool = True):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.block = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.proj_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        if zero_init:
            _zero_module(self.proj_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.proj_in(x))
        hidden = self.block(hidden)
        return self.proj_out(hidden)


class ControlNetConditioningEmbedding(nn.Module):
    def __init__(self, conditioning_channels: int = 3, pre_embed_channels: Optional[list[int]] = None):
        super().__init__()

        dims = [16, 32, 96, 256]

        pre_embed_channels = pre_embed_channels or []
        self.pre_blocks = nn.ModuleList([])

        in_channels = conditioning_channels
        for out_channels in pre_embed_channels:
            self.pre_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            in_channels = out_channels

        self.conv_in = nn.Conv2d(in_channels, dims[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([])

        for i in range(len(dims) - 1):
            channel_in = dims[i]
            channel_out = dims[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = nn.Conv2d(dims[-1], 320, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)  # zero module weight
        nn.init.zeros_(self.conv_out.bias)  # zero module bias

    def forward(self, x):
        for block in self.pre_blocks:
            x = block(x)
            x = F.silu(x)
        x = self.conv_in(x)
        x = F.silu(x)
        for block in self.blocks:
            x = block(x)
            x = F.silu(x)
        x = self.conv_out(x)
        return x


class SdxlControlNet(sdxl_original_unet.SdxlUNet2DConditionModel):
    def __init__(
        self,
        multiplier: Optional[float] = None,
        conditioning_channels: int = 3,
        conditioning_pre_embed_channels: Optional[list[int]] = None,
        alpha_head_indices: Optional[list[int]] = None,
        alpha_baseline_mode: str = "none",
        alpha_baseline_terrain_channel_index: int = 3,
        seam_adapter_enabled: bool = False,
        seam_adapter_band_px: int = 64,
        seam_adapter_scale: float = 1.0,
        seam_adapter_zero_init: bool = True,
        seam_adapter_target: str = "first_high_res",
        seam_adapter_conditioning_offset: int = -1,
        seam_adapter_per_edge: bool = False,
        seam_adapter_extrusion_mode: str = "decay",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.multiplier = multiplier
        self.alpha_head_indices = sorted(alpha_head_indices or [])
        self.alpha_baseline_mode = str(alpha_baseline_mode)
        self.alpha_baseline_terrain_channel_index = int(alpha_baseline_terrain_channel_index)
        self.seam_adapter_enabled = bool(seam_adapter_enabled)
        self.seam_adapter_band_px = int(seam_adapter_band_px)
        self.seam_adapter_scale = float(seam_adapter_scale)
        self.seam_adapter_target = str(seam_adapter_target)
        self.seam_adapter_conditioning_offset = int(seam_adapter_conditioning_offset)
        self.seam_adapter_per_edge = bool(seam_adapter_per_edge)
        self.seam_adapter_extrusion_mode = str(seam_adapter_extrusion_mode or "decay").strip().lower()
        self.seam_adapter_block_index = 1 if self.seam_adapter_target == "first_high_res" else 0

        # remove unet layers
        self.output_blocks = nn.ModuleList([])
        del self.out

        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_channels=conditioning_channels,
            pre_embed_channels=conditioning_pre_embed_channels,
        )

        dims = [320, 320, 320, 320, 640, 640, 640, 1280, 1280]
        self.controlnet_down_blocks = nn.ModuleList([])
        for dim in dims:
            self.controlnet_down_blocks.append(nn.Conv2d(dim, dim, kernel_size=1))
            nn.init.zeros_(self.controlnet_down_blocks[-1].weight)  # zero module weight
            nn.init.zeros_(self.controlnet_down_blocks[-1].bias)  # zero module bias

        self.controlnet_mid_block = nn.Conv2d(1280, 1280, kernel_size=1)
        nn.init.zeros_(self.controlnet_mid_block.weight)  # zero module weight
        nn.init.zeros_(self.controlnet_mid_block.bias)  # zero module bias

        self.controlnet_seam_adapter = None
        if self.seam_adapter_enabled:
            self.controlnet_seam_adapter = SeamLocalHiResAdapter(
                in_channels=(SEAM_ADAPTER_PER_EDGE_INPUT_CHANNELS if self.seam_adapter_per_edge else SEAM_ADAPTER_LEGACY_INPUT_CHANNELS),
                hidden_channels=128,
                out_channels=320,
                zero_init=bool(seam_adapter_zero_init),
            )

        self.controlnet_alpha_heads = nn.ModuleDict()
        for index in self.alpha_head_indices:
            dim = dims[index]
            self.controlnet_alpha_heads[str(index)] = nn.Conv2d(dim, 1, kernel_size=1)

        self.controlnet_alpha_baseline_heads = nn.ModuleDict()
        if self.alpha_baseline_mode in {"terrain_mask", "both"}:
            self.controlnet_alpha_baseline_heads["terrain_mask"] = nn.Conv2d(1, 1, kernel_size=1)
        if self.alpha_baseline_mode in {"pre_stem", "both"}:
            self.controlnet_alpha_baseline_heads["pre_stem"] = nn.Conv2d(320, 1, kernel_size=1)

    def init_from_unet(self, unet: sdxl_original_unet.SdxlUNet2DConditionModel):
        unet_sd = unet.state_dict()
        unet_sd = {k: v for k, v in unet_sd.items() if not k.startswith("out")}
        sd = super().state_dict()
        sd.update(unet_sd)
        info = super().load_state_dict(sd, strict=True, assign=True)
        return info

    def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = True) -> Any:
        # convert state_dict to SAI format
        unet_sd = {}
        for k in list(state_dict.keys()):
            if not k.startswith("controlnet_"):
                unet_sd[k] = state_dict.pop(k)
        unet_sd = convert_diffusers_unet_state_dict_to_sdxl(unet_sd)
        state_dict.update(unet_sd)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # convert state_dict to Diffusers format
        state_dict = super().state_dict(destination, prefix, keep_vars)
        control_net_sd = {}
        for k in list(state_dict.keys()):
            if k.startswith("controlnet_"):
                control_net_sd[k] = state_dict.pop(k)
        state_dict = convert_sdxl_unet_state_dict_to_diffusers(state_dict)
        state_dict.update(control_net_sd)
        return state_dict

    def forward(
        self,
        x: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        cond_image: Optional[torch.Tensor] = None,
        return_alpha: bool = False,
        return_diagnostics: bool = False,
        alpha_target_size: Optional[tuple[int, int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # broadcast timesteps to batch dimension
        timesteps = timesteps.expand(x.shape[0])

        t_emb = sdxl_original_unet.get_timestep_embedding(timesteps, self.model_channels, downscale_freq_shift=0)
        t_emb = t_emb.to(x.dtype)
        emb = self.time_embed(t_emb)

        assert x.shape[0] == y.shape[0], f"batch size mismatch: {x.shape[0]} != {y.shape[0]}"
        assert x.dtype == y.dtype, f"dtype mismatch: {x.dtype} != {y.dtype}"
        emb = emb + self.label_emb(y)

        def call_module(module, h, emb, context):
            x = h
            for layer in module:
                if isinstance(layer, sdxl_original_unet.ResnetBlock2D):
                    x = layer(x, emb)
                elif isinstance(layer, sdxl_original_unet.Transformer2DModel):
                    x = layer(x, context)
                else:
                    x = layer(x)
            return x

        h = x
        multiplier = self.multiplier if self.multiplier is not None else 1.0
        hs = []
        alpha_per_scale = []
        residual_norms = []
        activation_norms = []
        residual_to_activation_ratios = []
        cond_embedding_norm = None
        cond_embedding = None
        seam_adapter_maps = None
        seam_adapter_mask = None
        seam_invalid_mask = None
        seam_edge_valid_count = None
        seam_edge_valid_flags = None
        seam_combined_active_mask = None
        seam_per_edge_input_energy = None
        seam_per_edge_sobel_input_energy = None
        seam_per_edge_output_energy = None
        seam_per_edge_invalid_input_energy = None
        seam_per_edge_invalid_output_energy = None
        seam_per_edge_active_px = None
        seam_per_edge_ratio = None
        seam_corner_active_px = 0.0
        seam_residual_energy_map = None
        seam_input_energy = 0.0
        seam_output_energy = 0.0
        seam_invalid_input_energy = 0.0
        seam_invalid_output_energy = 0.0
        seam_adapter_ratio = 0.0
        seam_adapter_scale = float(self.seam_adapter_scale if self.seam_adapter_enabled else 0.0)
        seam_sobel_input_energy = 0.0

        if self.seam_adapter_enabled:
            seam_adapter_maps = kwargs.pop("seam_local_maps", None)
            seam_adapter_mask = kwargs.pop("seam_local_mask", None)
            seam_invalid_mask = kwargs.pop("seam_local_invalid_mask", None)
            seam_edge_valid_count = kwargs.pop("seam_local_edge_valid_count", None)
            seam_edge_valid_flags = kwargs.pop("seam_local_edge_valid_flags", None)
            seam_combined_active_mask = kwargs.pop("seam_local_combined_active_mask", None)
            if seam_adapter_maps is None or seam_adapter_mask is None:
                seam_payload = build_seam_local_adapter_maps_from_conditioning(
                    cond_image,
                    seam_conditioning_offset=self.seam_adapter_conditioning_offset,
                    band_px=self.seam_adapter_band_px,
                    seam_adapter_per_edge=self.seam_adapter_per_edge,
                    seam_adapter_extrusion_mode=self.seam_adapter_extrusion_mode,
                )
                seam_adapter_maps = seam_payload["adapter_input"]
                seam_adapter_mask = seam_payload["active_mask"]
                seam_invalid_mask = seam_payload["invalid_active_mask"]
                seam_edge_valid_count = seam_payload["edge_valid_count"]
                seam_edge_valid_flags = seam_payload.get("edge_valid_flags")
                seam_combined_active_mask = seam_payload.get("combined_active_mask")
            if seam_invalid_mask is None:
                seam_invalid_mask = torch.zeros_like(seam_adapter_mask)
            if seam_edge_valid_count is None:
                seam_edge_valid_count = torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)
            if seam_combined_active_mask is None:
                seam_combined_active_mask = seam_adapter_mask.sum(dim=1) if seam_adapter_mask.ndim == 5 else seam_adapter_mask
            if seam_adapter_maps.ndim == 5:
                seam_per_edge_input_energy = seam_adapter_maps.detach().float().reshape(seam_adapter_maps.shape[0], seam_adapter_maps.shape[1], -1).norm(dim=2)
                seam_per_edge_sobel_input_energy = seam_adapter_maps[:, :, 4:6].detach().float().reshape(
                    seam_adapter_maps.shape[0], seam_adapter_maps.shape[1], -1
                ).norm(dim=2)
                seam_per_edge_invalid_input_energy = (
                    (seam_adapter_maps.detach().float() * seam_invalid_mask.detach().float())
                    .reshape(seam_adapter_maps.shape[0], seam_adapter_maps.shape[1], -1)
                    .norm(dim=2)
                )
                seam_sobel_input_energy = float(seam_adapter_maps[:, :, 4:6].detach().float().norm().item())
            seam_input_energy = float(seam_adapter_maps.detach().float().norm().item())
            seam_invalid_input_energy = float((seam_adapter_maps.detach().float() * seam_invalid_mask.detach().float()).norm().item())

        for i, module in enumerate(self.input_blocks):
            h = call_module(module, h, emb, context)
            if i == 0:
                cond_embedding = self.controlnet_cond_embedding(cond_image)
                cond_embedding_norm = float(cond_embedding.detach().float().norm().item())
                h = cond_embedding + h
            if self.controlnet_seam_adapter is not None and i == self.seam_adapter_block_index:
                adapter_maps = seam_adapter_maps
                adapter_mask = seam_adapter_mask
                invalid_mask = seam_invalid_mask
                if adapter_maps is not None and adapter_mask is not None:
                    if adapter_maps.ndim == 5:
                        batch_size, edge_count, channels, map_h, map_w = adapter_maps.shape
                        flat_maps = adapter_maps.reshape(batch_size * edge_count, channels, map_h, map_w)
                        if flat_maps.shape[-2:] != h.shape[-2:]:
                            flat_maps = F.interpolate(flat_maps, size=h.shape[-2:], mode="bilinear", align_corners=False)
                        flat_mask = adapter_mask.reshape(batch_size * edge_count, 1, adapter_mask.shape[-2], adapter_mask.shape[-1])
                        if flat_mask.shape[-2:] != h.shape[-2:]:
                            flat_mask = F.interpolate(flat_mask, size=h.shape[-2:], mode="nearest")
                        flat_invalid = None
                        if invalid_mask is not None:
                            flat_invalid = invalid_mask.reshape(batch_size * edge_count, 1, invalid_mask.shape[-2], invalid_mask.shape[-1])
                            if flat_invalid.shape[-2:] != h.shape[-2:]:
                                flat_invalid = F.interpolate(flat_invalid, size=h.shape[-2:], mode="nearest")
                        edge_residual = self.controlnet_seam_adapter(flat_maps)
                        edge_residual = edge_residual.view(batch_size, edge_count, edge_residual.shape[1], h.shape[-2], h.shape[-1])
                        edge_mask = flat_mask.view(batch_size, edge_count, 1, h.shape[-2], h.shape[-1])
                        masked_residual = edge_residual * edge_mask * self.seam_adapter_scale
                        seam_per_edge_output_energy = masked_residual.detach().float().reshape(batch_size, edge_count, -1).norm(dim=2)
                        seam_per_edge_active_px = edge_mask.detach().float().sum(dim=(2, 3, 4))
                        mask_sum = edge_mask.sum(dim=1).clamp_min(1.0)
                        adapter_residual = masked_residual.sum(dim=1) / mask_sum
                        if flat_invalid is not None:
                            invalid_mask = flat_invalid.view(batch_size, edge_count, 1, h.shape[-2], h.shape[-1])
                            seam_per_edge_invalid_output_energy = (
                                (masked_residual.detach().float() * invalid_mask.detach().float()).reshape(batch_size, edge_count, -1).norm(dim=2)
                            )
                        adapter_mask = edge_mask.sum(dim=1).clamp_(0.0, 1.0)
                        seam_combined_active_mask = adapter_mask
                        seam_corner_active_px = float((edge_mask.sum(dim=1) > 1.0).detach().float().sum().item())
                    else:
                        if adapter_maps.shape[-2:] != h.shape[-2:]:
                            adapter_maps = F.interpolate(adapter_maps, size=h.shape[-2:], mode="bilinear", align_corners=False)
                        if adapter_mask.shape[-2:] != h.shape[-2:]:
                            adapter_mask = F.interpolate(adapter_mask, size=h.shape[-2:], mode="nearest")
                        if invalid_mask is not None and invalid_mask.shape[-2:] != h.shape[-2:]:
                            invalid_mask = F.interpolate(invalid_mask, size=h.shape[-2:], mode="nearest")
                        adapter_residual = self.controlnet_seam_adapter(adapter_maps) * adapter_mask * self.seam_adapter_scale
                        seam_combined_active_mask = adapter_mask
                    activation_norm = float(h.detach().float().norm().item())
                    h = h + adapter_residual
                    seam_output_energy = float(adapter_residual.detach().float().norm().item())
                    seam_adapter_ratio = seam_output_energy / max(activation_norm, 1e-12)
                    seam_residual_energy_map = adapter_residual.detach().float().pow(2.0).mean(dim=1, keepdim=True).sqrt()
                    if seam_per_edge_output_energy is not None:
                        seam_per_edge_ratio = seam_per_edge_output_energy / max(activation_norm, 1e-12)
                    if invalid_mask is not None:
                        seam_invalid_output_energy = float((adapter_residual.detach().float() * invalid_mask.detach().float()).norm().item())
            control_residual = self.controlnet_down_blocks[i](h) * multiplier
            hs.append(control_residual)
            if return_diagnostics:
                residual_norm = float(control_residual.detach().float().norm().item())
                activation_norm = float(h.detach().float().norm().item())
                residual_norms.append(residual_norm)
                activation_norms.append(activation_norm)
                residual_to_activation_ratios.append(residual_norm / max(activation_norm, 1e-12))
            if return_alpha and str(i) in self.controlnet_alpha_heads:
                alpha_per_scale.append((i, self.controlnet_alpha_heads[str(i)](control_residual)))

        h = call_module(self.middle_block, h, emb, context)
        h = self.controlnet_mid_block(h) * multiplier

        if not return_alpha:
            if return_diagnostics:
                return hs, h, {
                    "multiplier": float(multiplier),
                    "cond_embedding_norm": cond_embedding_norm,
                    "down_block_residual_norms": residual_norms,
                    "down_block_activation_norms": activation_norms,
                    "down_block_residual_to_activation_ratios": residual_to_activation_ratios,
                    "mid_block_residual_norm": float(h.detach().float().norm().item()),
                    "seam_adapter_enabled": bool(self.seam_adapter_enabled),
                    "seam_adapter_scale": seam_adapter_scale,
                    "seam_adapter_band_px": int(self.seam_adapter_band_px),
                    "seam_adapter_input_energy": seam_input_energy,
                    "seam_adapter_output_energy": seam_output_energy,
                    "seam_adapter_to_activation_ratio": seam_adapter_ratio,
                    "seam_adapter_active_px": float(seam_combined_active_mask.detach().float().sum().item()) if seam_combined_active_mask is not None else 0.0,
                    "seam_adapter_edge_valid_count": float(seam_edge_valid_count.detach().float().mean().item()) if seam_edge_valid_count is not None else 0.0,
                    "seam_adapter_edge_valid_flags": seam_edge_valid_flags.detach().float().cpu() if seam_edge_valid_flags is not None else None,
                    "seam_adapter_combined_active_px": float(seam_combined_active_mask.detach().float().sum().item()) if seam_combined_active_mask is not None else 0.0,
                    "seam_adapter_corner_active_px": seam_corner_active_px,
                    "seam_adapter_per_edge_input_energy": seam_per_edge_input_energy.detach().float().cpu() if seam_per_edge_input_energy is not None else None,
                    "seam_adapter_per_edge_output_energy": seam_per_edge_output_energy.detach().float().cpu() if seam_per_edge_output_energy is not None else None,
                    "seam_adapter_per_edge_sobel_input_energy": seam_per_edge_sobel_input_energy.detach().float().cpu() if seam_per_edge_sobel_input_energy is not None else None,
                    "seam_adapter_per_edge_invalid_input_energy": seam_per_edge_invalid_input_energy.detach().float().cpu() if seam_per_edge_invalid_input_energy is not None else None,
                    "seam_adapter_per_edge_invalid_output_energy": seam_per_edge_invalid_output_energy.detach().float().cpu() if seam_per_edge_invalid_output_energy is not None else None,
                    "seam_adapter_per_edge_active_px": seam_per_edge_active_px.detach().float().cpu() if seam_per_edge_active_px is not None else None,
                    "seam_adapter_per_edge_ratio": seam_per_edge_ratio.detach().float().cpu() if seam_per_edge_ratio is not None else None,
                    "seam_adapter_combined_active_mask": seam_combined_active_mask.detach().float().cpu() if seam_combined_active_mask is not None else None,
                    "seam_adapter_per_edge_mask": seam_adapter_mask.detach().float().cpu() if seam_adapter_mask is not None and seam_adapter_mask.ndim == 5 else None,
                    "seam_adapter_residual_energy_map": seam_residual_energy_map.detach().float().cpu() if seam_residual_energy_map is not None else None,
                    "seam_adapter_sobel_input_energy": seam_sobel_input_energy,
                    "seam_adapter_undefined_edge_input_energy": seam_invalid_input_energy,
                    "seam_adapter_undefined_edge_output_energy": seam_invalid_output_energy,
                }
            return hs, h

        fused_alpha_logits = None
        if alpha_per_scale:
            if alpha_target_size is None:
                raise ValueError("alpha_target_size is required when return_alpha=True")
            resized_logits = [
                F.interpolate(alpha_logits, size=alpha_target_size, mode="bilinear", align_corners=False)
                for _, alpha_logits in alpha_per_scale
            ]
            fused_alpha_logits = torch.stack(resized_logits, dim=0).mean(dim=0)

        baseline_logits = {}
        if alpha_target_size is not None and self.controlnet_alpha_baseline_heads:
            if "terrain_mask" in self.controlnet_alpha_baseline_heads:
                t_idx = self.alpha_baseline_terrain_channel_index
                if t_idx < 0 or t_idx >= cond_image.shape[1]:
                    raise ValueError(
                        f"terrain channel index {t_idx} out of range for cond_image channels={cond_image.shape[1]}"
                    )
                terrain_channel = cond_image[:, t_idx : t_idx + 1].contiguous()
                terrain_logits = self.controlnet_alpha_baseline_heads["terrain_mask"](terrain_channel)
                if terrain_logits.shape[-2:] != alpha_target_size:
                    terrain_logits = F.interpolate(
                        terrain_logits,
                        size=alpha_target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                baseline_logits["terrain_mask"] = terrain_logits
            if "pre_stem" in self.controlnet_alpha_baseline_heads:
                if cond_embedding is None:
                    raise RuntimeError("missing cond_embedding while computing pre_stem alpha baseline")
                pre_stem_logits = self.controlnet_alpha_baseline_heads["pre_stem"](cond_embedding)
                if pre_stem_logits.shape[-2:] != alpha_target_size:
                    pre_stem_logits = F.interpolate(
                        pre_stem_logits,
                        size=alpha_target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                baseline_logits["pre_stem"] = pre_stem_logits

        alpha_payload = {
            "per_scale_logits": alpha_per_scale,
            "fused_logits": fused_alpha_logits,
            "baseline_logits": baseline_logits,
        }
        if return_diagnostics:
            alpha_payload["diagnostics"] = {
                "multiplier": float(multiplier),
                "cond_embedding_norm": cond_embedding_norm,
                "down_block_residual_norms": residual_norms,
                "down_block_activation_norms": activation_norms,
                "down_block_residual_to_activation_ratios": residual_to_activation_ratios,
                "mid_block_residual_norm": float(h.detach().float().norm().item()),
                "seam_adapter_enabled": bool(self.seam_adapter_enabled),
                "seam_adapter_scale": seam_adapter_scale,
                "seam_adapter_band_px": int(self.seam_adapter_band_px),
                "seam_adapter_input_energy": seam_input_energy,
                "seam_adapter_output_energy": seam_output_energy,
                "seam_adapter_to_activation_ratio": seam_adapter_ratio,
                "seam_adapter_active_px": float(seam_combined_active_mask.detach().float().sum().item()) if seam_combined_active_mask is not None else 0.0,
                "seam_adapter_edge_valid_count": float(seam_edge_valid_count.detach().float().mean().item()) if seam_edge_valid_count is not None else 0.0,
                "seam_adapter_edge_valid_flags": seam_edge_valid_flags.detach().float().cpu() if seam_edge_valid_flags is not None else None,
                "seam_adapter_combined_active_px": float(seam_combined_active_mask.detach().float().sum().item()) if seam_combined_active_mask is not None else 0.0,
                "seam_adapter_corner_active_px": seam_corner_active_px,
                "seam_adapter_per_edge_input_energy": seam_per_edge_input_energy.detach().float().cpu() if seam_per_edge_input_energy is not None else None,
                "seam_adapter_per_edge_output_energy": seam_per_edge_output_energy.detach().float().cpu() if seam_per_edge_output_energy is not None else None,
                "seam_adapter_per_edge_sobel_input_energy": seam_per_edge_sobel_input_energy.detach().float().cpu() if seam_per_edge_sobel_input_energy is not None else None,
                "seam_adapter_per_edge_invalid_input_energy": seam_per_edge_invalid_input_energy.detach().float().cpu() if seam_per_edge_invalid_input_energy is not None else None,
                "seam_adapter_per_edge_invalid_output_energy": seam_per_edge_invalid_output_energy.detach().float().cpu() if seam_per_edge_invalid_output_energy is not None else None,
                "seam_adapter_per_edge_active_px": seam_per_edge_active_px.detach().float().cpu() if seam_per_edge_active_px is not None else None,
                "seam_adapter_per_edge_ratio": seam_per_edge_ratio.detach().float().cpu() if seam_per_edge_ratio is not None else None,
                "seam_adapter_combined_active_mask": seam_combined_active_mask.detach().float().cpu() if seam_combined_active_mask is not None else None,
                "seam_adapter_per_edge_mask": seam_adapter_mask.detach().float().cpu() if seam_adapter_mask is not None and seam_adapter_mask.ndim == 5 else None,
                "seam_adapter_residual_energy_map": seam_residual_energy_map.detach().float().cpu() if seam_residual_energy_map is not None else None,
                "seam_adapter_sobel_input_energy": seam_sobel_input_energy,
                "seam_adapter_undefined_edge_input_energy": seam_invalid_input_energy,
                "seam_adapter_undefined_edge_output_energy": seam_invalid_output_energy,
            }

        return hs, h, alpha_payload


class SdxlControlledUNet(sdxl_original_unet.SdxlUNet2DConditionModel):
    """
    This class is for training purpose only.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, timesteps=None, context=None, y=None, input_resi_add=None, mid_add=None, **kwargs):
        # broadcast timesteps to batch dimension
        timesteps = timesteps.expand(x.shape[0])

        hs = []
        t_emb = sdxl_original_unet.get_timestep_embedding(timesteps, self.model_channels, downscale_freq_shift=0)
        t_emb = t_emb.to(x.dtype)
        emb = self.time_embed(t_emb)

        assert x.shape[0] == y.shape[0], f"batch size mismatch: {x.shape[0]} != {y.shape[0]}"
        assert x.dtype == y.dtype, f"dtype mismatch: {x.dtype} != {y.dtype}"
        emb = emb + self.label_emb(y)

        def call_module(module, h, emb, context):
            x = h
            for layer in module:
                if isinstance(layer, sdxl_original_unet.ResnetBlock2D):
                    x = layer(x, emb)
                elif isinstance(layer, sdxl_original_unet.Transformer2DModel):
                    x = layer(x, context)
                else:
                    x = layer(x)
            return x

        h = x
        for module in self.input_blocks:
            h = call_module(module, h, emb, context)
            hs.append(h)

        h = call_module(self.middle_block, h, emb, context)
        h = h + mid_add

        for module in self.output_blocks:
            resi = hs.pop() + input_resi_add.pop()
            h = torch.cat([h, resi], dim=1)
            h = call_module(module, h, emb, context)

        h = h.type(x.dtype)
        h = call_module(self.out, h, emb, context)

        return h


if __name__ == "__main__":
    import time

    logger.info("create unet")
    unet = SdxlControlledUNet()
    unet.to("cuda", torch.bfloat16)
    unet.set_use_sdpa(True)
    unet.set_gradient_checkpointing(True)
    unet.train()

    logger.info("create control_net")
    control_net = SdxlControlNet()
    control_net.to("cuda")
    control_net.set_use_sdpa(True)
    control_net.set_gradient_checkpointing(True)
    control_net.train()

    logger.info("Initialize control_net from unet")
    control_net.init_from_unet(unet)

    unet.requires_grad_(False)
    control_net.requires_grad_(True)

    # 使用メモリ量確認用の疑似学習ループ
    logger.info("preparing optimizer")

    # optimizer = torch.optim.SGD(unet.parameters(), lr=1e-3, nesterov=True, momentum=0.9) # not working

    import bitsandbytes

    optimizer = bitsandbytes.adam.Adam8bit(control_net.parameters(), lr=1e-3)  # not working
    # optimizer = bitsandbytes.optim.RMSprop8bit(unet.parameters(), lr=1e-3)  # working at 23.5 GB with torch2
    # optimizer=bitsandbytes.optim.Adagrad8bit(unet.parameters(), lr=1e-3)  # working at 23.5 GB with torch2

    # import transformers
    # optimizer = transformers.optimization.Adafactor(unet.parameters(), relative_step=True)  # working at 22.2GB with torch2

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    logger.info("start training")
    steps = 10
    batch_size = 1

    for step in range(steps):
        logger.info(f"step {step}")
        if step == 1:
            time_start = time.perf_counter()

        x = torch.randn(batch_size, 4, 128, 128).cuda()  # 1024x1024
        t = torch.randint(low=0, high=1000, size=(batch_size,), device="cuda")
        txt = torch.randn(batch_size, 77, 2048).cuda()
        vector = torch.randn(batch_size, sdxl_original_unet.ADM_IN_CHANNELS).cuda()
        cond_img = torch.rand(batch_size, 3, 1024, 1024).cuda()

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            input_resi_add, mid_add = control_net(x, t, txt, vector, cond_img)
            output = unet(x, t, txt, vector, input_resi_add, mid_add)
            target = torch.randn_like(output)
            loss = torch.nn.functional.mse_loss(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    time_end = time.perf_counter()
    logger.info(f"elapsed time: {time_end - time_start} [sec] for last {steps - 1} steps")

    logger.info("finish training")
    sd = control_net.state_dict()

    from safetensors.torch import save_file

    save_file(sd, r"E:\Work\SD\Tmp\sdxl\ctrl\control_net.safetensors")
