import csv
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


CHANNEL_NAME_TO_INDEX = {
    "R": 0,
    "G": 1,
    "B": 2,
    "A": 3,
}

EDGE_INDEX_TO_NAME = ("north", "south", "east", "west")
EDGE_NAME_TO_INDEX = {name: index for index, name in enumerate(EDGE_INDEX_TO_NAME)}


@dataclass(frozen=True)
class SemanticChannelSpec:
    name: str
    source: str
    semantic_range: Tuple[float, float]
    clamp_range: Optional[Tuple[float, float]] = None
    disk_range: Optional[Tuple[float, float]] = None

    @property
    def atlas_name(self) -> str:
        return self.source.split(".", 1)[0]

    @property
    def channel_name(self) -> str:
        return self.source.split(".", 1)[1]


def _resolve_disk_range(array: np.ndarray, explicit_range: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    if explicit_range is not None:
        return explicit_range

    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        return float(info.min), float(info.max)

    return float(array.min()), float(array.max())


def _resize_tensor(
    tensor: torch.Tensor,
    size: Tuple[int, int],
    mode: str,
) -> torch.Tensor:
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    else:
        raise ValueError(f"unexpected tensor rank for resize: {tensor.shape}")

    kwargs = {}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False

    return F.interpolate(tensor, size=size, mode=mode, **kwargs).squeeze(0)


def _resolve_pil_resample(mode: str) -> Image.Resampling:
    mapping = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    if mode not in mapping:
        raise ValueError(f"unsupported image resize mode: {mode}")
    return mapping[mode]


def resolve_fixed_defined_edge_index(edge_name: str) -> int:
    return EDGE_NAME_TO_INDEX.get(str(edge_name or "").strip().lower(), -1)


def center_embed_spatial_tensor(tensor: torch.Tensor, halo_px: int, fill_value: float = 0.0) -> torch.Tensor:
    halo = int(max(0, halo_px))
    if halo <= 0:
        return tensor
    if tensor.ndim == 3:
        out = torch.full(
            (tensor.shape[0], tensor.shape[1] + (2 * halo), tensor.shape[2] + (2 * halo)),
            float(fill_value),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        out[:, halo : halo + tensor.shape[1], halo : halo + tensor.shape[2]] = tensor
        return out
    if tensor.ndim == 4:
        out = torch.full(
            (tensor.shape[0], tensor.shape[1], tensor.shape[2] + (2 * halo), tensor.shape[3] + (2 * halo)),
            float(fill_value),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        out[:, :, halo : halo + tensor.shape[2], halo : halo + tensor.shape[3]] = tensor
        return out
    raise ValueError(f"unexpected tensor rank for center embed: {tuple(tensor.shape)}")


def terrain_mask_to_occupancy(mask: torch.Tensor, black_is_terrain: bool) -> torch.Tensor:
    mask = mask.detach().float() if not mask.is_floating_point() else mask.float()
    mask = mask.clamp(0.0, 1.0)
    return (1.0 - mask) if black_is_terrain else mask


def _expand_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel_size = (radius * 2) + 1
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=radius)


def build_seam_supervision_mask(
    trusted_mask: torch.Tensor,
    edge_band_masks: torch.Tensor,
    edge_defined_flags: torch.Tensor,
    seam_config: Dict[str, object],
) -> torch.Tensor:
    mask = trusted_mask.float().clamp(0.0, 1.0)
    expand_px = int(max(0, int(seam_config.get("seam_supervision_expand_px", 0))))
    if expand_px > 0:
        mask = _expand_mask(mask, expand_px).clamp(0.0, 1.0)

    if bool(seam_config.get("force_defined_strip_supervision", True)):
        edge_defined = edge_defined_flags.float().unsqueeze(-1).unsqueeze(-1)
        defined_edge_band = (edge_band_masks.float() * edge_defined).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        mask = torch.maximum(mask, defined_edge_band)

    return mask


def build_seam_region_maps(
    edge_band_masks: torch.Tensor,
    seam_decay_maps: torch.Tensor,
    edge_defined_flags: torch.Tensor,
    seam_strip_width_px: torch.Tensor,
    supervision_mask: torch.Tensor,
    seam_config: Dict[str, object],
    expanded_halo_px: int = 0,
    continuation_valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    if edge_band_masks.ndim != 4 or seam_decay_maps.ndim != 4:
        raise ValueError("seam maps must have shape [batch, edges, height, width]")

    batch_size, _, height, width = edge_band_masks.shape
    device = edge_band_masks.device
    dtype = edge_band_masks.dtype

    strip_width = seam_strip_width_px.to(device=device, dtype=dtype).view(batch_size, 1, 1, 1).clamp_min(1.0)
    margin_inner_px = float(max(1, int(seam_config.get("margin_inner_px", 32))))
    continuation_width_px = float(
        max(
            1,
            int(
                seam_config.get(
                    "continuation_width_px",
                    seam_config.get("continuation_band_px", seam_config.get("interior_band_inner_px", 48)),
                )
            ),
        )
    )
    continuation_falloff_power = float(max(1e-3, float(seam_config.get("continuation_falloff_power", 2.0))))
    if continuation_valid_mask is None:
        continuation_valid_mask = torch.ones((batch_size, 1, height, width), device=device, dtype=dtype)
    else:
        continuation_valid_mask = continuation_valid_mask.to(device=device, dtype=dtype)
        if continuation_valid_mask.ndim != 4 or continuation_valid_mask.shape[1] != 1 or tuple(continuation_valid_mask.shape[-2:]) != (height, width):
            raise ValueError(
                "continuation_valid_mask must have shape [batch, 1, height, width]: "
                + f"got {tuple(continuation_valid_mask.shape)} expected {(batch_size, 1, height, width)}"
            )
        continuation_valid_mask = continuation_valid_mask.clamp(0.0, 1.0)

    yy = torch.arange(height, device=device, dtype=dtype).view(1, 1, height, 1).expand(batch_size, 1, height, width)
    xx = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width).expand(batch_size, 1, height, width)
    dist_top = yy
    dist_bottom = float(height - 1) - yy
    dist_right = float(width - 1) - xx
    dist_left = xx

    margin_inner_limit = torch.minimum(strip_width, torch.full_like(strip_width, margin_inner_px)).clamp_min(1.0)

    if int(expanded_halo_px) > 0:
        halo = float(int(expanded_halo_px))
        if halo <= 0.0 or halo * 2.0 >= float(min(height, width)):
            raise ValueError(
                "expanded seam geometry has invalid halo size for region construction: "
                + f"halo_px={expanded_halo_px} hw={(height, width)}"
            )

        interior_min_y = halo
        interior_max_y = float(height - 1) - halo
        interior_min_x = halo
        interior_max_x = float(width - 1) - halo

        north_dist_outside = (interior_min_y - yy).clamp(min=0.0)
        south_dist_outside = (yy - interior_max_y).clamp(min=0.0)
        east_dist_outside = (xx - interior_max_x).clamp(min=0.0)
        west_dist_outside = (interior_min_x - xx).clamp(min=0.0)

        north_active = north_dist_outside > 0.0
        south_active = south_dist_outside > 0.0
        east_active = east_dist_outside > 0.0
        west_active = west_dist_outside > 0.0
        outside_any = north_active | south_active | east_active | west_active

        inf = torch.full_like(north_dist_outside, float("inf"))
        side_dist_stack = torch.cat(
            [
                torch.where(north_active, north_dist_outside, inf),
                torch.where(south_active, south_dist_outside, inf),
                torch.where(east_active, east_dist_outside, inf),
                torch.where(west_active, west_dist_outside, inf),
            ],
            dim=1,
        )
        side_owner_idx = side_dist_stack.argmin(dim=1, keepdim=True)

        north_owner = outside_any & (side_owner_idx == 0)
        south_owner = outside_any & (side_owner_idx == 1)
        east_owner = outside_any & (side_owner_idx == 2)
        west_owner = outside_any & (side_owner_idx == 3)

        north_inner = (north_owner & (north_dist_outside <= margin_inner_limit)).to(dtype=dtype)
        south_inner = (south_owner & (south_dist_outside <= margin_inner_limit)).to(dtype=dtype)
        east_inner = (east_owner & (east_dist_outside <= margin_inner_limit)).to(dtype=dtype)
        west_inner = (west_owner & (west_dist_outside <= margin_inner_limit)).to(dtype=dtype)
        north_outer = (north_owner & (north_dist_outside > margin_inner_limit)).to(dtype=dtype)
        south_outer = (south_owner & (south_dist_outside > margin_inner_limit)).to(dtype=dtype)
        east_outer = (east_owner & (east_dist_outside > margin_inner_limit)).to(dtype=dtype)
        west_outer = (west_owner & (west_dist_outside > margin_inner_limit)).to(dtype=dtype)

        margin_inner_per_edge = torch.cat([north_inner, south_inner, east_inner, west_inner], dim=1)
        margin_outer_per_edge = torch.cat([north_outer, south_outer, east_outer, west_outer], dim=1)
    else:
        north_inner = (dist_top < margin_inner_limit).to(dtype=dtype)
        south_inner = (dist_bottom < margin_inner_limit).to(dtype=dtype)
        east_inner = (dist_right < margin_inner_limit).to(dtype=dtype)
        west_inner = (dist_left < margin_inner_limit).to(dtype=dtype)
        margin_inner_per_edge = torch.cat([north_inner, south_inner, east_inner, west_inner], dim=1) * edge_band_masks
        margin_outer_per_edge = (edge_band_masks - margin_inner_per_edge).clamp(0.0, 1.0)

    if int(expanded_halo_px) > 0:
        boundary_offset = torch.full_like(strip_width, float(int(expanded_halo_px)))
    else:
        boundary_offset = strip_width

    north_dist_interior = (yy - boundary_offset).clamp(min=0.0)
    south_dist_interior = ((float(height - 1) - boundary_offset) - yy).clamp(min=0.0)
    east_dist_interior = (((float(width - 1) - boundary_offset) - xx)).clamp(min=0.0)
    west_dist_interior = (xx - boundary_offset).clamp(min=0.0)

    continuation_norm_north = (1.0 - (north_dist_interior / float(continuation_width_px))).clamp(0.0, 1.0).pow(continuation_falloff_power)
    continuation_norm_south = (1.0 - (south_dist_interior / float(continuation_width_px))).clamp(0.0, 1.0).pow(continuation_falloff_power)
    continuation_norm_east = (1.0 - (east_dist_interior / float(continuation_width_px))).clamp(0.0, 1.0).pow(continuation_falloff_power)
    continuation_norm_west = (1.0 - (west_dist_interior / float(continuation_width_px))).clamp(0.0, 1.0).pow(continuation_falloff_power)

    continuation_binary_per_edge = torch.cat(
        [
            ((seam_decay_maps[:, 0:1] > 0.0) & (north_dist_interior <= float(continuation_width_px))).to(dtype=dtype),
            ((seam_decay_maps[:, 1:2] > 0.0) & (south_dist_interior <= float(continuation_width_px))).to(dtype=dtype),
            ((seam_decay_maps[:, 2:3] > 0.0) & (east_dist_interior <= float(continuation_width_px))).to(dtype=dtype),
            ((seam_decay_maps[:, 3:4] > 0.0) & (west_dist_interior <= float(continuation_width_px))).to(dtype=dtype),
        ],
        dim=1,
    )
    continuation_weighted_per_edge = torch.cat(
        [
            continuation_norm_north * (seam_decay_maps[:, 0:1] > 0.0).to(dtype=dtype),
            continuation_norm_south * (seam_decay_maps[:, 1:2] > 0.0).to(dtype=dtype),
            continuation_norm_east * (seam_decay_maps[:, 2:3] > 0.0).to(dtype=dtype),
            continuation_norm_west * (seam_decay_maps[:, 3:4] > 0.0).to(dtype=dtype),
        ],
        dim=1,
    )
    interior_core_per_edge = (seam_decay_maps > 0.0).to(dtype=dtype)
    interior_outer_per_edge = (interior_core_per_edge - continuation_binary_per_edge).clamp(0.0, 1.0)
    interior_inner_per_edge = continuation_binary_per_edge

    edge_defined = edge_defined_flags.to(device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
    continuation_valid_per_edge = continuation_valid_mask.expand(-1, 4, -1, -1)
    interior_inner_per_edge = interior_inner_per_edge * edge_defined * continuation_valid_per_edge
    interior_outer_per_edge = interior_outer_per_edge * edge_defined * continuation_valid_per_edge
    interior_core_per_edge = interior_core_per_edge * edge_defined * continuation_valid_per_edge
    continuation_weighted_per_edge = continuation_weighted_per_edge * edge_defined * continuation_valid_per_edge
    continuation_active_per_edge = (interior_inner_per_edge.sum(dim=(-2, -1), keepdim=True) > 0.0).to(dtype=dtype)

    if bool(seam_config.get("require_defined_for_margin_and_band", True)):
        margin_inner_per_edge = margin_inner_per_edge * edge_defined
        margin_outer_per_edge = margin_outer_per_edge * edge_defined

    margin_inner_per_edge = margin_inner_per_edge * continuation_active_per_edge
    margin_outer_per_edge = margin_outer_per_edge * continuation_active_per_edge

    margin_inner_map = margin_inner_per_edge.sum(dim=1, keepdim=True).clamp(0.0, 1.0) * supervision_mask
    margin_outer_map = margin_outer_per_edge.sum(dim=1, keepdim=True).clamp(0.0, 1.0) * supervision_mask
    interior_inner_map = interior_inner_per_edge.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    interior_outer_map = interior_outer_per_edge.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    interior_core_map = interior_core_per_edge.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    continuation_weighted_map = continuation_weighted_per_edge.sum(dim=1, keepdim=True).clamp(0.0, 1.0)

    return {
        "margin_inner": margin_inner_map,
        "margin_outer": margin_outer_map,
        "interior_inner": interior_inner_map,
        "interior_outer": interior_outer_map,
        "halo_inner": margin_inner_map,
        "halo_outer": margin_outer_map,
        "interior_continuation": interior_inner_map,
        "interior_core": interior_core_map,
        "continuation_distance_weighted": continuation_weighted_map,
        "margin_inner_per_edge": margin_inner_per_edge,
        "margin_outer_per_edge": margin_outer_per_edge,
        "interior_inner_per_edge": interior_inner_per_edge,
        "interior_outer_per_edge": interior_outer_per_edge,
        "interior_continuation_per_edge": interior_inner_per_edge,
        "interior_core_per_edge": interior_core_per_edge,
        "continuation_distance_weighted_per_edge": continuation_weighted_per_edge,
    }


def summarize_seam_edge_qualification(
    seam_maps: Dict[str, torch.Tensor],
    edge_defined_flags: torch.Tensor,
    seam_config: Dict[str, object],
) -> Dict[str, torch.Tensor]:
    device = edge_defined_flags.device
    dtype = edge_defined_flags.dtype if edge_defined_flags.is_floating_point() else torch.float32
    zero = torch.zeros_like(edge_defined_flags, dtype=dtype, device=device)

    margin_inner_per_edge = seam_maps.get("margin_inner_per_edge")
    margin_outer_per_edge = seam_maps.get("margin_outer_per_edge")
    continuation_per_edge = seam_maps.get("interior_inner_per_edge")
    continuation_weighted_per_edge = seam_maps.get("continuation_distance_weighted_per_edge")
    if margin_inner_per_edge is None or margin_outer_per_edge is None or continuation_per_edge is None or continuation_weighted_per_edge is None:
        raise ValueError("seam qualification requires per-edge seam maps")

    continuation_px = continuation_per_edge.sum(dim=(-2, -1))
    halo_inner_px = margin_inner_per_edge.sum(dim=(-2, -1))
    halo_outer_px = margin_outer_per_edge.sum(dim=(-2, -1))
    halo_total_px = halo_inner_px + halo_outer_px
    continuation_weight_sum = continuation_weighted_per_edge.sum(dim=(-2, -1))

    defined_mask = (edge_defined_flags.to(device=device, dtype=dtype) >= 0.5).to(dtype=dtype)
    qualified_mask = defined_mask.clone()
    if bool(seam_config.get("seam_qualified_sampling_enabled", False)):
        min_continuation_px = float(max(0.0, float(seam_config.get("seam_qualified_min_continuation_px", 0.0))))
        min_halo_px = float(max(0.0, float(seam_config.get("seam_qualified_min_halo_px", 0.0))))
        qualified_mask = qualified_mask * (continuation_px >= min_continuation_px).to(dtype=dtype)
        qualified_mask = qualified_mask * (halo_total_px >= min_halo_px).to(dtype=dtype)

    return {
        "qualified_edge_mask": qualified_mask,
        "continuation_px": continuation_px,
        "halo_inner_px": halo_inner_px,
        "halo_outer_px": halo_outer_px,
        "halo_total_px": halo_total_px,
        "continuation_weight_sum": continuation_weight_sum,
        "valid_edges_for_loss_count": qualified_mask.sum(dim=1),
        "defined_edges_count": defined_mask.sum(dim=1),
        "valid_edge_ratio": qualified_mask.sum(dim=1) / defined_mask.sum(dim=1).clamp_min(1.0),
        "zero": zero,
    }


def _has_native_alpha_channel(image: Image.Image) -> bool:
    return "A" in image.getbands()


def _to_tensor_image(
    image: Image.Image,
    size: Tuple[int, int],
    resize_mode: str,
    return_alpha: bool = False,
    synthesize_opaque_alpha: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
    native_alpha = _has_native_alpha_channel(image)
    resized = image.resize((size[1], size[0]), resample=_resolve_pil_resample(resize_mode)).convert("RGBA")
    array = np.asarray(resized, dtype=np.float32) / 255.0
    rgb = array[:, :, :3]
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    rgb_tensor = tensor * 2.0 - 1.0

    alpha_tensor: Optional[torch.Tensor] = None
    if return_alpha:
        if native_alpha or synthesize_opaque_alpha:
            alpha_tensor = torch.from_numpy(array[:, :, 3]).contiguous().float()
        else:
            alpha_tensor = None

    return rgb_tensor, alpha_tensor, native_alpha


class TerrainSemanticManifestDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        manifest_path: str,
        channel_specs: Sequence[SemanticChannelSpec],
        train_size: Tuple[int, int],
        prompt: str,
        prompt2: Optional[str] = None,
        min_trusted_mask_ratio: float = 0.05,
        image_resize_mode: str = "bicubic",
        semantic_resize_mode: str = "bilinear",
        latent_cache_dir: Optional[str] = None,
        latent_cache_version: str = "v1",
        latent_cache_vae_key: Optional[str] = None,
        enable_alpha_supervision: bool = False,
        strict_alpha: bool = False,
        seam_enabled: bool = False,
        seam_strip_width_px: int = 64,
        seam_state_all_defined_weight: float = 0.25,
        seam_state_partial_defined_weight: float = 0.50,
        seam_state_none_defined_weight: float = 0.25,
        seam_partial_one_edge_ratio: float = 0.45,
        seam_undefined_zero_prob: float = 0.40,
        seam_undefined_noise_prob: float = 0.40,
        seam_fixed_defined_edge_index: int = -1,
        seam_runtime_config: Optional[Dict[str, object]] = None,
        seam_seed: int = 1337,
        expanded_target_halo_px: int = 0,
        terrain_mask_black_is_terrain: bool = True,
        alpha_binary_threshold: float = 0.5,
        boundary_chunk_stride_px: int = 16,
        boundary_grid_offset_x_px: int = 0,
        boundary_grid_offset_y_px: int = 0,
        boundary_alignment_error_max_px: float = 0.5,
        boundary_consistency_error_max_px: float = 0.5,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.manifest_path = manifest_path
        self.channel_specs = list(channel_specs)
        self.train_size = train_size
        self.prompt = prompt
        self.prompt2 = prompt2 or prompt
        self.min_trusted_mask_ratio = min_trusted_mask_ratio
        self.image_resize_mode = image_resize_mode
        self.semantic_resize_mode = semantic_resize_mode
        self.latent_cache_dir = latent_cache_dir
        self.latent_cache_version = latent_cache_version
        self.latent_cache_vae_key = latent_cache_vae_key or "default"
        self.enable_alpha_supervision = enable_alpha_supervision
        self.strict_alpha = strict_alpha
        self.seam_enabled = bool(seam_enabled)
        self.seam_strip_width_px = int(max(1, seam_strip_width_px))
        self.seam_state_all_defined_weight = float(max(0.0, seam_state_all_defined_weight))
        self.seam_state_partial_defined_weight = float(max(0.0, seam_state_partial_defined_weight))
        self.seam_state_none_defined_weight = float(max(0.0, seam_state_none_defined_weight))
        self.seam_partial_one_edge_ratio = float(min(max(seam_partial_one_edge_ratio, 0.0), 1.0))
        self.seam_undefined_zero_prob = float(min(max(seam_undefined_zero_prob, 0.0), 1.0))
        self.seam_undefined_noise_prob = float(min(max(seam_undefined_noise_prob, 0.0), 1.0))
        fixed_edge = int(seam_fixed_defined_edge_index)
        self.seam_fixed_defined_edge_index = fixed_edge if fixed_edge in (0, 1, 2, 3) else -1
        runtime_config = dict(seam_runtime_config or {})
        if self.seam_fixed_defined_edge_index >= 0:
            runtime_config.setdefault("fixed_defined_edge", EDGE_INDEX_TO_NAME[self.seam_fixed_defined_edge_index])
        runtime_config.setdefault("margin_inner_px", 32)
        runtime_config.setdefault("continuation_width_px", 48)
        runtime_config.setdefault("continuation_falloff_power", 2.0)
        runtime_config.setdefault("require_defined_for_margin_and_band", True)
        runtime_config.setdefault("force_defined_strip_supervision", True)
        runtime_config.setdefault("seam_supervision_expand_px", 0)
        runtime_config.setdefault("seam_qualified_sampling_enabled", False)
        runtime_config.setdefault("seam_qualified_min_continuation_px", 0)
        runtime_config.setdefault("seam_qualified_min_halo_px", 0)
        self.seam_runtime_config = runtime_config
        self._seam_rng = random.Random(int(seam_seed))
        self.expanded_target_halo_px = int(max(0, expanded_target_halo_px))
        self.terrain_mask_black_is_terrain = bool(terrain_mask_black_is_terrain)
        self.alpha_binary_threshold = float(alpha_binary_threshold)
        self.terrain_mask_channel_index = self.channel_names.index("terrain_mask") if "terrain_mask" in self.channel_names else -1
        self.boundary_chunk_stride_px = int(max(1, boundary_chunk_stride_px))
        self.boundary_grid_offset_x_px = int(boundary_grid_offset_x_px)
        self.boundary_grid_offset_y_px = int(boundary_grid_offset_y_px)
        self.boundary_alignment_error_max_px = float(max(0.0, boundary_alignment_error_max_px))
        self.boundary_consistency_error_max_px = float(max(0.0, boundary_consistency_error_max_px))

        self._manifest_audit = {
            "total_rows": 0,
            "usable_rows": 0,
            "skipped_rejected_or_weight": 0,
            "skipped_bad_path": 0,
            "skipped_out_of_bounds": 0,
            "skipped_low_trusted_area": 0,
            "skipped_insufficient_halo_margin": 0,
            "path_remapped": 0,
            "native_alpha_rows": 0,
            "synthesized_opaque_rows": 0,
            "boundary_alignment_error_mean": 0.0,
            "boundary_alignment_error_p95": 0.0,
            "boundary_alignment_error_max": 0.0,
            "boundary_consistency_error_mean": 0.0,
            "boundary_consistency_error_p95": 0.0,
            "boundary_consistency_error_max": 0.0,
            "boundary_outlier_count": 0,
            "boundary_consistency_metric": "edge_alignment_variance_proxy",
            "boundary_consistency_is_proxy": True,
            "seam_qualified_records": 0,
            "seam_disqualified_records": 0,
            "seam_qualification_discoveries": 0,
        }
        self._records = self._load_manifest()
        self._sampling_weights = torch.tensor([record["sampling_weight"] for record in self._records], dtype=torch.float32)
        self._seam_edge_qualification_cache: Dict[Tuple[int, int], Dict[str, float]] = {}
        self._seam_record_eligibility_cache: Dict[int, bool] = {}
        self._cached_latents: List[Optional[torch.Tensor]] = [None] * len(self._records)
        if self.latent_cache_dir:
            os.makedirs(self.latent_cache_dir, exist_ok=True)
        self._recursive_latent_cache_index = self._index_existing_latent_cache_files()
        self._cached_latent_paths: List[Optional[str]] = [self._latent_cache_file(index) for index in range(len(self._records))]
        self._cached_latent_read_paths: List[Optional[str]] = [
            self._resolve_existing_latent_cache_path(index) for index in range(len(self._records))
        ]
        self._last_latent_cache_report: Dict[str, object] = {
            "total": len(self._records),
            "in_memory_hits": 0,
            "disk_hits": 0,
            "encoded": 0,
            "disk_misses": len(self._records),
            "cache_dir": self.latent_cache_dir,
        }

    @property
    def seam_channel_names(self) -> List[str]:
        if not self.seam_enabled:
            return []
        return [
            "seam_north_r",
            "seam_north_g",
            "seam_north_b",
            "seam_north_a",
            "seam_south_r",
            "seam_south_g",
            "seam_south_b",
            "seam_south_a",
            "seam_east_r",
            "seam_east_g",
            "seam_east_b",
            "seam_east_a",
            "seam_west_r",
            "seam_west_g",
            "seam_west_b",
            "seam_west_a",
            "seam_flag_north",
            "seam_flag_south",
            "seam_flag_east",
            "seam_flag_west",
        ]

    @property
    def full_conditioning_channel_names(self) -> List[str]:
        return self.channel_names + self.seam_channel_names

    @property
    def full_conditioning_channels(self) -> int:
        return len(self.full_conditioning_channel_names)

    def _load_manifest(self) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        boundary_alignment_errors: List[float] = []
        boundary_consistency_errors: List[float] = []
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                self._manifest_audit["total_rows"] += 1
                rejection_reason = (row.get("rejection_reason") or "").strip()
                sampling_weight = float(row.get("sampling_weight") or 0.0)
                if rejection_reason or sampling_weight <= 0.0:
                    self._manifest_audit["skipped_rejected_or_weight"] += 1
                    continue

                crop_x = int(float(row["crop_box_x"]))
                crop_y = int(float(row["crop_box_y"]))
                crop_w = int(float(row["crop_box_w"]))
                crop_h = int(float(row["crop_box_h"]))

                trusted_x = int(float(row["trusted_center_x"]))
                trusted_y = int(float(row["trusted_center_y"]))
                trusted_w = int(float(row["trusted_center_w"]))
                trusted_h = int(float(row["trusted_center_h"]))

                image_path = self._resolve_existing_path(row["image_path"])
                base_atlas_path = self._resolve_existing_path(row["base_atlas_path"])
                edge_atlas_path = self._resolve_existing_path(row["edge_atlas_path"])
                interior_atlas_path = self._resolve_existing_path(row["interior_atlas_path"])

                if any(path is None for path in [image_path, base_atlas_path, edge_atlas_path, interior_atlas_path]):
                    self._manifest_audit["skipped_bad_path"] += 1
                    continue

                try:
                    with Image.open(str(image_path)) as image_probe:
                        image_width, image_height = image_probe.size
                        has_native_alpha = _has_native_alpha_channel(image_probe)
                    with Image.open(str(base_atlas_path)) as base_probe:
                        base_width, base_height = base_probe.size
                    with Image.open(str(edge_atlas_path)) as edge_probe:
                        edge_width, edge_height = edge_probe.size
                    with Image.open(str(interior_atlas_path)) as interior_probe:
                        interior_width, interior_height = interior_probe.size
                except (OSError, ValueError) as e:
                    self._manifest_audit["skipped_bad_path"] += 1
                    continue

                if (
                    crop_x < 0
                    or crop_y < 0
                    or crop_w <= 0
                    or crop_h <= 0
                    or crop_x + crop_w > image_width
                    or crop_y + crop_h > image_height
                    or crop_x + crop_w > base_width
                    or crop_y + crop_h > base_height
                    or crop_x + crop_w > edge_width
                    or crop_y + crop_h > edge_height
                    or crop_x + crop_w > interior_width
                    or crop_y + crop_h > interior_height
                ):
                    self._manifest_audit["skipped_out_of_bounds"] += 1
                    continue

                if (
                    trusted_x < 0
                    or trusted_y < 0
                    or trusted_w <= 0
                    or trusted_h <= 0
                    or trusted_x + trusted_w > crop_w
                    or trusted_y + trusted_h > crop_h
                ):
                    self._manifest_audit["skipped_out_of_bounds"] += 1
                    continue

                trusted_area = trusted_w * trusted_h
                crop_area = max(1, crop_w * crop_h)
                trusted_ratio = trusted_area / crop_area
                if trusted_ratio < self.min_trusted_mask_ratio:
                    self._manifest_audit["skipped_low_trusted_area"] += 1
                    continue

                def _edge_mod_distance(value: int, offset: int, stride: int) -> float:
                    mod = (value - offset) % stride
                    return float(min(mod, stride - mod))

                left_err = _edge_mod_distance(crop_x, self.boundary_grid_offset_x_px, self.boundary_chunk_stride_px)
                right_err = _edge_mod_distance(crop_x + crop_w, self.boundary_grid_offset_x_px, self.boundary_chunk_stride_px)
                top_err = _edge_mod_distance(crop_y, self.boundary_grid_offset_y_px, self.boundary_chunk_stride_px)
                bottom_err = _edge_mod_distance(crop_y + crop_h, self.boundary_grid_offset_y_px, self.boundary_chunk_stride_px)
                edge_errors = np.asarray([left_err, right_err, top_err, bottom_err], dtype=np.float32)
                boundary_alignment_error = float(np.mean(edge_errors))
                # Proxy consistency score: lower variance means all four edges are similarly aligned to grid cadence.
                # This does not estimate full spatial distortion across neighboring chunks.
                boundary_consistency_error = float(np.var(edge_errors))
                boundary_alignment_errors.append(boundary_alignment_error)
                boundary_consistency_errors.append(boundary_consistency_error)

                record = {
                    "image_name": row["image_name"],
                    "image_path": str(image_path),
                    "has_native_alpha": has_native_alpha,
                    "base_atlas_path": str(base_atlas_path),
                    "edge_atlas_path": str(edge_atlas_path),
                    "interior_atlas_path": str(interior_atlas_path),
                    "crop_box": (crop_x, crop_y, crop_w, crop_h),
                    "trusted_box": (trusted_x, trusted_y, trusted_w, trusted_h),
                    "special_structure_tags": (row.get("special_structure_tags") or "").strip(),
                    "assigned_crop_class": (row.get("assigned_crop_class") or "").strip(),
                    "crop_size_class": row.get("crop_size_class") or "",
                    "generation_strategy": row.get("generation_strategy") or "",
                    "sampling_weight": sampling_weight,
                    "trusted_ratio": trusted_ratio,
                    "boundary_alignment_error": boundary_alignment_error,
                    "boundary_consistency_error": boundary_consistency_error,
                }
                records.append(record)

        self._manifest_audit["usable_rows"] = len(records)
        self._manifest_audit["native_alpha_rows"] = sum(1 for record in records if record["has_native_alpha"])
        self._manifest_audit["synthesized_opaque_rows"] = len(records) - self._manifest_audit["native_alpha_rows"]
        if boundary_alignment_errors:
            align_arr = np.asarray(boundary_alignment_errors, dtype=np.float32)
            consistency_arr = np.asarray(boundary_consistency_errors, dtype=np.float32)
            self._manifest_audit["boundary_alignment_error_mean"] = float(np.mean(align_arr))
            self._manifest_audit["boundary_alignment_error_p95"] = float(np.percentile(align_arr, 95.0))
            self._manifest_audit["boundary_alignment_error_max"] = float(np.max(align_arr))
            self._manifest_audit["boundary_consistency_error_mean"] = float(np.mean(consistency_arr))
            self._manifest_audit["boundary_consistency_error_p95"] = float(np.percentile(consistency_arr, 95.0))
            self._manifest_audit["boundary_consistency_error_max"] = float(np.max(consistency_arr))
            self._manifest_audit["boundary_outlier_count"] = int(
                np.sum(
                    (align_arr > self.boundary_alignment_error_max_px)
                    | (consistency_arr > self.boundary_consistency_error_max_px)
                )
            )

        # Debug print removed for production/cleanup
        if not records:
            raise ValueError(f"no usable samples found in manifest: {self.manifest_path}")

        if (
            self._manifest_audit["boundary_alignment_error_max"] > self.boundary_alignment_error_max_px
            or self._manifest_audit["boundary_consistency_error_max"] > self.boundary_consistency_error_max_px
        ):
            raise ValueError(
                "manifest failed boundary alignment validation: "
                + f"alignment_max={self._manifest_audit['boundary_alignment_error_max']:.4f} "
                + f"(limit={self.boundary_alignment_error_max_px:.4f}), "
                + f"consistency_max={self._manifest_audit['boundary_consistency_error_max']:.4f} "
                + f"(limit={self.boundary_consistency_error_max_px:.4f})"
            )
        return records

    def _resolve_path(self, path: str) -> str:
        normalized = path.strip()
        if os.path.isabs(normalized):
            return normalized
        return os.path.join(self.root_dir, normalized)

    def _resolve_existing_path(self, path: str) -> Optional[str]:
        candidate = self._resolve_path(path)
        if os.path.exists(candidate):
            return candidate

        remapped = path.replace("training_semantic_atlases/all_training_images/", "training_semantic_atlases/")
        if remapped != path:
            remapped_candidate = self._resolve_path(remapped)
            if os.path.exists(remapped_candidate):
                self._manifest_audit["path_remapped"] += 1
                return remapped_candidate

        return None

    @property
    def sampling_weights(self) -> torch.Tensor:
        return self._sampling_weights

    @property
    def channel_names(self) -> List[str]:
        return [spec.name for spec in self.channel_specs]

    @property
    def records(self) -> List[Dict[str, object]]:
        return self._records

    @property
    def manifest_audit(self) -> Dict[str, object]:
        return dict(self._manifest_audit)

    @property
    def alpha_source_summary(self) -> Dict[str, float]:
        total = len(self._records)
        native_count = sum(1 for record in self._records if record["has_native_alpha"])
        synthesized_count = total - native_count
        return {
            "total": float(total),
            "native_alpha_count": float(native_count),
            "synthesized_opaque_count": float(synthesized_count),
            "native_alpha_fraction": 0.0 if total == 0 else native_count / total,
            "synthesized_opaque_fraction": 0.0 if total == 0 else synthesized_count / total,
        }

    @property
    def latent_cache_report(self) -> Dict[str, object]:
        return dict(self._last_latent_cache_report)

    def _expected_latent_shape(self) -> Tuple[int, int, int]:
        return (4, self.train_size[0] // 8, self.train_size[1] // 8)

    def _index_existing_latent_cache_files(self) -> Dict[str, str]:
        if not self.latent_cache_dir or not os.path.isdir(self.latent_cache_dir):
            return {}

        cache_index: Dict[str, str] = {}
        for current_dir, _, filenames in os.walk(self.latent_cache_dir):
            for filename in filenames:
                if filename in cache_index:
                    continue
                cache_index[filename] = os.path.join(current_dir, filename)
        return cache_index

    def _latent_cache_file(self, index: int) -> Optional[str]:
        if not self.latent_cache_dir:
            return None

        record = self._records[index]
        cache_key_payload = {
            "version": self.latent_cache_version,
            "vae_key": self.latent_cache_vae_key,
            "image_resize_mode": self.image_resize_mode,
            "train_size": [int(self.train_size[0]), int(self.train_size[1])],
            "image_path": record["image_path"],
            "crop_box": [int(v) for v in record["crop_box"]],
        }
        digest = hashlib.sha256(json.dumps(cache_key_payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
        filename = f"{index:06d}_{digest}.pt"
        return os.path.join(self.latent_cache_dir, filename)

    def _resolve_existing_latent_cache_path(self, index: int) -> Optional[str]:
        cache_path = self._cached_latent_paths[index]
        if not cache_path:
            return None
        if os.path.isfile(cache_path):
            return cache_path

        filename = os.path.basename(cache_path)
        return self._recursive_latent_cache_index.get(filename, cache_path)

    def _load_cached_latent(self, index: int) -> Optional[torch.Tensor]:
        cache_path = self._cached_latent_read_paths[index]
        if not cache_path or not os.path.isfile(cache_path):
            return None

        try:
            payload = torch.load(cache_path, map_location="cpu")
            latent = payload["latents"] if isinstance(payload, dict) and "latents" in payload else payload
            if not isinstance(latent, torch.Tensor):
                return None
            latent = latent.contiguous().to(dtype=torch.float32)
            if tuple(latent.shape) != self._expected_latent_shape():
                return None
            return latent
        except Exception:
            return None

    def _save_cached_latent(self, index: int, latent: torch.Tensor) -> None:
        cache_path = self._cached_latent_paths[index]
        if not cache_path:
            return

        payload = {
            "latents": latent.detach().to("cpu", dtype=torch.float32).contiguous(),
            "shape": list(latent.shape),
            "version": self.latent_cache_version,
            "vae_key": self.latent_cache_vae_key,
        }
        torch.save(payload, cache_path)

    def cache_latents(self, vae, device: torch.device, dtype: torch.dtype, batch_size: int = 1) -> None:
        in_memory_hits = 0
        disk_hits = 0
        encoded = 0

        for index, latent in enumerate(self._cached_latents):
            if latent is not None:
                in_memory_hits += 1
                continue

            loaded = self._load_cached_latent(index)
            if loaded is not None:
                self._cached_latents[index] = loaded
                disk_hits += 1

        pending_indices = [index for index, latent in enumerate(self._cached_latents) if latent is None]
        self._last_latent_cache_report = {
            "total": len(self._cached_latents),
            "in_memory_hits": in_memory_hits,
            "disk_hits": disk_hits,
            "encoded": 0,
            "disk_misses": len(pending_indices),
            "cache_dir": self.latent_cache_dir,
        }
        if not pending_indices:
            return

        vae.eval()
        with torch.no_grad():
            for start in range(0, len(pending_indices), batch_size):
                batch_indices = pending_indices[start : start + batch_size]
                images = [self._load_resized_image(index) for index in batch_indices]
                batch = torch.stack(images, dim=0).to(device=device, dtype=dtype)
                latents = vae.encode(batch).latent_dist.sample().to("cpu")
                encoded += len(batch_indices)
                for local_index, sample_index in enumerate(batch_indices):
                    latent = latents[local_index].contiguous().to(dtype=torch.float32)
                    self._cached_latents[sample_index] = latent
                    self._save_cached_latent(sample_index, latent)

        self._last_latent_cache_report = {
            "total": len(self._cached_latents),
            "in_memory_hits": in_memory_hits,
            "disk_hits": disk_hits,
            "encoded": encoded,
            "disk_misses": len(pending_indices),
            "cache_dir": self.latent_cache_dir,
        }

    def _read_image(self, path: str) -> Image.Image:
        return Image.open(path)

    def _read_atlas(self, path: str) -> np.ndarray:
        return np.asarray(Image.open(path))

    def _crop_array(self, array: np.ndarray, crop_box: Tuple[int, int, int, int]) -> np.ndarray:
        crop_x, crop_y, crop_w, crop_h = crop_box
        return array[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

    def _crop_image(self, image: Image.Image, crop_box: Tuple[int, int, int, int]) -> Image.Image:
        crop_x, crop_y, crop_w, crop_h = crop_box
        return image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))

    def _expanded_crop_box(self, crop_box: Tuple[int, int, int, int], halo_px: Optional[int] = None) -> Tuple[int, int, int, int]:
        crop_x, crop_y, crop_w, crop_h = crop_box
        halo = int(self.expanded_target_halo_px if halo_px is None else halo_px)
        if halo <= 0:
            return crop_box
        expanded = (crop_x - halo, crop_y - halo, crop_w + (2 * halo), crop_h + (2 * halo))
        if expanded[0] < 0 or expanded[1] < 0:
            raise ValueError(f"expanded crop exceeds source image bounds: crop_box={crop_box} halo_px={halo}")
        return expanded

    def _expanded_source_extent(
        self,
        crop_box: Tuple[int, int, int, int],
        halo_px: Optional[int] = None,
    ) -> Tuple[float, float, float, float]:
        crop_x, crop_y, crop_w, crop_h = crop_box
        halo = int(self.expanded_target_halo_px if halo_px is None else halo_px)
        if halo <= 0:
            return float(crop_x), float(crop_y), float(crop_w), float(crop_h)

        source_margin_x = float(crop_w) * float(halo) / float(self.train_size[1])
        source_margin_y = float(crop_h) * float(halo) / float(self.train_size[0])
        return (
            float(crop_x) - source_margin_x,
            float(crop_y) - source_margin_y,
            float(crop_w) + (2.0 * source_margin_x),
            float(crop_h) + (2.0 * source_margin_y),
        )

    def _resample_image_extent(
        self,
        image: Image.Image,
        source_extent: Tuple[float, float, float, float],
        out_size: Tuple[int, int],
        resize_mode: str,
    ) -> Image.Image:
        extent_x, extent_y, extent_w, extent_h = source_extent
        if extent_w <= 0.0 or extent_h <= 0.0:
            raise ValueError(f"invalid source extent for resample: {source_extent}")

        return image.transform(
            (int(out_size[1]), int(out_size[0])),
            Image.Transform.EXTENT,
            (float(extent_x), float(extent_y), float(extent_x + extent_w), float(extent_y + extent_h)),
            resample=_resolve_pil_resample(resize_mode),
        )

    def _center_insert_tensor(self, tensor: torch.Tensor, halo_px: int, fill_value: float = 0.0) -> torch.Tensor:
        halo = int(max(0, halo_px))
        if halo <= 0:
            return tensor.contiguous()
        if tensor.ndim == 2:
            out = torch.full(
                (tensor.shape[0] + (2 * halo), tensor.shape[1] + (2 * halo)),
                float(fill_value),
                dtype=tensor.dtype,
            )
            out[halo : halo + tensor.shape[0], halo : halo + tensor.shape[1]] = tensor
            return out.contiguous()
        if tensor.ndim == 3:
            out = torch.full(
                (tensor.shape[0], tensor.shape[1] + (2 * halo), tensor.shape[2] + (2 * halo)),
                float(fill_value),
                dtype=tensor.dtype,
            )
            out[:, halo : halo + tensor.shape[1], halo : halo + tensor.shape[2]] = tensor
            return out.contiguous()
        raise ValueError(f"unexpected tensor rank for center insert: {tuple(tensor.shape)}")

    def _zero_undefined_expanded_halo(
        self,
        rgb_tensor: torch.Tensor,
        alpha_tensor: Optional[torch.Tensor],
        edge_defined_flags: Optional[torch.Tensor],
        halo_px: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        halo = int(max(0, halo_px))
        height = int(rgb_tensor.shape[-2])
        width = int(rgb_tensor.shape[-1])
        zero_mask = torch.zeros((height, width), dtype=torch.float32)
        if halo <= 0 or edge_defined_flags is None:
            return rgb_tensor.contiguous(), None if alpha_tensor is None else alpha_tensor.contiguous(), zero_mask.contiguous()

        flags = edge_defined_flags.detach().float().cpu().view(-1)
        if flags.numel() != 4:
            raise ValueError(f"expected four seam edge flags, got {flags.tolist()}")

        rgb_out = rgb_tensor.clone()
        alpha_out = None if alpha_tensor is None else alpha_tensor.clone()
        if flags[0] < 0.5:
            zero_mask[:halo, :] = 1.0
        if flags[1] < 0.5:
            zero_mask[height - halo :, :] = 1.0
        if flags[2] < 0.5:
            zero_mask[:, width - halo :] = 1.0
        if flags[3] < 0.5:
            zero_mask[:, :halo] = 1.0
        if zero_mask.any():
            expanded_mask = zero_mask.unsqueeze(0)
            rgb_out = torch.where(expanded_mask > 0.5, torch.full_like(rgb_out, -1.0), rgb_out)
            if alpha_out is not None:
                alpha_out = torch.where(zero_mask > 0.5, torch.zeros_like(alpha_out), alpha_out)
        return rgb_out.contiguous(), None if alpha_out is None else alpha_out.contiguous(), zero_mask.contiguous()

    def _extract_channel(self, atlas_arrays: Dict[str, np.ndarray], spec: SemanticChannelSpec) -> torch.Tensor:
        atlas = atlas_arrays[spec.atlas_name]
        if atlas.ndim == 2:
            raw_channel = atlas
        else:
            channel_index = CHANNEL_NAME_TO_INDEX[spec.channel_name]
            if channel_index >= atlas.shape[2]:
                raise ValueError(f"channel {spec.source} is missing from atlas with shape {atlas.shape}")
            raw_channel = atlas[:, :, channel_index]

        # Resolve disk range from original dtype (e.g. uint8 -> 0..255) before float conversion.
        disk_min, disk_max = _resolve_disk_range(raw_channel, spec.disk_range)
        channel = raw_channel.astype(np.float32)
        if math.isclose(disk_max, disk_min):
            normalized = np.zeros_like(channel, dtype=np.float32)
        else:
            normalized = (channel - disk_min) / (disk_max - disk_min)

        semantic_min, semantic_max = spec.semantic_range
        decoded = semantic_min + normalized * (semantic_max - semantic_min)
        if spec.clamp_range is not None:
            clamp_min, clamp_max = spec.clamp_range
            decoded = np.clip(decoded, clamp_min, clamp_max)

        return torch.from_numpy(decoded).float()

    def _load_semantic_tensor(self, index: int) -> torch.Tensor:
        record = self._records[index]
        crop_box = record["crop_box"]
        atlas_arrays = {
            "base": self._crop_array(self._read_atlas(record["base_atlas_path"]), crop_box),
            "edge": self._crop_array(self._read_atlas(record["edge_atlas_path"]), crop_box),
            "interior": self._crop_array(self._read_atlas(record["interior_atlas_path"]), crop_box),
        }

        channels = [self._extract_channel(atlas_arrays, spec) for spec in self.channel_specs]
        tensor = torch.stack(channels, dim=0)
        tensor = _resize_tensor(tensor, self.train_size, self.semantic_resize_mode)
        return tensor.contiguous()

    def _load_expanded_semantic_tensor(self, index: int, halo_px: Optional[int] = None) -> torch.Tensor:
        halo = int(self.expanded_target_halo_px if halo_px is None else halo_px)
        if halo <= 0:
            return self._load_semantic_tensor(index)

        record = self._records[index]
        source_extent = self._expanded_source_extent(record["crop_box"], halo)
        expanded_size = (int(self.train_size[0] + (2 * halo)), int(self.train_size[1] + (2 * halo)))
        atlas_arrays: Dict[str, np.ndarray] = {}
        for atlas_name, path_key in (
            ("base", "base_atlas_path"),
            ("edge", "edge_atlas_path"),
            ("interior", "interior_atlas_path"),
        ):
            with Image.open(record[path_key]) as atlas_image:
                sampled = self._resample_image_extent(
                    atlas_image,
                    source_extent,
                    expanded_size,
                    self.semantic_resize_mode,
                )
                atlas_arrays[atlas_name] = np.asarray(sampled)

        channels = [self._extract_channel(atlas_arrays, spec) for spec in self.channel_specs]
        return torch.stack(channels, dim=0).contiguous()

    def _build_trusted_mask(self, index: int) -> torch.Tensor:
        record = self._records[index]
        _, _, crop_w, crop_h = record["crop_box"]
        trusted_x, trusted_y, trusted_w, trusted_h = record["trusted_box"]

        mask = torch.zeros((crop_h, crop_w), dtype=torch.float32)
        if trusted_w > 0 and trusted_h > 0:
            mask[trusted_y : trusted_y + trusted_h, trusted_x : trusted_x + trusted_w] = 1.0

        resized = _resize_tensor(mask, self.train_size, mode="area").squeeze(0)
        return resized.contiguous()

    def _load_resized_image(self, index: int) -> torch.Tensor:
        image, _ = self._load_resized_image_with_alpha(index)
        return image

    def _load_resized_image_with_alpha(self, index: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        record = self._records[index]
        image = self._read_image(record["image_path"])
        cropped = self._crop_image(image, record["crop_box"])
        if self.enable_alpha_supervision and self.strict_alpha and not record["has_native_alpha"]:
            raise ValueError(f"strict alpha supervision requires native alpha: {record['image_path']}")

        rgb_tensor, alpha_tensor, _ = _to_tensor_image(
            cropped,
            self.train_size,
            self.image_resize_mode,
            return_alpha=self.enable_alpha_supervision,
            synthesize_opaque_alpha=not self.strict_alpha,
        )
        if self.enable_alpha_supervision and alpha_tensor is None:
            raise ValueError(f"alpha supervision enabled but alpha target missing for: {record['image_path']}")
        return rgb_tensor, alpha_tensor

    def _load_expanded_resized_image_with_alpha(
        self,
        index: int,
        edge_defined_flags: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        if self.expanded_target_halo_px <= 0:
            raise ValueError("expanded target requested but expanded_target_halo_px is disabled")

        record = self._records[index]
        halo_px = int(self.expanded_target_halo_px)
        image = self._read_image(record["image_path"])
        expanded_source_extent = self._expanded_source_extent(record["crop_box"], halo_px)
        if self.enable_alpha_supervision and self.strict_alpha and not record["has_native_alpha"]:
            raise ValueError(f"strict alpha supervision requires native alpha: {record['image_path']}")

        expanded_size = (int(self.train_size[0] + (2 * halo_px)), int(self.train_size[1] + (2 * halo_px)))
        cropped = self._resample_image_extent(image.convert("RGBA"), expanded_source_extent, expanded_size, self.image_resize_mode)
        rgba = np.asarray(cropped, dtype=np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgba[:, :, :3]).permute(2, 0, 1).contiguous().float() * 2.0 - 1.0
        alpha_tensor: Optional[torch.Tensor] = None
        if self.enable_alpha_supervision:
            alpha_tensor = torch.from_numpy(rgba[:, :, 3]).contiguous().float()
        rgb_tensor, alpha_tensor, zero_mask = self._zero_undefined_expanded_halo(rgb_tensor, alpha_tensor, edge_defined_flags, halo_px)
        return rgb_tensor, alpha_tensor, zero_mask, torch.tensor(expanded_source_extent, dtype=torch.float32)

    def _load_resized_rgba_tensor(self, index: int) -> torch.Tensor:
        record = self._records[index]
        image = self._read_image(record["image_path"])
        cropped = self._crop_image(image, record["crop_box"])
        resized = cropped.resize((self.train_size[1], self.train_size[0]), resample=_resolve_pil_resample(self.image_resize_mode)).convert("RGBA")
        rgba = np.asarray(resized, dtype=np.float32) / 255.0
        rgb = torch.from_numpy(rgba[:, :, :3]).permute(2, 0, 1).contiguous().float() * 2.0 - 1.0
        alpha = torch.from_numpy(rgba[:, :, 3]).unsqueeze(0).contiguous().float()
        return torch.cat([rgb, alpha], dim=0)

    def _build_seam_decay_maps(self, height: int, width: int) -> torch.Tensor:
        band = max(1, min(self.seam_strip_width_px, (min(height, width) - 1) // 2))
        yy = torch.arange(height, dtype=torch.float32).unsqueeze(1).expand(height, width)
        xx = torch.arange(width, dtype=torch.float32).unsqueeze(0).expand(height, width)

        north = torch.clamp(1.0 - (yy - float(band)) / float(max(1, band)), min=0.0, max=1.0)
        north = north * ((yy >= float(band)) & (yy < float(2 * band))).float()

        south_anchor = float(height - band - 1)
        south = torch.clamp(1.0 - (south_anchor - yy) / float(max(1, band)), min=0.0, max=1.0)
        south = south * ((yy <= south_anchor) & (yy > south_anchor - float(band))).float()

        west = torch.clamp(1.0 - (xx - float(band)) / float(max(1, band)), min=0.0, max=1.0)
        west = west * ((xx >= float(band)) & (xx < float(2 * band))).float()

        east_anchor = float(width - band - 1)
        east = torch.clamp(1.0 - (east_anchor - xx) / float(max(1, band)), min=0.0, max=1.0)
        east = east * ((xx <= east_anchor) & (xx > east_anchor - float(band))).float()
        return torch.stack([north, south, east, west], dim=0).contiguous()

    def _build_seam_geometry_from_flags(
        self,
        edge_defined_flags: torch.Tensor,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        height = int(self.train_size[0] if height is None else height)
        width = int(self.train_size[1] if width is None else width)
        band = max(1, min(self.seam_strip_width_px, (min(height, width) - 1) // 2))
        edge_flag_maps = edge_defined_flags.view(4, 1, 1).expand(4, height, width).contiguous()
        edge_band_masks = torch.zeros((4, height, width), dtype=torch.float32)
        edge_band_masks[0, :band, :] = 1.0
        edge_band_masks[1, height - band :, :] = 1.0
        edge_band_masks[2, :, width - band :] = 1.0
        edge_band_masks[3, :, :band] = 1.0
        seam_decay_maps = self._build_seam_decay_maps(height, width)
        return {
            "edge_defined_flags": edge_defined_flags.contiguous(),
            "edge_flag_maps": edge_flag_maps.contiguous(),
            "edge_band_masks": edge_band_masks.contiguous(),
            "seam_decay_maps": seam_decay_maps.contiguous(),
            "seam_strip_width_px": torch.tensor(float(band), dtype=torch.float32),
        }

    def _build_continuation_valid_mask(
        self,
        conditioning_images: torch.Tensor,
        alpha_target: Optional[torch.Tensor],
        halo_px: int,
    ) -> torch.Tensor:
        if self.terrain_mask_channel_index >= 0:
            continuation_valid_mask = terrain_mask_to_occupancy(
                conditioning_images[self.terrain_mask_channel_index : self.terrain_mask_channel_index + 1],
                self.terrain_mask_black_is_terrain,
            )
            continuation_valid_mask = (continuation_valid_mask >= self.alpha_binary_threshold).float().unsqueeze(0)
        elif alpha_target is not None:
            alpha_mask = alpha_target
            if alpha_mask.ndim == 2:
                alpha_mask = alpha_mask.unsqueeze(0)
            continuation_valid_mask = (alpha_mask >= self.alpha_binary_threshold).float().unsqueeze(0)
        else:
            continuation_valid_mask = torch.ones(
                (1, 1, conditioning_images.shape[-2], conditioning_images.shape[-1]),
                dtype=torch.float32,
            )
        if int(halo_px) > 0:
            continuation_valid_mask = center_embed_spatial_tensor(continuation_valid_mask, int(halo_px), fill_value=0.0)
        return continuation_valid_mask.contiguous()

    def _evaluate_seam_qualification(
        self,
        index: int,
        edge_defined_flags: torch.Tensor,
        conditioning_images: Optional[torch.Tensor] = None,
        alpha_target: Optional[torch.Tensor] = None,
        trusted_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        edge_flags = edge_defined_flags.detach().float().cpu().view(4)
        conditioning_images = self._load_semantic_tensor(index) if conditioning_images is None else conditioning_images.detach().float().cpu()
        if alpha_target is None:
            _, alpha_target = self._load_resized_image_with_alpha(index)
        else:
            alpha_target = alpha_target.detach().float().cpu()
        trusted_mask = self._build_trusted_mask(index) if trusted_mask is None else trusted_mask.detach().float().cpu()

        expanded_halo_px = int(self.expanded_target_halo_px)
        if expanded_halo_px > 0:
            expanded_geometry = self._build_expanded_seam_geometry(edge_flags)
            edge_band_masks = expanded_geometry["expanded_edge_band_masks"].unsqueeze(0)
            seam_decay_maps = expanded_geometry["expanded_seam_decay_maps"].unsqueeze(0)
            seam_strip_width_px = torch.tensor([float(self.seam_strip_width_px)], dtype=torch.float32)
            supervision_mask_base = center_embed_spatial_tensor(trusted_mask.unsqueeze(0).unsqueeze(0), expanded_halo_px, fill_value=0.0)
        else:
            geometry = self._build_seam_geometry_from_flags(edge_flags)
            edge_band_masks = geometry["edge_band_masks"].unsqueeze(0)
            seam_decay_maps = geometry["seam_decay_maps"].unsqueeze(0)
            seam_strip_width_px = geometry["seam_strip_width_px"].view(1)
            supervision_mask_base = trusted_mask.unsqueeze(0).unsqueeze(0)

        continuation_valid_mask = self._build_continuation_valid_mask(
            conditioning_images=conditioning_images,
            alpha_target=alpha_target,
            halo_px=expanded_halo_px,
        )
        edge_defined_batch = edge_flags.unsqueeze(0)
        seam_supervision_mask = build_seam_supervision_mask(
            trusted_mask=supervision_mask_base,
            edge_band_masks=edge_band_masks,
            edge_defined_flags=edge_defined_batch,
            seam_config=self.seam_runtime_config,
        )
        seam_maps = build_seam_region_maps(
            edge_band_masks=edge_band_masks,
            seam_decay_maps=seam_decay_maps,
            edge_defined_flags=edge_defined_batch,
            seam_strip_width_px=seam_strip_width_px,
            supervision_mask=seam_supervision_mask,
            seam_config=self.seam_runtime_config,
            expanded_halo_px=expanded_halo_px,
            continuation_valid_mask=continuation_valid_mask,
        )
        qualification = summarize_seam_edge_qualification(
            seam_maps=seam_maps,
            edge_defined_flags=edge_defined_batch,
            seam_config=self.seam_runtime_config,
        )
        return {
            "seam_qualified_edge_mask": qualification["qualified_edge_mask"].squeeze(0).contiguous(),
            "seam_qualified_continuation_px": qualification["continuation_px"].squeeze(0).contiguous(),
            "seam_qualified_halo_inner_px": qualification["halo_inner_px"].squeeze(0).contiguous(),
            "seam_qualified_halo_outer_px": qualification["halo_outer_px"].squeeze(0).contiguous(),
            "seam_qualified_continuation_weight_sum": qualification["continuation_weight_sum"].squeeze(0).contiguous(),
            "seam_qualified_valid_edges_count": qualification["valid_edges_for_loss_count"].squeeze(0).contiguous(),
        }

    def _get_edge_qualification(self, index: int, edge_idx: int) -> Dict[str, float]:
        key = (int(index), int(edge_idx))
        cached = self._seam_edge_qualification_cache.get(key)
        if cached is not None:
            return cached

        edge_flags = torch.zeros(4, dtype=torch.float32)
        edge_flags[int(edge_idx)] = 1.0
        qualification = self._evaluate_seam_qualification(index, edge_flags)
        stats = {
            "qualified": float(qualification["seam_qualified_edge_mask"][int(edge_idx)].item()),
            "continuation_px": float(qualification["seam_qualified_continuation_px"][int(edge_idx)].item()),
            "halo_inner_px": float(qualification["seam_qualified_halo_inner_px"][int(edge_idx)].item()),
            "halo_outer_px": float(qualification["seam_qualified_halo_outer_px"][int(edge_idx)].item()),
            "continuation_weight_sum": float(qualification["seam_qualified_continuation_weight_sum"][int(edge_idx)].item()),
        }
        self._seam_edge_qualification_cache[key] = stats
        return stats

    def _index_has_any_qualified_edge(self, index: int) -> bool:
        cached = self._seam_record_eligibility_cache.get(int(index))
        if cached is not None:
            return bool(cached)

        if self.seam_fixed_defined_edge_index >= 0:
            eligible = self._get_edge_qualification(index, self.seam_fixed_defined_edge_index)["qualified"] >= 0.5
        else:
            eligible = bool(self._qualified_edges_for_index(index))

        self._seam_record_eligibility_cache[int(index)] = bool(eligible)
        self._manifest_audit["seam_qualification_discoveries"] = len(self._seam_record_eligibility_cache)
        self._manifest_audit["seam_qualified_records"] = int(sum(1 for value in self._seam_record_eligibility_cache.values() if value))
        self._manifest_audit["seam_disqualified_records"] = int(
            sum(1 for value in self._seam_record_eligibility_cache.values() if not value)
        )
        return bool(eligible)

    def _resolve_sample_index(self, index: int, max_attempts: int = 64) -> int:
        if not self.seam_enabled or not bool(self.seam_runtime_config.get("seam_qualified_sampling_enabled", False)):
            return int(index)

        if self._index_has_any_qualified_edge(index):
            return int(index)

        if float(self._sampling_weights.sum().item()) <= 0.0:
            raise ValueError("sampling weights sum to zero while resolving seam-qualified sample index")

        for _ in range(int(max(1, max_attempts))):
            candidate = int(torch.multinomial(self._sampling_weights, num_samples=1, replacement=True).item())
            if self._index_has_any_qualified_edge(candidate):
                return candidate

        raise RuntimeError(
            "failed to resolve a seam-qualified sample index after repeated attempts; "
            + "lower thresholds or disable seam qualification"
        )

    def _qualified_edges_for_index(self, index: int) -> List[int]:
        return [edge_idx for edge_idx in range(4) if self._get_edge_qualification(index, edge_idx)["qualified"] >= 0.5]

    def _sample_seam_flags(self, index: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        if not self.seam_enabled:
            return torch.zeros(4, dtype=torch.float32), 0

        seam_qualified_sampling_enabled = bool(self.seam_runtime_config.get("seam_qualified_sampling_enabled", False))
        eligible_edges: Optional[List[int]] = None
        if seam_qualified_sampling_enabled and index is not None:
            eligible_edges = self._qualified_edges_for_index(index)
            if self.seam_fixed_defined_edge_index >= 0:
                if self.seam_fixed_defined_edge_index in eligible_edges:
                    eligible_edges = [self.seam_fixed_defined_edge_index]
                else:
                    eligible_edges = []
            if not eligible_edges:
                return torch.zeros(4, dtype=torch.float32), 5

        weights = [
            self.seam_state_all_defined_weight,
            self.seam_state_partial_defined_weight,
            self.seam_state_none_defined_weight,
        ]
        total = sum(weights)
        if total <= 0.0:
            if eligible_edges is not None:
                flags = torch.zeros(4, dtype=torch.float32)
                for edge_idx in eligible_edges:
                    flags[int(edge_idx)] = 1.0
                return flags, 1 if len(eligible_edges) == 4 else (2 if len(eligible_edges) == 1 else 3)
            flags = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
            return flags, 1

        draw = self._seam_rng.uniform(0.0, total)
        if draw <= weights[0]:
            if eligible_edges is not None:
                flags = torch.zeros(4, dtype=torch.float32)
                for edge_idx in eligible_edges:
                    flags[int(edge_idx)] = 1.0
                return flags, 1 if len(eligible_edges) == 4 else (2 if len(eligible_edges) == 1 else 3)
            return torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32), 1
        if draw <= (weights[0] + weights[1]):
            partial_draw = self._seam_rng.random()
            if partial_draw <= self.seam_partial_one_edge_ratio:
                if self.seam_fixed_defined_edge_index >= 0:
                    idx = int(self.seam_fixed_defined_edge_index)
                elif eligible_edges is not None:
                    idx = eligible_edges[self._seam_rng.randint(0, len(eligible_edges) - 1)]
                else:
                    idx = self._seam_rng.randint(0, 3)
                flags = torch.zeros(4, dtype=torch.float32)
                flags[idx] = 1.0
                return flags, 2

            adjacent_pairs = [(0, 2), (0, 3), (1, 2), (1, 3)]
            opposite_pairs = [(0, 1), (2, 3)]
            if eligible_edges is not None:
                adjacent_pairs = [pair for pair in adjacent_pairs if pair[0] in eligible_edges and pair[1] in eligible_edges]
                opposite_pairs = [pair for pair in opposite_pairs if pair[0] in eligible_edges and pair[1] in eligible_edges]
                if not adjacent_pairs and not opposite_pairs:
                    idx = eligible_edges[self._seam_rng.randint(0, len(eligible_edges) - 1)]
                    flags = torch.zeros(4, dtype=torch.float32)
                    flags[idx] = 1.0
                    return flags, 2
            if self._seam_rng.random() < 0.7:
                pair = adjacent_pairs[self._seam_rng.randint(0, len(adjacent_pairs) - 1)]
                state = 3
            else:
                pair = opposite_pairs[self._seam_rng.randint(0, len(opposite_pairs) - 1)]
                state = 4
            flags = torch.zeros(4, dtype=torch.float32)
            flags[pair[0]] = 1.0
            flags[pair[1]] = 1.0
            return flags, state

        return torch.zeros(4, dtype=torch.float32), 5

    def _build_seam_features(self, index: int) -> Dict[str, torch.Tensor]:
        height = int(self.train_size[0])
        width = int(self.train_size[1])
        band = max(1, min(self.seam_strip_width_px, (min(height, width) - 1) // 2))
        rgba = self._load_resized_rgba_tensor(index)

        seam_tensor = torch.zeros((16, height, width), dtype=torch.float32)
        seam_tensor[0:4, :band, :] = rgba[:, :band, :]
        seam_tensor[4:8, height - band :, :] = rgba[:, height - band :, :]
        seam_tensor[8:12, :, width - band :] = rgba[:, :, width - band :]
        seam_tensor[12:16, :, :band] = rgba[:, :, :band]

        edge_defined_flags, seam_state_label = self._sample_seam_flags(index)
        seam_geometry = self._build_seam_geometry_from_flags(edge_defined_flags, height=height, width=width)
        edge_flag_maps = seam_geometry["edge_flag_maps"]
        edge_band_masks = seam_geometry["edge_band_masks"]

        undefined_mode = torch.full((4,), 0, dtype=torch.long)
        for edge_idx in range(4):
            if edge_defined_flags[edge_idx] >= 0.5:
                continue
            draw = self._seam_rng.random()
            if draw < self.seam_undefined_zero_prob:
                undefined_mode[edge_idx] = 1
                seam_tensor[(edge_idx * 4) : (edge_idx * 4 + 4)] = 0.0
            elif draw < (self.seam_undefined_zero_prob + self.seam_undefined_noise_prob):
                undefined_mode[edge_idx] = 2
                mask = edge_band_masks[edge_idx].unsqueeze(0)
                seam_tensor[(edge_idx * 4) : (edge_idx * 4 + 4)] = torch.where(
                    mask > 0.5,
                    torch.randn_like(seam_tensor[(edge_idx * 4) : (edge_idx * 4 + 4)]),
                    seam_tensor[(edge_idx * 4) : (edge_idx * 4 + 4)],
                )
            else:
                undefined_mode[edge_idx] = 3

        return {
            "seam_strip_tensor": seam_tensor.contiguous(),
            "edge_defined_flags": edge_defined_flags.contiguous(),
            "edge_flag_maps": edge_flag_maps.contiguous(),
            "edge_band_masks": edge_band_masks.contiguous(),
            "seam_decay_maps": seam_geometry["seam_decay_maps"].contiguous(),
            "seam_state_label": torch.tensor(seam_state_label, dtype=torch.long),
            "seam_undefined_mode": undefined_mode.contiguous(),
            "seam_strip_width_px": torch.tensor(float(band), dtype=torch.float32),
        }

    def _build_expanded_seam_geometry(self, edge_defined_flags: torch.Tensor) -> Dict[str, torch.Tensor]:
        halo_px = int(self.expanded_target_halo_px)
        if halo_px <= 0:
            return {}
        expanded_height = int(self.train_size[0] + (2 * halo_px))
        expanded_width = int(self.train_size[1] + (2 * halo_px))
        edge_band_masks = torch.zeros((4, expanded_height, expanded_width), dtype=torch.float32)
        edge_band_masks[0, :halo_px, :] = 1.0
        edge_band_masks[1, expanded_height - halo_px :, :] = 1.0
        edge_band_masks[2, :, expanded_width - halo_px :] = 1.0
        edge_band_masks[3, :, :halo_px] = 1.0
        seam_decay_maps = self._build_seam_decay_maps(expanded_height, expanded_width)
        return {
            "expanded_edge_band_masks": edge_band_masks.contiguous(),
            "expanded_seam_decay_maps": seam_decay_maps.contiguous(),
            "expanded_edge_defined_flags": edge_defined_flags.contiguous(),
        }

    def _build_expanded_seam_strip(
        self,
        expanded_rgba: torch.Tensor,
        edge_defined_flags: torch.Tensor,
        halo_px: int,
        exclude_corners: bool = True,
    ) -> torch.Tensor:
        """Build a 16-channel seam strip at expanded size using actual halo pixel data.

        For each enabled edge the corresponding 4-channel RGBA band (width=halo_px) is filled
        with pixels from the expanded image.  Disabled edges remain zero.  Corner regions
        (overlap of two orthogonal halo bands) are zeroed when exclude_corners=True so the
        model never receives ambiguous corner pixels as a seam cue.  The interior region
        (inside all four halo bands) is always zero; only halo pixels are populated.
        """
        exp_H = expanded_rgba.shape[1]
        exp_W = expanded_rgba.shape[2]
        seam_tensor = torch.zeros((16, exp_H, exp_W), dtype=torch.float32)

        # Corner-exclusion mask: 1 inside non-corner halo regions, 0 at corners.
        corner_mask = torch.ones((exp_H, exp_W), dtype=torch.float32)
        if exclude_corners:
            corner_mask[:halo_px, :halo_px] = 0.0          # NW
            corner_mask[:halo_px, exp_W - halo_px :] = 0.0  # NE
            corner_mask[exp_H - halo_px :, :halo_px] = 0.0  # SW
            corner_mask[exp_H - halo_px :, exp_W - halo_px :] = 0.0  # SE

        # North (edge 0, channels 0:4): top halo_px rows, corners excluded.
        if edge_defined_flags[0] >= 0.5:
            band = expanded_rgba[:, :halo_px, :]  # (4, halo_px, exp_W)
            mask = corner_mask[:halo_px, :].unsqueeze(0)  # (1, halo_px, exp_W)
            seam_tensor[0:4, :halo_px, :] = band * mask

        # South (edge 1, channels 4:8): bottom halo_px rows, corners excluded.
        if edge_defined_flags[1] >= 0.5:
            band = expanded_rgba[:, exp_H - halo_px :, :]
            mask = corner_mask[exp_H - halo_px :, :].unsqueeze(0)
            seam_tensor[4:8, exp_H - halo_px :, :] = band * mask

        # East (edge 2, channels 8:12): rightmost halo_px columns, corners excluded.
        if edge_defined_flags[2] >= 0.5:
            band = expanded_rgba[:, :, exp_W - halo_px :]
            mask = corner_mask[:, exp_W - halo_px :].unsqueeze(0)
            seam_tensor[8:12, :, exp_W - halo_px :] = band * mask

        # West (edge 3, channels 12:16): leftmost halo_px columns, corners excluded.
        if edge_defined_flags[3] >= 0.5:
            band = expanded_rgba[:, :, :halo_px]
            mask = corner_mask[:, :halo_px].unsqueeze(0)
            seam_tensor[12:16, :, :halo_px] = band * mask

        return seam_tensor.contiguous()

    def build_expanded_target_diagnostic(self, index: int, edge_defined_flags: Sequence[float]) -> Dict[str, object]:
        flags = torch.tensor([float(v) for v in edge_defined_flags], dtype=torch.float32)
        if flags.numel() != 4:
            raise ValueError(f"expected four edge flags for diagnostic, got {list(edge_defined_flags)}")

        interior_rgb, interior_alpha = self._load_resized_image_with_alpha(index)
        expanded_rgb, expanded_alpha, zero_mask, expanded_crop_box = self._load_expanded_resized_image_with_alpha(index, flags)
        semantic = self._load_semantic_tensor(index)
        crop_h, crop_w = int(interior_rgb.shape[-2]), int(interior_rgb.shape[-1])
        exp_h, exp_w = int(expanded_rgb.shape[-2]), int(expanded_rgb.shape[-1])
        halo = int(self.expanded_target_halo_px)
        if tuple(semantic.shape[-2:]) != (crop_h, crop_w):
            raise ValueError(
                "semantic mask crop shape mismatch: "
                + f"semantic={tuple(semantic.shape[-2:])} crop={(crop_h, crop_w)}"
            )
        if (exp_h, exp_w) != (crop_h + (2 * halo), crop_w + (2 * halo)):
            raise ValueError(
                "expanded target shape mismatch: "
                + f"expanded={(exp_h, exp_w)} expected={(crop_h + (2 * halo), crop_w + (2 * halo))}"
            )
        return {
            "interior_rgb": interior_rgb,
            "interior_alpha": interior_alpha,
            "expanded_rgb": expanded_rgb,
            "expanded_alpha": expanded_alpha,
            "semantic_tensor": semantic,
            "edge_defined_flags": flags.contiguous(),
            "zero_mask": zero_mask,
            "crop_shape_hw": torch.tensor([crop_h, crop_w], dtype=torch.long),
            "expanded_shape_hw": torch.tensor([exp_h, exp_w], dtype=torch.long),
            "expanded_crop_box": expanded_crop_box,
            "crop_box": torch.tensor(self._records[index]["crop_box"], dtype=torch.long),
            "image_name": self._records[index]["image_name"],
            "channel_names": self.channel_names,
        }

    def build_debug_example(self, index: int) -> Dict[str, object]:
        sample = self[index]
        return {
            "image_name": sample["image_name"],
            "channel_names": self.channel_names,
            "full_conditioning_channel_names": self.full_conditioning_channel_names,
            "crop_box": sample["crop_box"],
            "trusted_box": sample["trusted_box"],
            "trusted_ratio": sample["trusted_ratio"],
            "images": sample["images"],
            "alpha_target": sample["alpha_target"],
            "conditioning_images": sample["conditioning_images"],
            "trusted_mask": sample["trusted_mask"],
            "seam_strip_tensor": sample.get("seam_strip_tensor"),
            "edge_defined_flags": sample.get("edge_defined_flags"),
            "edge_flag_maps": sample.get("edge_flag_maps"),
            "edge_band_masks": sample.get("edge_band_masks"),
            "seam_decay_maps": sample.get("seam_decay_maps"),
            "seam_state_label": sample.get("seam_state_label"),
            "seam_undefined_mode": sample.get("seam_undefined_mode"),
            "boundary_alignment_error": sample.get("boundary_alignment_error"),
            "boundary_consistency_error": sample.get("boundary_consistency_error"),
            "expanded_images": sample.get("expanded_images"),
            "expanded_alpha_target": sample.get("expanded_alpha_target"),
            "expanded_trusted_mask": sample.get("expanded_trusted_mask"),
            "expanded_target_sizes_hw": sample.get("expanded_target_sizes_hw"),
            "expanded_edge_band_masks": sample.get("expanded_edge_band_masks"),
            "expanded_seam_decay_maps": sample.get("expanded_seam_decay_maps"),
            "expanded_seam_strip_tensor": sample.get("expanded_seam_strip_tensor"),
            "expanded_zero_mask": sample.get("expanded_zero_mask"),
            "expanded_crop_box": sample.get("expanded_crop_box"),
        }

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        index = self._resolve_sample_index(index)
        record = self._records[index]
        crop_x, crop_y, crop_w, crop_h = record["crop_box"]

        if self._cached_latents[index] is None:
            self._cached_latents[index] = self._load_cached_latent(index)

        images, alpha_target = self._load_resized_image_with_alpha(index)
        conditioning_images = self._load_semantic_tensor(index)
        trusted_mask = self._build_trusted_mask(index)
        seam_features: Dict[str, torch.Tensor] = {}
        if self.seam_enabled:
            seam_features = self._build_seam_features(index)

        expanded_images: Optional[torch.Tensor] = None
        expanded_alpha_target: Optional[torch.Tensor] = None
        expanded_trusted_mask: Optional[torch.Tensor] = None
        expanded_target_sizes_hw: Optional[torch.Tensor] = None
        expanded_crop_box: Optional[torch.Tensor] = None
        expanded_zero_mask: Optional[torch.Tensor] = None
        expanded_geometry: Dict[str, torch.Tensor] = {}
        if self.expanded_target_halo_px > 0:
            expanded_images, expanded_alpha_target, expanded_zero_mask, expanded_crop_box = self._load_expanded_resized_image_with_alpha(
                index,
                seam_features.get("edge_defined_flags"),
            )
            expanded_trusted_mask = self._center_insert_tensor(trusted_mask, self.expanded_target_halo_px, fill_value=0.0)
            expanded_target_sizes_hw = torch.tensor(
                [self.train_size[0] + (2 * self.expanded_target_halo_px), self.train_size[1] + (2 * self.expanded_target_halo_px)],
                dtype=torch.long,
            )
            if tuple(conditioning_images.shape[-2:]) != tuple(images.shape[-2:]):
                raise ValueError(
                    "semantic mask crop must match original crop exactly: "
                    + f"semantic={tuple(conditioning_images.shape[-2:])} crop={tuple(images.shape[-2:])}"
                )
            if tuple(expanded_images.shape[-2:]) != tuple(expanded_target_sizes_hw.tolist()):
                raise ValueError(
                    "expanded target shape mismatch after load: "
                    + f"expanded={tuple(expanded_images.shape[-2:])} expected={tuple(expanded_target_sizes_hw.tolist())}"
                )
            if self.seam_enabled and seam_features.get("edge_defined_flags") is not None:
                expanded_geometry = self._build_expanded_seam_geometry(seam_features["edge_defined_flags"])
                if expanded_images is not None:
                    if expanded_alpha_target is None:
                        _exp_alpha = torch.ones(
                            1,
                            expanded_images.shape[1],
                            expanded_images.shape[2],
                            dtype=torch.float32,
                        )
                    else:
                        _exp_alpha = expanded_alpha_target
                        if _exp_alpha.ndim == 2:
                            _exp_alpha = _exp_alpha.unsqueeze(0)
                        elif _exp_alpha.ndim == 3 and _exp_alpha.shape[0] != 1:
                            _exp_alpha = _exp_alpha[:1]
                    _expanded_rgba = torch.cat([expanded_images, _exp_alpha], dim=0)
                    expanded_geometry["expanded_seam_strip_tensor"] = self._build_expanded_seam_strip(
                        _expanded_rgba,
                        seam_features["edge_defined_flags"],
                        int(self.expanded_target_halo_px),
                        exclude_corners=True,
                    )

        seam_qualification: Dict[str, torch.Tensor] = {}
        if self.seam_enabled and seam_features.get("edge_defined_flags") is not None:
            seam_qualification = self._evaluate_seam_qualification(
                index=index,
                edge_defined_flags=seam_features["edge_defined_flags"],
                conditioning_images=conditioning_images,
                alpha_target=alpha_target,
                trusted_mask=trusted_mask,
            )

        example: Dict[str, object] = {
            "image_name": record["image_name"],
            "images": images,
            "alpha_target": alpha_target,
            "alpha_has_native": torch.tensor(1.0 if record["has_native_alpha"] else 0.0, dtype=torch.float32),
            "conditioning_images": conditioning_images,
            "trusted_mask": trusted_mask,
            "latents": self._cached_latents[index],
            "prompt": self.prompt,
            "prompt2": self.prompt2,
            "original_sizes_hw": torch.tensor([crop_h, crop_w], dtype=torch.long),
            "crop_top_lefts": torch.tensor([0, 0], dtype=torch.long),
            "target_sizes_hw": torch.tensor([self.train_size[0], self.train_size[1]], dtype=torch.long),
            "crop_box": torch.tensor([crop_x, crop_y, crop_w, crop_h], dtype=torch.long),
            "trusted_box": torch.tensor(record["trusted_box"], dtype=torch.long),
            "trusted_ratio": torch.tensor(record["trusted_ratio"], dtype=torch.float32),
            "sampling_weight": torch.tensor(record["sampling_weight"], dtype=torch.float32),
            "channel_names": self.channel_names,
            "full_conditioning_channel_names": self.full_conditioning_channel_names,
            "special_structure_tags": record["special_structure_tags"],
            "assigned_crop_class": record.get("assigned_crop_class", ""),
            "crop_size_class": record["crop_size_class"],
            "generation_strategy": record["generation_strategy"],
            "seam_enabled": torch.tensor(1.0 if self.seam_enabled else 0.0, dtype=torch.float32),
            "boundary_alignment_error": torch.tensor(record.get("boundary_alignment_error", 0.0), dtype=torch.float32),
            "boundary_consistency_error": torch.tensor(record.get("boundary_consistency_error", 0.0), dtype=torch.float32),
            "expanded_images": expanded_images,
            "expanded_alpha_target": expanded_alpha_target,
            "expanded_trusted_mask": expanded_trusted_mask,
            "expanded_target_sizes_hw": expanded_target_sizes_hw,
            "expanded_zero_mask": expanded_zero_mask,
            "expanded_crop_box": expanded_crop_box,
        }
        example.update(seam_features)
        example.update(expanded_geometry)
        example.update(seam_qualification)
        return example
