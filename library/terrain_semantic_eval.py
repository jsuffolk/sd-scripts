from library.terrain_semantic_seam_geometry import center_crop_chw, center_crop_hw, expanded_hw, pad_chw_spatial
import csv
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from PIL import Image, ImageDraw, ImageFilter

from library import sdxl_model_util, sdxl_train_util
from library.device_utils import clean_memory_on_device
from library.terrain_semantic_conditioning import (
    ModelVisibleConditioningSpec,
    build_model_visible_conditioning_spec,
    compose_sample_aware_model_visible_conditioning,
    compose_model_visible_conditioning,
)
from library.terrain_semantic_manifest_dataset import (
    build_seam_region_maps as shared_build_seam_region_maps,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    eval_id: str
    category: str
    sample_key: str
    dataset_index: int
    image_name: str
    crop_box: Tuple[int, int, int, int]
    generation_strategy: str


@dataclass
class SwapPair:
    pair_id: str
    base_image: str
    base_sample_key: str
    base_dataset_index: int
    swap_image: str
    swap_sample_key: str
    swap_dataset_index: int
    edit_type: str                   # "global" or "local"
    primary_expected_effect: str
    allowed_effects: str
    disallowed_effects: str
    edit_mask_path: Optional[str]    # only set for local edits


def build_sample_key(image_name: str, crop_box: Sequence[int]) -> str:
    x, y, w, h = [int(v) for v in crop_box]
    safe_name = image_name.replace("/", "_").replace(" ", "_")
    return f"{safe_name}__x{x}_y{y}_w{w}_h{h}"


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().float().clamp(-1.0, 1.0)
    array = (array + 1.0) * 0.5
    array = (array * 255.0).round().to(torch.uint8)
    array = array.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def _mask_to_image(mask: torch.Tensor) -> Image.Image:
    arr = mask.detach().float().clamp(0.0, 1.0).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _float_to_grayscale_image(mask: torch.Tensor) -> Image.Image:
    arr = mask.detach().float().cpu().numpy()
    if np.isclose(arr.max(), arr.min()):
        norm = np.zeros_like(arr, dtype=np.float32)
    else:
        norm = (arr - arr.min()) / (arr.max() - arr.min())
    return Image.fromarray((norm * 255.0).round().astype(np.uint8), mode="L")


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x1 = x.flatten().float()
    y1 = y.flatten().float()
    x1 = x1 - x1.mean()
    y1 = y1 - y1.mean()
    denom = (x1.norm() * y1.norm()).item()
    if denom <= 1e-8:
        return 0.0
    return float((x1 * y1).sum().item() / denom)


def _speckle_ratio(prob: torch.Tensor) -> float:
    # Ratio of Laplacian energy to total energy as a simple high-frequency noise proxy.
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=prob.device,
        dtype=prob.dtype,
    ).view(1, 1, 3, 3)
    p = prob.unsqueeze(0).unsqueeze(0)
    lap = F.conv2d(p, kernel, padding=1).squeeze(0).squeeze(0)
    lap_energy = float((lap * lap).mean().item())
    total_energy = float((prob * prob).mean().item())
    if total_energy <= 1e-8:
        return 0.0
    return lap_energy / total_energy

# ---------------------------------------------------------------------------
# Semantic binding diagnostic helpers
# ---------------------------------------------------------------------------

def _compute_rgb_diff(
    img_a: Image.Image,
    img_b: Image.Image,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (abs_diff, signed_diff) as float32 numpy arrays in [0,255] and [-255,255]."""
    a = np.asarray(img_a.convert("RGB"), dtype=np.float32)
    b = np.asarray(img_b.convert("RGB"), dtype=np.float32)
    signed = a - b
    return np.abs(signed), signed


def _abs_diff_to_image(abs_diff: np.ndarray) -> Image.Image:
    """Convert abs diff float array [0,255] to a uint8 PIL RGB image."""
    return Image.fromarray(abs_diff.clip(0, 255).astype(np.uint8), mode="RGB")


def _signed_diff_to_image(signed_diff: np.ndarray) -> Image.Image:
    """Map signed diff [-255,255] to [0,255] with neutral grey at 128."""
    mapped = ((signed_diff + 255.0) * 0.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(mapped, mode="RGB")


def _smooth_edge_map(rgb_img: Image.Image, blur_radius: float = 1.2) -> np.ndarray:
    """Compute a Sobel edge magnitude map on a pre-blurred grayscale image.

    The Gaussian pre-blur suppresses high-frequency texture so that only
    structural edges contribute to the variance measurement.  Returns a
    float32 array in [0, 1] with the same spatial dimensions as rgb_img.
    """
    blurred = rgb_img.convert("L").filter(ImageFilter.GaussianBlur(radius=blur_radius))
    gray = np.asarray(blurred, dtype=np.float32) / 255.0
    padded = np.pad(gray, 1, mode="reflect")
    gx = (
        -padded[:-2, :-2] + padded[:-2, 2:]
        - 2.0 * padded[1:-1, :-2] + 2.0 * padded[1:-1, 2:]
        - padded[2:, :-2] + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2] - 2.0 * padded[:-2, 1:-1] - padded[:-2, 2:]
        + padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:]
    )
    mag = np.hypot(gx, gy)
    peak = mag.max()
    if peak > 0.0:
        mag = mag / peak
    return mag.astype(np.float32)


def _edge_map_variance(edge_maps: List[np.ndarray]) -> float:
    """Mean per-pixel variance of smooth edge maps across a list of renders."""
    if len(edge_maps) < 2:
        return 0.0
    stack = np.stack(edge_maps, axis=0)  # (N, H, W)
    return float(np.mean(np.var(stack, axis=0)))


def _compute_localization_score(
    abs_diff: np.ndarray,
    region_mask: np.ndarray,
) -> Tuple[float, float]:
    """Return (localization_score, total_mean_diff_per_pixel).

    localization_score = fraction of total diff energy inside region_mask.
    total_mean_diff_per_pixel = mean(abs_diff) across all pixels/channels.
    """
    per_pixel = abs_diff.mean(axis=2)  # (H, W)
    total_energy = float(per_pixel.sum())
    inside_energy = float((per_pixel * region_mask).sum())
    loc_score = inside_energy / max(total_energy, 1e-8)
    total_mean_diff = float(per_pixel.mean())
    return loc_score, total_mean_diff


def _normalize_diff(abs_diff: np.ndarray, ref_rgb: np.ndarray) -> float:
    """Normalized mean diff: mean(abs_diff) / mean(|ref_rgb|), both in [0, 255].

    Removes the confound of later checkpoints producing sharper outputs.
    Returns 0 if reference intensity is near zero.
    """
    mean_diff = float(abs_diff.mean())
    mean_intensity = float(np.abs(ref_rgb).mean())
    if mean_intensity < 1.0:
        return 0.0
    return mean_diff / mean_intensity


def _make_cond_override(
    cond: torch.Tensor,
    mode: str,
    terrain_mask_channel_index: int,
    rng_seed: int = 9999,
) -> torch.Tensor:
    """Build a modified conditioning tensor for eval-only ablation.

    mode:
        "zero"       — all channels zeroed.
        "mask_only"  — keep terrain_mask channel; zero all others.
        "shuffled"   — spatially shuffle all channels with a fixed seed.
        "nullspace"  — keep terrain_mask; spatially shuffle all other channels.
    """
    if mode == "zero":
        return torch.zeros_like(cond)

    if mode == "mask_only":
        result = torch.zeros_like(cond)
        idx = terrain_mask_channel_index
        result[:, idx : idx + 1] = cond[:, idx : idx + 1]
        return result

    # shuffled / nullspace: build a fixed spatial permutation on CPU then apply
    B, C, H, W = cond.shape
    gen = torch.Generator(device="cpu")
    gen.manual_seed(rng_seed)
    perm = torch.randperm(H * W, generator=gen)  # CPU

    flat = cond.reshape(B, C, -1).cpu()  # (B, C, H*W)
    shuffled_flat = flat[:, :, perm]
    shuffled = shuffled_flat.reshape(B, C, H, W).to(device=cond.device, dtype=cond.dtype)

    if mode == "shuffled":
        return shuffled

    if mode == "nullspace":
        result = cond.clone()
        for ch in range(C):
            if ch != terrain_mask_channel_index:
                result[:, ch] = shuffled[:, ch]
        return result

    raise ValueError(f"Unknown cond_override mode: '{mode}'")


def _histogram_counts(values: torch.Tensor, bin_edges: List[float]) -> List[int]:
    flat = values.detach().float().flatten().cpu().numpy()
    counts, _ = np.histogram(flat, bins=np.asarray(bin_edges, dtype=np.float32))
    return [int(v) for v in counts.tolist()]


def _expand_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    return F.max_pool2d(mask, kernel_size=(radius * 2) + 1, stride=1, padding=radius)


def _terrain_mask_to_occupancy(mask: torch.Tensor, black_is_terrain: bool) -> torch.Tensor:
    mask = mask.detach().float() if not mask.is_floating_point() else mask.float()
    mask = mask.clamp(0.0, 1.0)
    return (1.0 - mask) if black_is_terrain else mask


def _select_eval_alpha_logits(alpha_outputs: Optional[Dict[str, object]], output_source: str) -> torch.Tensor:
    if alpha_outputs is None:
        raise RuntimeError("alpha outputs are missing while selecting eval alpha logits")

    source = str(output_source).strip().lower()
    if source in {"main", "fused", "fused_logits"}:
        logits = alpha_outputs.get("fused_logits")
    elif source in {"terrain_mask", "terrain_baseline", "terrain_mask_baseline"}:
        logits = (alpha_outputs.get("baseline_logits") or {}).get("terrain_mask")
    elif source in {"pre_stem", "prestem", "pre_stem_baseline"}:
        logits = (alpha_outputs.get("baseline_logits") or {}).get("pre_stem")
    else:
        raise ValueError(f"unsupported evaluation alpha_output_source='{output_source}'")

    if logits is None:
        available_baselines = sorted(list((alpha_outputs.get("baseline_logits") or {}).keys()))
        raise RuntimeError(
            f"requested eval alpha output source '{output_source}' is unavailable; "
            f"available_baselines={available_baselines}"
        )
    return logits


def _compose_model_visible_conditioning(
    sample: Dict[str, object],
    base_conditioning: torch.Tensor,
    conditioning_spec: ModelVisibleConditioningSpec,
    expanded_conditioning_mode: str = "real_expanded",
) -> torch.Tensor:
    if not isinstance(base_conditioning, torch.Tensor):
        return base_conditioning
    full_conditioning, _ = compose_sample_aware_model_visible_conditioning(
        sample=sample,
        spec=conditioning_spec,
        base_conditioning=base_conditioning.float(),
        expanded_conditioning_mode=expanded_conditioning_mode,
    )
    return full_conditioning


def _spatial_shuffle_channels(cond: torch.Tensor, channels: List[int], seed: int) -> torch.Tensor:
    """Spatially shuffle selected channels of a (C,H,W) tensor with deterministic seed."""
    result = cond.clone()
    if not channels:
        return result
    c, h, w = cond.shape
    flat = cond.reshape(c, -1).cpu()
    gen = torch.Generator(device="cpu")
    for i, ch in enumerate(channels):
        if ch < 0 or ch >= c:
            continue
        gen.manual_seed(int(seed) + i)
        perm = torch.randperm(h * w, generator=gen)
        result[ch] = flat[ch, perm].view(h, w).to(device=cond.device, dtype=cond.dtype)
    return result


def _masked_mean_abs_diff(img_a: Image.Image, img_b: Image.Image, mask: np.ndarray) -> float:
    """Mean absolute RGB difference under mask, normalized to [0,1]."""
    a = np.asarray(img_a.convert("RGB"), dtype=np.float32)
    b = np.asarray(img_b.convert("RGB"), dtype=np.float32)
    per_pixel = np.mean(np.abs(a - b), axis=2) / 255.0
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    m = np.clip(m, 0.0, 1.0)
    denom = float(np.sum(m))
    if denom <= 1e-8:
        return float(np.mean(per_pixel))
    return float(np.sum(per_pixel * m) / denom)


def _to_luma(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0]) + (0.587 * rgb[..., 1]) + (0.114 * rgb[..., 2])


def _gradients_2d(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gy, gx = np.gradient(x.astype(np.float32))
    return gx, gy


def _mean_gradient_cosine(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    ax, ay = _gradients_2d(a)
    bx, by = _gradients_2d(b)
    dot = (ax * bx) + (ay * by)
    na = np.sqrt((ax * ax) + (ay * ay) + 1e-8)
    nb = np.sqrt((bx * bx) + (by * by) + 1e-8)
    cosine = dot / (na * nb + 1e-8)
    m = np.asarray(mask, dtype=np.float32)
    denom = float(np.sum(m))
    if denom <= 1e-8:
        return 0.0
    return float(np.sum(cosine * m) / denom)


def _masked_l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=np.float32)
    denom = float(np.sum(m))
    if denom <= 1e-8:
        return 0.0
    diff = np.mean(np.abs(a - b), axis=2)
    return float(np.sum(diff * m) / denom)


def _build_expected_expanded_seam_rgba(
    sample: Dict[str, object],
    interior_h: int,
    interior_w: int,
    halo_px: int,
) -> Optional[np.ndarray]:
    explicit_expanded_target = sample.get("expanded_target_rgba")
    seam_strip = sample.get("seam_strip_tensor")
    expanded_seam_strip = sample.get("expanded_seam_strip_tensor")
    expanded_images = sample.get("expanded_images")
    edge_defined = sample.get("edge_defined_flags")
    if seam_strip is None or edge_defined is None:
        if expanded_seam_strip is None or edge_defined is None:
            return None
    if not isinstance(edge_defined, torch.Tensor):
        return None

    halo = int(max(1, halo_px))
    exp_h, exp_w = expanded_hw(interior_h, interior_w, halo)
    expected = np.zeros((exp_h, exp_w, 4), dtype=np.float32)

    flags = edge_defined.detach().float().cpu().numpy()

    def to_01(x: np.ndarray) -> np.ndarray:
        rgb = np.clip((x[:3] + 1.0) * 0.5, 0.0, 1.0)
        a = np.clip(x[3:4], 0.0, 1.0)
        return np.concatenate([rgb, a], axis=0)

    if isinstance(explicit_expanded_target, torch.Tensor):
        target_rgba = explicit_expanded_target.detach().float().cpu().numpy()
        if target_rgba.shape[0] == 4 and tuple(int(v) for v in target_rgba.shape[-2:]) == (exp_h, exp_w):
            return np.clip(target_rgba.transpose(1, 2, 0), 0.0, 1.0)

    def _build_projected_expected_from_expanded_source(source_rgba: np.ndarray) -> np.ndarray:
        projected = np.zeros((exp_h, exp_w, 4), dtype=np.float32)

        if flags[0] >= 0.5:
            projected[:halo, :, :] = source_rgba[:halo, :, :]
        if flags[1] >= 0.5:
            projected[exp_h - halo :, :, :] = source_rgba[exp_h - halo :, :, :]
        if flags[2] >= 0.5:
            projected[:, exp_w - halo :, :] = source_rgba[:, exp_w - halo :, :]
        if flags[3] >= 0.5:
            projected[:, :halo, :] = source_rgba[:, :halo, :]

        interior_sum = np.zeros((interior_h, interior_w, 4), dtype=np.float32)
        interior_weight = np.zeros((interior_h, interior_w, 1), dtype=np.float32)

        if flags[0] >= 0.5:
            north_src = source_rgba[:halo, halo : halo + interior_w, :].mean(axis=0, keepdims=True)
            north_fill = np.repeat(north_src, halo, axis=0)
            interior_sum[:halo, :, :] += north_fill
            interior_weight[:halo, :, :] += 1.0
        if flags[1] >= 0.5:
            south_src = source_rgba[exp_h - halo :, halo : halo + interior_w, :].mean(axis=0, keepdims=True)
            south_fill = np.repeat(south_src, halo, axis=0)
            interior_sum[interior_h - halo :, :, :] += south_fill
            interior_weight[interior_h - halo :, :, :] += 1.0
        if flags[2] >= 0.5:
            east_src = source_rgba[halo : halo + interior_h, exp_w - halo :, :].mean(axis=1, keepdims=True)
            east_fill = np.repeat(east_src, halo, axis=1)
            interior_sum[:, interior_w - halo :, :] += east_fill
            interior_weight[:, interior_w - halo :, :] += 1.0
        if flags[3] >= 0.5:
            west_src = source_rgba[halo : halo + interior_h, :halo, :].mean(axis=1, keepdims=True)
            west_fill = np.repeat(west_src, halo, axis=1)
            interior_sum[:, :halo, :] += west_fill
            interior_weight[:, :halo, :] += 1.0

        valid = interior_weight > 0.0
        if np.any(valid):
            interior_projected = interior_sum / np.maximum(interior_weight, 1e-6)
            valid_rgb = valid[..., 0]
            interior_projected[~valid_rgb] = 0.0
            projected_interior = projected[halo : halo + interior_h, halo : halo + interior_w, :]
            projected_interior[valid_rgb] = interior_projected[valid_rgb]
            projected[halo : halo + interior_h, halo : halo + interior_w, :] = projected_interior

        return projected

    if isinstance(expanded_images, torch.Tensor):
        expanded_rgb = expanded_images.detach().float().cpu()
        if tuple(int(v) for v in expanded_rgb.shape[-2:]) == (exp_h, exp_w):
            expanded_alpha = sample.get("expanded_alpha_target")
            if isinstance(expanded_alpha, torch.Tensor):
                alpha = expanded_alpha.detach().float().cpu()
                if alpha.ndim == 2:
                    alpha = alpha.unsqueeze(0)
                elif alpha.ndim == 3 and alpha.shape[0] != 1:
                    alpha = alpha[:1]
            else:
                alpha = torch.ones((1, exp_h, exp_w), dtype=torch.float32)
            source_rgba = np.concatenate(
                [
                    np.clip(((expanded_rgb.numpy() + 1.0) * 0.5).transpose(1, 2, 0), 0.0, 1.0),
                    np.clip(alpha.numpy().transpose(1, 2, 0), 0.0, 1.0),
                ],
                axis=2,
            )
            return _build_projected_expected_from_expanded_source(source_rgba)

    if isinstance(expanded_seam_strip, torch.Tensor):
        expanded_strip = expanded_seam_strip.detach().float().cpu().numpy()
        if expanded_strip.shape[-2:] == (exp_h, exp_w):
            for edge_idx in range(4):
                if edge_idx >= len(flags) or flags[edge_idx] < 0.5:
                    continue
                edge_rgba = to_01(expanded_strip[edge_idx * 4 : edge_idx * 4 + 4]).transpose(1, 2, 0)
                expected = np.maximum(expected, edge_rgba)
            return expected

    if not isinstance(seam_strip, torch.Tensor):
        return None
    strip = seam_strip.detach().float().cpu().numpy()

    band = int(min(halo, strip.shape[1], strip.shape[2]))
    if band <= 0:
        return expected

    # north
    if flags[0] >= 0.5:
        north = to_01(strip[0:4, :band, :]).transpose(1, 2, 0)
        expected[halo - band : halo, halo : halo + interior_w, :] = north[:band, :interior_w, :]
    # south
    if flags[1] >= 0.5:
        south = to_01(strip[4:8, interior_h - band : interior_h, :]).transpose(1, 2, 0)
        expected[halo + interior_h : halo + interior_h + band, halo : halo + interior_w, :] = south[:band, :interior_w, :]
    # east
    if flags[2] >= 0.5:
        east = to_01(strip[8:12, :, interior_w - band : interior_w]).transpose(1, 2, 0)
        expected[halo : halo + interior_h, halo + interior_w : halo + interior_w + band, :] = east[:interior_h, :band, :]
    # west
    if flags[3] >= 0.5:
        west = to_01(strip[12:16, :, :band]).transpose(1, 2, 0)
        expected[halo : halo + interior_h, halo - band : halo, :] = west[:interior_h, :band, :]

    return expected


def _build_expanded_edge_masks(height: int, width: int, halo_px: int, band_px: int) -> Dict[str, Dict[str, np.ndarray]]:
    h = int(height)
    w = int(width)
    halo = int(max(1, halo_px))
    band = int(max(1, min(band_px, halo)))
    masks: Dict[str, Dict[str, np.ndarray]] = {}

    def z() -> np.ndarray:
        return np.zeros((h, w), dtype=np.float32)

    yy = np.arange(h, dtype=np.float32).reshape(h, 1)
    xx = np.arange(w, dtype=np.float32).reshape(1, w)
    interior_min_y = float(halo)
    interior_max_y = float(h - 1 - halo)
    interior_min_x = float(halo)
    interior_max_x = float(w - 1 - halo)

    north_dist_outside = np.clip(interior_min_y - yy, 0.0, None)
    south_dist_outside = np.clip(yy - interior_max_y, 0.0, None)
    east_dist_outside = np.clip(xx - interior_max_x, 0.0, None)
    west_dist_outside = np.clip(interior_min_x - xx, 0.0, None)
    north_dist_inside = np.clip(yy - interior_min_y, 0.0, None)
    south_dist_inside = np.clip(interior_max_y - yy, 0.0, None)
    east_dist_inside = np.clip(interior_max_x - xx, 0.0, None)
    west_dist_inside = np.clip(xx - interior_min_x, 0.0, None)

    north_active = north_dist_outside > 0.0
    south_active = south_dist_outside > 0.0
    east_active = east_dist_outside > 0.0
    west_active = west_dist_outside > 0.0
    interior_active = (
        (yy >= interior_min_y)
        & (yy <= interior_max_y)
        & (xx >= interior_min_x)
        & (xx <= interior_max_x)
    )
    corner_excluded = ((north_active.astype(np.int32) + south_active.astype(np.int32) + east_active.astype(np.int32) + west_active.astype(np.int32)) > 1)
    outside_single_side = (north_active | south_active | east_active | west_active) & (~corner_excluded)

    inf = np.full((h, w), np.inf, dtype=np.float32)
    owner_stack = np.stack(
        [
            np.where(north_active, north_dist_outside, inf),
            np.where(south_active, south_dist_outside, inf),
            np.where(east_active, east_dist_outside, inf),
            np.where(west_active, west_dist_outside, inf),
        ],
        axis=0,
    )
    # Deterministic tie-break for equal-distance ownership in priority order north->south->east->west.
    tie_break_eps = np.array([0.0, 1e-6, 2e-6, 3e-6], dtype=np.float32).reshape(4, 1, 1)
    owner_idx = np.argmin(owner_stack + tie_break_eps, axis=0)
    interior_owner_stack = np.stack(
        [
            np.where(interior_active, north_dist_inside, inf),
            np.where(interior_active, south_dist_inside, inf),
            np.where(interior_active, east_dist_inside, inf),
            np.where(interior_active, west_dist_inside, inf),
        ],
        axis=0,
    )
    interior_owner_idx = np.argmin(interior_owner_stack + tie_break_eps, axis=0)

    north_owner = outside_single_side & (owner_idx == 0)
    south_owner = outside_single_side & (owner_idx == 1)
    east_owner = outside_single_side & (owner_idx == 2)
    west_owner = outside_single_side & (owner_idx == 3)
    north_interior_owner = interior_active & (interior_owner_idx == 0)
    south_interior_owner = interior_active & (interior_owner_idx == 1)
    east_interior_owner = interior_active & (interior_owner_idx == 2)
    west_interior_owner = interior_active & (interior_owner_idx == 3)

    def _edge_payload(owner_mask: np.ndarray, distance_map: np.ndarray, interior_owner_mask: np.ndarray, interior_distance_map: np.ndarray) -> Dict[str, np.ndarray]:
        halo_all = owner_mask.astype(np.float32)
        halo_inner = (owner_mask & (distance_map <= float(band))).astype(np.float32)
        halo_outer = (owner_mask & (distance_map > float(band))).astype(np.float32)
        ring_1 = (owner_mask & (distance_map > 0.0) & (distance_map <= 1.0)).astype(np.float32)
        ring_4 = (owner_mask & (distance_map > 0.0) & (distance_map <= 4.0)).astype(np.float32)
        ring_8 = (owner_mask & (distance_map > 0.0) & (distance_map <= 8.0)).astype(np.float32)
        ring_16 = (owner_mask & (distance_map > 0.0) & (distance_map <= 16.0)).astype(np.float32)
        interior_ring_1 = (interior_owner_mask & (interior_distance_map >= 0.0) & (interior_distance_map <= 1.0)).astype(np.float32)
        interior_outer = (interior_owner_mask & (interior_distance_map <= float(band))).astype(np.float32)

        return {
            "halo_all": halo_all,
            "halo_inner": halo_inner,
            "halo_outer": halo_outer,
            "halo_inner_edge_1px": ring_1,
            "halo_inner_edge_4px": ring_4,
            "halo_inner_8px": ring_8,
            "halo_inner_16px": ring_16,
            "interior_edge_1px": interior_ring_1,
            "interior_outer": interior_outer,
            "corner_excluded": corner_excluded.astype(np.float32),
        }

    masks["top"] = _edge_payload(north_owner, north_dist_outside, north_interior_owner, north_dist_inside)
    masks["bottom"] = _edge_payload(south_owner, south_dist_outside, south_interior_owner, south_dist_inside)
    masks["right"] = _edge_payload(east_owner, east_dist_outside, east_interior_owner, east_dist_inside)
    masks["left"] = _edge_payload(west_owner, west_dist_outside, west_interior_owner, west_dist_inside)
    return masks


def _build_expanded_denoise_clamp_masks(
    height: int,
    width: int,
    halo_px: int,
    inner_px: int,
    feather_px: int,
    feather_profile: str,
    edge_defined_flags: Optional[torch.Tensor],
    alpha_gate: Optional[np.ndarray] = None,
    valid_source_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    source_sizes_hw: Optional[torch.Tensor] = None,
    expanded_source_box: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    h = int(height)
    w = int(width)
    halo = int(max(1, halo_px))
    inner = int(max(0, min(inner_px, halo)))
    feather = int(max(0, feather_px))
    edge_band_masks = torch.zeros((1, 4, h, w), dtype=torch.float32)
    edge_band_masks[0, 0, :halo, :] = 1.0
    edge_band_masks[0, 1, h - halo :, :] = 1.0
    edge_band_masks[0, 2, :, w - halo :] = 1.0
    edge_band_masks[0, 3, :, :halo] = 1.0

    yy = torch.arange(h, dtype=torch.float32).unsqueeze(1).expand(h, w)
    xx = torch.arange(w, dtype=torch.float32).unsqueeze(0).expand(h, w)
    band = halo
    north = torch.clamp(1.0 - (yy - float(band)) / float(max(1, band)), min=0.0, max=1.0)
    north = north * ((yy >= float(band)) & (yy < float(2 * band))).float()
    south_anchor = float(h - band - 1)
    south = torch.clamp(1.0 - (south_anchor - yy) / float(max(1, band)), min=0.0, max=1.0)
    south = south * ((yy <= south_anchor) & (yy > south_anchor - float(band))).float()
    west = torch.clamp(1.0 - (xx - float(band)) / float(max(1, band)), min=0.0, max=1.0)
    west = west * ((xx >= float(band)) & (xx < float(2 * band))).float()
    east_anchor = float(w - band - 1)
    east = torch.clamp(1.0 - (east_anchor - xx) / float(max(1, band)), min=0.0, max=1.0)
    east = east * ((xx <= east_anchor) & (xx > east_anchor - float(band))).float()
    seam_decay_maps = torch.stack([north, south, east, west], dim=0).unsqueeze(0)
    if isinstance(edge_defined_flags, torch.Tensor):
        flags = edge_defined_flags.detach().float().view(1, -1)
    else:
        flags = torch.ones((1, 4), dtype=torch.float32)

    valid_mask_tensor = torch.ones((1, 1, h, w), dtype=torch.float32)
    if valid_source_mask is not None:
        valid_mask_np = np.asarray(valid_source_mask, dtype=np.float32)
        if valid_mask_np.ndim == 3:
            valid_mask_np = valid_mask_np[..., 0]
        valid_mask_tensor = torch.from_numpy(np.clip(valid_mask_np, 0.0, 1.0)).unsqueeze(0).unsqueeze(0).float()
    elif alpha_gate is not None:
        alpha_valid = np.asarray(alpha_gate, dtype=np.float32)
        if alpha_valid.ndim == 3:
            alpha_valid = alpha_valid[..., 0]
        valid_mask_tensor = torch.from_numpy(np.clip(alpha_valid, 0.0, 1.0)).unsqueeze(0).unsqueeze(0).float()

    supervision_mask = valid_mask_tensor.clone()
    seam_maps = shared_build_seam_region_maps(
        edge_band_masks=edge_band_masks,
        seam_decay_maps=seam_decay_maps,
        edge_defined_flags=flags,
        seam_strip_width_px=torch.tensor([float(halo)], dtype=torch.float32),
        supervision_mask=supervision_mask,
        seam_config={
            "margin_inner_px": inner,
            "continuation_profile": str(feather_profile or "piecewise").strip().lower(),
            "continuation_width_px": feather,
            "continuation_base_width_px": feather,
            "continuation_min_width_px": feather,
            "continuation_max_width_px": feather,
            "continuation_noise_enabled": False,
            "continuation_corner_normalization_enabled": True,
            "require_defined_for_margin_and_band": True,
        },
        expanded_halo_px=halo,
        source_sizes_hw=source_sizes_hw,
        expanded_source_boxes=expanded_source_box,
        valid_expanded_source_mask=valid_mask_tensor,
        continuation_valid_mask=valid_mask_tensor,
        style_support_valid_mask=valid_mask_tensor,
    )

    hard_mask = seam_maps["hard_band_mask"][0, 0].detach().cpu().numpy().astype(np.float32)
    feather_mask = seam_maps["continuation_distance_weighted"][0, 0].detach().cpu().numpy().astype(np.float32)

    # Criterion 6: log geometry fingerprints so train/eval consistency can be checked per sample.
    _gap_mask = seam_maps.get("seam_gap_mask")
    _overlap_mask = seam_maps.get("seam_overlap_mask")
    _gap_px = float(_gap_mask[0, 0].sum().item()) if _gap_mask is not None else -1.0
    _overlap_px = float(_overlap_mask[0, 0].sum().item()) if _overlap_mask is not None else -1.0
    _hard_px = float(np.sum(hard_mask > 0.0))
    _valid_px = float(valid_mask_tensor[0, 0].sum().item())
    logger.info(
        "[seam/invariant/criterion6] eval geometry: hard_band_px=%.1f gap_px=%.1f overlap_px=%.1f "
        "valid_px=%.1f h=%d w=%d halo=%d inner=%d — "
        "compare with training log [seam/invariant/OK] for same sample to verify mask parity",
        _hard_px, _gap_px, _overlap_px, _valid_px, h, w, halo, inner,
    )
    if _gap_px > 0.0:
        logger.warning(
            "[seam/invariant/criterion6] WARN eval geometry has seam_gap_px=%.1f — "
            "inner_halo and hard_band do not cover all seam neighbourhood pixels in the eval path",
            _gap_px,
        )

    valid_mask_np = valid_mask_tensor[0, 0].detach().cpu().numpy().astype(np.float32)
    hard_mask = hard_mask * valid_mask_np
    feather_mask = feather_mask * valid_mask_np

    feather_mask = np.clip(feather_mask * (1.0 - (hard_mask > 0.0).astype(np.float32)), 0.0, 1.0)
    return hard_mask, feather_mask


def _ddpm_pred_original_sample_coeff(scheduler: DDPMScheduler, timestep: torch.Tensor) -> torch.Tensor:
    timestep_index = int(timestep.item())
    prev_timestep = timestep_index - (scheduler.config.num_train_timesteps // scheduler.num_inference_steps)
    alpha_prod_t = scheduler.alphas_cumprod[timestep_index]
    if prev_timestep >= 0:
        alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep]
    else:
        alpha_prod_t_prev = scheduler.one
    beta_prod_t = 1 - alpha_prod_t
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t
    coeff = (alpha_prod_t_prev.sqrt() * current_beta_t) / beta_prod_t.clamp_min(1e-8)
    return coeff.to(dtype=torch.float32)


def _prepare_seam_denoise_clamp_payload(
    sample: Dict[str, object],
    *,
    interior_h: int,
    interior_w: int,
    halo_px: int,
    inner_px: int,
    feather_px: int,
    feather_profile: str,
    hard_threshold: float,
    clamp_mode: str,
    latent_h: int,
    latent_w: int,
    vae: torch.nn.Module,
    vae_dtype: torch.dtype,
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    expected_rgba = _build_expected_expanded_seam_rgba(
        sample=sample,
        interior_h=interior_h,
        interior_w=interior_w,
        halo_px=halo_px,
    )
    if expected_rgba is None:
        return None

    exp_h, exp_w = expected_rgba.shape[:2]
    alpha_gate = (expected_rgba[..., 3] > 0.0).astype(np.float32)
    if float(np.sum(alpha_gate)) <= 0.0:
        alpha_gate = (np.abs(expected_rgba[..., :3]).sum(axis=2) > 1e-6).astype(np.float32)
    valid_source_mask = sample.get("valid_expanded_source_mask")
    hard_mask_np, feather_mask_np = _build_expanded_denoise_clamp_masks(
        height=exp_h,
        width=exp_w,
        halo_px=halo_px,
        inner_px=inner_px,
        feather_px=feather_px,
        feather_profile=feather_profile,
        edge_defined_flags=sample.get("edge_defined_flags"),
        alpha_gate=alpha_gate,
        valid_source_mask=valid_source_mask,
        source_sizes_hw=sample.get("original_sizes_hw"),
        expanded_source_box=sample.get("expanded_crop_box"),
    )
    if float(np.sum(hard_mask_np) + np.sum(feather_mask_np)) <= 0.0:
        return None

    hard_mask = torch.from_numpy(hard_mask_np).to(device=device, dtype=torch.float32).unsqueeze(0)
    feather_mask = torch.from_numpy(feather_mask_np).to(device=device, dtype=torch.float32).unsqueeze(0)
    combined_mask = (hard_mask + feather_mask).clamp(0.0, 1.0)
    seam_target_rgb = torch.from_numpy(expected_rgba[..., :3]).permute(2, 0, 1).to(device=device, dtype=torch.float32)
    seam_target_rgb = (seam_target_rgb * 2.0) - 1.0

    target_rgb = seam_target_rgb
    context_rgb: Optional[torch.Tensor] = None
    expanded_images = sample.get("expanded_images")
    if isinstance(expanded_images, torch.Tensor):
        if tuple(int(v) for v in expanded_images.shape[-2:]) == (exp_h, exp_w):
            context_rgb = expanded_images.detach().to(device=device, dtype=torch.float32)
    elif isinstance(sample.get("images"), torch.Tensor):
        images = sample["images"].detach()
        if tuple(int(v) for v in images.shape[-2:]) == (interior_h, interior_w):
            context_rgb = torch.zeros((3, exp_h, exp_w), device=device, dtype=torch.float32)
            context_rgb[:, halo : halo + interior_h, halo : halo + interior_w] = images.to(device=device, dtype=torch.float32)

    if context_rgb is not None:
        target_rgb = (context_rgb * (1.0 - hard_mask)) + (seam_target_rgb * hard_mask)
    else:
        target_rgb = seam_target_rgb * hard_mask

    payload: Dict[str, torch.Tensor] = {
        "target_rgb": target_rgb,
        "hard_mask": hard_mask,
        "feather_mask": feather_mask,
        "combined_mask": combined_mask,
    }
    combined_mask_np = np.clip(hard_mask_np + feather_mask_np, 0.0, 1.0)
    if isinstance(sample.get("edge_defined_flags"), torch.Tensor):
        flags = sample["edge_defined_flags"].detach().float().cpu().view(-1).numpy()
    else:
        flags = np.ones((4,), dtype=np.float32)
    undefined_hard_np, undefined_feather_np = _build_expanded_denoise_clamp_masks(
        height=exp_h,
        width=exp_w,
        halo_px=halo_px,
        inner_px=inner_px,
        feather_px=feather_px,
        feather_profile=feather_profile,
        edge_defined_flags=torch.from_numpy((flags < 0.5).astype(np.float32)),
        alpha_gate=alpha_gate,
        valid_source_mask=valid_source_mask,
        source_sizes_hw=sample.get("original_sizes_hw"),
        expanded_source_box=sample.get("expanded_crop_box"),
    )
    undefined_combined_np = np.clip(undefined_hard_np + undefined_feather_np, 0.0, 1.0)
    payload["applied_px"] = torch.tensor(float(np.sum(combined_mask_np > 0.0)), dtype=torch.float32)
    payload["hard_applied_px"] = torch.tensor(float(np.sum(hard_mask_np > 0.0)), dtype=torch.float32)
    payload["feather_applied_px"] = torch.tensor(float(np.sum(feather_mask_np > 0.0)), dtype=torch.float32)
    payload["undefined_overlap_px"] = torch.tensor(
        float(np.sum((combined_mask_np > 0.0) & (undefined_combined_np > 0.0))),
        dtype=torch.float32,
    )

    if str(clamp_mode).strip().lower() == "latent":
        with torch.no_grad():
            encoded = vae.encode(target_rgb.unsqueeze(0).to(dtype=vae_dtype)).latent_dist.mode()
        target_latent = encoded.to(dtype=torch.float32) * float(sdxl_model_util.VAE_SCALE_FACTOR)
        hard_latent = F.interpolate(hard_mask.unsqueeze(0), size=(latent_h, latent_w), mode="area").squeeze(0)
        feather_latent = F.interpolate(feather_mask.unsqueeze(0), size=(latent_h, latent_w), mode="area").squeeze(0)
        hard_latent = hard_latent.clamp(0.0, 1.0)
        feather_latent = feather_latent.clamp(0.0, 1.0)
        hard_binary = (hard_latent > float(hard_threshold)).float()
        combined_latent = (hard_latent + feather_latent).clamp(0.0, 1.0)
        soft_latent = (combined_latent - hard_binary).clamp(0.0, 1.0)
        # Criterion 7: the downsampled latent clamp should touch the seam-side support boundary,
        # and there should be no uncovered one-pixel boundary between support and hard/feather.
        if isinstance(sample.get("edge_defined_flags"), torch.Tensor):
            flags = sample["edge_defined_flags"].detach().float().cpu().view(-1).numpy()
        else:
            flags = np.ones((4,), dtype=np.float32)
        edge_masks_support = _build_expanded_edge_masks(
            exp_h,
            exp_w,
            halo_px=halo_px,
            band_px=max(1, min(int(inner_px), int(halo_px))),
        )
        valid_support_np: Optional[np.ndarray] = None
        if valid_source_mask is not None:
            valid_support_np = np.asarray(valid_source_mask, dtype=np.float32)
            if valid_support_np.ndim == 3:
                valid_support_np = valid_support_np[..., 0]
            valid_support_np = np.clip(valid_support_np, 0.0, 1.0)
        elif alpha_gate is not None:
            valid_support_np = np.asarray(alpha_gate, dtype=np.float32)
            if valid_support_np.ndim == 3:
                valid_support_np = valid_support_np[..., 0]
            valid_support_np = np.clip(valid_support_np, 0.0, 1.0)

        trusted_boundary_expanded: Optional[np.ndarray] = None
        if isinstance(sample.get("trusted_mask"), torch.Tensor):
            trusted_mask = sample["trusted_mask"].detach().float().cpu().numpy()
            if trusted_mask.ndim == 3:
                trusted_mask = trusted_mask[0]
            trusted_boundary_expanded = np.zeros((exp_h, exp_w), dtype=np.float32)
            trusted_h = min(int(interior_h), int(trusted_mask.shape[-2]))
            trusted_w = min(int(interior_w), int(trusted_mask.shape[-1]))
            trusted_boundary_expanded[halo_px : halo_px + trusted_h, halo_px : halo_px + trusted_w] = trusted_mask[:trusted_h, :trusted_w]

        support_boundary_latent = torch.zeros((1, latent_h, latent_w), device=device, dtype=torch.float32)
        support_side_specs = [
            (0, "top"),
            (1, "bottom"),
            (2, "right"),
            (3, "left"),
        ]
        for edge_idx, side_name in support_side_specs:
            if edge_idx >= len(flags) or float(flags[edge_idx]) < 0.5:
                continue
            side_masks = edge_masks_support[side_name]
            support_boundary_np = np.clip(
                side_masks["halo_inner_edge_1px"]
                + (
                    side_masks["interior_edge_1px"]
                    if trusted_boundary_expanded is None
                    else (trusted_boundary_expanded * side_masks["interior_edge_1px"])
                ),
                0.0,
                1.0,
            )
            if valid_support_np is not None:
                support_boundary_np = support_boundary_np * valid_support_np
            support_boundary_mask = F.interpolate(
                torch.from_numpy(support_boundary_np).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                size=(latent_h, latent_w),
                mode="area",
            ).clamp(0.0, 1.0).squeeze(0)
            support_boundary_latent = torch.maximum(support_boundary_latent, support_boundary_mask)

        support_boundary_binary = (support_boundary_latent > 1e-6).float()
        hard_touch_region = _expand_mask(hard_binary.unsqueeze(0), 1).squeeze(0)
        combined_positive = (combined_latent > 1e-6).float()
        combined_touch_region = _expand_mask(combined_positive.unsqueeze(0), 1).squeeze(0)
        support_boundary_px = float(support_boundary_binary.sum().item())
        hard_support_touch_px = float((support_boundary_binary * hard_touch_region).sum().item())
        boundary_gap_px = float((support_boundary_binary * (1.0 - combined_touch_region)).sum().item())
        boundary_gap_ratio = boundary_gap_px / max(support_boundary_px, 1e-6)
        logger.info(
            "[seam/invariant/criterion7] eval latent clamp boundary: support_boundary_px=%.1f "
            "hard_touch_px=%.1f boundary_gap_px=%.1f boundary_gap_ratio=%.4f latent_h=%d latent_w=%d halo_px=%d inner_px=%d",
            support_boundary_px,
            hard_support_touch_px,
            boundary_gap_px,
            boundary_gap_ratio,
            latent_h,
            latent_w,
            halo_px,
            inner_px,
        )
        if hard_support_touch_px <= 0.0:
            logger.warning(
                "[seam/invariant/criterion7] WARN eval latent hard clamp does not touch seam-side support: "
                "support_boundary_px=%.1f latent_h=%d latent_w=%d halo_px=%d inner_px=%d",
                support_boundary_px,
                latent_h,
                latent_w,
                halo_px,
                inner_px,
            )
        if boundary_gap_px > 0.0:
            logger.warning(
                "[seam/invariant/criterion7] WARN eval latent boundary gap detected: gap_px=%.1f ratio=%.4f — "
                "one-pixel seam-side support boundary is not touched by hard/feather clamp at latent scale",
                boundary_gap_px,
                boundary_gap_ratio,
            )
        if float(soft_latent.max().detach().item()) > 0.0:
            edge_masks_hard = _build_expanded_edge_masks(
                exp_h,
                exp_w,
                halo_px=halo_px,
                band_px=max(1, min(int(inner_px), int(halo_px))),
            )
            edge_masks_soft = _build_expanded_edge_masks(
                exp_h,
                exp_w,
                halo_px=halo_px,
                band_px=max(1, min(int(feather_px), int(halo_px))),
            )
            projected_sum_latent = torch.zeros_like(target_latent)
            projected_weight_latent = torch.zeros((1, 1, latent_h, latent_w), device=device, dtype=torch.float32)
            side_specs = [
                (0, "top", "north"),
                (1, "bottom", "south"),
                (2, "right", "east"),
                (3, "left", "west"),
            ]
            for edge_idx, side_name, edge_name in side_specs:
                if edge_idx >= len(flags) or float(flags[edge_idx]) < 0.5:
                    continue
                src_mask_np = edge_masks_hard[side_name]["halo_inner"]
                dst_mask_np = edge_masks_soft[side_name]["interior_outer"]
                src_mask_latent = F.interpolate(
                    torch.from_numpy(src_mask_np).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                    size=(latent_h, latent_w),
                    mode="area",
                ).clamp(0.0, 1.0)
                dst_mask_latent = F.interpolate(
                    torch.from_numpy(dst_mask_np).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                    size=(latent_h, latent_w),
                    mode="area",
                ).clamp(0.0, 1.0)
                dst_mask_latent = dst_mask_latent * soft_latent.unsqueeze(0)
                if float(src_mask_latent.max().detach().item()) <= 0.0 or float(dst_mask_latent.max().detach().item()) <= 0.0:
                    continue
                src_presence = src_mask_latent[0, 0]
                dst_presence = dst_mask_latent[0, 0]
                if edge_name in {"north", "south"}:
                    src_rows = torch.nonzero(src_presence.amax(dim=1) > 0.0, as_tuple=False).flatten()
                    dst_rows = torch.nonzero(dst_presence.amax(dim=1) > 0.0, as_tuple=False).flatten()
                    if src_rows.numel() == 0 or dst_rows.numel() == 0:
                        continue
                    src_y0 = int(src_rows[0].item())
                    src_y1 = int(src_rows[-1].item()) + 1
                    dst_y0 = int(dst_rows[0].item())
                    dst_y1 = int(dst_rows[-1].item()) + 1
                    src_slice = target_latent[:, :, src_y0:src_y1, :]
                    src_weight = src_mask_latent[:, :, src_y0:src_y1, :]
                    projected_edge_latent = F.interpolate(
                        src_slice * src_weight,
                        size=(max(1, dst_y1 - dst_y0), latent_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    projected_edge_weight = F.interpolate(
                        src_weight,
                        size=(max(1, dst_y1 - dst_y0), latent_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    projected_edge_latent = torch.where(
                        projected_edge_weight > 0.0,
                        projected_edge_latent / projected_edge_weight.clamp_min(1e-6),
                        torch.zeros_like(projected_edge_latent),
                    )
                    edge_target = torch.zeros_like(target_latent)
                    edge_gate = torch.zeros((1, 1, latent_h, latent_w), device=device, dtype=torch.float32)
                    edge_target[:, :, dst_y0:dst_y1, :] = projected_edge_latent
                    edge_gate[:, :, dst_y0:dst_y1, :] = dst_mask_latent[:, :, dst_y0:dst_y1, :]
                else:
                    src_cols = torch.nonzero(src_presence.amax(dim=0) > 0.0, as_tuple=False).flatten()
                    dst_cols = torch.nonzero(dst_presence.amax(dim=0) > 0.0, as_tuple=False).flatten()
                    if src_cols.numel() == 0 or dst_cols.numel() == 0:
                        continue
                    src_x0 = int(src_cols[0].item())
                    src_x1 = int(src_cols[-1].item()) + 1
                    dst_x0 = int(dst_cols[0].item())
                    dst_x1 = int(dst_cols[-1].item()) + 1
                    src_slice = target_latent[:, :, :, src_x0:src_x1]
                    src_weight = src_mask_latent[:, :, :, src_x0:src_x1]
                    projected_edge_latent = F.interpolate(
                        src_slice * src_weight,
                        size=(latent_h, max(1, dst_x1 - dst_x0)),
                        mode="bilinear",
                        align_corners=False,
                    )
                    projected_edge_weight = F.interpolate(
                        src_weight,
                        size=(latent_h, max(1, dst_x1 - dst_x0)),
                        mode="bilinear",
                        align_corners=False,
                    )
                    projected_edge_latent = torch.where(
                        projected_edge_weight > 0.0,
                        projected_edge_latent / projected_edge_weight.clamp_min(1e-6),
                        torch.zeros_like(projected_edge_latent),
                    )
                    edge_target = torch.zeros_like(target_latent)
                    edge_gate = torch.zeros((1, 1, latent_h, latent_w), device=device, dtype=torch.float32)
                    edge_target[:, :, :, dst_x0:dst_x1] = projected_edge_latent
                    edge_gate[:, :, :, dst_x0:dst_x1] = dst_mask_latent[:, :, :, dst_x0:dst_x1]
                projected_sum_latent = projected_sum_latent + (edge_target * edge_gate)
                projected_weight_latent = projected_weight_latent + edge_gate

            if float(projected_weight_latent.max().detach().item()) > 0.0:
                projected_target_latent = torch.where(
                    projected_weight_latent > 0.0,
                    projected_sum_latent / projected_weight_latent.clamp_min(1e-6),
                    target_latent,
                )
                target_latent = (target_latent * (1.0 - soft_latent.unsqueeze(0))) + (projected_target_latent * soft_latent.unsqueeze(0))
        payload["target_latent"] = target_latent
        payload["hard_mask_latent"] = hard_latent
        payload["hard_mask_latent_binary"] = hard_binary
        payload["feather_mask_latent"] = feather_latent
        payload["soft_mask_latent"] = soft_latent
        payload["combined_mask_latent"] = (hard_binary + soft_latent).clamp(0.0, 1.0)
        payload["combined_mask_latent_raw"] = combined_latent
        payload["hard_threshold"] = torch.tensor(float(hard_threshold), dtype=torch.float32)
        payload["latent_support_boundary_px"] = torch.tensor(support_boundary_px, dtype=torch.float32)
        payload["latent_hard_support_touch_px"] = torch.tensor(hard_support_touch_px, dtype=torch.float32)
        payload["latent_hard_touches_support"] = torch.tensor(1.0 if hard_support_touch_px > 0.0 else 0.0, dtype=torch.float32)
        payload["latent_boundary_gap_px"] = torch.tensor(boundary_gap_px, dtype=torch.float32)
        payload["latent_boundary_gap_ratio"] = torch.tensor(boundary_gap_ratio, dtype=torch.float32)
    return payload


def _apply_seam_denoise_clamp(
    pred_x0: torch.Tensor,
    *,
    clamp_mode: str,
    clamp_payload: Dict[str, torch.Tensor],
    vae: torch.nn.Module,
    vae_dtype: torch.dtype,
    capture_rgb_diagnostics: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    mode = str(clamp_mode or "latent").strip().lower()
    diagnostics: Dict[str, torch.Tensor] = {}
    if mode == "latent":
        blend = clamp_payload["combined_mask_latent"].to(device=pred_x0.device, dtype=pred_x0.dtype)
        hard = clamp_payload["hard_mask_latent_binary"].to(device=pred_x0.device, dtype=pred_x0.dtype)
        hard_raw = clamp_payload["hard_mask_latent"].to(device=pred_x0.device, dtype=pred_x0.dtype)
        soft = clamp_payload["soft_mask_latent"].to(device=pred_x0.device, dtype=pred_x0.dtype)
        target = clamp_payload["target_latent"].to(device=pred_x0.device, dtype=pred_x0.dtype)
        pred_x0_clamped = pred_x0 * (1.0 - hard - soft) + target * (hard + soft)
        diagnostics["clamp_delta_latent"] = (pred_x0_clamped - pred_x0).detach().float()
        diagnostics["clamp_hard_mask"] = hard.detach().float().cpu()
        diagnostics["clamp_combined_mask"] = blend.detach().float().cpu()
        diagnostics["clamp_hard_mask_raw"] = hard_raw.detach().float().cpu()
        diagnostics["clamp_soft_mask_latent"] = soft.detach().float().cpu()
        if clamp_payload.get("hard_threshold") is not None:
            diagnostics["clamp_hard_threshold"] = clamp_payload["hard_threshold"].detach().float().cpu()
        if capture_rgb_diagnostics:
            with torch.no_grad():
                decoded_pre = vae.decode((pred_x0 / sdxl_model_util.VAE_SCALE_FACTOR).to(dtype=vae_dtype)).sample
                decoded_post = vae.decode((pred_x0_clamped / sdxl_model_util.VAE_SCALE_FACTOR).to(dtype=vae_dtype)).sample
            diagnostics["clamp_decoded_pre"] = decoded_pre.detach().float().cpu().squeeze(0)
            diagnostics["clamp_decoded_post"] = decoded_post.detach().float().cpu().squeeze(0)
        return pred_x0_clamped, diagnostics

    with torch.no_grad():
        decoded = vae.decode((pred_x0 / sdxl_model_util.VAE_SCALE_FACTOR).to(dtype=vae_dtype)).sample
    decoded = decoded.to(dtype=torch.float32)
    target_rgb = clamp_payload["target_rgb"].unsqueeze(0).to(device=decoded.device, dtype=decoded.dtype)
    blend = clamp_payload["combined_mask"].unsqueeze(0).to(device=decoded.device, dtype=decoded.dtype)
    hard = clamp_payload["hard_mask"].unsqueeze(0).to(device=decoded.device, dtype=decoded.dtype)
    feather = (blend - hard).clamp(0.0, 1.0)

    decoded_clamped = decoded.clone()
    hard_bool = hard >= 0.999
    decoded_clamped = torch.where(hard_bool, target_rgb, decoded_clamped)
    if torch.count_nonzero(feather > 0.0).item() > 0:
        decoded_clamped = decoded_clamped * (1.0 - feather) + target_rgb * feather
    with torch.no_grad():
        encoded = vae.encode(decoded_clamped.to(dtype=vae_dtype)).latent_dist.mode()
        decoded_reencoded = vae.decode(encoded).sample.to(dtype=torch.float32)
    pred_x0_clamped = encoded.to(device=pred_x0.device, dtype=torch.float32) * float(sdxl_model_util.VAE_SCALE_FACTOR)
    diagnostics["clamp_pre_rgb_min"] = decoded.detach().float().amin().cpu()
    diagnostics["clamp_pre_rgb_max"] = decoded.detach().float().amax().cpu()
    diagnostics["clamp_pre_rgb_mean"] = decoded.detach().float().mean().cpu()
    diagnostics["clamp_target_rgb_min"] = target_rgb.detach().float().amin().cpu()
    diagnostics["clamp_target_rgb_max"] = target_rgb.detach().float().amax().cpu()
    diagnostics["clamp_target_rgb_mean"] = target_rgb.detach().float().mean().cpu()
    diagnostics["clamp_post_rgb_min"] = decoded_clamped.detach().float().amin().cpu()
    diagnostics["clamp_post_rgb_max"] = decoded_clamped.detach().float().amax().cpu()
    diagnostics["clamp_post_rgb_mean"] = decoded_clamped.detach().float().mean().cpu()
    diagnostics["clamp_decoded_pre"] = decoded.detach().float().cpu().squeeze(0)
    diagnostics["clamp_delta_rgb"] = (decoded_clamped - decoded).detach().float().cpu().squeeze(0)
    diagnostics["clamp_target_rgb"] = target_rgb.detach().float().cpu().squeeze(0)
    diagnostics["clamp_hard_mask"] = hard.detach().float().cpu().squeeze(0)
    diagnostics["clamp_combined_mask"] = blend.detach().float().cpu().squeeze(0)
    diagnostics["clamp_decoded_post"] = decoded_clamped.detach().float().cpu().squeeze(0)
    diagnostics["clamp_decoded_reencoded"] = decoded_reencoded.detach().float().cpu().squeeze(0)
    return pred_x0_clamped, diagnostics


def _render_one(
    sample: Dict[str, object],
    unet: torch.nn.Module,
    control_net: torch.nn.Module,
    vae: torch.nn.Module,
    scheduler: DDPMScheduler,
    cached_text: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    weight_dtype: torch.dtype,
    control_dtype: torch.dtype,
    vae_dtype: torch.dtype,
    steps: int,
    seed: int,
    write_latent_debug: bool,
    alpha_output_source: str,
    expanded_prediction_enabled: bool = False,
    expanded_halo_px: int = 0,
    seam_denoise_clamp: bool = False,
    seam_denoise_clamp_inner_px: int = 0,
    seam_denoise_clamp_feather_px: int = 160,
    seam_denoise_clamp_feather_profile: str = "smoothstep",
    seam_denoise_clamp_every_n_steps: int = 1,
    seam_denoise_clamp_mode: str = "latent",
    seam_denoise_clamp_hard_threshold: float = 0.5,
    conditioning_spec: Optional[ModelVisibleConditioningSpec] = None,
    override_conditioning: Optional[torch.Tensor] = None,
    override_full_conditioning: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    if override_full_conditioning is not None:
        cond = override_full_conditioning.unsqueeze(0).to(device=device, dtype=control_dtype)
    else:
        raw_cond = override_conditioning if override_conditioning is not None else sample["conditioning_images"]
        if conditioning_spec is None:
            full_channel_names = sample.get("full_conditioning_channel_names") or sample.get("channel_names") or []
            channel_names = sample.get("channel_names") or []
            seam_channel_count = 0
            if isinstance(sample.get("seam_strip_tensor"), torch.Tensor):
                seam_channel_count += int(sample["seam_strip_tensor"].shape[0])
            if isinstance(sample.get("edge_flag_maps"), torch.Tensor):
                seam_channel_count += int(sample["edge_flag_maps"].shape[0])
            style_start = min(len(full_channel_names), int(raw_cond.shape[0]) + seam_channel_count)
            conditioning_spec = build_model_visible_conditioning_spec(
                seam_enabled=isinstance(sample.get("seam_strip_tensor"), torch.Tensor),
                channel_names=channel_names,
                full_conditioning_channel_names=full_channel_names,
                style_conditioning_channel_names=full_channel_names[style_start:],
                seam_config={},
                style_ratio_config={},
                terrain_mask_channel_index=int(channel_names.index("terrain_mask")) if "terrain_mask" in channel_names else -1,
                terrain_mask_black_is_terrain=True,
                alpha_binary_threshold=0.5,
            )
        visible_cond = _compose_model_visible_conditioning(sample, raw_cond, conditioning_spec)
        cond = visible_cond.unsqueeze(0).to(device=device, dtype=control_dtype)

    interior_h = int(sample["target_sizes_hw"][0].item())
    interior_w = int(sample["target_sizes_hw"][1].item())
    use_expanded = bool(expanded_prediction_enabled and int(expanded_halo_px) > 0)
    halo_px = int(max(0, expanded_halo_px))
    target_h, target_w = (interior_h, interior_w)
    if use_expanded:
        target_h, target_w = expanded_hw(interior_h, interior_w, halo_px)
        if tuple(cond.shape[-2:]) == (interior_h, interior_w):
            cond = pad_chw_spatial(cond.squeeze(0), halo_px=halo_px, mode="constant").unsqueeze(0).to(device=device, dtype=control_dtype)
        elif tuple(cond.shape[-2:]) != (target_h, target_w):
            raise ValueError(
                "expanded generic eval conditioning shape mismatch: "
                + f"got={tuple(cond.shape[-2:])} expected={(target_h, target_w)}"
            )

    te1, te2, pool2 = cached_text
    text_embedding = torch.cat([te1, te2], dim=2)

    size_batch = {
        "original_sizes_hw": torch.tensor([[target_h, target_w]], device=device, dtype=torch.long),
        "crop_top_lefts": torch.tensor([[0, 0]], device=device, dtype=torch.long),
        "target_sizes_hw": torch.tensor([[target_h, target_w]], device=device, dtype=torch.long),
    }
    size_embeddings = sdxl_train_util.get_size_embeddings(
        size_batch["original_sizes_hw"],
        size_batch["crop_top_lefts"],
        size_batch["target_sizes_hw"],
        device,
    ).to(weight_dtype)
    vector_embedding = torch.cat([pool2, size_embeddings], dim=1)

    latent_h = int(target_h) // 8
    latent_w = int(target_w) // 8
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    noisy = torch.randn((1, 4, latent_h, latent_w), generator=gen, device=device, dtype=weight_dtype)

    scheduler.set_timesteps(max(2, steps), device=device)
    alpha_probs = None
    alpha_logits = None
    clamp_enabled = bool(seam_denoise_clamp) and use_expanded and halo_px > 0
    clamp_payload = None
    if clamp_enabled:
        clamp_payload = _prepare_seam_denoise_clamp_payload(
            sample=sample,
            interior_h=interior_h,
            interior_w=interior_w,
            halo_px=halo_px,
            inner_px=int(seam_denoise_clamp_inner_px),
            feather_px=int(seam_denoise_clamp_feather_px),
            feather_profile=str(seam_denoise_clamp_feather_profile),
            hard_threshold=float(seam_denoise_clamp_hard_threshold),
            clamp_mode=seam_denoise_clamp_mode,
            latent_h=latent_h,
            latent_w=latent_w,
            vae=vae,
            vae_dtype=vae_dtype,
            device=device,
        )
        clamp_enabled = clamp_payload is not None
    clamp_steps_applied = 0
    clamp_diagnostics: Dict[str, torch.Tensor] = {}
    pred_x0 = None
    with torch.no_grad():
        for step_index, timestep in enumerate(scheduler.timesteps):
            t = timestep.expand(1).to(device=device, dtype=torch.long)
            input_resi_add, mid_add, alpha_outputs = control_net(
                noisy.to(dtype=control_dtype),
                t,
                text_embedding.to(dtype=control_dtype),
                vector_embedding.to(dtype=control_dtype),
                cond,
                return_alpha=True,
                alpha_target_size=(target_h, target_w),
            )
            eps = unet(
                noisy.to(dtype=weight_dtype),
                t,
                text_embedding.to(dtype=weight_dtype),
                vector_embedding.to(dtype=weight_dtype),
                [x.to(dtype=weight_dtype) for x in input_resi_add],
                mid_add.to(dtype=weight_dtype),
            )
            step_out = scheduler.step(eps, timestep, noisy)
            pred_x0 = step_out.pred_original_sample.to(dtype=torch.float32)
            if clamp_enabled and (
                (step_index % max(1, int(seam_denoise_clamp_every_n_steps)) == 0)
                or (step_index == len(scheduler.timesteps) - 1)
            ):
                capture_rgb_diagnostics = step_index == len(scheduler.timesteps) - 1
                pred_x0_clamped, clamp_diagnostics = _apply_seam_denoise_clamp(
                    pred_x0,
                    clamp_mode=seam_denoise_clamp_mode,
                    clamp_payload=clamp_payload,
                    vae=vae,
                    vae_dtype=vae_dtype,
                    capture_rgb_diagnostics=capture_rgb_diagnostics,
                )
                coeff = _ddpm_pred_original_sample_coeff(scheduler, timestep).to(device=step_out.prev_sample.device, dtype=step_out.prev_sample.dtype)
                coeff = coeff.view(1, 1, 1, 1)
                noisy = step_out.prev_sample + coeff * (pred_x0_clamped.to(dtype=step_out.prev_sample.dtype) - pred_x0.to(dtype=step_out.prev_sample.dtype))
                pred_x0 = pred_x0_clamped
                clamp_steps_applied += 1
            else:
                noisy = step_out.prev_sample

        if pred_x0 is None:
            alpha_t = scheduler.alphas_cumprod[scheduler.timesteps[-1]].to(device=device, dtype=torch.float32).view(1, 1, 1, 1)
            pred_x0 = noisy.float() / alpha_t.sqrt()
        decoded = vae.decode((pred_x0 / sdxl_model_util.VAE_SCALE_FACTOR).to(dtype=vae_dtype)).sample[0]

        selected_logits = _select_eval_alpha_logits(alpha_outputs, alpha_output_source)
        alpha_logits = selected_logits.squeeze(0).squeeze(0).detach().float().cpu()
        alpha_probs = torch.sigmoid(alpha_logits)

    debug_latent = pred_x0.detach().cpu() if write_latent_debug else None
    decoded_cpu = decoded.detach().float().cpu()
    expanded_decoded_cpu = decoded_cpu.clone() if use_expanded else decoded_cpu
    expanded_alpha_logits = alpha_logits.clone() if use_expanded else alpha_logits
    expanded_alpha_probs = alpha_probs.clone() if use_expanded else alpha_probs

    expected_h = interior_h
    expected_w = interior_w
    if use_expanded:
        if decoded_cpu.shape[-2:] != (target_h, target_w):
            raise RuntimeError(
                "expanded render decode shape mismatch before crop: "
                + f"decoded={tuple(decoded_cpu.shape[-2:])} expected={(target_h, target_w)}"
            )
        decoded_cpu = center_crop_chw(decoded_cpu, out_h=expected_h, out_w=expected_w, halo_px=halo_px)
        alpha_logits = center_crop_hw(alpha_logits, out_h=expected_h, out_w=expected_w, halo_px=halo_px)
        alpha_probs = torch.sigmoid(alpha_logits)
    if decoded_cpu.shape[-2:] != (expected_h, expected_w):
        raise RuntimeError(
            "render decode shape mismatch before export: "
            + f"decoded={tuple(decoded_cpu.shape[-2:])} expected={(expected_h, expected_w)}"
        )

    del decoded
    del pred_x0
    del noisy
    del alpha_outputs
    del selected_logits
    del input_resi_add
    del mid_add
    del eps
    clean_memory_on_device(device)

    rgb = _tensor_to_image(decoded_cpu).convert("RGB")
    if rgb.size != (expected_w, expected_h):
        raise RuntimeError(
            "export image shape mismatch: "
            + f"rgb_size={rgb.size} expected={(expected_w, expected_h)}"
        )
    pred_alpha_img = _mask_to_image(alpha_probs)
    rgba = rgb.copy()
    rgba.putalpha(pred_alpha_img)

    output = {
        "rgb": rgb,
        "pred_alpha_logits": alpha_logits,
        "pred_alpha_prob": alpha_probs,
        "pred_alpha_img": pred_alpha_img,
        "rgba": rgba,
        "expanded_prediction_enabled": float(1.0 if use_expanded else 0.0),
        "expanded_halo_px": float(halo_px),
        "expanded_rgb_tensor": expanded_decoded_cpu,
        "expanded_alpha_prob": expanded_alpha_probs,
        "seam_denoise_clamp_enabled": float(1.0 if clamp_enabled else 0.0),
        "seam_denoise_clamp_steps_applied": float(clamp_steps_applied),
        "seam_denoise_clamp_expected_steps": float(
            0.0
            if not clamp_enabled
            else len(
                [
                    i
                    for i in range(len(scheduler.timesteps))
                    if (i % max(1, int(seam_denoise_clamp_every_n_steps)) == 0)
                    or (i == len(scheduler.timesteps) - 1)
                ]
            )
        ),
        "seam_denoise_clamp_mode": str(seam_denoise_clamp_mode),
    }
    if clamp_diagnostics:
        output["clamp_combined_mask"] = clamp_diagnostics.get("clamp_combined_mask")
        output["clamp_hard_mask"] = clamp_diagnostics.get("clamp_hard_mask")
        if clamp_diagnostics.get("clamp_decoded_pre") is not None:
            output["clamp_decoded_pre"] = clamp_diagnostics["clamp_decoded_pre"]
        if clamp_diagnostics.get("clamp_decoded_post") is not None:
            output["clamp_decoded_post"] = clamp_diagnostics["clamp_decoded_post"]
        if clamp_diagnostics.get("clamp_decoded_reencoded") is not None:
            output["clamp_decoded_reencoded"] = clamp_diagnostics["clamp_decoded_reencoded"]
        if clamp_diagnostics.get("clamp_target_rgb") is not None:
            output["clamp_target_rgb"] = clamp_diagnostics["clamp_target_rgb"]
        if clamp_diagnostics.get("clamp_delta_rgb") is not None:
            clamp_delta = clamp_diagnostics["clamp_delta_rgb"]
            output["clamp_delta_rgb"] = clamp_delta
            output["seam_denoise_clamp_delta_mean"] = float(clamp_delta.abs().mean().item())
            output["seam_denoise_clamp_delta_max"] = float(clamp_delta.abs().max().item())
        elif clamp_diagnostics.get("clamp_delta_latent") is not None:
            clamp_delta = clamp_diagnostics["clamp_delta_latent"]
            output["clamp_delta_latent"] = clamp_delta.detach().float().cpu()
            output["seam_denoise_clamp_delta_mean"] = float(clamp_delta.abs().mean().item())
            output["seam_denoise_clamp_delta_max"] = float(clamp_delta.abs().max().item())
        if clamp_diagnostics.get("clamp_hard_mask_raw") is not None:
            output["clamp_hard_mask_raw"] = clamp_diagnostics["clamp_hard_mask_raw"]
        if clamp_diagnostics.get("clamp_soft_mask_latent") is not None:
            output["clamp_soft_mask_latent"] = clamp_diagnostics["clamp_soft_mask_latent"]
        if clamp_diagnostics.get("clamp_hard_threshold") is not None:
            output["seam_denoise_clamp_hard_threshold"] = float(clamp_diagnostics["clamp_hard_threshold"].item())
        for stat_key in (
            "clamp_pre_rgb_min",
            "clamp_pre_rgb_max",
            "clamp_pre_rgb_mean",
            "clamp_target_rgb_min",
            "clamp_target_rgb_max",
            "clamp_target_rgb_mean",
            "clamp_post_rgb_min",
            "clamp_post_rgb_max",
            "clamp_post_rgb_mean",
        ):
            if clamp_diagnostics.get(stat_key) is not None:
                output[stat_key] = float(clamp_diagnostics[stat_key].item())
    if clamp_payload is not None:
        if output.get("clamp_target_rgb") is None and clamp_payload.get("target_rgb") is not None:
            output["clamp_target_rgb"] = clamp_payload["target_rgb"].detach().float().cpu()
        output["seam_denoise_clamp_applied_px"] = float(clamp_payload["applied_px"].item())
        output["seam_denoise_clamp_hard_applied_px"] = float(clamp_payload["hard_applied_px"].item())
        output["seam_denoise_clamp_feather_applied_px"] = float(clamp_payload["feather_applied_px"].item())
        output["seam_denoise_clamp_undefined_overlap_px"] = float(clamp_payload["undefined_overlap_px"].item())
        if clamp_payload.get("latent_support_boundary_px") is not None:
            output["seam_denoise_clamp_latent_support_boundary_px"] = float(clamp_payload["latent_support_boundary_px"].item())
        if clamp_payload.get("latent_hard_support_touch_px") is not None:
            output["seam_denoise_clamp_latent_hard_support_touch_px"] = float(clamp_payload["latent_hard_support_touch_px"].item())
        if clamp_payload.get("latent_hard_touches_support") is not None:
            output["seam_denoise_clamp_latent_hard_touches_support"] = float(clamp_payload["latent_hard_touches_support"].item())
        if clamp_payload.get("latent_boundary_gap_px") is not None:
            output["seam_denoise_clamp_latent_boundary_gap_px"] = float(clamp_payload["latent_boundary_gap_px"].item())
        if clamp_payload.get("latent_boundary_gap_ratio") is not None:
            output["seam_denoise_clamp_latent_boundary_gap_ratio"] = float(clamp_payload["latent_boundary_gap_ratio"].item())
        if clamp_payload.get("hard_threshold") is not None:
            output["seam_denoise_clamp_hard_threshold"] = float(clamp_payload["hard_threshold"].item())
    if debug_latent is not None:
        output["pred_x0_latent"] = debug_latent
    return output


def _write_json(path: str, payload: Dict[str, object]) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp, path)


def _pairwise_mse(images: List[np.ndarray]) -> float:
    if len(images) < 2:
        return 1.0
    max_mse = 0.0
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            mse = float(np.mean((images[i].astype(np.float32) - images[j].astype(np.float32)) ** 2) / (255.0 * 255.0))
            max_mse = max(max_mse, mse)
    return max_mse


def _build_contact_sheet(rows: List[Tuple[str, List[Image.Image]]], headers: List[str], out_path: str, tile_min_size: int) -> None:
    if not rows:
        return
    first = rows[0][1][0]
    tile_w = max(tile_min_size, first.width)
    tile_h = max(tile_min_size, first.height)
    label_w = 280
    pad = 8
    header_h = 36
    width = label_w + (tile_w + pad) * len(headers) + pad
    height = header_h + (tile_h + pad) * len(rows) + pad
    board = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(board)

    for idx, header in enumerate(headers):
        x = label_w + pad + idx * (tile_w + pad)
        draw.text((x + 4, 10), header, fill=(230, 230, 230))

    for r, (label, images) in enumerate(rows):
        y = header_h + pad + r * (tile_h + pad)
        draw.text((10, y + 8), label, fill=(240, 240, 240))
        for c, image in enumerate(images):
            x = label_w + pad + c * (tile_w + pad)
            board.paste(image.convert("RGB").resize((tile_w, tile_h), Image.Resampling.NEAREST), (x, y))

    board.save(out_path)


def resolve_eval_samples(
    dataset,
    eval_manifest_path: str,
    max_samples: int,
) -> List[EvalSample]:
    key_to_index: Dict[str, List[int]] = {}
    for idx, record in enumerate(dataset.records):
        key = build_sample_key(record["image_name"], record["crop_box"])
        key_to_index.setdefault(key, []).append(idx)

    if not os.path.isfile(eval_manifest_path):
        raise FileNotFoundError(f"eval manifest not found: {eval_manifest_path}")

    rows: List[EvalSample] = []
    with open(eval_manifest_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            sample_key = (row.get("sample_key") or "").strip()
            if not sample_key:
                raise ValueError(f"eval manifest row {i + 2} missing sample_key")
            matches = key_to_index.get(sample_key, [])
            if len(matches) != 1:
                raise ValueError(f"sample_key '{sample_key}' resolved to {len(matches)} matches")
            record = dataset.records[matches[0]]
            rows.append(
                EvalSample(
                    eval_id=(row.get("eval_id") or f"eval_{i:02d}"),
                    category=(row.get("category") or "uncategorized"),
                    sample_key=sample_key,
                    dataset_index=matches[0],
                    image_name=record["image_name"],
                    crop_box=tuple(int(v) for v in record["crop_box"]),
                    generation_strategy=str(record.get("generation_strategy") or ""),
                )
            )

    if not rows:
        raise ValueError("eval manifest resolved to 0 samples")

    return rows[: max(1, int(max_samples))]


def run_eval_step(
    *,
    step_label: str,
    output_dir: str,
    run_name: str,
    pretrain: bool,
    optimizer_steps_completed: int,
    dataset,
    resolved_samples: Sequence[EvalSample],
    unet: torch.nn.Module,
    control_net: torch.nn.Module,
    vae: torch.nn.Module,
    cached_text: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    eval_config: Dict[str, object],
    scheduler_config: Dict[str, object],
    device: torch.device,
    weight_dtype: torch.dtype,
    control_dtype: torch.dtype,
    vae_dtype: torch.dtype,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    step_dir = os.path.join(output_dir, step_label)
    os.makedirs(step_dir, exist_ok=True)

    unet.to(device=device, dtype=weight_dtype).eval()
    control_net.to(device=device, dtype=control_dtype).eval()
    vae.to(device=device, dtype=vae_dtype).eval()

    scheduler = DDPMScheduler(**scheduler_config)
    seeds = [int(s) for s in eval_config["seeds"]]
    primary_seed = seeds[0]

    run_config = {
        "run_name": run_name,
        "step_label": step_label,
        "pretrain": bool(pretrain),
        "optimizer_steps_completed": int(optimizer_steps_completed),
        "scheduler_config": scheduler_config,
        "scheduler_hash": hashlib.sha256(json.dumps(scheduler_config, sort_keys=True).encode("utf-8")).hexdigest(),
        "weight_dtype": str(weight_dtype),
        "control_dtype": str(control_dtype),
        "vae_dtype": str(vae_dtype),
        "inference_steps": int(eval_config["inference_steps"]),
        "seeds": seeds,
        "prompt": str(eval_config["prompt"]),
        "prompt2": str(eval_config["prompt2"]),
        "alpha_mode": {
            "capture_alpha_outputs": True,
            "alpha_preview_mode": str(eval_config.get("alpha_preview_mode", "mask")),
            "alpha_output_source": str(eval_config.get("alpha_output_source", "main")),
        },
        "expanded_prediction_enabled": bool(eval_config.get("expanded_prediction_enabled", False)),
        "expanded_halo_px": int(eval_config.get("expanded_halo_px", 0)),
        "seam_denoise_clamp": bool(eval_config.get("seam_denoise_clamp", False)),
        "seam_denoise_clamp_inner_px": int(eval_config.get("seam_denoise_clamp_inner_px", 0)),
        "seam_denoise_clamp_feather_px": int(eval_config.get("seam_denoise_clamp_feather_px", 160)),
        "seam_denoise_clamp_feather_profile": str(eval_config.get("seam_denoise_clamp_feather_profile", "smoothstep")),
        "seam_denoise_clamp_every_n_steps": int(eval_config.get("seam_denoise_clamp_every_n_steps", 1)),
        "seam_denoise_clamp_mode": str(eval_config.get("seam_denoise_clamp_mode", "latent")),
    }
    _write_json(os.path.join(step_dir, "eval_run_config.json"), run_config)

    rows_for_board: List[Tuple[str, List[Image.Image]]] = []
    metrics_rows: List[Dict[str, object]] = []
    resolved_rows: List[Dict[str, object]] = []
    collapse_images: List[np.ndarray] = []

    terrain_mask_index = dataset.channel_names.index("terrain_mask")
    terrain_black_is_terrain = bool(eval_config.get("terrain_mask_black_is_terrain", True))
    conditioning_spec = build_model_visible_conditioning_spec(
        seam_enabled=bool(getattr(dataset, "seam_enabled", False)),
        channel_names=getattr(dataset, "channel_names", []),
        full_conditioning_channel_names=getattr(dataset, "full_conditioning_channel_names", []),
        style_conditioning_channel_names=getattr(dataset, "style_conditioning_channel_names", []),
        seam_config=getattr(dataset, "seam_config", {}),
        style_ratio_config=getattr(dataset, "style_ratio_config", {}),
        terrain_mask_channel_index=int(getattr(dataset, "terrain_mask_channel_index", -1)),
        terrain_mask_black_is_terrain=bool(getattr(dataset, "terrain_mask_black_is_terrain", True)),
        alpha_binary_threshold=float(getattr(dataset, "alpha_binary_threshold", 0.5)),
    )
    full_scene_for_panel: Optional[Tuple[EvalSample, Dict[str, object], Dict[str, object]]] = None

    for sample_info in resolved_samples:
        sample = dataset[sample_info.dataset_index]
        sem_hash = hashlib.sha256(sample["conditioning_images"].detach().cpu().numpy().tobytes()).hexdigest()

        target_alpha = sample["alpha_target"]
        terrain_prior_raw = sample["conditioning_images"][terrain_mask_index].detach().float().clamp(0.0, 1.0)
        terrain_prior = _terrain_mask_to_occupancy(terrain_prior_raw, terrain_black_is_terrain)
        if target_alpha is None:
            target_alpha = terrain_prior.clone()

        primary_render = None
        seed_rgb_list: List[Image.Image] = []
        for seed in seeds:
            render = _render_one(
                sample=sample,
                unet=unet,
                control_net=control_net,
                vae=vae,
                scheduler=scheduler,
                cached_text=cached_text,
                device=device,
                weight_dtype=weight_dtype,
                control_dtype=control_dtype,
                vae_dtype=vae_dtype,
                steps=int(eval_config["inference_steps"]),
                seed=seed,
                write_latent_debug=bool(eval_config.get("write_latent_debug", False)) and seed == primary_seed,
                alpha_output_source=str(eval_config.get("alpha_output_source", "main")),
                expanded_prediction_enabled=bool(eval_config.get("expanded_prediction_enabled", False)),
                expanded_halo_px=int(eval_config.get("expanded_halo_px", 0)),
                seam_denoise_clamp=bool(eval_config.get("seam_denoise_clamp", False)),
                seam_denoise_clamp_inner_px=int(eval_config.get("seam_denoise_clamp_inner_px", 0)),
                seam_denoise_clamp_feather_px=int(eval_config.get("seam_denoise_clamp_feather_px", 160)),
                seam_denoise_clamp_feather_profile=str(eval_config.get("seam_denoise_clamp_feather_profile", "smoothstep")),
                seam_denoise_clamp_every_n_steps=int(eval_config.get("seam_denoise_clamp_every_n_steps", 1)),
                seam_denoise_clamp_mode=str(eval_config.get("seam_denoise_clamp_mode", "latent")),
                seam_denoise_clamp_hard_threshold=float(eval_config.get("seam_denoise_clamp_hard_threshold", 0.5)),
                conditioning_spec=conditioning_spec,
            )
            rgb_path = os.path.join(step_dir, f"{sample_info.eval_id}_seed{seed:06d}_rgb.png")
            pred_alpha_path = os.path.join(step_dir, f"{sample_info.eval_id}_seed{seed:06d}_pred_alpha.png")
            rgba_path = os.path.join(step_dir, f"{sample_info.eval_id}_seed{seed:06d}_rgba.png")
            render["rgb"].save(rgb_path)
            render["pred_alpha_img"].save(pred_alpha_path)
            render["rgba"].save(rgba_path)
            seed_rgb_list.append(render["rgb"])

            if seed == primary_seed:
                primary_render = render
                collapse_images.append(np.asarray(render["rgb"].convert("RGB"), dtype=np.uint8))
                _mask_to_image(target_alpha).save(os.path.join(step_dir, f"{sample_info.eval_id}_target_alpha.png"))
                _mask_to_image(terrain_prior).save(os.path.join(step_dir, f"{sample_info.eval_id}_terrain_prior.png"))
                _mask_to_image(terrain_prior_raw).save(os.path.join(step_dir, f"{sample_info.eval_id}_terrain_prior_raw.png"))
                if "pred_x0_latent" in render:
                    debug_dir = os.path.join(step_dir, "debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    torch.save(render["pred_x0_latent"], os.path.join(debug_dir, f"{sample_info.eval_id}_seed{seed:06d}_latent.pt"))
                if render.get("clamp_combined_mask") is not None:
                    debug_dir = os.path.join(step_dir, "debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    _mask_to_image(render["clamp_combined_mask"].squeeze(0) if render["clamp_combined_mask"].ndim == 3 else render["clamp_combined_mask"]).save(
                        os.path.join(debug_dir, f"{sample_info.eval_id}_seed{seed:06d}_clamp_applied_mask.png")
                    )
                if render.get("clamp_hard_mask") is not None:
                    debug_dir = os.path.join(step_dir, "debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    _mask_to_image(render["clamp_hard_mask"].squeeze(0) if render["clamp_hard_mask"].ndim == 3 else render["clamp_hard_mask"]).save(
                        os.path.join(debug_dir, f"{sample_info.eval_id}_seed{seed:06d}_clamp_hard_mask.png")
                    )
                if render.get("clamp_target_rgb") is not None:
                    debug_dir = os.path.join(step_dir, "debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    _tensor_to_image(render["clamp_target_rgb"]).save(
                        os.path.join(debug_dir, f"{sample_info.eval_id}_seed{seed:06d}_clamp_target_rgb.png")
                    )
                if render.get("clamp_decoded_pre") is not None:
                    debug_dir = os.path.join(step_dir, "debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    _tensor_to_image(render["clamp_decoded_pre"]).save(
                        os.path.join(debug_dir, f"{sample_info.eval_id}_seed{seed:06d}_preclamp_rgb.png")
                    )
                if render.get("clamp_decoded_post") is not None:
                    debug_dir = os.path.join(step_dir, "debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    _tensor_to_image(render["clamp_decoded_post"]).save(
                        os.path.join(debug_dir, f"{sample_info.eval_id}_seed{seed:06d}_postclamp_rgb.png")
                    )
                if render.get("clamp_delta_rgb") is not None:
                    debug_dir = os.path.join(step_dir, "debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    _float_to_grayscale_image(render["clamp_delta_rgb"].abs().mean(dim=0)).save(
                        os.path.join(debug_dir, f"{sample_info.eval_id}_seed{seed:06d}_clamp_delta_map.png")
                    )

        assert primary_render is not None
        edge_map_var = _edge_map_variance([_smooth_edge_map(img) for img in seed_rgb_list])

        halo_inner_recon_l1 = 0.0
        halo_outer_recon_l1 = 0.0
        halo_inner_edge_1px_rgb_loss = 0.0
        halo_inner_edge_4px_rgb_loss = 0.0
        halo_inner_8px_rgb_loss = 0.0
        halo_inner_16px_rgb_loss = 0.0
        pre_clamp_inner_8px_loss = 0.0
        post_clamp_inner_8px_loss = 0.0
        post_clamp_decoded_inner_8px_loss = 0.0
        final_inner_8px_loss = 0.0
        interior_continuation_l1 = 0.0
        halo_to_interior_alignment = 0.0
        halo_effect_strength = 0.0
        expanded_vs_direct_rgb_l1 = 0.0
        expanded_vs_direct_alpha_l1 = 0.0
        expanded_halo_copy_diff_mean = 0.0
        expanded_halo_copy_diff_max = 0.0
        use_expanded = bool(eval_config.get("expanded_prediction_enabled", False)) and int(eval_config.get("expanded_halo_px", 0)) > 0
        if use_expanded:
            halo_px = int(eval_config.get("expanded_halo_px", 0))
            interior_h = int(sample["target_sizes_hw"][0].item())
            interior_w = int(sample["target_sizes_hw"][1].item())
            exp_h, exp_w = expanded_hw(interior_h, interior_w, halo_px)

            expanded_rgb = primary_render["expanded_rgb_tensor"].detach().float().cpu().clamp(-1.0, 1.0)
            expanded_rgb = ((expanded_rgb + 1.0) * 0.5).permute(1, 2, 0).numpy()
            expanded_alpha = primary_render["expanded_alpha_prob"].detach().float().cpu().numpy()
            pred_rgba_exp = np.concatenate([expanded_rgb, expanded_alpha[..., None]], axis=2)
            pred_rgb_exp = pred_rgba_exp[..., :3]

            expected_rgba_exp = _build_expected_expanded_seam_rgba(
                sample=sample,
                interior_h=interior_h,
                interior_w=interior_w,
                halo_px=halo_px,
            )
            if expected_rgba_exp is not None and expected_rgba_exp.shape[0] == exp_h and expected_rgba_exp.shape[1] == exp_w:
                expected_rgb_exp = expected_rgba_exp[..., :3]
                band = max(1, min(int(eval_config.get("halo_inner_eval_px", 32)), halo_px))
                continuation_band = max(1, min(int(sample.get("seam_strip_width_px", torch.tensor(float(halo_px))).item()), halo_px))
                masks = _build_expanded_edge_masks(exp_h, exp_w, halo_px=halo_px, band_px=band)
                edge_flags = sample.get("edge_defined_flags")
                if isinstance(edge_flags, torch.Tensor):
                    ef = edge_flags.detach().float().cpu().numpy()
                else:
                    ef = np.ones((4,), dtype=np.float32)
                sides = ["top", "bottom", "right", "left"]

                halo_vals: List[float] = []
                halo_outer_vals: List[float] = []
                halo_edge_1_vals: List[float] = []
                halo_edge_4_vals: List[float] = []
                halo_edge_8_vals: List[float] = []
                halo_edge_16_vals: List[float] = []
                pre_clamp_edge_8_vals: List[float] = []
                post_clamp_edge_8_vals: List[float] = []
                final_edge_8_vals: List[float] = []
                interior_vals: List[float] = []
                align_vals: List[float] = []
                copy_diff_sums: List[float] = []
                copy_diff_counts: List[float] = []
                copy_diff_max_values: List[float] = []
                per_pixel_rgba_diff = np.mean(np.abs(pred_rgba_exp - expected_rgba_exp), axis=2)
                pre_clamp_rgb = primary_render.get("clamp_decoded_pre")
                post_clamp_rgb = primary_render.get("clamp_decoded_post")
                if isinstance(pre_clamp_rgb, torch.Tensor):
                    pre_clamp_rgb = ((pre_clamp_rgb.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).permute(1, 2, 0).numpy()
                else:
                    pre_clamp_rgb = None
                if isinstance(post_clamp_rgb, torch.Tensor):
                    post_clamp_rgb = ((post_clamp_rgb.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).permute(1, 2, 0).numpy()
                else:
                    post_clamp_rgb = None
                for side_idx, side in enumerate(sides):
                    if side_idx < len(ef) and ef[side_idx] < 0.5:
                        continue
                    m_h = masks[side]["halo_inner"]
                    m_ho = masks[side]["halo_outer"]
                    m_i = _build_expanded_edge_masks(exp_h, exp_w, halo_px=halo_px, band_px=continuation_band)[side]["interior_outer"]
                    m_all = masks[side]["halo_all"]
                    halo_vals.append(_masked_l1(pred_rgba_exp, expected_rgba_exp, m_h))
                    halo_outer_vals.append(_masked_l1(pred_rgba_exp, expected_rgba_exp, m_ho))
                    halo_edge_1_vals.append(_masked_l1(pred_rgb_exp, expected_rgb_exp, masks[side]["halo_inner_edge_1px"]))
                    halo_edge_4_vals.append(_masked_l1(pred_rgb_exp, expected_rgb_exp, masks[side]["halo_inner_edge_4px"]))
                    halo_edge_8_vals.append(_masked_l1(pred_rgb_exp, expected_rgb_exp, masks[side]["halo_inner_8px"]))
                    halo_edge_16_vals.append(_masked_l1(pred_rgb_exp, expected_rgb_exp, masks[side]["halo_inner_16px"]))
                    final_edge_8_vals.append(_masked_l1(pred_rgb_exp, expected_rgb_exp, masks[side]["halo_inner_8px"]))
                    if pre_clamp_rgb is not None:
                        pre_clamp_edge_8_vals.append(_masked_l1(pre_clamp_rgb, expected_rgb_exp, masks[side]["halo_inner_8px"]))
                    if post_clamp_rgb is not None:
                        post_clamp_edge_8_vals.append(_masked_l1(post_clamp_rgb, expected_rgb_exp, masks[side]["halo_inner_8px"]))
                    interior_vals.append(_masked_l1(pred_rgba_exp, expected_rgba_exp, m_i))
                    align_vals.append(
                        _mean_gradient_cosine(
                            _to_luma(pred_rgba_exp[..., :3]),
                            _to_luma(expected_rgba_exp[..., :3]),
                            m_i,
                        )
                    )
                    copy_diff_sums.append(float(np.sum(per_pixel_rgba_diff * m_all)))
                    copy_diff_counts.append(float(np.sum(m_all)))
                if halo_vals:
                    halo_inner_recon_l1 = float(np.mean(halo_vals))
                if halo_outer_vals:
                    halo_outer_recon_l1 = float(np.mean(halo_outer_vals))
                if halo_edge_4_vals:
                    halo_inner_edge_4px_rgb_loss = float(np.mean(halo_edge_4_vals))
                if halo_edge_8_vals:
                    halo_inner_8px_rgb_loss = float(np.mean(halo_edge_8_vals))
                if halo_edge_16_vals:
                    halo_inner_16px_rgb_loss = float(np.mean(halo_edge_16_vals))
                if pre_clamp_edge_8_vals:
                    pre_clamp_inner_8px_loss = float(np.mean(pre_clamp_edge_8_vals))
                if post_clamp_edge_8_vals:
                    post_clamp_inner_8px_loss = float(np.mean(post_clamp_edge_8_vals))
                    post_clamp_decoded_inner_8px_loss = float(np.mean(post_clamp_edge_8_vals))
                if final_edge_8_vals:
                    final_inner_8px_loss = float(np.mean(final_edge_8_vals))
                if interior_vals:
                    interior_continuation_l1 = float(np.mean(interior_vals))
                if align_vals:
                    halo_to_interior_alignment = float(np.mean(align_vals))
                if copy_diff_counts and sum(copy_diff_counts) > 0.0:
                    expanded_halo_copy_diff_mean = float(sum(copy_diff_sums) / max(sum(copy_diff_counts), 1e-8))
                if copy_diff_max_values:
                    expanded_halo_copy_diff_max = float(max(copy_diff_max_values))

                try:
                    full_cond = _compose_model_visible_conditioning(sample, sample["conditioning_images"], conditioning_spec)
                    if isinstance(full_cond, torch.Tensor):
                        cond_zero = full_cond.clone()
                        base_ch = int(sample["conditioning_images"].shape[0])
                        seam_ch_start = base_ch
                        seam_ch_end = min(base_ch + 16, int(cond_zero.shape[0]))
                        cond_zero[seam_ch_start:seam_ch_end] = 0.0
                        effect_render = _render_one(
                            sample=sample,
                            unet=unet,
                            control_net=control_net,
                            vae=vae,
                            scheduler=scheduler,
                            cached_text=cached_text,
                            device=device,
                            weight_dtype=weight_dtype,
                            control_dtype=control_dtype,
                            vae_dtype=vae_dtype,
                            steps=int(eval_config["inference_steps"]),
                            seed=primary_seed,
                            write_latent_debug=False,
                            alpha_output_source=str(eval_config.get("alpha_output_source", "main")),
                            expanded_prediction_enabled=use_expanded,
                            expanded_halo_px=halo_px,
                            conditioning_spec=conditioning_spec,
                            override_full_conditioning=cond_zero,
                        )
                        base_rgb = np.asarray(primary_render["rgb"].convert("RGB"), dtype=np.float32)
                        pert_rgb = np.asarray(effect_render["rgb"].convert("RGB"), dtype=np.float32)
                        halo_effect_strength = float(np.mean(np.abs(base_rgb - pert_rgb)) / 255.0)
                except Exception:
                    halo_effect_strength = 0.0

                try:
                    direct_render = _render_one(
                        sample=sample,
                        unet=unet,
                        control_net=control_net,
                        vae=vae,
                        scheduler=scheduler,
                        cached_text=cached_text,
                        device=device,
                        weight_dtype=weight_dtype,
                        control_dtype=control_dtype,
                        vae_dtype=vae_dtype,
                        steps=int(eval_config["inference_steps"]),
                        seed=primary_seed,
                        write_latent_debug=False,
                        alpha_output_source=str(eval_config.get("alpha_output_source", "main")),
                        expanded_prediction_enabled=False,
                        expanded_halo_px=0,
                        conditioning_spec=conditioning_spec,
                    )
                    base_rgb = np.asarray(primary_render["rgb"].convert("RGB"), dtype=np.float32)
                    direct_rgb = np.asarray(direct_render["rgb"].convert("RGB"), dtype=np.float32)
                    expanded_vs_direct_rgb_l1 = float(np.mean(np.abs(base_rgb - direct_rgb)) / 255.0)
                    direct_alpha = direct_render["pred_alpha_prob"].detach().float().cpu().numpy()
                    base_alpha = primary_render["pred_alpha_prob"].detach().float().cpu().numpy()
                    expanded_vs_direct_alpha_l1 = float(np.mean(np.abs(base_alpha - direct_alpha)))
                except Exception:
                    expanded_vs_direct_rgb_l1 = 0.0
                    expanded_vs_direct_alpha_l1 = 0.0

        p = primary_render["pred_alpha_prob"].detach().float().cpu()
        p_logits = primary_render["pred_alpha_logits"].detach().float().cpu()
        t = terrain_prior.detach().float().cpu()
        t_raw = terrain_prior_raw.detach().float().cpu()
        t_alpha = target_alpha.detach().float().cpu()
        threshold = float(eval_config.get("binary_threshold", 0.5))
        b = (p >= threshold).float()
        tbin = (t >= threshold).float()
        tbin_raw = (t_raw >= threshold).float()
        tbin_raw_inv = 1.0 - tbin_raw
        t_alpha_bin = (t_alpha >= threshold).float()
        inter = float((b * tbin).sum().item())
        union = float((b + tbin - b * tbin).sum().item())
        alpha_iou_terrain = inter / max(union, 1e-6)
        inter_target = float((b * t_alpha_bin).sum().item())
        union_target = float((b + t_alpha_bin - b * t_alpha_bin).sum().item())
        alpha_iou_target = inter_target / max(union_target, 1e-6)
        inter_raw = float((b * tbin_raw).sum().item())
        union_raw = float((b + tbin_raw - b * tbin_raw).sum().item())
        alpha_iou_terrain_rawpol = inter_raw / max(union_raw, 1e-6)
        inter_raw_inv = float((b * tbin_raw_inv).sum().item())
        union_raw_inv = float((b + tbin_raw_inv - b * tbin_raw_inv).sum().item())
        alpha_iou_terrain_inverted_rawpol = inter_raw_inv / max(union_raw_inv, 1e-6)

        supervision_mask = sample["trusted_mask"].detach().float().cpu().unsqueeze(0).unsqueeze(0)
        supervision_mask = _expand_mask(supervision_mask, int(eval_config.get("supervision_expand_px", 0))).squeeze(0).squeeze(0)
        supervision_mask = supervision_mask.clamp(0.0, 1.0)
        masked_inter_target = float((b * t_alpha_bin * supervision_mask).sum().item())
        masked_union_target = float(((b + t_alpha_bin - b * t_alpha_bin) * supervision_mask).sum().item())
        alpha_iou_target_masked = masked_inter_target / max(masked_union_target, 1e-6)

        alpha_bce = float(F.binary_cross_entropy(p.clamp(1e-6, 1.0 - 1e-6), tbin, reduction="mean").item())
        alpha_corr = _pearson_corr(p, t)
        alpha_occ = float(p.mean().item())
        alpha_speckle = _speckle_ratio(p)
        near_zero_prob_frac = float((p <= 0.01).float().mean().item())
        near_one_prob_frac = float((p >= 0.99).float().mean().item())
        logit_hist_bins = [-12.0, -8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 12.0]
        prob_hist_bins = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
        logit_hist_counts = _histogram_counts(p_logits, logit_hist_bins)
        prob_hist_counts = _histogram_counts(p, prob_hist_bins)

        metrics_rows.append(
            {
                "eval_id": sample_info.eval_id,
                "category": sample_info.category,
                "sample_key": sample_info.sample_key,
                "alpha_iou": alpha_iou_terrain,
                "alpha_iou_terrain_rawpol": alpha_iou_terrain_rawpol,
                "alpha_iou_terrain_inverted_rawpol": alpha_iou_terrain_inverted_rawpol,
                "alpha_iou_target": alpha_iou_target,
                "alpha_iou_target_masked": alpha_iou_target_masked,
                "alpha_bce": alpha_bce,
                "alpha_corr": alpha_corr,
                "alpha_occ": alpha_occ,
                "alpha_speckle": alpha_speckle,
                "pred_near0_01": near_zero_prob_frac,
                "pred_near1_99": near_one_prob_frac,
                "alpha_logits_hist_bins": json.dumps(logit_hist_bins),
                "alpha_logits_hist_counts": json.dumps(logit_hist_counts),
                "alpha_sigmoid_hist_bins": json.dumps(prob_hist_bins),
                "alpha_sigmoid_hist_counts": json.dumps(prob_hist_counts),
                "seed_edge_map_var": edge_map_var,
                "halo_inner_recon_l1": halo_inner_recon_l1,
                "halo_outer_recon_l1": halo_outer_recon_l1,
                "halo_inner_edge_1px_rgb_loss": halo_inner_edge_1px_rgb_loss,
                "halo_inner_edge_4px_rgb_loss": halo_inner_edge_4px_rgb_loss,
                "halo_inner_8px_rgb_loss": halo_inner_8px_rgb_loss,
                "halo_inner_16px_rgb_loss": halo_inner_16px_rgb_loss,
                "pre_clamp_inner_8px_loss": pre_clamp_inner_8px_loss,
                "post_clamp_inner_8px_loss": post_clamp_inner_8px_loss,
                "post_clamp_decoded_inner_8px_loss": post_clamp_decoded_inner_8px_loss,
                "final_inner_8px_loss": final_inner_8px_loss,
                "interior_continuation_l1": interior_continuation_l1,
                "halo_to_interior_alignment": halo_to_interior_alignment,
                "halo_effect_strength": halo_effect_strength,
                "expanded_vs_direct_rgb_l1": expanded_vs_direct_rgb_l1,
                "expanded_vs_direct_alpha_l1": expanded_vs_direct_alpha_l1,
                "expanded_halo_copy_diff_mean": expanded_halo_copy_diff_mean,
                "expanded_halo_copy_diff_max": expanded_halo_copy_diff_max,
                "seam_denoise_clamp_enabled": float(render.get("seam_denoise_clamp_enabled", 0.0)),
                "seam_denoise_clamp_steps_applied": float(render.get("seam_denoise_clamp_steps_applied", 0.0)),
                "seam_denoise_clamp_expected_steps": float(render.get("seam_denoise_clamp_expected_steps", 0.0)),
                "seam_denoise_clamp_applied_px": float(render.get("seam_denoise_clamp_applied_px", 0.0)),
                "seam_denoise_clamp_hard_applied_px": float(render.get("seam_denoise_clamp_hard_applied_px", 0.0)),
                "seam_denoise_clamp_feather_applied_px": float(render.get("seam_denoise_clamp_feather_applied_px", 0.0)),
                "seam_denoise_clamp_undefined_overlap_px": float(render.get("seam_denoise_clamp_undefined_overlap_px", 0.0)),
                "seam_denoise_clamp_delta_mean": float(render.get("seam_denoise_clamp_delta_mean", 0.0)),
                "seam_denoise_clamp_delta_max": float(render.get("seam_denoise_clamp_delta_max", 0.0)),
                "seam_denoise_clamp_latent_support_boundary_px": float(render.get("seam_denoise_clamp_latent_support_boundary_px", 0.0)),
                "seam_denoise_clamp_latent_hard_support_touch_px": float(render.get("seam_denoise_clamp_latent_hard_support_touch_px", 0.0)),
                "seam_denoise_clamp_latent_hard_touches_support": float(render.get("seam_denoise_clamp_latent_hard_touches_support", 0.0)),
                "seam_denoise_clamp_latent_boundary_gap_px": float(render.get("seam_denoise_clamp_latent_boundary_gap_px", 0.0)),
                "seam_denoise_clamp_latent_boundary_gap_ratio": float(render.get("seam_denoise_clamp_latent_boundary_gap_ratio", 0.0)),
            }
        )

        metric_tensors_dir = os.path.join(step_dir, "metric_tensors")
        os.makedirs(metric_tensors_dir, exist_ok=True)
        torch.save(
            {
                "pred_alpha_logits": p_logits,
                "pred_alpha_prob": p,
                "pred_alpha_bin": b,
                "terrain_prior": t,
                "terrain_prior_bin": tbin,
                "target_alpha": t_alpha,
                "target_alpha_bin": t_alpha_bin,
                "supervision_mask": supervision_mask,
                "threshold": threshold,
            },
            os.path.join(metric_tensors_dir, f"{sample_info.eval_id}_seed{primary_seed:06d}_iou_tensors.pt"),
        )

        resolved_rows.append(
            {
                "eval_id": sample_info.eval_id,
                "category": sample_info.category,
                "sample_key": sample_info.sample_key,
                "dataset_index": sample_info.dataset_index,
                "image_name": sample_info.image_name,
                "crop_box_x": sample_info.crop_box[0],
                "crop_box_y": sample_info.crop_box[1],
                "crop_box_w": sample_info.crop_box[2],
                "crop_box_h": sample_info.crop_box[3],
                "generation_strategy": sample_info.generation_strategy,
                "semantic_tensor_sha256": sem_hash,
            }
        )

        semantic_preview = _float_to_grayscale_image(sample["conditioning_images"][terrain_mask_index]).convert("RGB")
        rows_for_board.append(
            (
                f"{sample_info.eval_id} | {sample_info.category}",
                [
                    semantic_preview,
                    _mask_to_image(target_alpha).convert("RGB"),
                    _mask_to_image(terrain_prior).convert("RGB"),
                    primary_render["pred_alpha_img"].convert("RGB"),
                    primary_render["rgb"].convert("RGB"),
                    primary_render["rgba"].convert("RGB"),
                ],
            )
        )

        if full_scene_for_panel is None and sample_info.generation_strategy == "full_scene":
            full_scene_for_panel = (sample_info, sample, primary_render)

        del primary_render
        del seed_rgb_list
        del p
        del p_logits
        del t
        del t_raw
        del t_alpha
        del b
        del tbin
        del tbin_raw
        del tbin_raw_inv
        del t_alpha_bin
        del supervision_mask
        clean_memory_on_device(device)

    _write_csv(
        os.path.join(step_dir, "resolved_eval_manifest.csv"),
        resolved_rows,
        [
            "eval_id",
            "category",
            "sample_key",
            "dataset_index",
            "image_name",
            "crop_box_x",
            "crop_box_y",
            "crop_box_w",
            "crop_box_h",
            "generation_strategy",
            "semantic_tensor_sha256",
        ],
    )
    _write_csv(
        os.path.join(step_dir, "metrics_alpha_alignment.csv"),
        metrics_rows,
        [
            "eval_id",
            "category",
            "sample_key",
            "alpha_iou",
            "alpha_iou_terrain_rawpol",
            "alpha_iou_terrain_inverted_rawpol",
            "alpha_iou_target",
            "alpha_iou_target_masked",
            "alpha_bce",
            "alpha_corr",
            "alpha_occ",
            "alpha_speckle",
            "pred_near0_01",
            "pred_near1_99",
            "alpha_logits_hist_bins",
            "alpha_logits_hist_counts",
            "alpha_sigmoid_hist_bins",
            "alpha_sigmoid_hist_counts",
            "seed_edge_map_var",
            "halo_inner_recon_l1",
            "halo_outer_recon_l1",
            "halo_inner_edge_1px_rgb_loss",
            "halo_inner_edge_4px_rgb_loss",
            "halo_inner_8px_rgb_loss",
            "halo_inner_16px_rgb_loss",
            "pre_clamp_inner_8px_loss",
            "post_clamp_inner_8px_loss",
            "post_clamp_decoded_inner_8px_loss",
            "final_inner_8px_loss",
            "interior_continuation_l1",
            "halo_to_interior_alignment",
            "halo_effect_strength",
            "expanded_vs_direct_rgb_l1",
            "expanded_vs_direct_alpha_l1",
            "expanded_halo_copy_diff_mean",
            "expanded_halo_copy_diff_max",
            "seam_denoise_clamp_enabled",
            "seam_denoise_clamp_steps_applied",
            "seam_denoise_clamp_expected_steps",
            "seam_denoise_clamp_applied_px",
            "seam_denoise_clamp_hard_applied_px",
            "seam_denoise_clamp_feather_applied_px",
            "seam_denoise_clamp_undefined_overlap_px",
            "seam_denoise_clamp_delta_mean",
            "seam_denoise_clamp_delta_max",
            "seam_denoise_clamp_latent_support_boundary_px",
            "seam_denoise_clamp_latent_hard_support_touch_px",
            "seam_denoise_clamp_latent_hard_touches_support",
            "seam_denoise_clamp_latent_boundary_gap_px",
            "seam_denoise_clamp_latent_boundary_gap_ratio",
        ],
    )

    headers = ["semantic", "target_alpha", "terrain_prior", "pred_alpha", "generated_rgb", "rgba"]
    board_a_path = os.path.join(step_dir, "board_a_alpha_alignment.png")
    _build_contact_sheet(rows_for_board, headers, board_a_path, tile_min_size=int(eval_config.get("board_tile_min_size", 256)))

    full_scene_dir = os.path.join(output_dir, "full_scene")
    os.makedirs(full_scene_dir, exist_ok=True)
    if full_scene_for_panel is None:
        first = resolved_samples[0]
        sample = dataset[first.dataset_index]
        render = _render_one(
            sample=sample,
            unet=unet,
            control_net=control_net,
            vae=vae,
            scheduler=scheduler,
            cached_text=cached_text,
            device=device,
            weight_dtype=weight_dtype,
            control_dtype=control_dtype,
            vae_dtype=vae_dtype,
            steps=int(eval_config["inference_steps"]),
            seed=primary_seed,
            write_latent_debug=False,
            alpha_output_source=str(eval_config.get("alpha_output_source", "main")),
            expanded_prediction_enabled=bool(eval_config.get("expanded_prediction_enabled", False)),
            expanded_halo_px=int(eval_config.get("expanded_halo_px", 0)),
            conditioning_spec=conditioning_spec,
        )
        full_scene_for_panel = (first, sample, render)

    fs_info, fs_sample, fs_render = full_scene_for_panel
    fs_sem = _float_to_grayscale_image(fs_sample["conditioning_images"][terrain_mask_index]).convert("RGB")
    fs_prior_raw = fs_sample["conditioning_images"][terrain_mask_index].detach().float().clamp(0.0, 1.0)
    fs_prior = _terrain_mask_to_occupancy(fs_prior_raw, terrain_black_is_terrain)
    fs_target = fs_sample["alpha_target"] if fs_sample["alpha_target"] is not None else fs_prior
    fs_rows = [
        (
            f"full_scene | {fs_info.eval_id}",
            [
                fs_sem,
                _mask_to_image(fs_target).convert("RGB"),
                _mask_to_image(fs_prior).convert("RGB"),
                fs_render["pred_alpha_img"].convert("RGB"),
                fs_render["rgb"].convert("RGB"),
                fs_render["rgba"].convert("RGB"),
            ],
        )
    ]
    full_scene_panel_path = os.path.join(full_scene_dir, f"full_scene_panel_{step_label}.png")
    _build_contact_sheet(fs_rows, headers, full_scene_panel_path, tile_min_size=int(eval_config.get("full_scene_tile_min_size", 512)))

    max_pairwise_mse = _pairwise_mse(collapse_images)
    if max_pairwise_mse <= float(eval_config.get("collapse_mse_threshold", 1e-4)):
        raise RuntimeError(
            f"eval outputs collapsed at {step_label}: max_pairwise_mse={max_pairwise_mse:.8f} <= threshold"
        )

    means = {
        "alpha_iou": float(np.mean([row["alpha_iou"] for row in metrics_rows])),
        "alpha_iou_terrain_rawpol": float(np.mean([row["alpha_iou_terrain_rawpol"] for row in metrics_rows])),
        "alpha_iou_terrain_inverted_rawpol": float(np.mean([row["alpha_iou_terrain_inverted_rawpol"] for row in metrics_rows])),
        "alpha_iou_target": float(np.mean([row["alpha_iou_target"] for row in metrics_rows])),
        "alpha_iou_target_masked": float(np.mean([row["alpha_iou_target_masked"] for row in metrics_rows])),
        "alpha_bce": float(np.mean([row["alpha_bce"] for row in metrics_rows])),
        "alpha_corr": float(np.mean([row["alpha_corr"] for row in metrics_rows])),
        "alpha_occ": float(np.mean([row["alpha_occ"] for row in metrics_rows])),
        "alpha_speckle": float(np.mean([row["alpha_speckle"] for row in metrics_rows])),
        "pred_near0_01": float(np.mean([row["pred_near0_01"] for row in metrics_rows])),
        "pred_near1_99": float(np.mean([row["pred_near1_99"] for row in metrics_rows])),
        "max_pairwise_mse": max_pairwise_mse,
        "seed_edge_map_var": float(np.mean([row["seed_edge_map_var"] for row in metrics_rows])),
        "halo_inner_recon_l1": float(np.mean([row.get("halo_inner_recon_l1", 0.0) for row in metrics_rows])),
        "halo_outer_recon_l1": float(np.mean([row.get("halo_outer_recon_l1", 0.0) for row in metrics_rows])),
        "halo_inner_edge_1px_rgb_loss": float(np.mean([row.get("halo_inner_edge_1px_rgb_loss", 0.0) for row in metrics_rows])),
        "halo_inner_edge_4px_rgb_loss": float(np.mean([row.get("halo_inner_edge_4px_rgb_loss", 0.0) for row in metrics_rows])),
        "halo_inner_8px_rgb_loss": float(np.mean([row.get("halo_inner_8px_rgb_loss", 0.0) for row in metrics_rows])),
        "halo_inner_16px_rgb_loss": float(np.mean([row.get("halo_inner_16px_rgb_loss", 0.0) for row in metrics_rows])),
        "pre_clamp_inner_8px_loss": float(np.mean([row.get("pre_clamp_inner_8px_loss", 0.0) for row in metrics_rows])),
        "post_clamp_inner_8px_loss": float(np.mean([row.get("post_clamp_inner_8px_loss", 0.0) for row in metrics_rows])),
        "post_clamp_decoded_inner_8px_loss": float(np.mean([row.get("post_clamp_decoded_inner_8px_loss", 0.0) for row in metrics_rows])),
        "final_inner_8px_loss": float(np.mean([row.get("final_inner_8px_loss", 0.0) for row in metrics_rows])),
        "interior_continuation_l1": float(np.mean([row.get("interior_continuation_l1", 0.0) for row in metrics_rows])),
        "halo_to_interior_alignment": float(np.mean([row.get("halo_to_interior_alignment", 0.0) for row in metrics_rows])),
        "halo_effect_strength": float(np.mean([row.get("halo_effect_strength", 0.0) for row in metrics_rows])),
        "expanded_vs_direct_rgb_l1": float(np.mean([row.get("expanded_vs_direct_rgb_l1", 0.0) for row in metrics_rows])),
        "expanded_vs_direct_alpha_l1": float(np.mean([row.get("expanded_vs_direct_alpha_l1", 0.0) for row in metrics_rows])),
        "expanded_halo_copy_diff_mean": float(np.mean([row.get("expanded_halo_copy_diff_mean", 0.0) for row in metrics_rows])),
        "expanded_halo_copy_diff_max": float(np.mean([row.get("expanded_halo_copy_diff_max", 0.0) for row in metrics_rows])),
        "seam_denoise_clamp_enabled": float(np.mean([row.get("seam_denoise_clamp_enabled", 0.0) for row in metrics_rows])),
        "seam_denoise_clamp_steps_applied": float(np.mean([row.get("seam_denoise_clamp_steps_applied", 0.0) for row in metrics_rows])),
        "seam_denoise_clamp_expected_steps": float(np.mean([row.get("seam_denoise_clamp_expected_steps", 0.0) for row in metrics_rows])),
        "seam_denoise_clamp_applied_px": float(np.mean([row.get("seam_denoise_clamp_applied_px", 0.0) for row in metrics_rows])),
        "seam_denoise_clamp_hard_applied_px": float(np.mean([row.get("seam_denoise_clamp_hard_applied_px", 0.0) for row in metrics_rows])),
        "seam_denoise_clamp_feather_applied_px": float(np.mean([row.get("seam_denoise_clamp_feather_applied_px", 0.0) for row in metrics_rows])),
        "seam_denoise_clamp_undefined_overlap_px": float(np.mean([row.get("seam_denoise_clamp_undefined_overlap_px", 0.0) for row in metrics_rows])),
        "seam_denoise_clamp_delta_mean": float(np.mean([row.get("seam_denoise_clamp_delta_mean", 0.0) for row in metrics_rows])),
        "seam_denoise_clamp_delta_max": float(np.mean([row.get("seam_denoise_clamp_delta_max", 0.0) for row in metrics_rows])),
        "seam_denoise_clamp_latent_support_boundary_px": float(np.mean([row.get("seam_denoise_clamp_latent_support_boundary_px", 0.0) for row in metrics_rows])),
        "seam_denoise_clamp_latent_hard_support_touch_px": float(np.mean([row.get("seam_denoise_clamp_latent_hard_support_touch_px", 0.0) for row in metrics_rows])),
        "seam_denoise_clamp_latent_hard_touches_support": float(np.mean([row.get("seam_denoise_clamp_latent_hard_touches_support", 0.0) for row in metrics_rows])),
        "seam_denoise_clamp_latent_boundary_gap_px": float(np.mean([row.get("seam_denoise_clamp_latent_boundary_gap_px", 0.0) for row in metrics_rows])),
        "seam_denoise_clamp_latent_boundary_gap_ratio": float(np.mean([row.get("seam_denoise_clamp_latent_boundary_gap_ratio", 0.0) for row in metrics_rows])),
        "seam_margin_inner_recon_l1": float(np.mean([row.get("halo_inner_recon_l1", 0.0) for row in metrics_rows])),
        "seam_margin_outer_recon_l1": float(np.mean([row.get("halo_outer_recon_l1", 0.0) for row in metrics_rows])),
        "seam_interior_continuation_l1": float(np.mean([row.get("interior_continuation_l1", 0.0) for row in metrics_rows])),
        "expanded_prediction_enabled": float(1.0 if bool(eval_config.get("expanded_prediction_enabled", False)) else 0.0),
        "expanded_halo_px": float(int(eval_config.get("expanded_halo_px", 0))),
        "step_label": step_label,
    }
    _write_json(os.path.join(step_dir, "step_summary.json"), means)
    return means


def build_progression_boards(
    *,
    output_dir: str,
    run_name: str,
    resolved_samples: Sequence[EvalSample],
    step_labels: Sequence[str],
    primary_seed: int,
) -> None:
    if not step_labels:
        return
    eval_dir = output_dir
    rows_rgb: List[Tuple[str, List[Image.Image]]] = []
    rows_alpha: List[Tuple[str, List[Image.Image]]] = []

    for sample in resolved_samples:
        rgb_images: List[Image.Image] = []
        alpha_images: List[Image.Image] = []
        available = True
        for step_label in step_labels:
            step_dir = os.path.join(eval_dir, step_label)
            rgb_path = os.path.join(step_dir, f"{sample.eval_id}_seed{primary_seed:06d}_rgb.png")
            alpha_path = os.path.join(step_dir, f"{sample.eval_id}_seed{primary_seed:06d}_pred_alpha.png")
            if not os.path.isfile(rgb_path) or not os.path.isfile(alpha_path):
                available = False
                break
            rgb_images.append(Image.open(rgb_path).convert("RGB"))
            alpha_images.append(Image.open(alpha_path).convert("RGB"))
        if not available:
            continue

        rows_rgb.append((f"{sample.eval_id} | {sample.category}", rgb_images))
        rows_alpha.append((f"{sample.eval_id} | {sample.category}", alpha_images))

    if not rows_rgb:
        return

    headers = list(step_labels)
    rgb_out = os.path.join(eval_dir, f"progression_{run_name}_{'_'.join(step_labels)}.png")
    alpha_out = os.path.join(eval_dir, f"progression_alpha_{run_name}_{'_'.join(step_labels)}.png")
    _build_contact_sheet(rows_rgb, headers, rgb_out, tile_min_size=256)
    _build_contact_sheet(rows_alpha, headers, alpha_out, tile_min_size=256)


def summarize_attempt(
    *,
    output_dir: str,
    eval_step_summaries: Dict[str, Dict[str, float]],
    loss_trace: Sequence[Dict[str, float]],
    eval_config: Dict[str, object],
) -> Dict[str, object]:
    if not eval_step_summaries:
        summary = {
            "decision": "ESCALATE_REVIEW",
            "reason": "no_eval_step_summaries",
            "failed_threshold_keys": ["eval_step_summaries"],
        }
        _write_json(os.path.join(output_dir, "attempt_summary.json"), summary)
        return summary


    step0 = eval_step_summaries.get("step_0000_pretrain")
    step200 = eval_step_summaries.get("step_0200") or eval_step_summaries.get("step_0120")

    diff_losses = [float(row.get("diffusion_loss", row.get("loss", 0.0))) for row in loss_trace if isinstance(row, dict)]
    if diff_losses:
        tail_start = max(0, int(len(diff_losses) * 2 / 3))
        tail = np.array(diff_losses[tail_start:], dtype=np.float32)
        diffusion_tail_mean = float(tail.mean())
        if len(tail) >= 2:
            x = np.arange(len(tail), dtype=np.float32)
            slope = float(np.polyfit(x, tail, 1)[0])
        else:
            slope = 0.0
    else:
        diffusion_tail_mean = float("inf")
        slope = float("inf")

    failed: List[str] = []
    if step200 is None:
        failed.append("missing_step200_or_step120")
    else:
        if step200["alpha_iou"] < float(eval_config.get("alpha_iou_min", 0.35)):
            failed.append("alpha_iou_min")
        if step200["alpha_bce"] > float(eval_config.get("alpha_bce_max", 0.62)):
            failed.append("alpha_bce_max")
        if step200["alpha_corr"] < float(eval_config.get("alpha_corr_min", 0.30)):
            failed.append("alpha_corr_min")
        if step200["alpha_occ"] < float(eval_config.get("alpha_occ_min", 0.08)):
            failed.append("alpha_occ_min")
        if step200["alpha_occ"] > float(eval_config.get("alpha_occ_max", 0.92)):
            failed.append("alpha_occ_max")
        if step200["alpha_speckle"] > float(eval_config.get("alpha_speckle_max", 0.45)):
            failed.append("alpha_speckle_max")
        if step200["max_pairwise_mse"] <= float(eval_config.get("collapse_mse_threshold", 1e-4)):
            failed.append("collapse_mse_threshold")

    if step0 is not None and step200 is not None:
        if (step200["alpha_iou"] - step0["alpha_iou"]) < float(eval_config.get("alpha_iou_delta_min", 0.08)):
            failed.append("alpha_iou_delta_min")
        if (step0["alpha_bce"] - step200["alpha_bce"]) < float(eval_config.get("alpha_bce_delta_min", 0.05)):
            failed.append("alpha_bce_delta_min")

    if slope > float(eval_config.get("diffusion_tail_slope_max", 0.002)):
        failed.append("diffusion_tail_slope_max")

    severe_fail_keys = {
        "missing_step200_or_step120",
        "collapse_mse_threshold",
    }
    if any(key in severe_fail_keys for key in failed):
        decision = "ESCALATE_REVIEW"
    elif not failed:
        decision = "PROMOTE_TO_LONG_RUN"
    else:
        decision = "CONTINUE_ITERATION"
    summary = {
        "decision": decision,
        "failed_threshold_keys": failed,
        "diffusion_tail_mean": diffusion_tail_mean,
        "diffusion_tail_slope": slope,
        "step_summaries": eval_step_summaries,
        "thresholds": {
            "alpha_iou_min": float(eval_config.get("alpha_iou_min", 0.35)),
            "alpha_bce_max": float(eval_config.get("alpha_bce_max", 0.62)),
            "alpha_corr_min": float(eval_config.get("alpha_corr_min", 0.30)),
            "alpha_occ_min": float(eval_config.get("alpha_occ_min", 0.08)),
            "alpha_occ_max": float(eval_config.get("alpha_occ_max", 0.92)),
            "alpha_speckle_max": float(eval_config.get("alpha_speckle_max", 0.45)),
            "alpha_iou_delta_min": float(eval_config.get("alpha_iou_delta_min", 0.08)),
            "alpha_bce_delta_min": float(eval_config.get("alpha_bce_delta_min", 0.05)),
            "diffusion_tail_slope_max": float(eval_config.get("diffusion_tail_slope_max", 0.002)),
            "collapse_mse_threshold": float(eval_config.get("collapse_mse_threshold", 1e-4)),
        },
    }
    _write_json(os.path.join(output_dir, "attempt_summary.json"), summary)
    return summary


# ── swap pair manifest loading ────────────────────────────────────────────────

def resolve_swap_pairs(
    dataset,
    swap_manifest_path: str,
) -> List[SwapPair]:
    """Resolve a swap-pair CSV manifest to dataset indices.

    Each row must have ``base_sample_key`` and ``swap_sample_key`` that match
    exactly one entry in ``dataset.records``.  Mirrors ``resolve_eval_samples``.
    """
    key_to_index: Dict[str, List[int]] = {}
    for idx, record in enumerate(dataset.records):
        key = build_sample_key(record["image_name"], record["crop_box"])
        key_to_index.setdefault(key, []).append(idx)

    if not os.path.isfile(swap_manifest_path):
        raise FileNotFoundError(f"swap manifest not found: {swap_manifest_path}")

    pairs: List[SwapPair] = []
    with open(swap_manifest_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            base_key = (row.get("base_sample_key") or "").strip()
            swap_key = (row.get("swap_sample_key") or "").strip()
            if not base_key or not swap_key:
                raise ValueError(f"swap manifest row {i + 2} missing base_sample_key or swap_sample_key")
            base_matches = key_to_index.get(base_key, [])
            swap_matches = key_to_index.get(swap_key, [])
            if len(base_matches) != 1:
                raise ValueError(
                    f"swap pair '{row.get('pair_id')}': base_sample_key '{base_key}' "
                    f"resolved to {len(base_matches)} dataset matches (expected 1)"
                )
            if len(swap_matches) != 1:
                raise ValueError(
                    f"swap pair '{row.get('pair_id')}': swap_sample_key '{swap_key}' "
                    f"resolved to {len(swap_matches)} dataset matches (expected 1)"
                )
            edit_mask_path = (row.get("edit_mask_path") or "").strip() or None
            pairs.append(
                SwapPair(
                    pair_id=(row.get("pair_id") or f"pair_{i:02d}").strip(),
                    base_image=(row.get("base_image") or "").strip(),
                    base_sample_key=base_key,
                    base_dataset_index=base_matches[0],
                    swap_image=(row.get("swap_image") or "").strip(),
                    swap_sample_key=swap_key,
                    swap_dataset_index=swap_matches[0],
                    edit_type=(row.get("edit_type") or "global").strip(),
                    primary_expected_effect=(row.get("primary_expected_effect") or "").strip(),
                    allowed_effects=(row.get("allowed_effects") or "").strip(),
                    disallowed_effects=(row.get("disallowed_effects") or "").strip(),
                    edit_mask_path=edit_mask_path,
                )
            )
    if not pairs:
        raise ValueError("swap manifest resolved to 0 pairs")
    return pairs


# ── semantic binding eval ─────────────────────────────────────────────────────

def run_semantic_binding_eval(
    *,
    step_label: str,
    output_dir: str,
    run_name: str,
    dataset,
    swap_pairs: Sequence[SwapPair],
    unet: torch.nn.Module,
    control_net: torch.nn.Module,
    vae: torch.nn.Module,
    cached_text: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    binding_config: Dict[str, object],
    scheduler_config: Dict[str, object],
    device: torch.device,
    weight_dtype: torch.dtype,
    control_dtype: torch.dtype,
    vae_dtype: torch.dtype,
) -> Dict[str, object]:
    """Run the semantic-binding diagnostic suite for one checkpoint.

    For each swap pair this renders and saves:
      base_rgb        — base atlas, primary seed (and N-seed panel for consistency)
      swap_rgb        — swap atlas, primary seed; abs + signed diffs vs base_rgb
      zero_cond_rgb   — zero conditioning (ControlNet ablation)
      shuffled_rgb    — spatially shuffled conditioning (ControlNet ablation)
      maskonly_rgb    — terrain_mask channel only (mask-only comparison)
      ns_<group>_rgb  — non-mask channels scrambled per group (null-space test)

    Metrics per pair are written to ``binding_metrics.csv`` and a contact-sheet
    board is saved as ``binding_board_<step_label>.png``.
    """
    step_dir = os.path.join(output_dir, step_label + "_binding")
    os.makedirs(step_dir, exist_ok=True)

    unet.to(device=device, dtype=weight_dtype).eval()
    control_net.to(device=device, dtype=control_dtype).eval()
    vae.to(device=device, dtype=vae_dtype).eval()

    scheduler = DDPMScheduler(**scheduler_config)

    seeds_panel: List[int] = [int(s) for s in binding_config.get("seeds_panel", [1234, 5678, 9012])]
    primary_seed: int = seeds_panel[0]
    inference_steps: int = int(binding_config.get("inference_steps", 8))
    blur_radius: float = float(binding_config.get("blur_radius", 1.5))
    terrain_mask_ch: int = int(binding_config.get("terrain_mask_channel_index", 3))
    ablation_seed: int = int(binding_config.get("ablation_shuffle_seed", 9999))
    alpha_output_source: str = str(binding_config.get("alpha_output_source", "main"))
    conditioning_spec = build_model_visible_conditioning_spec(
        seam_enabled=bool(getattr(dataset, "seam_enabled", False)),
        channel_names=getattr(dataset, "channel_names", []),
        full_conditioning_channel_names=getattr(dataset, "full_conditioning_channel_names", []),
        style_conditioning_channel_names=getattr(dataset, "style_conditioning_channel_names", []),
        seam_config=getattr(dataset, "seam_config", {}),
        style_ratio_config=getattr(dataset, "style_ratio_config", {}),
        terrain_mask_channel_index=int(getattr(dataset, "terrain_mask_channel_index", -1)),
        terrain_mask_black_is_terrain=bool(getattr(dataset, "terrain_mask_black_is_terrain", True)),
        alpha_binary_threshold=float(getattr(dataset, "alpha_binary_threshold", 0.5)),
    )

    null_space_groups: List[List[int]] = [
        [int(ch) for ch in grp]
        for grp in binding_config.get("null_space_channel_groups", [[4, 5, 6, 7], [8, 9, 10, 11], [0, 1, 2]])
    ]
    null_space_group_names: List[str] = list(
        binding_config.get("null_space_group_names", ["edge_channels", "openness_channels", "base_semantic"])
    )
    while len(null_space_group_names) < len(null_space_groups):
        null_space_group_names.append(f"group_{len(null_space_group_names)}")

    ns_ablation_seed_offset = 100  # distinct seed per group to avoid correlated permutations

    def _render(sample_: Dict, override: Optional[torch.Tensor], seed: int) -> Image.Image:
        """Render and return only the RGB PIL image."""
        return _render_one(
            sample=sample_,
            unet=unet,
            control_net=control_net,
            vae=vae,
            scheduler=scheduler,
            cached_text=cached_text,
            device=device,
            weight_dtype=weight_dtype,
            control_dtype=control_dtype,
            vae_dtype=vae_dtype,
            steps=inference_steps,
            seed=seed,
            write_latent_debug=False,
            alpha_output_source=alpha_output_source,
            seam_denoise_clamp=bool(binding_config.get("seam_denoise_clamp", False)),
            seam_denoise_clamp_inner_px=int(binding_config.get("seam_denoise_clamp_inner_px", 0)),
            seam_denoise_clamp_feather_px=int(binding_config.get("seam_denoise_clamp_feather_px", 160)),
            seam_denoise_clamp_feather_profile=str(binding_config.get("seam_denoise_clamp_feather_profile", "smoothstep")),
            seam_denoise_clamp_every_n_steps=int(binding_config.get("seam_denoise_clamp_every_n_steps", 1)),
            seam_denoise_clamp_mode=str(binding_config.get("seam_denoise_clamp_mode", "latent")),
            seam_denoise_clamp_hard_threshold=float(binding_config.get("seam_denoise_clamp_hard_threshold", 0.5)),
            conditioning_spec=conditioning_spec,
            override_conditioning=override,
        )["rgb"]

    def _render_full(sample_: Dict, full_override: torch.Tensor, seed: int) -> Image.Image:
        """Render using a full model-visible conditioning tensor override."""
        return _render_one(
            sample=sample_,
            unet=unet,
            control_net=control_net,
            vae=vae,
            scheduler=scheduler,
            cached_text=cached_text,
            device=device,
            weight_dtype=weight_dtype,
            control_dtype=control_dtype,
            vae_dtype=vae_dtype,
            steps=inference_steps,
            seed=seed,
            write_latent_debug=False,
            alpha_output_source=alpha_output_source,
            seam_denoise_clamp=bool(binding_config.get("seam_denoise_clamp", False)),
            seam_denoise_clamp_inner_px=int(binding_config.get("seam_denoise_clamp_inner_px", 0)),
            seam_denoise_clamp_feather_px=int(binding_config.get("seam_denoise_clamp_feather_px", 160)),
            seam_denoise_clamp_feather_profile=str(binding_config.get("seam_denoise_clamp_feather_profile", "smoothstep")),
            seam_denoise_clamp_every_n_steps=int(binding_config.get("seam_denoise_clamp_every_n_steps", 1)),
            seam_denoise_clamp_mode=str(binding_config.get("seam_denoise_clamp_mode", "latent")),
            seam_denoise_clamp_hard_threshold=float(binding_config.get("seam_denoise_clamp_hard_threshold", 0.5)),
            conditioning_spec=conditioning_spec,
            override_full_conditioning=full_override,
        )["rgb"]

    def _cond_override(base_cond_3d: torch.Tensor, mode: str) -> torch.Tensor:
        """Apply a conditioning override mode; input/output are (C, H, W)."""
        return _make_cond_override(
            base_cond_3d.unsqueeze(0), mode, terrain_mask_ch, ablation_seed
        ).squeeze(0)

    def _ns_override(base_cond_3d: torch.Tensor, channels: List[int], g_seed_offset: int) -> torch.Tensor:
        """Scramble selected channels; input/output are (C, H, W)."""
        C, H, W = base_cond_3d.shape
        batched = base_cond_3d.unsqueeze(0)  # (1, C, H, W)
        gen = torch.Generator(device="cpu")
        result = batched.clone()
        flat = batched.reshape(1, C, -1).cpu()
        for i, ch in enumerate(channels):
            gen.manual_seed(ablation_seed + g_seed_offset + i)
            perm = torch.randperm(H * W, generator=gen)
            result[0, ch] = flat[0, ch, perm].view(H, W).to(device=batched.device, dtype=batched.dtype)
        return result.squeeze(0)

    metrics_rows: List[Dict[str, object]] = []
    board_rows: List[Tuple[str, List[Image.Image]]] = []

    for pair in swap_pairs:
        base_sample = dataset[pair.base_dataset_index]
        swap_sample = dataset[pair.swap_dataset_index]
        base_cond = base_sample["conditioning_images"].detach()     # (C, H, W)
        swap_cond = swap_sample["conditioning_images"].detach()     # (C, H, W)
        pair_dir = os.path.join(step_dir, pair.pair_id)
        os.makedirs(pair_dir, exist_ok=True)

        # ── seed panel (base atlas, N seeds) ───────────────────────────────
        seed_rgbs: List[Image.Image] = []
        for sd in seeds_panel:
            rgb = _render(base_sample, None, sd)
            rgb.save(os.path.join(pair_dir, f"seed{sd:06d}_base_rgb.png"))
            seed_rgbs.append(rgb)
        base_rgb = seed_rgbs[0]
        edge_maps = [_smooth_edge_map(img, blur_radius=blur_radius) for img in seed_rgbs]
        edge_var = _edge_map_variance(edge_maps)

        # ── semantic swap ──────────────────────────────────────────────────
        swap_rgb = _render(base_sample, swap_cond, primary_seed)
        swap_rgb.save(os.path.join(pair_dir, "swap_rgb.png"))
        swap_abs, swap_signed = _compute_rgb_diff(base_rgb, swap_rgb)
        _abs_diff_to_image(swap_abs).save(os.path.join(pair_dir, "swap_absdiff.png"))
        _signed_diff_to_image(swap_signed).save(os.path.join(pair_dir, "swap_signeddiff.png"))
        swap_diff_mag = float(swap_abs.mean()) / 255.0
        swap_diff_norm = _normalize_diff(swap_abs, np.asarray(base_rgb.convert("RGB"), dtype=np.float32))

        # ── local-edit localization (only if edit_mask_path is set) ────────
        loc_score: float = float("nan")
        if pair.edit_type == "local" and pair.edit_mask_path and os.path.isfile(pair.edit_mask_path):
            edit_mask = np.asarray(Image.open(pair.edit_mask_path).convert("L"), dtype=np.float32) / 255.0
            loc_score, _ = _compute_localization_score(swap_abs, edit_mask)

        # ── ControlNet ablations ────────────────────────────────────────────
        zero_rgb = _render(base_sample, _cond_override(base_cond, "zero"), primary_seed)
        zero_rgb.save(os.path.join(pair_dir, "zero_cond_rgb.png"))
        zero_abs, _ = _compute_rgb_diff(base_rgb, zero_rgb)
        _abs_diff_to_image(zero_abs).save(os.path.join(pair_dir, "zero_absdiff.png"))
        zero_diff_mag = float(zero_abs.mean()) / 255.0

        shuffled_rgb = _render(base_sample, _cond_override(base_cond, "shuffled"), primary_seed)
        shuffled_rgb.save(os.path.join(pair_dir, "shuffled_cond_rgb.png"))
        shuffled_abs, _ = _compute_rgb_diff(base_rgb, shuffled_rgb)
        _abs_diff_to_image(shuffled_abs).save(os.path.join(pair_dir, "shuffled_absdiff.png"))
        shuffled_diff_mag = float(shuffled_abs.mean()) / 255.0

        maskonly_rgb = _render(base_sample, _cond_override(base_cond, "mask_only"), primary_seed)
        maskonly_rgb.save(os.path.join(pair_dir, "maskonly_rgb.png"))
        maskonly_abs, _ = _compute_rgb_diff(base_rgb, maskonly_rgb)
        _abs_diff_to_image(maskonly_abs).save(os.path.join(pair_dir, "maskonly_absdiff.png"))
        maskonly_diff_mag = float(maskonly_abs.mean()) / 255.0

        # ── seam strip-only perturbation (when seam conditioning exists) ───
        strip_only_diff_mag = float("nan")
        strip_only_edge_localization = float("nan")
        strip_only_interior_drift = float("nan")
        strip_rgb: Optional[Image.Image] = None
        seam_strip = base_sample.get("seam_strip_tensor")
        edge_band_masks = base_sample.get("edge_band_masks")
        seam_strip_width_px = int(float(base_sample.get("seam_strip_width_px", 0.0) or 0.0))
        if isinstance(seam_strip, torch.Tensor):
            full_base_cond = _compose_model_visible_conditioning(base_sample, base_cond, conditioning_spec)
            strip_start = int(base_cond.shape[0])
            strip_end = strip_start + int(seam_strip.shape[0])
            perturbed_full = _spatial_shuffle_channels(full_base_cond, list(range(strip_start, strip_end)), ablation_seed + 700)
            strip_rgb = _render_full(base_sample, perturbed_full, primary_seed)
            strip_rgb.save(os.path.join(pair_dir, "strip_only_perturb_rgb.png"))
            strip_abs, _ = _compute_rgb_diff(base_rgb, strip_rgb)
            _abs_diff_to_image(strip_abs).save(os.path.join(pair_dir, "strip_only_perturb_absdiff.png"))
            strip_only_diff_mag = float(strip_abs.mean()) / 255.0

            if isinstance(edge_band_masks, torch.Tensor):
                edge_mask = edge_band_masks.float().sum(dim=0).clamp(0.0, 1.0).cpu().numpy()
            else:
                h, w = base_cond.shape[-2:]
                band = max(1, min(seam_strip_width_px if seam_strip_width_px > 0 else 32, (min(h, w) - 1) // 2))
                edge_mask = np.zeros((h, w), dtype=np.float32)
                edge_mask[:band, :] = 1.0
                edge_mask[h - band :, :] = 1.0
                edge_mask[:, :band] = 1.0
                edge_mask[:, w - band :] = 1.0

            interior_mask = 1.0 - np.clip(edge_mask, 0.0, 1.0)
            strip_only_interior_drift = _masked_mean_abs_diff(base_rgb, strip_rgb, interior_mask)
            edge_diff = _masked_mean_abs_diff(base_rgb, strip_rgb, edge_mask)
            total_diff = _masked_mean_abs_diff(base_rgb, strip_rgb, np.ones_like(edge_mask, dtype=np.float32))
            strip_only_edge_localization = edge_diff / max(total_diff, 1e-8)

        # ── null-space tests ────────────────────────────────────────────────
        ns_metrics: Dict[str, float] = {}
        for g_idx, (gname, ch_indices) in enumerate(zip(null_space_group_names, null_space_groups)):
            ns_cond = _ns_override(base_cond, ch_indices, g_seed_offset=g_idx * ns_ablation_seed_offset)
            ns_rgb = _render(base_sample, ns_cond, primary_seed)
            ns_rgb.save(os.path.join(pair_dir, f"ns_{gname}_rgb.png"))
            ns_abs, _ = _compute_rgb_diff(base_rgb, ns_rgb)
            _abs_diff_to_image(ns_abs).save(os.path.join(pair_dir, f"ns_{gname}_absdiff.png"))
            ns_metrics[f"ns_{gname}_diff_mag"] = float(ns_abs.mean()) / 255.0

        # ── save expected-effects annotation ────────────────────────────────
        _write_json(
            os.path.join(pair_dir, "expected_effects.json"),
            {
                "pair_id": pair.pair_id,
                "edit_type": pair.edit_type,
                "base_image": pair.base_image,
                "swap_image": pair.swap_image,
                "primary_expected_effect": pair.primary_expected_effect,
                "allowed_effects": pair.allowed_effects,
                "disallowed_effects": pair.disallowed_effects,
            },
        )

        # ── build contact-sheet row ─────────────────────────────────────────
        board_images: List[Image.Image] = [
            base_rgb, swap_rgb,
            _abs_diff_to_image(swap_abs), _signed_diff_to_image(swap_signed),
            zero_rgb, shuffled_rgb, maskonly_rgb,
        ]
        if strip_rgb is not None:
            board_images.append(strip_rgb)
        for gname in null_space_group_names:
            p = os.path.join(pair_dir, f"ns_{gname}_rgb.png")
            if os.path.isfile(p):
                board_images.append(Image.open(p).convert("RGB"))
        board_rows.append((f"{pair.pair_id}|{pair.base_image}→{pair.swap_image}", board_images))

        # ── assemble metrics row ────────────────────────────────────────────
        row: Dict[str, object] = {
            "pair_id": pair.pair_id,
            "step_label": step_label,
            "base_image": pair.base_image,
            "swap_image": pair.swap_image,
            "edit_type": pair.edit_type,
            "swap_diff_mag": swap_diff_mag,
            "swap_diff_norm": swap_diff_norm,
            "zero_diff_mag": zero_diff_mag,
            "shuffled_diff_mag": shuffled_diff_mag,
            "maskonly_diff_mag": maskonly_diff_mag,
            "full_vs_maskonly_gap": swap_diff_mag - maskonly_diff_mag,
            # controlnet_sensitivity: how much base vs zero diverges (>0 means ControlNet active)
            "controlnet_sensitivity": zero_diff_mag,
            # semantic_richness: how much maskonly differs from full atlas
            # high value means full atlas ≈ mask_only (richness unused); low means richness matters
            "maskonly_vs_full_ratio": maskonly_diff_mag / (swap_diff_mag + 1e-8),
            "strip_only_diff_mag": strip_only_diff_mag,
            "strip_only_edge_localization": strip_only_edge_localization,
            "strip_only_interior_drift": strip_only_interior_drift,
            "seed_edge_variance": edge_var,
            "localization_score": loc_score,
        }
        row.update(ns_metrics)
        metrics_rows.append(row)

    # ── write aggregate metrics CSV ──────────────────────────────────────────
    if metrics_rows:
        base_fields = [
            "pair_id", "step_label", "base_image", "swap_image", "edit_type",
            "swap_diff_mag", "swap_diff_norm", "zero_diff_mag", "shuffled_diff_mag",
            "maskonly_diff_mag", "full_vs_maskonly_gap", "controlnet_sensitivity", "maskonly_vs_full_ratio",
            "strip_only_diff_mag", "strip_only_edge_localization", "strip_only_interior_drift",
            "seed_edge_variance", "localization_score",
        ]
        ns_fields = sorted(k for k in metrics_rows[0] if k.startswith("ns_"))
        _write_csv(
            os.path.join(step_dir, "binding_metrics.csv"),
            metrics_rows,
            base_fields + ns_fields,
        )

    # ── build binding board ──────────────────────────────────────────────────
    if board_rows:
        headers = ["base_rgb", "swap_rgb", "abs_diff", "signed_diff",
                   "zero_cond", "shuffled", "mask_only"]
        if any(os.path.isfile(os.path.join(step_dir, pair.pair_id, "strip_only_perturb_rgb.png")) for pair in swap_pairs):
            headers.append("strip_only_perturb")
        headers += [f"ns_{g}" for g in null_space_group_names]
        _build_contact_sheet(
            board_rows,
            headers,
            os.path.join(step_dir, f"binding_board_{step_label}.png"),
            tile_min_size=256,
        )

    # ── aggregate summary ────────────────────────────────────────────────────
    def _mean(key: str) -> float:
        vals = [float(r[key]) for r in metrics_rows
                if not (isinstance(r[key], float) and r[key] != r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    summary: Dict[str, object] = {
        "step_label": step_label,
        "n_pairs": len(metrics_rows),
        "mean_swap_diff_mag": _mean("swap_diff_mag"),
        "mean_swap_diff_norm": _mean("swap_diff_norm"),
        "mean_zero_diff_mag": _mean("zero_diff_mag"),
        "mean_shuffled_diff_mag": _mean("shuffled_diff_mag"),
        "mean_maskonly_diff_mag": _mean("maskonly_diff_mag"),
        "mean_controlnet_sensitivity": _mean("controlnet_sensitivity"),
        "mean_strip_only_diff_mag": _mean("strip_only_diff_mag"),
        "mean_strip_only_edge_localization": _mean("strip_only_edge_localization"),
        "mean_strip_only_interior_drift": _mean("strip_only_interior_drift"),
        "mean_seed_edge_variance": _mean("seed_edge_variance"),
    }
    for gname in null_space_group_names:
        k = f"ns_{gname}_diff_mag"
        summary[f"mean_{k}"] = _mean(k)

    _write_json(os.path.join(step_dir, "binding_step_summary.json"), summary)
    return summary


