from library.terrain_semantic_seam_geometry import center_crop_chw, center_crop_hw, expanded_hw, pad_chw_spatial
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from PIL import Image, ImageDraw, ImageFilter

from library import sdxl_model_util, sdxl_train_util
from library.device_utils import clean_memory_on_device


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
) -> torch.Tensor:
    """Build model-visible conditioning for eval, including seam channels when present.

    Input/output tensors are channel-first 3D tensors (C, H, W).
    """
    if not isinstance(base_conditioning, torch.Tensor):
        return base_conditioning

    seam_strip = sample.get("seam_strip_tensor")
    edge_defined_flags = sample.get("edge_defined_flags")
    edge_flag_maps = sample.get("edge_flag_maps")
    if seam_strip is None or edge_defined_flags is None or edge_flag_maps is None:
        return base_conditioning

    if not isinstance(seam_strip, torch.Tensor) or not isinstance(edge_defined_flags, torch.Tensor) or not isinstance(edge_flag_maps, torch.Tensor):
        return base_conditioning

    if base_conditioning.shape[-2:] != seam_strip.shape[-2:]:
        return base_conditioning

    expected_full_channels = base_conditioning.shape[0] + seam_strip.shape[0] + edge_flag_maps.shape[0]
    if base_conditioning.shape[0] == expected_full_channels:
        return base_conditioning

    seam_gate = edge_defined_flags.float().view(4, 1, 1).repeat_interleave(4, dim=0)
    seam_visible = seam_strip.float() * seam_gate
    return torch.cat([base_conditioning.float(), seam_visible, edge_flag_maps.float()], dim=0)


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
    seam_strip = sample.get("seam_strip_tensor")
    edge_defined = sample.get("edge_defined_flags")
    if seam_strip is None or edge_defined is None:
        return None
    if not isinstance(seam_strip, torch.Tensor) or not isinstance(edge_defined, torch.Tensor):
        return None

    halo = int(max(1, halo_px))
    exp_h, exp_w = expanded_hw(interior_h, interior_w, halo)
    expected = np.zeros((exp_h, exp_w, 4), dtype=np.float32)

    strip = seam_strip.detach().float().cpu().numpy()
    flags = edge_defined.detach().float().cpu().numpy()

    def to_01(x: np.ndarray) -> np.ndarray:
        rgb = np.clip((x[:3] + 1.0) * 0.5, 0.0, 1.0)
        a = np.clip(x[3:4], 0.0, 1.0)
        return np.concatenate([rgb, a], axis=0)

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

    north_active = north_dist_outside > 0.0
    south_active = south_dist_outside > 0.0
    east_active = east_dist_outside > 0.0
    west_active = west_dist_outside > 0.0
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
    owner_idx = np.argmin(owner_stack, axis=0)

    north_owner = outside_single_side & (owner_idx == 0)
    south_owner = outside_single_side & (owner_idx == 1)
    east_owner = outside_single_side & (owner_idx == 2)
    west_owner = outside_single_side & (owner_idx == 3)

    def _edge_payload(owner_mask: np.ndarray, distance_map: np.ndarray, side: str) -> Dict[str, np.ndarray]:
        halo_all = owner_mask.astype(np.float32)
        halo_inner = (owner_mask & (distance_map <= float(band))).astype(np.float32)
        halo_outer = (owner_mask & (distance_map > float(band))).astype(np.float32)
        ring_1 = (owner_mask & (distance_map > 0.0) & (distance_map <= 1.0)).astype(np.float32)
        ring_4 = (owner_mask & (distance_map > 0.0) & (distance_map <= 4.0)).astype(np.float32)
        ring_8 = (owner_mask & (distance_map > 0.0) & (distance_map <= 8.0)).astype(np.float32)
        ring_16 = (owner_mask & (distance_map > 0.0) & (distance_map <= 16.0)).astype(np.float32)

        interior_outer = z()
        if side == "top":
            interior_outer[halo : min(h, halo + band), :] = 1.0
        elif side == "bottom":
            interior_outer[max(0, h - halo - band) : h - halo, :] = 1.0
        elif side == "right":
            interior_outer[:, max(0, w - halo - band) : w - halo] = 1.0
        else:
            interior_outer[:, halo : min(w, halo + band)] = 1.0

        return {
            "halo_all": halo_all,
            "halo_inner": halo_inner,
            "halo_outer": halo_outer,
            "halo_inner_edge_1px": ring_1,
            "halo_inner_edge_4px": ring_4,
            "halo_inner_8px": ring_8,
            "halo_inner_16px": ring_16,
            "interior_outer": interior_outer,
            "corner_excluded": corner_excluded.astype(np.float32),
        }

    masks["top"] = _edge_payload(north_owner, north_dist_outside, "top")
    masks["bottom"] = _edge_payload(south_owner, south_dist_outside, "bottom")
    masks["right"] = _edge_payload(east_owner, east_dist_outside, "right")
    masks["left"] = _edge_payload(west_owner, west_dist_outside, "left")
    return masks


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
    override_conditioning: Optional[torch.Tensor] = None,
    override_full_conditioning: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    if override_full_conditioning is not None:
        cond = override_full_conditioning.unsqueeze(0).to(device=device, dtype=control_dtype)
    else:
        raw_cond = override_conditioning if override_conditioning is not None else sample["conditioning_images"]
        visible_cond = _compose_model_visible_conditioning(sample, raw_cond)
        cond = visible_cond.unsqueeze(0).to(device=device, dtype=control_dtype)

    interior_h = int(sample["target_sizes_hw"][0].item())
    interior_w = int(sample["target_sizes_hw"][1].item())
    use_expanded = bool(expanded_prediction_enabled and int(expanded_halo_px) > 0)
    halo_px = int(max(0, expanded_halo_px))
    target_h, target_w = (interior_h, interior_w)
    if use_expanded:
        target_h, target_w = expanded_hw(interior_h, interior_w, halo_px)
        cond = pad_chw_spatial(cond.squeeze(0), halo_px=halo_px, mode="constant").unsqueeze(0).to(device=device, dtype=control_dtype)

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
    with torch.no_grad():
        for timestep in scheduler.timesteps:
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
            noisy = scheduler.step(eps, timestep, noisy).prev_sample

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
    }
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
    }
    _write_json(os.path.join(step_dir, "eval_run_config.json"), run_config)

    rows_for_board: List[Tuple[str, List[Image.Image]]] = []
    metrics_rows: List[Dict[str, object]] = []
    resolved_rows: List[Dict[str, object]] = []
    collapse_images: List[np.ndarray] = []

    terrain_mask_index = dataset.channel_names.index("terrain_mask")
    terrain_black_is_terrain = bool(eval_config.get("terrain_mask_black_is_terrain", True))
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

        assert primary_render is not None
        edge_map_var = _edge_map_variance([_smooth_edge_map(img) for img in seed_rgb_list])

        halo_inner_recon_l1 = 0.0
        halo_outer_recon_l1 = 0.0
        halo_inner_edge_1px_rgb_loss = 0.0
        halo_inner_edge_4px_rgb_loss = 0.0
        halo_inner_8px_rgb_loss = 0.0
        halo_inner_16px_rgb_loss = 0.0
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

            expected_rgba_exp = _build_expected_expanded_seam_rgba(
                sample=sample,
                interior_h=interior_h,
                interior_w=interior_w,
                halo_px=halo_px,
            )
            if expected_rgba_exp is not None and expected_rgba_exp.shape[0] == exp_h and expected_rgba_exp.shape[1] == exp_w:
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
                interior_vals: List[float] = []
                align_vals: List[float] = []
                copy_diff_sums: List[float] = []
                copy_diff_counts: List[float] = []
                copy_diff_max_values: List[float] = []
                per_pixel_rgba_diff = np.mean(np.abs(pred_rgba_exp - expected_rgba_exp), axis=2)
                for side_idx, side in enumerate(sides):
                    if side_idx < len(ef) and ef[side_idx] < 0.5:
                        continue
                    m_h = masks[side]["halo_inner"]
                    m_ho = masks[side]["halo_outer"]
                    m_i = _build_expanded_edge_masks(exp_h, exp_w, halo_px=halo_px, band_px=continuation_band)[side]["interior_outer"]
                    m_all = masks[side]["halo_all"]
                    halo_vals.append(_masked_l1(pred_rgba_exp, expected_rgba_exp, m_h))
                    halo_outer_vals.append(_masked_l1(pred_rgba_exp, expected_rgba_exp, m_ho))
                    halo_edge_1_vals.append(_masked_l1(pred_rgba_exp, expected_rgba_exp, masks[side]["halo_inner_edge_1px"]))
                    halo_edge_4_vals.append(_masked_l1(pred_rgba_exp, expected_rgba_exp, masks[side]["halo_inner_edge_4px"]))
                    halo_edge_8_vals.append(_masked_l1(pred_rgba_exp, expected_rgba_exp, masks[side]["halo_inner_8px"]))
                    halo_edge_16_vals.append(_masked_l1(pred_rgba_exp, expected_rgba_exp, masks[side]["halo_inner_16px"]))
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
                    if float(np.sum(m_all)) > 0.0:
                        copy_diff_max_values.append(float(np.max(per_pixel_rgba_diff[m_all > 0.0])))

                if halo_vals:
                    halo_inner_recon_l1 = float(np.mean(halo_vals))
                if halo_outer_vals:
                    halo_outer_recon_l1 = float(np.mean(halo_outer_vals))
                if halo_edge_1_vals:
                    halo_inner_edge_1px_rgb_loss = float(np.mean(halo_edge_1_vals))
                if halo_edge_4_vals:
                    halo_inner_edge_4px_rgb_loss = float(np.mean(halo_edge_4_vals))
                if halo_edge_8_vals:
                    halo_inner_8px_rgb_loss = float(np.mean(halo_edge_8_vals))
                if halo_edge_16_vals:
                    halo_inner_16px_rgb_loss = float(np.mean(halo_edge_16_vals))
                if interior_vals:
                    interior_continuation_l1 = float(np.mean(interior_vals))
                if align_vals:
                    halo_to_interior_alignment = float(np.mean(align_vals))
                if copy_diff_counts and sum(copy_diff_counts) > 0.0:
                    expanded_halo_copy_diff_mean = float(sum(copy_diff_sums) / max(sum(copy_diff_counts), 1e-8))
                if copy_diff_max_values:
                    expanded_halo_copy_diff_max = float(max(copy_diff_max_values))

                try:
                    full_cond = _compose_model_visible_conditioning(sample, sample["conditioning_images"])
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
                "interior_continuation_l1": interior_continuation_l1,
                "halo_to_interior_alignment": halo_to_interior_alignment,
                "halo_effect_strength": halo_effect_strength,
                "expanded_vs_direct_rgb_l1": expanded_vs_direct_rgb_l1,
                "expanded_vs_direct_alpha_l1": expanded_vs_direct_alpha_l1,
                "expanded_halo_copy_diff_mean": expanded_halo_copy_diff_mean,
                "expanded_halo_copy_diff_max": expanded_halo_copy_diff_max,
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
            "interior_continuation_l1",
            "halo_to_interior_alignment",
            "halo_effect_strength",
            "expanded_vs_direct_rgb_l1",
            "expanded_vs_direct_alpha_l1",
            "expanded_halo_copy_diff_mean",
            "expanded_halo_copy_diff_max",
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
        "interior_continuation_l1": float(np.mean([row.get("interior_continuation_l1", 0.0) for row in metrics_rows])),
        "halo_to_interior_alignment": float(np.mean([row.get("halo_to_interior_alignment", 0.0) for row in metrics_rows])),
        "halo_effect_strength": float(np.mean([row.get("halo_effect_strength", 0.0) for row in metrics_rows])),
        "expanded_vs_direct_rgb_l1": float(np.mean([row.get("expanded_vs_direct_rgb_l1", 0.0) for row in metrics_rows])),
        "expanded_vs_direct_alpha_l1": float(np.mean([row.get("expanded_vs_direct_alpha_l1", 0.0) for row in metrics_rows])),
        "expanded_halo_copy_diff_mean": float(np.mean([row.get("expanded_halo_copy_diff_mean", 0.0) for row in metrics_rows])),
        "expanded_halo_copy_diff_max": float(np.mean([row.get("expanded_halo_copy_diff_max", 0.0) for row in metrics_rows])),
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
            full_base_cond = _compose_model_visible_conditioning(base_sample, base_cond)
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


