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
from PIL import Image, ImageDraw

from library import sdxl_model_util, sdxl_train_util


@dataclass
class EvalSample:
    eval_id: str
    category: str
    sample_key: str
    dataset_index: int
    image_name: str
    crop_box: Tuple[int, int, int, int]
    generation_strategy: str


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


def _histogram_counts(values: torch.Tensor, bin_edges: List[float]) -> List[int]:
    flat = values.detach().float().flatten().cpu().numpy()
    counts, _ = np.histogram(flat, bins=np.asarray(bin_edges, dtype=np.float32))
    return [int(v) for v in counts.tolist()]


def _expand_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    return F.max_pool2d(mask, kernel_size=(radius * 2) + 1, stride=1, padding=radius)


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
) -> Dict[str, object]:
    cond = sample["conditioning_images"].unsqueeze(0).to(device=device, dtype=control_dtype)

    te1, te2, pool2 = cached_text
    text_embedding = torch.cat([te1, te2], dim=2)

    size_batch = {
        "original_sizes_hw": sample["original_sizes_hw"].unsqueeze(0).to(device),
        "crop_top_lefts": sample["crop_top_lefts"].unsqueeze(0).to(device),
        "target_sizes_hw": sample["target_sizes_hw"].unsqueeze(0).to(device),
    }
    size_embeddings = sdxl_train_util.get_size_embeddings(
        size_batch["original_sizes_hw"],
        size_batch["crop_top_lefts"],
        size_batch["target_sizes_hw"],
        device,
    ).to(weight_dtype)
    vector_embedding = torch.cat([pool2, size_embeddings], dim=1)

    latent_h = int(sample["target_sizes_hw"][0].item()) // 8
    latent_w = int(sample["target_sizes_hw"][1].item()) // 8
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
                alpha_target_size=tuple(sample["target_sizes_hw"].tolist()),
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
        alpha_logits = selected_logits.squeeze(0).squeeze(0).detach().float()
        alpha_probs = torch.sigmoid(alpha_logits)

    rgb = _tensor_to_image(decoded).convert("RGB")
    pred_alpha_img = _mask_to_image(alpha_probs)
    rgba = rgb.copy()
    rgba.putalpha(pred_alpha_img)

    output = {
        "rgb": rgb,
        "pred_alpha_logits": alpha_logits,
        "pred_alpha_prob": alpha_probs,
        "pred_alpha_img": pred_alpha_img,
        "rgba": rgba,
    }
    if write_latent_debug:
        output["pred_x0_latent"] = pred_x0.detach().cpu()
    return output


def _write_json(path: str, payload: Dict[str, object]) -> None:
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
    }
    _write_json(os.path.join(step_dir, "eval_run_config.json"), run_config)

    rows_for_board: List[Tuple[str, List[Image.Image]]] = []
    metrics_rows: List[Dict[str, object]] = []
    resolved_rows: List[Dict[str, object]] = []
    collapse_images: List[np.ndarray] = []

    terrain_mask_index = dataset.channel_names.index("terrain_mask")
    full_scene_for_panel: Optional[Tuple[EvalSample, Dict[str, object], Dict[str, object]]] = None

    for sample_info in resolved_samples:
        sample = dataset[sample_info.dataset_index]
        sem_hash = hashlib.sha256(sample["conditioning_images"].detach().cpu().numpy().tobytes()).hexdigest()

        target_alpha = sample["alpha_target"]
        if target_alpha is None:
            target_alpha = sample["conditioning_images"][terrain_mask_index].detach().float().clamp(0.0, 1.0)
        terrain_prior = sample["conditioning_images"][terrain_mask_index].detach().float().clamp(0.0, 1.0)

        primary_render = None
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
            )
            rgb_path = os.path.join(step_dir, f"{sample_info.eval_id}_seed{seed:06d}_rgb.png")
            pred_alpha_path = os.path.join(step_dir, f"{sample_info.eval_id}_seed{seed:06d}_pred_alpha.png")
            rgba_path = os.path.join(step_dir, f"{sample_info.eval_id}_seed{seed:06d}_rgba.png")
            render["rgb"].save(rgb_path)
            render["pred_alpha_img"].save(pred_alpha_path)
            render["rgba"].save(rgba_path)

            if seed == primary_seed:
                primary_render = render
                collapse_images.append(np.asarray(render["rgb"].convert("RGB"), dtype=np.uint8))
                _mask_to_image(target_alpha).save(os.path.join(step_dir, f"{sample_info.eval_id}_target_alpha.png"))
                _mask_to_image(terrain_prior).save(os.path.join(step_dir, f"{sample_info.eval_id}_terrain_prior.png"))
                if "pred_x0_latent" in render:
                    debug_dir = os.path.join(step_dir, "debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    torch.save(render["pred_x0_latent"], os.path.join(debug_dir, f"{sample_info.eval_id}_seed{seed:06d}_latent.pt"))

        assert primary_render is not None

        p = primary_render["pred_alpha_prob"].detach().float().cpu()
        p_logits = primary_render["pred_alpha_logits"].detach().float().cpu()
        t = terrain_prior.detach().float().cpu()
        t_alpha = target_alpha.detach().float().cpu()
        threshold = float(eval_config.get("binary_threshold", 0.5))
        b = (p >= threshold).float()
        tbin = (t >= threshold).float()
        t_alpha_bin = (t_alpha >= threshold).float()
        inter = float((b * tbin).sum().item())
        union = float((b + tbin - b * tbin).sum().item())
        alpha_iou_terrain = inter / max(union, 1e-6)
        inter_target = float((b * t_alpha_bin).sum().item())
        union_target = float((b + t_alpha_bin - b * t_alpha_bin).sum().item())
        alpha_iou_target = inter_target / max(union_target, 1e-6)

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
        )
        full_scene_for_panel = (first, sample, render)

    fs_info, fs_sample, fs_render = full_scene_for_panel
    fs_sem = _float_to_grayscale_image(fs_sample["conditioning_images"][terrain_mask_index]).convert("RGB")
    fs_target = fs_sample["alpha_target"] if fs_sample["alpha_target"] is not None else fs_sample["conditioning_images"][terrain_mask_index]
    fs_prior = fs_sample["conditioning_images"][terrain_mask_index].detach().float().clamp(0.0, 1.0)
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
        "alpha_iou_target": float(np.mean([row["alpha_iou_target"] for row in metrics_rows])),
        "alpha_iou_target_masked": float(np.mean([row["alpha_iou_target_masked"] for row in metrics_rows])),
        "alpha_bce": float(np.mean([row["alpha_bce"] for row in metrics_rows])),
        "alpha_corr": float(np.mean([row["alpha_corr"] for row in metrics_rows])),
        "alpha_occ": float(np.mean([row["alpha_occ"] for row in metrics_rows])),
        "alpha_speckle": float(np.mean([row["alpha_speckle"] for row in metrics_rows])),
        "pred_near0_01": float(np.mean([row["pred_near0_01"] for row in metrics_rows])),
        "pred_near1_99": float(np.mean([row["pred_near1_99"] for row in metrics_rows])),
        "max_pairwise_mse": max_pairwise_mse,
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
