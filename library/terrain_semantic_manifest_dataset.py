import csv
import hashlib
import json
import math
import os
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

        self._manifest_audit = {
            "total_rows": 0,
            "usable_rows": 0,
            "skipped_rejected_or_weight": 0,
            "skipped_bad_path": 0,
            "skipped_out_of_bounds": 0,
            "skipped_low_trusted_area": 0,
            "path_remapped": 0,
            "native_alpha_rows": 0,
            "synthesized_opaque_rows": 0,
        }
        self._records = self._load_manifest()
        self._sampling_weights = torch.tensor([record["sampling_weight"] for record in self._records], dtype=torch.float32)
        self._cached_latents: List[Optional[torch.Tensor]] = [None] * len(self._records)
        self._cached_latent_paths: List[Optional[str]] = [self._latent_cache_file(index) for index in range(len(self._records))]
        self._last_latent_cache_report: Dict[str, object] = {
            "total": len(self._records),
            "in_memory_hits": 0,
            "disk_hits": 0,
            "encoded": 0,
            "disk_misses": len(self._records),
            "cache_dir": self.latent_cache_dir,
        }

        if self.latent_cache_dir:
            os.makedirs(self.latent_cache_dir, exist_ok=True)

    def _load_manifest(self) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
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
                except (OSError, ValueError):
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
                    "crop_size_class": row.get("crop_size_class") or "",
                    "generation_strategy": row.get("generation_strategy") or "",
                    "sampling_weight": sampling_weight,
                    "trusted_ratio": trusted_ratio,
                }
                records.append(record)

        self._manifest_audit["usable_rows"] = len(records)
        self._manifest_audit["native_alpha_rows"] = sum(1 for record in records if record["has_native_alpha"])
        self._manifest_audit["synthesized_opaque_rows"] = len(records) - self._manifest_audit["native_alpha_rows"]
        if not records:
            raise ValueError(f"no usable samples found in manifest: {self.manifest_path}")
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
    def manifest_audit(self) -> Dict[str, int]:
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

    def _load_cached_latent(self, index: int) -> Optional[torch.Tensor]:
        cache_path = self._cached_latent_paths[index]
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

    def build_debug_example(self, index: int) -> Dict[str, object]:
        sample = self[index]
        return {
            "image_name": sample["image_name"],
            "channel_names": self.channel_names,
            "crop_box": sample["crop_box"],
            "trusted_box": sample["trusted_box"],
            "trusted_ratio": sample["trusted_ratio"],
            "images": sample["images"],
            "alpha_target": sample["alpha_target"],
            "conditioning_images": sample["conditioning_images"],
            "trusted_mask": sample["trusted_mask"],
        }

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self._records[index]
        crop_x, crop_y, crop_w, crop_h = record["crop_box"]

        if self._cached_latents[index] is None:
            self._cached_latents[index] = self._load_cached_latent(index)

        images, alpha_target = self._load_resized_image_with_alpha(index)
        conditioning_images = self._load_semantic_tensor(index)
        trusted_mask = self._build_trusted_mask(index)

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
            "special_structure_tags": record["special_structure_tags"],
            "crop_size_class": record["crop_size_class"],
            "generation_strategy": record["generation_strategy"],
        }
        return example
