from typing import Tuple

import torch
import torch.nn.functional as F


def expanded_hw(interior_h: int, interior_w: int, halo_px: int) -> Tuple[int, int]:
    halo = int(max(0, halo_px))
    return int(interior_h + (2 * halo)), int(interior_w + (2 * halo))


def pad_chw_spatial(tensor: torch.Tensor, halo_px: int, mode: str = "constant") -> torch.Tensor:
    halo = int(max(0, halo_px))
    if halo <= 0:
        return tensor
    if tensor.ndim != 3:
        raise ValueError(f"expected CHW tensor, got shape={tuple(tensor.shape)}")
    return F.pad(tensor, (halo, halo, halo, halo), mode=mode)


def center_crop_chw(tensor: torch.Tensor, out_h: int, out_w: int, halo_px: int) -> torch.Tensor:
    halo = int(max(0, halo_px))
    if tensor.ndim != 3:
        raise ValueError(f"expected CHW tensor, got shape={tuple(tensor.shape)}")
    h, w = int(tensor.shape[-2]), int(tensor.shape[-1])
    expected_h, expected_w = expanded_hw(out_h, out_w, halo)
    if (h, w) != (expected_h, expected_w):
        raise ValueError(
            "expanded crop shape mismatch: "
            + f"got={(h, w)} expected={(expected_h, expected_w)} halo={halo}"
        )
    top = halo
    left = halo
    return tensor[:, top : top + int(out_h), left : left + int(out_w)]


def center_crop_hw(tensor: torch.Tensor, out_h: int, out_w: int, halo_px: int) -> torch.Tensor:
    halo = int(max(0, halo_px))
    if tensor.ndim != 2:
        raise ValueError(f"expected HW tensor, got shape={tuple(tensor.shape)}")
    h, w = int(tensor.shape[-2]), int(tensor.shape[-1])
    expected_h, expected_w = expanded_hw(out_h, out_w, halo)
    if (h, w) != (expected_h, expected_w):
        raise ValueError(
            "expanded crop shape mismatch: "
            + f"got={(h, w)} expected={(expected_h, expected_w)} halo={halo}"
        )
    top = halo
    left = halo
    return tensor[top : top + int(out_h), left : left + int(out_w)]
