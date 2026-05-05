"""Regression test for P2 fix: style patch grid must extend into the
hard_band (0-32px) and near_band (32-64px) regions nearest the seam.

Prior behavior built ``style_support_mask`` as
``overlap_band_mask + soft_field_mask + interior_mask`` which excluded the
two innermost distance bands.  The patch grid loop in
``_compute_style_ratio_losses`` skips patches whose support mean is
``<= 0.05``, so excluding hard_band/near_band caused the visualized grid
to be sparse / absent in the area closest to the seam — the exact place
where it should be densest.
"""
from collections import Counter
from typing import Any

import torch

import sdxl_train_terrain_semantic_control_net as train_mod


def _stub_prototype_and_features(monkeypatch: Any) -> None:
    def fake_load_or_build_style_prototypes(**kwargs):
        return {
            "prototypes": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "image_paths": ["x.png"],
            "image_source_id": str(kwargs["style_id"]),
            "cache_status": "test",
            "cache_path": "",
            "prototype_norm_stats": {},
        }

    def fake_compute_handcrafted_style_features(pred_patch_rgb01, **kwargs):
        m = pred_patch_rgb01.mean(dim=(1, 2, 3), keepdim=False)
        return torch.stack((m, 1.0 - m), dim=1)

    monkeypatch.setattr(train_mod, "_load_or_build_style_prototypes", fake_load_or_build_style_prototypes)
    monkeypatch.setattr(
        train_mod, "_compute_handcrafted_style_features", fake_compute_handcrafted_style_features
    )


def test_style_support_mask_covers_hard_and_near_band(monkeypatch: Any) -> None:
    _stub_prototype_and_features(monkeypatch)

    height = width = 8
    pred_rgb = torch.zeros((1, 3, height, width), dtype=torch.float32)
    target_rgb = pred_rgb.clone()

    # Construct seam_maps where hard_band covers the left two columns,
    # near_band the next two, overlap the next two, soft_field zero,
    # interior zero.  Without the P2 fix, no patch would cover columns 0-3.
    hard_band = torch.zeros((1, 1, height, width), dtype=torch.float32)
    hard_band[..., :2] = 1.0
    near_band = torch.zeros((1, 1, height, width), dtype=torch.float32)
    near_band[..., 2:4] = 1.0
    overlap_band = torch.zeros((1, 1, height, width), dtype=torch.float32)
    overlap_band[..., 4:6] = 1.0

    seam_maps = {
        "near_band_mask_per_edge": torch.zeros((1, 4, height, width), dtype=torch.float32),
        "overlap_band_mask_per_edge": torch.zeros((1, 4, height, width), dtype=torch.float32),
        "hard_band_mask": hard_band,
        "near_band_mask": near_band,
        "overlap_band_mask": overlap_band,
        "soft_field_mask": torch.zeros((1, 1, height, width), dtype=torch.float32),
        "interior_mask": torch.zeros((1, 1, height, width), dtype=torch.float32),
        "valid_style_support_mask": torch.ones((1, 1, height, width), dtype=torch.float32),
        "style_ratio_ramp_mask": torch.ones((1, 1, height, width), dtype=torch.float32),
        "soft_field_q_per_edge": torch.full((1, 4, height, width), 0.25, dtype=torch.float32),
        "soft_field_q_interior": torch.full((1, 1, height, width), 0.5, dtype=torch.float32),
        "soft_field_influence_c": torch.full((1, 1, height, width), 0.5, dtype=torch.float32),
        "source_signed_distance_per_edge": torch.zeros((1, 4, height, width), dtype=torch.float32),
        "edge_defined_flags": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "source_local_x_map": torch.arange(width, dtype=torch.float32).view(1, 1, 1, width).expand(1, 1, height, width),
        "source_local_y_map": torch.arange(height, dtype=torch.float32).view(1, 1, height, 1).expand(1, 1, height, width),
    }

    style_entry = {
        "sample_id": "s",
        "style_family_id": "style_a",
        "reference_image_path": "x.png",
        "variant_image_paths": ["x.png"],
        "edge_style_ids": {"north": "style_a", "south": "style_a", "east": "style_a", "west": "style_a"},
        "interior_style_id": "style_a",
        "original_reference_class": True,
    }
    style_ratio_config = {
        "temperature": 1.0,
        "patch_sizes": [4, 6],
        "bucket_c_edges": [0.0, 0.5, 1.01],
        "bucket_distance_edges_px": [0.0, 1.0, 10.0],
        "overlap_band_end_px": 2.0,
        "soft_field_end_px": 8.0,
        "loss_weights": {
            "style_ratio_kl_weight_max": 0.0,
            "style_ratio_soft_weight": 0.0,
            "style_ratio_overlap_weight_start": 0.0,
            "style_ratio_overlap_weight_end": 0.0,
            "entropy_weight": 0.0,
            "patch_grid_smoothing_weight": 0.0,
        },
    }

    result = train_mod._compute_style_ratio_losses(
        pred_rgb=pred_rgb,
        target_rgb=target_rgb,
        seam_maps=seam_maps,
        style_entry=style_entry,
        style_pool_config={},
        style_ratio_config=style_ratio_config,
        prototype_cache={},
        prototype_cache_stats=Counter(),
        current_step=0,
        step_since_resume=0,
        near_normal_history=[],
    )

    patch_boxes = result["patch_boxes"]
    # At least one patch must cover the hard-band column (x=0) and one the near-band column (x=2).
    hard_covered = any(int(p["x0"]) <= 0 < int(p["x1"]) for p in patch_boxes)
    near_covered = any(int(p["x0"]) <= 2 < int(p["x1"]) for p in patch_boxes)
    assert hard_covered, (
        "P2 regression: style patch grid does not cover hard_band region (x=0); "
        "style_support_mask must include hard_band_mask"
    )
    assert near_covered, (
        "P2 regression: style patch grid does not cover near_band region (x=2); "
        "style_support_mask must include near_band_mask"
    )
