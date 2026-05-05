import math
import os
from collections import Counter
from typing import Any

import torch

import sdxl_train_terrain_semantic_control_net as train_mod


def test_collect_style_image_paths_collapses_same_source_regenerated_roles() -> None:
    style_entry = {
        "sample_id": "sample",
        "style_family_id": "style_family",
        "reference_image_path": "fallback.png",
        "variant_image_paths": ["fallback.png", "interior.png"],
        "seam_role_image_paths": ["fallback.png"],
        "edge_style_ids": {
            "north": "style_family",
            "south": "style_family",
            "east": "style_east",
            "west": "style_family",
        },
        "interior_style_id": "style_family",
        "selected_interior_prompt_kind": "regenerated_variant",
        "selected_interior_prompt_image_path": "interior.png",
        "selected_seam_image_source_path": "interior.png",
        "selected_seam_image_source_paths_by_edge": {
            "east": "interior.png",
        },
        "selected_seam_image_source_only": False,
    }

    assert train_mod._collect_style_image_paths(style_entry, "style_family") == ["interior.png"]
    assert train_mod._collect_style_image_paths(style_entry, "style_east") == ["interior.png"]


def test_collect_style_image_paths_preserves_distinct_seam_source_when_mismatched() -> None:
    style_entry = {
        "sample_id": "sample",
        "style_family_id": "style_family",
        "reference_image_path": "fallback.png",
        "variant_image_paths": ["fallback.png", "interior.png", "seam.png"],
        "seam_role_image_paths": ["seam.png"],
        "edge_style_ids": {
            "north": "style_family",
            "south": "style_family",
            "east": "style_east",
            "west": "style_family",
        },
        "interior_style_id": "style_family",
        "selected_interior_prompt_kind": "regenerated_variant",
        "selected_interior_prompt_image_path": "interior.png",
        "selected_seam_image_source_path": "seam.png",
        "selected_seam_image_source_paths_by_edge": {
            "east": "seam.png",
        },
        "selected_seam_image_source_only": True,
    }

    assert train_mod._collect_style_image_paths(style_entry, "style_east") == ["seam.png"]


def test_collect_style_image_paths_prefers_regenerated_seam_fallback_over_legacy_reference() -> None:
    style_entry = {
        "sample_id": "LeftOverhang",
        "style_family_id": "LeftOverhang",
        "reference_image_path": "/workspace/terrain-style/images/training_images/LeftOverhang.png",
        "variant_image_paths": [
            "/workspace/terrain-style/images/regenerated_training_images/run/leftoverhang/variant_02_minimal_terrain_05.png",
            "/workspace/terrain-style/images/regenerated_training_images/run/leftoverhang/original_rgba.png",
        ],
        "seam_role_image_paths": [
            "/workspace/terrain-style/images/regenerated_training_images/run/leftoverhang/original_rgba.png",
        ],
        "edge_style_ids": {
            "north": "interior::minimal_terrain_05",
            "south": "interior::guardrail_ceiling_shape_03",
            "east": "seam::leftoverhang",
            "west": "interior::minimal_terrain_05",
        },
        "interior_style_id": "interior::minimal_terrain_05",
        "selected_interior_prompt_kind": "regenerated_variant",
        "selected_interior_prompt_image_path": "/workspace/terrain-style/images/regenerated_training_images/run/leftoverhang/variant_02_minimal_terrain_05.png",
        "selected_seam_image_source_path": "",
        "selected_seam_image_source_paths_by_edge": {},
        "selected_seam_image_source_only": True,
    }

    assert train_mod._collect_style_image_paths(style_entry, "seam::leftoverhang") == [
        "/workspace/terrain-style/images/regenerated_training_images/run/leftoverhang/original_rgba.png",
    ]


def test_style_ratio_allows_seam_only_regenerated_original_within_family(monkeypatch: Any) -> None:
    def fake_load_or_build_style_prototypes(**kwargs):
        style_id = str(kwargs["style_id"])
        if style_id == "interior::variant_01":
            prototype_value = [0.0, 1.0]
            image_paths = ["interior.png"]
        else:
            prototype_value = [1.0, 0.0]
            image_paths = ["original_rgba.png"]
        return {
            "prototypes": torch.tensor([prototype_value], dtype=torch.float32),
            "image_paths": image_paths,
            "image_source_id": style_id,
            "cache_status": "test",
            "cache_path": "",
            "prototype_norm_stats": {},
        }

    def fake_compute_handcrafted_style_features(pred_patch_rgb01, **kwargs):
        patch_mean = pred_patch_rgb01.mean(dim=(1, 2, 3), keepdim=False)
        return torch.stack((patch_mean, 1.0 - patch_mean), dim=1)

    monkeypatch.setattr(train_mod, "_load_or_build_style_prototypes", fake_load_or_build_style_prototypes)
    monkeypatch.setattr(train_mod, "_compute_handcrafted_style_features", fake_compute_handcrafted_style_features)

    height = width = 8
    pred_rgb = torch.zeros((1, 3, height, width), dtype=torch.float32)
    target_rgb = pred_rgb.clone()
    seam_maps = {
        "near_band_mask_per_edge": torch.zeros((1, 4, height, width), dtype=torch.float32),
        "overlap_band_mask_per_edge": torch.zeros((1, 4, height, width), dtype=torch.float32),
        "overlap_band_mask": torch.ones((1, 1, height, width), dtype=torch.float32),
        "soft_field_mask": torch.ones((1, 1, height, width), dtype=torch.float32),
        "interior_mask": torch.ones((1, 1, height, width), dtype=torch.float32),
        "valid_style_support_mask": torch.ones((1, 1, height, width), dtype=torch.float32),
        "style_ratio_ramp_mask": torch.ones((1, 1, height, width), dtype=torch.float32),
        "soft_field_q_per_edge": torch.tensor(
            [[
                [[0.0] * width for _ in range(height)],
                [[0.6] * width for _ in range(height)],
                [[0.0] * width for _ in range(height)],
                [[0.0] * width for _ in range(height)],
            ]],
            dtype=torch.float32,
        ),
        "soft_field_q_interior": torch.full((1, 1, height, width), 0.4, dtype=torch.float32),
        "soft_field_influence_c": torch.full((1, 1, height, width), 0.6, dtype=torch.float32),
        "source_signed_distance_per_edge": torch.zeros((1, 4, height, width), dtype=torch.float32),
        "edge_defined_flags": torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
        "source_local_x_map": torch.arange(width, dtype=torch.float32).view(1, 1, 1, width).expand(1, 1, height, width),
        "source_local_y_map": torch.arange(height, dtype=torch.float32).view(1, 1, height, 1).expand(1, 1, height, width),
    }
    style_entry = {
        "sample_id": "convexleft1",
        "style_family_id": "convexleft1",
        "reference_image_path": "legacy_training_image.png",
        "variant_image_paths": ["interior.png", "original_rgba.png"],
        "seam_role_image_paths": ["original_rgba.png"],
        "edge_style_ids": {
            "north": "interior::variant_01",
            "south": "seam::original_rgba",
            "east": "interior::variant_01",
            "west": "interior::variant_01",
        },
        "interior_style_id": "interior::variant_01",
        "selected_interior_prompt_kind": "regenerated_variant",
        "selected_interior_prompt_image_path": "interior.png",
        "selected_seam_image_source_path": "original_rgba.png",
        "selected_seam_image_source_paths_by_edge": {
            "south": "original_rgba.png",
        },
        "selected_seam_image_source_only": True,
    }
    style_ratio_config = {
        "temperature": 1.0,
        "patch_sizes": [4],
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

    assert result["enabled"] == 1.0


def test_style_ratio_merges_distinct_roles_with_identical_source_paths(monkeypatch: Any) -> None:
    def fake_load_or_build_style_prototypes(**kwargs):
        style_id = str(kwargs["style_id"])
        if style_id == "interior::variant_01":
            return {
                "prototypes": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
                "image_paths": ["interior.png"],
                "image_source_id": style_id,
                "cache_status": "test",
                "cache_path": "",
                "prototype_norm_stats": {},
            }
        return {
            "prototypes": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "image_paths": ["shared_seam.png"],
            "image_source_id": style_id,
            "cache_status": "test",
            "cache_path": "",
            "prototype_norm_stats": {},
        }

    def fake_compute_handcrafted_style_features(pred_patch_rgb01, **kwargs):
        patch_mean = pred_patch_rgb01.mean(dim=(1, 2, 3), keepdim=False)
        return torch.stack((patch_mean, 1.0 - patch_mean), dim=1)

    monkeypatch.setattr(train_mod, "_load_or_build_style_prototypes", fake_load_or_build_style_prototypes)
    monkeypatch.setattr(train_mod, "_compute_handcrafted_style_features", fake_compute_handcrafted_style_features)

    height = width = 8
    pred_rgb = torch.zeros((1, 3, height, width), dtype=torch.float32)
    target_rgb = pred_rgb.clone()
    seam_maps = {
        "near_band_mask_per_edge": torch.zeros((1, 4, height, width), dtype=torch.float32),
        "overlap_band_mask_per_edge": torch.zeros((1, 4, height, width), dtype=torch.float32),
        "overlap_band_mask": torch.ones((1, 1, height, width), dtype=torch.float32),
        "soft_field_mask": torch.ones((1, 1, height, width), dtype=torch.float32),
        "interior_mask": torch.ones((1, 1, height, width), dtype=torch.float32),
        "valid_style_support_mask": torch.ones((1, 1, height, width), dtype=torch.float32),
        "style_ratio_ramp_mask": torch.ones((1, 1, height, width), dtype=torch.float32),
        "soft_field_q_per_edge": torch.tensor(
            [[
                [[0.0] * width for _ in range(height)],
                [[0.5] * width for _ in range(height)],
                [[0.5] * width for _ in range(height)],
                [[0.0] * width for _ in range(height)],
            ]],
            dtype=torch.float32,
        ),
        "soft_field_q_interior": torch.zeros((1, 1, height, width), dtype=torch.float32),
        "soft_field_influence_c": torch.ones((1, 1, height, width), dtype=torch.float32),
        "source_signed_distance_per_edge": torch.zeros((1, 4, height, width), dtype=torch.float32),
        "edge_defined_flags": torch.tensor([[0.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
        "source_local_x_map": torch.arange(width, dtype=torch.float32).view(1, 1, 1, width).expand(1, 1, height, width),
        "source_local_y_map": torch.arange(height, dtype=torch.float32).view(1, 1, height, 1).expand(1, 1, height, width),
    }
    style_entry = {
        "sample_id": "sample",
        "style_family_id": "sample",
        "reference_image_path": "legacy_reference.png",
        "variant_image_paths": ["interior.png", "shared_seam.png"],
        "seam_role_image_paths": ["shared_seam.png"],
        "edge_style_ids": {
            "north": "interior::variant_01",
            "south": "seam::south_alias",
            "east": "seam::east_alias",
            "west": "interior::variant_01",
        },
        "interior_style_id": "interior::variant_01",
        "selected_interior_prompt_kind": "regenerated_variant",
        "selected_interior_prompt_image_path": "interior.png",
        "selected_seam_image_source_path": "shared_seam.png",
        "selected_seam_image_source_paths_by_edge": {
            "south": "shared_seam.png",
            "east": "shared_seam.png",
        },
        "selected_seam_image_source_only": True,
    }
    style_ratio_config = {
        "temperature": 1.0,
        "patch_sizes": [4],
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

    assert result["enabled"] == 1.0
    assert result["active_style_count"] == 2.0
    assert result["active_style_ids"] == ["interior::variant_01", "seam::south_alias"]


def test_style_cache_file_path_caps_overlong_basenames() -> None:
    long_source_id = (
        "north_variant_02_natural_eroded_rock_02_"
        "south_variant_02_natural_eroded_rock_02_"
        "east_variant_02_natural_eroded_rock_02_"
        "west_variant_02_natural_eroded_rock_02"
    )
    cache_path = train_mod._style_cache_file_path(
        "/tmp/style-cache",
        style_id="interior::natural_eroded_rock_02",
        image_source_id=long_source_id,
        image_path_signature="c7c65628db42a4da",
        patch_size=32,
        prototype_count=3,
        clustering_seed=1337,
        clustering_mode="kmeans",
        prompt_embedding_mode="side_channel",
        lab_enabled=False,
    )
    changed_tail_cache_path = train_mod._style_cache_file_path(
        "/tmp/style-cache",
        style_id="interior::natural_eroded_rock_02",
        image_source_id=long_source_id + "_tailchange",
        image_path_signature="c7c65628db42a4da",
        patch_size=32,
        prototype_count=3,
        clustering_seed=1337,
        clustering_mode="kmeans",
        prompt_embedding_mode="side_channel",
        lab_enabled=False,
    )

    basename = os.path.basename(cache_path)
    assert len(basename) <= 240
    assert basename.endswith("_v1.pt")
    assert os.path.basename(cache_path) != os.path.basename(changed_tail_cache_path)


def test_resolve_seam_visual_debug_inputs_prefers_aligned_style_ratio_snapshot() -> None:
    base_pred = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    base_target = torch.ones((1, 3, 4, 4), dtype=torch.float32)
    base_mask = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    base_maps = {"margin_inner": torch.zeros((1, 1, 4, 4), dtype=torch.float32)}
    base_crop = {"pixel_x0": 0, "pixel_x1": 4, "pixel_y0": 0, "pixel_y1": 4}

    aligned_pred = torch.full((1, 3, 2, 2), 0.25, dtype=torch.float32)
    aligned_target = torch.full((1, 3, 2, 2), 0.75, dtype=torch.float32)
    aligned_mask = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    aligned_maps = {"margin_inner": torch.ones((1, 1, 2, 2), dtype=torch.float32)}
    aligned_crop = {"pixel_x0": 5, "pixel_x1": 7, "pixel_y0": 9, "pixel_y1": 11}

    resolved = train_mod._resolve_seam_visual_debug_inputs(
        base_pred,
        base_target,
        base_mask,
        base_maps,
        base_crop,
        {
            "crop_box": aligned_crop,
            "debug_pred_rgb": aligned_pred,
            "debug_target_rgb": aligned_target,
            "debug_supervision_mask": aligned_mask,
            "debug_seam_maps": aligned_maps,
        },
    )

    assert resolved[0] is aligned_pred
    assert resolved[1] is aligned_target
    assert resolved[2] is aligned_mask
    assert resolved[3] is aligned_maps
    assert resolved[4] == aligned_crop


def test_resolve_seam_visual_debug_inputs_ignores_incomplete_style_ratio_snapshot() -> None:
    base_pred = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    base_target = torch.ones((1, 3, 4, 4), dtype=torch.float32)
    base_mask = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    base_maps = {"margin_inner": torch.zeros((1, 1, 4, 4), dtype=torch.float32)}
    base_crop = {"pixel_x0": 0, "pixel_x1": 4, "pixel_y0": 0, "pixel_y1": 4}

    resolved = train_mod._resolve_seam_visual_debug_inputs(
        base_pred,
        base_target,
        base_mask,
        base_maps,
        base_crop,
        {
            "crop_box": {"pixel_x0": 5, "pixel_x1": 7, "pixel_y0": 9, "pixel_y1": 11},
            "debug_pred_rgb": torch.full((1, 3, 2, 2), 0.25, dtype=torch.float32),
        },
    )

    assert resolved[0] is base_pred
    assert resolved[1] is base_target
    assert resolved[2] is base_mask
    assert resolved[3] is base_maps
    assert resolved[4] == base_crop


def test_style_ratio_corner_debug_maps_average_overlapping_patches(monkeypatch: Any) -> None:
    def fake_load_or_build_style_prototypes(**kwargs):
        style_id = str(kwargs["style_id"])
        prototype_value = [0.0, 1.0] if style_id == "style_a" else [1.0, 0.0]
        return {
            "prototypes": torch.tensor([prototype_value], dtype=torch.float32),
            "image_paths": [f"{style_id}.png"],
            "image_source_id": style_id,
            "cache_status": "test",
            "cache_path": "",
            "prototype_norm_stats": {},
        }

    def fake_compute_handcrafted_style_features(pred_patch_rgb01, **kwargs):
        patch_mean = pred_patch_rgb01.mean(dim=(1, 2, 3), keepdim=False)
        return torch.stack((patch_mean, 1.0 - patch_mean), dim=1)

    monkeypatch.setattr(train_mod, "_load_or_build_style_prototypes", fake_load_or_build_style_prototypes)
    monkeypatch.setattr(train_mod, "_compute_handcrafted_style_features", fake_compute_handcrafted_style_features)

    height = 8
    width = 8
    x01 = torch.linspace(0.0, 1.0, width, dtype=torch.float32).view(1, 1, 1, width).expand(1, 1, height, width)
    pred_rgb01 = x01.expand(1, 3, height, width).contiguous()
    pred_rgb = (pred_rgb01 * 2.0) - 1.0
    target_rgb = pred_rgb.clone()

    seam_maps = {
        "near_band_mask_per_edge": torch.zeros((1, 4, height, width), dtype=torch.float32),
        "overlap_band_mask_per_edge": torch.zeros((1, 4, height, width), dtype=torch.float32),
        "overlap_band_mask": torch.ones((1, 1, height, width), dtype=torch.float32),
        "soft_field_mask": torch.zeros((1, 1, height, width), dtype=torch.float32),
        "interior_mask": torch.zeros((1, 1, height, width), dtype=torch.float32),
        "valid_style_support_mask": torch.ones((1, 1, height, width), dtype=torch.float32),
        "style_ratio_ramp_mask": torch.ones((1, 1, height, width), dtype=torch.float32),
        "soft_field_q_per_edge": torch.tensor(
            [
                [
                    [[0.4] * width for _ in range(height)],
                    [[0.0] * width for _ in range(height)],
                    [[0.4] * width for _ in range(height)],
                    [[0.0] * width for _ in range(height)],
                ]
            ],
            dtype=torch.float32,
        ),
        "soft_field_q_interior": torch.full((1, 1, height, width), 0.2, dtype=torch.float32),
        "soft_field_influence_c": torch.full((1, 1, height, width), 0.8, dtype=torch.float32),
        "source_signed_distance_per_edge": torch.zeros((1, 4, height, width), dtype=torch.float32),
        "edge_defined_flags": torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
        "source_local_x_map": torch.arange(width, dtype=torch.float32).view(1, 1, 1, width).expand(1, 1, height, width),
        "source_local_y_map": torch.arange(height, dtype=torch.float32).view(1, 1, height, 1).expand(1, 1, height, width),
    }

    style_entry = {
        "sample_id": "sample",
        "style_family_id": "style_a",
        "reference_image_path": "style_a.png",
        "variant_image_paths": ["style_a.png", "style_b.png"],
        "edge_style_ids": {
            "north": "style_a",
            "south": "style_a",
            "east": "style_b",
            "west": "style_a",
        },
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
            "style_ratio_kl_weight_max": 0.1,
            "style_ratio_soft_weight": 1.0,
            "style_ratio_overlap_weight_start": 0.0,
            "style_ratio_overlap_weight_end": 0.5,
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
    assert len(patch_boxes) >= 4

    pixel_y = 2
    pixel_x = 5
    covering_patches = [
        patch
        for patch in patch_boxes
        if int(patch["y0"]) <= pixel_y < int(patch["y1"]) and int(patch["x0"]) <= pixel_x < int(patch["x1"])
    ]
    assert len(covering_patches) > 1

    def patch_probability(patch):
        patch_rgb01 = pred_rgb01[:, :, int(patch["y0"]):int(patch["y1"]), int(patch["x0"]):int(patch["x1"])]
        feature_value = float(patch_rgb01.mean().item())
        feature_vector = torch.tensor([feature_value, 1.0 - feature_value], dtype=torch.float32)
        distance_a = float(torch.linalg.vector_norm(feature_vector - torch.tensor([0.0, 1.0], dtype=torch.float32)).item())
        distance_b = float(torch.linalg.vector_norm(feature_vector - torch.tensor([1.0, 0.0], dtype=torch.float32)).item())
        logits = torch.tensor([-distance_a, -distance_b], dtype=torch.float32)
        return torch.softmax(logits, dim=0)

    probabilities = [patch_probability(patch) for patch in covering_patches]
    averaged_probability = torch.stack(probabilities, dim=0).mean(dim=0)
    averaged_probability = averaged_probability / averaged_probability.sum().clamp_min(1e-6)
    expected_q = torch.tensor([0.6, 0.4], dtype=torch.float32)
    expected_kl = float(
        torch.sum(expected_q * (torch.log(expected_q.clamp_min(1e-6)) - torch.log(averaged_probability.clamp_min(1e-6)))).item()
    )
    last_patch_probability = probabilities[-1]
    last_patch_kl = float(
        torch.sum(expected_q * (torch.log(expected_q.clamp_min(1e-6)) - torch.log(last_patch_probability.clamp_min(1e-6)))).item()
    )

    p_class_maps = result["p_class_maps"][0, :, pixel_y, pixel_x]
    kl_proxy_value = float(result["kl_proxy_map"][0, 0, pixel_y, pixel_x].item())

    assert math.isclose(float(p_class_maps[0].item()), float(averaged_probability[0].item()), rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(float(p_class_maps[1].item()), float(averaged_probability[1].item()), rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(kl_proxy_value, expected_kl, rel_tol=1e-5, abs_tol=1e-5)
    assert not math.isclose(expected_kl, last_patch_kl, rel_tol=1e-5, abs_tol=1e-5)