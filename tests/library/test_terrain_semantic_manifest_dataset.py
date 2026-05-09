import random
from pathlib import Path

import pytest
import torch
from PIL import Image

from library.terrain_semantic_manifest_dataset import (
    EDGE_INDEX_TO_NAME,
    TerrainSemanticManifestDataset,
    _merge_regenerated_prompt_variants,
    build_seam_region_maps,
)


def _write_test_rgba(path: Path) -> None:
    Image.new("RGBA", (4, 4), (32, 64, 96, 255)).save(path)


def _build_dataset_stub(candidate_paths):
    dataset = TerrainSemanticManifestDataset.__new__(TerrainSemanticManifestDataset)
    dataset.rotation_seed = 1234
    dataset._same_family_regenerated_paths = lambda style_pool_entry: list(candidate_paths)
    return dataset


def _build_rotation_dataset_stub(interior_path: str, seam_variant_path: str):
    dataset = TerrainSemanticManifestDataset.__new__(TerrainSemanticManifestDataset)
    dataset.rotation_seed = 1234
    dataset.rotation_enabled = True
    dataset.rotation_phase_origin_step = 0
    dataset.style_ratio_config = {
        "ratio_warmup_steps": 0,
        "ratio_ramp_steps": 0,
    }
    dataset.rotation_config = {
        "rotate_prompts": True,
        "rotate_seam_images": True,
        "rotate_style_assignments": True,
        "warmup_block_steps": 25,
        "ramp_block_steps": 25,
        "steady_block_steps": 10,
        "single_edge_fraction": 1.0,
        "multi_edge_fraction": 0.0,
    }
    dataset._rotation_style_entries = [
        {
            "sample_id": "sample_rotation",
            "style_family_id": "family_rotation",
            "reference_image_path": seam_variant_path,
            "seam_role_image_paths": [seam_variant_path],
            "variant_image_paths": [interior_path, seam_variant_path],
            "interior_prompt_variants": [
                {
                    "prompt_name": "variant_a",
                    "image_path": interior_path,
                }
            ],
        }
    ]
    return dataset


def test_style_ratio_support_excludes_valid_expanded_halo() -> None:
    edge_band_masks = torch.zeros((1, 4, 10, 10), dtype=torch.float32)
    seam_decay_maps = torch.zeros_like(edge_band_masks)
    edge_defined_flags = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    seam_strip_width_px = torch.tensor([2.0], dtype=torch.float32)
    supervision_mask = torch.ones((1, 1, 10, 10), dtype=torch.float32)
    valid_expanded_source_mask = torch.ones((1, 1, 10, 10), dtype=torch.float32)
    q_valid_support_mask = torch.zeros((1, 1, 10, 10), dtype=torch.float32)
    q_valid_support_mask[:, :, 2:8, 2:8] = 1.0

    seam_maps = build_seam_region_maps(
        edge_band_masks=edge_band_masks,
        seam_decay_maps=seam_decay_maps,
        edge_defined_flags=edge_defined_flags,
        seam_strip_width_px=seam_strip_width_px,
        supervision_mask=supervision_mask,
        seam_config={},
        expanded_halo_px=2,
        source_sizes_hw=torch.tensor([[6.0, 6.0]], dtype=torch.float32),
        expanded_source_boxes=torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
        valid_expanded_source_mask=valid_expanded_source_mask,
        continuation_valid_mask=None,
        style_support_valid_mask=None,
        q_valid_support_mask=q_valid_support_mask,
        style_ratio_config={
            "hard_band_end_px": 1.0,
            "near_band_end_px": 2.0,
            "overlap_band_end_px": 3.0,
            "soft_field_end_px": 5.0,
            "valid_style_support_start_px": 1.0,
            "controlnet_style_ramp_start_px": 2.0,
            "controlnet_style_ramp_end_px": 3.0,
        },
    )

    signed_distance = seam_maps["source_signed_distance_per_edge"]
    source_interior_mask = (signed_distance >= 0.0).all(dim=1, keepdim=True).to(dtype=torch.float32)
    valid_halo_mask = ((1.0 - source_interior_mask) * valid_expanded_source_mask).clamp(0.0, 1.0)

    for field_name in (
        "hard_band_mask",
        "near_band_mask",
        "overlap_band_mask",
        "soft_field_mask",
        "valid_style_support_mask",
        "style_spatial_support_mask",
    ):
        field = seam_maps[field_name]
        assert float((field * valid_halo_mask).sum().item()) == 0.0, field_name

    assert torch.allclose(
        seam_maps["soft_field_q_sum"] * valid_halo_mask,
        torch.zeros_like(valid_halo_mask),
        atol=1e-5,
    )
    assert float((seam_maps["soft_field_q_sum"] * source_interior_mask).sum().item()) > 0.0
    assert float((seam_maps["soft_field_mask"] * source_interior_mask).sum().item()) > 0.0


def test_q_fields_follow_loss_support_not_style_support() -> None:
    height = 16
    width = 32
    edge_band_masks = torch.zeros((1, 4, height, width), dtype=torch.float32)
    seam_decay_maps = torch.zeros_like(edge_band_masks)
    edge_defined_flags = torch.tensor([[1.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    seam_strip_width_px = torch.tensor([2.0], dtype=torch.float32)
    regional_loss_valid_mask = torch.ones((1, 1, height, width), dtype=torch.float32)
    regional_loss_valid_mask[:, :, 6:10, 12:20] = 0.0
    style_support_valid_mask = torch.zeros((1, 1, height, width), dtype=torch.float32)
    style_support_valid_mask[:, :, 0:3, 14:18] = 1.0

    seam_maps = build_seam_region_maps(
        edge_band_masks=edge_band_masks,
        seam_decay_maps=seam_decay_maps,
        edge_defined_flags=edge_defined_flags,
        seam_strip_width_px=seam_strip_width_px,
        supervision_mask=torch.ones_like(regional_loss_valid_mask),
        seam_config={},
        source_sizes_hw=torch.tensor([[float(height), float(width)]], dtype=torch.float32),
        style_support_valid_mask=style_support_valid_mask,
        q_valid_support_mask=regional_loss_valid_mask,
        style_ratio_config={
            "edge_plateau_px": 2.0,
            "edge_radius_fraction": 1.0,
            "edge_sigma_fraction": 0.35,
            "q_normalize": True,
            "soft_field_end_px": 30.0,
        },
    )

    q_sum = seam_maps["soft_field_q_per_edge"].sum(dim=1, keepdim=True) + seam_maps["soft_field_q_interior"]
    assert torch.allclose(q_sum * regional_loss_valid_mask, regional_loss_valid_mask, atol=1e-5)
    assert float((q_sum * (1.0 - regional_loss_valid_mask)).sum().item()) == 0.0
    assert float(q_sum[0, 0, 4, 4].item()) > 0.99
    assert float(style_support_valid_mask[0, 0, 4, 4].item()) == 0.0


def test_q_fields_keep_interior_competing_on_edge_plateaus() -> None:
    height = 24
    width = 48
    edge_band_masks = torch.zeros((1, 4, height, width), dtype=torch.float32)
    seam_decay_maps = torch.zeros_like(edge_band_masks)
    edge_defined_flags = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    seam_strip_width_px = torch.tensor([2.0], dtype=torch.float32)
    supervision_mask = torch.ones((1, 1, height, width), dtype=torch.float32)

    seam_maps = build_seam_region_maps(
        edge_band_masks=edge_band_masks,
        seam_decay_maps=seam_decay_maps,
        edge_defined_flags=edge_defined_flags,
        seam_strip_width_px=seam_strip_width_px,
        supervision_mask=supervision_mask,
        seam_config={},
        source_sizes_hw=torch.tensor([[float(height), float(width)]], dtype=torch.float32),
        style_ratio_config={
            "edge_plateau_px": 4.0,
            "edge_radius_fraction": 1.0,
            "edge_sigma_fraction": 0.35,
            "q_normalize": True,
            "soft_field_end_px": 20.0,
        },
    )

    q_edges = seam_maps["soft_field_q_per_edge"]
    q_interior = seam_maps["soft_field_q_interior"]
    q_sum = q_edges.sum(dim=1, keepdim=True) + q_interior

    assert torch.allclose(q_sum, torch.ones_like(q_sum), atol=1e-5)
    assert float(q_edges[0, 0, 0, width // 2].item()) >= 0.999
    assert float(q_interior[0, 0, 0, width // 2].item()) <= 1e-6


def test_smooth_q_fields_decay_and_overlap_at_corners() -> None:
    height = 24
    width = 64
    edge_band_masks = torch.zeros((1, 4, height, width), dtype=torch.float32)
    seam_decay_maps = torch.zeros_like(edge_band_masks)
    edge_defined_flags = torch.tensor([[1.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    seam_strip_width_px = torch.tensor([2.0], dtype=torch.float32)
    supervision_mask = torch.ones((1, 1, height, width), dtype=torch.float32)

    seam_maps = build_seam_region_maps(
        edge_band_masks=edge_band_masks,
        seam_decay_maps=seam_decay_maps,
        edge_defined_flags=edge_defined_flags,
        seam_strip_width_px=seam_strip_width_px,
        supervision_mask=supervision_mask,
        seam_config={},
        source_sizes_hw=torch.tensor([[float(height), float(width)]], dtype=torch.float32),
        style_ratio_config={
            "edge_plateau_px": 2.0,
            "q_edge_sigma_px": 180.0,
            "q_normalize": True,
            "soft_field_end_px": 20.0,
        },
    )

    q_edges = seam_maps["soft_field_q_per_edge"]
    q_interior = seam_maps["soft_field_q_interior"]
    q_pre_mask = seam_maps["q_edge_pre_mask_per_edge"]
    q_distance = seam_maps["q_distance_px_per_edge"]
    max_edge_q = seam_maps["q_max_edge"]
    q_sum = q_edges.sum(dim=1, keepdim=True) + q_interior

    assert torch.allclose(q_sum, torch.ones_like(q_sum), atol=1e-5)
    assert float(q_distance[0, 0, 1, width // 2].item()) > float(q_distance[0, 0, 0, width // 2].item())
    assert float(q_distance[0, 3, height // 2, 1].item()) > float(q_distance[0, 3, height // 2, 0].item())
    assert float(q_pre_mask[0, 0, 0, width // 2].item()) >= 0.95
    assert float(q_interior[0, 0, 0, width // 2].item()) <= 0.05
    assert float(q_edges[0, 0, 3, width // 2].item()) < float(q_edges[0, 0, 0, width // 2].item())
    assert float(q_edges[0, 3, height // 2, 3].item()) < float(q_edges[0, 3, height // 2, 0].item())
    assert float(q_edges[0, 1].max().item()) == 0.0
    assert float(q_edges[0, 2].max().item()) == 0.0
    assert float(q_edges[0, 0, 8, 8].item()) > 0.0
    assert float(q_edges[0, 3, 8, 8].item()) > 0.0
    assert abs(float(q_edges[0, 0, 0, 0].item()) - 0.5) <= 1e-5
    assert abs(float(q_edges[0, 3, 0, 0].item()) - 0.5) <= 1e-5
    assert float(q_interior[0, 0, 0, 0].item()) <= 0.05
    assert float(max_edge_q[0, 0, 0, 0].item()) >= 0.95
    assert float(q_interior[0, 0, height // 2, width // 2].item()) > float(q_interior[0, 0, 0, 0].item())
    assert float(q_edges[0, 0, -1, width // 2].item()) < float(q_edges[0, 0, 0, width // 2].item())


def test_q_fields_use_source_space_seam_edge_priors() -> None:
    height = 32
    width = 64
    edge_band_masks = torch.zeros((1, 4, height, width), dtype=torch.float32)
    seam_decay_maps = torch.zeros_like(edge_band_masks)
    edge_defined_flags = torch.tensor([[1.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    seam_strip_width_px = torch.tensor([2.0], dtype=torch.float32)
    supervision_mask = torch.ones((1, 1, height, width), dtype=torch.float32)
    valid_generated_mask = torch.zeros((1, 1, height, width), dtype=torch.float32)
    valid_generated_mask[:, :, 0:4, 20:45] = 1.0
    valid_generated_mask[:, :, 10:21, 0:4] = 1.0

    seam_maps = build_seam_region_maps(
        edge_band_masks=edge_band_masks,
        seam_decay_maps=seam_decay_maps,
        edge_defined_flags=edge_defined_flags,
        seam_strip_width_px=seam_strip_width_px,
        supervision_mask=supervision_mask,
        seam_config={},
        source_sizes_hw=torch.tensor([[float(height), float(width)]], dtype=torch.float32),
        style_support_valid_mask=valid_generated_mask,
        style_ratio_config={
            "hard_band_end_px": 4.0,
            "near_band_end_px": 8.0,
            "overlap_band_end_px": 12.0,
            "soft_field_end_px": 30.0,
            "edge_plateau_px": 2.0,
            "q_edge_sigma_px": 180.0,
            "q_normalize": True,
        },
    )

    q_edges = seam_maps["soft_field_q_per_edge"]
    q_decay = seam_maps["q_inward_decay_per_edge"]
    q_distance = seam_maps["q_distance_px_per_edge"]
    q_interior = seam_maps["q_interior"]

    assert float(q_decay[0, 0, 0, 32].item()) >= 0.95
    assert float(q_edges[0, 0, 0, 32].item()) > 0.5
    assert float(q_edges[0, 0, 0, 2].item()) > 0.45
    assert float(q_edges[0, 3, 12, 0].item()) > 0.45
    assert abs(float(q_edges[0, 0, 1, 0].item()) - 0.5) <= 1e-5
    assert abs(float(q_edges[0, 3, 1, 0].item()) - 0.5) <= 1e-5
    assert float(q_edges[0, 0, 8, 8].item()) > 0.0
    assert float(q_edges[0, 3, 8, 8].item()) > 0.0
    assert float(q_distance[0, 0, 3, 32].item()) > float(q_distance[0, 0, 0, 32].item())
    assert float(q_distance[0, 3, 12, 3].item()) > float(q_distance[0, 3, 12, 0].item())
    assert float(q_decay[0, 0, 6, 32].item()) < float(q_decay[0, 0, 0, 32].item())
    assert float(q_decay[0, 3, 12, 6].item()) < float(q_decay[0, 3, 12, 0].item())
    assert float(q_interior[0, 0, 0, 32].item()) <= 0.05
    assert float(q_interior[0, 0, 12, 0].item()) <= 0.05


def test_q_edge_decay_saturates_then_balances_bottom_edge_and_interior() -> None:
    height = 640
    width = 128
    edge_band_masks = torch.zeros((1, 4, height, width), dtype=torch.float32)
    seam_decay_maps = torch.zeros_like(edge_band_masks)
    edge_defined_flags = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    seam_strip_width_px = torch.tensor([2.0], dtype=torch.float32)
    supervision_mask = torch.ones((1, 1, height, width), dtype=torch.float32)

    seam_maps = build_seam_region_maps(
        edge_band_masks=edge_band_masks,
        seam_decay_maps=seam_decay_maps,
        edge_defined_flags=edge_defined_flags,
        seam_strip_width_px=seam_strip_width_px,
        supervision_mask=supervision_mask,
        seam_config={},
        source_sizes_hw=torch.tensor([[float(height), float(width)]], dtype=torch.float32),
        style_ratio_config={
            "edge_plateau_px": 16.0,
            "q_edge_sigma_px": 180.0,
            "q_interior_sigma_px": 180.0,
            "q_floor": 0.0,
            "q_normalize": True,
        },
    )

    q_edges = seam_maps["soft_field_q_per_edge"]
    q_interior = seam_maps["soft_field_q_interior"]
    q_decay = seam_maps["q_inward_decay_per_edge"]
    q_distance = seam_maps["q_distance_px_per_edge"]
    center_x = width // 2

    assert 0.0 <= float(q_distance[0, 1, height - 1, center_x].item()) <= 0.5
    assert float(q_decay[0, 1, height - 1, center_x].item()) == 1.0
    assert float(q_decay[0, 1, height - 1 - 16, center_x].item()) >= 0.998
    assert 0.03 <= float(q_decay[0, 1, height - 1 - 600, center_x].item()) <= 0.05

    bottom_edge_sum = float(q_edges[:, 1:2].sum().item())
    interior_sum = float(q_interior.sum().item())
    balance_ratio = bottom_edge_sum / max(interior_sum, 1e-6)
    assert 0.65 <= balance_ratio <= 1.15


def _build_regional_candidate_dataset_stub(paths: list[str]) -> TerrainSemanticManifestDataset:
    dataset = TerrainSemanticManifestDataset.__new__(TerrainSemanticManifestDataset)
    dataset.regional_loss_enabled = True
    dataset._records = [{"image_name": "family_a", "crop_box": (0, 0, 4, 4)}]
    dataset.train_size = (4, 4)
    dataset.expanded_target_halo_px = 0
    dataset._regional_rng = random.Random(1234)
    dataset._regional_audit_logged = 0
    dataset._regional_audit_max = 0
    family = {
        "variants": [
            {"path": paths[0], "prompt": "prompt 0", "negative_prompt": "", "name": "variant_0"},
            {"path": paths[1], "prompt": "prompt 1", "negative_prompt": "", "name": "variant_1"},
        ],
        "full_image_set": list(paths),
    }
    dataset._resolve_family_for_image_name = lambda _image_name: family
    dataset._load_candidate_train_rgba = lambda image_path, _crop_box: torch.full(
        (4, 4, 4),
        float(paths.index(image_path) + 1) / 10.0,
        dtype=torch.float32,
    )
    return dataset


def test_regional_candidates_never_reuse_active_target_image(tmp_path: Path) -> None:
    paths = [str(tmp_path / f"candidate_{idx}.png") for idx in range(5)]
    dataset = _build_regional_candidate_dataset_stub(paths)

    candidates = dataset._compose_regional_candidates(
        index=0,
        edge_defined_flags=torch.ones((4,), dtype=torch.float32),
    )

    assert candidates is not None
    active_mask = candidates["candidate_active_mask"]
    active_names = [
        name
        for is_active, name in zip(active_mask.tolist(), candidates["candidate_image_names"])
        if is_active > 0.5
    ]
    assert len(active_names) == 5
    assert len(active_names) == len(set(active_names))


def test_regional_candidates_disable_slots_instead_of_reusing_images(tmp_path: Path) -> None:
    unique_paths = [str(tmp_path / f"candidate_{idx}.png") for idx in range(3)]
    duplicated_paths = unique_paths + [unique_paths[1], unique_paths[2], unique_paths[2]]
    dataset = _build_regional_candidate_dataset_stub(duplicated_paths)

    candidates = dataset._compose_regional_candidates(
        index=0,
        edge_defined_flags=torch.ones((4,), dtype=torch.float32),
    )

    assert candidates is not None
    active_mask = candidates["candidate_active_mask"]
    active_names = [
        name
        for is_active, name in zip(active_mask.tolist(), candidates["candidate_image_names"])
        if is_active > 0.5
    ]
    assert len(active_names) == len(set(active_names))
    assert int(active_mask.sum().item()) == len(set(duplicated_paths))


def test_regenerated_seam_sources_select_distinct_family_variant(tmp_path: Path) -> None:
    interior_path = tmp_path / "interior.png"
    seam_variant_path = tmp_path / "seam_variant.png"
    _write_test_rgba(interior_path)
    _write_test_rgba(seam_variant_path)

    dataset = _build_dataset_stub([str(interior_path), str(seam_variant_path)])
    style_pool_entry = {
        "sample_id": "sample_a",
        "style_family_id": "family_a",
    }

    seam_paths = dataset._select_regenerated_seam_source_paths_by_edge(
        index=0,
        record={"image_name": "sample_a"},
        style_pool_entry=style_pool_entry,
        rotation_state={"block_id": 0, "phase_name": "steady"},
        interior_source_image_path=str(interior_path),
        edge_defined_flags=torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
    )

    assert seam_paths == {
        edge_name: str(seam_variant_path)
        for edge_name in EDGE_INDEX_TO_NAME
    }


def test_same_family_regenerated_paths_excludes_legacy_reference_when_family_exists(tmp_path: Path) -> None:
    interior_path = tmp_path / "interior.png"
    seam_variant_path = tmp_path / "seam_variant.png"
    regenerated_original_path = tmp_path / "original_rgba.png"
    legacy_reference_path = tmp_path / "legacy_reference.png"
    _write_test_rgba(interior_path)
    _write_test_rgba(seam_variant_path)
    _write_test_rgba(regenerated_original_path)
    _write_test_rgba(legacy_reference_path)

    dataset = TerrainSemanticManifestDataset.__new__(TerrainSemanticManifestDataset)
    family_paths = dataset._same_family_regenerated_paths(
        {
            "reference_image_path": str(legacy_reference_path),
            "variant_image_paths": [str(interior_path), str(seam_variant_path)],
            "seam_role_image_paths": [str(seam_variant_path), str(regenerated_original_path)],
            "interior_prompt_variants": [
                {
                    "image_path": str(interior_path),
                }
            ],
        }
    )

    assert str(legacy_reference_path) not in family_paths
    assert family_paths == [
        str(interior_path),
        str(seam_variant_path),
        str(regenerated_original_path),
    ]


def test_merge_regenerated_prompt_variants_strips_legacy_reference_from_family_lists(tmp_path: Path) -> None:
    legacy_reference_path = tmp_path / "legacy_reference.png"
    foreign_path = tmp_path / "foreign_variant.png"
    regenerated_variant_path = tmp_path / "variant_01.png"
    regenerated_original_path = tmp_path / "original_rgba.png"
    _write_test_rgba(legacy_reference_path)
    _write_test_rgba(foreign_path)
    _write_test_rgba(regenerated_variant_path)
    _write_test_rgba(regenerated_original_path)

    mapping = {
        "LeftOverhang": {
            "sample_id": "LeftOverhang",
            "style_family_id": "LeftOverhang",
            "reference_image_path": str(legacy_reference_path),
            "variant_image_paths": [str(legacy_reference_path), str(foreign_path)],
            "seam_role_image_paths": [str(legacy_reference_path), str(foreign_path)],
        }
    }
    merged = _merge_regenerated_prompt_variants(
        mapping,
        {
            "LeftOverhang": [
                {
                    "prompt": "prompt",
                    "prompt2": "prompt",
                    "prompt_name": "variant_01",
                    "image_path": str(regenerated_variant_path),
                }
            ]
        },
        {
            "LeftOverhang": [str(regenerated_original_path)],
        },
    )

    entry = merged["LeftOverhang"]
    assert str(legacy_reference_path) not in entry["variant_image_paths"]
    assert str(legacy_reference_path) not in entry["seam_role_image_paths"]
    assert str(foreign_path) not in entry["variant_image_paths"]
    assert str(foreign_path) not in entry["seam_role_image_paths"]
    assert entry["variant_image_paths"] == [str(regenerated_variant_path)]
    assert entry["seam_role_image_paths"] == [str(regenerated_original_path)]


def test_regenerated_seam_sources_allow_same_family_explicit_variant(tmp_path: Path) -> None:
    interior_path = tmp_path / "interior.png"
    seam_variant_path = tmp_path / "seam_variant.png"
    _write_test_rgba(interior_path)
    _write_test_rgba(seam_variant_path)

    dataset = _build_dataset_stub([str(interior_path), str(seam_variant_path)])
    style_pool_entry = {
        "sample_id": "sample_b",
        "style_family_id": "family_b",
        "selected_seam_image_source_paths_by_edge": {
            "north": str(seam_variant_path),
            "south": str(interior_path),
        },
    }

    seam_paths = dataset._select_regenerated_seam_source_paths_by_edge(
        index=0,
        record={"image_name": "sample_b"},
        style_pool_entry=style_pool_entry,
        rotation_state={"block_id": 0, "phase_name": "steady"},
        interior_source_image_path=str(interior_path),
        edge_defined_flags=torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
    )

    assert seam_paths == {
        "north": str(seam_variant_path),
        "south": str(seam_variant_path),
    }


def test_regenerated_seam_sources_fail_loud_on_cross_family_explicit_variant(tmp_path: Path) -> None:
    interior_path = tmp_path / "interior.png"
    seam_variant_path = tmp_path / "seam_variant.png"
    foreign_variant_path = tmp_path / "foreign_variant.png"
    _write_test_rgba(interior_path)
    _write_test_rgba(seam_variant_path)
    _write_test_rgba(foreign_variant_path)

    dataset = _build_dataset_stub([str(interior_path), str(seam_variant_path)])
    style_pool_entry = {
        "sample_id": "sample_b",
        "style_family_id": "family_b",
        "selected_seam_image_source_paths_by_edge": {
            "north": str(foreign_variant_path),
        },
    }

    with pytest.raises(RuntimeError, match="stay within the interior variant family"):
        dataset._select_regenerated_seam_source_paths_by_edge(
            index=0,
            record={"image_name": "sample_b"},
            style_pool_entry=style_pool_entry,
            rotation_state={"block_id": 0, "phase_name": "steady"},
            interior_source_image_path=str(interior_path),
            edge_defined_flags=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        )


def test_rotation_state_selects_distinct_seam_family_variant(tmp_path: Path) -> None:
    interior_path = tmp_path / "interior.png"
    seam_variant_path = tmp_path / "seam_variant.png"
    _write_test_rgba(interior_path)
    _write_test_rgba(seam_variant_path)

    dataset = _build_rotation_dataset_stub(str(interior_path), str(seam_variant_path))

    rotation_state = dataset._resolve_rotation_state(training_global_step=0)

    assert rotation_state["selected_prompt_variant"]["image_path"] == str(interior_path)
    assert rotation_state["selected_seam_image_source_path"] == str(seam_variant_path)
    assert rotation_state["selected_seam_image_source_id"] == "seam_variant"


def test_rotated_style_entry_allows_same_family_distinct_source(tmp_path: Path) -> None:
    interior_path = tmp_path / "interior.png"
    seam_variant_path = tmp_path / "seam_variant.png"
    _write_test_rgba(interior_path)
    _write_test_rgba(seam_variant_path)

    dataset = _build_rotation_dataset_stub(str(interior_path), str(seam_variant_path))
    base_entry = {
        "sample_id": "sample_rotation",
        "style_family_id": "family_rotation",
    }
    rotation_state = {
        "enabled": True,
        "anchor_entry": dict(dataset._rotation_style_entries[0]),
        "selected_prompt_variant": {
            "prompt_name": "variant_a",
            "image_path": str(interior_path),
        },
        "selected_seam_image_source_path": str(seam_variant_path),
        "selected_seam_image_source_id": "seam_variant",
    }

    rotated_entry = dataset._resolve_rotated_style_pool_entry(base_entry, rotation_state)

    assert rotated_entry["selected_interior_prompt_image_path"] == str(interior_path)
    assert rotated_entry["selected_seam_image_source_path"] == str(seam_variant_path)
    assert rotated_entry["selected_seam_image_source_only"] is True
    assert rotated_entry["interior_style_id"] == "interior::variant_a"
    assert rotated_entry["selected_interior_style_id"] == "interior::variant_a"
    assert rotated_entry["edge_style_ids"] == {
        edge_name: "seam::seam_variant"
        for edge_name in EDGE_INDEX_TO_NAME
    }


def test_rotated_style_entry_fails_loud_on_cross_family_source(tmp_path: Path) -> None:
    interior_path = tmp_path / "interior.png"
    seam_variant_path = tmp_path / "seam_variant.png"
    foreign_variant_path = tmp_path / "foreign_variant.png"
    _write_test_rgba(interior_path)
    _write_test_rgba(seam_variant_path)
    _write_test_rgba(foreign_variant_path)

    dataset = _build_rotation_dataset_stub(str(interior_path), str(seam_variant_path))
    base_entry = {
        "sample_id": "sample_rotation",
        "style_family_id": "family_rotation",
    }
    rotation_state = {
        "enabled": True,
        "anchor_entry": dict(dataset._rotation_style_entries[0]),
        "selected_prompt_variant": {
            "prompt_name": "variant_a",
            "image_path": str(interior_path),
        },
        "selected_seam_image_source_path": str(foreign_variant_path),
        "selected_seam_image_source_id": "foreign_variant",
    }

    with pytest.raises(RuntimeError, match="must stay within the selected interior variant family"):
        dataset._resolve_rotated_style_pool_entry(base_entry, rotation_state)