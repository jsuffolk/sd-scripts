import torch

from library.regional_multitarget_loss import RegionalLossConfig, compute_regional_loss


def _build_two_candidate_case():
    pred_x0 = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
    targets = torch.zeros((1, 2, 4, 2, 2), dtype=torch.float32)
    targets[:, 0, :, :, 1] = 1.0
    targets[:, 1, :, :, 0] = 1.0
    q = torch.zeros((1, 2, 1, 2, 2), dtype=torch.float32)
    q[:, 0, :, :, 0] = 0.9
    q[:, 0, :, :, 1] = 0.1
    q[:, 1, :, :, 0] = 0.1
    q[:, 1, :, :, 1] = 0.9
    return {
        "pred_x0_latents": pred_x0,
        "targets_x0_latents": targets,
        "candidate_active_mask": torch.ones((1, 2), dtype=torch.float32),
        "candidate_q_field_latent": q,
        "noise": torch.zeros_like(pred_x0),
        "noise_pred": torch.zeros_like(pred_x0),
        "noisy_latents": torch.zeros_like(pred_x0),
        "sqrt_alpha_t": torch.ones((1, 1, 1, 1), dtype=torch.float32),
        "sqrt_one_minus_alpha_t": torch.ones((1, 1, 1, 1), dtype=torch.float32),
        "trusted_mask_latent": torch.ones((1, 1, 2, 2), dtype=torch.float32),
    }


def test_q_route_uses_conservative_sharpening_for_routing_entropy() -> None:
    cfg = RegionalLossConfig(enabled=True, kernel_lat=1, stride_lat=1, gamma_route=3.0)
    common = _build_two_candidate_case()

    phase1 = compute_regional_loss(
        **common,
        cfg=cfg,
        current_step=0,
        rho_q_route_override=1.0,
        phase_override=1,
    )["diagnostics"]
    phase2 = compute_regional_loss(
        **common,
        cfg=cfg,
        current_step=0,
        rho_q_route_override=1.0,
        phase_override=2,
    )["diagnostics"]

    assert torch.allclose(phase1["routing_entropy_mean"], phase1["q_boot_entropy_mean"], atol=1e-6)
    assert torch.allclose(phase2["routing_entropy_mean"], phase2["q_boot_entropy_mean"], atol=1e-6)
    assert phase1["q_boot_entropy_mean"] < phase1["q_entropy_mean"]
    assert torch.allclose(phase1["gamma_route"], torch.tensor(3.0), atol=1e-6)


def test_q_confidence_masks_q_regret_constraints() -> None:
    common = _build_two_candidate_case()
    common["candidate_q_field_latent"] = torch.full((1, 2, 1, 2, 2), 0.5, dtype=torch.float32)
    common["candidate_q_field_latent"][:, 0] = 0.51
    common["candidate_q_field_latent"][:, 1] = 0.49

    masked = compute_regional_loss(
        **common,
        cfg=RegionalLossConfig(
            enabled=True,
            kernel_lat=1,
            stride_lat=1,
            q_regret_loss_weight=1.0,
            q_regret_q_tol=0.0,
            q_confidence_threshold=0.2,
            q_confidence_mask_q_regret=True,
        ),
        current_step=0,
        rho_q_route_override=1.0,
        lambda_q_regret_override=1.0,
        phase_override=1,
    )["diagnostics"]
    unmasked = compute_regional_loss(
        **common,
        cfg=RegionalLossConfig(
            enabled=True,
            kernel_lat=1,
            stride_lat=1,
            q_regret_loss_weight=1.0,
            q_regret_q_tol=0.0,
            q_confidence_threshold=0.2,
            q_confidence_mask_q_regret=False,
        ),
        current_step=0,
        rho_q_route_override=1.0,
        lambda_q_regret_override=1.0,
        phase_override=1,
    )["diagnostics"]

    assert masked["q_confidence_mask_fraction"] < torch.tensor(1e-6)
    assert torch.allclose(masked["q_regret_loss"], torch.tensor(0.0), atol=1e-6)
    assert unmasked["q_regret_active_pair_fraction"] > torch.tensor(0.0)


def test_in_region_advantage_compares_self_to_competitors() -> None:
    cfg = RegionalLossConfig(enabled=True, kernel_lat=1, stride_lat=1, q_high_threshold=0.45)
    outputs = compute_regional_loss(
        **_build_two_candidate_case(),
        cfg=cfg,
        current_step=0,
        rho_q_route_override=1.0,
        phase_override=1,
    )
    diagnostics = outputs["diagnostics"]

    assert torch.allclose(diagnostics["R_self_slot0"], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(diagnostics["R_comp_slot0"], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(diagnostics["in_region_advantage_slot0"], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(diagnostics["min_in_region_advantage"], torch.tensor(1.0), atol=1e-6)


def test_terrain_content_mask_weights_q_and_gate_diagnostics() -> None:
    cfg = RegionalLossConfig(enabled=True, kernel_lat=1, stride_lat=1, q_high_threshold=0.45)
    terrain_soft = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    terrain_soft[:, :, :, 0] = 1.0

    diagnostics = compute_regional_loss(
        **_build_two_candidate_case(),
        terrain_soft_latent=terrain_soft,
        cfg=cfg,
        current_step=0,
        rho_q_route_override=1.0,
        phase_override=1,
    )["diagnostics"]

    assert torch.allclose(diagnostics["active_content_fraction"], torch.tensor(0.5), atol=1e-6)
    expected_route_slot0 = torch.tensor((0.9**3) / ((0.9**3) + (0.1**3)), dtype=torch.float32)
    expected_route_slot1 = torch.tensor((0.1**3) / ((0.9**3) + (0.1**3)), dtype=torch.float32)
    assert torch.allclose(diagnostics["q_mass_slot0"], expected_route_slot0, atol=1e-6)
    assert torch.allclose(diagnostics["q_mass_slot1"], expected_route_slot1, atol=1e-6)
    assert torch.allclose(diagnostics["q_raw_mass_slot0"], torch.tensor(0.9), atol=1e-6)
    assert torch.allclose(diagnostics["q_raw_mass_slot1"], torch.tensor(0.1), atol=1e-6)
    assert torch.allclose(diagnostics["R_self_slot0"], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(diagnostics["R_comp_slot0"], torch.tensor(1.0), atol=1e-6)


def test_terrain_content_mask_falls_back_when_too_small() -> None:
    cfg = RegionalLossConfig(
        enabled=True,
        kernel_lat=1,
        stride_lat=1,
        min_active_content_fraction=0.75,
    )
    terrain_soft = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    terrain_soft[:, :, 0, 0] = 1.0

    diagnostics = compute_regional_loss(
        **_build_two_candidate_case(),
        terrain_soft_latent=terrain_soft,
        cfg=cfg,
        current_step=0,
        rho_q_route_override=1.0,
        phase_override=1,
    )["diagnostics"]

    assert torch.allclose(diagnostics["active_content_mask_fallback"], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(diagnostics["active_content_fraction"], torch.tensor(1.0), atol=1e-6)


def test_terrain_empty_sample_does_not_fall_back_to_active_diffusion() -> None:
    cfg = RegionalLossConfig(enabled=True, kernel_lat=1, stride_lat=1, min_active_content_fraction=0.75)
    terrain_soft = torch.zeros((1, 1, 2, 2), dtype=torch.float32)

    diagnostics = compute_regional_loss(
        **_build_two_candidate_case(),
        terrain_soft_latent=terrain_soft,
        cfg=cfg,
        current_step=0,
        rho_q_route_override=1.0,
        phase_override=1,
    )["diagnostics"]

    assert torch.allclose(diagnostics["terrain_empty_sample_count"], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(diagnostics["active_content_mask_fallback"], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(diagnostics["active_content_fraction"], torch.tensor(0.0), atol=1e-6)


def test_low_sigma_downweights_candidate_competition_metrics() -> None:
    cfg = RegionalLossConfig(
        enabled=True,
        kernel_lat=1,
        stride_lat=1,
        q_high_threshold=0.45,
        candidate_competition_sigma_ref=0.2,
        candidate_competition_power=2.0,
        candidate_competition_min_weight=0.1,
    )
    common = _build_two_candidate_case()
    common["sqrt_one_minus_alpha_t"] = torch.full((1, 1, 1, 1), 0.01, dtype=torch.float32)

    diagnostics = compute_regional_loss(
        **common,
        cfg=cfg,
        current_step=0,
        rho_q_route_override=1.0,
        phase_override=1,
    )["diagnostics"]

    assert torch.allclose(diagnostics["candidate_competition_weight_mean"], torch.tensor(0.1), atol=1e-6)
    assert torch.allclose(diagnostics["R_self_slot0"], torch.tensor(0.45), atol=1e-6)
    assert torch.allclose(diagnostics["R_comp_slot0"], torch.tensor(0.55), atol=1e-6)
    assert torch.allclose(diagnostics["in_region_advantage_slot0"], torch.tensor(0.1), atol=1e-6)


def test_low_sigma_raw_spike_visible_safe_loss_bounded() -> None:
    cfg = RegionalLossConfig(
        enabled=True,
        kernel_lat=1,
        stride_lat=1,
        target_eps_sigma_floor=0.05,
        robust_diffusion_loss_enabled=True,
        safe_pixel_loss_cap=4.0,
    )
    common = _build_two_candidate_case()
    common["sqrt_one_minus_alpha_t"] = torch.full((1, 1, 1, 1), 0.01, dtype=torch.float32)

    diagnostics = compute_regional_loss(
        **common,
        cfg=cfg,
        current_step=0,
        rho_q_route_override=1.0,
        phase_override=1,
    )["diagnostics"]

    assert diagnostics["regional_diff_loss_raw"] > diagnostics["regional_diff_loss_safe"]
    assert diagnostics["regional_loss_raw_safe_ratio"] > torch.tensor(1.0)
    assert torch.allclose(diagnostics["target_eps_sigma_min_raw"], torch.tensor([0.01]), atol=1e-6)
    assert torch.allclose(diagnostics["target_eps_sigma_min_safe"], torch.tensor([0.05]), atol=1e-6)


def test_low_sigma_can_skip_candidate_competition() -> None:
    cfg = RegionalLossConfig(
        enabled=True,
        kernel_lat=1,
        stride_lat=1,
        regional_competition_skip_sigma_threshold=0.05,
        regional_competition_skip_snr_threshold=0.0,
    )
    common = _build_two_candidate_case()
    common["sqrt_one_minus_alpha_t"] = torch.full((1, 1, 1, 1), 0.01, dtype=torch.float32)

    diagnostics = compute_regional_loss(
        **common,
        cfg=cfg,
        current_step=0,
        rho_q_route_override=1.0,
        phase_override=1,
    )["diagnostics"]

    assert torch.allclose(diagnostics["candidate_competition_weight_mean"], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(diagnostics["regional_competition_skipped_low_sigma"], torch.tensor(1.0), atol=1e-6)


def test_competitive_gate_keeps_base_signal_under_negative_advantage() -> None:
    common = _build_two_candidate_case()
    common["candidate_q_field_latent"] = common["candidate_q_field_latent"].flip(1)
    outputs = compute_regional_loss(
        **common,
        cfg=RegionalLossConfig(
            enabled=True,
            kernel_lat=1,
            stride_lat=1,
            competitive_gate_beta=0.9,
            competitive_gate_tolerance=0.0,
            competitive_gate_scale=0.01,
            competitive_gate_max_delta_per_step=0.05,
            competitive_gate_base_weight_floor=0.6,
            competitive_gate_min_region_support=0.01,
            competitive_catchup_enabled=True,
            competitive_catchup_weight=0.02,
        ),
        current_step=0,
        rho_q_route_override=1.0,
        phase_override=1,
        advantage_ema_prev=torch.zeros((2,), dtype=torch.float32),
        competitive_gate_prev=torch.ones((2,), dtype=torch.float32),
    )
    diagnostics = outputs["diagnostics"]

    assert diagnostics["in_region_advantage_slot0"] < torch.tensor(0.0)
    assert torch.allclose(diagnostics["competitive_gate_slot0"], torch.tensor(0.95), atol=1e-6)
    assert torch.allclose(diagnostics["competitive_gate_delta_slot0"], torch.tensor(0.05), atol=1e-6)
    assert diagnostics["base_region_loss_slot0"] > torch.tensor(0.0)
    assert diagnostics["final_region_loss_slot0"] >= diagnostics["base_region_loss_slot0"]
    assert torch.allclose(diagnostics["would_hard_gate_suppress_slot0"], torch.tensor(1.0), atol=1e-6)
    assert diagnostics["catchup_loss_slot0"] > torch.tensor(0.0)