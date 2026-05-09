import math

import pytest
import torch

from library.regional_multitarget_loss import RegionalLossConfig, compute_regional_loss, masked_spatial_mean


def _equivalence_cfg() -> RegionalLossConfig:
    return RegionalLossConfig(
        enabled=True,
        kernel_lat=1,
        stride_lat=1,
        gate_blur_sigma_pooled=0.0,
        beta=0.0,
        q_regret_loss_weight=0.0,
        bind_preference_weight=0.0,
        rgb_aux_loss_weight=0.0,
        terrain_loss_mask_enabled=False,
        target_eps_sigma_floor=1e-8,
        robust_diffusion_loss_enabled=False,
        regional_loss_sigma_weight_power=0.0,
        candidate_competition_power=0.0,
        candidate_competition_min_weight=1.0,
        regional_competition_skip_sigma_threshold=0.0,
        regional_competition_skip_snr_threshold=0.0,
        competitive_gate_min_region_support=0.0,
        competitive_gate_base_weight_floor=1.0,
        competitive_catchup_enabled=False,
    )


def _fixed_batch(candidate_count: int = 1, offsets: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    torch.manual_seed(20260509)
    batch_size, latent_channels, height, width = 1, 4, 3, 5
    x0 = torch.linspace(-0.5, 0.75, batch_size * latent_channels * height * width, dtype=torch.float32).view(
        batch_size, latent_channels, height, width
    )
    noise = torch.randn_like(x0) * 0.17
    noise_pred = noise + (torch.randn_like(x0) * 0.031)
    sqrt_alpha = torch.full((batch_size, 1, 1, 1), math.sqrt(0.72), dtype=torch.float32)
    sqrt_sigma = torch.full((batch_size, 1, 1, 1), math.sqrt(0.28), dtype=torch.float32)
    noisy = (sqrt_alpha * x0) + (sqrt_sigma * noise)
    if offsets is None:
        targets = x0.unsqueeze(1).repeat(1, candidate_count, 1, 1, 1)
    else:
        targets = x0.unsqueeze(1) + offsets.view(1, candidate_count, 1, 1, 1)
    mask = torch.ones((batch_size, 1, height, width), dtype=torch.float32)
    mask[:, :, 0, 0] = 0.0
    mask[:, :, 2, 4] = 0.0
    return {
        "pred_x0_latents": (noisy - sqrt_sigma * noise_pred) / sqrt_alpha,
        "targets_x0_latents": targets,
        "candidate_active_mask": torch.ones((batch_size, candidate_count), dtype=torch.float32),
        "candidate_q_field_latent": torch.ones((batch_size, candidate_count, 1, height, width), dtype=torch.float32),
        "noise": noise,
        "noise_pred": noise_pred,
        "noisy_latents": noisy,
        "sqrt_alpha_t": sqrt_alpha,
        "sqrt_one_minus_alpha_t": sqrt_sigma,
        "trusted_mask_latent": mask,
    }


def _standard_diffusion_loss(noise_pred: torch.Tensor, noise: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    loss_map = (noise_pred.float() - noise.float()).pow(2)
    pixel_loss = loss_map.mean(dim=1, keepdim=True)
    return (pixel_loss * mask).sum() / mask.sum().clamp_min(1e-6), pixel_loss


def _run_regional(batch: dict[str, torch.Tensor], cfg: RegionalLossConfig | None = None, **kwargs) -> dict[str, torch.Tensor]:
    return compute_regional_loss(
        **batch,
        cfg=cfg or _equivalence_cfg(),
        current_step=0,
        rho_q_route_override=kwargs.pop("rho_q_route_override", 1.0),
        lambda_q_regret_override=kwargs.pop("lambda_q_regret_override", 0.0),
        lambda_bind_override=kwargs.pop("lambda_bind_override", 0.0),
        phase_override=kwargs.pop("phase_override", 1.0),
        **kwargs,
    )


def test_c1_regional_loss_matches_standard_diffusion_loss(record_property) -> None:
    batch = _fixed_batch(candidate_count=1)
    standard_loss, standard_pixel_loss = _standard_diffusion_loss(
        batch["noise_pred"], batch["noise"], batch["trusted_mask_latent"]
    )

    out = _run_regional(batch)
    diagnostics = out["diagnostics"]

    record_property("target_type", "epsilon")
    record_property("prediction_type", "epsilon")
    record_property("timestep", "fixed synthetic alpha_cumprod=0.72")
    record_property("sigma", float(diagnostics["sigma"].item()))
    record_property("snr", float(diagnostics["snr"].item()))
    record_property("loss_weighting", float(diagnostics["regional_low_sigma_weight"].item()))

    assert torch.allclose(out["loss"], standard_loss, atol=1e-7, rtol=1e-6)
    assert torch.allclose(diagnostics["regional_diff_loss_raw"], standard_loss, atol=1e-7, rtol=1e-6)
    assert torch.allclose(out["L_train_pixel_raw"][:, 0:1], standard_pixel_loss, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("routing", ["uniform", "one_hot"])
def test_identical_candidates_match_base_loss_without_candidate_factor(routing: str) -> None:
    candidate_count = 5
    batch = _fixed_batch(candidate_count=candidate_count)
    if routing == "one_hot":
        q = torch.zeros_like(batch["candidate_q_field_latent"])
        q[:, 3:4] = 1.0
        batch["candidate_q_field_latent"] = q

    standard_loss, _ = _standard_diffusion_loss(batch["noise_pred"], batch["noise"], batch["trusted_mask_latent"])
    out = _run_regional(batch)

    assert torch.allclose(out["loss"], standard_loss, atol=1e-7, rtol=1e-6)
    assert torch.allclose(out["diagnostics"]["regional_diff_loss_raw"], standard_loss, atol=1e-7, rtol=1e-6)
    assert out["loss"] < standard_loss * 1.000001
    assert out["loss"] > standard_loss / 1.000001
    assert not torch.allclose(out["loss"], standard_loss * candidate_count, rtol=1e-3, atol=1e-3)


def _perfect_candidate_batch(selected: int = 0, q_requires_grad: bool = False, mask_requires_grad: bool = False) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    batch_size, candidate_count, latent_channels, height, width = 1, 5, 4, 2, 3
    target = torch.randn((batch_size, latent_channels, height, width), dtype=torch.float32) * 0.2
    offsets = torch.tensor([0.0, 0.35, -0.5, 0.8, -1.1], dtype=torch.float32)
    targets = target.unsqueeze(1) + offsets.view(1, candidate_count, 1, 1, 1)
    noise = torch.randn_like(target) * 0.1
    noise_pred = noise.clone()
    sqrt_alpha = torch.full((batch_size, 1, 1, 1), 0.8, dtype=torch.float32)
    sqrt_sigma = torch.full((batch_size, 1, 1, 1), 0.6, dtype=torch.float32)
    noisy = (sqrt_alpha * target) + (sqrt_sigma * noise)
    mask = torch.ones((batch_size, 1, height, width), dtype=torch.float32)
    mask[:, :, 0, 2] = 0.0
    mask.requires_grad_(mask_requires_grad)
    q = torch.zeros((batch_size, candidate_count, 1, height, width), dtype=torch.float32)
    q[:, selected : selected + 1] = 1.0
    q.requires_grad_(q_requires_grad)
    return {
        "pred_x0_latents": target.clone(),
        "targets_x0_latents": targets,
        "candidate_active_mask": torch.ones((batch_size, candidate_count), dtype=torch.float32),
        "candidate_q_field_latent": q,
        "noise": noise,
        "noise_pred": noise_pred,
        "noisy_latents": noisy,
        "sqrt_alpha_t": sqrt_alpha,
        "sqrt_one_minus_alpha_t": sqrt_sigma,
        "trusted_mask_latent": mask,
    }


def test_perfect_candidate_synthetic_behavior_and_gradients() -> None:
    perfect = _perfect_candidate_batch(selected=0, q_requires_grad=True, mask_requires_grad=True)
    perfect["noise_pred"].requires_grad_(True)
    out = _run_regional(perfect)
    diagnostics = out["diagnostics"]

    assert torch.allclose(diagnostics["R_self_slot0"], torch.tensor(0.0), atol=1e-7)
    assert diagnostics["R_comp_slot0"] > diagnostics["R_self_slot0"]
    assert torch.all(out["winner_idx_pooled"] == 0)
    assert torch.allclose(out["loss"], torch.tensor(0.0), atol=1e-7)
    assert diagnostics["in_region_advantage_slot0"] > torch.tensor(0.0)
    out["loss"].backward()
    assert perfect["noise_pred"].grad is not None
    assert perfect["noise_pred"].grad.norm() < 1e-7
    assert perfect["candidate_q_field_latent"].grad is None
    assert perfect["trusted_mask_latent"].grad is None

    bad = _perfect_candidate_batch(selected=1, q_requires_grad=True, mask_requires_grad=True)
    bad["noise_pred"].requires_grad_(True)
    bad_out = _run_regional(bad)
    assert bad_out["loss"] > out["loss"] + 0.01
    bad_out["loss"].backward()
    assert bad["noise_pred"].grad is not None
    assert bad["noise_pred"].grad.norm() > 0.0
    assert bad["candidate_q_field_latent"].grad is None
    assert bad["trusted_mask_latent"].grad is None


def test_candidate_dimension_preserves_identity_before_winner_computation() -> None:
    offsets = torch.tensor([0.10, -0.50, 0.25, 1.50, -0.75], dtype=torch.float32)
    batch = _fixed_batch(candidate_count=5, offsets=offsets)
    batch["noise_pred"] = batch["noise"].clone()
    batch["pred_x0_latents"] = (batch["noisy_latents"] - batch["sqrt_one_minus_alpha_t"] * batch["noise_pred"]) / batch["sqrt_alpha_t"]
    out = _run_regional(batch)

    r_by_candidate = out["R_pixel"].mean(dim=(0, 2, 3))
    assert torch.allclose(r_by_candidate, offsets.abs(), atol=1e-6)
    assert int(out["diagnostics"]["candidate_with_max_loss"].item()) == 3
    assert int(out["diagnostics"]["candidate_with_max_loss"].item()) != 0
    assert int(out["diagnostics"]["pixel_max_candidate"].item()) == 3
    assert out["R_pooled"].shape[1] == 5


def test_mask_reduction_denominators_for_spatial_candidate_and_latent_axes() -> None:
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]], dtype=torch.float32)
    spatial = torch.arange(1, 5, dtype=torch.float32).view(1, 2, 2)
    latent = torch.arange(1, 9, dtype=torch.float32).view(1, 2, 2, 2)
    candidates = torch.stack([spatial, spatial + 10.0, spatial + 20.0], dim=1)
    candidate_latent = torch.stack([latent, latent + 10.0, latent + 20.0], dim=1)

    manual_spatial = (spatial * mask.squeeze(1)).sum() / mask.sum()
    manual_latent = (latent * mask).sum() / (mask.sum() * latent.shape[1])
    manual_candidates = (candidates * mask.squeeze(1).unsqueeze(1)).sum(dim=(-2, -1)) / mask.sum()
    manual_candidate_latent = (candidate_latent * mask.view(1, 1, 1, 2, 2)).sum(dim=(-3, -2, -1)) / (
        mask.sum() * candidate_latent.shape[2]
    )

    assert torch.allclose(masked_spatial_mean(spatial, mask), manual_spatial)
    assert torch.allclose(masked_spatial_mean(latent, mask), manual_latent)
    assert torch.allclose(masked_spatial_mean(candidates, mask, candidate_dim=1), manual_candidates)
    assert torch.allclose(masked_spatial_mean(candidate_latent, mask, candidate_dim=1), manual_candidate_latent)


def test_low_sigma_scaling_applies_to_regional_weighted_path(record_property) -> None:
    ratios = []
    for sigma in (0.02, 0.10, 0.20, 0.50):
        batch = _fixed_batch(candidate_count=1)
        sqrt_sigma = torch.full_like(batch["sqrt_one_minus_alpha_t"], sigma)
        sqrt_alpha = torch.sqrt(torch.clamp(1.0 - sqrt_sigma.square(), min=1e-6))
        x0 = batch["targets_x0_latents"][:, 0]
        batch["sqrt_alpha_t"] = sqrt_alpha
        batch["sqrt_one_minus_alpha_t"] = sqrt_sigma
        batch["noisy_latents"] = (sqrt_alpha * x0) + (sqrt_sigma * batch["noise"])
        batch["pred_x0_latents"] = (batch["noisy_latents"] - sqrt_sigma * batch["noise_pred"]) / sqrt_alpha
        cfg = _equivalence_cfg()
        cfg.regional_loss_sigma_ref = 0.2
        cfg.regional_loss_sigma_weight_power = 2.0
        cfg.regional_loss_sigma_weight_min = 0.05
        cfg.regional_competition_skip_sigma_threshold = 0.05
        out = _run_regional(batch, cfg=cfg)
        diag = out["diagnostics"]
        ratio = float((out["loss"] / diag["regional_diff_loss_raw"].clamp_min(1e-12)).item())
        ratios.append(ratio)
        record_property(f"sigma_{sigma}_summary", {
            "sigma": sigma,
            "snr": float(diag["snr"].item()),
            "base_diffusion_loss": float(_standard_diffusion_loss(batch["noise_pred"], batch["noise"], batch["trusted_mask_latent"])[0].item()),
            "regional_diff_loss_raw": float(diag["regional_diff_loss_raw"].item()),
            "regional_diff_loss_weighted": float(out["loss"].item()),
            "low_sigma_weight": float(diag["regional_low_sigma_weight"].item()),
            "regional_competition_skipped_low_sigma": float(diag["regional_competition_skipped_low_sigma"].item()),
        })
        assert math.isclose(ratio, float(diag["regional_low_sigma_weight"].item()), rel_tol=1e-5, abs_tol=1e-7)
        if sigma <= 0.05:
            assert torch.allclose(diag["regional_competition_skipped_low_sigma"], torch.tensor(1.0), atol=1e-6)
    assert ratios[0] < ratios[-1]


@pytest.mark.parametrize("candidate_count", [1, 5])
def test_fixed_batch_regional_overfit_decreases_loss(candidate_count: int) -> None:
    base = _perfect_candidate_batch(selected=0)
    if candidate_count == 1:
        base["targets_x0_latents"] = base["targets_x0_latents"][:, :1]
        base["candidate_active_mask"] = base["candidate_active_mask"][:, :1]
        base["candidate_q_field_latent"] = base["candidate_q_field_latent"][:, :1]
    target0 = base["targets_x0_latents"][:, 0]
    sqrt_alpha = base["sqrt_alpha_t"]
    sqrt_sigma = base["sqrt_one_minus_alpha_t"]
    noise_pred = (base["noise"] + 0.75).clone().requires_grad_(True)
    optimizer = torch.optim.SGD([noise_pred], lr=0.25)
    losses = []
    r_self = []
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        batch = dict(base)
        batch["noise_pred"] = noise_pred
        batch["pred_x0_latents"] = (batch["noisy_latents"] - sqrt_sigma * noise_pred) / sqrt_alpha
        out = _run_regional(batch)
        loss = out["loss"]
        assert torch.isfinite(loss)
        assert loss >= 0.0
        loss.backward()
        assert noise_pred.grad is not None and torch.isfinite(noise_pred.grad).all()
        optimizer.step()
        losses.append(float(loss.detach().item()))
        r_self.append(float(out["diagnostics"]["R_self_slot0"].item()))
    assert losses[-1] < losses[0] * 0.25
    assert r_self[-1] < r_self[0]
    assert torch.isfinite(noise_pred).all()
    assert torch.allclose((base["noisy_latents"] - sqrt_sigma * noise_pred.detach()) / sqrt_alpha, target0, atol=0.25)


def test_gradient_path_audit_candidate_reconstruction_only() -> None:
    batch = _perfect_candidate_batch(selected=1, q_requires_grad=True, mask_requires_grad=True)
    batch["noise_pred"] = (batch["noise"] + 0.2).clone().requires_grad_(True)
    out = _run_regional(batch)
    out["loss"].backward()

    assert batch["noise_pred"].grad is not None
    assert batch["noise_pred"].grad.norm() > 0.0
    assert batch["candidate_q_field_latent"].grad is None
    assert batch["trusted_mask_latent"].grad is None
    assert out["routing_pooled"].requires_grad is False
    assert out["gate"].requires_grad is False