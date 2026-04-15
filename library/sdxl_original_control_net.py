# some parts are modified from Diffusers library (Apache License 2.0)

import math
from types import SimpleNamespace
from typing import Any, Optional
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library import sdxl_original_unet
from library.sdxl_model_util import convert_sdxl_unet_state_dict_to_diffusers, convert_diffusers_unet_state_dict_to_sdxl


class ControlNetConditioningEmbedding(nn.Module):
    def __init__(self, conditioning_channels: int = 3, pre_embed_channels: Optional[list[int]] = None):
        super().__init__()

        dims = [16, 32, 96, 256]

        pre_embed_channels = pre_embed_channels or []
        self.pre_blocks = nn.ModuleList([])

        in_channels = conditioning_channels
        for out_channels in pre_embed_channels:
            self.pre_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            in_channels = out_channels

        self.conv_in = nn.Conv2d(in_channels, dims[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([])

        for i in range(len(dims) - 1):
            channel_in = dims[i]
            channel_out = dims[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = nn.Conv2d(dims[-1], 320, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)  # zero module weight
        nn.init.zeros_(self.conv_out.bias)  # zero module bias

    def forward(self, x):
        for block in self.pre_blocks:
            x = block(x)
            x = F.silu(x)
        x = self.conv_in(x)
        x = F.silu(x)
        for block in self.blocks:
            x = block(x)
            x = F.silu(x)
        x = self.conv_out(x)
        return x


class SdxlControlNet(sdxl_original_unet.SdxlUNet2DConditionModel):
    def __init__(
        self,
        multiplier: Optional[float] = None,
        conditioning_channels: int = 3,
        conditioning_pre_embed_channels: Optional[list[int]] = None,
        alpha_head_indices: Optional[list[int]] = None,
        alpha_baseline_mode: str = "none",
        alpha_baseline_terrain_channel_index: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.multiplier = multiplier
        self.alpha_head_indices = sorted(alpha_head_indices or [])
        self.alpha_baseline_mode = str(alpha_baseline_mode)
        self.alpha_baseline_terrain_channel_index = int(alpha_baseline_terrain_channel_index)

        # remove unet layers
        self.output_blocks = nn.ModuleList([])
        del self.out

        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_channels=conditioning_channels,
            pre_embed_channels=conditioning_pre_embed_channels,
        )

        dims = [320, 320, 320, 320, 640, 640, 640, 1280, 1280]
        self.controlnet_down_blocks = nn.ModuleList([])
        for dim in dims:
            self.controlnet_down_blocks.append(nn.Conv2d(dim, dim, kernel_size=1))
            nn.init.zeros_(self.controlnet_down_blocks[-1].weight)  # zero module weight
            nn.init.zeros_(self.controlnet_down_blocks[-1].bias)  # zero module bias

        self.controlnet_mid_block = nn.Conv2d(1280, 1280, kernel_size=1)
        nn.init.zeros_(self.controlnet_mid_block.weight)  # zero module weight
        nn.init.zeros_(self.controlnet_mid_block.bias)  # zero module bias

        self.controlnet_alpha_heads = nn.ModuleDict()
        for index in self.alpha_head_indices:
            dim = dims[index]
            self.controlnet_alpha_heads[str(index)] = nn.Conv2d(dim, 1, kernel_size=1)

        self.controlnet_alpha_baseline_heads = nn.ModuleDict()
        if self.alpha_baseline_mode in {"terrain_mask", "both"}:
            self.controlnet_alpha_baseline_heads["terrain_mask"] = nn.Conv2d(1, 1, kernel_size=1)
        if self.alpha_baseline_mode in {"pre_stem", "both"}:
            self.controlnet_alpha_baseline_heads["pre_stem"] = nn.Conv2d(320, 1, kernel_size=1)

    def init_from_unet(self, unet: sdxl_original_unet.SdxlUNet2DConditionModel):
        unet_sd = unet.state_dict()
        unet_sd = {k: v for k, v in unet_sd.items() if not k.startswith("out")}
        sd = super().state_dict()
        sd.update(unet_sd)
        info = super().load_state_dict(sd, strict=True, assign=True)
        return info

    def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = True) -> Any:
        # convert state_dict to SAI format
        unet_sd = {}
        for k in list(state_dict.keys()):
            if not k.startswith("controlnet_"):
                unet_sd[k] = state_dict.pop(k)
        unet_sd = convert_diffusers_unet_state_dict_to_sdxl(unet_sd)
        state_dict.update(unet_sd)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # convert state_dict to Diffusers format
        state_dict = super().state_dict(destination, prefix, keep_vars)
        control_net_sd = {}
        for k in list(state_dict.keys()):
            if k.startswith("controlnet_"):
                control_net_sd[k] = state_dict.pop(k)
        state_dict = convert_sdxl_unet_state_dict_to_diffusers(state_dict)
        state_dict.update(control_net_sd)
        return state_dict

    def forward(
        self,
        x: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        cond_image: Optional[torch.Tensor] = None,
        return_alpha: bool = False,
        return_diagnostics: bool = False,
        alpha_target_size: Optional[tuple[int, int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # broadcast timesteps to batch dimension
        timesteps = timesteps.expand(x.shape[0])

        t_emb = sdxl_original_unet.get_timestep_embedding(timesteps, self.model_channels, downscale_freq_shift=0)
        t_emb = t_emb.to(x.dtype)
        emb = self.time_embed(t_emb)

        assert x.shape[0] == y.shape[0], f"batch size mismatch: {x.shape[0]} != {y.shape[0]}"
        assert x.dtype == y.dtype, f"dtype mismatch: {x.dtype} != {y.dtype}"
        emb = emb + self.label_emb(y)

        def call_module(module, h, emb, context):
            x = h
            for layer in module:
                if isinstance(layer, sdxl_original_unet.ResnetBlock2D):
                    x = layer(x, emb)
                elif isinstance(layer, sdxl_original_unet.Transformer2DModel):
                    x = layer(x, context)
                else:
                    x = layer(x)
            return x

        h = x
        multiplier = self.multiplier if self.multiplier is not None else 1.0
        hs = []
        alpha_per_scale = []
        residual_norms = []
        cond_embedding_norm = None
        cond_embedding = None
        for i, module in enumerate(self.input_blocks):
            h = call_module(module, h, emb, context)
            if i == 0:
                cond_embedding = self.controlnet_cond_embedding(cond_image)
                cond_embedding_norm = float(cond_embedding.detach().float().norm().item())
                h = cond_embedding + h
            control_residual = self.controlnet_down_blocks[i](h) * multiplier
            hs.append(control_residual)
            if return_diagnostics:
                residual_norms.append(float(control_residual.detach().float().norm().item()))
            if return_alpha and str(i) in self.controlnet_alpha_heads:
                alpha_per_scale.append((i, self.controlnet_alpha_heads[str(i)](control_residual)))

        h = call_module(self.middle_block, h, emb, context)
        h = self.controlnet_mid_block(h) * multiplier

        if not return_alpha:
            if return_diagnostics:
                return hs, h, {
                    "multiplier": float(multiplier),
                    "cond_embedding_norm": cond_embedding_norm,
                    "down_block_residual_norms": residual_norms,
                    "mid_block_residual_norm": float(h.detach().float().norm().item()),
                }
            return hs, h

        fused_alpha_logits = None
        if alpha_per_scale:
            if alpha_target_size is None:
                raise ValueError("alpha_target_size is required when return_alpha=True")
            resized_logits = [
                F.interpolate(alpha_logits, size=alpha_target_size, mode="bilinear", align_corners=False)
                for _, alpha_logits in alpha_per_scale
            ]
            fused_alpha_logits = torch.stack(resized_logits, dim=0).mean(dim=0)

        baseline_logits = {}
        if alpha_target_size is not None and self.controlnet_alpha_baseline_heads:
            if "terrain_mask" in self.controlnet_alpha_baseline_heads:
                t_idx = self.alpha_baseline_terrain_channel_index
                if t_idx < 0 or t_idx >= cond_image.shape[1]:
                    raise ValueError(
                        f"terrain channel index {t_idx} out of range for cond_image channels={cond_image.shape[1]}"
                    )
                terrain_channel = cond_image[:, t_idx : t_idx + 1].contiguous()
                terrain_logits = self.controlnet_alpha_baseline_heads["terrain_mask"](terrain_channel)
                if terrain_logits.shape[-2:] != alpha_target_size:
                    terrain_logits = F.interpolate(
                        terrain_logits,
                        size=alpha_target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                baseline_logits["terrain_mask"] = terrain_logits
            if "pre_stem" in self.controlnet_alpha_baseline_heads:
                if cond_embedding is None:
                    raise RuntimeError("missing cond_embedding while computing pre_stem alpha baseline")
                pre_stem_logits = self.controlnet_alpha_baseline_heads["pre_stem"](cond_embedding)
                if pre_stem_logits.shape[-2:] != alpha_target_size:
                    pre_stem_logits = F.interpolate(
                        pre_stem_logits,
                        size=alpha_target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                baseline_logits["pre_stem"] = pre_stem_logits

        alpha_payload = {
            "per_scale_logits": alpha_per_scale,
            "fused_logits": fused_alpha_logits,
            "baseline_logits": baseline_logits,
        }
        if return_diagnostics:
            alpha_payload["diagnostics"] = {
                "multiplier": float(multiplier),
                "cond_embedding_norm": cond_embedding_norm,
                "down_block_residual_norms": residual_norms,
                "mid_block_residual_norm": float(h.detach().float().norm().item()),
            }

        return hs, h, alpha_payload


class SdxlControlledUNet(sdxl_original_unet.SdxlUNet2DConditionModel):
    """
    This class is for training purpose only.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, timesteps=None, context=None, y=None, input_resi_add=None, mid_add=None, **kwargs):
        # broadcast timesteps to batch dimension
        timesteps = timesteps.expand(x.shape[0])

        hs = []
        t_emb = sdxl_original_unet.get_timestep_embedding(timesteps, self.model_channels, downscale_freq_shift=0)
        t_emb = t_emb.to(x.dtype)
        emb = self.time_embed(t_emb)

        assert x.shape[0] == y.shape[0], f"batch size mismatch: {x.shape[0]} != {y.shape[0]}"
        assert x.dtype == y.dtype, f"dtype mismatch: {x.dtype} != {y.dtype}"
        emb = emb + self.label_emb(y)

        def call_module(module, h, emb, context):
            x = h
            for layer in module:
                if isinstance(layer, sdxl_original_unet.ResnetBlock2D):
                    x = layer(x, emb)
                elif isinstance(layer, sdxl_original_unet.Transformer2DModel):
                    x = layer(x, context)
                else:
                    x = layer(x)
            return x

        h = x
        for module in self.input_blocks:
            h = call_module(module, h, emb, context)
            hs.append(h)

        h = call_module(self.middle_block, h, emb, context)
        h = h + mid_add

        for module in self.output_blocks:
            resi = hs.pop() + input_resi_add.pop()
            h = torch.cat([h, resi], dim=1)
            h = call_module(module, h, emb, context)

        h = h.type(x.dtype)
        h = call_module(self.out, h, emb, context)

        return h


if __name__ == "__main__":
    import time

    logger.info("create unet")
    unet = SdxlControlledUNet()
    unet.to("cuda", torch.bfloat16)
    unet.set_use_sdpa(True)
    unet.set_gradient_checkpointing(True)
    unet.train()

    logger.info("create control_net")
    control_net = SdxlControlNet()
    control_net.to("cuda")
    control_net.set_use_sdpa(True)
    control_net.set_gradient_checkpointing(True)
    control_net.train()

    logger.info("Initialize control_net from unet")
    control_net.init_from_unet(unet)

    unet.requires_grad_(False)
    control_net.requires_grad_(True)

    # 使用メモリ量確認用の疑似学習ループ
    logger.info("preparing optimizer")

    # optimizer = torch.optim.SGD(unet.parameters(), lr=1e-3, nesterov=True, momentum=0.9) # not working

    import bitsandbytes

    optimizer = bitsandbytes.adam.Adam8bit(control_net.parameters(), lr=1e-3)  # not working
    # optimizer = bitsandbytes.optim.RMSprop8bit(unet.parameters(), lr=1e-3)  # working at 23.5 GB with torch2
    # optimizer=bitsandbytes.optim.Adagrad8bit(unet.parameters(), lr=1e-3)  # working at 23.5 GB with torch2

    # import transformers
    # optimizer = transformers.optimization.Adafactor(unet.parameters(), relative_step=True)  # working at 22.2GB with torch2

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    logger.info("start training")
    steps = 10
    batch_size = 1

    for step in range(steps):
        logger.info(f"step {step}")
        if step == 1:
            time_start = time.perf_counter()

        x = torch.randn(batch_size, 4, 128, 128).cuda()  # 1024x1024
        t = torch.randint(low=0, high=1000, size=(batch_size,), device="cuda")
        txt = torch.randn(batch_size, 77, 2048).cuda()
        vector = torch.randn(batch_size, sdxl_original_unet.ADM_IN_CHANNELS).cuda()
        cond_img = torch.rand(batch_size, 3, 1024, 1024).cuda()

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            input_resi_add, mid_add = control_net(x, t, txt, vector, cond_img)
            output = unet(x, t, txt, vector, input_resi_add, mid_add)
            target = torch.randn_like(output)
            loss = torch.nn.functional.mse_loss(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    time_end = time.perf_counter()
    logger.info(f"elapsed time: {time_end - time_start} [sec] for last {steps - 1} steps")

    logger.info("finish training")
    sd = control_net.state_dict()

    from safetensors.torch import save_file

    save_file(sd, r"E:\Work\SD\Tmp\sdxl\ctrl\control_net.safetensors")
