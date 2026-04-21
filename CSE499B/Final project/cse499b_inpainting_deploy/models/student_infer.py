# from typing import Dict, Any
# import numpy as np
# from PIL import Image, ImageFilter, ImageDraw

# from models.model_registry import get_model_config, select_benchmark_for_model


# class DummyStudentModel:
#     def __init__(self, checkpoint_path: str | None = None):
#         self.checkpoint_path = checkpoint_path

#     def infer(
#         self,
#         image: Image.Image,
#         mask: Image.Image,
#         steps: int = 50,
#         seed: int = 42,
#     ) -> Image.Image:
#         image = image.convert("RGB")
#         mask = mask.convert("L")

#         blurred = image.copy().filter(ImageFilter.GaussianBlur(radius=2))

#         image_np = np.array(image)
#         blur_np = np.array(blurred)
#         mask_np = np.array(mask)

#         out_np = image_np.copy()
#         region = mask_np > 127
#         out_np[region] = (0.65 * blur_np[region] + 0.35 * image_np[region]).astype(np.uint8)

#         out = Image.fromarray(out_np)
#         draw = ImageDraw.Draw(out)
#         draw.text((10, 10), f"A02 Placeholder | {steps} steps", fill=(255, 255, 255))

#         return out


# _LOADED_STUDENT_MODELS: Dict[str, DummyStudentModel] = {}


# def load_student_model(model_name: str):
#     cfg = get_model_config(model_name)
#     checkpoint_path = cfg.get("checkpoint_path")

#     if model_name not in _LOADED_STUDENT_MODELS:
#         _LOADED_STUDENT_MODELS[model_name] = DummyStudentModel(checkpoint_path=checkpoint_path)

#     return _LOADED_STUDENT_MODELS[model_name]


# def run_student_inference(
#     model_name: str,
#     image: Image.Image,
#     mask: Image.Image,
#     steps: int = 50,
#     seed: int = 42,
# ) -> tuple[Image.Image, Dict[str, Any]]:
#     cfg = get_model_config(model_name)
#     benchmark = select_benchmark_for_model(model_name, steps)
#     model = load_student_model(model_name)

#     output = model.infer(image=image, mask=mask, steps=steps, seed=seed)

#     info = {
#         "model_name": cfg["display_name"],
#         "model_type": cfg["type"],
#         "source": cfg.get("source"),
#         "dataset": cfg.get("dataset"),
#         "checkpoint_path": cfg.get("checkpoint_path"),
#         "params": cfg.get("params"),
#         "benchmark_label": benchmark["label"] if benchmark else "N/A",
#         "offline_metrics": benchmark if benchmark else {},
#         "status": "placeholder student inference active",
#     }

#     return output, info





from typing import Dict, Any
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from diffusers import AutoencoderKL, DDPMScheduler

from models.model_registry import get_model_config, select_benchmark_for_model


_LOADED_STUDENT_MODELS: Dict[str, "StudentA02Runtime"] = {}


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32


# -----------------------------
# Utilities
# -----------------------------
def pil_to_tensor_01(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    x = torch.from_numpy(
        __import__("numpy").array(image).astype("float32") / 255.0
    ).permute(2, 0, 1)
    return x


def pil_to_tensor_neg1_1(image: Image.Image) -> torch.Tensor:
    x = pil_to_tensor_01(image)
    return x * 2.0 - 1.0


def tensor_neg1_1_to_01(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)


def tensor_01_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(0, 1)
    x = (x * 255.0).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def pil_mask_to_tensor(mask_pil: Image.Image) -> torch.Tensor:
    import numpy as np
    mask_np = np.array(mask_pil.convert("L"), dtype=np.float32) / 255.0
    mask_np = (mask_np > 0.5).astype("float32")
    return torch.from_numpy(mask_np).unsqueeze(0)


def apply_mask_to_image(image_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
    import numpy as np
    image_np = np.array(image_pil.convert("RGB")).astype("uint8")
    mask_np = np.array(mask_pil.convert("L")) > 127
    masked_np = image_np.copy()
    masked_np[mask_np] = 0
    return Image.fromarray(masked_np)


# -----------------------------
# Student A02 architecture
# Must match notebook exactly
# -----------------------------
def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.norm1(h)
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, time_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        x = self.res(x, t_emb)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res = ResBlock(out_ch + skip_ch, out_ch, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        return x


class StudentUNet(nn.Module):
    def __init__(self, in_channels=9, out_channels=4, base_channels=64, time_dim=256):
        super().__init__()
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down1 = DownBlock(base_channels, base_channels * 2, time_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_dim)

        self.mid1 = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, time_dim)

        self.up2 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, time_dim)
        self.up1 = UpBlock(base_channels * 2, base_channels * 2, base_channels, time_dim)

        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_act = nn.SiLU()
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(timesteps, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x0 = self.conv_in(x)

        x1, skip1 = self.down1(x0, t_emb)
        x2, skip2 = self.down2(x1, t_emb)

        x = self.mid1(x2, t_emb)
        x = self.mid2(x, t_emb)

        x = self.up2(x, skip2, t_emb)
        x = self.up1(x, skip1, t_emb)

        x = self.out_norm(x)
        x = self.out_act(x)
        x = self.conv_out(x)
        return x


# -----------------------------
# Runtime wrapper
# -----------------------------
class StudentA02Runtime:
    def __init__(self, checkpoint_path: str, hf_vae_model_id: str | None = None):
        self.checkpoint_path = checkpoint_path
        self.device = _get_device()
        self.dtype = _get_dtype()

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg = ckpt.get("cfg", {})

        student_cfg = cfg.get("student", {})
        in_channels = student_cfg.get("in_channels", 9)
        out_channels = student_cfg.get("out_channels", 4)
        base_channels = student_cfg.get("base_channels", 64)

        self.model = StudentUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels
        ).to(self.device).float()

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Load VAE from SD2 source
        vae_source = hf_vae_model_id or "sd2-community/stable-diffusion-2-inpainting"
        self.vae = AutoencoderKL.from_pretrained(
            vae_source,
            subfolder="vae",
            torch_dtype=self.dtype
        ).to(self.device)
        self.vae.eval()

        # Scheduler for reverse diffusion
        self.scheduler = DDPMScheduler.from_pretrained(
            vae_source,
            subfolder="scheduler"
        )

    @torch.no_grad()
    def encode_image_to_latent(self, image_tensor: torch.Tensor) -> torch.Tensor:
        vae_dtype = next(self.vae.parameters()).dtype
        image_tensor = image_tensor.to(device=self.device, dtype=vae_dtype)
        latent = self.vae.encode(image_tensor).latent_dist.sample()
        latent = latent * 0.18215
        return latent

    def prepare_input(
        self,
        latents_noisy: torch.Tensor,
        mask: torch.Tensor,
        masked_latent: torch.Tensor
    ) -> torch.Tensor:
        mask_latent = F.interpolate(mask, size=latents_noisy.shape[-2:], mode="nearest")
        return torch.cat([latents_noisy, mask_latent, masked_latent], dim=1)

    @torch.no_grad()
    def infer(
        self,
        image: Image.Image,
        mask: Image.Image,
        steps: int = 100,
        seed: int = 42,
    ) -> Image.Image:
        image = image.convert("RGB")
        mask = mask.convert("L")

        image_t = pil_to_tensor_neg1_1(image).unsqueeze(0).to(self.device).float()
        mask_t = pil_mask_to_tensor(mask).unsqueeze(0).to(self.device).float()

        masked_pil = apply_mask_to_image(image, mask)
        masked_t = pil_to_tensor_neg1_1(masked_pil).unsqueeze(0).to(self.device).float()

        latents = self.encode_image_to_latent(image_t).float()
        masked_latent = self.encode_image_to_latent(masked_t).float()

        self.scheduler.set_timesteps(int(steps), device=self.device)

        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        latents = torch.randn(
            latents.shape,
            generator=generator,
            device=self.device,
            dtype=torch.float32
        )

        for t in self.scheduler.timesteps:
            t_batch = torch.full((latents.size(0),), int(t), device=self.device, dtype=torch.long)
            model_input = self.prepare_input(latents, mask_t, masked_latent).float()
            noise_pred = self.model(model_input, t_batch).float()
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample.float()

        vae_dtype = next(self.vae.parameters()).dtype
        decoded = self.vae.decode((latents / 0.18215).to(dtype=vae_dtype)).sample.float()

        pred_01 = tensor_neg1_1_to_01(decoded[0])
        pred_pil = tensor_01_to_pil(pred_01)
        return pred_pil


# -----------------------------
# Public API
# -----------------------------
def load_student_model(model_name: str):
    cfg = get_model_config(model_name)
    checkpoint_path = cfg.get("checkpoint_path")
    hf_vae_model_id = cfg.get("hf_vae_model_id", "sd2-community/stable-diffusion-2-inpainting")

    if not checkpoint_path:
        raise ValueError(f"No checkpoint_path configured for model '{model_name}'")

    if model_name not in _LOADED_STUDENT_MODELS:
        _LOADED_STUDENT_MODELS[model_name] = StudentA02Runtime(
            checkpoint_path=checkpoint_path,
            hf_vae_model_id=hf_vae_model_id,
        )

    return _LOADED_STUDENT_MODELS[model_name]


def run_student_inference(
    model_name: str,
    image: Image.Image,
    mask: Image.Image,
    steps: int = 100,
    seed: int = 42,
) -> tuple[Image.Image, Dict[str, Any]]:
    cfg = get_model_config(model_name)
    benchmark = select_benchmark_for_model(model_name, steps)
    model = load_student_model(model_name)

    output = model.infer(image=image, mask=mask, steps=steps, seed=seed)

    info = {
        "model_name": cfg["display_name"],
        "model_type": cfg["type"],
        "source": cfg.get("source"),
        "dataset": cfg.get("dataset"),
        "checkpoint_path": cfg.get("checkpoint_path"),
        "params": cfg.get("params"),
        "benchmark_label": benchmark["label"] if benchmark else "N/A",
        "offline_metrics": benchmark if benchmark else {},
        "status": f"real Student A02 inference active on {_get_device()}",
    }

    return output, info