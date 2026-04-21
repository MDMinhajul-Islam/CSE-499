from typing import Dict, Any
import torch
from PIL import Image

from diffusers import StableDiffusionInpaintPipeline

from models.model_registry import get_model_config, select_benchmark_for_model


_LOADED_SD2_MODELS: Dict[str, StableDiffusionInpaintPipeline] = {}


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32


def load_sd2_model(model_name: str):
    cfg = get_model_config(model_name)
    hf_model_id = cfg.get("hf_model_id")

    if model_name in _LOADED_SD2_MODELS:
        return _LOADED_SD2_MODELS[model_name]

    device = _get_device()
    dtype = _get_dtype()

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        hf_model_id,
        torch_dtype=dtype,
    )

    pipe = pipe.to(device)

    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    _LOADED_SD2_MODELS[model_name] = pipe
    return pipe


def _prepare_sd2_inputs(image: Image.Image, mask: Image.Image):
    image = image.convert("RGB")
    mask = mask.convert("L")

    # SD inpainting expects white = editable region, black = keep region
    # Your app already uses binary white masks, so this is fine.
    return image, mask


def run_sd2_inference(
    model_name: str,
    image: Image.Image,
    mask: Image.Image,
    steps: int = 50,
    seed: int = 42,
) -> tuple[Image.Image, Dict[str, Any]]:
    cfg = get_model_config(model_name)
    benchmark = select_benchmark_for_model(model_name, steps)

    pipe = load_sd2_model(model_name)
    device = _get_device()

    image, mask = _prepare_sd2_inputs(image, mask)

    generator = torch.Generator(device=device).manual_seed(int(seed))

    result = pipe(
        prompt="high quality realistic face inpainting",
        image=image,
        mask_image=mask,
        num_inference_steps=int(steps),
        guidance_scale=4.0,
        generator=generator,
    )

    output = result.images[0]

    info = {
        "model_name": cfg["display_name"],
        "model_type": cfg["type"],
        "source": cfg.get("source"),
        "dataset": cfg.get("dataset"),
        "hf_model_id": cfg.get("hf_model_id"),
        "params": cfg.get("params"),
        "benchmark_label": benchmark["label"] if benchmark else "N/A",
        "offline_metrics": benchmark if benchmark else {},
        "status": f"real pretrained SD2 inference active on {device}",
    }

    return output, info