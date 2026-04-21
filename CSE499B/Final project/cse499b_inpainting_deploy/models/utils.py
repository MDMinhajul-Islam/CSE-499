import os
import time
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw


def timer() -> float:
    return time.perf_counter()


def elapsed_seconds(start_time: float) -> float:
    return time.perf_counter() - start_time


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# def resize_image_and_mask(
#     image: Image.Image,
#     mask: Image.Image,
#     target_size: int,
# ) -> Tuple[Image.Image, Image.Image]:
#     image = image.convert("RGB").resize((target_size, target_size))
#     mask = mask.convert("L").resize((target_size, target_size))
#     return image, mask

def resize_image_and_mask(
    image: Image.Image,
    mask: Image.Image,
    target_size: int,
) -> Tuple[Image.Image, Image.Image]:
    image = image.convert("RGB").resize((target_size, target_size), Image.BICUBIC)
    mask = mask.convert("L").resize((target_size, target_size), Image.NEAREST)
    return image, mask


def ensure_binary_mask(mask: Image.Image, threshold: int = 127) -> Image.Image:
    arr = np.array(mask.convert("L"))
    arr = (arr > threshold).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def make_masked_preview(image: Image.Image, mask: Image.Image) -> Image.Image:
    image_arr = np.array(image.convert("RGB")).copy()
    mask_arr = np.array(mask.convert("L")) > 0

    preview_arr = image_arr.copy()
    preview_arr[mask_arr] = [255, 255, 255]

    return Image.fromarray(preview_arr)


def overlay_mask_on_image(image: Image.Image, mask: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    mask_arr = np.array(mask.convert("L"))

    alpha = (mask_arr > 0).astype(np.uint8) * 120

    rgba = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = alpha

    overlay = Image.fromarray(rgba, mode="RGBA")
    return Image.alpha_composite(image, overlay).convert("RGB")


# def save_comparison_panel(
#     save_path: str,
#     original: Image.Image,
#     masked: Image.Image,
#     output_1: Image.Image,
#     output_2: Image.Image,
#     title_1: str,
#     title_2: str,
# ):
#     original = original.convert("RGB")
#     masked = masked.convert("RGB")
#     output_1 = output_1.convert("RGB")
#     output_2 = output_2.convert("RGB")

#     w, h = original.size
#     header_h = 34
#     pad = 10

#     total_w = (w * 4) + (pad * 5)
#     total_h = h + header_h + (pad * 2)

#     canvas = Image.new("RGB", (total_w, total_h), color=(245, 245, 245))
#     draw = ImageDraw.Draw(canvas)

#     panels = [
#         ("Original", original),
#         ("Masked Preview", masked),
#         (title_1, output_1),
#         (title_2, output_2),
#     ]

#     x = pad
#     for title, img in panels:
#         draw.text((x, 8), title, fill=(0, 0, 0))
#         canvas.paste(img, (x, header_h))
#         x += w + pad

#     save_dir = os.path.dirname(save_path)
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)

#     canvas.save(save_path)

def save_comparison_panel(
    save_path: str,
    original: Image.Image,
    masked: Image.Image,
    output_1: Image.Image,
    output_2: Image.Image,
    title_1: str,
    title_2: str,
):
    original = original.convert("RGB")
    w, h = original.size

    masked = masked.convert("RGB").resize((w, h))
    output_1 = output_1.convert("RGB").resize((w, h))
    output_2 = output_2.convert("RGB").resize((w, h))

    header_h = 34
    pad = 10

    total_w = (w * 4) + (pad * 5)
    total_h = h + header_h + (pad * 2)

    canvas = Image.new("RGB", (total_w, total_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    panels = [
        ("Original", original),
        ("Masked Preview", masked),
        (title_1, output_1),
        (title_2, output_2),
    ]

    x = pad
    for title, img in panels:
        draw.text((x, 8), title, fill=(0, 0, 0))
        canvas.paste(img, (x, header_h))
        x += w + pad

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    canvas.save(save_path)