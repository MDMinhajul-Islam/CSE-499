import os
import time
from datetime import datetime
from typing import Optional, Any

import gradio as gr
from PIL import Image
import numpy as np

from models.model_registry import list_model_names, get_model_config
from models.student_infer import run_student_inference
from models.sd2_infer import run_sd2_inference
from models.utils import (
    ensure_binary_mask,
    make_masked_preview,
    resize_image_and_mask,
    save_comparison_panel,
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CUSTOM_CSS = """
.gradio-container {
    max-width: 1500px !important;
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%) !important;
}

.main-title {
    text-align: center;
    margin-bottom: 6px;
    font-size: 2.1rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: 0.3px;
}

.sub-title {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 10px;
    font-size: 1rem;
}

.app-subtext {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 18px;
    font-size: 0.96rem;
}

.gr-button-primary {
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
    border: none !important;
    color: white !important;
}

.gr-button-secondary {
    background: linear-gradient(90deg, #475569, #334155) !important;
    border: none !important;
    color: white !important;
}

textarea, .scroll-hide {
    font-size: 0.96rem !important;
    line-height: 1.45 !important;
}

.run-summary {
    background: linear-gradient(90deg, rgba(37,99,235,0.16), rgba(124,58,237,0.16));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 10px 14px;
    color: #e5e7eb;
    margin-top: 2px;
    margin-bottom: 8px;
    font-size: 0.96rem;
}

.live-run-heading {
    margin-bottom: 2px !important;
}

.live-row {
    margin-top: 0px !important;
    margin-bottom: 8px !important;
}

.live-row .gr-box,
.live-row .gr-group {
    min-height: 280px !important;
}

.offline-accordion {
    margin-top: 4px !important;
}

footer {
    visibility: hidden !important;
}
"""


def _extract_image_and_mask(editor_value: Any):
    if editor_value is None:
        return None, None

    if isinstance(editor_value, dict):
        background = editor_value.get("background")
        layers = editor_value.get("layers", [])
        composite = editor_value.get("composite")

        image = background or composite
        mask = layers[-1] if layers else None
        return image, mask

    if isinstance(editor_value, Image.Image):
        return editor_value, None

    return None, None


def _fmt(value, digits: int = 4):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _compute_mask_area_percent(mask: Image.Image) -> float:
    arr = np.array(mask.convert("L")) > 0
    return float(arr.mean() * 100.0)


def run_model_by_name(
    model_name: str,
    image: Image.Image,
    mask: Image.Image,
    seed: int,
    steps: int,
):
    cfg = get_model_config(model_name)
    model_type = cfg["type"]

    start_time = time.time()

    if model_type == "student":
        output, meta = run_student_inference(
            model_name=model_name,
            image=image,
            mask=mask,
            steps=steps,
            seed=seed,
        )
    elif model_type == "pretrained":
        output, meta = run_sd2_inference(
            model_name=model_name,
            image=image,
            mask=mask,
            steps=steps,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    meta["live_inference_time_sec"] = round(time.time() - start_time, 4)
    return output, meta


def build_run_summary_html(
    target_size: int,
    steps: int,
    seed: int,
    mask_area_percent: float,
    compare_path: str,
) -> str:
    return f"""
    <div class="run-summary">
        <b>Run Settings:</b> {target_size} × {target_size}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Steps:</b> {steps}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Seed:</b> {int(seed)}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Mask Area:</b> {_fmt(mask_area_percent, 2)}%
        <br><br>
        <b>Saved Panel:</b> {compare_path}
    </div>
    """


def build_model_live_info(model_name: str, meta: dict) -> str:
    lines = [
        f"Model: {model_name}",
        f"Type: {meta.get('model_type', 'N/A')}",
        f"Source: {meta.get('source', 'N/A')}",
        f"Params: {meta.get('params', 'N/A')}",   # ✅ ADD THIS
        f"Live Inference Time: {_fmt(meta.get('live_inference_time_sec'))} sec",
        f"Status: {meta.get('status', 'N/A')}",
    ]
    return "\n".join(lines)


def _format_offline_block(title: str, meta: dict) -> str:
    offline = meta.get("offline_metrics", {}) or {}

    lines = [
        title,
        f"Type: {meta.get('model_type', 'N/A')}",
        f"Source: {meta.get('source', 'N/A')}",
        f"Dataset: {meta.get('dataset', 'N/A')}",
        f"Params: {meta.get('params', 'N/A')}",
        f"Benchmark Setup: {meta.get('benchmark_label', 'N/A')}",
    ]

    if meta.get("checkpoint_path"):
        lines.append(f"Checkpoint: {meta.get('checkpoint_path')}")
    if meta.get("hf_model_id"):
        lines.append(f"HF Model ID: {meta.get('hf_model_id')}")

    lines.extend([
        f"FID: {_fmt(offline.get('fid'))}",
        f"PSNR_masked: {_fmt(offline.get('psnr_masked'))}",
        f"SSIM_masked: {_fmt(offline.get('ssim_masked'))}",
        f"LPIPS_masked: {_fmt(offline.get('lpips_masked'))}",
        f"Avg Inference Time: {_fmt(offline.get('avg_inference_time'))}",
        f"Benchmark Sampling Steps: {_fmt(offline.get('sampling_steps'), 0)}",
        f"Test Samples: {_fmt(offline.get('num_test_samples'), 0)}",
    ])
    return "\n".join(lines)


def build_offline_info_text(model_1_name: str, model_2_name: str, meta_1: dict, meta_2: dict) -> str:
    intro = "Offline benchmark results come from saved CelebA Dataset evaluations."
    block_1 = _format_offline_block(f"Model 1 Offline Benchmark: {model_1_name}", meta_1)
    block_2 = _format_offline_block(f"Model 2 Offline Benchmark: {model_2_name}", meta_2)
    return f"{intro}\n\n{block_1}\n\n{block_2}"


def run_compare(
    editor_value,
    uploaded_mask: Optional[Image.Image],
    model_1_name: str,
    model_2_name: str,
    target_size: int,
    seed: int,
    steps: int,
):
    image, drawn_mask = _extract_image_and_mask(editor_value)

    if image is None:
        raise gr.Error("Please upload an input image first.")

    mask = uploaded_mask if uploaded_mask is not None else drawn_mask
    if mask is None:
        raise gr.Error("Please draw or upload a mask first.")

    image, mask = resize_image_and_mask(image, mask, target_size)
    mask = ensure_binary_mask(mask)
    masked_preview = make_masked_preview(image, mask)

    mask_area_percent = _compute_mask_area_percent(mask)

    result_1, meta_1 = run_model_by_name(model_1_name, image, mask, seed, steps)
    result_2, meta_2 = run_model_by_name(model_2_name, image, mask, seed, steps)

    # force same display size for both outputs
    target_wh = image.size
    result_1 = result_1.convert("RGB").resize(target_wh)
    result_2 = result_2.convert("RGB").resize(target_wh)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    compare_path = os.path.join(OUTPUT_DIR, f"compare_{timestamp}.png")

    save_comparison_panel(
        save_path=compare_path,
        original=image,
        masked=masked_preview,
        output_1=result_1,
        output_2=result_2,
        title_1=model_1_name,
        title_2=model_2_name,
    )

    run_summary_html = build_run_summary_html(
        target_size=target_size,
        steps=steps,
        seed=seed,
        mask_area_percent=mask_area_percent,
        compare_path=compare_path,
    )

    model_1_live = build_model_live_info(model_1_name, meta_1)
    model_2_live = build_model_live_info(model_2_name, meta_2)

    offline_info = build_offline_info_text(
        model_1_name=model_1_name,
        model_2_name=model_2_name,
        meta_1=meta_1,
        meta_2=meta_2,
    )

    return (
        image,
        mask,
        masked_preview,
        result_1,
        result_2,
        run_summary_html,
        model_1_live,
        model_2_live,
        offline_info,
    )


def clear_all():
    return None, None, None, None, None, None, None, "", "", "", ""


def build_ui():
    model_names = list_model_names()

    default_model_1 = "student_a02" if "student_a02" in model_names else (model_names[0] if len(model_names) > 0 else None)
    default_model_2 = "sd2_baseline" if "sd2_baseline" in model_names else (model_names[1] if len(model_names) > 1 else default_model_1)

    with gr.Blocks(title="CSE499B Capstone Inpainting Demo") as demo:
        gr.HTML("""
            <div class="main-title">Lightweight Diffusion Image Inpainting Comparator</div>
            <div class="sub-title">CSE499B Capstone Project Showcase</div>
            <div class="app-subtext">Compare our lightweight student model with the pretrained SD2 baseline using the same input image and mask.</div>
        """)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=360):
                gr.Markdown("### Input and Controls")

                image_editor = gr.ImageEditor(
                    type="pil",
                    label="Upload Image and Draw Mask",
                    brush=gr.Brush(colors=["#FFFFFF"], default_size=18),
                )

                uploaded_mask = gr.Image(
                    type="pil",
                    label="Optional Uploaded Binary Mask (white = inpaint area)",
                )

                model_1 = gr.Dropdown(
                    choices=model_names,
                    value=default_model_1,
                    label="Model 1",
                )

                model_2 = gr.Dropdown(
                    choices=model_names,
                    value=default_model_2,
                    label="Model 2",
                )

                target_size = gr.Dropdown(
                    choices=[256, 512],
                    value=256,
                    label="Target Size",
                )

                seed = gr.Number(
                    value=42,
                    precision=0,
                    label="Seed",
                )

                steps = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Inference Steps",
                )

                with gr.Row():
                    run_btn = gr.Button("Run Comparison", variant="primary")
                    clear_btn = gr.Button("Clear All", variant="secondary")

            with gr.Column(scale=2, min_width=780):
                gr.Markdown("### Visual Comparison")

                with gr.Row():
                    out_original = gr.Image(type="pil", label="Original")
                    out_mask = gr.Image(type="pil", label="Binary Mask")

                out_masked = gr.Image(type="pil", label="Masked Preview")

                with gr.Row():
                    out_1 = gr.Image(type="pil", label="Model 1 Output")
                    out_2 = gr.Image(type="pil", label="Model 2 Output")

                gr.Markdown("### Live Run Info", elem_classes=["live-run-heading"])
                run_summary_html = gr.HTML(
    value="""
    <div class="run-summary">
        Run settings will appear here after you click <b>Run Comparison</b>.
    </div>
    """
)

                with gr.Row(elem_classes=["live-row"]):
                    model_1_live_box = gr.Textbox(
                        label="Model 1 Live Info",
                        lines=5.5,
                    )
                    model_2_live_box = gr.Textbox(
                        label="Model 2 Live Info",
                        lines=5.5,
                    )

                with gr.Accordion("Show Offline Benchmark Details",open=False, elem_classes=["offline-accordion"]):
                    offline_info_box = gr.Textbox(
                        label="Offline Benchmark Info",
                        lines=18,
                    )

        run_btn.click(
            fn=run_compare,
            inputs=[image_editor, uploaded_mask, model_1, model_2, target_size, seed, steps],
            outputs=[
                out_original,
                out_mask,
                out_masked,
                out_1,
                out_2,
                run_summary_html,
                model_1_live_box,
                model_2_live_box,
                offline_info_box,
            ],
        )

        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                image_editor,
                uploaded_mask,
                out_original,
                out_mask,
                out_masked,
                out_1,
                out_2,
                run_summary_html,
                model_1_live_box,
                model_2_live_box,
                offline_info_box,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(css=CUSTOM_CSS, share=True)