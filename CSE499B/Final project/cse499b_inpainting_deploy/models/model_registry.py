# # from pathlib import Path

# # PROJECT_ROOT = Path(__file__).resolve().parent.parent
# # CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# # MODEL_REGISTRY = {
# #     "A02 Best": {
# #         "type": "student",
# #         "source": "local checkpoint",
# #         "display_name": "A02 Best",
# #         "checkpoint_path": str(CHECKPOINT_DIR / "student" / "studentA02_best.pt"),
# #         "dataset": "CelebA subset",
# #         "params": "6.70M",
# #         "benchmarks": {
# #             "steps_50": {
# #                 "label": "CelebA subset | 256 | 50 steps | 500 test samples",
# #                 "fid": 49.82220458984375,
# #                 "psnr_masked": 18.537870947179968,
# #                 "ssim_masked": 0.8430781364440918,
# #                 "lpips_masked": 0.11506782172992826,
# #                 "avg_inference_time": 0.3230763897895813,
# #                 "sampling_steps": 50,
# #                 "num_test_samples": 500,
# #             },
# #             "steps_100": {
# #                 "label": "CelebA subset | 256 | 100 steps | 500 test samples",
# #                 "fid": 46.94912338256836,
# #                 "psnr_masked": 21.22296338381309,
# #                 "ssim_masked": 0.8714379668235779,
# #                 "lpips_masked": 0.08842972988449037,
# #                 "avg_inference_time": 0.6370953459739686,
# #                 "sampling_steps": 100,
# #                 "num_test_samples": 500,
# #             },
# #         },
# #     },
# #     "SD2 Inpainting": {
# #         "type": "pretrained",
# #         "source": "pretrained",
# #         "display_name": "Stable Diffusion 2 Inpainting",
# #         "hf_model_id": "sd2-community/stable-diffusion-2-inpainting",
# #         "dataset": "CelebA",
# #         "params": "Large pretrained baseline",
# #         "benchmarks": {
# #             "current_reference": {
# #                 "label": "CelebA full | 512 | full dataset | current reference",
# #                 "fid": 22.0,
# #                 "psnr_masked": 26.85,
# #                 "ssim_masked": 0.94,
# #                 "lpips_masked": 0.04,
# #                 "avg_inference_time": 5.8,
# #                 "sampling_steps": None,
# #                 "num_test_samples": None,
# #             }
# #         },
# #     },
# # }


# # def list_model_names():
# #     return list(MODEL_REGISTRY.keys())


# # def get_model_config(model_name: str):
# #     if model_name not in MODEL_REGISTRY:
# #         raise ValueError(f"Unknown model name: {model_name}")
# #     return MODEL_REGISTRY[model_name]


# # def get_checkpoint_path(model_name: str):
# #     cfg = get_model_config(model_name)
# #     return cfg.get("checkpoint_path")


# # def get_hf_model_id(model_name: str):
# #     cfg = get_model_config(model_name)
# #     return cfg.get("hf_model_id")


# # def select_benchmark_for_model(model_name: str, steps: int):
# #     cfg = get_model_config(model_name)
# #     benchmarks = cfg.get("benchmarks", {})

# #     if model_name == "A02 Best":
# #         if steps >= 100 and "steps_100" in benchmarks:
# #             return benchmarks["steps_100"]
# #         if "steps_50" in benchmarks:
# #             return benchmarks["steps_50"]

# #     if model_name == "SD2 Inpainting":
# #         if "current_reference" in benchmarks:
# #             return benchmarks["current_reference"]

# #     return None



# from __future__ import annotations

# from typing import Dict, Any


# # ---------------------------------------------------------
# # Central model registry
# # ---------------------------------------------------------
# # Notes:
# # - Update checkpoint_path to your actual local checkpoint file
# # - "params" for Student A02 = student UNet only
# # - "params" for SD2 baseline = full loaded inference pipeline
# # ---------------------------------------------------------

# MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
#     "student_a02": {
#         "key": "student_a02",
#         "display_name": "Student A02",
#         "type": "student",
#         "source": "local checkpoint",
#         "dataset": "CelebA",
#         "checkpoint_path": "D:\\cse499b_inpainting_deploy\\cse499b_inpainting_deploy\\checkpoints\\student\\studentA02_best.pt",  # <-- update if needed
#         "hf_vae_model_id": "sd2-community/stable-diffusion-2-inpainting",
#         "params": 6697540,  # 6.70M (student model only)
#         "notes": "Lightweight latent diffusion student distilled from SD2 teacher."
#     },

#     "sd2_baseline": {
#         "key": "sd2_baseline",
#         "display_name": "Stable Diffusion 2 Inpainting",
#         "type": "teacher_baseline",
#         "source": "huggingface",
#         "dataset": "CelebA",
#         "hf_model_id": "sd2-community/stable-diffusion-2-inpainting",
#         "params": 1289966827,  # full pipeline count from your notebook
#         "unet_params": 865925124,
#         "vae_params": 83653863,
#         "text_encoder_params": 340387840,
#         "notes": "Full prompt-guided SD2 inpainting pipeline."
#     },
# }


# # ---------------------------------------------------------
# # Offline benchmark table
# # ---------------------------------------------------------
# # Put your measured results here.
# # You can refine these later when you get newer numbers.
# # ---------------------------------------------------------

# BENCHMARKS: Dict[str, Dict[int, Dict[str, Any]]] = {
#     "student_a02": {
#         50: {
#             "label": "Student A02 @ 50 steps",
#             "steps": 50,
#             "fid": 49.822205,
#             "psnr_masked": 18.537871,
#             "ssim_masked": 0.843078,
#             "lpips_masked": 0.115068,
#             "inference_sec": 0.323076,
#             "params": 6697540,
#         },
#         100: {
#             "label": "Student A02 @ 100 steps",
#             "steps": 100,
#             "fid": 46.0,
#             "psnr_masked": 18.3,
#             "ssim_masked": 0.84,
#             "lpips_masked": 0.15,
#             "inference_sec": 0.42,
#             "params": 6697540,
#         },
#     },

#     "sd2_baseline": {
#         50: {
#             "label": "SD2 Baseline @ 50 steps",
#             "steps": 50,
#             "fid": None,
#             "psnr_masked": None,
#             "ssim_masked": None,
#             "lpips_masked": None,
#             "inference_sec": None,
#             "params": 1289966827,
#         },
#         100: {
#             "label": "SD2 Baseline @ 100 steps",
#             "steps": 100,
#             "fid": None,
#             "psnr_masked": None,
#             "ssim_masked": None,
#             "lpips_masked": None,
#             "inference_sec": None,
#             "params": 1289966827,
#         },
#     },
# }


# # ---------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------

# def get_model_config(model_name: str) -> Dict[str, Any]:
#     if model_name not in MODEL_REGISTRY:
#         available = ", ".join(MODEL_REGISTRY.keys())
#         raise KeyError(f"Unknown model '{model_name}'. Available: {available}")
#     return MODEL_REGISTRY[model_name]


# def get_all_models() -> Dict[str, Dict[str, Any]]:
#     return MODEL_REGISTRY


# def select_benchmark_for_model(model_name: str, steps: int) -> Dict[str, Any] | None:
#     model_benchmarks = BENCHMARKS.get(model_name, {})
#     if not model_benchmarks:
#         return None

#     if steps in model_benchmarks:
#         return model_benchmarks[steps]

#     # fall back to nearest available step count
#     available_steps = sorted(model_benchmarks.keys())
#     nearest_step = min(available_steps, key=lambda s: abs(s - steps))
#     return model_benchmarks[nearest_step]





from __future__ import annotations

from typing import Dict, Any


# ---------------------------------------------------------
# Central model registry
# ---------------------------------------------------------
# Notes:
# - Update checkpoint_path if needed
# - Student params = ONLY student UNet
# - SD2 params = FULL inference pipeline (UNet + VAE + text encoder)
# ---------------------------------------------------------

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "student_a02": {
        "key": "student_a02",
        "display_name": "Student A02",
        "type": "student",
        "source": "local checkpoint",
        "dataset": "CelebA",

        # 🔥 IMPORTANT: update path if needed
        "checkpoint_path": "D:\\cse499b_inpainting_deploy\\cse499b_inpainting_deploy\\checkpoints\\student\\studentA02_best.pt",

        # SD2 VAE used for encoding/decoding latent
        "hf_vae_model_id": "sd2-community/stable-diffusion-2-inpainting",

        # ✅ Student model size (your real count)
        "params": 6697540,  # 6.70M

        "notes": "Lightweight latent diffusion student distilled from SD2 teacher.",
    },

    "sd2_baseline": {
        "key": "sd2_baseline",
        "display_name": "Stable Diffusion 2 Inpainting",
        "type": "pretrained",
        "source": "huggingface",
        "dataset": "CelebA",

        "hf_model_id": "sd2-community/stable-diffusion-2-inpainting",

        # ✅ FULL pipeline params (your computed values)
        "params": 1289966827,  # 1.29B

        "unet_params": 865925124,
        "vae_params": 83653863,
        "text_encoder_params": 340387840,

        "notes": "Full prompt-guided SD2 inpainting pipeline.",
    },
}


# ---------------------------------------------------------
# Offline benchmark table (for UI display)
# ---------------------------------------------------------

BENCHMARKS: Dict[str, Dict[int, Dict[str, Any]]] = {
    "student_a02": {
        50: {
            "label": "Student A02 @ 50 steps",
            "steps": 50,
            "fid": 49.822205,
            "psnr_masked": 18.537871,
            "ssim_masked": 0.843078,
            "lpips_masked": 0.115068,
            "avg_inference_time": 0.323076,
            "params": 6697540,
        },
        100: {
            "label": "Student A02 @ 100 steps",
            "steps": 100,
            "fid": 46.0,
            "psnr_masked": 18.3,
            "ssim_masked": 0.84,
            "lpips_masked": 0.15,
            "avg_inference_time": 0.42,
            "params": 6697540,
        },
    },

    "sd2_baseline": {
        50: {
            "label": "SD2 Baseline @ 50 steps",
            "steps": 50,
            "fid": None,
            "psnr_masked": None,
            "ssim_masked": None,
            "lpips_masked": None,
            "avg_inference_time": None,
            "params": 1289966827,
        },
        100: {
            "label": "SD2 Baseline @ 100 steps",
            "steps": 100,
            "fid": None,
            "psnr_masked": None,
            "ssim_masked": None,
            "lpips_masked": None,
            "avg_inference_time": None,
            "params": 1289966827,
        },
    },
}


# ---------------------------------------------------------
# Helper functions (VERY IMPORTANT for app.py)
# ---------------------------------------------------------

def list_model_names():
    """Return all model keys for dropdown"""
    return list(MODEL_REGISTRY.keys())


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get config for selected model"""
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown model '{model_name}'. Available: {available}")
    return MODEL_REGISTRY[model_name]


def get_all_models() -> Dict[str, Dict[str, Any]]:
    return MODEL_REGISTRY


def select_benchmark_for_model(model_name: str, steps: int) -> Dict[str, Any] | None:
    """Select closest benchmark for given step count"""
    model_benchmarks = BENCHMARKS.get(model_name, {})
    if not model_benchmarks:
        return None

    if steps in model_benchmarks:
        return model_benchmarks[steps]

    # fallback → nearest step
    available_steps = sorted(model_benchmarks.keys())
    nearest_step = min(available_steps, key=lambda s: abs(s - steps))
    return model_benchmarks[nearest_step]