# CSE499B Inpainting Deploy

Starter scaffold for comparing your lightweight student checkpoint (A02 and future checkpoints) with Stable Diffusion 2 Inpainting in a simple Gradio app.

## Features in this scaffold
- Upload image
- Draw/upload mask
- Compare two models side by side
- Student checkpoint registry
- SD2 loader placeholder
- Shared preprocessing utilities
- Save outputs to `outputs/`

## Suggested workflow
1. Put your student checkpoint inside `checkpoints/student/`
2. Update `models/model_registry.py`
3. Replace the placeholder student architecture/inference with your real code
4. Install requirements
5. Run `python app.py`

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python app.py
```

## Important
This scaffold is designed to make your deployment structure clean. You still need to connect:
- your exact A02 model architecture
- your checkpoint loading keys
- your true diffusion sampling/inference code
- your SD2 model id / device settings
