<p align="center">
  
</p>

<h1 align="center">Design and Analysis of Lightweight Diffusion Models for Efficient Image Inpainting</h1>
<h3 align="center">A Teacher-Student Latent Diffusion Framework Based on Stable Diffusion 2</h3>

<p align="center">
  <strong>Author:</strong> Md. Minhajul Islam  
  <br>
  <strong>Course:</strong> CSE499B – Capstone Project  
  <br>
  <strong>Institution:</strong> North South University  
  <br>
  <strong>Supervisor:</strong> M. Shifat-E-Rabbi
</p>

---

## 📌 Project Overview

This project explores how to build a **lightweight diffusion-based image inpainting model** that can reconstruct missing image regions efficiently while significantly reducing computational cost.

Large pretrained models such as **Stable Diffusion 2 Inpainting** generate strong visual results but are computationally expensive and difficult to deploy in low-resource settings. To address this limitation, this work proposes a **teacher-student latent diffusion framework**, where a compact student model learns from a frozen Stable Diffusion 2 teacher.

The main goal is to achieve a strong **quality-efficiency tradeoff** by reducing model complexity while still producing reasonable inpainting quality.

---

## 🚀 Core Idea

- Use **Stable Diffusion 2 Inpainting** as a strong frozen teacher
- Build a **lightweight latent diffusion student**
- Train the student using:
  - diffusion loss
  - teacher distillation loss
- Evaluate the tradeoff between:
  - visual quality
  - parameter count
  - inference speed

---

## 🧠 Proposed Student Model (A02)

Our final lightweight model, **Student A02**, is a mask-conditioned latent diffusion model.

### Key characteristics:
- Operates in **latent space**
- Uses **masked image conditioning**
- Uses **hybrid masking**
- Trained with **teacher-student distillation**
- Designed for **fast and efficient inference**

### Input to the student model:
The student receives a 9-channel concatenated latent input:

- `z_t` → noisy latent
- `z_masked` → masked-image latent
- `m_latent` → latent-space binary mask

So the full student input is:

```text
[z_t, z_masked, m_latent]



🏗️ Model Architecture
Student A02
Lightweight latent diffusion U-Net
Input channels: 9
Base channels: 64
Total parameters: 6,697,540 (~6.70M)
Stable Diffusion 2 Baseline

The aligned SD2 baseline used in this project contains:

UNet: 865,925,124 (~865.93M)
VAE: 83,653,863 (~83.65M)
Text Encoder: 340,387,840 (~340.39M)
Full SD2 pipeline total:
1,289,966,827 (~1.29B) parameters
Parameter reduction

Compared to the SD2 UNet backbone, Student A02 reduces parameters by approximately 99.2%.


🧩 Method Overview
<p align="center">
  <img src="assets/system_diagram.jpg" alt="System Diagram" width="85%">
</p>
Training pipeline
Input image and mask are prepared
Original image and masked image are encoded using the SD2 VAE
Latents are obtained:
z
z_masked
m_latent
Noise is added to z to produce z_t
Inputs are concatenated:
[z_t, z_masked, m_latent]
Student U-Net predicts latent noise ε_pred
Frozen SD2 teacher predicts ε_teacher
Student is optimized using:
diffusion loss
teacher distillation loss
Inference pipeline
Start from random latent noise
Perform reverse diffusion using Student A02
Decode final latent using VAE decoder
Generate the inpainted image
🎯 Hybrid Masking Strategy

To improve robustness, the model is trained using multiple mask types:

Rectangle masks
Brush masks
Center masks
Blob masks

This hybrid masking setup helps the model learn under both structured and irregular missing-region conditions.


📊 Final Evaluation Results
Student A02 (100-step evaluation on CelebA fixed subset)
| Metric             | Value          |
| ------------------ | -------------- |
| FID                | 46.0           |
| PSNR (masked)      | 18.3           |
| SSIM (masked)      | 0.84           |
| LPIPS (masked)     | 0.15           |
| Avg Inference Time | 0.42 sec/image |
| Parameters         | 6.70M          |


⚖️ Comparison with SD2
| Model             | Parameters | Inference Type                    | Notes                    |
| ----------------- | ---------- | --------------------------------- | ------------------------ |
| Student A02       | 6.70M      | Lightweight latent diffusion      | Fast and efficient       |
| SD2 UNet          | 865.93M    | Backbone only                     | Strong teacher/reference |
| Full SD2 Pipeline | 1.29B      | Prompt-guided inpainting pipeline | Heavy baseline           |

Summary

Student A02 is much smaller and faster than SD2, while still maintaining a usable level of inpainting quality.

🖼️ Visual Results
<p align="center"> <img src="assets/results_grid.png" alt="Results Grid" width="90%"> </p>

The results show that Student A02 preserves global facial structure reasonably well, though masked-region quality is still less sharp than the SD2 baseline.

🎬 Demo
<p align="center"> <img src="assets/demo.gif" alt="Demo GIF" width="80%"> </p>

An interactive Gradio app is included for comparing:

Student A02
Stable Diffusion 2 baseline

using the same input image and mask.


🛠️ Technologies Used
Frameworks & Libraries
PyTorch
HuggingFace Diffusers
Transformers
TorchMetrics
LPIPS
PIL / OpenCV
Gradio
Compute Environment
Kaggle GPU
Google Colab
Local deployment environment (Python + Gradio)
📦 Dataset
CelebA Dataset

Used for training and evaluation.

Fixed subset used for reproducibility:
Train: 3000
Validation: 500
Test: 500
Dataset link
https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

📁 Project Structure
.
├── models/
│   ├── model_registry.py
│   ├── student_infer.py
│   ├── sd2_infer.py
│   └── utils.py
│
├── checkpoints/
│   └── student/
│       └── studentA02_best.pt
│
├── assets/
│   ├── banner.png
│   ├── system_diagram.png
│   ├── demo.gif
│   └── results_grid.png
│
├── outputs/
├── app.py
├── requirements.txt
└── README.md

▶️ How to Run the Project
1. Clone the repository
git clone https://github.com/MDMinhajul-Islam/CSE-499.git
cd CSE-499/CSE499B/cse499b_inpainting_deploy

2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Run the app
python app.py

Then open:http://127.0.0.1:7860

🧪 Current Limitations
Student A02 still produces blurred masked-region outputs in difficult facial areas
Visual quality remains below the SD2 baseline
The current student model still depends on latent-space preprocessing and diffusion scheduling
More work is needed to improve local realism and boundary blending


🔮 Future Work
Improve masked-region visual fidelity
Explore stronger mask-aware supervision
Add boundary-aware distillation
Increase resolution to 512×512
Investigate text-conditioned student models
Optimize deployment for real-time usage
👨‍🎓 Author

Md. Minhajul Islam
BSc in Computer Science and Engineering
North South University

Nur Ibne Kawsar Zitu
BSc in Computer Science and Engineering
North South University

Kazi Tazrian Mon
BSc in Computer Science and Engineering
North South University

🙏 Acknowledgements
Stable Diffusion 2 Inpainting
HuggingFace Diffusers
CelebA Dataset
PyTorch ecosystem
North South University




                                                            
Supervisor:
Dr. Mohammad Shifat-E-Rabbi
Assistant Professor,
Department of Electrical & Computer Engineering
North South University
                                              
  


