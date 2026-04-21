<p align="center">
  <img src="https://dummyimage.com/1600x300/4f46e5/ffffff&text=Diffusion+Models+for+Image+Inpainting:+DDPM,+Stable+Diffusion,+RePaint,+U-Net+Baseline" />
</p>

<h1 align="center">Diffusion Models for Image Inpainting</h1>
<h3 align="center">DDPM • Stable Diffusion • RePaint • U-Net Baseline</h3>

<p align="center">
  <strong>Authors:</strong> Md Minhajul Islam · Nur Ibne Kawsar Zitu · Kazi Tarzian Mon  
  <br>
  <strong>Semester:</strong> Fall 2025 (CSE499A – Project)  Group-4 
  <br>
  <strong>Supervisor:</strong> M. Shifat-E-Rabbi
</p>

---

##  **Project Overview**

This project provides a complete implementation and comparison of **classical** and **state-of-the-art** diffusion-based image inpainting models.

We explore:

- How diffusion models reconstruct missing image regions  
- How performance varies with model complexity & dataset scale  
- Quantitative comparison using **PSNR** & **SSIM**  
- Visual comparison across all models  

The study includes **four major components**:

---

## ✔ **Part A — DDPM From Scratch (CIFAR-10)**

- Implemented **Denoising Diffusion Probabilistic Model (DDPM)** manually  
- Built a **lightweight U-Net**, trained for 10 epochs  
- Performed generation & inpainting using RePaint-style masking  
- Demonstrates core diffusion behavior  
- **Limitation:** very low resolution (32×32), lacks semantic structure  

---

## ✔ **Part B — Stable Diffusion Inpainting (CelebA-HQ)**

- Used **HuggingFace diffusers** and `stable-diffusion-inpainting-2-0`  
- Produces **high-quality**, sharp, realistic inpainting  
- Handles **large missing regions** with strong semantic restoration  
- Achieved the **best PSNR/SSIM scores** among all models  

---

## ✔ **Part C — Supervised U-Net Inpainting (CelebA-HQ)**

- Built a **custom U-Net** trained using masked CelebA-HQ images  
- Optimized with **L1 reconstruction loss**  
- Produces **smooth** but slightly **blurry** results  
- Serves as a **baseline supervised** inpainting model  

---

## ✔ **Part D — RePaint Diffusion (CelebA-HQ, Pretrained)**

- Implemented **RePaint** using DDPM + backward–forward cycles  
- Excellent performance on **irregular masks**  
- Recovers global structure and texture  
- Competitive with Stable Diffusion in many cases  

---

## 📊 **Model Comparison Table**

| Model                 | Dataset   | Strengths                                   | Limitations                     |
| --------------------- | --------- | ------------------------------------------- | ------------------------------- |
| **DDPM (Scratch)**    | CIFAR-10  | Understands diffusion fundamentals          | Low resolution, weak semantics  |
| **UNet (Supervised)** | CelebA-HQ | Fast, stable                                | Blurry results, lacks structure |
| **Stable Diffusion**  | CelebA-HQ | Sharp, realistic, best visual quality        | Heavy pretrained model          |
| **RePaint**           | CelebA-HQ | Great for irregular masks, global coherence | Slowest inference               |

---

## 🧪 **Evaluation Metrics**

### **PSNR** — Peak Signal-to-Noise Ratio  
Higher → closer to original pixel values

### **SSIM** — Structural Similarity Index  
Higher → better structural & perceptual similarity  

> 📌 **Stable Diffusion and RePaint achieved the highest scores across both center and random masks.**



## 📁 **Project Structure**

```text
📦 Diffusion-Inpainting-Project
│
├── Part_A_DDPM_CIFAR10/
│   ├── ddpm_training.ipynb
│   ├── ddpm_inference.ipynb
│   └── outputs/
│
├── Part_B_Stable_Diffusion/
│   ├── inpainting_stable_diffusion.ipynb
│   └── results/
│
├── Part_C_UNet_Inpainting/
│   ├── unet_training.ipynb
│   └── outputs/
│
├── Part_D_RePaint/
│   ├── repaint_inference.ipynb
│   └── results/
│
└── report/
    └── Capstone_Final_Report.pdf





## 🧰 Technologies Used

 **Frameworks & Libraries**
- PyTorch
- HuggingFace Diffusers
- TorchVision
- Scikit-Image

#Compute
- Google Colab GPU(T4) 

#Models Implemented
- DDPM — custom PyTorch diffusion implementation  
- Stable Diffusion 2.0 Inpainting — pretrained, HF diffusers  
- RePaint Diffusion — DDPM with backward–forward jumps  
- Custom Supervised U-Net — trained on CelebA-HQ  



 📦 Datasets

1. CIFAR-10
- 60,000 color images (32×32)  
- Used for training DDPM from scratch  

2. CelebA-HQ
- High-resolution face dataset  
- Used for **Stable Diffusion**, **RePaint**, and **U-Net training**  



 ▶️ How to Run the Notebooks

 1️⃣ Clone the Repository
```bash
https://github.com/Minhajul-Islam-Rimon/CSE-499.git

```


2️⃣ Open Any Notebook in Google Colab

#Each notebook contains:

-Environment setup

-Model loading/initialization

-0Training & inference functions

-Automatic save-to-Google-Drive options

3️⃣ Login to HuggingFace (Required for Stable Diffusion & RePaint)
   from huggingface_hub import login
   login()

📝 Results Overview
<img width="956" height="645" alt="image" src="https://github.com/user-attachments/assets/96f90f89-8a29-4e28-9954-a18aedd8e2af" />
<img width="1445" height="807" alt="image" src="https://github.com/user-attachments/assets/b523aedb-cb61-45a5-86ed-575dec90a627" />
<img width="1532" height="322" alt="image" src="https://github.com/user-attachments/assets/63c59b84-c951-40e9-a0c5-b82b8b089eb4" />
<img width="1443" height="388" alt="image" src="https://github.com/user-attachments/assets/de306295-f83f-4a5a-8186-6ade21c31585" />

🔹 DDPM (Scratch)

-Blurry, low-resolution outputs

-Demonstrates fundamental diffusion behavior (noise → image)

🔹 Supervised U-Net

-Smooth reconstructions

-Lacks semantic detail & sharp textures

🔹 Stable Diffusion

-Sharpest and most realistic inpainting results

-Excellent semantic filling for large masked regions

🔹 RePaint

-Best for irregular masks

-Strong global structure maintenance



#Future Improvements

-Fine-tuning Stable Diffusion specifically for inpainting tasks

-Training models on larger or domain-specific datasets

-Exploring advanced diffusion variants:

   -Palette

   -SDEdit

   -ILVR

-Building a real-time inpainting web demo



📜 Citation References
1.CIFAR-10 Dataset

Krizhevsky, A. (2009). Learning multiple layers of features from tiny images.

2.CelebA-HQ Dataset

Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2018).
Progressive Growing of GANs for Improved Quality, Stability, and Variation.

<p align="center">
  <img src="https://dummyimage.com/1500x220/f43f5e/ffffff&text=❤️+Acknowledgements" />
</p>

<p align="center">
  <b>Department of Electrical & Computer Engineering</b><br>
  <b>North South University</b><br>
  <b>Supervisor: M. Shifat-E-Rabbi</b><br><br>
  Special thanks to the open-source community for providing<br>
  foundational implementations of:<br>
  <b>DDPM</b> • <b>Stable Diffusion</b> • <b>RePaint</b> • <b>U-Net</b>
</p>






