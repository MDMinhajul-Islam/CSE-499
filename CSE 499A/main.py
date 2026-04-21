"""
Main entry point for the Diffusion Inpainting Project
-----------------------------------------------------

This script serves as a launcher and guide for the project. Since the core
experiments (DDPM, Stable Diffusion, RePaint, U-Net training) are implemented
in Jupyter notebooks, please run the notebooks located in the 'support' 
directory for full functionality.

Notebooks included:
- PART_A.ipynb               (DDPM from scratch on CIFAR-10)
- Part_B_&_C.ipynb           (Stable Diffusion + U-Net baseline)
- Part_D.ipynb               (RePaint Diffusion)
- Full_Diffusion_Inpainting_Project_PartA_B_C_D_merged.ipynb (All parts combined)

To install dependencies:
    pip install -r requirements.txt

To run the project:
1. Open the notebooks in Google Colab or Jupyter.
2. Follow the execution steps inside each notebook.
3. For Stable Diffusion or RePaint, login to HuggingFace using:
       from huggingface_hub import login
       login()

Author: MD.Minhajul Islam, Nur Ibne Kawser Zitu, Kazi Tazrian Mon
Supervisor: M. Shifat-E-Rabbi
"""

def show_project_structure():
    print("\nProject Structure:")
    print(
        "    Diffusion-Inpainting-Project/\n"
        "    |-- main.py\n"
        "    |-- README.md\n"
        "    |-- requirements.txt\n"
        "    |-- data/\n"
        "    |-- support/\n"
        "    `-- others/\n"
    )
    print("Open the notebooks inside 'support/' to run the models.\n")


def main():
    print("\n===============================================")
    print("     Diffusion Inpainting Project Launcher     ")
    print("===============================================\n")

    print("This project is Jupyter Notebook based.")
    print("Please open the appropriate notebook from the 'support' folder:\n")
    print(" - PART_A.ipynb               -> DDPM (CIFAR-10)")
    print(" - Part_B_&_C.ipynb           -> Stable Diffusion + U-Net")
    print(" - Part_D.ipynb               -> RePaint Diffusion\n")

    print("NOTE: Install dependencies first:")
    print("    pip install -r requirements.txt\n")

    print("To use Stable Diffusion or RePaint, login to HuggingFace:")
    print("    from huggingface_hub import login")
    print("    login()\n")

    show_project_structure()


if __name__ == "__main__":
    main()
