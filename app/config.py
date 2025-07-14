import os
import torch
from pathlib import Path

# Configuration générale
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Configuration des modèles
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_MODEL_ID = "diffusers/controlnet-canny-sdxl-1.0"

# Chemins vers les LoRAs
WALL_LORA_PATH = MODEL_DIR / "Wall_Lora_2"
FLOOR_PLAN_LORA_PATH = MODEL_DIR / "floor_plans_a_v1"

# Configuration d'inférence
IMAGE_SIZE = 512
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.5
CONTROLNET_CONDITIONING_SCALE = 1.0

# Configuration S3
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "floor-plans-generated")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Configuration de l'appareil (M2 Max) - Temporairement CPU pour debug
#DEVICE = "cpu"  # Changé temporairement de "mps" à "cpu" pour résoudre les images noires
DEVICE = "mps" if hasattr(torch, "backends") and torch.backends.mps.is_available() else "cpu"

# Configuration Streamlit
STREAMLIT_CONFIG = {
    "title": "Floor Plan Generator",
    "description": "Générateur de plans d'étage avec LoRA",
    "max_prompt_length": 500,
    "example_prompts": [
        "A modern apartment floor plan with 2 bedrooms and 1 bathroom",
        "Floor plan of a house with open kitchen and living room",
        "Small studio apartment floor plan with efficient layout",
        "Three bedroom house floor plan with separate dining room"
    ]
}

# Créer les répertoires nécessaires
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)