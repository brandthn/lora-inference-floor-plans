import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import cv2
from pathlib import Path
import uuid
from datetime import datetime
import logging

from .config import (
    SDXL_MODEL_ID, CONTROLNET_MODEL_ID, WALL_LORA_PATH, FLOOR_PLAN_LORA_PATH,
    DEVICE, IMAGE_SIZE, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, 
    CONTROLNET_CONDITIONING_SCALE, OUTPUT_DIR
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FloorPlanLoRAInference:
    def __init__(self):
        self.wall_pipeline = None
        self.floor_plan_pipeline = None
        self.load_models()
    
    def load_models(self):
        """Charge les modèles et pipelines nécessaires"""
        try:
            logger.info("Chargement des modèles...")
            
            # Vérifier que les fichiers LoRA existent
            if not self._check_lora_files():
                raise FileNotFoundError("Fichiers LoRA manquants - voir logs ci-dessus")
            
            # 1. Pipeline pour Wall_Lora_2 (génération de base)
            logger.info("Chargement de Wall_Lora_2...")
            self.wall_pipeline = StableDiffusionXLPipeline.from_pretrained(
                SDXL_MODEL_ID,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            
            # Charger le Wall_Lora_2
            wall_lora_path = str(WALL_LORA_PATH)
            try:
                self.wall_pipeline.load_lora_weights(wall_lora_path)
                logger.info("Wall_Lora_2 chargé avec succès")
            except Exception as e:
                logger.error(f"Erreur chargement Wall_Lora_2: {e}")
                # Essayer de charger sans LoRA pour tester
                logger.warning("Tentative de chargement sans Wall_Lora_2...")
            
            self.wall_pipeline.to(DEVICE)
            
            # 2. Pipeline pour floor_plans_a_v1 avec ControlNet
            logger.info("Chargement de ControlNet...")
            controlnet = ControlNetModel.from_pretrained(
                CONTROLNET_MODEL_ID,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            
            logger.info("Chargement de floor_plans_a_v1...")
            self.floor_plan_pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                SDXL_MODEL_ID,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            
            # Charger le floor_plans_a_v1
            floor_plan_lora_path = str(FLOOR_PLAN_LORA_PATH)
            try:
                self.floor_plan_pipeline.load_lora_weights(floor_plan_lora_path)
                logger.info("floor_plans_a_v1 chargé avec succès")
            except Exception as e:
                logger.error(f"Erreur chargement floor_plans_a_v1: {e}")
                # Essayer de charger sans LoRA pour tester
                logger.warning("Tentative de chargement sans floor_plans_a_v1...")
            
            self.floor_plan_pipeline.to(DEVICE)
            
            # Optimisation pour M2 Max
            if DEVICE == "mps":
                self.wall_pipeline.enable_attention_slicing()
                self.floor_plan_pipeline.enable_attention_slicing()
            
            logger.info("Tous les modèles chargés avec succès!")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles: {e}")
            raise
    
    def _check_lora_files(self):
        """Vérifie la présence des fichiers LoRA"""
        wall_lora_files = list(WALL_LORA_PATH.glob("*.safetensors"))
        floor_plan_lora_files = list(FLOOR_PLAN_LORA_PATH.glob("*.safetensors"))
        
        if not wall_lora_files:
            logger.error(f"Aucun fichier .safetensors trouvé dans {WALL_LORA_PATH}")
            logger.error("Veuillez placer les fichiers Wall_Lora_2 dans models/Wall_Lora_2/")
            return False
        
        if not floor_plan_lora_files:
            logger.error(f"Aucun fichier .safetensors trouvé dans {FLOOR_PLAN_LORA_PATH}")
            logger.error("Veuillez placer les fichiers floor_plans_a_v1 dans models/floor_plans_a_v1/")
            return False
        
        logger.info(f"Fichiers Wall_Lora_2 trouvés: {[f.name for f in wall_lora_files]}")
        logger.info(f"Fichiers floor_plans_a_v1 trouvés: {[f.name for f in floor_plan_lora_files]}")
        
        return True
    
    def generate_wall_image(self, prompt: str, negative_prompt: str = None) -> Image.Image:
        """Génère l'image avec contours/edges avec Wall_Lora_2"""
        try:
            logger.info(f"Génération de l'image avec contours/edges avec Wall_Lora_2...")
            
            if negative_prompt is None:
                negative_prompt = "blurry, low quality, distorted, watermark, text"
            
            # Génération avec Wall_Lora_2 (contient déjà les contours/edges)
            wall_image = self.wall_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                width=IMAGE_SIZE,
                height=IMAGE_SIZE
                # Pas de generator pour seed aléatoire
            ).images[0]
            
            return wall_image
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de l'image Wall_Lora_2: {e}")
            raise
    
    def generate_canny_condition(self, image: Image.Image) -> Image.Image:
        """Génère la condition canny pour ControlNet"""
        try:
            logger.info("Génération de la condition canny...")
            
            # Convertir en canny edge
            canny_image = self.canny_detector(image)
            
            return canny_image
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération canny: {e}")
            raise
    
    def generate_floor_plan(self, prompt: str, wall_image: Image.Image, negative_prompt: str = None) -> Image.Image:
        """Génère le plan d'étage final avec floor_plans_a_v1 et ControlNet"""
        try:
            logger.info("Génération du plan d'étage final avec ControlNet...")
            
            if negative_prompt is None:
                negative_prompt = "blurry, low quality, distorted, watermark, text, people, furniture"
            
            # Utiliser directement l'image Wall_Lora_2 comme condition ControlNet
            # (elle contient déjà les contours/edges)
            floor_plan_image = self.floor_plan_pipeline(
                prompt=prompt,
                image=wall_image,  # Utiliser directement wall_image
                negative_prompt=negative_prompt,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
                width=IMAGE_SIZE,
                height=IMAGE_SIZE
                # Pas de generator pour seed aléatoire
            ).images[0]
            
            return floor_plan_image
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du plan d'étage: {e}")
            raise
    
    def generate_complete_floor_plan(self, prompt: str, negative_prompt: str = None) -> tuple[Image.Image, Image.Image]:
        """Pipeline complet de génération"""
        try:
            logger.info(f"Début de la génération complète pour: '{prompt}'")
            
            # Étape 1: Générer l'image avec contours/edges avec Wall_Lora_2
            wall_image = self.generate_wall_image(prompt, negative_prompt)
            
            # Étape 2: Utiliser directement l'image Wall_Lora_2 comme condition ControlNet
            # (PAS de Canny Edge - Wall_Lora_2 génère déjà les contours)
            floor_plan_image = self.generate_floor_plan(prompt, wall_image, negative_prompt)
            
            logger.info("Génération complète terminée avec succès!")
            
            return wall_image, floor_plan_image
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération complète: {e}")
            raise
    
    def save_images(self, images: list[Image.Image], filenames: list[str]) -> list[Path]:
        """Sauvegarde les images générées"""
        try:
            saved_paths = []
            
            for image, filename in zip(images, filenames):
                # Créer un nom de fichier unique
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_filename = f"{timestamp}_{filename}"
                filepath = OUTPUT_DIR / unique_filename
                
                # Sauvegarder l'image
                image.save(filepath, format="PNG", quality=95)
                saved_paths.append(filepath)
                
                logger.info(f"Image sauvegardée: {filepath}")
            
            return saved_paths
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            raise

# Instance globale
inference_model = None

def get_inference_model() -> FloorPlanLoRAInference:
    """Récupère l'instance du modèle d'inférence (singleton)"""
    global inference_model
    if inference_model is None:
        inference_model = FloorPlanLoRAInference()
    return inference_model