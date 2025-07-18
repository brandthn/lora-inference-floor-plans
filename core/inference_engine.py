import torch
import numpy as np
from PIL import Image
import cv2
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import random

from diffusers import StableDiffusionXLPipeline, ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.schedulers import (
    EulerDiscreteScheduler, 
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler
)

from .config import config_manager
from .generation_params import GenerationParams
from .utils.image_utils import save_image, create_canny_image

class InferenceEngine:
    """Moteur d'inférence principal pour la génération d'images"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipelines = {}
        self.controlnet_models = {}
        self.current_base_model = None
        
        print(f"🚀 Initialisation du moteur d'inférence sur {self.device}")
        
        # Vérifier la configuration
        if not self._check_models():
            print("⚠️  Certains modèles sont manquants. Vérifiez votre configuration.")
    
    def _check_models(self) -> bool:
        """Vérifie la disponibilité des modèles"""
        base_models = config_manager.get_base_models()
        lora_models = config_manager.get_lora_models()
        
        missing = []
        for name, model in base_models.items():
            if not os.path.exists(model.path):
                missing.append(f"Base model '{name}': {model.path}")
        
        for name, model in lora_models.items():
            if not os.path.exists(model.path):
                missing.append(f"LoRA '{name}': {model.path}")
        
        if missing:
            print("❌ Modèles manquants:")
            for m in missing:
                print(f"   - {m}")
            return False
        
        return True
    
    def _get_scheduler(self, sampler_name: str):
        """Retourne le scheduler correspondant au nom"""
        schedulers = {
            "Euler": EulerDiscreteScheduler,
            "Euler a": EulerAncestralDiscreteScheduler,
            "DPM++ 2M Karras": DPMSolverMultistepScheduler,
            "DPM++ SDE Karras": DPMSolverMultistepScheduler,
            "DDIM": DDIMScheduler,
        }
        
        return schedulers.get(sampler_name, DPMSolverMultistepScheduler)
    
    def _load_base_pipeline(self, model_name: str) -> StableDiffusionXLPipeline:
        """Charge le pipeline de base"""
        base_models = config_manager.get_base_models()
        
        if model_name not in base_models:
            raise ValueError(f"Modèle de base '{model_name}' non trouvé")
        
        model_path = base_models[model_name].path
        
        # Charger le pipeline
        pipeline = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True
        )
        
        pipeline = pipeline.to(self.device)
        
        # Optimisations pour la performance
        if self.device == "cuda":
            pipeline.enable_attention_slicing()
            pipeline.enable_model_cpu_offload()
        
        return pipeline
    
    def _load_controlnet_pipeline(self, model_name: str, controlnet_type: str = "canny") -> StableDiffusionXLControlNetPipeline:
        """Charge le pipeline avec ControlNet"""
        base_models = config_manager.get_base_models()
        controlnet_models = config_manager.get_controlnet_models()
        
        if model_name not in base_models:
            raise ValueError(f"Modèle de base '{model_name}' non trouvé")
        
        if controlnet_type not in controlnet_models:
            raise ValueError(f"ControlNet '{controlnet_type}' non trouvé")
        
        model_path = base_models[model_name].path
        controlnet_path = controlnet_models[controlnet_type].path
        
        # Charger le ControlNet
        controlnet = ControlNetModel.from_single_file(
            controlnet_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True
        )
        
        # Charger le pipeline avec ControlNet
        pipeline = StableDiffusionXLControlNetPipeline.from_single_file(
            model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True
        )
        
        pipeline = pipeline.to(self.device)
        
        # Optimisations pour la performance
        if self.device == "cuda":
            pipeline.enable_attention_slicing()
            pipeline.enable_model_cpu_offload()
        
        return pipeline
    
    def _apply_lora(self, pipeline, lora_name: str, weight: float = 0.8):
        """Applique un LoRA au pipeline"""
        lora_models = config_manager.get_lora_models()
        
        if lora_name not in lora_models:
            raise ValueError(f"LoRA '{lora_name}' non trouvé")
        
        lora_path = lora_models[lora_name].path
        
        # Charger le LoRA avec la nouvelle API
        try:
            pipeline.load_lora_weights(lora_path, adapter_name=lora_name)
            # Définir le poids du LoRA
            pipeline.set_adapters([lora_name], adapter_weights=[weight])
            
            print(f"✅ LoRA '{lora_name}' appliqué avec un poids de {weight}")
            return pipeline
        except Exception as e:
            print(f"❌ Erreur lors du chargement du LoRA '{lora_name}': {e}")
            # Essayer l'ancienne méthode en fallback
            try:
                pipeline.load_lora_weights(lora_path)
                pipeline.fuse_lora(lora_scale=weight)
                print(f"✅ LoRA '{lora_name}' appliqué (méthode fallback)")
                return pipeline
            except Exception as e2:
                raise Exception(f"Impossible de charger le LoRA: {e2}")
    
    
    def _prepare_seed(self, seed: int) -> int:
        """Prépare le seed pour la génération"""
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        
        # Définir le seed pour la reproductibilité
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        return seed
    
    def generate_single_lora(self, params: GenerationParams) -> Tuple[str, Optional[str]]:
        """Génère une image avec un seul LoRA"""
        print(f"🎨 Génération avec {params.lora_model} (approche single_lora)")
        
        # Préparer le seed
        actual_seed = self._prepare_seed(params.seed)
        
        # Charger le pipeline de base
        pipeline = self._load_base_pipeline(params.base_model)
        
        # Appliquer le LoRA
        pipeline = self._apply_lora(pipeline, params.lora_model, params.lora_weight)
        
        # Configurer le scheduler
        scheduler_class = self._get_scheduler(params.sampler)
        pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config)
        
        # Générer l'image
        generator = torch.Generator(device=self.device).manual_seed(actual_seed)
        
        result = pipeline(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            num_inference_steps=params.steps,
            guidance_scale=params.cfg_scale,
            width=params.width,
            height=params.height,
            num_images_per_prompt=params.batch_size,
            generator=generator
        )
        
        # Sauvegarder l'image
        image = result.images[0]
        filename = f"{params.get_filename_prefix()}_seed_{actual_seed}.png"
        image_path = save_image(image, filename, params.to_dict())
        
        # TODO: Upload vers S3 si configuré
        s3_url = None
        
        print(f"✅ Image générée: {image_path}")
        return image_path, s3_url
    
    def generate_combined_approach(self, params: GenerationParams) -> Tuple[str, Optional[str]]:
        """Génère une image avec l'approche combinée (wall_lora + controlnet + plan_lora)"""
        print("🎨 Génération avec approche combinée (wall_lora + controlnet + plan_lora)")
        
        # Préparer le seed
        actual_seed = self._prepare_seed(params.seed)
        
        # Étape 1: Générer l'image avec wall_lora
        print("📐 Étape 1: Génération des contours avec wall_lora")
        wall_pipeline = self._load_base_pipeline(params.base_model)
        wall_pipeline = self._apply_lora(wall_pipeline, "wall_lora", params.wall_lora_weight)
        
        # Configurer le scheduler
        scheduler_class = self._get_scheduler(params.sampler)
        wall_pipeline.scheduler = scheduler_class.from_config(wall_pipeline.scheduler.config)
        
        # Générer l'image des contours
        generator = torch.Generator(device=self.device).manual_seed(actual_seed)
        
        wall_result = wall_pipeline(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            num_inference_steps=params.steps,
            guidance_scale=params.cfg_scale,
            width=params.width,
            height=params.height,
            generator=generator
        )
        
        wall_image = wall_result.images[0]
        
        # Étape 2: Traitement canny edge
        print("🔍 Étape 2: Traitement Canny Edge")
        canny_image = create_canny_image(wall_image)
        
        # Étape 3: Génération finale avec ControlNet + plan_lora
        print("🎯 Étape 3: Génération finale avec ControlNet + plan_lora")
        controlnet_pipeline = self._load_controlnet_pipeline(params.base_model, "canny")
        
        # Appliquer le plan_lora
        plan_lora_name = params.lora_model if params.lora_model else "lora_plan_v1"
        controlnet_pipeline = self._apply_lora(controlnet_pipeline, plan_lora_name, params.plan_lora_weight)
        
        # Configurer le scheduler
        controlnet_pipeline.scheduler = scheduler_class.from_config(controlnet_pipeline.scheduler.config)
        
        # Générer l'image finale
        generator = torch.Generator(device=self.device).manual_seed(actual_seed)
        
        final_result = controlnet_pipeline(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            image=canny_image,
            num_inference_steps=params.steps,
            guidance_scale=params.cfg_scale,
            controlnet_conditioning_scale=params.controlnet_weight,
            width=params.width,
            height=params.height,
            generator=generator
        )
        
        # Sauvegarder l'image finale
        final_image = final_result.images[0]
        filename = f"{params.get_filename_prefix()}_combined_seed_{actual_seed}.png"
        image_path = save_image(final_image, filename, params.to_dict())
        
        # Sauvegarder aussi les images intermédiaires pour debug
        wall_filename = f"{params.get_filename_prefix()}_wall_seed_{actual_seed}.png"
        canny_filename = f"{params.get_filename_prefix()}_canny_seed_{actual_seed}.png"
        
        save_image(wall_image, wall_filename, {"step": "wall_generation"})
        save_image(canny_image, canny_filename, {"step": "canny_processing"})
        
        # TODO: Upload vers S3 si configuré
        s3_url = None
        
        print(f"✅ Image générée (approche combinée): {image_path}")
        return image_path, s3_url
    
    def generate(self, params: GenerationParams) -> Tuple[str, Optional[str]]:
        """Point d'entrée principal pour la génération"""
        if not params.validate():
            raise ValueError("Paramètres de génération invalides")
        
        try:
            if params.approach == "single_lora":
                return self.generate_single_lora(params)
            elif params.approach == "combined_approach":
                return self.generate_combined_approach(params)
            else:
                raise ValueError(f"Approche '{params.approach}' non supportée")
        
        except Exception as e:
            print(f"❌ Erreur lors de la génération: {e}")
            raise
        
        finally:
            # Nettoyer la mémoire GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne des informations sur les modèles disponibles"""
        base_models = config_manager.get_base_models()
        lora_models = config_manager.get_lora_models()
        controlnet_models = config_manager.get_controlnet_models()
        
        return {
            "device": self.device,
            "base_models": {name: {"path": model.path, "available": os.path.exists(model.path)} 
                          for name, model in base_models.items()},
            "lora_models": {name: {"path": model.path, "available": os.path.exists(model.path)} 
                          for name, model in lora_models.items()},
            "controlnet_models": {name: {"path": model.path, "available": os.path.exists(model.path)} 
                                for name, model in controlnet_models.items()},
            "memory_info": self._get_memory_info()
        }
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Retourne des informations sur la mémoire"""
        if torch.cuda.is_available():
            return {
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
                "gpu_name": torch.cuda.get_device_name(0)
            }
        else:
            return {"device": "cpu"}

# Instance globale du moteur d'inférence
inference_engine = InferenceEngine()