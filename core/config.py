import yaml
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class ModelConfig:
    path: str
    type: str = None
    default: bool = False
    default_weight: float = 1.0
    description: str = ""

@dataclass
class GenerationDefaults:
    steps: int
    cfg_scale: float
    sampler: str
    width: int
    height: int
    batch_size: int
    denoise_strength: float
    seed: int

class ConfigManager:
    def __init__(self, config_path: str = "config/models_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_paths()
    
    def _load_config(self) -> Dict:
        """Charge la configuration depuis le fichier YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def _validate_paths(self):
        """Valide l'existence des chemins des modèles"""
        missing_files = []
        
        # Vérifier les modèles de base
        for name, config in self.config.get('base_models', {}).items():
            if not os.path.exists(config['path']):
                missing_files.append(f"Base model '{name}': {config['path']}")
        
        # Vérifier les LoRA
        for name, config in self.config.get('lora_models', {}).items():
            if not os.path.exists(config['path']):
                missing_files.append(f"LoRA '{name}': {config['path']}")
        
        # Vérifier les ControlNet
        for name, config in self.config.get('controlnet_models', {}).items():
            if not os.path.exists(config['path']):
                missing_files.append(f"ControlNet '{name}': {config['path']}")
        
        if missing_files:
            print("⚠️  Attention: Fichiers modèles manquants:")
            for file in missing_files:
                print(f"   - {file}")
            print("L'application peut ne pas fonctionner correctement.")
    
    def get_base_models(self) -> Dict[str, ModelConfig]:
        """Retourne la configuration des modèles de base"""
        models = {}
        for name, config in self.config.get('base_models', {}).items():
            models[name] = ModelConfig(
                path=config['path'],
                type=config.get('type'),
                default=config.get('default', False)
            )
        return models
    
    def get_lora_models(self) -> Dict[str, ModelConfig]:
        """Retourne la configuration des modèles LoRA"""
        models = {}
        for name, config in self.config.get('lora_models', {}).items():
            models[name] = ModelConfig(
                path=config['path'],
                default_weight=config.get('default_weight', 1.0),
                description=config.get('description', '')
            )
        return models
    
    def get_controlnet_models(self) -> Dict[str, ModelConfig]:
        """Retourne la configuration des modèles ControlNet"""
        models = {}
        for name, config in self.config.get('controlnet_models', {}).items():
            models[name] = ModelConfig(
                path=config['path'],
                type=config.get('type')
            )
        return models
    
    def get_generation_defaults(self) -> GenerationDefaults:
        """Retourne les paramètres de génération par défaut"""
        defaults = self.config.get('generation_defaults', {})
        return GenerationDefaults(
            steps=defaults.get('steps', 30),
            cfg_scale=defaults.get('cfg_scale', 7.5),
            sampler=defaults.get('sampler', 'DPM++ 2M Karras'),
            width=defaults.get('width', 1024),
            height=defaults.get('height', 1024),
            batch_size=defaults.get('batch_size', 1),
            denoise_strength=defaults.get('denoise_strength', 0.8),
            seed=defaults.get('seed', -1)
        )
    
    def get_samplers(self) -> List[str]:
        """Retourne la liste des samplers disponibles"""
        return self.config.get('samplers', ['Euler', 'DPM++ 2M Karras'])
    
    def get_prompt_templates(self) -> Dict[str, str]:
        """Retourne les templates de prompts"""
        return self.config.get('prompt_templates', {})
    
    def get_room_types(self) -> List[str]:
        """Retourne la liste des types de pièces"""
        return self.config.get('room_types', [])
    
    def get_default_base_model(self) -> Optional[str]:
        """Retourne le nom du modèle de base par défaut"""
        for name, config in self.config.get('base_models', {}).items():
            if config.get('default', False):
                return name
        return None
    
    def create_directories(self):
        """Crée les dossiers nécessaires s'ils n'existent pas"""
        directories = [
            "data/models/base",
            "data/models/lora", 
            "data/models/controlnet",
            "data/generations",
            "config"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        print("✅ Dossiers créés/vérifiés")

# Instance globale du gestionnaire de configuration
config_manager = ConfigManager()