from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json
import hashlib
from datetime import datetime

@dataclass
class GenerationParams:
    """Paramètres pour la génération d'images"""
    
    # Paramètres de base
    prompt: str
    negative_prompt: str = "blurry, low quality, distorted, text, watermark"
    
    # Paramètres de sampling
    steps: int = 30
    cfg_scale: float = 7.5
    sampler: str = "DPM++ 2M Karras"
    seed: int = -1
    denoise_strength: float = 0.8
    
    # Paramètres d'image
    width: int = 1024
    height: int = 1024
    batch_size: int = 1
    
    # Paramètres de modèle
    base_model: str = "sdxl_base"
    approach: str = "single_lora"  # "single_lora", "combined_approach"
    
    # Paramètres LoRA
    lora_model: Optional[str] = None
    lora_weight: float = 0.8
    
    # Paramètres pour l'approche combinée
    wall_lora_weight: float = 0.7
    plan_lora_weight: float = 0.8
    controlnet_weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convertit en JSON"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationParams':
        """Crée une instance depuis un dictionnaire"""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'GenerationParams':
        """Crée une instance depuis une chaîne JSON"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_hash(self) -> str:
        """Génère un hash unique pour ces paramètres"""
        # Exclure le seed du hash pour permettre la comparaison
        hash_data = self.to_dict()
        hash_data.pop('seed', None)
        
        json_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()[:8]
    
    def get_filename_prefix(self) -> str:
        """Génère un préfixe de nom de fichier basé sur les paramètres"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_suffix = self.get_hash()
        return f"{timestamp}_{self.approach}_{hash_suffix}"
    
    def validate(self) -> bool:
        """Valide les paramètres"""
        errors = []
        
        if not self.prompt.strip():
            errors.append("Le prompt ne peut pas être vide")
        
        if self.steps < 1 or self.steps > 100:
            errors.append("Le nombre de steps doit être entre 1 et 100")
        
        if self.cfg_scale < 1.0 or self.cfg_scale > 20.0:
            errors.append("CFG scale doit être entre 1.0 et 20.0")
        
        if self.width <= 0 or self.height <= 0:
            errors.append("Les dimensions doivent être positives")
        
        if self.width % 8 != 0 or self.height % 8 != 0:
            errors.append("Les dimensions doivent être multiples de 8")
        
        if self.lora_weight < 0.0 or self.lora_weight > 2.0:
            errors.append("LoRA weight doit être entre 0.0 et 2.0")
        
        if self.approach == "single_lora" and not self.lora_model:
            errors.append("Un modèle LoRA doit être spécifié pour l'approche single_lora")
        
        if errors:
            print("❌ Erreurs de validation:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        return True

@dataclass
class GenerationResult:
    """Résultat d'une génération"""
    
    # Métadonnées de base
    id: str
    params: GenerationParams
    status: str = "pending"  # pending, processing, completed, failed
    
    # Résultats
    image_path: Optional[str] = None
    s3_url: Optional[str] = None
    error_message: Optional[str] = None
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    generation_time: Optional[float] = None
    
    # Métadonnées additionnelles
    model_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour stockage"""
        result = asdict(self)
        
        # Convertir les datetime en string
        if self.start_time:
            result['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result['end_time'] = self.end_time.isoformat()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationResult':
        """Crée une instance depuis un dictionnaire"""
        # Convertir les strings en datetime
        if 'start_time' in data and data['start_time']:
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if 'end_time' in data and data['end_time']:
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        
        # Reconstruire les paramètres
        if 'params' in data:
            data['params'] = GenerationParams.from_dict(data['params'])
        
        return cls(**data)
    
    def mark_started(self):
        """Marque le début de la génération"""
        self.status = "processing"
        self.start_time = datetime.now()
    
    def mark_completed(self, image_path: str, s3_url: str = None):
        """Marque la fin de la génération avec succès"""
        self.status = "completed"
        self.end_time = datetime.now()
        self.image_path = image_path
        self.s3_url = s3_url
        
        if self.start_time:
            self.generation_time = (self.end_time - self.start_time).total_seconds()
    
    def mark_failed(self, error_message: str):
        """Marque l'échec de la génération"""
        self.status = "failed"
        self.end_time = datetime.now()
        self.error_message = error_message
        
        if self.start_time:
            self.generation_time = (self.end_time - self.start_time).total_seconds()
    
    def get_duration_str(self) -> str:
        """Retourne la durée formatée"""
        if self.generation_time:
            return f"{self.generation_time:.1f}s"
        return "N/A"