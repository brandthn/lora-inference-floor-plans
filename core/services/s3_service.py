import boto3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid

from dotenv import load_dotenv

from ..generation_params import GenerationResult, GenerationParams
from ..utils.prompt_hash import generate_prompt_hash, extract_prompt_structure, PromptStructure

load_dotenv() 

@dataclass
class S3Paths:
    """Chemins S3 pour une génération"""
    main_image: str
    metadata: str
    debug_images: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class S3Metadata:
    """Métadonnées enrichies pour S3"""
    generation_id: str
    timestamp: str
    prompt_info: Dict[str, Any]
    approach: str
    model_config: Dict[str, Any]
    generation_params: Dict[str, Any]
    s3_paths: S3Paths
    generation_time: float
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ComparisonGroup:
    """Groupe de comparaison pour le même prompt"""
    comparison_id: str
    prompt_hash: str
    prompt_structure: Dict[str, Any]
    generations: List[Dict[str, Any]]
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class S3Service:
    """Service pour la gestion des uploads et métadonnées S3"""
    
    def __init__(self, bucket_name: str = None):
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET_NAME', 'floor-plan-gallery-3344')
        
        # Configuration AWS
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'eu-west-1')
        )
        
        # Vérifier la connectivité
        self._verify_connection()
    
    def _verify_connection(self):
        """Vérifie la connexion S3"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"✅ Connexion S3 vérifiée pour le bucket: {self.bucket_name}")
        except Exception as e:
            print(f"⚠️  Connexion S3 échouée: {e}")
            print("🔧 Vérifiez vos credentials AWS et le nom du bucket")
    
    def _generate_s3_key(self, approach: str, generation_id: str, file_type: str = "png") -> str:
        """Génère une clé S3 pour un fichier"""
        if file_type == "metadata":
            return f"metadata/by_generation/{generation_id}.json"
        elif file_type == "comparison":
            return f"metadata/comparisons/{generation_id}.json"
        elif file_type == "debug":
            return f"images/debug/{generation_id}.{file_type}"
        else:
            return f"images/by_approach/{approach}/{generation_id}.{file_type}"
    
    def _upload_file(self, file_path: str, s3_key: str, content_type: str = None) -> str:
        """Upload un fichier vers S3"""
        try:
            # Déterminer le content type
            if content_type is None:
                if s3_key.endswith('.png'):
                    content_type = 'image/png'
                elif s3_key.endswith('.jpg') or s3_key.endswith('.jpeg'):
                    content_type = 'image/jpeg'
                elif s3_key.endswith('.json'):
                    content_type = 'application/json'
                else:
                    content_type = 'binary/octet-stream'
            
            # Upload le fichier
            self.s3_client.upload_file(
                file_path,
                self.bucket_name,
                s3_key,
                ExtraArgs={'ContentType': content_type}
            )
            
            # Générer l'URL
            url = f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}"
            print(f"✅ Fichier uploadé: {s3_key}")
            return url
            
        except Exception as e:
            print(f"❌ Erreur lors de l'upload de {s3_key}: {e}")
            raise
    
    def _upload_json_data(self, data: Dict[str, Any], s3_key: str) -> str:
        """Upload des données JSON directement vers S3"""
        try:
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_str.encode('utf-8'),
                ContentType='application/json'
            )
            
            url = f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}"
            print(f"✅ JSON uploadé: {s3_key}")
            return url
            
        except Exception as e:
            print(f"❌ Erreur lors de l'upload JSON {s3_key}: {e}")
            raise
    
    def create_s3_metadata(self, result: GenerationResult) -> S3Metadata:
        """Crée les métadonnées S3 à partir d'un résultat de génération"""
        
        # Extraire la structure du prompt
        prompt_structure = extract_prompt_structure(result.params.prompt)
        prompt_hash = generate_prompt_hash(result.params.prompt)
        
        # Créer les informations du prompt
        prompt_info = {
            "original": result.params.prompt,
            "normalized": prompt_structure.normalized_prompt,
            "hash": prompt_hash,
            "structure": {
                "rooms": prompt_structure.rooms,
                "locations": prompt_structure.locations,
                "counts": prompt_structure.counts,
                "features": prompt_structure.features
            }
        }
        
        # Configuration du modèle
        model_config = {
            "base_model": result.params.base_model,
            "approach": result.params.approach
        }
        
        if result.params.approach == "single_lora":
            model_config.update({
                "lora_model": result.params.lora_model,
                "lora_weight": result.params.lora_weight
            })
        else:  # combined_approach
            model_config.update({
                "wall_lora_weight": result.params.wall_lora_weight,
                "plan_lora_weight": result.params.plan_lora_weight,
                "controlnet_weight": result.params.controlnet_weight
            })
        
        # Paramètres de génération
        generation_params = {
            "steps": result.params.steps,
            "cfg_scale": result.params.cfg_scale,
            "sampler": result.params.sampler,
            "seed": result.params.seed,
            "width": result.params.width,
            "height": result.params.height,
            "negative_prompt": result.params.negative_prompt
        }
        
        # Génération des chemins S3 (seront mis à jour après upload)
        s3_paths = S3Paths(
            main_image="",  # À remplir après upload
            metadata="",   # À remplir après upload
            debug_images=[]
        )
        
        # Génération des tags
        tags = self._generate_tags(prompt_structure, result.params)
        
        return S3Metadata(
            generation_id=result.id,
            timestamp=datetime.now().isoformat(),
            prompt_info=prompt_info,
            approach=result.params.approach,
            model_config=model_config,
            generation_params=generation_params,
            s3_paths=s3_paths,
            generation_time=result.generation_time or 0.0,
            tags=tags
        )
    
    def _generate_tags(self, prompt_structure: PromptStructure, params: GenerationParams) -> List[str]:
        """Génère des tags pour la génération"""
        tags = []
        
        # Tags basés sur le prompt
        tags.extend(prompt_structure.rooms)
        tags.extend(prompt_structure.features)
        
        # Tags basés sur les paramètres
        tags.append(params.approach)
        tags.append(params.base_model)
        
        if params.lora_model:
            tags.append(params.lora_model)
        
        # Tags basés sur les dimensions
        if params.width == params.height:
            tags.append("square")
        elif params.width > params.height:
            tags.append("landscape")
        else:
            tags.append("portrait")
        
        # Tags basés sur la qualité
        if params.steps >= 40:
            tags.append("high_quality")
        elif params.steps <= 20:
            tags.append("fast_generation")
        
        return list(set(tags))  # Supprimer les doublons
    
    def upload_generation(self, result: GenerationResult) -> S3Metadata:
        """Upload une génération complète vers S3"""
        
        print(f"📤 Upload de la génération {result.id[:8]} vers S3...")
        
        # Créer les métadonnées
        metadata = self.create_s3_metadata(result)
        
        try:
            # 1. Upload de l'image principale
            if result.image_path and os.path.exists(result.image_path):
                main_image_key = self._generate_s3_key(
                    result.params.approach, 
                    result.id, 
                    "png"
                )
                main_image_url = self._upload_file(result.image_path, main_image_key)
                metadata.s3_paths.main_image = main_image_url
            
            # 2. Upload des images de debug (si approche combinée)
            if result.params.approach == "combined_approach":
                debug_images = self._find_debug_images(result.id)
                debug_urls = []
                
                for debug_path in debug_images:
                    if os.path.exists(debug_path):
                        debug_filename = os.path.basename(debug_path)
                        debug_key = f"images/debug/{result.id}_{debug_filename}"
                        debug_url = self._upload_file(debug_path, debug_key)
                        debug_urls.append(debug_url)
                
                metadata.s3_paths.debug_images = debug_urls
            
            # 3. Upload des métadonnées
            metadata_key = self._generate_s3_key("", result.id, "metadata")
            metadata_url = self._upload_json_data(metadata.to_dict(), metadata_key)
            metadata.s3_paths.metadata = metadata_url
            
            # 4. Mise à jour des index
            self._update_indexes(metadata)
            
            # 5. Gestion des comparaisons
            self._handle_comparisons(metadata)
            
            print(f"✅ Génération {result.id[:8]} uploadée avec succès")
            return metadata
            
        except Exception as e:
            print(f"❌ Erreur lors de l'upload de {result.id[:8]}: {e}")
            raise
    
    def _find_debug_images(self, generation_id: str) -> List[str]:
        """Trouve les images de debug pour une génération"""
        debug_images = []
        generation_dir = Path("data/generations")
        
        # Patterns pour les images de debug
        patterns = [
            f"*{generation_id}*_wall_*.png",
            f"*{generation_id}*_canny_*.png"
        ]
        
        for pattern in patterns:
            debug_images.extend(generation_dir.glob(pattern))
        
        return [str(path) for path in debug_images]
    
    def _update_indexes(self, metadata: S3Metadata):
        """Met à jour les index S3 pour les recherches rapides"""
        
        # Index par approche
        approach_index_key = f"indexes/by_approach/{metadata.approach}.json"
        self._append_to_index(approach_index_key, {
            "generation_id": metadata.generation_id,
            "timestamp": metadata.timestamp,
            "prompt_hash": metadata.prompt_info["hash"],
            "model": metadata.model_config,
            "image_url": metadata.s3_paths.main_image
        })
        
        # Index par prompt_hash
        prompt_hash_index_key = f"indexes/by_prompt_hash/{metadata.prompt_info['hash']}.json"
        self._append_to_index(prompt_hash_index_key, {
            "generation_id": metadata.generation_id,
            "timestamp": metadata.timestamp,
            "approach": metadata.approach,
            "model": metadata.model_config,
            "image_url": metadata.s3_paths.main_image
        })
        
        # Index récent
        recent_index_key = "indexes/recent.json"
        self._append_to_index(recent_index_key, {
            "generation_id": metadata.generation_id,
            "timestamp": metadata.timestamp,
            "approach": metadata.approach,
            "prompt_hash": metadata.prompt_info["hash"],
            "image_url": metadata.s3_paths.main_image
        }, max_entries=100)
    
    def _append_to_index(self, index_key: str, entry: Dict[str, Any], max_entries: int = None):
        """Ajoute une entrée à un index S3"""
        try:
            # Essayer de récupérer l'index existant
            try:
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=index_key)
                existing_data = json.loads(response['Body'].read().decode('utf-8'))
                entries = existing_data.get('entries', [])
            except self.s3_client.exceptions.NoSuchKey:
                entries = []
            
            # Ajouter la nouvelle entrée
            entries.append(entry)
            
            # Limiter le nombre d'entrées si spécifié
            if max_entries and len(entries) > max_entries:
                entries = sorted(entries, key=lambda x: x['timestamp'], reverse=True)[:max_entries]
            
            # Mettre à jour l'index
            index_data = {
                'last_updated': datetime.now().isoformat(),
                'count': len(entries),
                'entries': entries
            }
            
            self._upload_json_data(index_data, index_key)
            
        except Exception as e:
            print(f"⚠️  Erreur lors de la mise à jour de l'index {index_key}: {e}")
    
    def _handle_comparisons(self, metadata: S3Metadata):
        """Gère les comparaisons automatiques pour le même prompt"""
        prompt_hash = metadata.prompt_info["hash"]
        
        # Chercher les générations existantes avec le même prompt_hash
        similar_generations = self.get_generations_by_prompt_hash(prompt_hash)
        
        if len(similar_generations) > 1:
            # Créer ou mettre à jour le groupe de comparaison
            comparison_id = f"comp_{prompt_hash}"
            
            comparison_group = ComparisonGroup(
                comparison_id=comparison_id,
                prompt_hash=prompt_hash,
                prompt_structure=metadata.prompt_info["structure"],
                generations=[
                    {
                        "generation_id": gen["generation_id"],
                        "approach": gen["approach"],
                        "model": gen["model"],
                        "timestamp": gen["timestamp"],
                        "image_url": gen["image_url"]
                    }
                    for gen in similar_generations
                ],
                created_at=datetime.now().isoformat()
            )
            
            # Sauvegarder le groupe de comparaison
            comparison_key = self._generate_s3_key("", comparison_id, "comparison")
            self._upload_json_data(comparison_group.to_dict(), comparison_key)
            
            print(f"🔍 Comparaison créée pour prompt_hash {prompt_hash}")
    
    def get_generations_by_prompt_hash(self, prompt_hash: str) -> List[Dict[str, Any]]:
        """Récupère toutes les générations pour un prompt_hash donné"""
        try:
            index_key = f"indexes/by_prompt_hash/{prompt_hash}.json"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=index_key)
            index_data = json.loads(response['Body'].read().decode('utf-8'))
            return index_data.get('entries', [])
        except self.s3_client.exceptions.NoSuchKey:
            return []
        except Exception as e:
            print(f"❌ Erreur lors de la récupération des générations: {e}")
            return []
    
    def get_recent_generations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Récupère les générations récentes"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key="indexes/recent.json")
            index_data = json.loads(response['Body'].read().decode('utf-8'))
            entries = index_data.get('entries', [])
            return entries[:limit]
        except self.s3_client.exceptions.NoSuchKey:
            return []
        except Exception as e:
            print(f"❌ Erreur lors de la récupération des générations récentes: {e}")
            return []
    
    def get_comparison_group(self, prompt_hash: str) -> Optional[ComparisonGroup]:
        """Récupère un groupe de comparaison"""
        try:
            comparison_key = f"metadata/comparisons/comp_{prompt_hash}.json"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=comparison_key)
            data = json.loads(response['Body'].read().decode('utf-8'))
            return ComparisonGroup(**data)
        except self.s3_client.exceptions.NoSuchKey:
            return None
        except Exception as e:
            print(f"❌ Erreur lors de la récupération du groupe de comparaison: {e}")
            return None

# Instance globale du service S3
s3_service = S3Service()