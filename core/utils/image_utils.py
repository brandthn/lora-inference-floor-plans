import os
import json
import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

def ensure_output_dir():
    """Crée le dossier de sortie s'il n'existe pas"""
    output_dir = Path("data/generations")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def save_image(image: Image.Image, filename: str, metadata: Dict[str, Any] = None) -> str:
    """Sauvegarde une image avec ses métadonnées"""
    output_dir = ensure_output_dir()
    image_path = output_dir / filename
    
    # Sauvegarder l'image
    image.save(image_path, format='PNG', optimize=True)
    
    # Sauvegarder les métadonnées si fournies
    if metadata:
        metadata_path = output_dir / f"{Path(filename).stem}_metadata.json"
        
        # Ajouter des métadonnées système
        full_metadata = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "image_size": image.size,
            "image_mode": image.mode,
            **metadata
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(full_metadata, f, indent=2, ensure_ascii=False)
    
    return str(image_path)

def load_image(image_path: str) -> Optional[Image.Image]:
    """Charge une image depuis un chemin"""
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'image {image_path}: {e}")
        return None

def create_canny_image(image: Image.Image, low_threshold: int = 50, high_threshold: int = 150) -> Image.Image:
    """Crée une image canny edge à partir d'une image PIL"""
    # Convertir en array numpy
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_array = np.array(image)
    
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Appliquer le filtre Canny
    canny = cv2.Canny(gray, low_threshold, high_threshold)
    
    # Convertir en image RGB (3 canaux)
    canny_rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    
    # Convertir en image PIL
    canny_image = Image.fromarray(canny_rgb)
    
    return canny_image

def resize_image(image: Image.Image, width: int, height: int, maintain_aspect: bool = True) -> Image.Image:
    """Redimensionne une image"""
    if maintain_aspect:
        # Maintenir le ratio d'aspect
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        
        # Créer une nouvelle image avec les dimensions exactes et coller l'image redimensionnée
        new_image = Image.new('RGB', (width, height), (255, 255, 255))
        
        # Centrer l'image
        x = (width - image.width) // 2
        y = (height - image.height) // 2
        new_image.paste(image, (x, y))
        
        return new_image
    else:
        # Redimensionner directement
        return image.resize((width, height), Image.Resampling.LANCZOS)

def create_image_grid(images: list, cols: int = 2, spacing: int = 10) -> Image.Image:
    """Crée une grille d'images"""
    if not images:
        return None
    
    # Calculer les dimensions
    rows = (len(images) + cols - 1) // cols
    img_width, img_height = images[0].size
    
    grid_width = cols * img_width + (cols - 1) * spacing
    grid_height = rows * img_height + (rows - 1) * spacing
    
    # Créer l'image de grille
    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    # Coller les images
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        
        x = col * (img_width + spacing)
        y = row * (img_height + spacing)
        
        grid.paste(img, (x, y))
    
    return grid

def add_watermark(image: Image.Image, text: str, position: str = "bottom-right", 
                  font_size: int = 20, opacity: int = 128) -> Image.Image:
    """Ajoute un watermark à une image"""
    from PIL import ImageDraw, ImageFont
    
    # Créer une copie de l'image
    watermarked = image.copy()
    
    # Créer un overlay transparent
    overlay = Image.new('RGBA', watermarked.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Essayer de charger une police, sinon utiliser la police par défaut
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Calculer la position du texte
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    if position == "bottom-right":
        x = watermarked.width - text_width - 10
        y = watermarked.height - text_height - 10
    elif position == "bottom-left":
        x = 10
        y = watermarked.height - text_height - 10
    elif position == "top-right":
        x = watermarked.width - text_width - 10
        y = 10
    elif position == "top-left":
        x = 10
        y = 10
    else:  # center
        x = (watermarked.width - text_width) // 2
        y = (watermarked.height - text_height) // 2
    
    # Dessiner le texte
    draw.text((x, y), text, font=font, fill=(0, 0, 0, opacity))
    
    # Combiner avec l'image originale
    watermarked = Image.alpha_composite(watermarked.convert('RGBA'), overlay)
    
    return watermarked.convert('RGB')

def enhance_image(image: Image.Image, brightness: float = 1.0, contrast: float = 1.0, 
                  saturation: float = 1.0, sharpness: float = 1.0) -> Image.Image:
    """Améliore une image avec des ajustements de base"""
    from PIL import ImageEnhance
    
    enhanced = image.copy()
    
    # Luminosité
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = enhancer.enhance(brightness)
    
    # Contraste
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(contrast)
    
    # Saturation
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(saturation)
    
    # Netteté
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(sharpness)
    
    return enhanced

def get_image_info(image_path: str) -> Dict[str, Any]:
    """Retourne des informations sur une image"""
    try:
        with Image.open(image_path) as img:
            return {
                "filename": os.path.basename(image_path),
                "size": img.size,
                "mode": img.mode,
                "format": img.format,
                "file_size": os.path.getsize(image_path),
                "exists": True
            }
    except Exception as e:
        return {
            "filename": os.path.basename(image_path),
            "error": str(e),
            "exists": False
        }

def cleanup_old_images(max_age_days: int = 7):
    """Nettoie les anciennes images générées"""
    output_dir = Path("data/generations")
    if not output_dir.exists():
        return
    
    cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
    deleted_count = 0
    
    for file_path in output_dir.iterdir():
        if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"❌ Erreur lors de la suppression de {file_path}: {e}")
    
    if deleted_count > 0:
        print(f"🗑️  {deleted_count} fichiers anciens supprimés")

def create_comparison_image(images: list, titles: list = None, spacing: int = 20) -> Image.Image:
    """Crée une image de comparaison côte à côte"""
    if not images:
        return None
    
    # S'assurer que toutes les images ont la même taille
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    
    resized_images = []
    for img in images:
        if img.size != (max_width, max_height):
            resized_img = resize_image(img, max_width, max_height, maintain_aspect=True)
            resized_images.append(resized_img)
        else:
            resized_images.append(img)
    
    # Calculer les dimensions de l'image finale
    total_width = len(resized_images) * max_width + (len(resized_images) - 1) * spacing
    total_height = max_height
    
    # Ajouter de l'espace pour les titres si fournis
    title_height = 30 if titles else 0
    total_height += title_height
    
    # Créer l'image de comparaison
    comparison = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    # Coller les images
    x_offset = 0
    for i, img in enumerate(resized_images):
        y_offset = title_height
        comparison.paste(img, (x_offset, y_offset))
        
        # Ajouter le titre si fourni
        if titles and i < len(titles):
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(comparison)
            
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            title = titles[i]
            text_bbox = draw.textbbox((0, 0), title, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            
            text_x = x_offset + (max_width - text_width) // 2
            text_y = 5
            
            draw.text((text_x, text_y), title, font=font, fill=(0, 0, 0))
        
        x_offset += max_width + spacing
    
    return comparison