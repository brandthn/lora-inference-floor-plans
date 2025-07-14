#!/usr/bin/env python3
"""
Script de vérification pour Floor Plan Generator
"""

import sys
import os
from pathlib import Path
import importlib

def check_dependencies():
    """Vérifie les dépendances Python"""
    print("🔍 Vérification des dépendances...")
    
    required_packages = [
        "torch", "diffusers", "transformers", "accelerate", 
        "peft", "controlnet_aux", "safetensors", "PIL", 
        "cv2", "numpy", "streamlit", "boto3"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "PIL":
                importlib.import_module("PIL")
            elif package == "cv2":
                importlib.import_module("cv2")
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MANQUANT")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Dépendances manquantes: {', '.join(missing_packages)}")
        print("Installez-les avec: pip install -r requirements.txt")
        return False
    
    print("✅ Toutes les dépendances sont installées")
    return True

def check_pytorch_mps():
    """Vérifie le support MPS pour PyTorch"""
    print("\n🔍 Vérification PyTorch MPS...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ MPS disponible - GPU Apple Silicon détecté")
            return True
        else:
            print("⚠️  MPS non disponible - utilisation CPU")
            return False
    except ImportError:
        print("❌ PyTorch non installé")
        return False

def check_model_files():
    """Vérifie la présence des fichiers modèles"""
    print("\n🔍 Vérification des fichiers modèles...")
    
    wall_lora_path = Path("models/Wall_Lora_2")
    floor_plan_lora_path = Path("models/floor_plans_a_v1")
    
    # Vérifier Wall_Lora_2
    wall_files = list(wall_lora_path.glob("*.safetensors"))
    if wall_files:
        print(f"✅ Wall_Lora_2 trouvé: {[f.name for f in wall_files]}")
    else:
        print(f"❌ Wall_Lora_2 manquant dans {wall_lora_path}")
        print("   Copiez vos fichiers .safetensors dans models/Wall_Lora_2/")
    
    # Vérifier floor_plans_a_v1
    floor_files = list(floor_plan_lora_path.glob("*.safetensors"))
    if floor_files:
        print(f"✅ floor_plans_a_v1 trouvé: {[f.name for f in floor_files]}")
    else:
        print(f"❌ floor_plans_a_v1 manquant dans {floor_plan_lora_path}")
        print("   Copiez vos fichiers .safetensors dans models/floor_plans_a_v1/")
    
    return bool(wall_files and floor_files)

def check_env_file():
    """Vérifie le fichier .env"""
    print("\n🔍 Vérification configuration .env...")
    
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ Fichier .env manquant")
        print("   Copiez .env.example vers .env et configurez vos variables")
        return False
    
    # Vérifier les variables importantes
    with open(env_path, 'r') as f:
        content = f.read()
    
    required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_BUCKET_NAME"]
    missing_vars = []
    
    for var in required_vars:
        if var not in content or f"{var}=your_" in content:
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Variables non configurées: {', '.join(missing_vars)}")
        print("   Editez .env avec vos vraies valeurs AWS")
    else:
        print("✅ Configuration .env OK")
    
    return True

def check_directories():
    """Vérifie les répertoires nécessaires"""
    print("\n🔍 Vérification des répertoires...")
    
    directories = [
        "models/Wall_Lora_2",
        "models/floor_plans_a_v1",
        "outputs",
        "app"
    ]
    
    for directory in directories:
        path = Path(directory)
        if path.exists():
            print(f"✅ {directory}")
        else:
            print(f"❌ {directory} - MANQUANT")
            path.mkdir(parents=True, exist_ok=True)
            print(f"   Créé: {directory}")
    
    return True

def main():
    """Vérification principale"""
    print("🏠 Floor Plan Generator - Vérification du setup")
    print("=" * 50)
    
    all_ok = True
    
    # Vérifications
    all_ok &= check_dependencies()
    all_ok &= check_pytorch_mps()
    all_ok &= check_directories()
    all_ok &= check_model_files()
    all_ok &= check_env_file()
    
    print("\n" + "=" * 50)
    
    if all_ok:
        print("🎉 Setup complet ! Vous pouvez lancer l'application.")
        print("Commande: ./run_streamlit.sh")
    else:
        print("❌ Setup incomplet. Corrigez les erreurs ci-dessus.")
        print("\nActions recommandées:")
        print("1. pip install -r requirements.txt")
        print("2. Copiez vos fichiers LoRA dans models/")
        print("3. Configurez .env avec vos credentials AWS")
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)