#!/usr/bin/env python3
"""
Script de setup pour Floor Plan Generator

Ce script configure l'environnement et vérifie les dépendances.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Vérifie la version de Python"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 ou supérieur requis")
        print(f"Version actuelle: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} détecté")
    return True

def check_cuda():
    """Vérifie la disponibilité de CUDA"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA disponible")
            return True
        else:
            print("⚠️  CUDA non détecté - utilisation du CPU")
            return False
    except FileNotFoundError:
        print("⚠️  NVIDIA-SMI non trouvé - utilisation du CPU")
        return False

def create_directory_structure():
    """Crée la structure de dossiers nécessaire"""
    directories = [
        "config",
        "data/models/base",
        "data/models/lora",
        "data/models/controlnet",
        "data/generations",
        "core/utils",
        "app/pages",
        "app/components"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Créé: {directory}")

def create_config_files():
    """Crée les fichiers de configuration par défaut"""
    
    # Fichier .env
    env_content = """
# Configuration pour Floor Plan Generator

# Chemins des modèles (à adapter selon votre installation)
BASE_MODEL_PATH=data/models/base
LORA_MODEL_PATH=data/models/lora
CONTROLNET_MODEL_PATH=data/models/controlnet

# Configuration AWS S3 (optionnel)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
S3_BUCKET_NAME=

# Configuration de l'application
MAX_WORKERS=1
DEBUG=false
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Fichier .env créé")

def install_requirements():
    """Installe les dépendances"""
    print("📦 Installation des dépendances...")
    
    # Utiliser le script d'installation personnalisé
    try:
        subprocess.run([sys.executable, "install_dependencies.py"], check=True)
        print("✅ Dépendances installées avec succès")
        return True
    except subprocess.CalledProcessError:
        print("❌ Erreur lors de l'installation des dépendances")
        print("💡 Essayez d'exécuter manuellement: python install_dependencies.py")
        return False

def check_model_files():
    """Vérifie la présence des fichiers modèles"""
    print("\n📋 Vérification des modèles...")
    
    required_files = {
        "Modèles de base": [
            "data/models/base/sd_xl_base_1.0.safetensors",
            "data/models/base/xl_juggernautXL_juggXIByRundiffusion.safetensors"
        ],
        "Modèles LoRA": [
            "data/models/lora/lora_plan_v1.safetensors",
            "data/models/lora/lora_plan_v2.safetensors", 
            "data/models/lora/wall_lora.safetensors"
        ],
        "Modèles ControlNet": [
            "data/models/controlnet/control_canny.safetensors"
        ]
    }
    
    missing_files = []
    
    for category, files in required_files.items():
        print(f"\n{category}:")
        for file_path in files:
            if os.path.exists(file_path):
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ {file_path} (manquant)")
                missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  {len(missing_files)} fichiers modèles manquants")
        print("📝 Placez vos modèles dans les dossiers appropriés selon la configuration")
        return False
    
    print("\n✅ Tous les modèles requis sont présents")
    return True

def create_launch_script():
    """Crée un script de lancement"""
    
    # Script pour Linux/macOS
    launch_sh = """#!/bin/bash
echo "🚀 Lancement de Floor Plan Generator..."
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
"""
    
    with open("launch.sh", "w") as f:
        f.write(launch_sh)
    
    # Rendre exécutable
    os.chmod("launch.sh", 0o755)
    
    # Script pour Windows
    launch_bat = """@echo off
echo 🚀 Lancement de Floor Plan Generator...
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
pause
"""
    
    with open("launch.bat", "w") as f:
        f.write(launch_bat)
    
    print("✅ Scripts de lancement créés (launch.sh et launch.bat)")

def main():
    """Fonction principale de setup"""
    print("🏠 Floor Plan Generator - Script de Setup")
    print("=" * 50)
    
    # Vérifications préliminaires
    if not check_python_version():
        return False
    
    check_cuda()
    
    print("\n📁 Création de la structure de dossiers...")
    create_directory_structure()
    
    print("\n⚙️  Création des fichiers de configuration...")
    create_config_files()
    
    print("\n📦 Installation des dépendances...")
    if not install_requirements():
        return False
    
    print("\n🚀 Création des scripts de lancement...")
    create_launch_script()
    
    print("\n📋 Vérification des modèles...")
    models_ok = check_model_files()
    
    print("\n" + "=" * 50)
    print("📊 Résumé du setup:")
    print("✅ Structure de dossiers créée")
    print("✅ Configuration générée")
    print("✅ Dépendances installées")
    print("✅ Scripts de lancement créés")
    
    if models_ok:
        print("✅ Modèles vérifiés")
        print("\n🎉 Setup terminé avec succès!")
        print("🚀 Lancez l'application avec: ./launch.sh (Linux/macOS) ou launch.bat (Windows)")
        print("🌐 Ou directement: streamlit run streamlit_app.py")
    else:
        print("⚠️  Modèles manquants")
        print("\n📝 Prochaines étapes:")
        print("1. Placez vos modèles dans les dossiers appropriés")
        print("2. Vérifiez la configuration dans config/models_config.yaml")
        print("3. Lancez l'application avec: ./launch.sh ou launch.bat")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)