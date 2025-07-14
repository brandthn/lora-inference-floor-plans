#!/usr/bin/env python3
"""
Script de setup pour Floor Plan Generator
"""

import os
import sys
from pathlib import Path

def create_directories():
    """Crée les répertoires nécessaires"""
    directories = [
        "models/Wall_Lora_2",
        "models/floor_plans_a_v1", 
        "outputs",
        "app"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Répertoire créé: {directory}")

def setup_env_file():
    """Configure le fichier .env"""
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if not env_path.exists() and env_example_path.exists():
        # Copier .env.example vers .env
        with open(env_example_path, 'r') as f:
            content = f.read()
        
        with open(env_path, 'w') as f:
            f.write(content)
        
        print("✓ Fichier .env créé depuis .env.example")
        print("⚠️  Veuillez éditer .env avec vos credentials AWS")
    else:
        print("✓ Fichier .env déjà existant")

def check_model_files():
    """Vérifie la présence des fichiers modèles"""
    wall_lora_path = Path("models/Wall_Lora_2")
    floor_plan_lora_path = Path("models/floor_plans_a_v1")
    
    if not any(wall_lora_path.glob("*.safetensors")):
        print("⚠️  Fichiers Wall_Lora_2 non trouvés dans models/Wall_Lora_2/")
        print("   Veuillez copier vos fichiers .safetensors dans ce répertoire")
    else:
        print("✓ Fichiers Wall_Lora_2 trouvés")
    
    if not any(floor_plan_lora_path.glob("*.safetensors")):
        print("⚠️  Fichiers floor_plans_a_v1 non trouvés dans models/floor_plans_a_v1/")
        print("   Veuillez copier vos fichiers .safetensors dans ce répertoire")
    else:
        print("✓ Fichiers floor_plans_a_v1 trouvés")

def install_requirements():
    """Installe les dépendances"""
    try:
        import subprocess
        print("Installation des dépendances...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✓ Dépendances installées avec succès")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'installation des dépendances: {e}")
        return False
    return True

def main():
    """Setup principal"""
    print("🏠 Floor Plan Generator - Setup")
    print("=" * 40)
    
    # Créer les répertoires
    create_directories()
    
    # Configurer .env
    setup_env_file()
    
    # Vérifier les modèles
    check_model_files()
    
    # Installer les dépendances
    if input("\nInstaller les dépendances Python ? (y/n): ").lower() == 'y':
        install_requirements()
    
    print("\n" + "=" * 40)
    print("Setup terminé ! 🎉")
    print("\nProchaines étapes:")
    print("1. Placez vos fichiers LoRA dans les répertoires models/")
    print("2. Configurez vos credentials AWS dans .env")
    print("3. Lancez l'application avec: ./run_streamlit.sh")
    print("   ou: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()