#!/usr/bin/env python3
"""
Script d'installation pour Floor Plan Generator
Gère les dépendances spécifiques et les problèmes de compatibilité
"""

import subprocess
import sys
import platform
import importlib.util

def check_module_installed(module_name):
    """Vérifie si un module est installé"""
    spec = importlib.util.find_spec(module_name)
    return spec is not None

def install_package(package):
    """Installe un package via pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def install_torch_nightly():
    """Installe PyTorch nightly"""
    print("🔥 Installation de PyTorch nightly...")
    
    # Détecter le système et l'architecture
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "darwin" and ("arm" in machine or "aarch64" in machine):
        # macOS Apple Silicon
        torch_url = "https://download.pytorch.org/whl/nightly/cpu"
        print("🍎 Détecté: macOS Apple Silicon")
    elif system == "darwin":
        # macOS Intel
        torch_url = "https://download.pytorch.org/whl/nightly/cpu"
        print("🍎 Détecté: macOS Intel")
    elif system == "linux":
        # Linux - vérifier CUDA
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True)
            torch_url = "https://download.pytorch.org/whl/nightly/cu121"
            print("🐧 Détecté: Linux avec CUDA")
        except:
            torch_url = "https://download.pytorch.org/whl/nightly/cpu"
            print("🐧 Détecté: Linux sans CUDA")
    else:
        # Windows
        torch_url = "https://download.pytorch.org/whl/nightly/cu121"
        print("🪟 Détecté: Windows")
    
    # Installer PyTorch nightly
    cmd = [
        sys.executable, "-m", "pip", "install", 
        "--pre", "torch", "torchvision", "torchaudio",
        "--index-url", torch_url
    ]
    
    try:
        subprocess.check_call(cmd)
        print("✅ PyTorch nightly installé avec succès")
        return True
    except subprocess.CalledProcessError:
        print("❌ Erreur lors de l'installation de PyTorch nightly")
        return False

def install_safetensors():
    """Installe safetensors avec gestion d'erreurs"""
    print("🔐 Installation de safetensors...")
    
    # Essayer d'abord la version stable
    packages_to_try = [
        "safetensors>=0.3.0",
        "safetensors==0.4.2",
        "safetensors==0.4.1",
        "safetensors==0.4.0",
        "safetensors==0.3.3"
    ]
    
    for package in packages_to_try:
        print(f"   Tentative: {package}")
        if install_package(package):
            print(f"✅ {package} installé avec succès")
            return True
        else:
            print(f"❌ Échec: {package}")
    
    print("❌ Impossible d'installer safetensors")
    return False

def install_core_dependencies():
    """Installe les dépendances principales"""
    print("📦 Installation des dépendances principales...")
    
    # Dépendances critiques à installer individuellement
    critical_deps = [
        "streamlit>=1.28.0",
        "diffusers>=0.21.0", 
        "transformers>=4.21.0",
        "accelerate>=0.20.0",
        "peft>=0.5.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0",
        "requests>=2.31.0",
        "tqdm>=4.65.0",
        "psutil>=5.9.0"
    ]
    
    failed_packages = []
    
    for package in critical_deps:
        print(f"   Installation: {package}")
        if install_package(package):
            print(f"   ✅ {package}")
        else:
            print(f"   ❌ {package}")
            failed_packages.append(package)
    
    return failed_packages

def install_optional_dependencies():
    """Installe les dépendances optionnelles"""
    print("🎁 Installation des dépendances optionnelles...")
    
    optional_deps = [
        "boto3>=1.28.0",
        "python-dotenv>=1.0.0",
        "streamlit-option-menu>=0.3.6"
    ]
    
    for package in optional_deps:
        print(f"   Installation: {package}")
        if install_package(package):
            print(f"   ✅ {package}")
        else:
            print(f"   ⚠️  {package} (optionnel)")

def verify_installation():
    """Vérifie que les modules critiques sont installés"""
    print("🔍 Vérification de l'installation...")
    
    critical_modules = [
        "streamlit",
        "torch",
        "diffusers", 
        "transformers",
        "peft",
        "PIL",
        "cv2",
        "numpy",
        "yaml",
        "safetensors"
    ]
    
    missing = []
    
    for module in critical_modules:
        if check_module_installed(module):
            print(f"   ✅ {module}")
        else:
            print(f"   ❌ {module}")
            missing.append(module)
    
    # Vérifier sqlite3 (intégré à Python)
    if check_module_installed("sqlite3"):
        print("   ✅ sqlite3 (intégré)")
    else:
        print("   ❌ sqlite3 (problème Python)")
        missing.append("sqlite3")
    
    return missing

def main():
    """Fonction principale d'installation"""
    print("🚀 Installation des dépendances pour Floor Plan Generator")
    print("=" * 60)
    
    # Vérifier si nous sommes dans un environnement virtuel
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Environnement virtuel détecté")
    else:
        print("⚠️  Aucun environnement virtuel détecté")
        response = input("Continuer quand même ? (y/N): ")
        if response.lower() != 'y':
            print("Installation annulée")
            return False
    
    print(f"🐍 Python {sys.version}")
    print(f"💻 Système: {platform.system()} {platform.machine()}")
    
    # Étape 1: Vérifier si PyTorch nightly est déjà installé
    if check_module_installed("torch"):
        print("✅ PyTorch déjà installé")
        try:
            import torch
            print(f"   Version: {torch.__version__}")
        except ImportError:
            print("   ⚠️  PyTorch installé mais non importable")
    else:
        print("📦 PyTorch non trouvé, installation...")
        if not install_torch_nightly():
            print("❌ Impossible d'installer PyTorch")
            return False
    
    # Étape 2: Installer safetensors
    if not check_module_installed("safetensors"):
        if not install_safetensors():
            print("⚠️  safetensors non installé, certaines fonctionnalités peuvent ne pas marcher")
    else:
        print("✅ safetensors déjà installé")
    
    # Étape 3: Installer les dépendances principales
    failed = install_core_dependencies()
    
    # Étape 4: Installer les dépendances optionnelles
    install_optional_dependencies()
    
    # Étape 5: Vérification finale
    missing = verify_installation()
    
    print("\n" + "=" * 60)
    print("📊 Résumé de l'installation:")
    
    if not missing:
        print("✅ Toutes les dépendances critiques sont installées")
        print("🎉 Installation terminée avec succès!")
        print("\n🚀 Vous pouvez maintenant lancer:")
        print("   streamlit run streamlit_app.py")
        return True
    else:
        print("❌ Dépendances manquantes:")
        for module in missing:
            print(f"   - {module}")
        print("\n🔧 Solutions possibles:")
        print("   1. Réexécuter ce script")
        print("   2. Installer manuellement: pip install <package>")
        print("   3. Vérifier votre environnement Python")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)