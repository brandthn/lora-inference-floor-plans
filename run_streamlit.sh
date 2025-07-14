#!/bin/bash

# Script pour lancer l'application Streamlit
echo "Lancement de l'application Floor Plan Generator..."

# Vérifier si .env existe
if [ ! -f .env ]; then
    echo "Fichier .env non trouvé. Copiez .env.example vers .env et configurez vos variables."
    exit 1
fi

# Vérifier si les dépendances sont installées
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit n'est pas installé. Installez les dépendances avec: pip install -r requirements.txt"
    exit 1
fi

# Créer les répertoires nécessaires
mkdir -p models/Wall_Lora_2
mkdir -p models/floor_plans_a_v1
mkdir -p outputs

# Lancer Streamlit
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

echo "Application lancée sur http://localhost:8501"