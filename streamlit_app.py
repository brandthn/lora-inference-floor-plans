import streamlit as st
import time
from PIL import Image
import io
import logging
from pathlib import Path
import sys

# Ajouter le répertoire app au path
sys.path.append(str(Path(__file__).parent / "app"))

from app.inference import get_inference_model
from app.s3_utils import get_s3_utils
from app.config import STREAMLIT_CONFIG

# Configuration de la page
st.set_page_config(
    page_title=STREAMLIT_CONFIG["title"],
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_session_state():
    """Initialise les variables de session"""
    if 'inference_model' not in st.session_state:
        st.session_state.inference_model = None
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    if 'current_images' not in st.session_state:
        st.session_state.current_images = None

def load_model():
    """Charge le modèle d'inférence"""
    try:
        with st.spinner("Chargement des modèles LoRA... Cela peut prendre quelques minutes."):
            st.session_state.inference_model = get_inference_model()
        st.success("Modèles chargés avec succès!")
        return True
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {e}")
        logger.error(f"Erreur chargement modèles: {e}")
        return False

def generate_floor_plan(prompt, negative_prompt):
    """Génère un plan d'étage"""
    try:
        with st.spinner("Génération du plan d'étage en cours..."):
            # Barre de progression
            progress_bar = st.progress(0)
            
            # Génération complète (2 étapes)
            wall_image, floor_plan_image = st.session_state.inference_model.generate_complete_floor_plan(
                prompt=prompt,
                negative_prompt=negative_prompt
            )
            
            progress_bar.progress(100)
            
            # Stocker les images dans la session
            st.session_state.current_images = {
                'wall': wall_image,
                'floor_plan': floor_plan_image
            }
            
            return True
            
    except Exception as e:
        st.error(f"Erreur lors de la génération: {e}")
        logger.error(f"Erreur génération: {e}")
        return False

def upload_to_s3(images, prompt):
    """Upload les images vers S3"""
    try:
        with st.spinner("Upload vers S3 en cours..."):
            s3_utils = get_s3_utils()
            
            # Noms des fichiers (2 images)
            filenames = [
                f"wall_{prompt[:30].replace(' ', '_')}.png",
                f"floor_plan_{prompt[:30].replace(' ', '_')}.png"
            ]
            
            # Upload
            urls = s3_utils.upload_multiple_images(
                [images['wall'], images['floor_plan']],
                filenames
            )
            
            return urls
            
    except Exception as e:
        st.error(f"Erreur lors de l'upload S3: {e}")
        logger.error(f"Erreur upload S3: {e}")
        return [None, None]

def main():
    """Interface principale"""
    init_session_state()
    
    # Titre
    st.title(STREAMLIT_CONFIG["title"])
    st.markdown(STREAMLIT_CONFIG["description"])
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Chargement du modèle
        if st.session_state.inference_model is None:
            if st.button("Charger les modèles", type="primary"):
                if load_model():
                    st.rerun()
        else:
            st.success("✅ Modèles chargés")
        
        # Paramètres avancés
        st.header("Paramètres avancés")
        
        # Prompts exemples
        st.subheader("Prompts d'exemple")
        for i, example in enumerate(STREAMLIT_CONFIG["example_prompts"]):
            if st.button(f"Exemple {i+1}", key=f"example_{i}"):
                st.session_state.example_prompt = example
    
    # Interface principale
    if st.session_state.inference_model is not None:
        # Zone de saisie
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Prompt principal
            prompt = st.text_area(
                "Décrivez le plan d'étage souhaité",
                value=st.session_state.get('example_prompt', ''),
                height=100,
                max_chars=STREAMLIT_CONFIG["max_prompt_length"],
                help="Décrivez le type de plan d'étage que vous voulez générer"
            )
            
            # Prompt négatif
            negative_prompt = st.text_area(
                "Prompt négatif (optionnel)",
                value="blurry, low quality, distorted, watermark, text, people, furniture",
                height=70,
                help="Éléments à éviter dans la génération"
            )
        
        with col2:
            st.markdown("### Actions")
            
            # Bouton de génération
            if st.button("Générer le plan", type="primary", disabled=not prompt.strip()):
                if generate_floor_plan(prompt, negative_prompt):
                    st.session_state.generation_history.append({
                        'prompt': prompt,
                        'negative_prompt': negative_prompt,
                        'timestamp': time.time()
                    })
                    st.rerun()
            
            # Bouton d'upload S3
            if st.session_state.current_images is not None:
                if st.button("Upload vers S3", type="secondary"):
                    urls = upload_to_s3(st.session_state.current_images, prompt)
                    if any(urls):
                        st.success("Images uploadées avec succès!")
                        for i, url in enumerate(urls):
                            if url:
                                st.markdown(f"[Image {i+1}]({url})")
        
        # Affichage des résultats
        if st.session_state.current_images is not None:
            st.markdown("---")
            st.subheader("Résultats de la génération")
            
            # Afficher les images en colonnes (2 étapes)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Étape 1: Wall LoRA (avec contours/edges)**")
                st.image(st.session_state.current_images['wall'], use_column_width=True)
                
                # Bouton de téléchargement
                buf = io.BytesIO()
                st.session_state.current_images['wall'].save(buf, format='PNG')
                st.download_button(
                    label="Télécharger",
                    data=buf.getvalue(),
                    file_name="wall_image.png",
                    mime="image/png"
                )
            
            with col2:
                st.markdown("**Étape 2: Plan d'étage final**")
                st.image(st.session_state.current_images['floor_plan'], use_column_width=True)
                
                # Bouton de téléchargement
                buf = io.BytesIO()
                st.session_state.current_images['floor_plan'].save(buf, format='PNG')
                st.download_button(
                    label="Télécharger",
                    data=buf.getvalue(),
                    file_name="floor_plan.png",
                    mime="image/png"
                )
        
        # Historique des générations
        if st.session_state.generation_history:
            st.markdown("---")
            st.subheader("Historique des générations")
            
            for i, generation in enumerate(reversed(st.session_state.generation_history[-5:])):
                with st.expander(f"Génération {len(st.session_state.generation_history) - i}"):
                    st.markdown(f"**Prompt:** {generation['prompt']}")
                    if generation['negative_prompt']:
                        st.markdown(f"**Prompt négatif:** {generation['negative_prompt']}")
                    st.markdown(f"**Timestamp:** {time.ctime(generation['timestamp'])}")
    
    else:
        st.info("Veuillez charger les modèles pour commencer à générer des plans d'étage.")

if __name__ == "__main__":
    main()