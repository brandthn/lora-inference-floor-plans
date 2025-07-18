import streamlit as st
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / "core"))

from core.config import config_manager
from core.inference_engine import inference_engine
from core.queue_manager import generation_queue
from core.generation_params import GenerationParams
from core.utils.image_utils import load_image, create_comparison_image

# Configuration de la page
st.set_page_config(
    page_title="Floor Plan Generator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .status-pending {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    
    .status-processing {
        background-color: #cce5ff;
        border: 1px solid #74b9ff;
        color: #004085;
    }
    
    .status-completed {
        background-color: #d4edda;
        border: 1px solid #00b894;
        color: #155724;
    }
    
    .status-failed {
        background-color: #f8d7da;
        border: 1px solid #e74c3c;
        color: #721c24;
    }
    
    .model-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_app():
    """Initialise l'application"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    
    if not st.session_state.initialized:
        with st.spinner("🚀 Initialisation de l'application..."):
            # Créer les dossiers nécessaires
            config_manager.create_directories()
            
            # Configurer la file d'attente
            generation_queue.set_generation_function(inference_engine.generate)
            generation_queue.start()
            
            st.session_state.initialized = True
            st.success("✅ Application initialisée avec succès!")

def render_sidebar():
    """Rendu de la sidebar avec les paramètres"""
    st.sidebar.title("⚙️ Paramètres de Génération")
    
    # Sélection du modèle de base
    base_models = config_manager.get_base_models()
    default_model = config_manager.get_default_base_model() or list(base_models.keys())[0]
    
    base_model = st.sidebar.selectbox(
        "Modèle de base",
        options=list(base_models.keys()),
        index=list(base_models.keys()).index(default_model) if default_model in base_models else 0
    )
    
    # Sélection de l'approche
    approach = st.sidebar.selectbox(
        "Approche de génération",
        options=["single_lora", "combined_approach"],
        format_func=lambda x: {
            "single_lora": "LoRA Simple",
            "combined_approach": "Approche Combinée (Wall + ControlNet + Plan)"
        }[x]
    )
    
    # Sélection du LoRA (pour single_lora)
    lora_models = config_manager.get_lora_models()
    lora_model = None
    
    if approach == "single_lora":
        lora_options = [k for k in lora_models.keys() if k != "wall_lora"]
        lora_model = st.sidebar.selectbox(
            "Modèle LoRA",
            options=lora_options
        )
    
    st.sidebar.markdown("---")
    
    # Paramètres de sampling
    st.sidebar.subheader("🎯 Paramètres de Sampling")
    
    defaults = config_manager.get_generation_defaults()
    
    steps = st.sidebar.slider(
        "Nombre de steps",
        min_value=10,
        max_value=50,
        value=defaults.steps,
        step=1
    )
    
    cfg_scale = st.sidebar.slider(
        "CFG Scale",
        min_value=1.0,
        max_value=15.0,
        value=defaults.cfg_scale,
        step=0.1
    )
    
    samplers = config_manager.get_samplers()
    sampler = st.sidebar.selectbox(
        "Sampler",
        options=samplers,
        index=samplers.index(defaults.sampler) if defaults.sampler in samplers else 0
    )
    
    seed = st.sidebar.number_input(
        "Seed (-1 pour aléatoire)",
        value=-1,
        min_value=-1,
        max_value=2**32-1
    )
    
    st.sidebar.markdown("---")
    
    # Paramètres LoRA
    st.sidebar.subheader("🔧 Paramètres LoRA")
    
    if approach == "single_lora":
        lora_weight = st.sidebar.slider(
            "Poids LoRA",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1
        )
        wall_lora_weight = 0.7
        plan_lora_weight = 0.8
        controlnet_weight = 1.0
    else:
        lora_weight = 0.8
        wall_lora_weight = st.sidebar.slider(
            "Poids Wall LoRA",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1
        )
        plan_lora_weight = st.sidebar.slider(
            "Poids Plan LoRA",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1
        )
        controlnet_weight = st.sidebar.slider(
            "Poids ControlNet",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1
        )
    
    st.sidebar.markdown("---")
    
    # Paramètres d'image
    st.sidebar.subheader("🖼️ Paramètres d'Image")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        width = st.selectbox(
            "Largeur",
            options=[512, 768, 1024],
            index=2
        )
    with col2:
        height = st.selectbox(
            "Hauteur", 
            options=[512, 768, 1024],
            index=2
        )
    
    batch_size = st.sidebar.slider(
        "Batch Size",
        min_value=1,
        max_value=4,
        value=1
    )
    
    # Créer l'objet de paramètres
    params = GenerationParams(
        prompt="",  # Sera rempli plus tard
        base_model=base_model,
        approach=approach,
        lora_model=lora_model,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler=sampler,
        seed=seed,
        width=width,
        height=height,
        batch_size=batch_size,
        lora_weight=lora_weight,
        wall_lora_weight=wall_lora_weight,
        plan_lora_weight=plan_lora_weight,
        controlnet_weight=controlnet_weight
    )
    
    return params

def render_prompt_section():
    """Rendu de la section de prompt"""
    st.subheader("📝 Configuration du Prompt")
    
    # Templates de prompts
    templates = config_manager.get_prompt_templates()
    room_types = config_manager.get_room_types()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Sélection du template
        template_names = list(templates.keys())
        selected_template = st.selectbox(
            "Template de prompt",
            options=["custom"] + template_names,
            format_func=lambda x: "Personnalisé" if x == "custom" else x.replace("_", " ").title()
        )
        
        if selected_template != "custom":
            # Sélection du type de pièce
            room_type = st.selectbox(
                "Type de pièce",
                options=room_types
            )
            
            # Générer le prompt à partir du template
            template_prompt = templates[selected_template].format(room_type=room_type)
            st.text_area(
                "Prompt généré",
                value=template_prompt,
                height=80,
                disabled=True
            )
        else:
            template_prompt = ""
            room_type = ""
    
    with col2:
        st.markdown("**💡 Conseils pour les prompts:**")
        st.markdown("""
        - Soyez précis sur le type de pièce
        - Mentionnez le style architectural
        - Incluez "floor plan"
        """)
    
    # Prompt principal
    if selected_template != "custom":
        prompt = st.text_area(
            "Prompt personnalisé (optionnel)",
            value="",
            height=100,
            placeholder="Ajoutez des détails supplémentaires au template..."
        )
        
        # Combiner template et prompt personnalisé
        if prompt.strip():
            final_prompt = f"{template_prompt}, {prompt}"
        else:
            final_prompt = template_prompt
    else:
        final_prompt = st.text_area(
            "Prompt personnalisé",
            value="",
            height=100,
            placeholder="Entrez votre prompt personnalisé..."
        )
    
    # Prompt négatif
    default_negative = ""
    
    negative_prompt = st.text_area(
        "Prompt négatif",
        value=default_negative,
        height=80
    )
    
    return final_prompt, negative_prompt

def render_generation_section(params):
    """Rendu de la section de génération"""
    st.subheader("🎨 Génération d'Image")
    
    prompt, negative_prompt = render_prompt_section()
    
    # Mise à jour des paramètres avec les prompts
    params.prompt = prompt
    params.negative_prompt = negative_prompt
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        generate_button = st.button(
            "🚀 Générer l'image",
            type="primary",
            disabled=not prompt.strip()
        )
    
    with col2:
        if st.button("📊 Statistiques"):
            stats = generation_queue.get_stats()
            st.json(stats)
    
    with col3:
        if st.button("🧹 Nettoyer"):
            generation_queue.clear_completed()
            st.success("✅ Tâches terminées supprimées")
    
    if generate_button and prompt.strip():
        if not params.validate():
            st.error("❌ Paramètres invalides")
            return
        
        # Soumettre la génération
        task_id = generation_queue.submit_generation(params)
        st.session_state.current_task_id = task_id
        st.success(f"✅ Génération soumise (ID: {task_id[:8]})")
        
        # Rerun pour afficher le suivi
        st.rerun()

def render_task_monitoring():
    """Rendu du suivi des tâches"""
    if 'current_task_id' not in st.session_state:
        return
    
    task_id = st.session_state.current_task_id
    result = generation_queue.get_result(task_id)
    
    if not result:
        return
    
    st.subheader("📈 Suivi de la Génération")
    
    # Barre de statut
    status_colors = {
        "pending": "🟡",
        "processing": "🔵", 
        "completed": "🟢",
        "failed": "🔴"
    }
    
    status_text = {
        "pending": "En attente",
        "processing": "En cours de traitement",
        "completed": "Terminé",
        "failed": "Échoué"
    }
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.metric("Statut", f"{status_colors[result.status]} {status_text[result.status]}")
    
    with col2:
        if result.start_time:
            st.metric("Durée", result.get_duration_str())
    
    with col3:
        queue_size = generation_queue.get_queue_size()
        st.metric("File d'attente", f"{queue_size} tâches")
    
    # Détails de la tâche
    with st.expander("📋 Détails de la tâche"):
        st.json({
            "id": result.id,
            "approach": result.params.approach,
            "model": result.params.base_model,
            "lora": result.params.lora_model,
            "steps": result.params.steps,
            "cfg_scale": result.params.cfg_scale,
            "seed": result.params.seed,
            "dimensions": f"{result.params.width}x{result.params.height}"
        })
    
    # Affichage du résultat
    if result.status == "completed" and result.image_path:
        st.subheader("🖼️ Résultat")
        
        image = load_image(result.image_path)
        if image:
            st.image(image, caption=f"Généré en {result.get_duration_str()}")
            
            # Boutons d'action
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💾 Sauvegarder vers S3"):
                    try:
                        # Upload vers S3
                        s3_url = inference_engine.upload_to_s3(result)
                        if s3_url:
                            st.success(f"✅ Image uploadée vers S3: {s3_url}")
                            result.s3_url = s3_url
                        else:
                            st.error("❌ Erreur lors de l'upload S3")
                    except Exception as e:
                        st.error(f"❌ Erreur S3: {e}")
            
            with col2:
                if st.button("🔄 Régénérer"):
                    # Régénérer avec les mêmes paramètres
                    new_params = result.params
                    new_params.seed = -1  # Nouveau seed aléatoire
                    new_task_id = generation_queue.submit_generation(new_params)
                    st.session_state.current_task_id = new_task_id
                    st.rerun()
            
            with col3:
                if st.button("📊 Comparer"):
                    st.session_state.comparison_image = result.image_path
                    st.success("Image ajoutée à la comparaison")
    
    elif result.status == "failed":
        st.error(f"❌ Génération échouée: {result.error_message}")
    
    elif result.status == "processing":
        st.info("🔄 Génération en cours...")
        # Auto-refresh toutes les 2 secondes
        time.sleep(2)
        st.rerun()

def render_model_info():
    """Rendu des informations sur les modèles"""
    st.subheader("ℹ️ Informations sur les Modèles")
    
    model_info = inference_engine.get_model_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Modèles de Base:**")
        for name, info in model_info["base_models"].items():
            status = "✅" if info["available"] else "❌"
            st.markdown(f"- {status} {name}")
        
        st.markdown("**Modèles LoRA:**")
        for name, info in model_info["lora_models"].items():
            status = "✅" if info["available"] else "❌"
            st.markdown(f"- {status} {name}")
    
    with col2:
        st.markdown("**Informations Système:**")
        st.markdown(f"- Device: {model_info['device']}")
        
        if "gpu_name" in model_info["memory_info"]:
            memory_info = model_info["memory_info"]
            st.markdown(f"- GPU: {memory_info['gpu_name']}")
            st.markdown(f"- Mémoire GPU utilisée: {memory_info['gpu_memory_allocated']:.1f} GB")
            st.markdown(f"- Mémoire GPU réservée: {memory_info['gpu_memory_reserved']:.1f} GB")

def render_recent_generations():
    """Rendu des générations récentes"""
    st.subheader("🕒 Générations Récentes")
    
    completed_tasks = generation_queue.get_completed_tasks()
    
    if not completed_tasks:
        st.info("Aucune génération terminée")
        return
    
    # Trier par date de fin (plus récent en premier)
    sorted_tasks = sorted(
        completed_tasks.items(),
        key=lambda x: x[1].end_time or 0,
        reverse=True
    )
    
    # Afficher les 6 plus récentes
    recent_tasks = sorted_tasks[:6]
    
    cols = st.columns(3)
    
    for i, (task_id, result) in enumerate(recent_tasks):
        col = cols[i % 3]
        
        with col:
            if result.image_path:
                image = load_image(result.image_path)
                if image:
                    st.image(image, caption=f"ID: {task_id[:8]}")
                    
                    if st.button(f"🔍 Voir détails", key=f"details_{task_id}"):
                        st.session_state.current_task_id = task_id
                        st.rerun()

def main():
    """Fonction principale"""
    st.markdown('<h1 class="main-header">🏠 Floor Plan Generator</h1>', unsafe_allow_html=True)
    
    # Initialiser l'application
    initialize_app()
    
    if not st.session_state.initialized:
        return
    
    # Sidebar avec paramètres
    params = render_sidebar()
    
    # Contenu principal
    tab1, tab2, tab3 = st.tabs(["🎨 Génération", "📊 Suivi", "ℹ️ Informations"])
    
    with tab1:
        render_generation_section(params)
        render_recent_generations()
    
    with tab2:
        render_task_monitoring()
        
        # Statistiques générales
        st.subheader("📈 Statistiques Générales")
        stats = generation_queue.get_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total", stats["total_tasks"])
        with col2:
            st.metric("En attente", stats["pending"])
        with col3:
            st.metric("Terminées", stats["completed"])
        with col4:
            st.metric("Échouées", stats["failed"])
    
    with tab3:
        render_model_info()
        
        # Nettoyage
        st.subheader("🧹 Maintenance")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Nettoyer anciennes images"):
                from core.utils.image_utils import cleanup_old_images
                cleanup_old_images()
                st.success("✅ Nettoyage effectué")
        
        with col2:
            if st.button("🔄 Redémarrer file d'attente"):
                generation_queue.stop()
                generation_queue.start()
                st.success("✅ File d'attente redémarrée")

if __name__ == "__main__":
    main()