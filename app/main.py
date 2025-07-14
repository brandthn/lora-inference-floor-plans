from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import logging
from PIL import Image
import io
import base64

from .inference import get_inference_model
from .s3_utils import get_s3_utils
from .config import STREAMLIT_CONFIG

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Floor Plan Generator API",
    description="API pour générer des plans d'étage avec LoRA",
    version="1.0.0"
)

# Modèles Pydantic
class FloorPlanRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    upload_to_s3: bool = False

class FloorPlanResponse(BaseModel):
    success: bool
    message: str
    images: Optional[dict] = None
    s3_urls: Optional[List[str]] = None

# Variables globales
inference_model = None
s3_utils = None

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage"""
    global inference_model, s3_utils
    try:
        logger.info("Initialisation des modèles...")
        inference_model = get_inference_model()
        s3_utils = get_s3_utils()
        logger.info("Modèles initialisés avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation: {e}")
        raise

@app.get("/")
async def root():
    """Route racine"""
    return {"message": "Floor Plan Generator API", "status": "active"}

@app.get("/health")
async def health_check():
    """Vérification de l'état du service"""
    return {
        "status": "healthy",
        "model_loaded": inference_model is not None,
        "s3_configured": s3_utils is not None
    }

@app.post("/generate", response_model=FloorPlanResponse)
async def generate_floor_plan(request: FloorPlanRequest):
    """Génère un plan d'étage"""
    try:
        if inference_model is None:
            raise HTTPException(status_code=503, detail="Modèle non initialisé")
        
        logger.info(f"Génération pour le prompt: {request.prompt}")
        
        # Génération des images
        wall_image, canny_image, floor_plan_image = inference_model.generate_complete_floor_plan(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt
        )
        
        # Convertir les images en base64
        def image_to_base64(img: Image.Image) -> str:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        
        images = {
            "wall": image_to_base64(wall_image),
            "canny": image_to_base64(canny_image),
            "floor_plan": image_to_base64(floor_plan_image)
        }
        
        response = FloorPlanResponse(
            success=True,
            message="Plan d'étage généré avec succès",
            images=images
        )
        
        # Upload vers S3 si demandé
        if request.upload_to_s3 and s3_utils is not None:
            try:
                filenames = [
                    f"wall_{request.prompt[:30].replace(' ', '_')}.png",
                    f"canny_{request.prompt[:30].replace(' ', '_')}.png",
                    f"floor_plan_{request.prompt[:30].replace(' ', '_')}.png"
                ]
                
                urls = s3_utils.upload_multiple_images(
                    [wall_image, canny_image, floor_plan_image],
                    filenames
                )
                
                response.s3_urls = urls
                logger.info("Images uploadées vers S3 avec succès")
                
            except Exception as e:
                logger.error(f"Erreur upload S3: {e}")
                response.message += f" (Erreur upload S3: {e})"
        
        return response
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/examples")
async def get_examples():
    """Retourne des exemples de prompts"""
    return {
        "examples": STREAMLIT_CONFIG["example_prompts"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)