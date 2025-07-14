import boto3
import logging
from pathlib import Path
from datetime import datetime
from PIL import Image
import io
from typing import Optional
from botocore.exceptions import NoCredentialsError, ClientError

from .config import S3_BUCKET_NAME, S3_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3Utils:
    def __init__(self):
        self.s3_client = None
        self.bucket_name = S3_BUCKET_NAME
        self.region = S3_REGION
        self.init_s3_client()
    
    def init_s3_client(self):
        """Initialise le client S3"""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=self.region
            )
            logger.info("Client S3 initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du client S3: {e}")
            raise
    
    def upload_image_from_path(self, file_path: Path, s3_key: str = None) -> Optional[str]:
        """Upload une image depuis un chemin local vers S3"""
        try:
            if s3_key is None:
                s3_key = f"floor_plans/{file_path.name}"
            
            # Upload du fichier
            self.s3_client.upload_file(
                str(file_path),
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': 'image/png',
                    'ACL': 'public-read'  # Rendre l'image publique
                }
            )
            
            # Générer l'URL publique
            url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            logger.info(f"Image uploadée avec succès: {url}")
            
            return url
            
        except NoCredentialsError:
            logger.error("Credentials AWS non trouvés")
            return None
        except ClientError as e:
            logger.error(f"Erreur cliente S3: {e}")
            return None
        except Exception as e:
            logger.error(f"Erreur lors de l'upload: {e}")
            return None
    
    def upload_image_from_pil(self, image: Image.Image, filename: str) -> Optional[str]:
        """Upload une image PIL directement vers S3"""
        try:
            # Convertir PIL Image en bytes
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # Générer la clé S3
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"floor_plans/{timestamp}_{filename}"
            
            # Upload vers S3
            self.s3_client.upload_fileobj(
                img_buffer,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': 'image/png',
                    'ACL': 'public-read'
                }
            )
            
            # Générer l'URL publique
            url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            logger.info(f"Image uploadée avec succès: {url}")
            
            return url
            
        except Exception as e:
            logger.error(f"Erreur lors de l'upload PIL: {e}")
            return None
    
    def upload_multiple_images(self, images: list[Image.Image], filenames: list[str]) -> list[Optional[str]]:
        """Upload plusieurs images vers S3"""
        try:
            urls = []
            
            for image, filename in zip(images, filenames):
                url = self.upload_image_from_pil(image, filename)
                urls.append(url)
            
            return urls
            
        except Exception as e:
            logger.error(f"Erreur lors de l'upload multiple: {e}")
            return [None] * len(images)
    
    def create_bucket_if_not_exists(self):
        """Crée le bucket S3 s'il n'existe pas"""
        try:
            # Vérifier si le bucket existe
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket {self.bucket_name} existe déjà")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket n'existe pas, le créer
                try:
                    if self.region == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region}
                        )
                    logger.info(f"Bucket {self.bucket_name} créé avec succès")
                except Exception as create_error:
                    logger.error(f"Erreur lors de la création du bucket: {create_error}")
                    raise
            else:
                logger.error(f"Erreur lors de la vérification du bucket: {e}")
                raise
    
    def delete_image(self, s3_key: str) -> bool:
        """Supprime une image de S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Image supprimée: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression: {e}")
            return False

# Instance globale
s3_utils = None

def get_s3_utils() -> S3Utils:
    """Récupère l'instance des utilitaires S3 (singleton)"""
    global s3_utils
    if s3_utils is None:
        s3_utils = S3Utils()
    return s3_utils