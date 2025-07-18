import threading
import queue
import uuid
from typing import Dict, Optional, Callable
from datetime import datetime
import time

from .generation_params import GenerationParams, GenerationResult

class GenerationQueue:
    """Gestionnaire de file d'attente pour les générations d'images"""
    
    def __init__(self, max_workers: int = 1):
        self.task_queue = queue.Queue()
        self.results: Dict[str, GenerationResult] = {}
        self.max_workers = max_workers
        self.workers = []
        self.is_running = False
        self.generation_func: Optional[Callable] = None
        self._lock = threading.Lock()
        
    def set_generation_function(self, func: Callable):
        """Définit la fonction de génération à utiliser"""
        self.generation_func = func
    
    def start(self):
        """Démarre les workers"""
        if self.is_running:
            return
        
        self.is_running = True
        
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker, name=f"GenerationWorker-{i}")
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        print(f"✅ File d'attente démarrée avec {self.max_workers} worker(s)")
    
    def stop(self):
        """Arrête les workers"""
        self.is_running = False
        
        # Ajouter des tâches poison pour arrêter les workers
        for _ in range(self.max_workers):
            self.task_queue.put(None)
        
        # Attendre que tous les workers se terminent
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.workers.clear()
        print("✅ File d'attente arrêtée")
    
    def submit_generation(self, params: GenerationParams) -> str:
        """Soumet une nouvelle génération à la file d'attente"""
        if not self.is_running:
            self.start()
        
        if not self.generation_func:
            raise ValueError("Fonction de génération non définie")
        
        # Créer un ID unique pour la tâche
        task_id = str(uuid.uuid4())
        
        # Créer le résultat initial
        result = GenerationResult(
            id=task_id,
            params=params,
            status="pending"
        )
        
        # Stocker le résultat
        with self._lock:
            self.results[task_id] = result
        
        # Ajouter à la file d'attente
        self.task_queue.put((task_id, params))
        
        print(f"📝 Tâche {task_id[:8]} ajoutée à la file d'attente")
        return task_id
    
    def get_result(self, task_id: str) -> Optional[GenerationResult]:
        """Récupère le résultat d'une tâche"""
        with self._lock:
            return self.results.get(task_id)
    
    def get_status(self, task_id: str) -> str:
        """Récupère le statut d'une tâche"""
        result = self.get_result(task_id)
        return result.status if result else "not_found"
    
    def get_queue_size(self) -> int:
        """Retourne la taille de la file d'attente"""
        return self.task_queue.qsize()
    
    def get_all_results(self) -> Dict[str, GenerationResult]:
        """Retourne tous les résultats"""
        with self._lock:
            return self.results.copy()
    
    def get_pending_tasks(self) -> Dict[str, GenerationResult]:
        """Retourne les tâches en attente"""
        with self._lock:
            return {k: v for k, v in self.results.items() if v.status == "pending"}
    
    def get_processing_tasks(self) -> Dict[str, GenerationResult]:
        """Retourne les tâches en cours de traitement"""
        with self._lock:
            return {k: v for k, v in self.results.items() if v.status == "processing"}
    
    def get_completed_tasks(self) -> Dict[str, GenerationResult]:
        """Retourne les tâches terminées"""
        with self._lock:
            return {k: v for k, v in self.results.items() if v.status == "completed"}
    
    def get_failed_tasks(self) -> Dict[str, GenerationResult]:
        """Retourne les tâches échouées"""
        with self._lock:
            return {k: v for k, v in self.results.items() if v.status == "failed"}
    
    def clear_completed(self):
        """Supprime les tâches terminées des résultats"""
        with self._lock:
            self.results = {k: v for k, v in self.results.items() 
                          if v.status not in ["completed", "failed"]}
    
    def _worker(self):
        """Worker principal qui traite les tâches"""
        while self.is_running:
            try:
                # Récupérer une tâche (timeout pour permettre l'arrêt)
                task = self.task_queue.get(timeout=1.0)
                
                # Tâche poison pour arrêter le worker
                if task is None:
                    break
                
                task_id, params = task
                
                # Récupérer le résultat et marquer comme en cours
                with self._lock:
                    result = self.results.get(task_id)
                    if not result:
                        continue
                
                result.mark_started()
                print(f"🔄 Traitement de la tâche {task_id[:8]}...")
                
                try:
                    # Exécuter la génération
                    image_path, s3_url = self.generation_func(params)
                    
                    # Marquer comme terminé
                    result.mark_completed(image_path, s3_url)
                    print(f"✅ Tâche {task_id[:8]} terminée en {result.get_duration_str()}")
                    
                except Exception as e:
                    # Marquer comme échoué
                    error_msg = str(e)
                    result.mark_failed(error_msg)
                    print(f"❌ Tâche {task_id[:8]} échouée: {error_msg}")
                
                finally:
                    # Marquer la tâche comme terminée dans la queue
                    self.task_queue.task_done()
                    
            except queue.Empty:
                # Timeout normal, continuer
                continue
            except Exception as e:
                print(f"❌ Erreur dans le worker: {e}")
                continue
    
    def wait_for_completion(self, task_id: str, timeout: float = 300.0) -> GenerationResult:
        """Attend la completion d'une tâche spécifique"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = self.get_result(task_id)
            if result and result.status in ["completed", "failed"]:
                return result
            
            time.sleep(0.5)
        
        # Timeout
        result = self.get_result(task_id)
        if result:
            result.mark_failed("Timeout de génération")
        
        return result
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques de la file d'attente"""
        with self._lock:
            total = len(self.results)
            pending = sum(1 for r in self.results.values() if r.status == "pending")
            processing = sum(1 for r in self.results.values() if r.status == "processing")
            completed = sum(1 for r in self.results.values() if r.status == "completed")
            failed = sum(1 for r in self.results.values() if r.status == "failed")
            
            return {
                "total_tasks": total,
                "pending": pending,
                "processing": processing,
                "completed": completed,
                "failed": failed,
                "queue_size": self.get_queue_size(),
                "is_running": self.is_running,
                "workers": len(self.workers)
            }

# Instance globale de la file d'attente
generation_queue = GenerationQueue()