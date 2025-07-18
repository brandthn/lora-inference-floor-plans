import re
import json
import hashlib
from typing import Dict, List, Set
from dataclasses import dataclass

@dataclass
class PromptStructure:
    """Structure extraite d'un prompt"""
    rooms: List[str]
    locations: List[str]
    counts: Dict[str, int]
    features: List[str]
    original_prompt: str
    normalized_prompt: str

class PromptHashGenerator:
    """Générateur de hash pour prompts de floor plans"""
    
    def __init__(self):
        self.common_prefixes = [
            "Floor plans.", "Floor plan.", "Architectural drawing.", 
            "Blueprint of", "Plan of", "Layout of", "Design of"
        ]
        
        self.room_patterns = [
            r'(bedroom|bed room)',
            r'(bathroom|bath room|toilet)',
            r'(lounge|living room|living area)',
            r'(dining room|dining area|dining)',
            r'(kitchen|cook area)',
            r'(storeroom|store room|storage|closet)',
            r'(office|study|work room)',
            r'(balcony|terrace|patio)',
            r'(hallway|corridor|passage)',
            r'(entrance|entry|foyer)'
        ]
        
        self.location_patterns = [
            r'(north|northern)',
            r'(south|southern)', 
            r'(east|eastern)',
            r'(west|western)',
            r'(center|centre|central|middle)',
            r'(corner|side)'
        ]
        
        self.count_patterns = [
            r'(first|1st|one)',
            r'(second|2nd|two)', 
            r'(third|3rd|three)',
            r'(fourth|4th|four)',
            r'(\d+)'
        ]
    
    def normalize_prompt(self, prompt: str) -> str:
        """Normalise le prompt pour la comparaison"""
        cleaned = prompt.strip()
        
        # Supprimer les préfixes communs
        for prefix in self.common_prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        # Normaliser les espaces et la ponctuation
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'[,\.\;\:\!]', '', cleaned)
        
        # Normaliser les cas
        cleaned = cleaned.lower()
        
        # Normaliser les variations linguistiques
        replacements = {
            'is located at': 'at',
            'are located at': 'at',
            'located at': 'at',
            'positioned at': 'at',
            'situated at': 'at',
            'and are': 'are',
            'are combined': 'combined'
        }
        
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        return cleaned
    
    def extract_rooms(self, prompt: str) -> List[str]:
        """Extrait les types de pièces du prompt"""
        rooms = []
        
        for pattern in self.room_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            for match in matches:
                # Normaliser le nom de la pièce
                normalized = match.lower()
                if 'bed' in normalized:
                    normalized = 'bedroom'
                elif 'bath' in normalized:
                    normalized = 'bathroom'
                elif 'living' in normalized or 'lounge' in normalized:
                    normalized = 'living_room'
                elif 'dining' in normalized:
                    normalized = 'dining_room'
                elif 'store' in normalized or 'storage' in normalized:
                    normalized = 'storeroom'
                
                rooms.append(normalized)
        
        return rooms
    
    def extract_locations(self, prompt: str) -> List[str]:
        """Extrait les positions/localisations du prompt"""
        locations = []
        
        for pattern in self.location_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            for match in matches:
                normalized = match.lower()
                if 'centre' in normalized or 'center' in normalized or 'central' in normalized:
                    normalized = 'center'
                elif 'middle' in normalized:
                    normalized = 'center'
                
                locations.append(normalized)
        
        return locations
    
    def extract_features(self, prompt: str) -> List[str]:
        """Extrait les caractéristiques spéciales du prompt"""
        features = []
        
        feature_patterns = [
            r'(combined|merged|joined)',
            r'(separate|separated|individual)',
            r'(open plan|open-plan)',
            r'(en-suite|ensuite)',
            r'(master|main)',
            r'(guest|visitor)',
            r'(private|public)',
            r'(large|small|big|tiny)',
            r'(modern|traditional|contemporary)',
            r'(luxury|premium|basic)'
        ]
        
        for pattern in feature_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            features.extend([match.lower() for match in matches])
        
        return features
    
    def count_rooms(self, rooms: List[str]) -> Dict[str, int]:
        """Compte les occurrences de chaque type de pièce"""
        counts = {}
        for room in rooms:
            counts[room] = counts.get(room, 0) + 1
        return counts
    
    def extract_structure(self, prompt: str) -> PromptStructure:
        """Extrait la structure complète du prompt"""
        normalized = self.normalize_prompt(prompt)
        
        rooms = self.extract_rooms(normalized)
        locations = self.extract_locations(normalized)
        features = self.extract_features(normalized)
        counts = self.count_rooms(rooms)
        
        return PromptStructure(
            rooms=sorted(set(rooms)),
            locations=sorted(set(locations)),
            counts=counts,
            features=sorted(set(features)),
            original_prompt=prompt,
            normalized_prompt=normalized
        )
    
    def generate_hash(self, prompt: str) -> str:
        """Génère un hash unique basé sur la structure du prompt"""
        structure = self.extract_structure(prompt)
        
        # Créer une représentation canonique
        canonical = {
            'rooms': structure.rooms,
            'locations': structure.locations,
            'counts': dict(sorted(structure.counts.items())),
            'features': structure.features
        }
        
        # Générer le hash
        canonical_str = json.dumps(canonical, sort_keys=True)
        hash_obj = hashlib.md5(canonical_str.encode('utf-8'))
        return hash_obj.hexdigest()[:12]
    
    def are_similar(self, prompt1: str, prompt2: str, threshold: float = 0.8) -> bool:
        """Détermine si deux prompts sont similaires"""
        struct1 = self.extract_structure(prompt1)
        struct2 = self.extract_structure(prompt2)
        
        # Calculer la similarité basée sur les éléments communs
        rooms_similarity = self._calculate_set_similarity(struct1.rooms, struct2.rooms)
        counts_similarity = self._calculate_counts_similarity(struct1.counts, struct2.counts)
        locations_similarity = self._calculate_set_similarity(struct1.locations, struct2.locations)
        
        # Moyenne pondérée
        overall_similarity = (
            rooms_similarity * 0.5 +
            counts_similarity * 0.3 +
            locations_similarity * 0.2
        )
        
        return overall_similarity >= threshold
    
    def _calculate_set_similarity(self, set1: List[str], set2: List[str]) -> float:
        """Calcule la similarité entre deux ensembles"""
        s1, s2 = set(set1), set(set2)
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_counts_similarity(self, counts1: Dict[str, int], counts2: Dict[str, int]) -> float:
        """Calcule la similarité entre deux dictionnaires de comptage"""
        all_rooms = set(counts1.keys()) | set(counts2.keys())
        
        if not all_rooms:
            return 1.0
        
        total_diff = 0
        total_max = 0
        
        for room in all_rooms:
            count1 = counts1.get(room, 0)
            count2 = counts2.get(room, 0)
            
            total_diff += abs(count1 - count2)
            total_max += max(count1, count2)
        
        return 1 - (total_diff / total_max) if total_max > 0 else 1.0

# Instance globale du générateur de hash
prompt_hash_generator = PromptHashGenerator()

# Fonctions utilitaires
def generate_prompt_hash(prompt: str) -> str:
    """Génère un hash pour un prompt donné"""
    return prompt_hash_generator.generate_hash(prompt)

def extract_prompt_structure(prompt: str) -> PromptStructure:
    """Extrait la structure d'un prompt"""
    return prompt_hash_generator.extract_structure(prompt)

def are_prompts_similar(prompt1: str, prompt2: str, threshold: float = 0.8) -> bool:
    """Vérifie si deux prompts sont similaires"""
    return prompt_hash_generator.are_similar(prompt1, prompt2, threshold)