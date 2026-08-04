"""
Service de vision par ordinateur.
Utilise CLIP pour la classification d'images avec des prompts industriels.
"""

import os
from typing import Dict, Any
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class VisionService:
    """
    Service pour l'analyse d'images avec CLIP.
    Utilise des prompts industriels pour détecter les défauts.
    """
    
    _instance = None
    _model = None
    _processor = None
    
    # Prompts pour la classification industrielle
    CONFORME_PROMPTS = [
        "defect-free industrial surface",
        "clean smooth metal surface",
        "uniform manufactured surface",
        "perfect industrial part",
        "acceptable industrial part",
        "passed quality inspection",
        "accepted quality inspection",
        "no visible surface defects",
        "a close-up photo of a conforming industrial product with a smooth, uniform and defect-free surface",
        "a manufactured industrial part accepted after visual quality inspection, with no scratches, cracks, corrosion or contamination",
        "a high-quality metal or plastic industrial component with a clean finish, uniform texture and no visible defects",
        "an industrial product with excellent surface quality, free of dents, stains, pits, oxidation and manufacturing flaws",
        "a production line part meeting quality standards, showing a flawless and homogeneous surface",
        "a visually perfect industrial component with consistent color, texture and finish across the entire surface"
    ]
    
    DEFAUT_PROMPTS = [
        "defective industrial surface",
        "corroded surface",
        "scratched surface",
        "cracked surface",
        "damaged metal surface",
        "damaged industrial part",
        "corroded industrial part",
        "failed quality inspection",
        "rejected quality inspection",
        "visible surface defects",
        "a close-up photo of a defective industrial product with visible surface defects",
        "a manufactured industrial part rejected during visual quality inspection because of scratches, cracks, corrosion or contamination",
        "a metal or plastic industrial component with damaged finish, rough texture, dents or manufacturing defects",
        "an industrial product showing corrosion, oxidation, pitting, stains, chipped areas or surface irregularities",
        "a production line part failing quality inspection due to visible surface damage or poor finishing",
        "a defective industrial component with obvious imperfections affecting surface quality"
    ]
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VisionService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self._load_model()
    
    def _load_model(self):
        """
        Charge le modèle CLIP.
        Le nom du modèle est lu depuis les variables d'environnement.
        """
        model_name = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
        self._model = CLIPModel.from_pretrained(model_name)
        self._processor = CLIPProcessor.from_pretrained(model_name)
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyse une image avec CLIP pour déterminer si le produit est conforme ou défectueux.
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Dictionnaire avec la catégorie et les scores de confiance
        """
        # Charger l'image
        image = Image.open(image_path)
        
        # Combiner tous les prompts
        all_prompts = self.CONFORME_PROMPTS + self.DEFAUT_PROMPTS
        
        # Préparer les entrées
        inputs = self._processor(
            text=all_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Faire la prédiction
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Obtenir les probabilités
        probs_list = probs[0].tolist()
        
        # Séparer les scores
        nb_conforme = len(self.CONFORME_PROMPTS)
        score_conforme = sum(probs_list[:nb_conforme])
        score_defaut = sum(probs_list[nb_conforme:])
        
        # Normaliser
        total = score_conforme + score_defaut
        if total > 0:
            score_conforme = score_conforme / total
            score_defaut = score_defaut / total
        
        # Déterminer la catégorie
        conformite_threshold = float(os.getenv("CONFORMITY_THRESHOLD", 0.6))
        
        if score_defaut > 0.5:
            return {
                "categorie": "Endommagé / Défectueux",
                "confiance": round(score_defaut, 3),
                "score_conforme": round(score_conforme, 3),
                "score_defaut": round(score_defaut, 3)
            }
        elif score_conforme > conformite_threshold:
            return {
                "categorie": "Conforme / Bon état",
                "confiance": round(score_conforme, 3),
                "score_conforme": round(score_conforme, 3),
                "score_defaut": round(score_defaut, 3)
            }
        else:
            return {
                "categorie": "Incertain",
                "confiance": round(max(score_conforme, score_defaut), 3),
                "score_conforme": round(score_conforme, 3),
                "score_defaut": round(score_defaut, 3)
            }
