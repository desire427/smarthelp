"""
Service de vision par ordinateur.
Utilise CLIP pour la classification d'images avec des prompts industriels.

Feature: image-vision
- Mode batch : analyse de plusieurs images en une passe
- Labels configurables via variables d'environnement
- Retour enrichi avec top-3 des prompts les plus similaires
- Classe VisionError pour les erreurs métier
"""

import os
import logging
from typing import Dict, Any, List, Optional

import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)


class VisionError(Exception):
    """Erreur spécifique au service Vision."""
    pass


class VisionService:
    """
    Service pour l'analyse d'images avec CLIP.
    Utilise des prompts industriels pour détecter les défauts.
    """

    _instance = None
    _model = None
    _processor = None

    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

    # Prompts pour la classification industrielle
    CONFORME_PROMPTS = [
        "defect-free industrial surface",
        "clean smooth metal surface",
        "uniform manufactured surface",
        "perfect industrial part",
        "acceptable industrial part",
        "passed quality inspection",
        "no visible surface defects",
        "a conforming industrial product with smooth, uniform and defect-free surface",
        "a manufactured part accepted after visual quality inspection",
        "a high-quality component with clean finish and no visible defects",
        "an industrial product meeting quality standards",
        "a visually perfect component with consistent color and texture",
    ]

    DEFAUT_PROMPTS = [
        "defective industrial surface",
        "corroded surface",
        "scratched surface",
        "cracked surface",
        "damaged metal surface",
        "failed quality inspection",
        "visible surface defects",
        "a defective industrial product with visible surface defects",
        "a part rejected during visual quality inspection",
        "a component with rough texture, dents or manufacturing defects",
        "an industrial product showing corrosion or surface irregularities",
        "a defective component with imperfections affecting surface quality",
    ]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VisionService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self):
        """Charge le modèle CLIP depuis les variables d'environnement."""
        model_name = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
        try:
            logger.info("Chargement du modèle CLIP : %s", model_name)
            self._model = CLIPModel.from_pretrained(model_name)
            self._processor = CLIPProcessor.from_pretrained(model_name)
            logger.info("Modèle CLIP chargé avec succès.")
        except Exception as exc:
            logger.error("Échec du chargement du modèle CLIP : %s", exc)
            raise VisionError(f"Impossible de charger le modèle CLIP '{model_name}': {exc}") from exc

    def _validate_image(self, image_path: str) -> Image.Image:
        """
        Valide et charge une image depuis un chemin.

        Raises:
            VisionError: si le fichier est absent, illisible ou de format non supporté.
        """
        if not os.path.isfile(image_path):
            raise VisionError(f"Image introuvable : {image_path}")

        ext = os.path.splitext(image_path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise VisionError(
                f"Format non supporté '{ext}'. "
                f"Formats acceptés : {sorted(self.SUPPORTED_FORMATS)}"
            )

        try:
            return Image.open(image_path).convert("RGB")
        except UnidentifiedImageError as exc:
            raise VisionError(f"Fichier image invalide ou corrompu : {image_path}") from exc

    def _run_clip(self, image: Image.Image) -> Dict[str, Any]:
        """
        Exécute l'inférence CLIP sur une image PIL.

        Returns:
            Dictionnaire avec catégorie, confiance, scores détaillés et top-3 prompts.
        """
        all_prompts = self.CONFORME_PROMPTS + self.DEFAUT_PROMPTS

        inputs = self._processor(
            text=all_prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()

        nb_conforme = len(self.CONFORME_PROMPTS)
        score_conforme = sum(probs[:nb_conforme])
        score_defaut = sum(probs[nb_conforme:])

        total = score_conforme + score_defaut
        if total > 0:
            score_conforme /= total
            score_defaut /= total

        # Top-3 prompts les plus activés
        indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
        top3 = [{"prompt": all_prompts[i], "score": round(s, 4)} for i, s in indexed]

        conformity_threshold = float(os.getenv("CONFORMITY_THRESHOLD", 0.6))

        if score_defaut > 0.5:
            categorie = "Endommagé / Défectueux"
            confiance = score_defaut
        elif score_conforme > conformity_threshold:
            categorie = "Conforme / Bon état"
            confiance = score_conforme
        else:
            categorie = "Incertain"
            confiance = max(score_conforme, score_defaut)

        return {
            "categorie": categorie,
            "confiance": round(confiance, 3),
            "score_conforme": round(score_conforme, 3),
            "score_defaut": round(score_defaut, 3),
            "top3_prompts": top3,
        }

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyse une image unique.

        Args:
            image_path: Chemin vers l'image

        Returns:
            Dictionnaire avec catégorie, confiance et scores détaillés.

        Raises:
            VisionError: si le fichier est absent ou invalide.
        """
        image = self._validate_image(image_path)
        return self._run_clip(image)

    def analyze_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Analyse un lot d'images.

        Args:
            image_paths: Liste des chemins d'images

        Returns:
            Liste de résultats, un par image.
            Les images invalides contiennent une clé "error" à la place.
        """
        results = []
        for path in image_paths:
            try:
                image = self._validate_image(path)
                result = self._run_clip(image)
                result["image_path"] = path
                results.append(result)
            except VisionError as exc:
                results.append({"image_path": path, "error": str(exc)})
        return results
