"""
Service Vision — analyse d'image avec CLIP.
Le modèle est chargé une seule fois en mémoire via @lru_cache.
"""

import os
from functools import lru_cache
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

PROMPTS_CONFORME = [
    "a product in perfect condition with no visible damage",
    "a new undamaged product",
    "a product that passed quality inspection",
]

PROMPTS_DEFAUT = [
    "a damaged or broken product",
    "a product with scratches, cracks or defects",
    "a product that failed quality inspection",
]


@lru_cache(maxsize=1)
def _get_clip():
    model_name = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def analyze_image(image_path: str) -> dict:
    """Analyse une image et retourne la catégorie et les scores de confiance."""
    model, processor = _get_clip()
    image = Image.open(image_path).convert("RGB")

    all_prompts = PROMPTS_CONFORME + PROMPTS_DEFAUT
    inputs = processor(text=all_prompts, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(dim=1)[0].tolist()

    n = len(PROMPTS_CONFORME)
    score_conforme = sum(probs[:n]) / n
    score_defaut = sum(probs[n:]) / len(PROMPTS_DEFAUT)

    # Normaliser
    total = score_conforme + score_defaut
    score_conforme /= total
    score_defaut /= total

    if score_defaut >= 0.5:
        categorie = "Endommagé / Défectueux"
        confiance = score_defaut
    elif score_conforme >= 0.6:
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
    }
