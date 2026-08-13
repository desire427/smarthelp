"""
Service d'embeddings — encodage de texte pour le RAG.
Le pipeline est chargé une seule fois en mémoire via @lru_cache.
"""

import os
from functools import lru_cache
import numpy as np
from transformers import pipeline


@lru_cache(maxsize=1)
def _get_embedding_pipeline():
    model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return pipeline("feature-extraction", model=model)


def embed(text: str) -> np.ndarray:
    """Retourne le vecteur d'embedding moyen pour un texte."""
    pipe = _get_embedding_pipeline()
    vectors = pipe(text)
    return np.array(vectors[0]).mean(axis=0)
