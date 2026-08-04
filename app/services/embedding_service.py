"""
Service d'embeddings.
Utilisé pour la recherche RAG avec des modèles de sentence-transformers.
"""

import os
import numpy as np
from transformers import pipeline

class EmbeddingService:
    """
    Service pour la génération d'embeddings de texte.
    Singleton pour éviter de charger le modèle plusieurs fois.
    """
    
    _instance = None
    _pipeline = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._pipeline is None:
            self._load_model()
    
    def _load_model(self):
        """
        Charge le modèle d'embeddings.
        Le nom du modèle est lu depuis les variables d'environnement.
        """
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self._pipeline = pipeline(
            "feature-extraction",
            model=model_name
        )
    
    def embed(self, text: str) -> np.ndarray:
        """
        Génère un embedding pour un texte.
        
        Args:
            text: Texte à encoder
            
        Returns:
            Vecteur d'embedding
        """
        vectors = self._pipeline(text)
        return np.array(vectors[0]).mean(axis=0)
