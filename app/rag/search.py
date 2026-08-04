"""
Logique de recherche RAG (Retrieval Augmented Generation).
Utilise des embeddings et la similarité cosinus pour trouver la politique applicable.
"""

import os
import numpy as np
from typing import Tuple, Optional, Dict, Any

from app.rag.knowledge_base import KnowledgeBase
from app.services.embedding_service import EmbeddingService

class RAGSearch:
    """
    Service de recherche RAG.
    """
    
    def __init__(self):
        self.kb = KnowledgeBase()
        self.embedding_service = EmbeddingService()
        self._embeddings_cache = None
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calcule la similarité cosinus entre deux vecteurs.
        
        Args:
            a: Premier vecteur
            b: Deuxième vecteur
            
        Returns:
            Score de similarité entre 0 et 1
        """
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    
    def _get_kb_embeddings(self):
        """
        Calcule les embeddings de la base de connaissances une seule fois.
        Utilise le cache pour éviter de recalculer.
        """
        if self._embeddings_cache is None:
            knowledge = self.kb.get_all()
            self._embeddings_cache = [
                self.embedding_service.embed(text) 
                for text in knowledge
            ]
        return self._embeddings_cache
    
    def search(self, query: str) -> Tuple[str, float]:
        """
        Recherche la règle la plus pertinente dans la base de connaissances.
        
        Args:
            query: Requête textuelle
            
        Returns:
            Tuple (règle trouvée, score de similarité)
        """
        if not query.strip():
            return "", 0.0
        
        # Embedding de la requête
        q_vec = self.embedding_service.embed(query)
        
        # Embeddings de la base de connaissances
        kb_vecs = self._get_kb_embeddings()
        knowledge = self.kb.get_all()
        
        # Calcul des similarités
        scores = [self._cosine_similarity(q_vec, kb_vecs[i]) for i in range(len(knowledge))]
        
        # Meilleur résultat
        best_idx = int(np.argmax(scores))
        return knowledge[best_idx], scores[best_idx]
    
    def propose_status(
        self, 
        matched_rule: str, 
        similarity: float, 
        vision_diagnosis: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Propose un statut pour le ticket en fonction de la règle trouvée et de l'analyse visuelle.
        
        Args:
            matched_rule: Règle correspondante
            similarity: Score de similarité
            vision_diagnosis: Diagnostic de l'image (optionnel)
            
        Returns:
            Statut proposé
        """
        threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.35))
        
        if similarity < threshold:
            return "À vérifier"
        
        rule = matched_rule.lower()
        
        # Analyse visuelle
        if vision_diagnosis:
            categorie = vision_diagnosis.get("categorie", "")
            confiance = vision_diagnosis.get("confiance", 0)
            confidence_threshold = float(os.getenv("VISION_CONFIDENCE_THRESHOLD", 0.5))
            
            if categorie == "Endommagé / Défectueux" and confiance > confidence_threshold:
                if "remboursable" in rule or "remboursement" in rule:
                    return "Remboursable"
                if "échange" in rule or "échangeable" in rule:
                    return "Échangeable"
            
            if categorie == "Conforme / Bon état" and ("défaut" in rule or "endommagé" in rule):
                return "À vérifier - Incohérence"
        
        # Hors garantie / refus explicite
        if any(kw in rule for kw in [
            "hors garantie", "n'est plus couvert", "aucun remboursement ou échange gratuit",
            "ne peut plus être retourné", "pas droit", "n'est plus éligible",
            "réparation payante",
        ]):
            return "Refusé - Hors garantie"

        # Logique basée sur les règles
        if "remboursable" in rule or "remboursement" in rule:
            return "Remboursable"
        if "échangeable" in rule or "échange" in rule:
            return "Échangeable"
        if "devis" in rule or "réparation" in rule:
            return "Réparation proposée"

        return "À vérifier"
