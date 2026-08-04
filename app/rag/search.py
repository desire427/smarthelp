"""
Logique de recherche RAG (Retrieval Augmented Generation).

Feature: rag-search
- top_k : retourne les k meilleures règles (pas seulement la première)
- Filtrage optionnel par catégorie avant la recherche
- Résultats enrichis avec métadonnées (id, categorie, mots_cles)
- Classe RAGError pour les erreurs métier
"""

import os
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

from app.rag.knowledge_base import KnowledgeBase
from app.services.embedding_service import EmbeddingService


class RAGError(Exception):
    """Erreur spécifique au service RAG."""
    pass


class RAGSearch:
    """
    Service de recherche RAG avec support top-k et filtrage par catégorie.
    """

    def __init__(self):
        self.kb = KnowledgeBase()
        self.embedding_service = EmbeddingService()
        self._embeddings_cache: Optional[List[np.ndarray]] = None

    # ------------------------------------------------------------------
    # Utilitaires internes
    # ------------------------------------------------------------------

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calcule la similarité cosinus entre deux vecteurs."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _get_kb_embeddings(self) -> List[np.ndarray]:
        """
        Calcule et met en cache les embeddings de la base de connaissances.
        Recalcule si le cache est vide (ex. après add_rule).
        """
        if self._embeddings_cache is None:
            knowledge = self.kb.get_all()
            self._embeddings_cache = [
                self.embedding_service.embed(text) for text in knowledge
            ]
        return self._embeddings_cache

    def invalidate_cache(self):
        """Force le recalcul des embeddings au prochain appel."""
        self._embeddings_cache = None

    # ------------------------------------------------------------------
    # Recherche
    # ------------------------------------------------------------------

    def search(self, query: str) -> Tuple[str, float]:
        """
        Retourne la règle la plus pertinente (interface de compatibilité).

        Args:
            query: Texte de la requête

        Returns:
            Tuple (texte de la règle, score de similarité)
        """
        results = self.search_top_k(query, k=1)
        if not results:
            return "", 0.0
        return results[0]["texte"], results[0]["score"]

    def search_top_k(
        self,
        query: str,
        k: int = 3,
        categorie: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retourne les k règles les plus pertinentes, avec leurs métadonnées.

        Args:
            query    : Texte de la requête
            k        : Nombre de résultats à retourner (défaut 3)
            categorie: Filtre optionnel sur la catégorie

        Returns:
            Liste de dictionnaires triés par score décroissant :
              {id, texte, categorie, mots_cles, score}

        Raises:
            RAGError: si la requête est vide ou k invalide.
        """
        if not query.strip():
            raise RAGError("La requête ne peut pas être vide.")
        if k < 1:
            raise RAGError(f"k doit être supérieur ou égal à 1, reçu : {k}")

        all_rules = self.kb.get_all_with_metadata()
        all_texts = self.kb.get_all()
        kb_embeddings = self._get_kb_embeddings()

        # Filtrage par catégorie
        if categorie:
            indices = [i for i, r in enumerate(all_rules) if r["categorie"] == categorie]
            if not indices:
                return []
        else:
            indices = list(range(len(all_rules)))

        # Embedding de la requête
        q_vec = self.embedding_service.embed(query)

        # Calcul des scores
        scored = [
            (i, self._cosine_similarity(q_vec, kb_embeddings[i]))
            for i in indices
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Construction des résultats top-k
        results = []
        for i, score in scored[:k]:
            rule = dict(all_rules[i])
            rule["score"] = round(score, 4)
            results.append(rule)

        return results

    # ------------------------------------------------------------------
    # Statut proposé
    # ------------------------------------------------------------------

    def propose_status(
        self,
        matched_rule: str,
        similarity: float,
        vision_diagnosis: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Propose un statut pour le ticket en fonction de la règle et de l'analyse visuelle.

        Args:
            matched_rule    : Texte de la règle correspondante
            similarity      : Score de similarité
            vision_diagnosis: Diagnostic image optionnel

        Returns:
            Statut proposé parmi : Remboursable, Échangeable, Refusé, À vérifier,
            À vérifier - Incohérence
        """
        threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.35))

        if similarity < threshold:
            return "À vérifier"

        rule = matched_rule.lower()

        # Prise en compte de l'analyse visuelle
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

        # Logique basée sur les mots-clés de la règle
        if "remboursable" in rule or "remboursement" in rule:
            return "Remboursable"
        if "ne peut plus être retourné" in rule or "pas droit" in rule:
            return "Refusé"
        if "échangeable" in rule or "échange" in rule:
            return "Échangeable"

        return "À vérifier"
