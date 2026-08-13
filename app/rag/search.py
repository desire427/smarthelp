"""
Recherche RAG — similarité cosinus entre la requête et la base de connaissances.
Les embeddings de la base sont mis en cache à la première utilisation.
"""

import os
from functools import lru_cache
import numpy as np

from app.rag.knowledge_base import load_knowledge
from app.services.embedding_service import embed


@lru_cache(maxsize=1)
def _get_kb_embeddings() -> tuple[list[str], list[np.ndarray]]:
    """Charge et encode la base de connaissances une seule fois."""
    rules = load_knowledge()
    vectors = [embed(rule) for rule in rules]
    return rules, vectors


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# Mots-clés du domaine support client e-commerce
_DOMAIN_KEYWORDS = {
    "produit", "article", "commande", "achat", "acheté", "livraison", "colis",
    "remboursement", "rembourser", "remboursé", "retour", "renvoyer", "renvoyé",
    "échange", "échanger", "échangé", "réparation", "réparer", "réparé",
    "défaut", "défectueux", "endommagé", "cassé", "abîmé", "panne", "fonctionne",
    "garantie", "réclamation", "ticket", "support", "service", "facture",
    "reçu", "reçu", "livré", "expédié", "erreur", "problème",
}


def is_relevant(query: str) -> bool:
    """Retourne True si la query contient au moins un mot-clé du domaine support."""
    words = set(query.lower().split())
    return bool(words & _DOMAIN_KEYWORDS)


# Mots-clés indiquant une responsabilité de l'utilisateur
_USER_FAULT_KEYWORDS = [
    "tombé", "tomber", "chute", "j'ai fait tomber",
    "cassé avec moi", "cassé par moi", "c'est moi qui",
    "gaté", "gâté", "gater", "gâter",
    "mauvaise manipulation", "ma faute", "de ma faute",
    "maladresse", "accident de ma part",
]


def is_user_fault(query: str) -> bool:
    """Retourne True si la query indique un dommage causé par l'utilisateur."""
    q = query.lower()
    return any(kw in q for kw in _USER_FAULT_KEYWORDS)


def search(query: str) -> tuple[str, float]:
    """Retourne la règle la plus proche et son score de similarité."""
    if not query.strip():
        return "", 0.0

    rules, vectors = _get_kb_embeddings()
    q_vec = embed(query)
    scores = [_cosine_similarity(q_vec, v) for v in vectors]
    best = int(np.argmax(scores))
    return rules[best], scores[best]


def propose_status(matched_rule: str, similarity: float, vision: dict | None, query: str = "") -> str:
    """Propose un statut ticket à partir de la règle RAG et du diagnostic image."""
    threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))

    if similarity < threshold:
        return "À vérifier"

    rule = matched_rule.lower()
    q = query.lower()

    # Casse ou dommage causé par l'utilisateur — priorité absolue
    _user_fault_kw = ["tombé", "tomber", "chute", "cassé avec moi", "gaté", "gâté", "gater", "mauvaise manipulation", "ma faute"]
    if any(kw in q for kw in _user_fault_kw):
        return "Refusé - Dommage causé par l'utilisateur"

    # Règle matchée indique une responsabilité utilisateur
    if any(kw in rule for kw in ["faute", "mauvaise manipulation", "cassé par", "endommagé par"]):
        return "Refusé - Dommage causé par l'utilisateur"

    # Cohérence visuelle
    if vision:
        categorie = vision.get("categorie", "")
        confiance = vision.get("confiance", 0)
        if categorie == "Endommagé / Défectueux" and confiance > 0.5:
            if any(kw in rule for kw in ["remboursement", "remboursable", "remboursé", "remboursée"]):
                return "Remboursable"
            if any(kw in rule for kw in ["échange", "échangeable", "échangé", "échangée"]):
                return "Échangeable"
        if categorie == "Conforme / Bon état" and ("défaut" in rule or "endommagé" in rule):
            return "À vérifier - Incohérence"

    # Hors garantie
    if any(kw in rule for kw in ["hors garantie", "n'est plus couvert", "n'est plus éligible", "réparation payante"]):
        return "Refusé - Hors garantie"

    if any(kw in rule for kw in ["remboursement", "remboursable", "remboursé", "remboursée"]):
        return "Remboursable"
    if any(kw in rule for kw in ["échange", "échangeable", "échangé", "échangée"]):
        return "Échangeable"
    if any(kw in rule for kw in ["réparation", "devis", "réparé"]):
        return "Réparation proposée"

    return "À vérifier"
