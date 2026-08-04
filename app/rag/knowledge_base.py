"""
Base de connaissances pour le système RAG.

Feature: rag-search
- Base enrichie avec des catégories et mots-clés pour chaque règle
- Support d'ajout dynamique de règles (add_rule)
- Méthode de filtrage par catégorie
- Métadonnées par règle (id, categorie, mots_cles)
"""

from typing import List, Dict, Any, Optional


class KnowledgeBase:
    """
    Base de connaissances interne contenant les politiques de support.
    Chaque règle est un dictionnaire enrichi de métadonnées.
    """

    _instance = None
    _knowledge = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KnowledgeBase, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._knowledge is None:
            self._load_knowledge()

    def _load_knowledge(self):
        """Charge la base de connaissances enrichie avec métadonnées."""
        self._knowledge = [
            {
                "id": 1,
                "texte": "Un produit endommagé à la livraison est remboursable intégralement si signalé sous 48h avec photo à l'appui.",
                "categorie": "livraison",
                "mots_cles": ["endommagé", "livraison", "remboursement", "48h", "photo"],
            },
            {
                "id": 2,
                "texte": "Un article ne correspondant pas à la description peut être retourné sous 14 jours pour remboursement ou échange.",
                "categorie": "retour",
                "mots_cles": ["description", "retour", "14 jours", "remboursement", "échange"],
            },
            {
                "id": 3,
                "texte": "Un colis en retard de livraison ne donne pas droit à un remboursement automatique, un dédommagement commercial peut être proposé.",
                "categorie": "livraison",
                "mots_cles": ["retard", "colis", "dédommagement", "remboursement"],
            },
            {
                "id": 4,
                "texte": "Un produit utilisé ou porté ne peut plus être retourné, sauf défaut de fabrication constaté.",
                "categorie": "retour",
                "mots_cles": ["utilisé", "porté", "retour", "défaut", "fabrication"],
            },
            {
                "id": 5,
                "texte": "Un défaut de fabrication détecté après réception nécessite une vérification technique avant tout remboursement.",
                "categorie": "qualite",
                "mots_cles": ["défaut", "fabrication", "vérification", "technique", "remboursement"],
            },
            {
                "id": 6,
                "texte": "Une erreur de couleur ou de taille lors de la commande est échangeable gratuitement sous 30 jours.",
                "categorie": "commande",
                "mots_cles": ["couleur", "taille", "échange", "gratuit", "30 jours"],
            },
            {
                "id": 7,
                "texte": "Un produit manquant dans un colis doit être signalé sous 72h pour déclencher un réapprovisionnement ou un remboursement partiel.",
                "categorie": "livraison",
                "mots_cles": ["manquant", "colis", "72h", "réapprovisionnement", "remboursement partiel"],
            },
            {
                "id": 8,
                "texte": "Une réclamation liée à une erreur de facturation doit être traitée sous 5 jours ouvrés avec correction de la facture.",
                "categorie": "facturation",
                "mots_cles": ["facturation", "erreur", "réclamation", "facture", "5 jours"],
            },
            {
                "id": 9,
                "texte": "Un produit hors garantie présentant un dysfonctionnement peut faire l'objet d'un devis de réparation gratuit.",
                "categorie": "garantie",
                "mots_cles": ["garantie", "dysfonctionnement", "réparation", "devis"],
            },
            {
                "id": 10,
                "texte": "Tout produit sous garantie constructeur bénéficie d'un échange standard sous 7 jours sans frais.",
                "categorie": "garantie",
                "mots_cles": ["garantie", "constructeur", "échange", "7 jours", "sans frais"],
            },
        ]

    def get_all(self) -> List[str]:
        """
        Retourne tous les textes de règles (interface de compatibilité).

        Returns:
            Liste des textes de règles.
        """
        return [r["texte"] for r in self._knowledge]

    def get_all_with_metadata(self) -> List[Dict[str, Any]]:
        """
        Retourne toutes les règles avec leurs métadonnées complètes.

        Returns:
            Liste de dictionnaires {id, texte, categorie, mots_cles}.
        """
        return list(self._knowledge)

    def get_rule(self, index: int) -> Optional[str]:
        """
        Retourne le texte d'une règle par son index.

        Args:
            index: Index dans la liste (0-based)

        Returns:
            Texte de la règle ou None si hors limites.
        """
        if 0 <= index < len(self._knowledge):
            return self._knowledge[index]["texte"]
        return None

    def get_by_id(self, rule_id: int) -> Optional[Dict[str, Any]]:
        """
        Retourne une règle complète par son identifiant.

        Args:
            rule_id: Identifiant de la règle

        Returns:
            Dictionnaire de la règle ou None si introuvable.
        """
        for rule in self._knowledge:
            if rule["id"] == rule_id:
                return rule
        return None

    def get_by_category(self, categorie: str) -> List[Dict[str, Any]]:
        """
        Retourne toutes les règles d'une catégorie donnée.

        Args:
            categorie: Catégorie à filtrer (livraison, retour, qualite, etc.)

        Returns:
            Liste des règles correspondantes.
        """
        return [r for r in self._knowledge if r["categorie"] == categorie]

    def get_categories(self) -> List[str]:
        """
        Retourne la liste des catégories disponibles (sans doublons).

        Returns:
            Liste triée des catégories.
        """
        return sorted({r["categorie"] for r in self._knowledge})

    def add_rule(self, texte: str, categorie: str, mots_cles: List[str]) -> Dict[str, Any]:
        """
        Ajoute une nouvelle règle à la base de connaissances.

        Args:
            texte     : Texte de la règle
            categorie : Catégorie de la règle
            mots_cles : Liste de mots-clés associés

        Returns:
            La règle créée avec son identifiant.
        """
        new_id = max(r["id"] for r in self._knowledge) + 1
        rule = {
            "id": new_id,
            "texte": texte,
            "categorie": categorie,
            "mots_cles": mots_cles,
        }
        self._knowledge.append(rule)
        return rule
