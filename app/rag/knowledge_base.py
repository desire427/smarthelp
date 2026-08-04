"""
Base de connaissances pour le système RAG.
Contient les politiques et règles de support client.
"""


class KnowledgeBase:
    """
    Base de connaissances interne contenant les politiques de support.
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
        """
        Charge la base de connaissances.
        Chaque règle est rédigée en langage naturel et couvre un cas client réel.
        Les variantes de formulation améliorent la similarité cosinus avec les
        descriptions clients imparfaites (fautes, langage parlé).
        """
        self._knowledge = [
            # --- Produit récent qui ne fonctionne plus (< 30 jours) ---
            "Un produit qui ne fonctionne plus ou qui tombe en panne peu après l'achat peut être échangé ou remboursé sous 30 jours.",
            "Un article acheté récemment qui ne marche plus, ne s'allume plus ou présente un dysfonctionnement peut faire l'objet d'un échange immédiat.",
            "Si votre produit ne fonctionne plus le lendemain de l'achat ou dans les premiers jours, un remplacement ou remboursement est possible.",

            # --- Produit en panne acheté il y a plus de 2 ans (hors garantie légale) ---
            "Un produit acheté il y a plus de 2 ans qui tombe en panne n'est plus couvert par la garantie légale ; un devis de réparation peut être proposé mais le remplacement gratuit est refusé.",
            "Un appareil acheté depuis plusieurs années (2 ans ou plus) et qui ne fonctionne plus est hors garantie légale ; aucun remboursement ou échange gratuit n'est dû.",
            "Un téléphone, ordinateur ou appareil électronique en panne après 2 ans d'utilisation n'est plus éligible à l'échange gratuit ; seule une réparation payante ou un devis peut être proposé.",

            # --- Produit en panne sous garantie (< 2 ans) ---
            "La garantie légale de conformité couvre tout produit défectueux dans les 2 ans suivant l'achat ; un échange ou remboursement est possible.",
            "Un produit en panne couvert par la garantie (moins de 2 ans) peut être réparé, échangé ou remboursé selon les conditions du fabricant.",
            "Tout produit sous garantie constructeur bénéficie d'un échange standard sans frais dans un délai de 7 jours après signalement.",

            # --- Échange / retour standard ---
            "Un article ne correspondant pas à la description peut être retourné sous 14 jours pour remboursement ou échange.",
            "Un produit que vous souhaitez renvoyer pour en prendre un autre peut être échangé gratuitement sous 30 jours si non utilisé.",
            "Il est possible de renvoyer un article et d'en commander un autre en échange s'il n'a pas été utilisé et que la demande est faite sous 14 jours.",

            # --- Produit endommagé à la livraison ---
            "Un produit endommagé à la livraison est remboursable intégralement si signalé sous 48h avec photo à l'appui.",
            "Si votre colis est arrivé abîmé ou cassé, vous pouvez demander un remboursement complet en signalant le problème dans les 48 heures.",

            # --- Défaut de fabrication ---
            "Un produit utilisé ou porté ne peut plus être retourné, sauf défaut de fabrication constaté.",
            "Un défaut de fabrication détecté après réception nécessite une vérification technique avant tout remboursement.",
            "Si votre produit présente un défaut dès la sortie de la boîte, un échange ou remboursement est possible après vérification.",

            # --- Livraison / retard ---
            "Un colis en retard de livraison ne donne pas droit à un remboursement automatique, un dédommagement commercial peut être proposé.",
            "Un colis non reçu après la date de livraison prévue donne droit à une enquête transporteur et un éventuel remboursement sous 15 jours.",

            # --- Erreur de commande ---
            "Une erreur de couleur ou de taille lors de la commande est échangeable gratuitement sous 30 jours.",
            "Si vous avez reçu le mauvais produit ou une mauvaise référence, un échange ou remboursement est possible sans frais.",

            # --- Facturation ---
            "Une réclamation liée à une erreur de facturation doit être traitée sous 5 jours ouvrés avec correction de la facture.",
        ]

    def get_all(self):
        """
        Retourne toutes les règles de la base de connaissances.

        Returns:
            Liste des règles
        """
        return self._knowledge

    def get_rule(self, index: int):
        """
        Retourne une règle spécifique.

        Args:
            index: Index de la règle

        Returns:
            Règle correspondante
        """
        if 0 <= index < len(self._knowledge):
            return self._knowledge[index]
        return None
