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
        """
        self._knowledge = [
            "Un produit endommagé à la livraison est remboursable intégralement si signalé sous 48h avec photo à l'appui.",
            "Un article ne correspondant pas à la description peut être retourné sous 14 jours pour remboursement ou échange.",
            "Un colis en retard de livraison ne donne pas droit à un remboursement automatique, un dédommagement commercial peut être proposé.",
            "Un produit utilisé ou porté ne peut plus être retourné, sauf défaut de fabrication constaté.",
            "Un défaut de fabrication détecté après réception nécessite une vérification technique avant tout remboursement.",
            "Une erreur de couleur ou de taille lors de la commande est échangeable gratuitement sous 30 jours.",
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
