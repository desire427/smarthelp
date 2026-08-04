"""
Tests unitaires pour KnowledgeBase.

Feature: rag-search
"""

import pytest
from app.rag.knowledge_base import KnowledgeBase


@pytest.fixture(autouse=True)
def reset_singleton():
    """Réinitialise le singleton entre les tests."""
    KnowledgeBase._instance = None
    KnowledgeBase._knowledge = None
    yield
    KnowledgeBase._instance = None
    KnowledgeBase._knowledge = None


class TestKnowledgeBaseBase:
    def test_get_all_returns_strings(self):
        kb = KnowledgeBase()
        rules = kb.get_all()
        assert isinstance(rules, list)
        assert all(isinstance(r, str) for r in rules)

    def test_get_all_not_empty(self):
        kb = KnowledgeBase()
        assert len(kb.get_all()) >= 6

    def test_get_all_with_metadata_has_required_keys(self):
        kb = KnowledgeBase()
        for rule in kb.get_all_with_metadata():
            assert "id" in rule
            assert "texte" in rule
            assert "categorie" in rule
            assert "mots_cles" in rule

    def test_get_rule_valid_index(self):
        kb = KnowledgeBase()
        rule = kb.get_rule(0)
        assert rule is not None
        assert isinstance(rule, str)

    def test_get_rule_invalid_index_returns_none(self):
        kb = KnowledgeBase()
        assert kb.get_rule(9999) is None


class TestKnowledgeBaseFilters:
    def test_get_categories_returns_list(self):
        kb = KnowledgeBase()
        cats = kb.get_categories()
        assert isinstance(cats, list)
        assert len(cats) > 0

    def test_get_categories_contains_livraison(self):
        kb = KnowledgeBase()
        assert "livraison" in kb.get_categories()

    def test_get_by_category_returns_correct_category(self):
        kb = KnowledgeBase()
        rules = kb.get_by_category("livraison")
        assert len(rules) > 0
        for rule in rules:
            assert rule["categorie"] == "livraison"

    def test_get_by_category_unknown_returns_empty(self):
        kb = KnowledgeBase()
        rules = kb.get_by_category("inexistant_xyz")
        assert rules == []

    def test_get_by_id_returns_correct_rule(self):
        kb = KnowledgeBase()
        rule = kb.get_by_id(1)
        assert rule is not None
        assert rule["id"] == 1

    def test_get_by_id_unknown_returns_none(self):
        kb = KnowledgeBase()
        assert kb.get_by_id(99999) is None


class TestKnowledgeBaseAddRule:
    def test_add_rule_increases_count(self):
        kb = KnowledgeBase()
        before = len(kb.get_all())
        kb.add_rule(
            texte="Une nouvelle règle de test pour vérifier l'ajout dynamique.",
            categorie="test",
            mots_cles=["test", "dynamique"],
        )
        assert len(kb.get_all()) == before + 1

    def test_add_rule_returns_new_rule(self):
        kb = KnowledgeBase()
        rule = kb.add_rule(
            texte="Règle ajoutée dynamiquement avec un texte suffisamment long.",
            categorie="garantie",
            mots_cles=["garantie", "ajout"],
        )
        assert rule["categorie"] == "garantie"
        assert "id" in rule
        assert rule["id"] > 0

    def test_add_rule_retrievable_by_id(self):
        kb = KnowledgeBase()
        added = kb.add_rule(
            texte="Règle récupérable par son identifiant unique.",
            categorie="facturation",
            mots_cles=["facture"],
        )
        retrieved = kb.get_by_id(added["id"])
        assert retrieved is not None
        assert retrieved["texte"] == added["texte"]
