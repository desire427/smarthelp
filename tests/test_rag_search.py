"""
Tests unitaires pour RAGSearch.

Feature: rag-search
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from app.rag.knowledge_base import KnowledgeBase
from app.rag.search import RAGSearch, RAGError


@pytest.fixture(autouse=True)
def reset_singletons():
    """Réinitialise les singletons entre les tests."""
    KnowledgeBase._instance = None
    KnowledgeBase._knowledge = None
    yield
    KnowledgeBase._instance = None
    KnowledgeBase._knowledge = None


def _make_rag_with_mock_embeddings():
    """
    Crée un RAGSearch avec EmbeddingService mocké.
    Chaque texte reçoit un vecteur déterministe basé sur sa longueur.
    """
    with patch("app.services.embedding_service.pipeline"):
        rag = RAGSearch()
        # Embedding mock : vecteur basé sur le hash du texte
        def fake_embed(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            return rng.random(32).astype(np.float32)

        rag.embedding_service.embed = fake_embed
        rag._embeddings_cache = None  # forcer recalcul avec fake_embed
        return rag


class TestRAGSearchBasic:
    def test_search_returns_tuple(self):
        rag = _make_rag_with_mock_embeddings()
        result, score = rag.search("produit endommagé à la livraison")
        assert isinstance(result, str)
        assert isinstance(score, float)

    def test_search_empty_query_returns_empty(self):
        rag = _make_rag_with_mock_embeddings()
        result, score = rag.search("   ")
        assert result == ""
        assert score == 0.0

    def test_search_top_k_returns_k_results(self):
        rag = _make_rag_with_mock_embeddings()
        results = rag.search_top_k("retard de livraison", k=3)
        assert len(results) == 3

    def test_search_top_k_results_have_required_keys(self):
        rag = _make_rag_with_mock_embeddings()
        results = rag.search_top_k("défaut de fabrication", k=2)
        for r in results:
            assert "id" in r
            assert "texte" in r
            assert "categorie" in r
            assert "score" in r

    def test_search_top_k_sorted_by_score_desc(self):
        rag = _make_rag_with_mock_embeddings()
        results = rag.search_top_k("erreur de commande", k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_k_empty_query_raises(self):
        rag = _make_rag_with_mock_embeddings()
        with pytest.raises(RAGError, match="vide"):
            rag.search_top_k("")

    def test_search_top_k_invalid_k_raises(self):
        rag = _make_rag_with_mock_embeddings()
        with pytest.raises(RAGError, match="k doit être"):
            rag.search_top_k("test", k=0)


class TestRAGSearchFiltering:
    def test_filter_by_category_livraison(self):
        rag = _make_rag_with_mock_embeddings()
        results = rag.search_top_k("colis endommagé", k=5, categorie="livraison")
        for r in results:
            assert r["categorie"] == "livraison"

    def test_filter_by_unknown_category_returns_empty(self):
        rag = _make_rag_with_mock_embeddings()
        results = rag.search_top_k("quelque chose", k=3, categorie="categorie_inconnue_xyz")
        assert results == []

    def test_invalidate_cache(self):
        rag = _make_rag_with_mock_embeddings()
        _ = rag._get_kb_embeddings()  # peuple le cache
        assert rag._embeddings_cache is not None
        rag.invalidate_cache()
        assert rag._embeddings_cache is None


class TestRAGSearchProposeStatus:
    def test_low_similarity_returns_a_verifier(self):
        rag = _make_rag_with_mock_embeddings()
        status = rag.propose_status("quelque règle", 0.1)
        assert status == "À vérifier"

    def test_remboursement_rule_returns_remboursable(self):
        rag = _make_rag_with_mock_embeddings()
        status = rag.propose_status("produit remboursable sous conditions", 0.8)
        assert status == "Remboursable"

    def test_echange_rule_returns_echangeable(self):
        rag = _make_rag_with_mock_embeddings()
        status = rag.propose_status("article échangeable gratuitement", 0.7)
        assert status == "Échangeable"

    def test_vision_defect_with_remboursement_rule(self):
        rag = _make_rag_with_mock_embeddings()
        vision = {"categorie": "Endommagé / Défectueux", "confiance": 0.9}
        status = rag.propose_status("produit remboursable intégralement", 0.75, vision)
        assert status == "Remboursable"
