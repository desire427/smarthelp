"""
Routes dédiées à la recherche RAG.

Feature: rag-search
- POST /rag/search         : recherche top-k dans la base de connaissances
- GET  /rag/categories     : liste les catégories disponibles
- GET  /rag/rules          : liste toutes les règles avec métadonnées
- POST /rag/rules          : ajoute une nouvelle règle dynamiquement
"""

from typing import List, Optional
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.rag.search import RAGSearch, RAGError
from app.rag.knowledge_base import KnowledgeBase

router = APIRouter(prefix="/rag", tags=["RAG Search"])


# --------------------------------------------------------------------------
# Schémas de requête / réponse
# --------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Texte de la requête")
    k: int = Field(default=3, ge=1, le=10, description="Nombre de résultats souhaités")
    categorie: Optional[str] = Field(default=None, description="Filtrer par catégorie")


class NewRuleRequest(BaseModel):
    texte: str = Field(..., min_length=10, description="Texte de la règle")
    categorie: str = Field(..., min_length=1, description="Catégorie de la règle")
    mots_cles: List[str] = Field(..., min_items=1, description="Mots-clés associés")


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@router.post("/search")
async def rag_search(body: SearchRequest):
    """
    Recherche les règles les plus pertinentes pour une requête donnée.

    - **query** : texte décrivant le problème du client
    - **k** : nombre de résultats (1 à 10, défaut 3)
    - **categorie** : filtre optionnel (livraison, retour, qualite, garantie, facturation, commande)

    Retourne les k règles les plus similaires avec leur score de similarité cosinus.
    """
    try:
        rag = RAGSearch()
        results = rag.search_top_k(
            query=body.query,
            k=body.k,
            categorie=body.categorie,
        )
        return {
            "query": body.query,
            "categorie_filtre": body.categorie,
            "k": body.k,
            "results": results,
        }
    except RAGError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Erreur serveur: {exc}"})


@router.get("/categories")
async def get_categories():
    """
    Liste toutes les catégories disponibles dans la base de connaissances.
    """
    kb = KnowledgeBase()
    return {"categories": kb.get_categories()}


@router.get("/rules")
async def get_rules(categorie: Optional[str] = Query(default=None)):
    """
    Liste toutes les règles de la base de connaissances.

    - **categorie** : filtre optionnel sur la catégorie
    """
    kb = KnowledgeBase()
    if categorie:
        rules = kb.get_by_category(categorie)
    else:
        rules = kb.get_all_with_metadata()
    return {"count": len(rules), "rules": rules}


@router.post("/rules", status_code=201)
async def add_rule(body: NewRuleRequest):
    """
    Ajoute une nouvelle règle à la base de connaissances en mémoire.

    Note : les règles ajoutées dynamiquement sont perdues au redémarrage
    (base non persistante dans cette version).
    """
    try:
        kb = KnowledgeBase()
        # Invalider le cache des embeddings après ajout
        rag = RAGSearch()
        rule = kb.add_rule(
            texte=body.texte,
            categorie=body.categorie,
            mots_cles=body.mots_cles,
        )
        rag.invalidate_cache()
        return {"message": "Règle ajoutée avec succès.", "rule": rule}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Erreur serveur: {exc}"})
