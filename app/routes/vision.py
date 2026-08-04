"""
Routes dédiées à l'analyse visuelle d'images.

Feature: image-vision
- POST /vision/check       : analyse une image unique, retourne catégorie + scores
- POST /vision/check/batch : analyse plusieurs images en une seule requête
"""

from typing import List
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from app.services.vision_service import VisionService, VisionError
from app.utils.file_utils import save_temp_file, cleanup_temp_files

router = APIRouter(prefix="/vision", tags=["Image Vision"])


@router.post("/check")
async def vision_check(image: UploadFile = File(...)):
    """
    Analyse une image pour déterminer si le produit est conforme ou défectueux.

    Formats supportés : .png, .jpg, .jpeg, .webp, .bmp, .tiff

    Retourne :
    - **categorie** : "Conforme / Bon état", "Endommagé / Défectueux" ou "Incertain"
    - **confiance** : score de confiance entre 0 et 1
    - **score_conforme** / **score_defaut** : scores normalisés
    - **top3_prompts** : les 3 prompts CLIP les plus activés
    """
    tmp_files = []
    try:
        path = save_temp_file(image)
        tmp_files.append(path)

        service = VisionService()
        result = service.analyze_image(path)

        return {"filename": image.filename, **result}

    except VisionError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Erreur serveur: {exc}"})
    finally:
        cleanup_temp_files(tmp_files)


@router.post("/check/batch")
async def vision_check_batch(images: List[UploadFile] = File(...)):
    """
    Analyse plusieurs images en une seule requête.

    Retourne une liste de résultats, un par image.
    Les images invalides incluent une clé "error" au lieu des scores.
    """
    tmp_files = []
    try:
        paths = []
        for img in images:
            path = save_temp_file(img)
            tmp_files.append(path)
            paths.append((img.filename, path))

        service = VisionService()
        results = []
        for filename, path in paths:
            try:
                result = service.analyze_image(path)
                results.append({"filename": filename, **result})
            except VisionError as exc:
                results.append({"filename": filename, "error": str(exc)})

        return {"count": len(results), "results": results}

    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Erreur serveur: {exc}"})
    finally:
        cleanup_temp_files(tmp_files)
