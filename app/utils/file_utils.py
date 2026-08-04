"""
Utilitaires pour la gestion des fichiers.
"""

import os
import shutil
import tempfile
from fastapi import UploadFile

# Extensions autorisées
ALLOWED_AUDIO = {".mp3", ".wav"}
ALLOWED_IMAGE = {".png", ".jpg", ".jpeg", ".webp"}

def save_temp_file(upload: UploadFile) -> str:
    """
    Sauvegarde un fichier uploadé dans un fichier temporaire.
    
    Args:
        upload: Fichier uploadé via FastAPI
        
    Returns:
        Chemin du fichier temporaire
    """
    suffix = os.path.splitext(upload.filename)[1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with tmp:
        shutil.copyfileobj(upload.file, tmp)
    return tmp.name

def cleanup_temp_files(files: list):
    """
    Supprime les fichiers temporaires.
    
    Args:
        files: Liste des chemins de fichiers à supprimer
    """
    for f in files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass  # Ignorer les erreurs de suppression
