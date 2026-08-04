# SmartHelp — Micro-service de support client multimodal

## Lancer le service

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Puis ouvrir **http://127.0.0.1:8000/docs** pour tester via Swagger.

## Endpoint

`POST /support-ticket` (multipart/form-data)

| Champ         | Type   | Obligatoire | Description                          |
|---------------|--------|-------------|----------------------------------------|
| `audio`       | file   | non         | `.mp3` ou `.wav` — note vocale client   |
| `image`       | file   | non         | `.png`/`.jpg` — photo du produit        |
| `description` | texte  | non         | Description écrite (utilisée si pas d'audio) |

### Réponse

```json
{
  "transcribed_text": "...",
  "vision_diagnosis": "...",
  "matched_policy": "...",
  "similarity_score": 0.62,
  "proposed_status": "Remboursable"
}
```

## Comment ça marche

1. **ASR** : si un audio est fourni, il est transcrit avec `openai/whisper-small` via `pipeline("automatic-speech-recognition", ...)`.
2. **Vision** : si une image est fournie, elle est classifiée avec `openai/clip-vit-base-patch32` via `pipeline("image-classification", ...)`.
3. **RAG** : le texte (transcription ou description) est comparé par similarité cosinus à une petite base de connaissances interne (règles de retour/remboursement), embeddée avec `sentence-transformers/all-MiniLM-L6-v2` via `pipeline("feature-extraction", ...)`.
4. **Diagnostic** : la règle la plus proche + le résultat vision déterminent le statut proposé (`Remboursable`, `À vérifier`, `Refusé`).

## Optimisation

Tous les modèles sont chargés une seule fois grâce à `@lru_cache` (pattern singleton), évitant de recharger Whisper/ViT/l'encodeur à chaque requête. Les fichiers temporaires sont systématiquement supprimés (`finally`), même en cas d'erreur.

## À compléter pour le livrable complet

- Board Kanban (lien à ajouter dans ce README)
- Git Flow (branches `feature/...`, PR vers `develop`, puis `main`)
- Vidéo de démo (max 3 min) via `/docs`
