# SmartHelp — Micro-service de support client multimodal

## Lancer le service
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Puis ouvrir **http://127.0.0.1:8000/docs** pour tester via Swagger.

## Features intégrées
- **support-ticket** : endpoint principal multimodal (audio + image + texte)
- **audio-transcription** : service ASR Whisper étendu avec gestion d'erreurs
- **image-vision** : service CLIP avec analyse batch et scores explicables
- **rag-search** : base de connaissances enrichie avec recherche top-k
- **fastapi-routes** : middleware CORS, logging, health-checks étendus

## Endpoint principal
`POST /support-ticket` (multipart/form-data)

| Champ         | Type   | Obligatoire | Description                                    |
|---------------|--------|-------------|------------------------------------------------|
| `audio`       | file   | non         | `.mp3` ou `.wav` — note vocale client          |
| `image`       | file   | non         | `.png`/`.jpg` — photo du produit               |
| `description` | texte  | non         | Description écrite (utilisée si pas d'audio)   |

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
1. **ASR** : si un audio est fourni, il est transcrit avec `openai/whisper-small`.
2. **Vision** : si une image est fournie, elle est classifiée avec `openai/clip-vit-base-patch32`.
3. **RAG** : le texte est comparé par similarité cosinus à la base de connaissances, embeddée avec `sentence-transformers/all-MiniLM-L6-v2`.
4. **Diagnostic** : la règle la plus proche + le résultat vision déterminent le statut proposé (`Remboursable`, `Échangeable`, `Refusé`, `À vérifier`).

## Autres endpoints
- `POST /audio/transcribe` — transcription simple
- `POST /audio/transcribe/full` — transcription avec segments horodatés
- `POST /vision/check` — analyse d'image unique
- `POST /vision/check/batch` — analyse de plusieurs images
- `POST /rag/search` — recherche top-k dans la base de connaissances
- `GET  /rag/categories` — liste des catégories
- `GET  /rag/rules` — liste des règles
- `GET  /health/details` — statut détaillé de l'application
- `GET  /health/ready` — readiness probe

## Optimisation
Tous les modèles sont chargés une seule fois grâce au pattern singleton, évitant de recharger Whisper/CLIP/l'encodeur à chaque requête. Les fichiers temporaires sont systématiquement supprimés (`finally`), même en cas d'erreur.
