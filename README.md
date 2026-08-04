# SmartHelp — Guide complet

SmartHelp est un micro-service de support client multimodal. Il accepte un audio, une image et/ou une description textuelle, puis retourne automatiquement une règle de politique applicable et un statut proposé (Remboursable, Échangeable, Refusé, etc.).

---

## Table des matières

1. [Architecture globale](#1-architecture-globale)
2. [Prérequis](#2-prérequis)
3. [Installation](#3-installation)
4. [Configuration](#4-configuration)
5. [Lancer le service](#5-lancer-le-service)
6. [Structure du projet](#6-structure-du-projet)
7. [Comment ça marche (pipeline complet)](#7-comment-ça-marche-pipeline-complet)
8. [Tous les endpoints](#8-tous-les-endpoints)
9. [Exemples de requêtes](#9-exemples-de-requêtes)
10. [Base de connaissances RAG](#10-base-de-connaissances-rag)
11. [Branches Git et workflow](#11-branches-git-et-workflow)
12. [Tests](#12-tests)
13. [Dépannage](#13-dépannage)

---

## 1. Architecture globale

```
Client (Swagger / curl / frontend)
        │
        ▼
┌─────────────────────────────────────────┐
│           FastAPI (main.py)             │
│  Middleware CORS + Logging + Erreurs    │
└────────────┬────────────────────────────┘
             │
    POST /support-ticket
             │
    ┌────────┴─────────┐
    │                  │
    ▼                  ▼
[Audio fourni?]   [Image fournie?]
    │                  │
    ▼                  ▼
ASRService        VisionService
(Whisper)         (CLIP)
    │                  │
    └────────┬─────────┘
             │
             ▼
       texte (transcrit
       ou description)
             │
             ▼
       EmbeddingService
    (all-MiniLM-L6-v2)
             │
             ▼
        RAGSearch
    (similarité cosinus)
             │
             ▼
    KnowledgeBase (22 règles)
             │
             ▼
      propose_status()
             │
             ▼
        Réponse JSON
```

---

## 2. Prérequis

| Outil | Version minimale | Vérification |
|-------|-----------------|--------------|
| Python | 3.10+ | `python3 --version` |
| pip | 23+ | `pip --version` |
| ffmpeg | toute version récente | `ffmpeg -version` |
| Git | 2.x | `git --version` |

### Installer ffmpeg (obligatoire pour l'audio)

```bash
sudo apt-get install -y ffmpeg
```

Sans ffmpeg, le service plante dès qu'un fichier audio est envoyé.

---

## 3. Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/desire427/smarthelp.git
cd smarthelp

# 2. Créer et activer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

Les dépendances installées :

| Package | Rôle |
|---------|------|
| `fastapi` | Framework web |
| `uvicorn[standard]` | Serveur ASGI |
| `python-multipart` | Upload de fichiers |
| `transformers` | Whisper, CLIP, sentence-transformers |
| `torch` | Moteur de calcul des modèles |
| `numpy` | Calcul de similarité cosinus |
| `pillow` | Chargement et traitement des images |
| `sentence-transformers` | Modèle d'embeddings |
| `python-dotenv` | Chargement du fichier `.env` |

---

## 4. Configuration

Le fichier `.env` à la racine du projet contient tous les paramètres configurables :

```env
# Modèles IA utilisés
ASR_MODEL=openai/whisper-small
CLIP_MODEL=openai/clip-vit-base-patch32
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Seuils de décision
SIMILARITY_THRESHOLD=0.35
VISION_CONFIDENCE_THRESHOLD=0.5
CONFORMITY_THRESHOLD=0.6
```

### Explication des seuils

| Variable | Défaut | Rôle |
|----------|--------|------|
| `SIMILARITY_THRESHOLD` | `0.35` | En dessous de ce score RAG, le statut est "À vérifier" |
| `VISION_CONFIDENCE_THRESHOLD` | `0.5` | Confiance minimale pour que la vision influence le statut |
| `CONFORMITY_THRESHOLD` | `0.6` | Score minimal pour classer une image comme "Conforme" |

### Changer de modèle Whisper

Pour traiter plus vite (moins précis) :
```env
ASR_MODEL=openai/whisper-tiny
```

Pour une meilleure précision (plus lent) :
```env
ASR_MODEL=openai/whisper-medium
```

---

## 5. Lancer le service

```bash
# Depuis la racine du projet, avec l'environnement virtuel activé
uvicorn app.main:app --reload
```

Le service démarre sur **http://127.0.0.1:8000**

- **Swagger UI** (interface de test) : http://127.0.0.1:8000/docs
- **ReDoc** (documentation) : http://127.0.0.1:8000/redoc
- **Health check** : http://127.0.0.1:8000/health

> Le flag `--reload` recharge automatiquement le serveur à chaque modification de fichier. À retirer en production.

---

## 6. Structure du projet

```
smarthelp/
├── .env                          # Variables d'environnement (non commité)
├── .gitignore
├── README.md
├── requirements.txt
│
├── app/
│   ├── __init__.py
│   ├── main.py                   # Point d'entrée FastAPI, middlewares, routes
│   │
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── logging_middleware.py # Ajoute X-Request-ID et X-Process-Time
│   │   └── error_handler.py     # Gestionnaires d'erreurs globaux (HTTP, validation, 500)
│   │
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── support.py            # POST /support-ticket (endpoint principal)
│   │   ├── audio.py              # POST /audio/transcribe et /audio/transcribe/full
│   │   ├── vision.py             # POST /vision/check et /vision/check/batch
│   │   ├── rag.py                # POST /rag/search, GET /rag/categories et /rag/rules
│   │   └── health.py             # GET /health, /health/details, /health/ready
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── asr_service.py        # Transcription audio avec Whisper
│   │   ├── vision_service.py     # Classification d'images avec CLIP
│   │   └── embedding_service.py  # Embeddings de texte avec all-MiniLM-L6-v2
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── knowledge_base.py     # 22 règles de politique client (singleton)
│   │   └── search.py             # Recherche par similarité cosinus + propose_status()
│   │
│   └── utils/
│       ├── __init__.py
│       └── file_utils.py         # save_temp_file(), cleanup_temp_files()
│
└── tests/
    ├── __init__.py
    ├── test_asr_service.py
    ├── test_vision_service.py
    ├── test_knowledge_base.py
    ├── test_rag_search.py
    └── test_health_routes.py
```

---

## 7. Comment ça marche (pipeline complet)

### Étape 1 — Réception de la requête

Le client envoie une requête `POST /support-ticket` avec un ou plusieurs de ces champs :
- `audio` : fichier `.mp3` ou `.wav`
- `image` : fichier `.png`, `.jpg`, `.jpeg` ou `.webp`
- `description` : texte libre

### Étape 2 — Transcription audio (ASR)

Si un audio est fourni, `ASRService` le transcrit avec le modèle Whisper via HuggingFace `transformers`.

- Le modèle est chargé **une seule fois** au premier appel (pattern singleton)
- Les fichiers de plus de 30 secondes sont découpés automatiquement en segments (`chunk_length_s=30`) et reconstitués
- Le texte transcrit remplace la description pour la recherche RAG

```python
# Comportement interne
result = pipeline("automatic-speech-recognition", model="openai/whisper-small")
result(audio_path, return_timestamps=True, chunk_length_s=30)
# → {"text": "J'ai acheté mon téléphone hier et...", "chunks": [...]}
```

### Étape 3 — Analyse d'image (Vision)

Si une image est fournie, `VisionService` l'analyse avec le modèle CLIP.

CLIP compare l'image à deux ensembles de prompts textuels :
- **14 prompts "Conforme"** : "clean smooth metal surface", "defect-free industrial surface", etc.
- **16 prompts "Défectueux"** : "corroded surface", "cracked surface", etc.

Les probabilités sont normalisées pour obtenir `score_conforme` et `score_defaut`.

**Règle de décision :**

| Condition | Catégorie retournée |
|-----------|---------------------|
| `score_defaut > 0.5` | Endommagé / Défectueux |
| `score_conforme > CONFORMITY_THRESHOLD (0.6)` | Conforme / Bon état |
| Sinon | Incertain |

### Étape 4 — Recherche RAG

`RAGSearch` compare le texte (transcription ou description) à la base de connaissances.

1. Le texte est transformé en vecteur par `EmbeddingService` (modèle `all-MiniLM-L6-v2`)
2. Chaque règle de la base a aussi son vecteur (calculé une fois, mis en cache)
3. La **similarité cosinus** est calculée entre le vecteur requête et chaque vecteur règle
4. La règle avec le score le plus élevé est retournée

```
score = (A · B) / (||A|| × ||B||)
```

### Étape 5 — Proposition de statut

`propose_status()` combine le score RAG et le diagnostic image pour décider :

| Condition | Statut |
|-----------|--------|
| `similarity < 0.35` | À vérifier |
| Règle contient "hors garantie", "n'est plus éligible"... | Refusé - Hors garantie |
| Image "Endommagée" + règle "remboursement" | Remboursable |
| Image "Conforme" + règle sur "défaut" | À vérifier - Incohérence |
| Règle contient "remboursable" ou "remboursement" | Remboursable |
| Règle contient "échange" ou "échangeable" | Échangeable |
| Règle contient "devis" ou "réparation" | Réparation proposée |

---

## 8. Tous les endpoints

### POST `/support-ticket`

Endpoint principal. Traite un ticket support multimodal.

**Paramètres (multipart/form-data) :**

| Champ | Type | Obligatoire | Formats acceptés |
|-------|------|-------------|-----------------|
| `audio` | file | non | `.mp3`, `.wav` |
| `image` | file | non | `.png`, `.jpg`, `.jpeg`, `.webp` |
| `description` | string | non | texte libre |

> Au moins un des trois champs doit être rempli pour une recherche RAG utile.

**Réponse 200 :**
```json
{
  "transcribed_text": "J'ai acheté mon téléphone hier et aujourd'hui ça ne marche plus.",
  "vision_diagnosis": {
    "categorie": "Conforme / Bon état",
    "confiance": 0.73,
    "score_conforme": 0.73,
    "score_defaut": 0.27
  },
  "matched_policy": "Un article acheté récemment qui ne marche plus peut faire l'objet d'un échange immédiat.",
  "similarity_score": 0.621,
  "proposed_status": "Échangeable"
}
```

---

### POST `/audio/transcribe`

Transcription simple d'un fichier audio.

**Paramètres :** `audio` (file) — `.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`, `.webm`

**Réponse :**
```json
{
  "filename": "note.mp3",
  "transcription": "Mon colis est arrivé cassé."
}
```

---

### POST `/audio/transcribe/full`

Transcription avec segments horodatés et langue détectée.

**Réponse :**
```json
{
  "filename": "note.mp3",
  "transcription": "Mon colis est arrivé cassé.",
  "language": "fr",
  "chunks": [
    {"timestamp": [0.0, 2.4], "text": "Mon colis est arrivé cassé."}
  ]
}
```

---

### POST `/vision/check`

Analyse une image unique.

**Paramètres :** `image` (file) — `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`

**Réponse :**
```json
{
  "filename": "produit.jpg",
  "categorie": "Endommagé / Défectueux",
  "confiance": 0.81,
  "score_conforme": 0.19,
  "score_defaut": 0.81
}
```

---

### POST `/vision/check/batch`

Analyse plusieurs images en une seule requête.

**Paramètres :** `images` (liste de files)

**Réponse :**
```json
{
  "count": 2,
  "results": [
    {"filename": "a.jpg", "categorie": "Conforme / Bon état", "confiance": 0.72, ...},
    {"filename": "b.jpg", "categorie": "Endommagé / Défectueux", "confiance": 0.88, ...}
  ]
}
```

---

### POST `/rag/search`

Recherche les règles les plus pertinentes pour une requête.

**Corps JSON :**
```json
{
  "query": "Mon produit est tombé en panne après 3 ans",
  "k": 3
}
```

**Réponse :**
```json
{
  "query": "Mon produit est tombé en panne après 3 ans",
  "k": 3,
  "results": [
    {
      "texte": "Un téléphone en panne après 2 ans n'est plus éligible à l'échange gratuit...",
      "score": 0.701
    }
  ]
}
```

---

### GET `/rag/rules`

Liste toutes les règles de la base de connaissances.

---

### GET `/health`

Retourne `{"status": "healthy"}` si le service tourne. Utilisé par les load balancers.

---

### GET `/health/details`

Retourne l'état détaillé : uptime, variables d'environnement, versions.

**Réponse :**
```json
{
  "status": "healthy",
  "uptime_seconds": 142.3,
  "started_at": "2026-08-04T13:30:00+00:00",
  "python_version": "3.12.0",
  "environment": {
    "asr_model": "openai/whisper-small",
    "clip_model": "openai/clip-vit-base-patch32",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
  }
}
```

---

### GET `/health/ready`

Readiness probe : retourne `200` si toutes les variables d'env sont définies, `503` sinon.

---

## 9. Exemples de requêtes

### curl — description seulement

```bash
curl -X POST http://127.0.0.1:8000/support-ticket \
  -F "description=J'ai acheté mon téléphone hier et il ne s'allume plus."
```

### curl — audio + image

```bash
curl -X POST http://127.0.0.1:8000/support-ticket \
  -F "audio=@note_vocale.mp3" \
  -F "image=@photo_produit.jpg"
```

### curl — transcription full

```bash
curl -X POST http://127.0.0.1:8000/audio/transcribe/full \
  -F "audio=@note_vocale.mp3"
```

### curl — recherche RAG directe

```bash
curl -X POST http://127.0.0.1:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "produit cassé à la livraison", "k": 2}'
```

---

## 10. Base de connaissances RAG

La base contient **22 règles** réparties en catégories. Chaque règle est formulée en langage naturel avec des variantes pour améliorer la correspondance avec les descriptions clients imparfaites.

| Catégorie | Nb de règles | Exemples de cas couverts |
|-----------|-------------|--------------------------|
| Panne récente (< 30j) | 3 | "ça ne marche plus", "ne s'allume plus" |
| Hors garantie (> 2 ans) | 3 | "acheté il y a 3 ans", "plusieurs années" |
| Sous garantie (< 2 ans) | 3 | "garantie constructeur", "garantie légale" |
| Retour / échange | 3 | "renvoyer", "prendre un autre" |
| Livraison endommagée | 2 | "colis abîmé", "arrivé cassé" |
| Défaut de fabrication | 3 | "défaut dès la boîte", "défaut constaté" |
| Livraison / retard | 2 | "retard", "colis non reçu" |
| Erreur de commande | 2 | "mauvaise couleur", "mauvaise référence" |
| Facturation | 1 | "erreur de facturation" |

---

## 11. Branches Git et workflow

Le projet suit un workflow Git Flow simplifié.

```
master          ← production (miroir de develop)
  └── develop   ← intégration de toutes les features
        ├── feature/support-ticket-api    ← code de base
        ├── feature/audio-transcription   ← service ASR
        ├── feature/image-vision          ← service CLIP
        ├── feature/rag-search            ← moteur RAG
        └── feature/fastapi-routes        ← middlewares et health-checks
```

### Rôle de chaque branche

| Branche | Contenu spécifique |
|---------|-------------------|
| `feature/support-ticket-api` | Structure de base FastAPI, endpoint `/support-ticket`, tous les services initiaux |
| `feature/audio-transcription` | `asr_service.py` amélioré, route `/audio/transcribe`, support `.ogg/.flac/.m4a/.webm` |
| `feature/image-vision` | `vision_service.py` amélioré, route `/vision/check`, mode batch, `top3_prompts` |
| `feature/rag-search` | `knowledge_base.py` enrichi (22 règles), route `/rag/search`, filtrage par catégorie |
| `feature/fastapi-routes` | Middleware CORS et logging, gestionnaires d'erreurs globaux, routes `/health` |
| `develop` | Merge de toutes les features — version complète de l'application |
| `master` | Miroir de `develop` — version stable de référence |

---

## 12. Tests

Les tests unitaires utilisent `pytest` avec des mocks pour éviter de charger les modèles IA.

```bash
# Lancer tous les tests
pytest tests/ -v

# Lancer un fichier de tests spécifique
pytest tests/test_rag_search.py -v

# Avec couverture de code
pytest tests/ --cov=app
```

| Fichier de test | Ce qui est testé |
|----------------|-----------------|
| `test_asr_service.py` | Formats supportés, transcription, fichier manquant, format invalide |
| `test_vision_service.py` | Validation image, analyse batch, fichier manquant, format invalide |
| `test_knowledge_base.py` | Chargement, filtrage par catégorie, ajout dynamique de règle |
| `test_rag_search.py` | Recherche top-k, filtrage, tri par score, propose_status() |
| `test_health_routes.py` | Statuts HTTP, headers X-Request-ID et X-Process-Time, readiness |

---

## 13. Dépannage

### `ffmpeg was not found`
```bash
sudo apt-get install -y ffmpeg
```

### `You have passed more than 3000 mel input features`
Le fichier audio dépasse 30 secondes. Ce bug est corrigé dans `asr_service.py` avec `chunk_length_s=30`. Si tu vois encore cette erreur, vérifie que le serveur a bien rechargé le fichier (`--reload` ou redémarre manuellement).

### `ModuleNotFoundError: No module named 'app'`
Tu lances uvicorn depuis le mauvais répertoire. Il faut être à la racine du projet :
```bash
cd ~/smarthelp
uvicorn app.main:app --reload
```

### Le modèle Whisper ne se télécharge pas
Whisper se télécharge automatiquement depuis HuggingFace au premier lancement (~250 Mo pour `whisper-small`). Si le téléchargement échoue, crée un compte sur huggingface.co et définis la variable :
```bash
export HF_TOKEN=ton_token_ici
```

### Statut toujours "À vérifier"
Le score de similarité est en dessous de `0.35`. Cela peut arriver si :
- La description est très courte ou sans mots-clés clairs
- La requête est en anglais alors que la base est en français

Solution : enrichis la description avec plus de détails, ou ajoute des règles correspondantes dans `knowledge_base.py`.
