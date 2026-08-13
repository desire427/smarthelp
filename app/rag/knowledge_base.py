"""
Base de connaissances — lit les fichiers .txt du dossier data/.
Chaque ligne non vide est une règle indexée dans le RAG.
"""

import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_knowledge() -> list[str]:
    """Charge toutes les règles depuis les fichiers .txt de data/."""
    rules = []
    for filepath in sorted(DATA_DIR.glob("*.txt")):
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rules.append(line)
    return rules
