"""
voxtral_server.py — Serveur STT + Reranker natif Apple Silicon.

Tourne HORS Docker, directement sur le Mac, pour exploiter le GPU Metal via MLX
— exactement comme Ollama pour le LLM. L'app (dans Docker) l'appelle en HTTP via
http://host.docker.internal:8600.

STT : Whisper large-v3 (MLX, GPU) — MULTILINGUE avec auto-détection de langue
      (français ET anglais transcrits nativement, sans forcer la langue).
Rerank : bge-reranker-v2-m3 (cross-encoder) pour la pertinence du RAG.

Lancement :
    ./run.sh
ou :
    .venv/bin/uvicorn voxtral_server:app --host 127.0.0.1 --port 8600
"""

import os
import tempfile
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import mlx_whisper

# Whisper large-v3 (MLX) — multilingue, auto-détection. Turbo possible pour + de vitesse :
#   STT_MODEL=mlx-community/whisper-large-v3-turbo
STT_MODEL = os.getenv("STT_MODEL", "mlx-community/whisper-large-v3-mlx")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

_state = {"reranker": None, "stt_ready": False}


def _get_reranker():
    """Charge (une fois) le cross-encoder de reranking (pertinence calibrée)."""
    if _state["reranker"] is None:
        from sentence_transformers import CrossEncoder
        print(f"⏳ Chargement du reranker ({RERANK_MODEL})...")
        _state["reranker"] = CrossEncoder(RERANK_MODEL, max_length=512)
        print("✅ Reranker prêt")
    return _state["reranker"]


def _stt(audio, language=None) -> dict:
    """Transcrit (path ou np.ndarray) avec Whisper large-v3.
    language=None → auto-détection ; sinon force la langue."""
    return mlx_whisper.transcribe(audio, path_or_hf_repo=STT_MODEL, language=language)


def _warmup_stt():
    """Compile/charge Whisper sur du silence pour que la 1re vraie requête soit rapide."""
    print(f"⏳ Chargement de Whisper STT ({STT_MODEL})...")
    _stt(np.zeros(16000, dtype=np.float32))   # 1 s de silence à 16 kHz
    _state["stt_ready"] = True
    print("✅ Whisper STT prêt sur http://127.0.0.1:8600")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _warmup_stt()
    _get_reranker()          # précharge le reranker au démarrage
    yield
    _state["reranker"] = None


app = FastAPI(title="AI Services (STT large-v3 + Rerank) — alélo", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": STT_MODEL,
            "loaded": _state["stt_ready"],
            "reranker": _state["reranker"] is not None}


class RerankRequest(BaseModel):
    query: str
    documents: list[str]


@app.post("/rerank")
def rerank(req: RerankRequest):
    """Reclasse des documents par pertinence réelle (cross-encoder).
    Renvoie un score calibré 0-1 par document (sigmoïde du logit)."""
    import math
    if not req.documents:
        return {"scores": []}
    model = _get_reranker()
    logits = model.predict([(req.query, d) for d in req.documents])
    scores = [1.0 / (1.0 + math.exp(-float(s))) for s in logits]
    return {"scores": scores}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Reçoit un fichier audio (WAV), renvoie {'text': ..., 'language': ...}.
    La langue est DÉTECTÉE automatiquement (français, anglais, …)."""
    data = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        f.write(data)
        f.flush()
        r = _stt(f.name)                                  # auto-détection
        lang = (r.get("language") or "").lower()
        # L'app ne cible que le FR (primaire) et l'EN. Le français fortement accentué
        # (ex. ivoirien) est parfois détecté comme créole/autre → transcription phonétique
        # fausse. Dans ce cas, on RE-TRANSCRIT en forçant le français.
        if lang not in ("fr", "en"):
            r = _stt(f.name, language="fr")
            lang = "fr"
    return {"text": (r.get("text") or "").strip(), "language": lang}
