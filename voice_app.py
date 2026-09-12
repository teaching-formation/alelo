"""
voice_app.py — Mode Vocal immersif de la CGECI.ai (page dédiée type "voice mode").

Sert une page avec un orbe animé qui écoute, réfléchit et répond à voix haute,
en conversation continue. Orchestre toute la chaîne 100% locale :

    navigateur (micro)  →  Voxtral (STT, GPU natif)  →  RAG (ChromaDB+Ollama)
                        →  Piper (TTS)  →  audio renvoyé au navigateur

Lancé dans le conteneur Docker sur le port 8502 (voir docker-compose : service "voice").
"""

import os
import json
import base64
import logging

import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_app")

HERE = os.path.dirname(os.path.abspath(__file__))
VOXTRAL_URL = os.getenv("VOXTRAL_URL", "http://host.docker.internal:8600")
DEFAULT_MODEL = os.getenv("VOICE_MODEL", "cgeci")

app = FastAPI(title="CGECI.ai — Mode Vocal")

# Chargés une seule fois au démarrage
_vectordb = None


def _get_vectordb():
    global _vectordb
    if _vectordb is None:
        from rag_chain import load_vectordb, _build_bm25_index
        logger.info("Chargement ChromaDB + index BM25...")
        _vectordb = load_vectordb()
        _build_bm25_index(_vectordb)
        logger.info("Base prête.")
    return _vectordb


@app.on_event("startup")
def _startup():
    # Pré-chauffe la base et la voix Piper pour éviter la latence au 1er tour
    try:
        _get_vectordb()
        from voice import _get_piper
        _get_piper()
        logger.info("Mode Vocal prêt.")
    except Exception as e:
        logger.warning("Préchauffage partiel : %s", e)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "voice_mode.html"))


@app.post("/voice")
async def voice(
    audio: UploadFile = File(...),
    history: str = Form("[]"),
    model: str = Form(DEFAULT_MODEL),
    org: str = Form(""),
):
    """
    Reçoit un tour de parole (WAV), renvoie transcription + réponse + audio (base64).
    """
    audio_bytes = await audio.read()

    # 1) STT — Voxtral (GPU natif), repli Whisper via voice.transcribe
    from voice import transcribe, synthesize
    transcript = ""
    try:
        transcript = transcribe(audio_bytes)
    except Exception as e:
        logger.warning("STT échec : %s", e)

    if not transcript or len(transcript.strip()) < 2:
        return JSONResponse({"transcript": "", "answer": "", "audio": "", "sources": []})

    # 2) RAG conversationnel (garde la mémoire via l'historique côté client)
    try:
        hist = json.loads(history) if history else []
    except Exception:
        hist = []

    from rag_chain import chat as rag_chat
    org_filter = org.strip().upper() or None
    if org_filter not in (None, "ANSUT", "CGECI"):
        org_filter = None
    result = rag_chat(_get_vectordb(), transcript, history=hist, model=model, org=org_filter)
    answer = result.get("answer", "").strip()

    # 3) TTS — Piper (français, dans le conteneur)
    audio_b64 = ""
    try:
        wav = synthesize(answer)
        audio_b64 = base64.b64encode(wav).decode("ascii")
    except Exception as e:
        logger.warning("TTS échec : %s", e)

    return JSONResponse({
        "transcript": transcript,
        "answer": answer,
        "audio": audio_b64,
        "sources": [s.get("title") or s.get("url") for s in result.get("sources", [])][:3],
    })
