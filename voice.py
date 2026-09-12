"""
voice.py — Reconnaissance et synthèse vocale 100% locales (offline).

STT (voix → texte)  : faster-whisper (modèle Whisper optimisé CPU)
TTS (texte → voix)  : Piper (voix française neuronale fr_FR-siwis-medium)

Les modèles sont chargés une seule fois (paresseux) puis gardés en mémoire.
Aucune connexion internet n'est requise une fois les modèles téléchargés.
"""

import io
import os
import re
import wave
import difflib
import logging

import requests

logger = logging.getLogger(__name__)

# ── Normalisation post-STT (garde-fou : corrige les sigles/entités mal transcrits) ──
# Les STT (même Voxtral) massacrent les sigles ivoiriens : « CGECI » → « CQSI »/« c'est jessie »,
# « ANSUT » → « an sûr ». Une requête mal transcrite = mauvais retrieval. On corrige AVANT.
_STT_PATTERNS = [
    # CGECI
    (re.compile(r"\bc'?\s*est\s+jess?[iy]e?\b", re.I), "CGECI"),
    (re.compile(r"\bs[ée]?j[ée]ss?[iy]e?\b", re.I), "CGECI"),
    (re.compile(r"\bc\.?\s?g\.?\s?e\.?\s?c\.?\s?i\.?\b", re.I), "CGECI"),
    (re.compile(r"\b[cs][qg][ée]?[sc]?i\b", re.I), "CGECI"),
    (re.compile(r"\bc[ée]g[ée]ci\b", re.I), "CGECI"),
    (re.compile(r"\b[cs][ée]s[iy]\b", re.I), "CGECI"),          # cesi, cési, sesi
    (re.compile(r"\bj[ée]ss?[iy]e?\b", re.I), "CGECI"),         # jessie, jési
    # ANSUT
    (re.compile(r"\ban\s*s[uûü]r?t?\b", re.I), "ANSUT"),
    (re.compile(r"\bans?ou?te?\b", re.I), "ANSUT"),
    (re.compile(r"\ba\.?\s?n\.?\s?s\.?\s?u\.?\s?t\.?\b", re.I), "ANSUT"),
    (re.compile(r"\bax[cs]u?[it]e?\b", re.I), "ANSUT"),
]

# Vocabulaire du domaine pour un rattrapage flou (fautes proches d'un sigle).
_DOMAIN_VOCAB = ["ansut", "cgeci"]
_FR_STOPWORDS = {"aussi", "merci", "assez", "ainsi", "celui", "cette", "comme",
                 "quels", "quelles", "avec", "pour", "dans", "vous", "sont"}


def _correct_entities(text: str) -> str:
    """Corrige les sigles/entités du domaine mal transcrits (CGECI, ANSUT…).
    Uniquement via des motifs regex CIBLÉS et testés — pas de matching flou,
    qui corromprait des mots courants (« ceci » → « CGECI »)."""
    if not text:
        return text
    for pat, repl in _STT_PATTERNS:
        text = pat.sub(repl, text)
    return text

# ── Configuration ────────────────────────────────────────────────────────────
MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")
WHISPER_SIZE = os.getenv("WHISPER_MODEL", "small")          # tiny|base|small|medium
PIPER_ONNX = os.path.join(MODELS_DIR, "piper", "fr_FR-siwis-medium.onnx")

# Serveur Voxtral natif (hors Docker, GPU Apple Silicon) — comme Ollama pour le LLM
VOXTRAL_URL = os.getenv("VOXTRAL_URL", "http://host.docker.internal:8600")

# Caches modules (chargés une seule fois)
_whisper_model = None
_piper_voice = None


# ── STT : voix → texte ───────────────────────────────────────────────────────
# Priorité : serveur natif Whisper large-v3 (GPU, multilingue, auto-détection FR/EN).
# Repli    : faster-whisper local dans le conteneur (CPU) si le serveur est éteint.

def _transcribe_native(audio_bytes: bytes) -> tuple[str, str]:
    """Transcription via le serveur natif (Whisper large-v3, MLX/Metal).
    Retourne (texte, langue_détectée)."""
    resp = requests.post(
        f"{VOXTRAL_URL}/transcribe",
        files={"audio": ("audio.wav", audio_bytes, "audio/wav")},
        timeout=120,
    )
    resp.raise_for_status()
    j = resp.json()
    return (j.get("text") or "").strip(), (j.get("language") or "fr")


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        logger.info("Chargement du modèle Whisper '%s'...", WHISPER_SIZE)
        _whisper_model = WhisperModel(
            WHISPER_SIZE,
            device="cpu",
            compute_type="int8",
            download_root=os.path.join(MODELS_DIR, "whisper"),
        )
    return _whisper_model


def _transcribe_whisper_auto(audio_bytes: bytes) -> tuple[str, str]:
    """Transcription Whisper avec AUTO-DÉTECTION de langue (FR/EN/…).
    Retourne (langue_détectée, texte)."""
    model = _get_whisper()
    segments, info = model.transcribe(
        io.BytesIO(audio_bytes),
        language=None,              # None → Whisper détecte la langue lui-même
        beam_size=1,
        vad_filter=True,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    lang = getattr(info, "language", None) or "fr"
    return lang, text


def transcribe(audio_bytes: bytes) -> str:
    """
    Transcrit un enregistrement audio (WAV) en texte, MULTILINGUE.
    Serveur natif Whisper large-v3 (GPU, auto-détection FR/EN) en primaire ;
    faster-whisper local (CPU, auto-détection) en repli si le serveur est éteint.
    Corrige ensuite les sigles/entités du domaine — uniquement sur du français.
    """
    try:
        raw, lang = _transcribe_native(audio_bytes)         # Whisper large-v3 GPU
    except Exception as e:
        logger.warning("STT natif indisponible (%s) → faster-whisper local.", e)
        lang, raw = _transcribe_whisper_auto(audio_bytes)

    logger.info("Langue détectée : %s", lang)
    # Correction des sigles FR (motifs phonétiques français) uniquement sur du français.
    if lang.startswith("fr"):
        corrected = _correct_entities(raw)
        if corrected != raw:
            logger.info("Correction post-STT : %r → %r", raw, corrected)
        return corrected
    return raw


# ── TTS : texte → voix (Piper, multi-voix) ───────────────────────────────────
# Catalogue de voix françaises (l'utilisateur choisit à la 1re visite).
PIPER_DIR = os.path.join(MODELS_DIR, "piper")
VOICES = {
    "siwis": {"label": "Siwis", "genre": "femme", "file": "fr_FR-siwis-medium.onnx"},
    "tom":   {"label": "Tom",   "genre": "homme", "file": "fr_FR-tom-medium.onnx"},
    "upmc":  {"label": "UPMC",  "genre": "femme", "file": "fr_FR-upmc-medium.onnx"},
    "gilles":{"label": "Gilles","genre": "homme", "file": "fr_FR-gilles-low.onnx"},
}
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "siwis")
# Voix anglaise (non listée : utilisée automatiquement si la réponse est en anglais)
EN_VOICE_FILE = os.getenv("EN_VOICE_FILE", "en_US-amy-medium.onnx")
_piper_voices: dict = {}   # cache par voix

# ── Détection de langue pour router la synthèse (FR/EN) ──
_FR_SW = {"le", "la", "les", "des", "un", "une", "est", "que", "qui", "pour", "vous", "je",
          "de", "du", "dans", "avec", "sur", "et", "à", "ce", "vos", "votre", "au", "aux",
          "pas", "plus", "être", "faire", "cette", "sont", "peut"}
_EN_SW = {"the", "a", "an", "is", "are", "you", "your", "to", "of", "for", "and", "in",
          "with", "on", "can", "this", "that", "it", "we", "our", "have", "will", "please",
          "here", "there", "how", "what"}


def _speech_lang(text: str) -> str:
    """Heuristique FR/EN sur le texte à lire (mots-outils). 'en' seulement si clairement anglais."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 4:
        return "fr"
    fr = sum(w in _FR_SW for w in words)
    en = sum(w in _EN_SW for w in words)
    return "en" if en > fr and en >= 2 else "fr"


def _get_piper_en():
    """Charge (et met en cache) la voix anglaise, si le fichier est présent."""
    key = "__en__"
    if key not in _piper_voices:
        from piper import PiperVoice
        onnx = os.path.join(PIPER_DIR, EN_VOICE_FILE)
        logger.info("Chargement de la voix anglaise Piper '%s'...", EN_VOICE_FILE)
        _piper_voices[key] = PiperVoice.load(onnx, config_path=onnx + ".json")
    return _piper_voices[key]


def list_voices() -> list[dict]:
    """Voix disponibles (fichier présent sur disque)."""
    out = []
    for vid, v in VOICES.items():
        if os.path.exists(os.path.join(PIPER_DIR, v["file"])):
            out.append({"id": vid, "label": v["label"], "genre": v["genre"]})
    return out


def _get_piper(voice_id: str = DEFAULT_VOICE):
    if voice_id not in VOICES:
        voice_id = DEFAULT_VOICE
    if voice_id not in _piper_voices:
        from piper import PiperVoice
        onnx = os.path.join(PIPER_DIR, VOICES[voice_id]["file"])
        logger.info("Chargement de la voix Piper '%s'...", voice_id)
        _piper_voices[voice_id] = PiperVoice.load(onnx, config_path=onnx + ".json")
    return _piper_voices[voice_id]


def _clean_for_speech(text: str, max_chars: int = 1500) -> str:
    """Nettoie le markdown pour une lecture vocale naturelle."""
    text = re.sub(r"[*_`#>]", "", text)                     # markdown
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)     # liens [txt](url) → txt
    text = re.sub(r"https?://\S+", "", text)                # URLs nues
    text = re.sub(r"\n{2,}", ". ", text)                    # paragraphes → pause
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(".", 1)[0] + "."
    return text


def synthesize(text: str, voice_id: str = DEFAULT_VOICE) -> bytes:
    """Génère un WAV (bytes) de la lecture du texte.
    Route automatiquement vers la voix anglaise si la réponse est en anglais
    (une voix française lisant de l'anglais sonne faux), sinon la voix FR choisie."""
    clean = _clean_for_speech(text)
    voice = None
    if _speech_lang(clean) == "en" and os.path.exists(os.path.join(PIPER_DIR, EN_VOICE_FILE)):
        try:
            voice = _get_piper_en()
        except Exception as e:
            logger.warning("Voix anglaise indisponible (%s) → voix FR.", e)
    if voice is None:
        voice = _get_piper(voice_id)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize(clean, wf)
    return buf.getvalue()


# ── Préchargement (appelé au démarrage de l'app) ─────────────────────────────
def warmup() -> None:
    """Charge la voix par défaut en mémoire pour éviter la latence au 1er usage."""
    _get_piper(DEFAULT_VOICE)
