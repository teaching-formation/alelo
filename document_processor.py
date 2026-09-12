"""
document_processor.py — Extrait le texte de fichiers uploadés par l'utilisateur.
Supporte : PDF, DOCX, TXT, images (via Ollama llava).
"""

import base64
import logging
from io import BytesIO

import requests
import pypdf
from docx import Document

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = __import__("os").getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def extract_pdf(file_bytes: bytes) -> str:
    reader = pypdf.PdfReader(BytesIO(file_bytes))
    pages = [p.extract_text() for p in reader.pages if p.extract_text()]
    return "\n\n".join(pages)


def extract_docx(file_bytes: bytes) -> str:
    doc = Document(BytesIO(file_bytes))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")


def describe_image(file_bytes: bytes, filename: str) -> str:
    """Envoie l'image à llava via Ollama et retourne la description."""
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": "llava",
                "prompt": (
                    "Décris précisément le contenu de cette image en français. "
                    "Si c'est un document, extrait tout le texte visible. "
                    "Si c'est un graphique ou tableau, décris les données."
                ),
                "images": [b64],
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        logger.warning(f"Erreur llava: {e}")
        return f"[Image: {filename} — modèle llava non disponible. Lance: ollama pull llava]"


def process_upload(file_bytes: bytes, filename: str) -> dict:
    """
    Détecte le type de fichier et extrait son contenu.
    Retourne {"text": str, "type": str, "filename": str}
    """
    name = filename.lower()

    if name.endswith(".pdf"):
        text = extract_pdf(file_bytes)
        ftype = "pdf"
    elif name.endswith(".docx"):
        text = extract_docx(file_bytes)
        ftype = "docx"
    elif name.endswith(".txt") or name.endswith(".md"):
        text = extract_txt(file_bytes)
        ftype = "text"
    elif name.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
        text = describe_image(file_bytes, filename)
        ftype = "image"
    else:
        text = file_bytes.decode("utf-8", errors="replace")
        ftype = "unknown"

    return {
        "text": text,
        "type": ftype,
        "filename": filename,
        "word_count": len(text.split()),
    }
