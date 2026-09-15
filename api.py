"""
api.py — API HTTP du RAG CGECI/ANSUT pour le front Next.js.

Expose :
  GET  /api/config      → modèles, périmètres, voix disponibles
  POST /api/chat        → réponse en streaming (SSE) + sources
  POST /api/transcribe  → audio (WAV) → texte (Voxtral + correction)
  POST /api/tts         → texte + voix → audio WAV

Réutilise rag_chain (RAG + garde-fous) et voice (STT/TTS multi-voix).
Tourne dans Docker (port 8080), appelle Ollama + Voxtral+Reranker natifs.
"""

import os
import json
import time
import logging
from contextlib import asynccontextmanager

import requests

import tracing

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

MODELS = [
    {"id": "alelo",    "label": "alélo",     "flag": "🇨🇮", "recommended": True},
    {"id": "qwen2.5",  "label": "Qwen 2.5",  "flag": "🇨🇳"},
    {"id": "mistral",  "label": "Mistral",   "flag": "🇫🇷"},
    {"id": "llama3.1", "label": "Llama 3.1", "flag": "🦙"},
    {"id": "llama3",   "label": "Llama 3",   "flag": "🦙"},
]
_THEME_EMOJI = {
    "Démarches": "📋", "Fiscalité": "🧾", "Finances": "💰", "Transports": "🚗",
    "Entreprise": "🏢", "Commerce": "📦", "Foncier": "🏘️", "Données": "🔐",
    "Social": "👨‍👩‍👧", "Textes": "⚖️", "Gouvernement": "🏛️", "Numérique": "📡",
    "Développement": "📈", "Environnement": "🌿", "Culture": "🎭", "Médias": "📰",
    "Défense": "🛡️", "Agriculture": "🐄", "Contacts": "📇",
}

def _installed_models() -> set:
    """Noms des modèles réellement présents dans Ollama (pour n'exposer que ceux-là)."""
    base = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    try:
        r = requests.get(f"{base}/api/tags", timeout=4)
        r.raise_for_status()
        names = set()
        for m in r.json().get("models", []):
            n = m.get("name", "")
            if n:
                names.add(n)
                names.add(n.split(":")[0])       # 'qwen2.5:7b' → aussi 'qwen2.5'
        return names
    except Exception as e:
        logger.warning("Liste des modèles Ollama indisponible : %s", e)
        return set()


def available_models() -> list:
    """MODELS filtré aux modèles installés (alelo toujours gardé). Fallback : liste complète."""
    inst = _installed_models()
    if not inst:
        return MODELS
    keep = [m for m in MODELS if m["id"] in inst or m["id"] == "alelo"]
    return keep or MODELS


# Cerveau du Mode Expert : un modèle plus gros (14B) si disponible, sinon repli sur alelo (7B).
_expert = {"checked": False, "model": "alelo"}


def expert_model() -> str:
    if not _expert["checked"]:
        want = os.getenv("EXPERT_MODEL", "alelo-14b")
        inst = _installed_models()
        _expert["model"] = want if (want in inst or f"{want}:latest" in inst) else "alelo"
        _expert["checked"] = True
        logger.info("Modèle Expert : %s", _expert["model"])
    return _expert["model"]


def build_orgs():
    """Construit la liste des périmètres (Tout + institutions) depuis la carte."""
    from rag_chain import INSTITUTIONS
    orgs = [{"id": "", "label": "Tout", "emoji": "🌍", "theme": ""}]
    for org, info in sorted(INSTITUTIONS.items(), key=lambda x: (x[1]["theme"], x[1]["label"])):
        orgs.append({"id": org, "label": info["label"],
                     "emoji": _THEME_EMOJI.get(info["theme"], "🏛️"), "theme": info["theme"]})
    return orgs

_state = {"db": None}


def get_db():
    if _state["db"] is None:
        from rag_chain import load_vectordb, _build_bm25_index
        logger.info("Chargement ChromaDB + BM25...")
        db = load_vectordb()
        _build_bm25_index(db)
        _state["db"] = db
        logger.info("Base prête.")
    return _state["db"]


def _warmup_models():
    """Précharge LLM + embedding + reranker : le 1er vrai message ne paie plus le cold-start."""
    import rag_chain as rc
    try:
        rc.generate("Bonjour", "Document de préchauffage.", rc.LLM_MODEL)
    except Exception as e:
        logger.warning("Warmup LLM : %s", e)
    try:
        db = _state.get("db")
        if db is not None:
            rc.retrieve(db, "préchauffage service public", k=2)   # embedding + BM25 + reranker
    except Exception as e:
        logger.warning("Warmup retrieval : %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_db()
        from voice import warmup
        warmup()
        _warmup_models()
    except Exception as e:
        logger.warning("Préchauffage partiel : %s", e)
    yield


app = FastAPI(title="alélo API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    from voice import list_voices, DEFAULT_VOICE
    return {
        "models": available_models(),
        "orgs": build_orgs(),
        "voices": list_voices(),
        "defaultVoice": DEFAULT_VOICE,
        "defaultModel": "alelo",
    }


class ChatReq(BaseModel):
    message: str
    history: list = []
    mode: str = "auto"          # "auto" | "expert"
    org: str | None = None      # "", "ANSUT", "CGECI"
    model: str = "alelo"


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


def _gen_document(db, req, doc_req):
    """Génère un document (PDF/Word/Excel) : contenu RAG → fichier → carte de téléchargement."""
    import documents
    from rag_chain import chat as rag_chat, doc_title, _detect_lang

    formats = doc_req.get("formats") or [doc_req.get("format", "pdf")]
    # 1) Contenu (généré UNE fois) : soit la dernière réponse (« mets ça en PDF »), soit une réponse fraîche.
    # `topic` = titre fourni par le modèle (tool-call) ; à défaut, la question précédente.
    body, topic = "", (doc_req.get("topic") or "").strip()
    if doc_req.get("refers_prior"):
        body = next((m["content"] for m in reversed(req.history or [])
                     if m.get("role") == "assistant" and m.get("content", "").strip()), "")
        if not topic:                      # pas de titre du modèle → sujet = question précédente
            topic = next((m["content"] for m in reversed(req.history or [])
                          if m.get("role") == "user" and m.get("content", "").strip()), "")
    if not body:
        res = rag_chat(db, topic or req.message, history=req.history, detail=True)
        body = res.get("answer", "")
        if not topic:
            topic = req.message
    title = doc_title(topic)

    # 2) Un fichier PAR format demandé
    docs_out = []
    for fmt in formats:
        info = documents.generate(fmt, title, body)
        docs_out.append({"id": info["id"], "filename": info["filename"], "format": fmt,
                         "label": info["label"], "url": f"/api/download/{info['id']}"})

    # 3) Message de confirmation (langue de l'utilisateur) + streaming pour l'animation/voix
    en = _detect_lang(req.message) == "en"
    labels = (" and " if en else " et ").join(d["label"] for d in docs_out)
    if en:
        confirm = f"Done ✅ I've prepared your {labels} document(s) “{title}”. You can download them just below."
    else:
        confirm = f"C'est prêt ✅ J'ai préparé ton document {labels} « {title} ». Tu peux le télécharger juste en dessous."
    for word in confirm.split(" "):
        yield _sse({"type": "token", "text": word + " "})

    yield _sse({"type": "done", "answer": confirm, "sources": [],
                "document": docs_out[0], "documents": docs_out})


@app.post("/api/chat")
def chat(req: ChatReq):
    from rag_chain import chat_stream, agent_stream, detect_doc_request
    db = get_db()

    # Le mode change la PROFONDEUR *et* le cerveau :
    #   auto   → alelo (7B) : pipeline déterministe rapide, périmètre routé automatiquement ;
    #   expert → alelo-14b (si dispo) : AGENTIQUE par tool-calling — le modèle orchestre lui-même
    #            la recherche (agent_stream), avec repli sur le pipeline vérifié en cas d'échec.
    detail = (req.mode == "expert")
    model = expert_model() if detail else "alelo"
    org = (req.org or None) if req.mode == "expert" else None

    # Demande de document : regex = pré-filtre rapide ; puis le MODÈLE décide (tool-calling)
    # et extrait formats/titre/source. Repli sur la regex si le modèle échoue mais qu'un format
    # est cité explicitement. (Fini l'extraction fragile « ces informations / mets ça… ».)
    from rag_chain import llm_doc_request
    doc_req = None
    _hint = detect_doc_request(req.message)
    if _hint:
        _args = llm_doc_request(req.message, req.history, model="alelo")
        if _args:
            doc_req = {"formats": _args["formats"], "topic": _args["titre"],
                       "refers_prior": _args["source"] == "reponse_precedente"}
        elif _hint.get("explicit"):
            doc_req = _hint

    # ── Observabilité locale (tracing.py) : une trace par requête ────────────
    trace_id = tracing.new_id()
    t0 = time.monotonic()
    steps: list = []

    def _finish(answer: str, sources: list, kind: str = "chat", err: str | None = None):
        tracing.record(id=trace_id, kind=kind, mode=req.mode, model=model,
                       question=(req.message or "")[:1000], answer=answer or "",
                       sources=[s.get("url") for s in (sources or []) if isinstance(s, dict)][:5],
                       steps=steps, latency_ms=round((time.monotonic() - t0) * 1000),
                       error=err)

    def gen():
        answer, sources = "", []
        try:
            if doc_req:                       # « crée-moi un PDF/Word/Excel … »
                for ev in _gen_document(db, req, doc_req):
                    yield ev
                _finish("(document généré)", [], kind="document")
                return
            # Expert → agent tool-calling (le modèle orchestre) ; Auto → pipeline déterministe.
            _stream = (agent_stream(db, req.message, history=req.history, model=model, org=org)
                       if detail else
                       chat_stream(db, req.message, history=req.history,
                                   model=model, org=org, detail=detail))
            for chunk in _stream:
                if isinstance(chunk, str):
                    yield _sse({"type": "token", "text": chunk})
                elif "step" in chunk:                 # étape agentique (Mode Expert)
                    steps.append(chunk["step"])
                    yield _sse({"type": "step", "text": chunk["step"]})
                else:
                    answer, sources = chunk.get("answer", ""), chunk.get("sources", [])
                    yield _sse({"type": "done", "answer": answer,
                                "sources": sources, "traceId": trace_id})
            _finish(answer, sources)
        except Exception as e:
            _finish(answer, sources, err=str(e))
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/download/{file_id}")
def download(file_id: str):
    import documents
    r = documents.get_file(file_id)
    if not r:
        return JSONResponse({"error": "document introuvable ou expiré"}, status_code=404)
    path, mime, filename = r
    return FileResponse(path, media_type=mime, filename=filename)


FEEDBACK_FILE = os.path.join("data", "feedback.jsonl")   # data/ est monté (persistant)


class FeedbackReq(BaseModel):
    rating: str                 # "up" | "down"
    question: str = ""
    answer: str = ""
    comment: str = ""
    sources: list = []
    model: str = "alelo"


@app.post("/api/feedback")
def feedback(req: FeedbackReq):
    """Journalise un pouce 👍/👎 (question, réponse, sources) → base d'amélioration d'alélo."""
    if req.rating not in ("up", "down"):
        return JSONResponse({"ok": False, "error": "rating invalide"}, status_code=400)
    from datetime import datetime, timezone
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rating": req.rating,
        "question": (req.question or "")[:1000],
        "answer": (req.answer or "")[:4000],
        "comment": (req.comment or "")[:1000],
        "sources": [s.get("url") for s in (req.sources or []) if isinstance(s, dict)][:5],
        "model": req.model,
    }
    try:
        os.makedirs(os.path.dirname(FEEDBACK_FILE) or ".", exist_ok=True)
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("Feedback non enregistré : %s", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=200)
    return {"ok": True}


@app.get("/api/feedback/stats")
def feedback_stats():
    """Bilan des pouces + derniers 👎 (avec commentaire) pour prioriser les corrections."""
    up = down = 0
    recent_down = []
    try:
        with open(FEEDBACK_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("rating") == "up":
                    up += 1
                elif r.get("rating") == "down":
                    down += 1
                    recent_down.append({"question": r.get("question", ""),
                                        "comment": r.get("comment", ""),
                                        "ts": r.get("ts", "")})
    except FileNotFoundError:
        pass
    total = up + down
    return {"up": up, "down": down, "total": total,
            "satisfaction": round(up / total, 2) if total else None,
            "recent_down": recent_down[-25:]}


@app.get("/api/traces")
def traces(limit: int = 100):
    """Traces récentes (question, mode, étapes, latence) + 👍/👎 joints → observabilité locale."""
    return tracing.load_traces(limit=min(max(limit, 1), 500))


@app.get("/api/traces/view")
def traces_view():
    """Petit tableau de bord HTML autonome (100 % local) au-dessus de /api/traces."""
    return Response(content=_TRACES_HTML, media_type="text/html; charset=utf-8")


_TRACES_HTML = """<!doctype html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>alélo · traces</title>
<style>
:root{--bg:#0b0f14;--card:#141b24;--fg:#e7edf3;--dim:#8aa0b3;--line:#233240;--up:#2ecc71;--down:#ff5c5c;--exp:#c39bff;--auto:#5cc8ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
header{padding:18px 22px;border-bottom:1px solid var(--line);display:flex;gap:22px;align-items:baseline;flex-wrap:wrap}
h1{font-size:17px;margin:0;font-weight:650}.stat{color:var(--dim)}.stat b{color:var(--fg)}
main{padding:14px 22px;max-width:1100px}
.row{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:10px 0}
.top{display:flex;gap:10px;align-items:center;flex-wrap:wrap;color:var(--dim);font-size:12px}
.badge{padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600}
.b-auto{background:rgba(92,200,255,.15);color:var(--auto)}.b-exp{background:rgba(195,155,255,.15);color:var(--exp)}
.q{font-weight:600;margin:6px 0 4px}.a{color:#cdd8e2;white-space:pre-wrap}
.up{color:var(--up)}.down{color:var(--down)}
details{margin-top:6px}summary{cursor:pointer;color:var(--dim);font-size:12px}
.step{color:var(--dim);font-size:12px;padding:1px 0}
.refresh{margin-left:auto;color:var(--dim);cursor:pointer;border:1px solid var(--line);padding:4px 10px;border-radius:8px;background:none;font:inherit}
@media(prefers-color-scheme:light){:root{--bg:#f6f8fb;--card:#fff;--fg:#0b1620;--dim:#5a6b7b;--line:#e2e8ee}}
</style></head><body>
<header><h1>alélo · traces <span class=stat>(observabilité locale)</span></h1>
<span class=stat>total <b id=t>–</b></span><span class=stat>latence moy <b id=l>–</b></span>
<span class=stat>p95 <b id=p>–</b></span><span class=stat>expert <b id=e>–</b></span>
<span class=stat>👍 <b id=u>–</b> · 👎 <b id=d>–</b></span>
<button class=refresh onclick=load()>↻ rafraîchir</button></header>
<main id=list></main>
<script>
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function load(){
 const r=await fetch('/api/traces?limit=200');const j=await r.json();const s=j.stats||{};
 t.textContent=s.total??'–';l.textContent=s.avg_latency_ms?s.avg_latency_ms+' ms':'–';
 p.textContent=s.p95_latency_ms?s.p95_latency_ms+' ms':'–';e.textContent=s.expert??'–';
 u.textContent=s.up??0;d.textContent=s.down??0;
 list.innerHTML=(j.traces||[]).map(row=>{
  const mode=row.mode==='expert'?'<span class="badge b-exp">expert</span>':'<span class="badge b-auto">auto</span>';
  const rate=row.rating==='up'?'<span class=up>👍</span>':row.rating==='down'?'<span class=down>👎</span>':'';
  const open=OPEN.has(row.id)?' open':'';
  const steps=(row.steps||[]).length?`<details data-id="${esc(row.id||'')}"${open}><summary>🔎 ${row.steps.length} étape(s)</summary>${row.steps.map(x=>`<div class=step>${esc(x)}</div>`).join('')}</details>`:'';
  const lat=row.latency_ms!=null?row.latency_ms+' ms':'';
  return `<div class=row><div class=top>${mode}<span>${lat}</span><span>${esc(row.ts||'')}</span>${row.error?'<span class=down>erreur</span>':''}${rate}</div>
   <div class=q>${esc(row.question||'')}</div><div class=a>${esc((row.answer||'').slice(0,700))}</div>${steps}</div>`;
 }).join('')||'<p class=stat>Aucune trace pour l’instant — pose une question dans l’app.</p>';
 list.querySelectorAll('details[data-id]').forEach(d=>d.ontoggle=()=>{d.open?OPEN.add(d.dataset.id):OPEN.delete(d.dataset.id)});
}
const OPEN=new Set();
load();setInterval(load,5000);
</script></body></html>"""


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    from voice import transcribe as stt
    data = await audio.read()
    try:
        text = stt(data)
    except Exception as e:
        return JSONResponse({"text": "", "error": str(e)}, status_code=200)
    return {"text": text}


class TTSReq(BaseModel):
    text: str
    voice: str = "siwis"


@app.post("/api/tts")
def tts(req: TTSReq):
    from voice import synthesize
    if not req.text.strip():
        return Response(status_code=204)
    try:
        wav = synthesize(req.text, voice_id=req.voice)
    except Exception as e:
        logger.warning("TTS échec : %s", e)
        return JSONResponse({"error": "synthèse vocale indisponible"}, status_code=503)
    return Response(content=wav, media_type="audio/wav")
