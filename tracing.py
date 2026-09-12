"""
tracing.py — Observabilité locale légère (le « monde ouvert »).

Chaque requête chat est journalisée dans data/traces.jsonl : question, mode, modèle,
langue, étapes agentiques, sources, latence. On y joint ensuite les 👍/👎 (feedback.jsonl)
par correspondance de réponse. 100 % local, 0 conteneur, désactivable par TRACING=0.

Point d'extension : le jour où alélo tourne sur un vrai serveur, on branche ici un
exporteur Langfuse (même point d'instrumentation) sans toucher au reste.

Le traçage ne doit JAMAIS casser une requête → tout est encapsulé, aucune exception ne remonte.
"""
import os
import json
import uuid

TRACING_ENABLED = os.getenv("TRACING", "1") == "1"
TRACES_FILE = os.path.join("data", "traces.jsonl")
FEEDBACK_FILE = os.path.join("data", "feedback.jsonl")

MAX_ANSWER = 4000
MAX_STEPS = 40


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def record(**fields) -> None:
    """Ajoute une ligne de trace. Ne lève jamais (le traçage ne casse pas la requête)."""
    if not TRACING_ENABLED:
        return
    try:
        from datetime import datetime, timezone
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        rec.update(fields)
        if "answer" in rec and isinstance(rec["answer"], str):
            rec["answer"] = rec["answer"][:MAX_ANSWER]
        if "steps" in rec and isinstance(rec["steps"], list):
            rec["steps"] = rec["steps"][:MAX_STEPS]
        os.makedirs(os.path.dirname(TRACES_FILE) or ".", exist_ok=True)
        with open(TRACES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _norm_ans(s: str) -> str:
    return " ".join((s or "").lower().split())[:160]


def _feedback_map() -> dict:
    """{réponse normalisée → {rating, comment}} pour joindre les 👍/👎 aux traces."""
    fb = {}
    try:
        with open(FEEDBACK_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                key = _norm_ans(r.get("answer", ""))
                if key:
                    fb[key] = {"rating": r.get("rating"), "comment": r.get("comment", "")}
    except FileNotFoundError:
        pass
    return fb


def load_traces(limit: int = 100) -> dict:
    """Traces récentes (plus récentes d'abord), enrichies du 👍/👎 correspondant + stats."""
    rows = []
    try:
        with open(TRACES_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        return {"traces": [], "stats": {"total": 0}}

    fb = _feedback_map()
    for r in rows:
        f = fb.get(_norm_ans(r.get("answer", "")))
        if f:
            r["rating"] = f["rating"]
            r["comment"] = f.get("comment", "")

    total = len(rows)
    lat = [r.get("latency_ms") for r in rows if isinstance(r.get("latency_ms"), (int, float))]
    down = sum(1 for r in rows if r.get("rating") == "down")
    up = sum(1 for r in rows if r.get("rating") == "up")
    stats = {
        "total": total,
        "avg_latency_ms": round(sum(lat) / len(lat)) if lat else None,
        "p95_latency_ms": (sorted(lat)[int(len(lat) * 0.95)] if len(lat) >= 20 else (max(lat) if lat else None)),
        "expert": sum(1 for r in rows if r.get("mode") == "expert"),
        "up": up, "down": down,
    }
    rows.reverse()   # plus récentes d'abord
    return {"traces": rows[:limit], "stats": stats}
