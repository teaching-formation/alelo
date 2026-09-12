"""Comparaison rigoureuse qwen2.5 (cgeci) vs mistral (cgeci-mistral).
Contexte récupéré UNE fois par question (identique aux 2 modèles).
Options de génération IDENTIQUES. Seul le FROM du modèle change."""
import time, json, requests, os
from rag_chain import (load_vectordb, _build_bm25_index, retrieve, _build_context,
                       _build_chat_prompt, _route_org, _faithfulness_check)

BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPTS = {"temperature": 0.15, "top_p": 0.7, "repeat_penalty": 1.05,
        "num_ctx": 8192, "num_predict": 500}
MODELS = ["cgeci", "cgeci-mistral"]

QUESTIONS = [
    ("synthese",   "Quelles sont les conditions ET le coût pour adhérer à la CGECI ?", "CGECI"),
    ("synthese",   "Explique le rôle de l'ANSUT et ce qu'est le service universel des télécommunications.", "ANSUT"),
    ("ambiguite",  "Combien coûte l'adhésion ?", None),
    ("refus_part", "Quel est le coût d'adhésion à la CGECI et le nombre exact de ses membres aujourd'hui ?", "CGECI"),
    ("redaction",  "Présente la CGECI en quelques phrases à un chef d'entreprise.", "CGECI"),
    ("redaction",  "Explique simplement à un citoyen le rôle de l'ANSUT.", "ANSUT"),
    ("factuel",    "Qui préside la CGECI ?", "CGECI"),
    ("refus",      "Quel est le salaire du directeur général de l'ANSUT ?", "ANSUT"),
]

def gen(model, prompt):
    t = time.time()
    r = requests.post(f"{BASE}/api/generate", json={
        "model": model, "prompt": prompt, "stream": False,
        "keep_alive": "30m", "options": OPTS}, timeout=180)
    r.raise_for_status()
    return r.json().get("response", "").strip(), time.time() - t

db = load_vectordb(); _build_bm25_index(db)

# 1) Contexte identique par question
prepared = []
for typ, q, org in QUESTIONS:
    o = org or _route_org(q)
    docs = retrieve(db, q, org=o)
    ctx = _build_context(docs)
    prompt = _build_chat_prompt(q, ctx, [], "x")
    prepared.append({"type": typ, "q": q, "org": o, "ctx": ctx, "prompt": prompt})

# 2) Génération par modèle (chaque modèle chauffé puis passé sur toutes les questions)
results = {m: [] for m in MODELS}
for m in MODELS:
    gen(m, "Bonjour")  # warmup
    for p in prepared:
        ans, dt = gen(m, p["prompt"])
        faith = _faithfulness_check(ans, p["ctx"])
        results[m].append({"ans": ans, "dt": dt,
                           "grounded": faith["grounded"],
                           "bad_nums": faith["unsupported_numbers"],
                           "words": len(ans.split())})

# 3) Tableau récap
print("="*90)
print(f"{'TYPE':11s} | {'qwen2.5 (s)':>11s} {'mots':>5s} {'faith':>6s} | {'mistral (s)':>11s} {'mots':>5s} {'faith':>6s}")
print("-"*90)
tot = {m: 0.0 for m in MODELS}
for i, p in enumerate(prepared):
    a = results["cgeci"][i]; b = results["cgeci-mistral"][i]
    tot["cgeci"] += a["dt"]; tot["cgeci-mistral"] += b["dt"]
    fa = "OK" if a["grounded"] else "!"+",".join(a["bad_nums"][:2])
    fb = "OK" if b["grounded"] else "!"+",".join(b["bad_nums"][:2])
    print(f"{p['type']:11s} | {a['dt']:11.1f} {a['words']:5d} {fa:>6s} | {b['dt']:11.1f} {b['words']:5d} {fb:>6s}")
print("-"*90)
n = len(prepared)
print(f"{'MOYENNE':11s} | {tot['cgeci']/n:11.1f} {'':5s} {'':6s} | {tot['cgeci-mistral']/n:11.1f}")
print("="*90)

# 4) Réponses complètes pour l'axe FRANÇAIS (rédaction + synthèse)
for i, p in enumerate(prepared):
    if p["type"] in ("redaction", "synthese", "refus_part"):
        print(f"\n### [{p['type']}] {p['q']}")
        print(f"--- qwen2.5 ({results['cgeci'][i]['dt']:.1f}s) ---\n{results['cgeci'][i]['ans']}")
        print(f"--- mistral ({results['cgeci-mistral'][i]['dt']:.1f}s) ---\n{results['cgeci-mistral'][i]['ans']}")
