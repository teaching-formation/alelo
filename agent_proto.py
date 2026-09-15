"""
agent_proto.py — PROTOTYPE (isolé, ne touche pas l'app) d'un alélo AGENTIQUE par tool-calling.

Le modèle orchestre lui-même la recherche : il décide QUAND et QUOI chercher (au lieu de notre
routeur/décompo regex), lit les documents rendus par l'outil, et répond — ou répond directement
pour la culture générale. Objectif : valider la fiabilité avant toute intégration.

Lancement (dans le conteneur api) :
    docker compose exec -e AGENT_Q="qui dirige le CEPICI ?" api python agent_proto.py
"""
import os
import json
import requests

import rag_chain as rc

MODEL = os.getenv("AGENT_MODEL", "alelo-14b")
OLLAMA = os.getenv("OLLAMA_BASE_URL", rc.OLLAMA_BASE_URL)
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "4"))

_TOOLS = [{"type": "function", "function": {
    "name": "rechercher_documents",
    "description": ("Recherche dans les documents officiels des institutions et services publics "
                    "de Côte d'Ivoire. À APPELER pour toute question factuelle (démarche, responsable, "
                    "coût, délai, mission d'une institution, composition du gouvernement). Peut être "
                    "appelé plusieurs fois (une par sous-question)."),
    "parameters": {"type": "object", "properties": {
        "requete": {"type": "string", "description": "La requête de recherche, reformulée clairement."},
        "institution": {"type": "string", "description": (
            "Sigle si connu : GOUVERNEMENT (ministres/président/PM), ANSUT, CEPICI, DGI, TRESOR, "
            "SNEDAI, SERVICEPUBLIC, SANTE, AGRICULTURE, JUSTICE, INTERIEUR, EDUCATION, ENERGIE, "
            "COMMERCE, NUMERIQUE… Laisser vide si incertain.")}},
        "required": ["requete"]}}}]

_SYSTEM = (rc.SYSTEM_PROMPT + "\n\nTu disposes de l'outil `rechercher_documents`. Pour toute "
           "question sur une institution, un service public, une démarche ou un responsable "
           "ivoirien, APPELLE cet outil AVANT de répondre, puis réponds UNIQUEMENT à partir des "
           "documents qu'il renvoie. Pour une question à plusieurs volets, appelle l'outil une fois "
           "par volet. Pour de la culture générale ou un calcul, réponds directement sans l'outil.")


def _run_search(db, args) -> str:
    """Recherche enrichie comme le pipeline déterministe : fiches du graphe (faisant autorité) +
    composition du gouvernement si pertinent + chunks bruts rerankés."""
    req = (args.get("requete") or "").strip()
    inst = (args.get("institution") or "").strip().upper() or None
    if inst and inst not in rc.INSTITUTIONS:
        inst = None
    docs = rc.retrieve(db, req, k=6, org=inst or rc._route_org(req))
    # matching des fiches/composition sur requête + institution (le modèle passe parfois une
    # requête minimale « dirigeant » en s'appuyant sur le champ institution).
    fq = (req + " " + (inst or "")).strip()
    fiches = rc._match_service_fiches(fq)
    graph_ctx = rc._fiche_context(fiches) if fiches else ""
    gov_ctx = rc._gov_composition_context(fq)
    raw = rc._build_context(docs) if docs else ""
    full = (gov_ctx + graph_ctx + raw).strip()
    return full[:4800] or "(aucun document trouvé pour cette requête)"


def agent_answer(db, question: str, history: list | None = None):
    msgs = [{"role": "system", "content": _SYSTEM}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in (history or [])]
    msgs.append({"role": "user", "content": question})
    trace = []
    for step in range(MAX_STEPS):
        r = requests.post(f"{OLLAMA}/api/chat",
                          json={"model": MODEL, "stream": False, "messages": msgs,
                                "tools": _TOOLS, "keep_alive": rc.KEEP_ALIVE,
                                "options": {"temperature": 0.15, "num_ctx": 8192}},
                          timeout=180)
        r.raise_for_status()
        m = r.json().get("message", {}) or {}
        tcs = m.get("tool_calls") or []
        if not tcs:
            return (m.get("content") or "").strip(), trace
        msgs.append(m)                                   # message assistant portant les tool_calls
        for tc in tcs:
            fn = tc.get("function") or {}
            if fn.get("name") == "rechercher_documents":
                args = fn.get("arguments") or {}
                trace.append(f"🔎 rechercher({args.get('requete','')[:50]} | {args.get('institution','')})")
                result = _run_search(db, args)
                msgs.append({"role": "tool", "content": result})
    # garde-fou : forcer une réponse finale sans outil
    msgs.append({"role": "user", "content": "Réponds maintenant à partir de ce que tu as trouvé."})
    r = requests.post(f"{OLLAMA}/api/chat",
                      json={"model": MODEL, "stream": False, "messages": msgs,
                            "keep_alive": rc.KEEP_ALIVE, "options": {"temperature": 0.15}}, timeout=180)
    return (r.json().get("message", {}).get("content") or "").strip(), trace


if __name__ == "__main__":
    db = rc.load_vectordb()
    q = os.getenv("AGENT_Q", "qui dirige le CEPICI ?")
    ans, trace = agent_answer(db, q)
    print("QUESTION:", q)
    print("ÉTAPES  :", " | ".join(trace) if trace else "(aucune recherche)")
    print("RÉPONSE :", ans[:400])
