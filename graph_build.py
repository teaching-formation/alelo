"""
graph_build.py — Construit un GRAPHE DE CONNAISSANCES ciblé des services citoyens.

Pour chaque service clé (passeport, CNI, permis…), on récupère ses meilleurs chunks (RAG hybride)
et on extrait une FICHE structurée {institution, documents, coût, délai} avec le modèle 14B.
Le résultat (services + relations service↔institution↔document) est écrit dans
data/knowledge_graph.json, utilisé ensuite comme contexte faisant autorité (cf. graph.py).

Lancement (api arrêtée ou non, lecture seule ChromaDB) :
    docker compose run --rm --no-deps api python graph_build.py
"""

import os
import re
import json
import requests

import rag_chain as rc

MODEL = os.getenv("GRAPH_MODEL", "alelo-14b")
OUT = os.path.join("data", "knowledge_graph.json")
OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

# Services citoyens clés : (id, requête de récupération, org à privilégier)
SERVICES = [
    ("passeport",              "démarches documents coût délai pour obtenir un passeport", "SNEDAI"),
    ("cni",                    "démarches documents coût délai carte nationale d'identité CNI", "SERVICEPUBLIC"),
    ("permis_conduire",        "démarches documents pour obtenir un permis de conduire", "TRANSPORTS"),
    ("acte_naissance",         "extrait d'acte de naissance documents coût délai", "SERVICEPUBLIC"),
    ("certificat_nationalite", "certificat de nationalité documents coût délai", "SERVICEPUBLIC"),
    ("casier_judiciaire",      "casier judiciaire extrait documents coût délai", "SERVICEPUBLIC"),
    ("carte_sejour",           "carte de séjour étranger documents coût délai", "SERVICEPUBLIC"),
    ("acte_mariage",           "acte de mariage certificat documents coût délai", "SERVICEPUBLIC"),
    ("creation_entreprise",    "création d'entreprise formalités CEPICI guichet unique documents coût délai", "CEPICI"),
    ("impots",                 "paiement des impôts déclaration e-impots démarche", "DGI"),
    ("carte_grise",            "carte grise certificat d'immatriculation véhicule documents coût", "TRANSPORTS"),
    ("carte_resident",         "carte de résident étranger documents coût délai", "SERVICEPUBLIC"),
]

_EXTRACT_PROMPT = (
    "Extrais de ce TEXTE une fiche du service « {name} » en JSON STRICT avec EXACTEMENT ces clés :\n"
    '{{"service": str, "institution": str|null, "documents": [str], "cout": str|null, '
    '"delai": str|null, "ou": str|null}}\n'
    "Règles : n’invente RIEN ; mets null (ou []) si l’info est absente du texte ; garde les "
    "montants et délais tels qu’écrits. Réponds UNIQUEMENT le JSON.\n\nTEXTE:\n{ctx}\n\nJSON:"
)


def _extract(name: str, ctx: str) -> dict | None:
    prompt = _EXTRACT_PROMPT.format(name=name, ctx=ctx[:2200])
    try:
        r = requests.post(f"{OLLAMA}/api/generate",
                          json={"model": MODEL, "prompt": prompt, "stream": False,
                                "options": {"temperature": 0, "num_predict": 400, "num_ctx": 4096}},
                          timeout=180)
        r.raise_for_status()
        txt = r.json().get("response", "")
    except Exception as e:
        print("  ! extraction échouée :", e)
        return None
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    db = rc.load_vectordb()
    graph = {"services": {}, "institutions": {}, "generated_model": MODEL}
    for sid, query, pref_org in SERVICES:
        print(f"=== {sid} ===")
        org = pref_org if pref_org in rc.INSTITUTIONS else None
        docs = rc.retrieve(db, query, k=4, org=org)
        if not docs:
            print("  (aucun doc)")
            continue
        ctx = "\n\n".join(d.page_content[:900] for d in docs[:3])
        fiche = _extract(sid, ctx)
        if not fiche:
            print("  (extraction vide)")
            continue
        srcs = list({d.metadata.get("url", "") for d in docs[:3] if d.metadata.get("url")})
        orgs = list({d.metadata.get("org") for d in docs[:3] if d.metadata.get("org")})
        fiche["id"] = sid
        fiche["orgs"] = orgs
        fiche["sources"] = srcs[:3]
        graph["services"][sid] = fiche
        for o in orgs:
            graph["institutions"].setdefault(o, {"label": rc.org_label(o), "services": []})
            if sid not in graph["institutions"][o]["services"]:
                graph["institutions"][o]["services"].append(sid)
        print("  →", fiche.get("institution"), "| docs:", len(fiche.get("documents") or []),
              "| coût:", fiche.get("cout"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Graphe écrit : {OUT} — {len(graph['services'])} services, "
          f"{len(graph['institutions'])} institutions.")


if __name__ == "__main__":
    main()
