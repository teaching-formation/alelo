#!/usr/bin/env python3
"""
gen_golden.py — Génère la batterie de vérité DEPUIS les sources qui font autorité.

Idée : les faits ÉNUMÉRABLES (les membres du gouvernement, les services du graphe) ne
doivent plus être écrits à la main. On les DÉRIVE de la donnée source, si bien que la
batterie est exhaustive ET se met à jour toute seule quand la source change.

    golden.json  =  golden_core.json (curé, non-énumérable)  +  cas générés
                    ├─ ministres      ← data/raw_gouvernement/composition.json
                    └─ services_inst  ← data/knowledge_graph.json (institution par service)

Le curé (golden_core.json) garde ce qui n'est PAS dérivable : comptes/rangs, culture
générale, honnêteté, langue, anti-esquive, cas experts.

Usage :
    python3 eval/gen_golden.py            # (ré)écrit eval/golden.json
    python3 eval/gen_golden.py --dry-run  # affiche ce qui serait généré, sans écrire
"""
import argparse
import json
import os
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORE = os.path.join(HERE, "golden_core.json")
OUT = os.path.join(HERE, "golden.json")
COMPOSITION = os.path.join(ROOT, "data", "raw_gouvernement", "composition.json")
GRAPH = os.path.join(ROOT, "data", "knowledge_graph.json")

_STOP = {"ministre", "ministere", "de", "du", "des", "la", "le", "les", "l", "d", "et",
         "a", "au", "aux", "delegue", "deleguee", "charge", "chargee", "etat", "en"}


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    for ch in ("'", "’", "-", ",", "(", ")", "."):
        s = s.replace(ch, " ")
    return " ".join(s.split())


def _slug(role: str, taken: set) -> str:
    toks = [t for t in _norm(role).split() if t not in _STOP]
    base = "-".join(toks[:2]) or "role"
    sid = "gov-" + base
    i = 2
    while sid in taken:
        sid = f"gov-{base}-{i}"
        i += 1
    taken.add(sid)
    return sid


def _clean_role(role: str) -> str:
    """Enlève les parenthèses explicatives ; garde le poste complet (désambiguïse les ministres d'État)."""
    role = re.sub(r"\([^)]*\)", "", role)
    return " ".join(role.split()).strip(" ,")


def gen_gouvernement() -> list:
    """Un cas 'qui est le <poste> ?' par ligne « Poste : Nom » de la section MEMBRES."""
    with open(COMPOSITION, encoding="utf-8") as f:
        content = json.load(f).get("content", "")

    # On ne garde que la section listant les membres (évite le résumé et les entêtes).
    m = re.search(r"MEMBRES DU GOUVERNEMENT\s*:(.*?)(?:\nSource\s*:|\Z)", content, re.S | re.I)
    section = m.group(1) if m else content

    cases, taken = [], set()
    for raw_line in section.split("\n"):
        line = raw_line.strip()
        if " : " not in line:
            continue
        role, name = line.split(" : ", 1)
        role, name = _clean_role(role), name.strip().rstrip(".").strip()
        # garde-fous : c'est bien un poste nominatif, et un nom plausible (2 mots mini)
        if not re.search(r"ministre|president|secretaire|gouverneur|prefet|maire|directeur", role, re.I):
            continue
        if len(name.split()) < 2 or len(name) > 60:
            continue
        cases.append({
            "id": _slug(role, taken),
            "category": "ministres",
            "mode": "auto",
            "q": f"Qui est le {role[0].lower() + role[1:]} ?",
            # Nom complet BRUT : le lanceur normalise les deux côtés (accents/apostrophes → espaces).
            "expect_any": [name],
            "_src": "composition.json",
        })
    return cases


def gen_services() -> list:
    """Un cas 'quelle institution pour <service> ?' par fiche du graphe ayant une institution."""
    if not os.path.exists(GRAPH):
        return []
    with open(GRAPH, encoding="utf-8") as f:
        g = json.load(f)
    cases, taken = [], set()
    for sid, fiche in (g.get("services") or {}).items():
        inst = (fiche.get("institution") or "").strip()
        label = (fiche.get("service") or sid).strip()
        if not inst or len(inst) > 70:
            continue
        cid = "svc-" + re.sub(r"[^a-z0-9]+", "-", _norm(sid)).strip("-")
        if cid in taken:
            continue
        taken.add(cid)
        cases.append({
            "id": cid,
            "category": "services_inst",
            "mode": "auto",
            "q": f"Quelle institution s'occupe de : {label} ?",
            "expect_any": [inst],
            "_src": "knowledge_graph.json",
        })
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    # Les cas services sont OPT-IN : le graphe est extrait par LLM (pas une vérité sûre) —
    # on n'y assoit une assertion « vérité » qu'après avoir vérifié ses institutions.
    ap.add_argument("--services", action="store_true",
                    help="générer aussi les cas services depuis le graphe (à vérifier avant)")
    args = ap.parse_args()

    with open(CORE, encoding="utf-8") as f:
        core = json.load(f)

    generated = gen_gouvernement()
    if args.services:
        generated += gen_services()

    # Fusion : le curé d'abord, puis le généré (les ids en double sont ignorés au profit du curé).
    seen = {c["id"] for c in core}
    merged = list(core)
    dropped = 0
    for c in generated:
        if c["id"] in seen:
            dropped += 1
            continue
        seen.add(c["id"])
        merged.append(c)

    by_cat = {}
    for c in merged:
        by_cat[c.get("category", "?")] = by_cat.get(c.get("category", "?"), 0) + 1

    print(f"curé : {len(core)} · généré : {len(generated)}"
          f" (dont {dropped} ignorés car id déjà curé) · total : {len(merged)}")
    for cat in sorted(by_cat):
        print(f"  {cat:<16} {by_cat[cat]}")

    if args.dry_run:
        print("\n--- aperçu des cas générés ---")
        for c in generated:
            exp = " / ".join(c.get("expect_any", []))
            print(f"  [{c['category']:<13}] {c['id']:<26} {c['q']}  →  {exp}")
        print("\n(dry-run : rien écrit)")
        return 0

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)
    print(f"\n✅ écrit : {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
