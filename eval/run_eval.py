#!/usr/bin/env python3
"""
run_eval.py — Banc de vérité d'alélo (« golden eval »).

Rejoue une batterie de questions-vérité contre l'API (/api/chat, SSE) et vérifie
chaque réponse par assertions simples. Objectif : attraper une régression AVANT
l'utilisateur. Une erreur déjà connue ne doit jamais repasser.

Usage :
    python3 eval/run_eval.py                 # toute la batterie
    python3 eval/run_eval.py -k min          # seulement les ids/catégories contenant "min"
    python3 eval/run_eval.py --category ministres
    python3 eval/run_eval.py --mode auto     # ignore les cas expert (lents)
    python3 eval/run_eval.py --url http://localhost:8088
    python3 eval/run_eval.py -v               # montre la réponse même quand ça passe

Schéma d'un cas (eval/golden.json) :
    id, category, mode ("auto"|"expert"), q,
    expect_all : liste d'exigences — chaîne (doit apparaître) ou liste (au moins une),
    expect_any : au moins une des chaînes doit apparaître,
    forbid     : aucune de ces chaînes ne doit apparaître (en plus de l'anti-esquive global),
    expect_lang: "fr"|"en" — langue attendue de la réponse.
Comparaison insensible à la casse et aux accents.

Code de sortie : 0 si tout passe, 1 si au moins un échec (utilisable en garde-fou/CI).
"""
import argparse
import json
import os
import sys
import time
import unicodedata
import urllib.request
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden.json")
REPORT_MD = os.path.join(HERE, "report.md")
REPORT_JSON = os.path.join(HERE, "last_run.json")

# Formulations d'esquive INTERDITES sur toutes les réponses (le modèle a déjà scrapé l'info).
GLOBAL_FORBID = [
    "service communication",
    "service de communication",
    "je vous invite a contacter",
    "je vous invite a consulter",
    "veuillez contacter le service",
    "rapprochez vous du service",
    "consultez le site officiel",
    "contacter directement le service",
    "contactez directement le service",
]

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


def norm(s: str) -> str:
    """minuscule + sans accents + apostrophes/traits d'union → espaces."""
    s = (s or "").lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    for ch in ("'", "\u2019", "-", "\u2013", "\u2014"):
        s = s.replace(ch, " ")
    return " ".join(s.split())


def guess_lang(text: str) -> str:
    """Heuristique légère fr/en (assez pour un contrôle de stickiness)."""
    t = " " + norm(text) + " "
    fr = sum(t.count(f" {w} ") for w in
             ["le", "la", "les", "est", "vous", "pour", "une", "des", "du",
              "ministre", "gouvernement", "je", "au", "avec", "cette"])
    en = sum(t.count(f" {w} ") for w in
             ["the", "is", "you", "for", "and", "of", "government", "minister",
              "please", "who", "what", "this", "with", "can"])
    return "en" if en > fr else "fr"


def ask(url: str, q: str, mode: str, timeout: int) -> str:
    """POST /api/chat en SSE ; renvoie le texte final de la réponse."""
    payload = json.dumps({"message": q, "history": [], "mode": mode}).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/api/chat", data=payload,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    answer, tokens = "", ""
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except Exception:
                continue
            typ = ev.get("type")
            if typ == "token":
                tokens += ev.get("text", "")
            elif typ == "done":
                answer = ev.get("answer") or tokens
            elif typ == "error":
                raise RuntimeError(ev.get("message", "erreur API"))
    return (answer or tokens).strip()


def check(case: dict, answer: str) -> list:
    """Renvoie la liste des raisons d'échec ([] = succès)."""
    A = norm(answer)
    fails = []

    if not A:
        return ["réponse vide"]

    for req in case.get("expect_all", []):
        if isinstance(req, list):
            if not any(norm(x) in A for x in req):
                fails.append("aucun de " + "/".join(req))
        elif norm(req) not in A:
            fails.append("manque « %s »" % req)

    any_list = case.get("expect_any", [])
    if any_list and not any(norm(x) in A for x in any_list):
        fails.append("aucun de " + "/".join(any_list))

    for bad in case.get("forbid", []) + GLOBAL_FORBID:
        if norm(bad) in A:
            fails.append("interdit vu « %s »" % bad)

    want_lang = case.get("expect_lang")
    if want_lang and guess_lang(answer) != want_lang:
        fails.append("langue %s attendue (vu %s)" % (want_lang, guess_lang(answer)))

    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description="Banc de vérité alélo")
    ap.add_argument("--url", default=os.getenv("EVAL_URL", "http://localhost:8088"))
    ap.add_argument("-k", "--filter", default="", help="sous-chaîne id/catégorie")
    ap.add_argument("--category", default="", help="ne garder qu'une catégorie")
    ap.add_argument("--mode", default="", choices=["", "auto", "expert"],
                    help="ne garder qu'un mode (expert = lent)")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    with open(GOLDEN, encoding="utf-8") as f:
        cases = json.load(f)

    if args.filter:
        kf = args.filter.lower()
        cases = [c for c in cases if kf in c["id"].lower() or kf in c.get("category", "").lower()]
    if args.category:
        cases = [c for c in cases if c.get("category") == args.category]
    if args.mode:
        cases = [c for c in cases if c.get("mode", "auto") == args.mode]

    if not cases:
        print("Aucun cas ne correspond au filtre.")
        return 0

    print(f"\n{BOLD}alélo — banc de vérité{RESET}  ({len(cases)} cas · {args.url})\n")
    results, by_cat = [], {}
    t0 = time.time()

    for i, c in enumerate(cases, 1):
        mode = c.get("mode", "auto")
        cat = c.get("category", "?")
        print(f"{DIM}[{i:>2}/{len(cases)}]{RESET} {DIM}{cat:<16}{RESET} {c['id']:<20} ", end="", flush=True)
        t = time.time()
        try:
            ans = ask(args.url, c["q"], mode, args.timeout)
            fails = check(c, ans)
            err = None
        except (urllib.error.URLError, RuntimeError, TimeoutError, OSError) as e:
            ans, fails, err = "", ["ERREUR: %s" % e], str(e)
        dt = time.time() - t

        ok = not fails
        by_cat.setdefault(cat, [0, 0])
        by_cat[cat][0] += 1 if ok else 0
        by_cat[cat][1] += 1
        results.append({"id": c["id"], "category": cat, "mode": mode, "q": c["q"],
                        "ok": ok, "fails": fails, "answer": ans, "sec": round(dt, 1)})

        if ok:
            print(f"{GREEN}✅{RESET} {DIM}{dt:4.1f}s{RESET}")
            if args.verbose:
                print(f"      {DIM}{ans[:160]}{RESET}")
        else:
            print(f"{RED}❌{RESET} {DIM}{dt:4.1f}s{RESET}")
            for r in fails:
                print(f"      {YELLOW}· {r}{RESET}")
            if not err:
                print(f"      {DIM}→ {ans[:200]}{RESET}")

    total = len(results)
    passed = sum(1 for r in results if r["ok"])
    dur = time.time() - t0

    print(f"\n{BOLD}Résumé par catégorie{RESET}")
    for cat in sorted(by_cat):
        p, n = by_cat[cat]
        col = GREEN if p == n else RED
        print(f"  {cat:<18} {col}{p}/{n}{RESET}")

    fails = [r for r in results if not r["ok"]]
    verdict = f"{GREEN}✅ TOUT VERT{RESET}" if not fails else f"{RED}❌ {len(fails)} ÉCHEC(S){RESET}"
    print(f"\n{BOLD}TOTAL {passed}/{total}{RESET}  {verdict}   {DIM}({dur:.0f}s){RESET}")
    if fails:
        print("  Échecs : " + ", ".join(r["id"] for r in fails))

    _write_reports(results, passed, total, dur)
    print(f"{DIM}Rapport : {REPORT_MD}{RESET}\n")
    return 0 if not fails else 1


def _write_reports(results, passed, total, dur):
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump({"passed": passed, "total": total, "sec": round(dur, 1),
                   "results": results}, f, ensure_ascii=False, indent=2)

    lines = [f"# alélo — rapport banc de vérité", "",
             f"**{passed}/{total}** questions-vérité passées  ·  {dur:.0f}s", ""]
    fails = [r for r in results if not r["ok"]]
    if fails:
        lines += ["## ❌ Échecs", "", "| id | catégorie | problème | réponse |", "|---|---|---|---|"]
        for r in fails:
            ans = (r["answer"] or "").replace("|", "\\|").replace("\n", " ")[:120]
            lines.append(f"| `{r['id']}` | {r['category']} | {'; '.join(r['fails'])} | {ans} |")
        lines.append("")
    else:
        lines += ["## ✅ Tout vert", "", "Aucune régression détectée.", ""]
    lines += ["## Détail par cas", "", "| id | mode | ok | s | question |", "|---|---|---|---|---|"]
    for r in results:
        mark = "✅" if r["ok"] else "❌"
        lines.append(f"| `{r['id']}` | {r['mode']} | {mark} | {r['sec']} | {r['q']} |")
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
