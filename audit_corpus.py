"""Nettoyage à blanc v2 — mesure le corpus après filtres renforcés (langue, domaine,
boilerplate étendu). READ-ONLY : n'indexe rien."""
import os, json, glob
from collections import Counter
from urllib.parse import urlparse
import indexer

dirs = sorted(glob.glob("data/raw_*"))
skip = {"data/raw", "data/raw_ansut"}

print(f"{'institution':22s} {'brut':>5s} {'chunks(av)':>10s} {'chunks(ap)':>10s}")
print("-" * 55)
tot_raw = tot_after = 0

# comptage AVANT (sans filtre domaine/langue) pour comparaison : on neutralise INDEX_DOMAIN
for d in dirs:
    if d in skip:
        continue
    files = glob.glob(f"{d}/*.json")
    pages = []
    for f in files:
        try: pages.append(json.load(open(f)))
        except: pass
    if not pages:
        continue
    doms = Counter(urlparse(p.get("url", "")).netloc.replace("www.", "") for p in pages)
    expected = doms.most_common(1)[0][0] if doms else ""

    # APRÈS : filtres domaine + langue + boilerplate étendu
    os.environ["INDEX_DOMAIN"] = expected
    texts_after, _ = indexer.chunk_pages(pages)

    # AVANT : sans filtre domaine/langue (on simule l'ancien en désactivant le domaine)
    os.environ["INDEX_DOMAIN"] = ""
    # (le boilerplate étendu et le filtre langue restent, donc "avant" ici = ancien pipeline
    #  approximatif ; l'important est l'effet du domaine)
    texts_before, _ = indexer.chunk_pages(pages)

    name = d.replace("data/raw_", "")
    print(f"{name:22s} {len(pages):5d} {len(texts_before):10d} {len(texts_after):10d}")
    tot_raw += len(pages); tot_after += len(texts_after)

print("-" * 55)
print(f"{'TOTAL':22s} {tot_raw:5d} {'':10s} {tot_after:10d}")
