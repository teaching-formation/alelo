"""Re-vérif nettoyage + estimation du temps d'indexation (mesure du débit bge-m3).
READ-ONLY : n'écrit RIEN dans ChromaDB."""
import os, json, glob, time
from collections import Counter
from urllib.parse import urlparse
import indexer

dirs = [d for d in sorted(glob.glob("data/raw_*")) if d not in {"data/raw", "data/raw_ansut"}]

# ── 1) Recompte total + collecte de tous les chunks propres (pour l'échantillon) ──
total_chunks = 0
per_inst = {}
sample_texts = []
for d in dirs:
    pages = []
    for f in glob.glob(f"{d}/*.json"):
        try: pages.append(json.load(open(f)))
        except: pass
    if not pages:
        continue
    doms = Counter(urlparse(p.get("url", "")).netloc.replace("www.", "") for p in pages)
    os.environ["INDEX_DOMAIN"] = doms.most_common(1)[0][0] if doms else ""
    texts, _ = indexer.chunk_pages(pages)
    per_inst[d.replace("data/raw_", "")] = len(texts)
    total_chunks += len(texts)
    if texts and len(sample_texts) < 80:
        sample_texts.extend(texts[:3])   # qq chunks par institution pour l'échantillon

print(f"TOTAL chunks propres (recompte) : {total_chunks}")
print()

# ── 2) Vérif qualité : 2 chunks réels de 4 institutions variées ──
print("=== Échantillon de chunks NETTOYÉS (doivent être du vrai contenu) ===")
for inst in ["servicepublic", "dgi", "transports", "cepici"]:
    d = f"data/raw_{inst}"
    pages = [json.load(open(f)) for f in glob.glob(f"{d}/*.json")[:60]]
    doms = Counter(urlparse(p.get("url", "")).netloc.replace("www.", "") for p in pages)
    os.environ["INDEX_DOMAIN"] = doms.most_common(1)[0][0] if doms else ""
    texts, _ = indexer.chunk_pages(pages)
    print(f"\n[{inst}] ({len(texts)} chunks)")
    for t in texts[:2]:
        print("   •", " ".join(t.split())[:150])

# ── 3) Mesure du débit d'embedding bge-m3 (sur 60 chunks réels) ──
print("\n=== Estimation du temps d'indexation ===")
from langchain_community.embeddings import OllamaEmbeddings
emb = OllamaEmbeddings(model="bge-m3", base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"))
probe = sample_texts[:60] or ["test"] * 60
emb.embed_documents(probe[:5])          # warmup (charge le modèle en VRAM)
t0 = time.time()
emb.embed_documents(probe)
dt = time.time() - t0
rate = len(probe) / dt
print(f"Débit mesuré : {rate:.1f} chunks/s ({len(probe)} chunks en {dt:.1f}s)")
mins = total_chunks / rate / 60
print(f"→ Estimation pour {total_chunks} chunks : ~{mins:.0f} min d'embedding "
      f"(+ ~1-2 min d'écriture ChromaDB)")
