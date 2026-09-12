# alélo 🇨🇮

**Assistant IA citoyen pour les services publics de Côte d'Ivoire — 100 % local.**

alélo répond en langage naturel aux questions sur les démarches administratives et les
institutions ivoiriennes (passeport, CNI, impôts, création d'entreprise, composition du
gouvernement…), en s'appuyant sur le contenu **officiel** des sites publics. Tout tourne
**en local** : aucune donnée citoyenne ne quitte la machine.

> Le problème : l'information administrative est éparpillée sur des dizaines de sites et de
> guichets. alélo la centralise derrière une seule conversation, à l'écrit comme à la voix,
> 24h/24 — sans inventer.

---

## Ce que fait alélo

- 💬 **Réponses sourcées** sur les démarches et institutions, avec citation des sources.
- 🧠 **Deux cerveaux** : *Auto* (RAG simple, rapide) et *Expert* (**RAG agentique** : décomposition
  des questions, correction de la récupération, outils, auto-vérification).
- 🕸️ **GraphRAG** : un graphe de connaissances des services citoyens (service → institution →
  documents → coût → délai) injecté comme contexte faisant autorité.
- 🗣️ **Voix multilingue** : compréhension **et** réponse en **français et en anglais** (STT + TTS).
- 📄 **Génération de documents** (PDF / Word / Excel) sur simple demande, même à la voix.
- 🛡️ **Garde-fous** : anti-hallucination (vérification des chiffres), anti-esquive, cloisonnement
  des institutions, qualification par date des faits nominatifs, stickiness de langue.
- ✅ **Banc de vérité** (`eval/`) : batterie de questions-vérité anti-régression, feu vert/rouge.
- 🔍 **Observabilité locale** : chaque requête tracée (`/api/traces/view`), 👍/👎 joints.

## Architecture

```
Navigateur ──HTTP──> Next.js (web/, :3000) ──SSE──> FastAPI (api.py, :8088)
                                                          │
                                    ┌─────────────────────┼───────────────────────┐
                                    ▼                      ▼                       ▼
                          Ollama (:11434)        ChromaDB (data/)      Serveur natif (:8600)
                          alelo (7B) / 14B       17k+ chunks           Whisper large-v3 (STT)
                          bge-m3 (embeddings)    bge-m3                bge-reranker-v2-m3
```

Le pipeline de récupération est **hybride** : BM25 (mots-clés) + dense (`bge-m3`) → fusion **RRF**
→ **reranker** cross-encoder. En mode Expert, une boucle **agentique** en 4 étapes s'ajoute
(corrective RAG, décomposition, outils, auto-vérification). Un **routeur anti-mélange** cloisonne
les institutions.

### Stack

| Composant | Technologie |
|-----------|-------------|
| LLM | Ollama — `alelo` (7B) / `alelo-14b`, `FROM qwen2.5` + persona (Modelfile) |
| Embeddings | `bge-m3` (1024-dim) via Ollama |
| Reranker | `bge-reranker-v2-m3` (cross-encoder, servi en natif) |
| Vector store | ChromaDB (persisté) |
| STT | Whisper large-v3 via `mlx-whisper` (GPU Metal), fallback faster-whisper |
| TTS | Piper (multi-voix FR + EN) |
| API | FastAPI (SSE streaming) |
| Front | Next.js 15 (interface type ChatGPT, mode vocal) |
| Documents | python-docx, openpyxl, fpdf2 |

## Démarrage

**Prérequis** : macOS (GPU Metal recommandé), [Docker](https://www.docker.com/),
[Ollama](https://ollama.com), Python 3.10+.

### 1. Modèles Ollama (natifs)

```bash
ollama pull qwen2.5:7b && ollama pull qwen2.5:14b && ollama pull bge-m3
ollama create alelo     -f Modelfile
ollama create alelo-14b -f Modelfile.14b
```

### 2. Serveur natif STT + reranker (GPU Metal)

```bash
cd voxtral-server && pip install -r requirements.txt
python voxtral_server.py          # écoute sur :8600
```

### 3. Construire la base de connaissances

Le corpus n'est **pas** versionné (contenu de sites publics — voir *Données*). On le reconstruit :

```bash
python scraper.py     # crawl des sites publics → data/raw_*/
python indexer.py     # chunks + embeddings bge-m3 → data/chroma_db/
```

### 4. Lancer l'application

```bash
docker compose up -d          # api (:8088) + web (:3000)
```

Ouvrir **http://localhost:3000**.

## Banc de vérité (anti-régression)

```bash
python eval/gen_golden.py     # régénère la batterie depuis les sources (ex. composition du gouvernement)
python eval/run_eval.py       # rejoue les questions-vérité → rapport vert/rouge (exit 0/1)
```

Règle : chaque mauvaise surprise (un 👎, un bug en démo) → on **ajoute une question-vérité**.
Détails dans [`eval/README.md`](eval/README.md).

## Observabilité

Dashboard local des requêtes (question, mode, étapes agentiques, latence, 👍/👎) :
**http://localhost:8088/api/traces/view**

## Structure du projet

```
├── api.py             API FastAPI (chat SSE, transcribe, tts, feedback, traces)
├── rag_chain.py       Cœur RAG : hybride + reranker + agentique + garde-fous + GraphRAG
├── scraper.py         Crawl des sites publics → data/raw_*/
├── indexer.py         Chunks + embeddings → ChromaDB
├── graph_build.py     Construit le graphe de connaissances (data/knowledge_graph.json)
├── documents.py       Génération PDF / Word / Excel
├── voice.py           STT/TTS (Whisper + Piper)
├── tracing.py         Observabilité locale (traces.jsonl)
├── eval/              Banc de vérité (golden_core.json, gen_golden.py, run_eval.py)
├── voxtral-server/    Serveur natif Whisper + reranker (:8600)
├── web/               Front Next.js
├── Modelfile(.14b)    Personas alélo (Ollama)
└── docker-compose.yml api + web
```

## Données

Le corpus est **scrapé de sites publics** ivoiriens. Il n'est **pas** redistribué dans ce repo
(taille + respect du contenu source) : lance `scraper.py` puis `indexer.py` pour le reconstruire.
Seuls quelques artefacts curés, petits et publics, sont versionnés
(`data/raw_gouvernement/composition.json`, `data/knowledge_graph.json`). Les journaux d'usage
(`feedback.jsonl`, `traces.jsonl`) ne sont pas versionnés (vie privée).

## Licence & auteur

Projet développé par **Diakité Mamadou Youssouf**. Voir [`MANIFESTE.md`](MANIFESTE.md) pour la vision.
Construit avec l'aide de [Claude Code](https://claude.com/claude-code).

Licence : MIT (voir `LICENSE`).
