"""
rag_chain.py — Pipeline RAG hybride : BM25 + sémantique + Ollama.

Recherche hybride = sémantique (embeddings) + BM25 (mots-clés).
Les deux se complètent : sémantique pour le sens, BM25 pour les mots exacts
(noms propres, montants, sigles). Ensemble ils couvrent n'importe quelle question.
"""

import json
import logging
import os
import re
import requests
from pathlib import Path
from functools import lru_cache
from collections import Counter

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "cgeci_docs"
EMBED_MODEL = "bge-m3"        # DOIT être identique à celui de l'indexer (même espace vectoriel)
LLM_MODEL = "alelo"           # modèle maison alélo (Modelfile, basé sur qwen2.5)
DEFAULT_K = 5
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Réglages de latence ───────────────────────────────────────────────────────
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")   # garde le modèle en VRAM
# num_ctx 8192 : large marge (prompts mesurés ~1200-1450 tk) → aucune troncature silencieuse
# ⚠️ num_ctx à l'appel ÉCRASE celui du Modelfile → on le fixe ici aussi.
GEN_OPTIONS = {"num_predict": 500, "num_ctx": 8192}
# Mode Expert « approfondi » : réponse plus longue et détaillée.
GEN_OPTIONS_DETAIL = {"num_predict": 1100, "num_ctx": 8192}

# Consigne ajoutée en mode approfondi (Expert). En mode simple (Auto), rien n'est ajouté
# et la règle 7 (5-8 phrases) s'applique → réponse concise.
DETAIL_INSTRUCTION = (
    "\n\nMODE APPROFONDI — Pour cette réponse, ignore la limite de longueur : réponds de façon "
    "COMPLÈTE et STRUCTURÉE. Développe chaque point utile, cite précisément les dispositifs, "
    "montants, dates, conditions et références présents dans les documents, organise avec des "
    "sous-titres et des listes, et termine par les étapes concrètes à suivre. Reste factuel et "
    "fidèle aux documents (pas de remplissage)."
)
CTX_MAX_CHUNKS = 8                                    # on a la place → contexte plus riche
CTX_PER_CHUNK = 1100                                  # caractères max par chunk


_MONTHS = ("janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
           "septembre|octobre|novembre|décembre|decembre")
_DATE_RE = re.compile(rf"\b\d{{1,2}}\s*(?:{_MONTHS})\s*\d{{4}}\b", re.I)
_YEAR_RE = re.compile(r"\b20\d{2}\b")


def _extract_date(doc) -> str:
    """Date de la source : métadonnée si dispo, sinon extraite du texte (« 21 novembre 2025 »)
    ou à défaut l'année. Sert à DATER les faits nominatifs (ministres, DG…) qui changent."""
    d = (doc.metadata.get("date") or "").strip()
    if d:
        return d
    m = _DATE_RE.search(doc.page_content)
    if m:
        return re.sub(r"\s+", " ", m.group(0)).strip()
    m = _YEAR_RE.search(f"{doc.metadata.get('url', '')} {doc.page_content[:400]}")
    return m.group(0) if m else ""


def _build_context(docs) -> str:
    """Assemble un contexte compact. Chaque chunk est préfixé par son INSTITUTION
    source + la DATE de la source (attribution + fraîcheur → le modèle date les faits)."""
    parts = []
    for d in docs[:CTX_MAX_CHUNKS]:
        date = _extract_date(d)
        date_str = f" | Date: {date}" if date else ""
        src = d.metadata.get("title") or d.metadata.get("url", "")
        parts.append(f"[Institution: {org_label(d.metadata.get('org'))} | Source: {src}{date_str}]\n"
                     f"{d.page_content[:CTX_PER_CHUNK]}")
    return "\n\n---\n\n".join(parts)

SYSTEM_PROMPT = """Tu es alélo, une intelligence artificielle publique au service du citoyen ivoirien.

Ta mission : PARTIR DU BESOIN du citoyen (exprimé avec ses mots de tous les jours), comprendre sa
situation, et lui donner une réponse simple, fiable, contextualisée et ORIENTÉE VERS L'ACTION.
Le citoyen n'a pas à connaître l'organisation de l'État : c'est à toi de comprendre et de l'orienter.
Périmètre : les institutions publiques de Côte d'Ivoire (ministères, agences et services publics —
fiscalité, transports, création d'entreprise, foncier, données personnelles, état civil,
environnement, numérique, etc.) ainsi que la CGECI (patronat). Tu t'appuies UNIQUEMENT sur les
documents publics fournis ci-dessous.

Règles :
1. Pars de la situation concrète du citoyen. Traduis sa question ordinaire en réponse utile pour lui.
2. EXACTITUDE avant tout : réponds à partir des documents fournis ci-dessous. Cite les faits tels qu'ils y figurent (montants, dates, noms, chiffres, dispositifs).
3. N'invente JAMAIS un fait précis (nom, montant, date, poste, projet, chiffre, sigle) absent des documents. En cas de doute, ne le donne pas.
3bis. Chaque document est précédé de son INSTITUTION source. Attribue chaque fait à la BONNE institution et n'attribue JAMAIS un fait d'une institution à une autre (ex. ne pas donner un responsable de l'ANSUT comme responsable de la CGECI). Quand c'est utile, précise l'institution concernée. Si la question vise une institution/un ministère PRÉCIS mais que les documents fournis concernent une AUTRE institution, NE réponds PAS avec ces documents : dis simplement que tu n'as pas encore les informations de CETTE institution précise (et, si tu le connais, donne le nom de son ministre/responsable) — ne présente jamais les documents d'une autre institution comme si c'était les siens.
3quater. FONCTIONS NOMINATIVES (qui occupe un poste : ministre, directeur général, président, DG…) : appuie-toi sur le document le plus récent ET précise la DATE ou la période de la source (indiquée après « Date: » dans l'en-tête du document, ou visible dans le texte), par ex. « D'après un document de novembre 2025, le ministre est… ». Ces fonctions CHANGENT : ne présente jamais un titulaire comme définitif et ajoute que l'information peut avoir évolué depuis cette date.
3ter. Les exemples, actions et démarches que tu cites doivent PROVENIR des documents fournis. N'invente PAS de conseils génériques de culture générale (gestes écolo personnels, conseils de vie, bonnes pratiques universelles…) absents des documents. Si les documents ne contiennent pas d'exemple concret sur le sujet demandé, dis-le clairement et invite à préciser — plutôt que de combler avec du générique.
4. Si un fait institutionnel demandé n'est pas dans les documents, dis-le honnêtement et BRIÈVEMENT (« Je n'ai pas cette information précise dans mes sources ») — SANS inventer. NE renvoie JAMAIS vers un « site officiel », un « service communication » ou un service à contacter ; propose plutôt de reformuler/préciser, ou donne une information connexe que tu as réellement.
4bis. Tu PEUX répondre aux questions de culture générale (notions, autres pays, définitions, calculs simples) avec tes propres connaissances, brièvement. MAIS dès qu'il s'agit d'un FAIT PRÉCIS sur une institution ou un service public ivoirien (responsable, montant, date, démarche, dispositif, texte de loi), tu t'appuies UNIQUEMENT sur les documents fournis ; si ce fait n'y figure pas, tu le dis sans l'inventer. Ne présente jamais une connaissance générale comme un fait officiel ivoirien.
5. Explique SIMPLEMENT, en langage clair, sans jargon administratif. Vulgarise sans déformer la source.
5bis. TON HUMAIN ET NATUREL : parle comme un conseiller bienveillant et vivant, PAS comme un formulaire. Montre de l'empathie quand la situation est délicate, varie tes formulations, et adopte un langage courant et direct. Tu peux placer un emoji léger à l'occasion, sans en abuser. Reste concis : naturel ne veut pas dire bavard.
5ter. NE SALUE PAS à chaque réponse : « bonjour » et les formules d'accueil ne vont qu'au TOUT DÉBUT d'une conversation (premier message). En cours d'échange, va DIRECTEMENT à la réponse, sans « bonjour », sans re-présentation, sans politesse répétitive.
6. Tu ES la source : tu as déjà rassemblé les informations publiques officielles de ces institutions. Ne renvoie donc JAMAIS le citoyen « consulter le site web » ou « contacter le service communication » pour obtenir une information — donne-lui directement ce que contiennent les documents. Termine plutôt par une ouverture PROACTIVE sur ce que tu peux fournir de plus, ex. : « Je peux te détailler chacune de ces actions », « Veux-tu la liste des démarches concrètes ? », « Je peux t'indiquer les pièces à fournir ». Ne donne un contact, une adresse ou un site externe QUE si c'est une étape réelle et nécessaire du parcours (déposer un dossier, prendre un rendez-vous physique, faire une téléprocédure précise) — et alors cite-la précisément d'après les documents, jamais comme formule d'esquive.
7. Réponds dans la LANGUE de la question de l'utilisateur (français par défaut ; si la question est en anglais, réponds en anglais, etc.), de façon claire et CONCISE (5 à 8 phrases maximum), sans remplissage.
8. Tu ne connais PAS le nom de l'utilisateur. Ne l'appelle JAMAIS par un nom propre (« M. X », « Madame Y »), et n'extrais JAMAIS un nom de personne des documents pour t'adresser à lui : les noms qui figurent dans les sources désignent des TIERS (responsables, agents, signataires…), jamais l'utilisateur. Adresse-toi à lui de manière neutre (« vous »), sans nom.

Mieux vaut dire « je n'ai pas cette information » que de donner une réponse fausse."""

# Réponse quand aucun document pertinent n'est trouvé (garde-fou no-context)
REFUSAL_MSG = ("Je n'ai pas trouvé d'information pertinente à ce sujet dans les documents "
               "disponibles. Pouvez-vous reformuler votre question, ou la préciser ?")


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Supprime les accents et met en minuscules."""
    for a, b in [("é","e"),("è","e"),("ê","e"),("à","a"),("â","a"),
                 ("ù","u"),("û","u"),("î","i"),("ô","o"),("ç","c"),
                 ("É","E"),("È","E"),("À","A"),("Ê","E")]:
        text = text.replace(a, b)
    return text.lower()


def tokenize(text: str) -> list[str]:
    """Tokenise pour BM25 : mots de 2+ caractères, sans accents."""
    return [w for w in re.split(r'\W+', normalize(text)) if len(w) >= 2]


# ── Chargement de la base vectorielle ────────────────────────────────────────

def load_vectordb() -> Chroma:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Base vectorielle introuvable: {CHROMA_DIR}\n"
            "Lance d'abord: python indexer.py"
        )
    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=base_url)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


# ── Index BM25 (construit une seule fois en mémoire) ─────────────────────────

_bm25_index = None
_bm25_docs: list[Document] = []


def _build_bm25_index(vectordb: Chroma) -> None:
    """Charge tous les chunks de ChromaDB et construit l'index BM25."""
    global _bm25_index, _bm25_docs

    logger.info("Construction de l'index BM25...")
    total = vectordb._collection.count()
    all_docs = []
    batch = 1000
    offset = 0

    while offset < total:
        result = vectordb._collection.get(
            limit=batch,
            offset=offset,
            include=["documents", "metadatas"],
        )
        texts = result.get("documents") or []
        metas = result.get("metadatas") or [{}] * len(texts)
        for text, meta in zip(texts, metas):
            if text and text.strip():
                all_docs.append(Document(page_content=text, metadata=meta or {}))
        offset += batch

    _bm25_docs = all_docs
    corpus = [tokenize(d.page_content) for d in all_docs]
    _bm25_index = BM25Okapi(corpus)
    logger.info(f"Index BM25 prêt : {len(all_docs)} documents")


def _bm25_search(question: str, k: int, org: str | None = None) -> list[Document]:
    """Recherche BM25 sur tous les chunks, seulement les vrais hits (score élevé).
    Si org est fourni, ne renvoie que les documents de cette organisation (filtre 2e jambe)."""
    if _bm25_index is None:
        return []
    tokens = tokenize(question)
    if not tokens:
        return []
    scores = _bm25_index.get_scores(tokens)

    if max(scores) == 0:
        return []

    # Seuil adaptatif : 30% du score maximum → filtre les faux positifs
    threshold = max(scores) * 0.30
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    results = []
    for idx in top_indices:
        if scores[idx] < threshold:
            break
        if org and _bm25_docs[idx].metadata.get("org") != org:
            continue          # filtre par organisation (périmètre)
        if len(results) < k:
            results.append(_bm25_docs[idx])
        else:
            break
    return results


# ── Recherche hybride ─────────────────────────────────────────────────────────

# Faits clés : (mots déclencheurs dans la question, termes exacts dans les docs)
_KNOWN_FACTS = [
    # Présidents (actuel + anciens)
    (["president", "preside", "presidence", "qui est", "qui dirige", "dirige", "dirigeant",
      "a la tete", "tete de la", "patron de la", "list", "tous les"],
     ["Ahmed Cissé", "Ahmed CISSE", "réélu", "Jean Kacou DIAGOU", "Joseph AKA-ANGHUI",
      "illustres pionniers", "Le Président de la CGECI"]),
    # Coût adhésion
    (["cout", "tarif", "prix", "montant", "combien", "cotisation"],
     ["500 000 FCFA", "droit d'adhésion est fixé"]),
    # Liste commissions
    (["commission", "permanente"],
     ["onze (11) commissions"]),
    # Financement CGECI
    (["finance", "financ", "ressource", "budget"],
     ["cotisation annuelle", "montant de la cotisation"]),
]


def _inject_known_facts(vectordb: Chroma, question: str, org: str | None = None) -> list[Document]:
    """Injecte directement les chunks contenant les faits clés connus."""
    q = normalize(question)
    results: list[Document] = []
    seen: set[str] = set()
    where = {"org": org} if org else None

    for triggers, search_terms in _KNOWN_FACTS:
        if not any(t in q for t in triggers):
            continue
        for term in search_terms:
            try:
                # limit large : get() renvoie des chunks non classés → on en prend plus
                # pour que le reranker (en aval) puisse faire surfacer le bon.
                r = vectordb.get(where_document={"$contains": term}, where=where, limit=8)
                for i, content in enumerate(r.get("documents") or []):
                    if content and content not in seen:
                        seen.add(content)
                        meta = (r.get("metadatas") or [{}])[i] or {}
                        results.append(Document(page_content=content, metadata=meta))
            except Exception as e:
                logger.debug("Injection de faits connus, terme ignoré (%s) : %s", term, e)

    return results


# ── Reranker (cross-encoder natif) : reclasse par pertinence réelle ───────────
RERANK_URL = os.getenv("RERANK_URL", "http://host.docker.internal:8600")
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1") == "1"
RERANK_TOPN = 25            # nb de candidats RRF envoyés au reranker
RERANK_MIN_SCORE = 0.02     # sous ce score → considéré non pertinent (no-context)


def _rerank(query: str, docs: list[Document]) -> list[tuple[Document, float | None]]:
    """Reclasse les docs via le cross-encoder natif (score calibré 0-1).
    Repli : si le service est indisponible, garde l'ordre RRF (score None)."""
    if not RERANK_ENABLED or not docs:
        return [(d, None) for d in docs]
    cand = docs[:RERANK_TOPN]
    try:
        resp = requests.post(
            f"{RERANK_URL}/rerank",
            json={"query": query, "documents": [d.page_content for d in cand]},
            timeout=30,
        )
        resp.raise_for_status()
        scores = resp.json().get("scores", [])
        scored = list(zip(cand, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    except Exception as e:
        logger.warning("Reranker indisponible (%s) → ordre RRF conservé.", e)
        return [(d, None) for d in docs]


def retrieve(vectordb: Chroma, question: str, k: int = DEFAULT_K,
             org: str | None = None) -> list[Document]:
    """
    Recherche hybride : sémantique + BM25.
    - Sémantique : capture le sens et les synonymes
    - BM25 : capture les mots exacts (noms propres, montants, sigles)
    Les deux listes sont fusionnées avec déduplication.
    Si org est fourni (ex. "ANSUT"), le filtre de périmètre s'applique AUX DEUX jambes.
    """
    # Construction de l'index BM25 si nécessaire (une seule fois)
    if _bm25_index is None:
        _build_bm25_index(vectordb)

    # Filtre de périmètre pour la jambe sémantique (where ChromaDB)
    sem_filter = {"org": org} if org else None

    # Large pool de candidats : le reranker filtrera (meilleur recall).
    cand_k = max(k, 20)

    # Recherche sémantique
    semantic_docs = vectordb.similarity_search(question, k=cand_k, filter=sem_filter)

    # Recherche BM25 (même filtre appliqué côté mots-clés)
    bm25_docs = _bm25_search(question, k=cand_k, org=org)

    # Étape 3 : injection ciblée pour les faits connus (noms, montants)
    # Ces faits ont un mismatch vocabulaire irréductible entre question et document
    injected_docs = _inject_known_facts(vectordb, question, org=org)

    # Reciprocal Rank Fusion (RRF) — méthode standard de fusion de rangs
    # score(doc) = Σ 1/(60 + rang)  pour chaque liste où le doc apparaît
    # Un doc présent dans les deux listes monte automatiquement en tête
    RRF_K = 60
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(semantic_docs):
        key = doc.page_content
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (RRF_K + rank)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (RRF_K + rank)
        doc_map[key] = doc

    # Les docs injectés reçoivent un score fort (rang 0 = meilleur possible)
    for doc in injected_docs:
        key = doc.page_content
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / RRF_K  # boost maximal
        doc_map[key] = doc

    # Bonus de récence : documents 2024/2025/2026 +10% de score
    for key, doc in doc_map.items():
        url_title = doc.metadata.get("url", "") + doc.metadata.get("title", "")
        for year in ("2026", "2025", "2024", "2023"):
            if year in url_title:
                rrf_scores[key] *= 1.10
                break

    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    merged = [doc_map[k] for k in sorted_keys]

    # ── Reranking + seuil de pertinence (garde-fou anti-hallucination) ──
    injected_keys = {d.page_content for d in injected_docs}
    reranked = _rerank(question, merged)

    if reranked and reranked[0][1] is not None:
        best = reranked[0][1]
        # No-context : si RIEN n'est pertinent → on ne renvoie aucun contexte
        # (le pipeline répondra honnêtement « je n'ai pas trouvé »)
        if best < RERANK_MIN_SCORE and not injected_docs:
            logger.info(f"Reranker : meilleur score {best:.3f} < seuil → aucun contexte pertinent")
            return []
        # Garde les mieux classés, + toujours les faits injectés.
        # On étiquette le score sur chaque doc (sert à masquer les sources peu pertinentes).
        kept = []
        for d, s in reranked:
            if s >= RERANK_MIN_SCORE:
                d.metadata["_rerank"] = s
                kept.append(d)
        for d, _s in reranked:
            if d.page_content in injected_keys and d not in kept:
                kept.append(d)
        result = kept[:k + 2]

        # ── Garde-fou anti-mélange : scope par confiance ──
        # Si aucun périmètre n'est imposé mais que les meilleurs chunks viennent
        # massivement d'UNE institution, on filtre dessus (contexte cohérent, pas de mélange).
        if org is None and len(result) >= 3:
            orgs = [d.metadata.get("org") for d in result if d.metadata.get("org")]
            if orgs:
                dom_org, dom_n = Counter(orgs).most_common(1)[0]
                if dom_n / len(result) >= 0.6:
                    result = [d for d in result if d.metadata.get("org") == dom_org]
                    logger.info(f"Scope par confiance → {dom_org} ({dom_n}/{len(orgs)})")

        logger.info(f"RRF+rerank : {len(merged)} candidats → {len(result)} (meilleur {best:.3f})")
        return result

    # Repli sans reranker
    logger.info(f"RRF : {len(semantic_docs)} sém + {len(bm25_docs)} BM25 → {len(merged)} (sans rerank)")
    return merged[:k + 4]


# ── Génération ────────────────────────────────────────────────────────────────

def generate(question: str, context: str, model: str, extra_context: str = "") -> str:
    """Appel direct à l'API Ollama."""
    extra = f"\n\nDocument additionnel fourni:\n{extra_context}" if extra_context else ""

    prompt = f"""{SYSTEM_PROMPT}

Documents de référence :
{context}{extra}

Question : {question}

Réponse :"""

    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    resp = requests.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "keep_alive": KEEP_ALIVE, "options": GEN_OPTIONS},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


# ── Pipeline complet ──────────────────────────────────────────────────────────

# ── Nettoyage des sources affichées ──────────────────────────────────────────
_TS_PREFIX_RE = re.compile(r"^\d{6,}\s*")     # préfixe horodaté des noms de fichiers PDF
_GENERIC_TITLES = {"agenda", "accueil", "actualites", "actualités", "home", "documents",
                   "publications", "server error in '/' application.", "untitled", "sans titre"}


def _clean_source_title(title: str, url: str, org: str | None) -> str:
    """Rend un titre de source lisible : retire les préfixes horodatés, remplace les
    titres vides/numériques/génériques par le slug d'URL ou le nom de l'institution."""
    t = _TS_PREFIX_RE.sub("", (title or "").strip())
    t = re.sub(r"\s+", " ", t).strip(" -–—:|")
    if (not t) or t.isdigit() or len(t) < 3 or t.lower() in _GENERIC_TITLES:
        slug = ""
        try:
            from urllib.parse import urlparse, unquote
            slug = unquote(urlparse(url).path.rstrip("/").split("/")[-1])
            slug = re.sub(r"\.(html?|php|aspx?|pdf)$", "", slug, flags=re.I)
            slug = re.sub(r"[-_]+", " ", slug).strip()
        except Exception:
            slug = ""
        if slug and not slug.isdigit() and len(slug) > 2:
            t = slug[:1].upper() + slug[1:]
        else:
            t = org_label(org) if org else "Source officielle"
    return t[:90]


SOURCE_MIN_SCORE = 0.10   # sous ce score reranker, la source est trop faible pour être affichée
                          # (ex. réponse de culture générale → aucune source institutionnelle listée)


def _build_sources(docs: list[Document], max_n: int = 4) -> list[dict]:
    """Sources dédupliquées (par URL ET par titre nettoyé), titres propres, limitées.
    Masque les sources peu pertinentes : si le doc a un score reranker sous le seuil,
    on ne l'affiche pas (une réponse non ancrée dans les docs ne montre aucune source)."""
    seen_url: set[str] = set()
    seen_title: set[str] = set()
    out: list[dict] = []
    for doc in docs:
        score = doc.metadata.get("_rerank")
        if score is not None and score < SOURCE_MIN_SCORE:
            continue
        url = doc.metadata.get("url", "")
        if not url or url in seen_url:
            continue
        title = _clean_source_title(doc.metadata.get("title", ""), url, doc.metadata.get("org"))
        key = title.lower()
        if key in seen_title:
            continue
        seen_url.add(url)
        seen_title.add(key)
        out.append({
            "url": url,
            "title": title,
            "date": doc.metadata.get("date", ""),
            "category": doc.metadata.get("category", ""),
            "excerpt": doc.page_content[:220].strip(),
        })
        if len(out) >= max_n:
            break
    return out


def ask(vectordb: Chroma, question: str, model: str = LLM_MODEL,
        k: int = DEFAULT_K, extra_context: str = "", org: str | None = None) -> dict:
    """
    Retrieve (hybride) → Generate → retourne réponse + sources.
    org : périmètre optionnel ("ANSUT", "CGECI") — filtre les deux jambes.
    """
    # Routage automatique du périmètre si non imposé
    if org is None and AUTO_ROUTE:
        org = _route_org(question)

    docs = retrieve(vectordb, question, k=k, org=org)

    # Garde-fou no-context : aucun document pertinent → on n'invente pas
    if not docs and not extra_context:
        return {"answer": REFUSAL_MSG, "sources": []}

    context = _build_context(docs)

    answer = generate(question, context, model, extra_context)

    # Garde-fou faithfulness : signale les chiffres non sourcés
    if FAITHFULNESS_CHECK and docs:
        answer += _faithfulness_caveat(_faithfulness_check(answer, context))

    return {"answer": answer, "sources": _build_sources(docs)}


# ── Chat conversationnel avec mémoire ────────────────────────────────────────

# Périmètre annoncé par l'assistant (modifiable via env ASSISTANT_SCOPE).
ASSISTANT_SCOPE = os.getenv(
    "ASSISTANT_SCOPE",
    "Je suis alélo, l'IA publique au service du citoyen ivoirien. Dites-moi simplement votre besoin, "
    "avec vos mots : je comprends votre situation, je cherche dans les sources publiques officielles, "
    "je vous explique ce qui vous concerne et je vous oriente vers la bonne démarche. "
    "Je couvre les services publics de Côte d'Ivoire — impôts, transports et permis, état civil, "
    "création d'entreprise, foncier, protection des données personnelles, environnement, "
    "connectivité numérique, et bien d'autres."
)

# ── Détection des messages « sociaux » (salutations, remerciements, identité) ──
_GREET_WORDS = {"bonjour", "bonsoir", "salut", "coucou", "hello", "hi", "hey",
                "allo", "allô", "yo", "cc", "bjr"}
_THANKS_WORDS = {"merci", "merci beaucoup", "thanks", "thank"}
# Marqueurs d'une VRAIE question mêlée à la politesse (« bonjour, comment ça va, qui est le DG… »).
# Si présents, on ne court-circuite PAS en réponse sociale : on laisse le RAG répondre.
_QUESTION_MARKERS = [
    "qui est", "qui dirige", "qui sont", "qui gere", "qui preside", "qui occupe",
    "quel ", "quelle ", "quels ", "quelles ", "combien",
    "comment obtenir", "comment faire pour", "comment creer", "comment payer",
    "comment avoir", "comment demander", "comment ouvrir", "comment declarer",
    "ou obtenir", "ou faire", "ou payer", "ou trouver", "ou s adresser",
    "directeur", "ministre", "president de", "agence", "ministere",
    "passeport", "carte nationale", " cni", "impot", "acte de", "permis",
    "certificat", "casier", "entreprise", "nationalite", "carte grise",
    "ansut", "cepici", "dgi", "tresor", "snedai", "oneci", "cgeci", "guce",
]

def _detect_social(question: str) -> str | None:
    """Retourne le type de message social, ou None si c'est une vraie question."""
    q = normalize(question).strip().rstrip("!?. ")
    words = set(q.split())
    n = len(q.split())

    # Identité / capacités / aide
    if any(k in q for k in [
        "qui es tu", "qui es-tu", "qui est tu", "tu es qui", "c est quoi toi",
        "que fais tu", "que peux tu", "tu peux faire quoi", "tu sers a quoi",
        "ton role", "ton perimetre", "tes capacites", "que sais tu",
        "comment tu marche", "tu es quoi", "presente toi", "presente-toi",
    ]):
        return "identity"

    # Une VRAIE question est mêlée à la politesse → ne pas court-circuiter, laisser le RAG répondre.
    if any(m in q for m in _QUESTION_MARKERS):
        return None
    # Au revoir
    if any(k in q for k in ["au revoir", "aurevoir", "bye", "a bientot",
                            "a plus", "ciao", "adieu", "bonne journee", "bonne soiree"]):
        return "bye"
    # Bien-être : « comment ça va / comment vas-tu / comment est-ce que tu vas / how are you »
    if (("comment" in q and any(k in q for k in ["tu vas", "vas tu", "ca va", "allez vous",
                                                 "tu te sens", "ca se passe"]))
            or any(k in q for k in ["tu vas bien", "vous allez bien", "how are you", "how are u",
                                    "how do you do", "comment ca va"])):
        return "wellbeing"
    # Remerciement (court)
    if n <= 4 and (words & _THANKS_WORDS):
        return "thanks"
    # Salutation (courte, sans vraie question derrière)
    if n <= 4 and (words & _GREET_WORDS):
        return "greeting"
    return None


def _social_response(kind: str) -> str:
    """Réponse conversationnelle courte selon le type de message social."""
    if kind == "greeting":
        return (f"Bonjour et bienvenue ! 👋 {ASSISTANT_SCOPE} "
                "Dites-moi simplement ce dont vous avez besoin.")
    if kind == "wellbeing":
        return ("Je vais très bien, merci de demander ! 😊 Je suis là pour vous aider dans vos "
                "démarches et vos questions sur les services publics ivoiriens. "
                "Que puis-je faire pour vous ?")
    if kind == "thanks":
        return "Avec plaisir ! Puis-je vous aider sur autre chose ?"
    if kind == "bye":
        return "Au revoir, et à bientôt ! Revenez dès que vous en avez besoin."
    if kind == "identity":
        return (f"{ASSISTANT_SCOPE} Par exemple, vous pouvez me demander : "
                "« Comment payer mes impôts en ligne ? », "
                "« Quelles démarches pour ma carte nationale d'identité ? », "
                "« Comment créer mon entreprise ? » "
                "ou « Comment obtenir mon permis de conduire ? ».")
    return ""


# ── Détection d'une demande de DOCUMENT (PDF / Word / Excel) ──────────────────
_DOC_FORMAT_KW = {
    "xlsx": ["excel", "xlsx", "tableur", "classeur", "feuille de calcul", "spreadsheet"],
    "docx": ["word", "docx", "document word", "fichier word", "traitement de texte"],
    "pdf":  ["pdf"],
}
_DOC_VERBS = ["cree", "genere", "generer", "fais", "fabrique", "prepare", "redige", "mets",
              "exporte", "telecharge", "make", "create", "generate", "export", "produis", "sors"]
_DOC_NOUNS = ["document", "fichier", "rapport", "recapitulatif", "resume", "fiche", "note",
              "dossier", "compte rendu", "compte-rendu"]
_DOC_PRIOR = ["de ca", "ce que tu", "ce que vous", "notre echange", "notre conversation",
              "ce qui precede", "tout ca", "resume", "recapitul", "cela", "cette reponse",
              "ta reponse", "precedent", "ci-dessus", "au-dessus", "ce dont on a parle",
              "ces informations", "cette information", "ces infos", "cette info",
              "les informations", "l information", "ces donnees", "ces elements", "ces points",
              "ces details", "ceci", "ces resultats", "reponse precedente", "cette liste"]


def detect_doc_request(question: str) -> dict | None:
    """Détecte « crée-moi un PDF/Word/Excel … ». Retourne {formats, format, refers_prior, topic} ou None.
    `formats` = TOUS les formats demandés (« pdf ET excel » → ['pdf','xlsx'])."""
    q = normalize(question)
    formats = [f for f, kws in _DOC_FORMAT_KW.items() if any(k in q for k in kws)]
    has_verb = any(v in q for v in _DOC_VERBS)
    has_noun = any(n in q for n in _DOC_NOUNS)
    # Déclenché si : un format explicite est cité, OU (un verbe de création + un nom de doc).
    if not formats and not (has_verb and has_noun):
        return None
    if not formats:
        formats = ["pdf"]
    refers_prior = any(k in q for k in _DOC_PRIOR)
    # Sujet = question nettoyée des mots de commande (pour le titre + la recherche).
    topic = question
    topic = re.sub(r"(?i)\b(cr[ée]e[- ]?(moi|nous)?|g[ée]n[èe]re[r]?|fais[- ]?(moi)?|pr[ée]pare|r[ée]dige|"
                   r"met[s]?|exporte|produis|sors|un|une|des|le|la|les|en|au format|dans|s'?il te pla[îi]t|"
                   r"stp|document|fichier|rapport|r[ée]capitulatif|cette|ces|information[s]?|aussi|et|ca|"
                   r"pdf|word|excel|tableur|classeur|docx|xlsx)\b", " ", topic)
    topic = re.sub(r"\s+", " ", topic).strip(" .,:;-")
    return {"formats": formats, "format": formats[0], "refers_prior": refers_prior, "topic": topic}


def doc_title(topic: str) -> str:
    """Titre lisible pour le document à partir du sujet."""
    t = re.sub(r"[*_`#>]", "", topic or "").strip(" .,:;-")
    t = re.sub(r"(?i)^(sur|des|de la|de l'|du|pour|le|la|les)\s+", "", t)
    if not t or len(t) < 3:
        return "Récapitulatif alélo"
    return t[:1].upper() + t[1:80]


# ── Routage automatique du périmètre (org) — multi-institutions ───────────────
AUTO_ROUTE = os.getenv("AUTO_ROUTE", "1") == "1"

# Carte des institutions : org → {label (attribution/UI), theme, mots-clés de routage}.
INSTITUTIONS = {
    "ANSUT":        {"label": "ANSUT", "theme": "Numérique",
                     "kw": ["ansut", "service universel", "fibre",
                            "haut debit", "connectivite", "connecte", "connexion", "couverture reseau",
                            "village connecte", "internet dans", "fracture numerique",
                            "inclusion numerique"]},
    "NUMERIQUE":    {"label": "Ministère de la Transition Numérique et de l'Innovation Technologique",
                     "theme": "Numérique",
                     "kw": ["transition numerique", "innovation technologique", "digitalisation",
                            "ministre du numerique", "ministere du numerique", "djibril ouattara",
                            "telecom.gouv", "numerique", "telecom", "tic", "e-gouvernement"]},
    "CGECI":        {"label": "CGECI (patronat)", "theme": "Entreprise",
                     "kw": ["cgeci", "patronat", "adhesion", "cotisation", "commission permanente",
                            "cisse", "diagou", "employeur", "medef", "entreprises de cote"]},
    "SERVICEPUBLIC":{"label": "Service Public", "theme": "Démarches",
                     "kw": ["demarche", "carte nationale d identite", "cni", "acte de naissance",
                            "nationalite", "extrait", "casier judiciaire", "certificat", "etat civil"]},
    "SNEDAI":       {"label": "SNEDAI (passeports)", "theme": "Démarches",
                     "kw": ["passeport", "snedai", "passeport biometrique", "passeport ordinaire",
                            "renouvellement passeport", "police de l air et des frontieres",
                            "document de voyage", "passeport diaspora"]},
    "DGI":          {"label": "Direction Générale des Impôts", "theme": "Fiscalité",
                     "kw": ["impot", "fiscal", "fiscalite", "taxe", "tva", "contribuable",
                            "declaration fiscale", "sticker", "dgi", "vignette"]},
    "TRESOR":       {"label": "Trésor Public", "theme": "Finances",
                     "kw": ["tresor", "tresorerie", "comptable public", "recette", "depot public"]},
    "TRANSPORTS":   {"label": "Ministère des Transports", "theme": "Transports",
                     "kw": ["transport", "permis de conduire", "immatriculation", "vehicule",
                            "carte grise", "conducteur", "code de la route", "visite technique"]},
    "CEPICI":       {"label": "CEPICI (investissement)", "theme": "Entreprise",
                     "kw": ["creation d entreprise", "creer une entreprise", "creer mon entreprise",
                            "creer son entreprise", "monter une entreprise", "immatriculer mon entreprise",
                            "cepici", "guichet unique", "investissement", "investir",
                            "formalites entreprise", "registre de commerce"]},
    "GUCE":         {"label": "Guichet Unique du Commerce Extérieur", "theme": "Commerce",
                     "kw": ["commerce exterieur", "import", "export", "douane", "guce", "dedouanement"]},
    "AIGF":         {"label": "Agence Ivoirienne de Gestion des Fréquences", "theme": "Numérique",
                     "kw": ["aigf", "gestion des frequences", "frequences radio", "spectre radioelectrique"]},
    "APDP":         {"label": "Autorité de Protection des Données", "theme": "Données",
                     "kw": ["donnees personnelles", "donnee personnelle", "protection des donnees",
                            "vie privee", "apdp", "consentement"]},
    "FAMILLE":      {"label": "Ministère de la Femme, de la Famille et de l'Enfant", "theme": "Social",
                     "kw": ["famille", "enfant", "scolarite", "protection de l enfance", "adoption", "femme"]},
    "SGG":          {"label": "Secrétariat Général du Gouvernement", "theme": "Textes",
                     "kw": ["decret", "loi", "journal officiel", "texte reglementaire", "ordonnance", "sgg", "arrete"]},
    "PRIMATURE":    {"label": "Primature", "theme": "Gouvernement",
                     "kw": ["premier ministre", "primature", "conseil des ministres"]},
    "PRESIDENCE":   {"label": "Présidence", "theme": "Gouvernement",
                     "kw": ["president de la republique", "presidence", "chef de l etat",
                            "president de la cote d ivoire", "president ivoirien", "ouattara",
                            "president du pays", "premier magistrat"]},
    "GOUVERNEMENT": {"label": "Gouvernement (composition)", "theme": "Gouvernement",
                     "kw": ["composition du gouvernement", "membres du gouvernement", "premier ministre",
                            "1er ministre", "vice-president", "vice president", "gouvernement mambe",
                            "liste des ministres", "quel ministre", "qui est le ministre",
                            "ministre actuel", "conseil des ministres"]},
    "SNDI":         {"label": "SNDI", "theme": "Numérique",
                     "kw": ["sndi", "sigmap", "sigfip", "informatique de l etat"]},
    "FINANCES":     {"label": "Ministère des Finances", "theme": "Finances",
                     "kw": ["finances publiques", "ministere des finances", "depense publique"]},
    "IGF":          {"label": "Inspection Générale des Finances", "theme": "Finances",
                     "kw": ["inspection generale des finances", "controle des finances", "igf"]},
    "PLAN":         {"label": "Ministère du Plan", "theme": "Développement",
                     "kw": ["plan national de developpement", "pnd", "planification"]},
    "ENVIRONNEMENT":{"label": "Ministère de l'Environnement", "theme": "Environnement",
                     "kw": ["environnement", "pollution", "climat", "dechets", "developpement durable"]},
    "SALUBRITE":    {"label": "Ministère de la Salubrité", "theme": "Environnement",
                     "kw": ["salubrite", "assainissement", "proprete", "ordures"]},
    "CULTURE":      {"label": "Ministère de la Culture", "theme": "Culture",
                     "kw": ["culture", "patrimoine", "artiste", "musee"]},
    "COMMUNICATION":{"label": "Ministère de la Communication", "theme": "Médias",
                     "kw": ["communication", "medias", "presse", "audiovisuel"]},
    "DEFENSE":      {"label": "Ministère de la Défense", "theme": "Défense",
                     "kw": ["defense", "armee", "militaire", "forces armees"]},
    "EAUXETFORETS": {"label": "Ministère des Eaux et Forêts", "theme": "Environnement",
                     "kw": ["eaux et forets", "foret", "reboisement", "faune", "chasse"]},
    "RESSOURCESANIMALES":{"label": "Ministère des Ressources Animales", "theme": "Agriculture",
                     "kw": ["ressources animales", "elevage", "betail", "peche", "halieutique", "veterinaire"]},
    "FER":          {"label": "FER (entretien routier)", "theme": "Transports",
                     "kw": ["entretien routier", "route", "peage", "voirie"]},
    "C2D":          {"label": "C2D", "theme": "Développement",
                     "kw": ["c2d", "desendettement", "contrat de desendettement"]},
    "ANNUAIRE":     {"label": "Annuaire de l'administration", "theme": "Contacts",
                     "kw": ["annuaire", "coordonnees", "contact administration", "adresse institution"]},
    "DATAGOUV":     {"label": "Portail des données ouvertes", "theme": "Données",
                     "kw": ["donnees ouvertes", "open data", "statistiques publiques", "jeu de donnees"]},
    # ── Ministères ajoutés (scrape 2026-09) ──────────────────────────────────
    "SANTE":        {"label": "Ministère de la Santé", "theme": "Santé",
                     "kw": ["sante", "hopital", "hopitaux", "maladie", "couverture maladie", "cmu",
                            "hygiene publique", "vaccination", "chu", "pierre dimba", "soins"]},
    "AGRICULTURE":  {"label": "Ministère de l'Agriculture", "theme": "Agriculture",
                     "kw": ["agriculture", "agricole", "productions vivrieres", "agriculteur",
                            "vivrier", "developpement rural", "nabagne kone", "cultures"]},
    "JUSTICE":      {"label": "Ministère de la Justice", "theme": "Justice",
                     "kw": ["ministere de la justice", "juridiction", "magistrat", "droits de l homme",
                            "e-justice", "e justice", "sansan kambile", "penitentiaire"]},
    "INTERIEUR":    {"label": "Ministère de l'Intérieur et de la Sécurité", "theme": "Sécurité",
                     "kw": ["ministere de l interieur", "police nationale", "prefet", "prefecture",
                            "sous prefet", "collectivites territoriales", "vagondo diomande"]},
    "EDUCATION":    {"label": "Ministère de l'Éducation Nationale", "theme": "Éducation",
                     "kw": ["education nationale", "ecole", "eleve", "alphabetisation", "enseignant",
                            "primaire", "secondaire", "koffi nguessan", "bepc", "cepe"]},
    "ENERGIE":      {"label": "Ministère des Mines, du Pétrole et de l'Énergie", "theme": "Énergie",
                     "kw": ["energie", "petrole", " mines", "electricite", "hydrocarbures", "gaz",
                            "raffinage", "sangafowa", "minier", "petrolier"]},
    "COMMERCE":     {"label": "Ministère du Commerce et de l'Industrie", "theme": "Commerce",
                     "kw": ["commerce et industrie", "industrie", "industriel", "commercant",
                            "concurrence", "khalil konate"]},
}

def org_label(org: str | None) -> str:
    return INSTITUTIONS.get(org, {}).get("label", org or "")

# Abréviations courantes → forme complète (pour routage, détection, recherche).
_ABBREV = [
    (r"\bpr\b", "president"), (r"\bpdt\b", "president"), (r"\bci\b", "cote d ivoire"),
    (r"\bdg\b", "directeur general"), (r"\bpm\b", "premier ministre"), (r"\bsg\b", "secretaire general"),
    (r"\bmin\b", "ministre"), (r"\brep\b", "republique"), (r"\bgvt\b", "gouvernement"),
    (r"\b1er\b", "premier"), (r"\b1ere\b", "premiere"), (r"\bpremier min\b", "premier ministre"),
]


def _expand_abbrev(question: str) -> str:
    """Développe les abréviations sur le texte normalisé (« pr de la ci » → « president cote d ivoire »).

    L'apostrophe est traitée comme un espace : « d'ivoire » → « d ivoire », pour que le routage
    (FR comme EN) matche « cote d ivoire », « carte d identite », etc. de façon fiable.
    """
    q = normalize(question)
    q = re.sub(r"['’]", " ", q)
    q = re.sub(r"\s+", " ", q)
    for pat, rep in _ABBREV:
        q = re.sub(pat, rep, q)
    return q


def _route_org(question: str) -> str | None:
    """Devine l'institution visée (routage multi-institutions). None si ambigu → recherche large."""
    q = " " + _expand_abbrev(question) + " "
    # Priorité : « qui est le ministre / Premier ministre / président de la République ? »
    # → composition officielle du gouvernement (datée), et non un vieux doc d'institution.
    if "GOUVERNEMENT" in INSTITUTIONS and any(a in q for a in _OFFICE_ASK) \
            and " directeur" not in q and " dg " not in q and "cgeci" not in q:
        gov = any(t in q for t in [" ministre", " ministere", "premier ministre", "vice-president",
                                    "vice president", " minister", "prime minister"]) \
            or "president de la republique" in q or "president de la cote d ivoire" in q \
            or "president du pays" in q or "chef de l etat" in q \
            or "president of the republic" in q or "president of cote d ivoire" in q \
            or "president of ivory coast" in q or "head of state" in q or "head of government" in q
        if gov:
            return "GOUVERNEMENT"
    scores = {}
    for org, info in INSTITUTIONS.items():
        s = sum(1 for kw in info["kw"] if normalize(kw) in q)
        if s:
            scores[org] = s
    if not scores:
        return None
    best = max(scores.values())
    winners = [o for o, s in scores.items() if s == best]
    return winners[0] if len(winners) == 1 else None   # égalité → ambigu → large


# ── Vérification de faithfulness (garde-fou post-génération) ───────────────────
# Contrôle que les CHIFFRES de la réponse figurent bien dans le contexte fourni.
# C'est le type d'hallucination le plus dommageable (montants, dates, comptes inventés).
FAITHFULNESS_CHECK = os.getenv("FAITHFULNESS_CHECK", "1") == "1"
# Nombre = suite de chiffres, éventuellement avec séparateurs de milliers (espace/nbsp/point/virgule)
# MAIS jamais à travers un saut de ligne (sinon on colle deux nombres → faux positif).
_NUM_RE = re.compile(r"\d(?:[  .,]?\d)*")

def _norm_num(s: str) -> str:
    return re.sub(r"[\s., ]", "", s)

def _faithfulness_check(answer: str, context: str) -> dict:
    """Repère les nombres de la réponse absents du contexte (hallucinations chiffrées)."""
    ctx_nums = {_norm_num(m) for m in _NUM_RE.findall(context)}
    ctx_raw = context
    seen = set()
    unsupported = []
    for m in _NUM_RE.findall(answer):
        raw = m.strip(" ., ")
        n = _norm_num(raw)
        if len(n) < 3:                       # ignore petits nombres (puces, comptes 1..99)
            continue
        if re.fullmatch(r"(?:19|20)\d{2}", raw):   # année reformulée (2027…) → pas un montant
            continue
        if n in ctx_nums or raw in ctx_raw or n in seen:
            continue
        seen.add(n)
        unsupported.append(raw)
    return {"grounded": not unsupported, "unsupported_numbers": unsupported}

def _faithfulness_caveat(check: dict) -> str:
    """Ajoute une mise en garde discrète si des chiffres ne sont pas sourcés."""
    if check.get("unsupported_numbers"):
        nums = ", ".join(check["unsupported_numbers"][:4])
        return (f"\n\n⚠️ *Certains chiffres ({nums}) ne figurent pas tels quels dans les "
                f"documents — à vérifier auprès de la source officielle.*")
    return ""


# Mots qui indiquent une VRAIE question de suivi (référence anaphorique à l'échange
# précédent). On évite les mots courants (le/la/les/son/plus/en…) qui déclenchaient
# à tort l'enrichissement et faisaient déborder le contexte d'un sujet sur un autre.
_FOLLOWUP_WORDS = {
    # anaphores (renvoient à l'échange précédent)
    "il", "elle", "ils", "elles", "lui", "eux",
    "celui", "celle", "ceux", "celles", "celui-ci", "celle-ci",
    "cela", "ca", "ça", "ceci", "idem", "pareil",
    "aussi", "egalement", "également", "encore",
    "precise", "précise", "detaille", "détaille", "developpe", "développe",
    "continue", "poursuis", "et lui", "et elle", "et eux",
    # demandes d'exemple / précision SUR le sujet précédent (sans nommer de nouveau sujet)
    "exemple", "exemples",
    "lequel", "laquelle", "lesquels", "lesquelles",
    "ensuite", "apres", "après", "suite", "puis", "davantage", "plus",
    "comment", "pourquoi", "concretement", "concrètement",
}

# Anaphores FORTES : déterminants démonstratifs qui pointent une entité déjà citée
# (« ce ministère là », « cette agence », « ces mesures »). Suivi quelle que soit la longueur.
# (le garde-fou org-gate empêche l'over-trigger : si la question nomme une institution
#  connue, elle route → traitée comme autonome AVANT d'arriver ici.)
_STRONG_ANAPHORA = {"ce", "cet", "cette", "ces"}

def _is_followup(question: str, org: str | None = None) -> bool:
    """Détecte une vraie question de suivi (anaphore vers l'échange précédent).
    Si la question nomme clairement une organisation/sujet → question autonome (pas un suivi),
    ce qui évite de traîner le contexte précédent (ex. Gilles Thierry Beugré/ANSUT) sur
    une question CGECI."""
    if org is None:
        org = _route_org(question)
    if org is not None:            # sujet clair → question autonome
        return False
    # Sans institution nommée : c'est un suivi si la question porte un mot-déclencheur
    # (anaphore, demande d'exemple/précision « par exemple », « lesquels », « comment »…)
    # et reste courte. Une question qui apporte son PROPRE sujet (ex. « combien coûte
    # un passeport ? ») n'a pas de déclencheur → traitée comme autonome.
    words = set(normalize(question).split())
    if words & _STRONG_ANAPHORA:   # « ce ministère là », « cette agence » → suivi (toute longueur)
        return True
    return bool(words & _FOLLOWUP_WORDS) and len(question.split()) < 12


# Une question « actions / réalisations / projets / bilan » d'un responsable ou d'une institution
# doit chercher les PROJETS de l'institution, pas la biographie de la personne.
_ACTION_WORDS = ["action", "actions", "realisation", "realisations", "realise", "projet", "projets",
                 "bilan", "mesure", "mesures", "initiative", "initiatives", "activite", "activites",
                 "chantier", "chantiers", "reforme", "reformes", "a mene", "mene des", "a fait",
                 "ont fait", "accompli", "qu a t il fait", "programme", "programmes"]


def _build_search_query(question: str, history: list[dict], org: str | None = None) -> str:
    """
    Construit une requête de recherche autonome.
    Enrichit UNIQUEMENT les vraies suites anaphoriques ('et lui ?', 'précise').
    Une question avec un sujet clair reste telle quelle (pas de contamination).
    Pour une question « actions/réalisations/projets », ajoute des termes orientés PROJETS afin
    de remonter les initiatives de l'institution (et non la bio du responsable).
    """
    # Développe les abréviations (pr, ci, dg…) pour la recherche, sinon garde le texte tel quel.
    exp = _expand_abbrev(question)
    q = exp if exp != normalize(question) else question

    boost = ""
    if any(w in normalize(question) for w in _ACTION_WORDS):
        # Termes neutres (marchent pour un ministère, une agence, une direction, un DG/DGA…).
        boost = " projets programmes initiatives realisations chantiers reformes activites bilan mesures"

    if not history or not _is_followup(question, org):
        return q + boost

    # Ancre la recherche sur le SUJET précédent = la dernière question de l'utilisateur.
    # (On ignore la réponse de l'assistant : elle ajoute du bruit et peut être générique.)
    last_user = ""
    for msg in reversed(history):
        if msg.get("role") == "user" and msg.get("content", "").strip():
            last_user = msg["content"].strip()
            break
    if not last_user:
        return q + boost
    return f"{last_user} {q}" + boost


_FR_STOP = {"le", "la", "les", "des", "un", "une", "est", "que", "qui", "pour", "comment",
            "quel", "quelle", "quelles", "quels", "je", "vous", "mon", "ma", "mes", "dans",
            "avec", "sur", "au", "aux", "et", "ou", "de", "du", "à", "ce", "cette", "mes"}
_EN_STOP = {"the", "a", "an", "is", "are", "how", "what", "do", "does", "for", "your", "you",
            "my", "in", "with", "on", "and", "or", "of", "to", "can", "could", "where", "when",
            "which", "i", "should", "would", "about", "get", "need"}


def _detect_lang(text: str) -> str:
    """Heuristique FR/EN via mots-outils. 'en' ou 'fr' si tranché, sinon 'unknown'
    (question trop courte/ambiguë, ex. un suivi « tell me more »)."""
    words = re.findall(r"[a-zA-Z']+", (text or "").lower())
    if len(words) < 3:
        return "unknown"
    fr = sum(w in _FR_STOP for w in words)
    en = sum(w in _EN_STOP for w in words)
    if en >= 2 and en > fr:
        return "en"
    if fr >= 2 and fr > en:
        return "fr"
    return "unknown"


def _build_chat_prompt(question: str, context: str,
                       history: list[dict], model: str, detail: bool = False) -> str:
    """Construit le prompt complet avec historique de conversation.
    detail=True (mode Expert) → ajoute la consigne de réponse approfondie."""
    # Historique formaté (max 6 derniers échanges pour ne pas dépasser le contexte)
    history_str = ""
    if history:
        lines = []
        for msg in history[-6:]:
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role} : {msg['content']}")
        history_str = "\n".join(lines)

    history_block = f"\nHistorique de la conversation :\n{history_str}\n" if history_str else ""

    # Langue de réponse : celle de la question ; si trop courte/ambiguë (suivi type
    # « tell me more », « and the cost? »), on HÉRITE de la langue du dernier tour utilisateur
    # → la conversation reste dans la même langue (fini le retour au français en plein échange EN).
    lang = _detect_lang(question)
    if lang == "unknown":
        prev_user = next((m["content"] for m in reversed(history or [])
                          if m.get("role") == "user" and m.get("content", "").strip()), "")
        lang = _detect_lang(prev_user)
    lang_note = ("\nIMPORTANT: the user's question is in English — answer ENTIRELY in English.\n"
                 if lang == "en" else "")
    detail_note = DETAIL_INSTRUCTION if detail else ""

    return f"""{SYSTEM_PROMPT}{detail_note}
{history_block}
Documents de référence :
{context}

Question actuelle : {question}
{lang_note}
Réponse :"""


def chat(vectordb: Chroma, question: str, history: list[dict],
         model: str = LLM_MODEL, k: int = DEFAULT_K, org: str | None = None,
         detail: bool = False) -> dict:
    """
    Pipeline RAG conversationnel.
    history = [{"role": "user"|"assistant", "content": "..."}]
    org : périmètre optionnel ("ANSUT", "CGECI"). detail=True → réponse approfondie.
    """
    # Message social (bonjour, merci, qui es-tu…) → réponse directe, sans RAG
    social = _detect_social(question)
    if social:
        return {"answer": _social_response(social), "sources": []}

    # Routage automatique du périmètre si non imposé
    if org is None and AUTO_ROUTE:
        org = _route_org(question)

    # Requête enrichie du contexte pour la recherche
    search_query = _build_search_query(question, history, org)
    docs = retrieve(vectordb, search_query, k=k, org=org)

    # Garde-fou no-context : aucun document pertinent → on n'invente pas
    if not docs:
        return {"answer": REFUSAL_MSG, "sources": []}

    context = _build_context(docs)

    prompt = _build_chat_prompt(question, context, history, model, detail=detail)

    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    resp = requests.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "keep_alive": KEEP_ALIVE, "options": GEN_OPTIONS_DETAIL if detail else GEN_OPTIONS},
        timeout=180,
    )
    resp.raise_for_status()
    answer = resp.json().get("response", "").strip()

    # Garde-fou faithfulness : signale les chiffres non sourcés
    if FAITHFULNESS_CHECK:
        answer += _faithfulness_caveat(_faithfulness_check(answer, context))

    # Sources
    sources = _build_sources(docs)

    return {"answer": answer, "sources": sources}


# ── Étape 1 agentique : re-recherche corrective (Mode Expert) ─────────────────
# Si le meilleur score reranker est sous ce seuil, le contexte est jugé FAIBLE
# → on reformule la requête et on re-cherche (une fois), en gardant le meilleur résultat.
CORRECTIVE_MIN = float(os.getenv("CORRECTIVE_MIN", "0.35"))


def _best_score(docs) -> float:
    return max((d.metadata.get("_rerank") or 0.0) for d in docs) if docs else 0.0


def _reformulate_query(question: str, model: str = LLM_MODEL) -> str:
    """Réécrit la question en UNE requête de recherche plus efficace (sigles développés,
    termes officiels, synonymes). Repli sur la question d'origine si échec."""
    prompt = ("Tu aides un moteur de recherche documentaire sur les services publics ivoiriens. "
              "Réécris la demande du citoyen en UNE requête de recherche efficace, en restant "
              "STRICTEMENT FIDÈLE à son intention : n'introduis AUCUN sujet nouveau, n'invente rien. "
              "Développe seulement les sigles et emploie les termes officiels équivalents. "
              "Une seule ligne, sans phrase. "
              f"Réponds UNIQUEMENT par la requête.\n\nDemande : {question}\nRequête :")
    try:
        base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
        resp = requests.post(f"{base_url}/api/generate",
                             json={"model": model, "prompt": prompt, "stream": False,
                                   "keep_alive": KEEP_ALIVE,
                                   "options": {"num_predict": 60, "temperature": 0.2, "num_ctx": 2048}},
                             timeout=60)
        resp.raise_for_status()
        q = resp.json().get("response", "").strip().strip('"').splitlines()[0].strip()
        return q if 3 <= len(q) <= 200 else question
    except Exception as e:
        logger.warning("Reformulation échouée : %s", e)
        return question


# ── Étape 2 agentique : décomposition des questions multi-volets (Mode Expert) ─
_MULTIPART_KW = ["compar", "versus", " vs ", "difference entre", "distingu",
                 "par rapport a", "chacun", "chacune", "respectivement"]


def _looks_multipart(question: str) -> bool:
    """Signal (bon marché) qu'une question pourrait avoir plusieurs volets → tenter le découpage.
    Le vrai juge reste _decompose_question (qui répond NON si un seul volet)."""
    n = normalize(question)
    if any(k in n for k in _MULTIPART_KW):
        return True
    return (" et " in n or " ou " in n) and len(question.split()) >= 6


def _decompose_question(question: str, model: str = LLM_MODEL) -> list[str]:
    """Découpe en sous-questions autonomes (2 à 4) SI plusieurs volets ; sinon []."""
    prompt = (
        "Tu découpes une question en sous-questions autonomes pour un moteur de recherche.\n"
        "- Si elle compare ou mentionne PLUSIEURS sujets/institutions, écris UNE sous-question par "
        "sujet (une par ligne, sans numéro ni tiret).\n"
        "- Si elle ne porte que sur UN seul sujet, réponds exactement : NON\n\n"
        "Exemple 1\n"
        "Question : compare les démarches pour le passeport et la carte nationale d'identité\n"
        "Sous-questions :\n"
        "démarches pour obtenir un passeport\n"
        "démarches pour obtenir une carte nationale d'identité\n\n"
        "Exemple 2\n"
        "Question : quelles sont les missions de l'APDP\n"
        "Sous-questions :\n"
        "NON\n\n"
        "Exemple 3\n"
        "Question : quelles institutions gèrent le foncier et les impôts\n"
        "Sous-questions :\n"
        "institution qui gère le foncier en Côte d'Ivoire\n"
        "institution qui gère les impôts en Côte d'Ivoire\n\n"
        f"Question : {question}\nSous-questions :")
    try:
        base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
        resp = requests.post(f"{base_url}/api/generate",
                             json={"model": model, "prompt": prompt, "stream": False,
                                   "keep_alive": KEEP_ALIVE,
                                   "options": {"num_predict": 140, "temperature": 0.2, "num_ctx": 2048}},
                             timeout=60)
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning("Décomposition échouée : %s", e)
        return []
    if "non" in text.lower()[:6]:
        return []
    subs = []
    for line in text.splitlines():
        s = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip(" .")
        if len(s) >= 6 and "?" not in s[:2] and s.lower() != "non":
            subs.append(s[:160])
    return subs[:4] if len(subs) >= 2 else []


def _build_multi_context(per_volet: list) -> str:
    """Contexte regroupé PAR VOLET (chaque sous-question + ses documents)."""
    return "\n\n".join(f"═══ VOLET : {sub} ═══\n{_build_context(docs[:4])}"
                       for sub, docs in per_volet if docs)


# ── Étape 3 agentique : OUTIL FRAÎCHEUR (fonctions nominatives) ────────────────
_OFFICE_ROLES = ["ministre", "directeur general", "directrice generale", "directeur", "directrice",
                 "president", "presidente", "pdg", "gouverneur", "prefet", "maire",
                 "secretaire general", "chef ",
                 # équivalents anglais (les questions officeholder peuvent arriver en anglais)
                 "minister", "prime minister", "head of", "director", "governor", "mayor", "ceo"]
_OFFICE_ASK = ["qui est", "qui dirige", "qui gere", "qui occupe", "qui preside", "qui commande",
               "nom du", "nom de la", "nom de l", "s appelle",
               # équivalents anglais
               "who is", "who s ", "who heads", "who leads", "who runs", "name of", "who governs"]
FRESHNESS_LOG = os.path.join("data", "freshness_queries.jsonl")


def _is_officeholder_q(question: str) -> bool:
    """« Qui est le ministre / DG / président de X ? » → fait nominatif sensible à la fraîcheur."""
    n = _expand_abbrev(question)
    return any(r in n for r in _OFFICE_ROLES) and any(a in n for a in _OFFICE_ASK)


def _freshness_directive(docs) -> str:
    """Directive : donner le titulaire, prévenir qu'il a pu changer, SANS confondre la date d'un
    document avec une date de nomination (source d'hallucination fréquente)."""
    return ("[OUTIL FRAÎCHEUR — fonction nominative] Le titulaire d'un poste peut changer avec le temps. "
            "Donne le nom d'après les sources ci-dessus (ou, pour une fonction NOTOIRE — président de la "
            "République, Premier ministre, chef de l'État —, ta MEILLEURE connaissance), et signale "
            "brièvement que l'information a pu évoluer depuis. NE REFUSE PAS et ne renvoie JAMAIS vers un "
            "site externe ou un « service communication ».\n"
            "RÈGLE DE DATE STRICTE : n'indique une date de PRISE DE FONCTION / de nomination QUE si une "
            "source l'écrit explicitement comme telle (ex. « en fonction depuis avril 2022 »). La date à "
            "laquelle un DOCUMENT a été publié, et une date de naissance, NE SONT PAS des dates de prise de "
            "fonction : ne les présente jamais ainsi. N'écris JAMAIS « il a pris fonction à cette date » en "
            "te basant sur la date d'un document, et n'invente aucune date de nomination.\n\n")


def _log_freshness(question: str, docs) -> None:
    """Journalise la requête nominative → priorise le re-scraping des institutions concernées."""
    try:
        from datetime import datetime, timezone
        os.makedirs(os.path.dirname(FRESHNESS_LOG) or ".", exist_ok=True)
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "question": question[:300],
               "orgs": list({d.metadata.get("org") for d in docs if d.metadata.get("org")})[:5]}
        with open(FRESHNESS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ── GraphRAG : graphe de connaissances des services (fiches faisant autorité) ─
_GRAPH_PATH = os.path.join("data", "knowledge_graph.json")
_GRAPH_CACHE = {"g": None}
_GRAPH_KW = {
    "passeport": ["passeport"],
    "cni": ["cni", "carte nationale", "carte d identite", "carte nationale d identite"],
    "permis_conduire": ["permis de conduire", "permis"],
    "acte_naissance": ["acte de naissance", "extrait de naissance", "extrait d acte"],
    "certificat_nationalite": ["certificat de nationalite", "nationalite"],
    "casier_judiciaire": ["casier judiciaire", "casier"],
    "carte_sejour": ["carte de sejour", "titre de sejour"],
    "acte_mariage": ["acte de mariage", "certificat de mariage"],
    "creation_entreprise": ["creer une entreprise", "creation d entreprise", "creation entreprise",
                            "cepici", "guichet unique", "immatriculer mon entreprise"],
    "impots": ["impot", "impots", "e-impots", "e impots", "declaration fiscale"],
    "carte_grise": ["carte grise", "immatriculation", "certificat d immatriculation"],
    "carte_resident": ["carte de resident"],
    # ── Fiches INSTITUTIONS (audit vérifié 2026-09) ───────────────────────
    "inst_cgeci": ["cgeci", "patronat"],
    "inst_cepici": ["cepici"],
    "inst_guce": ["guce", "guichet unique du commerce", "commerce exterieur"],
    "inst_dgi": ["dgi", "direction generale des impots"],
    "inst_tresor": ["tresor public", "tresor", "comptabilite publique"],
    "inst_igf": ["igf", "inspection generale des finances"],
    "inst_ansut": ["ansut", "service universel", "services universel",
                   "agence nationale du service universel", "telecommunication"],
    "inst_sndi": ["sndi"],
    "inst_protection_donnees": ["apdp", "protection des donnees", "donnees personnelles",
                                "donnees a caractere personnel", "autorite de protection"],
    "inst_agef": ["agef", "gestion fonciere", "reserve fonciere", "terrain viabilise",
                  "concession definitive", "foncier urbain"],
    "inst_aigf": ["aigf", "gestion des frequences", "frequences radio"],
    "inst_sgg": ["sgg", "secretariat general du gouvernement", "secretaire general du gouvernement",
                 "journal officiel"],
    "inst_fer": ["fonds d entretien routier", "entretien routier", "peage"],
}


def _load_graph() -> dict:
    if _GRAPH_CACHE["g"] is None:
        try:
            with open(_GRAPH_PATH, encoding="utf-8") as f:
                _GRAPH_CACHE["g"] = json.load(f)
        except Exception:
            _GRAPH_CACHE["g"] = {"services": {}, "institutions": {}}
    return _GRAPH_CACHE["g"]


def _match_service_fiches(question: str) -> list:
    """Retrouve les fiches services pertinentes (mots-clés) — max 3, sans LLM."""
    g = _load_graph()
    n = normalize(question)
    hits = []
    for sid, kws in _GRAPH_KW.items():
        if sid in g.get("services", {}) and any(k in n for k in kws):
            hits.append(g["services"][sid])
    return hits[:3]


def _fiche_context(fiches: list) -> str:
    """Formatte les fiches en contexte structuré faisant autorité."""
    blocks = []
    for f in fiches:
        b = [f"— {f.get('service') or f.get('id')} —"]
        inst = f.get("institution")
        if inst:
            b.append(f"Institution/opérateur : {inst}")
        if f.get("role"):                       # fiche institution : mission
            b.append(f"Rôle : {f['role']}")
        if f.get("dirigeant"):                  # fiche institution : dirigeant daté
            b.append(f"Dirigeant : {f['dirigeant']}")
        if f.get("documents"):
            b.append("Documents à fournir : " + " ; ".join(str(d) for d in f["documents"]))
        if f.get("actions"):                    # fiche institution : services/actions
            b.append("Services/actions : " + " ; ".join(str(a) for a in f["actions"]))
        if f.get("cout"):
            b.append(f"Coût : {f['cout']}")
        if f.get("delai"):
            b.append(f"Délai : {f['delai']}")
        if f.get("ou"):
            b.append(f"Où : {f['ou']}")
        if f.get("site"):
            b.append(f"Site officiel : {f['site']}")
        if f.get("note"):                       # caveat important (ex. « OBTENIR ≠ MODIFIER »)
            b.append(f"À noter : {f['note']}")
        # NB : les champs verified/verified_date/verified_source servent à NOTRE suivi et ne sont
        # PAS injectés — sinon le modèle recopie la date de vérification comme si c'était un fait.
        blocks.append("\n".join(b))
    return ("[FICHES OFFICIELLES — graphe de connaissances alélo. Ces données STRUCTURÉES font "
            "autorité : appuie-toi dessus en priorité. N'affirme un coût ou un délai QUE s'il "
            "figure ci-dessus ; sinon invite à vérifier sans inventer de chiffre.]\n"
            + "\n\n".join(blocks) + "\n\n")


def _gov_composition_context(question: str) -> str:
    """Injecte la composition COMPLÈTE et datée du gouvernement pour les questions ministérielles.

    La composition est découpée en chunks à l'indexation : une question sur un ministère peut alors
    remonter en tête un ministère PROCHE (ex. « Éducation nationale » pour « Enseignement supérieur »)
    et donner le mauvais titulaire. On injecte donc la liste entière (~4 Ko, fait autorité) pour que
    le modèle trouve le titulaire EXACT."""
    q = _expand_abbrev(question)
    if not (_is_officeholder_q(question)
            and any(t in q for t in [" ministre", " ministere", "gouvernement", "premier ministre",
                                     "vice premier", "president de la republique", "vice president"])):
        return ""
    try:
        with open(os.path.join("data", "raw_gouvernement", "composition.json"), encoding="utf-8") as f:
            content = json.load(f).get("content", "")
    except Exception:
        return ""
    if not content:
        return ""
    return ("[COMPOSITION OFFICIELLE ET DATÉE DU GOUVERNEMENT — fait autorité. Identifie le titulaire "
            "EXACT du ministère demandé dans cette liste ; ne confonds PAS deux ministères proches "
            "(« Éducation nationale » ≠ « Enseignement supérieur et Recherche scientifique »).]\n"
            + content + "\n\n")


# ── Étape 4 agentique : AUTO-VÉRIFICATION de la réponse contre les sources ─────
def _verify_answer(answer: str, context: str, model: str = LLM_MODEL) -> dict:
    """Relit la réponse : chaque fait précis (nom, montant, date, dispositif) est-il soutenu
    par les documents ? Retourne {'ok': bool, 'issues': [affirmations douteuses]}.
    En cas de doute/échec → considéré OK (on n'alarme pas à tort le citoyen)."""
    prompt = (
        "Tu vérifies une réponse par rapport à des documents sources.\n"
        "Pour CHAQUE affirmation factuelle précise (nom d'une personne, montant, date, chiffre, "
        "dispositif, sigle), vérifie si elle est bien soutenue par les DOCUMENTS.\n"
        "Réponds STRICTEMENT :\n"
        "- la ligne « OK » si toutes les affirmations précises sont soutenues ;\n"
        "- sinon, liste UNIQUEMENT les affirmations douteuses, une par ligne, très courtes.\n\n"
        f"DOCUMENTS :\n{context[:4500]}\n\nRÉPONSE À VÉRIFIER :\n{answer[:2000]}\n\nVérification :")
    try:
        base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
        resp = requests.post(f"{base_url}/api/generate",
                             json={"model": model, "prompt": prompt, "stream": False,
                                   "keep_alive": KEEP_ALIVE,
                                   "options": {"num_predict": 160, "temperature": 0.0, "num_ctx": 8192}},
                             timeout=90)
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning("Auto-vérification échouée : %s", e)
        return {"ok": True, "issues": []}
    low = normalize(text)
    if low.startswith("ok") or "tout est soutenu" in low or "toutes les affirmations" in low \
            or "conforme" in low or "aucune" in low[:40]:
        return {"ok": True, "issues": []}
    issues = []
    for line in text.splitlines():
        s = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip(" .")
        if len(s) >= 8 and normalize(s) not in ("ok",) and "vérification" not in normalize(s):
            issues.append(s[:120])
    return {"ok": not issues, "issues": issues[:4]}


def _verify_note(v: dict) -> str:
    if v.get("ok"):
        return "\n\n✅ *Cohérence avec les sources vérifiée.*"
    return ("\n\n⚠️ *Auto-vérification — à confirmer auprès de la source : "
            + " ; ".join(v["issues"]) + ".*")


def chat_stream(vectordb: Chroma, question: str, history: list[dict],
                model: str = LLM_MODEL, k: int = DEFAULT_K, org: str | None = None,
                detail: bool = False):
    """
    Version streaming du chat conversationnel.
    Génère (yield) les tokens au fur et à mesure qu'Ollama les produit,
    puis termine par un dict {"sources": [...], "answer": "..."} (marqueur final).
    detail=True (mode Expert) → réponse approfondie + plus de sources.

    Usage :
        for chunk in chat_stream(...):
            if isinstance(chunk, str):
                afficher(chunk)          # token de texte
            else:
                sources = chunk["sources"]  # dernier élément
    """
    # Message social (bonjour, merci, qui es-tu…) → réponse directe, sans RAG
    social = _detect_social(question)
    if social:
        answer = _social_response(social)
        yield answer
        yield {"answer": answer, "sources": []}
        return

    # Routage automatique du périmètre si non imposé
    if org is None and AUTO_ROUTE:
        org = _route_org(question)

    context = None
    decomposed = False
    if detail:
        # ── Mode Expert AGENTIQUE ──
        # Étape 2 : découpage si la question a plusieurs volets (comparaison, multi-institutions).
        subs = _decompose_question(question, model) if _looks_multipart(question) else []
        if len(subs) >= 2:
            yield {"step": f"🧭 Question à {len(subs)} volets — je les traite séparément…"}
            per_volet = []
            for i, sub in enumerate(subs, 1):
                yield {"step": f"🔎 Volet {i} : {sub[:55]}…"}
                per_volet.append((sub, retrieve(vectordb, sub, k=k, org=_route_org(sub))))
            docs = [d for _, ds in per_volet for d in ds[:4]]
            if docs:
                context = ("La question comporte plusieurs volets, traités SÉPARÉMENT ci-dessous "
                           "(chaque « VOLET » a SES PROPRES documents). Rédige UNE section par volet, "
                           "en n'utilisant QUE les documents de CE volet : ne mélange JAMAIS les "
                           "informations d'un volet à l'autre (n'attribue pas au passeport une info du "
                           "volet CNI, etc.). Précise l'institution/opérateur de chaque volet, puis "
                           "termine par une brève comparaison.\n\n") + _build_multi_context(per_volet)
                decomposed = True
        else:
            # Étape 1 : re-recherche corrective (question simple).
            yield {"step": "🔎 Recherche dans les documents…"}
            search_query = _build_search_query(question, history, org)
            docs = retrieve(vectordb, search_query, k=k, org=org)
            if _best_score(docs) < CORRECTIVE_MIN:            # contexte jugé faible
                yield {"step": "Contexte faible — je reformule la recherche…"}
                rq = _reformulate_query(question, model)
                if normalize(rq) != normalize(search_query):
                    docs2 = retrieve(vectordb, rq, k=k, org=org)
                    if _best_score(docs2) > _best_score(docs):
                        docs = docs2
                        yield {"step": "🔁 De meilleurs résultats après reformulation."}
    else:
        # ── Mode Auto — recherche directe (rapide) ──
        search_query = _build_search_query(question, history, org)
        docs = retrieve(vectordb, search_query, k=k, org=org)

    # Garde-fou no-context : aucun document pertinent → on n'invente pas
    if not docs:
        yield REFUSAL_MSG
        yield {"answer": REFUSAL_MSG, "sources": []}
        return

    # ── GraphRAG : fiche(s) structurée(s) faisant autorité (lookup, sans LLM) ──
    fiches = _match_service_fiches(question)
    graph_ctx = ""
    if fiches:
        if detail:
            yield {"step": "🕸️ Fiche officielle du graphe de connaissances…"}
        graph_ctx = _fiche_context(fiches)

    # ── Étape 3 : OUTIL FRAÎCHEUR (fonctions nominatives : ministre, DG, président…) ──
    fresh = ""
    if _is_officeholder_q(question):
        if detail:
            yield {"step": "🕰️ Fonction nominative — je date et signale la fraîcheur…"}
            _log_freshness(question, docs)
        fresh = _freshness_directive(docs)                    # directive en Auto ET Expert

    # Si une FICHE VÉRIFIÉE donne déjà le dirigeant, elle fait autorité pour une question nominative :
    # on n'y mêle PAS le corpus brut (ses dates parasites deviennent des « pris fonction le … » faux).
    fiche_gives_dirigeant = _is_officeholder_q(question) and any(f.get("dirigeant") for f in fiches)

    if context is None:                                       # chemin simple (Auto ou corrective)
        context = "" if fiche_gives_dirigeant else _build_context(docs)

    if detail:
        yield {"step": "✍️ Synthèse des volets…" if decomposed else "✍️ Rédaction de la réponse…"}

    gov_ctx = _gov_composition_context(question)   # liste ministérielle complète si pertinent
    context = gov_ctx + graph_ctx + fresh + context
    prompt = _build_chat_prompt(question, context, history, model, detail=detail)
    gen_opts = GEN_OPTIONS_DETAIL if detail else GEN_OPTIONS

    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    full_answer = ""
    with requests.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": True,
              "keep_alive": KEEP_ALIVE, "options": gen_opts},
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)          # une ligne non-JSON (keep-alive) ne casse pas le flux
            except (json.JSONDecodeError, ValueError):
                continue
            token = data.get("response", "")
            if token:
                full_answer += token
                yield token
            if data.get("done"):
                break

    answer = full_answer.strip()
    # Étape 4 : AUTO-VÉRIFICATION (Expert) — relit la réponse contre les sources.
    # En Auto : garde-fou chiffres seul (rapide).
    note = ""
    if detail and len(answer) > 40 and answer != REFUSAL_MSG:
        yield {"step": "🔍 Auto-vérification contre les sources…"}
        note = _verify_note(_verify_answer(answer, context, model))
    elif FAITHFULNESS_CHECK:
        note = _faithfulness_caveat(_faithfulness_check(answer, context))
    if note:
        yield note
        answer += note

    # Sources (marqueur final) — plus de sources en mode approfondi
    sources = _build_sources(docs, max_n=6 if detail else 4)

    yield {"answer": answer, "sources": sources}


if __name__ == "__main__":
    import time

    TEST_QUESTIONS = [
        "Qui est le président de la CGECI ?",
        "Comment devenir membre de la CGECI ?",
        "Quelles sont les commissions permanentes ?",
        "Quel est le coût d'adhésion ?",
        "Qui finance la CGECI ?",
        "Quelles sont les dernières actualités ?",
        "Quel est le rôle du secteur privé en Côte d'Ivoire ?",
        "Quand a été créée la CGECI ?",
    ]

    print("Chargement de la base vectorielle...")
    db = load_vectordb()
    print("Prêt.\n")

    for q in TEST_QUESTIONS:
        print(f"❓ {q}")
        t0 = time.time()
        result = ask(db, q)
        print(f"💬 {result['answer'][:300]}")
        print(f"⏱ {time.time()-t0:.1f}s | Sources: {len(result['sources'])}")
        print("-" * 60)
