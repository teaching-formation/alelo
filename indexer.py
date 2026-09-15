"""
indexer.py — Charge les pages scrapées, les découpe en chunks,
et les indexe dans ChromaDB avec des embeddings Ollama.
"""

import json
import logging
import os
import re
from pathlib import Path
from collections import Counter

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

from urllib.parse import urlparse

RAW_DIR = Path(os.getenv("INDEX_RAW_DIR", "data/raw"))
CHROMA_DIR = Path("data/chroma_db")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "cgeci_docs"
CHUNK_SIZE = 800          # ~130-160 mots : garde ensemble montant/date et ce qu'ils qualifient
CHUNK_OVERLAP = 120
MAX_PAGE_WORDS = 12000    # au-delà = page-dump (listings/PDF multi-sujets) → ignorée
EMBED_MODEL = "bge-m3"    # multilingue, fort en français, MÊME famille que le reranker BGE-M3
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Vocabulaire contrôlé (métadonnées scalables) ───────────────────────────────
# org : organisation source (valeurs figées → filtres fiables, pas de "ansut" vs "ANSUT")
ALLOWED_ORGS = {"ANSUT", "CGECI", "SERVICEPUBLIC", "PRIMATURE", "SNDI",
                "APDP", "PRESIDENCE", "AIGF",
                # Vague 2 : institutions gouv.ci découvertes pendant le crawl
                "ANNUAIRE", "DGI", "SGG", "FAMILLE", "C2D", "IGF",
                # Vague 3 : ministères et agences gouv.ci
                "FINANCES", "TRANSPORTS", "ENSEIGNEMENT", "EMPLOI", "ENVIRONNEMENT",
                "SALUBRITE", "CULTURE", "PLAN", "COMMUNICATION", "DEFENSE",
                "EAUXETFORETS", "RESSOURCESANIMALES", "CEPICI", "GUCE", "PARTICIPATION",
                "MODERNISATION", "EADMIN", "TRESOR", "BUDGET", "CICG", "FER", "DATAGOUV",
                # Ministère du Numérique (telecom.gouv.ci) — distinct de l'agence ANSUT
                "NUMERIQUE",
                # SNEDAI — opérateur officiel des passeports (comble le trou passeport)
                "SNEDAI",
                # Composition officielle du gouvernement (président, PM, tous les ministres, datée)
                "GOUVERNEMENT",
                # Vague 4 : ministères manquants (scrape Playwright 2026-09)
                "SANTE", "AGRICULTURE", "JUSTICE", "INTERIEUR", "EDUCATION", "ENERGIE", "COMMERCE"}
INDEX_ORG = os.getenv("INDEX_ORG", "").strip().upper()
if INDEX_ORG and INDEX_ORG not in ALLOWED_ORGS:
    raise SystemExit(f"INDEX_ORG='{INDEX_ORG}' invalide. Autorisés : {sorted(ALLOWED_ORGS)}")

INDEX_RESET = os.getenv("INDEX_RESET", "0") == "1"   # vide la collection avant d'indexer

# doc_type : type de document (valeurs figées)
ALLOWED_DOC_TYPES = {
    "presentation", "service", "actualite", "reglementation",
    "formulaire", "publication", "document", "page", "presse",
}


def derive_doc_type(url: str, title: str, raw_type: str) -> str:
    """Dérive un doc_type normalisé (vocabulaire contrôlé) depuis l'URL/type."""
    u = (url or "").lower()
    if raw_type == "html_external":
        return "presse"
    if raw_type == "pdf":
        if any(k in u for k in ["formulaire", "fiche"]):
            return "formulaire"
        if any(k in u for k in ["reglement", "statut", "texte", "loi", "decret",
                                "arrete", "annexe-fiscale", "circulaire", "juridique"]):
            return "reglementation"
        if any(k in u for k in ["bulletin", "lettre", "booklet", "plaquette",
                                "annuaire", "ldp", "veille", "rapport"]):
            return "publication"
        return "document"
    # HTML interne
    if any(k in u for k in ["actualite", "/news", "/article", "/agenda",
                            "communique", "on-parle", "phototheque", "ansut-tv", "presse"]):
        return "actualite"
    if any(k in u for k in ["texte-reglementaire", "reglement", "statut", "juridique", "fiscal"]):
        return "reglementation"
    if any(k in u for k in ["formulaire", "adhesion", "fiche"]):
        return "formulaire"
    if any(k in u for k in ["service", "infrastructure", "axe", "feuille-de-route",
                            "vulgarisation", "patrimoine", "universel"]):
        return "service"
    if any(k in u for k in ["qui-sommes", "equipe", "mot-du", "mission", "cooperation",
                            "partenariat", "recrutement", "faq", "a-propos", "sommes-nous"]):
        return "presentation"
    return "page"


# ── Nettoyage de contenu (anti-bruit / anti-hallucination) ────────────────────
# Le scrape capture du boilerplate (newsletter, nav, formulaires) qui pollue le RAG.
_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")           # images markdown
_EMPTY_LINK_RE = re.compile(r"\[\s*\]\([^)]*\)")         # liens vides []()
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")          # [texte](url) → texte

_BOILERPLATE_MARKERS = [
    "abonnez-vous", "newsletter", "entrez votre adresse", "veuillez laisser ce champ",
    "verifiez votre boite", "vérifiez votre boîte", "indésirables", "indesirables",
    "confirmer votre abonnement", "ansut news", "champ vide", "laissez ce champ",
    "suivez-nous", "tous droits réservés", "mentions légales", "politique de confidentialité",
    "politique de cookies", "gestion des cookies", "© ansut", "© cgeci", "retour en haut",
    "partager sur", "lire la suite", "read more",
    # ── Étendu après audit gouv.ci (réseaux sociaux, cookies, footers, contacts) ──
    "reseaux sociaux", "réseaux sociaux", "nous suivre", "retrouvez-nous", "retrouvez nous",
    "saisissez votre e-mail", "saisissez votre email", "votre adresse e-mail", "votre e-mail",
    "gerer le consentement", "gérer le consentement", "preferences", "préférences",
    "mesure d'audience", "mesure d’audience", "autoriser la mesure", "cookie",
    "conception et realisation", "conception et réalisation", "conception & realisation",
    "a ete enregistree", "a été enregistrée", "votre requete", "votre réquête",
    "plan du site", "nous contacter", "copyright", "tous les droits", "flux rss",
    # ── En-têtes / navigation résiduels (audit v2) ──
    "menu principal", "formulaire de recherche", "rechercher fermer", "aller au contenu",
    "vous etes ici", "vous êtes ici", "fil d'ariane", "fil d’ariane", "skip to content",
    "foire aux questions", "menu de navigation", "barre de recherche",
]

# Détecte une ligne de menu mono-ligne (beaucoup de puces/segments courts) ou un fil d'Ariane
def _is_menu_line(l: str) -> bool:
    if l.count("›") >= 1:                       # fil d'Ariane (Accueil › … › …)
        return True
    segs = [s.strip() for s in re.split(r"\s\*\s|\s›\s|\s\|\s|\s•\s", l) if s.strip()]
    if len(segs) >= 5:                          # ligne éclatée en 5+ segments
        avg = sum(len(s.split()) for s in segs) / len(segs)
        if avg <= 3:                            # …courts → c'est un menu, pas du texte
            return True
    return False

# Lignes de contact pures (tél, e-mail, adresse) — bruit répétitif dans les footers
_CONTACT_RE = re.compile(
    r"^\s*(\*\s*)?((\+?\d[\d\s().-]{6,})|(tél|tel|fax|email|e-mail|adresse|bp)\s*[:.]|"
    r"[\w.+-]+@[\w.-]+\.\w+)", re.I)


def clean_content(text: str) -> str:
    """Retire images/liens vides et lignes de boilerplate (newsletter, nav, cookies,
    réseaux sociaux, contacts, menus finissant par « »)."""
    text = _IMG_RE.sub("", text)
    text = _EMPTY_LINK_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)          # garde le texte du lien, jette l'URL
    out = []
    for line in text.split("\n"):
        l = line.strip()
        if len(l) < 3:
            continue
        low = l.lower()
        if any(m in low for m in _BOILERPLATE_MARKERS):
            continue
        if l.startswith("![") or l.startswith("[]("):
            continue
        # Item de menu : ligne courte finissant par « » (fil de navigation)
        if l.rstrip().endswith("»") and len(l.split()) <= 5:
            continue
        # Menu mono-ligne (puces * en excès) ou fil d'Ariane (›)
        if _is_menu_line(l):
            continue
        # Ligne de contact pure (téléphone / e-mail / adresse)
        if _CONTACT_RE.match(l):
            continue
        out.append(l)
    return "\n".join(out)


# ── Filtre de langue : garder le français, écarter l'anglais (ex. Présidence bilingue) ──
_EN_MARKERS = [" the ", " and ", " of ", " for ", " with ", " is ", " are ", " to the ", " this ", " that "]
_FR_MARKERS = [" le ", " la ", " les ", " et ", " des ", " pour ", " est ", " une ", " dans ", " sur "]

def is_english(text: str) -> bool:
    """Heuristique simple : la page est-elle majoritairement en anglais ?"""
    t = " " + text.lower()[:2500] + " "
    en = sum(t.count(w) for w in _EN_MARKERS)
    fr = sum(t.count(w) for w in _FR_MARKERS)
    return en > fr and en >= 4


_BULLET = ("*", "-", "#", "•", "▪", "►", "+", "▸")

def _looks_like_nav(chunk: str) -> bool:
    """Détecte un chunk surtout composé de menu/navigation (lignes très courtes ou
    puces courtes de menu, fils d'Ariane) plutôt que du contenu rédigé. Les vraies
    listes de contenu (puces = phrases longues) sont conservées."""
    lines = [l for l in chunk.split("\n") if l.strip()]
    if len(lines) >= 4:
        navish = 0
        for l in lines:
            w = l.split()
            if len(w) <= 3:                                   # ligne ultra-courte
                navish += 1
            elif w and w[0] in _BULLET and len(w) <= 6:       # puce courte = label de menu
                navish += 1
        if navish / len(lines) > 0.6:
            return True
    # Fil d'Ariane / séparateurs de navigation en excès
    if chunk.count("»") >= 3 or chunk.count("›") >= 2 or chunk.count(" | ") >= 4:
        return True
    # Beaucoup de puces courtes sur peu de lignes = liste de menu mono-bloc
    if chunk.count(" * ") >= 6 and len(lines) <= 3:
        return True
    return False


_DATE_RE = re.compile(r"/(20\d\d)/(\d{2})/")

def derive_date(url: str, existing: str) -> str:
    """Garde la date existante, sinon tente d'extraire AAAA-MM depuis l'URL."""
    if existing:
        return existing
    m = _DATE_RE.search(url or "")
    return f"{m.group(1)}-{m.group(2)}" if m else ""

# ── Règles de catégorisation ──────────────────────────────────────────────────
CATEGORY_RULES = [
    ("gouvernance", ["bio", "president", "conseil-administration",
                     "gouvernance", "election", "apf", "fopao", "dirigeant"]),
    ("adhesion",    ["adhesion", "adhesion", "membre", "cotisation",
                     "fiche-adhesion", "fiche-engagement", "rejoindre"]),
    ("reglementation", ["reglement-interieur", "statuts", "circulaire",
                        "juridique", "fiscal", "annexe-fiscale"]),
    ("commissions", ["commission", "thematique", "permanente", "energie",
                     "numerique", "pme", "douane", "emploi", "rse"]),
    ("actualites",  ["news", "actualite", "category", "communique",
                     "seminaire", "forum", "conference", "partenariat"]),
    ("publications",["lettre-patronat", "ldp", "bulletin", "veille",
                     "booklet", "plaquette", "annuaire", "publication"]),
    ("economie",    ["economie", "financement", "investissement", "swedfund",
                     "eurobond", "budget", "fiscalite", "secteur-prive"]),
    ("covid",       ["covid", "coronavirus", "pandemie", "2020"]),
]


def categorize(url: str, title: str, content: str) -> str:
    """Détermine la catégorie d'un document à partir de son URL et titre."""
    text = (url + " " + title).lower()
    for category, keywords in CATEGORY_RULES:
        if any(kw in text for kw in keywords):
            return category
    # Fallback : cherche dans les premiers mots du contenu
    snippet = content[:200].lower()
    for category, keywords in CATEGORY_RULES:
        if any(kw in snippet for kw in keywords):
            return category
    return "general"


def load_raw_pages() -> list[dict]:
    """Charge tous les fichiers JSON du dossier data/raw/."""
    pages = []
    for path in RAW_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("content") and len(data["content"]) > 100:
                pages.append(data)
        except Exception as e:
            logger.warning(f"Erreur lecture {path.name}: {e}")
    logger.info(f"{len(pages)} pages chargées depuis {RAW_DIR}/")
    return pages


def chunk_pages(pages: list[dict]) -> tuple[list[str], list[dict]]:
    """
    Découpe chaque page en chunks et retourne (texts, metadatas).
    Les métadonnées conservent url, titre et date pour les citations.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_texts = []
    all_metas = []
    seen_keys: set[str] = set()      # dédup exact (nav/footer répétés)
    n_dup = n_short = n_nav = 0
    n_dump = n_en = n_leak = 0

    index_domain = os.getenv("INDEX_DOMAIN", "").strip().lower().replace("www.", "")
    # "auto" → domaine majoritaire des pages (fiable, évite les erreurs de saisie)
    if index_domain == "auto":
        doms = Counter(urlparse(p.get("url", "")).netloc.replace("www.", "").lower()
                       for p in pages if p.get("url"))
        index_domain = doms.most_common(1)[0][0] if doms else ""
        logger.info(f"INDEX_DOMAIN=auto → domaine détecté : {index_domain}")

    for page in pages:
        url = page.get("url", "") or ""
        # Filtre de domaine : garder uniquement les pages de l'institution (anti-fuite)
        if index_domain:
            dom = urlparse(url).netloc.replace("www.", "").lower()
            if dom != index_domain:
                n_leak += 1
                continue
        raw = page.get("content") or ""
        # Filtre de langue : écarter les pages anglaises (sites bilingues, ex. Présidence)
        if is_english(raw):
            n_en += 1
            continue
        content = clean_content(raw)
        wc = len(content.split())
        if wc < 30:                        # page vide après nettoyage → ignorée
            continue
        if wc > MAX_PAGE_WORDS:            # page-dump multi-sujets (contamine le RAG)
            n_dump += 1
            continue
        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            chunk = (chunk or "").strip()
            # Garde-fou longueur : chunk trop court = fragment bruité
            if len(chunk.split()) < 15:
                n_short += 1
                continue
            # Garde-fou nav/menu : lignes majoritairement très courtes
            if _looks_like_nav(chunk):
                n_nav += 1
                continue
            # Garde-fou dédup : même contenu répété sur plusieurs pages
            key = " ".join(chunk.lower().split())
            if key in seen_keys:
                n_dup += 1
                continue
            seen_keys.add(key)

            url = page["url"]
            raw_type = page.get("type", "html")
            all_texts.append(chunk)
            all_metas.append({
                "url": url,
                "title": page.get("title", ""),
                "date": derive_date(url, page.get("date", "")),
                "type": raw_type,
                "category": categorize(url, page.get("title", ""), content),
                "chunk_index": i,
                # ── Métadonnées scalables (vocabulaire contrôlé) ──
                "org": INDEX_ORG,                            # organisation (obligatoire)
                "source": urlparse(url).netloc.replace("www.", ""),  # domaine précis
                "doc_type": derive_doc_type(url, page.get("title", ""), raw_type),
            })

    logger.info(f"{len(all_texts)} chunks créés depuis {len(pages)} pages "
                f"(filtrés : {n_dump} pages-dumps, {n_dup} doublons, {n_short} courts, {n_nav} nav/menu)")
    return all_texts, all_metas


def build_index(texts: list[str], metadatas: list[dict]) -> Chroma:
    """Crée ou met à jour la base vectorielle ChromaDB."""
    logger.info(f"Chargement du modèle d'embedding: {EMBED_MODEL} ({OLLAMA_BASE_URL})")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    if INDEX_RESET:
        logger.warning("INDEX_RESET=1 → suppression de la collection existante avant réindexation")
        try:
            Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings,
                   persist_directory=str(CHROMA_DIR)).delete_collection()
        except Exception as e:
            logger.warning(f"Reset : {e}")

    logger.info(f"Indexation de {len(texts)} chunks dans ChromaDB (org={INDEX_ORG or 'CGECI'})...")
    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    logger.info(f"Index sauvegardé dans {CHROMA_DIR}/")
    return vectordb


def load_index() -> Chroma:
    """Charge un index ChromaDB existant depuis le disque."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return vectordb


if __name__ == "__main__":
    # Garde-fou : org obligatoire à l'indexation (évite un tag silencieux erroné)
    if not INDEX_ORG:
        raise SystemExit(
            "❌ INDEX_ORG obligatoire. Ex : INDEX_ORG=ANSUT INDEX_RAW_DIR=data/raw_ansut python indexer.py"
        )

    pages = load_raw_pages()

    if not pages:
        print("❌ Aucune page trouvée. Lance d'abord: python scraper.py")
        exit(1)

    texts, metadatas = chunk_pages(pages)
    vectordb = build_index(texts, metadatas)

    count = vectordb._collection.count()
    print(f"\n✅ Index ChromaDB créé: {count} chunks indexés dans {CHROMA_DIR}/")
