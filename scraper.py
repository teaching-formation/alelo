"""
scraper.py — Crawl complet de cgeci.com :
  - Pages HTML avec rendu JavaScript (crawl4ai + playwright)
  - Documents PDF (pypdf)
  - Liens externes trouvés sur le site (profondeur 1)
"""

import os
import re
import json
import time
import asyncio
import hashlib
import logging
import requests
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from datetime import datetime

import pypdf
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Configurable par variables d'environnement (défauts = CGECI).
# Pour scraper un autre site : SCRAPE_BASE_URL, SCRAPE_DOMAIN, SCRAPE_RAW_DIR, SCRAPE_SEEDS.
RAW_DIR = Path(os.getenv("SCRAPE_RAW_DIR", "data/raw"))
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = os.getenv("SCRAPE_BASE_URL", "https://cgeci.com").rstrip("/")
HOME_DOMAIN = os.getenv("SCRAPE_DOMAIN", "cgeci.com").replace("www.", "")
DELAY_SECONDS = float(os.getenv("SCRAPE_DELAY", "1.2"))

# Domaines externes à ne PAS crawler (réseaux sociaux, trackers...)
BLOCKED_EXTERNAL_DOMAINS = {
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "youtube.com", "linkedin.com", "tiktok.com",
    "google.com", "google.fr", "googleapis.com",
    "goo.gl", "bit.ly", "t.co", "amzn.to",
    "cloudflare.com", "wp.com", "wordpress.com",
    "gravatar.com", "doubleclick.net", "googlesyndication.com",
}

SEED_PATHS = [
    "/",
    "/la-cgeci",
    "/membres",
    "/adhesion",
    "/commissions-permanentes",
    "/plate-forme-humanitaire",
    "/category/news/",
    "/mot-du-president/",
    "/les-agendas-du-patronat",
    "/catalogues-bulletins",
    "/documentation",
    "/presse-medias",
    "/category/infos-gouv/",
    "/enquetes-sondages/",
    "/contact/",
    "/category/commission-economie-numerique-et-entreprise-digitale/",
    "/category/commissions-2/developpement-des-pme-financement/",
    "/category/douanes-integration/",
    "/category/promotion-de-lentrepreneuriat-national/",
    "/category/economie-diversification/",
    "/category/commissions-2/emploi-relations-sociales/",
    "/category/energie-ghse/",
    "/category/environnement-des-affaires-competitivite/",
    "/category/juridique-fiscale/",
    "/category/gouvernance-ethique-rse/",
    *[f"/category/news/page/{i}/" for i in range(2, 20)],
]

# Surcharge des seeds via env (liste séparée par des virgules). Sinon, défauts ci-dessus.
_env_seeds = os.getenv("SCRAPE_SEEDS", "").strip()
if _env_seeds:
    SEED_PATHS = [p.strip() for p in _env_seeds.split(",") if p.strip()]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CGECIRagBot/1.0)",
    "Accept-Language": "fr-FR,fr;q=0.9",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def url_to_filename(url: str) -> str:
    slug = re.sub(r"https?://[^/]+", "", url).strip("/").replace("/", "_")
    slug = re.sub(r"[^\w\-]", "_", slug) or "index"
    uid = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{slug}_{uid}.json"


def is_blocked(url: str) -> bool:
    blocked_paths = ["/wp-admin", "/wp-login", "/feed", "xmlrpc.php",
                     "/cart", "/checkout", "?add-to-cart", "/wp-json"]
    if any(p in url for p in blocked_paths):
        return True
    domain = urlparse(url).netloc.replace("www.", "")
    return domain in BLOCKED_EXTERNAL_DOMAINS


def is_internal(url: str) -> bool:
    domain = urlparse(url).netloc.replace("www.", "")
    return domain == HOME_DOMAIN


def is_pdf(url: str) -> bool:
    return bool(re.search(r"\.pdf$", urlparse(url).path, re.I))


def is_binary(url: str) -> bool:
    return bool(re.search(
        r"\.(jpg|jpeg|png|gif|svg|zip|exe|mp4|mp3|avi|doc|xls|ppt)$",
        urlparse(url).path, re.I
    ))


def clean_markdown(text: str) -> str:
    """
    Nettoie le markdown produit par crawl4ai :
    - Supprime les lignes qui sont uniquement des liens de navigation
    - Supprime les lignes trop courtes (< 30 chars)
    - Supprime les doublons de lignes consécutives
    """
    lines = text.split("\n")
    cleaned = []
    prev = ""

    for line in lines:
        stripped = line.strip()

        # Ignore les lignes vides multiples
        if not stripped:
            if prev != "":
                cleaned.append("")
            prev = ""
            continue

        # Ignore les lignes qui ne contiennent qu'un lien markdown : * [texte](url)
        if re.match(r"^\*?\s*\[.+?\]\(.+?\)\s*$", stripped):
            continue

        # Ignore les lignes de navigation courtes (menu items)
        if len(stripped) < 30 and re.match(r"^\*?\s*\[.+?\]", stripped):
            continue

        # Ignore les doublons consécutifs
        if stripped == prev:
            continue

        cleaned.append(line)
        prev = stripped

    return "\n".join(cleaned).strip()


# ── Extraction PDF ────────────────────────────────────────────────────────────

def extract_pdf(content_bytes: bytes, url: str) -> dict:
    try:
        reader = pypdf.PdfReader(BytesIO(content_bytes))
        pages_text = [
            p.extract_text() for p in reader.pages
            if p.extract_text() and p.extract_text().strip()
        ]
        content = "\n\n".join(pages_text)
        if not content.strip():
            return {}

        filename = urlparse(url).path.split("/")[-1]
        title = re.sub(r"[-_]", " ", filename.replace(".pdf", "")).strip()

        return {
            "url": url,
            "title": title,
            "content": content,
            "date": "",
            "type": "pdf",
            "pages": len(reader.pages),
            "scraped_at": datetime.utcnow().isoformat(),
            "word_count": len(content.split()),
        }
    except Exception as e:
        logger.warning(f"Erreur PDF {url}: {e}")
        return {}


def fetch_pdf(url: str) -> bytes | None:
    try:
        # verify=False : certificats gouv.ci parfois invalides/expirés
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        r = requests.get(url, headers=HEADERS, timeout=20, verify=False)
        r.raise_for_status()
        return r.content
    except Exception as e:
        logger.warning(f"Échec PDF {url}: {e}")
        return None


# ── Découverte des liens depuis le markdown crawl4ai ─────────────────────────

def discover_links_from_html(html: str, base_url: str) -> tuple[list, list, list]:
    """
    Retourne (liens_internes, liens_externes, liens_pdf)
    depuis le HTML brut d'une page.
    """
    soup = BeautifulSoup(html, "lxml")
    internal, external, pdfs = set(), set(), set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("mailto:"):
            continue

        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        if parsed.scheme not in ("http", "https"):
            continue
        if is_blocked(full_url):
            continue

        clean = parsed._replace(fragment="", query="").geturl()

        if is_pdf(clean):
            pdfs.add(clean)
        elif is_binary(clean):
            continue
        elif is_internal(clean):
            internal.add(clean)
        else:
            external.add(clean)

    return list(internal), list(external), list(pdfs)


# ── Crawl principal ───────────────────────────────────────────────────────────

async def crawl_all(
    max_internal: int = 150,
    max_external: int = 50,
    max_pdfs: int = 60,
) -> list[dict]:

    results = []
    visited_internal = set()
    visited_external = set()
    visited_pdfs = set()

    internal_queue = [urljoin(BASE_URL, p) for p in SEED_PATHS]
    external_queue = []
    pdf_queue = []

    # ignore_https_errors : certains sites gouv.ci ont des certificats invalides/expirés
    browser_cfg = BrowserConfig(headless=True, verbose=False, ignore_https_errors=True)
    run_cfg = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:

        # ── Pages internes (JS rendu) ─────────────────────────────────────────
        logger.info(f"=== Crawl interne (max {max_internal} pages) ===")

        while internal_queue and len(visited_internal) < max_internal:
            url = internal_queue.pop(0)
            if url in visited_internal:
                continue
            visited_internal.add(url)

            logger.info(f"[INT {len(visited_internal)}/{max_internal}] {url}")
            try:
                result = await crawler.arun(url=url, config=run_cfg)

                if not result.success:
                    logger.warning(f"  Échec: {result.error_message}")
                    continue

                content = result.markdown_v2.raw_markdown if result.markdown_v2 else result.markdown
                content = clean_markdown(content or "")
                if not content or len(content.split()) < 50:
                    logger.info(f"  Contenu insuffisant, ignoré")
                    continue

                # Titre depuis le HTML
                title = ""
                if result.html:
                    soup = BeautifulSoup(result.html, "lxml")
                    if soup.find("h1"):
                        title = soup.find("h1").get_text(strip=True)
                    elif soup.title:
                        title = soup.title.get_text(strip=True)

                page_data = {
                    "url": url,
                    "title": title,
                    "content": content,
                    "date": "",
                    "type": "html",
                    "scraped_at": datetime.utcnow().isoformat(),
                    "word_count": len(content.split()),
                }

                filepath = RAW_DIR / url_to_filename(url)
                filepath.write_text(
                    json.dumps(page_data, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                results.append(page_data)
                logger.info(f"  → {page_data['word_count']} mots")

                # Découverte de liens
                if result.html:
                    new_int, new_ext, new_pdfs = discover_links_from_html(result.html, url)
                    for link in new_int:
                        if link not in visited_internal and link not in internal_queue:
                            internal_queue.append(link)
                    for link in new_ext:
                        if link not in visited_external and link not in external_queue:
                            external_queue.append(link)
                    for pdf in new_pdfs:
                        if pdf not in visited_pdfs and pdf not in pdf_queue:
                            pdf_queue.append(pdf)

            except Exception as e:
                logger.warning(f"  Erreur: {e}")

            await asyncio.sleep(DELAY_SECONDS)

        # ── Pages externes (profondeur 1) ─────────────────────────────────────
        logger.info(f"\n=== Crawl externe (max {max_external} pages) ===")
        logger.info(f"{len(external_queue)} liens externes découverts")

        for url in external_queue[:max_external]:
            if url in visited_external:
                continue
            visited_external.add(url)

            logger.info(f"[EXT {len(visited_external)}/{max_external}] {url}")
            try:
                result = await crawler.arun(url=url, config=run_cfg)
                if not result.success:
                    continue

                content = result.markdown_v2.raw_markdown if result.markdown_v2 else result.markdown
                if not content or len(content.split()) < 80:
                    continue

                soup = BeautifulSoup(result.html or "", "lxml")
                title = ""
                if soup.find("h1"):
                    title = soup.find("h1").get_text(strip=True)

                page_data = {
                    "url": url,
                    "title": title,
                    "content": content,
                    "date": "",
                    "type": "html_external",
                    "scraped_at": datetime.utcnow().isoformat(),
                    "word_count": len(content.split()),
                }

                filepath = RAW_DIR / url_to_filename(url)
                filepath.write_text(
                    json.dumps(page_data, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                results.append(page_data)
                logger.info(f"  → {page_data['word_count']} mots")

                # Découverte PDFs sur les pages externes aussi
                if result.html:
                    _, _, new_pdfs = discover_links_from_html(result.html, url)
                    for pdf in new_pdfs:
                        if pdf not in visited_pdfs and pdf not in pdf_queue:
                            pdf_queue.append(pdf)

            except Exception as e:
                logger.warning(f"  Erreur: {e}")

            await asyncio.sleep(DELAY_SECONDS)

    # ── PDFs (hors async crawler) ─────────────────────────────────────────────
    logger.info(f"\n=== PDFs (max {max_pdfs}) ===")
    logger.info(f"{len(pdf_queue)} PDFs découverts")

    for url in pdf_queue[:max_pdfs]:
        if url in visited_pdfs:
            continue
        visited_pdfs.add(url)

        logger.info(f"[PDF {len(visited_pdfs)}/{max_pdfs}] {url}")
        raw = fetch_pdf(url)
        if not raw:
            continue

        pdf_data = extract_pdf(raw, url)
        if pdf_data and pdf_data.get("word_count", 0) > 30:
            filepath = RAW_DIR / url_to_filename(url)
            filepath.write_text(
                json.dumps(pdf_data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            results.append(pdf_data)
            logger.info(f"  → {pdf_data.get('pages')} pages, {pdf_data['word_count']} mots")

        time.sleep(DELAY_SECONDS)

    # ── Résumé ────────────────────────────────────────────────────────────────
    html_n = sum(1 for r in results if r["type"] == "html")
    ext_n  = sum(1 for r in results if r["type"] == "html_external")
    pdf_n  = sum(1 for r in results if r["type"] == "pdf")
    logger.info(f"\nTerminé: {html_n} pages internes + {ext_n} pages externes + {pdf_n} PDFs")
    return results


if __name__ == "__main__":
    results = asyncio.run(crawl_all(
        max_internal=int(os.getenv("SCRAPE_MAX_INTERNAL", "150")),
        max_external=int(os.getenv("SCRAPE_MAX_EXTERNAL", "20")),
        max_pdfs=int(os.getenv("SCRAPE_MAX_PDFS", "80")),
    ))
    html_n = sum(1 for r in results if r["type"] == "html")
    ext_n  = sum(1 for r in results if r["type"] == "html_external")
    pdf_n  = sum(1 for r in results if r["type"] == "pdf")
    print(f"\n✅ {html_n} pages cgeci.com + {ext_n} pages externes + {pdf_n} PDFs → data/raw/")
