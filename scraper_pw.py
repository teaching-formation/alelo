"""
scraper_pw.py — Crawler Playwright DIRECT (contourne le bug crawl4ai sur les sites .gouv.ci).

Gère : pages d'intro « ENTREZ SUR LE SITE », rendu JavaScript (SPA), crawl interne borné.
Sort des JSON au format attendu par indexer.py : {url, title, date, type, content, ...}.

Paramétrage par env :
    SCRAPE_BASE_URL, SCRAPE_DOMAIN, SCRAPE_RAW_DIR, SCRAPE_MAX_INTERNAL (défaut 40).

Lancement (dans le conteneur api) :
    docker compose exec -e SCRAPE_BASE_URL=… -e SCRAPE_DOMAIN=… -e SCRAPE_RAW_DIR=… api python scraper_pw.py
"""
import os
import re
import json
import asyncio
import hashlib
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timezone

from playwright.async_api import async_playwright

BASE = os.getenv("SCRAPE_BASE_URL", "").rstrip("/")
DOMAIN = os.getenv("SCRAPE_DOMAIN", urlparse(BASE).netloc).replace("www.", "")
RAW = Path(os.getenv("SCRAPE_RAW_DIR", "data/raw_x"))
RAW.mkdir(parents=True, exist_ok=True)
MAX_PAGES = int(os.getenv("SCRAPE_MAX_INTERNAL", "40"))
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
BAD_EXT = (".pdf", ".jpg", ".jpeg", ".png", ".zip", ".mp4", ".doc", ".docx",
           ".xls", ".xlsx", ".gif", ".svg", ".webp", ".mp3", ".rar")


def same_domain(u: str) -> bool:
    try:
        return urlparse(u).netloc.replace("www.", "").endswith(DOMAIN)
    except Exception:
        return False


def fname(u: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", urlparse(u).path.lower()).strip("_")[:60] or "home"
    return f"{slug}_{hashlib.md5(u.encode()).hexdigest()[:8]}.json"


async def links_of(page) -> list:
    try:
        hrefs = await page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
    except Exception:
        return []
    out = []
    for h in hrefs:
        h = (h or "").split("#")[0].rstrip("/")
        if h and same_domain(h) and not h.lower().endswith(BAD_EXT):
            out.append(h)
    return out


async def render(page, url: str):
    await page.goto(url, wait_until="domcontentloaded", timeout=45000)
    await page.wait_for_timeout(3000)          # laisse le JS peupler la page
    try:
        txt = await page.inner_text("body")   # préserve les \n entre blocs (nécessaire à clean_content)
    except Exception:
        txt = ""
    title = (await page.title()) or ""
    txt = txt or ""
    txt = re.sub(r"[ \t]+", " ", txt)          # espaces multiples → un seul
    txt = re.sub(r"\n[ \t]*", "\n", txt)       # trim en début de ligne
    txt = re.sub(r"\n{2,}", "\n", txt).strip() # lignes vides multiples → une
    return title.strip(), txt


async def main():
    if not BASE:
        print("SCRAPE_BASE_URL manquant"); return
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(user_agent=UA, ignore_https_errors=True)
        page = await ctx.new_page()

        # 1) Home + éventuelle page d'INTRO (« ENTREZ SUR LE SITE »)
        title, txt = await render(page, BASE)
        start = BASE
        if len(txt) < 250:
            ls = await links_of(page)
            enter = next((l for l in ls if re.search(r"entr|accueil|home|site|welcome", l, re.I)), None)
            enter = enter or next((l for l in ls if l.rstrip("/") != BASE), None)
            if enter:
                start = enter
                print("  (page d'intro détectée → entrée sur", enter, ")")

        # 2) BFS borné
        seen = {start}
        queue = [start]
        saved = 0
        while queue and saved < MAX_PAGES:
            url = queue.pop(0)
            try:
                title, txt = await render(page, url)
            except Exception as e:
                print("  ! échec", url[:60], str(e)[:40])
                continue
            if len(txt) > 200:
                rec = {"url": url, "title": title, "date": "", "type": "html",
                       "content": txt, "scraped_at": datetime.now(timezone.utc).isoformat(),
                       "word_count": len(txt.split())}
                (RAW / fname(url)).write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
                saved += 1
                print(f"  [{saved}/{MAX_PAGES}] {url[:62]} ({len(txt)} car)")
            for l in await links_of(page):
                if l not in seen and len(seen) < MAX_PAGES * 5:
                    seen.add(l)
                    queue.append(l)
        await browser.close()
        print(f"=== {DOMAIN} : {saved} pages sauvées dans {RAW} ===")


if __name__ == "__main__":
    asyncio.run(main())
