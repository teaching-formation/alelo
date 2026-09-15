#!/usr/bin/env bash
# Scrape séquentiel des 7 ministères manquants via le crawler Playwright maison (scraper_pw.py),
# dans le conteneur api (chromium présent). crawl4ai est bogué → on ne l'utilise pas.
set -u
cd "$(dirname "$0")"

# org | base_url | domain
SITES=(
  "sante|https://sante.gouv.ci|sante.gouv.ci"
  "agriculture|https://agriculture.gouv.ci|agriculture.gouv.ci"
  "justice|https://justice.gouv.ci|justice.gouv.ci"
  "interieur|https://interieur.gouv.ci|interieur.gouv.ci"
  "education|https://education.gouv.ci|education.gouv.ci"
  "energie|https://energie.gouv.ci|energie.gouv.ci"
  "commerce|https://commerce.gouv.ci|commerce.gouv.ci"
)

for row in "${SITES[@]}"; do
  IFS='|' read -r org url domain <<< "$row"
  echo "======================================================================"
  echo "### $org  ($url)"
  rm -f "data/raw_$org"/*.json 2>/dev/null
  docker compose exec -T \
    -e SCRAPE_BASE_URL="$url" -e SCRAPE_DOMAIN="$domain" \
    -e SCRAPE_RAW_DIR="data/raw_$org" -e SCRAPE_MAX_INTERNAL=40 \
    api python scraper_pw.py 2>&1 | grep -E "^\s*\[|intro|===|échec" | tail -6
done
echo "===== TERMINÉ ====="
for org in sante agriculture justice interieur education energie commerce; do
  echo "  $org: $(find data/raw_$org -name '*.json' 2>/dev/null | wc -l | tr -d ' ') docs"
done
