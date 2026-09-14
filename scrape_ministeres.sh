#!/usr/bin/env bash
# Scrape séquentiel des ministères manquants (dans le conteneur api : crawl4ai + chromium).
# Usage : ./scrape_ministeres.sh            (les 6 restants ; justice déjà fait à part)
set -u
cd "$(dirname "$0")"

# org  | base_url                       | domain
SITES=(
  "sante|https://sante.gouv.ci|sante.gouv.ci"
  "agriculture|https://agriculture.gouv.ci|agriculture.gouv.ci"
  "interieur|https://interieur.gouv.ci|interieur.gouv.ci"
  "education|https://education.gouv.ci|education.gouv.ci"
  "energie|https://energie.gouv.ci|energie.gouv.ci"
  "commerce|https://commerce.gouv.ci|commerce.gouv.ci"
)

for row in "${SITES[@]}"; do
  IFS='|' read -r org url domain <<< "$row"
  echo "======================================================================"
  echo "### SCRAPE $org  ($url)"
  echo "======================================================================"
  docker compose exec -T \
    -e SCRAPE_BASE_URL="$url" -e SCRAPE_DOMAIN="$domain" \
    -e SCRAPE_RAW_DIR="data/raw_$org" \
    -e SCRAPE_MAX_INTERNAL=45 -e SCRAPE_MAX_PDFS=25 -e SCRAPE_MAX_EXTERNAL=4 \
    api python scraper.py 2>&1 | grep -iE "INT |PDF |EXT |termine|fini|erreur|error|pages" | tail -6
  n=$(find "data/raw_$org" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
  echo "→ $org : $n docs"
done
echo "===== SCRAPING TERMINÉ ====="
for org in sante agriculture interieur education energie commerce justice; do
  echo "  $org: $(find data/raw_$org -name '*.json' 2>/dev/null | wc -l | tr -d ' ') docs"
done
