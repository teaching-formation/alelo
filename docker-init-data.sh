#!/bin/bash
# Lance le scraping et l'indexation dans le conteneur app.
# Ollama doit tourner nativement sur l'hôte : ollama serve

set -e
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  CGECI RAG — Init des données"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Vérifie qu'Ollama est accessible sur l'hôte
if ! curl -sf http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Ollama n'est pas démarré. Lance : ollama serve"
    exit 1
fi

echo "→ Scraping cgeci.ci..."
docker exec cgeci-app python scraper.py

echo "→ Indexation dans ChromaDB..."
docker exec cgeci-app python indexer.py

echo ""
echo "✅ Données prêtes. Ouvre http://localhost:8501"
