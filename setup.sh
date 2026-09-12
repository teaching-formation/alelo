#!/bin/bash
# setup.sh — Installation complète du projet CGECI RAG (macOS)
set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  CGECI RAG — Installation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Vérification Python
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 requis. Installe-le depuis https://python.org"
    exit 1
fi
echo "✅ Python: $(python3 --version)"

# 2. Vérification Ollama
if ! command -v ollama &>/dev/null; then
    echo "❌ Ollama non trouvé. Installe-le depuis https://ollama.com"
    exit 1
fi
echo "✅ Ollama: $(ollama --version 2>/dev/null || echo 'installé')"

# 3. Environnement virtuel
if [ ! -d ".venv" ]; then
    echo "→ Création de l'environnement virtuel..."
    python3 -m venv .venv
fi
source .venv/bin/activate
echo "✅ Environnement virtuel activé"

# 4. Dépendances Python
echo "→ Installation des dépendances..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✅ Dépendances installées"

# 5. Modèles Ollama
echo "→ Téléchargement des modèles Ollama (si absents)..."
ollama pull llama3
ollama pull nomic-embed-text
echo "✅ Modèles Ollama prêts"

# 6. Dossiers de données
mkdir -p data/raw data/chroma_db
echo "✅ Dossiers data/ créés"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Installation terminée !"
echo ""
echo "  Prochaines étapes :"
echo "  1. source .venv/bin/activate"
echo "  2. python scraper.py       # Scrape cgeci.ci (~5 min)"
echo "  3. python indexer.py       # Indexe dans ChromaDB"
echo "  4. streamlit run app.py    # Lance l'interface"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
