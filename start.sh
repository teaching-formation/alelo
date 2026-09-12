#!/usr/bin/env bash
# Démarre toute la stack CGECI.ai en une commande.
#   - Vérifie Ollama (natif)
#   - Démarre Voxtral + Reranker (natif GPU) si besoin
#   - Lance l'API + le front Next.js (Docker)
set -e
cd "$(dirname "$0")"
echo "▶ CGECI.ai — démarrage de la stack"

# 1) Ollama (natif, LLM + embeddings)
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "  ✓ Ollama (:11434)"
else
  echo "  ⚠️  Ollama ne répond pas. Lance-le (app Ollama ou 'ollama serve'), puis relance."
fi

# 2) Voxtral + Reranker (natif, GPU Apple Silicon)
if curl -s http://127.0.0.1:8600/health >/dev/null 2>&1; then
  echo "  ✓ Voxtral + Reranker (:8600)"
else
  echo "  ▶ Démarrage Voxtral + Reranker (natif)…"
  ( cd voxtral-server && nohup ./run.sh > /tmp/voxtral.log 2>&1 & )
  echo "    (chargement des modèles en cours — voir /tmp/voxtral.log)"
fi

# 3) API + Front (Docker)
echo "  ▶ Docker : API + front Next.js…"
docker compose up -d --build

echo ""
echo "✅ Prêt :"
echo "   • Application  : http://localhost:3000"
echo "   • API (santé)  : http://localhost:8088/api/health"
echo ""
echo "Pour arrêter :  docker compose down   (et  pkill -f 'uvicorn voxtral_server'  pour la voix)"
