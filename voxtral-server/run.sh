#!/usr/bin/env bash
# Lance le serveur STT (Whisper large-v3, GPU Apple Silicon) + Reranker, hors Docker.
# À démarrer AVANT l'app, comme Ollama.
set -e
cd "$(dirname "$0")"
# large-v3 multilingue (auto-détection FR/EN). Turbo pour + de vitesse :
#   export STT_MODEL=mlx-community/whisper-large-v3-turbo
export STT_MODEL="${STT_MODEL:-mlx-community/whisper-large-v3-mlx}"
exec .venv/bin/uvicorn voxtral_server:app --host 127.0.0.1 --port 8600
