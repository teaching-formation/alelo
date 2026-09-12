# Serveur Voxtral STT (natif Mac, GPU)

Transcription vocale française haute qualité (Voxtral Mini 3B de Mistral), exécutée
**nativement sur le Mac** pour exploiter le GPU Apple Silicon via MLX — exactement
comme Ollama pour le LLM. L'app Streamlit (Docker) l'appelle en HTTP.

```
┌─ Docker : app Streamlit ─┐         ┌─ Mac natif (GPU Metal) ──────┐
│ voice.py ─ POST /transcribe ───────▶ Voxtral  :8600  (STT)        │
│ Piper (TTS) dans le conteneur │host │ Ollama   :11434 (LLM)        │
└──────────────────────────┘.internal└──────────────────────────────┘
```

## Démarrer (AVANT de lancer l'app)

```bash
cd voxtral-server
./run.sh
```

Attendre le message `✅ Voxtral prêt sur http://127.0.0.1:8600`
(le 1er lancement charge ~9 Go de modèle, déjà téléchargé dans `~/.cache/huggingface`).

Vérifier : `curl http://127.0.0.1:8600/health`

## Arrêter

`Ctrl-C` dans le terminal, ou : `pkill -f "uvicorn voxtral_server"`

## Repli automatique

Si ce serveur est éteint, l'app bascule seule sur **Whisper local** (CPU, dans le
conteneur) — la voix continue de marcher, en moins précis. Aucune config requise.

## Réglages (optionnels)

- Modèle plus léger/rapide (4-bit, ~2,5 Go) :
  `VOXTRAL_MODEL=mlx-community/Voxtral-Mini-3B-2507-TurboQuant-MLX-4bit ./run.sh`
  *(nécessite `uv pip install <repo>` la 1re fois si non présent)*
- Port : édite `run.sh` et `VOXTRAL_URL` dans `docker-compose.yml`.

## Stack

Python 3.11 (venv `.venv`) · mlx-audio · torch (MPS) · mistral-common · librosa ·
FastAPI + uvicorn. Dépendances gelées dans `requirements.txt`.
```bash
uv venv --python 3.11 && uv pip install -r requirements.txt
```
