"""
app.py — Interface CGECI.ai : onglet Recherche (Google-style) + onglet Chat (conversationnel).
"""

import os
import time
import logging
import streamlit as st

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="Assistant CGECI",
    page_icon="🏢",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ── Global ── */
body, .stApp { background: #ffffff; }
.block-container { padding-top: 2rem; max-width: 760px; }

/* ── Logo ── */
.logo {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
}
.logo span:nth-child(1) { color: #1a3a5c; }
.logo span:nth-child(2) { color: #e67e22; }
.logo span:nth-child(3) { color: #1a3a5c; }
.logo span:nth-child(4) { color: #27ae60; }
.logo span:nth-child(5) { color: #e74c3c; }
.logo span:nth-child(6) { color: #f39c12; }
.tagline {
    text-align: center;
    color: #888;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

/* ── Boutons modèles ── */
div[data-testid="stButton"] button {
    border-radius: 24px;
    background: #1a3a5c;
    color: white;
    border: none;
    padding: 10px 28px;
    font-size: 0.95rem;
    width: 100%;
}

/* ── Input texte ── */
div[data-testid="stTextInput"] input {
    border: none !important;
    box-shadow: none !important;
    font-size: 1rem;
}
div[data-testid="stTextInput"] > div {
    border: 1.5px solid #dadce0 !important;
    border-radius: 28px !important;
    padding: 6px 16px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}

/* ── Suggestions ── */
.suggestions-title {
    text-align: center;
    color: #aaa;
    font-size: 0.8rem;
    margin: 1.5rem 0 0.6rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Chat : bulles ── */
.stChatMessage { border-radius: 12px; }

/* Cache la sidebar toggle */
[data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Modèles disponibles ────────────────────────────────────────────────────────
# "cgeci" = notre modèle maison (Modelfile, basé sur qwen2.5). Les autres restent
# disponibles pour comparaison. Ordre = ordre d'affichage dans la liste.
MODELS = {
    "cgeci":     {"label": "CGECI.ai (maison)", "flag": "🇨🇮"},
    "qwen2.5":   {"label": "Qwen 2.5",  "flag": "🇨🇳"},
    "mistral":   {"label": "Mistral",   "flag": "🇫🇷"},
    "llama3.1":  {"label": "Llama 3.1", "flag": "🦙"},
    "llama3":    {"label": "Llama 3",   "flag": "🦙"},
}

SUGGESTIONS = [
    "Qu'est-ce que la CGECI ?",
    "Comment adhérer à la CGECI ?",
    "Quelles sont les commissions permanentes ?",
    "Quel est le coût d'adhésion ?",
    "Quelles sont les dernières actualités ?",
    "Qui est le président de la CGECI ?",
]

# ── État session ───────────────────────────────────────────────────────────────
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "cgeci"
if "question" not in st.session_state:
    st.session_state["question"] = ""
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []   # [{"role": "user"|"assistant", "content": "..."}]

# ── Logo ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="logo">
  <span>C</span><span>G</span><span>E</span><span>C</span><span>I</span><span>.</span>ai
</div>
<div class="tagline">L'assistant intelligent du Patronat Ivoirien</div>
""", unsafe_allow_html=True)

# ── Sélecteur de modèle (liste déroulante compacte) ────────────────────────────
_model_ids = list(MODELS.keys())
_current = st.session_state["selected_model"]
if _current not in _model_ids:
    _current = _model_ids[0]

def _fmt_model(mid):
    m = MODELS[mid]
    return f"{m['flag']}  {m['label']}"

# Périmètre = filtre de source (org). "Tout" = pas de filtre (les deux jambes).
SCOPES = {"🌍 Tout": None, "📡 ANSUT": "ANSUT", "🏢 CGECI": "CGECI"}
_scope_labels = list(SCOPES.keys())

col_model, col_scope = st.columns([2, 2])
with col_model:
    st.session_state["selected_model"] = st.selectbox(
        "Modèle",
        _model_ids,
        index=_model_ids.index(_current),
        format_func=_fmt_model,
        label_visibility="collapsed",
    )
with col_scope:
    _scope_label = st.selectbox(
        "Périmètre",
        _scope_labels,
        index=0,
        label_visibility="collapsed",
        help="Filtrer les réponses sur une organisation précise",
    )
    st.session_state["scope_org"] = SCOPES[_scope_label]

st.markdown("<br>", unsafe_allow_html=True)

# ── Chargement ChromaDB + BM25 (une seule fois) ────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_vectordb():
    from rag_chain import load_vectordb, _build_bm25_index
    db = load_vectordb()
    _build_bm25_index(db)
    return db


# ── Accès au Mode Vocal (page orbe immersive, port 8502) ───────────────────────
VOICE_MODE_URL = os.getenv("VOICE_MODE_URL", "http://localhost:8502")
_, col_vm = st.columns([2, 1])
with col_vm:
    st.link_button("🎙️  Mode Vocal", VOICE_MODE_URL, use_container_width=True)

# ── Onglets ────────────────────────────────────────────────────────────────────
tab_search, tab_chat = st.tabs(["🔍  Recherche", "💬  Chat"])


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — RECHERCHE (interface Google)
# ══════════════════════════════════════════════════════════════════════════════
with tab_search:

    # Upload fichier
    uploaded_file = st.file_uploader(
        "Joindre un fichier",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
        help="PDF, Word, texte ou image",
        key="search_upload",
    )
    if uploaded_file:
        st.markdown(
            f'<div style="background:#f0f7ff;border-radius:8px;padding:8px 14px;'
            f'font-size:0.85rem;margin-bottom:0.5rem">'
            f'📎 <b>{uploaded_file.name}</b> · {uploaded_file.size // 1024} KB</div>',
            unsafe_allow_html=True,
        )

    # Barre de recherche
    question = st.text_input(
        "Votre question",
        placeholder="🔍  Posez votre question sur la CGECI...",
        key="question_input",
        label_visibility="collapsed",
        value=st.session_state.get("question", ""),
    )
    search_clicked = st.button("Rechercher", use_container_width=True,
                               type="primary", key="btn_search")

    if search_clicked and question.strip():
        try:
            with st.spinner("Chargement de la base..."):
                vectordb = get_vectordb()

            extra_context = ""
            if uploaded_file:
                from document_processor import process_upload
                doc = process_upload(uploaded_file.read(), uploaded_file.name)
                extra_context = doc["text"][:3000]
                st.caption(f"📄 {doc['word_count']} mots extraits ({doc['type']})")

            from rag_chain import ask
            with st.spinner("Génération de la réponse..."):
                t0 = time.time()
                result = ask(
                    vectordb,
                    question,
                    model=st.session_state["selected_model"],
                    extra_context=extra_context,
                    org=st.session_state.get("scope_org"),
                )
                elapsed = time.time() - t0

            model_label = MODELS[st.session_state["selected_model"]]["label"]
            st.markdown("### 💬 Réponse")
            st.info(result["answer"] or "Aucune réponse générée.")
            st.caption(f"⏱ {elapsed:.1f}s · {model_label}")

            if result["sources"]:
                st.markdown("**📎 Sources**")
                for src in result["sources"]:
                    with st.expander(f"📄 {src['title'] or src['url']}"):
                        st.markdown(f"[{src['url']}]({src['url']})")
                        st.write(src['excerpt'][:300])

        except FileNotFoundError:
            st.error("Base vectorielle introuvable. Lance : `python indexer.py`")
        except Exception as e:
            st.error(f"Erreur : {e}")
            st.exception(e)

    elif search_clicked:
        st.warning("Veuillez saisir une question.")

    # Suggestions (uniquement si pas de recherche en cours)
    if not (search_clicked and question.strip()):
        st.markdown('<div class="suggestions-title">Suggestions</div>',
                    unsafe_allow_html=True)
        cols2 = st.columns(2)
        for i, suggestion in enumerate(SUGGESTIONS):
            with cols2[i % 2]:
                if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                    st.session_state["question"] = suggestion
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — CHAT CONVERSATIONNEL
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:

    # Bouton effacer l'historique
    col_title, col_clear = st.columns([5, 1])
    with col_title:
        st.markdown("**Conversation avec la CGECI.ai**")
    with col_clear:
        if st.button("🗑️", key="clear_chat", help="Effacer la conversation"):
            st.session_state["chat_history"] = []
            st.rerun()

    # Affiche l'historique
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"],
                             avatar="🧑" if msg["role"] == "user" else "🏢"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} source(s)", expanded=False):
                    for src in msg["sources"][:4]:
                        st.markdown(
                            f"**[{src['title'] or src['url']}]({src['url']})**  \n"
                            f"_{src['excerpt'][:150]}_"
                        )

    # Message de bienvenue si conversation vide
    if not st.session_state["chat_history"]:
        st.markdown(
            '<div style="text-align:center;color:#aaa;padding:2rem 0;font-size:0.95rem">'
            '💬 Posez votre première question pour démarrer la conversation.<br>'
            '<small>Je me souviens de tout ce que vous me dites dans cette session.</small>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Zone de saisie texte
    user_input = st.chat_input(
        "Écrivez votre message...",
        key="chat_input",
    )

    if user_input and user_input.strip():
        # Affiche la question de l'utilisateur
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        # Génère la réponse en streaming (tokens affichés au fur et à mesure)
        with st.chat_message("assistant", avatar="🏢"):
            try:
                vectordb = get_vectordb()
                from rag_chain import chat_stream as rag_chat_stream

                # Capture les sources émises comme dernier élément du générateur
                captured = {"sources": []}

                def token_generator():
                    for chunk in rag_chat_stream(
                        vectordb,
                        user_input,
                        history=st.session_state["chat_history"],
                        model=st.session_state["selected_model"],
                        org=st.session_state.get("scope_org"),
                    ):
                        if isinstance(chunk, str):
                            yield chunk
                        else:  # marqueur final : dict avec sources + answer complète
                            captured["sources"] = chunk.get("sources", [])
                            captured["answer"] = chunk.get("answer", "")

                t0 = time.time()
                streamed_text = st.write_stream(token_generator())
                elapsed = time.time() - t0

                answer = captured.get("answer") or streamed_text
                model_label = MODELS[st.session_state["selected_model"]]["label"]
                st.caption(f"⏱ {elapsed:.1f}s · {model_label}")

                result = {"answer": answer, "sources": captured["sources"]}

                if result["sources"]:
                    with st.expander(
                        f"📎 {len(result['sources'])} source(s)", expanded=False
                    ):
                        for src in result["sources"][:4]:
                            st.markdown(
                                f"**[{src['title'] or src['url']}]({src['url']})**  \n"
                                f"_{src['excerpt'][:150]}_"
                            )

            except Exception as e:
                result = {"answer": f"Erreur : {e}", "sources": []}
                st.error(result["answer"])

        # Sauvegarde dans l'historique
        st.session_state["chat_history"].append(
            {"role": "user", "content": user_input}
        )
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": result["answer"],
             "sources": result.get("sources", [])}
        )

        # Limite l'historique à 20 échanges (10 tours)
        if len(st.session_state["chat_history"]) > 20:
            st.session_state["chat_history"] = st.session_state["chat_history"][-20:]


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>100% local · Ollama + ChromaDB + Hybrid RAG · "
    "<a href='https://cgeci.com' target='_blank'>cgeci.com</a></small></center>",
    unsafe_allow_html=True,
)
