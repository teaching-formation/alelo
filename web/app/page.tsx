"use client";
import { useEffect, useRef, useState } from "react";
import { Config, Msg, getConfig, chatStream, transcribe, tts } from "@/lib/api";
import { Recorder } from "@/lib/recorder";
import { SpeechQueue } from "@/lib/speech";
import Message from "@/components/Message";
import VoicePicker from "@/components/VoicePicker";
import AgentAvatar from "@/components/AgentAvatar";
import VoiceMode from "@/components/VoiceMode";

const SUGGESTIONS = [
  "Comment payer mes impôts en ligne ?",
  "Quelles démarches pour ma carte nationale d'identité ?",
  "Comment créer mon entreprise ?",
  "Comment obtenir mon permis de conduire ?",
];

type Convo = { id: string; title: string; messages: Msg[]; updatedAt: number };
function blankConvo(): Convo {
  const id = typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
  return { id, title: "", messages: [], updatedAt: Date.now() };
}

export default function Home() {
  const [cfg, setCfg] = useState<Config | null>(null);
  const [convos, setConvos] = useState<Convo[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const messages = convos.find((c) => c.id === activeId)?.messages ?? [];
  const setMessages = (u: Msg[] | ((m: Msg[]) => Msg[])) =>
    setConvos((prev) => prev.map((c) =>
      c.id === activeId
        ? { ...c, messages: typeof u === "function" ? (u as (m: Msg[]) => Msg[])(c.messages) : u, updatedAt: Date.now() }
        : c));
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<"auto" | "expert">("auto");
  const [model, setModel] = useState("cgeci");
  const [org, setOrg] = useState("");
  const [voice, setVoice] = useState("siwis");
  const [voiceReply, setVoiceReply] = useState(true);
  const [showPicker, setShowPicker] = useState(false);
  const [sidebar, setSidebar] = useState(false);
  const [recording, setRecording] = useState(false);
  const [micLevel, setMicLevel] = useState(0);
  const [theme, setTheme] = useState<"system" | "light" | "dark">("system");
  const [editingId, setEditingId] = useState("");
  const [editVal, setEditVal] = useState("");
  const [voiceMode, setVoiceMode] = useState(false);
  const [cfgError, setCfgError] = useState(false);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [confirmDel, setConfirmDel] = useState("");

  const recRef = useRef<Recorder | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const streamingRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  function autosize() {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
  }

  useEffect(() => {
    getConfig().then((c) => {
      setCfg(c);
      setCfgError(false);
      setModel(c.defaultModel);
      const saved = typeof window !== "undefined" ? localStorage.getItem("cgeci_voice") : null;
      if (saved && c.voices.some((v) => v.id === saved)) setVoice(saved);
      else setShowPicker(true);
    }).catch(() => setCfgError(true));
  }, []);

  // Raccourcis clavier : Cmd/Ctrl+K = nouvelle conversation, Échap = fermer.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); newConvo(); }
      else if (e.key === "Escape") {
        if (voiceMode) setVoiceMode(false);
        else if (sidebar) setSidebar(false);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // Chargement des conversations sauvegardées
  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("cgeci_convos") || "[]");
      if (Array.isArray(saved) && saved.length) { setConvos(saved); setActiveId(saved[0].id); return; }
    } catch {}
    const c = blankConvo(); setConvos([c]); setActiveId(c.id);
  }, []);

  // Persistance (débouncée : coalesce les rafales de tokens en une seule écriture)
  useEffect(() => {
    if (!convos.length) return;
    const t = setTimeout(() => {
      try { localStorage.setItem("cgeci_convos", JSON.stringify(convos)); } catch {}
    }, 400);
    return () => clearTimeout(t);
  }, [convos]);

  // Auto-scroll qui SUIT la génération, sauf si l'utilisateur a remonté volontairement.
  const lastLen = messages[messages.length - 1]?.content.length ?? 0;
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 160;
    if (nearBottom) el.scrollTo({ top: el.scrollHeight });
  }, [messages.length, lastLen, busy]);

  function onScroll() {
    const el = scrollRef.current;
    if (!el) return;
    setShowScrollBtn(el.scrollHeight - el.scrollTop - el.clientHeight > 240);
  }
  function scrollToBottom() {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }

  function newConvo() {
    const cur = convos.find((c) => c.id === activeId);
    if (cur && cur.messages.length === 0) { setSidebar(false); return; }
    const c = blankConvo();
    setConvos((p) => [c, ...p]); setActiveId(c.id); setSidebar(false);
  }
  function selectConvo(id: string) { setActiveId(id); setSidebar(false); }
  // Le Mode Vocal démarre TOUJOURS une conversation neuve (sauf si la courante est déjà vide).
  function openVoiceMode() {
    const cur = convos.find((c) => c.id === activeId);
    if (!cur || cur.messages.length > 0) {
      const c = blankConvo();
      setConvos((p) => [c, ...p]);
      setActiveId(c.id);
    }
    setSidebar(false);
    setVoiceMode(true);
  }
  function renameConvo(id: string, title: string) {
    setConvos((p) => p.map((c) => c.id === id ? { ...c, title: title.trim() || c.title } : c));
    setEditingId("");
  }

  // Thème
  useEffect(() => {
    const t = localStorage.getItem("cgeci_theme");
    if (t === "dark" || t === "light") setTheme(t);
  }, []);
  function cycleTheme() {
    const order: ("system" | "light" | "dark")[] = ["system", "light", "dark"];
    const t = order[(order.indexOf(theme) + 1) % 3];
    setTheme(t);
    if (t === "system") { localStorage.removeItem("cgeci_theme"); document.documentElement.removeAttribute("data-theme"); }
    else { localStorage.setItem("cgeci_theme", t); document.documentElement.setAttribute("data-theme", t); }
  }
  function deleteConvo(id: string) {
    setConvos((p) => {
      const next = p.filter((c) => c.id !== id);
      if (next.length === 0) { const c = blankConvo(); setActiveId(c.id); return [c]; }
      if (id === activeId) setActiveId(next[0].id);
      return next;
    });
  }

  function validateVoice(id: string) {
    setVoice(id);
    localStorage.setItem("cgeci_voice", id);
    setShowPicker(false);
  }

  async function playTTS(text: string) {
    try {
      const blob = await tts(text, voice);
      const url = URL.createObjectURL(blob);
      const a = new Audio(url);
      a.onended = () => URL.revokeObjectURL(url);
      a.play().catch(() => {});
    } catch {}
  }

  async function send(text: string, spoken = false, base?: Msg[]) {
    if (!text.trim() || busy) return;
    setInput("");
    if (taRef.current) taRef.current.style.height = "auto";
    const baseMsgs = base ?? messages;
    const history = baseMsgs.map((m) => ({ role: m.role, content: m.content }));
    const next = [...baseMsgs, { role: "user", content: text, ts: Date.now() } as Msg, { role: "assistant", content: "" } as Msg];
    setMessages(next);
    // Titre de la conversation = 1er message
    setConvos((p) => p.map((c) => c.id === activeId && !c.title ? { ...c, title: text.slice(0, 44) } : c));
    setBusy(true);
    streamingRef.current = true;
    const idx = next.length - 1;

    // TTS progressive si la question vient de la voix (audio dès la 1re phrase).
    const sq = spoken && voiceReply ? new SpeechQueue(voice) : null;

    const ac = new AbortController();
    abortRef.current = ac;
    let answer = "", sources: any[] = [], document: any = undefined;
    try {
      for await (const ev of chatStream({
        message: text, history,
        mode, org: mode === "expert" ? org : null,
        model: "alelo",
      }, ac.signal)) {
        if (ev.type === "step") {
          setMessages((m) => { const c = [...m]; c[idx] = { ...c[idx], steps: [...(c[idx].steps || []), ev.text] }; return c; });
        } else if (ev.type === "token") {
          answer += ev.text;
          setMessages((m) => { const c = [...m]; c[idx] = { ...c[idx], content: answer }; return c; });
          sq?.push(ev.text);
        } else if (ev.type === "done") {
          answer = ev.answer || answer; sources = ev.sources || []; document = ev.document;
        } else if (ev.type === "error") {
          answer = answer + "\n\n⚠️ La réponse a été interrompue (" + ev.message + "). Réessayez.";
        }
      }
    } catch (e: any) {
      if (e?.name === "AbortError") {
        answer = answer || "_(réponse arrêtée)_";
      } else {
        answer = answer || "⚠️ Impossible de joindre le service. Vérifiez votre connexion et réessayez.";
      }
    }
    abortRef.current = null;
    sq?.finish();
    streamingRef.current = false;
    setMessages((m) => { const c = [...m]; c[idx] = { ...c[idx], role: "assistant", content: answer, sources, document, ts: Date.now() }; return c; });
    setBusy(false);
  }

  function stopGen() {
    abortRef.current?.abort();
  }

  // Régénère la dernière réponse : rejoue la dernière question avec l'historique amont.
  function regenerate() {
    if (busy) return;
    let lastUser = -1;
    for (let i = messages.length - 1; i >= 0; i--) if (messages[i].role === "user") { lastUser = i; break; }
    if (lastUser < 0) return;
    send(messages[lastUser].content, false, messages.slice(0, lastUser));
  }

  // Persiste un tour du Mode Vocal comme une vraie conversation (texte + sources).
  function appendVoiceTurn(user: string, assistant: string, sources: any[], document?: any) {
    setMessages((m) => [...m, { role: "user", content: user, ts: Date.now() } as Msg,
                             { role: "assistant", content: assistant, sources, document, ts: Date.now() } as Msg]);
    setConvos((p) => p.map((c) => c.id === activeId && !c.title ? { ...c, title: user.slice(0, 44) } : c));
  }

  async function toggleMic() {
    if (recording) {
      const wav = recRef.current?.stop();
      setRecording(false); setMicLevel(0);
      if (!wav) return;
      setBusy(true);
      const text = await transcribe(wav);
      setBusy(false);
      if (text) send(text, true);
      return;
    }
    const rec = new Recorder();
    rec.onLevel = (l) => setMicLevel(l);
    rec.onAutoStop = () => { if (recRef.current === rec) toggleMic(); };
    recRef.current = rec;
    try {
      await rec.start();
      setRecording(true);
    } catch {
      alert("Micro refusé. Autorisez l'accès au microphone.");
    }
  }

  const empty = messages.length === 0;

  return (
    <div className="flex h-screen" style={{ background: "var(--bg)" }}>
      {showPicker && cfg && (
        <VoicePicker voices={cfg.voices} current={voice} onValidate={validateVoice} />
      )}
      {voiceMode && (
        <VoiceMode
          voice={voice}
          org={mode === "expert" ? org : null}
          history={messages.map((m) => ({ role: m.role, content: m.content }))}
          onTurn={appendVoiceTurn}
          onClose={() => setVoiceMode(false)}
        />
      )}

      {/* ── Sidebar ── */}
      <aside
        className={`fixed md:static z-40 h-full w-72 flex flex-col transition-transform ${sidebar ? "translate-x-0" : "-translate-x-full md:translate-x-0"}`}
        style={{ background: "var(--panel)", borderRight: "1px solid var(--border)" }}>
        <div className="p-4 flex items-center gap-2">
          <AgentAvatar size={34} />
          <div>
            <div className="text-2xl font-bold tracking-tight lowercase" style={{ color: "var(--accent)" }}>
              alélo
            </div>
            <div className="text-xs" style={{ color: "var(--muted)" }}>L&apos;IA publique au service du citoyen</div>
          </div>
        </div>

        <button onClick={newConvo}
                className="mx-4 mb-2 rounded-xl py-2.5 text-sm font-medium"
                style={{ background: "var(--bg-elev)", border: "1px solid var(--border)" }}>
          ＋ Nouvelle conversation
        </button>
        <button onClick={openVoiceMode}
                className="mx-4 mb-3 rounded-xl py-2.5 text-sm font-semibold text-white flex items-center justify-center gap-2"
                style={{ background: "var(--accent)" }}>
          🎙️ Mode Vocal
        </button>

        {/* Mode */}
        <div className="px-4">
          <div className="text-xs mb-1.5" style={{ color: "var(--muted)" }}>Mode</div>
          <div className="flex rounded-xl p-1 mb-4" style={{ background: "var(--bg-elev)" }}>
            {(["auto", "expert"] as const).map((m) => (
              <button key={m} onClick={() => setMode(m)}
                      className="flex-1 rounded-lg py-1.5 text-sm font-medium capitalize"
                      style={{ background: mode === m ? "var(--accent)" : "transparent", color: mode === m ? "#fff" : "var(--text)" }}>
                {m === "auto" ? "Auto" : "Expert"}
              </button>
            ))}
          </div>
          <div className="text-xs mb-4" style={{ color: "var(--muted)" }}>
            {mode === "auto"
              ? "Réponse simple et directe. Idéal au quotidien — périmètre choisi automatiquement."
              : "Réponse approfondie : détaillée, structurée, avec citations et sources complètes."}
          </div>

          {mode === "expert" && cfg && (
            <div className="space-y-3 mb-4">
              <div>
                <div className="text-xs mb-1" style={{ color: "var(--muted)" }}>Périmètre (optionnel)</div>
                <select value={org} onChange={(e) => setOrg(e.target.value)}
                        aria-label="Périmètre institutionnel"
                        className="w-full rounded-lg px-3 py-2 text-sm"
                        style={{ background: "var(--bg-elev)", border: "1px solid var(--border)", color: "var(--text)" }}>
                  {cfg.orgs.map((o) => (
                    <option key={o.id} value={o.id}>{o.emoji} {o.label}</option>
                  ))}
                </select>
              </div>
            </div>
          )}
        </div>

        {/* Historique des conversations */}
        <div className="flex-1 overflow-y-auto px-2 mt-1">
          <div className="text-xs px-2 mb-1" style={{ color: "var(--muted)" }}>Conversations</div>
          {[...convos].sort((a, b) => b.updatedAt - a.updatedAt).map((c) => (
            <div key={c.id} onClick={() => editingId !== c.id && selectConvo(c.id)}
                 className="group flex items-center gap-1 rounded-lg px-2 py-2 cursor-pointer text-sm"
                 style={{ background: c.id === activeId ? "var(--bg-elev)" : "transparent" }}>
              <span className="opacity-60">💬</span>
              {editingId === c.id ? (
                <input autoFocus value={editVal}
                       onChange={(e) => setEditVal(e.target.value)}
                       onBlur={() => renameConvo(c.id, editVal)}
                       onKeyDown={(e) => { if (e.key === "Enter") renameConvo(c.id, editVal); if (e.key === "Escape") setEditingId(""); }}
                       onClick={(e) => e.stopPropagation()}
                       className="flex-1 min-w-0 bg-transparent outline-none border-b"
                       style={{ color: "var(--text)", borderColor: "var(--accent)" }} />
              ) : (
                <span className="flex-1 truncate"
                      onDoubleClick={(e) => { e.stopPropagation(); setEditingId(c.id); setEditVal(c.title); }}>
                  {c.title || "Nouvelle conversation"}
                </span>
              )}
              <button onClick={(e) => { e.stopPropagation(); setEditingId(c.id); setEditVal(c.title); }}
                      title="Renommer" aria-label="Renommer la conversation"
                      className="opacity-0 group-hover:opacity-60 hover:!opacity-100 px-0.5"
                      style={{ color: "var(--muted)" }}>✏️</button>
              {confirmDel === c.id ? (
                <>
                  <button onClick={(e) => { e.stopPropagation(); deleteConvo(c.id); setConfirmDel(""); }}
                          title="Confirmer la suppression" aria-label="Confirmer la suppression"
                          className="px-1 font-semibold" style={{ color: "#e74c3c" }}>✓</button>
                  <button onClick={(e) => { e.stopPropagation(); setConfirmDel(""); }}
                          title="Annuler" aria-label="Annuler la suppression"
                          className="px-1" style={{ color: "var(--muted)" }}>✕</button>
                </>
              ) : (
                <button onClick={(e) => { e.stopPropagation(); setConfirmDel(c.id); }}
                        title="Supprimer" aria-label="Supprimer la conversation"
                        className="opacity-0 group-hover:opacity-60 hover:!opacity-100 px-1"
                        style={{ color: "var(--muted)" }}>×</button>
              )}
            </div>
          ))}
        </div>

        <div className="p-4 space-y-3" style={{ borderTop: "1px solid var(--border)" }}>
          <label className="flex items-center justify-between text-sm">
            <span>🔊 Réponse vocale</span>
            <input type="checkbox" checked={voiceReply} onChange={(e) => setVoiceReply(e.target.checked)} />
          </label>
          <button onClick={() => setShowPicker(true)} className="text-sm w-full text-left" style={{ color: "var(--muted)" }}>
            🎙️ Changer de voix {cfg && <span>({cfg.voices.find((v) => v.id === voice)?.label})</span>}
          </button>
          <button onClick={cycleTheme} aria-label="Changer de thème" className="text-sm w-full text-left" style={{ color: "var(--muted)" }}>
            {theme === "system" ? "🖥️ Thème : Système" : theme === "light" ? "☀️ Thème : Clair" : "🌙 Thème : Sombre"}
          </button>
        </div>
      </aside>

      {sidebar && <div className="fixed inset-0 z-30 md:hidden" style={{ background: "rgba(0,0,0,.4)" }} onClick={() => setSidebar(false)} />}

      {/* ── Zone principale ── */}
      <main className="flex-1 flex flex-col min-w-0">
        <header className="flex items-center gap-3 p-3 md:hidden" style={{ borderBottom: "1px solid var(--border)" }}>
          <button onClick={() => setSidebar(true)} className="text-xl" aria-label="Ouvrir le menu">☰</button>
          <div className="font-bold text-lg lowercase" style={{ color: "var(--accent)" }}>alélo</div>
        </header>

        {cfgError && (
          <div className="px-4 py-2 text-sm text-center"
               style={{ background: "#fdecea", color: "#b02a1a", borderBottom: "1px solid #f5c6cb" }}>
            ⚠️ Impossible de contacter le service alélo. Vérifiez qu&apos;il est démarré, puis rechargez la page.
          </div>
        )}

        <div ref={scrollRef} onScroll={onScroll} className="flex-1 overflow-y-auto relative">
          <div className="max-w-3xl mx-auto w-full px-4">
            {empty ? (
              <div className="h-full flex flex-col items-center justify-center text-center py-20">
                <div className="mb-5"><AgentAvatar size={76} /></div>
                <h1 className="text-2xl font-semibold mb-2">Quel est votre besoin ?</h1>
                <p className="text-sm mb-8 max-w-md" style={{ color: "var(--muted)" }}>
                  Posez votre question comme vous la pensez — par écrit ou à la voix.
                  Je comprends votre situation et je vous oriente vers la bonne réponse.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-xl">
                  {SUGGESTIONS.map((s) => (
                    <button key={s} onClick={() => send(s)}
                            className="text-left rounded-xl px-4 py-3 text-sm hover:opacity-80"
                            style={{ background: "var(--bg-elev)", border: "1px solid var(--border)" }}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="py-4" aria-live="polite">
                {messages.map((m, i) => (
                  <Message key={i} msg={m} voice={voice}
                           question={m.role === "assistant" ? messages[i - 1]?.content : undefined}
                           streaming={m.role === "assistant" && i === messages.length - 1 && busy && streamingRef.current}
                           canRegenerate={m.role === "assistant" && i === messages.length - 1 && !busy}
                           onRegenerate={regenerate} />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* ── Composer ── */}
        <div className="px-4 pb-4 pt-2 relative">
          {showScrollBtn && !empty && (
            <button onClick={scrollToBottom} aria-label="Défiler vers le bas" title="Défiler vers le bas"
                    className="absolute left-1/2 -translate-x-1/2 -top-8 w-9 h-9 rounded-full flex items-center justify-center shadow-md z-10"
                    style={{ background: "var(--panel)", border: "1px solid var(--border)", color: "var(--text)" }}>↓</button>
          )}
          <div className="max-w-3xl mx-auto">
            <div className="flex items-end gap-2 rounded-2xl px-3 py-2"
                 style={{ background: "var(--bg-elev)", border: "1px solid var(--border)" }}>
              <button onClick={toggleMic} disabled={busy && !recording}
                      className="flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center relative"
                      title="Parler"
                      style={{ background: recording ? "#e74c3c" : "var(--panel)", border: "1px solid var(--border)" }}>
                {recording && (
                  <span className="absolute inset-0 rounded-full"
                        style={{ background: "#e74c3c", animation: "pulse-ring 1s ease-out infinite",
                                 transform: `scale(${1 + Math.min(micLevel * 4, 0.8)})`, opacity: 0.4 }} />
                )}
                <span className="relative">{recording ? "⏹️" : "🎤"}</span>
              </button>
              <textarea
                ref={taRef}
                value={input} rows={1}
                aria-label="Votre message"
                onChange={(e) => { setInput(e.target.value); autosize(); }}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); } }}
                placeholder={recording ? "Je vous écoute…" : "Écrivez votre message…"}
                className="flex-1 resize-none bg-transparent outline-none py-2 max-h-40"
                style={{ color: "var(--text)" }} />
              {busy ? (
                <button onClick={stopGen} aria-label="Arrêter la génération" title="Arrêter"
                        className="flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center text-white"
                        style={{ background: "var(--accent)" }}>⏹</button>
              ) : (
                <button onClick={() => send(input)} disabled={!input.trim()} aria-label="Envoyer"
                        className="flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center text-white disabled:opacity-40"
                        style={{ background: "var(--accent)" }}>➤</button>
              )}
            </div>
            <div className="text-center text-xs mt-2" style={{ color: "var(--muted)" }}>
              {mode === "auto" ? "Mode Auto · réponse simple" : `Mode Expert · approfondi · ${cfg?.orgs.find(o => o.id === org)?.label ?? "Tout"}`}
              {" · "}100% local
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
