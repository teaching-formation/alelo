"use client";
import { useState } from "react";
import { Msg, tts, apiUrl, sendFeedback } from "@/lib/api";
import AgentAvatar from "@/components/AgentAvatar";

const DOC_ICON: Record<string, string> = { pdf: "📕", docx: "📘", xlsx: "📗" };

// Temps relatif « il y a … » (fr).
function relativeTime(ts?: number): string {
  if (!ts) return "";
  const s = Math.floor((Date.now() - ts) / 1000);
  if (s < 45) return "à l'instant";
  const m = Math.floor(s / 60);
  if (m < 60) return `il y a ${m} min`;
  const h = Math.floor(m / 60);
  if (h < 24) return `il y a ${h} heure${h > 1 ? "s" : ""}`;
  const d = Math.floor(h / 24);
  return `il y a ${d} jour${d > 1 ? "s" : ""}`;
}

// Icônes ligne (style sobre, currentColor).
const svg = (p: string) => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" dangerouslySetInnerHTML={{ __html: p }} />
);
const IconCopy = svg('<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/>');
const IconCheck = svg('<path d="M20 6 9 17l-5-5"/>');
const IconSpeaker = svg('<path d="M11 5 6 9H2v6h4l5 4z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M18.5 5.5a9 9 0 0 1 0 13"/>');
const IconUp = svg('<path d="M7 10v11"/><path d="M18 21H5V10l5-8a2 2 0 0 1 3 2l-1 6h6a2 2 0 0 1 2 2.4l-1.6 7A2 2 0 0 1 16.5 21z"/>');
const IconDown = svg('<path d="M17 14V3"/><path d="M6 3h13v11l-5 8a2 2 0 0 1-3-2l1-6H6a2 2 0 0 1-2-2.4l1.6-7A2 2 0 0 1 7.5 3z"/>');
const IconRefresh = svg('<path d="M21 12a9 9 0 1 1-2.6-6.4"/><path d="M21 3v6h-6"/>');

// Rendu markdown minimal (titres, gras, italique, code, listes, liens) — sans dépendance.
// esc() échappe AUSSI les guillemets : sans ça, une URL contenant " casse l'attribut href (XSS).
function renderMarkdown(text: string): string {
  const esc = (s: string) => s
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  const inline = (s: string) =>
    s.replace(/`([^`]+)`/g, "<code>$1</code>")
     .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
     .replace(/\*(?!\s)([^*\n]+?)\*/g, "<em>$1</em>")
     .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');

  const lines = esc(text).split("\n");
  let html = "", inUl = false, inOl = false, inCode = false, codeBuf = "";
  const closeLists = () => {
    if (inUl) { html += "</ul>"; inUl = false; }
    if (inOl) { html += "</ol>"; inOl = false; }
  };
  for (const raw of lines) {
    if (raw.trim().startsWith("```")) {
      if (inCode) { html += `<pre><code>${codeBuf}</code></pre>`; codeBuf = ""; inCode = false; }
      else { closeLists(); inCode = true; }
      continue;
    }
    if (inCode) { codeBuf += (codeBuf ? "\n" : "") + raw; continue; }
    const l = raw.trim();
    const h = l.match(/^(#{1,4})\s+(.*)/);
    const ul = l.match(/^[-*•]\s+(.*)/);
    const ol = l.match(/^\d+[.)]\s+(.*)/);
    if (h) { closeLists(); html += `<h3>${inline(h[2])}</h3>`; }
    else if (ul) {
      if (inOl) { html += "</ol>"; inOl = false; }
      if (!inUl) { html += "<ul>"; inUl = true; }
      html += `<li>${inline(ul[1])}</li>`;
    } else if (ol) {
      if (inUl) { html += "</ul>"; inUl = false; }
      if (!inOl) { html += "<ol>"; inOl = true; }
      html += `<li>${inline(ol[1])}</li>`;
    } else {
      closeLists();
      if (l) html += `<p>${inline(l)}</p>`;
    }
  }
  if (inCode) html += `<pre><code>${codeBuf}</code></pre>`;
  closeLists();
  return html;
}

export default function Message({ msg, voice, streaming, canRegenerate, onRegenerate, question }:
  { msg: Msg; voice: string; streaming?: boolean; canRegenerate?: boolean; onRegenerate?: () => void; question?: string }) {
  const [playing, setPlaying] = useState(false);
  const [copied, setCopied] = useState(false);
  const [rated, setRated] = useState<"up" | "down" | null>(null);
  const [askWhy, setAskWhy] = useState(false);
  const isUser = msg.role === "user";

  function rate(r: "up" | "down", comment?: string) {
    setRated(r);
    sendFeedback({ rating: r, question: question || "", answer: msg.content,
                   comment, sources: msg.sources });
    setAskWhy(r === "down" && !comment);   // 👎 → propose de dire pourquoi
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(msg.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {}
  }

  async function speak() {
    if (playing) return;
    setPlaying(true);
    try {
      const blob = await tts(msg.content, voice);
      const url = URL.createObjectURL(blob);
      const a = new Audio(url);
      a.onended = () => { setPlaying(false); URL.revokeObjectURL(url); };
      a.play().catch(() => setPlaying(false));
    } catch { setPlaying(false); }
  }

  return (
    <div className="flex gap-3 py-4 px-1">
      <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm overflow-hidden"
           style={{ background: isUser ? "var(--accent)" : "transparent", border: isUser ? "1px solid var(--border)" : "none" }}>
        {isUser ? "🧑" : <AgentAvatar size={32} />}
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-xs mb-1" style={{ color: "var(--muted)" }}>
          {isUser ? "Vous" : "alélo"}
        </div>
        {isUser ? (
          <div className="prose-chat whitespace-pre-wrap break-words">{msg.content}</div>
        ) : (
          <>
            {/* Étapes agentiques (Mode Expert) : en direct pendant la génération, repliées ensuite */}
            {msg.steps && msg.steps.length > 0 && (
              streaming ? (
                <div className="steps-live mb-2">
                  {msg.steps.map((s, i) => (
                    <div key={i} className={`step-line${i === msg.steps!.length - 1 && !msg.content ? " active" : ""}`}>
                      <span className="step-dot" />{s}
                    </div>
                  ))}
                </div>
              ) : (
                <details className="text-xs mb-2" style={{ color: "var(--muted)" }}>
                  <summary className="cursor-pointer">🔎 Démarche · {msg.steps.length} étape{msg.steps.length > 1 ? "s" : ""}</summary>
                  <div className="mt-1 space-y-0.5 pl-1">
                    {msg.steps.map((s, i) => <div key={i}>{s}</div>)}
                  </div>
                </details>
              )
            )}
            {streaming && !msg.content ? (
              (!msg.steps || msg.steps.length === 0) ? (
                <div className="typing-dots" aria-label="alélo écrit…"><span /><span /><span /></div>
              ) : null
            ) : (
              <div className={`prose-chat break-words ${streaming ? "cursor-blink" : ""}`}
                   dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }} />
            )}
          </>
        )}

        {msg.document && !streaming && (
          <a href={apiUrl(msg.document.url)} target="_blank" rel="noreferrer"
             className="mt-3 inline-flex items-center gap-3 rounded-xl px-4 py-3 hover:opacity-90 transition"
             style={{ background: "var(--bg-elev)", border: "1px solid var(--border)" }}>
            <span className="text-2xl">{DOC_ICON[msg.document.format] || "📄"}</span>
            <span className="flex flex-col min-w-0">
              <span className="text-sm font-medium truncate" style={{ color: "var(--text)" }}>{msg.document.filename}</span>
              <span className="text-xs" style={{ color: "var(--accent)" }}>⬇ Télécharger le {msg.document.label}</span>
            </span>
          </a>
        )}

        {!isUser && !streaming && msg.content && (
          <>
            <div className="flex items-center gap-1 mt-2 -ml-1" style={{ color: "var(--muted)" }}>
              <button onClick={copy} title="Copier" aria-label="Copier"
                      className="p-1.5 rounded-md hover:bg-black/5 dark:hover:bg-white/10">
                {copied ? IconCheck : IconCopy}
              </button>
              <button onClick={speak} title="Écouter" aria-label="Écouter"
                      className="p-1.5 rounded-md hover:bg-black/5 dark:hover:bg-white/10"
                      style={{ color: playing ? "var(--accent)" : undefined }}>
                {IconSpeaker}
              </button>
              <button onClick={() => rate("up")} title="Bonne réponse" aria-label="Bonne réponse"
                      className="p-1.5 rounded-md hover:bg-black/5 dark:hover:bg-white/10"
                      style={{ color: rated === "up" ? "var(--accent)" : undefined }}>
                {IconUp}
              </button>
              <button onClick={() => rate("down")} title="Réponse à améliorer" aria-label="Réponse à améliorer"
                      className="p-1.5 rounded-md hover:bg-black/5 dark:hover:bg-white/10"
                      style={{ color: rated === "down" ? "#e74c3c" : undefined }}>
                {IconDown}
              </button>
              {canRegenerate && onRegenerate && (
                <button onClick={onRegenerate} title="Régénérer" aria-label="Régénérer"
                        className="p-1.5 rounded-md hover:bg-black/5 dark:hover:bg-white/10">
                  {IconRefresh}
                </button>
              )}
              {msg.ts ? <span className="text-xs ml-1.5 opacity-70">{relativeTime(msg.ts)}</span> : null}
            </div>

            {askWhy && (
              <div className="mt-2 flex items-center gap-2">
                <input autoFocus placeholder="Qu'est-ce qui n'allait pas ? (optionnel)"
                       onKeyDown={(e) => {
                         if (e.key === "Enter") { rate("down", (e.target as HTMLInputElement).value); setAskWhy(false); }
                         if (e.key === "Escape") setAskWhy(false);
                       }}
                       className="text-xs rounded-lg px-3 py-1.5 w-full max-w-sm outline-none"
                       style={{ background: "var(--bg-elev)", border: "1px solid var(--border)", color: "var(--text)" }} />
                <button onClick={() => setAskWhy(false)} className="text-xs" style={{ color: "var(--muted)" }}>Ignorer</button>
              </div>
            )}
            {rated && !askWhy && (
              <div className="text-xs mt-1" style={{ color: "var(--muted)" }}>Merci, c'est noté 🙏</div>
            )}

            {msg.sources && msg.sources.length > 0 && (
              <details className="text-xs mt-2" style={{ color: "var(--muted)" }}>
                <summary className="cursor-pointer">📎 {msg.sources.length} source(s)</summary>
                <div className="mt-1 space-y-1">
                  {msg.sources.slice(0, 4).map((s, i) => (
                    <a key={i} href={s.url} target="_blank" rel="noreferrer"
                       className="block hover:underline" style={{ color: "var(--accent)" }}>
                      {s.title || s.url}
                    </a>
                  ))}
                </div>
              </details>
            )}
          </>
        )}
      </div>
    </div>
  );
}
