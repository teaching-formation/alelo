"use client";
import { useEffect, useRef, useState } from "react";
import { Recorder } from "@/lib/recorder";
import { transcribe, chatStream } from "@/lib/api";
import { SpeechQueue } from "@/lib/speech";
import AgentAvatar from "@/components/AgentAvatar";

type Status = "listening" | "thinking" | "speaking";
type Turn = { role: string; content: string };

// Retire le markdown pour un affichage vocal propre (pas de ** ni de [lien](url)).
function stripMd(s: string): string {
  return s
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")   // [texte](url) → texte
    .replace(/https?:\/\/\S+/g, "")             // URLs nues
    .replace(/[*_`#>]/g, "")                     // symboles markdown
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\n{2,}/g, "\n")
    .trim();
}

export default function VoiceMode({
  voice, org, history, onTurn, onClose,
}: {
  voice: string;
  org?: string | null;
  history?: Turn[];
  onTurn?: (user: string, assistant: string, sources: any[], document?: any) => void;
  onClose: () => void;
}) {
  const [status, setStatus] = useState<Status>("listening");
  const [transcript, setTranscript] = useState("");
  const [answer, setAnswer] = useState("");
  const [level, setLevel] = useState(0);
  const [micError, setMicError] = useState(false);

  const activeRef = useRef(true);
  const recRef = useRef<Recorder | null>(null);
  const histRef = useRef<Turn[]>((history || []).map((m) => ({ role: m.role, content: m.content })));
  const sqRef = useRef<SpeechQueue | null>(null);

  useEffect(() => {
    activeRef.current = true;
    startListening();
    return () => {
      activeRef.current = false;
      try { recRef.current?.stop(); } catch {}
      try { sqRef.current?.stop(); } catch {}
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function startListening() {
    if (!activeRef.current) return;
    setTranscript(""); setAnswer(""); setStatus("listening"); setLevel(0); setMicError(false);
    const rec = new Recorder();
    rec.onLevel = (l) => setLevel(l);
    rec.onAutoStop = () => { if (recRef.current === rec) finishTurn(); };
    recRef.current = rec;
    try { await rec.start(); }
    catch { setMicError(true); }
  }

  async function finishTurn() {
    const rec = recRef.current;
    if (!rec) return;
    const wav = rec.stop(); recRef.current = null; setLevel(0);
    if (!wav) { if (activeRef.current) startListening(); return; }

    setStatus("thinking");
    let text = "";
    try { text = await transcribe(wav); } catch {}
    if (!text) { if (activeRef.current) startListening(); return; }
    setTranscript(text);
    histRef.current.push({ role: "user", content: text });

    // TTS progressive : on joue phrase par phrase pendant que la réponse se génère.
    const sq = new SpeechQueue(voice);
    sq.onLevel = (l) => setLevel(l);
    sq.onStart = () => { if (activeRef.current) setStatus("speaking"); };
    sqRef.current = sq;

    let ans = "";
    let sources: any[] = [];
    let document: any = undefined;
    try {
      for await (const ev of chatStream({ message: text, history: histRef.current, mode: "auto", org })) {
        if (ev.type === "token") { ans += ev.text; setAnswer(ans); sq.push(ev.text); }
        else if (ev.type === "done") { ans = ev.answer || ans; sources = ev.sources || []; document = ev.document; }
      }
    } catch { ans = ans || "Désolé, une erreur s'est produite."; }
    sq.finish();

    histRef.current.push({ role: "assistant", content: ans });
    if (histRef.current.length > 12) histRef.current = histRef.current.slice(-12);
    // Persiste le tour comme une vraie conversation (texte + sources + éventuel document).
    onTurn?.(text, ans, sources, document);

    await sq.done();
    sqRef.current = null;
    if (activeRef.current) startListening();
  }

  const scale = 1 + Math.min(level * 2.2, 0.5);
  const statusText = status === "listening" ? "Je vous écoute…"
    : status === "thinking" ? "Réflexion…" : "alélo répond…";

  return (
    <div role="dialog" aria-modal="true" aria-label="Mode vocal"
         className="fixed inset-0 z-50 flex flex-col items-center justify-center px-6"
         style={{ background: "var(--bg)" }}>
      <button onClick={onClose} aria-label="Fermer le mode vocal"
              className="absolute top-5 left-5 text-sm" style={{ color: "var(--muted)" }}>
        ← Retour au chat
      </button>

      {/* Tête animée */}
      <div className="relative flex items-center justify-center" style={{ width: 240, height: 240 }}>
        <div className="absolute rounded-full"
             style={{
               inset: -30,
               background: "radial-gradient(circle, rgba(107,140,255,.35), transparent 65%)",
               transform: `scale(${1 + level * 1.6})`,
               transition: "transform .1s ease-out",
               opacity: status === "thinking" ? 0.5 : 1,
             }} />
        <div style={{ transform: `scale(${scale})`, transition: "transform .1s ease-out" }}
             className={status === "thinking" ? "animate-pulse" : ""}>
          <AgentAvatar size={170} />
        </div>
      </div>

      {micError ? (
        <div className="mt-8 flex flex-col items-center gap-3">
          <div className="text-base" style={{ color: "#e74c3c" }}>🎤 Micro non autorisé</div>
          <div className="text-sm text-center max-w-sm" style={{ color: "var(--muted)" }}>
            Autorisez l&apos;accès au microphone dans votre navigateur, puis réessayez.
          </div>
          <button onClick={startListening}
                  className="rounded-full px-5 py-2 font-medium text-white" style={{ background: "var(--accent)" }}>
            Réessayer
          </button>
        </div>
      ) : (
        <div className="mt-8 text-base" style={{ color: "var(--muted)" }}>{statusText}</div>
      )}
      {transcript && (
        <div className="mt-4 max-w-xl text-center italic" style={{ color: "var(--muted)" }}>
          « {transcript} »
        </div>
      )}
      {answer && (
        <div className="mt-3 max-w-xl text-center text-lg leading-relaxed max-h-[38vh] overflow-y-auto whitespace-pre-line" style={{ color: "var(--text)" }}>
          {stripMd(answer)}
        </div>
      )}

      <div className="absolute bottom-10 text-xs" style={{ color: "var(--muted)" }}>
        Parlez naturellement — je détecte quand vous avez fini.
      </div>
      <button onClick={onClose}
              className="absolute bottom-20 rounded-full px-6 py-3 font-semibold text-white"
              style={{ background: "#e74c3c" }}>
        ⏹️ Terminer
      </button>
    </div>
  );
}
