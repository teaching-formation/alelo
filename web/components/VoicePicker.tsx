"use client";
import { useState } from "react";
import { VoiceInfo, tts } from "@/lib/api";

export default function VoicePicker({
  voices, current, onValidate,
}: {
  voices: VoiceInfo[];
  current: string;
  onValidate: (id: string) => void;
}) {
  const [sel, setSel] = useState(current || voices[0]?.id);
  const [playing, setPlaying] = useState<string | null>(null);

  async function preview(id: string) {
    setPlaying(id);
    try {
      const blob = await tts("Bonjour, je suis alélo, votre assistant public. Comment puis-je vous aider ?", id);
      const url = URL.createObjectURL(blob);
      const a = new Audio(url);
      a.onended = () => { setPlaying(null); URL.revokeObjectURL(url); };
      a.play().catch(() => setPlaying(null));
    } catch { setPlaying(null); }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
         style={{ background: "rgba(0,0,0,.55)", backdropFilter: "blur(3px)" }}>
      <div role="dialog" aria-modal="true" aria-label="Choix de la voix"
           className="w-full max-w-md rounded-2xl p-6 shadow-2xl"
           style={{ background: "var(--bg-elev)", border: "1px solid var(--border)" }}>
        <div className="text-lg font-semibold mb-1">Choisissez votre voix 🎙️</div>
        <div className="text-sm mb-4" style={{ color: "var(--muted)" }}>
          L&apos;assistant vous répondra avec cette voix. Vous pourrez la changer plus tard.
        </div>
        <div className="space-y-2">
          {voices.map((v) => (
            <div key={v.id}
                 onClick={() => setSel(v.id)}
                 className="flex items-center justify-between rounded-xl px-4 py-3 cursor-pointer transition"
                 style={{
                   border: `1.5px solid ${sel === v.id ? "var(--accent)" : "var(--border)"}`,
                   background: sel === v.id ? "var(--user-bubble)" : "transparent",
                 }}>
              <div className="flex items-center gap-3">
                <span className="text-xl">{v.genre === "homme" ? "👨" : "👩"}</span>
                <div>
                  <div className="font-medium">{v.label}</div>
                  <div className="text-xs" style={{ color: "var(--muted)" }}>Voix {v.genre}</div>
                </div>
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); preview(v.id); }}
                className="rounded-full px-3 py-1.5 text-sm font-medium"
                style={{ background: "var(--panel)", border: "1px solid var(--border)" }}>
                {playing === v.id ? "🔊 …" : "▶ Écouter"}
              </button>
            </div>
          ))}
        </div>
        <button
          onClick={() => onValidate(sel)}
          className="mt-5 w-full rounded-xl py-3 font-semibold text-white"
          style={{ background: "var(--accent)" }}>
          Valider
        </button>
      </div>
    </div>
  );
}
