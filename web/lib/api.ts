// Client de l'API RAG (chat SSE, transcription, TTS, config).
const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8088";

// URL absolue vers l'API (pour les liens de téléchargement de documents).
export function apiUrl(path: string): string {
  return path.startsWith("http") ? path : `${BASE}${path}`;
}

export type DocInfo = { id: string; filename: string; format: string; label: string; url: string };

export type Msg = { role: "user" | "assistant"; content: string; sources?: Source[]; ts?: number; document?: DocInfo; steps?: string[] };
export type Source = { url: string; title: string; excerpt?: string };
export type VoiceInfo = { id: string; label: string; genre: string };
export type ModelInfo = { id: string; label: string; flag?: string; recommended?: boolean };
export type OrgInfo = { id: string; label: string; emoji?: string };
export type Config = {
  models: ModelInfo[];
  orgs: OrgInfo[];
  voices: VoiceInfo[];
  defaultVoice: string;
  defaultModel: string;
};

export async function getConfig(): Promise<Config> {
  const r = await fetch(`${BASE}/api/config`);
  if (!r.ok) throw new Error("config");
  return r.json();
}

export type ChatBody = {
  message: string;
  history: { role: string; content: string }[];
  mode: "auto" | "expert";
  org?: string | null;
  model?: string;
};

// Génère les évènements du flux : {type:"token"|"done"|"error", ...}
export async function* chatStream(body: ChatBody, signal?: AbortSignal): AsyncGenerator<any> {
  const r = await fetch(`${BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.body) throw new Error("no stream");
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() || "";
    for (const p of parts) {
      const line = p.trim();
      if (line.startsWith("data:")) {
        try {
          yield JSON.parse(line.slice(5).trim());
        } catch {}
      }
    }
  }
}

export async function transcribe(wav: Blob): Promise<string> {
  const fd = new FormData();
  fd.append("audio", wav, "turn.wav");
  const r = await fetch(`${BASE}/api/transcribe`, { method: "POST", body: fd });
  const d = await r.json();
  return (d.text || "").trim();
}

export async function sendFeedback(p: {
  rating: "up" | "down"; question: string; answer: string; comment?: string; sources?: Source[];
}): Promise<void> {
  try {
    await fetch(`${BASE}/api/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(p),
    });
  } catch {}
}

export async function tts(text: string, voice: string): Promise<Blob> {
  const r = await fetch(`${BASE}/api/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice }),
  });
  return r.blob();
}
