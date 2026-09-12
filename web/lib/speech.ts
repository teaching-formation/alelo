// File d'attente de synthèse vocale PROGRESSIVE : on synthétise et on joue
// phrase par phrase au fil du streaming → l'audio démarre après la 1re phrase,
// pas après toute la réponse (voice-first, faible latence).
import { tts } from "@/lib/api";

const MIN_CHARS = 6; // ignore les fragments trop courts ("1.", "-", …)
const alnum = (s: string) => s.replace(/[^0-9A-Za-zÀ-ÿ]/g, "").length;

export class SpeechQueue {
  private voice: string;
  private buf = "";
  private queue: string[] = [];
  private playing = false;
  private closed = false;
  private started = false;
  private ctx?: AudioContext;
  private raf: number | null = null;
  private resolveDone!: () => void;
  private donePromise: Promise<void>;

  onLevel?: (l: number) => void;
  onStart?: () => void;

  constructor(voice: string) {
    this.voice = voice;
    this.donePromise = new Promise((res) => (this.resolveDone = res));
  }

  /** Ajoute du texte en flux ; extrait et met en file les phrases complètes. */
  push(chunk: string) {
    this.buf += chunk;
    while (true) {
      const m = this.buf.match(/^[\s\S]*?[.!?…\n]+/);
      if (!m) break;
      const sent = m[0];
      this.buf = this.buf.slice(sent.length);
      const clean = sent.trim();
      if (alnum(clean) >= MIN_CHARS) { this.queue.push(clean); this.pump(); }
    }
  }

  /** Signale la fin du flux ; joue le reste et résout done(). */
  finish() {
    const rest = this.buf.trim();
    this.buf = "";
    if (alnum(rest) >= 2) this.queue.push(rest);
    this.closed = true;
    this.pump();
    if (!this.playing && this.queue.length === 0) this.resolveDone();
  }

  done(): Promise<void> { return this.donePromise; }

  stop() {
    this.closed = true;
    this.queue = [];
    if (this.raf) cancelAnimationFrame(this.raf);
    try { this.ctx?.close(); } catch {}
    this.resolveDone();
  }

  private async pump() {
    if (this.playing) return;
    const text = this.queue.shift();
    if (text === undefined) { if (this.closed) this.resolveDone(); return; }
    this.playing = true;
    if (!this.started) { this.started = true; this.onStart?.(); }
    try {
      const blob = await tts(text, this.voice);
      if (!this.closed || this.queue.length >= 0) await this.play(blob);
    } catch {}
    this.playing = false;
    if (this.queue.length) this.pump();
    else if (this.closed) this.resolveDone();
  }

  private play(blob: Blob): Promise<void> {
    return new Promise((resolve) => {
      const url = URL.createObjectURL(blob);
      const el = new Audio(url);
      try {
        if (!this.ctx) this.ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
        const ctx = this.ctx;
        const src = ctx.createMediaElementSource(el);
        const an = ctx.createAnalyser(); an.fftSize = 512;
        src.connect(an); an.connect(ctx.destination);
        const bufA = new Uint8Array(an.frequencyBinCount);
        const tick = () => {
          an.getByteTimeDomainData(bufA);
          let sum = 0;
          for (let i = 0; i < bufA.length; i++) { const v = (bufA[i] - 128) / 128; sum += v * v; }
          this.onLevel?.(Math.sqrt(sum / bufA.length));
          this.raf = requestAnimationFrame(tick);
        };
        el.onended = () => { if (this.raf) cancelAnimationFrame(this.raf); this.onLevel?.(0); URL.revokeObjectURL(url); resolve(); };
        el.onerror = () => { URL.revokeObjectURL(url); resolve(); };
        el.play().then(() => tick()).catch(() => resolve());
      } catch {
        el.onended = () => resolve();
        el.play().catch(() => resolve());
      }
    });
  }
}
