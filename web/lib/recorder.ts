// Enregistreur micro → WAV (débit natif, normalisé). Détection de fin de parole optionnelle.
function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buf);
  const w = (o: number, s: string) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); w(8, "WAVE");
  w(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  w(36, "data"); view.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true); off += 2;
  }
  return new Blob([view], { type: "audio/wav" });
}

export class Recorder {
  private ctx?: AudioContext;
  private stream?: MediaStream;
  private src?: MediaStreamAudioSourceNode;
  private proc?: ScriptProcessorNode;
  private chunks: Float32Array[] = [];
  private recording = false;

  // VAD
  private speechStarted = false;
  private lastVoice = 0;
  private speechStart = 0;
  private uttStart = 0;

  onLevel?: (level: number) => void;
  onAutoStop?: () => void;

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    this.ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    if (this.ctx.state === "suspended") await this.ctx.resume();
    this.src = this.ctx.createMediaStreamSource(this.stream);
    this.proc = this.ctx.createScriptProcessor(4096, 1, 1);
    this.chunks = [];
    this.speechStarted = false; this.lastVoice = 0; this.speechStart = 0;
    this.uttStart = performance.now();
    this.recording = true;

    const START = 0.014, STOP = 0.009, SILENCE_MS = 1400, MIN_SPEECH_MS = 700, MAX_MS = 15000;
    this.proc.onaudioprocess = (e) => {
      if (!this.recording) return;
      const input = e.inputBuffer.getChannelData(0);
      this.chunks.push(new Float32Array(input));
      let sum = 0;
      for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
      const rms = Math.sqrt(sum / input.length);
      this.onLevel?.(rms);
      const now = performance.now();
      if (rms > START) {
        if (!this.speechStarted) { this.speechStarted = true; this.speechStart = now; }
        this.lastVoice = now;
      }
      const silenceFor = now - this.lastVoice;
      const spokeEnough = this.speechStarted && now - this.speechStart > MIN_SPEECH_MS;
      if ((spokeEnough && rms < STOP && silenceFor > SILENCE_MS) ||
          (this.speechStarted && now - this.uttStart > MAX_MS)) {
        this.onAutoStop?.();
      }
    };
    this.src.connect(this.proc);
    this.proc.connect(this.ctx.destination);
  }

  stop(): Blob | null {
    if (!this.recording || !this.ctx) return null;
    this.recording = false;
    try { this.proc?.disconnect(); this.src?.disconnect(); this.stream?.getTracks().forEach((t) => t.stop()); } catch {}
    let total = this.chunks.reduce((n, c) => n + c.length, 0);
    const merged = new Float32Array(total);
    let off = 0;
    for (const c of this.chunks) { merged.set(c, off); off += c.length; }
    // Normalisation du niveau
    let peak = 0;
    for (let i = 0; i < merged.length; i++) { const a = Math.abs(merged[i]); if (a > peak) peak = a; }
    if (peak > 0.001 && peak < 0.5) {
      const gain = Math.min(0.9 / peak, 8);
      for (let i = 0; i < merged.length; i++) merged[i] *= gain;
    }
    const rate = this.ctx.sampleRate;
    this.ctx.close();
    if (merged.length < rate * 0.3) return null; // trop court
    return encodeWav(merged, rate);
  }
}
