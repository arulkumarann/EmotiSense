import { useCallback, useEffect, useRef, useState } from "react";
import { transcodeToWav } from "../lib/wav";

export type RecorderStatus = "idle" | "requesting" | "recording" | "processing" | "done" | "error";

const MIME_CANDIDATES = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];

/** Microphone recording that ends as a 16 kHz mono WAV file, plus a live input level for visuals. */
export function useRecorder() {
  const [status, setStatus] = useState<RecorderStatus>("idle");
  const [elapsed, setElapsed] = useState(0);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const level = useRef(0);

  const recorder = useRef<MediaRecorder | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const chunks = useRef<Blob[]>([]);
  const startedAt = useRef(0);
  const timer = useRef(0);
  const raf = useRef(0);
  const audioCtx = useRef<AudioContext | null>(null);

  const cleanupAudio = useCallback(() => {
    cancelAnimationFrame(raf.current);
    window.clearInterval(timer.current);
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    void audioCtx.current?.close();
    audioCtx.current = null;
    level.current = 0;
  }, []);

  useEffect(() => () => cleanupAudio(), [cleanupAudio]);
  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);

  const reset = useCallback(() => {
    cleanupAudio();
    setStatus("idle");
    setElapsed(0);
    setFile(null);
    setPreviewUrl(null);
    setError(null);
  }, [cleanupAudio]);

  const start = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser can't record audio. Try Chrome, Edge or Firefox over localhost.");
      setStatus("error");
      return;
    }
    setStatus("requesting");
    setError(null);
    try {
      // Keep the signal natural: processing that flattens loudness or removes "noise" can flatten emotion too.
      const media = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } });
      stream.current = media;
      const mimeType = MIME_CANDIDATES.find((m) => MediaRecorder.isTypeSupported(m));
      const rec = new MediaRecorder(media, mimeType ? { mimeType } : undefined);
      chunks.current = [];
      rec.ondataavailable = (e) => e.data.size > 0 && chunks.current.push(e.data);
      rec.onstop = async () => {
        const raw = new Blob(chunks.current, { type: rec.mimeType || "audio/webm" });
        cleanupAudio();
        setStatus("processing");
        try {
          const wav = await transcodeToWav(raw, `recording-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}`);
          setFile(wav);
          setPreviewUrl(URL.createObjectURL(wav));
          setStatus("done");
        } catch {
          setError("The recording could not be processed. Please try again.");
          setStatus("error");
        }
      };
      recorder.current = rec;

      const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const ctx = new Ctx();
      audioCtx.current = ctx;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      ctx.createMediaStreamSource(media).connect(analyser);
      const data = new Float32Array(analyser.fftSize);
      const loop = () => {
        analyser.getFloatTimeDomainData(data);
        let sum = 0;
        for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
        const rms = Math.sqrt(sum / data.length);
        level.current += (Math.min(1, rms * 5.5) - level.current) * 0.28;
        raf.current = requestAnimationFrame(loop);
      };
      loop();

      rec.start(250);
      startedAt.current = performance.now();
      setElapsed(0);
      timer.current = window.setInterval(() => setElapsed((performance.now() - startedAt.current) / 1000), 200);
      setStatus("recording");
    } catch (e) {
      cleanupAudio();
      const denied = e instanceof DOMException && (e.name === "NotAllowedError" || e.name === "SecurityError");
      setError(denied ? "Microphone access was blocked. Allow it in the address bar, then try again." : "No microphone could be opened.");
      setStatus("error");
    }
  }, [cleanupAudio]);

  const stop = useCallback(() => {
    if (recorder.current && recorder.current.state !== "inactive") recorder.current.stop();
  }, []);

  return { status, elapsed, file, previewUrl, error, level, start, stop, reset };
}
