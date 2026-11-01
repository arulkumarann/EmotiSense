/** Browser-side audio helpers: turn any decodable recording into a 16 kHz mono WAV the model reads directly. */

export function encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  write(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

/** Decode with the browser's own codecs (webm, m4a, mp3, ogg, wav...) and resample to 16 kHz mono. */
export async function transcodeToWav(blob: Blob, name = "recording"): Promise<File> {
  const bytes = await blob.arrayBuffer();
  const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const ctx = new AudioCtx();
  try {
    const decoded = await ctx.decodeAudioData(bytes.slice(0));
    const target = 16000;
    const frames = Math.max(1, Math.ceil(decoded.duration * target));
    const offline = new OfflineAudioContext(1, frames, target);
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start();
    const rendered = await offline.startRendering();
    return new File([encodeWav(rendered.getChannelData(0), target)], `${name}.wav`, { type: "audio/wav" });
  } finally {
    void ctx.close();
  }
}

/** Formats the server reads natively; everything else is converted in the browser first. */
export const NATIVE_EXTENSIONS = ["wav", "mp3", "ogg", "oga", "opus", "flac"];
export const ACCEPTED_EXTENSIONS = [...NATIVE_EXTENSIONS, "m4a", "aac", "webm", "mp4"];

export function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot < 0 ? "" : name.slice(dot + 1).toLowerCase();
}

export function isAcceptable(file: File): boolean {
  return file.type.startsWith("audio/") || ACCEPTED_EXTENSIONS.includes(extensionOf(file.name));
}
