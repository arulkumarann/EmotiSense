import { useCallback, useEffect, useMemo, useRef, useState } from "react";

export interface Player {
  duration: number;
  /** Current playback time in seconds (smoothed). */
  getTime(): number;
  /** Called every animation frame while playing, and once after every seek. Returns an unsubscribe function. */
  subscribe(fn: (t: number) => void): () => void;
  seek(t: number): void;
  play(): void;
  pause(): void;
  toggle(): void;
  skip(delta: number): void;
  setRate(rate: number): void;
  setVolume(volume: number): void;
}

export interface PlayerState {
  playing: boolean;
  ready: boolean;
  error: string | null;
  rate: number;
  volume: number;
  /** Low-frequency copy of the time for text (about 10 updates a second). */
  displayTime: number;
}

/**
 * One audio element, one clock. Visual components subscribe to the clock and move DOM nodes directly, so a
 * 60 fps playhead never causes a React re-render. While playing, the time is eased toward the audio's own clock
 * to hide browsers that only update `currentTime` in coarse steps.
 */
export function usePlayer(src: string | null, duration: number): { player: Player; state: PlayerState } {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const listeners = useRef(new Set<(t: number) => void>());
  const timeRef = useRef(0);
  const rafRef = useRef(0);
  const [state, setState] = useState<PlayerState>({ playing: false, ready: false, error: null, rate: 1, volume: 0.9, displayTime: 0 });
  const lastText = useRef(0);

  const emit = useCallback((t: number) => {
    timeRef.current = t;
    listeners.current.forEach((fn) => fn(t));
    const now = performance.now();
    if (now - lastText.current > 90) {
      lastText.current = now;
      setState((s) => (Math.abs(s.displayTime - t) < 0.05 ? s : { ...s, displayTime: t }));
    }
  }, []);

  const stopLoop = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    rafRef.current = 0;
  }, []);

  const startLoop = useCallback(() => {
    if (rafRef.current) return;
    const tick = () => {
      const audio = audioRef.current;
      if (!audio) return;
      const target = audio.currentTime;
      const prev = timeRef.current;
      // Jumps (seeks) are applied at once; normal playback is eased to hide coarse currentTime steps.
      const next = Math.abs(target - prev) > 0.35 ? target : prev + (target - prev) * 0.4;
      emit(Math.min(next, duration || next));
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, [duration, emit]);

  useEffect(() => {
    if (!src) return undefined;
    const audio = new Audio();
    audio.preload = "auto";
    audio.src = src;
    audio.volume = 0.9;
    audioRef.current = audio;
    timeRef.current = 0;
    const onPlay = () => {
      setState((s) => ({ ...s, playing: true }));
      startLoop();
    };
    const onPause = () => {
      setState((s) => ({ ...s, playing: false }));
      stopLoop();
      emit(audio.currentTime);
    };
    const onEnded = () => {
      setState((s) => ({ ...s, playing: false }));
      stopLoop();
      emit(duration);
    };
    const onReady = () => setState((s) => ({ ...s, ready: true, error: null }));
    const onError = () => setState((s) => ({ ...s, error: "This browser can't play the audio file", ready: false }));
    const onSeeked = () => emit(audio.currentTime);
    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("ended", onEnded);
    audio.addEventListener("canplay", onReady);
    audio.addEventListener("error", onError);
    audio.addEventListener("seeked", onSeeked);
    return () => {
      stopLoop();
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("ended", onEnded);
      audio.removeEventListener("canplay", onReady);
      audio.removeEventListener("error", onError);
      audio.removeEventListener("seeked", onSeeked);
      audioRef.current = null;
    };
  }, [src, duration, emit, startLoop, stopLoop]);

  const player = useMemo<Player>(() => {
    const seek = (t: number) => {
      const audio = audioRef.current;
      const clamped = Math.min(Math.max(0, t), duration || t);
      if (audio) audio.currentTime = clamped;
      emit(clamped);
    };
    return {
      duration,
      getTime: () => timeRef.current,
      subscribe: (fn) => {
        listeners.current.add(fn);
        fn(timeRef.current);
        return () => listeners.current.delete(fn);
      },
      seek,
      play: () => {
        const audio = audioRef.current;
        if (!audio) return;
        if (audio.ended || audio.currentTime >= duration - 0.05) audio.currentTime = 0;
        void audio.play().catch(() => setState((s) => ({ ...s, error: "Playback was blocked. Press play again." })));
      },
      pause: () => audioRef.current?.pause(),
      toggle: () => {
        const audio = audioRef.current;
        if (!audio) return;
        if (audio.paused) player.play();
        else audio.pause();
      },
      skip: (delta) => seek(timeRef.current + delta),
      setRate: (rate) => {
        if (audioRef.current) audioRef.current.playbackRate = rate;
        setState((s) => ({ ...s, rate }));
      },
      setVolume: (volume) => {
        if (audioRef.current) audioRef.current.volume = volume;
        setState((s) => ({ ...s, volume }));
      },
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [duration, emit]);

  return { player, state };
}
