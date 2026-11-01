import { useEffect, useMemo, useRef } from "react";
import type { Player } from "./usePlayer";

/** A silent, looping clock with the same shape as the real player, used for the landing-page demos. */
export function useDemoPlayer(duration: number, active = true, speed = 1): Player {
  const listeners = useRef(new Set<(t: number) => void>());
  const time = useRef(0);
  const activeRef = useRef(active);
  activeRef.current = active;

  useEffect(() => {
    let raf = 0;
    let prev = performance.now();
    const tick = (now: number) => {
      const dt = (now - prev) / 1000;
      prev = now;
      if (activeRef.current) {
        time.current = (time.current + dt * speed) % (duration + 2.5);
        const t = Math.min(time.current, duration);
        listeners.current.forEach((fn) => fn(t));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [duration, speed]);

  return useMemo<Player>(() => ({
    duration,
    getTime: () => Math.min(time.current, duration),
    subscribe: (fn) => {
      listeners.current.add(fn);
      fn(Math.min(time.current, duration));
      return () => listeners.current.delete(fn);
    },
    seek: (t) => { time.current = Math.max(0, Math.min(duration, t)); },
    play: () => undefined,
    pause: () => undefined,
    toggle: () => undefined,
    skip: (d) => { time.current = Math.max(0, Math.min(duration, time.current + d)); },
    setRate: () => undefined,
    setVolume: () => undefined,
  }), [duration]);
}
