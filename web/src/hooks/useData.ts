import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../lib/api";
import type { Health, SessionCard } from "../lib/types";

/** Model/server status; polls quickly while the model is still loading. */
export function useHealth() {
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  useEffect(() => {
    let cancelled = false;
    let timer = 0;
    const poll = async () => {
      try {
        const h = await api.health();
        if (cancelled) return;
        setHealth(h);
        setOffline(false);
        timer = window.setTimeout(poll, h.model === "loading" ? 1500 : 20000);
      } catch {
        if (cancelled) return;
        setOffline(true);
        timer = window.setTimeout(poll, 4000);
      }
    };
    void poll();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, []);
  return { health, offline };
}

export function useSessions() {
  const [sessions, setSessions] = useState<SessionCard[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const refresh = useCallback(async () => {
    try {
      setSessions(await api.sessions());
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not load sessions");
    }
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);
  return { sessions, error, refresh, setSessions };
}
