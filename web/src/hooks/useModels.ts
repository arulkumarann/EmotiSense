import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../lib/api";
import type { ModelsResponse } from "../lib/types";
import { useToast } from "./useToasts";

/** The model catalogue plus the saved default, with an optimistic `choose` that rolls back on failure. */
export function useModels() {
  const toast = useToast();
  const [data, setData] = useState<ModelsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setData(await api.models());
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not load the models");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const choose = useCallback(async (id: string) => {
    let previous: ModelsResponse | null = null;
    setData((cur) => {
      previous = cur;
      return cur ? { ...cur, active: id, timeline: cur.timeline.map((m) => ({ ...m, active: m.id === id })) } : cur;
    });
    try {
      await api.setActiveModel(id);
    } catch (e) {
      setData(previous);
      toast((e as Error).message, "error");
    }
  }, [toast]);

  const active = data?.timeline.find((m) => m.id === data.active) ?? null;
  return { data, error, active, choose, refresh };
}
