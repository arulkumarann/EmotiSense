import { NATIVE_EXTENSIONS, extensionOf, transcodeToWav } from "./wav";
import type { EscalationSettings, Health, JobState, Note, SessionCard, SessionDetail } from "./types";

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, init);
  } catch {
    throw new ApiError("Can't reach the local EmotiSense server. Is it running?", 0);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : detail;
    } catch {
      /* not json */
    }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/health"),
  sessions: (params: { tag?: string; q?: string } = {}) => {
    const qs = new URLSearchParams();
    if (params.tag) qs.set("tag", params.tag);
    if (params.q) qs.set("q", params.q);
    return request<SessionCard[]>(`/sessions${qs.size ? `?${qs}` : ""}`);
  },
  session: (id: string) => request<SessionDetail>(`/sessions/${id}`),
  rename: (id: string, patch: { title?: string; tag?: string }) => request<{ ok: boolean }>(`/sessions/${id}`, json("PATCH", patch)),
  remove: (id: string) => request<{ ok: boolean }>(`/sessions/${id}`, { method: "DELETE" }),
  audioUrl: (id: string) => `/api/sessions/${id}/audio`,
  addNote: (id: string, t: number, text: string) => request<Note>(`/sessions/${id}/notes`, json("POST", { t, text })),
  removeNote: (noteId: number) => request<{ ok: boolean }>(`/notes/${noteId}`, { method: "DELETE" }),
  settings: () => request<{ values: EscalationSettings; defaults: EscalationSettings }>("/settings"),
  saveSettings: (values: Partial<EscalationSettings>) =>
    request<{ values: EscalationSettings; defaults: EscalationSettings }>("/settings", json("PUT", values)),
  job: (id: string) => request<JobState>(`/jobs/${id}`),

  /** Upload a recording (converted to WAV in the browser if the server can't read its format). */
  async upload(file: File, meta: { title?: string; tag?: string } = {}): Promise<{ job_id: string; session_id: string }> {
    let toSend = file;
    if (!NATIVE_EXTENSIONS.includes(extensionOf(file.name))) {
      try {
        toSend = await transcodeToWav(file, file.name.replace(/\.[^.]+$/, "") || "recording");
      } catch {
        toSend = file; // let the server try (it can use ffmpeg when available)
      }
    }
    const form = new FormData();
    form.append("file", toSend, toSend.name);
    form.append("title", meta.title ?? "");
    form.append("tag", meta.tag ?? "");
    return request("/sessions", { method: "POST", body: form });
  },
};

/** Poll a job until it finishes, reporting each state. */
export async function watchJob(jobId: string, onUpdate: (job: JobState) => void, signal?: AbortSignal): Promise<JobState> {
  for (;;) {
    if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
    const job = await api.job(jobId);
    onUpdate(job);
    if (job.status === "done" || job.status === "error") return job;
    await new Promise((resolve) => setTimeout(resolve, 350));
  }
}
