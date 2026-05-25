import type {
  AtomInfo,
  JobStatus,
  ResultsResponse,
  ScenarioDetail,
  ScenarioSummary,
  StreamMessage,
} from "./types";

async function jget<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json() as Promise<T>;
}

async function jpost<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const text = await r.text();
    throw new Error(`${r.status} ${r.statusText}: ${text}`);
  }
  return r.json() as Promise<T>;
}

export const api = {
  health: () => jget<{ ok: boolean }>("/api/health"),
  listAtoms: () => jget<AtomInfo[]>("/api/atoms"),
  listScenarios: () => jget<ScenarioSummary[]>("/api/scenarios"),
  getScenario: (name: string) => jget<ScenarioDetail>(`/api/scenarios/${encodeURIComponent(name)}`),
  generate: (scenarioName: string, backgroundLines: string[]) =>
    jpost<{ job_id: string; status: string }>("/api/generate", {
      scenarioName,
      backgroundLines,
    }),
  run: (topologyName: string, scenarioOverrides: Record<string, unknown> = {}) =>
    jpost<{ job_id: string; status: string }>("/api/run", {
      topologyName,
      scenarioOverrides,
    }),
  getJob: (jobId: string) => jget<JobStatus>(`/api/jobs/${jobId}`),
  results: (name: string) => jget<ResultsResponse>(`/api/results/${encodeURIComponent(name)}`),
};

export function openJobStream(
  jobId: string,
  onMessage: (msg: StreamMessage) => void,
  onClose?: () => void,
): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${window.location.host}/api/jobs/${jobId}/stream`;
  const ws = new WebSocket(url);
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data) as StreamMessage;
      onMessage(msg);
    } catch (e) {
      console.warn("ws parse error", e, ev.data);
    }
  };
  ws.onclose = () => {
    onClose?.();
  };
  ws.onerror = (e) => {
    console.warn("ws error", e);
  };
  return ws;
}
