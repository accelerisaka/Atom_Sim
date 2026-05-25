export interface AtomInfo {
  sim_id: string;
  class_name: string;
  name: string;
  docstring: string;
  file: string;
  module: string;
  category: "physical" | "social" | "other";
  base_classes: string[];
}

export type Classification = "reused" | "inherited" | "new" | "unknown";

export interface SimulatorRef {
  sim_id: string;
  class: string;
  group: string;
  params: Record<string, unknown>;
  classification: Classification;
  file: string | null;
  docstring: string;
  display_name: string;
}

export interface ConnectionRef {
  connection_id: string;
  source_sim: string;
  source_port: string;
  target_sim: string;
  target_port: string;
  strategy: string;
  transform: string;
  description: string;
}

export interface ScenarioDetail {
  name: string;
  file: string;
  time_groups: Record<string, { dt: number; description?: string }>;
  simulators: SimulatorRef[];
  connections: ConnectionRef[];
  scenario: Record<string, unknown>;
}

export interface ScenarioSummary {
  name: string;
  file: string;
  has_run_script: boolean;
  run_script: string | null;
}

export interface JobStatus {
  job_id: string;
  kind: "generate" | "run" | string;
  name: string;
  status: "pending" | "running" | "done" | "error" | "cancelled" | string;
  exit_code: number | null;
  meta: Record<string, any>;
  line_count: number;
  lines?: string[];
}

export interface OutputFile {
  filename: string;
  url: string;
  size: number;
  ext?: string;
}

export interface ResultsResponse {
  name: string;
  files: OutputFile[];
}

export type StreamMessage =
  | { type: "line"; text: string }
  | { type: "status"; status: string; exit_code?: number | null }
  | { type: "end"; meta?: Record<string, any> }
  | { type: "ping" }
  | { type: "error"; message: string };
