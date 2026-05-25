import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import {
  Background,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";
import { api } from "../api";
import type { ConnectionRef, ScenarioDetail, SimulatorRef } from "../types";
import AtomCircleNode from "../components/AtomCircle";
import DetailPanel from "../components/DetailPanel";
import LogStream from "../components/LogStream";
import ScenarioConfig from "../components/ScenarioConfig";
import ResultViewer from "../components/ResultViewer";

const NODE_TYPES = { atomCircle: AtomCircleNode };

type SelectionKind = "node" | "edge" | null;

interface Selection {
  kind: SelectionKind;
  id: string | null;
}

function layoutByGroup(simulators: SimulatorRef[], timeGroups: Record<string, any>) {
  const groupOrder = Object.keys(timeGroups);
  const groupIndex: Record<string, number> = {};
  groupOrder.forEach((g, i) => (groupIndex[g] = i));

  const grouped: Record<string, SimulatorRef[]> = {};
  for (const s of simulators) {
    const g = s.group || "_default";
    if (!grouped[g]) grouped[g] = [];
    grouped[g].push(s);
  }

  const xStep = 240;
  const yStep = 180;
  const positions: Record<string, { x: number; y: number }> = {};
  const groups = Object.keys(grouped);
  groups.sort((a, b) => (groupIndex[a] ?? 99) - (groupIndex[b] ?? 99));
  groups.forEach((g, col) => {
    const items = grouped[g];
    items.forEach((s, row) => {
      positions[s.sim_id] = {
        x: col * xStep,
        y: row * yStep,
      };
    });
  });
  return positions;
}

export default function ScenarioPage() {
  const { name } = useParams<{ name: string }>();
  const [detail, setDetail] = useState<ScenarioDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [sel, setSel] = useState<Selection>({ kind: null, id: null });
  const [configOpen, setConfigOpen] = useState(false);
  const [runJobId, setRunJobId] = useState<string | null>(null);
  const [runDone, setRunDone] = useState(false);
  const [results, setResults] = useState<Awaited<ReturnType<typeof api.results>> | null>(null);

  const fetchDetail = useCallback(async () => {
    if (!name) return;
    setLoading(true);
    try {
      const d = await api.getScenario(name);
      setDetail(d);
    } catch (e) {
      console.error(e);
      setDetail(null);
    } finally {
      setLoading(false);
    }
  }, [name]);

  useEffect(() => {
    fetchDetail();
  }, [fetchDetail]);

  const nodes: Node[] = useMemo(() => {
    if (!detail) return [];
    const pos = layoutByGroup(detail.simulators, detail.time_groups);
    return detail.simulators.map((s) => ({
      id: s.sim_id,
      type: "atomCircle",
      position: pos[s.sim_id] ?? { x: 0, y: 0 },
      data: {
        sim_id: s.sim_id,
        display_name: s.display_name || s.class.split(".").pop() || s.sim_id,
        classification: s.classification,
      },
      selected: sel.kind === "node" && sel.id === s.sim_id,
    }));
  }, [detail, sel]);

  const edges: Edge[] = useMemo(() => {
    if (!detail) return [];
    return detail.connections.map((c) => ({
      id: c.connection_id,
      source: c.source_sim,
      target: c.target_sim,
      label: c.transform,
      animated: c.strategy === "EVENT",
      markerEnd: { type: MarkerType.ArrowClosed, color: "#6b7488" },
      selected: sel.kind === "edge" && sel.id === c.connection_id,
      style: c.strategy === "EVENT" ? { strokeDasharray: "4 4" } : undefined,
      labelStyle: { fontSize: 10 },
      labelBgStyle: {
        fill: "var(--bg-elev)",
        stroke: "var(--border)",
      },
    }));
  }, [detail, sel]);

  const selectedSim = useMemo(() => {
    if (!detail || sel.kind !== "node" || !sel.id) return null;
    return detail.simulators.find((s) => s.sim_id === sel.id) ?? null;
  }, [detail, sel]);

  const selectedConn = useMemo<ConnectionRef | null>(() => {
    if (!detail || sel.kind !== "edge" || !sel.id) return null;
    return detail.connections.find((c) => c.connection_id === sel.id) ?? null;
  }, [detail, sel]);

  function classificationCounts() {
    if (!detail) return { reused: 0, inherited: 0, new: 0, unknown: 0 };
    const c = { reused: 0, inherited: 0, new: 0, unknown: 0 };
    for (const s of detail.simulators) {
      (c as any)[s.classification] = ((c as any)[s.classification] ?? 0) + 1;
    }
    return c;
  }

  async function handleConfirmRun(overrides: Record<string, unknown>) {
    if (!detail) return;
    setConfigOpen(false);
    setRunDone(false);
    setResults(null);
    try {
      const r = await api.run(detail.name, overrides);
      setRunJobId(r.job_id);
    } catch (e: any) {
      alert(`运行启动失败: ${e?.message ?? e}`);
    }
  }

  async function handleRunDone(meta: Record<string, any>) {
    setRunDone(true);
    if (detail) {
      try {
        const r = await api.results(detail.name);
        setResults(r);
      } catch (e) {
        console.warn(e);
      }
    }
  }

  if (loading) {
    return <div className="loading">加载场景中…</div>;
  }
  if (!detail) {
    return <div className="loading">场景未找到。</div>;
  }

  const counts = classificationCounts();

  return (
    <div className="scenario-page">
      <div className="flow-region">
        <div className="flow-toolbar">
          <div>
            <div className="title">{detail.name}</div>
            <div className="subtitle">
              {detail.file} · {detail.simulators.length} sims · {detail.connections.length} connections
            </div>
          </div>
          <div className="legend" style={{ marginLeft: 16 }}>
            <div className="legend-item">
              <span className="legend-dot reused" /> 复用 {counts.reused}
            </div>
            <div className="legend-item">
              <span className="legend-dot inherited" /> 继承 {counts.inherited}
            </div>
            <div className="legend-item">
              <span className="legend-dot new" /> 新增 {counts.new}
            </div>
          </div>
          <div className="spacer" />
          <button className="btn primary" onClick={() => setConfigOpen(true)} disabled={!!runJobId && !runDone}>
            {runJobId && !runDone ? "运行中…" : "运行联合仿真"}
          </button>
        </div>
        <div className="flow-canvas">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={NODE_TYPES}
            fitView
            fitViewOptions={{ padding: 0.25 }}
            minZoom={0.2}
            maxZoom={1.6}
            onNodeClick={(_, n) => setSel({ kind: "node", id: n.id })}
            onEdgeClick={(_, e) => setSel({ kind: "edge", id: e.id })}
            onPaneClick={() => setSel({ kind: null, id: null })}
            proOptions={{ hideAttribution: true }}
          >
            <Background gap={20} size={1} color="#1f2532" />
            <Controls position="bottom-left" />
            <MiniMap position="bottom-right" pannable zoomable maskColor="rgba(15,17,21,0.7)" />
          </ReactFlow>
        </div>
      </div>

      <div className="detail-side">
        {runJobId ? (
          <div className="detail-body" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <LogStream jobId={runJobId} title={`运行 run_${detail.name}.py`} onDone={handleRunDone} height={300} />
            {runDone && results && <ResultViewer files={results.files} />}
            <button
              className="btn"
              onClick={() => {
                setRunJobId(null);
                setRunDone(false);
                setResults(null);
              }}
            >
              关闭运行面板
            </button>
          </div>
        ) : selectedSim ? (
          <SimDetail sim={selectedSim} />
        ) : selectedConn ? (
          <ConnDetail conn={selectedConn} />
        ) : (
          <ScenarioOverview detail={detail} />
        )}
      </div>

      {configOpen && (
        <ScenarioConfig
          scenario={detail.scenario || {}}
          onCancel={() => setConfigOpen(false)}
          onConfirm={handleConfirmRun}
        />
      )}
    </div>
  );
}

function SimDetail({ sim }: { sim: SimulatorRef }) {
  return (
    <DetailPanel
      title={`${sim.sim_id} · ${sim.display_name}`}
      subtitle={sim.file ?? sim.class}
      rows={[
        {
          key: "classification",
          value: <span className={`tag ${sim.classification}`}>{sim.classification}</span>,
        },
        { key: "class", value: <code>{sim.class}</code> },
        { key: "group", value: <code>{sim.group}</code> },
        {
          key: "params",
          value: (
            <pre style={{ margin: 0 }}>
              {JSON.stringify(sim.params, null, 2)}
            </pre>
          ),
        },
      ]}
      docstring={sim.docstring || "（该原子未提供 docstring）"}
    />
  );
}

function ConnDetail({ conn }: { conn: ConnectionRef }) {
  return (
    <DetailPanel
      title={`Connection · ${conn.connection_id}`}
      subtitle={`${conn.source_sim}.${conn.source_port} → ${conn.target_sim}.${conn.target_port}`}
      rows={[
        { key: "source", value: <code>{`${conn.source_sim}.${conn.source_port}`}</code> },
        { key: "target", value: <code>{`${conn.target_sim}.${conn.target_port}`}</code> },
        { key: "strategy", value: <code>{conn.strategy}</code> },
        { key: "transform", value: <code>{conn.transform}</code> },
      ]}
      docstring={conn.description || "（该连接未提供描述）"}
    />
  );
}

function ScenarioOverview({ detail }: { detail: ScenarioDetail }) {
  return (
    <DetailPanel
      title="场景概览"
      subtitle={detail.file}
      rows={[
        { key: "name", value: detail.name },
        { key: "simulators", value: detail.simulators.length },
        { key: "connections", value: detail.connections.length },
        {
          key: "time_groups",
          value: (
            <code>
              {Object.entries(detail.time_groups)
                .map(([k, v]: any) => `${k}(${v.dt ?? "?"}s)`)
                .join(", ")}
            </code>
          ),
        },
      ]}
    >
      <div style={{ marginTop: 8, fontSize: 12, color: "var(--text-dim)" }}>
        点击左侧任一原子（圆）或连接（箭头）查看详情。
      </div>
      <div style={{ marginTop: 8, fontSize: 12, color: "var(--text-dim)" }}>
        scenario 默认值（运行前可在弹窗中覆盖）：
      </div>
      <pre style={{ marginTop: 6 }}>{JSON.stringify(detail.scenario || {}, null, 2)}</pre>
    </DetailPanel>
  );
}
