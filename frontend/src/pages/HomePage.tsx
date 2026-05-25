import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import type { AtomInfo, ScenarioSummary } from "../types";
import DetailPanel from "../components/DetailPanel";
import GenerateModal from "../components/GenerateModal";

export default function HomePage() {
  const navigate = useNavigate();
  const [atoms, setAtoms] = useState<AtomInfo[]>([]);
  const [scenarios, setScenarios] = useState<ScenarioSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [modalOpen, setModalOpen] = useState(false);

  async function refresh() {
    setLoading(true);
    try {
      const [a, s] = await Promise.all([api.listAtoms(), api.listScenarios()]);
      setAtoms(a);
      setScenarios(s);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  const selectedAtom = useMemo(
    () => atoms.find((a) => a.sim_id === selected) ?? null,
    [atoms, selected],
  );

  const physical = atoms.filter((a) => a.category === "physical");
  const social = atoms.filter((a) => a.category === "social");

  function handleNavScenario(name: string) {
    navigate(`/scenario/${encodeURIComponent(name)}`);
  }

  function renderAtomGrid(list: AtomInfo[]) {
    if (list.length === 0) {
      return <div className="empty">（无）</div>;
    }
    return (
      <div className="atom-grid">
        {list.map((a) => (
          <div
            key={a.sim_id + a.file}
            className={`atom-cell ${selected === a.sim_id ? "selected" : ""}`}
            onClick={() => setSelected(a.sim_id)}
          >
            <div className={`atom-circle cat-${a.category}`}>{a.sim_id}</div>
            <div className="atom-label">
              <div className="name">{a.class_name}</div>
              <div>{a.name.replace(`${a.class_name}`, "").replace(/^[\s—\-:：]+/, "")}</div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  const fireScenario = scenarios.find((s) => s.name === "fire");
  const gridScenario = scenarios.find((s) => s.name === "grid");
  const otherScenarios = scenarios.filter((s) => s.name !== "fire" && s.name !== "grid");

  return (
    <div className="home">
      <div className="home-main">
        <div className="section">
          <div className="section-header">
            <span className="section-title">入口</span>
            <span className="section-subtitle">生成新联合仿真环境 / 运行内置演示</span>
          </div>
          <div className="section-body">
            <div className="action-cards">
              <div className="card primary" onClick={() => setModalOpen(true)}>
                <div className="card-tag">cursor_agent</div>
                <div className="card-title">+ 生成联合仿真环境</div>
                <div className="card-desc">
                  输入场景名与背景，调用 Cursor Agent 自动生成原子、transforms、topology
                  与可视化入口。
                </div>
              </div>
              <div
                className="card"
                onClick={() => handleNavScenario("fire")}
                style={{ opacity: fireScenario ? 1 : 0.5, pointerEvents: fireScenario ? "auto" : "none" }}
              >
                <div className="card-tag">演示</div>
                <div className="card-title">run_fire · 建筑火灾疏散</div>
                <div className="card-desc">
                  TD1/FL3/CM1/PD3/BP6/S2/SD4/P6/FL4/BP5 多原子耦合，温度/烟雾 → 心理 → 疏散行为。
                </div>
              </div>
              <div
                className="card"
                onClick={() => handleNavScenario("grid")}
                style={{ opacity: gridScenario ? 1 : 0.5, pointerEvents: gridScenario ? "auto" : "none" }}
              >
                <div className="card-tag">演示</div>
                <div className="card-title">run_grid · 智能电网负载均衡</div>
                <div className="card-desc">
                  夏季高温下的需求侧响应：户外温度 → 空调负荷 → 电价 → 用户调温 闭环。
                </div>
              </div>
              {otherScenarios.map((s) => (
                <div key={s.name} className="card" onClick={() => handleNavScenario(s.name)}>
                  <div className="card-tag">已生成</div>
                  <div className="card-title">{s.name}</div>
                  <div className="card-desc">
                    {s.file}
                    {!s.has_run_script && (
                      <span style={{ color: "var(--warn)", marginLeft: 6 }}>· 未找到 run_{s.name}.py</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="section">
          <div className="section-header">
            <span className="section-title">已有的原子仿真器</span>
            <span className="section-subtitle">
              扫描自 atoms/physical/ 与 atoms/social/，共 {atoms.length} 个
            </span>
            <span style={{ flex: 1 }} />
            <button className="btn" onClick={refresh} disabled={loading}>
              {loading ? "扫描中…" : "刷新"}
            </button>
          </div>
          <div className="section-body">
            <div style={{ marginBottom: 6, color: "var(--text-dim)", fontSize: 12 }}>
              <span className="tag">physical</span> 物理原子（{physical.length}）
            </div>
            {renderAtomGrid(physical)}
            <div style={{ marginTop: 16, marginBottom: 6, color: "var(--text-dim)", fontSize: 12 }}>
              <span className="tag">social</span> 社会/心理原子（{social.length}）
            </div>
            {renderAtomGrid(social)}
          </div>
        </div>
      </div>

      <div>
        {selectedAtom ? (
          <DetailPanel
            title={`${selectedAtom.sim_id} · ${selectedAtom.class_name}`}
            subtitle={selectedAtom.file}
            rows={[
              { key: "category", value: <span className="tag">{selectedAtom.category}</span> },
              { key: "module", value: <code>{selectedAtom.module}</code> },
              {
                key: "base",
                value: selectedAtom.base_classes.join(", ") || "—",
              },
            ]}
            docstring={selectedAtom.docstring || "（该文件未提供 docstring）"}
          />
        ) : (
          <DetailPanel emptyHint="点击左侧任一原子查看其类信息与文件开头注释" />
        )}
      </div>

      <GenerateModal
        open={modalOpen}
        onClose={() => {
          setModalOpen(false);
          refresh();
        }}
      />
    </div>
  );
}
