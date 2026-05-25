import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import LogStream from "./LogStream";

interface Props {
  open: boolean;
  onClose: () => void;
}

export default function GenerateModal({ open, onClose }: Props) {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [background, setBackground] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [finished, setFinished] = useState(false);
  const [topologyName, setTopologyName] = useState<string | null>(null);

  if (!open) return null;

  async function handleSubmit() {
    setError(null);
    const lines = background
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    if (!name.trim() || lines.length === 0) {
      setError("请填写场景名，并至少输入一条场景背景");
      return;
    }
    setSubmitting(true);
    try {
      const r = await api.generate(name.trim(), lines);
      setJobId(r.job_id);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setSubmitting(false);
    }
  }

  function handleDone(meta: Record<string, any>) {
    setFinished(true);
    const t = (meta?.topology_name as string | undefined) ?? null;
    if (t) setTopologyName(t);
  }

  function handleClose() {
    setName("");
    setBackground("");
    setError(null);
    setJobId(null);
    setFinished(false);
    setTopologyName(null);
    onClose();
  }

  function goScenario() {
    if (topologyName) {
      navigate(`/scenario/${encodeURIComponent(topologyName)}`);
      handleClose();
    } else {
      handleClose();
    }
  }

  return (
    <div className="modal-mask" onClick={(e) => e.target === e.currentTarget && !jobId && handleClose()}>
      <div className="modal" style={{ width: jobId ? 720 : 560 }}>
        <div className="modal-header">
          <div className="modal-title">
            {jobId ? "正在生成联合仿真环境 …" : "新建联合仿真环境"}
          </div>
        </div>
        <div className="modal-body">
          {!jobId && (
            <>
              <div>
                <div className="form-label">场景名称（如：智能电网负载均衡）</div>
                <input
                  className="input"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="例：智能电网负载均衡"
                />
              </div>
              <div>
                <div className="form-label">
                  场景背景（每行一条，至少 1 条；将自动编号填入 prompt [2]）
                </div>
                <textarea
                  className="textarea"
                  value={background}
                  onChange={(e) => setBackground(e.target.value)}
                  placeholder={
                    "夏季高温日，户外大量空调开启\n" +
                    "局部变压器接近容量上限，可能过载\n" +
                    "动态电价随负载实时上升\n" +
                    "用户对电价敏感，会上调空调设定温度"
                  }
                />
              </div>
              {error && <div style={{ color: "var(--danger)", fontSize: 12 }}>{error}</div>}
            </>
          )}
          {jobId && (
            <>
              <LogStream jobId={jobId} title="cursor_agent 流式输出" onDone={handleDone} height={360} />
              {finished && (
                <div style={{ marginTop: 8, fontSize: 13 }}>
                  {topologyName ? (
                    <>
                      已生成新拓扑：
                      <code style={{ marginLeft: 6 }}>topology_{topologyName}.yaml</code>
                    </>
                  ) : (
                    <span style={{ color: "var(--warn)" }}>
                      未检测到新生成的 topology_*.yaml，请检查日志。
                    </span>
                  )}
                </div>
              )}
            </>
          )}
        </div>
        <div className="modal-footer">
          {!jobId && (
            <>
              <button className="btn" onClick={handleClose} disabled={submitting}>
                取消
              </button>
              <button className="btn primary" onClick={handleSubmit} disabled={submitting}>
                {submitting ? "提交中…" : "生成"}
              </button>
            </>
          )}
          {jobId && !finished && (
            <button className="btn" onClick={handleClose}>
              隐藏窗口（任务继续运行）
            </button>
          )}
          {jobId && finished && (
            <>
              <button className="btn" onClick={handleClose}>
                关闭
              </button>
              <button className="btn primary" onClick={goScenario} disabled={!topologyName}>
                查看联合仿真环境
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
