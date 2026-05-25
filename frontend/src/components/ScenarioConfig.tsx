import { useEffect, useState } from "react";

interface Props {
  scenario: Record<string, unknown>;
  onCancel: () => void;
  onConfirm: (overrides: Record<string, unknown>) => void;
  running?: boolean;
}

type FieldKind = "scalar" | "json";

function detectKind(v: unknown): FieldKind {
  if (typeof v === "number" || typeof v === "string" || typeof v === "boolean" || v === null) {
    return "scalar";
  }
  return "json";
}

function toEditable(v: unknown, kind: FieldKind): string {
  if (kind === "scalar") {
    return v === null || v === undefined ? "" : String(v);
  }
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

function parseEditable(raw: string, kind: FieldKind, original: unknown): { ok: true; value: unknown } | { ok: false; error: string } {
  if (kind === "scalar") {
    if (typeof original === "number") {
      const n = Number(raw);
      if (Number.isNaN(n)) return { ok: false, error: "需要数字" };
      return { ok: true, value: n };
    }
    if (typeof original === "boolean") {
      const lc = raw.trim().toLowerCase();
      if (lc === "true") return { ok: true, value: true };
      if (lc === "false") return { ok: true, value: false };
      return { ok: false, error: "需要 true / false" };
    }
    return { ok: true, value: raw };
  }
  // json
  try {
    return { ok: true, value: JSON.parse(raw) };
  } catch (e: any) {
    return { ok: false, error: `JSON 解析失败: ${e?.message ?? e}` };
  }
}

export default function ScenarioConfig({ scenario, onCancel, onConfirm, running }: Props) {
  const initialKeys = Object.keys(scenario);
  const initial: Record<string, string> = {};
  const kinds: Record<string, FieldKind> = {};
  for (const k of initialKeys) {
    kinds[k] = detectKind(scenario[k]);
    initial[k] = toEditable(scenario[k], kinds[k]);
  }
  const [vals, setVals] = useState<Record<string, string>>(initial);
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    // 当 scenario 改变时重置
    const obj: Record<string, string> = {};
    for (const k of Object.keys(scenario)) {
      obj[k] = toEditable(scenario[k], detectKind(scenario[k]));
    }
    setVals(obj);
    setErrors({});
  }, [scenario]);

  function handleConfirm() {
    const overrides: Record<string, unknown> = {};
    const nextErr: Record<string, string> = {};
    for (const k of Object.keys(vals)) {
      const kind = kinds[k] ?? detectKind(scenario[k]);
      const parsed = parseEditable(vals[k], kind, scenario[k]);
      if (!parsed.ok) {
        nextErr[k] = parsed.error;
      } else {
        // 只有改动过的才写入 override，避免无意义覆盖
        const before = toEditable(scenario[k], kind);
        if (before !== vals[k]) {
          overrides[k] = parsed.value;
        }
      }
    }
    setErrors(nextErr);
    if (Object.keys(nextErr).length > 0) return;
    onConfirm(overrides);
  }

  return (
    <div className="modal-mask" onClick={(e) => e.target === e.currentTarget && !running && onCancel()}>
      <div className="modal" style={{ width: 640, maxHeight: "85vh", display: "flex", flexDirection: "column" }}>
        <div className="modal-header">
          <div className="modal-title">运行前配置 · scenario 参数</div>
        </div>
        <div className="modal-body" style={{ overflow: "auto", maxHeight: "60vh" }}>
          <div style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 4 }}>
            读取 topology 中的 scenario 字段，可在此覆盖默认值。仅修改过的项会被提交。
          </div>
          {Object.keys(vals).length === 0 && (
            <div className="empty">该拓扑未声明 scenario 字段，将使用 yaml 默认值。</div>
          )}
          {Object.keys(vals).map((k) => {
            const kind = kinds[k] ?? "scalar";
            return (
              <div key={k} style={{ marginTop: 12 }}>
                <div className="form-label">
                  {k}
                  <span style={{ marginLeft: 6, color: "var(--text-dim)" }}>({kind})</span>
                </div>
                {kind === "scalar" ? (
                  <input
                    className="input"
                    value={vals[k]}
                    onChange={(e) => setVals({ ...vals, [k]: e.target.value })}
                  />
                ) : (
                  <textarea
                    className="textarea"
                    value={vals[k]}
                    onChange={(e) => setVals({ ...vals, [k]: e.target.value })}
                    style={{ minHeight: 100 }}
                  />
                )}
                {errors[k] && <div style={{ color: "var(--danger)", fontSize: 12, marginTop: 4 }}>{errors[k]}</div>}
              </div>
            );
          })}
        </div>
        <div className="modal-footer">
          <button className="btn" onClick={onCancel} disabled={running}>
            取消
          </button>
          <button className="btn primary" onClick={handleConfirm} disabled={running}>
            开始运行
          </button>
        </div>
      </div>
    </div>
  );
}
