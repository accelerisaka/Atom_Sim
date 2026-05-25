import { useEffect, useRef, useState } from "react";
import { openJobStream } from "../api";
import type { StreamMessage } from "../types";

interface Props {
  jobId: string;
  title?: string;
  onDone?: (meta: Record<string, any>, status: string) => void;
  height?: number;
}

export default function LogStream({ jobId, title = "运行日志", onDone, height }: Props) {
  const [lines, setLines] = useState<string[]>([]);
  const [status, setStatus] = useState<string>("connecting");
  const [exitCode, setExitCode] = useState<number | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;

  useEffect(() => {
    let closed = false;
    setLines([]);
    setStatus("connecting");
    setExitCode(null);

    const ws = openJobStream(jobId, (msg: StreamMessage) => {
      if (closed) return;
      if (msg.type === "line") {
        setLines((prev) => {
          const next = [...prev, msg.text];
          if (next.length > 1500) next.splice(0, next.length - 1500);
          return next;
        });
      } else if (msg.type === "status") {
        setStatus(msg.status);
        if (msg.exit_code !== undefined && msg.exit_code !== null) {
          setExitCode(msg.exit_code);
        }
      } else if (msg.type === "end") {
        onDoneRef.current?.(msg.meta || {}, status);
        try {
          ws.close();
        } catch {}
      } else if (msg.type === "error") {
        setLines((prev) => [...prev, `[error] ${msg.message}`]);
      }
    });

    return () => {
      closed = true;
      try {
        ws.close();
      } catch {}
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  useEffect(() => {
    const el = bodyRef.current;
    if (!el) return;
    // 自动滚动到底部
    el.scrollTop = el.scrollHeight;
  }, [lines]);

  const pillClass =
    status === "running" || status === "connecting"
      ? "running"
      : status === "done"
        ? "done"
        : "error";

  return (
    <div className="log-panel">
      <div className="log-panel-header">
        <span>{title}</span>
        <span className={`status-pill ${pillClass}`}>
          {status}
          {exitCode !== null ? ` · exit=${exitCode}` : ""}
        </span>
        <span style={{ flex: 1 }} />
        <span style={{ fontFamily: "var(--mono)" }}>job {jobId}</span>
      </div>
      <div className="log-panel-body" ref={bodyRef} style={height ? { height } : undefined}>
        {lines.length === 0 ? (
          <div style={{ color: "var(--text-dim)" }}>（等待输出 …）</div>
        ) : (
          lines.map((l, i) => (
            <div key={i} className={`log-line${l.includes("[error]") || l.includes("Error") ? " err" : ""}`}>
              {l}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
