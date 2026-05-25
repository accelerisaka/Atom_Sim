interface Props {
  title?: string;
  subtitle?: string;
  emptyHint?: string;
  docstring?: string;
  rows?: { key: string; value: React.ReactNode }[];
  children?: React.ReactNode;
}

export default function DetailPanel({
  title,
  subtitle,
  emptyHint = "点击左侧任一原子或连接查看详情",
  docstring,
  rows,
  children,
}: Props) {
  if (!title) {
    return (
      <div className="detail-panel">
        <div className="detail-empty">{emptyHint}</div>
      </div>
    );
  }
  return (
    <div className="detail-panel">
      <div className="detail-header">
        <div className="detail-title">{title}</div>
        {subtitle && <div className="detail-sub">{subtitle}</div>}
      </div>
      <div className="detail-body">
        {rows && rows.length > 0 && (
          <table className="kv-table" style={{ marginBottom: 12 }}>
            <tbody>
              {rows.map((r) => (
                <tr key={r.key}>
                  <td className="k">{r.key}</td>
                  <td>{r.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {docstring && (
          <div style={{ marginTop: 4 }}>
            <pre>{docstring}</pre>
          </div>
        )}
        {children}
      </div>
    </div>
  );
}
