import { useMemo } from "react";
import type { OutputFile } from "../types";

interface Props {
  files: OutputFile[];
}

export default function ResultViewer({ files }: Props) {
  const mp4 = useMemo(() => files.find((f) => f.filename.toLowerCase().endsWith(".mp4")), [files]);
  const gif = useMemo(() => files.find((f) => f.filename.toLowerCase().endsWith(".gif")), [files]);
  const png = useMemo(() => files.find((f) => f.filename.toLowerCase().endsWith(".png")), [files]);

  if (files.length === 0) {
    return <div className="empty">尚未发现结果文件（gif/mp4）。</div>;
  }

  return (
    <div className="result-viewer">
      {mp4 ? (
        <video className="result-media" src={mp4.url} controls loop autoPlay muted />
      ) : gif ? (
        <img className="result-media" src={gif.url} alt={gif.filename} />
      ) : png ? (
        <img className="result-media" src={png.url} alt={png.filename} />
      ) : (
        <div className="empty">仅找到非媒体文件，请使用下方列表下载。</div>
      )}
      <div className="result-files">
        {files.map((f) => (
          <a key={f.filename} className="card" href={f.url} target="_blank" rel="noreferrer" style={{ minWidth: 0 }}>
            <div className="card-title">{f.filename}</div>
            <div className="card-desc">{(f.size / 1024).toFixed(1)} KB · {f.ext ?? "?"}</div>
          </a>
        ))}
      </div>
    </div>
  );
}
