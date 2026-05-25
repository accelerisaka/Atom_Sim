import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { Classification } from "../types";

export interface AtomNodeData {
  sim_id: string;
  display_name: string;
  classification: Classification;
  selected?: boolean;
  [key: string]: unknown;
}

export default function AtomCircleNode(props: NodeProps) {
  const data = props.data as AtomNodeData;
  const cls = data.classification;
  const cn = `atom-node cls-${cls} ${props.selected ? "selected" : ""}`;
  const labelMap: Record<Classification, string> = {
    reused: "复用",
    inherited: "继承",
    new: "新增",
    unknown: "未知",
  };
  return (
    <div className={cn} title={data.display_name}>
      <Handle type="target" position={Position.Left} />
      <div className="sim-id">{data.sim_id}</div>
      <div className="sim-name">{data.display_name}</div>
      <div className="cls-badge">{labelMap[cls]}</div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
