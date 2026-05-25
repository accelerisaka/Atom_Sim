"""
Atom_Sim 前端可视化后端 (FastAPI)。

API 概览
--------
GET  /api/atoms                       列出 atoms/{physical,social}/ 下所有原子仿真器
GET  /api/scenarios                   列出 config/topology_*.yaml
GET  /api/scenarios/{name}            返回 topology 详情 + simulators 分类
POST /api/generate                    启动 cursor_agent 生成新场景
GET  /api/jobs                        列出所有 jobs
GET  /api/jobs/{job_id}                查询 job 状态
WS   /api/jobs/{job_id}/stream         订阅流式日志
POST /api/run                         运行 run_xxx.py
GET  /api/results/{name}              查询 output_{name}/ 下的 gif/mp4 列表
GET  /api/output/{name}/{filename}    流式下载结果文件

运行
----
cd 仓库根
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.services.atom_scanner import scan_all_atoms
from backend.services.classifier import classify_simulators
from backend.services.manifest import load_manifest, record_generation
from backend.services.process_manager import JobState, process_manager
from backend.services.snapshot import diff_atoms, snapshot_atoms
from backend.services.topology_parser import (
    find_topology_path,
    list_topologies,
    load_topology,
    write_temp_topology,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backend")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


app = FastAPI(title="Atom_Sim Frontend Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------
# Pydantic schemas
# ----------------------------------------------------------


class GenerateRequest(BaseModel):
    scenarioName: str = Field(..., min_length=1)
    backgroundLines: List[str] = Field(default_factory=list)


class RunRequest(BaseModel):
    topologyName: str
    scenarioOverrides: Dict[str, Any] = Field(default_factory=dict)


# ----------------------------------------------------------
# 静态 / 健康检查
# ----------------------------------------------------------


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "platform": sys.platform}


# ----------------------------------------------------------
# Atoms / Scenarios
# ----------------------------------------------------------


@app.get("/api/atoms")
async def get_atoms() -> List[Dict[str, Any]]:
    return scan_all_atoms()


@app.get("/api/scenarios")
async def get_scenarios() -> List[Dict[str, Any]]:
    return list_topologies()


@app.get("/api/scenarios/{name}")
async def get_scenario(name: str) -> Dict[str, Any]:
    topo = load_topology(name)
    if topo is None:
        raise HTTPException(404, detail=f"scenario '{name}' not found")
    # 若该场景由 cursor_agent 生成过，则使用持久化的基线做分类，
    # 这样刷新页面后仍能看到 reused / inherited / new。
    manifest = load_manifest(name)
    pre_snap = set(manifest["baseline_atom_files"]) if manifest else None
    classification = classify_simulators(topo["simulators"], pre_snapshot=pre_snap)
    atoms = {a["sim_id"]: a for a in scan_all_atoms()}
    atoms_by_class = {a["class_name"]: a for a in atoms.values()}

    enriched_sims: List[Dict[str, Any]] = []
    for sim in topo["simulators"]:
        cls_ref = sim["class"]
        tail = cls_ref.rsplit(".", 1)[-1]
        atom_info = atoms_by_class.get(tail)
        cinfo = classification.get(sim["sim_id"], {})
        enriched_sims.append({
            **sim,
            "classification": cinfo.get("classification", "unknown"),
            "file": cinfo.get("file"),
            "docstring": atom_info["docstring"] if atom_info else "",
            "display_name": atom_info["name"] if atom_info else tail,
        })

    return {
        "name": topo["name"],
        "file": topo["file"],
        "time_groups": topo["time_groups"],
        "simulators": enriched_sims,
        "connections": topo["connections"],
        "scenario": topo["scenario"],
    }


# ----------------------------------------------------------
# Generate (cursor_agent)
# ----------------------------------------------------------


@app.post("/api/generate")
async def generate_scenario(req: GenerateRequest) -> Dict[str, Any]:
    if not req.scenarioName.strip():
        raise HTTPException(400, "scenarioName required")
    if not req.backgroundLines:
        raise HTTPException(400, "backgroundLines required (at least 1)")

    root = project_root()
    agent_dir = root / "cursor_agent"
    if not agent_dir.exists():
        raise HTTPException(500, "cursor_agent directory not found")

    # 快照
    pre_snap = snapshot_atoms()
    pre_topologies = {t["name"] for t in list_topologies()}

    # 写背景文件到临时位置
    bg_path = agent_dir / ".background.tmp.txt"
    bg_path.write_text("\n".join(req.backgroundLines), encoding="utf-8")

    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    cmd = [
        npm_cmd,
        "start",
        "--silent",
        "--",
        "--non-interactive",
        "--scenario-name",
        req.scenarioName,
        "--background-file",
        str(bg_path),
    ]

    async def on_complete(job: JobState) -> None:
        new_files = diff_atoms(pre_snap)
        post_topologies = {t["name"] for t in list_topologies()}
        new_topologies = sorted(post_topologies - pre_topologies)
        # 兜底：读 cursor_agent/last_run.json
        last_run_path = agent_dir / "last_run.json"
        last_run: Dict[str, Any] = {}
        if last_run_path.exists():
            try:
                last_run = json.loads(last_run_path.read_text(encoding="utf-8"))
            except Exception:
                last_run = {}
        topology_name: Optional[str] = None
        if new_topologies:
            topology_name = new_topologies[-1]
        elif last_run.get("topologyYaml"):
            yfile = str(last_run["topologyYaml"])
            stem = Path(yfile).stem
            if stem.startswith("topology_"):
                topology_name = stem[len("topology_"):]
        job.meta["pre_snapshot"] = sorted(pre_snap)
        job.meta["new_atoms"] = new_files
        job.meta["new_topologies"] = new_topologies
        job.meta["topology_name"] = topology_name
        job.meta["last_run"] = last_run
        # 持久化分类基线（供后续 GET /api/scenarios/{name} 刷新后仍能正确着色）
        if topology_name:
            try:
                record_generation(
                    name=topology_name,
                    baseline_atom_files=list(pre_snap),
                    new_atom_files=new_files,
                    extras={"scenarioName": req.scenarioName},
                )
            except Exception as e:
                logger.warning("record_generation failed: %s", e)
        # 删临时背景文件
        try:
            bg_path.unlink(missing_ok=True)
        except Exception:
            pass

    job = await process_manager.spawn(
        kind="generate",
        name=req.scenarioName,
        cmd=cmd,
        cwd=str(agent_dir),
        meta={"scenarioName": req.scenarioName},
        on_complete=on_complete,
    )
    return {"job_id": job.job_id, "status": job.status}


# ----------------------------------------------------------
# Run a scenario (python run_xxx.py -c ...)
# ----------------------------------------------------------


@app.post("/api/run")
async def run_scenario(req: RunRequest) -> Dict[str, Any]:
    name = req.topologyName
    root = project_root()
    run_script = root / f"run_{name}.py"
    if not run_script.exists():
        raise HTTPException(404, f"run_{name}.py not found in repo root")

    if find_topology_path(name) is None:
        raise HTTPException(404, f"topology '{name}' not found")

    # 写运行时 yaml
    tmp_yaml = write_temp_topology(name, req.scenarioOverrides or {})
    if tmp_yaml is None:
        raise HTTPException(500, "failed to write runtime yaml")

    rel_yaml = tmp_yaml.relative_to(root).as_posix()
    python_exe = sys.executable or "python"
    cmd = [python_exe, run_script.name, "-c", rel_yaml]

    async def on_complete(job: JobState) -> None:
        out_dir = root / f"output_{name}"
        files: List[Dict[str, Any]] = []
        if out_dir.exists():
            for f in sorted(out_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in (".gif", ".mp4"):
                    files.append({
                        "filename": f.name,
                        "url": f"/api/output/{name}/{f.name}",
                        "size": f.stat().st_size,
                    })
        job.meta["output_files"] = files

    job = await process_manager.spawn(
        kind="run",
        name=name,
        cmd=cmd,
        cwd=str(root),
        meta={"topologyName": name, "runtime_yaml": rel_yaml},
        on_complete=on_complete,
    )
    return {"job_id": job.job_id, "status": job.status}


# ----------------------------------------------------------
# Jobs API
# ----------------------------------------------------------


@app.get("/api/jobs")
async def list_jobs() -> List[Dict[str, Any]]:
    return process_manager.list()


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    job = process_manager.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return {
        **job.to_public(),
        "lines": job.lines[-500:],
    }


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> Dict[str, Any]:
    ok = await process_manager.cancel(job_id)
    if not ok:
        raise HTTPException(404, "job not found or already finished")
    return {"ok": True}


@app.websocket("/api/jobs/{job_id}/stream")
async def stream_job(ws: WebSocket, job_id: str) -> None:
    await ws.accept()
    q = await process_manager.subscribe(job_id)
    if q is None:
        await ws.send_json({"type": "error", "message": "job not found"})
        await ws.close()
        return
    try:
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=60.0)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping"})
                continue
            await ws.send_json(msg)
            if msg.get("type") == "end":
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("ws send failed")
    finally:
        await process_manager.unsubscribe(job_id, q)
        try:
            await ws.close()
        except Exception:
            pass


# ----------------------------------------------------------
# Results / static output files
# ----------------------------------------------------------


_SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")
_SAFE_FILE = re.compile(r"^[A-Za-z0-9_\-\.]+\.(gif|mp4|png|json)$", re.IGNORECASE)


@app.get("/api/results/{name}")
async def get_results(name: str) -> Dict[str, Any]:
    if not _SAFE_NAME.match(name):
        raise HTTPException(400, "invalid name")
    out_dir = project_root() / f"output_{name}"
    files: List[Dict[str, Any]] = []
    if out_dir.exists():
        for f in sorted(out_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in (".gif", ".mp4", ".png", ".json"):
                files.append({
                    "filename": f.name,
                    "url": f"/api/output/{name}/{f.name}",
                    "size": f.stat().st_size,
                    "ext": f.suffix.lower().lstrip("."),
                })
    return {"name": name, "files": files}


@app.get("/api/output/{name}/{filename}")
async def get_output_file(name: str, filename: str) -> FileResponse:
    if not _SAFE_NAME.match(name):
        raise HTTPException(400, "invalid name")
    if not _SAFE_FILE.match(filename):
        raise HTTPException(400, "invalid filename")
    p = project_root() / f"output_{name}" / filename
    if not p.exists():
        raise HTTPException(404, "file not found")
    ext = p.suffix.lower()
    media = {
        ".gif": "image/gif",
        ".mp4": "video/mp4",
        ".png": "image/png",
        ".json": "application/json",
    }.get(ext, "application/octet-stream")
    return FileResponse(str(p), media_type=media, filename=filename)


# ----------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
