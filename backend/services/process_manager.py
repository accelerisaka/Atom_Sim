"""
子进程管理 + WebSocket 流式日志广播。

每个 Job 用一个唯一 ID 标识，对应一个外部子进程（cursor_agent 或 run_xxx.py）。
- spawn 时把 stdout/stderr 合并行级读取，存入 ring buffer 并广播给所有订阅者。
- 进程结束后会保留状态，前端可以再 GET /api/jobs/{id} 查询。
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("process_manager")


@dataclass
class JobState:
    job_id: str
    kind: str  # "generate" / "run"
    name: str
    cwd: str
    cmd: List[str]
    status: str = "pending"  # pending / running / done / error / cancelled
    exit_code: Optional[int] = None
    lines: List[str] = field(default_factory=list)
    subscribers: List[asyncio.Queue] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    on_complete: Optional[Callable[["JobState"], Awaitable[None]]] = None
    process: Optional[asyncio.subprocess.Process] = None

    def to_public(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "name": self.name,
            "status": self.status,
            "exit_code": self.exit_code,
            "meta": self.meta,
            "line_count": len(self.lines),
        }


class ProcessManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobState] = {}

    def get(self, job_id: str) -> Optional[JobState]:
        return self._jobs.get(job_id)

    def list(self) -> List[Dict[str, Any]]:
        return [j.to_public() for j in self._jobs.values()]

    async def spawn(
        self,
        kind: str,
        name: str,
        cmd: List[str],
        cwd: str,
        env: Optional[Dict[str, str]] = None,
        meta: Optional[Dict[str, Any]] = None,
        on_complete: Optional[Callable[[JobState], Awaitable[None]]] = None,
    ) -> JobState:
        job_id = uuid.uuid4().hex[:12]
        job = JobState(
            job_id=job_id,
            kind=kind,
            name=name,
            cwd=cwd,
            cmd=cmd,
            meta=meta or {},
            on_complete=on_complete,
        )
        self._jobs[job_id] = job

        merged_env = os.environ.copy()
        merged_env.setdefault("PYTHONIOENCODING", "utf-8")
        merged_env.setdefault("PYTHONUNBUFFERED", "1")
        if env:
            merged_env.update(env)

        async def _runner() -> None:
            try:
                logger.info("spawn job=%s cmd=%s cwd=%s", job_id, cmd, cwd)
                if sys.platform == "win32":
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        cwd=cwd,
                        env=merged_env,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        creationflags=getattr(asyncio.subprocess, "CREATE_NO_WINDOW", 0),
                    )
                else:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        cwd=cwd,
                        env=merged_env,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                    )
                job.process = proc
                job.status = "running"
                await self._broadcast(job, {"type": "status", "status": "running"})

                assert proc.stdout is not None
                while True:
                    chunk = await proc.stdout.readline()
                    if not chunk:
                        break
                    try:
                        text = chunk.decode("utf-8", errors="replace").rstrip("\r\n")
                    except Exception:
                        text = repr(chunk)
                    job.lines.append(text)
                    if len(job.lines) > 5000:
                        del job.lines[: len(job.lines) - 5000]
                    await self._broadcast(job, {"type": "line", "text": text})

                rc = await proc.wait()
                job.exit_code = rc
                job.status = "done" if rc == 0 else "error"
                await self._broadcast(job, {"type": "status", "status": job.status, "exit_code": rc})

                if job.on_complete is not None:
                    try:
                        await job.on_complete(job)
                    except Exception as e:
                        logger.exception("on_complete callback failed")
                        await self._broadcast(job, {"type": "line", "text": f"[backend] on_complete error: {e}"})

                await self._broadcast(job, {"type": "end", "meta": job.meta})
            except FileNotFoundError as e:
                job.status = "error"
                job.exit_code = -1
                logger.exception("spawn failed (file not found)")
                await self._broadcast(job, {"type": "line", "text": f"[error] {e}"})
                await self._broadcast(job, {"type": "status", "status": "error", "exit_code": -1})
                await self._broadcast(job, {"type": "end", "meta": job.meta})
            except Exception as e:
                job.status = "error"
                job.exit_code = -1
                logger.exception("job runner crashed")
                await self._broadcast(job, {"type": "line", "text": f"[error] {e}"})
                await self._broadcast(job, {"type": "status", "status": "error", "exit_code": -1})
                await self._broadcast(job, {"type": "end", "meta": job.meta})

        asyncio.create_task(_runner())
        return job

    async def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.process is None:
            return False
        try:
            job.process.terminate()
        except ProcessLookupError:
            return False
        job.status = "cancelled"
        return True

    async def subscribe(self, job_id: str) -> Optional[asyncio.Queue]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        q: asyncio.Queue = asyncio.Queue(maxsize=1024)
        job.subscribers.append(q)
        # 回放历史
        for line in job.lines:
            await q.put({"type": "line", "text": line})
        await q.put({
            "type": "status",
            "status": job.status,
            "exit_code": job.exit_code,
        })
        if job.status in ("done", "error", "cancelled"):
            await q.put({"type": "end", "meta": job.meta})
        return q

    async def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        try:
            job.subscribers.remove(q)
        except ValueError:
            pass

    async def _broadcast(self, job: JobState, message: Dict[str, Any]) -> None:
        dead: List[asyncio.Queue] = []
        for q in list(job.subscribers):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            try:
                job.subscribers.remove(q)
            except ValueError:
                pass


process_manager = ProcessManager()
