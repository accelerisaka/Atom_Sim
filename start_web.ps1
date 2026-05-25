# ============================================================
# Atom_Sim Web 可视化一键启动脚本 (Windows / PowerShell)
# ------------------------------------------------------------
# 用法：
#   pwsh ./start_web.ps1            # 启动 backend + frontend
#   pwsh ./start_web.ps1 -Install   # 先安装依赖再启动
# ============================================================

param(
    [switch]$Install
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

if ($Install) {
    Write-Host "[setup] pip install backend deps ..." -ForegroundColor Cyan
    & python -m pip install -r "$root/backend/requirements.txt"

    Write-Host "[setup] npm install frontend deps ..." -ForegroundColor Cyan
    Push-Location "$root/frontend"
    & npm install
    Pop-Location

    Write-Host "[setup] npm install cursor_agent deps ..." -ForegroundColor Cyan
    Push-Location "$root/cursor_agent"
    & npm install
    Pop-Location
}

# 控制台 UTF-8（避免中文乱码）
chcp 65001 | Out-Null
$env:PYTHONIOENCODING = "utf-8"

Write-Host "[run] starting backend on http://127.0.0.1:8000" -ForegroundColor Green
$backend = Start-Process -PassThru -FilePath "python" -ArgumentList @(
    "-m", "uvicorn", "backend.main:app", "--port", "8000", "--host", "127.0.0.1"
) -WorkingDirectory $root -NoNewWindow

Start-Sleep -Seconds 2

Write-Host "[run] starting frontend on http://127.0.0.1:5173" -ForegroundColor Green
try {
    Push-Location "$root/frontend"
    & npm run dev
} finally {
    Pop-Location
    if ($backend -and -not $backend.HasExited) {
        Write-Host "[exit] stopping backend (pid=$($backend.Id))" -ForegroundColor Yellow
        Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    }
}
