# start.ps1 — 在 cursor_agent 目录下执行: .\start.ps1 [传给 index.ts 的参数...]
$ErrorActionPreference = "Stop"
$CursorSdk = "D:\Cursor\CursorSdk"
$env:NODE_PATH = Join-Path $CursorSdk "node_modules"
$rg = "D:\Cursor\CursorSdk\node_modules\@cursor\sdk-win32-x64\bin\rg.exe"
if (-not (Test-Path $rg)) {
  throw "未找到 rg.exe，请在 D:\Cursor\CursorSdk 执行 npm install"
}
$env:CURSOR_RIPGREP_PATH = $rg
$tsx = Join-Path $CursorSdk "node_modules\.bin\tsx.cmd"

if (-not (Test-Path $tsx)) {
  throw "未找到 tsx，请先在 $CursorSdk 执行 npm install"
}

& $tsx (Join-Path $PSScriptRoot "src\index.ts") @args