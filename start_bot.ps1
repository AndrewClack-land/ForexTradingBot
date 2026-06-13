# ── Убиваем старые процессы и чистим PID-файл ──────────────────────────────
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
Remove-Item "$PSScriptRoot\ai_data\bot.pid" -ErrorAction SilentlyContinue

# ── UTF-8 console encoding (fixes UnicodeEncodeError for → and other symbols) ─
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING  = "utf-8"
$env:PYTHONUTF8        = "1"

# ── Переменные окружения ─────────────────────────────────────────────────────
$env:MT5_EXECUTION      = "1"
$env:MT5_SERVER         = "FxPro-MT5 Demo"
$env:MT5_MAGIC          = "20260318"
$env:MT5_RISK_PCT       = "0.01"
$env:DEBUG_RAW_SIGNALS  = "0"   # поставь "1" для отладки сигналов
$env:PYTHONUNBUFFERED   = "1"   # отключаем буферизацию — логи видны сразу

# ── Загружаем секреты из .env ─────────────────────────────────────────────────
$envFile = "$PSScriptRoot\.env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
        }
    }
} else {
    Write-Host "[WARN] .env file not found — MT5_LOGIN and MT5_PASSWORD must be set manually" -ForegroundColor Yellow
}
# ── Активация venv (если используется) ──────────────────────────────────────
$venv = "$PSScriptRoot\.venv\Scripts\Activate.ps1"
if (Test-Path $venv) { & $venv }

# ── Запуск ───────────────────────────────────────────────────────────────────
Set-Location $PSScriptRoot
python main.py *>&1 | Tee-Object -FilePath .\ai_data\bot.log
