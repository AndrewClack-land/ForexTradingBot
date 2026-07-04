# ── Windows Forms + Drawing ───────────────────────────────────────────────────
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# P/Invoke для приостановки/возобновления процесса на уровне ОС
if (-not ([System.Management.Automation.PSTypeName]'NtProcess').Type) {
    Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class NtProcess {
    [DllImport("ntdll.dll")]
    public static extern int NtSuspendProcess(IntPtr processHandle);
    [DllImport("ntdll.dll")]
    public static extern int NtResumeProcess(IntPtr processHandle);
}
"@
}

# ── UTF-8 ─────────────────────────────────────────────────────────────────────
$OutputEncoding            = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding  = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING      = "utf-8"
$env:PYTHONUTF8            = "1"

# ── Переменные окружения ──────────────────────────────────────────────────────
$env:MT5_EXECUTION      = "1"
$env:MT5_SERVER         = "FxPro-MT5 Demo"
$env:MT5_MAGIC          = "20260318"
$env:MT5_RISK_PCT       = "0.01"
$env:DEBUG_RAW_SIGNALS  = "0"
$env:PYTHONUNBUFFERED   = "1"

# ── Загружаем секреты из .env ─────────────────────────────────────────────────
$envFile = "$PSScriptRoot\.env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
        }
    }
} else {
    [System.Windows.Forms.MessageBox]::Show(
        ".env file not found - MT5_LOGIN and MT5_PASSWORD must be set manually",
        "Warning",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Warning
    ) | Out-Null
}

# ── Python executable (venv или системный) ────────────────────────────────────
$pythonExe = if (Test-Path "$PSScriptRoot\.venv\Scripts\python.exe") {
    "$PSScriptRoot\.venv\Scripts\python.exe"
} else { "python" }

$logPath = "$PSScriptRoot\ai_data\bot.log"

# ── Форма ─────────────────────────────────────────────────────────────────────
$form                 = New-Object System.Windows.Forms.Form
$form.Text            = "Forex Bot Control"
$form.Size            = New-Object System.Drawing.Size(320, 200)
$form.StartPosition   = "CenterScreen"
$form.FormBorderStyle = "FixedSingle"
$form.MaximizeBox     = $false
$form.BackColor       = [System.Drawing.Color]::FromArgb(28, 28, 28)
$form.ForeColor       = [System.Drawing.Color]::WhiteSmoke

$lblTitle            = New-Object System.Windows.Forms.Label
$lblTitle.Text       = "Forex Bot Control"
$lblTitle.Location   = New-Object System.Drawing.Point(0, 14)
$lblTitle.Size       = New-Object System.Drawing.Size(304, 24)
$lblTitle.Font       = New-Object System.Drawing.Font("Segoe UI", 11, [System.Drawing.FontStyle]::Bold)
$lblTitle.TextAlign  = "MiddleCenter"
$form.Controls.Add($lblTitle)

$lblStatus           = New-Object System.Windows.Forms.Label
$lblStatus.Text      = "Status: STOPPED"
$lblStatus.Location  = New-Object System.Drawing.Point(0, 48)
$lblStatus.Size      = New-Object System.Drawing.Size(304, 22)
$lblStatus.Font      = New-Object System.Drawing.Font("Segoe UI", 10)
$lblStatus.TextAlign = "MiddleCenter"
$lblStatus.ForeColor = [System.Drawing.Color]::Tomato
$form.Controls.Add($lblStatus)

# ── Кнопки ───────────────────────────────────────────────────────────────────
function New-BotButton($text, $x, $bgColor, $fgColor) {
    $btn                            = New-Object System.Windows.Forms.Button
    $btn.Text                       = $text
    $btn.Location                   = New-Object System.Drawing.Point($x, 96)
    $btn.Size                       = New-Object System.Drawing.Size(84, 40)
    $btn.BackColor                  = $bgColor
    $btn.ForeColor                  = $fgColor
    $btn.FlatStyle                  = "Flat"
    $btn.FlatAppearance.BorderSize  = 0
    $btn.Font                       = New-Object System.Drawing.Font("Segoe UI", 9, [System.Drawing.FontStyle]::Bold)
    $btn.Cursor                     = [System.Windows.Forms.Cursors]::Hand
    $form.Controls.Add($btn)
    return $btn
}

$btnStart = New-BotButton "▶  Start"  18  ([System.Drawing.Color]::FromArgb(40, 167, 69))  ([System.Drawing.Color]::White)
$btnPause = New-BotButton "⏸  Pause" 110  ([System.Drawing.Color]::FromArgb(255, 193, 7))  ([System.Drawing.Color]::Black)
$btnStop  = New-BotButton "⏹  Stop"  202  ([System.Drawing.Color]::FromArgb(220, 53, 69))  ([System.Drawing.Color]::White)

$btnPause.Enabled = $false
$btnStop.Enabled  = $false

# ── Состояние ─────────────────────────────────────────────────────────────────
$script:botProcess = $null
$script:isPaused   = $false
$script:outJob     = $null
$script:errJob     = $null

# ── Обработчики ──────────────────────────────────────────────────────────────
$btnStart.Add_Click({
    # Возобновление из паузы
    if ($script:isPaused) {
        [NtProcess]::NtResumeProcess($script:botProcess.Handle) | Out-Null
        $script:isPaused  = $false
        $lblStatus.Text   = "Status: RUNNING"
        $lblStatus.ForeColor = [System.Drawing.Color]::LimeGreen
        $btnStart.Text    = "▶  Start"
        $btnStart.Enabled = $false
        $btnPause.Enabled = $true
        return
    }

    if ($script:botProcess -and !$script:botProcess.HasExited) { return }

    # Останавливаем предыдущий экземпляр бота по PID-файлу
    # (не трогаем остальные python-процессы на машине)
    $pidFile = "$PSScriptRoot\ai_data\bot.pid"
    if (Test-Path $pidFile) {
        $oldPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
        if ($oldPid -match '^\d+$') {
            $oldProc = Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue
            if ($oldProc -and $oldProc.ProcessName -like 'python*') {
                Stop-Process -Id $oldProc.Id -Force -ErrorAction SilentlyContinue
            }
        }
        Remove-Item $pidFile -ErrorAction SilentlyContinue
    }
    "" | Set-Content -LiteralPath $logPath -Encoding UTF8

    # Запускаем бота
    $psi                        = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName               = $pythonExe
    $psi.Arguments              = "main.py"
    $psi.WorkingDirectory       = $PSScriptRoot
    $psi.UseShellExecute        = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.CreateNoWindow         = $true

    $script:botProcess = New-Object System.Diagnostics.Process
    $script:botProcess.StartInfo = $psi

    $lp = $logPath
    $script:outJob = Register-ObjectEvent -InputObject $script:botProcess -EventName OutputDataReceived -MessageData $lp -Action {
        if ($null -ne $Event.SourceEventArgs.Data) {
            Add-Content -LiteralPath $Event.MessageData -Value $Event.SourceEventArgs.Data -Encoding UTF8
        }
    }
    $script:errJob = Register-ObjectEvent -InputObject $script:botProcess -EventName ErrorDataReceived -MessageData $lp -Action {
        if ($null -ne $Event.SourceEventArgs.Data) {
            Add-Content -LiteralPath $Event.MessageData -Value $Event.SourceEventArgs.Data -Encoding UTF8
        }
    }

    $script:botProcess.Start() | Out-Null
    $script:botProcess.BeginOutputReadLine()
    $script:botProcess.BeginErrorReadLine()

    $lblStatus.Text      = "Status: RUNNING"
    $lblStatus.ForeColor = [System.Drawing.Color]::LimeGreen
    $btnStart.Enabled    = $false
    $btnPause.Enabled    = $true
    $btnStop.Enabled     = $true
})

$btnPause.Add_Click({
    if ($script:botProcess -and !$script:botProcess.HasExited -and !$script:isPaused) {
        [NtProcess]::NtSuspendProcess($script:botProcess.Handle) | Out-Null
        $script:isPaused     = $true
        $lblStatus.Text      = "Status: PAUSED"
        $lblStatus.ForeColor = [System.Drawing.Color]::Gold
        $btnStart.Text       = "▶  Resume"
        $btnStart.Enabled    = $true
        $btnPause.Enabled    = $false
    }
})

$btnStop.Add_Click({
    if ($script:outJob) { Unregister-Event -SourceIdentifier $script:outJob.Name -ErrorAction SilentlyContinue; Remove-Job $script:outJob -Force -ErrorAction SilentlyContinue; $script:outJob = $null }
    if ($script:errJob) { Unregister-Event -SourceIdentifier $script:errJob.Name -ErrorAction SilentlyContinue; Remove-Job $script:errJob -Force -ErrorAction SilentlyContinue; $script:errJob = $null }
    if ($script:botProcess -and !$script:botProcess.HasExited) {
        if ($script:isPaused) { [NtProcess]::NtResumeProcess($script:botProcess.Handle) | Out-Null }
        $script:botProcess.Kill()
        $script:botProcess.WaitForExit(3000) | Out-Null
    }
    $script:botProcess   = $null
    $script:isPaused     = $false
    $lblStatus.Text      = "Status: STOPPED"
    $lblStatus.ForeColor = [System.Drawing.Color]::Tomato
    $btnStart.Text       = "▶  Start"
    $btnStart.Enabled    = $true
    $btnPause.Enabled    = $false
    $btnStop.Enabled     = $false
})

$form.Add_FormClosing({
    if ($script:outJob) { Unregister-Event -SourceIdentifier $script:outJob.Name -ErrorAction SilentlyContinue; Remove-Job $script:outJob -Force -ErrorAction SilentlyContinue }
    if ($script:errJob) { Unregister-Event -SourceIdentifier $script:errJob.Name -ErrorAction SilentlyContinue; Remove-Job $script:errJob -Force -ErrorAction SilentlyContinue }
    if ($script:botProcess -and !$script:botProcess.HasExited) {
        if ($script:isPaused) { [NtProcess]::NtResumeProcess($script:botProcess.Handle) | Out-Null }
        $script:botProcess.Kill()
    }
})

[System.Windows.Forms.Application]::Run($form)
