<#
.SYNOPSIS
    Starts the paper execution server and a Cloudflare tunnel, then prints the
    public webhook URL to paste into TradingView.

.DESCRIPTION
    Both processes are detached and keep running after this script exits. Their
    PIDs are recorded in logs/run_bot.pids so -Stop can tear down exactly what
    was started, rather than every cloudflared on the machine.

    The quick-tunnel hostname is regenerated on every launch, so the TradingView
    alert has to be re-pointed each time this is run.

.EXAMPLE
    .\run_bot.ps1
    .\run_bot.ps1 -Stop
#>
[CmdletBinding()]
param(
    [switch]$Stop,
    [int]$Port = 5000
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root    = $PSScriptRoot
$LogDir  = Join-Path $Root 'logs'
$PidFile = Join-Path $LogDir 'run_bot.pids'
$EnvFile = Join-Path $Root '.env'
$Python  = Join-Path $Root '.venv\Scripts\python.exe'
$SrvOut  = Join-Path $LogDir 'server.log'
$SrvErr  = Join-Path $LogDir 'server.err'
$CfOut   = Join-Path $LogDir 'cloudflared.log'
$CfErr   = Join-Path $LogDir 'cloudflared.err'

function Stop-Stack {
    if (Test-Path $PidFile) {
        foreach ($line in Get-Content $PidFile) {
            if ($line -match '^\d+$') {
                Get-Process -Id ([int]$line) -ErrorAction SilentlyContinue |
                    Stop-Process -Force -ErrorAction SilentlyContinue
            }
        }
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }
    # Anything else squatting on the port would make uvicorn die with WinError 10048.
    Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }

    # Ctrl-C during startup can orphan a tunnel that outlives the PID file, and
    # it would keep serving a hostname with nothing behind it. Match on the
    # command line so an unrelated cloudflared on this machine is left alone.
    Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*localhost:$Port*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

    Start-Sleep -Seconds 1
}

if ($Stop) {
    Stop-Stack
    Write-Host 'Stopped server and tunnel.' -ForegroundColor Yellow
    return
}

# --- preflight -------------------------------------------------------------
if (-not (Test-Path $Python)) {
    throw "Virtualenv python not found at $Python. Create it with: python -m venv .venv"
}

# A shell opened before the cloudflared MSI ran still holds the old PATH.
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
            [Environment]::GetEnvironmentVariable('Path', 'User')

$Cloudflared = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $Cloudflared) {
    $Cloudflared = @(
        'C:\Program Files (x86)\cloudflared\cloudflared.exe',
        'C:\Program Files\cloudflared\cloudflared.exe'
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $Cloudflared) {
    throw 'cloudflared not found. Install it with: winget install Cloudflare.cloudflared'
}

# The tunnel is public, so the endpoint must require a token before it opens.
$secret = $null
if (Test-Path $EnvFile) {
    $hit = Select-String -Path $EnvFile -Pattern '^\s*WEBHOOK_SECRET\s*=\s*(\S+)' | Select-Object -First 1
    if ($hit) { $secret = $hit.Matches[0].Groups[1].Value }
}
if (-not $secret) {
    # Get-Random -Count samples without replacement, so asking for more chars
    # than the alphabet holds silently returns a shuffle of it. Use the crypto
    # RNG and base64url instead: 24 bytes -> 32 URL-safe chars, 192 bits.
    $bytes = [byte[]]::new(24)
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $secret = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
    $raw = if (Test-Path $EnvFile) { Get-Content -Raw $EnvFile } else { '' }
    if ($raw -and -not $raw.EndsWith("`n")) { Add-Content -Path $EnvFile -Value '' }
    Add-Content -Path $EnvFile -Value "WEBHOOK_SECRET=$secret"
    Write-Host 'Generated WEBHOOK_SECRET and appended it to .env' -ForegroundColor Yellow
}

# config.py calls load_dotenv(override=False), so a WEBHOOK_SECRET already in
# the environment silently outranks .env. The child process would then enforce
# a different token than the URL printed below, which reads as a 401 with no
# obvious cause. Reconcile the two before launching.
if ($env:WEBHOOK_SECRET -and $env:WEBHOOK_SECRET -ne $secret) {
    Write-Host 'WARNING: WEBHOOK_SECRET is set in this shell and overrides .env.' -ForegroundColor Yellow
    Write-Host '         Using the shell value. To prefer .env instead, run:' -ForegroundColor Yellow
    Write-Host '           Remove-Item Env:WEBHOOK_SECRET' -ForegroundColor Yellow
    $secret = $env:WEBHOOK_SECRET
}
$env:WEBHOOK_SECRET = $secret

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Stop-Stack

# --- launch ----------------------------------------------------------------
Write-Host 'Starting paper execution server ...' -NoNewline
$server = Start-Process -FilePath $Python -ArgumentList '-m', 'core.webhook_server' `
    -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $SrvOut -RedirectStandardError $SrvErr
# Recorded before the readiness wait so -Stop can clean up after a Ctrl-C here.
"$($server.Id)" | Set-Content -Path $PidFile

$health = $null
foreach ($i in 1..60) {
    Start-Sleep -Milliseconds 500
    try { $health = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 5; break } catch { }
}
if (-not $health) {
    Write-Host ' FAILED' -ForegroundColor Red
    Get-Content $SrvErr -ErrorAction SilentlyContinue | Select-Object -Last 15
    Stop-Stack
    throw "Server did not answer on port $Port. See $SrvErr"
}
Write-Host " ok (model $($health.model), auth_required $($health.auth_required))" -ForegroundColor Green

Write-Host 'Opening Cloudflare tunnel ...' -NoNewline
Remove-Item $CfOut, $CfErr -Force -ErrorAction SilentlyContinue
$tunnel = Start-Process -FilePath $Cloudflared `
    -ArgumentList 'tunnel', '--url', "http://localhost:$Port" `
    -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $CfOut -RedirectStandardError $CfErr
Add-Content -Path $PidFile -Value "$($tunnel.Id)"

$publicUrl = $null
foreach ($i in 1..60) {
    Start-Sleep -Seconds 1
    $text = (Get-Content $CfOut, $CfErr -ErrorAction SilentlyContinue) -join "`n"
    $m = [regex]::Match($text, 'https://[a-z0-9-]+\.trycloudflare\.com')
    if ($m.Success) { $publicUrl = $m.Value; break }
}
if (-not $publicUrl) {
    Write-Host ' FAILED' -ForegroundColor Red
    Get-Content $CfErr -ErrorAction SilentlyContinue | Select-Object -Last 20
    Stop-Stack
    throw "cloudflared never published a URL. See $CfErr"
}
Write-Host ' ok' -ForegroundColor Green

# Cloudflare warns the hostname takes a moment to become reachable; confirm it
# rather than handing over a URL that fails on first use.
#
# This network's resolver (10.100.70.x) serves trycloudflare.com but not its
# freshly created subdomains, so resolve through a public resolver and pin the
# address. TradingView resolves independently and is unaffected by that policy.
Write-Host 'Verifying public reachability ...' -NoNewline
$hostName = ([Uri]$publicUrl).Host

# The A record is published a few seconds after the tunnel registers, so the
# lookup is retried alongside the request. The whole check is capped, because a
# verification step is not worth making the launch feel hung.
$edgeIp = $null
$code = ''
$deadline = (Get-Date).AddSeconds(60)
while ((Get-Date) -lt $deadline) {
    if (-not $edgeIp) {
        # A lookup issued before the record is published leaves an NXDOMAIN in
        # the Windows resolver cache, and every later retry reads that instead
        # of asking again. Drop the cache each time or the loop can never win.
        Clear-DnsClientCache -ErrorAction SilentlyContinue
        try {
            $edgeIp = (Resolve-DnsName $hostName -Type A -Server 1.1.1.1 -QuickTimeout -ErrorAction Stop |
                       Where-Object IPAddress | Select-Object -First 1).IPAddress
        } catch { }
    }
    if ($edgeIp) {
        $code = curl.exe -s -o NUL -m 10 -w '%{http_code}' --resolve "$($hostName):443:$edgeIp" "$publicUrl/health"
        if ($code -eq '200') { break }
    }
    Start-Sleep -Seconds 2
}

$localDnsOk = $true
try { Resolve-DnsName $hostName -Type A -ErrorAction Stop | Out-Null } catch { $localDnsOk = $false }

if ($code -eq '200') {
    Write-Host ' ok (HTTP 200 from the public edge)' -ForegroundColor Green
} else {
    Write-Host " could not confirm (last HTTP code '$code')" -ForegroundColor Yellow
}
if (-not $localDnsOk -and $edgeIp) {
    Write-Host '  note: this machine cannot resolve the hostname (local DNS policy).' -ForegroundColor DarkGray
    Write-Host '        TradingView is unaffected. To test from here, use:' -ForegroundColor DarkGray
    Write-Host "        curl.exe --resolve $($hostName):443:$edgeIp $publicUrl/health" -ForegroundColor DarkGray
}

# --- report ----------------------------------------------------------------
$webhookUrl = "$publicUrl/webhook?token=$secret"

Write-Host ''
Write-Host '==================== TRADINGVIEW ALERT ====================' -ForegroundColor Cyan
Write-Host ''
Write-Host '  Notifications tab -> Webhook URL:' -ForegroundColor Cyan
Write-Host "    $webhookUrl"
Write-Host ''
Write-Host '  Settings tab -> Message:' -ForegroundColor Cyan
Write-Host '    {{strategy.order.alert_message}}'
Write-Host ''
Write-Host '===========================================================' -ForegroundColor Cyan
Write-Host ''
Write-Host "  server pid : $($server.Id)   tunnel pid : $($tunnel.Id)"
Write-Host "  logs       : $LogDir"
Write-Host "  trades     : $($health.trade_log)"
Write-Host '  stop       : .\run_bot.ps1 -Stop'
Write-Host ''
