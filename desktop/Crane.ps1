# Crane desktop launcher: starts Crane Hands (computer use) if needed, then opens Crane as an app window.
$ErrorActionPreference = 'SilentlyContinue'
$Dir   = Join-Path $env:LOCALAPPDATA 'Crane'
$Url   = 'https://aibruh-crane-builder-v1.static.hf.space/'
$Token = ''

$listening = Get-NetTCPConnection -LocalPort 8765 -State Listen
if (-not $listening) {
  $pyw = (Get-Command pyw).Source
  if (-not $pyw) { $pyw = (Get-Command pythonw).Source }
  if ($pyw) {
    Start-Process -FilePath $pyw -ArgumentList "`"$Dir\crane_hands.py`"" -WindowStyle Hidden
    for ($i = 0; $i -lt 20 -and -not (Test-Path "$env:USERPROFILE\.crane\token.txt"); $i++) { Start-Sleep -Milliseconds 250 }
  }
}
if (Test-Path "$env:USERPROFILE\.crane\token.txt") { $Token = (Get-Content "$env:USERPROFILE\.crane\token.txt" -Raw).Trim() }
if ($Token) { $Url = "$Url#cu=$Token" }

$browsers = @(
  "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
  "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
  "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe",
  "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
  "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe"
)
$browser = $browsers | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($browser) { Start-Process -FilePath $browser -ArgumentList "--app=$Url", '--start-maximized' }
else { Start-Process $Url }
