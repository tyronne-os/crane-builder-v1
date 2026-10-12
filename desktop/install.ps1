# One-time setup: puts a Crane icon on the desktop and installs the Crane Hands computer-use agent.
# Run in Windows PowerShell:
#   irm https://raw.githubusercontent.com/tyronne-os/crane-builder-v1/main/desktop/install.ps1 | iex
$ErrorActionPreference = 'Stop'
$Raw = 'https://raw.githubusercontent.com/tyronne-os/crane-builder-v1/main'
$Dir = Join-Path $env:LOCALAPPDATA 'Crane'
New-Item -ItemType Directory -Force -Path $Dir | Out-Null

Write-Host 'Downloading Crane files...'
Invoke-WebRequest "$Raw/desktop/Crane.ps1"      -OutFile "$Dir\Crane.ps1"
Invoke-WebRequest "$Raw/desktop/crane.ico"      -OutFile "$Dir\crane.ico"
Invoke-WebRequest "$Raw/scripts/crane_hands.py" -OutFile "$Dir\crane_hands.py"
Invoke-WebRequest "$Raw/scripts/voice_server.py" -OutFile "$Dir\voice_server.py"

$py = (Get-Command py -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command python -ErrorAction SilentlyContinue).Source }
if ($py) {
  Write-Host 'Installing computer-use and voice dependencies (pyautogui, pillow, kokoro)...'
  & $py -m pip install --user --quiet --upgrade pyautogui pillow kokoro soundfile fastapi uvicorn
} else {
  Write-Warning 'Windows Python not found. Crane will open, but computer use needs Python: winget install Python.Python.3.12  then re-run this installer.'
}

$lnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Crane.lnk'
$sh = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
$sh.TargetPath   = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$sh.Arguments    = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Dir\Crane.ps1`""
$sh.IconLocation = "$Dir\crane.ico"
$sh.WorkingDirectory = $Dir
$sh.Description  = 'Crane Builder'
$sh.Save()
Write-Host "Done. Double-click 'Crane' on your desktop."
