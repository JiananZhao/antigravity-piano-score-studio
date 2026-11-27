param(
    [switch]$InstallAll,
    [switch]$InstallTools,
    [switch]$DownloadSalamander,
    [switch]$DownloadMuScriptor,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if ($InstallAll) {
    $InstallTools = $true
    $DownloadSalamander = $true
    $DownloadMuScriptor = $true
}

Write-Host "=== Piano Score Studio Plugin Bootstrap (Windows) ==="
if (-not (Test-Path ".venv")) {
    & $PythonExe -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create Python virtual environment." }
}

$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
& $VenvPy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }
& $VenvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Failed to install Python dependencies from requirements.txt." }

if (-not (Test-Path "config.json")) {
    Copy-Item "config.example.json" "config.json"
    Write-Host "Created config.json from config.example.json"
}

if ($InstallTools) {
    $msCandidates = @(
        "C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
        "C:\Program Files (x86)\MuseScore 4\bin\MuseScore4.exe",
        "E:\AI\Playground\PianoTranscription\tools\MuseScorePortable\App\MuseScore\bin\MuseScore4.exe"
    )
    $msFound = ($msCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1)
    if (-not $msFound) {
        $msCmd = Get-Command MuseScore4, mscore, musescore -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($msCmd) { $msFound = $msCmd.Source }
    }
    if (-not $msFound) {
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if ($winget) {
            Write-Host "Installing MuseScore 4 via winget..."
            & winget install --id MuseScore.MuseScore -e --accept-source-agreements --accept-package-agreements
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "winget could not install MuseScore 4 automatically. Please install manually from https://musescore.org"
            }
        } else {
            Write-Warning "winget not found. Please install MuseScore 4 manually from https://musescore.org"
        }
    } else {
        Write-Host "MuseScore 4 already present: $msFound"
    }
}

$SetupScript = Join-Path $Root "skills\piano-score-studio\scripts\setup_assets.py"
$SetupArgs = @("--write-config", "--check")
if ($InstallTools) { $SetupArgs += "--download-sfizz" }
if ($DownloadSalamander) { $SetupArgs += "--download-salamander" }
if ($DownloadMuScriptor) { $SetupArgs += "--download-muscriptor" }

& $VenvPy $SetupScript @SetupArgs
if ($LASTEXITCODE -ne 0) { throw "setup_assets.py failed." }

if (-not ($InstallTools -or $DownloadSalamander -or $DownloadMuScriptor)) {
    Write-Host "`nTip: Run '.\install.ps1 -InstallAll' to auto-install MuseScore 4, sfizz_render, Salamander Grand Piano V3 (~714 MB), and MuScriptor (if HF logged in)."
}
