# Install the validated desktop environment in this checkout.
$ErrorActionPreference = 'Stop'
$lunarRoot = Split-Path -Parent $PSScriptRoot
$lunarVenv = Join-Path $lunarRoot '.venv'
$lunarPython = Join-Path $lunarVenv 'Scripts\python.exe'
$lunarRequirements = Join-Path $lunarRoot 'requirements-validated-windows.txt'

function Assert-LunarPython([string]$Executable) {
    & $Executable -c "import sys, struct; assert sys.version_info[:2] == (3,13) and struct.calcsize('P') == 8, 'Lunar OD requires 64-bit Python 3.13'; assert sys.prefix != sys.base_prefix, 'A virtual environment is required'"
    if ($LASTEXITCODE -ne 0) { throw 'Use 64-bit Python 3.13. Existing environments are preserved.' }
}

function Find-LunarPython {
    foreach ($lunarCommandName in @('py', 'python')) {
        $lunarCommand = Get-Command $lunarCommandName -CommandType Application -ErrorAction SilentlyContinue
        if (-not $lunarCommand) { continue }
        $lunarArguments = @('-c', "import sys, struct; assert sys.version_info[:2] == (3,13) and struct.calcsize('P') == 8; print(sys.executable)")
        if ($lunarCommandName -eq 'py') { $lunarArguments = @('-3.13') + $lunarArguments }
        try { $lunarCandidate = & $lunarCommand.Source @lunarArguments 2>$null } catch { continue }
        if ($LASTEXITCODE -eq 0 -and $lunarCandidate -and (Test-Path -LiteralPath $lunarCandidate)) {
            return [string]$lunarCandidate
        }
    }
    throw 'Install 64-bit Python 3.13 from python.org, then run setup_windows.cmd again.'
}

try {
    Push-Location -LiteralPath $lunarRoot
    try {
        if (-not (Test-Path -LiteralPath $lunarPython)) {
            if (Test-Path -LiteralPath $lunarVenv) {
                throw '.venv exists without a usable Python executable. Preserve or rename it before retrying.'
            }
            $lunarBasePython = Find-LunarPython
            & $lunarBasePython -m venv $lunarVenv
            if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv.' }
        }
        Assert-LunarPython $lunarPython
        & $lunarPython -m pip install -r $lunarRequirements
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check the pip output above.' }
        & $lunarPython -m pip install -e $lunarRoot
        if ($LASTEXITCODE -ne 0) { throw 'Package installation failed.' }
        & $lunarPython -m pip check
        if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
        & $lunarPython -c "import lunar_od, scipy, numba, spiceypy, pandas, matplotlib, pyqtgraph, h5py; from PyQt5 import QtWidgets; print('Lunar OD desktop imports: OK')"
        if ($LASTEXITCODE -ne 0) { throw 'Desktop import check failed.' }
        Write-Host 'Setup complete. Double-click launch_lunar_od.cmd.'
        Write-Host 'Scientific runs also require SPICE kernels: see docs/windows-quickstart.md.'
    } finally {
        Pop-Location
    }
} catch {
    Write-Host ('Lunar OD setup failed: ' + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
exit 0
