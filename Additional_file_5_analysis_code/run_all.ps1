param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$archiveRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$dataDir = Join-Path $archiveRoot "data"
$inputFile = Get-ChildItem -LiteralPath $dataDir -Filter "*.xlsx" -File | Select-Object -First 1
$inputPath = if ($null -eq $inputFile) { "" } else { $inputFile.FullName }
$expectedHash = "2c4e4f510154ab74ebfd1d9d17b2be7d88b98d3d51f2007a8f029cb39d5861aa"
$scriptRoot = Join-Path $archiveRoot "analysis\scripts\v4.0"

if ([string]::IsNullOrWhiteSpace($inputPath)) {
    throw "Required authorized .xlsx workbook not found in $dataDir"
}

$sha256 = [System.Security.Cryptography.SHA256]::Create()
$stream = [System.IO.File]::OpenRead($inputPath)
try {
    $actualHash = ([System.BitConverter]::ToString($sha256.ComputeHash($stream))).Replace("-", "").ToLowerInvariant()
}
finally {
    $stream.Dispose()
    $sha256.Dispose()
}
if ($actualHash -ne $expectedHash) {
    throw "Input SHA-256 mismatch. Expected $expectedHash; observed $actualHash"
}

$scripts = @(
    "44_cpp_rhgh_integrated_analysis.py",
    "49_audit_hospital_icc.py",
    "50_audit_growth_methods.py",
    "60_gnrha_secondary_analysis.py",
    "67_audit_gnrha_record_classification.py",
    "53_nature_figures_audited.py",
    "63_plot_gnrha_effectiveness.py",
    "69_redraw_figure1_professional_labels.py"
)

foreach ($script in $scripts) {
    $scriptPath = Join-Path $scriptRoot $script
    Write-Host "Running $script"
    & $Python -X utf8 $scriptPath
    if ($LASTEXITCODE -ne 0) {
        throw "$script failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Analysis pipeline completed."
