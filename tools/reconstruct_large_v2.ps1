param(
    [string]$ManifestPath = (Join-Path $PSScriptRoot '..\models\faster-whisper-large-v2\release-manifest.json'),
    [string]$PartDirectory = (Join-Path $PSScriptRoot '..\release-assets\large-v2'),
    [string]$ModelDirectory = (Join-Path $PSScriptRoot '..\models\faster-whisper-large-v2'),
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

function Get-Sha256Hex {
    param([string]$Path)
    return (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLowerInvariant()
}

if (-not (Test-Path $ManifestPath)) {
    throw "Manifest not found: $ManifestPath"
}

$manifest = Get-Content -Path $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $ModelDirectory | Out-Null

$targetPath = Join-Path $ModelDirectory $manifest.target_file
if ((Test-Path $targetPath) -and -not $Force) {
    $currentHash = Get-Sha256Hex -Path $targetPath
    if ($currentHash -eq $manifest.final_sha256) {
        Write-Host "model.bin already exists and matches manifest."
        exit 0
    }
}

foreach ($part in $manifest.parts) {
    $partPath = Join-Path $PartDirectory $part.file
    if (-not (Test-Path $partPath)) {
        throw "Missing part file: $partPath"
    }
    $partHash = Get-Sha256Hex -Path $partPath
    if ($partHash -ne $part.sha256) {
        throw "Checksum mismatch for $($part.file). Expected $($part.sha256), got $partHash"
    }
}

$tempPath = "$targetPath.tmp"
if (Test-Path $tempPath) {
    Remove-Item -LiteralPath $tempPath -Force
}

$output = [System.IO.File]::Open($tempPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
try {
    foreach ($part in $manifest.parts) {
        $partPath = Join-Path $PartDirectory $part.file
        $input = [System.IO.File]::OpenRead($partPath)
        try {
            $input.CopyTo($output)
        }
        finally {
            $input.Dispose()
        }
    }
}
finally {
    $output.Dispose()
}

$finalHash = Get-Sha256Hex -Path $tempPath
if ($finalHash -ne $manifest.final_sha256) {
    Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
    throw "Final model hash mismatch. Expected $($manifest.final_sha256), got $finalHash"
}

Move-Item -LiteralPath $tempPath -Destination $targetPath -Force
Set-Content -Path (Join-Path $ModelDirectory 'model.bin.sha256') -Value $finalHash -Encoding ASCII

Write-Host "Reconstructed model:"
Write-Host $targetPath
Write-Host "SHA256: $finalHash"
