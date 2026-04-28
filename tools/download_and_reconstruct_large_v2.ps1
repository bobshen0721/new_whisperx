param(
    [Parameter(Mandatory = $true)][string]$Owner,
    [Parameter(Mandatory = $true)][string]$Repo,
    [string]$Tag = 'model-large-v2',
    [string]$ManifestPath = (Join-Path $PSScriptRoot '..\models\faster-whisper-large-v2\release-manifest.json'),
    [string]$PartDirectory = (Join-Path $PSScriptRoot '..\release-assets\large-v2'),
    [switch]$ForceDownload,
    [switch]$ForceRebuild
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Get-Sha256Hex {
    param([string]$Path)
    return (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLowerInvariant()
}

if (-not (Test-Path $ManifestPath)) {
    throw "Manifest not found: $ManifestPath"
}

$manifest = Get-Content -Path $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $PartDirectory | Out-Null

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$driveName = [System.IO.Path]::GetPathRoot($repoRoot).TrimEnd('\').TrimEnd(':')
$drive = Get-PSDrive -Name $driveName
$requiredBytes = ([int64]$manifest.final_size_bytes * 2) + 1073741824
if ($drive.Free -lt $requiredBytes) {
    $freeGb = [Math]::Round($drive.Free / 1GB, 2)
    $requiredGb = [Math]::Round($requiredBytes / 1GB, 2)
    throw "Not enough free disk space on drive ${driveName}:. Free: ${freeGb} GB. Required: at least ${requiredGb} GB."
}

$baseUrl = "https://github.com/$Owner/$Repo/releases/download/$Tag"

foreach ($part in $manifest.parts) {
    $partPath = Join-Path $PartDirectory $part.file
    $partialPath = "$partPath.partial"
    $needsDownload = $true

    if ((Test-Path $partPath) -and -not $ForceDownload) {
        $existingHash = Get-Sha256Hex -Path $partPath
        if ($existingHash -eq $part.sha256) {
            $needsDownload = $false
            Write-Host "Using existing part $($part.file)"
        }
    }

    if ($needsDownload) {
        $url = "$baseUrl/$($part.file)"
        Write-Host "Downloading $($part.file)..."
        if (Test-Path $partialPath) {
            Remove-Item -LiteralPath $partialPath -Force
        }
        try {
            Invoke-WebRequest -Uri $url -OutFile $partialPath -UseBasicParsing
        }
        catch {
            throw "Failed to download $($part.file) from GitHub Release. Check network/proxy access to github.com. Details: $($_.Exception.Message)"
        }
        Move-Item -LiteralPath $partialPath -Destination $partPath -Force
        $downloadedHash = Get-Sha256Hex -Path $partPath
        if ($downloadedHash -ne $part.sha256) {
            throw "Checksum mismatch after download for $($part.file)"
        }
    }
}

& (Join-Path $PSScriptRoot 'reconstruct_large_v2.ps1') -ManifestPath $ManifestPath -PartDirectory $PartDirectory -Force:$ForceRebuild
