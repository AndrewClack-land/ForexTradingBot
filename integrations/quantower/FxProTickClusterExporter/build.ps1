[CmdletBinding()]
param(
    [string]$QuantowerBin = "E:\Quantower\TradingPlatform\v1.146.14\bin",
    [string]$DotnetPath = "dotnet",
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$projectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$project = Join-Path $projectDirectory "FxProTickClusterExporter.csproj"
$nugetConfig = Join-Path $projectDirectory "NuGet.Config"

if (-not (Test-Path -LiteralPath (Join-Path $QuantowerBin "TradingPlatform.BusinessLayer.dll"))) {
    throw "TradingPlatform.BusinessLayer.dll was not found under QuantowerBin."
}

& $DotnetPath restore $project `
    --configfile $nugetConfig `
    "-p:QuantowerBin=$QuantowerBin"
if ($LASTEXITCODE -ne 0) {
    throw "dotnet restore failed with exit code $LASTEXITCODE."
}

& $DotnetPath build $project `
    --configuration $Configuration `
    --no-restore `
    "-p:QuantowerBin=$QuantowerBin"
if ($LASTEXITCODE -ne 0) {
    throw "dotnet build failed with exit code $LASTEXITCODE."
}

$output = Join-Path $projectDirectory "bin\$Configuration\net10.0-windows"
Write-Host "Built Quantower exporter: $output"
