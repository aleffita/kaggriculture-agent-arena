# Recompila o dxgi_hook.dll usando o MSVC do Visual Studio
$ErrorActionPreference = "Stop"

$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vsWhere -latest -property installationPath
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvarsall.bat"

if (-not (Test-Path $vcvars)) {
    Write-Error "vcvarsall.bat não foi encontrado em: $vcvars"
}

$repoRoot = (Get-Item $PSScriptRoot).Parent.FullName
$shimsDir = Join-Path $repoRoot "shims"
$pkgDir = Join-Path $repoRoot "src\litert_explore"

Write-Host "Compilando dxgi_hook.cpp via MSVC x64..." -ForegroundColor Cyan

cmd.exe /c "`"$vcvars`" x64 && cd /d `"$shimsDir`" && cl /nologo /O2 /LD /EHsc dxgi_hook.cpp /link /OUT:dxgi_hook.dll"

if ($LASTEXITCODE -eq 0) {
    Copy-Item (Join-Path $shimsDir "dxgi_hook.dll") (Join-Path $pkgDir "dxgi_hook.dll") -Force
    Write-Host "dxgi_hook.dll compilado e sincronizado com sucesso!" -ForegroundColor Green
} else {
    Write-Error "Falha na compilação do dxgi_hook.dll."
}
