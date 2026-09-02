$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$model = Join-Path $root 'models\qwen3.5-2b-q4km\Qwen3.5-2B-Q4_K_M.gguf'
$bench = Join-Path $root 'runtime\llama_cpp_cuda\llama-bench.exe'
$resultDir = Join-Path $root 'results\baseline'
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

if (-not (Test-Path $model)) { throw "Model not found: $model" }
if (-not (Test-Path $bench)) { throw "llama-bench not found: $bench" }

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$meta = [ordered]@{
    timestamp = (Get-Date).ToString('o')
    model = $model
    model_sha256 = (Get-FileHash $model -Algorithm SHA256).Hash
    command = $MyInvocation.MyCommand.Path
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null)
    llama_version = (& $bench --version 2>&1 | Out-String).Trim()
}
$meta | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $resultDir "${stamp}_environment.json") -Encoding utf8

$common = @('-m', $model, '-p', '128', '-n', '32', '-t', '8', '-b', '512', '-ub', '128', '-r', '2', '-o', 'json')
foreach ($case in @(@{name='cpu'; ngl='0'}, @{name='gpu'; ngl='99'})) {
    $out = Join-Path $resultDir "${stamp}_$($case.name).txt"
    & $bench @common '-ngl' $case.ngl 2>&1 | Tee-Object -FilePath $out
    if ($LASTEXITCODE -ne 0) { throw "llama-bench failed for $($case.name): $LASTEXITCODE" }
}

Write-Output "Baseline results written to $resultDir"
