param(
    [string]$PromptFile = "data\calibration_prompts.txt",
    [string]$OutputRoot = "results\calibration_20260902",
    [int]$Threads = 8
)

$ErrorActionPreference = "Stop"
$model = "models\qwen3.5-2b-q4km\Qwen3.5-2B-Q4_K_M.gguf"
$probe = "vendor\build-probe\ffn_probe.exe"
$prompts = Get-Content -LiteralPath $PromptFile
$index = 0
foreach ($prompt in $prompts) {
    if ([string]::IsNullOrWhiteSpace($prompt)) { continue }
    $id = "p{0:D2}" -f $index
    $out = Join-Path $OutputRoot $id
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $promptPath = Join-Path $out "prompt.txt"
    Set-Content -LiteralPath $promptPath -Value $prompt -Encoding UTF8
    $log = Join-Path $out "run.log"
    $stdout = Join-Path $out "stdout.log"
    $stderr = Join-Path $out "stderr.log"
    $escapedPrompt = $prompt.Replace('"', '\\"')
    $argLine = "--model `"$model`" --out `"$out`" --prompt `"$escapedPrompt`" --threads $Threads --capture-ffn"
    $proc = Start-Process -FilePath $probe -ArgumentList $argLine -Wait -PassThru -NoNewWindow -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Get-Content -LiteralPath $stdout, $stderr | Set-Content -LiteralPath $log -Encoding utf8
    $code = $proc.ExitCode
    if ($code -ne 0) { throw "probe failed for $id with exit code $code" }
    $index++
}
Write-Output "completed prompts=$index output=$OutputRoot"
