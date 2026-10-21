# CUBDL speed-of-sound comparison: one command for each step (Windows PowerShell).
#   .\run.ps1 build     build the Docker image (GPU image if an NVIDIA GPU is available)
#   .\run.ps1 test      installation test on a synthetic file (true speed of sound 1500 m/s)
#   .\run.ps1 check     beamformer check on one file of each institution in data\
#   .\run.ps1 run       comparison on all files in data\ (resumes after a stop)
#   .\run.ps1 summary   summary table and figure in results\
#   .\run.ps1 all       build, test, check, run and summary
# Options: $env:CUBDL_DATA = "D:\other\folder"   $env:CUBDL_CPU = "1" (do not use the GPU)
#   Extra arguments after "run" go to run_sos_comparison.py, e.g.  .\run.ps1 run --files OSL002 MYO001
param([string]$Cmd = "help", [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest = @())
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Here = $PSScriptRoot
$Data = if ($env:CUBDL_DATA) { $env:CUBDL_DATA } else { Join-Path $Here "data" }
$UseGpu = (-not $env:CUBDL_CPU) -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($UseGpu) { $Img = "cubdl-sos-gpu"; $Dockerfile = "Dockerfile.gpu"; $Gpu = @("--gpus", "all") }
else { $Img = "cubdl-sos"; $Dockerfile = "Dockerfile"; $Gpu = @() }

function Build { Write-Host "== build $Img ($Dockerfile)"; docker build -f $Dockerfile -t $Img .; if ($LASTEXITCODE) { throw "build failed" } }
function NeedImage { docker image inspect $Img *> $null; if ($LASTEXITCODE) { Build } }

function TestInstall {
  NeedImage; New-Item -ItemType Directory -Force "$Here\test" | Out-Null
  docker run --rm @Gpu -v "${Here}\test:/work" --entrypoint python $Img /tool/make_synthetic.py --out /work/synthetic/SYN001.hdf5 --c 1500 --angles 31 --max-angle 15
  Remove-Item "$Here\test\results\results.csv" -ErrorAction SilentlyContinue
  docker run --rm @Gpu -v "${Here}\test:/work" $Img --data /work/synthetic --out /work/results/results.csv
  Write-Host "== expected: grid_cf and gd_cf within about 5 m/s of 1500"
}

function Check {
  NeedImage; New-Item -ItemType Directory -Force "$Here\test" | Out-Null
  $files = Get-ChildItem $Data -File | Where-Object { $_.Extension -in ".hdf5", ".h5" } |
           Group-Object { $_.Name.Substring(0, 3) } | ForEach-Object { $_.Group[0].Name }
  if (-not $files) { throw "no .hdf5 / .h5 files in $Data" }
  foreach ($f in $files) { docker run --rm @Gpu -v "${Data}:/data:ro" -v "${Here}\test:/work" $Img --check "/data/$f" }
  Write-Host "== the images are in test\check_<name>.png; the best NCC of each file must be 0.6 or more"
}

function RunAll {
  NeedImage; New-Item -ItemType Directory -Force "$Here\results" | Out-Null
  docker run --rm @Gpu -v "${Data}:/data:ro" -v "${Here}\results:/work/results" $Img --data /data --out /work/results/results.csv @Rest
}

function Summary {
  NeedImage
  docker run --rm -v "${Here}\results:/work/results" --entrypoint python $Img /tool/summarize.py /work/results/results.csv
}

switch ($Cmd) {
  "build" { Build } "test" { TestInstall } "check" { Check } "run" { RunAll } "summary" { Summary }
  "all" { Build; TestInstall; Check; RunAll; Summary }
  default { Get-Content $PSCommandPath -TotalCount 9 | Select-Object -Skip 1 }
}
