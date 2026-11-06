param(
  [Parameter(Mandatory=$true)][string]$IsaacPython,
  [Parameter(Mandatory=$true)][string]$SceneDir,
  [string]$Tag = 'repro01',
  [string]$Pylib = ''
)
# The final run's launch, with the environment that was set outside the script. run_live_episode.py sets the rest.
# Runs for many hours. Exit code 0 means every strict gate passed, which the stale kettle rule rules out: read the JSON, see DISCLOSURES.md.
$ErrorActionPreference = 'Stop'
if ($Tag -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Tag must be a single filename component.' }
$repo = Split-Path -Parent $PSScriptRoot
$out = Join-Path $repo 'out\logs'
New-Item -ItemType Directory -Path $out -Force | Out-Null
$log = Join-Path $out ($Tag + '.log')
if (Test-Path -LiteralPath $log) { throw 'This tag already has a log. Use a new tag.' }
$env:H12_SCENE_DIR = $SceneDir
if ($Pylib) { $env:H12_PYLIB = $Pylib }
$env:P_SCOOP = 'real'
$ErrorActionPreference = 'Continue'
& $IsaacPython (Join-Path $repo 'code\run_live_episode.py') $Tag *>&1 | Out-File -LiteralPath $log -Encoding utf8
Write-Output ("exit code " + $LASTEXITCODE + ", log " + $log)
