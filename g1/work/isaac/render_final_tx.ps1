# Re-render every film clip with the new texture layers (looks_room / looks_props / looks_exterior).
# Same shots, cameras and frame ranges as the current film; water is replayed from the existing caches.
# Usage: powershell -File render_final_tx.ps1 [a|b]   (a and b split the clips over two Isaac processes)
param([string]$part = "all")
$work = (Resolve-Path "$PSScriptRoot\..").Path
$py = if ($env:ISAAC_PY) { $env:ISAAC_PY } else { "D:\isaacsim62\python.bat" }
$w = "$work\isaac\water\water_render.py"
$sh = "$work\isaac\shots"
$d = "$work\isaac\renders"
$kstir = ((Get-Content "$work\isaac\water\stir_kw.json" -Raw) -replace "`r?`n", " ") -replace '"', '\"'
$kcup = ((Get-Content "$work\isaac\water\cup_kw.json" -Raw) -replace "`r?`n", " ") -replace '"', '\"'
$jobs = @{
  a = @(
    @("$sh\step1h", "step1h_eye_tx", "--cam", "eye", "--no-water"),
    @("$sh\step1", "step1_film_tx", "--cam", "film", "--no-water"),
    @("$work\agents\kettle\shots\pour35", "pour35h_eye_tx", "--cam", "eye", "--frames", "1800:2701",
      "--water-cache", "$work\agents\kettle\water\pour35", "--fill", "400", "--tea-from", "2460"),
    @("$work\agents\kettle\shots\pour35", "pour35_film_tx", "--cam", "film", "--view", "1.40,2.20,1.10:1.72,2.24,1.00:65",
      "--frames", "1800:2701", "--water-cache", "$work\agents\kettle\water\pour35", "--fill", "400", "--tea-from", "2460"),
    @("$sh\cupg2h", "cupg2h_eye_tx", "--cam", "eye", "--water-cache", "$work\isaac\water\cache\cupg2_150",
      "--fill", "150", "--tea-from", "0", "--water-kw", $kcup)
  )
  b = @(
    @("$sh\tb8h", "tb8h_eye_tx", "--cam", "eye", "--no-water"),
    @("$sh\tb8h", "tb8_film_tx", "--cam", "film", "--no-water"),
    @("$sh\stir34h", "stir34h_eye_tx", "--cam", "eye", "--frames", "1350:2987", "--water-cache",
      "$work\isaac\water\cache\stir34", "--fill", "155", "--tea-from", "1350", "--water-kw", $kstir),
    @("$sh\stir34h", "stir34_close_tx", "--cam", "film", "--view", "1.72,2.40,1.12:1.72,2.22,0.94:70", "--frames", "1350:2987",
      "--water-cache", "$work\isaac\water\cache\stir34", "--fill", "155", "--tea-from", "1350", "--water-kw", $kstir),
    @("$sh\cupg2h", "cupg2_film_tx", "--cam", "film", "--water-cache", "$work\isaac\water\cache\cupg2_150",
      "--fill", "150", "--tea-from", "0", "--water-kw", $kcup)
  )
}
$sel = if ($part -eq "all") { $jobs.a + $jobs.b } else { $jobs[$part] }
foreach ($j in $sel) {
  $shot, $name = $j[0], $j[1]
  $rest = $j[2..($j.Count - 1)]
  & $py $w $shot "$d\$name.mp4" --mode rt @rest *> "$d\$name.log"
  "$name : " + ((Get-Content "$d\$name.log" | Select-String "^done|Traceback" | Select-Object -Last 1).Line)
}
