# Stretch plate-to-rack skill in N different generated rooms, Isaac Lab. One run at a time, guarded. AI-operated.
# usage: pz_rooms.ps1 -Tag r30a -Envs 30 [-Secs 260] [-Extra '--cam']
param([string]$Tag, [int]$Envs = 30, [int]$Secs = 260, [int]$Timeout = 5400, [string[]]$Extra = @())
. C:\Users\raresense\isaaclab_work\scripts\env.ps1
Set-Location "$W\scripts"
$name = "rooms_$Tag"
if (Test-Path "$W\runs\$name") { Remove-Item -Recurse -Force "$W\runs\$name" }
& "$W\tools\uvenv\Scripts\python.exe" "$W\scripts\run_guarded.py" --name $name --min-commit-gb 14 --min-phys-gb 6 --stop-commit-gb 5 --vram-cap-gb 21 --timeout-s $Timeout -- "$W\env_isaaclab\Scripts\python.exe" "$W\scripts\pz_rooms.py" --tag $Tag --num_envs $Envs --spacing 9 --secs $Secs @Extra | Select-Object -Last 2
if (Test-Path "$W\runs\$name\run.log") { Select-String -Path "$W\runs\$name\run.log" -Pattern 'ROOMS SUMMARY|rooms generated|Traceback|Error' | Select-Object -Last 5 | ForEach-Object { $_.Line.Substring(0, [Math]::Min(700, $_.Line.Length)) } }
