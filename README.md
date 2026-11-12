<div align="center">

# Robot Episodes

### Simulated robot manipulation in a digital twin of a real kitchen

Three robots. Household tasks. Videos, recordings, datasets and code.

<a href="https://drive.google.com/file/d/1spM1i6wIExMvC1jkMVOc9FQAzvbNt_F0/view"><img src="g1/videos/previews/g1_hero.jpg" width="880" alt="G1 tea film"></a>

</div>

---

<table>
<tr>
<td align="center" width="33%"><a href="stretch/README.md"><img src="stretch/videos/previews/stretch_card.jpg" width="100%" alt="Stretch placing a pizza in the oven"><br><b>Stretch</b></a><br>Pizza in and out of an oven<br>Isaac Sim, CPU PhysX</td>
<td align="center" width="33%"><a href="g1/README.md"><img src="g1/videos/previews/g1_card.jpg" width="100%" alt="G1 putting a tea bag into the mug"><br><b>G1</b></a><br>Making tea, and the <a href="g1/oven/README.md">full oven task</a> in 5 rooms<br>MuJoCo, rendered in Isaac Sim</td>
<td align="center" width="33%"><a href="h1_2/README.md"><img src="h1_2/videos/previews/h12_carry.jpg" width="270" alt="H1-2"><br><b>H1-2</b></a><br>Making tea with live water<br>Isaac Sim, GPU PhysX</td>
</tr>
</table>

All runs are simulation and AI-operated.

## New: G1 full oven task, 5 rooms, 5 different ovens

A Unitree G1 walks to the counter, opens the oven, puts the plate on the rack, closes the door, turns the knob, heats the food, opens the door, takes the plate out, puts it back on the counter and closes the door. Each episode is about 8 minutes of whole-body torque-controlled physics in a different room with a different oven, objects moved by contact only, and every episode passes all 20 validator checks. Main, head-camera and hand-camera views.

<a href="g1/oven/README.md"><img src="g1/oven/hero.gif" width="880" alt="G1 oven: main, head and hand views"></a>

<table>
<tr>
<td align="center"><a href="g1/oven/README.md"><img src="g1/oven/ep01_oven_v01/previews/main.gif" width="165" alt="Episode 1"></a><br>Oven v01</td>
<td align="center"><a href="g1/oven/README.md"><img src="g1/oven/ep02_oven_v13/previews/main.gif" width="165" alt="Episode 2"></a><br>Oven v13</td>
<td align="center"><a href="g1/oven/README.md"><img src="g1/oven/ep03_oven_v06/previews/main.gif" width="165" alt="Episode 3"></a><br>Oven v06</td>
<td align="center"><a href="g1/oven/README.md"><img src="g1/oven/ep04_oven_v15/previews/main.gif" width="165" alt="Episode 4"></a><br>Oven v15</td>
<td align="center"><a href="g1/oven/README.md"><img src="g1/oven/ep05_oven_v02/previews/main.gif" width="165" alt="Episode 5"></a><br>Oven v02</td>
</tr>
</table>

[Episodes, recordings, validator results and full videos](g1/oven/README.md)

| Robot | Folder | Videos | Data |
|---|---|---|---|
| Stretch 3 | [stretch/](stretch/README.md) | [Videos](stretch/videos/README.md) | Full recording and LeRobot dataset; [30 generated rooms](stretch/rooms/README.md), 30 recorded episodes |
| Unitree G1 | [g1/](g1/README.md) | [Videos](g1/videos/README.md) | Per-step motion data, MuJoCo recordings and LeRobot dataset; [G1 oven](g1/oven/README.md): 5 full 8-minute episodes in 5 rooms with main, head and hand videos |
| Unitree H1-2 | [h1_2/](h1_2/README.md) | [Videos](h1_2/videos/README.md) | [Saved runs](h1_2/data/README.md), recordings and water particles |

Each robot folder holds its own code, data, model, videos index and examples.
