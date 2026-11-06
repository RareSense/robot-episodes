"""Summarise a water stats file (water_layer stats.json / *_water_stats.json): pour onset, flow, final volumes.
usage: venv python pour_summary.py <stats.json> [every_n_frames]"""
import json
import sys

S = json.load(open(sys.argv[1]))
step = int(sys.argv[2]) if len(sys.argv) > 2 else 30
onset = next((s for s in S if s["mug_ml"] > 1.0), None)
peak = max(((S[i]["mug_ml"] - S[i - 6]["mug_ml"]) * 10, S[i]["k"]) for i in range(6, len(S)))
print(f"frames {len(S)}; pour onset frame {onset['k'] if onset else '-'} (t {onset['t'] if onset else '-'} s, "
      f"tilt {onset['tilt_deg'] if onset else '-'} deg); peak flow into the mug {peak[0]:.0f} ml/s at frame {peak[1]}")
e = S[-1]
print(f"end (frame {e['k']}): mug {e['mug_ml']} ml (level {e.get('mug_level_mm', 0)} mm), kettle {e['kettle_ml']} ml, "
      f"elsewhere {e['other_ml']} ml (below counter {e.get('below_counter_ml', 0)} ml)")
print(" frame    t   tilt   mug  kettle  other  level")
for s in S[::step]:
    print(f"{s['k']:6d} {s['t']:5.2f} {s['tilt_deg']:6.1f} {s['mug_ml']:5.0f} {s['kettle_ml']:7.0f} {s['other_ml']:6.0f} "
          f"{s.get('mug_level_mm', 0):6.1f}")
