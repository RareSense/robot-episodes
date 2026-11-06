"""Read a LeRobot v3.0 dataset folder directly with pandas. Needs pandas and pyarrow.
usage: python examples/load_lerobot.py data/lerobot"""
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/lerobot")
info = json.loads((root / "meta" / "info.json").read_text())
df = pd.concat(pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet")))
print(f"{info['total_episodes']} episodes, {len(df)} frames at {info['fps']} fps, robot {info['robot_type']}")
print("columns:", list(df.columns))
print("state size:", len(df["observation.state"].iloc[0]), "action size:", len(df["action"].iloc[0]))
