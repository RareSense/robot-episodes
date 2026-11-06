"""Explicit task props added to the existing cafe; no changes during execution."""

import mujoco
import numpy as np

SHIFT_X = 0.65
CLUTTER = ["cleanup_red", "cleanup_blue", "cleanup_yellow"]
CLUTTER_XY = [(6.0, 6.78), (6.0, 6.68), (6.12, 6.74)]
BIN_BOUNDS = [[6.34 + SHIFT_X, 6.59, 0.67], [6.56 + SHIFT_X, 6.81, 0.84]]
DROP_POINT = [6.45 + SHIFT_X, 6.7, 1.05]


def add_cleanup_fixture(spec, seed=0, variation_m=0.0):
    rng = np.random.default_rng(seed)
    positions = np.array(CLUTTER_XY) + rng.uniform(-variation_m, variation_m, (3, 2))
    box = mujoco.mjtGeom.mjGEOM_BOX
    table = spec.worldbody.add_body(name="cleanup_table", pos=[SHIFT_X, 0, 0])
    table.add_geom(
        name="cleanup_table_top",
        type=box,
        pos=[6.0, 6.58, 0.78],
        size=[0.27, 0.23, 0.02],
        rgba=[0.25, 0.42, 0.48, 1],
        friction=[0.7, 0.005, 0.001],
    )
    for x in [5.75, 6.25]:
        for y in [6.4, 6.76]:
            table.add_geom(
                type=box,
                pos=[x, y, 0.38],
                size=[0.025, 0.025, 0.38],
                rgba=[0.24, 0.26, 0.28, 1],
            )
    tray = spec.worldbody.add_body(name="cleanup_bin", pos=[SHIFT_X, 0, 0])
    tray.add_geom(
        name="cleanup_bin_floor",
        type=box,
        pos=[6.45, 6.7, 0.66],
        size=[0.12, 0.12, 0.02],
        rgba=[0.12, 0.48, 0.28, 1],
    )
    for x in [6.32, 6.58]:
        tray.add_geom(
            type=box,
            pos=[x, 6.7, 0.76],
            size=[0.01, 0.14, 0.1],
            rgba=[0.12, 0.48, 0.28, 1],
        )
    for y in [6.57, 6.83]:
        tray.add_geom(
            type=box,
            pos=[6.45, y, 0.76],
            size=[0.12, 0.01, 0.1],
            rgba=[0.12, 0.48, 0.28, 1],
        )
    for x in [6.35, 6.55]:
        for y in [6.6, 6.8]:
            tray.add_geom(
                type=box,
                pos=[x, y, 0.32],
                size=[0.015, 0.015, 0.32],
                rgba=[0.24, 0.26, 0.28, 1],
            )
    colors = [[0.85, 0.25, 0.12, 1], [0.12, 0.38, 0.82, 1], [0.95, 0.7, 0.12, 1]]
    for name, (x, y), color in zip(CLUTTER, positions, colors, strict=True):
        obj = spec.worldbody.add_body(name=name, pos=[x + SHIFT_X, y, 0.84])
        obj.add_freejoint(name=name + "_free")
        obj.add_geom(
            name=name + "_shape",
            type=box,
            size=[0.018, 0.018, 0.04],
            mass=0.08,
            rgba=color,
            friction=[0.8, 0.005, 0.001],
        )
    return {
        "added_task_fixtures": ["fixed sorting table", "open collection bin"],
        "added_clutter": [
            {
                "name": n,
                "size_m": [0.036, 0.036, 0.08],
                "mass_kg": 0.08,
                "position": [x + SHIFT_X, y, 0.84],
            }
            for n, (x, y) in zip(CLUTTER, positions, strict=True)
        ],
        "bin_bounds": BIN_BOUNDS,
        "note": (
            "Task staging only; source cafe remains unchanged. "
            "No runtime object attachment or pose writes."
        ),
    }
