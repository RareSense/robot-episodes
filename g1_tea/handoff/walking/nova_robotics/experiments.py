"""A capability catalog, not claims of task success or trained locomotion."""

EXPERIMENTS = [
    {
        "id": "passive",
        "label": "Gravity and contact",
        "requires": "none",
        "detail": "Passive settling; no motor commands.",
    },
    {
        "id": "actuator_step",
        "label": "Motor step response",
        "requires": "actuator",
        "detail": "A bounded step on one selected native actuator.",
    },
    {
        "id": "actuator_sweep",
        "label": "Motor command sweep",
        "requires": "actuator",
        "detail": "A small sinusoidal native command, not a joint-limit certification.",
    },
    {
        "id": "push",
        "label": "Push response",
        "requires": "free",
        "detail": "A brief horizontal external force on a selected free body.",
    },
    {
        "id": "drop",
        "label": "Drop and settling",
        "requires": "free",
        "detail": "Start a free body above its source pose and observe impact.",
    },
    {
        "id": "slide",
        "label": "Sliding response",
        "requires": "free",
        "detail": "Initial horizontal velocity; observe contact and sliding.",
    },
    {
        "id": "hinge",
        "label": "Door / hinge response",
        "requires": "hinge",
        "detail": "Apply torque to a selected environment hinge.",
    },
    {
        "id": "slider",
        "label": "Drawer / slider response",
        "requires": "slide",
        "detail": "Apply force to a selected environment slider.",
    },
    {
        "id": "load",
        "label": "External load response",
        "requires": "free",
        "detail": "Apply downward force; this does not change mass or inertia.",
    },
    {
        "id": "release",
        "label": "Motor release response",
        "requires": "actuator",
        "detail": "Apply a bounded command, then disable actuation.",
    },
]


def validate_program(value):
    """Data-only open-loop command sequence. Never Python, pickle or executable code."""
    import math

    if not isinstance(value, dict) or set(value) != {"format", "frames"}:
        raise ValueError("Controller JSON needs only format and frames")
    if value["format"] != "nova.actuator-sequence/v1":
        raise ValueError("Supported controller format: nova.actuator-sequence/v1")
    frames = value["frames"]
    if not isinstance(frames, list) or not 1 <= len(frames) <= 500:
        raise ValueError("Controller needs 1–500 command frames")
    last = -1
    for frame in frames:
        if not isinstance(frame, dict) or set(frame) != {"time_s", "controls"}:
            raise ValueError("Each frame needs time_s and controls")
        t, controls = frame["time_s"], frame["controls"]
        if type(t) not in (int, float) or not math.isfinite(t) or not last < t <= 30 or t < 0:
            raise ValueError("Frame times must increase from zero through at most 30 seconds")
        if not isinstance(controls, dict) or not 1 <= len(controls) <= 64:
            raise ValueError("Each frame needs 1–64 named actuator values")
        for name, v in controls.items():
            if not isinstance(name, str) or not name or len(name) > 128:
                raise ValueError("Invalid actuator identifier")
            if type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 10000:
                raise ValueError("Actuator values must be finite and bounded")
        last = t
    if frames[0]["time_s"] != 0:
        raise ValueError("First controller frame must start at time_s 0")
    return value
