# Go2 walking demonstration policy

Downloaded from Eximius-Labs/efferent, commit
`f51e1af953afe1016dae9ceeff2b5328fa51781d`, path
`examples/go2_walk/policy.onnx`.

Source: https://github.com/Eximius-Labs/efferent/tree/f51e1af953afe1016dae9ceeff2b5328fa51781d/examples/go2_walk

SHA-256: `61f5174e862659305a7c219026bc02f215034ae26f1fdfdaf18fc907c809c138`.

Upstream describes this as the rl_sar robot_lab/Isaac Lab-trained Go2 velocity
policy converted from TorchScript to ONNX. This is a community policy, not an
official Unitree controller. Upstream's manifest does not identify the original
training commit. No hardware reliability or training-data provenance is claimed.

Apache-2.0 license is included. The upstream configuration carries
Copyright (c) 2024-2025 Ziqi Fan; it is included unchanged.

Nova's simulation-only runner binds joints by name and applies PD torques at the
physics rate. It preserves original uploaded XML/assets but disables their native
actuators during controlled execution, to avoid applying two controllers at once.
Torques are limited to 23.5 Nm; desired joint positions respect source limits.
Policy runs at 50 Hz, physics at 500 Hz. The demo uses a 1.5-second standing hold,
then a body-forward 0.4 m/s command, initially facing world +Y. This does not
guarantee that actual speed matches the command, or that the full course can be
traversed. Mud remains rigid geometry. No hardware interface is included.
