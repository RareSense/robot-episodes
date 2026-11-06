# Unitree H1 walking policy

Converted from unitreerobotics/unitree_rl_gym, commit `276801e`, path
`deploy/pre_train/h1/motion.pt` (Unitree's pretrained sim2sim walking policy,
LSTM(64) -> MLP(32) -> PD joint targets).

Source: https://github.com/unitreerobotics/unitree_rl_gym

SHA-256: `b6e92edc9e40a6b83ec9b0ed6ea37ce23105c4c87b2e2ccec0f949f242429ae9`

Conversion: TorchScript -> ONNX with the LSTM hidden/cell state as explicit
inputs/outputs (h_in/c_in, h_out/c_out), done locally on 2026-10-14 with
tools/export_onnx.py (kept in robotic_readme/humanoid_walk/tools). Numerical
parity against the TorchScript original was verified at export time
(parity_ort.py / parity_torch.py, same directory).

BSD 3-Clause license (Unitree Robotics) is included unchanged. This is
Unitree's published pretrained policy, not an official hardware controller;
no hardware reliability is claimed.

Nova's locomotion runner (byor/locomotion.py) binds the 50 Hz policy to an
imported robot by its own leg joint names, feeds it the observation layout of
its training configuration (deploy/deploy_mujoco/configs/h1.yaml), and applies
its PD joint targets through the robot's torque-limited motors. The base is
never pushed and no joint is ever set directly.
