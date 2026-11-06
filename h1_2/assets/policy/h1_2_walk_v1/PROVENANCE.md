# H1-2 walking policy

Unitree's pretrained sim2sim walking policy from `unitreerobotics/unitree_rl_gym`, commit `276801e46c5d433564f24658bac64f254b7d2d4b`,
path `deploy/pre_train/h1_2/motion.pt`, sha256 `ea3003268d5d2e4b5004eb26d0969fbfdedced265ceeec7bf0b335309259cc80`.
LSTM(64) then MLP(32), 47 observations, 12 leg actions, 50 Hz.

Converted from TorchScript to ONNX with the LSTM hidden and cell state as explicit inputs and outputs. The weight tensors were compared with `motion.pt` and are identical.
`policy.onnx` sha256: `77956801313e8d45a8b18c45beeafbe93690a7e1091efebeb613e721b515dfc4`.

`gait.json` holds the deployment values from Unitree's `h1_2.yaml` plus this project's standing and hold gains.

BSD-3-Clause, see `LICENSE`. This is Unitree's published policy, not an official hardware controller, and no hardware reliability is claimed.
