"""Generate a gait from the trained speed network (option A) at any speed.

    cd spider_cw
    python -m part2_nn.predict 0.357              # -> outputs/nn_speed_0.357/best_gait.csv
    python -m part2_nn.predict 0.12 0.24 0.357 0.45

Writes the same best_gait.csv format as the GA (header of motor names, 300 rows of motor angles),
so the Webots gait_player and the trainer's measure= option walk it unchanged:
    controllerArgs ["measure=nn_speed_0.357" "cycles=4"]   or   ["nn_speed_0.357"] for gait_player
"""
import json
import os
import sys

import numpy as np

from part1_ga.gait import FRAMES, MOTOR_HI, MOTOR_LO, MOTOR_NAMES
from part2_nn import nn
from part2_nn.speed_data import ROOT, inputs, unscale
from part2_nn.train_speed import MODEL_PATH


def gait_for_speed(speed, model_path=MODEL_PATH):
    """Speed in metres per cycle -> (300, 24) motor angles from the saved network."""
    model, extra = nn.load(model_path)
    x = inputs(speed, float(extra["speed_mean"]), float(extra["speed_std"]))
    return unscale(model.forward(x))


def write_gait(speed, model_path=MODEL_PATH):
    frames = gait_for_speed(speed, model_path)
    assert frames.shape == (FRAMES, 24) and (frames >= MOTOR_LO).all() and (frames <= MOTOR_HI).all()
    out = os.path.join(ROOT, "outputs", f"nn_speed_{speed:.3f}")
    os.makedirs(out, exist_ok=True)
    np.savetxt(os.path.join(out, "best_gait.csv"), frames, delimiter=",", fmt="%.6f",
               header=",".join(MOTOR_NAMES), comments="")
    with open(os.path.join(out, "nn_info.json"), "w") as f:
        json.dump({"source": "part2_nn speed network", "model": os.path.relpath(model_path, ROOT),
                   "requested_speed_m_per_cycle": speed}, f, indent=1)
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["0.357"]:
        print(write_gait(float(arg)))
