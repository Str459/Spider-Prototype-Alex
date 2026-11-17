"""Use the trained network: make a gait for any speed, and optionally walk it in Webots.

    python predict.py 0.177                  -> outputs/nn/speed_0.177/ (a folder gait_player can play)
    python predict.py 0.177 0.25 --webots    also walks each one in Webots and prints what happened

To watch one in Webots: open webots/worlds/spider.wbt and set controllerArgs to "nn/speed_0.177".
"""
import os
import shutil
import sys

import numpy as np

import data
import nn

MODEL = os.path.join(data.ROOT, "outputs", "nn", "speed_model.npz")
# All gaits in the family share the timing of the fastest gait (same period and duty), so the
# feet that are down in each frame are the same for every speed.
REF = "tournament_twopoint_seed1"


def gait_for_speed(speed):
    """speed (m/s) -> 300 x 24 motor angles."""
    model, extra = nn.load(MODEL)
    return data.unscale(model.forward(data.inputs(speed, float(extra["mean"]), float(extra["std"]))))


def write_gait(speed):
    angles = gait_for_speed(speed)
    name = f"nn/speed_{speed:.3f}"
    out = os.path.join(data.ROOT, "outputs", name)
    os.makedirs(out, exist_ok=True)
    np.savetxt(os.path.join(out, "joint_table.csv"), angles, delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(out, "stand.csv"), angles[:1], delimiter=",", fmt="%.6f")  # frame 0: all 8 feet down
    shutil.copy(data.gait_path(REF, "stance.csv"), os.path.join(out, "stance.csv"))
    meta = dict(np.genfromtxt(data.gait_path(REF, "meta.csv"), delimiter=",", names=True, dtype=None,
                              encoding=None).tolist())
    meta["speed"] = speed
    meta["stride"] = speed * meta["duty"] * meta["period"]
    with open(os.path.join(out, "meta.csv"), "w") as f:
        f.write("key,value\n" + "".join(f"{k},{v}\n" for k, v in meta.items()))
    return name


if __name__ == "__main__":
    args = sys.argv[1:]
    webots = "--webots" in args
    for v in [float(a) for a in args if a != "--webots"] or [0.177]:
        name = write_gait(v)
        print(f"wrote outputs/{name}")
        if webots:
            sys.path.insert(0, os.path.join(data.ROOT, "webots"))
            from measure import measure
            r = measure(name)
            print(f"  asked for {v:.3f} m/s, walked {r['speed_mps']:.3f} m/s in Webots "
                  f"({100 * (r['speed_mps'] - v) / v:+.1f}%), tilt {r['max_tilt_deg']:.1f} deg, "
                  f"self-contacts {r['self_contacts']}")
