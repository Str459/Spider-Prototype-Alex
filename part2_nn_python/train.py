"""Train the speed network: [sin phase, cos phase, speed] -> 24 motor angles.

    python train.py            all experiments, then saves the best model and results

Every experiment changes ONE setting of the base configuration, so its effect can be seen:
learning rate (Adam and SGD with momentum), hidden activation, network size, batch size.
Settings are chosen by the loss on the validation speed; the test speed is only used at the end.
Saves outputs/nn/speed_model.npz (weights + normalisation) and outputs/nn/results.json.
"""
import copy
import json
import os
import time

import numpy as np

import data
import nn

OUT = os.path.join(data.ROOT, "outputs", "nn")
BASE = {"sizes": [64, 64], "hidden": "tanh", "opt": "adam", "lr": 1e-3, "batch": 64}
EXPERIMENTS = {
    "learning rate (Adam)": [{"lr": v} for v in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)],
    "learning rate (SGD + momentum 0.9)": [{"opt": "sgd", "lr": v} for v in (1e-3, 1e-2, 3e-2, 1e-1)],
    "hidden activation": [{"hidden": v} for v in ("tanh", "relu", "sigmoid")],
    "network size": [{"sizes": v} for v in ([16, 16], [32, 32], [64, 64], [128, 128], [64], [64, 64, 64])],
    "batch size": [{"batch": v} for v in (16, 64, 256)],
}
EPOCHS, PATIENCE, SEED = 3000, 300, 0


def degrees_rmse(model, X, Y):
    """Root mean square angle error in degrees (after converting back to motor angles)."""
    return float(np.degrees(np.sqrt(np.mean((data.unscale(model.forward(X)) - data.unscale(Y)) ** 2))))


def run(cfg, d):
    model = nn.MLP([3] + cfg["sizes"] + [24], hidden=cfg["hidden"], output="tanh", seed=SEED)
    opt = nn.Adam(cfg["lr"]) if cfg["opt"] == "adam" else nn.SGD(cfg["lr"], momentum=0.9)
    t = time.time()
    hist = nn.train(model, d["X"], d["Y"], d["Xval"], d["Yval"], opt, epochs=EPOCHS,
                    batch_size=cfg["batch"], patience=PATIENCE, seed=SEED)
    return model, {"cfg": cfg, "train": hist["train"], "val": hist["val"], "best_epoch": hist["best_epoch"],
                   "seconds": time.time() - t, "val_loss": min(hist["val"]),
                   "val_deg": degrees_rmse(model, d["Xval"], d["Yval"])}


def main():
    os.makedirs(OUT, exist_ok=True)
    d = data.load()
    runs, groups, best = [], {}, None
    for name, changes in EXPERIMENTS.items():
        groups[name] = []
        for ch in changes:
            cfg = {**BASE, **ch}
            done = next((i for i, r in enumerate(runs) if r["cfg"] == cfg), None)   # don't train twice
            if done is None:
                model, r = run(cfg, d)
                runs.append(r)
                done = len(runs) - 1
                print(f"{name:36s} {str(ch):28s} val loss {r['val_loss']:.2e}  val {r['val_deg']:.2f} deg  "
                      f"epochs {len(r['train'])}  {r['seconds']:.0f} s", flush=True)
                if best is None or r["val_loss"] < runs[best[0]]["val_loss"]:
                    best = (done, copy.deepcopy(model))
            groups[name].append(done)

    i, model = best
    test_deg = degrees_rmse(model, d["Xtest"], d["Ytest"])
    # baseline: blend the two training gaits on either side of the test speed
    sp = d["speeds"]
    lo, hi = "speed_0.15", "speed_0.25"
    w = (sp[data.TEST_RUN] - sp[lo]) / (sp[hi] - sp[lo])
    blend = (1 - w) * np.loadtxt(data.gait_path(lo, "joint_table.csv"), delimiter=",") \
        + w * np.loadtxt(data.gait_path(hi, "joint_table.csv"), delimiter=",")
    target = data.unscale(d["Ytest"])
    blend_deg = float(np.degrees(np.sqrt(np.mean((blend - target) ** 2))))
    grad_err = nn.gradient_check(copy.deepcopy(model), d["X"][:20], d["Y"][:20])

    nn.save(model, os.path.join(OUT, "speed_model.npz"), mean=d["mean"], std=d["std"])
    summary = {"best_run": i, "best_cfg": runs[i]["cfg"], "test_run": data.TEST_RUN, "val_run": data.VAL_RUN,
               "test_speed": sp[data.TEST_RUN], "test_deg": test_deg, "blend_baseline_deg": blend_deg,
               "gradient_check": grad_err, "speeds": sp}
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({"summary": summary, "groups": groups, "runs": runs}, f)
    print(f"\nbest: {runs[i]['cfg']}  (val {runs[i]['val_deg']:.2f} deg)")
    print(f"test speed {sp[data.TEST_RUN]:.3f} m/s (never seen): {test_deg:.2f} deg RMSE, "
          f"blending the two nearest gaits gives {blend_deg:.2f} deg")
    print(f"gradient check (backprop vs finite differences): {grad_err:.1e}")


if __name__ == "__main__":
    main()
