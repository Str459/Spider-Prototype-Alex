"""Plots for Part 2 (run train.py first). Saved in outputs/nn/.

    python plots.py

loss_curves.png         training and validation loss per epoch of the chosen model
experiments.png         validation loss of every experiment (one setting changed at a time)
learning_rate.png       validation loss against learning rate, Adam and SGD
predicted_vs_target.png the never-seen test speed: network angles against the GA's angles
poses.png               3D poses of the test speed, GA (target) and network (predicted)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import data
import nn

OUT = os.path.join(data.ROOT, "outputs", "nn")
LEGS = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
HIP_X = [0.4875, 0.1625, -0.1625, -0.4875, -0.4875, -0.1625, 0.1625, 0.4875]
SIDE = [1, 1, 1, 1, -1, -1, -1, -1]


def leg_points(q):
    """Forward kinematics of the Webots robot (same as part1_ga_matlab/leg_fk.m).
    q: 24 motor angles -> 8 x 4 x 3 points (hip, thigh joint, knee, foot) in metres."""
    P = np.zeros((8, 4, 3))
    c = 0.5 * np.cos(np.pi / 4)
    for leg in range(8):
        s = SIDE[leg]
        q1, q2, q3 = q[3 * leg:3 * leg + 3]
        a2, a3 = np.pi / 4 + s * q2, -np.pi / 4 + s * (q2 + q3)
        u = [0, c, c + 0.6 * np.cos(a2), c + 0.6 * np.cos(a2) + 0.7 * np.cos(a3)]
        z = [0, c, c + 0.6 * np.sin(a2), c + 0.6 * np.sin(a2) + 0.7 * np.sin(a3)]
        axis = np.array([0, -s, 1]) / np.sqrt(2)
        for k in range(4):
            v = np.array([0, s * u[k], z[k]])
            v = v * np.cos(q1) + np.cross(axis, v) * np.sin(q1) + axis * (axis @ v) * (1 - np.cos(q1))
            P[leg, k] = v + [HIP_X[leg], 0.2 * s, 0]
    return P


def main():
    with open(os.path.join(OUT, "results.json")) as f:
        res = json.load(f)
    runs, groups, summ = res["runs"], res["groups"], res["summary"]
    best = runs[summ["best_run"]]

    plt.figure(figsize=(7, 4))
    plt.semilogy(best["train"], label="training (5 speeds)")
    plt.semilogy(best["val"], label=f"validation ({data.VAL_RUN})")
    plt.axvline(best["best_epoch"], color="grey", ls="--", label="kept (lowest validation)")
    plt.xlabel("epoch"); plt.ylabel("MSE loss (scaled angles)"); plt.legend(); plt.grid(alpha=0.3)
    plt.title(f"Training of the chosen network {best['cfg']['sizes']}, {best['cfg']['hidden']}, "
              f"{best['cfg']['opt']} lr {best['cfg']['lr']}")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "loss_curves.png"), dpi=120); plt.close()

    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 4))
    base = runs[groups["learning rate (Adam)"][2]]["cfg"]
    for ax, (name, ids) in zip(axes, groups.items()):
        labels = [", ".join(f"{k}={v}" for k, v in runs[i]["cfg"].items() if v != base[k]) or "base" for i in ids]
        ax.bar(range(len(ids)), [runs[i]["val_loss"] for i in ids], color="tab:blue")
        ax.set_yscale("log"); ax.set_xticks(range(len(ids))); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(name); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("best validation loss")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "experiments.png"), dpi=120); plt.close()

    plt.figure(figsize=(6, 4))
    for name, mark in (("learning rate (Adam)", "o-"), ("learning rate (SGD + momentum 0.9)", "s-")):
        ids = groups[name]
        plt.loglog([runs[i]["cfg"]["lr"] for i in ids], [runs[i]["val_loss"] for i in ids], mark, label=name)
    plt.xlabel("learning rate"); plt.ylabel("best validation loss"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "learning_rate.png"), dpi=120); plt.close()

    d = data.load()
    model, _ = nn.load(os.path.join(OUT, "speed_model.npz"))
    pred, target = data.unscale(model.forward(d["Xtest"])), data.unscale(d["Ytest"])
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for ax, j in zip(axes.flat[:6], [0, 1, 2, 12, 13, 14]):
        ax.plot(np.degrees(target[:, j]), "k", label="GA (target)")
        ax.plot(np.degrees(pred[:, j]), "r--", label="network")
        ax.set_title(f"{LEGS[j // 3]} {['hip', 'thigh', 'knee'][j % 3]}"); ax.set_xlabel("frame"); ax.set_ylabel("deg")
    axes.flat[0].legend()
    ax = axes.flat[6]
    ax.plot(np.degrees(target.ravel()), np.degrees(pred.ravel()), ".", ms=1)
    ax.plot([-120, 80], [-120, 80], "k-", lw=0.5); ax.set_xlabel("target (deg)"); ax.set_ylabel("predicted (deg)")
    ax.set_title("all 24 angles, all frames")
    ax = axes.flat[7]
    ax.hist(np.degrees(np.abs(pred - target)).ravel(), bins=50); ax.set_xlabel("|error| (deg)"); ax.set_title("error")
    fig.suptitle(f"Test speed {summ['test_speed']:.3f} m/s (never seen in training): "
                 f"RMSE {summ['test_deg']:.2f} deg")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "predicted_vs_target.png"), dpi=120); plt.close()

    fig = plt.figure(figsize=(16, 4.5))
    for k, f in enumerate([0, 75, 150, 225]):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        for q, col, lab in ((target[f], "k", "GA"), (pred[f], "r", "network")):
            P = leg_points(q)
            for leg in range(8):
                ax.plot(P[leg, :, 0], P[leg, :, 1], P[leg, :, 2], col, lw=1.5, label=lab if leg == 0 else None)
        ax.plot([0.65, 0.65, -0.65, -0.65, 0.65], [0.2, -0.2, -0.2, 0.2, 0.2], 0, "grey")
        ax.set_title(f"frame {f}"); ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_zlim(-0.6, 0.8)
        ax.view_init(25, -60)
        if k == 0:
            ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "poses.png"), dpi=120); plt.close()
    print("saved plots in", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
