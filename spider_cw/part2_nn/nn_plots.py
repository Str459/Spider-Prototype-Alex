"""Figures for option A (speed network). Run after train_speed:

    cd spider_cw
    python -m part2_nn.nn_plots          # -> outputs/nn_speed/*.png

    loss_curves.png        training loss of the final model + validation loss of its 3 folds
    lr_sweep.png           leave-one-speed-out error vs learning rate, Adam and SGD + momentum
    experiments.png        error of every configuration: choosing score and held-out test speed
    predicted_vs_target.png  joint angles over the cycle on the held-out speed: GA target, network,
                           and the no-learning baseline (blend of the neighbouring gaits)
    poses.png              3D poses at 4 points of the cycle: GA target vs network
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from part1_ga.body import plot_pose
from part1_ga.gait import MOTOR_NAMES
from part2_nn.predict import gait_for_speed
from part2_nn.speed_data import HELD_OUT, RUNS, load_frames, neighbours
from part2_nn.train_speed import OUT


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def loss_curves(res):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    final = res["final"]
    ax.semilogy(final["train"], lw=2, label=f"final model: training loss ({final['epochs']} epochs, all 5 speeds)")
    for f in final["loso"]["folds"]:
        ax.semilogy(f["val"], lw=1, alpha=0.8, label=f"fold: validation loss on hidden {f['val_run']}")
        ax.axvline(f["best_epoch"], color="grey", lw=0.5, ls=":")
    ax.set(xlabel="Epoch", ylabel="MSE (scaled angles)", title="Training progress (dotted: early-stopping epochs)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _save(fig, "loss_curves.png")


def lr_sweep(res):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, style in (("learning rate (Adam)", "o-"), ("learning rate (SGD + momentum 0.9)", "s--")):
        runs = res["experiments"][name]
        ax.semilogx([r["config"]["lr"] for r in runs], [r["cv_rmse_deg"] for r in runs], style, label=name)
    ax.axhline(res["baseline_rmse_deg"], color="k", ls=":", label="baseline (blend of neighbours)")
    ax.set(xlabel="Learning rate", ylabel="Leave-one-speed-out RMSE (deg)", title="Learning-rate sweep")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _save(fig, "lr_sweep.png")


def experiments(res):
    names = list(res["experiments"])
    fig, axes = plt.subplots(len(names), 1, figsize=(8, 2.3 * len(names)))
    for ax, name in zip(axes, names):
        runs = res["experiments"][name]
        x = np.arange(len(runs))
        ax.bar(x - 0.2, [r["cv_rmse_deg"] for r in runs], 0.4, label="leave-one-speed-out (chooses)")
        ax.bar(x + 0.2, [r["test_rmse_deg"] for r in runs], 0.4, label="held-out test speed (reported)")
        ax.axhline(res["baseline_rmse_deg"], color="k", ls=":", lw=1)
        ax.set_xticks(x, [r["label"] for r in runs], fontsize=8)
        ax.set(title=name, ylabel="RMSE (deg)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, "experiments.png")


def predicted_vs_target():
    target = load_frames(HELD_OUT)
    pred = gait_for_speed(RUNS[HELD_OUT])
    lo, hi = neighbours(HELD_OUT)
    w = (RUNS[HELD_OUT] - RUNS[lo]) / (RUNS[hi] - RUNS[lo])
    blend = (1 - w) * load_frames(lo) + w * load_frames(hi)
    joints = ["L1_hip_joint", "L1_thigh_joint", "L1_knee_joint", "L4_hip_joint", "L4_thigh_joint", "L4_knee_joint"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for ax, j in zip(axes.flat, joints):
        k = MOTOR_NAMES.index(j)
        ax.plot(np.degrees(target[:, k]), "k", lw=2, label="GA gait (target)")
        ax.plot(np.degrees(pred[:, k]), "tab:red", lw=1.5, label="network")
        ax.plot(np.degrees(blend[:, k]), "tab:blue", lw=1, ls="--", label="baseline blend")
        ax.set(title=j.replace("_joint", ""), ylabel="deg")
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Frame")
    axes[0, 0].legend(fontsize=8)
    err = np.degrees(np.sqrt(np.mean((pred - target) ** 2)))
    fig.suptitle(f"Held-out speed {RUNS[HELD_OUT]:.3f} m/cycle (never seen in training): network RMSE {err:.2f} deg")
    fig.tight_layout()
    return _save(fig, "predicted_vs_target.png")


def poses():
    target, pred = load_frames(HELD_OUT), gait_for_speed(RUNS[HELD_OUT])
    fig = plt.figure(figsize=(16, 8))
    for i, k in enumerate((0, 75, 150, 225)):
        plot_pose(fig.add_subplot(2, 4, i + 1, projection="3d"), target[k], f"GA target, frame {k}")
        plot_pose(fig.add_subplot(2, 4, i + 5, projection="3d"), pred[k], f"network, frame {k}")
    fig.suptitle(f"Held-out speed {RUNS[HELD_OUT]:.3f} m/cycle: GA gait (top) vs network output (bottom)")
    return _save(fig, "poses.png")


if __name__ == "__main__":
    with open(os.path.join(OUT, "experiments.json")) as f:
        res = json.load(f)
    for p in (loss_curves(res), lr_sweep(res), experiments(res), predicted_vs_target(), poses()):
        print(p)
