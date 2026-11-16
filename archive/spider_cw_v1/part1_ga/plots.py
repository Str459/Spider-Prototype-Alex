"""Plots and animations for the spider-gait GA (Part 1).

Every function saves one figure (PNG, or GIF for the animation) to `path` and returns `path`.
`history` is the list of per-generation dicts written by ga.py:
    {"gen", "best", "mean", "worst", "sigma", "best_chrom" (gait.N_GENES,), "components" {name: value}}
`frames` is a decoded gait, shape (300, 24) - see gait.decode().

Run the self-test from spider_cw/:  python -m part1_ga.plots
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from spider_model import LEG_LABELS, feet, plot_pose
from part1_ga.gait import decode

STANCE_TOL = 0.1          # a foot within this height of the lowest foot is "on the ground"
JOINT_NAMES = ["coxa yaw", "femur pitch", "tibia pitch"]


def _save(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _col(history, key):
    return np.array([h[key] for h in history], dtype=float)


def plot_fitness(history, path):
    """Best / mean / worst fitness per generation.
    Lets the marker judge convergence: best should rise then plateau; the mean closing in on
    the best shows the population converging; a persistent gap shows diversity is kept."""
    gens = _col(history, "gen")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gens, _col(history, "best"), label="best", lw=2)
    ax.plot(gens, _col(history, "mean"), label="mean", lw=1.5)
    ax.plot(gens, _col(history, "worst"), label="worst", lw=1, alpha=0.7)
    ax.set(xlabel="Generation", ylabel="Fitness", title="Fitness progression")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)


def plot_components(history, path):
    """Each fitness term of the best individual per generation, one small panel each.
    Shows WHICH part of the objective the GA is improving (e.g. distance up, foot slip down)
    and whether one term is being traded against another."""
    gens = _col(history, "gen")
    names = list(history[0]["components"])
    ncols = min(3, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        ax.plot(gens, [h["components"][name] for h in history], lw=1.8)
        ax.set(title=name, xlabel="Generation", ylabel="Value")
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.suptitle("Fitness components of the best individual")
    fig.tight_layout()
    return _save(fig, path)


def plot_mutation_sigma(history, path):
    """Mutation step size per generation - evidence that the adaptive mutation is working
    (sigma should shrink as the search converges, or jump back up when progress stalls)."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(_col(history, "gen"), _col(history, "sigma"), lw=2, color="tab:purple")
    ax.set(xlabel="Generation", ylabel="Mutation sigma", title="Adaptive mutation step size")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_pose_snapshots(history, path):
    """Frame-0 pose of the best individual at generation 0, ~1/3, ~2/3 and the end.
    Lets the marker see the body posture evolving from a random tangle to a plausible stance."""
    n = len(history)
    picks = sorted({0, n // 3, 2 * n // 3, n - 1})
    fig = plt.figure(figsize=(4.5 * len(picks), 4.5))
    for i, k in enumerate(picks):
        ax = fig.add_subplot(1, len(picks), i + 1, projection="3d")
        h = history[k]
        plot_pose(ax, decode(h["best_chrom"])[0], f"Gen {h['gen']}  fitness {h['best']:.3g}")
    fig.suptitle("Best individual's pose (frame 0) across generations")
    return _save(fig, path)


def plot_joint_angles(frames, path):
    """3 joint angles of each leg over the gait cycle, one panel per leg.
    Shows the gait is smooth and periodic, and lets the marker compare phase between legs."""
    frames = np.asarray(frames)
    t = np.arange(len(frames))
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
    for leg, ax in enumerate(axes.flat):
        for j, name in enumerate(JOINT_NAMES):
            ax.plot(t, np.rad2deg(frames[:, 3 * leg + j]), label=name)
        ax.set_title(LEG_LABELS[leg])
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Frame")
    for ax in axes[:, 0]:
        ax.set_ylabel("Angle (deg)")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Joint angles over the gait cycle")
    fig.tight_layout()
    return _save(fig, path)


def plot_foot_heights(frames, path):
    """Top: foot height per leg over time. Bottom: footfall (gait) diagram - a filled bar
    wherever that foot is in stance. A good gait shows alternating groups of legs on the
    ground (e.g. tetrapod / alternating tetrapod) with several feet always down for support."""
    z = feet(np.asarray(frames))[..., 2]                      # (300, 8)
    stance = z <= z.min(axis=1, keepdims=True) + STANCE_TOL
    t = np.arange(len(z))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    for leg in range(z.shape[1]):
        ax1.plot(t, z[:, leg], label=LEG_LABELS[leg])
    ax1.set(ylabel="Foot height z", title="Foot height per leg")
    ax1.grid(alpha=0.3)
    ax1.legend(ncol=8, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.04))

    for leg in range(z.shape[1]):
        # contiguous runs of stance frames -> (start, length) bars
        edges = np.diff(np.concatenate([[0], stance[:, leg].astype(int), [0]]))
        runs = [(s, e - s) for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
        ax2.broken_barh(runs, (leg - 0.4, 0.8), color="k")
    ax2.set(yticks=range(z.shape[1]), yticklabels=LEG_LABELS, xlabel="Frame",
            title=f"Gait diagram (black = stance, foot within {STANCE_TOL} of lowest foot)")
    ax2.set_xlim(0, len(z))
    ax2.invert_yaxis()
    fig.tight_layout()
    return _save(fig, path)


def animate_gait(frames, path, step=3, fps=20):
    """GIF of the evolved gait (every `step`-th frame) drawn with the MATLAB-style plot_pose."""
    frames = np.asarray(frames)
    idx = range(0, len(frames), step)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    anim = FuncAnimation(fig, lambda i: plot_pose(ax, frames[i], f"Frame {i}"), frames=idx)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def compare_runs(results, path):
    """Mean best-fitness curve (+/- 1 std shaded) over several seeds per configuration.
    Used for the operator experiments: a curve that is higher AND has a narrow band means that
    configuration is better and reliably so, not just lucky on one seed."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, runs in results.items():
        curves = [_col(h, "best") for h in runs]
        n = max(len(c) for c in curves)
        # early-stopped runs keep their final value for the remaining generations
        padded = np.array([np.pad(c, (0, n - len(c)), mode="edge") for c in curves])
        mean, std = padded.mean(axis=0), padded.std(axis=0)
        gens = np.arange(n)
        ax.plot(gens, mean, lw=2, label=f"{label} (n={len(runs)})")
        ax.fill_between(gens, mean - std, mean + std, alpha=0.2)
    ax.set(xlabel="Generation", ylabel="Best fitness", title="Configuration comparison (mean +/- 1 std over seeds)")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)


if __name__ == "__main__":
    from part1_ga.gait import random_population

    rng = np.random.default_rng(0)
    out = os.path.join("outputs", "_plots_selftest")

    def fake_history(n_gen, seed):
        r = np.random.default_rng(seed)
        pop = random_population(n_gen, r)
        best = np.maximum.accumulate(np.log1p(np.arange(n_gen)) + r.normal(0, 0.2, n_gen))
        return [{"gen": g, "best": best[g], "mean": best[g] - 1, "worst": best[g] - 3,
                 "sigma": 0.3 * 0.95 ** g, "best_chrom": pop[g],
                 "components": {"distance": best[g], "slip": 1 / (1 + g), "stability": g / n_gen,
                                "height": 0.5 + 0.1 * np.sin(g)}}
                for g in range(n_gen)]

    hist = fake_history(40, 1)
    frames = decode(hist[-1]["best_chrom"])
    files = [
        plot_fitness(hist, os.path.join(out, "fitness.png")),
        plot_components(hist, os.path.join(out, "components.png")),
        plot_mutation_sigma(hist, os.path.join(out, "sigma.png")),
        plot_pose_snapshots(hist, os.path.join(out, "snapshots.png")),
        plot_joint_angles(frames, os.path.join(out, "joints.png")),
        plot_foot_heights(frames, os.path.join(out, "feet.png")),
        animate_gait(frames, os.path.join(out, "gait.gif"), step=50),
        compare_runs({"A": [fake_history(40, s) for s in range(3)],
                      "B": [fake_history(30 + 5 * s, 10 + s) for s in range(3)]},
                     os.path.join(out, "compare.png")),
    ]
    for f in files:
        assert os.path.getsize(f) > 0, f
    print("plots self-check passed:", out)
