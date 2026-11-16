"""Run the GA once and save everything needed to judge the result.

    cd spider_cw
    python -m part1_ga.run_ga                                  # defaults from GAConfig
    python -m part1_ga.run_ga --generations 300 --selection roulette --crossover uniform --seed 1
    python -m part1_ga.run_ga --target-speed 2.0 --name speed_2.0   # gait for a set speed (Part 2 data)

Every GAConfig field is a command-line flag (--pop-size, --mutation-rate, ...).
Outputs go to outputs/part1/<name>/:
    best_gait.csv      300 x 24 joint angles in MATLAB order - feed any row to plot_spider_pose.m
    best_chrom.json    the winning genes (see gait.py) + config + final fitness components
    history.json       per-generation best/mean/worst/sigma/components (re-plottable)
    *.png, gait.gif    fitness progression, components, poses, joint angles, footfall diagram, animation
"""
import argparse
import dataclasses
import json
import os

import numpy as np

from part1_ga import fitness, gait, plots
from part1_ga.ga import GAConfig, run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WRAP = gait.WRAP                                 # phase genes are angles: wrap instead of clipping


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(GAConfig):
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            typ = lambda s: s.lower() in ("1", "true", "yes")
        elif f.default is None:                       # optional fields: seed is an int, target_fitness a float
            typ = int if f.name == "seed" else float
        else:
            typ = type(f.default)
        p.add_argument(flag, type=typ, default=f.default)
    p.add_argument("--target-speed", type=float, default=None, help="evolve for this distance per cycle")
    p.add_argument("--model", choices=["matlab", "webots"], default="matlab",
                   help="body to evolve for: the lecturer's MATLAB model or the Webots spider (part1_ga/webots_model.py)")
    p.add_argument("--name", default=None, help="output folder name (default: built from the settings)")
    p.add_argument("--quiet", action="store_true", help="no per-generation console log")
    return p.parse_args(argv)


def evolve(config, target_speed=None, log=True):
    """Run one GA. Returns (best_chrom, best_score, history)."""
    def log_gen(r):
        comps = "  ".join(f"{k} {v:+.3f}" for k, v in r["components"].items())
        print(f"gen {r['gen']:4d}  best {r['best']:9.3f}  mean {r['mean']:9.3f}  sigma {r['sigma']:.3f}  | {comps}")

    return run(lambda pop: fitness.evaluate_population(pop, target_speed=target_speed),
               gait.LOW, gait.HIGH, config, wrap=WRAP, on_generation=log_gen if log else None)


def history_to_json(history):
    return [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in history]


def save_run(out, config, target_speed, best, score, history, model="matlab"):
    os.makedirs(out, exist_ok=True)
    frames = gait.decode(best)
    np.savetxt(os.path.join(out, "best_gait.csv"), frames, delimiter=",", fmt="%.6f")
    _, comps = fitness.evaluate(best, target_speed=target_speed)
    with open(os.path.join(out, "best_chrom.json"), "w") as f:
        json.dump({"model": model, "fitness": float(score), "components": {k: float(v) for k, v in comps.items()},
                   "target_speed": target_speed, "config": dataclasses.asdict(config),
                   "chromosome": best.tolist()}, f, indent=1)
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump(history_to_json(history), f)

    plots.plot_fitness(history, os.path.join(out, "fitness.png"))
    plots.plot_components(history, os.path.join(out, "components.png"))
    plots.plot_mutation_sigma(history, os.path.join(out, "sigma.png"))
    plots.plot_pose_snapshots(history, os.path.join(out, "pose_snapshots.png"))
    plots.plot_joint_angles(frames, os.path.join(out, "joint_angles.png"))
    plots.plot_foot_heights(frames, os.path.join(out, "foot_heights.png"))
    plots.animate_gait(frames, os.path.join(out, "gait.gif"))


def main(argv=None):
    args = parse_args(argv)
    config = GAConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(GAConfig)})
    if args.model == "webots":
        from part1_ga import webots_model
        webots_model.activate()
    name = args.name or f"{'webots_' if args.model == 'webots' else ''}{config.selection}_{config.crossover}_seed{config.seed}"
    out = os.path.join(ROOT, "outputs", "part1", name)

    best, score, history = evolve(config, args.target_speed, log=not args.quiet)
    print(f"\nFinished after {len(history)} generations. Best fitness {score:.3f}")
    save_run(out, config, args.target_speed, best, score, history, args.model)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
