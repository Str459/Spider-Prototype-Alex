"""Compare GA components: each variant is run with several seeds so differences aren't luck.

    cd spider_cw
    python -m part1_ga.experiments                     # all experiments, 5 seeds, 150 generations
    python -m part1_ga.experiments --seeds 3 --generations 80     # quicker

Experiments (all other settings stay at the GAConfig defaults):
    selection   tournament vs roulette
    crossover   uniform vs two_point vs blend
    mutation    adaptive sigma on vs off, and three mutation rates
    population  50 vs 100 vs 200
Outputs in outputs/part1/experiments/: one comparison plot per experiment (mean +-1 std of best
fitness over generations) and summary.txt with the final best fitness per variant.
"""
import argparse
import dataclasses
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from part1_ga import plots
from part1_ga.ga import GAConfig
from part1_ga.run_ga import ROOT, evolve

EXPERIMENTS = {
    "selection": {s: {"selection": s} for s in ("tournament", "roulette")},
    "crossover": {c: {"crossover": c} for c in ("uniform", "two_point", "blend")},
    "mutation": {"adaptive": {"adaptive_mutation": True}, "fixed": {"adaptive_mutation": False},
                 "rate 0.01": {"mutation_rate": 0.01}, "rate 0.05": {"mutation_rate": 0.05},
                 "rate 0.2": {"mutation_rate": 0.2}},
    "population": {f"pop {n}": {"pop_size": n} for n in (50, 100, 200)},
}


def _history(config):
    """One GA run (top-level so worker processes can pickle it)."""
    return evolve(config, log=False)[2]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--generations", type=int, default=150)
    p.add_argument("--only", choices=list(EXPERIMENTS), help="run just one experiment")
    args = p.parse_args(argv)

    out = os.path.join(ROOT, "outputs", "part1", "experiments")
    os.makedirs(out, exist_ok=True)
    # Every (experiment, variant, seed) run is independent, so they all go to a process pool.
    # stall_generations = generations so every run has the same length to compare.
    jobs = {(exp, label, seed): dataclasses.replace(GAConfig(), generations=args.generations,
                                                    stall_generations=args.generations, seed=seed, **overrides)
            for exp, variants in EXPERIMENTS.items() if not args.only or exp == args.only
            for label, overrides in variants.items() for seed in range(args.seeds)}
    print(f"Running {len(jobs)} GA runs on {os.cpu_count()} cores...")
    with ProcessPoolExecutor() as pool:
        histories = dict(zip(jobs, pool.map(_history, jobs.values())))

    lines = []
    for exp, variants in EXPERIMENTS.items():
        if args.only and exp != args.only:
            continue
        results = {label: [histories[exp, label, s] for s in range(args.seeds)] for label in variants}
        for label, runs in results.items():
            finals = np.array([h[-1]["best"] for h in runs])
            line = f"{exp:10s} {label:12s} final best {finals.mean():9.3f} +- {finals.std():7.3f}  (n={args.seeds})"
            print(line)
            lines.append(line)
        plots.compare_runs(results, os.path.join(out, f"{exp}.png"))

    with open(os.path.join(out, "summary.txt"), "a") as f:
        f.write(f"# seeds={args.seeds} generations={args.generations}\n" + "\n".join(lines) + "\n")
    print(f"Saved plots and summary to {out}")


if __name__ == "__main__":
    main()
