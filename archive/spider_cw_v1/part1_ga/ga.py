"""Generic real-valued Genetic Algorithm (used to evolve the spider gait, see gait.py).

The GA knows nothing about spiders: it receives an `evaluate_population` function and the gene
bounds, so the same code is validated below on standard test functions before being trusted on
the (slow, noisy-looking) gait fitness.

Loop per generation:
    evaluate -> record -> check termination -> elitism -> selection -> crossover -> mutation -> repair

Run the self-check:  python -m part1_ga.ga   (from spider_cw/)
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class GAConfig:
    pop_size: int = 100
    generations: int = 300             # generations of breeding after the random generation 0
    # (gait walkers often only emerge after 60-200 generations of slow improvement, see docs/ga.txt)
    selection: str = "tournament"      # "tournament" | "roulette"
    tournament_k: int = 3
    crossover: str = "blend"           # "uniform" | "two_point" | "blend" (BLX-alpha)
    crossover_rate: float = 0.9
    blend_alpha: float = 0.5
    mutation_rate: float = 0.05        # per-gene probability
    mutation_sigma: float = 0.1        # Gaussian std as a FRACTION of each gene's range (HIGH-LOW)
    adaptive_mutation: bool = True     # sigma shrinks with progress and with stalling (see adaptive_sigma)
    elitism: int = 2
    stall_generations: int = 300       # terminate if best hasn't improved for this many gens (= off by default)
    target_fitness: float | None = None
    seed: int | None = None


# ----------------------------------------------------------------------------- initialisation

def init_population(n, low, high, rng):
    """Uniform random chromosomes inside the bounds.
    Uniform (not e.g. centred) so generation 0 covers the whole search space without bias."""
    return rng.uniform(low, high, size=(n, low.size))


# ----------------------------------------------------------------------------- selection

def tournament_select(fitness, n_select, k, rng):
    """Pick n_select parent indices; each is the fittest of k random individuals.
    Why the default: it only compares fitnesses (rank-based), so it works with negative scores and
    is unaffected by the fitness scale; k directly sets selection pressure (k=2 gentle, k>=5 greedy)."""
    contestants = rng.integers(0, fitness.size, size=(n_select, k))
    return contestants[np.arange(n_select), np.argmax(fitness[contestants], axis=1)]


def roulette_select(fitness, n_select, rng):
    """Fitness-proportionate selection.
    Probabilities must be >= 0, but our scores can be negative, so shift by the minimum: the worst
    individual gets ~0 weight, the rest are proportional to how much better than the worst they are.
    The epsilon (relative to the spread) keeps the worst selectable and avoids 0/0 when every score
    is equal (then selection is uniform). Weakness offered for comparison: pressure depends on the
    fitness scale, so one outlier can dominate early or differences can vanish late."""
    w = fitness - fitness.min()
    w = w + 1e-9 + 1e-3 * w.max()
    return rng.choice(fitness.size, size=n_select, p=w / w.sum())


# ----------------------------------------------------------------------------- crossover
# All operators take parent arrays a, b of shape (pairs, n_genes) and return two children arrays.

def uniform_crossover(a, b, rng):
    """Each gene independently comes from either parent (coin flip).
    Maximum mixing, ignores gene order; but it can only reuse existing gene values, never new ones."""
    mask = rng.random(a.shape) < 0.5
    return np.where(mask, a, b), np.where(mask, b, a)


def two_point_crossover(a, b, rng):
    """Swap the segment between two random cut points.
    Keeps blocks of neighbouring genes together (in the gait encoding, a leg's three joints are
    adjacent), so good building blocks survive; mixing is lower than uniform."""
    pairs, n = a.shape
    cuts = np.sort(rng.integers(0, n + 1, size=(pairs, 2)), axis=1)
    idx = np.arange(n)
    mask = (idx >= cuts[:, :1]) & (idx < cuts[:, 1:])
    return np.where(mask, b, a), np.where(mask, a, b)


def blend_crossover(a, b, rng, alpha, wrap, rng_span):
    """BLX-alpha: each child gene is sampled uniformly from [min - alpha*d, max + alpha*d], d=|a-b|.
    Why the default for real genes: children can land between AND slightly beyond the parents, so it
    creates new values (exploration) while the interval shrinks as the population converges
    (automatic fine-tuning). alpha=0.5 is the standard value that keeps population variance stable.
    Circular genes: parent b is first moved to the copy of itself nearest to a (e.g. phases 0.1 and
    6.2 rad are 0.18 rad apart, not 6.1), otherwise blending would jump to the opposite phase."""
    if wrap is not None and wrap.any():
        half = rng_span / 2
        b = np.where(wrap, a + np.mod(b - a + half, rng_span) - half, b)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    d = alpha * (hi - lo)
    return rng.uniform(lo - d, hi + d), rng.uniform(lo - d, hi + d)


# ----------------------------------------------------------------------------- mutation & repair

def adaptive_sigma(cfg, gen, stall):
    """Mutation step size (fraction of gene range) for this generation.
    Rule: sigma = sigma0 * (1 - 0.9*gen/G) * 0.5**(stall/stall_generations), floored at 5% of sigma0.
    Why: big steps early explore the space; as the run progresses the population sits near a good
    gait and big steps mostly destroy it, so steps shrink linearly to 10%. When the best stops
    improving we are probably on a peak that needs finer moves, so sigma halves for every
    stall_generations without improvement (reset as soon as the best improves)."""
    if not cfg.adaptive_mutation:
        return cfg.mutation_sigma
    progress = 1 - 0.9 * min(gen / max(cfg.generations, 1), 1.0)
    return cfg.mutation_sigma * max(progress * 0.5 ** (stall / cfg.stall_generations), 0.05)


def gaussian_mutation(pop, rate, sigma, rng_span, rng):
    """Each gene mutates with probability `rate` by adding N(0, (sigma*range)^2).
    Gaussian: mostly small tweaks, occasionally larger jumps. Scaling by each gene's range makes one
    sigma meaningful for genes with very different units (0.4 rad amplitudes vs 2*pi phases).
    The per-gene rate (~1/n_genes*4) changes a few genes per child, keeping most of what worked."""
    mask = rng.random(pop.shape) < rate
    return pop + mask * rng.normal(0.0, 1.0, pop.shape) * sigma * rng_span


def repair(pop, low, high, wrap):
    """Put genes back in bounds: clip ordinary genes, wrap circular ones (phase 2*pi+0.1 == 0.1).
    Clipping a phase would pile individuals up at 0 and 2*pi, which are the same gait, and would
    stop the GA from moving 'through' 0; wrapping preserves the real meaning of the gene."""
    out = np.clip(pop, low, high)
    if wrap is not None and wrap.any():
        out = np.where(wrap, low + np.mod(pop - low, high - low), out)
    return out


# ----------------------------------------------------------------------------- main loop

def _crossover(parents, cfg, rng, wrap, rng_span):
    """Pair up parents, cross each pair with probability crossover_rate, else copy the parents."""
    a, b = parents[0::2], parents[1::2]
    if cfg.crossover == "uniform":
        c1, c2 = uniform_crossover(a, b, rng)
    elif cfg.crossover == "two_point":
        c1, c2 = two_point_crossover(a, b, rng)
    elif cfg.crossover == "blend":
        c1, c2 = blend_crossover(a, b, rng, cfg.blend_alpha, wrap, rng_span)
    else:
        raise ValueError(f"unknown crossover {cfg.crossover!r}")
    do = rng.random((a.shape[0], 1)) < cfg.crossover_rate
    return np.concatenate([np.where(do, c1, a), np.where(do, c2, b)])


def run(evaluate_population, low, high, config, wrap=None, on_generation=None):
    """wrap: optional bool mask (n_genes,) of circular genes -> wrap modulo range instead of clipping.
    on_generation(record) callback called each generation (for console logging).
    Returns (best_chrom ndarray, best_score float, history list[dict])."""
    cfg = config
    low, high = np.asarray(low, float), np.asarray(high, float)
    span = high - low
    wrap = None if wrap is None else np.asarray(wrap, bool)
    if cfg.selection not in ("tournament", "roulette"):
        raise ValueError(f"unknown selection {cfg.selection!r}")
    if not 0 <= cfg.elitism < cfg.pop_size:
        raise ValueError("elitism must be in [0, pop_size)")
    rng = np.random.default_rng(cfg.seed)

    pop = init_population(cfg.pop_size, low, high, rng)
    history, best_chrom, best_score, stall = [], None, -np.inf, 0
    for gen in range(cfg.generations + 1):
        scores, components = evaluate_population(pop)
        scores = np.asarray(scores, float)
        i = int(np.argmax(scores))
        if scores[i] > best_score:
            best_chrom, best_score, stall = pop[i].copy(), float(scores[i]), 0
        elif gen > 0:
            stall += 1
        sigma = adaptive_sigma(cfg, gen, stall)
        record = {"gen": gen, "best": float(scores[i]), "mean": float(scores.mean()),
                  "worst": float(scores.min()), "sigma": float(sigma), "best_chrom": pop[i].copy(),
                  "components": {k: float(np.asarray(v)[i]) for k, v in components.items()}}
        history.append(record)
        if on_generation is not None:
            on_generation(record)

        if (gen == cfg.generations or stall >= cfg.stall_generations
                or (cfg.target_fitness is not None and best_score >= cfg.target_fitness)):
            break

        # Elitism: the top few are copied unchanged so the best solution can never be lost by
        # crossover/mutation (makes `best` monotone); kept small so they don't take over the gene pool.
        elites = pop[np.argsort(scores)[::-1][:cfg.elitism]]
        n_children = cfg.pop_size - cfg.elitism
        n_parents = n_children + n_children % 2          # crossover needs an even number
        if cfg.selection == "tournament":
            idx = tournament_select(scores, n_parents, cfg.tournament_k, rng)
        else:
            idx = roulette_select(scores, n_parents, rng)
        children = _crossover(pop[idx], cfg, rng, wrap, span)[:n_children]
        children = gaussian_mutation(children, cfg.mutation_rate, sigma, span, rng)
        pop = np.concatenate([elites, repair(children, low, high, wrap)])
    return best_chrom, best_score, history


# ----------------------------------------------------------------------------- self-check

if __name__ == "__main__":
    N = 72
    LOW = np.concatenate([np.tile([-0.6, -1.5, -2.2], 8), np.zeros(24), np.zeros(24)])
    HIGH = np.concatenate([np.tile([0.6, 0.8, 0.5], 8), np.tile([0.5, 0.4, 0.4], 8),
                           np.full(24, 2 * np.pi)])
    WRAP = np.arange(N) >= 48
    SPAN = HIGH - LOW
    # Optimum: inside bounds for normal genes; phase optimum at 0.05 rad, right next to the wrap
    # point, so the GA only finds it if wrapping (not clipping) and circular blending work.
    OPT = LOW + 0.3 * SPAN
    OPT[WRAP] = 0.05

    def normalised_error(pop):
        """Distance to OPT per gene in units of gene range; circular distance for phases."""
        d = (pop - OPT) / SPAN
        d[:, WRAP] = (d[:, WRAP] + 0.5) % 1.0 - 0.5
        return d

    def neg_sphere(pop):
        d = normalised_error(pop)
        return -np.sum(d ** 2, axis=1), {"sq_err": np.sum(d ** 2, axis=1)}

    def neg_rastrigin(pop):
        z = normalised_error(pop) * 5.12 * 2      # map +-0.5 range to Rastrigin's +-5.12
        return -(10 * N + np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z), axis=1)), {}

    def checked(f):
        """Wrap a fitness so every population the GA evaluates is checked against the bounds."""
        def g(pop):
            assert pop.shape[1] == N and np.all(pop >= LOW) and np.all(pop <= HIGH), "bounds broken"
            assert np.all(pop[:, WRAP] < HIGH[WRAP]), "wrapped gene reached 2*pi"
            return f(pop)
        return g

    print(f"{'function':<10}{'selection':<12}{'crossover':<11}{'gen0 best':>11}{'final best':>12}{'gens':>6}")
    for fname, f in [("sphere", neg_sphere), ("rastrigin", neg_rastrigin)]:
        for sel in ("tournament", "roulette"):
            for cx in ("uniform", "two_point", "blend"):
                cfg = GAConfig(selection=sel, crossover=cx, seed=1)
                chrom, score, hist = run(checked(f), LOW, HIGH, cfg, wrap=WRAP)
                g0, bests = hist[0]["best"], [h["best"] for h in hist]
                print(f"{fname:<10}{sel:<12}{cx:<11}{g0:>11.3f}{score:>12.3f}{len(hist) - 1:>6}")
                assert all(b2 >= b1 for b1, b2 in zip(bests, bests[1:])), "elitism: best decreased"
                assert score == max(bests) and np.allclose(f(chrom[None])[0], score)
                # "substantially": close at least 60% of the gap between generation 0 and the optimum
                opt = 0.0
                assert score - g0 > 0.6 * (opt - g0), f"{fname}/{sel}/{cx} barely improved"

    # Wrapping really used: the sphere optimum's phases (0.05) are found from both sides of 0.
    chrom, _, _ = run(neg_sphere, LOW, HIGH, GAConfig(seed=2), wrap=WRAP)
    assert np.all(np.abs(((chrom[WRAP] - 0.05 + np.pi) % (2 * np.pi)) - np.pi) < 0.5)

    # Reproducible: same seed -> identical history.
    h1 = run(neg_rastrigin, LOW, HIGH, GAConfig(seed=7, generations=30), wrap=WRAP)[2]
    h2 = run(neg_rastrigin, LOW, HIGH, GAConfig(seed=7, generations=30), wrap=WRAP)[2]
    assert [h["best"] for h in h1] == [h["best"] for h in h2]
    assert all(np.array_equal(a["best_chrom"], b["best_chrom"]) for a, b in zip(h1, h2))

    # Stall termination: a flat fitness never improves after gen 0 -> stops after stall_generations.
    flat = lambda pop: (np.zeros(len(pop)), {})
    h = run(flat, LOW, HIGH, GAConfig(seed=0, stall_generations=10), wrap=WRAP)[2]
    assert len(h) == 11 and h[-1]["gen"] == 10, len(h)

    # Target termination and adaptive sigma shrinking.
    h = run(neg_sphere, LOW, HIGH, GAConfig(seed=0, target_fitness=-0.5), wrap=WRAP)[2]
    assert h[-1]["best"] >= -0.5 and h[-2]["best"] < -0.5
    h = run(neg_sphere, LOW, HIGH, GAConfig(seed=0, generations=50), wrap=WRAP)[2]
    assert h[-1]["sigma"] < h[0]["sigma"]

    print("ga self-check passed")
