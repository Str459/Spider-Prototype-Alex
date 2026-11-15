# Spider kinematics with a GA and a neural network (M33174)

Part 1 (genetic algorithm) is written in **MATLAB**. Part 2 (neural network) will be in **Python** and uses the gaits Part 1 exports.

## Folder layout

```
spider_cw/
  part1_ga/        MATLAB code for Part 1 (run everything from this folder)
  docs/            explanations for the marker, one .txt per component
  outputs/part1/   results: one folder per run, plus experiments/
```

The lecturer's files stay where they were supplied, in `../AI cw/`. The code adds that folder to the path only to cross-check against `forward_leg_kinematics2.m` and to draw with `plot_spider_pose.m`.

## Requirements

Plain MATLAB, no toolboxes. Developed on R2026b. The figure theme call is skipped automatically on older versions.

## How to run (MATLAB, current folder `spider_cw/part1_ga`)

| Command | What it does | Time |
|---|---|---|
| `test_part1` | Self-checks: kinematics vs the lecturer's code, stability maths, decoder, fitness sanity, GA on a test function | ~5 s |
| `run_pose_ga` | Stage A: evolve one standing pose (a 1×24 chromosome) | ~5 s |
| `run_gait_ga` | Stage B: evolve a walking gait (seed 0) and save every plot and the GIF | ~15 s GA + ~1 min plots |
| `run_gait_ga('seed', 3)` | Same with another seed | |
| `run_gait_ga('selection','roulette','name','roulette_s0')` | Any GA setting by name | |
| `run_gait_ga('w_slip', 3)` | Any fitness weight, with the prefix `w_` | |
| `run_gait_ga('targetSpeed', 2.0, 'name', 'speed_2.0')` | Evolve a gait that walks exactly 2.0 per cycle (Part 2 training data) | |
| `run_experiments` | Compare selection / crossover / mutation / elitism / population size over 5 seeds | ~15 min |

From a terminal: `matlab -batch "run_gait_ga('seed',0)"` (run it inside `part1_ga`).

All GA settings are in `ga_default_config.m`, and all fitness settings and weights are in `fitness_config.m`.

## Outputs of a gait run (`outputs/part1/<name>/`)

| File | Content |
|---|---|
| `best_gait.csv` | **300 × 24** joint angles in radians, one pose per row, in the spec's order `[L1a L1b L1c … R1a R1b R1c]`. Any row can be passed to `plot_spider_pose`. |
| `best_gait_stance.csv` | 300 × 8, 1 = foot planted (same leg order) |
| `best_chrom.csv` | The 27 genes of the winning chromosome |
| `history.csv` | Per generation: best / mean / median / worst fitness, diversity, sigma, every fitness term of the best |
| `summary.txt` | Settings, stopping reason, every fitness term, every gene |
| `fitness.png`, `components.png` | Is the GA working? |
| `pose_snapshots.png`, `gait.gif`, `foot_heights.png`, `joint_angles.png`, `stability.png` | Is the gait sensible? |
| `lecturer_view.png` | Frame 1 drawn by the provided `plot_spider_pose.m` |
| `result.mat` | Everything above as MATLAB variables |

`docs/part1_visualisation.txt` explains how to read each plot.

## Code map (Part 1)

| File | Role |
|---|---|
| `spider_geometry.m` | Constants of the lecturer's model (segment lengths, leg positions, leg order) |
| `spider_fk.m` | Vectorised forward kinematics, identical to `forward_leg_kinematics2.m` (checked to 1e-14) and thousands of times faster |
| `gait_genes.m`, `gait_decode.m` | The 27-gene chromosome and how it becomes 300 × 24 angles |
| `fitness_config.m`, `gait_fitness.m`, `support_margin.m` | Fitness function and its stability maths |
| `ga_default_config.m`, `ga_run.m` | The GA loop: initialise, evaluate, elitism, breed, adapt, stop |
| `ga_selection.m`, `ga_crossover.m`, `ga_mutation.m` | Operators: tournament / roulette / rank, blend / uniform / one- and two-point, adaptive Gaussian |
| `run_gait_ga.m`, `run_pose_ga.m`, `run_experiments.m` | Entry points |
| `plot_ga_history.m`, `plot_gait_analysis.m`, `animate_gait.m`, `draw_spider.m`, `new_figure.m` | Visualisation |
| `pose_fitness.m` | Stage A fitness |
| `test_part1.m` | Self-checks |

## Webots walking gait (`robot_gait/`)

The same GA engine evolves 11 gait parameters for the Webots SpiderOctopod robot: alternating support groups A = L1 L3 R2 R4 and B = L2 L4 R1 R3, straight-line stance strokes, lifted swings, and exact inverse kinematics. Every candidate is then tested in Webots physics with self-collision on, and the best measured one becomes the gait the world plays.

```
cd spider_cw/robot_gait
matlab -batch "test_robot_gait; run_robot_gait_ga(0:4)"   # evolve, ~2 min
python select_in_webots.py                                 # test in Webots, pick the best, ~15 s
```

To watch it, open `worlds/spider_gait.wbt` in Webots. `docs/webots_gait.txt` explains why the old gait crawled, what changed, and the measured results.

## Explanations (`docs/`)

- `part1_encoding.txt`: why the chromosome looks the way it does, and what failed before
- `part1_fitness.txt`: every fitness term, the weights, and the tuning evidence
- `part1_operators.txt`: GA loop, selection, crossover, mutation, parameters, and experiment results
- `part1_visualisation.txt`: how to read each plot
