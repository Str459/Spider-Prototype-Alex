# Spider gait GA (M33174 Part 1)

A genetic algorithm that evolves a 300-frame walking gait for an 8-leg x 3-joint spider.
The kinematics are a Python port of the lecturer's MATLAB `forward_leg_kinematics2` / `plot_spider_pose`.

Requires Python 3 with numpy, matplotlib and pillow. Run everything from this folder.

```bash
python -m part1_ga.run_ga --seed 0 --name seed0      # one GA run, about 1-1.5 min
python -m part1_ga.experiments                       # operator comparisons, 5 seeds each
python -m part1_ga.run_ga --help                     # every GAConfig setting is a flag
```

Self-checks: `python -m spider_model`, `python -m part1_ga.gait`, `.fitness`, `.ga`, `.plots`.

## Files
| File | What it does |
|---|---|
| `spider_model.py` | MATLAB kinematics port: `forward_kinematics`, `feet`, `plot_pose` |
| `part1_ga/gait.py` | 27-gene chromosome -> 300 x 24 joint angles (see `docs/encoding.txt`) |
| `part1_ga/fitness.py` | kinematic fitness: progress, stability, clearance, slip, coordination, smoothness, lift (see `docs/fitness.txt`) |
| `part1_ga/ga.py` | `GAConfig` + `run`: tournament/roulette, uniform/two-point/blend crossover, adaptive Gaussian mutation, elitism (see `docs/ga.txt`) |
| `part1_ga/plots.py` | all figures and the GIF (see `docs/visualisation.txt`) |
| `part1_ga/run_ga.py` | CLI for one run, saves the outputs below |
| `part1_ga/experiments.py` | runs every operator variant over several seeds in parallel |

## Outputs (`outputs/part1/<name>/`)
| File | Content |
|---|---|
| `best_gait.csv` | 300 x 24 joint angles in MATLAB order (L1..L4, R4..R1); any row works with `plot_spider_pose.m` |
| `best_chrom.json` | winning genes, config, final fitness and its components |
| `history.json` | per-generation best/mean/worst fitness, sigma, components |
| `fitness.png`, `components.png`, `sigma.png` | how the run progressed |
| `pose_snapshots.png` | best individual at generations 0, 1/3, 2/3, end |
| `joint_angles.png`, `foot_heights.png` | the final gait's joint curves and footfall diagram |
| `gait.gif` | animation of the final gait |

`outputs/part1/seed0..4` are the reference runs (all five walk); `outputs/part1/experiments/` has the operator comparisons.

## Watch it in Webots
Open `../Spider_world_demo/worlds/spider_world_1.wbt` in Webots and press play. The world runs the `gait_player` controller with `selfCollision TRUE`. It plays `outputs/part1/webots_seed2/` at 3 s per cycle and prints the distance walked after every cycle.

Gaits evolved for the lecturer's MATLAB model (`seed0..4`) mostly shuffle in place in Webots, because the Webots spider is built differently: shorter legs pointing straight out sideways, tilted hip axes and other joint limits. So there is a second body to evolve for:
```bash
python -m part1_ga.run_ga --model webots --seed 2 --name webots_seed2
```
`part1_ga/webots_model.py` holds the Webots leg geometry, motor limits and a self-collision check (leg segments as capsules, against the neighbouring legs and the body box). The chromosome, the GA and the fitness terms are unchanged.

Webots results (3 s cycles, self-collision on, about 45 s of simulated walking each):

| run | Webots distance per cycle | notes |
|---|---|---|
| webots_seed2 | 0.13 m | straight (default) |
| webots_seed1 | 0.15 m | veers left |
| webots_seed4 | 0.01 m | barely moves |
| webots_seed0 | 0.03 m | |
| webots_seed3 | 0 m | body on the floor |

- **Another run or speed:** set the robot's `controllerArgs` to e.g. `["webots_seed1", "2"]`. The player reads `best_chrom.json` to know whether the angles are Webots motor angles or MATLAB angles that need converting (see the top of `controllers/gait_player/gait_player.py`; check it with `python gait_player.py --selftest`).
- **Standing controller:** set `controller "my_controller_s"` in the world file.
