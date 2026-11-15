# Handoff: Spider GA + NN coursework (M33174), status 15 Nov 2026

Paste this into your Claude session. **First, check that the files below actually exist on your machine.** An earlier handoff described files that were never copied over.

## The task

- Spec: `AI cw/M33174_cw_item1_.pdf`. Deadline **2 Dec 2026, 13:00**. Group work, so a contribution sheet is required.
- **Part 1 (100 marks):** a genetic algorithm that evolves the 24 joint angles (8 legs × 3 joints) of the lecturer's MATLAB spider, as a full gait of about 300 poses. Marked on encoding, fitness, operators, algorithm design, visualisation, explanations and code quality.
- **Part 2 (100 marks):** a neural network that outputs the 24 joint angles. Marks 1–7 need a network written **from scratch**. PyTorch/TensorFlow only count for the 5-mark comparison. Required: loss plots, predicted vs target, a saved model, and the details (layers, activations, learning rate, loss, optimiser, data).
- The lecturer's model: `AI cw/plot_spider_pose.m` and `AI cw/forward_leg_kinematics2.m`.

## Decisions made

- **Part 1 GA in MATLAB, Part 2 NN in Python.**
- The GA is written from scratch (no toolboxes) and also drives the Webots robot's gait.
- `worlds/spider_world_1.wbt` and `controllers/my_controller_s` are the original demo and haven't been changed. The walking demo is `worlds/spider_gait.wbt`.
- `Spider_world_demo/Spider_world_demo/` is an old duplicate. Ignore it.

## What to copy

The whole `Spider_world_demo/` folder, including `AI cw/`, `spider_cw/`, `controllers/gait_player/`, `worlds/spider_gait.wbt` and `protos/`. The code uses relative paths between these folders.

## Environment

- MATLAB, plain with no toolboxes. Developed on R2026b; older versions should work, and the theme call is skipped automatically.
- Webots R2025a. If `webots.exe` isn't at `C:\Program Files\Webots\msys64\mingw64\bin\webots.exe`, set `WEBOTS_EXE`.
- Python 3. The Webots controller and the selection script use only the standard library.
- For Part 2: numpy and matplotlib aren't installed on the original machine yet, and torch is only needed for the comparison.

## Part 1: done (`spider_cw/part1_ga/`, MATLAB)

- **Encoding:** 27 real-valued genes describe a gait. There are 6 per front-to-back leg pair, mirrored left/right (coxa offset and swing, femur and tibia stance and lift), plus 3 timing genes (`phaseLR`, `phaseFB`, `duty`). `gait_decode.m` expands them into 300×24 angles.
  - During stance only the coxa moves. It turns about the vertical axis, so a planted foot keeps its height, which solves the "GA stands still" trap.
  - Stage A (`run_pose_ga.m`) also evolves one standing pose directly as a 1×24 chromosome.
- **Fitness (`gait_fitness.m`):** goal first. Distance walked per cycle is the only reward. Ground contact, ≥4 feet down, stability margin, swing clearance, foot collisions, slip and drift are requirements that only cost when broken.
- **GA (`ga_run.m` + `ga_selection/crossover/mutation.m`):** tournament (or roulette, rank), two-point (or blend, uniform, one-point), adaptive Gaussian mutation, elitism, and stall/target stopping. Every parameter is in `ga_default_config.m`.
- **Results**, default settings, seeds 0–4: fitness 11.2–12.1, distance 4.11–4.39 per cycle, ≥4 feet down in every frame, about 13 s per run.
- **Operator experiments** (`run_experiments.m`, 11 variants × 5 seeds, all 55 runs walk): tournament 11.1 vs roulette 7.6 vs rank 8.8; two-point crossover 11.6 > uniform 11.35 > blend 11.1; no crossover 9.3; adaptive mutation beats fixed (11.1 vs 10.6); no elitism 10.2. The default was switched to two-point because of these results.
- **Self-checks (`test_part1.m`):** my fast kinematics match the lecturer's `forward_leg_kinematics2.m` to 1e-14.
- **Outputs (`spider_cw/outputs/part1/seed0..4/`):** `best_gait.csv` (300×24 radians in spec order), plots, `gait.gif`, and `lecturer_view.png` (drawn by the provided `plot_spider_pose`).
- **Docs:** `spider_cw/docs/part1_*.txt` (encoding, fitness, operators with results, visualisation) and `spider_cw/README.md`.

## Webots walking demo: done (`spider_cw/robot_gait/` + `controllers/gait_player/`)

- **Gait model for the robot:** the same GA engine evolves 11 gait parameters for the Webots SpiderOctopod. Legs move in alternating groups A = L1 L3 R2 R4 and B = L2 L4 R1 R3. Planted feet move straight back at equal speed, swings are lifted and touch down at matching ground speed, and the joint angles come from exact inverse kinematics.
- **Fitness:** forward speed, under the robot's real limits (joint limits, 1.6 rad/s, 42 Nm, leg clearance, ≥4 feet, stability). Standing still scores −5.
- **Selection:** `select_in_webots.py` tests every candidate in Webots physics with self-collision on, and copies the best to `spider_cw/outputs/robot_gait/selected/`. The world plays that folder by default.
- **Measured, selected gait:** 0.30 m/s, drift 0.8 cm per m, body at 0.41 m, bounce 1.8 mm, tilt 0.6°, swings ≥15 cm, planted feet slide 0.5 cm per step, zero self-contacts. The old gait slid 3.6 cm per step at 0.11 m/s.
- **Explanation:** `spider_cw/docs/webots_gait.txt`. The previous version is backed up in `spider_cw/backups/2026-11-15_webots_gait_v1/`.
- **Limits:** flat ground and straight walking only (it can't turn). Only verified headless, plus screenshots.

## How to run

```
cd spider_cw/part1_ga
matlab -batch "test_part1"                      # self-checks, ~5 s
matlab -batch "run_gait_ga('seed',0)"           # GA + plots + GIF, ~1-2 min
matlab -batch "run_experiments"                 # operator comparison, ~15 min

cd ../robot_gait
matlab -batch "test_robot_gait; run_robot_gait_ga(0:4)"   # ~2 min
python select_in_webots.py                                 # Webots selection, ~15 s
```

To watch the walk, open `worlds/spider_gait.wbt` in Webots and press play.

## Part 2: NOT STARTED (plan)

1. **Training data from Part 1.** Use `run_gait_ga('targetSpeed', s, 'name', sprintf('speed_%.1f', s))` for several speeds, e.g. 1.0–4.0, which makes `outputs/part1/speed_*/best_gait.csv` (300×24).
2. **Inputs and outputs.** Input `[sin φ, cos φ, speed]`, where φ = frame/300 × 2π. Output the 24 angles, scaled to [-1, 1]. Hold out one speed to test generalisation.
3. **Network in numpy, from scratch.** A 3→64→64→24 MLP, tanh hidden layers, linear output, MSE loss, hand-written backpropagation.
   - Prove backpropagation with a numerical gradient check.
   - Compare SGD+momentum with Adam, sweep the learning rate, and use early stopping.
   - Save the model as `.npz`.
4. **Plots.** Loss curves, predicted vs target angles, and 3D poses. The kinematics need porting to Python (formula in `part1_ga/spider_fk.m`). Optionally, score the NN's gait with the Part 1 fitness.
5. **PyTorch comparison.** Same architecture, compared on accuracy and training time.
6. **Write-up.** `docs/nn_details.txt` covering every required item, plus a README section.

## Open items

- **`docs/part1_encoding.txt` section 3** quotes numbers from an "earlier prototype" (the 72-gene sine encoding: standing local optimum, a median of −6.2 with random phases). Those came from a teammate's handoff, not from runs on this machine. Keep them only if your group really ran that prototype; otherwise cut them.
- **Part 1 gait quirks to mention honestly:** about 25–30% slip (arc-shaped foot paths), coxa swing stuck at its 0.5 bound, and high swing lifts.
- **The group contribution sheet** isn't done.
