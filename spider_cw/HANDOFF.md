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

## Part 1: done (`spider_cw/part1_ga/`, MATLAB, simplified to 4 files)

- `run_ga.m` (GA loop, tournament/roulette, two-point/uniform, Gaussian mutation, elitism, plots), `decode_gait.m` (27 genes to 300×24 angles), `gait_fitness.m`, `foot_positions.m`.
- Run `run_ga` in `part1_ga`; results go to `outputs/part1/<selection>_<crossover>_seed<n>/`.
- Everything is explained in `docs/part1_explanation.txt`, including the operator comparison (tournament + two-point best, mean 7.8).
- The old, bigger version is in git history (commit 29a7542).

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
matlab -batch "run_ga;"                         # GA + plots + GIF, ~30 s

cd ../robot_gait
matlab -batch "test_robot_gait; run_robot_gait_ga(0:4)"   # ~2 min
python select_in_webots.py                                 # Webots selection, ~15 s
```

To watch the walk, open `worlds/spider_gait.wbt` in Webots and press play.

## Part 2: NOT STARTED (plan)

1. **Training data from Part 1.** Use the `best_gait.csv` files in `outputs/part1/*/` (300×24 each, one per GA run). The old target-speed option was removed when Part 1 was simplified.
2. **Inputs and outputs.** Input `[sin φ, cos φ, speed]`, where φ = frame/300 × 2π. Output the 24 angles, scaled to [-1, 1]. Hold out one speed to test generalisation.
3. **Network in numpy, from scratch.** A 3→64→64→24 MLP, tanh hidden layers, linear output, MSE loss, hand-written backpropagation.
   - Prove backpropagation with a numerical gradient check.
   - Compare SGD+momentum with Adam, sweep the learning rate, and use early stopping.
   - Save the model as `.npz`.
4. **Plots.** Loss curves, predicted vs target angles, and 3D poses. The kinematics need porting to Python (formula in `part1_ga/spider_fk.m`). Optionally, score the NN's gait with the Part 1 fitness.
5. **PyTorch comparison.** Same architecture, compared on accuracy and training time.
6. **Write-up.** `docs/nn_details.txt` covering every required item, plus a README section.

## Open items

- **Part 1 gait quirks to mention honestly:** some slip (arc-shaped foot paths) and leg segments can cross, see section 7 of docs/part1_explanation.txt.
- **The group contribution sheet** isn't done.
