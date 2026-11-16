# Handoff: Spider GA + NN coursework (M33174)

Paste this into a new session (or give it to a teammate) to continue the work.

## The task
University coursework, `Info/M33174_cw_item1_.pdf`. Deadline **2 Dec 2026, 13:00**.
- **Part 1 (100 marks):** a Genetic Algorithm that evolves a walking gait for an 8-leg × 3-joint spider: a 1×24 joint-angle pose, about 300 frames per gait. Marks for encoding, fitness, operators, algorithm design, visualisation, explanations and code quality.
- **Part 2 (100 marks):** a neural network that outputs the 24 joint angles, trained (for example) on the GA's gaits. Backprop must be written from scratch: libraries give no marks in categories 1–7, and 5 marks go to comparing against PyTorch. It needs loss plots, predicted vs target comparisons and a saved model.
- The lecturer's MATLAB model is in `Info/plot_spider_pose.m.txt` and `Info/forward_leg_kinematics2.m.txt`.

## Environment
- Windows 11, Python 3.13. Installed: numpy 2.5.4, matplotlib 3.11.2, pillow. **torch is not installed** (it's needed for the Part 2 comparison).
- Run everything from `spider_cw/` as modules, for example `python -m part1_ga.run_ga`.
- Not a git repository.
- A "Fact-Forcing Gate" hook blocks the first Write/Edit/Bash on each file and every destructive command. It asks you to state facts: who calls the file, that no duplicate exists, and the user's instruction. State them and retry the identical call.

## Decisions already made
- The old neuroevolution controller `Spider_world_demo/controllers/spider_ga_nn` was scrapped at the user's request. Its files are deleted. The empty folder is locked (Webots or OneDrive), so delete it once it's free.
- `worlds/spider_world_1.wbt` now uses `controller "my_controller_s"`.
- The GA runs offline against a Python port of the MATLAB kinematics (milliseconds per gait). Webots is only an optional demo at the end.
- Work is split across parallel agents with fixed interfaces.

## Files (`spider_cw/`)
| File | Status | Purpose |
|---|---|---|
| `spider_model.py` | done, self-check passes | MATLAB kinematics port, vectorised. `forward_kinematics`, `feet`, `plot_pose` |
| `part1_ga/gait.py` | done, 27 genes | chromosome → 300×24 frames (`decode`, `LOW`, `HIGH`, `random_population`) |
| `part1_ga/fitness.py` | done | `evaluate_population(pop, target_speed=None)` returns (scores, components); `evaluate`; `distance`; `hand_gait()` reference |
| `part1_ga/ga.py` | done, self-check passes | `GAConfig` + `run(evaluate_population, low, high, config, wrap, on_generation)`. Tournament/roulette, uniform/two-point/blend (BLX-α), adaptive Gaussian mutation, elitism, stall/target termination |
| `part1_ga/plots.py` | done, self-check passes | fitness, components, sigma, pose snapshots, joint angles, foot heights + gait diagram, GIF, `compare_runs` |
| `part1_ga/run_ga.py` | done | CLI (every `GAConfig` field is a flag, plus `--target-speed`, `--name`). Writes `outputs/part1/<name>/` (best_gait.csv 300×24 in MATLAB order, best_chrom.json, history.json, PNGs, gait.gif) |
| `part1_ga/experiments.py` | done (parallel process pool) | selection / crossover / mutation / population comparisons over several seeds → `outputs/part1/experiments/` |
| `docs/ga.txt`, `docs/fitness.txt`, `docs/encoding.txt`, `docs/visualisation.txt` | written, up to date | marker-facing explanations |

## Part 1 status: done
- **Encoding:** 27 genes. That's the left legs' pose (12) and amplitudes (12), mirrored to the right side, plus 3 wave phase genes. The two sides always alternate (LR = π fixed). See `docs/encoding.txt` for why: the 72- and 52-gene versions stood still.
- **Fitness:** goal-first. Progress counts only on statically stable steps. The stability bonus is capped (a requirement, not a prize). There's a lift term so legs keep stepping, and an idle-leg penalty (-1 per leg that never touches down). See `docs/fitness.txt` for the evidence behind each change.
- **Results:** `outputs/part1/seed0..4` all walk 1.9-3.7 per cycle on all 8 legs with at least 4 feet down. All self-checks pass.
- **Operator experiments:** in `outputs/part1/experiments/` (5 seeds × 150 generations, about 23 minutes on 16 cores). The default configuration (tournament / blend / adaptive / 0.05 / pop 100) scores 6.04 ± 1.57; only pop 200 is higher (7.12, within one std). The table and discussion are in `docs/ga.txt`.
- **Known limits:**
  - The stability margin sits at about 0.
  - With the left/right lag free, the GA got stuck on in-sync pairs (lr ≈ 0, a "rowing" gait), so it is now fixed at π. See encoding.txt step 4.
  - Before the idle-leg fix, about 1 run in 5 stood still. Other seeds could still fail.

## Next steps
1. ~~Check the walking fix.~~ Done.
2. ~~Operator experiments.~~ Done (about 23 minutes for all 65 runs).
3. **Optional:** load a `best_gait.csv` row into MATLAB `plot_spider_pose` to confirm the port matches.
4. ~~README.md~~ Done.
5. **Part 2:**
   - Data: GA gaits at several `--target-speed` values (use `fitness.distance` for the actual distance walked).
   - Input `[sin φ, cos φ, speed]` → 24 angles, scaled to [-1, 1].
   - numpy MLP 3→64→64→24 with tanh hidden layers, linear output and MSE loss.
   - Hand-written backprop with a numerical gradient check; SGD+momentum vs Adam; a learning-rate sweep; early stopping.
   - Save to `model.npz`, with `predict.py`.
   - Plots: loss curves, predicted vs target, 3D poses, the NN gait scored by the Part 1 fitness.
   - `compare_pytorch.py` (install torch).
   - `docs/nn_details.txt` covering every required item.
6. ~~Webots `gait_player`~~ Done. MATLAB-model gaits shuffle in Webots, so `run_ga --model webots` evolves for the Webots body (`part1_ga/webots_model.py`, with self-collision); `selfCollision TRUE` is on in the world. The best result, `webots_seed2`, walks about 0.13 m per cycle in Webots: slow, because the fitness has no physics. See README.
7. **Group deliverables:** the contribution sheet is the group's job.
