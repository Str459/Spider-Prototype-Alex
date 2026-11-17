# Spider gait: GA in MATLAB, neural network in Python (M33174)

- **Part 1:** a genetic algorithm in MATLAB evolves a walking gait (300 poses × 24 joint angles) for the Webots spider robot. Every result walks in Webots with self-collision on.
- **Part 2:** a neural network, written from scratch in numpy, learns to make a gait for any speed. It is trained on gaits the GA made, and its gaits walk in Webots at the speed they were asked for.

`build.md` explains every decision in detail.

## Folders
| Folder | What is in it |
|---|---|
| `part1_ga_matlab/` | `run_ga.m` (GA, settings, plots), `decode_gait.m` (genes → gait), `gait_fitness.m`, `leg_ik.m` / `leg_fk.m` (robot kinematics), `make_speed_family.m` |
| `part2_nn_python/` | `nn.py` (the network, from scratch), `data.py`, `train.py`, `predict.py`, `plots.py`, `compare_pytorch.py` |
| `webots/` | `worlds/spider.wbt`, the robot (`protos/`), `controllers/gait_player/` (plays and measures a gait), `measure.py` |
| `outputs/` | `part1/<run>/` GA results, `nn/` the trained model, plots and network gaits |

## Requirements
- MATLAB, no toolboxes (developed on R2026b).
- Webots R2025a. If it isn't in `C:\Program Files\Webots`, set `WEBOTS_EXE`.
- Python 3 with `pip install -r requirements.txt`. torch is only needed for the comparison.

## Run it, in this order
```
cd part1_ga_matlab
matlab -batch "run_ga"                        # fastest gait -> outputs/part1/tournament_twopoint_seed1/  (~30 s)
matlab -batch "make_speed_family"             # 6 gaits at 0.10 ... 0.35 m/s for the network  (~3 min)

cd ..
python webots/measure.py part1/tournament_twopoint_seed1 part1/speed_0.10 part1/speed_0.15 part1/speed_0.20 part1/speed_0.25 part1/speed_0.30 part1/speed_0.35

cd part2_nn_python
python data.py && python nn.py                # self-checks
python train.py                               # experiments + final model -> outputs/nn/  (~8 min)
python plots.py
python compare_pytorch.py
python predict.py 0.177 0.15 0.30 --webots    # network gaits, walked in Webots
```
To watch a gait, open `webots/worlds/spider.wbt` in Webots and press play. It plays `part1/tournament_twopoint_seed1`. For a network gait, set SpiderOctopod → controllerArgs to e.g. `nn/speed_0.300`.

## Results
- **GA (fastest gait):** walks 0.34 m/s in Webots, body level (0.4° tilt once walking), 0 self-contacts. One run takes about 25 s.
- **Network:** on a speed it never saw, its angles are 0.12° RMSE from the GA's gait. Its gaits walk in Webots within 0.2–2.3% of the speed asked for.
- **Against PyTorch:** same start and batches give the same result (gradients agree to 3e-17).
