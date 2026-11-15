# Spider GA and NN coursework (M33174)

Part 1 (genetic algorithm) is in MATLAB. Part 2 (neural network) will be in Python.

## Part 1

Plain MATLAB, no toolboxes needed. The lecturer's files are used from `../AI cw/`.

In MATLAB, go to `spider_cw/part1_ga` and run:

```
run_ga                              % default: tournament selection, two-point crossover, seed 1
run_ga('roulette', 'uniform', 2)    % other selection / crossover / seed
```

It prints the progress, then saves everything in `outputs/part1/<selection>_<crossover>_seed<n>/`:
`best_gait.csv` (the 300 x 24 gait), `fitness.png`, `feet.png`, `pose.png` and `gait.gif`.

Files:
- `run_ga.m`: settings, the GA loop, selection, crossover, mutation and plots
- `decode_gait.m`: chromosome to 300 poses
- `gait_fitness.m`: fitness function
- `foot_positions.m`: forward kinematics

The explanation of every design choice and the operator comparison is in `docs/part1_explanation.txt`.

## Webots demo (extra)

`robot_gait/` evolves a gait for the Webots spider robot, and `worlds/spider_gait.wbt` plays it. Open the world in Webots and press play. `docs/webots_gait.txt` has the details.
