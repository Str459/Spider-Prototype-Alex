"""Play an evolved gait on the Webots spider: spider_cw/outputs/<run>/best_gait.csv (written by
the ga_trainer controller: a header row of motor names, then 300 rows of motor angles).

controllerArgs (optional):  [ "<run name or csv path>", "<seconds per cycle>" ]   default: run0, 2 s
Prints the distance walked after every cycle.
"""
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "..", "..", "spider_cw", "outputs")
SETTLE_SECONDS = 1.5


def load_gait(path):
    """-> (motor names, frames as lists of floats)."""
    with open(path) as f:
        rows = list(csv.reader(f))
    names, frames = rows[0], [[float(v) for v in r] for r in rows[1:] if r]
    if not frames or any(len(r) != len(names) for r in frames):
        raise ValueError(f"{path}: expected a header of motor names and rows of {len(names)} angles")
    return names, frames


def main():
    from controller import Robot

    args = sys.argv[1:]
    run = args[0] if args else "run0"
    period = float(args[1]) if len(args) > 1 else 2.0
    path = run if run.endswith(".csv") else os.path.join(RUNS, run, "best_gait.csv")
    names, frames = load_gait(path)
    print(f"gait_player: {len(frames)} frames from {os.path.normpath(path)}, {period:.1f} s per cycle")

    robot = Robot()
    step = int(robot.getBasicTimeStep())
    motors = [robot.getDevice(n) for n in names]
    gps = robot.getDevice("gps")
    gps.enable(step)

    def command(angles):
        for m, a in zip(motors, angles):
            m.setPosition(a)

    n, start, cycle = len(frames), None, -1
    while robot.step(step) != -1:
        t = robot.getTime()
        if t < SETTLE_SECONDS:
            command(frames[0])
            continue
        pos = gps.getValues()
        start = start or pos
        phase = (t - SETTLE_SECONDS) / period * n
        k, w = int(phase), phase - int(phase)
        if k // n != cycle:
            cycle = k // n
            dx, dy = pos[0] - start[0], pos[1] - start[1]
            print(f"cycle {cycle:3d}: moved x {dx:+.3f}  y {dy:+.3f}  (total {math.hypot(dx, dy):.3f} m)"
                  f"  body height {pos[2]:.3f}")
        a, b = frames[k % n], frames[(k + 1) % n]
        command([u + w * (v - u) for u, v in zip(a, b)])


if __name__ == "__main__":
    main()
