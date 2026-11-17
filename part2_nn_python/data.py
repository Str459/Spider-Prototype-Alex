"""Training data for the network: the speed family of gaits made by the MATLAB GA.

Each gait is outputs/part1/<run>/joint_table.csv (300 poses x 24 motor angles, one cycle).
The speed of each gait is the speed MEASURED in Webots (outputs/part1/<run>/webots_result.json,
written by webots/measure.py), not the GA's target: the network should learn what the gait
really does on the robot.

Input  (3 numbers): sin(phase), cos(phase), speed
    phase = 2 pi k / 300 for frame k. Two numbers instead of one so frame 299 sits next to frame 0.
    speed is normalised with the mean and std of the training speeds.
Output (24 numbers): the motor angles scaled to about [-0.9, 0.9] with the motor limits,
    y = (angle - middle) / (1.1 * half range). The tanh output layer can then only give angles
    inside the limits (+10% margin, clipped when converting back).

Split by WHOLE gaits, not random frames: frames of one gait are almost copies of each other,
so a random split would only test memory. Test = one speed never seen in training,
validation = another speed (used for early stopping and for choosing settings).
"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUNS = ["speed_0.10", "speed_0.15", "speed_0.20", "speed_0.25", "speed_0.30", "speed_0.35",
        "tournament_twopoint_seed1"]
TEST_RUN = "speed_0.20"
VAL_RUN = "speed_0.30"
FRAMES = 300

LOW = np.tile([-1.5708, -1.5708, -2.0], 8)      # motor limits: hip, thigh, knee for each leg
HIGH = np.tile([1.5708, 1.5708, 1.0], 8)
MID = (HIGH + LOW) / 2
HALF = 1.1 * (HIGH - LOW) / 2


def gait_path(run, file):
    return os.path.join(ROOT, "outputs", "part1", run, file)


def measured_speed(run):
    with open(gait_path(run, "webots_result.json")) as f:
        return json.load(f)["speed_mps"]


def scale(angles):
    return (angles - MID) / HALF


def unscale(y):
    return np.clip(MID + HALF * y, LOW, HIGH)


def inputs(speed, mean, std):
    phase = 2 * np.pi * np.arange(FRAMES) / FRAMES
    return np.column_stack([np.sin(phase), np.cos(phase), np.full(FRAMES, (speed - mean) / std)])


def load():
    """-> dict with X/Y for train, val and test, and the normalisation constants."""
    speeds = {r: measured_speed(r) for r in RUNS}
    train_runs = [r for r in RUNS if r not in (TEST_RUN, VAL_RUN)]
    mean = np.mean([speeds[r] for r in train_runs])
    std = np.std([speeds[r] for r in train_runs])

    def block(runs):
        X = np.vstack([inputs(speeds[r], mean, std) for r in runs])
        Y = np.vstack([scale(np.loadtxt(gait_path(r, "joint_table.csv"), delimiter=",")) for r in runs])
        return X, Y

    d = {"speeds": speeds, "mean": mean, "std": std, "train_runs": train_runs}
    d["X"], d["Y"] = block(train_runs)
    d["Xval"], d["Yval"] = block([VAL_RUN])
    d["Xtest"], d["Ytest"] = block([TEST_RUN])
    return d


if __name__ == "__main__":
    d = load()
    print("measured speeds (m/s):", {r: round(v, 3) for r, v in d["speeds"].items()})
    print("train", d["X"].shape, "val", d["Xval"].shape, "test", d["Xtest"].shape)
    assert d["X"].shape == (5 * FRAMES, 3) and d["Y"].shape == (5 * FRAMES, 24)
    assert np.abs(d["Y"]).max() < 1 / 1.1 + 1e-9                      # every target inside tanh's range
    a = np.loadtxt(gait_path(TEST_RUN, "joint_table.csv"), delimiter=",")
    assert np.allclose(unscale(scale(a)), a)                            # scaling round trip
    print("data self-check passed")
