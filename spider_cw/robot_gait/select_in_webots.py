"""Choose the final gait by testing every GA candidate in Webots physics.

    python select_in_webots.py                 # every outputs/robot_gait/seed*/
    python select_in_webots.py seed0 seed3     # only these
    python select_in_webots.py --cycles 8

For each candidate it starts Webots headless on worlds/spider_gait.wbt (self-
collision on), lets controllers/gait_player stand up and walk, and reads the
measurements the controller writes. The MATLAB GA already chose each
candidate with a kinematic fitness; this is the second, physical fitness
stage, which decides which candidate becomes outputs/robot_gait/selected/.

Physics score (higher is better), measured over all cycles after the first:
    forward speed (m/s)
  - 0.3  x sideways drift per metre walked
  - 0.01 x heading change (deg)
  - 0.01 x worst roll/pitch (deg)
  - 0.002 x worst body bounce (mm)
  - 0.2  x planted-foot sliding per metre walked
  - 0.5  if the robot touched itself at all (self-collision)
  - 0.5  if any swing lifted a foot less than 3 cm
  - 1.0  if the body sank below 0.25 m (fell / crawling)
Standing still scores at most 0 minus penalties, so it can never be chosen
over a candidate that walks.

Needs Webots (set WEBOTS_EXE if it is not in the default place).
"""

import csv
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(PROJECT, "spider_cw", "outputs", "robot_gait")
WORLD = os.path.join(PROJECT, "worlds", "spider_gait.wbt")
WEBOTS = os.environ.get("WEBOTS_EXE", r"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe")
FILES = ["joint_table.csv", "stance.csv", "stand.csv", "meta.csv",
         "startup.csv", "startup_stance.csv"]


def physics_score(r):
    return (r["speed_mps"]
            - 0.3 * abs(r["sideways_per_m"])
            - 0.01 * abs(r["heading_change_deg"])
            - 0.01 * r["max_tilt_deg"]
            - 0.002 * r["bounce_mm"]
            - 0.2 * r["slide_per_m_walked"]
            - 0.5 * (r["self_contacts"] > 0)
            - 0.5 * (r["min_swing_clearance_cm"] < 3.0)
            - 1.0 * (r["height_mean_m"] < 0.25))


def run(folder, cycles, port):
    result = os.path.join(folder, "webots_result.json")
    if os.path.exists(result):
        os.remove(result)
    env = dict(os.environ, GAIT_DIR=folder, GAIT_QUIT_AFTER=str(cycles), GAIT_RESULT=result)
    cmd = [WEBOTS, f"--port={port}", "--batch", "--mode=fast", "--minimize", "--no-rendering",
           "--stdout", "--stderr", WORLD]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    log = [l for l in proc.stdout.splitlines() + proc.stderr.splitlines() if "[gait_player]" in l]
    with open(os.path.join(folder, "webots_log.txt"), "w") as fh:
        fh.write("\n".join(log) + "\n")
    if not os.path.exists(result):
        raise RuntimeError(f"no result from Webots for {folder}; see webots_log.txt")
    with open(result) as fh:
        return json.load(fh)


def main():
    args = sys.argv[1:]
    cycles = 5           # steady cycles measured after the 4 start-up cycles
    if "--cycles" in args:
        i = args.index("--cycles")
        cycles = int(args[i + 1])
        del args[i:i + 2]
    names = args or sorted(d for d in os.listdir(OUT)
                           if d.startswith("seed") and os.path.exists(os.path.join(OUT, d, "joint_table.csv")))
    rows = []
    for k, name in enumerate(names):
        folder = os.path.join(OUT, name)
        print(f"testing {name} in Webots ...", flush=True)
        r = run(folder, cycles, 1300 + k)
        r["candidate"] = name
        r["physics_score"] = physics_score(r)
        rows.append(r)
        print(f"  speed {r['speed_mps']:.3f} m/s, drift {r['sideways_per_m']:+.3f} m/m, "
              f"heading {r['heading_change_deg']:+.1f} deg, height {r['height_mean_m']:.3f} m, "
              f"bounce {r['bounce_mm']:.1f} mm, tilt {r['max_tilt_deg']:.1f} deg, "
              f"lowest swing {r['min_swing_clearance_cm']:.1f} cm, slide {r['slide_per_m_walked']:.3f} m/m, "
              f"self-contacts {r['self_contacts']}  ->  score {r['physics_score']:.3f}", flush=True)

    best = max(rows, key=lambda r: r["physics_score"])
    sel = os.path.join(OUT, "selected")
    os.makedirs(sel, exist_ok=True)
    for f in FILES:
        shutil.copy(os.path.join(OUT, best["candidate"], f), os.path.join(sel, f))
    keys = ["candidate", "physics_score"] + [k for k in rows[0] if k not in ("candidate", "physics_score")]
    with open(os.path.join(OUT, "selection.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in sorted(rows, key=lambda r: -r["physics_score"]):
            w.writerow(r)
    with open(os.path.join(sel, "selected_from.txt"), "w") as fh:
        fh.write(f"{best['candidate']} (physics score {best['physics_score']:.3f}); "
                 f"all candidates in ../selection.csv\n")
    print(f"SELECTED {best['candidate']} (physics score {best['physics_score']:.3f}) -> {sel}")


if __name__ == "__main__":
    main()
