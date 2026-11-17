"""Walk gait folders in Webots (headless, self-collision on) and record what really happened.

    python webots/measure.py part1/speed_0.10 part1/speed_0.15 ...   (folders inside outputs/)
    python webots/measure.py --cycles 10 part1/tournament_twopoint_seed1

For each folder it starts Webots on worlds/spider.wbt, lets controllers/gait_player stand
up and walk, and saves the measurements (speed, drift, tilt, sliding, self-contacts...) to
<folder>/webots_result.json. Needs Webots; set WEBOTS_EXE if it is not in the default place.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(HERE, "..", "outputs")
WORLD = os.path.join(HERE, "worlds", "spider.wbt")
WEBOTS = os.environ.get("WEBOTS_EXE", r"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe")


def measure(name, cycles=5, port=1330):
    folder = os.path.normpath(os.path.join(OUTPUTS, name))
    result = os.path.join(folder, "webots_result.json")
    if os.path.exists(result):
        os.remove(result)
    env = dict(os.environ, GAIT_DIR=folder, GAIT_QUIT_AFTER=str(cycles), GAIT_RESULT=result)
    subprocess.run([WEBOTS, f"--port={port}", "--batch", "--mode=fast", "--minimize", "--no-rendering",
                    "--stdout", "--stderr", WORLD], env=env, capture_output=True, text=True, timeout=900)
    with open(result) as f:
        return json.load(f)


if __name__ == "__main__":
    args = sys.argv[1:]
    cycles = 5
    if "--cycles" in args:
        i = args.index("--cycles")
        cycles = int(args[i + 1])
        del args[i:i + 2]
    for name in args:
        r = measure(name, cycles)
        print(f"{name}: {r['speed_mps']:.3f} m/s, drift {100 * r['sideways_per_m']:+.1f} cm/m, "
              f"heading {r['heading_change_deg']:+.1f} deg, tilt {r['max_tilt_deg']:.1f} deg, "
              f"slide {r['slide_per_step_cm']:.2f} cm/step, self-contacts {r['self_contacts']}", flush=True)
