"""gait_player - plays a gait on the Webots SpiderOctopod and measures it.

A gait folder (written by the MATLAB GA, part1_ga_matlab/run_ga.m, or by the
network, part2_nn_python/predict.py) contains
    joint_table.csv   frames x 24 motor angles (one pose per row)
    stance.csv        frames x 8 (1 = foot planted)
    stand.csv         1 x 24 standing pose
    meta.csv          key,value: frames, period, stride, duty, ...

controllerArgs: [gait folder inside outputs/], default "part1/tournament_twopoint_seed1".
Environment (used by webots/measure.py):
    GAIT_DIR        full path of a gait folder (overrides controllerArgs)
    GAIT_QUIT_AFTER stop the simulation after this many cycles
    GAIT_RESULT     write the measurements to this JSON file when quitting
    GAIT_SNAPSHOTS  folder: save 6 screenshots during cycle 2

Printed every cycle (all MEASURED in the simulation): distance walked,
sideways drift, heading change, body height, roll/pitch, the lowest swing
height any foot reached, how far planted feet slid on the floor, and how
many times the robot touched itself (self-collision is on).
"""

import csv
import json
import math
import os
import sys

from controller import Supervisor

LEGS = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]    # order of every CSV
JOINTS = ("hip", "thigh", "knee")
HIP_X = {"1": 0.4875, "2": 0.1625, "3": -0.1625, "4": -0.4875}
LIMITS = [(-1.5708, 1.5708), (-1.5708, 1.5708), (-2.0, 1.0)]
STAND_SEED = {"L": [0.0, -1.07, 0.58], "R": [0.0, 1.07, -0.58]}
FOOT_RADIUS = 0.02
TUCK_REACH = 0.75        # feet are pulled in this close before pushing up [m]
STAND_SECONDS = 3.0      # tuck, push up, spread out
BLEND_SECONDS = 1.5      # standing pose -> first frame of the gait
R2 = math.sqrt(0.5)


# ------------------------------------------------ robot kinematics (proto) --
def rot(axis, a):
    x, y, z = axis
    c, s, C = math.cos(a), math.sin(a), 1 - math.cos(a)
    return [[x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c]]


def mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mv(a, v):
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def robot_foot(leg, q):
    """Foot position of a leg in the body frame (hip -> coxa -> thigh -> knee chain)."""
    s = 1.0 if leg[0] == "L" else -1.0
    anchor = [HIP_X[leg[1]], 0.2 * s, 0.0]
    r_hip = mm(rot((0.0, -R2 * s, R2), q[0]), rot((1.0, 0.0, 0.0), s * math.pi / 4))
    p = [a + b for a, b in zip(anchor, mv(r_hip, [0.0, 0.5 * s, 0.0]))]
    r_thigh = mm(r_hip, rot((1.0, 0.0, 0.0), q[1]))
    p = [a + b for a, b in zip(p, mv(r_thigh, [0.0, 0.6 * s, 0.0]))]
    r_knee = mm(r_thigh, rot((1.0, 0.0, 0.0), q[2]))
    return [a + b for a, b in zip(p, mv(r_knee, [0.0, 0.0, -0.7]))]


def solve3(a, b):
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(3):
        piv = max(range(c, 3), key=lambda r: abs(m[r][c]))
        m[c], m[piv] = m[piv], m[c]
        for r in range(c + 1, 3):
            f = m[r][c] / m[c][c]
            for k in range(c, 4):
                m[r][k] -= f * m[c][k]
    x = [0.0, 0.0, 0.0]
    for r in (2, 1, 0):
        x[r] = (m[r][3] - sum(m[r][k] * x[k] for k in range(r + 1, 3))) / m[r][r]
    return x


def ik(leg, target, q0):
    """Damped least squares, only used for the stand-up motion."""
    q = list(q0)
    for _ in range(60):
        f = robot_foot(leg, q)
        e = [t - x for t, x in zip(target, f)]
        if math.sqrt(sum(v * v for v in e)) < 1e-7:
            break
        jac = [[0.0] * 3 for _ in range(3)]
        for j in range(3):
            dq = list(q)
            dq[j] += 1e-6
            fj = robot_foot(leg, dq)
            for i in range(3):
                jac[i][j] = (fj[i] - f[i]) / 1e-6
        jtj = [[sum(jac[k][i] * jac[k][j] for k in range(3)) + (1e-6 if i == j else 0.0)
                for j in range(3)] for i in range(3)]
        jte = [sum(jac[k][i] * e[k] for k in range(3)) for i in range(3)]
        step = solve3(jtj, jte)
        q = [min(max(q[i] + step[i], LIMITS[i][0]), LIMITS[i][1]) for i in range(3)]
    return q


# ----------------------------------------------------------------- gait io --
def read_rows(path):
    with open(path, newline="") as fh:
        return [[float(v) for v in row] for row in csv.reader(fh) if row]


def load_gait():
    here = os.path.dirname(os.path.abspath(__file__))
    args = [a for a in sys.argv[1:] if a]
    folder = os.environ.get("GAIT_DIR") or os.path.join(
        here, "..", "..", "..", "outputs", args[0] if args else "part1/tournament_twopoint_seed1")
    table = read_rows(os.path.join(folder, "joint_table.csv"))
    stance = [[v > 0.5 for v in row] for row in read_rows(os.path.join(folder, "stance.csv"))]
    stand = read_rows(os.path.join(folder, "stand.csv"))[0]
    meta = {}
    with open(os.path.join(folder, "meta.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            meta[row["key"]] = float(row["value"])
    as_legs = lambda r: [r[3 * i:3 * i + 3] for i in range(8)]
    # Start-up (stride grows one step at a time) is played once, before the loop.
    startup, startup_stance = [], []
    if os.path.exists(os.path.join(folder, "startup.csv")):
        startup = [as_legs(r) for r in read_rows(os.path.join(folder, "startup.csv"))]
        startup_stance = [[v > 0.5 for v in row]
                          for row in read_rows(os.path.join(folder, "startup_stance.csv"))]
    frames = startup + [as_legs(r) for r in table]
    return folder, frames, startup_stance + stance, len(startup), as_legs(stand), meta


# --------------------------------------------------------------------- run --
def main():
    folder, table, stance, n_start, stand, meta = load_gait()
    n = int(meta["frames"])                  # frames per cycle of the loop
    period = meta["period"]
    start_cycles = n_start // n              # cycles spent starting up
    quit_after = int(os.environ.get("GAIT_QUIT_AFTER", "0"))   # steady cycles to measure
    snap_dir = os.environ.get("GAIT_SNAPSHOTS")

    robot = Supervisor()
    dt = int(robot.getBasicTimeStep())
    me = robot.getSelf()
    motors, sensors = {}, {}
    for leg in LEGS:
        for j, name in enumerate(JOINTS):
            m = robot.getDevice(f"{leg}_{name}_joint")
            m.setVelocity(m.getMaxVelocity())
            motors[(leg, j)] = m
            ps = robot.getDevice(f"{leg}_{name}_joint_sensor")
            ps.enable(dt)
            sensors[(leg, j)] = ps
    gps = robot.getDevice("gps")
    gps.enable(dt)
    try:
        me.enableContactPointsTracking(dt, True)
    except AttributeError:
        pass
    print(f"[gait_player] gait {os.path.basename(os.path.normpath(folder))}: {n} frames, "
          f"period {period:.2f} s, stride {meta['stride']:.3f} m, duty {meta['duty']:.2f}, "
          f"step height {meta['stepHeight']:.3f} m, body height {meta['bodyHeight']:.2f} m, "
          f"planned speed {meta['speed']:.3f} m/s")

    def command(angles):
        for i, leg in enumerate(LEGS):
            for j in range(3):
                motors[(leg, j)].setPosition(angles[i][j])

    def step():
        if robot.step(dt) == -1:
            sys.exit(0)

    # Stand up: tuck the feet in, push the body up, spread out to the gait's
    # own standing pose. (Pushing up with the feet far out needs more than
    # the motors' 50 Nm and the robot stays on its belly.)
    home = [robot_foot(leg, stand[i]) for i, leg in enumerate(LEGS)]

    def target(i, leg, u):
        s = 1.0 if leg[0] == "L" else -1.0
        x, z_home = HIP_X[leg[1]], home[i][2]
        if u < 0.25:
            return [x, s * TUCK_REACH, -0.05]
        if u < 0.75:
            k = (u - 0.25) / 0.5
            return [x, s * TUCK_REACH, -0.05 + k * (z_home + 0.05)]
        k = (u - 0.75) / 0.25
        return [x + k * (home[i][0] - x), s * TUCK_REACH + k * (home[i][1] - s * TUCK_REACH), z_home]

    steps = int(STAND_SECONDS * 1000 / dt)
    q = [STAND_SEED[leg[0]] for leg in LEGS]
    plan = []
    for k in range(1, steps + 1):
        q = [ik(leg, target(i, leg, k / steps), q[i]) for i, leg in enumerate(LEGS)]
        plan.append(q)
    for angles in plan:
        command(angles)
        step()
    command(stand)
    for _ in range(int(500 / dt)):
        step()
    print(f"[gait_player] stood up: body height {gps.getValues()[2]:.3f} m")
    steps = int(BLEND_SECONDS * 1000 / dt)
    for k in range(1, steps + 1):
        a = k / steps
        command([[(1 - a) * stand[i][j] + a * table[0][i][j] for j in range(3)] for i in range(8)])
        step()

    # Walk and measure -----------------------------------------------------------
    def body():
        p = gps.getValues()
        r = me.getOrientation()
        return p, r

    def world_feet(p, r):
        out = []
        for leg in LEGS:
            f = robot_foot(leg, [sensors[(leg, j)].getValue() for j in range(3)])
            out.append([p[k] + r[3 * k] * f[0] + r[3 * k + 1] * f[1] + r[3 * k + 2] * f[2]
                        for k in range(3)])
        return out

    p0, r0 = body()
    start, yaw0 = p0, math.atan2(r0[3], r0[0])
    print(f"[gait_player] walking from x={start[0]:.3f} y={start[1]:.3f}")
    cycles = []
    cur = None
    prev_feet = None
    prev_frame = None
    t = 0.0
    cycle = 0
    shots = set()

    def new_cycle(p):
        return {"x0": p[0], "y0": p[1], "heights": [], "tilt": 0.0, "swing_peak": [0.0] * 8,
                "slide": 0.0, "stances": 0, "self": 0, "floor": 0, "samples": 0}

    down_at = [None] * 8
    cur = new_cycle(p0)
    while True:
        # Frames 0 .. n_start-1 are the start-up (played once), then the
        # n-frame loop repeats. Both run at n frames per period.
        t += dt / 1000.0
        g = t / period * n
        if g < n_start:
            f0, f1 = int(g), int(g) + 1          # f1 == n_start is loop frame 0
        else:
            k = int(g - n_start) % n
            f0, f1 = n_start + k, n_start + (k + 1) % n
        a = g - int(g)
        pos = (t / period) % 1.0 * n             # position within the cycle (snapshots)
        command([[(1 - a) * table[f0][i][j] + a * table[f1][i][j] for j in range(3)]
                 for i in range(8)])
        step()

        p, r = body()
        feet = world_feet(p, r)
        roll = math.degrees(math.atan2(r[7], r[8]))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r[6]))))
        cur["heights"].append(p[2])
        cur["tilt"] = max(cur["tilt"], abs(roll), abs(pitch))
        for i in range(8):
            h = feet[i][2] - FOOT_RADIUS
            if not stance[f0][i]:
                cur["swing_peak"][i] = max(cur["swing_peak"][i], h)
            # Sliding = how far a planted foot moved over the floor between
            # touch-down and lift-off (net), so physics jitter does not add up.
            was = prev_frame is not None and stance[prev_frame][i]
            if stance[f0][i] and not was:
                down_at[i] = feet[i]
            elif was and not stance[f0][i] and down_at[i] is not None:
                cur["slide"] += math.hypot(prev_feet[i][0] - down_at[i][0],
                                           prev_feet[i][1] - down_at[i][1])
                cur["stances"] += 1
        prev_feet, prev_frame = feet, f0
        try:
            pts = me.getContactPoints(True)
            hi = sum(1 for c in pts if c.getPoint()[2] > 0.06)
            cur["self"] += hi
            cur["floor"] += len(pts) - hi
        except Exception:
            pass
        cur["samples"] += 1

        if snap_dir and cycle == start_cycles + 1:
            k = int(pos * 6 / n)
            if k not in shots:
                shots.add(k)
                robot.exportImage(os.path.join(snap_dir, f"walk_{k}.jpg"), 90)

        if int(t / period) > cycle:
            cycle = int(t / period)
            hs = cur["heights"]
            mean_h = sum(hs) / len(hs)
            std_h = math.sqrt(sum((h - mean_h) ** 2 for h in hs) / len(hs))
            yaw = math.degrees(math.atan2(r[3], r[0]) - yaw0)
            rec = {"cycle": cycle, "dx": p[0] - cur["x0"], "dy": p[1] - cur["y0"],
                   "yaw_total": yaw, "height": mean_h, "bounce": std_h, "tilt": cur["tilt"],
                   "min_swing": min(cur["swing_peak"]),
                   "min_swing_leg": LEGS[cur["swing_peak"].index(min(cur["swing_peak"]))],
                   "slide": cur["slide"], "stances": cur["stances"], "self": cur["self"],
                   "floor_per_step": cur["floor"] / max(cur["samples"], 1)}
            cycles.append(rec)
            label = "start-up" if cycle <= start_cycles else "walking"
            print(f"[gait_player] cycle {cycle} ({label}): forward {rec['dx']:+.3f} m, sideways {rec['dy']:+.3f} m, "
                  f"heading {yaw:+.1f} deg | height {mean_h:.3f} m (bounce {std_h * 1000:.1f} mm), "
                  f"max tilt {rec['tilt']:.1f} deg | lowest swing {rec['min_swing'] * 100:.1f} cm "
                  f"({rec['min_swing_leg']}) | planted feet slid {rec['slide'] * 100:.1f} cm over {rec['stances']} steps | "
                  f"self-contacts {rec['self']} (floor contacts/step {rec['floor_per_step']:.1f})")
            cur = new_cycle(p)
            if quit_after and cycle >= start_cycles + quit_after:
                write_result(cycles, period, start, p, start_cycles)
                robot.simulationQuit(0)
                step()
                return


def write_result(cycles, period, start, end, skip):
    """Summary of the steady walking cycles (after the start-up), plus the
    worst values seen during the start-up."""
    path = os.environ.get("GAIT_RESULT")
    use = cycles[skip:]
    early = cycles[:skip] or cycles[:1]
    dist = sum(c["dx"] for c in use)
    side = sum(c["dy"] for c in use)
    secs = len(use) * period
    yaw_before = cycles[skip - 1]["yaw_total"] if skip > 0 else 0.0
    res = {
        "cycles_measured": len(use),
        "speed_mps": dist / secs,
        "sideways_per_m": side / dist if dist > 1e-6 else 99.0,
        "heading_change_deg": use[-1]["yaw_total"] - yaw_before,
        "startup_heading_change_deg": yaw_before,
        "startup_max_tilt_deg": max(c["tilt"] for c in early),
        "startup_bounce_mm": 1000 * max(c["bounce"] for c in early),
        "startup_slide_per_step_cm": 100 * sum(c["slide"] for c in early) / max(1, sum(c["stances"] for c in early)),
        "height_mean_m": sum(c["height"] for c in use) / len(use),
        "bounce_mm": 1000 * max(c["bounce"] for c in use),
        "max_tilt_deg": max(c["tilt"] for c in use),
        "min_swing_clearance_cm": 100 * min(c["min_swing"] for c in use),
        "slide_per_m_walked": sum(c["slide"] for c in use) / dist if dist > 1e-6 else 99.0,
        "slide_per_step_cm": 100 * sum(c["slide"] for c in use) / max(1, sum(c["stances"] for c in use)),
        "self_contacts": sum(c["self"] for c in use),
        "total_forward_m": end[0] - start[0],
    }
    print("[gait_player] RESULT " + json.dumps(res))
    if path:
        with open(path, "w") as fh:
            json.dump(res, fh, indent=1)


main()
