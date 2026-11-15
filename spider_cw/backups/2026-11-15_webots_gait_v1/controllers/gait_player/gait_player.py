"""gait_player - walks the Webots spider with a gait evolved by the Part 1 GA.

What it does
1. Reads spider_cw/outputs/part1/<run>/best_gait.csv (300 x 24 joint angles of
   the lecturer's MATLAB model, leg order L1 L2 L3 L4 R4 R3 R2 R1) and
   best_gait_stance.csv (which feet are planted).
2. Works out where each MATLAB foot is in every frame (same maths as
   spider_fk.m / forward_leg_kinematics2.m).
3. Scales those foot paths to the Webots robot and solves the robot's own
   joint angles for them (inverse kinematics).

Why not send the MATLAB angles straight to the motors? The two spiders are
built differently: the Webots hip turns about an axis tilted 45 degrees, so
the same hip angle also raises or lowers the foot, and the segments have
different lengths (0.5/0.6/0.7 vs 1.2/0.7/1.0). Copying angles would drag
the planted feet. Copying the FOOT PATHS keeps what the GA optimised:
planted feet stay at one height and push backwards, swinging feet lift, and
the legs keep the evolved timing.

The world runs with selfCollision TRUE, so the legs must never touch each
other or the body. Feet are therefore planted spread out front-to-back
(SPREAD) and strides are limited (MAX_SWEEP).

Every cycle it prints, measured in the simulation (joint sensors + GPS +
body orientation): distance walked, body height and tilt, the LOWEST peak
height any foot reached while swinging (proves every step lifts the foot),
and the number of self-contacts (robot touching itself, should be 0).

controllerArgs (in the world file): [run, cycle seconds], e.g. ["seed3", "4"].
Set the environment variable GAIT_QUIT_AFTER=N to stop the simulation after
N cycles (used for automatic testing).
"""

import csv
import math
import os
import sys

from controller import Supervisor

# ---------------------------------------------------------------- settings --
RUN = "seed0"            # which GA run to play (folder in outputs/part1)
CYCLE_SECONDS = 4.0      # time for one full gait cycle (300 frames)
MAX_SWEEP = 0.15         # largest forward/back foot movement from its centre [m]
MAX_LIFT = 0.12          # foot lift at the top of EVERY swing [m]. Each leg's
                         # lift is scaled on its own: the GA gave some legs a
                         # lift 5x smaller than others, which on this robot
                         # would only be ~2 cm, i.e. dragging the foot.
SPREAD = 1.6             # planted feet sit at SPREAD x their hip's forward
                         # position: hips are only 0.325 m apart, so feet under
                         # their hips would hit their neighbours mid-stride.
HOME_REACH = 0.90        # how far out to the side each foot is planted [m]
HOME_DROP = 0.45         # how far below the body centre the ground is [m]
                         # The robot weighs ~43 kg and each motor gives at most
                         # 50 Nm. With the feet 1.33 m out (my_controller_s pose)
                         # 4 planted legs need ~80 Nm at the thigh, so the body
                         # sags and pitches. Closer in they need ~40 Nm.
TUCK_REACH = 0.75        # feet are pulled in this close before pushing down [m]
STAND_SECONDS = 3.0      # time to stand up before walking (tuck, push, spread)
BLEND_SECONDS = 2.0      # time to move from standing into the first gait frame
FOOT_RADIUS = 0.02       # radius of the tibia tip (from the proto)

LEGS = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]   # MATLAB / CSV order
JOINTS = ("hip", "thigh", "knee")

# Starting guess for inverse kinematics: [hip, thigh, knee]
STAND = {"L": [0.0, -1.07, 0.58], "R": [0.0, 1.07, -0.58]}
LIMITS = [(-1.5708, 1.5708), (-1.5708, 1.5708), (-2.0, 1.0)]

# -------------------------------------------- lecturer's MATLAB model (FK) --
M_SEG = (1.2, 0.7, 1.0)
M_A, M_B = 1.5, 1.0
M_BASE = [math.radians(a) for a in (45, 75, 105, 135, -135, -105, -75, -45)]
M_ELEV = math.radians(30)


def matlab_foot(i, t1, t2, t3):
    """Foot of MATLAB leg i (0..7). Same closed form as spider_fk.m."""
    b = M_BASE[i]
    psi = b + t1
    p = M_ELEV + t2
    t = p + (1.0 if math.cos(p) >= 0 else -1.0) * t3
    h = M_SEG[0] * math.cos(M_ELEV) + M_SEG[1] * math.cos(p) + M_SEG[2] * math.cos(t)
    v = M_SEG[0] * math.sin(M_ELEV) + M_SEG[1] * math.sin(p) + M_SEG[2] * math.sin(t)
    return (M_A * math.cos(b) + h * math.cos(psi), M_B * math.sin(b) + h * math.sin(psi), v)


# ------------------------------------------- Webots robot (SpiderOctopod) --
# Read from protos/SpiderOctopod.proto. s = +1 for left legs, -1 for right.
HIP_X = {"1": 0.4875, "2": 0.1625, "3": -0.1625, "4": -0.4875}
R2 = math.sqrt(0.5)


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
    """Foot position of a Webots leg in the robot body frame for angles q."""
    s = 1.0 if leg[0] == "L" else -1.0
    anchor = [HIP_X[leg[1]], 0.2 * s, 0.0]
    r_hip = mm(rot((0.0, -R2 * s, R2), q[0]), rot((1.0, 0.0, 0.0), s * math.pi / 4))
    p = [a + b for a, b in zip(anchor, mv(r_hip, [0.0, 0.5 * s, 0.0]))]
    r_thigh = mm(r_hip, rot((1.0, 0.0, 0.0), q[1]))
    p = [a + b for a, b in zip(p, mv(r_thigh, [0.0, 0.6 * s, 0.0]))]
    r_knee = mm(r_thigh, rot((1.0, 0.0, 0.0), q[2]))
    return [a + b for a, b in zip(p, mv(r_knee, [0.0, 0.0, -0.7]))]


def home_foot(leg):
    """Where a planted foot sits at the centre of its stride (body frame)."""
    s = 1.0 if leg[0] == "L" else -1.0
    return [SPREAD * HIP_X[leg[1]], s * HOME_REACH, -HOME_DROP]


def solve3(a, b):
    """Solve the 3x3 system a x = b (Gaussian elimination, partial pivoting)."""
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
    """Damped least-squares inverse kinematics. Returns (angles, error in m)."""
    q = list(q0)
    err = 1.0
    for _ in range(60):
        f = robot_foot(leg, q)
        e = [t - x for t, x in zip(target, f)]
        err = math.sqrt(sum(v * v for v in e))
        if err < 1e-7:
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
    return q, err


# ---------------------------------------------------- build the joint table --
def read_csv(path):
    with open(path, newline="") as fh:
        return [[float(v) for v in row] for row in csv.reader(fh) if row]


def build_table(run):
    here = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(here, "..", "..", "spider_cw", "outputs", "part1", run)
    gait = read_csv(os.path.join(folder, "best_gait.csv"))
    stance = [[v > 0.5 for v in row]
              for row in read_csv(os.path.join(folder, "best_gait_stance.csv"))]
    n = len(gait)

    # MATLAB foot paths, and per leg: ground height, centre of the stride and
    # highest lift during the swing.
    feet = [[matlab_foot(i, *row[3 * i:3 * i + 3]) for i in range(8)] for row in gait]
    centre, ground, peak = [], [], []
    for i in range(8):
        planted = [feet[f][i] for f in range(n) if stance[f][i]]
        centre.append((sum(p[0] for p in planted) / len(planted),
                       sum(p[1] for p in planted) / len(planted)))
        ground.append(sum(p[2] for p in planted) / len(planted))
        peak.append(max(feet[f][i][2] - ground[i] for f in range(n)))

    sweep = max(max(abs(feet[f][i][0] - centre[i][0]) for f in range(n)) for i in range(8))
    k_xy = MAX_SWEEP / sweep                       # same for all legs: keeps the stride shape

    table = [[None] * 8 for _ in range(n)]
    stand = [None] * 8
    worst = 0.0
    for i, leg in enumerate(LEGS):
        home = home_foot(leg)
        q, err = ik(leg, home, STAND[leg[0]])
        stand[i] = q
        worst = max(worst, err)
        k_z = MAX_LIFT / peak[i]                   # per leg: every swing lifts MAX_LIFT
        for f in range(n):
            x, y, z = feet[f][i]
            lift = 0.0 if stance[f][i] else k_z * max(0.0, z - ground[i])
            target = [home[0] + k_xy * (x - centre[i][0]),
                      home[1] + k_xy * (y - centre[i][1]),
                      home[2] + lift]
            q, err = ik(leg, target, q)             # warm start from previous frame
            worst = max(worst, err)
            table[f][i] = q
    print(f"[gait_player] {run}: {n} frames, stride scale {k_xy:.3f}, lift {MAX_LIFT * 100:.0f} cm "
          f"on every swing, worst IK error {worst * 1000:.2f} mm")
    return table, stand, stance


# --------------------------------------------------------------- run robot --
def main():
    global RUN, CYCLE_SECONDS
    args = [a for a in sys.argv[1:] if a]
    if args:
        RUN = args[0]
    if len(args) > 1:
        CYCLE_SECONDS = float(args[1])
    quit_after = int(os.environ.get("GAIT_QUIT_AFTER", "0"))

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

    table, stand, stance = build_table(RUN)
    n = len(table)

    def command(angles):
        for i, leg in enumerate(LEGS):
            for j in range(3):
                motors[(leg, j)].setPosition(angles[i][j])

    def foot_heights():
        """Measured height above the floor of the lowest point of each foot."""
        p = gps.getValues()
        r = me.getOrientation()
        out = []
        for leg in LEGS:
            f = robot_foot(leg, [sensors[(leg, j)].getValue() for j in range(3)])
            out.append(p[2] + r[6] * f[0] + r[7] * f[1] + r[8] * f[2] - FOOT_RADIUS)
        return out

    def contacts():
        """(self, floor) contact points. Points well above the floor = the
        robot touching itself. Floor points prove the detection is working."""
        try:
            pts = me.getContactPoints(True)
        except Exception:
            return 0, 0
        high = sum(1 for c in pts if c.getPoint()[2] > 0.06)
        return high, len(pts) - high

    # Stand up like a spider: tuck the feet in close, push straight down, then
    # spread out. Pushing with the feet far out from a lying start needs more
    # than the motors' 50 Nm and the robot stays on its belly.
    def stand_targets(leg, u):
        s = 1.0 if leg[0] == "L" else -1.0
        x = HIP_X[leg[1]]
        home = home_foot(leg)
        if u < 0.25:                                   # tuck, feet above the floor
            return [x, s * TUCK_REACH, -0.05]
        if u < 0.75:                                   # push the body up
            k = (u - 0.25) / 0.5
            return [x, s * TUCK_REACH, -0.05 - k * (HOME_DROP - 0.05)]
        k = (u - 0.75) / 0.25                          # spread to the walking stance
        return [x + k * (home[0] - x), s * TUCK_REACH + k * (home[1] - s * TUCK_REACH), -HOME_DROP]

    steps = int(STAND_SECONDS * 1000 / dt)
    q = [STAND[leg[0]] for leg in LEGS]
    plan = []
    for k in range(1, steps + 1):
        q = [ik(leg, stand_targets(leg, k / steps), q[i])[0] for i, leg in enumerate(LEGS)]
        plan.append(q)
    for angles in plan:
        command(angles)
        if robot.step(dt) == -1:
            return
    command(stand)
    for _ in range(int(500 / dt)):
        if robot.step(dt) == -1:
            return
    print(f"[gait_player] stood up: body height {gps.getValues()[2]:.3f} m")

    steps = int(BLEND_SECONDS * 1000 / dt)
    for s in range(1, steps + 1):
        a = s / steps
        command([[(1 - a) * stand[i][j] + a * table[0][i][j] for j in range(3)] for i in range(8)])
        if robot.step(dt) == -1:
            return

    start = gps.getValues()
    print(f"[gait_player] walking from x={start[0]:.3f} y={start[1]:.3f} z={start[2]:.3f}")
    t = 0.0
    cycle = 0
    swing_peak = [0.0] * 8            # highest point of each foot while swinging
    stance_max = 0.0                  # highest point of any foot while planted
    touches = 0
    floor_pts = 0
    samples = 0
    while robot.step(dt) != -1:
        t += dt / 1000.0
        pos = (t / CYCLE_SECONDS) % 1.0 * n
        f0 = int(pos) % n
        f1 = (f0 + 1) % n
        a = pos - int(pos)
        command([[(1 - a) * table[f0][i][j] + a * table[f1][i][j] for j in range(3)]
                 for i in range(8)])

        h = foot_heights()
        for i in range(8):
            if stance[f0][i]:
                stance_max = max(stance_max, h[i])
            else:
                swing_peak[i] = max(swing_peak[i], h[i])
        hi, lo = contacts()
        touches += hi
        floor_pts += lo
        samples += 1

        if int(t / CYCLE_SECONDS) > cycle:
            cycle = int(t / CYCLE_SECONDS)
            p = gps.getValues()
            r = me.getOrientation()
            roll = math.degrees(math.atan2(r[7], r[8]))
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r[6]))))
            low = min(range(8), key=lambda i: swing_peak[i])
            print(f"[gait_player] cycle {cycle}: walked {p[0] - start[0]:+.3f} m forward, "
                  f"{p[1] - start[1]:+.3f} m sideways, height {p[2]:.3f} m, "
                  f"roll {roll:+.1f}, pitch {pitch:+.1f} deg | lowest swing peak "
                  f"{swing_peak[low] * 100:.1f} cm ({LEGS[low]}), planted feet up to "
                  f"{stance_max * 100:.1f} cm | self-contacts {touches} "
                  f"(floor contacts per step {floor_pts / max(samples, 1):.1f})")
            swing_peak = [0.0] * 8
            stance_max = 0.0
            touches = 0
            floor_pts = 0
            samples = 0
            if quit_after and cycle >= quit_after:
                print("[gait_player] quitting")
                robot.simulationQuit(0)
                robot.step(dt)
                return


main()
