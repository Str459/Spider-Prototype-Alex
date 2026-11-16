"""The Webots spider (Spider_world_demo/protos/SpiderOctopod.proto) as a second body to evolve for.

    python -m part1_ga.run_ga --model webots --seed 0      # -> outputs/part1/webots_seed0/

Gaits evolved for the lecturer's MATLAB model do not transfer: the Webots spider has shorter
legs (coxa 0.5, thigh 0.6, tibia 0.7 m vs 1.2/0.7/1.0), legs pointing straight out sideways
from a 1.3 x 0.4 x 0.3 m box, hip axes tilted 45 deg and different joint limits, so the same
angles put the feet somewhere else and the stroke mostly scrapes. activate() swaps this body
into gait/fitness/plots; the chromosome layout (27 genes, LR = pi) and every fitness term stay
the same, so the GA itself is untouched. Genes are Webots motor angles, so the CSV plays as-is.

Differences from the MATLAB setup, all taken from the proto:
* kinematics(): each leg is hip anchor -> coxa (raised 45 deg, turned by the tilted hip axis)
  -> thigh -> tibia (90 deg to the thigh at knee 0). The thigh and knee both turn about the
  coxa's local x axis, so after the hip the leg is a planar chain.
* Mirror: the right legs' hip, thigh and knee are all the negated left angles (my_controller_s
  stands with R thigh 1.07 / knee -0.58, L -1.07 / 0.58).
* Angles are clipped to the motor limits before kinematics, as the Webots motors would.
* Self-collision (as with selfCollision TRUE): every leg segment is a capsule with the proto's
  thickness; a frame where any segment comes within COLLISION_GAP of a segment of the next or
  next-but-one leg, or a thigh/tibia enters the body box, counts as a collision. The fitness
  charges COLLISION_PENALTY x the fraction of colliding frames plus the overlap depth.
* Length-based fitness constants are scaled to the ~0.6x smaller robot.
"""
import numpy as np

HIP_X = np.array([0.4875, 0.1625, -0.1625, -0.4875])
SIDE = np.repeat([1.0, -1.0], 4)                               # MATLAB leg order L1..L4, R4..R1
HIP_POS = np.stack([np.concatenate([HIP_X, HIP_X[::-1]]), 0.2 * SIDE, np.zeros(8)], axis=1)
SEG_LEN = np.array([0.5, 0.6, 0.7])
SEG_RADIUS = np.array([0.03, 0.028, 0.02])                     # coxa cylinder, thigh 0.04 box, tibia cylinder
BODY_HALF = np.array([0.65, 0.2, 0.15])
COLLISION_GAP = 0.02                                           # clearance kept between leg surfaces (m)
COLLISION_PENALTY = 5.0                                        # per fraction of frames with a collision

MOTOR_LO = np.tile([-1.5708, -1.5708, -2.0], 8)                # per joint, from the proto
MOTOR_HI = np.tile([1.5708, 1.5708, 1.0], 8)
# Left-leg gene bounds. A right knee is the negated left knee and both must fit in [-2, 1],
# so the left knee pose is kept in [-1, 1]; hip pose +-0.8 keeps neighbouring legs apart.
POSE_LO, POSE_HI = np.tile([-0.8, -1.5, -1.0], 4), np.tile([0.8, 1.5, 1.0], 4)
AMP_HI = np.tile([0.5, 0.5, 0.5], 4)

_c, _s = np.cos(np.pi / 4), np.sin(np.pi / 4)
_HIP_AXIS = np.stack([np.zeros(8), -_c * SIDE, np.full(8, _c)], axis=1)
_COXA0 = np.stack([np.zeros(8), _c * SIDE, np.full(8, _s)], axis=1)      # Rot(x, side*45deg) @ (0, side, 0)
_DOWN0 = np.stack([np.zeros(8), _s * SIDE, np.full(8, -_c)], axis=1)     # Rot(x, side*45deg) @ (0, 0, -1)


def _rodrigues(v, k, a):
    """Rotate vectors v about unit axes k by angles a (broadcasting; v, k (...,3), a (...))."""
    a = a[..., None]
    return v * np.cos(a) + np.cross(k, v) * np.sin(a) + k * (k * v).sum(-1, keepdims=True) * (1 - np.cos(a))


def forward_kinematics(angles):
    """Webots motor angles (..., 24) in MATLAB leg order -> [hip, knee1, knee2, foot] (..., 8, 4, 3)."""
    a = np.clip(np.asarray(angles, dtype=float), MOTOR_LO, MOTOR_HI)
    a = a.reshape(*a.shape[:-1], 8, 3)
    hip, thigh, knee = a[..., 0], a[..., 1], a[..., 2]
    coxa = _rodrigues(np.broadcast_to(_COXA0, a.shape), _HIP_AXIS, hip)
    down = _rodrigues(np.broadcast_to(_DOWN0, a.shape), _HIP_AXIS, hip)
    ex = _rodrigues(np.broadcast_to(np.eye(3)[0], a.shape), _HIP_AXIS, hip)   # coxa's local x axis
    thigh_dir = _rodrigues(coxa, ex, thigh)
    tibia_dir = _rodrigues(down, ex, thigh + knee)         # thigh and knee turn about the same axis
    p1 = HIP_POS + SEG_LEN[0] * coxa
    p2 = p1 + SEG_LEN[1] * thigh_dir
    p3 = p2 + SEG_LEN[2] * tibia_dir
    return np.stack([np.broadcast_to(HIP_POS, p1.shape), p1, p2, p3], axis=-2)


def kinematics(angles):
    """Same contract as fitness._kinematics: (..., 24) -> [knee1, knee2, foot] (..., 8, 3, 3)."""
    return forward_kinematics(angles)[..., 1:, :]


def feet(angles):
    return forward_kinematics(angles)[..., 3, :]


def _segment_distance(p1, q1, p2, q2):
    """Closest distance between segments p1-q1 and p2-q2 (broadcasting over leading axes)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = (d1 * d1).sum(-1), (d2 * d2).sum(-1)
    b, c, f = (d1 * d2).sum(-1), (d1 * r).sum(-1), (d2 * r).sum(-1)
    denom = a * e - b * b
    s = np.where(denom > 1e-12, np.clip((b * f - c * e) / np.maximum(denom, 1e-12), 0, 1), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0, np.clip(-c / a, 0, 1), np.where(t > 1, np.clip((b - c) / a, 0, 1), s))
    t = np.clip(t, 0, 1)
    return np.linalg.norm(p1 + d1 * s[..., None] - p2 - d2 * t[..., None], axis=-1)


def self_collision(joints):
    """joints (n,T,8,3,3) [knee1, knee2, foot] -> (fraction of colliding frames, mean overlap) (n,)."""
    pts = np.concatenate([np.broadcast_to(HIP_POS[:, None, :], joints[..., :1, :].shape), joints], axis=-2)
    start, end = pts[..., :-1, :], pts[..., 1:, :]                         # (n,T,8,3 segments,3)
    overlap = []
    for shift in (1, 2):                                                   # next and next-but-one leg on the ring
        o_start, o_end = np.roll(start, -shift, axis=-3), np.roll(end, -shift, axis=-3)
        d = _segment_distance(start[..., :, None, :], end[..., :, None, :],
                              o_start[..., None, :, :], o_end[..., None, :, :])   # (n,T,8,3,3)
        reach = SEG_RADIUS[:, None] + SEG_RADIUS[None, :] + COLLISION_GAP
        overlap.append(np.maximum(reach - d, 0).sum(axis=(-1, -2, -3)))
    # thigh and tibia must stay out of the body box (the coxa is jointed to the body, so exempt)
    w = np.linspace(0, 1, 5)[:, None]
    seg_pts = start[..., 1:, None, :] + w * (end - start)[..., 1:, None, :]            # (n,T,8,2,5,3)
    inside = (np.abs(seg_pts) < BODY_HALF + SEG_RADIUS[1]).all(-1).any(axis=(-1, -2, -3))
    total = overlap[0] + overlap[1]
    return ((total > 0) | inside).mean(axis=1), total.mean(axis=1)


def plot_pose(ax, angles, title=None):
    """Draw one Webots-body pose (body box outline, coxa black, thigh blue, tibia red)."""
    pts = forward_kinematics(angles)
    ax.cla()
    x, y = BODY_HALF[0], BODY_HALF[1]
    ax.plot([x, -x, -x, x, x], [y, y, -y, -y, y], [0] * 5, "k-", lw=3)
    ax.plot([x + 0.1], [0], [0], "r^", ms=10)                                # head marker
    for leg in range(8):
        for seg, colour in enumerate("kbr"):
            ax.plot(*pts[leg, seg:seg + 2].T, colour + "-", lw=2)
        ax.plot(*pts[leg, 3], "ro", ms=4)
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-0.75, 0.75), xlabel="X", ylabel="Y", zlabel="Z")
    ax.set_box_aspect((4, 4, 2))
    ax.view_init(45, -45)
    if title:
        ax.set_title(title)


def activate():
    """Evolve for the Webots body from now on (this process only): patch gait, fitness and plots."""
    from part1_ga import fitness, gait, plots

    gait.LOW = np.concatenate([POSE_LO, np.zeros(12), np.zeros(gait.N_PHASES)])
    gait.HIGH = np.concatenate([POSE_HI, AMP_HI, np.full(gait.N_PHASES, 2 * np.pi)])
    gait._MIRROR = np.where(gait._RIGHT == 1, -1.0, 1.0)
    fitness._kinematics = kinematics
    fitness.JOINT_LO, fitness.JOINT_HI = MOTOR_LO, MOTOR_HI
    fitness.LOW, fitness.HIGH = gait.LOW, gait.HIGH
    # A foot only counts as lifted 8 cm up, and lift pays up to 15 cm: with 3 / 8 cm the evolved
    # feet skimmed the floor and in Webots (body sags and tilts a few cm) they just scraped.
    fitness.STANCE_TOL, fitness.MIN_HEIGHT, fitness.MARGIN_CAP, fitness.LIFT_CAP = 0.08, 0.25, 0.12, 0.15
    fitness.FOOT_GAP = 0.0                                  # replaced by the full segment check below
    base_clearance = fitness.clearance

    def clearance(joints, ground):
        frac, depth = self_collision(joints)
        return base_clearance(joints, ground) - COLLISION_PENALTY * frac - depth

    fitness.clearance = clearance
    plots.feet, plots.plot_pose = feet, plot_pose


if __name__ == "__main__":
    import math

    def rot(axis, a):                                       # independent matrix version, as in the proto
        x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
        c, s, C = math.cos(a), math.sin(a), 1 - math.cos(a)
        return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                         [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                         [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])

    rng = np.random.default_rng(0)
    ang = rng.uniform(MOTOR_LO, MOTOR_HI)
    fk = forward_kinematics(ang)
    for leg in range(8):
        s = SIDE[leg]
        h, t, k = ang[3 * leg:3 * leg + 3]
        rc = rot([0, -0.707108 * s, 0.707105], h) @ rot([1, 0, 0], s * 0.7854)
        rt = rc @ rot([1, 0, 0], t)
        p1 = HIP_POS[leg] + rc @ [0, 0.5 * s, 0]
        p2 = p1 + rt @ [0, 0.6 * s, 0]
        p3 = p2 + rt @ rot([1, 0, 0], k) @ [0, 0, -0.7]
        assert np.allclose(fk[leg, 1:], [p1, p2, p3], atol=1e-4), leg
    assert np.allclose(np.linalg.norm(np.diff(fk, axis=-2), axis=-1), SEG_LEN)

    stand = np.tile([0.0, -1.07, 0.58], 8) * np.repeat(SIDE, 3)         # my_controller_s: L (-1.07, 0.58), R mirrored
    assert np.allclose(stand[21:], [0.0, 1.07, -0.58])
    z = feet(stand)[:, 2]
    assert np.ptp(z) < 1e-9 and z[0] < -0.15, z                # all feet level, below the body box
    frac, _ = self_collision(kinematics(stand)[None, None])
    assert frac[0] == 0, "standing pose must not collide"
    crossed = stand.copy()
    crossed[0], crossed[3] = 1.2, -1.2                          # L1 swung back, L2 swung forward into it
    frac, depth = self_collision(kinematics(crossed)[None, None])
    assert frac[0] == 1 and depth[0] > 0, "crossed legs must collide"

    activate()
    from part1_ga import fitness, gait
    pop = gait.random_population(20, rng)
    scores, comps = fitness.evaluate_population(pop)
    assert scores.shape == (20,) and np.isfinite(scores).all()
    print(f"webots_model self-check passed (standing body height {-z[0]:.3f} m)")
