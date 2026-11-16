"""Geometry of the Webots spider (SpiderOctopod.proto): leg kinematics, a self-collision
pre-check and a 3D pose plot. The fitness itself comes from the Webots physics run; this module
lets the trainer skip simulating gaits whose commanded motion makes legs pass through each other,
and lets plots.py draw poses.

Each leg: hip anchor on the 1.3 x 0.4 x 0.3 m body box -> coxa 0.5 m (raised 45 deg, turned by a
hip axis tilted 45 deg) -> thigh 0.6 m -> tibia 0.7 m (at 90 deg to the thigh when the knee is
0). Thigh and knee both turn about the coxa's local x axis, so after the hip the leg is a
planar chain. Verified against a matrix version written straight from the proto (self-check).
"""
import numpy as np

from part1_ga.gait import MOTOR_LO, MOTOR_HI

HIP_X = np.array([0.4875, 0.1625, -0.1625, -0.4875])
SIDE = np.repeat([1.0, -1.0], 4)                               # leg order L1..L4, R4..R1
HIP_POS = np.stack([np.concatenate([HIP_X, HIP_X[::-1]]), 0.2 * SIDE, np.zeros(8)], axis=1)
SEG_LEN = np.array([0.5, 0.6, 0.7])
SEG_RADIUS = np.array([0.03, 0.028, 0.02])                     # coxa cylinder, thigh 0.04 box, tibia cylinder
BODY_HALF = np.array([0.65, 0.2, 0.15])
COLLISION_GAP = 0.02                                           # clearance kept between leg surfaces (m)

_c = np.cos(np.pi / 4)
_HIP_AXIS = np.stack([np.zeros(8), -_c * SIDE, np.full(8, _c)], axis=1)
_COXA0 = np.stack([np.zeros(8), _c * SIDE, np.full(8, _c)], axis=1)       # Rot(x, side*45deg) @ (0, side, 0)
_DOWN0 = np.stack([np.zeros(8), _c * SIDE, np.full(8, -_c)], axis=1)      # Rot(x, side*45deg) @ (0, 0, -1)


def _rodrigues(v, k, a):
    """Rotate vectors v about unit axes k by angles a (broadcasting)."""
    a = a[..., None]
    return v * np.cos(a) + np.cross(k, v) * np.sin(a) + k * (k * v).sum(-1, keepdims=True) * (1 - np.cos(a))


def forward_kinematics(angles):
    """Motor angles (..., 24) -> [hip, knee1, knee2, foot] positions (..., 8, 4, 3), body frame."""
    a = np.clip(np.asarray(angles, dtype=float), MOTOR_LO, MOTOR_HI)
    a = a.reshape(*a.shape[:-1], 8, 3)
    hip, thigh, knee = a[..., 0], a[..., 1], a[..., 2]
    coxa = _rodrigues(np.broadcast_to(_COXA0, a.shape), _HIP_AXIS, hip)
    down = _rodrigues(np.broadcast_to(_DOWN0, a.shape), _HIP_AXIS, hip)
    ex = _rodrigues(np.broadcast_to(np.eye(3)[0], a.shape), _HIP_AXIS, hip)   # coxa's local x axis
    p1 = HIP_POS + SEG_LEN[0] * coxa
    p2 = p1 + SEG_LEN[1] * _rodrigues(coxa, ex, thigh)
    p3 = p2 + SEG_LEN[2] * _rodrigues(down, ex, thigh + knee)
    return np.stack([np.broadcast_to(HIP_POS, p1.shape), p1, p2, p3], axis=-2)


def _segment_distance(p1, q1, p2, q2):
    """Closest distance between segments p1-q1 and p2-q2 (broadcasting)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = (d1 * d1).sum(-1), (d2 * d2).sum(-1)
    b, c, f = (d1 * d2).sum(-1), (d1 * r).sum(-1), (d2 * r).sum(-1)
    denom = a * e - b * b
    s = np.where(denom > 1e-12, np.clip((b * f - c * e) / np.maximum(denom, 1e-12), 0, 1), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0, np.clip(-c / a, 0, 1), np.where(t > 1, np.clip((b - c) / a, 0, 1), s))
    t = np.clip(t, 0, 1)
    return np.linalg.norm(p1 + d1 * s[..., None] - p2 - d2 * t[..., None], axis=-1)


def collision_fraction(frames):
    """frames (..., T, 24) -> fraction of frames (...) where a leg segment touches a segment of the
    next or next-but-one leg, or a thigh/tibia enters the body box (as Webots selfCollision would)."""
    pts = forward_kinematics(frames)
    start, end = pts[..., :-1, :], pts[..., 1:, :]                         # (..., T, 8, 3 segs, 3)
    hit = np.zeros(pts.shape[:-3], bool)
    reach = SEG_RADIUS[:, None] + SEG_RADIUS[None, :] + COLLISION_GAP
    for shift in (1, 2):                                                   # neighbours on the leg ring
        o_start, o_end = np.roll(start, -shift, axis=-3), np.roll(end, -shift, axis=-3)
        d = _segment_distance(start[..., :, None, :], end[..., :, None, :],
                              o_start[..., None, :, :], o_end[..., None, :, :])
        hit |= (d < reach).any(axis=(-1, -2, -3))
    w = np.linspace(0, 1, 5)[:, None]                                      # coxa is jointed to the body: exempt
    seg_pts = start[..., 1:, None, :] + w * (end - start)[..., 1:, None, :]
    hit |= (np.abs(seg_pts) < BODY_HALF + SEG_RADIUS[1]).all(-1).any(axis=(-1, -2, -3))
    return hit.mean(axis=-1)


def plot_pose(ax, angles, title=None):
    """One pose on a matplotlib 3D axis: body box outline, coxa black, thigh blue, tibia red."""
    pts = forward_kinematics(angles)
    ax.cla()
    x, y = BODY_HALF[:2]
    ax.plot([x, -x, -x, x, x], [y, y, -y, -y, y], [0] * 5, "k-", lw=3)
    ax.plot([x + 0.1], [0], [0], "r^", ms=10)                                # head marker
    for leg in range(8):
        for seg, colour in enumerate("kbr"):
            ax.plot(*pts[leg, seg:seg + 2].T, colour + "-", lw=2)
        ax.plot(*pts[leg, 3], "ro", ms=4)
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-0.75, 0.75), xlabel="X", ylabel="Y", zlabel="Z")
    ax.set_box_aspect((4, 4, 2))
    ax.view_init(30, -60)
    if title:
        ax.set_title(title)


if __name__ == "__main__":
    import math

    def rot(axis, a):
        x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
        c, s, C = math.cos(a), math.sin(a), 1 - math.cos(a)
        return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                         [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                         [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])

    rng = np.random.default_rng(0)
    ang = rng.uniform(MOTOR_LO, MOTOR_HI)
    fk = forward_kinematics(ang)
    for leg in range(8):                                   # proto: HingeJoint axis, endPoint rotation, offsets
        s = SIDE[leg]
        h, t, k = ang[3 * leg:3 * leg + 3]
        rc = rot([0, -0.707108 * s, 0.707105], h) @ rot([1, 0, 0], s * 0.7854)
        rt = rc @ rot([1, 0, 0], t)
        p1 = HIP_POS[leg] + rc @ [0, 0.5 * s, 0]
        p2 = p1 + rt @ [0, 0.6 * s, 0]
        p3 = p2 + rt @ rot([1, 0, 0], k) @ [0, 0, -0.7]
        assert np.allclose(fk[leg, 1:], [p1, p2, p3], atol=1e-4), leg

    stand = np.tile([0.0, -1.07, 0.58], 8) * np.repeat(SIDE, 3)          # stock standing pose
    feet_z = forward_kinematics(stand)[:, 3, 2]
    assert np.ptp(feet_z) < 1e-9 and feet_z[0] < -BODY_HALF[2]
    assert collision_fraction(stand[None])[()] == 0
    crossed = stand.copy()
    crossed[0], crossed[3] = 1.2, -1.2                     # L1 swung back into L2 swung forward
    assert collision_fraction(crossed[None])[()] == 1
    print(f"body self-check passed (standing: body {-feet_z[0]:.3f} m above the feet)")
