"""Python port of the module's MATLAB spider model (plot_spider_pose.m + forward_leg_kinematics2.m).

Pose vector: 1x24 joint angles in radians, legs ordered [L1, L2, L3, L4, R4, R3, R2, R1],
three joints per leg (coxa yaw, femur pitch, tibia pitch) - exactly the MATLAB input format.

forward_kinematics() is vectorised: it accepts any array whose last axis is 24
(one pose, a 300-frame gait, or a whole population of gaits) so the GA can score fast.
"""
import numpy as np

N_LEGS = 8
LEG_LABELS = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
SEGMENTS = np.array([1.2, 0.7, 1.0])          # coxa, femur, tibia
BODY_A, BODY_B = 1.5, 1.0                     # body ellipse semi-axes (front of spider is +X)
BASE_ANGLES = np.deg2rad([45, 75, 105, 135, -135, -105, -75, -45])
BASE_POS = np.stack([BODY_A * np.cos(BASE_ANGLES), BODY_B * np.sin(BASE_ANGLES),
                     np.zeros(N_LEGS)], axis=1)                  # (8, 3) leg roots on the body
COXA_ELEVATION = np.deg2rad(30)
Z = np.array([0.0, 0.0, 1.0])


def _rotate(v, axis, angle):
    """Rodrigues rotation of vectors v about (unit-normalised) axis by angle; broadcasts over leading dims."""
    axis = axis / np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
    c, s = np.cos(angle)[..., None], np.sin(angle)[..., None]
    k_dot_v = np.sum(axis * v, axis=-1, keepdims=True)
    return v * c + np.cross(axis, v) * s + axis * k_dot_v * (1 - c)


def forward_kinematics(angles):
    """angles (..., 24) -> joint positions (..., 8 legs, 4 points [base, knee1, knee2, foot], 3 xyz)."""
    a = np.asarray(angles, dtype=float).reshape(*np.shape(angles)[:-1], N_LEGS, 3)
    th1, th2, th3 = a[..., 0], a[..., 1], a[..., 2]

    yaw = BASE_ANGLES + th1
    horiz = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=-1)
    coxa_dir = _rotate(horiz, np.cross(horiz, Z), np.full_like(yaw, COXA_ELEVATION))
    femur_dir = _rotate(coxa_dir, np.cross(coxa_dir, Z), th2)
    tibia_dir = _rotate(femur_dir, np.cross(femur_dir, Z), th3)

    j1 = np.broadcast_to(BASE_POS, coxa_dir.shape)
    j2 = j1 + SEGMENTS[0] * coxa_dir
    j3 = j2 + SEGMENTS[1] * femur_dir
    j4 = j3 + SEGMENTS[2] * tibia_dir
    return np.stack([j1, j2, j3, j4], axis=-2)


def feet(angles):
    """angles (..., 24) -> foot positions (..., 8, 3)."""
    return forward_kinematics(angles)[..., 3, :]


def plot_pose(ax, angles, title=None):
    """Draw one pose on a matplotlib 3D axis using the MATLAB figure's colours (coxa black, femur blue, tibia red)."""
    pts = forward_kinematics(angles)
    ax.cla()
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(BODY_A * np.cos(t), BODY_B * np.sin(t), 0 * t, "k-", lw=3)
    ax.plot([BODY_A + 0.2], [0], [0], "r^", ms=10)                # head marker
    for leg in range(N_LEGS):
        p = pts[leg]
        for seg, colour in enumerate("kbr"):
            ax.plot(*p[seg:seg + 2].T, colour + "-", lw=2)
        ax.plot(*p[3], "ro", ms=4)
        lbl = BASE_POS[leg] + 0.2 * np.array([np.cos(BASE_ANGLES[leg]), np.sin(BASE_ANGLES[leg]), 0])
        ax.text(*lbl, LEG_LABELS[leg], fontsize=9, fontweight="bold")
    ax.set(xlim=(-4, 4), ylim=(-4, 4), zlim=(-2, 2), xlabel="X", ylabel="Y", zlabel="Z")
    ax.set_box_aspect((8, 8, 4))
    ax.view_init(45, -45)                                       # = MATLAB view(45, 45); mpl azimuth is offset by -90
    if title:
        ax.set_title(title)


if __name__ == "__main__":
    # Self-check against hand-derived geometry.
    zero = np.zeros(24)
    p = forward_kinematics(zero)
    # theta2 = theta3 = 0 -> the leg is one straight line along the 30-degree-elevated coxa direction
    d = np.array([np.cos(COXA_ELEVATION) * np.cos(BASE_ANGLES[0]),
                  np.cos(COXA_ELEVATION) * np.sin(BASE_ANGLES[0]), np.sin(COXA_ELEVATION)])
    assert np.allclose(p[0, 3], BASE_POS[0] + SEGMENTS.sum() * d)
    # Segment lengths are preserved for random poses and batch shapes work.
    rnd = np.random.default_rng(0).uniform(-1, 1, (5, 300, 24))
    q = forward_kinematics(rnd)
    assert q.shape == (5, 300, 8, 4, 3)
    assert np.allclose(np.linalg.norm(np.diff(q, axis=-2), axis=-1), SEGMENTS)
    # Positive femur pitch lifts the knee (same sign convention as the MATLAB code).
    up = zero.copy(); up[1] = 0.3
    assert forward_kinematics(up)[0, 2, 2] > p[0, 2, 2]
    print("spider_model self-check passed")
