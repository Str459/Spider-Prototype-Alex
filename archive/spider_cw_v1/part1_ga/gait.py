"""Chromosome encoding: 27 real genes -> a 300-frame gait (300 x 24 joint angles).

Gene layout (legs L1,L2,L3,L4 | R4,R3,R2,R1 as in the MATLAB pose; joints coxa yaw, femur, tibia):
    chrom[0:12]   base pose of the left legs L1..L4 (12 joint angles). The right legs are its
                  mirror image (R_k = L_k with the coxa yaw negated), so these 12 genes give the
                  full 1x24 base pose: pose(chrom) = [left, mirror(left)].
    chrom[12:24]  swing amplitude of the same 12 joints (rad), mirrored the same way.
    chrom[24]     fb     - phase lag between neighbouring legs on one side, front to back (rad).
    chrom[25]     femur  - phase of every femur relative to its own leg's coxa (rad).
    chrom[26]     tibia  - phase of every tibia relative to its own leg's coxa (rad).

    leg_phase  = LR * is_right + fb * position    (position L1..L4 = 0..3, R4..R1 = 3..0)
    LR = pi is fixed: the two sides always alternate, as in a real spider (see docs/encoding.txt).
    angle_j(t) = offset_j + amp_j * sin(2*pi*t/FRAMES + leg_phase + joint_phase_j)
    joint_phase = 0 (coxa) | femur | tibia;  right-side coxa offsets AND amplitudes are negated.

Why: see docs/encoding.txt. In short, one sine per joint keeps the gait smooth and periodic;
coupling the phases as a metachronal wave (4 genes instead of 24 free phases) makes every
chromosome a coordinated gait; and bilateral symmetry halves the pose genes, so the four feet
of a support group start at matching heights instead of the GA having to line up 8 legs by luck.
"""
import numpy as np

FRAMES = 300
N_JOINTS = 24
N_POSE = 12                          # left-side joints; the right side mirrors them
N_PHASES = 3
LR = np.pi                           # right side half a cycle behind the left (was a gene; see docs)
N_GENES = 2 * N_POSE + N_PHASES

# Per joint-type limits (coxa yaw, femur pitch, tibia pitch). Femur pitch is kept inside
# (-120, +60) deg because there the femur would point straight up/down and the MATLAB
# kinematics' tibia rotation axis (cross(femur_dir, z)) becomes undefined.
JOINT_LO = np.tile([-0.6, -1.5, -2.2], 8)      # per joint of the full 24-angle pose
JOINT_HI = np.tile([0.6, 0.8, 0.5], 8)
_AMP_HI = np.tile([0.5, 0.4, 0.4], 4)

LOW = np.concatenate([JOINT_LO[:N_POSE], np.zeros(N_POSE), np.zeros(N_PHASES)])
HIGH = np.concatenate([JOINT_HI[:N_POSE], _AMP_HI, np.full(N_PHASES, 2 * np.pi)])
WRAP = np.arange(N_GENES) >= 2 * N_POSE      # phase genes are angles: wrap instead of clipping

_T = 2 * np.pi * np.arange(FRAMES) / FRAMES
# full joint j (0..23) <- left-side gene: R4,R3,R2,R1 copy L4,L3,L2,L1
_SOURCE = np.concatenate([np.arange(12), np.arange(12).reshape(4, 3)[::-1].ravel()])
_RIGHT = np.repeat([0, 0, 0, 0, 1, 1, 1, 1], 3)
_MIRROR = np.where((_RIGHT == 1) & (np.arange(N_JOINTS) % 3 == 0), -1.0, 1.0)   # negate right coxa yaw
_POSITION = np.repeat([0, 1, 2, 3, 3, 2, 1, 0], 3)
_JOINT = np.tile([0, 1, 2], 8)                 # 0 coxa, 1 femur, 2 tibia


def base_pose(chrom):
    """chrom (..., 27) -> the 1x24 base pose (..., 24) in MATLAB order."""
    return _MIRROR * np.asarray(chrom, dtype=float)[..., _SOURCE]


def phases(chrom):
    """chrom (..., 27) -> per-joint phase (..., 24) implied by the 3 wave genes."""
    c = np.asarray(chrom, dtype=float)
    fb = c[..., 24:25]
    rel = np.concatenate([np.zeros_like(fb), c[..., 25:27]], axis=-1)[..., _JOINT]
    return LR * _RIGHT + fb * _POSITION + rel


def decode(chrom):
    """chrom (..., 27) -> frames (..., 300, 24). Works on one chromosome or a whole population."""
    c = np.asarray(chrom, dtype=float)
    amp = _MIRROR * c[..., N_POSE + _SOURCE]
    return base_pose(c)[..., None, :] + amp[..., None, :] * np.sin(_T[:, None] + phases(c)[..., None, :])


def random_population(n, rng):
    """n chromosomes drawn uniformly inside the gene bounds."""
    return rng.uniform(LOW, HIGH, size=(n, N_GENES))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pop = random_population(4, rng)
    g = decode(pop)
    assert g.shape == (4, FRAMES, N_JOINTS)
    assert np.allclose(decode(pop[0]), g[0])
    c = pop[0].copy()
    # mirror: R1 (joints 21..23) is L1 (0..2) with the coxa yaw negated
    assert np.allclose(base_pose(c)[21:], c[:3] * [-1, 1, 1]) and np.allclose(base_pose(c)[:12], c[:12])
    # amp = 0 -> every frame equals the base pose
    c[12:24] = 0
    assert np.allclose(decode(c), base_pose(c))
    # fb = pi (with LR = pi): L1 and R1 femurs in antiphase, L1 and L3 in phase (alternating tetrapod)
    c[:12] = 0; c[12:24] = 0.3; c[24:] = [np.pi, 0, 0]
    f = decode(c)
    assert np.allclose(f[:, 1], -f[:, 22]) and np.allclose(f[:, 1], f[:, 7])
    print("gait self-check passed")
