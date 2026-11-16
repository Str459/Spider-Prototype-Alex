"""Fitness function for the spider gait GA (Part 1).

The spec asks for plausible, stable, goal-directed poses but not full physics, so the score is
built purely from the kinematics: we decode a chromosome into 300 poses, run forward kinematics
and judge the foot trajectories against a simple ground/stance model.

Ground/stance model (shared by every term):
    * The body frame is fixed (body centre at the origin, z = 0). Per frame the ground is the
      plane through the LOWEST foot, so the body's height above ground is -min(foot z).
    * A foot is in STANCE when it is within STANCE_TOL of that ground, otherwise it is SWINGING.
    * Grounded feet are assumed not to slip, so when they sweep backwards in the body frame the
      body is carried forwards by the same amount - but only while the body is statically stable
      (>= 4 feet down and the body centre inside their support polygon). An unstable body would
      fall over, so a step that is not stable at both ends moves it nowhere.

Every term is one small function returning an (n,) array where higher is better; the score is
sum(WEIGHTS[k] * term_k). The components dict returns the unweighted terms so plots can show
which behaviour the GA is improving.
"""
import numpy as np

from spider_model import forward_kinematics, BODY_A, BODY_B, BASE_ANGLES, BASE_POS, SEGMENTS, COXA_ELEVATION
from part1_ga.gait import decode, FRAMES, JOINT_LO, JOINT_HI, LOW, HIGH, N_GENES

STANCE_TOL = 0.05   # model units (tibia = 1.0): a foot this close to the lowest foot is on the ground
MIN_HEIGHT = 0.5    # body must stay at least this far above the ground (about half a tibia)
LIMIT_MARGIN = 0.1  # rad: joints closer than this to their joint limits start to be penalised
FOOT_GAP = 0.3      # neighbouring feet closer than this (in XY) are treated as colliding legs
MARGIN_CAP = 0.2    # stability margin beyond this earns nothing: stability is a requirement, not a prize
UNSTABLE_PENALTY = 5.0  # per fraction of frames that are not statically stable (a constraint penalty)
LIFT_CAP = 0.15     # a foot raised this high above the ground has clearly stepped; more earns nothing
IDLE_PENALTY = 4.0  # per fraction of legs that never touch down (x weight 2 = -1 per idle leg)

# Unit directions for the support-function margin (see stability). 32 directions bound the
# discretisation error by r*(1-cos(pi/32)) ~ 0.5% of the polygon size - far below the cap.
_DIRS = np.stack([np.cos(np.linspace(0, 2 * np.pi, 32, endpoint=False)),
                  np.sin(np.linspace(0, 2 * np.pi, 32, endpoint=False))], axis=1)

# Most terms are already in body-length units per cycle, so weights start at 1. Progress is 3 so
# that walking (which inevitably skates a little with one sine per joint, slip ~ -1 per unit of
# progress) clearly beats standing still; clearance is 2 because its violations are small numbers
# but physically serious. Lift is worth at most 2 x 0.15 = 0.3, small next to progress: it only
# has to make "every leg steps" better than standing still. Tuning numbers are in docs/fitness.txt.
WEIGHTS = {
    "progress": 3.0,
    "stability": 1.0,
    "clearance": 2.0,
    "slip": 1.0,
    "coordination": 1.0,
    "smoothness": 0.2,
    "lift": 2.0,
}


# ----------------------------------------------------------------------------- shared geometry
def _kinematics(angles):
    """angles (...,24) -> [knee1, knee2, foot] positions (...,8,3,3); same result as spider_model.

    spider_model.forward_kinematics uses general Rodrigues rotations and takes ~0.4 s for a
    population of 100, too slow for a GA. In that model every femur/tibia rotation axis is
    cross(segment, z), i.e. horizontal and perpendicular to the leg's yaw, so each leg stays in a
    vertical plane and a rotation just adds to the segment's elevation angle. The sign flips when
    the segment points past vertical (the axis cross(d, z) flips), hence sign(cos e). The
    self-check asserts equality with forward_kinematics; this version is ~4x faster.
    """
    a = np.asarray(angles, dtype=float).reshape(*np.shape(angles)[:-1], 8, 3)
    yaw = BASE_ANGLES + a[..., 0]
    e2 = COXA_ELEVATION + a[..., 1]                    # coxa is at +30 deg, so cos > 0: no flip
    e3 = e2 + np.sign(np.cos(e2)) * a[..., 2]
    e = np.stack([np.full_like(yaw, COXA_ELEVATION), e2, e3], axis=-1)
    r = np.cumsum(SEGMENTS * np.cos(e), axis=-1)      # horizontal reach of each joint from base
    z = np.cumsum(SEGMENTS * np.sin(e), axis=-1)
    return np.stack([BASE_POS[:, 0, None] + r * np.cos(yaw)[..., None],
                     BASE_POS[:, 1, None] + r * np.sin(yaw)[..., None], z], axis=-1)


def _stance(foot_z):
    """foot_z (n,T,8) -> (stance mask (n,T,8), ground height (n,T))."""
    ground = foot_z.min(axis=-1)
    return foot_z - ground[..., None] < STANCE_TOL, ground


def _stance_steps(foot_xy, stance):
    """Per-frame displacement of feet grounded at both ends of the step (cyclic: the gait loops).

    Returns (dxy (n,T,8,2), mask (n,T,8), count (n,T)). Only feet that stay down for the whole
    step can push the body; a foot touching down or lifting off mid-step contributes nothing.
    """
    dxy = np.roll(foot_xy, -1, axis=1) - foot_xy
    mask = stance & np.roll(stance, -1, axis=1)
    return dxy, mask, mask.sum(axis=-1)


def _margin(foot_xy, stance):
    """Signed static-stability margin per frame (n,T): distance of the body centre (origin) inside
    the convex support polygon of the stance feet, negative when outside.

    It equals min over unit directions u of max over stance feet of u.x (the hull's support
    function), so no hull has to be built and it vectorises over population, frames and legs.
    """
    proj = np.where(stance[..., None], foot_xy @ _DIRS.T, -np.inf)   # (n,T,8,K)
    return proj.max(axis=2).min(axis=-1)


def _stable(margin, stance):
    """Statically stable frames (n,T): >= 4 feet down and the body centre inside their polygon."""
    return (stance.sum(axis=-1) >= 4) & (margin > 0)


def _body_velocity(dxy, mask, count, stable):
    """Body velocity per frame (n,T,2) = minus the mean displacement of the grounded feet.

    A step only moves the body if at least 4 feet stay down through it and the body is statically
    stable at both ends of it; otherwise the body would tip over rather than be carried forwards.
    History: the first version only asked for >= 3 feet, because the hill-climb self-check
    "walked" by paddling one or two feet. GA runs then found fast walkers that leaned on 3 feet
    for part of the cycle (stability -0.3..-0.6); gating on stability removes that reward.
    """
    v = -(dxy * mask[..., None]).sum(axis=2) / np.maximum(count, 1)[..., None]
    ok = (count >= 4) & stable & np.roll(stable, -1, axis=1)
    return v * ok[..., None]


# ----------------------------------------------------------------------------- fitness terms
def progress(vel, target_speed=None):
    """Distance walked forwards (+X) over one cycle - the goal of the whole exercise.

    With target_speed (distance per cycle) we instead reward being close to that speed, so Part 2
    can generate gaits at several chosen speeds rather than always the fastest one.
    """
    dist = vel[..., 0].sum(axis=1)
    return dist if target_speed is None else -np.abs(dist - target_speed)


def stability(margin, stable):
    """Static stability, treated as a constraint rather than a prize.

    Without dynamics a gait can only be trusted if it is statically stable at every frame (see
    _margin/_stable). The reward part is the average of the mean and the worst-frame margin (a
    gait is only as safe as its worst moment), capped at MARGIN_CAP = 0.2: with a cap of 1.0,
    standing with all 8 feet spread out earned as much as a good walk, so the GA stood still.
    The penalty part is UNSTABLE_PENALTY x the fraction of frames that are not stable, large
    enough that no amount of extra speed pays for leaning on fewer than 4 feet.
    """
    m = np.minimum(margin, MARGIN_CAP)
    return 0.5 * (m.mean(axis=1) + m.min(axis=1)) - UNSTABLE_PENALTY * (~stable).mean(axis=1)


def clearance(joints, ground):
    """Body carried at a sensible, steady height, with legs clear of the floor and of each other.

    Penalises (a) the body sagging below MIN_HEIGHT (dragging the belly), (b) the body height
    changing over the cycle (bobbing wastes energy and is not what a real walker does),
    (c) knees dipping below the ground plane, which would mean the leg passes through the floor,
    and (d) neighbouring feet coming within FOOT_GAP of each other. (d) was added after the
    hill-climb self-check: refining the hand gait, it swung neighbouring legs to within 0.04 of
    each other, which on a real robot is a leg collision.
    shortcut: (d) checks feet only, not whole segments; upgrade to segment-segment distance if
    evolved gaits are seen crossing legs with their feet apart.
    """
    height = -ground
    sag = np.maximum(MIN_HEIGHT - height, 0).mean(axis=1)
    bob = height.std(axis=1)
    knees_z = joints[..., :2, 2]                                       # (n,T,8,2)
    through_floor = np.maximum(ground[..., None, None] - knees_z, 0).max(axis=(-1, -2)).mean(axis=1)
    feet_xy = joints[..., 2, :2]
    gap = np.linalg.norm(feet_xy - np.roll(feet_xy, -1, axis=-2), axis=-1)  # ring neighbours
    collide = np.maximum(FOOT_GAP - gap, 0).sum(axis=-1).max(axis=1)    # one collision is enough
    return -(sag + bob + through_floor + collide)


def slip(dxy, mask, count, vel):
    """Grounded feet must move together and the walk must not drift sideways.

    If the body is rigid and feet do not skate, every stance foot has the same body-frame
    velocity. The RMS spread of stance-foot displacements, summed over the cycle, is the total
    skating distance; we add the sideways (Y) distance walked, which the goal does not want.
    Both are distances per cycle, so they are directly comparable to progress.
    """
    dev = dxy + vel[:, :, None, :]                                     # foot step minus mean step
    rms = np.sqrt((np.sum(dev ** 2, axis=-1) * mask).sum(axis=-1) / np.maximum(count, 1))
    return -(rms.sum(axis=1) + np.abs(vel[..., 1]).sum(axis=1))


def coordination(stance):
    """Neighbouring legs should not be in the air together (alternating, tetrapod-like gait).

    Leg order L1,L2,L3,L4,R4,R3,R2,R1 is a ring around the body, so legs i and i+1 (mod 8) are
    exactly the neighbour pairs: same-side front/back neighbours plus L1-R1 and L4-R4. Lifting two
    neighbours at once leaves a large gap in the support ring. We penalise the fraction of
    (pair, frame) samples where both are swinging. Standing still scores 0 (best): rewarding
    motion is left to progress, so this term cannot be farmed by merely wiggling legs.
    """
    swing = ~stance
    return -(swing & np.roll(swing, -1, axis=-1)).mean(axis=(1, 2))


def lift(foot_z, ground, stance):
    """Every leg must step: each foot's peak height above the ground over the cycle, capped at
    LIFT_CAP, averaged over the 8 legs - counted only for feet that also touch down (a step is
    up AND down; a leg held in the air all cycle earns nothing).

    A walk is defined by legs taking steps; a leg that never leaves the ground can only be dragged.
    Without this term the GA's first move was to make the body stable by shrinking every
    amplitude, collapsing to standing still (amplitude 0 is a sticky bound), from which walking
    needs several genes to change together. Rewarding stable stepping keeps the legs moving while
    the GA finds the coordination; the cap stops it paying for ever-higher steps.
    """
    peak = np.minimum((foot_z - ground[..., None]).max(axis=1), LIFT_CAP)
    touches = stance.any(axis=1)
    # a leg held in the air all cycle (seen: hind legs raised, walking on 6) costs IDLE_PENALTY
    return (peak * touches).mean(axis=-1) - IDLE_PENALTY * (~touches).mean(axis=-1)


def smoothness(angles):
    """Avoid violent joint motion and joints pressed against their limits.

    The sine encoding is already smooth, so the acceleration part mostly discourages needlessly
    large, fast swings. It is scaled by (FRAMES/2pi)^2 so a sine of amplitude A scores about A.
    Joint limits are the base-pose gene bounds (gait.JOINT_LO/HI); working near them is
    unrealistic for a real servo and near the femur limits the kinematics become degenerate.
    """
    acc = np.roll(angles, -1, axis=1) - 2 * angles + np.roll(angles, 1, axis=1)
    acc_term = np.abs(acc).mean(axis=(1, 2)) * (FRAMES / (2 * np.pi)) ** 2
    lo, hi = JOINT_LO + LIMIT_MARGIN, JOINT_HI - LIMIT_MARGIN
    over = (np.maximum(lo - angles, 0) + np.maximum(angles - hi, 0)).mean(axis=(1, 2))
    return -(acc_term + 10.0 * over)


# ----------------------------------------------------------------------------- public API
def evaluate_population(pop, target_speed=None):
    """pop (n,N_GENES) -> (scores (n,), components dict[str, (n,)]). Higher is better. Vectorised."""
    angles = decode(np.atleast_2d(pop))                  # (n,T,24)
    joints = _kinematics(angles)                         # (n,T,8,3,3) knee1, knee2, foot
    foot = joints[..., 2, :]
    stance, ground = _stance(foot[..., 2])
    foot_xy = foot[..., :2]
    margin = _margin(foot_xy, stance)
    stable = _stable(margin, stance)
    dxy, mask, count = _stance_steps(foot_xy, stance)
    vel = _body_velocity(dxy, mask, count, stable)

    comps = {
        "progress": progress(vel, target_speed),
        "stability": stability(margin, stable),
        "clearance": clearance(joints, ground),
        "slip": slip(dxy, mask, count, vel),
        "coordination": coordination(stance),
        "smoothness": smoothness(angles),
        "lift": lift(foot[..., 2], ground, stance),
    }
    comps = {k: np.nan_to_num(v, nan=-1e3, posinf=1e3, neginf=-1e3) for k, v in comps.items()}
    scores = sum(WEIGHTS[k] * comps[k] for k in WEIGHTS)
    return scores, comps


def evaluate(chrom, target_speed=None):
    """single chromosome (N_GENES,) -> (score float, components dict[str, float])."""
    s, c = evaluate_population(np.asarray(chrom)[None], target_speed)
    return float(s[0]), {k: float(v[0]) for k, v in c.items()}


def distance(pop):
    """Forward distance per cycle actually walked (useful when evaluating with a target_speed)."""
    angles = decode(np.atleast_2d(pop))
    foot = _kinematics(angles)[..., 2, :]
    stance, _ = _stance(foot[..., 2])
    stable = _stable(_margin(foot[..., :2], stance), stance)
    return _body_velocity(*_stance_steps(foot[..., :2], stance), stable)[..., 0].sum(axis=1)


# ----------------------------------------------------------------------------- self-check
def hand_gait(coxa_amp=0.3, femur_amp=0.15, femur=-0.8, tibia=-1.2):
    """A hand-built alternating tetrapod: groups {L1,L3,R2,R4} and {L2,L4,R1,R3} in antiphase.

    In the wave encoding that is gait.LR = pi (right side half a cycle behind) and fb = pi
    (neighbouring legs alternate). Coxa yaw = a*sin(t + leg phase) sweeps a foot backwards while
    sin is rising (gait.decode mirrors the right side). The femur phase 1.5*pi puts each femur
    lowest in the middle of its backward stroke and highest during the forward swing, so
    whichever group is lower is the one pushing.
    """
    return np.concatenate([np.tile([0.0, femur, tibia], 4), np.tile([coxa_amp, femur_amp, 0.0], 4),
                           [np.pi, 1.5 * np.pi, 0.0]])


def _hill_climb(start, seconds, rng):
    """(1+100) hill climb: a cheap adversary used only to look for fitness exploits."""
    import time
    best, best_s = start, evaluate(start)[0]
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        step = rng.normal(0, 0.05, (100, N_GENES)) * (HIGH - LOW) * (rng.random((100, N_GENES)) < 0.2)
        kids = np.clip(best + step, LOW, HIGH)
        s, _ = evaluate_population(kids)
        if s.max() > best_s:
            best, best_s = kids[np.argmax(s)], s.max()
    return best


if __name__ == "__main__":
    import time
    from part1_ga.gait import random_population

    np.set_printoptions(precision=2, suppress=True)
    rng = np.random.default_rng(0)
    fmt = lambda c: ", ".join(f"{k}={v:+.3f}" for k, v in c.items())

    rand = random_population(2000, rng)
    assert np.allclose(_kinematics(decode(rand[:20])), forward_kinematics(decode(rand[:20]))[..., 1:, :])
    rs, rc = evaluate_population(rand)
    assert all(np.isfinite(v).all() for v in rc.values()) and np.isfinite(rs).all()
    print("random  median score %.3f;  median terms: %s"
          % (np.median(rs), fmt({k: np.median(v) for k, v in rc.items()})))

    hs, hc = evaluate(hand_gait())
    print("hand    score %.3f;  %s" % (hs, fmt(hc)))
    assert hc["progress"] > 0.5, "hand tetrapod should walk forwards"
    assert hs > np.median(rs) + 2.0 and hs > np.percentile(rs, 99), "hand gait must clearly beat random"

    ss, sc = evaluate(hand_gait(coxa_amp=0, femur_amp=0))
    print("stand   score %.3f;  %s" % (ss, fmt(sc)))
    assert abs(sc["progress"]) < 1e-9
    assert abs(sc["stability"] - MARGIN_CAP) < 1e-9 and sc["clearance"] > -1e-9 and sc["coordination"] == 0
    assert hs > ss + 1.0, "walking must clearly beat standing still"

    # Target-speed mode: a gait scores best at the speed it actually walks.
    d = distance(hand_gait())[0]
    assert evaluate(hand_gait(), d)[1]["progress"] > -1e-9 > evaluate(hand_gait(), d + 1)[1]["progress"]

    evaluate_population(rand[:100])                     # warm-up
    t0 = time.perf_counter()
    for _ in range(10):
        evaluate_population(rand[:100])
    dt = (time.perf_counter() - t0) / 10
    print(f"evaluate_population(100): {dt * 1000:.0f} ms")
    assert dt < 1.0

    # Adversarial check: does a blind optimiser find something that scores well but is not a walk?
    for name, start in [("random", rand[np.argmax(rs)]), ("hand", hand_gait())]:
        best = _hill_climb(start, 15, rng)
        bs, bc = evaluate(best)
        joints = _kinematics(decode(best))
        st, ground = _stance(joints[..., 2, 2])
        feet_xy = joints[..., 2, :2]
        gap = np.linalg.norm(feet_xy - np.roll(feet_xy, -1, axis=-2), axis=-1).min()
        print(f"climbed from {name}: score {bs:.3f};  {fmt(bc)}")
        print(f"   feet down min/mean {st.sum(1).min()}/{st.sum(1).mean():.1f}, stance share per leg "
              f"{st.mean(0)}, body height {-ground.mean():.2f}, closest neighbour feet {gap:.2f}")
        assert st.sum(1).min() >= 3 and gap > 0.1 and bc["clearance"] > -0.2, "degenerate optimum"
    print("fitness self-check passed")
