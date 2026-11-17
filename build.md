# Build guide: every decision, and why

This explains how the project was built and why each choice was made, in the order you would build it again. Every number here was measured. Where a number comes from Webots, the robot really walked it, headless, with self-collision on.

---

## 0. The task and the two versions we compared

**The task** (M33174 coursework):
- **Part 1:** a GA that finds the 24 joint angles of the 8-leg spider as a full gait of about 300 poses.
- **Part 2:** a neural network that outputs the 24 joint angles. Marks 1–7 need it written from scratch. PyTorch only counts for the comparison.

**Language decision.** We chose the GA in MATLAB and the NN in Python. MATLAB is where the lecturer's model lives and is good for matrix maths and plots. Python has numpy for a hand-written network and PyTorch for the comparison.

**Which spider.** The spec provides two:
- the lecturer's simplified MATLAB model (legs 1.2 / 0.7 / 1.0, no physics);
- the Webots robot `SpiderOctopod` (legs 0.5 / 0.6 / 0.7, real physics).

The spec allows any language and provides Webots for testing. We target the **Webots robot**, so that every gait can be checked by actually walking it. A gait that only looks right on a model with no physics proves little. We learnt that the hard way: an earlier gait evolved on the lecturer's model was copied onto the robot and only dragged its feet.

### 0.1 The two approaches we had
| | A: our MATLAB GA (this folder) | B: teammate's Python GA |
|---|---|---|
| Scoring a gait | Kinematic rules of the real robot (reach, joint limits, motor speed, leg clearance, stability), then the winner is checked in Webots | Every candidate is walked in Webots physics and scored on what happened |
| Encoding | 8 numbers describing **where the feet go**; inverse kinematics gives the angles | 11–13 numbers describing **joint motions** directly |
| Time per GA run | ~25 s | hours (each candidate is a physics simulation) |
| Language | MATLAB (as decided) | Python |

### 0.2 Measured head to head
Same Webots world and the same instrumented player (`webots/controllers/gait_player`), 10 cycles each:

| gait | speed | drift per m | heading change | max tilt | bounce | planted-foot slide | self-contacts |
|---|---|---|---|---|---|---|---|
| **A: our fastest** | **0.345 m/s** | 5.9 cm | −3.8° (all during start-up) | 2.8° at start-up, **0.4° once walking** | 2 mm once walking | 1.0 cm/step | 0 |
| B: `ref10_2` (their less-sliding best) | 0.240 m/s | 19 cm | −6.7° | 6.0° | 35 mm | 1.1 cm/step | 0 |
| B: `pi0` (their best level walk) | 0.137 m/s in this player (0.256 in their own) | 39 cm | −24.5° | 11.5° | 25 mm | 4.1 cm/step | 0 |

**Caveat.** B's gaits were evolved from their own start pose, and our player stands the robot up its own way. That hurts `pi0` here (0.137 against 0.256 m/s in their setup). `ref10_2` walks the same speed in both.

**Decision: approach A for the GA, B's network core for Part 2.**
- **Faster and steadier.** A walks faster (0.345 against 0.256 m/s at B's own best) and keeps the body level (0.4° against 3–12°).
- **Quicker to run.** A run takes seconds rather than hours, so the GA can be tuned and experimented with properly. The spec asks us to compare GA components, which means many runs.
- **The GA is in MATLAB,** as decided.
- **What B does better:** it needs no model of the robot, because the fitness is real physics. Its network code (`nn.py`) is well tested. So we reuse B's `nn.py` (written by our teammate) as the from-scratch core for Part 2, and retrain it on A's gaits.

---

## 1. The robot (facts the code relies on)
Read from `webots/protos/SpiderOctopod.proto`:
- **Axes and legs:** body centre is the origin, +x forward, +y left, +z up. Legs are L1..L4 on the left (+y) and R1..R4 on the right. Hips are at x = ±0.4875 and ±0.1625, y = ±0.2.
- **Leg segments:** coxa 0.5 m tilted 45° up, thigh 0.6 m, tibia 0.7 m.
- **Joints:** the thigh and knee bend the leg inside a vertical plane. The **hip turns that plane about an axis tilted 45°**: (0, −s, 1)/√2, with s = +1 on the left and −1 on the right.
- **Joint order:** every CSV and every 1×24 vector uses L1 L2 L3 L4 R4 R3 R2 R1, each with (hip, thigh, knee). That is the spec's order.
- **Limits:** hip and thigh ±1.5708 rad, knee [−2, 1] rad, motors 2 rad/s and 50 Nm.
- **Body:** 43 kg, a 1.3 × 0.4 × 0.3 m box. Time step 32 ms.

---

## 2. Part 1: the GA (MATLAB, `part1_ga_matlab/`)

### 2.1 Encoding: 8 numbers, not 7200
A gait is 300 poses × 24 angles = 7200 numbers. We don't evolve them directly:
- **Too many.** A GA can't search 7200 dimensions in sensible time.
- **Not a walk.** A random mix of 7200 angles is never a walk, and nearly every random chromosome would be useless.

Instead the chromosome describes **how the robot walks**, and `decode_gait.m` builds the 300 × 24 angles from it. That is still "encoding the 24 joint angles": every angle in the gait is decided by the chromosome.

| gene | range | meaning | why this range |
|---|---|---|---|
| stride | 0.05–0.45 m | how far a planted foot moves back per step | below 0.05 is shuffling; above 0.45 feet leave their reach |
| stepHeight | 0.06–0.20 m | foot lift at the top of a swing | ≥ 6 cm so feet clear the floor even when the body bounces |
| duty | 0.52–0.75 | fraction of the cycle a foot is down | > 0.5 so both groups overlap at each change; ≤ 0.75 or the steps get too quick |
| period | 1.5–6 s | time for one cycle | shorter needs joint speeds above the motors' 2 rad/s |
| bodyHeight | 0.42–0.60 m | body centre above the floor | the GA always takes the lowest allowed; at 0.30 the body hugged the floor in Webots (looked like crawling), and 0.42 walked as well as 0.38 |
| reach | 0.65–1.00 m | sideways distance of the feet from the centre line | wider is more stable but needs more motor torque; with the feet far out the real robot sagged (motors are 50 Nm) |
| spread | 1.0–2.2 | foot placed at spread × its hip's x | fans the legs out so neighbours don't hit each other |
| xShift | −0.15–0.15 m | moves all footholds forward or back | lets the GA centre the support under the body |

**Real numbers, not binary:** every gene is a physical quantity, so a small change in a gene is a small change in the walk, which suits Gaussian mutation.

**Fixed structure, chosen by us** (the GA tunes the numbers):
- **Two support groups, A = L1 L3 R2 R4 and B = L2 L4 R1 R3, half a cycle apart.** Each group is 4 feet that surround the body (front-left, middle-left, middle-right, back-right), so 4 feet always hold it up. This is the classic tetrapod gait of 8-legged animals.
  - We kept timing out of the genes on purpose. An earlier version on the lecturer's model evolved the phases freely. It found fast but messy gaits, and those gaits didn't transfer to the robot.
- **Stance (foot down):** the foot moves in a straight line backwards at a constant speed, and every planted foot does the same. A rigid body on feet that all move the same way slides over none of them.
  - The old approach turned the hip to sweep the foot, which moves it on an arc. Eight feet on eight arcs can't all stay put, so they slid (3.6 cm per step measured, against 1 cm now).
- **Swing (foot up):**
  - **Lift.** The height follows (1 − cos 2πu)/2 × stepHeight, so vertical speed is zero at lift-off and touch-down, and the foot is set down gently.
  - **Forward path.** Forward position is a cubic (Hermite) curve from the back of the stroke to the front.
  - **Ground-speed matching.** The curve's end slopes are chosen so the foot leaves and meets the ground moving backwards at exactly the stance speed. It then lands with zero speed over the floor and doesn't skid. We tested this on the robot earlier: it was the change that cut sliding the most.
- **Inverse kinematics for every frame (`leg_ik.m`).** On this robot the hip axis is tilted, so turning the hip alone also raises or lowers the foot. Working in foot positions and solving for the angles keeps planted feet exactly on the floor.

### 2.2 Kinematics (`leg_fk.m`, `leg_ik.m`)
**Forward kinematics** (angles → points). In the leg's own plane (u = outwards, z = up), measured from the hip:
- the coxa end is at (0.5 cos45°, 0.5 sin45°);
- the thigh points at 45° + s·q_thigh;
- the tibia points at −45° + s·(q_thigh + q_knee).

These in-plane points are mapped to 3D as (0, s·u, z) and rotated about the hip axis by q_hip with Rodrigues' formula: v' = v cosθ + (a×v) sinθ + a(a·v)(1 − cosθ).

**Inverse kinematics** (foot target → angles), exact and in closed form:
1. **Hip angle:** choose the q_hip that turns the leg's plane through the target. Rotating the target back by q_hip must give x = 0, so q_hip = atan2(d_x·sign(c), |c|) with c = (−s·d_z − d_y)/√2.
2. **Thigh and knee:** in that plane, thigh plus tibia is a 2-link arm. Use the cosine rule for the knee bend and take the **knee-up** solution (like a real spider, and it keeps the knee inside its limits). An unreachable target is clamped, and the distance it is out of reach is returned so the fitness can punish it.

**Check:** `run_ga.m` pushes the IK angles back through the FK. Every foot lands within **7e-16 m** of its target, so the two functions agree exactly.

### 2.3 Fitness (`gait_fitness.m`): reward the goal, make everything else a rule
`score = 10 × speed − 50 × outOfReach − 20 × pastLimits − 5 × tooFast − 50 × legsTooClose − 20 × unstable`

| term | how it is measured | why |
|---|---|---|
| **speed** (the only reward) | stride / (duty × period) m/s: a planted foot moves back `stride` while it is down, and the body moves forward the same | the goal is to walk |
| outOfReach | mean distance (m) of foot targets outside a leg's reach | the robot can't do it |
| pastLimits | mean amount (rad) any joint goes beyond its limit minus 0.05 rad | the motors stop at the limits; the margin allows for tracking error |
| tooFast | how far the fastest joint speed goes above 1.6 rad/s | motors do 2 rad/s; 80% leaves room for physics |
| legsTooClose | for neighbouring legs on the same side: knee, mid-tibia and foot points closer than 0.15 m | self-collision is on in Webots, so legs that touch would jam |
| unstable | fraction of frames with fewer than 4 feet down, or with the body centre not inside the planted feet | it would tip over |

**Stability test:** look from the body centre at the planted feet. If two neighbouring feet (by angle) are 180° or more apart, the centre is outside their polygon and the robot tips over. We demand under 150° for a safety margin. It is cheap to compute and easy to explain.

**Why rules instead of rewards.** In the first version, stability earned points as well as distance. The GA found it easier to stand still in a wide, very stable pose than to walk. Now only speed earns points, and every rule only costs when it is broken. A gait that breaks no rule scores exactly 10 × its speed.

**Why the weights are large** (50, 20): a broken rule must always cost more than the speed it buys. 1 cm out of reach costs 0.5, the same as walking 0.05 m/s slower. The best gait breaks none (all terms 0.0000).

**Not modelled** (checked in Webots instead): motor torque, body bounce, and sliding caused by physics.

### 2.4 The GA loop (`run_ga.m`)
1. **Initialise:** 60 random chromosomes, each gene uniform inside its range.
2. **Evaluate** every chromosome.
3. **Repeat** for at most 200 generations:
   - **Sort** by fitness and record the best and mean (for the plot).
   - **Stop** if the best hasn't improved by 1e-4 for 40 generations (stall). It has converged, and more generations only cost time.
   - **Elitism:** copy the 2 best unchanged into the next generation, so the best gait can never be lost and the best fitness never drops.
   - **Fill the other 58** by selection → crossover → mutation.
   - **Evaluate** the 58 children (elites are not re-evaluated, since their score can't change).

**Selection: tournament, size 3** (`selectParent`). Pick 3 at random; the fittest is a parent.
- **Why:** it only compares scores, so negative fitness is fine, which happens early, when every gait breaks rules. The pressure is set by one number, the tournament size: 3 is mild and keeps variety. There's no need to scale fitness.
- **Roulette** (chance proportional to fitness, shifted to be positive) is implemented for comparison (§2.6).

**Crossover: two-point, rate 0.9.** Swap the block of genes between two random cut points.
- **Why:** genes that work together sit next to each other (stride, stepHeight, duty, period), so a block swap tends to pass a working combination on whole.
- **Uniform** (each gene swapped with 50% chance) is implemented for comparison.
- **Rate 0.9:** 10% of children are plain copies of a parent, plus mutation, which lets good parents survive almost unchanged.

**Mutation: Gaussian.** Each gene has a 10% chance (about 0.8 genes per child) to get a nudge with a standard deviation of 10% of its range. It is then clipped back inside its limits.
- **Why:** small steps fine-tune, the occasional large one escapes local optima, and scaling by the range treats metres and seconds the same.

**Every setting is a named variable at the top of `run_ga.m`.** Selection type, crossover type and random seed are arguments: `run_ga('roulette', 'uniform', 2)`.

### 2.5 Results (default run: tournament, two-point, seed 1)
- **GA:** converged at generation 125 (stall stop) in 26 s, with fitness 3.712 (planned speed 0.371 m/s) and no rule broken.
- **Genes:** stride 0.388 m, stepHeight 0.20, duty 0.52, period 2.01 s, bodyHeight 0.42, reach 1.00, spread 2.2, xShift ≈ 0.
- **The GA pushes genes to their limits:** the highest step, the lowest body and the widest spread. Those limits are what make the walk physically sensible, which is why their ranges were chosen carefully (§2.1).
- **In Webots** (10 cycles):
  - **speed:** 0.345 m/s, 7% below planned, because physics loses a little at every step;
  - **heading:** turns −3.8° during start-up, then walks straight;
  - **body:** height 0.41 m, tilt 0.4° and bounce 2 mm once walking;
  - **feet:** planted feet slide about 1 cm per step;
  - **self-collision:** 0 self-contacts.

### 2.6 Comparing GA operators
`run_ga(selection, crossover, seed)` for every combination, 3 seeds each. The table shows best fitness, with the generations run in brackets (200 = it never stalled):

| selection | crossover | seed 1 | seed 2 | seed 3 | mean fitness | mean generations |
|---|---|---|---|---|---|---|
| tournament | two-point | 3.712 (125) | 3.768 (109) | 3.385 (94) | 3.622 | 109 |
| tournament | uniform | 3.768 (132) | 3.621 (200) | 3.561 (76) | 3.650 | 136 |
| roulette | two-point | 3.386 (200) | 3.779 (177) | 3.784 (126) | 3.649 | 168 |
| roulette | uniform | 3.768 (144) | 3.747 (156) | 3.611 (200) | 3.709 | 167 |

What this shows:
- **The operators barely matter here.** The means differ by at most 0.09, while seeds of the same setting differ by up to 0.4. With 3 seeds we can't call any combination better.
  - The fitness has a ceiling of about 3.78 (stride, step height, body height and spread all at their limits). Several runs reach it, and others stop at a local optimum around 3.39.
  - With only 8 genes the search is easy enough that every reasonable operator gets there.
- **Tournament converges faster:** 109–136 generations against about 167 for roulette.
  - Tournament compares ranks, so the pressure stays the same all run.
  - Roulette's pressure fades once most of the population breaks no rule, because the scores (10 × speed) then sit close together and every gait gets nearly the same chance.
- **We kept tournament + two-point as the default.** It converges fastest, it doesn't depend on how the fitness is scaled, and it won clearly in our earlier 27-gene version on the lecturer's model (tournament 11.1 against roulette 7.6, two-point best of the crossovers). There, the larger search made the operators matter.
- **Seed matters more than operator.** To be sure of the best gait, run a few seeds and keep the best. A larger population or more mutation would also help escape the 3.39 optimum.

### 2.7 Visualisation (saved in `outputs/part1/<run>/`)
- **`fitness.png`:** best and mean fitness per generation.
  - The best only rises (elitism) and flattens when the GA converges.
  - The gap to the mean shows the population still has variety.
- **`feet.png`:**
  - **top:** each foot's height above the floor over one cycle (zero while down, humps for steps);
  - **bottom:** which feet are down (the two groups alternating, with the short overlap).
- **`pose.png` / `gait.gif`:** the robot drawn from `leg_fk.m`, with feet on the floor in red and the floor shaded.
- **The console:** prints progress every 10 generations, every fitness term, every gene and the IK check.
- **Webots:** `gait_player` prints per cycle the distance, drift, heading, height, bounce, tilt, lowest swing, sliding and self-contacts.

### 2.8 Making gaits at a chosen speed (training data for Part 2)
`run_ga('tournament', 'twopoint', 1, v)` evolves a gait that walks at v m/s. `make_speed_family.m` does it for 0.10, 0.15 … 0.35. In this mode:
- **Fitness:** `−50 × |speed − v| − 2 × change` plus the same rules. Here `change` is the mean gene difference from the fastest gait, divided by each gene's range.
- **Start:** the population starts around the fastest gait.
- **Fixed timing:** period and duty are fixed to the fastest gait's values.

**Why:** the network learns how a gait changes with speed.
- **Similar neighbours.** If the GA found unrelated gaits for neighbouring speeds, there would be nothing smooth to learn. So we ask for the smallest change that reaches the speed.
- **Same timing.** Every gait has the same timing, so "phase" means the same thing in all of them, and the feet that are down in each frame are identical.

**What the GA did:** it changed only the stride, which is the cheapest change, and kept every other gene. Neighbouring speeds differ by 0.8° per joint on average.

**Measured in Webots** (`webots/measure.py`, 5 cycles):

| gait | planned | walked | tilt | self-contacts |
|---|---|---|---|---|
| speed_0.10 | 0.10 | 0.088 m/s | 4.2° | 0 |
| speed_0.15 | 0.15 | 0.131 | 4.5° | 0 |
| speed_0.20 | 0.20 | 0.177 | 4.6° | 0 |
| speed_0.25 | 0.25 | 0.220 | 4.4° | 0 |
| speed_0.30 | 0.30 | 0.273 | 2.9° | 0 |
| speed_0.35 | 0.35 | 0.318 | 2.8° | 0 |
| fastest | 0.371 | 0.339 | 2.8° | 0 |

(Tilt here is the worst moment, which is during start-up.)

---

## 3. Webots (`webots/`)
- **`worlds/spider.wbt`:** flat floor, the robot with `selfCollision TRUE` and `supervisor TRUE` (needed to read the body's orientation and contact points), and controller `gait_player`.
- **`controllers/gait_player/gait_player.py`:** plays a gait folder (`joint_table.csv`, `stance.csv`, `stand.csv`, `meta.csv`).
  1. **Stand up.** The robot starts lying on its belly. It tucks its feet in close (pushing up with feet far out needs more than the motors' 50 Nm), pushes up, spreads to the standing pose, then blends into frame 1.
  2. **Walk.** It loops the 300 frames at the gait's period.
  3. **Measure everything, every cycle,** from the simulation: GPS and orientation, joint sensors through the FK for the feet, and the contact points for self-collision.
- **`measure.py`:** runs any number of gait folders headless and saves `webots_result.json` in each. It produces every Webots number in this guide.

---

## 4. Part 2: the neural network (Python, `part2_nn_python/`)

### 4.1 Task and input/output format
**Task:** the network makes the gait for any requested speed, which the GA can only do with a 25 s run per speed.

**Input:** 3 numbers per frame, `[sin φ, cos φ, speed]`.
- φ = 2π k / 300 is the position in the cycle. Two numbers instead of one so frame 299 sits next to frame 0: the gait is a loop, and a single number would put a jump there.
- Speed is the speed **measured in Webots** (§2.8), not the GA's target, so the network learns what the gait really does on the robot. It is normalised with the training speeds' mean and standard deviation, so all inputs have a similar size.

**Output:** the 24 motor angles, scaled per joint with the motor limits to `y = (angle − middle) / (1.1 × half range)`.
- With a **tanh** output layer the network can only produce angles inside the motor limits, so the outputs are valid by construction.
- The 10% margin means targets at a limit don't need tanh = ±1, which would need infinite weights.

**One gait = 300 samples.** A whole gait is the network evaluated at the 300 phases.

### 4.2 Data (`data.py`)
**Gaits:** the 7 GA gaits of §2.8, i.e. 2100 samples.

**Split by whole gaits, not random frames.** Frames of one gait are almost copies of their neighbours, so a random split would test memory, not whether the network can make a new speed.
- **Test:** `speed_0.20` (0.177 m/s), in the middle of the range, never used for training or for choosing anything.
- **Validation:** `speed_0.30` (0.273 m/s), used for early stopping and for picking settings.
- **Train:** the other 5 gaits (1500 samples).

**Self-check (`python data.py`):** shapes, all targets inside tanh's range, and scaling round-trips exactly.

### 4.3 The network (`nn.py`, numpy only, written by our teammate and reused)
- **Model:** `MLP([3, 64, 64, 24], hidden="tanh", output="tanh")`, a fully connected network with 2 hidden layers of 64 neurons.
- **Forward:** a_(l+1) = f(a_l W_l + b_l).
- **Loss:** MSE = mean((ŷ − y)²) over all 24 outputs and all samples, with dL/dŷ = 2(ŷ − y)/N.
  - **Why MSE:** this is regression of continuous angles, and MSE punishes big errors more than small ones.
- **Backpropagation:**
  - **Output layer:** δ_L = dL/dŷ ⊙ f′(ŷ).
  - **Moving back one layer:** δ_l = (δ_(l+1) W_(l+1)ᵀ) ⊙ f′(a_l).
  - **Gradients:** dL/dW_l = a_lᵀ δ and dL/db_l = Σ δ.
  - The derivatives use the stored outputs: tanh′ = 1 − y², sigmoid′ = y(1 − y), relu′ = [y > 0].
- **Initialisation:** Xavier (variance 2/(n_in + n_out)) for tanh and sigmoid, He (2/n_in) for ReLU, so signals neither vanish nor explode at the start.
- **Optimisers:**
  - **SGD with momentum:** v = 0.9v − lr·g, then p += v.
  - **Adam:** running means of g and g², bias-corrected; step = lr·m̂/(√v̂ + 1e-8).
- **Training:** mini-batches with reshuffling every epoch. **Early stopping:** stop after 300 epochs without a better validation loss, and keep the weights from the best epoch.
- **Proof that backprop is right:** `gradient_check` compares every backprop gradient with a central finite difference, (L(p + ε) − L(p − ε))/2ε. The largest relative difference on the final model is **9.7e-7**. `python nn.py` also checks every activation combination.
- **Saving:** `outputs/nn/speed_model.npz` holds the weights, the layer sizes, the activations and the speed normalisation, so `predict.py` needs nothing else.

### 4.4 Choosing the settings (`train.py`): one change at a time
Base: [64, 64], tanh, Adam, lr 1e-3, batch 64, up to 3000 epochs, patience 300. Each experiment changes one thing; the score is the best validation loss.

| experiment | values (validation loss / validation RMSE) | conclusion |
|---|---|---|
| Adam learning rate | 1e-4: 3.84e-6 / 0.19° · 3e-4: 3.04e-6 · **1e-3: 2.92e-6 / 0.17°** · 3e-3: 3.04e-6 · 1e-2: 4.96e-6 / 0.22° | flat between 3e-4 and 3e-3; 1e-4 is slow (2830 epochs), 1e-2 is noisy. 1e-3 is best |
| SGD + momentum learning rate | 1e-3: 8.94e-4 / 2.91° · 1e-2: 5.33e-5 · 3e-2: 2.28e-5 · 0.1: 1.34e-5 / 0.36° | still improving at 3000 epochs; even the best SGD is 4.6× worse than Adam. Adam's per-weight step sizes suit outputs on very different scales |
| hidden activation | **tanh: 2.92e-6** · sigmoid: 2.89e-6 · relu: 8.11e-5 / 0.88° | the angles are smooth functions of the phase; tanh and sigmoid are smooth, while ReLU is made of straight pieces and needs far more of them |
| size | 16-16: 3.40e-6 · 32-32: 3.99e-6 · 64-64: 2.92e-6 · 128-128: 3.11e-6 · 64: 3.05e-6 · 64-64-64: 2.90e-6 | all similar: the mapping is simple (§4.6). 64-64 is a safe middle |
| batch size | **16: 2.68e-6 / 0.16°** · 64: 2.92e-6 · 256: 3.97e-6 | smaller batches mean more updates per epoch and a bit of helpful noise |

**Chosen:** [64, 64], tanh, Adam, lr 1e-3, batch 16 (the lowest validation loss), stopped at epoch 1386. Plots: `loss_curves.png`, `experiments.png`, `learning_rate.png`.

### 4.5 Results
- **On the test speed (0.177 m/s, never seen):** RMSE **0.12°** over all 24 angles and 300 frames, with the largest error about 0.6° (`predicted_vs_target.png`, `poses.png`).
- **In Webots** (`python predict.py 0.177 0.15 0.30 --webots`):

| asked for | walked | error | tilt | self-contacts |
|---|---|---|---|---|
| 0.177 m/s (test speed) | 0.177 | −0.3% | 4.7° | 0 |
| 0.150 (new speed) | 0.147 | −2.3% | 4.5° | 0 |
| 0.300 (new speed) | 0.301 | +0.2% | 2.2° | 0 |

  The network's gaits walk at the speed they were asked for. They don't just look right, they work on the robot.

### 4.6 Honest limits of Part 2
- **Simple data.** The GA family only varies the stride (§2.8), so the angles change smoothly and nearly linearly with speed. A straight blend of the two nearest training gaits gives 0.06° on the test speed, better than the network's 0.12°.
  - The network's value is that it is **one small model** (5,976 numbers) that gives any speed directly and continuously, and that it learned the non-linear inverse-kinematics shape of every joint from data.
  - **Improvement:** evolve a richer family (let step height and body height vary with speed too), where blending stops working.
- **Fixed timing.** Period and duty are fixed, so the network can't change the walking rhythm.
- **Range.** Outside the trained speeds (0.088–0.339 m/s) the network is extrapolating.

### 4.7 Comparison with PyTorch (`compare_pytorch.py`, 5 marks)
**Setup:** the same architecture is built in PyTorch with the **same starting weights**, the same mini-batches in the same order, Adam with lr 1e-3, and the same 1386 epochs.
- **Gradients on one batch:** largest difference **2.8e-17**, i.e. our hand-written backprop equals autograd to rounding error.
- **After training:** validation loss 5.008e-6 (ours) against 5.005e-6 (PyTorch), and test RMSE 0.119° for both. The tiny gap is float rounding building up over 130,000 updates.
- **Time** (forward + backward + update only): ours 31 s, PyTorch 102 s.
  - At this size (batches of 16, matrices of 64 × 64) PyTorch's per-call overhead dominates. For big networks or GPUs PyTorch would win easily.
- **What PyTorch does for you:** autograd, ready-made optimisers and GPU support. What we wrote by hand is exactly what it automates.

---

## 5. How to rebuild it from nothing (order and checks)
1. **Kinematics.** Write `leg_fk.m` from the proto, then `leg_ik.m`. Check that IK → FK puts every foot back on its target (< 1e-12 m).
2. **Decoder.** Write `decode_gait.m`. Plot one gait's foot heights: flat while down, humps while up, two alternating groups.
3. **Fitness.** Write `gait_fitness.m`. A gait with stride 0 must score 0, and a sensible hand-made gait must break no rule.
4. **GA.** Write the loop and operators in `run_ga.m`. The best fitness must never drop (elitism), and the run must converge.
5. **Webots.** Play the result with `gait_player` and measure it with `measure.py`. Fix the gene ranges if the real robot disagrees; this is how the bodyHeight and reach limits were set.
6. **Data.** Make the speed family (`make_speed_family.m`), measure each gait in Webots, and run `python data.py`.
7. **Network.** Run `python nn.py` (the gradient checks must pass), then `python train.py` and `python plots.py`.
8. **Use and check.** `python predict.py <speed> --webots`: the walked speed must match the request. Finish with `python compare_pytorch.py`.

## 6. Known weaknesses (whole project)
- **Model-based fitness.** The fitness uses a model of the robot (kinematics and rules), not physics. Torque, bounce and physics sliding are only checked afterwards in Webots, with the start-up showing the worst numbers (tilt 2.8–4.7°).
- **Fixed gait structure.** The two groups and straight strokes are our choice. The GA tunes them but can't invent a different kind of gait.
- **Limited conditions.** Flat ground and straight walking only; the robot can't turn.
- **Start-up.** The start-up costs about 4° of heading before the walk settles.
