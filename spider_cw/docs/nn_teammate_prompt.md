# Prompt for a teammate: the Part 2 neural network

Copy everything below the line into your AI assistant, or follow it yourself. You need the shared project folder: the `spider_cw/` folder and `Spider_world_demo/` next to it.

---

## Context
- **Coursework:** M33174 group coursework, "Using GAs and Neural Networks for Solving the Spider Kinematics Problem". The spec is `Info/M33174_cw_item1_.pdf`; the deadline is 2 Dec 2026, 13:00.
- **Part 1 (done):** a genetic algorithm, written in Python, evolves walking gaits for the Webots 8-leg spider (`Spider_world_demo/protos/SpiderOctopod.proto`). Each gait is 300 frames × 24 joint angles (radians, Webots motor angles).
  - **Leg order:** L1, L2, L3, L4, R4, R3, R2, R1, each with (hip, thigh, knee). The column names are `part1_ga.gait.MOTOR_NAMES`.
  - **Output:** saved runs are in `spider_cw/outputs/<run>/best_gait.csv`, with a header row of motor names followed by 300 rows of angles.
- **Part 2 (now):** a neural network that outputs the 24 joint angles.
  - **Marking scheme (100):**
    - architecture 20;
    - activation and loss functions 15;
    - training method and learning rate 15;
    - backpropagation and convergence 15;
    - data handling 10;
    - visualisation 10;
    - explanation 10;
    - comparison with PyTorch/TensorFlow 5.
  - **Library code earns nothing in categories 1–7**, so the network must be written from scratch in numpy. PyTorch is only for the comparison.
  - **Required deliverables:**
    - the number of layers and neurons, activations, learning rate, loss and training method;
    - how the dataset was made;
    - a saved model;
    - loss-over-epochs plots;
    - example output poses;
    - predicted vs target comparisons.

## Useful code that already exists (`spider_cw/part1_ga/`)
- `gait.py`: `MOTOR_NAMES`, `MOTOR_LO` / `MOTOR_HI` (the 24 motor limits), `decode(chromosome)` → (300, 24).
- `body.py`: the Webots spider geometry.
  - `forward_kinematics(angles)`: (..., 24) → (..., 8, 4, 3), giving [hip, knee1, knee2, foot] positions per leg in the body frame (metres).
  - `plot_pose(ax, angles)`: draws a pose on a matplotlib 3D axis.
- **Self-checks:** run `python -m part1_ga.body` and `python -m part1_ga.gait` from `spider_cw/`.
- **Environment:** Python 3 with numpy and matplotlib; Webots R2025a.

## The plan: two networks sharing one from-scratch core
- **Option A (main network):** input `[sin φ, cos φ, speed]` → 24 joint angles, where φ is the phase within the gait cycle. Trained on a family of GA gaits evolved at 6 target speeds. One whole speed is held out for testing; the network's gait for that speed is walked in Webots.
- **Option E (second network, your part):** inverse kinematics. Input: the 8 target foot positions (24 numbers) → output: the 24 joint angles that put the feet there.

## Shared core: `spider_cw/part2_nn/nn.py`, numpy only
It's being written now. Use it rather than writing your own, so the two networks share the same tested code. The interface:
- `MLP(sizes, hidden="tanh", output="tanh"|"linear", seed=0)`
  - `.forward(X)` → predictions; `.backward(dY)` fills `.grads`; `.params` lists the weights and biases.
  - Weights use Xavier/He initialisation.
- **Activations:** tanh, relu and sigmoid, each with its derivative.
- `mse(pred, target)` → `(loss, dpred)`.
- **Optimisers:** `SGD(lr, momentum)` and `Adam(lr, beta1, beta2, eps)`, each with `.step(params, grads)`.
- `train(model, X, Y, Xval, Yval, optimizer, epochs, batch_size, patience, l2=0.0)` → a history dict of train and validation loss per epoch. It uses mini-batches, shuffling and early stopping. `l2` is optional weight decay.
- **Lesson from option A:** choose settings with a validation set that tests what you actually care about. Random frames of the training gaits picked the networks that generalised worst. For IK, that means validating on new random poses, which is already the natural split.
- `save(model, path)` / `load(path)`, as an `.npz` file with the weights plus the layer sizes and activations.
- `gradient_check(model, X, Y)`: compares the backprop gradients with finite differences. This is the evidence for marking category 4.

## Your part: option E, the inverse-kinematics network (`spider_cw/part2_nn/ik/`)
1. **Data** (`ik_data.py`):
   - Sample random joint angles inside realistic ranges. For left legs: hip [-1.2, 1.2], thigh [-1.5, 0], knee [-0.5, 1]. Right legs are the mirror image: R = −L for all three joints, see `gait.py`.
   - Compute the feet with `body.forward_kinematics(angles)[..., 3, :]` and flatten to 24 numbers.
   - Make about 50k samples, split into train / validation / test.
   - Normalise the inputs (per-coordinate mean and standard deviation, fitted on the training set only). Scale the outputs to [-1, 1] using the motor limits, so a tanh output layer always gives a valid angle.
2. **Network:** 24 → H → H → 24, with tanh hidden layers and a tanh output, and MSE loss.
   - Experiment and record results for:
     - learning rate (e.g. 1e-4 … 1e-1);
     - SGD+momentum vs Adam;
     - tanh vs ReLU;
     - H = 32 / 64 / 128 and 1 vs 2 hidden layers;
     - batch size.
   - Keep the best model and save it as `part2_nn/ik/ik_model.npz`.
3. **The metric that matters:** push the predicted angles back through `forward_kinematics` and report the foot position error in centimetres (mean and 95th percentile), not only the angle MSE.
   - **Explain the multiple-solutions issue:** several angle sets can put a foot in the same place, so the network averages them. The restricted angle ranges keep this small.
   - **Also note:** each leg's angles depend only on that leg's foot. A per-leg 3 → 3 network shared by all legs is an interesting comparison.
4. **Plots** (`ik_plots.py`):
   - train and validation loss per epoch;
   - the learning-rate sweep;
   - predicted vs target angles (scatter);
   - a histogram of foot errors;
   - a few 3D poses comparing the target pose with the predicted one (`body.plot_pose`).
5. **PyTorch comparison** (`compare_pytorch.py`, needs `pip install torch`): the same architecture, data split, optimiser and learning rate. Compare loss curves, final error and training time, and explain any differences.
6. **Explanation** (`docs/nn_ik.txt`): cover every required item:
   - layers and neurons;
   - activations;
   - learning rate and how it was tuned;
   - loss;
   - optimiser and batch size;
   - how the dataset was generated and normalised;
   - the saved model;
   - results;
   - trade-offs and improvements.

## Rules
- Run everything from `spider_cw/` as modules, e.g. `python -m part2_nn.ik.train`.
- Don't edit `part1_ga/` or `nn.py` without telling the rest of the group. If the core needs a change, agree on it first, so option A keeps working.
- Every script should have a small self-check (asserts) or print a clear result.
- Keep random seeds fixed, so results can be reproduced.
