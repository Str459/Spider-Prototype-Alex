"""A small fully connected neural network written from scratch in numpy (Part 2).

No ML library is used: forward pass, backpropagation, loss, optimisers, mini-batch training with
early stopping, saving/loading and a numerical gradient check are all here, so every step of
learning can be explained and checked. PyTorch appears only in compare_pytorch.py. (Shared core written by our teammate.)

    model = MLP([3, 64, 64, 24], hidden="tanh", output="tanh")
    hist = train(model, X, Y, Xval, Yval, Adam(1e-3), epochs=2000, batch_size=64, patience=200)
    save(model, "model.npz"); model, extra = load("model.npz")

Shapes: X (n_samples, n_in), Y (n_samples, n_out). Self-check: python nn.py
"""
import numpy as np

# Each activation is (f(x), f'(x) written in terms of the output y = f(x)); expressing the
# derivative through the stored output saves keeping the pre-activations.
ACTIVATIONS = {
    "tanh": (np.tanh, lambda y: 1.0 - y ** 2),
    "relu": (lambda x: np.maximum(x, 0.0), lambda y: (y > 0).astype(float)),
    "sigmoid": (lambda x: 1.0 / (1.0 + np.exp(-x)), lambda y: y * (1.0 - y)),
    "linear": (lambda x: x, lambda y: np.ones_like(y)),
}


class MLP:
    """Multi-layer perceptron: sizes [n_in, h1, ..., n_out]; `hidden` activation on every hidden
    layer, `output` on the last. Weights: He initialisation for ReLU (variance 2/n_in), Xavier for
    the others (variance 2/(n_in+n_out)), so activations neither vanish nor blow up at the start."""

    def __init__(self, sizes, hidden="tanh", output="tanh", seed=0):
        if hidden not in ACTIVATIONS or output not in ACTIVATIONS:
            raise ValueError(f"activations must be one of {list(ACTIVATIONS)}")
        rng = np.random.default_rng(seed)
        self.sizes, self.hidden, self.output = list(sizes), hidden, output
        self.params = []                                   # [W0, b0, W1, b1, ...]
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            act = output if i == len(sizes) - 2 else hidden
            std = np.sqrt(2.0 / n_in) if act == "relu" else np.sqrt(2.0 / (n_in + n_out))
            self.params += [rng.normal(0.0, std, (n_in, n_out)), np.zeros(n_out)]
        self.grads = [np.zeros_like(p) for p in self.params]

    def _acts(self):
        n = len(self.sizes) - 1
        return [self.output if i == n - 1 else self.hidden for i in range(n)]

    def forward(self, X):
        """X (n, n_in) -> output (n, n_out). Keeps every layer's output for backward()."""
        self.outs = [np.asarray(X, dtype=float)]
        for i, act in enumerate(self._acts()):
            W, b = self.params[2 * i], self.params[2 * i + 1]
            self.outs.append(ACTIVATIONS[act][0](self.outs[-1] @ W + b))
        return self.outs[-1]

    def backward(self, dY):
        """dY = dLoss/dOutput (n, n_out) -> fills self.grads (same order as self.params).
        Chain rule, layer by layer from the output: delta = dL/d(pre-activation) of that layer."""
        delta = None
        for i in reversed(range(len(self.sizes) - 1)):
            act = self._acts()[i]
            y, a_prev = self.outs[i + 1], self.outs[i]
            delta = (dY if delta is None else delta @ self.params[2 * (i + 1)].T) * ACTIVATIONS[act][1](y)
            self.grads[2 * i] = a_prev.T @ delta              # dL/dW = input^T . delta
            self.grads[2 * i + 1] = delta.sum(axis=0)        # dL/db = sum of delta over the batch
        return self.grads


def mse(pred, target):
    """Mean squared error over every output of every sample -> (loss, dLoss/dpred)."""
    diff = pred - target
    return float(np.mean(diff ** 2)), 2.0 * diff / diff.size


class SGD:
    """Stochastic gradient descent with momentum: v = m v - lr g; p += v. Momentum averages the
    noisy mini-batch gradients and speeds up travel along shallow, consistent directions."""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr, self.momentum, self.v = lr, momentum, None

    def step(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.v):
            v *= self.momentum
            v -= self.lr * g
            p += v


class Adam:
    """Adam (Kingma & Ba 2015): per-weight step sizes from running means of the gradient (m) and
    its square (v), bias-corrected for the zero start. Less sensitive to the learning rate than SGD."""

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            m_hat, v_hat = m / (1 - self.b1 ** self.t), v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def train(model, X, Y, Xval, Yval, optimizer, epochs=1000, batch_size=64, patience=100, seed=0, log_every=0,
          l2=0.0):
    """Mini-batch training with shuffling and early stopping on the validation loss.
    l2 > 0 adds weight decay: loss + l2/2 * sum(W^2) over the weight matrices (not biases), i.e.
    gradient + l2 * W, which keeps weights small and the learned function smooth.
    Returns {"train": [...], "val": [...], "best_epoch": k}; the model ends with the weights of
    the best validation epoch (so a model that later overfits is not kept)."""
    rng = np.random.default_rng(seed)
    hist = {"train": [], "val": [], "best_epoch": 0}
    best, best_params, wait = np.inf, [p.copy() for p in model.params], 0
    for epoch in range(epochs):
        order = rng.permutation(len(X))
        for start in range(0, len(X), batch_size):
            idx = order[start:start + batch_size]
            _, d = mse(model.forward(X[idx]), Y[idx])
            grads = model.backward(d)
            if l2:
                grads = [g + l2 * p if i % 2 == 0 else g for i, (g, p) in enumerate(zip(grads, model.params))]
            optimizer.step(model.params, grads)
        hist["train"].append(mse(model.forward(X), Y)[0])
        hist["val"].append(mse(model.forward(Xval), Yval)[0])
        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:5d}  train {hist['train'][-1]:.6f}  val {hist['val'][-1]:.6f}", flush=True)
        if hist["val"][-1] < best:
            best, best_params, wait, hist["best_epoch"] = hist["val"][-1], [p.copy() for p in model.params], 0, epoch
        else:
            wait += 1
            if wait >= patience:
                break
    for p, q in zip(model.params, best_params):
        p[...] = q
    return hist


def save(model, path, **extra):
    """Weights + architecture (+ any extra arrays, e.g. normalisation constants) in one .npz."""
    arrays = {f"p{i}": p for i, p in enumerate(model.params)}
    np.savez(path, sizes=np.array(model.sizes), hidden=model.hidden, output=model.output, **arrays,
             **{f"extra_{k}": np.asarray(v) for k, v in extra.items()})


def load(path):
    """-> (model, extra dict) as saved by save()."""
    with np.load(path) as f:
        model = MLP(f["sizes"].tolist(), str(f["hidden"]), str(f["output"]))
        for i in range(len(model.params)):
            model.params[i][...] = f[f"p{i}"]
        extra = {k[6:]: f[k] for k in f.files if k.startswith("extra_")}
    return model, extra


def gradient_check(model, X, Y, eps=1e-6):
    """Largest relative difference between backprop gradients and central finite differences
    (dL/dp ~ (L(p+eps) - L(p-eps)) / 2 eps) over every weight. ~1e-7 or less means backprop is right."""
    _, d = mse(model.forward(X), Y)
    analytic = [g.copy() for g in model.backward(d)]
    worst = 0.0
    for p, g in zip(model.params, analytic):
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[i]
            p[i] = old + eps
            up = mse(model.forward(X), Y)[0]
            p[i] = old - eps
            down = mse(model.forward(X), Y)[0]
            p[i] = old
            num = (up - down) / (2 * eps)
            worst = max(worst, abs(num - g[i]) / max(1e-8, abs(num) + abs(g[i])))
    return worst


if __name__ == "__main__":
    import os
    import tempfile

    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(8, 3)), rng.uniform(-0.9, 0.9, size=(8, 2))
    for hidden in ("tanh", "relu", "sigmoid"):
        for output in ("tanh", "linear"):
            err = gradient_check(MLP([3, 5, 4, 2], hidden, output, seed=1), X, Y)
            assert err < 1e-5, (hidden, output, err)
    print("gradient check passed for tanh/relu/sigmoid hidden x tanh/linear output")

    # learn a smooth function of a phase, the kind of mapping option A needs
    t = rng.uniform(0, 2 * np.pi, 400)
    Xs, Ys = np.column_stack([np.sin(t), np.cos(t)]), 0.8 * np.column_stack([np.sin(2 * t), np.cos(3 * t)])
    for opt in (SGD(0.05, 0.9), Adam(0.01)):
        m = MLP([2, 32, 32, 2], "tanh", "tanh", seed=0)
        h = train(m, Xs[:300], Ys[:300], Xs[300:], Ys[300:], opt, epochs=400, batch_size=32, patience=100)
        assert h["val"][h["best_epoch"]] < 0.01 * h["val"][0], (type(opt).__name__, h["val"][0], min(h["val"]))
        print(f"{type(opt).__name__:4s} converged: val loss {h['val'][0]:.4f} -> {min(h['val']):.6f}")

    norms = []
    for l2 in (0.0, 1e-2):                                  # weight decay must shrink the weights
        m_l2 = MLP([2, 32, 32, 2], "tanh", "tanh", seed=0)
        train(m_l2, Xs[:300], Ys[:300], Xs[300:], Ys[300:], Adam(0.01), epochs=100, patience=100, l2=l2)
        norms.append(sum(float((w ** 2).sum()) for w in m_l2.params[0::2]))
    assert norms[1] < norms[0], norms
    print(f"weight decay shrinks the weights: sum W^2 {norms[0]:.1f} -> {norms[1]:.1f}")

    path = os.path.join(tempfile.mkdtemp(), "m.npz")
    save(m, path, mean=np.arange(3.0))
    m2, extra = load(path)
    assert np.allclose(m2.forward(Xs), m.forward(Xs)) and np.allclose(extra["mean"], np.arange(3.0))
    print("save/load round trip passed\nnn self-check passed")
