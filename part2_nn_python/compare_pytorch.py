"""Compare our from-scratch network (nn.py) with the same network in PyTorch (run train.py first).

    python compare_pytorch.py

Both start from the SAME weights, see the same mini-batches in the same order and use the same
optimiser and learning rate, so any difference comes from the implementation, not from luck.
1. gradients of one batch: our backprop against PyTorch's autograd
2. training for the same number of epochs: loss curves, final error, time
Saves outputs/nn/pytorch_comparison.png and pytorch_comparison.txt.
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import data
import nn

OUT = os.path.join(data.ROOT, "outputs", "nn")
torch.set_default_dtype(torch.float64)
ACT = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU, "sigmoid": torch.nn.Sigmoid}


def torch_copy(model):
    """A PyTorch network with exactly the same layers and weights as our MLP."""
    layers = []
    for i in range(len(model.sizes) - 1):
        lin = torch.nn.Linear(model.sizes[i], model.sizes[i + 1])
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(model.params[2 * i].T))     # torch stores (out, in)
            lin.bias.copy_(torch.tensor(model.params[2 * i + 1]))
        layers += [lin, ACT[model.output if i == len(model.sizes) - 2 else model.hidden]()]
    return torch.nn.Sequential(*layers)


def main():
    with open(os.path.join(OUT, "results.json")) as f:
        res = json.load(f)
    cfg = res["summary"]["best_cfg"]
    epochs = len(res["runs"][res["summary"]["best_run"]]["train"])
    d = data.load()
    ours = nn.MLP([3] + cfg["sizes"] + [24], hidden=cfg["hidden"], output="tanh", seed=0)
    theirs = torch_copy(ours)
    X, Y = d["X"], d["Y"]

    # 1. gradients on one batch
    _, dY = nn.mse(ours.forward(X[:64]), Y[:64])
    g_ours = ours.backward(dY)
    theirs.zero_grad()
    torch.nn.functional.mse_loss(theirs(torch.tensor(X[:64])), torch.tensor(Y[:64])).backward()
    g_theirs = []
    for m in theirs:
        if isinstance(m, torch.nn.Linear):
            g_theirs += [m.weight.grad.numpy().T, m.bias.grad.numpy()]
    grad_diff = max(float(np.abs(a - b).max()) for a, b in zip(g_ours, g_theirs))

    # 2. same training, same batches
    opt_ours = nn.Adam(cfg["lr"]) if cfg["opt"] == "adam" else nn.SGD(cfg["lr"], 0.9)
    opt_theirs = (torch.optim.Adam(theirs.parameters(), lr=cfg["lr"]) if cfg["opt"] == "adam"
                  else torch.optim.SGD(theirs.parameters(), lr=cfg["lr"], momentum=0.9))
    rng = np.random.default_rng(0)
    curves = {"ours": [], "pytorch": []}
    times = {"ours": 0.0, "pytorch": 0.0}
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for start in range(0, len(X), cfg["batch"]):
            idx = order[start:start + cfg["batch"]]
            t = time.perf_counter()
            _, dY = nn.mse(ours.forward(X[idx]), Y[idx])
            opt_ours.step(ours.params, ours.backward(dY))
            times["ours"] += time.perf_counter() - t
            t = time.perf_counter()
            opt_theirs.zero_grad()
            torch.nn.functional.mse_loss(theirs(Xt[idx]), Yt[idx]).backward()
            opt_theirs.step()
            times["pytorch"] += time.perf_counter() - t
        with torch.no_grad():
            curves["pytorch"].append(float(torch.nn.functional.mse_loss(theirs(torch.tensor(d["Xval"])),
                                                                        torch.tensor(d["Yval"]))))
        curves["ours"].append(nn.mse(ours.forward(d["Xval"]), d["Yval"])[0])

    with torch.no_grad():
        pred_theirs = theirs(torch.tensor(d["Xtest"])).numpy()
    test = {"ours": ours.forward(d["Xtest"]), "pytorch": pred_theirs}
    target = data.unscale(d["Ytest"])
    deg = {k: float(np.degrees(np.sqrt(np.mean((data.unscale(v) - target) ** 2)))) for k, v in test.items()}

    lines = [f"network {cfg}, {epochs} epochs, same start weights and batches",
             f"largest gradient difference on one batch: {grad_diff:.1e}",
             f"final validation loss: ours {curves['ours'][-1]:.6e}, PyTorch {curves['pytorch'][-1]:.6e}",
             f"test speed RMSE: ours {deg['ours']:.3f} deg, PyTorch {deg['pytorch']:.3f} deg",
             f"training time (forward + backward + update only): ours {times['ours']:.1f} s, "
             f"PyTorch {times['pytorch']:.1f} s"]
    print("\n".join(lines))
    with open(os.path.join(OUT, "pytorch_comparison.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    plt.figure(figsize=(7, 4))
    plt.semilogy(curves["ours"], label="ours (numpy, from scratch)")
    plt.semilogy(curves["pytorch"], "--", label="PyTorch")
    plt.xlabel("epoch"); plt.ylabel("validation loss"); plt.legend(); plt.grid(alpha=0.3)
    plt.title("Same network, same start, same batches")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "pytorch_comparison.png"), dpi=120); plt.close()


if __name__ == "__main__":
    main()
