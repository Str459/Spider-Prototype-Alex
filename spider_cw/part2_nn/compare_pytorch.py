"""Compare the from-scratch numpy network with the same network in PyTorch (marking category 8).

    cd spider_cw
    python -m part2_nn.compare_pytorch        # after train_speed; needs torch

The PyTorch model is built to be identical: same layer sizes and activations, the SAME initial
weights (copied from the numpy model), float64, the same optimiser and learning rate, and the same
mini-batches in the same order (same shuffling seed). Torch's SGD momentum (v = m v + g; p -= lr v)
is the same update as ours (v = m v - lr g; p += v) for a constant learning rate.
1. Gradients: our hand-written backprop vs torch autograd on one batch (should agree to ~1e-12).
2. Training: both trained for the final model's epoch count; loss curves, held-out test error,
   and wall time compared.
Outputs: outputs/nn_speed/pytorch_comparison.png and pytorch_comparison.txt.
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from part2_nn import nn
from part2_nn.speed_data import build
from part2_nn.train_speed import OUT, make_model, rmse_deg

ACT = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU, "sigmoid": torch.nn.Sigmoid, "linear": torch.nn.Identity}


def torch_copy(model):
    """A torch.nn.Sequential with the same architecture and weights as a numpy MLP."""
    layers, acts = [], model._acts()
    for i, act in enumerate(acts):
        W, b = model.params[2 * i], model.params[2 * i + 1]
        lin = torch.nn.Linear(W.shape[0], W.shape[1]).double()
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(W.T.copy()))     # torch stores (out, in)
            lin.bias.copy_(torch.from_numpy(b.copy()))
        layers += [lin, ACT[act]()]
    return torch.nn.Sequential(*layers)


def torch_optimizer(cfg, params):
    if cfg["opt"] == "adam":
        return torch.optim.Adam(params, lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8)
    return torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9)


def gradient_agreement(cfg, d):
    ours, _ = make_model(cfg)
    net = torch_copy(ours)
    X, Y = d["X_train"][:64], d["Y_train"][:64]
    _, dpred = nn.mse(ours.forward(X), Y)
    g_ours = ours.backward(dpred)
    loss = torch.nn.functional.mse_loss(net(torch.from_numpy(X)), torch.from_numpy(Y))
    loss.backward()
    g_torch = []
    for m in net:
        if isinstance(m, torch.nn.Linear):
            g_torch += [m.weight.grad.numpy().T, m.bias.grad.numpy()]
    return max(float(np.abs(a - b).max()) for a, b in zip(g_ours, g_torch))


def train_torch(cfg, d, epochs, net):
    opt = torch_optimizer(cfg, net.parameters())
    rng = np.random.default_rng(cfg["seed"])                    # same batches as nn.train
    X, Y = torch.from_numpy(d["X_train"]), torch.from_numpy(d["Y_train"])
    losses = []
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for start in range(0, len(X), cfg["batch"]):
            idx = torch.from_numpy(order[start:start + cfg["batch"]])
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(X[idx]), Y[idx])
            if cfg["l2"]:                                           # same weight decay as nn.train
                loss = loss + cfg["l2"] / 2 * sum((m.weight ** 2).sum() for m in net if isinstance(m, torch.nn.Linear))
            loss.backward()
            opt.step()
        with torch.no_grad():
            losses.append(float(torch.nn.functional.mse_loss(net(X), Y)))
    return losses


def main():
    with open(os.path.join(OUT, "experiments.json")) as f:
        final = json.load(f)["final"]
    cfg, epochs = final["config"], final["epochs"]
    torch.set_num_threads(1)                                    # fair against single-threaded numpy
    d = build()

    grad_diff = gradient_agreement(cfg, d)

    ours, opt = make_model(cfg)
    net = torch_copy(ours)                                      # same starting weights
    t0 = time.perf_counter()
    hist = nn.train(ours, d["X_train"], d["Y_train"], d["X_val"], d["Y_val"], opt, epochs=epochs,
                    batch_size=cfg["batch"], patience=10 ** 9, seed=cfg["seed"], l2=cfg["l2"])
    t_ours = time.perf_counter() - t0
    t0 = time.perf_counter()
    torch_losses = train_torch(cfg, d, epochs, net)
    t_torch = time.perf_counter() - t0
    with torch.no_grad():
        pred_torch = net(torch.from_numpy(d["X_test"])).numpy()
    rmse_ours, rmse_torch = rmse_deg(ours.forward(d["X_test"]), d["Y_test"]), rmse_deg(pred_torch, d["Y_test"])
    weight_diff = max(float(np.abs(W - m.weight.detach().numpy().T).max())
                      for W, m in zip(ours.params[0::2], [m for m in net if isinstance(m, torch.nn.Linear)]))

    text = (f"config {cfg}, {epochs} epochs, float64, same initial weights and batches\n"
            f"gradient agreement on one batch: max |numpy - torch autograd| = {grad_diff:.2e}\n"
            f"final training loss: numpy {hist['train'][-1]:.6e}  torch {torch_losses[-1]:.6e}\n"
            f"largest weight difference after training: {weight_diff:.2e}\n"
            f"held-out test speed RMSE: numpy {rmse_ours:.3f} deg  torch {rmse_torch:.3f} deg\n"
            f"training time: numpy {t_ours:.1f} s  torch {t_torch:.1f} s (1 thread)\n")
    print(text, flush=True)
    with open(os.path.join(OUT, "pytorch_comparison.txt"), "w") as f:
        f.write(text)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(hist["train"], lw=3, alpha=0.6, label=f"numpy, from scratch ({t_ours:.1f} s)")
    ax.semilogy(torch_losses, "k--", lw=1, label=f"PyTorch ({t_torch:.1f} s)")
    ax.set(xlabel="Epoch", ylabel="Training MSE", title="Same network, same start, same batches: numpy vs PyTorch")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.savefig(os.path.join(OUT, "pytorch_comparison.png"), dpi=120, bbox_inches="tight")
    print(os.path.join(OUT, "pytorch_comparison.png"))


if __name__ == "__main__":
    main()
