"""Run from repository root: python -m experiments.train --dataset synthetic."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from nn.layers import Dense, BatchNorm, Dropout
from nn.activations import ReLU
from nn.model import Sequential
from nn.optimizers import Adam
from nn.data import mnist, synthetic
from nn.training import train, evaluate
from nn.visualization import plots

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset", choices=["synthetic", "mnist"], default="synthetic")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    if args.dataset == "mnist":
        x, y, xt, yt = mnist()
        order = rng.permutation(len(x))
        xv, yv = x[order[:5000]], y[order[:5000]]
        x, y = x[order[5000:]], y[order[5000:]]
        widths, dropout = [784, 256, 128, 10], .1
    else:
        all_x, all_y = synthetic(args.seed)
        x, y = all_x[:1600], all_y[:1600]
        xv, yv = all_x[1600:2000], all_y[1600:2000]
        xt, yt = all_x[2000:], all_y[2000:]
        widths, dropout = [2, 64, 64, 3], 0.
    layers = []
    for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(Dense(a, b, rng, "xavier" if i == len(widths)-2 else "he"))
        if i < len(widths)-2:
            layers.extend([BatchNorm(b), ReLU(), Dropout(dropout, rng)])
    model = Sequential(layers)
    started = time.perf_counter()
    history = train(model, Adam(), x, y, xv, yv, epochs=args.epochs,
                    seed=args.seed, l2=1e-5, patience=8)
    loss, accuracy = evaluate(model, xt, yt)
    output = Path("results")/args.dataset
    output.mkdir(parents=True, exist_ok=True)
    model.save(output/"model.npz")
    before = model.forward(xt[:16]).copy()
    model.load(output/"model.npz")
    np.testing.assert_array_equal(before, model.forward(xt[:16]))
    predictions = np.concatenate([model.forward(xt[i:i+512]).argmax(axis=1) for i in range(0, len(xt), 512)])
    report = dict(dataset=args.dataset, seed=args.seed, architecture=widths,
                  train_examples=len(x), validation_examples=len(xv), test_examples=len(xt),
                  test_loss=loss, test_accuracy=accuracy, epochs_completed=len(history),
                  best_epoch=min(history, key=lambda r:r["validation_loss"])["epoch"],
                  elapsed_seconds=time.perf_counter()-started, numpy_version=np.__version__, history=history)
    (output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    plots(history, yt, predictions, output)
    print(f"Test loss: {loss:.5f}; test accuracy: {accuracy:.4%}")

if __name__ == "__main__":
    main()
