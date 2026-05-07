#!/usr/bin/env python3
"""Generate a small prediction grid to verify checkpoints and plotting."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODES = ROOT / "codes"
sys.path.insert(0, str(CODES))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mynn as nn
from test_model import load_mnist_test


DEFAULT_MODEL = CODES / "part_c_results/recipe_full/recipe_combo/best_model.pickle"
DEFAULT_OUT = CODES / "part_c_results/smoke_test/prediction_grid.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--num-samples", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {args.model_path}. "
            "Run `python scripts/download_checkpoints.py` first."
        )

    images, labels = load_mnist_test(flatten=False)
    num_samples = min(args.num_samples, images.shape[0])

    model = nn.models.Model_CNN()
    model.load_model(args.model_path)
    model.eval()

    logits = model(images[:num_samples])
    probs = nn.op.softmax(logits)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    accuracy = float(np.mean(preds == labels[:num_samples]))

    cols = min(4, num_samples)
    rows = int(np.ceil(num_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.2))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, image, label, pred, conf in zip(
        axes,
        images[:num_samples, 0],
        labels[:num_samples],
        preds,
        confs,
    ):
        color = "green" if int(label) == int(pred) else "red"
        ax.imshow(image, cmap="gray")
        ax.set_title(f"T:{int(label)} P:{int(pred)} {conf:.2f}", fontsize=9, color=color)

    fig.suptitle(f"Quick visualization check, accuracy on {num_samples} samples: {accuracy:.3f}")
    fig.tight_layout()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"saved {args.out_path.relative_to(ROOT)}")
    print(f"accuracy_on_grid {accuracy:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
