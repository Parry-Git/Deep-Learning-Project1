#!/usr/bin/env python3
"""Run a compact grading-oriented verification for Project 1."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODES = ROOT / "codes"
DATASET_FILES = [
    CODES / "dataset/MNIST/train-images-idx3-ubyte.gz",
    CODES / "dataset/MNIST/train-labels-idx1-ubyte.gz",
    CODES / "dataset/MNIST/t10k-images-idx3-ubyte.gz",
    CODES / "dataset/MNIST/t10k-labels-idx1-ubyte.gz",
]
CHECKPOINTS = {
    "mlp": CODES / "best_models/mlp/best_model.pickle",
    "cnn": CODES / "best_models/cnn/best_model.pickle",
    "recipe_combo": CODES / "part_c_results/recipe_full/recipe_combo/best_model.pickle",
}


def require_files(paths: list[Path], label: str) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        lines = "\n".join(f"- {path.relative_to(ROOT)}" for path in missing)
        raise FileNotFoundError(f"Missing {label} files:\n{lines}")


def run(command: list[str], cwd: Path) -> str:
    print("$ " + " ".join(command))
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = completed.stdout.strip()
    if output:
        print(output)
    return output


def parse_accuracy(output: str) -> float:
    return float(output.strip().splitlines()[-1])


def main() -> int:
    require_files(DATASET_FILES, "MNIST dataset")
    require_files(list(CHECKPOINTS.values()), "checkpoint")

    py_files = [
        "test_train.py",
        "test_model.py",
        "experiment_part_c_recipe.py",
        "analysis_visualization.py",
        *[str(path.relative_to(CODES)) for path in sorted((CODES / "mynn").glob("*.py"))],
    ]
    run([sys.executable, "-m", "py_compile", *py_files], CODES)

    mlp_acc = parse_accuracy(run([sys.executable, "test_model.py", "--model", "mlp"], CODES))
    cnn_acc = parse_accuracy(run([sys.executable, "test_model.py", "--model", "cnn"], CODES))
    combo_acc = parse_accuracy(
        run(
            [
                sys.executable,
                "test_model.py",
                "--model",
                "cnn",
                "--model-path",
                "part_c_results/recipe_full/recipe_combo/best_model.pickle",
            ],
            CODES,
        )
    )

    print("\nSummary")
    print(f"MLP baseline test accuracy: {mlp_acc:.4f}")
    print(f"CNN baseline test accuracy: {cnn_acc:.4f}")
    print(f"Best recipe CNN test accuracy: {combo_acc:.4f}")

    if mlp_acc < 0.94 or cnn_acc < 0.97 or combo_acc < 0.98:
        raise RuntimeError("One or more checkpoint accuracies are below expected thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
