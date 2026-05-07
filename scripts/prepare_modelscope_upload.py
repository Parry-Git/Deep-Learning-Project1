#!/usr/bin/env python3
"""Create a local ModelScope upload directory from trained checkpoints."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / ".modelscope_upload"

CHECKPOINTS = {
    "codes/best_models/mlp/best_model.pickle": "checkpoints/mlp/best_model.pickle",
    "codes/best_models/cnn/best_model.pickle": "checkpoints/cnn/best_model.pickle",
    "codes/part_c_results/recipe_full/sgd/best_model.pickle": "checkpoints/recipe_full/sgd/best_model.pickle",
    "codes/part_c_results/recipe_full/momentum/best_model.pickle": "checkpoints/recipe_full/momentum/best_model.pickle",
    "codes/part_c_results/recipe_full/sgd_step/best_model.pickle": "checkpoints/recipe_full/sgd_step/best_model.pickle",
    "codes/part_c_results/recipe_full/adam/best_model.pickle": "checkpoints/recipe_full/adam/best_model.pickle",
    "codes/part_c_results/recipe_full/l2/best_model.pickle": "checkpoints/recipe_full/l2/best_model.pickle",
    "codes/part_c_results/recipe_full/dropout/best_model.pickle": "checkpoints/recipe_full/dropout/best_model.pickle",
    "codes/part_c_results/recipe_full/recipe_combo/best_model.pickle": "checkpoints/recipe_full/recipe_combo/best_model.pickle",
}

METRICS = {
    "codes/part_c_results/recipe_full/summary.json": "metrics/recipe_full_summary.json",
    "codes/part_c_results/error_analysis/analysis_summary.json": "metrics/error_analysis_summary.json",
}

MODEL_CARD = """---
frameworks:
- NumPy
license: Apache License 2.0
tags:
- mnist
- numpy
- handwritten-convolution
- neural-network-course
tasks:
- image-classification
---

# Deep-Learning-Project1

This ModelScope repository stores the trained checkpoint files for the MNIST
NumPy implementation used in Project 1 of Neural Network and Deep Learning.

Code repository:
https://github.com/Parry-Git/Deep-Learning-Project1

## Checkpoints

| Path | Description | Test accuracy |
| --- | --- | ---: |
| `checkpoints/mlp/best_model.pickle` | Part A MLP baseline | 0.9544 |
| `checkpoints/cnn/best_model.pickle` | Part B CNN baseline | 0.9779 |
| `checkpoints/recipe_full/recipe_combo/best_model.pickle` | Best Part C recipe CNN | 0.9856 |

The `checkpoints/recipe_full/` directory also includes the saved models for
SGD, Momentum, StepLR, Adam, L2 weight decay, and dropout experiments.

## Download

```bash
git clone https://www.modelscope.cn/ParryY/Deep-Learning-Project1.git
```

or from the project repository:

```bash
python scripts/download_checkpoints.py
python scripts/verify_submission.py
```

The models are Python pickle files saved by the NumPy classes in `codes/mynn/`.
They are not PyTorch checkpoints.
"""


def copy_file(src_rel: str, dst_rel: str) -> None:
    src = ROOT / src_rel
    dst = OUT_DIR / dst_rel
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"copied {src_rel} -> {dst_rel}")


def main() -> int:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    for src, dst in CHECKPOINTS.items():
        copy_file(src, dst)
    for src, dst in METRICS.items():
        copy_file(src, dst)

    manifest = {
        "code_repository": "https://github.com/Parry-Git/Deep-Learning-Project1",
        "modelscope_model": "https://modelscope.cn/models/ParryY/Deep-Learning-Project1",
        "checkpoints": list(CHECKPOINTS.values()),
        "metrics": list(METRICS.values()),
        "expected_test_accuracy": {
            "mlp": 0.9544,
            "cnn": 0.9779,
            "recipe_combo": 0.9856,
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT_DIR / "README.md").write_text(MODEL_CARD)
    print(f"\nModelScope upload directory prepared at {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
