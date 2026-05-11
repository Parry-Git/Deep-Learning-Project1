#!/usr/bin/env python3
"""Download Project 1 checkpoints and MNIST dataset from ModelScope."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "ParryY/Deep-Learning-Project1"

CHECKPOINT_MAP = {
    "checkpoints/mlp/best_model.pickle": "codes/best_models/mlp/best_model.pickle",
    "checkpoints/cnn/best_model.pickle": "codes/best_models/cnn/best_model.pickle",
    "checkpoints/recipe_full/sgd/best_model.pickle": "codes/part_c_results/recipe_full/sgd/best_model.pickle",
    "checkpoints/recipe_full/momentum/best_model.pickle": "codes/part_c_results/recipe_full/momentum/best_model.pickle",
    "checkpoints/recipe_full/sgd_step/best_model.pickle": "codes/part_c_results/recipe_full/sgd_step/best_model.pickle",
    "checkpoints/recipe_full/adam/best_model.pickle": "codes/part_c_results/recipe_full/adam/best_model.pickle",
    "checkpoints/recipe_full/l2/best_model.pickle": "codes/part_c_results/recipe_full/l2/best_model.pickle",
    "checkpoints/recipe_full/dropout/best_model.pickle": "codes/part_c_results/recipe_full/dropout/best_model.pickle",
    "checkpoints/recipe_full/recipe_combo/best_model.pickle": "codes/part_c_results/recipe_full/recipe_combo/best_model.pickle",
}

DATASET_MAP = {
    "dataset/MNIST/train-images-idx3-ubyte.gz": "codes/dataset/MNIST/train-images-idx3-ubyte.gz",
    "dataset/MNIST/train-labels-idx1-ubyte.gz": "codes/dataset/MNIST/train-labels-idx1-ubyte.gz",
    "dataset/MNIST/t10k-images-idx3-ubyte.gz": "codes/dataset/MNIST/t10k-images-idx3-ubyte.gz",
    "dataset/MNIST/t10k-labels-idx1-ubyte.gz": "codes/dataset/MNIST/t10k-labels-idx1-ubyte.gz",
}

PROXY_ENV_VARS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]


def env_without_proxy() -> dict[str, str]:
    env = os.environ.copy()
    for key in PROXY_ENV_VARS:
        env.pop(key, None)
    return env


def snapshot_with_sdk(model_id: str) -> Path | None:
    try:
        from modelscope import snapshot_download  # type: ignore
    except Exception:
        return None
    try:
        cache_dir = Path(tempfile.mkdtemp(prefix="pj1_modelscope_cache_"))
        return Path(snapshot_download(model_id, cache_dir=str(cache_dir)))
    except Exception as exc:
        print(f"ModelScope SDK download failed, falling back to git clone: {exc}", file=sys.stderr)
        return None


def snapshot_with_git(model_id: str) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="pj1_modelscope_"))
    repo_url = f"https://www.modelscope.cn/{model_id}.git"
    command = ["git", "clone", "--depth", "1", repo_url, str(temp_dir)]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            f"git clone with current network environment failed ({exc}); retrying without proxy env.",
            file=sys.stderr,
        )
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="pj1_modelscope_"))
        command[-1] = str(temp_dir)
        subprocess.run(command, check=True, env=env_without_proxy())
    return temp_dir


def copy_files(source_dir: Path, file_map: dict[str, str], label: str) -> None:
    missing = []
    for src_rel, dst_rel in file_map.items():
        src = source_dir / src_rel
        dst = ROOT / dst_rel
        if not src.exists():
            missing.append(src_rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied {src_rel} -> {dst_rel}")
    if missing:
        raise FileNotFoundError(
            f"Missing {label} files in ModelScope snapshot:\n"
            + "\n".join(f"- {item}" for item in missing)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = snapshot_with_sdk(args.model_id)
    if source_dir is None:
        source_dir = snapshot_with_git(args.model_id)
    copy_files(source_dir, CHECKPOINT_MAP, "checkpoint")
    copy_files(source_dir, DATASET_MAP, "dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
