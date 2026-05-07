#!/usr/bin/env python3
"""Download Project 1 checkpoints from ModelScope into local grading paths."""

from __future__ import annotations

import argparse
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


def snapshot_with_sdk(model_id: str) -> Path | None:
    try:
        from modelscope import snapshot_download  # type: ignore
    except Exception:
        return None
    return Path(snapshot_download(model_id))


def snapshot_with_git(model_id: str) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="pj1_modelscope_"))
    repo_url = f"https://www.modelscope.cn/{model_id}.git"
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(temp_dir)],
        check=True,
    )
    return temp_dir


def copy_checkpoints(source_dir: Path) -> None:
    missing = []
    for src_rel, dst_rel in CHECKPOINT_MAP.items():
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
            "Missing checkpoint files in ModelScope snapshot:\n"
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
    copy_checkpoints(source_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
