# MNIST Classification with NumPy-only MLP and CNN

All core operators (linear layer, conv2D, softmax cross-entropy, max pooling, dropout)
are implemented from scratch in NumPy without deep-learning frameworks.

## File Structure

| File | Description |
|---|---|
| `mynn/op.py` | Core layers and loss functions |
| `mynn/models.py` | `Model_MLP` and `Model_CNN` definitions |
| `mynn/optimizer.py` | SGD, Momentum, AdaGrad, RMSProp, Adam |
| `mynn/lr_scheduler.py` | StepLR, MultiStepLR, ExponentialLR |
| `mynn/runner.py` | Training loop, evaluation, early stopping |
| `test_train.py` | Part A / Part B training entry point |
| `test_model.py` | Evaluate a saved model on the test set |
| `experiment_part_c_recipe.py` | Part C optimization and regularization experiments |
| `analysis_visualization.py` | Confusion matrix, misclassified examples, weight and kernel visualization |

## Quick Start

```bash
# 1. Environment
conda env create -f ../environment.yml
conda activate dl-pj1

# 2. Place MNIST data
#    Put the four gzip files under codes/dataset/MNIST/

# 3. Download trained checkpoints (optional, for evaluation only)
python ../scripts/download_checkpoints.py

# 4. Verify
python ../scripts/verify_submission.py

# 5. Train from scratch
python test_train.py --model mlp
python test_train.py --model cnn

# 6. Evaluate
python test_model.py --model mlp
python test_model.py --model cnn

# 7. Part C experiments
python experiment_part_c_recipe.py \
  --recipes sgd,momentum,sgd_step,adam,l2,dropout,recipe_combo \
  --epochs 5 --eval-batch-size 1000 \
  --out-dir part_c_results/recipe_full

# 8. Error analysis and visualization
python analysis_visualization.py
```

## Troubleshooting

### `_pickle.UnpicklingError: invalid load key, 'v'.`

Model checkpoints are stored with **Git LFS** on ModelScope. If `git-lfs` is not
installed, `git clone` downloads only pointer files (~131 bytes) instead of the
actual weights. Fix:

```bash
# macOS
brew install git-lfs
# Linux (Debian/Ubuntu)
sudo apt install git-lfs

git lfs install
# then re-download checkpoints
python ../scripts/download_checkpoints.py
```

### `ModuleNotFoundError: No module named 'numpy._core.numeric'`

Checkpoints were serialized with **NumPy >= 2.0**, which uses the internal path
`numpy._core`. If your environment has NumPy 1.x (`numpy.core`), pickle will
fail. The evaluation scripts (`test_model.py`, `analysis_visualization.py`)
include a compatibility shim, but the simplest fix is to use the provided
environment:

```bash
conda env create -f ../environment.yml
conda activate dl-pj1
```
