# Deep Learning Project 1

NumPy implementation for MNIST classification in Project 1 of Neural Network and
Deep Learning. The repository contains the code, report source, generated figures,
and reproducibility scripts. It intentionally does not include the MNIST dataset
or trained checkpoint pickle files.

## Repository Layout

```text
codes/
  mynn/                         Core NumPy layers, models, optimizer, runner
  test_train.py                 Train the Part A MLP or Part B CNN baseline
  test_model.py                 Evaluate saved MLP/CNN checkpoints on MNIST test set
  experiment_part_c_recipe.py   Optimization and regularization experiments
  analysis_visualization.py     Confusion matrices, misclassified samples, kernels
  figs/                         Learning curves used by the report
  part_c_results/               Result JSON files and visualization figures
report/
  report.tex                    Report source
  report.pdf                    Submitted report PDF
scripts/
  download_checkpoints.py       Download checkpoints from ModelScope
  quick_visualization_check.py  Generate a small prediction grid smoke test
  prepare_modelscope_upload.py  Build a local ModelScope upload directory
  verify_submission.py          One-command sanity check for grading
environment.yml                 Conda environment used for the final checks
```

## One-Command Grading Flow

The intended review flow is:

```bash
git clone https://github.com/Parry-Git/Deep-Learning-Project1.git
cd Deep-Learning-Project1
conda env create -f environment.yml
conda activate dl-pj1
```

Place the provided MNIST gzip files under:

```text
codes/dataset/MNIST/
  train-images-idx3-ubyte.gz
  train-labels-idx1-ubyte.gz
  t10k-images-idx3-ubyte.gz
  t10k-labels-idx1-ubyte.gz
```

The dataset is not tracked in Git because the project PDF asks not to upload it.

Then download checkpoints, run the compact evaluation check, and generate a
small visualization smoke test:

```bash
python scripts/download_checkpoints.py
python scripts/verify_submission.py
python scripts/quick_visualization_check.py
```

Expected checkpoint evaluation results:

```text
MLP baseline test accuracy: 0.9544
CNN baseline test accuracy: 0.9779
Best recipe CNN test accuracy: 0.9856
```

The quick visualization script writes:

```text
codes/part_c_results/smoke_test/prediction_grid.png
```

## Checkpoints

The trained checkpoints are hosted on ModelScope:

```text
https://modelscope.cn/models/ParryY/Deep-Learning-Project1
```

`scripts/download_checkpoints.py` first tries the ModelScope SDK, then falls back
to `git clone https://www.modelscope.cn/ParryY/Deep-Learning-Project1.git`.
It copies the checkpoint files into the local paths expected by `test_model.py`
and the analysis scripts.

## Reproduce Main Experiments

Run from `codes/`:

```bash
cd codes
python test_train.py --model mlp
python test_train.py --model cnn
python test_model.py --model mlp
python test_model.py --model cnn
python experiment_part_c_recipe.py \
  --recipes sgd,momentum,sgd_step,adam,l2,dropout,recipe_combo \
  --epochs 5 --eval-batch-size 1000 \
  --out-dir part_c_results/recipe_full
python analysis_visualization.py
```

No deep learning framework modules are used for the neural network operators.
Core layers and optimizers are implemented with NumPy.
